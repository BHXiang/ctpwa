#ifndef PARAMETERS_CUH
#define PARAMETERS_CUH

#include <torch/torch.h>

#include <complex>
#include <vector>

// ============================================================
// Parameters: 统一管理耦合参数 (vector) 和共振态参数 (theta)
//
// 职责：
//   1. 存储约束定义（替代分散在 NLLFunction / analysis 的静态成员）
//   2. 提供 free → extended 向量变换（施加约束）
//   3. 提供 extended → free 梯度收缩（通过约束雅可比转置）
//   4. 作为参数维度的唯一数据源
// ============================================================
class Parameters {
public:
    // 单个约束组：一组 vector 索引通过线性关系约束在一起
    struct ConstraintGroup {
        int origin_id;                 // 自由参数在 extended 向量中的下标（组内最小 id）
        int origin_idx_in_group;       // origin 在 ext_indices 中的位置
        std::vector<int> ext_indices;  // 该组所有下标（含 origin）
        std::vector<float> real_ratios; // ext_val.real / origin_val.real
        std::vector<float> imag_ratios; // ext_val.imag / origin_val.imag
    };

    Parameters() = default;

    // 用约束数据初始化
    //   n_free_vector: 自由耦合参数个数（= n_gls - n_constraint_groups）
    //   con_ids:      每个约束组的扩展下标列表
    //   con_values:   每个约束组对应的复系数列表
    void initialize(int n_free_vector,
                    const std::vector<std::vector<int>>& con_ids,
                    const std::vector<std::vector<std::complex<double>>>& con_values);

    // ---------- 向量变换 ----------

    // 将自由向量扩展到含约束的完整向量
    //   free_vector: [nFreeVector] complex, 在 GPU 上
    //   返回: [nExtended] complex, 在 device 上
    torch::Tensor extendVector(const torch::Tensor& free_vector,
                               const torch::Device& device) const;

    // 将扩展梯度收缩回自由参数梯度（应用约束雅可比转置）
    //   extended_grad: [nExtended] complex, 在 GPU 上
    //   返回: [original_size] complex, 在 extended_grad 的设备上
    torch::Tensor collapseVectorGrad(const torch::Tensor& extended_grad,
                                    int original_size) const;

    // ---------- 维度查询 ----------

    int nFreeVector() const { return n_free_vector_; }
    int nFreeTheta() const { return n_free_theta_; }
    int nExtended() const { return n_extended_; }
    int nTotalFree() const { return n_free_vector_ + n_free_theta_; }
    bool hasConstraints() const { return !groups_.empty(); }
    bool initialized() const { return initialized_; }

    // ---------- theta 维度 ----------

    void setNFreeTheta(int n) { n_free_theta_ = n; }

    // ---------- 约束访问 ----------

    const std::vector<ConstraintGroup>& constraintGroups() const { return groups_; }

    const std::vector<std::vector<int>>& constraintsIndex() const { return con_indices_; }
    const std::vector<std::vector<std::complex<double>>>& constraintsValues() const { return con_values_; }

private:
    int n_free_vector_ = 0;
    int n_free_theta_ = 0;
    int n_extended_ = 0;
    std::vector<ConstraintGroup> groups_;
    std::vector<std::vector<int>> con_indices_;          // 原始约束下标（兼容 getConstraintsIndex）
    std::vector<std::vector<std::complex<double>>> con_values_; // 原始约束值（兼容 getConstraintsValues）
    bool initialized_ = false;
};

// ============================================================
// 实现
// ============================================================

inline void Parameters::initialize(
    int n_free_vector,
    const std::vector<std::vector<int>>& con_ids,
    const std::vector<std::vector<std::complex<double>>>& con_values)
{
    n_free_vector_ = n_free_vector;
    n_extended_ = n_free_vector;
    con_indices_ = con_ids;
    con_values_ = con_values;

    // 找到最大 ID 以确定扩展后向量大小
    for (const auto& vecid : con_ids) {
        if (!vecid.empty()) {
            auto max_it = std::max_element(vecid.begin(), vecid.end());
            n_extended_ = std::max(n_extended_, *max_it + 1);
        }
    }

    groups_.clear();
    for (size_t i = 0; i < con_ids.size(); ++i) {
        const auto& vecid = con_ids[i];
        const auto& values = con_values[i];

        if (vecid.empty() || values.empty() || vecid.size() != values.size())
            continue;

        // 找到原始 ID（组内最小值）
        auto min_it = std::min_element(vecid.begin(), vecid.end());
        int origin_idx = static_cast<int>(std::distance(vecid.begin(), min_it));
        int origin_id = vecid[origin_idx];

        if (origin_id < 0 || origin_id >= n_free_vector)
            continue;

        std::complex<double> origin_coeff = values[origin_idx];
        double oc_re = std::real(origin_coeff);
        double oc_im = std::imag(origin_coeff);

        if (std::abs(oc_re) < 1e-10 || std::abs(oc_im) < 1e-10)
            continue;

        ConstraintGroup group;
        group.origin_id = origin_id;
        group.origin_idx_in_group = origin_idx;

        for (size_t j = 0; j < vecid.size(); ++j) {
            int ext_id = vecid[j];
            if (ext_id < 0 || ext_id >= n_extended_ || j >= values.size())
                continue;

            group.ext_indices.push_back(ext_id);

            if (static_cast<int>(j) == origin_idx) {
                group.real_ratios.push_back(1.0f);
                group.imag_ratios.push_back(1.0f);
            } else {
                std::complex<double> ec = values[j];
                group.real_ratios.push_back(static_cast<float>(std::real(ec) / oc_re));
                group.imag_ratios.push_back(static_cast<float>(std::imag(ec) / oc_im));
            }
        }

        groups_.push_back(group);
    }

    initialized_ = true;
}

inline torch::Tensor Parameters::extendVector(
    const torch::Tensor& free_vector,
    const torch::Device& device) const
{
    TORCH_CHECK(free_vector.is_complex(), "Input vector must be complex");
    TORCH_CHECK(free_vector.dim() == 1, "Input vector must be 1-dimensional");

    const int original_size = free_vector.numel();

    // 实际扩展大小为 max(n_extended_, original_size)
    int ext_size = n_extended_;
    if (ext_size < original_size) ext_size = original_size;

    if (ext_size == original_size) {
        return free_vector.to(device);
    }

    // 创建扩展向量，初始化为零
    auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(device);
    torch::Tensor ext = torch::zeros({ext_size}, options);

    // 复制原始元素
    torch::Tensor indices = torch::arange(0, original_size, torch::kLong).to(device);
    ext.index_copy_(0, indices, free_vector.to(device));

    // 施加约束
    for (const auto& g : groups_) {
        // 获取原始值
        torch::Tensor origin_val = ext[g.origin_id];

        // 分离实部和虚部
        torch::Tensor real_part = (origin_val + torch::conj(origin_val)) / 2.0f;
        torch::Tensor imag_part =
            (origin_val - torch::conj(origin_val)) / (2.0f * c10::complex<float>(0, 1));

        for (size_t j = 0; j < g.ext_indices.size(); ++j) {
            if (static_cast<int>(j) == g.origin_idx_in_group) continue;

            int ext_id = g.ext_indices[j];
            torch::Tensor ext_val =
                g.real_ratios[j] * real_part + c10::complex<float>(0, 1) * g.imag_ratios[j] * imag_part;
            ext[ext_id] = ext_val;
        }
    }

    return ext;
}

inline torch::Tensor Parameters::collapseVectorGrad(
    const torch::Tensor& extended_grad,
    int original_size) const
{
    torch::Device device = extended_grad.device();
    auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(device);
    torch::Tensor grad = torch::zeros({original_size}, options);

    // 复制原始元素的梯度
    if (original_size > 0) {
        grad.copy_(extended_grad.slice(0, 0, original_size));
    }

    if (groups_.empty()) return grad;

    for (const auto& g : groups_) {
        // 收集扩展元素的梯度（不含 origin）
        std::vector<int> ext_idx_vec;
        std::vector<float> rr_vec, ir_vec;
        for (size_t j = 0; j < g.ext_indices.size(); ++j) {
            if (static_cast<int>(j) == g.origin_idx_in_group) continue;
            int eid = g.ext_indices[j];
            if (eid < 0 || eid >= extended_grad.numel()) continue;
            ext_idx_vec.push_back(eid);
            rr_vec.push_back(g.real_ratios[j]);
            ir_vec.push_back(g.imag_ratios[j]);
        }

        if (ext_idx_vec.empty()) continue;

        torch::Tensor ext_idx_t = torch::tensor(ext_idx_vec, torch::kLong).to(device);
        torch::Tensor rr_t = torch::tensor(rr_vec, torch::kFloat).to(device);
        torch::Tensor ir_t = torch::tensor(ir_vec, torch::kFloat).to(device);

        torch::Tensor ext_grads = extended_grad.index_select(0, ext_idx_t);

        // 分离实部和虚部
        torch::Tensor ext_re = (ext_grads + torch::conj(ext_grads)) / 2.0f;
        torch::Tensor ext_im =
            (ext_grads - torch::conj(ext_grads)) / (2.0f * c10::complex<float>(0, 1));

        // 累加贡献到 origin
        torch::Tensor contrib =
            (rr_t * ext_re + c10::complex<float>(0, 1) * ir_t * ext_im).sum();

        grad[g.origin_id] = grad[g.origin_id] + contrib;
    }

    return grad;
}

#endif // PARAMETERS_CUH
