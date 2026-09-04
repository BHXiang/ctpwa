#ifndef PARAMETERS_CUH
#define PARAMETERS_CUH

#include <torch/torch.h>
#include "ComplexType.h"

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

// ============================================================
// Coupling matrix data structures (must precede Parameters)
// ============================================================

struct SLKey { int S; int L; };

struct StepCouplingDef {
    std::string label;
    std::string key;
    std::vector<SLKey> sl_list;
    int first_free_idx = -1;
    int n_sl() const { return static_cast<int>(sl_list.size()); }
};

struct AmpStepSLMap {
    int amp_idx;
    std::string chain_key;
    std::vector<std::pair<int,int>> step_sl;
};

struct CouplingMatrixResult {
    int n_amps;
    int n_step_free;
    int n_chain_free;
    int n_free;
    std::vector<StepCouplingDef> steps;
    std::vector<std::string> chain_names;
    std::vector<AmpStepSLMap> amp_map;
    std::vector<int> amp_chain;
    std::vector<std::vector<int>> amp_step_params;
    std::vector<double> amp_chain_ratio;  // 1.0 or trans ratio for folded chains

    void print(std::ostream& os = std::cout) const;
};

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

    // ---------- 统一参数拆分 ----------

    // 将统一 params [real(v_0..n-1), imag(v_0..n-1), θ_0..P-1] 拆分为复数 vector 和 theta
    // params: [2*nFreeVector + nFreeTheta] float64, 在 GPU 上
    // 返回: {vector [nFreeVector] complex, theta [nFreeTheta] float64}
    std::pair<torch::Tensor, torch::Tensor> splitParams(const torch::Tensor& params) const;

    // ---------- 维度查询 ----------

    int nFreeVector() const { return n_free_vector_; }
    int nFreeTheta() const { return n_free_theta_; }
    int nExtended() const { return n_extended_; }
    int nTotalFree() const { return n_free_vector_ + n_free_theta_; }
    int nParams() const { return 2 * n_free_vector_ + n_free_theta_; }
    bool hasConstraints() const { return !groups_.empty(); }
    bool initialized() const { return initialized_; }

    // ---------- theta 维度 ----------

    void setNFreeTheta(int n) { n_free_theta_ = n; }

    // ---------- 约束访问 ----------

    const std::vector<ConstraintGroup>& constraintGroups() const { return groups_; }

    const std::vector<std::vector<int>>& constraintsIndex() const { return con_indices_; }
    const std::vector<std::vector<std::complex<double>>>& constraintsValues() const { return con_values_; }

    // ---------- 耦合矩阵 ----------

    void setCouplingMatrix(const CouplingMatrixResult& cm);
    const CouplingMatrixResult& couplingMatrix() const { return coupling_matrix_; }
    bool hasCouplingMatrix() const { return has_coupling_matrix_; }
    // 主设备（表上传与 coupling kernel 固定在其上，防多卡错卡读写）
    void setPrimaryDevice(int d) { primary_dev_ = d; }

    // 拟合参数 → 旧格式: d_in[Re_p, Im_p, θ] → d_out[Re_v, Im_v, θ]
    void extendCouplingParams(const double* d_in, double* d_out, int ncf, int nt) const;
    // v_ext = coupling(p), d_v: [n_amps] complex
    void applyCouplingMatrix(const double* d_params, ctComplex* d_v_out) const;
    // ∂L/∂v → ∂L/∂p  (d_grad_v: ctComplex [n_amps], d_v: ctComplex [n_amps])
    void transformCouplingGradient(const ctComplex* d_grad_v, const ctComplex* d_v,
                                   const double* d_params, double* d_grad_p) const;
    void freeCouplingData();

    // ---------- Hessian 雅可比变换 ----------

    // 预计算雅可比元素 w[a][j] = v[a] / p[j] 及 v[a] 值
    void precomputeJacobian(const double* d_params);
    // J_full^T · H_ext · J_full + Σ g·∇²v → H_fitting (single kernel)
    // d_params: 拟合参数 [2*ncf+nt] (用于二阶项 w_jk = w_j/p_k)
    // d_g_v: 振幅空间梯度 [2*na] ([Re(∂L/∂v), Im(∂L/∂v)])
    void transformExtendedHessian(const double* d_H_ext,
                                  const double* d_params, const double* d_g_v,
                                  double* d_H_fitting, int na, int ncf, int nt) const;

    // Device pointer accessors (for CouplingFunction backward etc.)
    int* dAmpChain() const { return d_amp_chain_; }
    int* dStepOffsets() const { return d_step_offsets_; }
    int* dStepData() const { return d_step_data_; }

    // ---------- 参数名 ----------

    void setParamNames(const std::vector<std::string>& names) { param_names_ = names; }
    const std::vector<std::string>& paramNames() const { return param_names_; }

    // ---------- gauss_constr（高斯罚项约束）----------
    // 仅作用于 theta 参数；name = "resName_paramName"，idx = slots_ 中的 theta 下标
    void setGaussConstr(const std::map<std::string, int>& name_to_idx,
                        const std::map<std::string, double>& sigma,
                        const std::map<std::string, double>& mu);
    int nGaussConstr() const { return static_cast<int>(theta_name_to_idx_.size()); }
    // 罚项 Σ (x-μ)²/(2σ²)（theta 为 CUDA float64 张量；返回罚值）
    double gaussPenalty(const torch::Tensor& theta) const;
    // 罚项梯度 (x-μ)/σ² 累加到 grad_theta（同设备）
    void gaussPenaltyGrad(const torch::Tensor& theta, torch::Tensor& grad_theta) const;
    // Hessian theta 对角块 += 1/σ²（theta 块行/列起点 = theta_offset）
    void addGaussHessianDiag(torch::Tensor& hess, int theta_offset) const;

private:
    void uploadCouplingData();

    int n_free_vector_ = 0;
    int n_free_theta_ = 0;
    int n_extended_ = 0;
    std::vector<ConstraintGroup> groups_;
    std::vector<std::vector<int>> con_indices_;
    std::vector<std::vector<std::complex<double>>> con_values_;
    CouplingMatrixResult coupling_matrix_;
    bool has_coupling_matrix_ = false;
    std::vector<std::string> param_names_;
    bool initialized_ = false;

    // ---- gauss_constr 约束数据（theta 参数）----
    std::map<std::string, int> theta_name_to_idx_;   // name → theta 下标
    std::map<std::string, double> gauss_constr_sigma_;
    std::map<std::string, double> gauss_constr_mu_;

private:

    int primary_dev_ = 0;   // 主设备（analysis 构造时设置；>=0 时 coupling 系列固定其上）

    // Device data for coupling kernels
    int* d_amp_chain_ = nullptr;
    int* d_step_offsets_ = nullptr;
    int* d_step_data_ = nullptr;
    double* d_amp_chain_ratio_ = nullptr;
    int step_data_len_ = 0;

    // Device data for Hessian Jacobian transform
    double* d_jac_re_ = nullptr;   // Re(w[a][j]) — sparse, same as step_data layout
    double* d_jac_im_ = nullptr;   // Im(w[a][j])
    double* d_jac_p_re_ = nullptr; // Re(p[j]) — [ncf] (for w_jk = w_j / p_k)
    double* d_jac_p_im_ = nullptr; // Im(p[j])
    double* d_v_re_ = nullptr;     // Re(v[a]) — [na]
    double* d_v_im_ = nullptr;     // Im(v[a])
};

// ============================================================
// CouplingMatrixBuilder: chain-level × step-level SL coupling params
// ============================================================

class CouplingMatrixBuilder {
public:
    int addStep(const std::string& key, const std::string& label,
                const std::vector<SLKey>& sl_list) {
        for (size_t si = 0; si < steps_.size(); ++si)
            if (steps_[si].key == key) return static_cast<int>(si);
        StepCouplingDef s;
        s.key = key; s.label = label; s.sl_list = sl_list;
        steps_.push_back(s);
        // Track per-chain step ordering for position-based trans folding
        auto pos = key.find("___");
        if (pos != std::string::npos)
            chain_step_order_[key.substr(0, pos)].push_back(
                static_cast<int>(steps_.size() - 1));
        return static_cast<int>(steps_.size() - 1);
    }

    void addAmplitude(int amp_idx, const std::string& chain_key,
                      const std::vector<std::pair<int,int>>& step_sl) {
        AmpStepSLMap m;
        m.amp_idx = amp_idx; m.chain_key = chain_key; m.step_sl = step_sl;
        amp_map_.push_back(m);
    }

    const std::vector<StepCouplingDef>& getSteps() const { return steps_; }

    CouplingMatrixResult build() const { return buildWithTrans({}, {}); }

    // trans_pairs[i] = {chainA_substr, chainB_substr} with value ratio
    CouplingMatrixResult buildWithTrans(
        const std::vector<std::vector<std::string>>& trans_names,
        const std::vector<std::complex<double>>& trans_values) const;

private:
    std::vector<StepCouplingDef> steps_;
    std::map<std::string, std::vector<int>> chain_step_order_;
    std::vector<AmpStepSLMap> amp_map_;
};

#endif // PARAMETERS_CUH
