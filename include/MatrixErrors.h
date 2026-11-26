// include/MatrixErrors.h — 干涉矩阵的统计误差传播（显式输入 + 少量回调）
//
// P4：`analysis::computeMatrixErrors` 有 292 行，读 9 个成员、调 4 个成员函数、
// 还会临时改写 `theta_cur_` 做中心差分。它不能整体搬走（那需要把 analysis 整个搬走），
// 但可以把**输入显式化**：状态走结构体字段，行为走回调。
//
// 这也是 P4 反复用的模板：字段 = 它真正读的状态，回调 = 它真正调用的算法。
// 编译器会逐个字段检查调用方有没有漏；漏了直接编译错，而不是运行时错。
#pragma once

#include <torch/extension.h>

#include <functional>
#include <string>
#include <vector>

namespace ctpwa {

struct MatrixErrorCtx {
    // ---- 只读状态（指向 analysis 持有的数据）----
    const std::vector<std::vector<int>>* events = nullptr;       // 每 GPU 的 phsp 事件数
    const std::vector<double*>* phsp_weights = nullptr;          // 每 GPU phsp 权重（可空 = 等权）
    const std::vector<int>* n_slvectors = nullptr;               // 每分波的振幅数（分块边界）
    const void* d_phsp_matrix = nullptr;                         // B̄：ctComplex*（float 档）
    const void* d_phsp_matrix_double = nullptr;                  // B̄：cuDoubleComplex*（优先）
    int n_amplitudes = 0;
    int n_theta = 0;          // 自由共振态参数个数
    int primary_dev = 0;
    bool has_gen = false;     // 是否有 phsp_truth（除重建侧外的产生侧样本）

    // ---- 行为回调（这些确实属于 analysis 的算法，本轮不搬）----
    std::function<int()> n_vector;                                        // getNVector()
    std::function<torch::Tensor(const torch::Tensor&)> free_params_to_amplitudes;
    std::function<torch::Tensor(const torch::Tensor&)> hessian;           // getHessian(params)
    std::function<void(const torch::Tensor&)> set_theta;                  // 中心差分时改 theta
    std::function<torch::Tensor()> get_theta;                             // 还原用
    // 一趟全样本积分：样本名 + 振幅向量 → 对角/散射矩阵（npartials×npartials）
    std::function<void(const std::string&, const torch::Tensor&,
                       double*, double*, int)> batch_integrals;
    // θ 方向扫描需要扰动振幅（每方向一对 ±）与更多输出块；
    // 形状与 analysis::computeFileBatchIntegrals 一致，便于直接转发。
    std::function<double(const std::string&, const torch::Tensor&,
                         const std::vector<torch::Tensor>&,
                         const std::vector<torch::Tensor>&,
                         double*, double*, double*,
                         double*, double*, double*, double*, int)> batch_integrals_full;
};

// 返回 {rec 误差矩阵, gen 误差矩阵}；任一侧不可用时返回空张量 [0,0]。
std::vector<torch::Tensor> computeMatrixErrors(const MatrixErrorCtx& ctx,
                                               const torch::Tensor& params,
                                               const torch::Tensor& hessian_in);

}  // namespace ctpwa
