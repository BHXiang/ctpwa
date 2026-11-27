// include/FitFractions.h — 分支比（fit fractions）、效率、以及按样本取干涉矩阵
//
// P4：这三个函数（137 + 151 + 43 行）依赖形态完全一样——都是"读配置里有没有某个
// 样本 + 分波边界 + 主设备"，再调三个**已经抽成自由函数**的底层：
//   freeParamsToAmplitudes（本文件用回调）/ computeFileBatchIntegrals / phspMatrixFromGram
//
// 所以它们能一次搬走：底层先搬（第六~七步）的回报就在这里。
#pragma once

#include <torch/extension.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ctpwa {

struct FitFractionCtx {
    // ---- 只读状态 ----
    const std::map<std::string, std::vector<std::string>>* data_files = nullptr;
    const std::vector<int>* n_slvectors = nullptr;
    int primary_dev = 0;

    // ---- 算法回调（这些仍属 analysis：耦合矩阵映射、批扫描积分、B̄ 收缩）----
    std::function<torch::Tensor(const torch::Tensor&)> free_params_to_amplitudes;
    std::function<double(const std::string&, const torch::Tensor&,
                         const std::vector<torch::Tensor>&,
                         const std::vector<torch::Tensor>&,
                         double*, double*, double*,
                         double*, double*, double*, double*, int)> batch_integrals;
    std::function<torch::Tensor(const torch::Tensor&)> phsp_matrix_from_gram;
    // 耦合方向的 Hessian（Jacobian 用）：analysis 仍在用，暂以回调接入
    std::function<torch::Tensor(const torch::Tensor&)> coupling_hessian;
};

// 分支比 [npartials, 2]（值, 误差）。没有 phsp_truth 时返回空张量 [0,2]。
torch::Tensor fitFractions(const FitFractionCtx& ctx, const torch::Tensor& vector,
                           const torch::Tensor& hessian_in);

// 效率 [npartials, 2]。没有 phsp_truth 时返回空张量 [0,2]。
torch::Tensor efficiency(const FitFractionCtx& ctx, const torch::Tensor& vector,
                         const torch::Tensor& hessian_in);

// 按样本取干涉矩阵："phsp" 走常驻 B̄ 收缩；"phsp_truth"/"truth" 走一趟批扫描。
torch::Tensor matrixForSample(const FitFractionCtx& ctx, const torch::Tensor& vector,
                              std::string sample);

}  // namespace ctpwa
