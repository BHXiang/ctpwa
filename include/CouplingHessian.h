// include/CouplingHessian.h — 耦合方向的 Hessian（Jacobian 用）
//
// P4：`analysis::computeCouplingHessian`（137 行）。它的六个依赖里有五个已经自由化
// （computeQuadraticForm / computePhspHessian / reduceHessianWithConstraints /
// hessianSegmentContrib / hessianFastPathEnabled），只剩 `freeParamsToAmplitudes`
// 还需要回调——所以这次搬得很干净，也顺带把 FitFractions 里那个
// `coupling_hessian` 回调去掉。
#pragma once

#include "ComplexType.h"

#include <Parameters.cuh>

#include <torch/extension.h>

#include <functional>
#include <vector>

namespace ctpwa {

struct CouplingHessianCtx {
    const std::vector<ctComplex*>* d_all_amplitudes = nullptr;
    const std::vector<std::vector<int>>* events = nullptr;
    const std::vector<std::vector<int>>* amp_offsets = nullptr;
    const std::vector<double*>* data_weights = nullptr;
    const std::vector<double*>* bkg_weights = nullptr;
    const ctComplex* d_phsp_matrix = nullptr;   // B̄
    const Parameters* params = nullptr;         // 耦合矩阵/约束
    int n_amplitudes = 0;
    int n_polar = 0;
    bool float_amps = false;
    int primary_dev = 0;
    double data_total_weight = 0.0;
    double bkg_integral = 0.0;

    std::function<torch::Tensor(const torch::Tensor&)> free_params_to_amplitudes;
};

// 返回 [2·n_amps, 2·n_amps] 的耦合方向 Hessian（已按 trans 约束约化前）。
torch::Tensor computeCouplingHessian(const CouplingHessianCtx& ctx,
                                     const torch::Tensor& vector);

}  // namespace ctpwa
