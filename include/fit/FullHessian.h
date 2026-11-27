// include/FullHessian.h — 统一 Hessian（耦合 + θ 的 (2n+nt)² 矩阵）
//
// P4：`analysis::getHessian` 是 god class 里最大的单体（685 行）。它的 12 个依赖
// 在这二十轮里已经逐个自由化——`computePhspHessian` / `computeQuadraticForm` /
// `reduceHessianWithConstraints` / `computeFactorNLL(F)` / `hessianSegmentContrib` /
// `computeCouplingHessian` 都不再是成员函数，只剩 `freeParamsToAmplitudes` 一个回调。
//
// 这正是 P4 "先搬底层"策略的终点：最后搬的是最大的那个，而它的接口反而最小。
#pragma once

#include "ComplexType.h"

#include <AmpGen.cuh>    // AmpCalc
#include <Parameters.cuh>

#include <torch/extension.h>

#include <functional>
#include <vector>

namespace ctpwa {

struct FullHessianCtx {
    // ---- 布局/设备 ----
    std::vector<ctComplex*>* d_all_amplitudes = nullptr;
    const std::vector<std::vector<int>>* events = nullptr;
    const std::vector<std::vector<int>>* events_offsets = nullptr;
    const std::vector<std::vector<int>>* amp_offsets = nullptr;
    const std::vector<double*>* data_weights = nullptr;
    const std::vector<double*>* bkg_weights = nullptr;
    const std::vector<double*>* phsp_weights = nullptr;
    ctComplex* d_phsp_matrix = nullptr;
    Parameters* params = nullptr;
    int n_amplitudes = 0;
    int n_polar = 0;
    int primary_dev = 0;
    bool float_amps = false;
    bool phsp_freed = false;
    double data_total_weight = 0.0;
    double bkg_integral = 0.0;

    AmpCalc* amp_calc = nullptr;

    // ---- 唯一剩下的算法回调（乘法链映射）----
    std::function<torch::Tensor(const torch::Tensor&)> free_params_to_amplitudes;
};

// 返回统一 Hessian [2n+nt, 2n+nt]（float64，在主设备上）。
torch::Tensor getHessian(FullHessianCtx& ctx, const torch::Tensor& params);

}  // namespace ctpwa
