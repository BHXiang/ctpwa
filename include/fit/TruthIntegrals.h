// include/TruthIntegrals.h — 产生侧（phsp_truth）分波积分（设备 kernel 编排）
//
// P4：`analysis::computeTruthIntegrals` 是 const 成员，读的类成员只有 nSLvectors_ /
// float_amps_ / n_polar_ / n_amplitudes_，其余全靠入参。把这些显式化后即可搬出
// god class（依赖的 upcast 工具已在 MixedPrecision.h 共享）。
#pragma once

#include "ComplexType.h"

#include <torch/extension.h>

#include <vector>

namespace ctpwa {

// 对每个 GPU 计算产生侧分波积分：
//   out_partial    [npartials]     对角
//   out_scattering [npartials²]    非对角
//   out_square     [npartials] 或 nullptr
void computeTruthIntegrals(
    const torch::Tensor& extended_vector,
    const std::vector<ctComplex*>& d_amps,
    const std::vector<int>& ev_per_gpu,
    double* out_partial,
    double* out_scattering,
    double* out_square,
    int npartials,
    const std::vector<ctComplex*>& ws_complex,
    const std::vector<ctComplex*>& ws_matrix,
    const std::vector<ctComplex*>& ws_upcast,
    const std::vector<double*>& d_weights_per_gpu,
    const std::vector<int>& n_slvectors,
    bool float_amps,
    int n_polar,
    int n_amplitudes,
    int primary_dev);

}  // namespace ctpwa
