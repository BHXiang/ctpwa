// include/HessianContrib.h — 单个 (GPU, 样本段) 的数据/本底 Hessian 贡献
//
// P4：`analysis::hessianSegmentContrib`（58 行）只读 8 个成员，且调用的
// computeDataHessianContrib / computeDataHessianContribFast 本来就是 ComputeHessian.cu
// 里的自由函数——所以它不需要任何回调，纯参数化即可搬走。
//
// 这是 getHessian 那一簇里最先能搬的一块：把它的数据/权重布局从类状态改成参数之后，
// 剩下的 getHessian 依赖面会小一些。
#pragma once

#include "ComplexType.h"

#include <vector>

namespace ctpwa {

void hessianSegmentContrib(
    const std::vector<ctComplex*>& d_all_amplitudes,
    const std::vector<std::vector<int>>& events,
    const std::vector<std::vector<int>>& amp_offsets,
    const std::vector<double*>& data_weights,
    const std::vector<double*>& bkg_weights,
    bool float_amps,
    int n_amplitudes,
    int n_polar,
    int gpu, int seg,
    const ctComplex* d_v, double* d_hess,
    int n_amp, bool fast);

}  // namespace ctpwa
