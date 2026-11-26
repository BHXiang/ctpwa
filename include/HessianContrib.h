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

#include <cstdlib>
#include <iostream>
#include <vector>

namespace ctpwa {

// fast / legacy Hessian 路径选择（P4：从 Analysis.h 上移，供抽出的模块共用）
inline bool hessianFastPathEnabled()
{
#if defined(CTPWA_BACKEND_CPU)
    // CPU 后端只移植了 fast 路径（gnBlockReduce + 加权 Gram）。legacy 路径
    // （perEventHessian + reduceHessian 的 chunk 循环）在 CPU 上尚未验证，
    // 会读到未初始化的 chunk 缓冲（实测 NaN）→ 这里强制走 fast 并明确告知。
    const char* e = getenv("CTPWA_HESS_FAST");
    if (e != nullptr && atoi(e) == 0) {
        std::cerr << "[ctpwa] CPU 后端不支持 CTPWA_HESS_FAST=0（legacy Hessian 路径"
                     "尚未移植），已改用 fast 路径\n";
    }
    return true;
#else
    const char* e = getenv("CTPWA_HESS_FAST");
    return e == nullptr || atoi(e) != 0;
#endif
}

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
