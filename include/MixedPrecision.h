// include/MixedPrecision.h — 混合精度（驻留振幅存 float2）的上转工具
//
// P4：`upcastAmpSegToDouble` 原本是 Analysis.h 里的 `static` 函数，被 truth 积分、
// 矩阵误差、写入器等多条路径使用。搬出去的模块需要它，但"复制一份"会让两边悄悄
// 分叉，所以上移成 inline 共享实现。
//
// 它只依赖 kernel 的**声明**（ComputeHessian.cuh）——定义在 ComputeHessian.cu 里，
// 所以这里放进头文件是安全的（不会多重定义）。
#pragma once

#include "BackendHelpers.h"
#include "ComplexType.h"

#include <ComputeHessian.cuh>   // castF2ToDouble2Kernel 声明

#include <torch/extension.h>

#include <cuda_runtime.h>

// float2 存储的振幅段上转成 double（cuDoubleComplex 语义）。
//   baseF : 按 float2 解释的源段
//   nElem : 元素个数
//   buf   : 可选的复用缓冲（nullptr 则临时 cudaMalloc，调用方负责释放）
// 返回上转后的缓冲；nElem==0 时返回 nullptr。
inline ctComplex* upcastAmpSegToDouble(const ctComplex* baseF, size_t nElem,
                                       ctComplex* buf = nullptr)
{
    if (nElem == 0) return nullptr;
    TORCH_CHECK(!ctIsFloatComplex,
        "upcastAmpSegToDouble 只适用于 double .so 的混合精度模式；"
        "原生 float .so 的 ctComplex 已是 float2，不应再上转");
    ctComplex* dbl = buf;
    if (dbl == nullptr) {
        // 返回缓冲按 cuDoubleComplex 语义使用，必须按 16B/元素分配。
        cudaError_t e = cudaMalloc(&dbl, nElem * sizeof(cuDoubleComplex));
        TORCH_CHECK(e == cudaSuccess, "upcastAmpSegToDouble: cudaMalloc 失败: ",
                    cudaGetErrorString(e));
    }
    int g = (int)((nElem + 255) / 256);
    CTPWA_LAUNCH(castF2ToDouble2Kernel, g, 256, 0, reinterpret_cast<const float2*>(baseF),
        reinterpret_cast<cuDoubleComplex*>(dbl), (int)nElem);
    return dbl;
}
