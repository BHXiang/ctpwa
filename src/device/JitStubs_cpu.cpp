// JitStubs_cpu.cpp — NVRTC 运行时 JIT 在 CPU 后端的空实现
//
// CPU 构建不编译 src/JITCustom.cu（它拼 CUDA C 源码 → nvrtc → cuModuleLoadData）。
// 但 AmpGen.cu 会调用下列入口，因此这里给出等价语义的实现：
//   jitEnabled()      → false（调用方据此整块跳过 JIT，走已有的解释器回退）
//   jitPrepare*()     → false（表示"未就绪"，调用方自动回退解释器）
//   jitLaunch*()      → 不应被调用（prepare 已返回 false），保险起见空实现
//   jitDestroy()      → 空
//
// 这与 `CTPWA_NO_JIT=1` 在 CUDA 后端上的行为一致：数值走解释器，逐位等价。

#include "Backend.h"      // 定义 CTPWA_HAS_CUDA（决定本文件是否为空）
#include "JITCustom.cuh"

#if !defined(CTPWA_HAS_CUDA)

bool jitCompileBlock(JitBlockState& /*st*/, const std::vector<JitNodeSpec>& /*nodes*/,
                     const std::vector<std::vector<double>>& /*aux_list*/,
                     int /*hessian_target*/)
{
    return false;
}

bool jitEnabled()
{
    return false;
}

bool jitPrepareGrad(JitBlockState& /*st*/, int /*gpu*/, const void* /*d_momenta*/,
                    const void* /*d_mom_tab*/, const double* /*all_params*/,
                    const int* /*h_res_param_off*/, const void* /*d_slComb*/,
                    int /*decayChain_size*/, int /*nEvents*/, int /*nSL*/,
                    int /*nSigma*/)
{
    return false;
}

bool jitPrepareFull(JitBlockState& /*st*/, int /*gpu*/, const void* /*d_momenta*/,
                    const void* /*d_mom_tab*/, const double* /*all_params*/,
                    const int* /*h_res_param_off*/, const void* /*d_slComb*/,
                    int /*decayChain_size*/, int /*nEvents*/, int /*evt_offset*/,
                    int /*nSL*/, int /*nSigma*/)
{
    return false;
}

void jitLaunchGrad(const JitBlockState& /*st*/, int /*gpu*/) {}

void jitLaunchFull(const JitBlockState& /*st*/, int /*gpu*/) {}

void jitDestroy(JitBlockState& /*st*/) {}

#endif  // !CTPWA_HAS_CUDA
