// include/BackendHelpers.h — 后端无关的小工具（被 analysis 与各抽出模块共用）
//
// 这些 helper 原来只存在于 Analysis.h 内部（cpu/cuda 两条分支各一份），任何搬出去的
// 模块都得复制一遍。P4 期间把它们放到这里，各处 include 同一份。
#pragma once

#include <torch/extension.h>

#include <cuda_runtime.h>

#include <utility>

// ---------------------------------------------------------------------------
// 后端无关的设备 helper：CPU 后端下所有张量都在 CPU 上，CUDA 后端保持原语义。
// ---------------------------------------------------------------------------
#if defined(CTPWA_BACKEND_CPU)
static inline torch::Device ctpwa_dev(int) { return torch::Device(torch::kCPU); }
static inline bool ctpwa_is_cuda_like(const torch::Tensor& t) { return t.device().is_cpu(); }
static inline bool ctpwa_on_primary(const torch::Tensor& t, int) { return t.device().is_cpu(); }
#else
static inline torch::Device ctpwa_dev(int i) { return torch::Device(torch::kCUDA, i); }
static inline bool ctpwa_is_cuda_like(const torch::Tensor& t) { return t.is_cuda(); }
static inline bool ctpwa_on_primary(const torch::Tensor& t, int primary)
{
    return t.device().is_cuda() && t.device().index() == primary;
}
#endif

// RAII：函数中途抛异常时也要还原内部状态/关闭文件。
template <class F>
struct ScopeExit {
    F f;
    ~ScopeExit() { f(); }
};
template <class F>
ScopeExit<F> makeScopeExit(F f) { return ScopeExit<F>{std::move(f)}; }

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA 分配包装：失败时打印设备与前后的显存信息（比裸 TORCH_CHECK 好定位 OOM）
inline void ctpwaCheckedMalloc(void** ptr, size_t bytes, const char* what)
{
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e != cudaSuccess) {
        int dev = 0;
        cudaGetDevice(&dev);
        size_t mf = 0, mt = 0;
        cudaMemGetInfo(&mf, &mt);
        TORCH_CHECK(false, "[ctpwa] cudaMalloc(", what, ") 失败: 需要 ", bytes >> 20,
                    " MiB, GPU ", dev, " 空闲 ", mf >> 20, " / ", mt >> 20, " MiB: ",
                    cudaGetErrorString(e));
    }
}
