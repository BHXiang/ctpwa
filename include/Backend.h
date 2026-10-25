// Backend.h — CPU / CUDA 双后端的唯一抽象层
//
// 设计目标
// ---------
//   1. 让"同一份 kernel 源码"在两种后端下都能编译：
//        CTPWA_BACKEND_CUDA : nvcc 编译，CTPWA_KERNEL=__global__，CTPWA_LAUNCH 展开成 <<<>>>
//        CTPWA_BACKEND_CPU  : g++  编译，CTPWA_KERNEL=空，   CTPWA_LAUNCH 展开成 OpenMP 循环
//   2. 把 cudaMalloc / cudaMemcpy / cublas 这三类主机侧调用收敛到少数几个入口，
//      避免 720 处 cuda* 调用散落在业务代码里。
//   3. **零第三方数值依赖**：不链接 OpenBLAS / MKL / Eigen，BLAS 形状全部自实现
//      （见 hw:: 命名空间）。理由是发布物要适配机器多，少一个 ABI 就少一类故障。
//
// 使用约定
// ---------
//   * 只有 include/Backend.h、src/Backend.cpp 允许出现 #if CTPWA_HAS_CUDA 的运行时分支；
//     业务代码只用 CTPWA_* 宏与 ctpwa:: / hw:: 接口。
//   * CTPWA_LAUNCH 的参数表达式必须无副作用（CPU 展开后每个工作项都会求值一次）。
//   * CTPWA_TID() 只在"一线程一工作项"的扁平 kernel（T1）里使用；需要 blockIdx.y
//     等二维语义的 kernel 在 CPU 后端应改写成显式循环（见 doc/cpu-backend-plan.md §P1）。
//
// 确定性
// ------
//   hw:: 的每个输出元素都是**串行**点积（并行只发生在输出元素之间），因此结果与
//   线程数无关。归约类（T2/T3）另有固定分块策略，见 doc/cpu-backend-plan.md §4.2。

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "CudaCompat.h"
#include "ComplexType.h"

#if defined(CTPWA_HAS_CUDA)
#  include <cuda_runtime.h>   // cudaLaunchKernel（CTPWA_LAUNCH_P 用）
#endif

// ---- 后端选择 --------------------------------------------------------------
#if defined(CTPWA_HAS_CUDA) && defined(CTPWA_BACKEND_CPU)
#  error "CTPWA_HAS_CUDA 与 CTPWA_BACKEND_CPU 同时定义：请只选一个后端"
#endif
#if defined(CTPWA_BACKEND_CPU) && !defined(CTPWA_HAS_CUDA)
// CPU 后端固定 double 精度：float 档位是为省显存设计的，RAM 上不需要，
// 而且少一个精度分支就少一半 kernel 变体（见 doc/cpu-backend-plan.md §1.3）。
#  ifndef CTPWA_DOUBLE_COMPLEX
#    define CTPWA_DOUBLE_COMPLEX 1
#  endif
#endif
#if !defined(CTPWA_HAS_CUDA)
#  define CTPWA_BACKEND_CPU 1
#endif

// ============================================================================
// kernel 声明 / 线程下标 / 启动
// ============================================================================

#if defined(CTPWA_HAS_CUDA)

#  define CTPWA_KERNEL __global__
#  define CTPWA_HD     __host__ __device__
#  define CTPWA_TID()  (blockIdx.x * blockDim.x + threadIdx.x)
#  define CTPWA_LAUNCH(K, g, b, shm, ...) K<<<g, b, shm>>>(__VA_ARGS__)
// 模板实参含逗号的 kernel（如 kernel<0, 1024>）不能直接作为宏实参——预处理器
// 会在 <> 里的逗号上切断。用 CTPWA_LAUNCH_P((&kernel<0, 1024>), ...) 规避：
// 整个函数指针加括号后成为一个宏实参，两种后端都走"函数指针 + 参数包"的路径。
#  define CTPWA_LAUNCH_P(K, g, b, shm, ...)                                  \
        ::ctpwa::launch_ptr((K), (g), (b), (shm), __VA_ARGS__)
// "一次调用算完"的 kernel（块内协作完成一个逻辑任务，如二次型/全量扫描）：
// CPU 后端只调用一次（kernel 内改成串行循环），CUDA 仍按 grid/block 启动。
#  define CTPWA_LAUNCH_SERIAL(K, g, b, shm, ...) K<<<g, b, shm>>>(__VA_ARGS__)
// CUDA 下与普通启动等价（GPU 侧 atomicAdd 本来就是非确定性顺序，CPU 才需要串行化）
#  define CTPWA_LAUNCH_SEQ(K, g, b, shm, ...) K<<<g, b, shm>>>(__VA_ARGS__)
#  define CTPWA_SMEM_ATTR(K, bytes)                                        \
        ::ctpwa::set_smem_bytes(reinterpret_cast<const void*>(K), (bytes))
#  define CTPWA_SYNC() ::ctpwa::sync()

#else  // ---------------- CPU 后端 ----------------

#  define CTPWA_KERNEL
#  define CTPWA_HD
#  define CTPWA_TID()  (::ctpwa::detail::tid())
#  define CTPWA_LAUNCH(K, g, b, shm, ...)                                  \
        ::ctpwa::launch(::ctpwa::grid_threads((g), (b)),                    \
                        [&](int) { K(__VA_ARGS__); })
#  define CTPWA_LAUNCH_P(K, g, b, shm, ...)                                  \
        ::ctpwa::launch_ptr((K), (g), (b), (shm), __VA_ARGS__)
#  define CTPWA_LAUNCH_SERIAL(K, g, b, shm, ...) K(__VA_ARGS__)
// "顺序遍历工作项"：语义与 CTPWA_LAUNCH 相同，但**按固定顺序单线程执行**。
// 用于内部用 atomicAdd 累加的 kernel（Hessian 的 vθ 块等）：原子竞争消失、
// 累加顺序固定 → 结果与线程数无关，可逐位复现。代价是这部分不并行。
#  define CTPWA_LAUNCH_SEQ(K, g, b, shm, ...)                                \
        ::ctpwa::launch_seq(::ctpwa::grid_threads((g), (b)),                 \
                            [&](int) { K(__VA_ARGS__); })
#  define CTPWA_SMEM_ATTR(K, bytes) ((void)0)
#  define CTPWA_SYNC() ((void)0)

#endif

namespace ctpwa {

// ============================================================================
// 设备与内存
// ============================================================================

enum class Dir { H2D, D2H, D2D };

// 分配/释放设备内存。CPU 后端返回 64B 对齐的普通堆内存。
void* alloc(std::size_t bytes);
void  dealloc(void* p);

// 拷贝。CPU 后端三种方向等价（同一地址空间）。
void  copy(void* dst, const void* src, std::size_t bytes, Dir dir = Dir::H2D);

// 填充字节（等价 cudaMemset）。
void  memset(void* p, int value, std::size_t bytes);

// 等待设备完成。CPU 后端为空操作。
void  sync();

// 设备选择。CPU 后端只认 index == 0。
bool  set_device(int index);
int   device_count();
int   current_device();

// 当前后端是否有可用设备（CUDA 下 = 有 GPU；CPU 下恒为 true）。
bool  available();

// 动态 shared memory 上限设置（cudaFuncSetAttribute 的封装）。CPU 后端为空操作。
void  set_smem_bytes(const void* kernel, std::size_t bytes);

// 后端名字（"cuda" / "cpu"），用于 banner 与 build_info()。
const char* backend_name();

// 可用内存（CUDA: 当前设备空闲显存；CPU: 可用 RAM）。单位 byte。
std::size_t available_memory();

// 总内存（CUDA: 当前设备总显存；CPU: 总 RAM）。单位 byte。
std::size_t total_memory();

// ============================================================================
// 启动（CPU 后端）
// ============================================================================

// ---------------------------------------------------------------------------
// CTPWA_LAUNCH_P 的实现：以"函数指针 + 参数包"形式启动。
// CUDA 侧用 cudaLaunchKernel（参数逐个取地址），CPU 侧展开成 launch() 循环。
// 仅用于模板实参含逗号、无法作为宏实参的 kernel；其余仍走 CTPWA_LAUNCH 的快路径。
// ---------------------------------------------------------------------------
#if defined(CTPWA_HAS_CUDA)

template <class K, class... Args>
inline void launch_ptr(K kernel, dim3 grid, dim3 block, std::size_t shm_bytes,
                       Args... args)
{
    void* argv[] = { const_cast<void*>(static_cast<const void*>(&args))... };
    cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, argv,
                     shm_bytes, nullptr);
}

#endif  // CTPWA_HAS_CUDA

#if !defined(CTPWA_HAS_CUDA)
namespace detail {
// 工作项下标。由 launch() 在每个工作项前写入；线程局部，OpenMP 线程池下安全。
int& tid();
}  // namespace detail
#endif

#if !defined(CTPWA_HAS_CUDA)
// 把 grid/block 折算成"总工作项数"。CPU 后端按线性一维工作项处理。
inline std::size_t grid_threads(dim3 g, dim3 b)
{
    return static_cast<std::size_t>(g.x) * g.y * g.z *
           static_cast<std::size_t>(b.x) * b.y * b.z;
}

// 顺序无关的并行 for：每个工作项独立写自己的输出（element-wise kernel 的 CPU 形态）。
// 不提供 reduction 语义——归约请显式写"每线程局部 + 固定顺序合并"。
template <class F>
inline void launch(std::size_t n, F&& f)
{
    if (n == 0)
        return;
#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        detail::tid() = static_cast<int>(i);
        f(static_cast<int>(i));
    }
}


template <class F, class... Args>
inline void launch_ptr(F kernel, dim3 grid, dim3 block, std::size_t /*shm_bytes*/,
                       Args... args)
{
    launch(grid_threads(grid, block), [&](int) { kernel(args...); });
}

// 顺序执行全部工作项（单线程、固定顺序）。用于带 atomicAdd 的 kernel 以获得
// 逐位可复现性；工作项下标仍通过 detail::tid() 暴露给 CTPWA_TID()。
template <class F>
inline void launch_seq(std::size_t n, F&& f)
{
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        detail::tid() = static_cast<int>(i);
        f(static_cast<int>(i));
    }
}
#endif  // !CTPWA_HAS_CUDA

// ============================================================================
// hw:: — 自实现 BLAS（不依赖任何外部数值库）
// ============================================================================
//
// 约定与 cublas 一致：矩阵**列主序**（column-major），lda 是列主元直径。
// 所有函数都是"输出元素之间并行、单个输出元素内部串行"，因此结果与线程数无关。
// CPU 后端只支持 double（ctComplex = cuDoubleComplex）；float 档位不启用。

namespace hw {

using idx_t = std::ptrdiff_t;

// 矩阵算子（与 cublas 对齐）：N 原样、T 转置、C 共轭转置。
// ⚠️ T 与 C 不能混用：A^H·A 用 C，A^T·A 用 T；早期垫片把 C 当 T，Gram 矩阵少了共轭。
enum class Op { N = 0, T = 1, C = 2 };

// y = alpha * op(A) * x + beta * y
//   trans == false: A 是 m×n, x 长度 n, y 长度 m
//   trans == true : A 是 n×m, x 长度 m, y 长度 n
void gemv(Op op, int m, int n,
          ctComplex alpha, const ctComplex* A, int lda,
          const ctComplex* x,
          ctComplex beta, ctComplex* y);

// 兼容重载：bool 只区分 N/T
inline void gemv(bool trans, int m, int n, ctComplex alpha, const ctComplex* A,
                 int lda, const ctComplex* x, ctComplex beta, ctComplex* y)
{
    gemv(trans ? Op::T : Op::N, m, n, alpha, A, lda, x, beta, y);
}

// C = alpha * op(A) * op(B) + beta * C
//   op(A): m×k, op(B): k×n, C: m×n
void gemm(Op op_a, Op op_b, int m, int n, int k,
          ctComplex alpha, const ctComplex* A, int lda,
          const ctComplex* B, int ldb,
          ctComplex beta, ctComplex* C, int ldc);

inline void gemm(bool trans_a, bool trans_b, int m, int n, int k,
                 ctComplex alpha, const ctComplex* A, int lda,
                 const ctComplex* B, int ldb, ctComplex beta, ctComplex* C, int ldc)
{
    gemm(trans_a ? Op::T : Op::N, trans_b ? Op::T : Op::N, m, n, k, alpha, A, lda, B,
         ldb, beta, C, ldc);
}

// 批量 gemm（stride 以元素计，等价 cublas gemmStridedBatched）
void gemm_batched(bool trans_a, bool trans_b, int m, int n, int k,
                  ctComplex alpha, const ctComplex* A, int lda, idx_t stride_a,
                  const ctComplex* B, int ldb, idx_t stride_b,
                  ctComplex beta, ctComplex* C, int ldc, idx_t stride_c,
                  int batch);

// C = diag(x) * A（left） 或 A * diag(x)（!left），逐元素，等价 cublas?dgmm
void dgmm(bool left, int m, int n,
          const ctComplex* A, int lda,
          const ctComplex* x,
          ctComplex* C, int ldc);

// C = diag(x) * A（x 为实数对角线）
void dgmm_real(bool left, int m, int n,
               const ctComplex* A, int lda,
               const double* x,
               ctComplex* C, int ldc);

// y = alpha * x + y（实数）
void daxpy(idx_t n, double alpha, const double* x, double* y);

// 共轭点积 sum_i conj(x_i) * y_i（等价 cublas?dotc）
ctComplex dotc(idx_t n, const ctComplex* x, const ctComplex* y);

}  // namespace hw
}  // namespace ctpwa

namespace ctpwa {
// CPU 后端尚未移植的 kernel 路径：明确失败（不静默返回垃圾）。
// 用在块内协作/归约类 kernel 的 #else 分支里——那里可能位于 OpenMP 并行域，
// 抛异常会 std::terminate，因此用 abort 并打印清楚的位置。
[[noreturn]] inline void not_implemented(const char* what)
{
    std::fprintf(stderr,
                 "[ctpwa] CPU 后端尚未移植: %s\n"
                 "        见 doc/cpu-backend-plan.md 的 P3 清单。\n", what);
    std::abort();
}
}  // namespace ctpwa
