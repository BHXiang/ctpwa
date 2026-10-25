// Backend.cpp — 双后端实现（由 g++ 编译，CUDA 构建下亦然）
//
// 为什么用 .cpp 而不是 .cu：
//   这里的代码全是主机侧（分配/拷贝/BLAS 调用），不需要 nvcc。用 .cpp 编进两种
//   构建可以保证"同一份实现"，避免 CPU/CUDA 两处逻辑漂移。
//
// CUDA 构建下本文件只调用 cuda* / cublas*（不 launch 任何 kernel）。
// CPU 构建下本文件是纯 C++，**不依赖任何外部数值库**。

#include "Backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(CTPWA_HAS_CUDA)
#  include <cuda_runtime.h>
#  include <cublas_v2.h>
#else
#  include <unistd.h>
#endif

namespace ctpwa {

// ============================================================================
// 设备与内存
// ============================================================================

namespace detail {
#if !defined(CTPWA_HAS_CUDA)
int& tid()
{
    static thread_local int v = 0;
    return v;
}
#endif
}  // namespace detail

#if defined(CTPWA_HAS_CUDA)

// ---------------- CUDA 实现 ----------------

void* alloc(std::size_t bytes)
{
    if (bytes == 0)
        return nullptr;
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ctpwa] cudaMalloc(%zu) failed: %s\n", bytes,
                     cudaGetErrorString(err));
        return nullptr;
    }
    return p;
}

void dealloc(void* p)
{
    if (p)
        cudaFree(p);
}

void copy(void* dst, const void* src, std::size_t bytes, Dir dir)
{
    if (bytes == 0)
        return;
    cudaMemcpyKind kind = cudaMemcpyHostToDevice;
    switch (dir) {
        case Dir::H2D: kind = cudaMemcpyHostToDevice; break;
        case Dir::D2H: kind = cudaMemcpyDeviceToHost; break;
        case Dir::D2D: kind = cudaMemcpyDeviceToDevice; break;
    }
    cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
    if (err != cudaSuccess)
        std::fprintf(stderr, "[ctpwa] cudaMemcpy(%zu, dir=%d) failed: %s\n", bytes,
                     (int)dir, cudaGetErrorString(err));
}

void memset(void* p, int value, std::size_t bytes)
{
    if (bytes == 0)
        return;
    cudaMemset(p, value, bytes);
}

void sync() { cudaDeviceSynchronize(); }

bool set_device(int index)
{
    if (index < 0)
        return false;
    return cudaSetDevice(index) == cudaSuccess;
}

int device_count()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 0)
        return 0;
    return n;
}

int current_device()
{
    int d = 0;
    cudaGetDevice(&d);
    return d;
}

bool available() { return device_count() > 0; }

void set_smem_bytes(const void* kernel, std::size_t bytes)
{
    if (!kernel)
        return;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)bytes);
}

const char* backend_name() { return "cuda"; }

std::size_t available_memory()
{
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b;
}

std::size_t total_memory()
{
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return total_b;
}

#else  // ---------------- CPU 实现 ----------------

void* alloc(std::size_t bytes)
{
    if (bytes == 0)
        return nullptr;
    void* p = nullptr;
    // 64B 对齐：与 CUDA 的默认对齐一致，便于将来做 SIMD / 向量化
    if (posix_memalign(&p, 64, bytes) != 0) {
        std::fprintf(stderr, "[ctpwa] alloc(%zu) failed\n", bytes);
        return nullptr;
    }
    return p;
}

void dealloc(void* p)
{
    ::free(p);
}

void copy(void* dst, const void* src, std::size_t bytes, Dir /*dir*/)
{
    if (bytes == 0 || dst == src)
        return;
    // memmove：CPU 后端同一地址空间，允许重叠
    std::memmove(dst, src, bytes);
}

void memset(void* p, int value, std::size_t bytes)
{
    if (bytes == 0)
        return;
    std::memset(p, value, bytes);
}

void sync() {}

bool set_device(int index) { return index == 0; }

int device_count() { return 1; }

int current_device() { return 0; }

bool available() { return true; }

void set_smem_bytes(const void* /*kernel*/, std::size_t /*bytes*/) {}

const char* backend_name() { return "cpu"; }

std::size_t available_memory()
{
#ifdef _SC_AVPHYS_PAGES
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_sz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_sz > 0)
        return (std::size_t)pages * (std::size_t)page_sz;
#endif
    return 0;
}

std::size_t total_memory()
{
#ifdef _SC_PHYS_PAGES
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_sz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_sz > 0)
        return (std::size_t)pages * (std::size_t)page_sz;
#endif
    return 0;
}

#endif  // CTPWA_HAS_CUDA

// ============================================================================
// hw:: — 自实现 BLAS
// ============================================================================
//
// 共同约定（与 cublas 一致）:
//   * 列主序: A(i, j) = A[i + j * lda]
//   * 输出元素之间并行，单个输出元素内部串行 → 结果与线程数无关（可复现）
//
// CUDA 后端转发到 cublas（行为与现状完全一致，P4 迁移调用点后性能不变）；
// CPU 后端为纯 C++ 循环，无第三方依赖。

#if defined(CTPWA_HAS_CUDA)

namespace {

// cublas handle 按设备缓存。cublas handle 绑定设备，多卡下必须一卡一个。
constexpr int kMaxDevices = 16;
cublasHandle_t g_handles[kMaxDevices] = {nullptr};

cublasHandle_t handle()
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices)
        dev = 0;
    if (!g_handles[dev]) {
        if (cublasCreate(&g_handles[dev]) != CUBLAS_STATUS_SUCCESS)
            g_handles[dev] = nullptr;
    }
    return g_handles[dev];
}

inline cublasOperation_t op_of(ctpwa::hw::Op op)
{
    switch (op) {
        case ctpwa::hw::Op::T: return CUBLAS_OP_T;
        case ctpwa::hw::Op::C: return CUBLAS_OP_C;
        default: return CUBLAS_OP_N;
    }
}

}  // namespace

namespace hw {

void gemv(Op op, int m, int n, ctComplex alpha, const ctComplex* A, int lda,
          const ctComplex* x, ctComplex beta, ctComplex* y)
{
    cublasZgemv(handle(), op_of(op), m, n, &alpha, A, lda, x, 1, &beta, y, 1);
}

void gemm(Op op_a, Op op_b, int m, int n, int k,
          ctComplex alpha, const ctComplex* A, int lda,
          const ctComplex* B, int ldb, ctComplex beta, ctComplex* C, int ldc)
{
    cublasZgemm(handle(), op_of(op_a), op_of(op_b), m, n, k, &alpha, A, lda,
                B, ldb, &beta, C, ldc);
}

void gemm_batched(bool trans_a, bool trans_b, int m, int n, int k,
                  ctComplex alpha, const ctComplex* A, int lda, idx_t stride_a,
                  const ctComplex* B, int ldb, idx_t stride_b,
                  ctComplex beta, ctComplex* C, int ldc, idx_t stride_c, int batch)
{
    cublasZgemmStridedBatched(handle(), trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                              trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
                              &alpha, A, lda, (long long)stride_a, B, ldb,
                              (long long)stride_b, &beta, C, ldc,
                              (long long)stride_c, batch);
}

void dgmm(bool left, int m, int n, const ctComplex* A, int lda, const ctComplex* x,
          ctComplex* C, int ldc)
{
    cublasZdgmm(handle(), left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, m, n, A,
                lda, x, 1, C, ldc);
}

void dgmm_real(bool left, int m, int n, const ctComplex* A, int lda,
               const double* x, ctComplex* C, int ldc)
{
    // cublas 没有"实数对角线 × 复数矩阵"的 dgmm，这里转成复数对角线再调。
    // 冷路径（结果写出），临时缓冲开销可接受。
    const int len = left ? m : n;
    ctComplex* d = (ctComplex*)alloc((std::size_t)len * sizeof(ctComplex));
    if (!d) {
        std::fprintf(stderr, "[ctpwa] dgmm_real: alloc failed\n");
        return;
    }
    ctComplex* h = (ctComplex*)std::malloc((std::size_t)len * sizeof(ctComplex));
    for (int i = 0; i < len; ++i)
        h[i] = ctMake(x[i], 0.0);
    copy(d, h, (std::size_t)len * sizeof(ctComplex), Dir::H2D);
    std::free(h);
    dgmm(left, m, n, A, lda, d, C, ldc);
    dealloc(d);
}

void daxpy(idx_t n, double alpha, const double* x, double* y)
{
    cublasDaxpy(handle(), (int)n, &alpha, x, 1, y, 1);
}

ctComplex dotc(idx_t n, const ctComplex* x, const ctComplex* y)
{
    ctComplex out = ctMake(0.0, 0.0);
    cublasZdotc(handle(), (int)n, x, 1, y, 1, &out);
    return out;
}

}  // namespace hw

#else  // ---------------- CPU 实现 ----------------

namespace hw {

namespace {

inline ctComplex cmul(ctComplex a, ctComplex b)
{
    return ctMake(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline ctComplex cadd(ctComplex a, ctComplex b)
{
    return ctMake(a.x + b.x, a.y + b.y);
}

// 列主序取元素
inline ctComplex at(const ctComplex* A, int lda, int i, int j)
{
    return A[(std::ptrdiff_t)i + (std::ptrdiff_t)j * lda];
}

}  // namespace

void gemv(Op op, int m, int n, ctComplex alpha, const ctComplex* A, int lda,
          const ctComplex* x, ctComplex beta, ctComplex* y)
{
    // op(A): m×n。N 时 A 是 m×n；T/C 时 A 是 n×m。
    const bool trans = (op != Op::N);
    const bool conj  = (op == Op::C);
    const int n_out = trans ? n : m;
    const int n_in = trans ? m : n;
#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n_out; ++i) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < n_in; ++j) {
            ctComplex a = trans ? at(A, lda, j, i) : at(A, lda, i, j);
            if (conj) a.y = -a.y;
            const ctComplex xv = x[j];
            re += a.x * xv.x - a.y * xv.y;
            im += a.x * xv.y + a.y * xv.x;
        }
        ctComplex acc = ctMake(re, im);
        acc = cmul(alpha, acc);
        if (beta.x != 0.0 || beta.y != 0.0) {
            ctComplex old = y[i];
            acc = cadd(acc, cmul(beta, old));
        }
        y[i] = acc;
    }
}

void gemm(Op op_a, Op op_b, int m, int n, int k,
          ctComplex alpha, const ctComplex* A, int lda,
          const ctComplex* B, int ldb, ctComplex beta, ctComplex* C, int ldc)
{
    // op(A): m×k, op(B): k×n
    const bool trans_a = (op_a != Op::N), conj_a = (op_a == Op::C);
    const bool trans_b = (op_b != Op::N), conj_b = (op_b == Op::C);
    const int a_rows = trans_a ? k : m, a_cols = trans_a ? m : k;
    const int b_rows = trans_b ? n : k, b_cols = trans_b ? k : n;
    (void)a_rows;
    (void)b_rows;
#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double re = 0.0, im = 0.0;
            for (int p = 0; p < k; ++p) {
                ctComplex a = trans_a ? at(A, lda, p, i) : at(A, lda, i, p);
                ctComplex b = trans_b ? at(B, ldb, j, p) : at(B, ldb, p, j);
                if (conj_a) a.y = -a.y;
                if (conj_b) b.y = -b.y;
                re += a.x * b.x - a.y * b.y;
                im += a.x * b.y + a.y * b.x;
            }
            ctComplex acc = cmul(alpha, ctMake(re, im));
            ctComplex& dst = C[(std::ptrdiff_t)i + (std::ptrdiff_t)j * ldc];
            if (beta.x != 0.0 || beta.y != 0.0)
                acc = cadd(acc, cmul(beta, dst));
            dst = acc;
        }
    }
    (void)a_cols;
    (void)b_cols;
}

void gemm_batched(bool trans_a, bool trans_b, int m, int n, int k,
                  ctComplex alpha, const ctComplex* A, int lda, idx_t stride_a,
                  const ctComplex* B, int ldb, idx_t stride_b,
                  ctComplex beta, ctComplex* C, int ldc, idx_t stride_c, int batch)
{
    for (int b = 0; b < batch; ++b)
        gemm(trans_a ? Op::T : Op::N, trans_b ? Op::T : Op::N, m, n, k, alpha,
             A + (idx_t)b * stride_a, lda, B + (idx_t)b * stride_b, ldb, beta,
             C + (idx_t)b * stride_c, ldc);
}

void dgmm(bool left, int m, int n, const ctComplex* A, int lda, const ctComplex* x,
          ctComplex* C, int ldc)
{
    if (left) {
        // C = diag(x) * A : 只缩放行
#ifdef _OPENMP
#      pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < m; ++i) {
            const ctComplex xi = x[i];
            for (int j = 0; j < n; ++j)
                C[(std::ptrdiff_t)i + (std::ptrdiff_t)j * ldc] =
                    cmul(xi, at(A, lda, i, j));
        }
    } else {
#ifdef _OPENMP
#      pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < n; ++j) {
            const ctComplex xj = x[j];
            for (int i = 0; i < m; ++i)
                C[(std::ptrdiff_t)i + (std::ptrdiff_t)j * ldc] =
                    cmul(at(A, lda, i, j), xj);
        }
    }
}

void dgmm_real(bool left, int m, int n, const ctComplex* A, int lda,
               const double* x, ctComplex* C, int ldc)
{
    if (left) {
#ifdef _OPENMP
#      pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < m; ++i) {
            const double xi = x[i];
            for (int j = 0; j < n; ++j) {
                const ctComplex a = at(A, lda, i, j);
                C[(std::ptrdiff_t)i + (std::ptrdiff_t)j * ldc] =
                    ctMake(a.x * xi, a.y * xi);
            }
        }
    } else {
#ifdef _OPENMP
#      pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < n; ++j) {
            const double xj = x[j];
            for (int i = 0; i < m; ++i) {
                const ctComplex a = at(A, lda, i, j);
                C[(std::ptrdiff_t)i + (std::ptrdiff_t)j * ldc] =
                    ctMake(a.x * xj, a.y * xj);
            }
        }
    }
}

void daxpy(idx_t n, double alpha, const double* x, double* y)
{
    if (alpha == 1.0) {
#ifdef _OPENMP
#      pragma omp parallel for schedule(static)
#endif
        for (idx_t i = 0; i < n; ++i)
            y[i] += x[i];
        return;
    }
#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
    for (idx_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

ctComplex dotc(idx_t n, const ctComplex* x, const ctComplex* y)
{
    // 串行累加：与线程数无关（可复现），且 dot 长度通常很小
    double re = 0.0, im = 0.0;
    for (idx_t i = 0; i < n; ++i) {
        const ctComplex xi = x[i];
        const ctComplex yi = y[i];
        re += xi.x * yi.x + xi.y * yi.y;
        im += xi.x * yi.y - xi.y * yi.x;
    }
    return ctMake(re, im);
}

}  // namespace hw

#endif  // CTPWA_HAS_CUDA

}  // namespace ctpwa
