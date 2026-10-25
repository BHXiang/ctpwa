// cpu_shims/cublas_v2.h — cuBLAS 的 CPU 垫片
//
// 只出现在 CTPWA_BACKEND=cpu 构建的 include 路径最前面。
// 转发到 src/Backend.cpp 的 hw:: 自实现 BLAS（零第三方依赖）。
//
// 约定:
//   列主序、incx/incy == 1（全库调用点都是 1；其它值直接报错而不是静默算错）。
//   handle 被忽略——CPU 后端不需要它，但保留参数以匹配现有调用形状。

#pragma once

#include "Backend.h"

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------- 类型
using cublasHandle_t = void*;
using cublasStatus_t = int;

constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;

enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2
};

enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1
};

// ---------------------------------------------------------------- 生命周期
inline cublasStatus_t cublasCreate(cublasHandle_t* handle)
{
    static int dummy = 0;
    *handle = &dummy;
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasDestroy(cublasHandle_t /*handle*/)
{
    return CUBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------- 内部工具
namespace ctpwa_cublas_detail {

inline bool check_inc(const char* fn, int incx, int incy)
{
    if (incx == 1 && incy == 1)
        return true;
    std::fprintf(stderr,
                 "[ctpwa] CPU backend: %s 暂不支持 incx=%d/incy=%d（全库调用点均为 1）\n",
                 fn, incx, incy);
    std::abort();
}

inline bool trans_of(cublasOperation_t op) { return op != CUBLAS_OP_N; }

inline ctpwa::hw::Op op_of(cublasOperation_t op)
{
    switch (op) {
        case CUBLAS_OP_T: return ctpwa::hw::Op::T;
        case CUBLAS_OP_C: return ctpwa::hw::Op::C;   // 共轭转置（Gram 矩阵必须走这条）
        default:          return ctpwa::hw::Op::N;
    }
}

}  // namespace ctpwa_cublas_detail

// ---------------------------------------------------------------- gemv / gemm
inline cublasStatus_t cublasZgemv(cublasHandle_t, cublasOperation_t trans, int m, int n,
                                  const ctComplex* alpha, const ctComplex* A, int lda,
                                  const ctComplex* x, int incx, const ctComplex* beta,
                                  ctComplex* y, int incy)
{
    ctpwa_cublas_detail::check_inc("cublasZgemv", incx, incy);
    ctpwa::hw::gemv(ctpwa_cublas_detail::op_of(trans), m, n, *alpha, A, lda, x,
                    *beta, y);
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasZgemm(cublasHandle_t, cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const ctComplex* alpha, const ctComplex* A, int lda,
                                  const ctComplex* B, int ldb, const ctComplex* beta,
                                  ctComplex* C, int ldc)
{
    ctpwa::hw::gemm(ctpwa_cublas_detail::op_of(transa),
                    ctpwa_cublas_detail::op_of(transb), m, n, k, *alpha, A, lda, B,
                    ldb, *beta, C, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasZgemmStridedBatched(
    cublasHandle_t, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
    int k, const ctComplex* alpha, const ctComplex* A, int lda, long long strideA,
    const ctComplex* B, int ldb, long long strideB, const ctComplex* beta, ctComplex* C,
    int ldc, long long strideC, int batchCount)
{
    ctpwa::hw::gemm_batched(ctpwa_cublas_detail::trans_of(transa),
                            ctpwa_cublas_detail::trans_of(transb), m, n, k, *alpha, A,
                            lda, (ctpwa::hw::idx_t)strideA, B, ldb,
                            (ctpwa::hw::idx_t)strideB, *beta, C, ldc,
                            (ctpwa::hw::idx_t)strideC, batchCount);
    return CUBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------- dgmm
inline cublasStatus_t cublasZdgmm(cublasHandle_t, cublasSideMode_t mode, int m, int n,
                                  const ctComplex* A, int lda, const ctComplex* x,
                                  int incx, ctComplex* C, int ldc)
{
    ctpwa_cublas_detail::check_inc("cublasZdgmm", incx, 1);
    ctpwa::hw::dgmm(mode == CUBLAS_SIDE_LEFT, m, n, A, lda, x, C, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------- axpy / dot
inline cublasStatus_t cublasDaxpy(cublasHandle_t, int n, const double* alpha,
                                  const double* x, int incx, double* y, int incy)
{
    ctpwa_cublas_detail::check_inc("cublasDaxpy", incx, incy);
    ctpwa::hw::daxpy((ctpwa::hw::idx_t)n, *alpha, x, y);
    return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasZdotc(cublasHandle_t, int n, const ctComplex* x, int incx,
                                  const ctComplex* y, int incy, ctComplex* result)
{
    ctpwa_cublas_detail::check_inc("cublasZdotc", incx, incy);
    *result = ctpwa::hw::dotc((ctpwa::hw::idx_t)n, x, y);
    return CUBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------- float 变体
// CPU 后端固定 double 精度（CTPWA_DOUBLE_COMPLEX），float 路径不应被实例化。
// 一旦有人误用，给一个明确的运行期错误而不是链接错误。
#define CTPWA_CPU_NO_FLOAT_BLAS(fn)                                          \
    template <class... Args>                                                 \
    inline cublasStatus_t fn(Args&&...)                                      \
    {                                                                        \
        std::fprintf(stderr, "[ctpwa] CPU backend 不支持 float BLAS: %s\n",  \
                     #fn);                                                   \
        std::abort();                                                        \
    }

CTPWA_CPU_NO_FLOAT_BLAS(cublasCgemv)
CTPWA_CPU_NO_FLOAT_BLAS(cublasCgemm)
CTPWA_CPU_NO_FLOAT_BLAS(cublasCdgmm)
CTPWA_CPU_NO_FLOAT_BLAS(cublasCdotc)

#undef CTPWA_CPU_NO_FLOAT_BLAS
