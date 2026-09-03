// ComplexType.h — complex 精度统一抽象层（double 默认 / float 编译时切换）
//
// 设计原则:
//   整个代码库中只有这一个文件包含 #ifdef CTPWA_DOUBLE_COMPLEX。
//   业务代码通过 ctComplex / ctFloat / ctMake 等类型和函数选择精度，
//   不出现 ifdef。
//   精度默认 double（高精度求导/数值；释放 phsp 后显存足够）；
//   float 为显式选择（超大统计量时配 free_phsp_amplitudes 使用）:
//     CTPWA_COMPLEX=float  或  CTPWA_DOUBLE_COMPLEX=0  → float 版
//   编译示例:
//     python3 setup.py build_ext --inplace            # double（默认）
//     CTPWA_COMPLEX=float python3 setup.py build_ext --inplace  # float
//   用户侧通过 config.yml 的 `precision: float|double` 声明请求的精度，
//   初始化时自动检查是否与 .so 编译精度一致（不匹配报清晰错误）。
//
// 使用约定:
//   cuComplex            → ctComplex
//   make_cuComplex(r,i)  → ctMake(r,i)
//   float factor         → ctFloat factor
//   float2               → ctFloat2
//   0.0f / 1.0f          → CTF(0.0) / CTF(1.0)
//   logf(x)              → ctLog(x)
//   fminf / fmaxf        → ctFmin / ctFmax
//   (float)expr          → ctCastFloat(expr)   // double 版不窄化
//   thrust::complex<float> resAmp → ctResAmp resAmp
//   kComplexFloat        → TORCH_COMPLEX
//   float* d_phsp_r      → ctPhspReal* d_phsp_r
//   cublasCgemv 等       → CUBLAS_CGEMV 等宏

#pragma once

#include <cuComplex.h>        // cuComplex / cuDoubleComplex
#include <cublas_v2.h>        // cublasCgemv / cublasZgemv
#include <cuda_fp16.h>        // __shfl_down_sync
#include <cmath>              // log / logf, fmin / fminf, fmax / fmaxf
#include <thrust/complex.h>   // thrust::complex<>
#include <torch/types.h>      // kComplexFloat / kComplexDouble

// ====================================================================
// double 精度（cuDoubleComplex / double2）
// ====================================================================

#ifdef CTPWA_DOUBLE_COMPLEX

using ctComplex = cuDoubleComplex;
inline __host__ __device__ ctComplex ctMake(double re, double im)
{
    return make_cuDoubleComplex(re, im);
}

#define CUBLAS_CGEMV cublasZgemv
#define CUBLAS_CGEMM cublasZgemm
#define CUBLAS_CGEMM_STRIDED_BATCHED cublasZgemmStridedBatched
#define CUBLAS_CDGMM cublasZdgmm

using ctFloat = double;
using ctFloat2 = double2;
#define CTF(x) (x)                // 字面量无需 f 后缀
#define ctLog(x) log(x)
#define ctFmin(x, y) fmin(x, y)
#define ctFmax(x, y) fmax(x, y)
inline __host__ __device__ double ctCastFloat(double d) { return d; }  // 不窄化

// warp 规约: double 用 hi/lo 拆位两路 shfl（保持精度）
inline __device__ double ctShflDown(double v, int off, int width = 32)
{
    unsigned hi = __double2hiint(v);
    unsigned lo = __double2loint(v);
    hi = __shfl_down_sync(0xffffffffu, hi, off, width);
    lo = __shfl_down_sync(0xffffffffu, lo, off, width);
    return __hiloint2double(hi, lo);
}

using ctResAmp = thrust::complex<double>;  // resAmp 与 slamps 同精度
using ctPhspReal = double;                 // quadratic form 输出

constexpr auto TORCH_COMPLEX = torch::kComplexDouble;
constexpr auto TORCH_FLOAT = torch::kFloat64;   // complex 的实/虚部 dtype
using ctC10Complex = c10::complex<double>;   // 与 TORCH_COMPLEX 匹配的 c10 复数
constexpr auto TORCH_COMPLEX_STR = "complex128";
constexpr auto PRECISION_NAME = "double";

// ====================================================================
// float 精度（cuComplex / float2）— 显式选择（CTPWA_COMPLEX=float，超大统计量配 free_phsp_amplitudes）
// ====================================================================

#else

using ctComplex = cuComplex;
inline __host__ __device__ ctComplex ctMake(double re, double im)
{
    return make_cuComplex((float)re, (float)im);
}

#define CUBLAS_CGEMV cublasCgemv
#define CUBLAS_CGEMM cublasCgemm
#define CUBLAS_CGEMM_STRIDED_BATCHED cublasCgemmStridedBatched
#define CUBLAS_CDGMM cublasCdgmm

using ctFloat = float;
using ctFloat2 = float2;
#define CTF(x) x##f
#define ctLog(x) logf(x)
#define ctFmin(x, y) fminf(x, y)
#define ctFmax(x, y) fmaxf(x, y)
inline __host__ __device__ float ctCastFloat(double d) { return (float)d; }  // double→float 窄化

// warp 规约: float 直接 shfl
inline __device__ float ctShflDown(float v, int off, int width = 32)
{
    return __shfl_down_sync(0xffffffffu, v, off, width);
}

using ctResAmp = thrust::complex<float>;   // 窄化点（float 版行为）
using ctPhspReal = float;

constexpr auto TORCH_COMPLEX = torch::kComplexFloat;
constexpr auto TORCH_FLOAT = torch::kFloat;      // complex 的实/虚部 dtype
using ctC10Complex = c10::complex<float>;    // 与 TORCH_COMPLEX 匹配的 c10 复数
constexpr auto TORCH_COMPLEX_STR = "complex64";
constexpr auto PRECISION_NAME = "float";

#endif

// ====================================================================
// 通用 float2 复数辅助（混合精度用: double .so + config precision:float 时
// A/S/w 等内存大户按 float2 存储/计算。不依赖编译精度宏，两分支均可用）
// ====================================================================
inline __host__ __device__ float2 cfMake(float re, float im) { return make_float2(re, im); }
inline __host__ __device__ float2 cfConj(float2 a) { return make_float2(a.x, -a.y); }
inline __host__ __device__ float2 cfMul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline __host__ __device__ float2 cfAdd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ float2 cfScale(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
inline __host__ __device__ double cfAbsSq(float2 a) { return (double)a.x * a.x + (double)a.y * a.y; }
