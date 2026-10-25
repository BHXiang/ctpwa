// CudaCompat.h — 非 CUDA 构建下 CUDA 内建类型 / 函数属性的最小替身
//
// 目的:
//   让同一份源码既能被 nvcc 编译（CTPWA_BACKEND_CUDA），也能被 g++ 以纯 C++ 编译
//   （CTPWA_BACKEND_CPU）。只替身"类型与属性"，不替身任何运行时 API——运行时 API
//   由 Backend.h 的 ctpwa::alloc/copy/launch 负责。
//
// 约定:
//   - 本文件在 CUDA 构建下**完全展开为空**（__CUDACC__ 或 CTPWA_BACKEND_CUDA 已定义）。
//   - 替身类型的布局必须与 CUDA 逐位一致（{x,y} 连续），否则 host↔device 的
//     reinterpret_cast 会错乱。目前 CPU 后端不做跨设备拷贝，但仍保持一致性。
//   - 只提供代码库实际用到的类型（float2 / double2 / cuComplex / cuDoubleComplex /
//     dim3），不引入 float4 等未使用类型。

#pragma once

#if defined(__CUDACC__) || defined(CTPWA_BACKEND_CUDA)
#  define CTPWA_HAS_CUDA 1
#endif

#ifndef CTPWA_HAS_CUDA

// ---- 函数属性 ----
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#  ifndef __forceinline__
#    define __forceinline__ inline
#  endif
#  ifndef __noinline__
#    define __noinline__
#  endif
// kernel 声明/定义在 CPU 后端里退化成普通函数。**故意不定义** threadIdx /
// blockIdx / blockDim / gridDim：未移植的 kernel 会在这些名字上编译失败，
// 失败点就是待移植清单（移植后的写法见 CTPWA_KERNEL / CTPWA_TID）。
#  ifndef __global__
#    define __global__
#  endif

// CUDA 头文件会把 isnan / max / min 等带进全局命名空间，CPU 后端的代码沿用了
// 这种写法（如 src/Amplitude.cu 的 isnan / max）。这里补同样的可见性。
#  include <algorithm>
#  include <cmath>
#  include <math.h>
using std::isnan;
using std::isinf;
using std::isfinite;
using std::max;
using std::min;

// ---- 向量类型（与 CUDA 同名、同布局）----
struct float2  { float  x, y; };
struct double2 { double x, y; };

// 与 CUDA 一致：cuComplex / cuDoubleComplex **就是** float2 / double2
// （CUDA 的 cuComplex.h 里也是 typedef）。因此 PrecCpx.h 里按 float2/double2
// 写的 cpxPrecOf<> 特化自动覆盖 ctComplex，无需另加。
using cuComplex       = float2;
using cuDoubleComplex = double2;

inline float2  make_float2(float x, float y)   { float2  v; v.x = x; v.y = y; return v; }
inline double2 make_double2(double x, double y){ double2 v; v.x = x; v.y = y; return v; }

inline cuComplex make_cuComplex(float x, float y)
{
    cuComplex v; v.x = x; v.y = y; return v;
}
inline cuDoubleComplex make_cuDoubleComplex(double x, double y)
{
    cuDoubleComplex v; v.x = x; v.y = y; return v;
}

// ---- 启动配置（仅主机侧用作"工作量"描述；CPU 后端由 CTPWA_LAUNCH 折算成循环上界）----
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

// ---- ctComplex / cuComplex 的算术 ----
// CUDA 的 cuComplex 没有 operator，业务代码一律手写 .x/.y。CPU 后端保留 .x/.y
// （数十处 kernel 代码依赖），但额外补一组算子，让宿主侧"自然写法"也能编过。
inline cuComplex operator+(cuComplex a, cuComplex b) { return {a.x + b.x, a.y + b.y}; }
inline cuComplex operator-(cuComplex a, cuComplex b) { return {a.x - b.x, a.y - b.y}; }
inline cuComplex operator*(cuComplex a, cuComplex b)
{ return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
inline cuComplex operator*(cuComplex a, float s) { return {a.x * s, a.y * s}; }
inline cuComplex operator*(float s, cuComplex a) { return {a.x * s, a.y * s}; }
inline cuComplex operator/(cuComplex a, float s) { return {a.x / s, a.y / s}; }
inline cuComplex operator-(cuComplex a) { return {-a.x, -a.y}; }
inline cuComplex conj(cuComplex a) { return {a.x, -a.y}; }
inline float cnorm(cuComplex a) { return a.x * a.x + a.y * a.y; }
inline cuComplex& operator+=(cuComplex& a, cuComplex b) { a.x += b.x; a.y += b.y; return a; }

inline cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b)
{ return {a.x + b.x, a.y + b.y}; }
inline cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b)
{ return {a.x - b.x, a.y - b.y}; }
inline cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b)
{ return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
inline cuDoubleComplex operator*(cuDoubleComplex a, double s) { return {a.x * s, a.y * s}; }
inline cuDoubleComplex operator*(double s, cuDoubleComplex a) { return {a.x * s, a.y * s}; }
inline cuDoubleComplex operator/(cuDoubleComplex a, double s) { return {a.x / s, a.y / s}; }
inline cuDoubleComplex operator-(cuDoubleComplex a) { return {-a.x, -a.y}; }
inline cuDoubleComplex conj(cuDoubleComplex a) { return {a.x, -a.y}; }
inline double cnorm(cuDoubleComplex a) { return a.x * a.x + a.y * a.y; }
inline cuDoubleComplex& operator+=(cuDoubleComplex& a, cuDoubleComplex b)
{ a.x += b.x; a.y += b.y; return a; }

// ---- atomicAdd ----
// CUDA 的 atomicAdd 在这里退化为 OpenMP atomic，**保留返回 void**（全库调用点都忽略
// 返回值）。注意：原子累加的求和顺序不确定 → 结果与线程数/调度有关，只能用于
// 诊断量（BF、直方图）。NLL 热路径必须用"每线程局部 + 固定顺序合并"（见
// doc/cpu-backend-plan.md §4.2），相关 kernel 在 T3 重写时替换。
inline void atomicAdd(double* address, double val)
{
#ifdef _OPENMP
#  pragma omp atomic
#endif
    *address += val;
}

inline void atomicAdd(float* address, float val)
{
#ifdef _OPENMP
#  pragma omp atomic
#endif
    *address += val;
}

inline void atomicAdd(int* address, int val)
{
#ifdef _OPENMP
#  pragma omp atomic
#endif
    *address += val;
}

#endif  // !CTPWA_HAS_CUDA
