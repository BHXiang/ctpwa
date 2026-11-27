// ============================================================
// PrecCpx.h —— 运行期精度档位基础设施（模板化 float/double 双实例路线）
//
// 背景: ComplexType.h 的 ctComplex 是【编译期】类型（CTPWA_DOUBLE_COMPLEX）。
// 本文件提供运行期档位 PrecMode 的:
//   - 类型映射 PrecTraits<P>: ct(=float2/double2)、real、cbytes、name
//   - 模板化复数算术 helpers（float2/double2 双实例，__host__ __device__），
//     供"float 走 float 计算、double 走 double 计算"的模板 kernel 使用。
//
// 档位语义（config precision）:
//   double → PrecMode::Double（全 double，现状）
//   hybrid → 旧 float-A 混合（A float2 + double 核心；auto 缺省同此）
//   float  → PrecMode::Float（全 float 档，逐批接线中）
// ============================================================
#pragma once

#include <cuda_runtime.h>
#include <type_traits>

enum class PrecMode : int {
    Float = 0,
    Double = 1,
};

// CT → PrecMode（float2→Float, double2→Double）
template <typename CT> struct cpxPrecOf;
template <> struct cpxPrecOf<float2>  { static constexpr PrecMode value = PrecMode::Float; };
template <> struct cpxPrecOf<double2> { static constexpr PrecMode value = PrecMode::Double; };

// ---- 档位类型映射 ----
template <PrecMode P>
struct PrecTraits;

template <>
struct PrecTraits<PrecMode::Float> {
    using ct = float2;                 // 与 cuComplex 布局一致（8B）
    using real = float;
    static constexpr size_t cbytes = 8;
    static constexpr const char* name = "float";
};

template <>
struct PrecTraits<PrecMode::Double> {
    using ct = double2;                // 与 cuDoubleComplex 布局一致（16B）
    using real = double;
    static constexpr size_t cbytes = 16;
    static constexpr const char* name = "double";
};

// ---- 复数构造/算术（CT 由实参推导，重载到 float2/double2）----
template <typename CT>
__host__ __device__ inline CT cpxMake(typename PrecTraits<cpxPrecOf<CT>::value>::real re,
                                      typename PrecTraits<cpxPrecOf<CT>::value>::real im);

template <>
__host__ __device__ inline float2 cpxMake<float2>(float re, float im)
{
    return make_float2(re, im);
}
template <>
__host__ __device__ inline double2 cpxMake<double2>(double re, double im)
{
    return make_double2(re, im);
}

// 由 double 输入构造（host 数据/参考用，窄化到档位 real）
template <typename CT>
__host__ __device__ inline CT cpxFromDouble(double re, double im)
{
    using R = typename PrecTraits<cpxPrecOf<CT>::value>::real;
    return cpxMake<CT>((R)re, (R)im);
}

template <typename CT>
__host__ __device__ inline CT cpxMul(CT a, CT b)
{
    using R = typename PrecTraits<cpxPrecOf<CT>::value>::real;
    return cpxMake<CT>((R)(a.x * b.x - a.y * b.y), (R)(a.x * b.y + a.y * b.x));
}

template <typename CT>
__host__ __device__ inline CT cpxConj(CT a)
{
    using R = typename PrecTraits<cpxPrecOf<CT>::value>::real;
    return cpxMake<CT>(a.x, (R)(-a.y));
}

template <typename CT>
__host__ __device__ inline CT cpxScale(CT a, typename PrecTraits<cpxPrecOf<CT>::value>::real s)
{
    using R = typename PrecTraits<cpxPrecOf<CT>::value>::real;
    return cpxMake<CT>((R)(a.x * s), (R)(a.y * s));
}

template <typename CT>
__host__ __device__ inline typename PrecTraits<cpxPrecOf<CT>::value>::real cpxAbs2(CT a)
{
    using R = typename PrecTraits<cpxPrecOf<CT>::value>::real;
    return (R)(a.x * a.x + a.y * a.y);
}
