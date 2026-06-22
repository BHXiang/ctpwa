#ifndef RES_MODEL_CUH
#define RES_MODEL_CUH

#include <cmath>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>

#include <AutoDiff.cuh>
#include <Resonance.cuh>

// ============================================================================
// ResResult<T>: 传播子返回类型萃取
// T = double              → thrust::complex<double>
// T = Var<double, N, WH>  → ComplexVar<double, N, WH>
// ============================================================================
template <typename T>
struct ResResult {
    using type = thrust::complex<double>;
    __host__ __device__
    static type make(T re, T im) { return type(re, im); }
};

// 特化：Var 类型 → ComplexVar
template <typename T, int N, bool WH>
struct ResResult<Var<T, N, WH>> {
    using type = ComplexVar<T, N, WH>;
    __host__ __device__
    static type make(const Var<T, N, WH>& re, const Var<T, N, WH>& im) {
        return type(re, im);
    }
};

// ============================================================================
// 辅助：Var/标量统一的条件判断
// ============================================================================
template <typename T>
__host__ __device__ bool val_ge_zero(const T& x) {
    if constexpr (std::is_arithmetic_v<T>) return x >= 0.0;
    else return x.val >= 0.0;
}
template <typename T>
__host__ __device__ bool val_gt_zero(const T& x) {
    if constexpr (std::is_arithmetic_v<T>) return x > 0.0;
    else return x.val > 0.0;
}

// ============================================================================
// 模板化传播子函数声明
// ============================================================================

template <typename T> __host__ __device__ T Bf(int L, const T &q, const T &q0, double d);

template <typename T>
__host__ __device__ auto BWR(T& m, T& m0, T& gamma0, int L, T& q, T& q0, double d)
    -> typename ResResult<T>::type;

template <typename T>
__host__ __device__ auto BW(T& m, T& m0, T& gamma0)
    -> typename ResResult<T>::type;

// Flatte: 模板化版本。channel_masses 始终为 double（物理常数，不需要导数）
template <typename T>
__host__ __device__ auto Flatte(T m, T m0,
    int n_channels, const T* g, const double* channel_masses)
    -> typename ResResult<T>::type;

// ============================================================================
// 统一共振态因子计算（__device__，模板化 scalar type）
// ============================================================================

// 计算带 AD 的 breakup momentum q0
template <typename T>
__device__ T computeQ0AD(T m0, T md1, T md2);

// 计算单个 DecayNode 的振幅因子（BWR·Bf / BW / ONE / Flatte·Bf / Bf only）
// params[0]=mass, params[1]=width (BWR/BW) 或 couplings[0] (Flatte)
// channels: Flatte 道质量数组指针（非 Flatte 时为 nullptr）
template <typename T>
__device__ auto computeNodeFactor(
    int L, T mm, T q_ad, T q0_ad,
    const T* params, int param_count,
    ResModelType model_type,
    const double* channels, int n_channels,
    double bf_d
) -> typename ResResult<T>::type;

#endif // RES_MODEL_CUH
