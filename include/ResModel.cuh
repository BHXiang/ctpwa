#ifndef RES_MODEL_CUH
#define RES_MODEL_CUH

#include <cmath>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>

#include <AutoDiff.cuh>

// template <typename T>
// T make_constant(double val, double deriv = 0.0)
// {
//     // 对于算术类型（如double, float, int等），直接转换
//     if constexpr (std::is_arithmetic_v<T>)
//     {
//         return T(val);
//     }
//     else
//     {
//         // 对于AutoDiff等类型，使用(val, deriv)构造函数
//         // 假设类型T有构造函数 T(double, double)
//         return T(val, deriv);
//     }
// }

template <typename T> __host__ __device__ T Bf(int L, const T &q, const T &q0);

template <typename T>
// __host__ __device__ thrust::complex<double> BWR(T &m, T &m0, T &gamma0, int
// L, T &q, T &q0, T *real, T *imag)
__host__ __device__ thrust::complex<double> BWR(T &m, T &m0, T &gamma0, int L,
                                                T &q, T &q0);

__device__ double BlattWeisskopf(int L, double q, double q0);

__host__ __device__ thrust::complex<double> Flatte(double x, double m0,
                                                   double g_pi, double g_K);

// __device__ thrust::complex<double> BWR(double m, double m0, double gamma0,
// int L, double q, double q0);

#endif // RES_MODEL_CUH