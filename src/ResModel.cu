#include <ResModel.cuh>

// 设备端函数实现
__device__ double BlattWeisskopf(int L, double q, double q0)
{
    const double d = 3.0;
    if (q0 <= 0)
        return 1.0; // 防止除以零
    const double z = q * d;
    const double z0 = q0 * d;

    switch (L) {
    case 0:
        return 1.0;
    case 1:
        return sqrt((1.0 + z0 * z0) / (1.0 + z * z));
    case 2:
        return sqrt((9.0 + 3.0 * z0 * z0 + z0 * z0 * z0 * z0) /
                    (9.0 + 3.0 * z * z + z * z * z * z));
    case 3:
        return sqrt((pow(z0, 6) + 6.0 * pow(z0, 4) + 45.0 * z0 * z0 + 225.0) /
                    (pow(z, 6) + 6.0 * pow(z, 4) + 45.0 * z * z + 225.0));
    case 4:
        return sqrt((pow(z0, 8) + 10.0 * pow(z0, 6) + 135.0 * pow(z0, 4) +
                     1575.0 * z0 * z0 + 11025.0) /
                    (pow(z, 8) + 10.0 * pow(z, 6) + 135.0 * pow(z, 4) +
                     1575.0 * z * z + 11025.0));
    case 5:
        return sqrt((pow(z0, 10) + 15.0 * pow(z0, 8) + 315.0 * pow(z0, 6) +
                     6300.0 * pow(z0, 4) + 99225.0 * z0 * z0 + 893025.0) /
                    (pow(z, 10) + 15.0 * pow(z, 8) + 315.0 * pow(z, 6) +
                     6300.0 * pow(z, 4) + 99225.0 * z * z + 893025.0));
    case 6:
        return sqrt((pow(z0, 12) + 21.0 * pow(z0, 10) + 630.0 * pow(z0, 8) +
                     17325.0 * pow(z0, 6) + 363825.0 * pow(z0, 4) +
                     6185025.0 * z0 * z0 + 540326025.0) /
                    (pow(z, 12) + 21.0 * pow(z, 10) + 630.0 * pow(z, 8) +
                     17325.0 * pow(z, 6) + 363825.0 * pow(z, 4) +
                     6185025.0 * z * z + 540326025.0));
    default:
        return 1.0; // 更高角动量返回1.0
    }
}

// __device__ thrust::complex<double> BWR(double m, double m0, double gamma0,
// int L, double q, double q0)
// {
//     // 计算能量依赖的宽度
//     const double gamma = gamma0 * pow(q / q0, 2 * L + 1) * (m0 / m) *
//     pow(BlattWeisskopf(L, q, q0), 2); double x = m0 * m0 - m * m; double y =
//     m0 * gamma; double s = x * x + y * y;

//     return thrust::complex<double>(x / s, y / s);
// }

// 模板化BlattWeisskopf函数，支持double和AutoDiff类型
template <typename T> __host__ __device__ T Bf(int L, const T &q, const T &q0)
{
    // 引入标准数学函数，ADL将找到Dual版本的重载
    // using ::pow;   // 不需要，ADL会自动找到正确的重载
    // using ::sqrt;  // 不需要，ADL会自动找到正确的重载

    const double d = 3.0;

    // printf("BlattWeisskopf called with L=%d, q=%f, q0=%f, d=%f\n", L, q, q0,
    // d);

    // // 辅助函数：获取T类型的值（对AutoDiff取.value，对double直接返回）
    // auto get_value = [](const T &x) -> double
    // {
    //     if constexpr (std::is_same_v<T, AutoDiff>)
    //     // if constexpr ()
    //     {
    //         return x.value;
    //     }
    //     else
    //     {
    //         return x;
    //     }
    // };

    // // 检查q0 <= 0的情况
    // if (get_value(q0) <= 0)
    // {
    //     return make_constant<T>(1.0);
    // }

    T z = q * d;
    T z0 = q0 * d;

    switch (L) {
    case 0:
        return 1.0;
    case 1:
        return sqrt((1.0 + z0 * z0) / (1.0 + z * z));
    case 2:
        return sqrt((9.0 + 3.0 * z0 * z0 + z0 * z0 * z0 * z0) /
                    (9.0 + 3.0 * z * z + z * z * z * z));
    case 3:
        return sqrt((pow(z0, 6) + 6.0 * pow(z0, 4) + 45.0 * z0 * z0 + 225.0) /
                    (pow(z, 6) + 6.0 * pow(z, 4) + 45.0 * z * z + 225.0));
    case 4:
        return sqrt((pow(z0, 8) + 10.0 * pow(z0, 6) + 135.0 * pow(z0, 4) +
                     1575.0 * z0 * z0 + 11025.0) /
                    (pow(z, 8) + 10.0 * pow(z, 6) + 135.0 * pow(z, 4) +
                     1575.0 * z * z + 11025.0));
    case 5:
        return sqrt((pow(z0, 10) + 15.0 * pow(z0, 8) + 315.0 * pow(z0, 6) +
                     6300.0 * pow(z0, 4) + 99225.0 * z0 * z0 + 893025.0) /
                    (pow(z, 10) + 15.0 * pow(z, 8) + 315.0 * pow(z, 6) +
                     6300.0 * pow(z, 4) + 99225.0 * z * z + 893025.0));
    case 6:
        return sqrt((pow(z0, 12) + 21.0 * pow(z0, 10) + 630.0 * pow(z0, 8) +
                     17325.0 * pow(z0, 6) + 363825.0 * pow(z0, 4) +
                     6185025.0 * z0 * z0 + 540326025.0) /
                    (pow(z, 12) + 21.0 * pow(z, 10) + 630.0 * pow(z, 8) +
                     17325.0 * pow(z, 6) + 363825.0 * pow(z, 4) +
                     6185025.0 * z * z + 540326025.0));
    default:
        return 1.0; // 更高角动量返回1.0
    }
}

template <typename T>
// __host__ __device__ thrust::complex<double> BWR(T &m, T &m0, T &gamma0, int
// L, T &q, T &q0, T *real, T *imag)
__host__ __device__ thrust::complex<double> BWR(T &m, T &m0, T &gamma0, int L,
                                                T &q, T &q0)
{
    // using ::pow;

    // 计算能量依赖的宽度
    const T gamma =
        gamma0 * pow(q / q0, 2 * L + 1) * (m0 / m) * pow(Bf<T>(L, q, q0), 2);
    T x = m0 * m0 - m * m;
    T y = m0 * gamma;
    T s = x * x + y * y;

    // if (real != nullptr || imag != nullptr)
    // {
    //     if (real != nullptr)
    //         *real = x / s;
    //     if (imag != nullptr)
    //         *imag = y / s;
    // }

    return thrust::complex<double>(x / s, y / s);
}

__host__ __device__ thrust::complex<double> csqrt_real(double x)
{
    if (x >= 0.0)
        // 实数情况，虚部为0
        return thrust::complex<double>(std::sqrt(x), 0.0);
    else
        // 纯虚数情况，结果为 i * sqrt(-x)
        return thrust::complex<double>(0.0, std::sqrt(-x));
}

__host__ __device__ thrust::complex<double> Flatte(double x, double m0,
                                                   double g_pi, double g_K)
{
    double s = x * x;

    const double m_pi_plus = 0.13957;
    const double m_pi0 = 0.13498;
    const double m_K_plus = 0.49368;
    const double m_K0 = 0.49761;

    // ρπ (实数)
    double sqrt_pi_plus = std::sqrt(1.0 - 4.0 * m_pi_plus * m_pi_plus / s);
    double sqrt_pi0 = std::sqrt(1.0 - 4.0 * m_pi0 * m_pi0 / s);
    double rho_pi = (2.0 / 3.0) * sqrt_pi_plus + (1.0 / 3.0) * sqrt_pi0;

    // ρK (复数)
    thrust::complex<double> sqrt_K_plus =
        csqrt_real(1.0 - 4.0 * m_K_plus * m_K_plus / s);
    thrust::complex<double> sqrt_K0 = csqrt_real(1.0 - 4.0 * m_K0 * m_K0 / s);
    thrust::complex<double> rho_K = 0.5 * sqrt_K_plus + 0.5 * sqrt_K0;

    // 复数分母: m0^2 - s - i*(g_pi*rho_pi + g_K*rho_K)
    double real_part = m0 * m0 - s;
    thrust::complex<double> i_term =
        thrust::complex<double>(0.0, 1.0) *
        (thrust::complex<double>(g_pi * rho_pi, 0.0) +
         thrust::complex<double>(g_K, 0.0) * rho_K);
    thrust::complex<double> denominator =
        thrust::complex<double>(real_part, 0.0) - i_term;

    return thrust::complex<double>(1.0, 0.0) / denominator;
    // return inv.real() * inv.real() + inv.imag() * inv.imag();
}