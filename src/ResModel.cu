#include <ResModel.cuh>

// 设备端函数实现
// __device__ double BlattWeisskopf(int L, double q, double q0)
// {
//     const double d = 3.0;
//     if (q0 <= 0)
//         return 1.0; // 防止除以零
//     const double z = q * d;
//     const double z0 = q0 * d;

//     switch (L) {
//     case 0:
//         return 1.0;
//     case 1:
//         return sqrt((1.0 + z0 * z0) / (1.0 + z * z));
//     case 2:
//         return sqrt((9.0 + 3.0 * z0 * z0 + z0 * z0 * z0 * z0) /
//                     (9.0 + 3.0 * z * z + z * z * z * z));
//     case 3:
//         return sqrt((pow(z0, 6) + 6.0 * pow(z0, 4) + 45.0 * z0 * z0 + 225.0) /
//                     (pow(z, 6) + 6.0 * pow(z, 4) + 45.0 * z * z + 225.0));
//     case 4:
//         return sqrt((pow(z0, 8) + 10.0 * pow(z0, 6) + 135.0 * pow(z0, 4) +
//                      1575.0 * z0 * z0 + 11025.0) /
//                     (pow(z, 8) + 10.0 * pow(z, 6) + 135.0 * pow(z, 4) +
//                      1575.0 * z * z + 11025.0));
//     case 5:
//         return sqrt((pow(z0, 10) + 15.0 * pow(z0, 8) + 315.0 * pow(z0, 6) +
//                      6300.0 * pow(z0, 4) + 99225.0 * z0 * z0 + 893025.0) /
//                     (pow(z, 10) + 15.0 * pow(z, 8) + 315.0 * pow(z, 6) +
//                      6300.0 * pow(z, 4) + 99225.0 * z * z + 893025.0));
//     case 6:
//         return sqrt((pow(z0, 12) + 21.0 * pow(z0, 10) + 630.0 * pow(z0, 8) +
//                      17325.0 * pow(z0, 6) + 363825.0 * pow(z0, 4) +
//                      6185025.0 * z0 * z0 + 540326025.0) /
//                     (pow(z, 12) + 21.0 * pow(z, 10) + 630.0 * pow(z, 8) +
//                      17325.0 * pow(z, 6) + 363825.0 * pow(z, 4) +
//                      6185025.0 * z * z + 540326025.0));
//     default:
//         return 1.0; // 更高角动量返回1.0
//     }
// }

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
template <typename T> __host__ __device__ T Bf(int L, const T &q, const T &q0, double d)
{

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
__host__ __device__ thrust::complex<double> BWR(T& m, T& m0, T& gamma0, int L, T& q, T& q0, double d)
{
    // using ::pow;

    // 计算能量依赖的宽度
    const T gamma =
        gamma0 * pow(q / q0, 2 * L + 1) * (m0 / m) * pow(Bf<T>(L, q, q0, d), 2);
    T x = m0 * m0 - m * m;
    T y = m0 * gamma;
    T s = x * x + y * y;

    return thrust::complex<double>(x / s, y / s);
}

template <typename T>
__host__ __device__ thrust::complex<double> BW(T& m, T& m0, T& gamma0)
{
    // using ::pow;

    // 计算能量依赖的宽度
    T x = m0 * m0 - m * m;
    T y = m0 * gamma0;
    T s = x * x + y * y;

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

__host__ __device__ thrust::complex<double> Flatte(double m, double m0,
    int n_channels, const double* g, const double* channel_masses)
{
    double s = m * m;

    thrust::complex<double> i_term(0.0, 0.0);

    for (int i = 0; i < n_channels; ++i) {
        double m_a = channel_masses[2 * i];
        double m_b = channel_masses[2 * i + 1];
        double sum = m_a + m_b;
        double diff = m_a - m_b;

        // ρ_i(s) = sqrt(1 - (m_a+m_b)²/s) * sqrt(1 - (m_a-m_b)²/s)
        // 第一因子在阈值以下解析延拓为纯虚数
        thrust::complex<double> factor1 =
            csqrt_real(1.0 - sum * sum / s);
        double factor2_sq = 1.0 - diff * diff / s;
        double factor2 = (factor2_sq > 0.0) ? std::sqrt(factor2_sq) : 0.0;
        thrust::complex<double> rho = factor1 * factor2;

        i_term = i_term + thrust::complex<double>(g[i], 0.0) * rho;
    }

    // denominator = m0² - s - i * Σ g_i * ρ_i(s)
    double real_part = m0 * m0 - s;
    thrust::complex<double> denominator =
        thrust::complex<double>(real_part, 0.0)
        - thrust::complex<double>(0.0, 1.0) * i_term;

    return thrust::complex<double>(1.0, 0.0) / denominator;
}