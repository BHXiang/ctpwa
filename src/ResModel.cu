#include <ResModel.cuh>

// ============================================================================
// 统一共振态因子计算
// ============================================================================

template <typename T>
__device__ T computeQ0AD(T m0, T md1, T md2)
{
    T s_md = md1 + md2;
    T d_md = md1 - md2;
    T m0sq = m0 * m0;
    T q0sq = (m0sq - s_md * s_md) * (m0sq - d_md * d_md) / (T(4.0) * m0sq);
    if constexpr (std::is_arithmetic_v<T>) {
        if (q0sq < 0.0) q0sq = 0.0;
        return T(std::sqrt(q0sq));
    } else {
        q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
        return sqrt(q0sq);
    }
}

template <typename T>
__device__ auto computeNodeFactor(
    int L, T mm, T q_ad, T q0_ad,
    const T* params, int param_count,
    ResModelType model_type,
    const double* channels, int n_channels,
    double bf_d
) -> typename ResResult<T>::type
{
    switch (model_type) {
        case ResModelType::BWR: {
            // params = [mass, width]
            T m0 = params[0], g0 = params[1];
            auto bw = BWR<T>(mm, m0, g0, L, q_ad, q0_ad, bf_d);
            auto bf = Bf<T>(L, q_ad, q0_ad, bf_d);
            if constexpr (std::is_arithmetic_v<T>) {
                return ResResult<T>::make(bw.real() * bf, bw.imag() * bf);
            } else {
                return ResResult<T>::make(bw.real * bf, bw.imag * bf);
            }
        }
        case ResModelType::BW: {
            T m0 = params[0], g0 = params[1];
            return BW<T>(mm, m0, g0);
        }
        case ResModelType::ONE: {
            auto bf = Bf<T>(L, q_ad, q0_ad, bf_d);
            return ResResult<T>::make(bf, T(0.0));
        }
        case ResModelType::Flatte: {
            // params = [mass, coupling0, coupling1, ...]
            T mass = params[0];
            T couplings[4];
            for (int i = 0; i < n_channels && i < 4; ++i)
                couplings[i] = params[1 + i];
            auto fl = Flatte<T>(mm, mass, n_channels, couplings, channels);
            auto bf = Bf<T>(L, q_ad, q0_ad, bf_d);
            if constexpr (std::is_arithmetic_v<T>) {
                return ResResult<T>::make(fl.real() * bf, fl.imag() * bf);
            } else {
                return ResResult<T>::make(fl.real * bf, fl.imag * bf);
            }
        }
        default:
            return ResResult<T>::make(T(1.0), T(0.0));
    }
}

// 模板化BlattWeisskopf函数，支持double和AutoDiff类型
template <typename T> __host__ __device__ T Bf(int L, const T &q, const T &q0, double d)
{
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
__host__ __device__ auto BWR(T& m, T& m0, T& gamma0, int L, T& q, T& q0, double d)
    -> typename ResResult<T>::type
{
    // 计算能量依赖的宽度
    const T gamma =
        gamma0 * pow(q / q0, 2 * L + 1) * (m0 / m) * pow(Bf<T>(L, q, q0, d), 2);
    T x = m0 * m0 - m * m;
    T y = m0 * gamma;
    T s = x * x + y * y;

    return ResResult<T>::make(x / s, y / s);
}

template <typename T>
__host__ __device__ auto BW(T& m, T& m0, T& gamma0)
    -> typename ResResult<T>::type
{
    T x = m0 * m0 - m * m;
    T y = m0 * gamma0;
    T s = x * x + y * y;

    return ResResult<T>::make(x / s, y / s);
}

// ============================================================================
// Flatte 模板化实现
// channel_masses 始终为 double*（物理常数，不需要导数）
// ============================================================================

// 辅助：复平方根 csqrt(x) → (re, im)，对 Var 和 double 统一处理
template <typename T>
__host__ __device__ void csqrt_real_t(T x, T& out_re, T& out_im) {
    if (val_ge_zero(x)) {
        out_re = sqrt(x);
        out_im = 0.0;
    } else {
        out_re = 0.0;
        out_im = sqrt(T(0.0) - x);
    }
}

template <typename T>
__host__ __device__ auto Flatte(T m, T m0,
    int n_channels, const T* g, const double* channel_masses)
    -> typename ResResult<T>::type
{
    T s = m * m;
    T real_part = m0 * m0 - s;         // Re(denominator): m0² - s
    T imag_part = 0.0;                 // Im(denominator): -Σ g_i * Re(ρ_i)

    // 实部额外贡献: i_term.imag = Σ g_i * Im(ρ_i)
    // denominator = (m0²-s, 0) - (0, 1) * i_term
    // 其中 i_term = Σ g_i * ρ_i, ρ_i = factor1 * factor2 (complex × real)
    // den_real = real_part + i_term.imag = real_part + Σ g_i * Im(ρ_i)
    // den_imag = -i_term.real = -Σ g_i * Re(ρ_i)
    T i_term_real = 0.0;              // Re(i_term) = Σ g_i * Re(ρ_i)
    T i_term_imag = 0.0;              // Im(i_term) = Σ g_i * Im(ρ_i)

    for (int i = 0; i < n_channels; ++i) {
        double m_a = channel_masses[2 * i];
        double m_b = channel_masses[2 * i + 1];
        double sum = m_a + m_b;
        double diff = m_a - m_b;

        // ρ_i = factor1 * factor2, factor1 = csqrt(1-sum²/s), factor2 = sqrt(max(0,1-diff²/s))
        T f1_sq = 1.0 - T(sum * sum) / s;
        T f1_re, f1_im;
        csqrt_real_t(f1_sq, f1_re, f1_im);

        T f2_sq = 1.0 - T(diff * diff) / s;
        T factor2 = val_gt_zero(f2_sq) ? sqrt(f2_sq) : T(0.0);

        // ρ_i = (f1_re + i*f1_im) * factor2
        T rho_re = f1_re * factor2;
        T rho_imag = f1_im * factor2;

        // i_term += g[i] * (rho_re + i*rho_imag)
        i_term_real = i_term_real + g[i] * rho_re;
        i_term_imag = i_term_imag + g[i] * rho_imag;
    }

    // denominator = (real_part, 0) - (0, 1) * (i_term_real, i_term_imag)
    //             = (real_part, 0) - (-i_term_imag, i_term_real)
    //             = (real_part + i_term_imag, -i_term_real)
    T den_re = real_part + i_term_imag;
    T den_im = T(0.0) - i_term_real;
    T den_sq = den_re * den_re + den_im * den_im;

    // 1/denominator = conj(denom) / |denom|² = (den_re - i*den_im) / den_sq
    return ResResult<T>::make(den_re / den_sq, (T(0.0) - den_im) / den_sq);
}
