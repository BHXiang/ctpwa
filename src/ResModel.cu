#include <ResModel.cuh>
#include <CustomExpr.cuh>

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

// Hist 形状表查表（定义在文件下方）
__device__ inline double lookupHistTable(double m, const double* aux, int off);

template <typename T>
__device__ auto computeNodeFactor(
    int L, T mm, T q_ad, T q0_ad,
    const T* params, int param_count,
    ResModelType model_type,
    const double* channels, int n_channels,
    const double* aux, int aux_offset,
    double bf_d
) -> typename ResResult<T>::type
{
    switch (model_type) {
        case ResModelType::BWR: {
            // params = [mass, width]
            T m0 = params[0], g0 = params[1];
            auto bw = BWR<T>(mm, m0, g0, L, q_ad, q0_ad, bf_d);
            auto bf = Bf<T>(L, q_ad, q0_ad, bf_d);
            // printf("bf = %f\n", bf);
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
        case ResModelType::Hist: {
            // 直方图形状查表（无自由参数 → AD 梯度为 0）
            // aux[offset+0] = m_min, [offset+1] = m_max, [offset+2] = n_bins, 之后为 bin 值
            double mval;
            if constexpr (std::is_arithmetic_v<T>) mval = (double)mm;
            else mval = (double)mm.val;
            double f = lookupHistTable(mval, aux, aux_offset);
            return ResResult<T>::make(T(f), T(0.0));
        }
        case ResModelType::Custom: {
            // DSL 字节码解释: aux = [P, n_seg, seg...]
            // 段 0 = F, 段 1..P = ∂F/∂θ_j, 段 P+1.. = ∂²F/∂θ_j∂θ_k
            // 输出 (re, im); 对 Var 类型组装成 ComplexVar（与现有 AD 一致）
            double mval, qval, q0val;
            if constexpr (std::is_arithmetic_v<T>) {
                mval = (double)mm; qval = (double)q_ad; q0val = (double)q0_ad;
            } else {
                mval = mm.val; qval = q_ad.val; q0val = q0_ad.val;
            }
            int P = (int)aux[aux_offset];
            int n_seg = (int)aux[aux_offset + 1];
            int seg_off = aux_offset + 2;
            // 参数值转 double（Var 类型取 .val）
            double pvals[3];
            for (int i = 0; i < param_count && i < 3; ++i) {
                if constexpr (std::is_arithmetic_v<T>) pvals[i] = (double)params[i];
                else pvals[i] = params[i].val;
            }
            double F_re = 0, F_im = 0;
            // 逐段执行: 段 0 = F
            {
                int n_instr = (int)aux[seg_off];
                double out[2];
                evalCustomSeg(aux + seg_off + 1, n_instr, mval, qval, q0val, L, bf_d,
                    pvals, out);
                F_re = out[0]; F_im = out[1];
                seg_off += 1 + 3 * n_instr;
            }
            if constexpr (std::is_arithmetic_v<T>) {
                return ResResult<T>::make(T(F_re), T(F_im));
            } else {
                // 组装 ComplexVar<double,N>: grad/hess 数组
                using VarT = T;   // Var<double, N, WH>
                constexpr int N = VarN<T>::value;
                VarT re_v(F_re), im_v(F_im);
                // 一阶段 (段 1..P)
                for (int j = 0; j < P && j < N; ++j) {
                    int n_instr = (int)aux[seg_off];
                    double out[2];
                    evalCustomSeg(aux + seg_off + 1, n_instr, mval, qval, q0val, L, bf_d,
                        pvals, out);
                    re_v.grad[j] = out[0];
                    im_v.grad[j] = out[1];
                    seg_off += 1 + 3 * n_instr;
                }
                // 二阶段 (段 P+1.., j ≤ k)
                for (int j = 0; j < P && j < N; ++j)
                    for (int k = j; k < P && k < N; ++k) {
                        int n_instr = (int)aux[seg_off];
                        double out[2];
                        evalCustomSeg(aux + seg_off + 1, n_instr, mval, qval, q0val, L, bf_d,
                            pvals, out);
                        re_v.hess[j][k] = out[0];
                        im_v.hess[j][k] = out[1];
                        if (j != k) {
                            re_v.hess[k][j] = out[0];
                            im_v.hess[k][j] = out[1];
                        }
                        seg_off += 1 + 3 * n_instr;
                    }
                return ResResult<T>::make(re_v, im_v);
            }
        }
        default:
            return ResResult<T>::make(T(1.0), T(0.0));
    }
}

// ============================================================================
// Hist 形状表线性插值查表
// aux[off+0] = m_min, [off+1] = m_max, [off+2] = n_bins, [off+3..] = bin 值
// 边界外: 默认 clip 到端点值
// ============================================================================
__device__ inline double lookupHistTable(double m, const double* aux, int off)
{
    if (aux == nullptr || off < 0) return 0.0;
    if (!(m >= 0.0)) return 0.0;   // NaN/负数 → 0（!(m>=0) 同时拦截 NaN 和负值）
    double m_min = aux[off];
    double m_max = aux[off + 1];
    int n = (int)aux[off + 2];
    const double* vals = aux + off + 3;
    if (n < 2 || m_max <= m_min) return 0.0;
    if (m <= m_min) return vals[0];
    if (m >= m_max) return vals[n - 1];
    double x = (m - m_min) / (m_max - m_min) * n;
    // 防御: clamp 索引（浮点误差/异常值下也不越界）
    if (x < 0.0) x = 0.0;
    if (x >= (double)(n - 1)) x = (double)(n - 2);
    int i = (int)x;
    double frac = x - i;
    return vals[i] * (1.0 - frac) + vals[i + 1] * frac;
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

    // printf("BWR: m=%.3f, m0=%.3f, gamma0=%.3f, L=%d, q=%.3f, q0=%.3f, d=%.3f, gamma=%.3f, x=%.3f, y=%.3f, (%.3f, %.3f)\n", m, m0, gamma0, L, q, q0, d, gamma, x, y, x/s, y/s);

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
