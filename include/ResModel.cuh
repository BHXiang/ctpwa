#ifndef RES_MODEL_CUH
#define RES_MODEL_CUH

#include <cmath>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <Resonance.cuh>
#include <CustomExpr.cuh>  // evalCustomSeg（Custom case 需要）

// ============================================================================
// ResResult<T>: 传播子返回类型萃取（仅 double 实例化，符号微分时代遗留模板）
// ============================================================================
template <typename T>
struct ResResult {
    using type = thrust::complex<double>;
    __host__ __device__
    static type make(T re, T im) { return type(re, im); }
};

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

// 计算 breakup momentum q0（host/device 通用，模板仅 double 实例化）
template <typename T>
__host__ __device__ T computeQ0AD(T m0, T md1, T md2);

// 计算单个 DecayNode 的振幅因子（BWR·Bf / BW / ONE / Flatte·Bf / Bf only）
// params[0]=mass, params[1]=width (BWR/BW) 或 couplings[0] (Flatte)
// channels: Flatte 道质量数组指针（非 Flatte 时为 nullptr）
template <typename T>
__device__ auto computeNodeFactor(
    int L, T mm, T q_ad, T q0_ad,
    const T* params, int param_count,
    ResModelType model_type,
    const double* channels, int n_channels,
    const double* aux, int aux_offset,   // Hist 形状表（与 channels 同辅助段；非 Hist 传 nullptr/0）
    double bf_d, bool has_bf = true
) -> typename ResResult<T>::type;

// ============================================================================
// 模板实现（头文件内，多 TU 共享）
// ============================================================================

// Hist 形状表查表
__device__ inline double lookupHistTable(double m, const double* aux, int off)
{
    if (aux == nullptr || off < 0) return 0.0;
    if (!(m >= 0.0)) return 0.0;   // NaN/负数 → 0
    double m_min = aux[off];
    double m_max = aux[off + 1];
    int n = (int)aux[off + 2];
    const double* vals = aux + off + 3;
    if (n < 2 || m_max <= m_min) return 0.0;
    if (m <= m_min) return vals[0];
    if (m >= m_max) return vals[n - 1];
    double x = (m - m_min) / (m_max - m_min) * n;
    if (x < 0.0) x = 0.0;
    if (x >= (double)(n - 1)) x = (double)(n - 2);
    int i = (int)x;
    double frac = x - i;
    return vals[i] * (1.0 - frac) + vals[i + 1] * frac;
}

// 带 AD 的 breakup momentum q0
template <typename T>
__host__ __device__ T computeQ0AD(T m0, T md1, T md2)
{
    T s_md = md1 + md2;
    T d_md = md1 - md2;
    T m0sq = m0 * m0;
    T q0sq = (m0sq - s_md * s_md) * (m0sq - d_md * d_md) / (T(4.0) * m0sq);
    if (q0sq < 0.0) q0sq = 0.0;
    return T(std::sqrt(q0sq));
}

// 统一共振态因子计算
// L: 顶点 Bf 的逐波轨道角动量（Bf(L, q, q0, d)）
// Lmin: BWR 宽度 Γ 的 L（该节点 SL 列表的最小 L）
template <typename T>
__device__ auto computeNodeFactor(
    int L, int Lmin, T mm, T q_ad, T q0_ad,
    const T* params, int param_count,
    ResModelType model_type,
    const double* channels, int n_channels,
    const double* aux, int aux_offset,
    double bf_d, bool has_bf
) -> typename ResResult<T>::type
{
    switch (model_type) {
        case ResModelType::BWR: {
            T m0 = params[0], g0 = params[1];
            // has_bf=false 时传播子内部宽度也不含 Bf（BWR 的 gamma 含 Bf²）
            // → 传 d=0.0 使 Bf≡1，与 bf_d=0.0 完全等价
            // 宽度 L 用 Lmin，顶点 Bf 用逐波 L
            auto bw = BWR<T>(mm, m0, g0, Lmin, q_ad, q0_ad, has_bf ? bf_d : 0.0);
            // 势垒因子门控: has_bf=false 时传播子不含 Bf
            if (has_bf) {
                auto bf = Bf<T>(L, q_ad, q0_ad, bf_d);
                return ResResult<T>::make(bw.real() * bf, bw.imag() * bf);
            } else {
                return ResResult<T>::make(bw.real(), bw.imag());
            }
        }
        case ResModelType::BW: {
            T m0 = params[0], g0 = params[1];
            return BW<T>(mm, m0, g0);
        }
        case ResModelType::ONE: {
            // ONE = 无传播子（单位因子）; 有 Bf 时为 Bf，否则为 1
            if (has_bf) {
                auto bf = Bf<T>(L, q_ad, q0_ad, bf_d);
                return ResResult<T>::make(bf, T(0.0));
            }
            return ResResult<T>::make(T(1.0), T(0.0));
        }
        case ResModelType::Flatte: {
            T mass = params[0];
            T couplings[4];
            for (int i = 0; i < n_channels && i < 4; ++i)
                couplings[i] = params[1 + i];
            auto fl = Flatte<T>(mm, mass, n_channels, couplings, channels);
            // 势垒因子门控: has_bf=false 时 Flatte 不含 Bf
            if (has_bf) {
                auto bf = Bf<T>(L, q_ad, q0_ad, bf_d);
                return ResResult<T>::make(fl.real() * bf, fl.imag() * bf);
            } else {
                return ResResult<T>::make(fl.real(), fl.imag());
            }
        }
        case ResModelType::Hist: {
            double mval = (double)mm;
            double f = lookupHistTable(mval, aux, aux_offset);
            return ResResult<T>::make(T(f), T(0.0));
        }
        case ResModelType::Custom: {
            // DSL 字节码求值（值段 F；∂F/∂θ 由符号微分 aux 在
            // computeCustomAmpsKernel / resonanceGradientKernelRuntime 中计算）
            double mval = (double)mm, qval = (double)q_ad, q0val = (double)q0_ad;
            int P = (int)aux[aux_offset];
            int seg_off = aux_offset + 2;
            double pvals[3];
            for (int i = 0; i < param_count && i < 3; ++i)
                pvals[i] = (double)params[i];
            double F_re = 0, F_im = 0;
            {
                int n_instr = (int)aux[seg_off];
                double out[2];
                evalCustomSeg(aux + seg_off + 1, n_instr, mval, qval, q0val, L, bf_d,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pvals, out);
                F_re = out[0]; F_im = out[1];
                seg_off += 1 + 3 * n_instr;
            }
            return ResResult<T>::make(T(F_re), T(F_im));
        }
        default:
            return ResResult<T>::make(T(1.0), T(0.0));
    }
}

// Bf 模板化实现
template <typename T> __host__ __device__ T Bf(int L, const T &q, const T &q0, double d)
{
    T z = q * d;
    T z0 = q0 * d;
    switch (L) {
    case 0: return 1.0;
    case 1: return sqrt((1.0 + z0 * z0) / (1.0 + z * z));
    case 2: return sqrt((9.0 + 3.0 * z0 * z0 + z0 * z0 * z0 * z0) /
                        (9.0 + 3.0 * z * z + z * z * z * z));
    case 3: return sqrt((pow(z0, 6) + 6.0 * pow(z0, 4) + 45.0 * z0 * z0 + 225.0) /
                        (pow(z, 6) + 6.0 * pow(z, 4) + 45.0 * z * z + 225.0));
    case 4: return sqrt((pow(z0, 8) + 10.0 * pow(z0, 6) + 135.0 * pow(z0, 4) +
                         1575.0 * z0 * z0 + 11025.0) /
                        (pow(z, 8) + 10.0 * pow(z, 6) + 135.0 * pow(z, 4) +
                         1575.0 * z * z + 11025.0));
    case 5: return sqrt((pow(z0, 10) + 15.0 * pow(z0, 8) + 315.0 * pow(z0, 6) +
                         6300.0 * pow(z0, 4) + 99225.0 * z0 * z0 + 893025.0) /
                        (pow(z, 10) + 15.0 * pow(z, 8) + 315.0 * pow(z, 6) +
                         6300.0 * pow(z, 4) + 99225.0 * z * z + 893025.0));
    case 6: return sqrt((pow(z0, 12) + 21.0 * pow(z0, 10) + 630.0 * pow(z0, 8) +
                         17325.0 * pow(z0, 6) + 363825.0 * pow(z0, 4) +
                         6185025.0 * z0 * z0 + 540326025.0) /
                        (pow(z, 12) + 21.0 * pow(z, 10) + 630.0 * pow(z, 8) +
                         17325.0 * pow(z, 6) + 363825.0 * pow(z, 4) +
                         6185025.0 * z * z + 540326025.0));
    default: return 1.0;
    }
}

template <typename T>
__host__ __device__ auto BWR(T& m, T& m0, T& gamma0, int L, T& q, T& q0, double d)
    -> typename ResResult<T>::type
{
    const T gamma =
        gamma0 * pow(q / q0, 2 * L + 1) * (m0 / m) * pow(Bf<T>(L, q, q0, d), 2);
    T x = m0 * m0 - m * m;
    T y = m0 * gamma;
    T s = x * x + y * y;
    return ResResult<T>::make(x / s, y / s);
}

// ============================================================
// 统一插值（Interp 模型，hist=0 / linear=1 / spline=2, Catmull-Rom）
// aux 格式: [method, N, x_min, dx, re_0..re_{N-1}, im_0..im_{N-1}] (等距)
//           [-(method+1), N, x_0..x_{N-1}, re_0..re_{N-1}, im_0..im_{N-1}] (非等距)
// ============================================================
__device__ inline double interp1d(const double* y, int i, double frac, int method, int N)
{
    switch (method) {
    case 0: return y[i];                                        // hist
    case 1: return y[i] + (y[i+1] - y[i]) * frac;               // linear
    case 2: {                                                   // spline
        double y0 = (i > 0) ? y[i-1] : y[i] - (y[i+1]-y[i]);
        double y1 = y[i], y2 = y[i+1];
        double y3 = (i+2 < N) ? y[i+2] : y[i+1] + (y[i+1]-y[i]);
        double f2 = frac * frac, f3 = f2 * frac;
        return 0.5 * ((2*y1) + (-y0 + y2) * frac +
                      (2*y0 - 5*y1 + 4*y2 - y3) * f2 +
                      (-y0 + 3*y1 - 3*y2 + y3) * f3);
    }
    default: return y[i];
    }
}

__device__ inline void interpEval(const double* tab, double x,
    double& Fr, double& Fi, double* dFr, double* dFi, int P)
{
    int hdr = (int)tab[0];
    int method = (hdr >= 0) ? hdr : -(hdr + 1);
    int N = (int)tab[1];

    int i; double frac;
    const double* re; const double* im;
    if (hdr >= 0) {
        // 等距 bin
        double xmin = tab[2], dx = tab[3];
        re = tab + 4;
        im = tab + 4 + N;
        double pos = (x - xmin) / dx;
        i = (int)pos;
        if (i < 0) { i = 0; frac = 0.0; }
        else if (i >= N - 1) { i = N - 2; frac = 1.0; }
        else { frac = pos - (double)i; }
    } else {
        // 非等距点：二分查找
        const double* xv = tab + 2;
        re = tab + 2 + N;
        im = tab + 2 + 2 * N;
        int lo = 0, hi = N - 1;
        while (lo < hi - 1) {
            int mid = (lo + hi) / 2;
            if (x < xv[mid]) hi = mid; else lo = mid;
        }
        i = lo;
        if (i < 0) { i = 0; frac = 0.0; }
        else if (i >= N - 1) { i = N - 2; frac = 1.0; }
        else { frac = (x - xv[i]) / (xv[i+1] - xv[i]); }
        if (method == 2) method = 0;  // 非均匀 spline 未实现 → 回退 hist
    }
    Fr = interp1d(re, i, frac, method, N);
    Fi = interp1d(im, i, frac, method, N);
    // 梯度：bin 高度为拟合参数时（极少见；Interp 无自由参数 → 正常为 0）
    for (int j = 0; j < P && j < 16; ++j) dFr[j] = dFi[j] = 0.0;
    if (P > 0 && method < 2) {
        if (hdr >= 0 && i < P)
            dFr[i] = (method == 0) ? 1.0 : 1.0 - frac;
        if (hdr >= 0 && i + 1 < P && method == 1)
            dFr[i + 1] = frac;
    }
}

// ============================================================
// Bf 因子对质量参数的一阶/二阶导数辅助（梯度/Hessian kernel 共用）
// O = Π_i Bf(L_i, qq_i, q0_i), q0_i = breakup(m0_i, m1_i, m2_i)
// 子粒子为共振态时 q0 依赖其 m0 参数 → ∂lnBf/∂m = ∂lnBf/∂q0 · ∂q0/∂m
// ============================================================

// ∂q(m,m1,m2)/∂m_which (which=1: m1, which=2: m2)
// qsq ≤ 0 时 q 被钳位为 0（常数）→ 导数为 0
__device__ inline double breakup_dq_dm(int which, double m, double m1, double m2)
{
    double s = m1 + m2, d = m1 - m2;
    double A = m * m - s * s, B = m * m - d * d;
    double qsq = A * B;
    if (qsq <= 0.0) return 0.0;
    double dqsq = (which == 1) ? (-2.0 * s * B - 2.0 * d * A)
                               : (-2.0 * s * B + 2.0 * d * A);
    return dqsq / (4.0 * m * sqrt(qsq));
}

// ∂²q/∂m_which²（q = sqrt(Q)/(2m), Q = A·B）
// d²q = (d²Q·Q − dQ²/2) / (4m·Q^1.5)；Q≤0 时导数为 0
__device__ inline double breakup_d2q_dm2(int which, double m, double m1, double m2)
{
    double s = m1 + m2, d = m1 - m2;
    double A = m * m - s * s, B = m * m - d * d;
    double Q = A * B;
    if (Q <= 0.0) return 0.0;
    double dA = -2.0 * s, dB = (which == 1) ? (-2.0 * d) : (2.0 * d);
    double d2A = -2.0, d2B = (which == 1) ? -2.0 : 2.0;
    double dQ = dA * B + A * dB;
    double d2Q = d2A * B + 2.0 * dA * dB + A * d2B;
    return (d2Q * Q - 0.5 * dQ * dQ) / (4.0 * m * Q * sqrt(Q));
}

// ∂²q/∂m1∂m2（两个子粒子同为同一共振态质量参数时的混合项；
// 与 breakup_d2q_dm2 同公式，∂Q 取 ∂/∂m1 × ∂/∂m2）
__device__ inline double breakup_d2q_dm1dm2(double m, double m1, double m2)
{
    double s = m1 + m2, d = m1 - m2;
    double A = m * m - s * s, B = m * m - d * d;
    double Q = A * B;
    if (Q <= 0.0) return 0.0;
    double dA1 = -2.0 * s, dB1 = -2.0 * d;   // ∂/∂m1
    double dA2 = -2.0 * s, dB2 = +2.0 * d;   // ∂/∂m2
    double dQ1 = dA1 * B + A * dB1;
    double dQ2 = dA2 * B + A * dB2;
    double d2Q = -2.0 * B + dA1 * dB2 + dA2 * dB1 + 2.0 * A;
    return (d2Q * Q - 0.5 * dQ1 * dQ2) / (4.0 * m * Q * sqrt(Q));
}

// ∂ln Bf(L,q,q0,d)/∂q0 = 0.5·N0'(z0)/N0(z0)·d, z0 = q0·d
// ∂²ln Bf/∂q0² = 0.5·d²·(N0''/N0 − (N0'/N0)²)
// N0(z0) 多项式系数与上方 Bf 完全一致
__device__ inline double dlnBf_dq0(int L, double q0, double bf_d)
{
    double z0 = q0 * bf_d;
    double z2 = z0 * z0;
    double n0, dn0;
    switch (L) {
    case 0: n0 = 1.0; dn0 = 0.0; break;
    case 1: n0 = 1.0 + z2; dn0 = 2.0 * z0; break;
    case 2: n0 = 9.0 + z2 * (3.0 + z2); dn0 = z0 * (6.0 + 4.0 * z2); break;
    case 3: n0 = 225.0 + z2 * (45.0 + z2 * (6.0 + z2));
            dn0 = z0 * (90.0 + z2 * (24.0 + 6.0 * z2)); break;
    case 4: n0 = 11025.0 + z2 * (1575.0 + z2 * (135.0 + z2 * (10.0 + z2)));
            dn0 = z0 * (3150.0 + z2 * (540.0 + z2 * (60.0 + 8.0 * z2))); break;
    case 5: n0 = 893025.0 + z2 * (99225.0 + z2 * (6300.0 + z2 * (315.0 + z2 * (15.0 + z2))));
            dn0 = z0 * (198450.0 + z2 * (25200.0 + z2 * (1890.0 + z2 * (120.0 + 10.0 * z2)))); break;
    case 6: n0 = 540326025.0 + z2 * (6185025.0 + z2 * (363825.0 + z2 * (17325.0 + z2 * (630.0 + z2 * (21.0 + z2)))));
            dn0 = z0 * (12370050.0 + z2 * (1455300.0 + z2 * (103950.0 + z2 * (5040.0 + z2 * (210.0 + 12.0 * z2))))); break;
    default: return 0.0;
    }
    return 0.5 * dn0 / n0 * bf_d;
}

__device__ inline double d2lnBf_dq0q0(int L, double q0, double bf_d)
{
    double z0 = q0 * bf_d;
    double z2 = z0 * z0;
    double n0, dn0, d2n0;   // N, N'(z0), N''(z0)
    switch (L) {
    case 0: n0 = 1.0; dn0 = 0.0; d2n0 = 0.0; break;
    case 1: n0 = 1.0 + z2; dn0 = 2.0 * z0; d2n0 = 2.0; break;
    case 2: n0 = 9.0 + z2 * (3.0 + z2); dn0 = z0 * (6.0 + 4.0 * z2);
            d2n0 = 6.0 + 12.0 * z2; break;
    case 3: n0 = 225.0 + z2 * (45.0 + z2 * (6.0 + z2));
            dn0 = z0 * (90.0 + z2 * (24.0 + 6.0 * z2));
            d2n0 = 90.0 + z2 * (72.0 + 30.0 * z2); break;
    case 4: n0 = 11025.0 + z2 * (1575.0 + z2 * (135.0 + z2 * (10.0 + z2)));
            dn0 = z0 * (3150.0 + z2 * (540.0 + z2 * (60.0 + 8.0 * z2)));
            d2n0 = 3150.0 + z2 * (1620.0 + z2 * (300.0 + 56.0 * z2)); break;
    case 5: n0 = 893025.0 + z2 * (99225.0 + z2 * (6300.0 + z2 * (315.0 + z2 * (15.0 + z2))));
            dn0 = z0 * (198450.0 + z2 * (25200.0 + z2 * (1890.0 + z2 * (120.0 + 10.0 * z2))));
            d2n0 = 198450.0 + z2 * (75600.0 + z2 * (9450.0 + z2 * (840.0 + 90.0 * z2))); break;
    case 6: n0 = 540326025.0 + z2 * (6185025.0 + z2 * (363825.0 + z2 * (17325.0 + z2 * (630.0 + z2 * (21.0 + z2)))));
            dn0 = z0 * (12370050.0 + z2 * (1455300.0 + z2 * (103950.0 + z2 * (5040.0 + z2 * (210.0 + 12.0 * z2)))));
            d2n0 = 12370050.0 + z2 * (4365900.0 + z2 * (519750.0 + z2 * (35280.0 + z2 * (1890.0 + 132.0 * z2)))); break;
    default: return 0.0;
    }
    return 0.5 * bf_d * bf_d * (d2n0 / n0 - (dn0 / n0) * (dn0 / n0));
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

template <typename T>
__host__ __device__ void csqrt_real_t(T x, T& out_re, T& out_im) {
    if (x >= 0.0) {
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
    // D = m0² - m² - i·m0·Σ gᵢ·(qᵢ/m)，qᵢ/m = ρᵢ/2
    // （ρᵢ = 2q/m 为下方 f1·f2）→ 虚部项 = (m0/2)·Σ gᵢ·ρᵢ。
    // 耦合常数 gᵢ 含义（宽度 ∝ m0·gᵢ）。
    T s = m * m;
    T real_part = m0 * m0 - s;
    T imag_part = 0.0;
    T i_term_real = 0.0;
    T i_term_imag = 0.0;
    for (int i = 0; i < n_channels; ++i) {
        double m_a = channel_masses[2 * i];
        double m_b = channel_masses[2 * i + 1];
        double sum = m_a + m_b;
        double diff = m_a - m_b;
        T f1_sq = 1.0 - T(sum * sum) / s;
        T f1_re, f1_im;
        csqrt_real_t(f1_sq, f1_re, f1_im);
        T f2_sq = 1.0 - T(diff * diff) / s;
        T factor2 = (f2_sq > 0.0) ? sqrt(f2_sq) : T(0.0);
        T rho_re = f1_re * factor2;
        T rho_imag = f1_im * factor2;
        T gw = (m0 / T(2.0)) * g[i];           // FlatteC 宽度项系数
        i_term_real = i_term_real + gw * rho_re;
        i_term_imag = i_term_imag + gw * rho_imag;
    }
    T den_re = real_part + i_term_imag;
    T den_im = T(0.0) - i_term_real;
    T den_sq = den_re * den_re + den_im * den_im;
    return ResResult<T>::make(den_re / den_sq, (T(0.0) - den_im) / den_sq);
}

#endif // RES_MODEL_CUH
