#include <Amplitude.cuh>
#include <cuda_runtime.h>
#include <iostream>

__device__ double factorial(float n)
{
    // 处理负数和非法输入
    if (n < 0.0 || isnan(n)) {
        return 0.0;
    }

    // 预计算查表：0.0到11.5的阶乘值
    static const double factorial_table[24] = {
        /* 0.0! */ 1.0,
        /* 0.5! */ 0.886226925452758, // sqrt(pi)/2
        /* 1.0! */ 1.0,
        /* 1.5! */ 1.329340388179137, // 3*sqrt(pi)/4
        /* 2.0! */ 2.0,
        /* 2.5! */ 3.323350970447842, // 15*sqrt(pi)/8
        /* 3.0! */ 6.0,
        /* 3.5! */ 11.63172839656745, // 105*sqrt(pi)/16
        /* 4.0! */ 24.0,
        /* 4.5! */ 52.34277778455352, // 945*sqrt(pi)/32
        /* 5.0! */ 120.0,
        /* 5.5! */ 287.8852778150444, // 10395*sqrt(pi)/64
        /* 6.0! */ 720.0,
        /* 6.5! */ 1871.254305797788, // 135135*sqrt(pi)/128
        /* 7.0! */ 5040.0,
        /* 7.5! */ 14034.40729348341, // 2027025*sqrt(pi)/256
        /* 8.0! */ 40320.0,
        /* 8.5! */ 119292.461994609, // 34459425*sqrt(pi)/512
        /* 9.0! */ 362880.0,
        /* 9.5! */ 1133278.388948785, // 654729075*sqrt(pi)/1024
        /* 10.0! */ 3628800.0,
        /* 10.5! */ 11899423.0839622, // 13749310575*sqrt(pi)/2048
        /* 11.0! */ 39916800.0,
        /* 11.5! */ 136843365.9027724 // 316234143225*sqrt(pi)/4096
    };

    // 判断是否为整数
    double rounded_int = round(n);
    bool is_integer = (fabs(n - rounded_int) < 1e-10);

    // 判断是否为半整数
    double half_rounded = round(2.0 * n) / 2.0;
    bool is_half_integer = (!is_integer && fabs(n - half_rounded) < 1e-10);

    // 处理0-11.5范围内的值（查表法）
    if (n >= 0.0 && n <= 11.5) {
        // 整数情况
        if (is_integer) {
            int int_n = static_cast<int>(rounded_int);
            if (int_n >= 0 && int_n <= 11) {
                return factorial_table[2 * int_n];
            }
        }
        // 半整数情况
        else if (is_half_integer) {
            // 计算半整数索引：n = 2*(x-0.5) = 2x-1
            int n_index = static_cast<int>(round(2.0 * n - 1.0));
            if (n_index >= 0 && n_index <= 22) {
                return factorial_table[n_index + 1]; // 半整数在表中的奇数位置
            }
        }
    }

    // 对于12及以上的整数，使用斯特林近似
    if (is_integer) {
        int int_n = static_cast<int>(rounded_int);
        if (int_n >= 12) {
            // 斯特林公式: n! ≈ sqrt(2πn) * (n/e)^n * (1 + 1/(12n))
            double stirling =
                sqrt(2.0 * M_PI * int_n) * pow(int_n / M_E, int_n);
            // 添加修正项提高精度
            double correction =
                1.0 + 1.0 / (12.0 * int_n) + 1.0 / (288.0 * int_n * int_n);
            return stirling * correction;
        }
    }

    // 对于其他情况（如大于11.5的非整数或半整数），使用Gamma函数
    // 调用原始Gamma函数
    const double g = 7.0;
    const double lanczos_coeff[] = {
        0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
        771.32342877765313,   -176.61502916214059,   12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

    double x = n - 1.0;
    double a = lanczos_coeff[0];
    double t = x + g + 0.5;

    for (int i = 1; i < 9; i++) {
        a += lanczos_coeff[i] / (x + i);
    }

    return sqrt(2 * M_PI) * pow(t, x + 0.5) * exp(-t) * a;
}

__device__ double ClebschGordan(int dim_j1, int m1_x2, int dim_j2, int m2_x2,
                                int dim_J, int M_x2)
{
    float j1 = (dim_j1 - 1) / 2.0f;
    float j2 = (dim_j2 - 1) / 2.0f;
    float J = (dim_J - 1) / 2.0f;
    float m1 = m1_x2 / 2.0f;
    float m2 = m2_x2 / 2.0f;
    float M = M_x2 / 2.0f;

    // 基本检查
    if (fabs(M - (m1 + m2)) > 1e-10)
        return 0.0;
    if (J < fabs(j1 - j2) || J > j1 + j2)
        return 0.0;
    if (fabs(m1) > j1 || fabs(m2) > j2 || fabs(M) > J)
        return 0.0;

    double term1 =
        sqrt((2.0 * J + 1.0) * factorial(j1 + j2 - J) * factorial(j1 - j2 + J) *
             factorial(-j1 + j2 + J) / factorial(j1 + j2 + J + 1.0));

    double term2 =
        sqrt(factorial(J + M) * factorial(J - M) * factorial(j1 - m1) *
             factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2));

    // 求和的界限
    int kmin = (int)ceil(fmax(0.0, fmax(j2 - J - m1, j1 - J + m2)));
    int kmax = (int)floor(fmin(j1 + j2 - J, fmin(j1 - m1, j2 + m2)));

    double sum = 0.0;
    for (int k = kmin; k <= kmax; k++) {
        double denominator = factorial(k) * factorial(j1 + j2 - J - k) *
                             factorial(j1 - m1 - k) * factorial(j2 + m2 - k) *
                             factorial(J - j2 + m1 + k) *
                             factorial(J - j1 - m2 + k);

        if (fabs(denominator) > 1e-15) {
            double sign = (k % 2 == 0) ? 1.0 : -1.0;
            sum += sign / denominator;
        }
    }

    return term1 * term2 * sum;
}

__device__ double associated_legendre_poly(int l, int m, double x)
{
    if (l < 0 || m < 0 || m > l)
        return 0.0;

    if (m == 0) {
        if (l == 0)
            return 1.0;
        if (l == 1)
            return x;
        if (l == 2)
            return 0.5 * (3.0 * x * x - 1.0);
        if (l == 3)
            return 0.5 * (5.0 * x * x * x - 3.0 * x);

        double P_prev2 = 1.0;
        double P_prev1 = x;
        double P_current = 0.0;

        for (int n = 2; n <= l; ++n) {
            P_current =
                ((2.0 * n - 1.0) * x * P_prev1 - (n - 1.0) * P_prev2) / n;
            P_prev2 = P_prev1;
            P_prev1 = P_current;
        }
        return P_current;
    }

    double P_mm = 1.0;
    for (int i = 1; i <= m; ++i) {
        P_mm *= (2 * i - 1) * sqrt(1.0 - x * x);
    }
    P_mm *= pow(-1.0, m);

    if (l == m)
        return P_mm;

    double P_mp1_m = x * (2.0 * m + 1.0) * P_mm;
    if (l == m + 1)
        return P_mp1_m;

    double P_prev2 = P_mm;
    double P_prev1 = P_mp1_m;
    double P_current = 0.0;

    for (int n = m + 2; n <= l; ++n) {
        P_current =
            ((2.0 * n - 1.0) * x * P_prev1 - (n + m - 1.0) * P_prev2) / (n - m);
        P_prev2 = P_prev1;
        P_prev1 = P_current;
    }

    return P_current;
}

__device__ void spherical_harmonic(int l, int m, double theta, double phi,
                                   double *real_part, double *imag_part)
{
    if (l < 0 || abs(m) > l) {
        *real_part = 0.0;
        *imag_part = 0.0;
        return;
    }

    if (l == 0 && m == 0) {
        *real_part = 0.5 * sqrt(1.0 / M_PI);
        *imag_part = 0.0;
        return;
    }

    double x = cos(theta);
    int abs_m = abs(m);
    double P_lm = associated_legendre_poly(l, abs_m, x);

    double norm_factor = sqrt((2.0 * l + 1.0) * factorial(l - abs_m) /
                              (4.0 * M_PI * factorial(l + abs_m)));

    double phase = 1.0;
    if (m < 0) {
        phase = (abs_m % 2 == 0) ? 1.0 : -1.0;
    }

    double amplitude = phase * norm_factor * P_lm;
    // double m_phi = -1 * m * phi;
    double m_phi = m * phi;

    *real_part = amplitude * cos(m_phi);
    *imag_part = amplitude * sin(m_phi);
}

// 质心系振幅元素计算
__device__ thrust::complex<double>
cmf_element(int sgm_x2, int sgm1_x2, int sgm2_x2, int dim_j, int dim_j1,
            int dim_j2, int dim_S, int dL, LorentzVector p1, LorentzVector p2)
{
    // 计算质心系角度
    double m1 = p1.M();
    double m2 = p2.M();
    double p1p2 = p1.Dot(p2);
    double m = sqrt(m1 * m1 + m2 * m2 + 2 * p1p2);
    double lamb = (2 * m * p1.E + m * m + m1 * m1 - m2 * m2) /
                  (2 * m * (m + p1.E + p2.E));
    double qs = sqrt(pow(m, 4) + pow(m1 * m1 - m2 * m2, 2) -
                     2 * m * m * (m1 * m1 + m2 * m2)) /
                (2 * m);
    double nsx = (p1.Px - lamb * (p1.Px + p2.Px)) / qs;
    double nsy = (p1.Py - lamb * (p1.Py + p2.Py)) / qs;
    double nsz = (p1.Pz - lamb * (p1.Pz + p2.Pz)) / qs;
    double theta = acos(nsz);
    double phi = atan2(nsy, nsx);

    // 计算球谐函数 - 修正相位
    double real_part, imag_part;
    spherical_harmonic(dL, (sgm_x2 - sgm1_x2 - sgm2_x2) / 2, theta, -1.0 * phi,
                       &real_part, &imag_part);
    thrust::complex<double> spherical(real_part, imag_part);

    double cg1 = ClebschGordan(dim_j1, sgm1_x2, dim_j2, sgm2_x2, dim_S,
                               sgm1_x2 + sgm2_x2);
    double cg2 = ClebschGordan(dim_S, sgm1_x2 + sgm2_x2, 2 * dL + 1,
                               sgm_x2 - sgm1_x2 - sgm2_x2, dim_j, sgm_x2);

    return (pow(qs, dL) / sqrt(dim_j)) * cg1 * cg2 * spherical;
}

__device__ thrust::complex<double> DFunc(int dim, int m1_x2, int m2_x2,
                                         double beta)
{
    float j = (dim - 1) / 2.0f;
    float m1 = m1_x2 / 2.0f;
    float m2 = m2_x2 / 2.0f;

    if (abs(m1) > j || abs(m2) > j) {
        return thrust::complex<double>(0.0, 0.0);
    }

    thrust::complex<double> sum(0.0, 0.0);
    int k_min = (m1 < m2) ? 0 : m1 - m2;
    int k_max = (m1 > -m2) ? j - m2 : j + m1;

    for (int k = k_min; k <= k_max; ++k) {
        double term1 = factorial(j + m2);
        double term2 = factorial(j - m2);
        double term3 = factorial(j + m1);
        double term4 = factorial(j - m1);

        double numerator = term1 * term2 * term3 * term4;

        double denom1 = factorial(k);
        double denom2 = factorial(j - m2 - k);
        double denom3 = factorial(j + m1 - k);
        double denom4 = factorial(k - m1 + m2);

        double denominator = denom1 * denom2 * denom3 * denom4;

        double comb = sqrt(numerator) / denominator;

        double cos_half = cos(beta / 2.0);
        double sin_half = sin(beta / 2.0);

        double cos_term = pow(cos_half, 2 * j + m1 - m2 - 2 * k);
        double sin_term = pow(sin_half, 2 * k - m1 + m2);

        double phase = pow(-1.0, k - m1 + m2);
        double term = phase * comb * cos_term * sin_term;

        sum += term;
    }

    return sum;
}

__device__ thrust::complex<double> wignerD_element(int dim, int m1_x2,
                                                   int m2_x2, double alpha,
                                                   double beta, double gamma)
{
    thrust::complex<double> d_val = DFunc(dim, m1_x2, m2_x2, beta);

    float m1 = (m1_x2) / 2.0f;
    float m2 = (m2_x2) / 2.0f;
    return thrust::exp(thrust::complex<double>(0.0, m1 * alpha)) * d_val *
           thrust::exp(thrust::complex<double>(0.0, m2 * gamma));
}

__device__ void MassiveTrans(thrust::complex<double> *trans, LorentzVector p,
                             LorentzVector q, int dim,
                             thrust::complex<double> *shared_buf)
{
    int matrix_size = dim * dim;

    // 初始化变换矩阵为单位矩阵
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            trans[i * dim + j] = (i == j) ? thrust::complex<double>(1.0, 0.0)
                                          : thrust::complex<double>(0.0, 0.0);
        }
    }

    double Pp = p.P();
    double Pq = q.P();

    double hat_x = p.Py * q.Pz - p.Pz * q.Py;
    double hat_y = p.Pz * q.Px - p.Px * q.Pz;
    double hat_z = p.Px * q.Py - p.Py * q.Px;

    if (fabs(hat_x) < 1e-10 && fabs(hat_y) < 1e-10 && fabs(hat_z) < 1e-10) {
        return;
    }

    double m1 = p.M();
    double gamma1 = p.E / m1;
    double m2 = q.M();
    double gamma2 = q.E / m2;

    double cos_angle =
        (p.Px * q.Px + p.Py * q.Py + p.Pz * q.Pz) / (-1.0 * Pp * Pq);
    double sin_angle = sqrt(1 - cos_angle * cos_angle);

    double norm = sqrt(hat_x * hat_x + hat_y * hat_y + hat_z * hat_z);
    double nhat_x = hat_x / norm;
    double nhat_y = hat_y / norm;
    double nhat_z = hat_z / norm;

    double theta = acos(nhat_z);
    double phi = atan2(nhat_y, nhat_x);

    double numerator = sin_angle * sqrt((gamma1 - 1.0) * (gamma2 - 1.0));
    double denominator = sqrt((gamma1 + 1.0) * (gamma2 + 1.0)) +
                         cos_angle * sqrt((gamma1 - 1.0) * (gamma2 - 1.0));
    double psi = 2.0 * atan2(numerator, denominator);

    // 使用共享内存存储中间矩阵
    thrust::complex<double> *wd1 = shared_buf;
    thrust::complex<double> *wd2 = wd1 + matrix_size;

    // 计算Wigner D矩阵元素
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            int m1_x2 = dim - 1 - 2 * i;
            int m2_x2 = dim - 1 - 2 * j;
            wd1[i * dim + j] =
                wignerD_element(dim, m1_x2, m2_x2, -phi, -theta, -psi);
            wd2[i * dim + j] =
                wignerD_element(dim, m1_x2, m2_x2, 0.0, theta, phi);
        }
    }

    // 矩阵乘法：trans = wd1 * wd2
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            thrust::complex<double> sum(0.0, 0.0);
            for (int k = 0; k < dim; k++) {
                sum += wd1[i * dim + k] * wd2[k * dim + j];
            }
            trans[i * dim + j] = sum;
        }
    }
}

__device__ void pwa_amp(thrust::complex<double> *amp, LorentzVector p1,
                        int dim_j1, LorentzVector p2, int dim_j2, int dim_j,
                        int dim_S, int dL, thrust::complex<double> *shared_buf)
{

    // 重新计算维度
    int total_amp_size = dim_j * dim_j1 * dim_j2;
    int trans1_size = dim_j1 * dim_j1;
    int trans2_size = dim_j2 * dim_j2;
    int kron_dim = dim_j1 * dim_j2;

    // 内存布局修正
    thrust::complex<double> *shared_amp = shared_buf;
    thrust::complex<double> *shared_trans1 = shared_amp + total_amp_size;
    thrust::complex<double> *shared_trans2 = shared_trans1 + trans1_size;
    thrust::complex<double> *shared_temp = shared_trans2 + trans2_size;

    // MassiveTrans需要的共享内存 (每个需要2倍矩阵大小)
    int max_dim = max(dim_j1, dim_j2);
    // int massive_shared_size = 2 * max_dim * max_dim;
    thrust::complex<double> *massive_shared = shared_temp + total_amp_size;

    // 1. 初始化振幅到共享内存
    for (int i = 0; i < total_amp_size; i++) {
        shared_amp[i] = thrust::complex<double>(0.0, 0.0);
    }

    // 2. 计算原始振幅
    LorentzVector P_total = p1 + p2;

    for (int sgm_idx = 0; sgm_idx < dim_j; sgm_idx++) {
        int sgm_x2 = dim_j - 1 - 2 * sgm_idx;

        for (int sgm1_idx = 0; sgm1_idx < dim_j1; sgm1_idx++) {
            int sgm1_x2 = dim_j1 - 1 - 2 * sgm1_idx;

            for (int sgm2_idx = 0; sgm2_idx < dim_j2; sgm2_idx++) {
                int sgm2_x2 = dim_j2 - 1 - 2 * sgm2_idx;

                // 计算未变换的振幅元素
                thrust::complex<double> untransformed_amp =
                    cmf_element(sgm_x2, sgm1_x2, sgm2_x2, dim_j, dim_j1, dim_j2,
                                dim_S, dL, p1, p2);

                int idx = sgm_idx * kron_dim + sgm1_idx * dim_j2 + sgm2_idx;
                shared_amp[idx] = untransformed_amp;
            }
        }
    }

    // 3. 计算变换矩阵
    LorentzVector zero_vec;
    zero_vec.Px = 0;
    zero_vec.Py = 0;
    zero_vec.Pz = 0;
    zero_vec.E = 0;

    MassiveTrans(shared_trans1, P_total, p1, dim_j1, massive_shared);
    MassiveTrans(shared_trans2, P_total, p2, dim_j2, massive_shared);

    // 4. 将振幅复制到临时存储
    for (int i = 0; i < total_amp_size; i++) {
        shared_temp[i] = shared_amp[i];
    }

    // 5. 矩阵乘法：
    for (int sgm_idx = 0; sgm_idx < dim_j; sgm_idx++) {
        int base_idx = sgm_idx * kron_dim;
        // 第一步：对trans1进行矩阵乘法
        for (int sgm1_new = 0; sgm1_new < dim_j1; sgm1_new++) {
            for (int sgm2_old = 0; sgm2_old < dim_j2; sgm2_old++) {
                thrust::complex<double> sum(0.0, 0.0);

                for (int sgm1_old = 0; sgm1_old < dim_j1; sgm1_old++) {
                    int amp_idx = base_idx + sgm1_old * dim_j2 + sgm2_old;
                    // int trans1_idx = sgm1_new * dim_j1 + sgm1_old;
                    int trans1_idx = sgm1_new + dim_j1 * sgm1_old;

                    sum += shared_trans1[trans1_idx] * shared_temp[amp_idx];
                }

                // 存储中间结果到共享内存的临时位置
                int temp_idx = sgm1_new * dim_j2 + sgm2_old;
                // 可以重用shared_temp的后半部分存储中间结果
                // 这里假设shared_temp有足够空间
                shared_temp[total_amp_size + temp_idx] = sum;
            }
        }
        // 第二步：对trans2进行矩阵乘法
        for (int sgm1_new = 0; sgm1_new < dim_j1; sgm1_new++) {
            for (int sgm2_new = 0; sgm2_new < dim_j2; sgm2_new++) {
                thrust::complex<double> sum(0.0, 0.0);

                for (int sgm2_old = 0; sgm2_old < dim_j2; sgm2_old++) {
                    int temp_idx = sgm1_new * dim_j2 + sgm2_old;
                    // int trans2_idx = sgm2_new * dim_j2 + sgm2_old;
                    int trans2_idx = sgm2_new + dim_j2 * sgm2_old;

                    sum += shared_temp[total_amp_size + temp_idx] *
                           shared_trans2[trans2_idx];
                }

                // 存储最终结果
                int new_idx = base_idx + sgm1_new * dim_j2 + sgm2_new;
                shared_amp[new_idx] = sum;
            }
        }
    }

    // 6. 复制回全局内存
    for (int i = 0; i < total_amp_size; i++) {
        amp[i] = shared_amp[i];
    }
}