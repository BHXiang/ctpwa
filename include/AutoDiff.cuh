#ifndef AUTODIFF_CUH
#define AUTODIFF_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <iostream>

// ============================================================================
// 第一部分：实数自动微分 Var<T,N> 模板
// ============================================================================

template <typename T, int N>
struct Var {
    T val;                     // 函数值
    T grad[N];                 // 一阶偏导
    T hess[N][N];              // 二阶偏导（Hessian 矩阵）

    // 默认构造（常数值，导数为零）
    __host__ __device__
        Var(T v = 0.0) : val(v) {
        for (int i = 0; i < N; ++i) {
            grad[i] = 0.0;
            for (int j = 0; j < N; ++j)
                hess[i][j] = 0.0;
        }
    }

    // 从值、梯度、Hessian 构造（用于运算结果）
    __host__ __device__
        Var(T v, const T* g, const T h[N][N]) : val(v) {
        for (int i = 0; i < N; ++i) {
            grad[i] = g[i];
            for (int j = 0; j < N; ++j)
                hess[i][j] = h[i][j];
        }
    }

    // 拷贝构造
    __host__ __device__
        Var(const Var& other) : val(other.val) {
        for (int i = 0; i < N; ++i) {
            grad[i] = other.grad[i];
            for (int j = 0; j < N; ++j)
                hess[i][j] = other.hess[i][j];
        }
    }

    // 赋值运算符
    __host__ __device__
        Var& operator=(const Var& other) {
        if (this != &other) {
            val = other.val;
            for (int i = 0; i < N; ++i) {
                grad[i] = other.grad[i];
                for (int j = 0; j < N; ++j)
                    hess[i][j] = other.hess[i][j];
            }
        }
        return *this;
    }
};

// 创建独立变量（第 idx 个变量，值为 x）
template <typename T, int N>
__host__ __device__
Var<T, N> make_variable(int idx, T x) {
    Var<T, N> v(x);
    v.grad[idx] = 1.0;          // ∂x_i/∂x_i = 1
    // Hessian 矩阵全为 0（独立变量二阶导为零）
    return v;
}

// -------------------- 运算符重载（正向自动微分）--------------------
// 加法
template <typename T, int N>
__host__ __device__
Var<T, N> operator+(const Var<T, N>& a, const Var<T, N>& b) {
    Var<T, N> r;
    r.val = a.val + b.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = a.grad[i] + b.grad[i];
        for (int j = 0; j < N; ++j)
            r.hess[i][j] = a.hess[i][j] + b.hess[i][j];
    }
    return r;
}

// 减法
template <typename T, int N>
__host__ __device__
Var<T, N> operator-(const Var<T, N>& a, const Var<T, N>& b) {
    Var<T, N> r;
    r.val = a.val - b.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = a.grad[i] - b.grad[i];
        for (int j = 0; j < N; ++j)
            r.hess[i][j] = a.hess[i][j] - b.hess[i][j];
    }
    return r;
}

// 乘法
template <typename T, int N>
__host__ __device__
Var<T, N> operator*(const Var<T, N>& a, const Var<T, N>& b) {
    Var<T, N> r;
    r.val = a.val * b.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = a.grad[i] * b.val + a.val * b.grad[i];
        for (int j = 0; j < N; ++j) {
            // 二阶导：a''*b + a'*b' + a'*b' + a*b''
            r.hess[i][j] = a.hess[i][j] * b.val
                + a.grad[i] * b.grad[j]
                + a.grad[j] * b.grad[i]
                + a.val * b.hess[i][j];
        }
    }
    return r;
}

// 除法
template <typename T, int N>
__host__ __device__
Var<T, N> operator/(const Var<T, N>& a, const Var<T, N>& b) {
    Var<T, N> r;
    T inv_b = 1.0 / b.val;
    r.val = a.val * inv_b;
    T b2 = b.val * b.val;
    T b4 = b2 * b2;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = (a.grad[i] * b.val - a.val * b.grad[i]) / b2;
        for (int j = 0; j < N; ++j) {
            T dM_dxj = a.hess[i][j] * b.val
                + a.grad[i] * b.grad[j]
                - a.grad[j] * b.grad[i]
                - a.val * b.hess[i][j];
            T dD_dxj = 2.0 * b.val * b.grad[j];
            r.hess[i][j] = (dM_dxj * b2 - (a.grad[i] * b.val - a.val * b.grad[i]) * dD_dxj) / b4;
        }
    }
    return r;
}

// 正弦函数
template <typename T, int N>
__host__ __device__
Var<T, N> sin(const Var<T, N>& a) {
    Var<T, N> r;
    r.val = sin(a.val);
    T cos_a = cos(a.val);
    T neg_sin_a = -sin(a.val);
    for (int i = 0; i < N; ++i) {
        r.grad[i] = cos_a * a.grad[i];
        for (int j = 0; j < N; ++j) {
            r.hess[i][j] = neg_sin_a * a.grad[i] * a.grad[j] + cos_a * a.hess[i][j];
        }
    }
    return r;
}

// 余弦函数
template <typename T, int N>
__host__ __device__
Var<T, N> cos(const Var<T, N>& a) {
    Var<T, N> r;
    r.val = cos(a.val);
    T neg_sin_a = -sin(a.val);
    T neg_cos_a = -cos(a.val);
    for (int i = 0; i < N; ++i) {
        r.grad[i] = neg_sin_a * a.grad[i];
        for (int j = 0; j < N; ++j) {
            r.hess[i][j] = neg_cos_a * a.grad[i] * a.grad[j] + neg_sin_a * a.hess[i][j];
        }
    }
    return r;
}

// 指数函数
template <typename T, int N>
__host__ __device__
Var<T, N> exp(const Var<T, N>& a) {
    Var<T, N> r;
    T exp_a = exp(a.val);
    r.val = exp_a;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = exp_a * a.grad[i];
        for (int j = 0; j < N; ++j) {
            r.hess[i][j] = exp_a * (a.grad[i] * a.grad[j] + a.hess[i][j]);
        }
    }
    return r;
}

// 常量乘以 Var（支持 double * Var）
template <typename T, int N>
__host__ __device__
Var<T, N> operator*(T c, const Var<T, N>& a) {
    Var<T, N> r;
    r.val = c * a.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = c * a.grad[i];
        for (int j = 0; j < N; ++j)
            r.hess[i][j] = c * a.hess[i][j];
    }
    return r;
}

// 自然对数函数 log(Var)
template <typename T, int N>
__host__ __device__
Var<T, N> log(const Var<T, N>& a) {
    Var<T, N> r;
    r.val = log(a.val);                     // 函数值
    T inv_a = 1.0 / a.val;                  // 1/a
    T inv_a2 = inv_a * inv_a;               // 1/a^2
    for (int i = 0; i < N; ++i) {
        r.grad[i] = inv_a * a.grad[i];      // ∂y/∂x_i = (1/a) * ∂a/∂x_i
        for (int j = 0; j < N; ++j) {
            // ∂²y/∂x_i∂x_j = (-1/a²) * (∂a/∂x_i)*(∂a/∂x_j) + (1/a) * (∂²a/∂x_i∂x_j)
            r.hess[i][j] = -inv_a2 * a.grad[i] * a.grad[j] + inv_a * a.hess[i][j];
        }
    }
    return r;
}

// Var * 常量
template <typename T, int N>
__host__ __device__
Var<T, N> operator*(const Var<T, N>& a, T c) {
    return c * a;
}

// Var / 常量
template <typename T, int N>
__host__ __device__
Var<T, N> operator/(const Var<T, N>& a, T c) {
    return a * (1.0 / c);
}

// ============================================================================
// 第二部分：复数自动微分 ComplexVar<T,N> 模板
// ============================================================================

template <typename T, int N>
struct ComplexVar {
    Var<T, N> real;  // 实部（包含梯度/Hessian）
    Var<T, N> imag;  // 虚部（包含梯度/Hessian）

    // 默认构造：零复数
    __host__ __device__
        ComplexVar(T re = 0.0, T im = 0.0) : real(re), imag(im) {}

    // 从实部、虚部 Var 构造
    __host__ __device__
        ComplexVar(const Var<T, N>& re, const Var<T, N>& im) : real(re), imag(im) {}

    // 拷贝构造
    __host__ __device__
        ComplexVar(const ComplexVar& other) : real(other.real), imag(other.imag) {}

    // 赋值运算符
    __host__ __device__
        ComplexVar& operator=(const ComplexVar& other) {
        if (this != &other) {
            real = other.real;
            imag = other.imag;
        }
        return *this;
    }
};

// 创建独立复数变量
// real_idx: 实部在实数变量列表中的索引 (0..N-1)
// imag_idx: 虚部在实数变量列表中的索引 (0..N-1)
// real_val: 实部初始值
// imag_val: 虚部初始值
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> make_complex_variable(int real_idx, int imag_idx, T real_val, T imag_val) {
    Var<T, N> re_var(real_val);
    Var<T, N> im_var(imag_val);
    re_var.grad[real_idx] = 1.0;  // ∂re/∂(re) = 1
    im_var.grad[imag_idx] = 1.0;  // ∂im/∂(im) = 1
    // 注意：实部和虚部是独立的，所以实部对 imag_idx 的偏导为 0（已默认），虚部对 real_idx 的偏导也为 0
    return ComplexVar<T, N>(re_var, im_var);
}

// -------------------- 复数运算 --------------------
// 加法
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> operator+(const ComplexVar<T, N>& a, const ComplexVar<T, N>& b) {
    return ComplexVar<T, N>(a.real + b.real, a.imag + b.imag);
}

// 减法
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> operator-(const ComplexVar<T, N>& a, const ComplexVar<T, N>& b) {
    return ComplexVar<T, N>(a.real - b.real, a.imag - b.imag);
}

// 复数乘法: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> operator*(const ComplexVar<T, N>& a, const ComplexVar<T, N>& b) {
    Var<T, N> re_part = a.real * b.real - a.imag * b.imag;
    Var<T, N> im_part = a.real * b.imag + a.imag * b.real;
    return ComplexVar<T, N>(re_part, im_part);
}

// 复数除法: (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c^2+d^2)
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> operator/(const ComplexVar<T, N>& a, const ComplexVar<T, N>& b) {
    Var<T, N> denom = b.real * b.real + b.imag * b.imag;
    Var<T, N> re_part = (a.real * b.real + a.imag * b.imag) / denom;
    Var<T, N> im_part = (a.imag * b.real - a.real * b.imag) / denom;
    return ComplexVar<T, N>(re_part, im_part);
}

// 共轭
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> conj(const ComplexVar<T, N>& z) {
    return ComplexVar<T, N>(z.real, -1.0 * z.imag);
}

// 模方 |z|^2 = real^2 + imag^2 (返回标量 Var)
template <typename T, int N>
__host__ __device__
Var<T, N> abs2(const ComplexVar<T, N>& z) {
    return z.real * z.real + z.imag * z.imag;
}

// 复数乘以标量常数 (标量 * 复数)
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> operator*(T c, const ComplexVar<T, N>& z) {
    return ComplexVar<T, N>(c * z.real, c * z.imag);
}

// 复数乘以标量常数 (复数 * 标量)
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> operator*(const ComplexVar<T, N>& z, T c) {
    return c * z;
}

// 复数除以标量常数
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> operator/(const ComplexVar<T, N>& z, T c) {
    return ComplexVar<T, N>(z.real / c, z.imag / c);
}

// 复数指数函数 (欧拉公式): exp(a+bi) = exp(a)*(cos(b) + i*sin(b))
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> exp(const ComplexVar<T, N>& z) {
    T exp_real = ::exp(z.real.val);
    Var<T, N> exp_re_part = exp_real * cos(z.imag);
    Var<T, N> exp_im_part = exp_real * sin(z.imag);
    return ComplexVar<T, N>(exp_re_part, exp_im_part);
}

// 辅助函数：从实部和虚部值创建常数复数（无梯度信息）
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> make_complex_constant(T re, T im) {
    return ComplexVar<T, N>(re, im);
}

// ============================================================================
// 第三部分：cuDoubleComplex / cuComplex 支持
// ============================================================================

// cuDoubleComplex 到 ComplexVar 的转换（常数，无梯度）
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> cucomplex_to_complexvar(const cuDoubleComplex& z) {
    // 注意：cuDoubleComplex 的成员是 x (实部) 和 y (虚部)
    return make_complex_constant<T, N>(z.x, z.y);
}

// cuComplex 到 ComplexVar 的转换 (float 版本)
template <typename T, int N>
__host__ __device__
ComplexVar<T, N> cucomplex_to_complexvar(const cuComplex& z) {
    return make_complex_constant<T, N>(z.x, z.y);
}


// 核函数：在设备端将 cuComplex 数组转换为 ComplexVar 数组
template <typename T, int N>
__global__ void init_complex_vars_kernel(
    const cuComplex* d_input,           // 输入复数数组 (长度 num_res)
    ComplexVar<T, N>* d_output,         // 输出 ComplexVar 数组
    int num_res
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_res) {
        cuComplex val = d_input[idx];
        // 实部索引 = 2*idx，虚部索引 = 2*idx+1
        d_output[idx] = make_complex_variable<T, N>(2 * idx, 2 * idx + 1, val.x, val.y);
    }
}

template <typename T, int N>
__global__ void extract_hessian_kernel(
    const Var<T, N>* d_var,   // 包含 Hessian 的结构体
    T* d_hessian_out          // 输出缓冲区，大小 N*N
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const Var<T, N>& v = *d_var;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                d_hessian_out[i * N + j] = v.hess[i][j];
            }
        }
    }
}

#endif // AUTODIFF_CUH