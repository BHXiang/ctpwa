#ifndef AUTODIFF_CUH
#define AUTODIFF_CUH

#include <cuda_runtime.h>
#include <ComplexType.h>
#include <cmath>
#include <iostream>

// ============================================================================
// 第一部分：实数自动微分 Var<T,N,WithHess> 模板
// WithHess = true  → val + grad + hess (默认，向后兼容)
// WithHess = false → val + grad only (跳过 Hessian 计算，节省寄存器和运算)
// ============================================================================

template <typename T, int N, bool WithHess = true>
struct Var {
    T val;                     // 函数值
    T grad[N];                 // 一阶偏导
    T hess[N][N];              // 二阶偏导（Hessian 矩阵）；WithHess=false 时仅占位不计算

    // 默认构造（常数值，导数为零）
    __host__ __device__
    Var(T v = 0.0) : val(v) {
        for (int i = 0; i < N; ++i) {
            grad[i] = 0.0;
        }
        if constexpr (WithHess) {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    hess[i][j] = 0.0;
        }
    }

    // 从值、梯度、Hessian 构造（用于运算结果）
    __host__ __device__
    Var(T v, const T* g, const T h[N][N]) : val(v) {
        for (int i = 0; i < N; ++i) {
            grad[i] = g[i];
        }
        if constexpr (WithHess) {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    hess[i][j] = h[i][j];
        }
    }

    // 拷贝构造
    __host__ __device__
    Var(const Var& other) : val(other.val) {
        for (int i = 0; i < N; ++i) {
            grad[i] = other.grad[i];
        }
        if constexpr (WithHess) {
            for (int i = 0; i < N; ++i)
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
            }
            if constexpr (WithHess) {
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j)
                        hess[i][j] = other.hess[i][j];
            }
        }
        return *this;
    }
};

// 创建独立变量（第 idx 个变量，值为 x）
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> make_variable(int idx, T x) {
    Var<T, N, WithHess> v(x);
    v.grad[idx] = 1.0;          // ∂x_i/∂x_i = 1
    return v;
}

// -------------------- 运算符重载（正向自动微分）--------------------
// 加法
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator+(const Var<T, N, WithHess>& a, const Var<T, N, WithHess>& b) {
    Var<T, N, WithHess> r;
    r.val = a.val + b.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = a.grad[i] + b.grad[i];
        if constexpr (WithHess) {
            for (int j = 0; j < N; ++j)
                r.hess[i][j] = a.hess[i][j] + b.hess[i][j];
        }
    }
    return r;
}

// 减法
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator-(const Var<T, N, WithHess>& a, const Var<T, N, WithHess>& b) {
    Var<T, N, WithHess> r;
    r.val = a.val - b.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = a.grad[i] - b.grad[i];
        if constexpr (WithHess) {
            for (int j = 0; j < N; ++j)
                r.hess[i][j] = a.hess[i][j] - b.hess[i][j];
        }
    }
    return r;
}

// 乘法
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator*(const Var<T, N, WithHess>& a, const Var<T, N, WithHess>& b) {
    Var<T, N, WithHess> r;
    r.val = a.val * b.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = a.grad[i] * b.val + a.val * b.grad[i];
        if constexpr (WithHess) {
            for (int j = 0; j < N; ++j) {
                r.hess[i][j] = a.hess[i][j] * b.val
                    + a.grad[i] * b.grad[j]
                    + a.grad[j] * b.grad[i]
                    + a.val * b.hess[i][j];
            }
        }
    }
    return r;
}

// 除法
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator/(const Var<T, N, WithHess>& a, const Var<T, N, WithHess>& b) {
    Var<T, N, WithHess> r;
    T inv_b = 1.0 / b.val;
    r.val = a.val * inv_b;
    T b2 = b.val * b.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = (a.grad[i] * b.val - a.val * b.grad[i]) / b2;
        if constexpr (WithHess) {
            T b4 = b2 * b2;
            for (int j = 0; j < N; ++j) {
                T dM_dxj = a.hess[i][j] * b.val
                    + a.grad[i] * b.grad[j]
                    - a.grad[j] * b.grad[i]
                    - a.val * b.hess[i][j];
                T dD_dxj = 2.0 * b.val * b.grad[j];
                r.hess[i][j] = (dM_dxj * b2 - (a.grad[i] * b.val - a.val * b.grad[i]) * dD_dxj) / b4;
            }
        }
    }
    return r;
}

// 正弦函数
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> sin(const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    r.val = sin(a.val);
    T cos_a = cos(a.val);
    for (int i = 0; i < N; ++i) {
        r.grad[i] = cos_a * a.grad[i];
        if constexpr (WithHess) {
            T neg_sin_a = -sin(a.val);
            for (int j = 0; j < N; ++j) {
                r.hess[i][j] = neg_sin_a * a.grad[i] * a.grad[j] + cos_a * a.hess[i][j];
            }
        }
    }
    return r;
}

// 余弦函数
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> cos(const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    r.val = cos(a.val);
    T neg_sin_a = -sin(a.val);
    for (int i = 0; i < N; ++i) {
        r.grad[i] = neg_sin_a * a.grad[i];
        if constexpr (WithHess) {
            T neg_cos_a = -cos(a.val);
            for (int j = 0; j < N; ++j) {
                r.hess[i][j] = neg_cos_a * a.grad[i] * a.grad[j] + neg_sin_a * a.hess[i][j];
            }
        }
    }
    return r;
}

// 指数函数
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> exp(const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    T exp_a = exp(a.val);
    r.val = exp_a;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = exp_a * a.grad[i];
        if constexpr (WithHess) {
            for (int j = 0; j < N; ++j) {
                r.hess[i][j] = exp_a * (a.grad[i] * a.grad[j] + a.hess[i][j]);
            }
        }
    }
    return r;
}

// 幂函数 pow(Var, T)
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> pow(const Var<T, N, WithHess>& a, T exponent) {
    Var<T, N, WithHess> r;
    T pow_val = pow(a.val, exponent);
    r.val = pow_val;
    T deriv1 = exponent * pow(a.val, exponent - 1.0);
    for (int i = 0; i < N; ++i) {
        r.grad[i] = deriv1 * a.grad[i];
        if constexpr (WithHess) {
            T deriv2 = exponent * (exponent - 1.0) * pow(a.val, exponent - 2.0);
            for (int j = 0; j < N; ++j) {
                r.hess[i][j] = deriv2 * a.grad[i] * a.grad[j] + deriv1 * a.hess[i][j];
            }
        }
    }
    return r;
}

// sqrt 函数
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> sqrt(const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    T sqrt_val = sqrt(a.val);
    r.val = sqrt_val;
    T inv_2sqrt = 0.5 / sqrt_val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = inv_2sqrt * a.grad[i];
        if constexpr (WithHess) {
            T neg_inv_4a = -0.25 / (a.val * sqrt_val);
            for (int j = 0; j < N; ++j) {
                r.hess[i][j] = neg_inv_4a * a.grad[i] * a.grad[j] + inv_2sqrt * a.hess[i][j];
            }
        }
    }
    return r;
}

// 常量乘以 Var（支持 double * Var）
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator*(T c, const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    r.val = c * a.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = c * a.grad[i];
        if constexpr (WithHess) {
            for (int j = 0; j < N; ++j)
                r.hess[i][j] = c * a.hess[i][j];
        }
    }
    return r;
}

// 自然对数函数 log(Var)
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> log(const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    r.val = log(a.val);
    T inv_a = 1.0 / a.val;
    for (int i = 0; i < N; ++i) {
        r.grad[i] = inv_a * a.grad[i];
        if constexpr (WithHess) {
            T inv_a2 = inv_a * inv_a;
            for (int j = 0; j < N; ++j) {
                r.hess[i][j] = -inv_a2 * a.grad[i] * a.grad[j] + inv_a * a.hess[i][j];
            }
        }
    }
    return r;
}

// Var * 常量
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator*(const Var<T, N, WithHess>& a, T c) {
    return c * a;
}

// Var / 常量
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator/(const Var<T, N, WithHess>& a, T c) {
    return a * (1.0 / c);
}

// 标量 + Var
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator+(T c, const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    r.val = c + a.val;
    for (int i = 0; i < N; ++i) r.grad[i] = a.grad[i];
    if constexpr (WithHess) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.hess[i][j] = a.hess[i][j];
    }
    return r;
}

// Var + 标量
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator+(const Var<T, N, WithHess>& a, T c) {
    return c + a;
}

// 标量 - Var
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator-(T c, const Var<T, N, WithHess>& a) {
    Var<T, N, WithHess> r;
    r.val = c - a.val;
    for (int i = 0; i < N; ++i) r.grad[i] = -a.grad[i];
    if constexpr (WithHess) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.hess[i][j] = -a.hess[i][j];
    }
    return r;
}

// Var - 标量
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> operator-(const Var<T, N, WithHess>& a, T c) {
    Var<T, N, WithHess> r;
    r.val = a.val - c;
    for (int i = 0; i < N; ++i) r.grad[i] = a.grad[i];
    if constexpr (WithHess) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.hess[i][j] = a.hess[i][j];
    }
    return r;
}

// pow(Var, int) — 方便 pow(z0, 6) 这种调用
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> pow(const Var<T, N, WithHess>& a, int exponent) {
    return pow(a, T(exponent));
}

// ============================================================================
// 第二部分：复数自动微分 ComplexVar<T,N,WithHess> 模板
// ============================================================================

template <typename T, int N, bool WithHess = true>
struct ComplexVar {
    Var<T, N, WithHess> real;  // 实部（包含梯度/Hessian）
    Var<T, N, WithHess> imag;  // 虚部（包含梯度/Hessian）

    // 默认构造：零复数
    __host__ __device__
    ComplexVar(T re = 0.0, T im = 0.0) : real(re), imag(im) {}

    // 从实部、虚部 Var 构造
    __host__ __device__
    ComplexVar(const Var<T, N, WithHess>& re, const Var<T, N, WithHess>& im) : real(re), imag(im) {}

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
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> make_complex_variable(int real_idx, int imag_idx, T real_val, T imag_val) {
    Var<T, N, WithHess> re_var(real_val);
    Var<T, N, WithHess> im_var(imag_val);
    re_var.grad[real_idx] = 1.0;
    im_var.grad[imag_idx] = 1.0;
    return ComplexVar<T, N, WithHess>(re_var, im_var);
}

// -------------------- 复数运算 --------------------
// 加法
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> operator+(const ComplexVar<T, N, WithHess>& a, const ComplexVar<T, N, WithHess>& b) {
    return ComplexVar<T, N, WithHess>(a.real + b.real, a.imag + b.imag);
}

// 减法
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> operator-(const ComplexVar<T, N, WithHess>& a, const ComplexVar<T, N, WithHess>& b) {
    return ComplexVar<T, N, WithHess>(a.real - b.real, a.imag - b.imag);
}

// 复数乘法: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> operator*(const ComplexVar<T, N, WithHess>& a, const ComplexVar<T, N, WithHess>& b) {
    return ComplexVar<T, N, WithHess>(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real);
}

// 复数除法: (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c^2+d^2)
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> operator/(const ComplexVar<T, N, WithHess>& a, const ComplexVar<T, N, WithHess>& b) {
    Var<T, N, WithHess> denom = b.real * b.real + b.imag * b.imag;
    return ComplexVar<T, N, WithHess>(
        (a.real * b.real + a.imag * b.imag) / denom,
        (a.imag * b.real - a.real * b.imag) / denom);
}

// 共轭
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> conj(const ComplexVar<T, N, WithHess>& z) {
    return ComplexVar<T, N, WithHess>(z.real, -1.0 * z.imag);
}

// 模方 |z|^2 = real^2 + imag^2 (返回标量 Var)
template <typename T, int N, bool WithHess>
__host__ __device__
Var<T, N, WithHess> abs2(const ComplexVar<T, N, WithHess>& z) {
    return z.real * z.real + z.imag * z.imag;
}

// 复数乘以标量常数 (标量 * 复数)
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> operator*(T c, const ComplexVar<T, N, WithHess>& z) {
    return ComplexVar<T, N, WithHess>(c * z.real, c * z.imag);
}

// 复数乘以标量常数 (复数 * 标量)
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> operator*(const ComplexVar<T, N, WithHess>& z, T c) {
    return c * z;
}

// 复数除以标量常数
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> operator/(const ComplexVar<T, N, WithHess>& z, T c) {
    return ComplexVar<T, N, WithHess>(z.real / c, z.imag / c);
}

// 复数指数函数 (欧拉公式): exp(a+bi) = exp(a)*(cos(b) + i*sin(b))
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> exp(const ComplexVar<T, N, WithHess>& z) {
    T exp_real = ::exp(z.real.val);
    return ComplexVar<T, N, WithHess>(exp_real * cos(z.imag), exp_real * sin(z.imag));
}

// 辅助函数：从实部和虚部值创建常数复数（无梯度信息）
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> make_complex_constant(T re, T im) {
    return ComplexVar<T, N, WithHess>(re, im);
}

// ============================================================================
// 第三部分：cuDoubleComplex / ctComplex 支持
// ============================================================================

// ctComplex 到 ComplexVar 的转换（常数，无梯度）
// （ctComplex 在 float 版为 cuComplex，double 版为 cuDoubleComplex）
template <typename T, int N, bool WithHess>
__host__ __device__
ComplexVar<T, N, WithHess> cucomplex_to_complexvar(const ctComplex& z) {
    return make_complex_constant<T, N, WithHess>(z.x, z.y);
}


// 核函数：在设备端将 ctComplex 数组转换为 ComplexVar 数组
template <typename T, int N, bool WithHess>
__global__ void init_complex_vars_kernel(
    const ctComplex* d_input,
    ComplexVar<T, N, WithHess>* d_output,
    int num_res
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_res) {
        ctComplex val = d_input[idx];
        d_output[idx] = make_complex_variable<T, N, WithHess>(2 * idx, 2 * idx + 1, val.x, val.y);
    }
}

template <typename T, int N, bool WithHess>
__global__ void extract_hessian_kernel(
    const Var<T, N, WithHess>* d_var,
    T* d_hessian_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const Var<T, N, WithHess>& v = *d_var;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                d_hessian_out[i * N + j] = v.hess[i][j];
            }
        }
    }
}

#endif // AUTODIFF_CUH
