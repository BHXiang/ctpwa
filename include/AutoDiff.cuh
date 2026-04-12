#ifndef AutoDiff_CUH
#define AutoDiff_CUH

#include <cstddef>

// AutoDiff：多变量自动梯度 - 支持CUDA的多变量自动微分
template <size_t N> struct AutoDiff {
  double value;          // 函数值
  double derivatives[N]; // N个变量的偏导数

  // 从值和导数数组构造（主机/设备）
  __host__ __device__ AutoDiff(double v, const double d[N]) : value(v) {
    for (size_t i = 0; i < N; ++i)
      derivatives[i] = d[i];
  }

  // 从值和单个导数值构造（所有导数相同）
  __host__ __device__ AutoDiff(double v, double d = 0.0) : value(v) {
    for (size_t i = 0; i < N; ++i)
      derivatives[i] = d;
  }

  // 默认构造函数
  __host__ __device__ AutoDiff() : value(0) {
    for (size_t i = 0; i < N; ++i)
      derivatives[i] = 0.0;
  }

  // 从值和初始化列表构造（仅主机）
#ifdef __CUDA_ARCH__
  // 在设备代码中，无法使用std::initializer_list，因此提供替代方案
  __device__ AutoDiff(double v, double d0, double d1) : value(v) {
    static_assert(N == 2, "This constructor only works for N=2");
    derivatives[0] = d0;
    derivatives[1] = d1;
    for (size_t i = 2; i < N; ++i)
      derivatives[i] = 0.0;
  }
#else
  // 主机代码可以使用initializer_list
  AutoDiff(double v, std::initializer_list<double> init) : value(v) {
    size_t i = 0;
    for (double d : init) {
      if (i < N)
        derivatives[i++] = d;
    }
    for (; i < N; ++i)
      derivatives[i] = 0.0;
  }
#endif

  // 获取第i个变量的导数
  __host__ __device__ double derivative(size_t i) const {
    return (i < N) ? derivatives[i] : 0.0;
  }

  // 设置第i个变量的导数
  __host__ __device__ void set_derivative(size_t i, double d) {
    if (i < N)
      derivatives[i] = d;
  }

  // 加法运算符
  __host__ __device__ AutoDiff<N> operator+(const AutoDiff<N> &other) const {
    double result_d[N];
    for (size_t i = 0; i < N; ++i) {
      result_d[i] = derivatives[i] + other.derivatives[i];
    }
    return AutoDiff<N>(value + other.value, result_d);
  }

  // 减法运算符
  __host__ __device__ AutoDiff<N> operator-(const AutoDiff<N> &other) const {
    double result_d[N];
    for (size_t i = 0; i < N; ++i) {
      result_d[i] = derivatives[i] - other.derivatives[i];
    }
    return AutoDiff<N>(value - other.value, result_d);
  }

  // 乘法运算符
  __host__ __device__ AutoDiff<N> operator*(const AutoDiff<N> &other) const {
    double result_d[N];
    for (size_t i = 0; i < N; ++i) {
      result_d[i] = value * other.derivatives[i] + derivatives[i] * other.value;
    }
    return AutoDiff<N>(value * other.value, result_d);
  }

  // 与标量的加法（右侧）
  __host__ __device__ AutoDiff<N> operator+(double other) const {
    return AutoDiff<N>(value + other, derivatives); // 导数不变
  }

  // 与标量的减法（右侧）
  __host__ __device__ AutoDiff<N> operator-(double other) const {
    return AutoDiff<N>(value - other, derivatives);
  }

  // 与标量的乘法（右侧）
  __host__ __device__ AutoDiff<N> operator*(double other) const {
    double result_d[N];
    for (size_t i = 0; i < N; ++i) {
      result_d[i] = derivatives[i] * other;
    }
    return AutoDiff<N>(value * other, result_d);
  }

  // 除法运算符
  __host__ __device__ AutoDiff<N> operator/(const AutoDiff<N> &other) const {
    double inv_value = 1.0 / other.value;
    double inv_value2 = inv_value * inv_value;
    double result_d[N];
    for (size_t i = 0; i < N; ++i) {
      result_d[i] =
          (derivatives[i] * other.value - value * other.derivatives[i]) *
          inv_value2;
    }
    return AutoDiff<N>(value * inv_value, result_d);
  }

  // 除以标量（右侧）
  __host__ __device__ AutoDiff<N> operator/(double other) const {
    double inv_other = 1.0 / other;
    double result_d[N];
    for (size_t i = 0; i < N; ++i) {
      result_d[i] = derivatives[i] * inv_other;
    }
    return AutoDiff<N>(value * inv_other, result_d);
  }
};

// 标量左加
template <size_t N>
__host__ __device__ AutoDiff<N> operator+(double a, const AutoDiff<N> &x) {
  return AutoDiff<N>(a + x.value, x.derivatives);
}

// 标量左减
template <size_t N>
__host__ __device__ AutoDiff<N> operator-(double a, const AutoDiff<N> &x) {
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = -x.derivatives[i];
  }
  return AutoDiff<N>(a - x.value, result_d);
}

// 标量左乘
template <size_t N>
__host__ __device__ AutoDiff<N> operator*(double a, const AutoDiff<N> &x) {
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = a * x.derivatives[i];
  }
  return AutoDiff<N>(a * x.value, result_d);
}

// 标量左除
template <size_t N>
__host__ __device__ AutoDiff<N> operator/(double a, const AutoDiff<N> &x) {
  double inv_value = 1.0 / x.value;
  double inv_value2 = inv_value * inv_value;
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = -a * x.derivatives[i] * inv_value2;
  }
  return AutoDiff<N>(a * inv_value, result_d);
}

// AutoDiff的数学函数

// 正弦函数
template <size_t N> __host__ __device__ AutoDiff<N> sin(const AutoDiff<N> &x) {
  double cos_val = ::cos(x.value);
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = cos_val * x.derivatives[i];
  }
  return AutoDiff<N>(::sin(x.value), result_d);
}

// 平方根函数
template <size_t N> __host__ __device__ AutoDiff<N> sqrt(const AutoDiff<N> &x) {
  double sqrt_val = ::sqrt(x.value);
  double factor = 0.5 / sqrt_val;
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = factor * x.derivatives[i];
  }
  return AutoDiff<N>(sqrt_val, result_d);
}

// 幂函数（整数指数）
template <size_t N>
__host__ __device__ AutoDiff<N> pow(const AutoDiff<N> &x, int n) {
  if (n == 0) {
    return AutoDiff<N>(1.0);
  } else if (n == 1) {
    return x;
  } else if (n < 0) {
    AutoDiff<N> xn = pow(x, -n);
    return 1.0 / xn;
  } else {
    // n >= 2
    double val_pow = ::pow(x.value, n);
    double factor = n * ::pow(x.value, n - 1);
    double result_d[N];
    for (size_t i = 0; i < N; ++i) {
      result_d[i] = factor * x.derivatives[i];
    }
    return AutoDiff<N>(val_pow, result_d);
  }
}

// 幂函数（实数指数）
template <size_t N>
__host__ __device__ AutoDiff<N> pow(const AutoDiff<N> &x, double a) {
  // 实数幂：x^a = exp(a * log(x))
  double val_pow = ::pow(x.value, a);
  double factor = a * ::pow(x.value, a - 1);
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = factor * x.derivatives[i];
  }
  return AutoDiff<N>(val_pow, result_d);
}

// 指数函数
template <size_t N> __host__ __device__ AutoDiff<N> exp(const AutoDiff<N> &x) {
  double exp_val = ::exp(x.value);
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = exp_val * x.derivatives[i];
  }
  return AutoDiff<N>(exp_val, result_d);
}

// 自然对数函数
template <size_t N> __host__ __device__ AutoDiff<N> log(const AutoDiff<N> &x) {
  double log_val = ::log(x.value);
  double factor = 1.0 / x.value;
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = factor * x.derivatives[i];
  }
  return AutoDiff<N>(log_val, result_d);
}

// 余弦函数
template <size_t N> __host__ __device__ AutoDiff<N> cos(const AutoDiff<N> &x) {
  double cos_val = ::cos(x.value);
  double sin_val = -::sin(x.value); // cos的导数是-sin
  double result_d[N];
  for (size_t i = 0; i < N; ++i) {
    result_d[i] = sin_val * x.derivatives[i];
  }
  return AutoDiff<N>(cos_val, result_d);
}

// 辅助函数：创建常量AutoDiff（主机/设备）
template <size_t N>
__host__ __device__ AutoDiff<N> make_AutoDiff_constant(double val,
                                                       double deriv = 0.0) {
  return AutoDiff<N>(val, deriv);
}

#endif // AutoDiff_CUH