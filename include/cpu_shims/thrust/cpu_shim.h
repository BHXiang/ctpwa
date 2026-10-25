// cpu_shims/thrust/cpu_shim.h — thrust 的最小 CPU 替身（真正的实现都在这里）
//
// 只出现在 CTPWA_BACKEND=cpu 构建的 include 路径最前面。
// 全库实际用到的 thrust 面非常窄：complex / device_ptr / reduce / plus / exp，
// 以及一批"只 include 不使用"的头（device_vector、sort、transform…）。
// 因此这里不需要仿制 thrust 的迭代器体系，只要让这些名字有等价物即可。

#pragma once

// 先给出 __host__ / __device__ 等属性的替身：有些头文件在包含 ComplexType.h
// 之前就先 include thrust（如 Amplitude.cuh:4），否则 __device__ 会未定义。
#include "CudaCompat.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

namespace thrust {

// ---- complex ----
// ComplexType.h 在 CPU 分支已经定义过同名 alias，用同一守卫避免重复定义
#ifndef CTPWA_THRUST_COMPLEX_DEFINED
#  define CTPWA_THRUST_COMPLEX_DEFINED 1
template <class T>
using complex = std::complex<T>;
#endif

using std::exp;

// ---- vector ----
template <class T, class A = std::allocator<T>>
using host_vector = std::vector<T, A>;

template <class T, class A = std::allocator<T>>
using device_vector = std::vector<T, A>;

// ---- functional ----
template <class T>
using plus = std::plus<T>;

// ---- device_ptr ----
// 同一地址空间下的"指针包装"，只需支持 ++/+/[]/*/比较。
template <class T>
struct device_ptr {
    using value_type = typename std::remove_const<T>::type;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::random_access_iterator_tag;

    T* p;

    device_ptr() : p(nullptr) {}
    explicit device_ptr(T* raw) : p(raw) {}

    T* get() const { return p; }
    reference operator*() const { return *p; }
    reference operator[](difference_type i) const { return p[i]; }

    device_ptr operator+(difference_type i) const { return device_ptr(p + i); }
    device_ptr operator-(difference_type i) const { return device_ptr(p - i); }
    difference_type operator-(const device_ptr& o) const { return p - o.p; }
    device_ptr& operator++() { ++p; return *this; }
    device_ptr operator++(int) { device_ptr t(*this); ++p; return t; }

    bool operator==(const device_ptr& o) const { return p == o.p; }
    bool operator!=(const device_ptr& o) const { return p != o.p; }
    bool operator<(const device_ptr& o) const { return p < o.p; }
};

template <class T>
inline T* raw_pointer_cast(const device_ptr<T>& d) { return d.get(); }

// ---- reduce ----
// 串行累加：小数组（权重求和）用，可复现且与线程数无关。
template <class It>
inline typename std::iterator_traits<It>::value_type reduce(It first, It last)
{
    using V = typename std::iterator_traits<It>::value_type;
    V acc = V();
    for (It it = first; it != last; ++it)
        acc = acc + *it;
    return acc;
}

template <class It, class T>
inline T reduce(It first, It last, T init)
{
    for (It it = first; it != last; ++it)
        init = init + *it;
    return init;
}

template <class It, class T, class Op>
inline T reduce(It first, It last, T init, Op op)
{
    for (It it = first; it != last; ++it)
        init = op(init, *it);
    return init;
}

// ---- 算法（只 include 不使用，给个等价物即可）----
using std::binary_search;
using std::lower_bound;
using std::upper_bound;
using std::sort;

}  // namespace thrust
