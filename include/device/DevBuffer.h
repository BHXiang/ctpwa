// include/DevBuffer.h — 单一所有权的设备缓冲（move-only）
//
// 解决的问题（modularization-plan §2.4b）
// --------------------------------------
// `AmpCasDecay` 曾把同一批设备指针同时登记在 `d_mom_sigma_` 与
// `d_momenta_structs_/d_momenta_arrays_` 两处，释放时只能靠"按指针匹配把它从
// 另一个登记表里摘掉"来避免二次 free（CPU 后端直接 abort：
// `free(): double free detected`）。那是两套簿记在争夺一个所有权。
//
// 本类型把"谁负责 free"变成类型属性：
//   * 构造即分配，析构即释放（在**自己的设备**上，调用方不需要先切设备）；
//   * 禁止拷贝，只能 move —— 一个缓冲区永远只有一个所有者；
//   * `data()` 给的是裸指针视图，**不含所有权**，可以随便传。
//
// 关于进程退出：CUDA 上下文可能已被 torch 销毁，此时裸 `cudaFree` 会段错误。
// 这正是 `AmpCasDecay` 析构里 `cudaFree(nullptr)` 探针存在的原因。对应地，
// 本类型提供 `disarm()`：丢弃指针但**不** free，交还给驱动在上下文销毁时回收。
#pragma once

#include "Backend.h"
#include "DeviceGuard.h"

#include <cstddef>
#include <utility>

namespace ctpwa {

template <class T>
class DevBuffer {
public:
    DevBuffer() = default;

    // 在 dev 上分配 n 个 T（内容未初始化）
    DevBuffer(int dev, std::size_t n) { allocate(dev, n); }

    ~DevBuffer() { reset(); }

    DevBuffer(DevBuffer&& other) noexcept
        : ptr_(other.ptr_), dev_(other.dev_), n_(other.n_)
    {
        other.ptr_ = nullptr;
        other.n_ = 0;
        other.bytes_ = 0;
        other.dev_ = -1;
    }

    DevBuffer& operator=(DevBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            dev_ = other.dev_;
            n_ = other.n_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.n_ = 0;
            other.bytes_ = 0;
            other.dev_ = -1;
        }
        return *this;
    }

    DevBuffer(const DevBuffer&) = delete;
    DevBuffer& operator=(const DevBuffer&) = delete;

    // 按字节分配（元素宽度随精度档位变化时用，例如 dF 存 float2 或 double2）
    //
    // ⚠ 0 字节也要分配 1 个元素：`cudaMalloc(0)` 在 CUDA 上返回成功且指针非空，
    // 旧代码依赖这个行为（调用方随后用 `if (!ptr) throw` 判失败）。若这里直接返回
    // 空指针，零长度表（如 deep_4pi0 的 nSL 组合数为 0）会被误判成"分配失败"。
    void allocate_bytes(int dev, std::size_t bytes)
    {
        reset();
        const std::size_t ask = bytes ? bytes : sizeof(T);
        void* p = nullptr;
        {
            DeviceGuard guard(dev);
            p = alloc(ask);
        }
        if (p) {
            ptr_ = static_cast<T*>(p);
            dev_ = dev;
            n_ = bytes / sizeof(T);
            bytes_ = ask;
        }
    }

    std::size_t bytes() const { return bytes_ ? bytes_ : n_ * sizeof(T); }

    // 释放后重新分配；分配失败时保持空（不抛异常，行为与原 cudaMalloc 一致）。
    void allocate(int dev, std::size_t n)
    {
        reset();
        // n == 0 也分配 1 个元素（理由见 allocate_bytes 的注释：cudaMalloc(0) 非空）
        const std::size_t ask = n ? n : 1;
        void* p = nullptr;
        {
            DeviceGuard guard(dev);   // 在目标设备上分配
            p = alloc(ask * sizeof(T));
        }
        if (p) {
            ptr_ = static_cast<T*>(p);
            dev_ = dev;
            n_ = n;
        }
    }

    // 立即释放（在分配它的设备上）
    void reset()
    {
        if (ptr_) {
            DeviceGuard guard(dev_);
            dealloc(ptr_);
            ptr_ = nullptr;
        }
        n_ = 0;
        bytes_ = 0;
        dev_ = -1;
    }

    // 只丢弃指针，不 free（进程退出期，上下文可能已销毁）
    void disarm() noexcept
    {
        ptr_ = nullptr;
        n_ = 0;
        bytes_ = 0;
        dev_ = -1;
    }

    T* data() const { return ptr_; }
    std::size_t size() const { return n_; }
    int device() const { return dev_; }
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
    int dev_ = -1;
    std::size_t n_ = 0;
    std::size_t bytes_ = 0;
};

}  // namespace ctpwa
