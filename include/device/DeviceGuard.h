// include/DeviceGuard.h — 设备作用域（RAII）与全项目唯一设备切换入口
//
// 为什么存在
// ----------
// `cudaSetDevice` 改的是**线程局部环境状态**。"进入函数时当前设备是主卡"这个约定
// 不可检查、不可继承，也无法在 CPU 构建上验证（CPU 后端 current_device() 恒为 0）。
// 历史故障：bc22b52 —— 逐 GPU 权重归约把 current device 留在最后一张卡，副卡 kernel
// 直接读主卡指针（compute-sanitizer 才发现）。
//
// 三个原语对应三类调用点（219 处的归类见 doc/modularization-plan.md §2.2）：
//
//   ensure_device(dev)      粘性设置 —— 取代函数入口的 cudaSetDevice(primary_dev_)
//                           （语义完全不变：只切，不还原）
//   DeviceGuard g(dev)      作用域 —— 进入切、退出还原
//                           （取代成对的 set(gpu) … set(primary)）
//   with_device(dev, fn)    作用域的函数式写法，给多卡循环用
//
// 约定：src/ 与 include/ 里除 Backend.cpp（后端实现）与 DeviceManager.cu（设备模块）
// 之外**不得**出现裸 cudaSetDevice —— 由 scripts/check_device_calls.sh 机械检查。
#pragma once

#include "Backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace ctpwa {

// 当前设备（后端无关：CUDA 下是 cudaGetDevice，CPU 下恒 0）
inline int device_now() { return current_device(); }

// 严格模式：越界/切换失败直接 abort，而不是留下一个"大概在某张卡上"的状态。
// 默认关闭（生产拟合不希望因为一次探测失败就中断），CI 与调试时置 1。
inline bool device_strict()
{
    static const bool v = [] {
        const char* e = std::getenv("CTPWA_STRICT_DEVICE");
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

// 切到 dev（带错误检查）。失败返回 false；严格模式下 abort。
// 注意：不还原。需要还原请用 DeviceGuard。
inline bool device_set(int dev)
{
    if (dev < 0) {
        std::fprintf(stderr, "[ctpwa] device_set: 非法设备号 %d\n", dev);
        if (device_strict()) std::abort();
        return false;
    }
    if (set_device(dev)) return true;
    std::fprintf(stderr,
                 "[ctpwa] device_set(%d) 失败（设备不存在，或 CPU 后端下 index != 0）\n",
                 dev);
    if (device_strict()) std::abort();
    return false;
}

// 粘性：确保当前设备是 dev。已经是则零开销（不触发 cudaSetDevice）。
inline void ensure_device(int dev)
{
    if (current_device() != dev) device_set(dev);
}

// RAII 设备作用域：构造时保存并切换，析构时还原。
// 禁止拷贝：一个作用域只能有一份"该还原到哪"的记录。
class DeviceGuard {
public:
    explicit DeviceGuard(int dev)
        : saved_(current_device()), target_(dev)
    {
        if (saved_ != target_) device_set(target_);
    }

    ~DeviceGuard()
    {
        // 还原失败也不抛（析构期抛异常会 terminate）；device_set 已打印。
        if (current_device() != saved_) device_set(saved_);
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&&) = delete;
    DeviceGuard& operator=(DeviceGuard&&) = delete;

    int device() const { return target_; }
    int saved() const { return saved_; }

private:
    int saved_;
    int target_;
};

// 作用域 + 回调，给多卡循环用：
//   for (int g = 0; g < n_gpu; ++g) with_device(g, [&] { ... });
template <class F>
inline void with_device(int dev, F&& fn)
{
    DeviceGuard guard(dev);
    std::forward<F>(fn)();
}

}  // namespace ctpwa
