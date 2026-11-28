// include/MultiGpuStorage.h — 每卡一份的 DeviceBuffer 集合及裸指针视图
//
// 所有权仍由 DevBuffer 持有；本类只负责：
//   * 多卡 buffer 的整体 resize/clear；
//   * 为既有 kernel 接口提供 std::vector<T*> 裸指针视图；
//   * move 后刷新视图，防止指针指向 moved-from 的旧存储。
#pragma once

#include "DevBuffer.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace ctpwa {

template <class T>
class MultiGpuStorage {
public:
    MultiGpuStorage() = default;

    MultiGpuStorage(MultiGpuStorage&& other) noexcept
        : bufs_(std::move(other.bufs_))
    {
        refreshViews();
        other.clearViews();
    }

    MultiGpuStorage& operator=(MultiGpuStorage&& other) noexcept
    {
        if (this != &other) {
            bufs_ = std::move(other.bufs_);
            refreshViews();
            other.clearViews();
        }
        return *this;
    }

    MultiGpuStorage(const MultiGpuStorage&) = delete;
    MultiGpuStorage& operator=(const MultiGpuStorage&) = delete;

    void resize(std::size_t n)
    {
        bufs_.resize(n);
        refreshViews();
    }

    void clear()
    {
        bufs_.clear();       // 每个 DevBuffer 在自己的设备上 reset
        views_.clear();
    }

    void reset() { clear(); }

    std::size_t size() const { return bufs_.size(); }
    bool empty() const { return bufs_.empty(); }

    T* operator[](std::size_t i) { return bufs_[i].data(); }
    T* operator[](std::size_t i) const { return bufs_[i].data(); }

    DevBuffer<T>& at(std::size_t i) { refreshViews(); return bufs_[i]; }
    const DevBuffer<T>& at(std::size_t i) const { refreshViews(); return bufs_[i]; }

    // 既有 kernel 的只读视图：不转移所有权；本对象 clear/resize/move 后需重新获取。
    std::vector<T*>& data() { refreshViews(); return views_; }
    const std::vector<T*>& data() const { refreshViews(); return views_; }

private:
    void refreshViews() const
    {
        views_.clear();
        views_.reserve(bufs_.size());
        for (const auto& b : bufs_) views_.push_back(b.data());
    }

    void clearViews() { views_.clear(); }

    std::vector<DevBuffer<T>> bufs_;
    mutable std::vector<T*> views_;
};

}  // namespace ctpwa
