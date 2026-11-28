#include <DevBuffer.h>
#include <MultiGpuStorage.h>

#include <cassert>
#include <cstddef>
#include <utility>

int main() {
    // allocate_bytes 可能不是元素宽度的整数倍；move 后必须保留原始分配字节数。
    ctpwa::DevBuffer<double> a;
    a.allocate_bytes(0, 5);
    assert(a);
    assert(a.bytes() == 5);
    assert(a.device() == 0);

    ctpwa::DevBuffer<double> b(std::move(a));
    assert(!a);
    assert(b);
    assert(b.bytes() == 5);   // 回归：移动构造曾丢 bytes_，退化为 0
    assert(b.device() == 0);

    ctpwa::DevBuffer<double> c;
    c = std::move(b);
    assert(!b);
    assert(c);
    assert(c.bytes() == 5);
    c.reset();
    assert(!c);
    assert(c.bytes() == 0);
    assert(c.device() == -1);
    c.reset();               // reset 必须幂等

    // 零长度表沿用旧 cudaMalloc(0) 非空约定，并保留一个元素的分配宽度。
    ctpwa::DevBuffer<double> z;
    z.allocate(0, 0);
    assert(z);
    assert(z.size() == 0);
    assert(z.bytes() == sizeof(double));
    z.disarm();
    assert(!z);
    assert(z.device() == -1);

    // MultiGpuStorage 的视图与所有权分离：move 后仍能按卡访问，clear 后视图失效。
    ctpwa::MultiGpuStorage<double> mg;
    mg.resize(2);
    for (int g = 0; g < 2; ++g) {
        mg.at(g).allocate(g, 3);
        assert(mg[g] != nullptr);
        assert(mg.data()[g] == mg[g]);
    }
    ctpwa::MultiGpuStorage<double> moved(std::move(mg));
    assert(mg.empty());
    assert(moved.size() == 2);
    for (int g = 0; g < 2; ++g) {
        assert(moved[g] != nullptr);
        assert(moved.data()[g] == moved[g]);
    }
    moved.clear();
    assert(moved.empty());
    assert(moved.data().empty());

    return 0;
}
