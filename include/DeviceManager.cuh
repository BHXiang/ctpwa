// DeviceManager.cuh — 统一设备管理：GPU 枚举、属性、内存估算与容量预检
//
// 设计目标:
//   1. 集中管理 GPU 个数、型号、算力、显存信息（避免散落的 cudaGetDeviceCount）
//   2. 根据事件数/振幅数等估算各 GPU 缓冲区内存，预检输入数据能否被设备承载
//   3. 预留 CPU 后端（DeviceType::CPU）与双精度 complex（ComplexPrecision::Double）
//      的扩展点，当前不实现
//
// 用法:
//   DeviceManager devmgr;
//   devmgr.detect();                    // 初始化时枚举设备
//   devmgr.print();                     // 打印设备列表
//   auto result = devmgr.checkCapacity(events_per_gpu, n_amps, n_pol, ...);
//   if (result.overall == DeviceManager::CapacityStatus::FAIL) { ... 中止 ... }

#pragma once

#include <string>
#include <vector>
#include <utility>
#include <cstddef>

// 设备类型（预留 CPU 后端，当前仅 GPU）
enum class DeviceType {
    GPU = 0,
    CPU = 1   // 预留：CPU 计算能力尚未实现
};

// Complex 精度模式（预留 double 精度，当前仅 float）
enum class ComplexPrecision {
    Float = 0,   // cuComplex (float2), 8 bytes — 当前行为
    Double = 1   // cuDoubleComplex (double2), 16 bytes — 预留
};

// 单个设备的静态信息
struct DeviceInfo {
    int    index;              // 设备索引
    std::string name;          // 型号名，如 "NVIDIA GeForce RTX 3060"
    int    cc_major;           // 算力主版本 (sm_86 → 8)
    int    cc_minor;           // 算力次版本 (sm_86 → 6)
    size_t total_memory;       // 总显存 (bytes)
    size_t free_memory;        // detect() 时的空闲显存快照 (bytes)
    bool   available;          // 是否可达
};

class DeviceManager {
public:
    // ============================================================
    // 设备检测
    // ============================================================

    // 枚举所有 GPU，记录型号/算力/显存。GPU 为 0 时 devices_ 为空。
    // 预留：未来 CPU 后端在此追加 CPU 设备到 types_。
    void detect();

    int numDevices() const { return static_cast<int>(devices_.size()); }
    bool hasDevices() const { return !devices_.empty(); }
    const DeviceInfo& device(int i) const;
    const DeviceInfo& primary() const;   // 主设备 = 设备 0

    // ============================================================
    // 设备切换（带错误检查的 cudaSetDevice 包装）
    // ============================================================

    bool setDevice(int i) const;
    int  currentDevice() const;

    // ============================================================
    // Complex 精度模式（预留 double 精度切换）
    // ============================================================

    void setComplexPrecision(ComplexPrecision p) { complex_precision_ = p; }
    ComplexPrecision complexPrecision() const { return complex_precision_; }
    // 单个 complex 值的内存占用: Float→8B, Double→16B
    size_t complexSize() const;

    // ============================================================
    // 显示
    // ============================================================

    void print() const;

    // ============================================================
    // 内存估算与容量预检
    // ============================================================

    struct MemEstimate {
        double total_bytes_gpu;   // 单 GPU 峰值总内存（事件相关部分 + 固定部分）
        double total_bytes_other; // 跨 GPU 分配（仅主 GPU 等，不乘事件数）
        std::vector<std::pair<std::string, double>> breakdown;  // 明细
    };

    // 估算单个 GPU 的内存需求。
    //   n_events_per_gpu: 该 GPU 的事件数（data + phsp 分开后按 GPU 分到的最大者）
    //   n_amplitudes: 振幅总数
    //   n_polar: 极化数
    //   n_slcombs: 单链 SL 组合数（d_slamps 用）
    //   n_particles: 末态粒子数（d_momenta 用）
    //   has_bkg: 是否有背景样本
    //   n_theta: 自由共振态参数数（d_grad_res 用）
    MemEstimate estimate(int n_events_per_gpu, int n_amplitudes,
                         int n_polar, int n_slcombs, int n_particles,
                         bool has_bkg, int n_theta = 0) const;

    enum class CapacityStatus { OK = 0, WARN = 1, FAIL = 2 };

    struct CapacityResult {
        CapacityStatus overall = CapacityStatus::OK;
        int    failing_device = -1;      // FAIL 时第一个失败的 GPU
        std::string failing_buffer;      // FAIL 时具体的 buffer 名
        double required_bytes = 0.0;     // FAIL 时该 GPU 需要的字节
        double available_bytes = 0.0;    // FAIL 时该 GPU 的可用显存
    };

    // 预检：给定的每-GPU 事件分布能否被设备集承载。
    //   events_per_gpu: 每个 GPU 的事件数（data + phsp 分开后的最大值）
    //   warn_fraction: 占用率超过该比例 → WARN（默认 80%）
    //   任一 GPU 需求超过可用显存 → FAIL
    CapacityResult checkCapacity(const std::vector<int>& events_per_gpu,
                                  int n_amplitudes, int n_polar, int n_slcombs,
                                  int n_particles, bool has_bkg, int n_theta = 0,
                                  double warn_fraction = 0.8) const;

private:
    std::vector<DeviceInfo> devices_;
    ComplexPrecision complex_precision_ = ComplexPrecision::Float;
};
