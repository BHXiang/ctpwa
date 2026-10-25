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

// 设备类型
//   GPU: CUDA 设备（CTPWA_BACKEND_CUDA 构建）
//   CPU: 伪设备（CTPWA_BACKEND_CPU 构建）。只有一个，index==0，
//        "显存"用可用 RAM 替代，容量检查语义不变。
enum class DeviceType {
    GPU = 0,
    CPU = 1
};

// Complex 精度模式（预留 double 精度，当前仅 float）
enum class ComplexPrecision {
    Float = 0,   // ctComplex (float2), 8 bytes — 当前行为
    Double = 1   // cuDoubleComplex (double2), 16 bytes — 预留
};

// 单个设备的静态信息
struct DeviceInfo {
    int    index;              // 设备索引
    std::string name;          // 型号名，如 "NVIDIA GeForce RTX 3060" / "CPU (16 threads)"
    int    cc_major;           // 算力主版本 (sm_86 → 8)；CPU 为 0
    int    cc_minor;           // 算力次版本 (sm_86 → 6)；CPU 为 0
    size_t total_memory;       // 总显存 (bytes)；CPU 为总 RAM
    size_t free_memory;        // detect() 时的空闲显存快照 (bytes)；CPU 为可用 RAM
    bool   available;          // 是否可达
    DeviceType type = DeviceType::GPU;  // 设备类型（CPU 后端为 CPU）
};

class DeviceManager {
public:
    // ============================================================
    // 设备检测
    // ============================================================

    // 枚举设备。
    //   CTPWA_BACKEND_CUDA: 枚举 GPU（GPU 为 0 时 devices_ 为空）
    //   CTPWA_BACKEND_CPU:  固定一个 CPU 伪设备（index 0）
    void detect();

    // 当前构建的后端类型（由构建宏决定，与是否有卡无关）
    DeviceType backend() const;
    bool isCpu() const { return backend() == DeviceType::CPU; }

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

    // 运行时存储档位说明（由 analysis 初始化时注入; print() 展示）:
    // 例: "A float2(8B) | dF double2(16B) | 核心 double (precision: auto)"
    void setStorageNote(const std::string& n) { storage_note_ = n; }
    const std::string& storageNote() const { return storage_note_; }

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
    // 最近一次 checkCapacity 得到的每 GPU 预计占用（bytes, 与 devices_ 同下标;
    // 为空表示还没估算过——print() 此时回退显示空闲量）。checkCapacity 是
    // const 接口, 故为 mutable。
    mutable std::vector<double> estimated_usage_;
    ComplexPrecision complex_precision_ = ComplexPrecision::Float;
    std::string storage_note_;   // 运行时存储档位（analysis 注入; 空则不打印）
};
