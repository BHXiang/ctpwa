// DeviceManager.cu — 设备管理实现
//
// 注意（多 GPU 常见 bug）:
//   遍历每个 GPU 执行操作（cudaSetDevice(i)）后，必须恢复主设备
//   （cudaSetDevice(primary)），否则后续操作会在错误的设备上执行。
//   DeviceManager::detect() 已处理该问题；调用方也应通过 setDevice()
//   包装而非直接调用 cudaSetDevice。

#include <DeviceManager.cuh>
#include <ComplexType.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace {

// 格式化字节数: 123456789 → "117.7 MiB"
std::string formatBytes(double bytes)
{
    char buf[64];
    if (bytes >= 1e9)
        snprintf(buf, sizeof(buf), "%.2f GiB", bytes / 1e9);
    else if (bytes >= 1e6)
        snprintf(buf, sizeof(buf), "%.2f MiB", bytes / 1e6);
    else if (bytes >= 1e3)
        snprintf(buf, sizeof(buf), "%.2f KiB", bytes / 1e3);
    else
        snprintf(buf, sizeof(buf), "%.0f B", bytes);
    return buf;
}

}  // namespace

// ============================================================
// 设备检测
// ============================================================

void DeviceManager::detect()
{
    devices_.clear();

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        std::fprintf(stderr,
            "DeviceManager: 未检测到 CUDA 设备 (%s)。"
            "CPU 后端尚未实现，无法继续。\n",
            cudaGetErrorString(err));
        return;  // devices_ 为空
    }

    int primary = 0;
    cudaGetDevice(&primary);

    devices_.reserve(count);
    for (int i = 0; i < count; ++i) {
        // 切到目标设备读取属性 + 空闲显存
        if (cudaSetDevice(i) != cudaSuccess) {
            DeviceInfo d;
            d.index = i;
            d.name = "(unavailable)";
            d.available = false;
            devices_.push_back(d);
            continue;
        }

        cudaDeviceProp prop;
        std::memset(&prop, 0, sizeof(prop));
        cudaError_t perr = cudaGetDeviceProperties(&prop, i);

        DeviceInfo d;
        d.index = i;
        d.available = (perr == cudaSuccess);
        if (perr == cudaSuccess) {
            d.name = prop.name;
            d.cc_major = prop.major;
            d.cc_minor = prop.minor;
            d.total_memory = prop.totalGlobalMem;
        } else {
            d.name = "(props unavailable)";
        }

        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            d.free_memory = free_mem;
        else
            d.free_memory = 0;

        devices_.push_back(d);
    }

    // ⚠️ 关键：遍历后恢复主设备，避免后续操作跑错设备
    cudaSetDevice(primary);
}

const DeviceInfo& DeviceManager::device(int i) const
{
    static const DeviceInfo empty = {-1, "", 0, 0, 0, 0, false};
    if (i < 0 || i >= (int)devices_.size())
        return empty;
    return devices_[i];
}

const DeviceInfo& DeviceManager::primary() const
{
    return device(0);
}

// ============================================================
// 设备切换
// ============================================================

bool DeviceManager::setDevice(int i) const
{
    if (i < 0 || i >= (int)devices_.size()) {
        std::fprintf(stderr, "DeviceManager: 设备索引 %d 越界 (共 %d 个)\n",
                     i, (int)devices_.size());
        return false;
    }
    cudaError_t err = cudaSetDevice(i);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "DeviceManager: cudaSetDevice(%d) 失败: %s\n",
                     i, cudaGetErrorString(err));
        return false;
    }
    return true;
}

int DeviceManager::currentDevice() const
{
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

// ============================================================
// Complex 精度模式
// ============================================================

size_t DeviceManager::complexSize() const
{
    // 编译时精度决定（ctComplex = cuComplex 8B / cuDoubleComplex 16B），
    // ComplexPrecision 枚举保留为运行时查询接口
    return sizeof(ctComplex);
}

// ============================================================
// 显示
// ============================================================

void DeviceManager::print() const
{
    std::printf("DeviceManager: %d 个设备\n", (int)devices_.size());
    for (const auto& d : devices_) {
        std::printf("  [%d] %s (sm_%d%d), 显存 %s / %s%s\n",
                    d.index, d.name.c_str(), d.cc_major, d.cc_minor,
                    formatBytes((double)d.free_memory).c_str(),
                    formatBytes((double)d.total_memory).c_str(),
                    d.available ? "" : " [不可用]");
    }
    std::printf("  complex 精度: %s (%zu B/值)\n",
                (complex_precision_ == ComplexPrecision::Double) ? "double" : "float",
                complexSize());
}

// ============================================================
// 内存估算
// ============================================================

DeviceManager::MemEstimate DeviceManager::estimate(
    int n_events_per_gpu, int n_amplitudes,
    int n_polar, int n_slcombs, int n_particles,
    bool has_bkg, int n_theta) const
{
    MemEstimate m;
    m.total_bytes_gpu = 0.0;
    m.total_bytes_other = 0.0;

    const double ne = (double)n_events_per_gpu;
    const double cs = (double)complexSize();       // complex 值大小 (8/16)
    const double sl = 16.0;                        // thrust::complex<double> 固定 16B
    const double db = 8.0;                         // double

    auto add = [&](const char* name, double bytes) {
        m.breakdown.emplace_back(name, bytes);
    };

    // ---- 事件相关（per GPU，用最大样本的事件数估算峰值）----
    double amp = ne * n_amplitudes * n_polar * cs;       // d_all_amplitudes
    double slamp = ne * n_polar * n_slcombs * sl;        // d_slamps
    double mom = ne * n_particles * 32.0;                // d_momenta (LorentzVector = 4×double)
    double T = ne * n_polar * cs;                        // d_T
    double w_data = ne * n_polar * cs;                   // data w buffer
    double w_bkg = has_bkg ? ne * n_polar * cs : 0.0;    // bkg w buffer
    double weights = ne * db * 3.0;                      // data/phsp/bkg 权重

    add("d_all_amplitudes", amp);
    add("d_slamps", slamp);
    add("d_momenta", mom);
    add("d_T", T);
    add("data w buffer", w_data);
    if (has_bkg) add("bkg w buffer", w_bkg);
    add("weights x3", weights);

    m.total_bytes_gpu = amp + slamp + mom + T + w_data + w_bkg + weights;

    // ---- 固定部分（每 GPU 一份的小 buffer + 主 GPU 独有）----
    double grad_buf = n_amplitudes * cs;                 // d_grad_global / d_grad_buf
    double grad_res = n_theta * db;                      // d_grad_res
    add("grad buffers", grad_buf + grad_res);
    m.total_bytes_gpu += grad_buf + grad_res;

    // ---- 仅主 GPU（跨 GPU 分配）----
    double phsp_mat = (double)n_amplitudes * n_amplitudes * cs;  // d_phsp_matrix
    add("d_phsp_matrix (主 GPU)", phsp_mat);
    m.total_bytes_other = phsp_mat;

    return m;
}

DeviceManager::CapacityResult DeviceManager::checkCapacity(
    const std::vector<int>& events_per_gpu,
    int n_amplitudes, int n_polar, int n_slcombs, int n_particles,
    bool has_bkg, int n_theta, double warn_fraction) const
{
    CapacityResult result;

    if (devices_.empty()) {
        result.overall = CapacityStatus::FAIL;
        result.failing_device = -1;
        result.failing_buffer = "(no devices)";
        return result;
    }

    // 每个 GPU 的事件数用 data/phsp 的最大者估算峰值内存
    // （调用方传入的是"该 GPU 分配到的最大样本事件数"）
    for (size_t i = 0; i < devices_.size() && i < events_per_gpu.size(); ++i) {
        const DeviceInfo& d = devices_[i];
        if (!d.available) continue;

        MemEstimate m = estimate(events_per_gpu[i], n_amplitudes, n_polar,
                                 n_slcombs, n_particles, has_bkg, n_theta);
        double need = m.total_bytes_gpu + m.total_bytes_other;

        // 可用显存：detect 时的空闲快照。若快照为 0（未知），用总显存兜底
        double avail = (d.free_memory > 0) ? (double)d.free_memory
                                           : (double)d.total_memory;

        if (need > avail) {
            // 找出超限的具体 buffer（用于报错信息）
            const char* worst = "";
            double worst_need = 0.0;
            for (const auto& [name, bytes] : m.breakdown) {
                if (bytes > worst_need) { worst_need = bytes; worst = name.c_str(); }
            }
            result.overall = CapacityStatus::FAIL;
            result.failing_device = (int)i;
            result.failing_buffer = worst;
            result.required_bytes = need;
            result.available_bytes = avail;
            return result;
        }

        if (need > avail * warn_fraction) {
            result.overall = CapacityStatus::WARN;   // 继续检查后续 GPU，保留最差状态
        }
    }

    return result;
}
