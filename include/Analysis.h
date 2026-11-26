// include/Analysis.h — `analysis` 类的定义（P4 迁移中间态）
//
// 历史：`src/Analysis.cu` 是一个 7624 行的文件，`class analysis` 的所有方法都写成
// 类内内联定义，而且没有头文件——main.cu 只能靠 `#include "Analysis.cu"` 把整个
// god class 文本内联进来，任何方法都无法拆到别的翻译单元去。
//
// P4 的第一步是**让它成为可被单独包含的类型**：本头保留原文件全部内容（工具函数 +
// 类定义），`src/Analysis.cu` 只剩一行 include。因为仍是同一个 TU、同一份代码，
// 行为完全不变；换来的是：
//
//   * 新模块可以 `#include <Analysis.h>` 看到 `analysis` 的声明；
//   * 后续每搬走一个方法，就把它从"类内定义"改成"类内声明 + 新 TU 定义"，
//     god class 可以**逐方法**缩小，而不必一次性重写 6000 行。
//
// 目标态（P4 完成时）：本文件只剩成员声明与少量 inline，实现分散到 io/ fit/ output/。
#pragma once

// #include <pybind11/pybind11.h>
#include <chrono>
#include <cstdio>
#include "ComplexType.h"
#include "PrecCpx.h"
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <torch/extension.h>
#include "Backend.h"  // CTPWA_KERNEL / CTPWA_LAUNCH / ctpwa:: 原语
#include <DataReader.h>  // 数据文件读取（P4：从本文件抽出）
#include "DeviceGuard.h"  // ctpwa::ensure_device / DeviceGuard
#include <MatrixStats.h> // ctpwa::couplingCovariance（P4 抽出）
#include <PhspGram.h>   // ctpwa::phspMatrixFromGram（P4 抽出）
#include <MatrixErrors.h> // ctpwa::computeMatrixErrors（P4 抽出）
#include <SLAmpsIO.h>   // ctpwa::slAmpsTensor / saveSLAmpsToFile（P4 抽出）
#include <MixedPrecision.h> // upcastAmpSegToDouble（P4：从本头上移）
#include <TruthIntegrals.h> // ctpwa::computeTruthIntegrals（P4 抽出）
#include <BatchIntegrals.h> // ctpwa::computeFileBatchIntegrals（P4 抽出）
#include <FitFractions.h>   // fitFractions / efficiency / matrixForSample（P4 抽出）
#include <Polarization.h>   // initializePolarization（P4 抽出）
#include <HessianContrib.h> // hessianSegmentContrib（P4 抽出）
#include <CouplingHessian.h> // computeCouplingHessian（P4 抽出）
#include <FullHessian.h>    // getHessian（P4 抽出）

// ---------------------------------------------------------------------------
// 后端无关的设备 helper：CPU 后端下所有张量都在 CPU 上，CUDA 后端保持原语义。
// 目的是让 7000 行的编排代码只有这一处知道"当前是哪个后端"。
// ---------------------------------------------------------------------------
#include <BackendHelpers.h>   // ctpwa_dev / ctpwa_is_cuda_like / makeScopeExit / CUDA_CHECK

#include <unordered_set>
#include <vector>

#include <AmpGen.cuh>
#include <DeviceManager.cuh>
// #include <ComputeGrad.cuh>
#include <ComputeNLL.cuh>
#include <ComputeResults.cuh>
#include <PhaseSpace.cuh>
#include <Config.cuh>
#include <Figure.cuh>
#include <ComputeHessian.cuh>
#include <Info.cuh>
#include <Parameters.cuh>
#include <ComputeBF.cuh>

#include <TFile.h>
#include <TLorentzVector.h>
#include <TMatrixD.h>
#include <TObjString.h>
#include <TTree.h>

#include <utility>


//////////////////////////////////////////////
////////////////////////////////////////
// 诊断辅助：2J+1 记号 -> "J^P" 字符串（如 "3/2+"）
static std::string jpLabel(int two_j_plus_1, int parity)
{
    int two_j = two_j_plus_1 - 1;
    std::string j;
    if (two_j % 2 == 0)
        j = std::to_string(two_j / 2);
    else
        j = std::to_string(two_j) + "/2";
    return j + (parity > 0 ? "+" : "-");
}



// 读取权重文件的全部数值到主机内存（不做 GPU 切分）。
// 支持三种写法（与 data_weights / bkg_weights / phsp_weights 一致）:
//   [<常数>]                          → 展开成 expected 个相同权重
//   [dat, <file>]                     → 每行一个权重
//   [ROOT, <file>, <tree>, <branch>]  → 每事件一个权重
// expected < 0 表示长度未知（仅 dat/ROOT 可省略；常数形式必须给）。
// 失败返回空 vector，由调用方决定是报错还是降级。


inline std::vector<double*> readWeightsFromFile(const std::vector<std::string>& fileinfo, const std::vector<int>& events_per_gpu)
{
    int total_weights = 0;
    for (int ev : events_per_gpu)
    {
        total_weights += ev;
    }

    // 读取所有权重到主机内存
    std::vector<double> weights = readWeightValuesFromFile(fileinfo, total_weights);
    if (weights.empty() && total_weights > 0) {
        // 读取失败（文件缺失/类型未知/空文件）: 保持旧行为——调用方按"无权重"
        // 处理。绝不在这里补零：全零权重会让 NLL 恒为 0，拟合看起来"收敛"。
        std::cerr << "Warning: failed to read weights; falling back to equal weights."
                  << std::endl;
        return {};
    }

    // 检查权重数量是否匹配
    if (weights.size() != static_cast<size_t>(total_weights))
    {
        std::cerr << "Error: Weights size " << weights.size() << " does not match total events " << total_weights << std::endl;
        // 不足补零，多余截断（保持旧行为：权重文件长度错位时不静默换布局）
        if (weights.size() < static_cast<size_t>(total_weights))
        {
            std::cerr << "Warning: Not enough weights in file. Padding with zeros." << std::endl;
            weights.resize(total_weights, 0.0);
        }
        else
        {
            std::cerr << "Warning: Too many weights in file. Truncating." << std::endl;
            weights.resize(total_weights);
        }
    }

    // 为每个GPU分配设备内存并复制数据
    std::vector<double*> d_weights_ptrs(events_per_gpu.size(), nullptr);
    size_t offset = 0;
    for (size_t i = 0; i < events_per_gpu.size(); ++i)
    {
        int n_weights = events_per_gpu[i];
        if (n_weights <= 0)
        {
            // 该GPU没有权重，保持nullptr
            continue;
        }

        // 设置当前GPU设备
        cudaError_t cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "Error: cudaSetDevice failed for GPU " << i << ": " << cudaGetErrorString(cudaStatus) << std::endl;
            // 清理已分配的内存
            for (size_t j = 0; j < i; ++j)
            {
                if (d_weights_ptrs[j] != nullptr)
                {
                    cudaSetDevice(j);
                    cudaFree(d_weights_ptrs[j]);
                }
            }
            return {};
        }

        // 分配设备内存
        double* d_weights_i = nullptr;
        cudaStatus = cudaMalloc(&d_weights_i, n_weights * sizeof(double));
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "Error: cudaMalloc failed for GPU " << i << ": " << cudaGetErrorString(cudaStatus) << std::endl;
            // 清理已分配的内存
            for (size_t j = 0; j < i; ++j)
            {
                if (d_weights_ptrs[j] != nullptr)
                {
                    cudaSetDevice(j);
                    cudaFree(d_weights_ptrs[j]);
                }
            }
            return {};
        }

        // 复制数据到设备
        cudaStatus = cudaMemcpy(d_weights_i, weights.data() + offset, n_weights * sizeof(double),
            cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "Error: cudaMemcpy failed for GPU " << i << ": " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(d_weights_i);
            // 清理已分配的内存
            for (size_t j = 0; j < i; ++j)
            {
                if (d_weights_ptrs[j] != nullptr)
                {
                    cudaSetDevice(j);
                    cudaFree(d_weights_ptrs[j]);
                }
            }
            return {};
        }

        d_weights_ptrs[i] = d_weights_i;
        offset += n_weights;
    }

    return d_weights_ptrs;
}

inline std::vector<LorentzVector*> convertToLorentzVectors(
    const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta,
    const std::map<std::string, int>& particleToIndex)
{
    std::vector<LorentzVector*> d_momenta(finalMomenta.size(), nullptr);
    // cudaMalloc(&d_momenta, sizeof(DeviceMomenta));
    // for (size_t i = 0; i < finalMomenta.size(); ++i)
    // {
    //     DeviceMomenta* d_momenta_i;
    //     cudaMalloc(&d_momenta_i, sizeof(DeviceMomenta));
    //     d_momenta.push_back(d_momenta_i);
    // }

    for (size_t i = 0; i < finalMomenta.size(); ++i)
    {
        cudaSetDevice(i);

        // 获取事件数量和粒子数量
        int n_events = finalMomenta[i].begin()->second.size();
        int n_particles = particleToIndex.size();

        // 在主机端分配所有粒子的四动量数组
        std::vector<LorentzVector> host_momenta(n_events * n_particles);
        std::fill(host_momenta.begin(), host_momenta.end(), LorentzVector());

        // 创建粒子计算状态标记
        std::vector<bool> particle_calculated(n_particles, false);

        // 第一步：将末态粒子的四动量复制到对应位置
        for (const auto& particle_momenta : finalMomenta[i])
        {
            const std::string& particle_name = particle_momenta.first;
            const std::vector<LorentzVector>& momenta_vec = particle_momenta.second;

            auto it = particleToIndex.find(particle_name);
            if (it != particleToIndex.end())
            {
                int particle_idx = it->second;
                for (int event_idx = 0; event_idx < n_events; ++event_idx)
                {
                    host_momenta[event_idx * n_particles + particle_idx] = momenta_vec[event_idx];
                }
                particle_calculated[particle_idx] = true;
            }
        }

        // 第二步：将数据复制到设备
        // 在设备端分配四动量数组
        LorentzVector* d_momenta_i;
        cudaMalloc(&d_momenta_i, host_momenta.size() * sizeof(LorentzVector));
        cudaMemcpy(d_momenta_i, host_momenta.data(), host_momenta.size() * sizeof(LorentzVector), cudaMemcpyHostToDevice);

        d_momenta[i] = d_momenta_i;
    }

    return d_momenta;
}


inline std::vector<std::map<std::string, std::vector<LorentzVector>>> mergeMaps(
    const std::vector<std::map<std::string, std::vector<LorentzVector>>>& maps, std::vector<std::vector<int>>& events)
{
    if (maps.empty())
        return {};

    // 结果 map
    std::vector<std::map<std::string, std::vector<LorentzVector>>> result;

    // 跟踪每种数据类型在各GPU间的全局偏移
    int n_types = maps.size();
    std::vector<int> global_offset(n_types, 0);

    for (size_t gpu = 0; gpu < events.size(); ++gpu)
    {
        std::map<std::string, std::vector<LorentzVector>> tmp;

        // 为当前GPU切出每种数据类型的对应事件
        for (int t = 0; t < n_types; ++t) {
            int n_ev = events[gpu][t];
            if (n_ev <= 0) continue;
            const auto& m = maps[t];
            for (const auto& [key, vec] : m) {
                auto& target = tmp[key];
                target.insert(target.end(),
                    vec.begin() + global_offset[t],
                    vec.begin() + global_offset[t] + n_ev);
            }
            global_offset[t] += n_ev;
        }

        result.push_back(tmp);
    }

    return result;
}

//////////////////////////////////////////////////////////////
// Replace ROOT-unsafe chars in object names
static std::string sanitizeROOTName(const std::string& s) {
    // → is UTF-8 \xE2\x86\x92 (3 bytes), iterate as bytes
    std::string r = s;
    for (size_t i = 0; i < r.size(); ) {
        unsigned char c = static_cast<unsigned char>(r[i]);
        if (c == 0xE2 && i + 2 < r.size() &&
            static_cast<unsigned char>(r[i+1]) == 0x86 &&
            static_cast<unsigned char>(r[i+2]) == 0x92) {
            r.replace(i, 3, "_"); continue;  // →
        }
        // if (!std::isalnum(c) && c != '_' && c != '.' && c != '-') {
        if (!std::isalnum(c) && c != '_') {
            r[i] = '_';
        }
        ++i;
    }
    return r;
}

//////////////////////////////////////////////////////////////
/// Hessian fast-path switch: CTPWA_HESS_FAST=0 → legacy CGEMM path
///////////////////////////////////////////////////////////////

// ----------------------------------------------------------------
// phsp 流式模式辅助 kernel（Constraints.free_phsp_amplitudes）
// ----------------------------------------------------------------
// 批振幅 float→double 转换（供 cublasZgemm 累加 double phsp 矩阵）
static CTPWA_KERNEL void castPhspBatchToDoubleKernel(
    const ctComplex* __restrict__ src, cuDoubleComplex* __restrict__ dst, int total)
{
    const int i = CTPWA_TID();
    if (i >= total) return;
    dst[i] = make_cuDoubleComplex((double)src[i].x, (double)src[i].y);
}

// 双精度 phsp Gram 的逐事例权重缩放（与 scalePhspAmpsKernel 同语义）：
//   amp[e,p,a] *= sqrt(w[e] / W_total)
// 流式 double 矩阵必须和 float32 矩阵用同一归一化，否则 double 矩阵少除 W_total，
// 所有读它的消费者（W_rec、writeResult 的 interference_rec、误差矩阵）会大 N 倍。
static CTPWA_KERNEL void scalePhspAmpsDoubleKernel(
    cuDoubleComplex* __restrict__ d_amp, const double* __restrict__ d_weights,
    int nEvents, int nPolar, int nAmp, double inv_W_total)
{
    const std::ptrdiff_t idx = (std::ptrdiff_t)CTPWA_TID();
    const std::ptrdiff_t total = (std::ptrdiff_t)nEvents * nPolar * nAmp;
    if (idx >= total) return;
    const int evt = (int)((idx / nAmp) / nPolar);
    const double scale = (d_weights != nullptr)
        ? sqrt(d_weights[evt] * inv_W_total)
        : sqrt(inv_W_total);
    d_amp[idx].x *= scale;
    d_amp[idx].y *= scale;
}
static void scalePhspAmpsDouble(cuDoubleComplex* d_amp, const double* d_w,
    int nEvents, int nPolar, int nAmp, double inv_W_total)
{
    const std::ptrdiff_t total = (std::ptrdiff_t)nEvents * nPolar * nAmp;
    if (total <= 0) return;
    const int grid = (int)((total + 255) / 256);
    CTPWA_LAUNCH(scalePhspAmpsDoubleKernel, grid, 256, 0, d_amp, d_w,
        nEvents, nPolar, nAmp, inv_W_total);
}

// phsp θ 梯度: S_ev,p ← S_ev,p · (w_ev / W_total)
// （computeResonanceGradient 把 S 当"逐事件权重"消费；先乘 w/W 即得
//  Σ (w/W) ∂f/∂θ，配合 phsp_sign=-N_eff/phsp_factor）
static CTPWA_KERNEL void scalePhspSByWeightKernel(
    ctComplex* __restrict__ d_S, const double* __restrict__ d_w,
    int nEvents, int nPolar, double inv_W)
{
    const int evt = CTPWA_TID();
    if (evt >= nEvents) return;
    const double s = (d_w ? d_w[evt] : 1.0) * inv_W;
    for (int p = 0; p < nPolar; ++p) {
        const int idx = evt * nPolar + p;
        d_S[idx] = ctMake(d_S[idx].x * s, d_S[idx].y * s);
    }
}

// y += x（cuDoubleComplex 版 axpy）
static CTPWA_KERNEL void axpyZComplexKernel(
    cuDoubleComplex* __restrict__ y, const cuDoubleComplex* __restrict__ x,
    std::ptrdiff_t n)
{
    const std::ptrdiff_t i = (std::ptrdiff_t)CTPWA_TID();
    if (i >= n) return;
    y[i].x += x[i].x;
    y[i].y += x[i].y;
}
static void axpyZComplex(cuDoubleComplex* y, const cuDoubleComplex* x, std::ptrdiff_t n)
{
    const int grid = (int)((n + 255) / 256);
    CTPWA_LAUNCH(axpyZComplexKernel, grid, 256, 0, y, x, n);
}
// 带诊断的 cudaMalloc：OOM 时直接报"哪个缓冲/多大/哪张卡/还剩多少"，
// 而不是把 cudaMalloc 的失败结果（nullptr/未定义指针）喂进 kernel，最后报成
// 一句无从下手的 "illegal memory access"。

//////////////////////////////////////////////////////////////
/// NLLFunction 类定义
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
class NLLFunction : public torch::autograd::Function<NLLFunction>
{
public:
    // 多 GPU 前向传播：输入统一 params = [real(v_0..n-1), imag(v_0..n-1), θ_0..P-1] float64
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor params_tensor,           // 统一参数 [2*nFreeVector + nFreeTheta] float64
        Parameters* params_mgr,                // 参数管理器（含约束 & 维度信息 & vspace flag）
        std::vector<ctComplex*>& d_all_amplitudes_list,
        AmpCalc* amp_calc,                     // 共振态管理器
        ctComplex* d_phsp_matrix_,
        cuDoubleComplex* d_phsp_matrix_double_,   // 非空 = phsp 流式模式（phsp_sum 走 double 矩阵）
        const std::vector<std::vector<int>>& events_list,
        const std::vector<std::vector<int>>& events_offsets_list,
        const std::vector<std::vector<int>>& amp_offsets_list,
        std::vector<double*>& d_data_weights_list,
        std::vector<double*>& d_phsp_weights_list,
        std::vector<double*>& d_bkg_weights_list,
        double bkg_integral_,
        int n_amplitudes_,
        int n_polar_)
    {
        TORCH_CHECK(params_mgr && params_mgr->initialized(), "Parameters not initialized");
        bool tprof = getenv("CTPWA_PROF") != nullptr;
        auto tF0 = std::chrono::high_resolution_clock::now();
        // phsp 梯度段的持久化缓冲（懒分配，避免每次 forward 的 cublasCreate/cudaMalloc）
        static std::vector<cublasHandle_t> s_phsp_handle;   // phsp CGEMV（θ 段 S 也用）
        static std::vector<ctComplex*> s_d_S_bufs;          // θ 段 S = A^T·v
        static std::vector<int> s_dS_alloc_sz;
        static std::vector<ctComplex*> s_d_P_part;          // free-θ: 每 GPU 局部 d_P_vec
        static std::vector<int> s_d_P_part_sz;
        static ctComplex* s_d_P_peer = nullptr;             // free-θ: 主 GPU 规约临时
        static int s_d_P_peer_sz = 0;

        // 0. 拆分 params → vector + theta
        torch::Tensor vector, theta;
        if (params_mgr->hasCouplingMatrix()) {
            const auto& cm = params_mgr->couplingMatrix();
            int ncf = cm.n_free;
            int na  = cm.n_amps;
            int nt  = params_mgr->nFreeTheta();

            // params_tensor: [Re_p, Im_p, θ] (2*ncf+nt) → extend → [Re_v, Im_v, θ] (2*na+nt)
            auto ext = torch::empty({2 * na + nt},
                torch::TensorOptions().dtype(torch::kFloat64).device(params_tensor.device()));
            params_mgr->extendCouplingParams(
                params_tensor.data_ptr<double>(), ext.data_ptr<double>(), ncf, nt);

            // Save original params for backward gradient transform
            ctx->saved_data["original_params"] = params_tensor;

            vector = torch::complex(
                ext.slice(0, 0, na).to(TORCH_FLOAT),
                ext.slice(0, na, 2 * na).to(TORCH_FLOAT));
            theta = (nt > 0) ? ext.slice(0, 2 * na, 2 * na + nt) : torch::Tensor();
        } else {
            std::tie(vector, theta) = params_mgr->splitParams(params_tensor);
        }

        int num_gpus = d_all_amplitudes_list.size();
        TORCH_CHECK(num_gpus > 0, "No GPUs provided");
        TORCH_CHECK(ctpwa_is_cuda_like(vector), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");

        // 1. 将 vector 扩展向量
        torch::Tensor extended_vector = params_mgr->extendVector(vector, vector.device());
        int extended_n_gls = extended_vector.numel();
        // torch 在 CPU 张量上返回 -1（没有设备索引）。这里统一归一化到 0：
        // 该值随后被当作设备号使用（ensure_device / cudaMemcpy / peer copy）。
        const int primary_dev = ctpwa_is_cuda_like(vector) ? vector.get_device() : 0;

        // 2. 将extended_vector分发到每个GPU（实际拷贝）
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < num_gpus; ++i) {
            extended_vec_per_gpu.push_back(
                extended_vector.to(ctpwa_dev(i)));
        }

#if defined(CTPWA_BACKEND_CPU)
        if (getenv("CTPWA_CPU_DEBUG")) {
            auto hv = extended_vector.to(torch::kCPU).contiguous();
            const auto* zp = hv.data_ptr<c10::complex<double>>();
            const int n = std::min<int>(6, (int)hv.numel());
            std::printf("[CPU-DBG] extended_vector n=%d:", (int)hv.numel());
            for (int i = 0; i < n; ++i)
                std::printf(" (%.6g,%.6g)", zp[i].real(), zp[i].imag());
            std::printf("\n");
        }
#endif
        auto tF1 = std::chrono::high_resolution_clock::now();  // extend+分发完成
        auto t2_reamp = tF1, t6_nll = tF1, t7_rg = tF1;
        auto tF2 = tF1, tF3 = tF1, tF4 = tF1;

        // === 新增：Step 2.5: 更新振幅 & 预计算有效耦合 ===
        int n_free_res = 0;
        if (amp_calc) n_free_res = amp_calc->nFreeResParams();
        if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
            // 重新计算 d_all_amplitudes（用新的共振态参数）
            auto t0_reamp = std::chrono::high_resolution_clock::now();
            amp_calc->reComputeAmps(d_all_amplitudes_list,
                reinterpret_cast<const double*>(theta.data_ptr()),
                n_amplitudes_, events_offsets_list, amp_offsets_list, n_polar_,
                primary_dev);
            cudaDeviceSynchronize();
            auto t1_reamp = std::chrono::high_resolution_clock::now();

            // 预计算有效耦合 T——已停用: d_T 全库无消费者（死代码; 每 forward 全事件
            // kernel + nEv×nPol×16B/卡 表纯浪费）。若未来 Hessian 需 T 中间量，
            // 请先实现消费者再恢复此调用。
            // for (int gpu = 0; gpu < num_gpus; ++gpu) { ... computeEffectiveCoupling ... }
            t2_reamp = std::chrono::high_resolution_clock::now();
            if (getenv("CTPWA_PROF")) printf("[PROF] reComputeAmps: %.2f ms, +T: %.2f ms\n",
                std::chrono::duration<double, std::milli>(t1_reamp - t0_reamp).count(),
                std::chrono::duration<double, std::milli>(t2_reamp - t1_reamp).count());

        }

        tF2 = std::chrono::high_resolution_clock::now();  // reAmp+T 完成
        // (free-θ 的 B̄ 重建已移除: d_P_vec 由下方 phsp_factor 融合扫描直接给出,
        //  免 7.2GB 加权副本; fixed-θ 仍用构造期预建 B̄ 的快速 matvec)

        // 3. 全局量: d_P_vec和phsp_factor（小矩阵M<100，用自定义核替代cuBLAS更快）
        //   fixed-θ（常驻或流式）: B̄ 构造期就绪 → computeQuadraticForm 快速 matvec
        //   free-θ:  B̄ 不再每 forward 重建（删 7.2GB 加权副本）;
        //            phsp_factor 与 d_P_vec 由一次融合扫描直接给出
        //            d_P_vec = Σ_ev,p (w/W)·A·conj(S) ≡ B̄·v̄
        bool free_theta = (amp_calc && n_free_res > 0 && theta.numel() > 0);
        ctpwa::ensure_device(primary_dev);
        ctComplex* d_P_vec;
        cudaMalloc(&d_P_vec, n_amplitudes_ * sizeof(ctComplex));
        double phsp_sum = 0.0;
        // phsp 归一化分母: Σ w（无权重 = 事件数）。与 d_phsp_matrix_ 的
        // 1/W_total 以及梯度 d_P_vec 的 w/W_total 必须用同一个 W_total。
        double W_total = 0.0;
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            int nP = events_list[gpu][0];
            if (nP == 0) continue;
            if (gpu < (int)d_phsp_weights_list.size() && d_phsp_weights_list[gpu] != nullptr) {
                cudaSetDevice(gpu);
                thrust::device_ptr<double> dp(d_phsp_weights_list[gpu]);
                W_total += thrust::reduce(dp, dp + nP);
            } else {
                W_total += (double)nP;
            }
        }
        if (W_total <= 0.0) W_total = 1.0;
        // 流式 double 矩阵与 float32 矩阵同归一化：B̄ = Σ(w/W)|A|²，
        // Re(v^H B̄ v) 已经是 Σ(w/W)|S|²，不能再除 Σw。
        bool phsp_stream_normalized = false;
        if (d_phsp_matrix_double_ != nullptr) {
            // phsp 流式模式: 原始 phsp 振幅不驻留，phsp_sum = Re(v^H M_double v)（主 GPU）
            TORCH_CHECK(n_free_res == 0,
                "phsp 流式模式（free_phsp_amplitudes）要求无自由共振态参数");
            ctpwa::ensure_device(primary_dev);
            const ctComplex* d_v_primary = reinterpret_cast<const ctComplex*>(
                extended_vector.data_ptr());
            phsp_sum = computeDoublePhspSum(d_phsp_matrix_double_, d_v_primary,
                                            n_amplitudes_);
            phsp_stream_normalized = true;
            // fixed-θ（n_free_res==0 必成立）: 构造期预建 B̄ → matvec
            ctPhspReal* d_phsp_r, * d_phsp_i;
            cudaMalloc(&d_phsp_r, sizeof(ctPhspReal));
            cudaMalloc(&d_phsp_i, sizeof(ctPhspReal));
            {
                torch::Tensor extended_vector_conj = extended_vector.conj();
                const ctComplex* d_vec_conj = reinterpret_cast<const ctComplex*>(extended_vector_conj.data_ptr());
                computeQuadraticForm(d_phsp_matrix_, d_vec_conj, d_P_vec,
                    d_phsp_r, d_phsp_i, n_amplitudes_);
            }
            cudaFree(d_phsp_r);
            cudaFree(d_phsp_i);
        } else if (free_theta) {
            // free-θ 常驻: phsp_factor 与 d_P_vec 一次融合扫描（免 B̄ 重建）
            //（phsp_factor 语义与 computePhspMeanSum 一致: Σ w|S|²，下面除 W_total）
            ctpwa::ensure_device(primary_dev);
            if ((int)s_d_P_part.size() < num_gpus) {
                s_d_P_part.resize(num_gpus, nullptr);
                s_d_P_part_sz.assign(num_gpus, 0);
            }
            if (s_d_P_peer_sz < n_amplitudes_) {
                if (s_d_P_peer) cudaFree(s_d_P_peer);
                cudaMalloc(&s_d_P_peer, n_amplitudes_ * sizeof(ctComplex));
                s_d_P_peer_sz = n_amplitudes_;
            }
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                int nP = events_list[gpu][0];
                if (nP == 0) continue;
                cudaSetDevice(gpu);
                if ((int)s_phsp_handle.size() < num_gpus) s_phsp_handle.resize(num_gpus, nullptr);
                if (!s_phsp_handle[gpu]) cublasCreate(&s_phsp_handle[gpu]);   // θ 段 S CGEMV 共用
                if (s_d_P_part_sz[gpu] < n_amplitudes_) {
                    if (s_d_P_part[gpu]) cudaFree(s_d_P_part[gpu]);
                    cudaMalloc(&s_d_P_part[gpu], n_amplitudes_ * sizeof(ctComplex));
                    s_d_P_part_sz[gpu] = n_amplitudes_;
                }
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(
                    extended_vec_per_gpu[gpu].data_ptr());
                const double* d_w = (gpu < (int)d_phsp_weights_list.size())
                    ? d_phsp_weights_list[gpu] : nullptr;
                bool fA = amp_calc && amp_calc->floatOutput();
                if (fA) {
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu]);
                    phsp_sum += computePhspMeanSumAndGradPF(d_ampF, d_v_gpu,
                        d_w, nP, n_polar_, n_amplitudes_, 1.0 / W_total, s_d_P_part[gpu]);
                } else {
                    phsp_sum += computePhspMeanSumAndGradP(d_all_amplitudes_list[gpu], d_v_gpu,
                        d_w, nP, n_polar_, n_amplitudes_, 1.0 / W_total, s_d_P_part[gpu]);
                }
            }
            // 跨 GPU 规约局部 d_P_vec → 主 GPU d_P_vec
            ctpwa::ensure_device(primary_dev);
            cudaMemset(d_P_vec, 0, n_amplitudes_ * sizeof(ctComplex));
            {
                ctComplex one = ctMake(1.0f, 0.0f);
                for (int gpu = 0; gpu < num_gpus; ++gpu) {
                    int nP = events_list[gpu][0];
                    if (nP == 0) continue;
                    if (gpu == primary_dev) {
                        axpyComplex(d_P_vec, s_d_P_part[gpu], one, n_amplitudes_);
                    } else {
                        cudaMemcpyPeer(s_d_P_peer, primary_dev, s_d_P_part[gpu], gpu,
                            n_amplitudes_ * sizeof(ctComplex));
                        ctpwa::ensure_device(primary_dev);
                        axpyComplex(d_P_vec, s_d_P_peer, one, n_amplitudes_);
                    }
                }
            }
        } else {
            // fixed-θ 常驻: B̄ 构造期预建（forward 不重建）→ matvec（快速路径）
            // W_total 的逐 GPU 权重归约把 current device 留在最后一张卡。
            // d_phsp_matrix_ 与 d_P_vec 都在主卡，必须先切回主卡再做二次型；
            // 否则副卡 kernel 会直接读主卡指针（compute-sanitizer 已定位）。
            ctpwa::ensure_device(primary_dev);
            ctPhspReal* d_phsp_r, * d_phsp_i;
            cudaMalloc(&d_phsp_r, sizeof(ctPhspReal));
            cudaMalloc(&d_phsp_i, sizeof(ctPhspReal));
            {
                torch::Tensor extended_vector_conj = extended_vector.conj();
                const ctComplex* d_vec_conj = reinterpret_cast<const ctComplex*>(extended_vector_conj.data_ptr());
                computeQuadraticForm(d_phsp_matrix_, d_vec_conj, d_P_vec,
                    d_phsp_r, d_phsp_i, n_amplitudes_);
            }
            cudaFree(d_phsp_r);
            cudaFree(d_phsp_i);
            // phsp_factor = Σ w f / Σ w：Σ w f 用 double 累加（float32 phsp 矩阵
            // 求和在 50 万事件上有 ~1e-4 相对系统偏差）
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                int nP = events_list[gpu][0];
                if (nP == 0) continue;
                cudaSetDevice(gpu);
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(
                    extended_vec_per_gpu[gpu].data_ptr());
                const double* d_w = (gpu < (int)d_phsp_weights_list.size())
                    ? d_phsp_weights_list[gpu] : nullptr;
                bool fA = amp_calc && amp_calc->floatOutput();
                if (fA) {
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu]);
                    phsp_sum += computePhspMeanSumF(d_ampF, d_v_gpu,
                                                   nP, n_polar_, n_amplitudes_, d_w);
                } else {
                    phsp_sum += computePhspMeanSum(d_all_amplitudes_list[gpu], d_v_gpu,
                                                   nP, n_polar_, n_amplitudes_, d_w);
                }
            }
        }
        // 加权 phsp 均值: Σ w f / Σ w。流式 double 矩阵已在构造期乘过
        // sqrt(w/W_total)，所以 phsp_sum 已经是归一化值（denom=1）。
        double phsp_denom = phsp_stream_normalized ? 1.0 : W_total;
        double phsp_factor = (phsp_denom > 0.0) ? phsp_sum / phsp_denom : 0.0;
        ctpwa::ensure_device(primary_dev);
        tF3 = std::chrono::high_resolution_clock::now();  // phsp_factor/d_P_vec（free-θ 融合扫描）完成
        tF4 = tF3;  // （原 B̄ CGEMM 段已删；D2H 计入 tF3 段）

        // //输出 d_P_vec 检查
        // std::vector<ctComplex> h_P_vec(n_amplitudes_);
        // cudaMemcpy(h_P_vec.data(), d_P_vec, n_amplitudes_ * sizeof(ctComplex),
        //     cudaMemcpyDeviceToHost);
        // std::cout << "P_vec: ";
        // for (int i = 0; i < n_amplitudes_; ++i) {
        //     std::cout << "(" << h_P_vec[i].x << ", " << h_P_vec[i].y << ") ";
        // }
        // std::cout << std::endl;

        // 4. 多GPU计算data+bkg似然值和梯度（纯data/bkg，不含phsp项）
        double total_data_nll = 0.0;
        double total_bkg_nll = 0.0;
        int totalDataEvents = 0;
        int totalBkgEvents = 0;
        double totalDataWeight = 0.0;   // Σ data 权重（无权重时 = 事件数）
        double bkg_integral = bkg_integral_;

        // 预分配持久化buffer（首次分配，之后复用）
        static ctComplex* s_d_grad_global = nullptr;
        static ctComplex* s_d_grad_buf = nullptr;
        static std::vector<ctComplex*> s_d_grad_per_gpu;
        static std::vector<ctComplex*> s_d_w_bufs;       // per-GPU data w buffer
        static std::vector<ctComplex*> s_d_w_bkg_bufs;   // per-GPU bkg  w buffer
        static std::vector<int> s_w_buf_sizes;            // data
        static std::vector<int> s_w_bkg_buf_sizes;        // bkg
        static int s_alloc_n = 0;

        if (s_alloc_n < extended_n_gls || s_d_grad_per_gpu.size() < (size_t)num_gpus) {
            // 释放后必须置空 + sizes 归零，否则 resize 不重置已有元素，
            // 悬空指针会被复用（cudaMemcpy invalid argument 的根因）
            if (s_d_grad_global) { cudaFree(s_d_grad_global); s_d_grad_global = nullptr; }
            if (s_d_grad_buf) { cudaFree(s_d_grad_buf); s_d_grad_buf = nullptr; }
            for (auto& p : s_d_grad_per_gpu) if (p) { cudaFree(p); p = nullptr; }
            for (auto& p : s_d_w_bufs) if (p) { cudaFree(p); p = nullptr; }
            for (auto& p : s_d_w_bkg_bufs) if (p) { cudaFree(p); p = nullptr; }

            ctpwa::ensure_device(primary_dev);
            cudaMalloc(&s_d_grad_global, extended_n_gls * sizeof(ctComplex));
            cudaMalloc(&s_d_grad_buf, extended_n_gls * sizeof(ctComplex));
            s_d_grad_per_gpu.resize(num_gpus, nullptr);
            s_d_w_bufs.resize(num_gpus, nullptr);
            s_d_w_bkg_bufs.resize(num_gpus, nullptr);
            s_w_buf_sizes.assign(num_gpus, 0);
            s_w_bkg_buf_sizes.assign(num_gpus, 0);
            for (int g = 0; g < num_gpus; ++g) {
                cudaSetDevice(g);
                cudaMalloc(&s_d_grad_per_gpu[g], extended_n_gls * sizeof(ctComplex));
            }
            s_alloc_n = extended_n_gls;
        }

        ctComplex* d_grad_global = s_d_grad_global;
        ctComplex* d_grad_buf = s_d_grad_buf;
        ctpwa::ensure_device(primary_dev);
        cudaMemset(d_grad_global, 0, extended_n_gls * sizeof(ctComplex));

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);
            const ctComplex* d_vec_gpu = reinterpret_cast<const ctComplex*>(
                extended_vec_per_gpu[gpu].data_ptr());
            ctComplex* d_grad = s_d_grad_per_gpu[gpu];

            // --- data ---
            int nData_gpu = events_list[gpu][1];
            if (nData_gpu > 0) {
                ctComplex* d_amp = d_all_amplitudes_list[gpu] + amp_offsets_list[gpu][1];
                const double* d_w_data = (gpu < (int)d_data_weights_list.size())
                    ? d_data_weights_list[gpu] : nullptr;

                // 分配/检查 w buffer（仅当有 theta 即做共振态拟合时）
                ctComplex* d_w_out = nullptr;
                if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
                    int nTotal = nData_gpu * n_polar_;
                    if (s_w_buf_sizes[gpu] < nTotal) {
                        if (s_d_w_bufs[gpu]) cudaFree(s_d_w_bufs[gpu]);
                        cudaMalloc(&s_d_w_bufs[gpu], nTotal * sizeof(ctComplex));
                        s_w_buf_sizes[gpu] = nTotal;
                    }
                    d_w_out = s_d_w_bufs[gpu];
                }

                auto t0_nll = std::chrono::high_resolution_clock::now();
                bool fA = amp_calc && amp_calc->floatOutput();
                double nll;
                if (fA) {
                    // amp_offsets 为元素数偏移：float2 模式下从基址 reinterpret 后偏移
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu])
                        + amp_offsets_list[gpu][1];
                    // w 缓冲保持 double（共振梯度按 ctComplex 读）：F 输出 float2 临时 → 上转
                    float2* d_w_f = nullptr;
                    if (d_w_out) {
                        cudaMalloc(&d_w_f, (size_t)nData_gpu * n_polar_ * sizeof(float2));
                    }
                    nll = computeFactorNLLF(d_ampF, d_vec_gpu,
                        d_grad, nData_gpu, n_polar_, n_amplitudes_, d_w_data, d_w_f);
                    if (d_w_f) {
                        int nW = nData_gpu * n_polar_;
                        if (ctIsFloatComplex) {
                            // 原生 float .so：float2 与 ctComplex(cuComplex) 同布局，
                            // 直接 D2D 拷贝；再"上转 double"会按 16B/元素越界写。
                            cudaMemcpy(d_w_out, d_w_f,
                                (size_t)nW * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                        } else {
                            CTPWA_LAUNCH(castF2ToDouble2Kernel, (nW + 255) / 256, 256, 0,
                                d_w_f, reinterpret_cast<cuDoubleComplex*>(d_w_out), nW);
                        }
                        cudaFree(d_w_f);
                    }
                } else {
                    nll = computeFactorNLL(d_amp, d_vec_gpu,
                        d_grad, nData_gpu, n_polar_, n_amplitudes_, d_w_data, d_w_out);
                }

#if defined(CTPWA_BACKEND_CPU)
                if (getenv("CTPWA_CPU_DEBUG"))
                    std::printf("[CPU-DBG] gpu %d data nll=%.10f (nData=%d nPol=%d nAmp=%d)\n",
                                gpu, nll, nData_gpu, n_polar_, n_amplitudes_);
#endif
                total_data_nll += nll;
                totalDataEvents += nData_gpu;
                // data 总权重（phsp 归一化 N_eff 用；无权重时 = 事件数）
                if (d_w_data != nullptr) {
                    cudaSetDevice(gpu);
                    thrust::device_ptr<const double> dp(d_w_data);
                    totalDataWeight += thrust::reduce(dp, dp + nData_gpu);
                }
                else {
                    totalDataWeight += nData_gpu;
                }
                auto t1_nll = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] computeFactorNLL(data): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_nll - t0_nll).count());
                // P2P累加到global (正号)
                if (gpu == primary_dev) {
                    axpyComplex(d_grad_global, d_grad, ctMake(1.0f, 0.0f), extended_n_gls);
                }
                else {
                    cudaMemcpyPeer(d_grad_buf, primary_dev, d_grad, gpu, extended_n_gls * sizeof(ctComplex));
                    ctpwa::ensure_device(primary_dev);
                    axpyComplex(d_grad_global, d_grad_buf, ctMake(1.0f, 0.0f), extended_n_gls);
                }
            }

            // --- bkg ---
            int nBkg_gpu = (events_list[gpu].size() > 2) ? events_list[gpu][2] : 0;
            if (nBkg_gpu > 0) {
                cudaSetDevice(gpu);
                ctComplex* d_amp = d_all_amplitudes_list[gpu] + amp_offsets_list[gpu][2];
                const double* d_w = d_bkg_weights_list[gpu];

                // 分配/检查 bkg  w buffer
                ctComplex* d_w_bkg_out = nullptr;
                if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
                    int nTotal = nBkg_gpu * n_polar_;
                    if (s_w_bkg_buf_sizes[gpu] < nTotal) {
                        if (s_d_w_bkg_bufs[gpu]) cudaFree(s_d_w_bkg_bufs[gpu]);
                        cudaMalloc(&s_d_w_bkg_bufs[gpu], nTotal * sizeof(ctComplex));
                        s_w_bkg_buf_sizes[gpu] = nTotal;
                    }
                    d_w_bkg_out = s_d_w_bkg_bufs[gpu];
                }

                auto t0_nllb = std::chrono::high_resolution_clock::now();
                bool fA = amp_calc && amp_calc->floatOutput();
                double nll;
                if (fA) {
                    const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[gpu])
                        + amp_offsets_list[gpu][2];
                    float2* d_w_f = nullptr;
                    if (d_w_bkg_out) {
                        cudaMalloc(&d_w_f, (size_t)nBkg_gpu * n_polar_ * sizeof(float2));
                    }
                    nll = computeFactorNLLF(d_ampF, d_vec_gpu,
                        d_grad, nBkg_gpu, n_polar_, n_amplitudes_, d_w, d_w_f);
                    if (d_w_f) {
                        int nW = nBkg_gpu * n_polar_;
                        if (ctIsFloatComplex) {
                            cudaMemcpy(d_w_bkg_out, d_w_f,
                                (size_t)nW * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                        } else {
                            CTPWA_LAUNCH(castF2ToDouble2Kernel, (nW + 255) / 256, 256, 0,
                                d_w_f, reinterpret_cast<cuDoubleComplex*>(d_w_bkg_out), nW);
                        }
                        cudaFree(d_w_f);
                    }
                } else {
                    nll = computeFactorNLL(d_amp, d_vec_gpu,
                        d_grad, nBkg_gpu, n_polar_, n_amplitudes_, d_w, d_w_bkg_out);
                }

                total_bkg_nll += nll;
                totalBkgEvents += nBkg_gpu;
                auto t1_nllb = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] computeFactorNLL(bkg): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_nllb - t0_nllb).count());
                if (gpu == primary_dev) {
                    axpyComplex(d_grad_global, d_grad, ctMake(-1.0f, 0.0f), extended_n_gls);
                }
                else {
                    cudaMemcpyPeer(d_grad_buf, primary_dev, d_grad, gpu, extended_n_gls * sizeof(ctComplex));
                    ctpwa::ensure_device(primary_dev);
                    axpyComplex(d_grad_global, d_grad_buf, ctMake(-1.0f, 0.0f), extended_n_gls);
                }
            }
        }

        t6_nll = std::chrono::high_resolution_clock::now();  // computeFactorNLL data+bkg 完成

        // === 新增：计算共振态参数梯度 ===
        torch::Tensor grad_theta;
        if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
            ctpwa::ensure_device(primary_dev);
            // 持久化懒分配（n_free_res 固定，避免每次 forward 的 malloc/free）
            static double* s_d_grad_res = nullptr;
            static int s_grad_res_sz = 0;
            if (s_grad_res_sz < n_free_res) {
                if (s_d_grad_res) cudaFree(s_d_grad_res);
                ctpwa::ensure_device(primary_dev);
                cudaMalloc(&s_d_grad_res, n_free_res * sizeof(double));
                s_grad_res_sz = n_free_res;
            }
            double* d_grad_res = s_d_grad_res;
            ctpwa::ensure_device(primary_dev);
            cudaMemset(d_grad_res, 0, n_free_res * sizeof(double));

            // data 贡献 (sign=+1, d_T/d_momenta 需加 phsp 偏移)
            if (totalDataEvents > 0) {
                std::vector<int> n_data_events(num_gpus);
                std::vector<int> phsp_offsets(num_gpus);
                std::vector<ctComplex*> d_v_ptrs(num_gpus);
                for (int g = 0; g < num_gpus; ++g) {
                    n_data_events[g] = events_list[g][1];
                    phsp_offsets[g] = events_list[g][0];
                    d_v_ptrs[g] = const_cast<ctComplex*>(
                        reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[g].data_ptr()));
                }
                auto t0_rg = std::chrono::high_resolution_clock::now();
                amp_calc->computeResonanceGradient(primary_dev, s_d_w_bufs, n_data_events, d_grad_res,
                    +1.0, phsp_offsets, d_v_ptrs);
                auto t1_rg = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] resonanceGradient(data): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_rg - t0_rg).count());
            }

            // bkg 贡献 (sign=-1, loss = data_nll - bkg_nll)
            if (totalBkgEvents > 0) {
                std::vector<int> n_bkg_events(num_gpus);
                std::vector<int> bkg_offsets(num_gpus);
                std::vector<ctComplex*> d_v_ptrs(num_gpus);
                for (int g = 0; g < num_gpus; ++g) {
                    n_bkg_events[g] = events_list[g][2];
                    bkg_offsets[g] = events_list[g][0] + events_list[g][1]; // phsp + data 偏移
                    d_v_ptrs[g] = const_cast<ctComplex*>(
                        reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[g].data_ptr()));
                }
                auto t0_rgb = std::chrono::high_resolution_clock::now();
                amp_calc->computeResonanceGradient(primary_dev, s_d_w_bkg_bufs, n_bkg_events, d_grad_res,
                    -1.0, bkg_offsets, d_v_ptrs);
                auto t1_rgb = std::chrono::high_resolution_clock::now();
                if (getenv("CTPWA_PROF")) printf("[PROF] resonanceGradient(bkg): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1_rgb - t0_rgb).count());
            }

            // phsp 贡献: ∂((N_data-W_bkg)*log(phsp))/∂θ
            {
                int total_phsp = 0;
                for (int g = 0; g < num_gpus; ++g) total_phsp += events_list[g][0];
                double effective_data = totalDataWeight - bkg_integral_;
                if (total_phsp > 0 && phsp_factor > 1e-30) {
                    // S 下面会乘 w/W_total（scalePhspSByWeightKernel），
                    // 因此 sign 不再除 total_phsp（W_total 已在 S 里）。
                    double phsp_sign = -effective_data / phsp_factor;

                    std::vector<ctComplex*> d_S_bufs(num_gpus, nullptr);
                    std::vector<int> n_phsp_evts(num_gpus);
                    if ((int)s_d_S_bufs.size() < num_gpus) {
                        s_d_S_bufs.resize(num_gpus, nullptr);
                        s_dS_alloc_sz.assign(num_gpus, 0);
                    }
                    for (int g = 0; g < num_gpus; ++g) {
                        cudaSetDevice(g);
                        int nP = events_list[g][0];
                        n_phsp_evts[g] = nP;
                        if (nP == 0) continue;
                        int nTot = nP * n_polar_;
                        if (s_dS_alloc_sz[g] < nTot) {
                            if (s_d_S_bufs[g]) cudaFree(s_d_S_bufs[g]);
                            cudaMalloc(&s_d_S_bufs[g], nTot * sizeof(ctComplex));
                            s_dS_alloc_sz[g] = nTot;
                        }
                        d_S_bufs[g] = s_d_S_bufs[g];
                        const ctComplex* d_vg = reinterpret_cast<const ctComplex*>(
                            extended_vec_per_gpu[g].data_ptr());
                        bool fA = amp_calc && amp_calc->floatOutput();
                        if (fA) {
                            // float2 A 直读算 S（无 7GB double 临时；v double、S 输出 double）
                            const float2* d_ampF = reinterpret_cast<const float2*>(d_all_amplitudes_list[g]);
                            int gS = (nP + 255) / 256;
                            CTPWA_LAUNCH(computeSFromFloatAmpsKernel, gS, 256, 0, 
                                s_d_S_bufs[g], d_ampF, d_vg, nP, n_polar_, n_amplitudes_);
                        } else {
                            ctComplex* d_amp = d_all_amplitudes_list[g];
                            ctComplex a = ctMake(1.0f, 0.0f);
                            ctComplex b = ctMake(0.0f, 0.0f);
                            CUBLAS_CGEMV(s_phsp_handle[g], CUBLAS_OP_T, n_amplitudes_, nTot,
                                &a, d_amp, n_amplitudes_, d_vg, 1, &b, d_S_bufs[g], 1);
                        }
                        const double* d_w = (g < (int)d_phsp_weights_list.size())
                            ? d_phsp_weights_list[g] : nullptr;
                        int gS = (nP + 255) / 256;
                        CTPWA_LAUNCH(scalePhspSByWeightKernel, gS, 256, 0,
                            d_S_bufs[g], d_w, nP, n_polar_, 1.0 / W_total);
                    }

                    ctpwa::ensure_device(primary_dev);  // d_grad_res is on primary_dev
                    std::vector<ctComplex*> d_v_ptrs(num_gpus);
                    for (int g = 0; g < num_gpus; ++g)
                        d_v_ptrs[g] = const_cast<ctComplex*>(
                            reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[g].data_ptr()));
                    auto t0_rgp = std::chrono::high_resolution_clock::now();
                    amp_calc->computeResonanceGradient(primary_dev, d_S_bufs, n_phsp_evts, d_grad_res,
                        phsp_sign, {}, d_v_ptrs);
                    auto t1_rgp = std::chrono::high_resolution_clock::now();
                    if (getenv("CTPWA_PROF")) printf("[PROF] resonanceGradient(phsp): %.2f ms\n",
                        std::chrono::duration<double, std::milli>(t1_rgp - t0_rgp).count());

                    ctpwa::ensure_device(primary_dev);
                }
            }
            t7_rg = std::chrono::high_resolution_clock::now();  // resonanceGradient 各段完成

            ctpwa::ensure_device(primary_dev);
            grad_theta = torch::empty({ n_free_res },
                torch::TensorOptions().dtype(torch::kFloat64).device(ctpwa_dev(primary_dev)));
            cudaMemcpy(grad_theta.data_ptr(), d_grad_res,
                n_free_res * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        else {
            grad_theta = torch::empty({ 0 },
                torch::TensorOptions().dtype(torch::kFloat64).device(vector.device()));
        }

        // // 输出d_grad_global检查
        // std::vector<ctComplex> h_grad_global(extended_n_gls);
        // cudaMemcpy(h_grad_global.data(), d_grad_global, extended_n_gls * sizeof(ctComplex),
        //     cudaMemcpyDeviceToHost);
        // std::cout << "Global gradient: ";
        // for (int i = 0; i < extended_n_gls; ++i) {
        //     std::cout << "(" << h_grad_global[i].x << ", " << h_grad_global[i].y << ") ";
        // }
        // std::cout << std::endl;

        // // 输出d_vec_gpu检查
        // std::vector<ctComplex> h_vec_gpu(extended_n_gls);
        // cudaMemcpy(h_vec_gpu.data(), extended_vec_per_gpu[0].data_ptr(),
        //     extended_n_gls * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // std::cout << "Extended vector (GPU 0): ";
        // for (int i = 0; i < extended_n_gls; ++i) {
        //     std::cout << "(" << h_vec_gpu[i].x << ", " << h_vec_gpu[i].y << ") ";
        // }
        // std::cout << std::endl;

        // 5. 合并data和bkg的NLL与梯度（loss = data_nll - bkg_nll）
        if (phsp_factor <= 1e-30 || isnan(phsp_factor) || isinf(phsp_factor)) {
            // std::cerr << "WARNING: phsp_factor invalid (" << phsp_factor << "), clamping to 1e-30" << std::endl;
            phsp_factor = 1e-30;
        }
        if (getenv("CTPWA_PROF")) printf("PY_NLL data_nll=%.10f bkg_nll=%.10f phsp_factor=%.10f totalData=%d totalBkg=%d bkg_integral=%.6f\n",
            total_data_nll, total_bkg_nll, phsp_factor, totalDataEvents, totalBkgEvents, bkg_integral_);

        double loss = total_data_nll - total_bkg_nll + (totalDataWeight - bkg_integral_) * log(phsp_factor);
        // Constraints.gauss_constr: 罚项 Σ(x-μ)²/(2σ²)
        // （NaN 情况下由下方 loss_reset 逻辑一并处理）
        if (params_mgr->nGaussConstr() > 0 && n_free_res > 0 && theta.numel() > 0)
            loss += params_mgr->gaussPenalty(theta);
        bool loss_reset = false;
        if (isnan(loss) || isinf(loss)) {
            // std::cerr << "WARNING: loss is " << (isnan(loss) ? "NaN" : "Inf") << ", resetting to 1e30" << std::endl;
            loss = 1e30;
            cudaMemset(d_grad_global, 0, extended_n_gls * sizeof(ctComplex));
            // theta 梯度也要清零，否则 NaN 会传播到最终梯度
            if (n_free_res > 0)
                cudaMemset(grad_theta.data_ptr<double>(), 0, n_free_res * sizeof(double));
            loss_reset = true;
        }
        // loss 重置时跳过 phsp 梯度累加——phsp_factor 已 clamp 到 1e-30，
        // scale_phsp ~ 1e33 会把 d_P_vec 中的任何 NaN/Inf 重新注入梯度
        if (!loss_reset) {
            ctComplex scale_phsp = ctMake(
                static_cast<float>(totalDataWeight - bkg_integral_) / static_cast<float>(phsp_factor), 0.0f);
            axpyComplex(d_grad_global, d_P_vec, scale_phsp, extended_n_gls);
        }
        cudaFree(d_P_vec);

        // gauss_constr 罚项梯度 (x-μ)/σ² 累加到 grad_theta
        // （loss_reset 时 grad_theta 已清零，跳过以免注入 NaN）
        if (!loss_reset && params_mgr->nGaussConstr() > 0 && n_free_res > 0
            && grad_theta.numel() > 0) {
            params_mgr->gaussPenaltyGrad(theta, grad_theta);
        }

        // 6. 保存梯度和loss到ctx
        ctpwa::ensure_device(primary_dev);
        torch::Tensor global_extended_grad = torch::empty({ extended_n_gls },
            TORCH_COMPLEX).to(vector.device());
        cudaMemcpy(global_extended_grad.data_ptr(), d_grad_global,
            extended_n_gls * sizeof(ctComplex), cudaMemcpyDeviceToDevice);

        ctx->save_for_backward({ params_tensor });
        ctx->saved_data["global_extended_grad"] = global_extended_grad;
        ctx->saved_data["grad_theta"] = grad_theta;
        ctx->saved_data["params_ptr"] = reinterpret_cast<int64_t>(params_mgr);

        // std::cout << "Total data NLL: " << loss << std::endl;
        if (tprof) {
            auto tFend = std::chrono::high_resolution_clock::now();
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count(); };
            printf("[PROF] f.extend: %.2f | f.reAmp+T: %.2f | f.phspFactor+dPvec: %.2f | f.nll: %.2f | f.gradRes: %.2f | f.tail: %.2f\n",
                ms(tF0, tF1), ms(tF1, t2_reamp), ms(t2_reamp, tF3),
                ms(tF3, t6_nll), ms(t6_nll, t7_rg), ms(t7_rg, tFend));
        }
        // 恢复主设备再返回：forward 尾部各 per-GPU 段可能把 raw current device
        // 留在非主卡，torch 缓存又只随 torch 自身调用更新——保持 raw 与 torch
        // 主设备一致，避免后续 python/torch 的裸/张量操作落到错误物理卡。
        ctpwa::ensure_device(primary_dev);
        return torch::tensor(loss, torch::kDouble).to(vector.device());
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::tensor_list& grad_outputs)
    {
        bool tprof = getenv("CTPWA_PROF") != nullptr;
        auto tB0 = std::chrono::high_resolution_clock::now();
        const auto saved = ctx->get_saved_variables();
        const auto& params_tensor = saved[0];
        const auto& global_extended_grad = ctx->saved_data["global_extended_grad"].toTensor();
        const auto& grad_theta = ctx->saved_data["grad_theta"].toTensor();
        auto* params_mgr = reinterpret_cast<Parameters*>(ctx->saved_data["params_ptr"].toInt());

        int nv = params_mgr->nFreeVector();
        int nt = params_mgr->nFreeTheta();

        torch::Tensor grad_params;

        if (params_mgr->hasCouplingMatrix()) {
            // Coupling matrix mode: transform ∂L/∂v → ∂L/∂p
            const auto& cm = params_mgr->couplingMatrix();
            int na = cm.n_amps;
            int ncf = cm.n_free;

            auto original_params = ctx->saved_data["original_params"].toTensor();
            const double* d_p = original_params.data_ptr<double>();

            // Current v values for gradient transform
            auto v_buf = torch::empty({na}, torch::TensorOptions()
                .dtype(TORCH_COMPLEX).device(global_extended_grad.device()));
            ctComplex* d_v = reinterpret_cast<ctComplex*>(v_buf.data_ptr());
            params_mgr->applyCouplingMatrix(d_p, d_v);

            // Gradient w.r.t coupling params [2*ncf] in [Re_p, Im_p] format
            auto grad_p = torch::empty({2 * ncf},
                torch::TensorOptions().dtype(torch::kFloat64).device(global_extended_grad.device()));
            double* d_grad_p = grad_p.data_ptr<double>();
            ctComplex* d_grad_v = reinterpret_cast<ctComplex*>(global_extended_grad.data_ptr());
            params_mgr->transformCouplingGradient(d_grad_v, d_v, d_p, d_grad_p);

            grad_params = (nt > 0 && grad_theta.numel() > 0)
                ? torch::cat({grad_p, grad_theta})
                : grad_p;
        } else {
            // Legacy mode: collapseVectorGrad + real/imag extraction
            torch::Tensor grad_vec = params_mgr->collapseVectorGrad(global_extended_grad, nv);
            torch::Tensor grad_real = (torch::real(grad_vec) * 2.0).to(torch::kFloat64);
            torch::Tensor grad_imag = (torch::imag(grad_vec) * 2.0).to(torch::kFloat64);

            if (nt > 0 && grad_theta.numel() > 0) {
                grad_params = torch::cat({grad_real, grad_imag, grad_theta});
            } else {
                grad_params = torch::cat({grad_real, grad_imag});
            }
        }

        auto tB1 = std::chrono::high_resolution_clock::now();
        if (tprof) printf("[PROF] backward: %.2f ms\n",
            std::chrono::duration<double, std::milli>(tB1 - tB0).count());
        return { grad_params * grad_outputs[0],
                torch::Tensor(),   // params_mgr
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor() };  // d_phsp_matrix_double_（流式 phsp 矩阵，裸指针无梯度）
    }
};

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
// float2 A 段 → 新分配 double 临时（writeResult/hessian 输出等低频消费路径用；
// 调用方负责 cudaFree 返回值。返回 nullptr 表示非 float 模式）
// 只允许 double .so 的混合精度模式，原生 float .so 下 A 已经是 ctComplex，
// 再"上转"会按 16B/元素写进 8B/元素缓冲区（越界）并让下游按 float 错读。

class analysis
{
public:
    analysis(const std::string& config_file = "config.yml", int fit_mode = 0)
        : config_parser_(config_file), n_amplitudes_(0), n_polar_(0), d_all_amplitudes_(), initialized_(false)
    {
        // fit mode 必须在 initialize()（→ initializeDecayChains）之前定好：
        // params_ 的耦合映射在那里一次性构建，构造后再 setFitMode 改不动参数化。
        fit_mode_ = fit_mode;
        frozen_fit_mode_ = fit_mode;
        if (!config_parser_.isValid()) {
            std::cerr << "Warning: Config file \"" << config_file
                      << "\" is empty or not found. Analysis not initialized." << std::endl;
            return;
        }
        initialize();
        initialized_ = true;
    }

    // 正常运行中显式释放本对象持有的设备资源；析构时也会调用。
    // 进程退出 GC 阶段 CUDA 上下文可能已销毁，此时只清空 host 簿记，
    // 让驱动在上下文销毁时统一回收，避免旧实现析构段错误（rc=139）。
    void release();

    // 析构函数，用于释放 CUDA 内存
    ~analysis() { release(); }

    bool isValid() const { return initialized_; }

    torch::Tensor getNLL(torch::Tensor params);

    // ---- fit mode ----
    // 注意：参数化（params_ 的耦合映射）在构造期 initializeDecayChains() 里一次性
    // 决定。构造后调用 setFitMode 只能改 getNVector/getParamNames 的语义，改不了
    // 实际拟合——需要 VSPACE 请用 ctpwa.analysis(config, 1)。
    void setFitMode(int mode) {
        if (mode != frozen_fit_mode_) {
            std::cerr << "[ctpwa] Warning: setFitMode(" << mode << ") 在构造后调用"
                         "无法改变参数化（构造期为 " << frozen_fit_mode_ << "，"
                         "params 结构已固定）；如需模式 " << mode
                      << " 请在构造时指定：ctpwa.analysis(config, " << mode << ")。"
                      << std::endl;
        }
        fit_mode_ = mode;
    }
    int getFitMode() const { return fit_mode_; }

    // legends: config 的 legends 规则按展开后的链顺序解析出的图例名 (getLegends() 结果)
    std::vector<std::string> getLegends() const { return legends_; }

    int getNVector() const {
        if (fit_mode_ == 1) return params_.nFreeVector();  // vspace: 自由振幅 (trans 折叠后)
        if (params_.hasCouplingMatrix()) return params_.couplingMatrix().n_free;
        return params_.nFreeVector();
    }
    int getNFreeTheta() const { return params_.nFreeTheta(); }
    std::vector<std::string> getParamNames() const {
        if (fit_mode_ == 1) {
            // vspace: 非折叠振幅名 + 共振态参数名（从实际 slots 生成）
            std::vector<std::string> names;
            if (params_.hasCouplingMatrix()) {
                const auto& cm = params_.couplingMatrix();
                std::map<int, std::string> free_idx_to_name;
                for (int ai = 0; ai < n_amplitudes_; ++ai) {
                    if (std::abs(cm.amp_chain_ratio[ai] - 1.0) < 1e-10) {
                        free_idx_to_name[cm.amp_chain[ai]] = amplitude_names_[ai];
                    }
                }
                for (const auto& [idx, name] : free_idx_to_name)
                    names.push_back(name);
            } else {
                names = amplitude_names_;
            }
            // theta 名从实际槽位生成（真相来源）：
            // trans 折叠后同名共振态只有一份 slot，避免 resonanceParamNames 的重复
            for (const auto& s : amp_calc_.slots())
                names.push_back(s.name);
            return names;
        }
        return params_.paramNames();
    }
    int getNParams() const { return 2 * getNVector() + getNFreeTheta(); }
    const Parameters& getParams() const { return params_; }

    torch::Tensor getSLVectors() const
    {
        torch::Device dev = ctpwa_dev(0);
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt).device(dev);
        return torch::tensor(nSLvectors_, options);
    }

    // waves: 可选, 只画选中分波子集 |Σ_{i∈S} A_i·v_i|² 的分布（空=全部=现状）。
    // 实现: kernel 用 mask 覆盖每事件总值 + 子集积分, 下游(直方图/TTree/归一化)不变。
    void writeResult(torch::Tensor params, const std::string& filename,
                     const int is_saved_weight = 0,
                     const std::vector<int>& waves = {},
                     torch::Tensor hessian_in = torch::Tensor());

    // =====================================================================
    // writeInterfResult: 保存指定波对的逐事件干涉形状。
    //   输出 TTree saved_weight: totalweight / weight_<i>(全部分波) /
    //   interf_<i>_<j>(只含选中对) / 末态粒子四动量。
    //   容量 = len(pairs) × N_phsp（用户按需选择, 不会因全波数×大MC爆显存）。
    //   pairs: [[i,j], ...] 波对(partial 索引, 与 weight_<i> 分支序一致)。
    // =====================================================================
    // ---------------------------------------------------------------------
    // [可删 · 自 v0.3.22 标记] 或合并进 writeResult(..., pairs=[[i,j],...])
    // 逐事例干涉形状 TTree；而且它把树也命名成 "saved_weight"，与 writeResult
    // 的逐事例树重名，读文件时容易拿错。目前 example/ 与 tests/ 均无调用方。
    // ---------------------------------------------------------------------
    void writeInterfResult(torch::Tensor params, const std::string& filename,
                           const std::vector<std::vector<int>>& pairs);


    // data/bkg 段 Hessian 贡献（多 GPU 循环内逐卡调用）。float-A 模式按窗口上转
    // A 段（峰值 = 窗口×16B，而非整段 16B double 临时——大样本下整段临时可达
    // 数百 MB~GB，直接吃掉"省一半显存"的收益）；double 模式直读常驻表。
    // bkg 段权重取负（与调用点原语义一致）；weights 随窗口偏移传入。
    void hessianSegmentContrib(int gpu, int seg,
                               const ctComplex* d_v, double* d_hess,
                               int n_amp, bool fast)
    {
        // P4：实现已搬到 ctpwa::hessianSegmentContrib（include/HessianContrib.h）
        ctpwa::hessianSegmentContrib(d_all_amplitudes_, events_, amp_offsets_,
                                     data_weights_, bkg_weights_,
                                     float_amps_, n_amplitudes_, n_polar_,
                                     gpu, seg, d_v, d_hess, n_amp, fast);
    }

    // ---- 内部 Hessian 计算 ----

    // 自由耦合参数 p (complex [n_free]) → 扩展振幅 v (complex [n_amps])
    // 乘法链耦合矩阵（trans 共享 / step 参数）时用 applyCouplingMatrix 映射；
    // 无耦合矩阵时 n_free == n_amps，恒等返回。
    // 注意: 不能用 params_.extendVector —— 它只处理 legacy 约束组，
    // 乘法链（hasCouplingMatrix）下是恒等，会把 p 误当振幅。
    // 布局: applyCouplingMatrix 期望 grouped 排列 [re0..re_{n-1}, im0..im_{n-1}]
    // （与 getNLL forward 的 params_tensor 一致），view_as_real 给的是
    // interleaved [re0,im0,re1,im1,...]，必须先转置。
    torch::Tensor freeParamsToAmplitudes(const torch::Tensor& p_vec) {
        if (!params_.hasCouplingMatrix()) return p_vec;
        const auto& cm = params_.couplingMatrix();
        torch::Device dev = p_vec.device();
        torch::Tensor v_ext = torch::empty({cm.n_amps},
            torch::TensorOptions().dtype(TORCH_COMPLEX).device(dev));
        torch::Tensor p64 = torch::view_as_real(p_vec).t().reshape({-1})
                                .to(torch::kFloat64).contiguous();
        params_.applyCouplingMatrix(p64.data_ptr<double>(),
            reinterpret_cast<ctComplex*>(v_ext.data_ptr()));
        return v_ext;
    }

    // 耦合参数 Hessian [2n × 2n]（被 getFitFractions 作为 hessian_in 缺省回退）
    // 输入: 自由耦合 p（complex [n_free]）；内部先映射为振幅 v 再对 v 求 Hessian。
    torch::Tensor computeCouplingHessian(torch::Tensor vector);

    // ========== 统一 params 接口 ==========

    // 完整 Hessian: 返回 (2n+P) × (2n+P)，params = [real(v), imag(v), θ] float64
    torch::Tensor getHessian(torch::Tensor params);

    // Helper: compute partial + scattering (+square) integrals from given amplitudes.
    // Accumulates into out_partial/out_scattering/out_square (caller must zero-init;
    // out_square may be nullptr when Σ|A_i|⁴ 不需要, e.g. 拟合分数).
    // d_weights_per_gpu: 可选逐 GPU 逐事件权重（空 vector 或元素 nullptr = 无权重）。
    //   加权语义见 ComputeBF.cuh: partial/scattering/total 累加 w·x, square 累加 (w·x)²。
    void computeTruthIntegrals(
        const torch::Tensor& extended_vector,
        const std::vector<ctComplex*>& d_amps,
        const std::vector<int>& ev_per_gpu,
        double* out_partial,
        double* out_scattering,
        double* out_square,
        int npartials,
        const std::vector<ctComplex*>& ws_complex,
        const std::vector<ctComplex*>& ws_matrix,
        const std::vector<ctComplex*>& ws_upcast,
        const std::vector<double*>& d_weights_per_gpu) const
    {
        // P4：实现已搬到 ctpwa::computeTruthIntegrals（include/TruthIntegrals.h）
        ctpwa::computeTruthIntegrals(extended_vector, d_amps, ev_per_gpu,
                                     out_partial, out_scattering, out_square, npartials,
                                     ws_complex, ws_matrix, ws_upcast, d_weights_per_gpu,
                                     nSLvectors_, float_amps_, n_polar_, n_amplitudes_, primary_dev_);
    }

    // =====================================================================
    // 通用: 在指定数据文件（phsp_truth/phsp 等）上分批积分
    //   out_partial [npartials]          Σ w·|A_i|²（中心）
    //   out_scattering [npartials²]      Σ w·2Re(A_i A_j*)（中心）
    //   out_square [npartials]           Σ (w·|A_i|²)²（中心, 可传 nullptr）
    //   out_p/m_partial/scattering        ±ε 扰动后的积分（可为 nullptr 跳过扰动）
    //
    // 权重（可选）: config 的 <file_key>_weights（如 phsp_truth_weights），写法同
    // data_weights/phsp_weights（常数 / dat 每行一个 / ROOT 分支）。权重语义与
    // 拟合 NLL 的归一化一致：I = Σ w·f 且 N = Σ w（w≡1 时逐位退化为事件数），
    // 所以"带少量模型产生的 truth 样本 + w∝1/|M_gen|²"可以直接当平坦 PHSP MC 用。
    // 返回该文件的 Σw（无权重时 = 事件数），供 getEfficiency 归一化。
    // =====================================================================
    double computeFileBatchIntegrals(
        const std::string& file_key,
        const torch::Tensor& ev_center,
        const std::vector<torch::Tensor>& ev_perturbed_p,
        const std::vector<torch::Tensor>& ev_perturbed_m,
        double* out_partial,
        double* out_scattering,
        double* out_square,
        double* out_p_partial,
        double* out_p_scattering,
        double* out_m_partial,
        double* out_m_scattering,
        int npartials);

    // =====================================================================
    // 从统一 Hessian 取"耦合自由方向"的协方差（误差传播用）。
    //
    //   H 维度 = 2n（仅耦合）或 2n+ntheta（联合 Hessian, 与 getHessian 同源）。
    //   联合时 **必须** 用边际协方差 inv(H_joint) 的耦合块，
    //   而不是 inv(H_coupling)（那是"θ 固定"的条件协方差，会系统性偏小）。
    //   参考方向 Re v0 / Im v0（下标 0, n）按约定固定，不进入自由块。
    //
    // 返回 [nf, nf] CPU 张量；nf = 耦合自由方向数。
    //   nf == 0        → 空张量, ok = true（没有参数误差可言）
    //   非正定/非有限 → 空张量, ok = false（调用方应跳过误差矩阵并说明原因）
    // =====================================================================
    // P4：数学部分已抽到 ctpwa::couplingCovariance（include/MatrixStats.h）
    torch::Tensor couplingCovariance(const torch::Tensor& H, int n,
                                     int& nf_out, bool& ok) const
    {
        return ctpwa::couplingCovariance(H, n, nf_out, ok);
    }

    // =====================================================================
    // 拟合分数: FF_i = truth_partial[i] / Σ_j truth_partial[j]
    //           = ∫|A_i|² / Σ_j ∫|A_j|²（无干涉项的分母, Σ_i FF_i = 1）。
    // 只用 phsp_truth（无效率 MC）→ 不依赖效率 MC 与数据归一化, 跨实验可比。
    // 配了 phsp_truth_weights 时积分为 Σ w·|A_i|²（权重在分子/分母中同源相消，
    // 等价于把带模型产生的 truth 样本重加权成平坦 PHSP MC）。
    // 注意: tfpwa 标准 fit fraction 分母用 ∫|ΣA|²（含干涉, Σ_i f_i ≠ 1），
    // 强干涉时两者略有差别, 报告时建议注明定义。
    // hessian_in: 可选统一 Hessian。分波误差与拟合 getHessian 同源。
    // 返回 [npartials, 2]: [center, error]
    // =====================================================================
    // ---------------------------------------------------------------------
    // [可删 · 自 v0.3.22 标记]
    // 中心值 = diag(interference_gen)/Σdiag，已经由 writeResult 写进
    //   weight_result.root（实测两条路径一致到 2e-15），脚本里一行就能算；
    // 误差是按"非相干分母"传播的，和 ROOT 里"相干分母"口径的
    //   interference_gen_err 不是同一个量（实测 0.0476 vs 0.2797），无法由 ROOT 复现；
    // 而非相干口径本身不是文献定义（Laura++/LHCb 用相干分母），去留待定。
    // 删除后：需要份额的人用 getFitFractionMatrix 或直接读 ROOT 表；
    //   或者给矩阵 API 加 denominator="coherent"|"incoherent" 开关。
    // 前置条件：example/fit.py 的 compute_fit_fractions 改掉再删。
    // ---------------------------------------------------------------------
    // 份额/效率/按样本取矩阵的共同输入（P4 抽出的三个模块函数共用）
    ctpwa::FitFractionCtx fitFractionCtx()
    {
        ctpwa::FitFractionCtx ctx;
        ctx.data_files = &config_parser_.getDataFiles();
        ctx.n_slvectors = &nSLvectors_;
        ctx.primary_dev = primary_dev_;
        ctx.free_params_to_amplitudes =
            [this](const torch::Tensor& cpl) { return freeParamsToAmplitudes(cpl); };
        ctx.batch_integrals = [this](const std::string& key, const torch::Tensor& ev,
                                     const std::vector<torch::Tensor>& ep,
                                     const std::vector<torch::Tensor>& em,
                                     double* a, double* b, double* c,
                                     double* d, double* e, double* f, double* g, int np) {
            return computeFileBatchIntegrals(key, ev, ep, em, a, b, c, d, e, f, g, np);
        };
        ctx.phsp_matrix_from_gram =
            [this](const torch::Tensor& v) { return phspMatrixFromGram(v); };
        ctx.coupling_hessian =
            [this](const torch::Tensor& v) { return computeCouplingHessian(v); };
        return ctx;
    }

    torch::Tensor getFitFractions(torch::Tensor vector)
    {
        return getFitFractions(vector, torch::Tensor());
    }
    torch::Tensor getFitFractions(torch::Tensor vector,
        torch::Tensor hessian_in)
    {
        // P4：实现已搬到 ctpwa::fitFractions（include/FitFractions.h）
        return ctpwa::fitFractions(fitFractionCtx(), vector, hessian_in);
    }

    // =====================================================================
    // 分波积分矩阵（原始值，未归一化；行/列顺序与 getFitFractions 一致）
    //   M[i][i] = Σ w|A_i|²            （= getFitFractions 的分子）
    //   M[i][j] = Σ w·2Re(A_i A_j*)    （i≠j, 对称）
    //
    // sample 选择在哪个人群上积分:
    //   "truth"（默认, = phsp_truth, 无效率 MC）: 文献里的拟合分数/干涉项分母
    //   "phsp"（带效率 MC）: 效率加权版本，与 writeResult 的 interference 矩阵同源
    //   两者都吃各自的权重（phsp_truth_weights / phsp_weights）。
    //
    // 两种拟合分数口径都从这里出:
    //   ctpwa（非相干分母）: FF_i = M[i][i] / Σ_j M[j][j]        （本库 getFitFractions）
    //   tf-pwa / 文献（相干分母, Laura++ Eq.22-23 / LHCb Eq.4）:
    //       W = Σ_i M[i][i] + Σ_{i<j} M[i][j]      （= ∫w|ΣA|²）
    //       FF_i = M[i][i] / W,  FF_ij = M[i][j] / W,  Σ_i FF_i + Σ_{i<j} FF_ij = 1
    //   两个样本的 W 之比 = 模型加权的平均效率 ε̄（BF 换算要用，见
    //   doc/branching_fraction_recipe.md）。
    // 缺对应数据键 → 返回空 [0,0]。不传 Hessian：这里只要积分，不要误差。
    // =====================================================================
    // 重建侧矩阵：常驻 B̄ = Σ_e (w_e/W_total) A_e A_e^H 与 v 的二次型。
    // B̄ 在构造期、reCalcAmp(θ)、getHessian 里都会刷新，所以这里不读文件、不跑逐事例 kernel，
    // 只做 n×n 收缩；返回 Σ w A A^H 口径（乘回 W_total）。
    torch::Tensor phspMatrixFromGram(const torch::Tensor& vector, double* W_out = nullptr,
                                     double* Wtotal_out = nullptr)
    {
        // P4：计算在 ctpwa::phspMatrixFromGram（include/PhspGram.h），这里只把状态显式化。
        ctpwa::PhspGramInputs in;
        in.events = &events_;
        in.phsp_weights = &phsp_weights_;
        in.nSLvectors = &nSLvectors_;
        in.d_matrix = d_phsp_matrix_;
        in.d_matrix_double = d_phsp_matrix_double_;
        in.n_amplitudes = n_amplitudes_;
        in.primary_dev = primary_dev_;
        return ctpwa::phspMatrixFromGram(in, vector, W_out, Wtotal_out);
    }

    // 内部用：给定耦合向量算某样本的原始积分矩阵（rec 走常驻 B̄，truth 走一趟扫描）。
    torch::Tensor matrixForSample(torch::Tensor vector, std::string sample)
    {
        // P4：实现已搬到 ctpwa::matrixForSample（include/FitFractions.h）
        return ctpwa::matrixForSample(fitFractionCtx(), vector, std::move(sample));
    }

    // =====================================================================
    // 两张干涉矩阵的**参数误差**（只含拟合参数，不含 MC 统计——后者是系统项）
    //
    // 线性化传播：σ(F_ij) = sqrt( J_ij^T · Cov · J_ij )，
    //   F_ij = M_ij / W（与 writeResult 写出的归一化表同口径），
    //   J 的列 = 自由方向：耦合（去掉参考方向 Re v_0, Im v_0）+ θ（质量/宽度）。
    //   Cov = H[free, free]^{-1}，H 为统一 Hessian（与 getHessian 同源；不传则内部算）。
    //   耦合方向用 ±ε 差分（ε = 5e-6·max(|p|,1)），θ 用 ±ε（ε = 1e-4·max(|θ|,1e-3)）。
    //
    // 成本 = 2·n_free 趟积分（每趟 phsp + phsp_truth 各一次全样本扫描）。θ 全固定时
    // 只有 2n-2 趟；比 getFitFractions 的扰动多，因为这里要整张矩阵。
    //
    // 返回 {err_rec, err_gen}: 两个 [npartials, npartials] float64（未归一化误差为 0 的
    // 情况返回空张量）。缺 phsp_truth 时 err_gen 返回空。
    // =====================================================================
    // 内部用：两张矩阵的参数误差（耦合方向解析；θ 见下方 TODO）。
    std::vector<torch::Tensor> computeMatrixErrors(
        torch::Tensor params, torch::Tensor hessian_in);

    // =====================================================================
    // 效率: ε_i = (Σ_{phsp} w|A_i|² / Σ_{phsp} w)
    //          / (Σ_{phsp_truth} w|A_i|² / Σ_{phsp_truth} w)
    //   phsp（如 cut 后 MC）= 带效率样本; phsp_truth = 无效率 MC truth。
    //   ⇒ 分波加权的探测/选择效率: ∫|A_i|²ε(x)dΦ / ∫|A_i|²dΦ。
    //   ε 是拟合依赖的（振幅形状含共振参数）, 用拟合后 v 计算。
    //   权重来自 config 的 phsp_weights / phsp_truth_weights（缺省 w≡1）。
    // 误差 = 参数误差（Jacobian @ H⁻¹ @ Jᵀ, 与 getFitFractions 同机制）⊕
    //        MC 统计误差（tf-pwa add_int_error 同款: σ²/ε² = S_eff/I²_eff + S_tr/I²_tr,
    //        S = Σ(w|A_i|²)², I = Σ w|A_i|²；w≡1 时退回 Σ|A_i|⁴ / Σ|A_i|²）。
    // 返回 [npartials, 2]: [center, error]
    // =====================================================================
    // ---------------------------------------------------------------------
    // [暂留 · 到 v0.4 再评估]
    // 中心值可由 ROOT 直接算：ε_i = (P_ii/N_rec)/(T_ii/N_gen)；
    // 但**MC 统计误差只在这里**（σ²/ε² = S_eff/I² + S_tr/I² 那一项）。
    // 一旦把 MC 误差也写进 weight_result.root（integrals 树或单独一列），
    // 本函数即可与 getFitFractions 一起删。
    // ---------------------------------------------------------------------
    torch::Tensor getEfficiency(torch::Tensor vector)
    {
        return getEfficiency(vector, torch::Tensor());
    }
    torch::Tensor getEfficiency(torch::Tensor vector,
        torch::Tensor hessian_in)
    {
        // P4：实现已搬到 ctpwa::efficiency（include/FitFractions.h）
        return ctpwa::efficiency(fitFractionCtx(), vector, hessian_in);
    }

    ////////////////////////
    // 驻留振幅段（gpu0）→ double 复杂张量克隆。float 模式（A 存 float2, 8B/元素）
    // 时按 元素偏移 上转 double 再克隆——保持 double .so 的 complex128 契约,
    // 同时不改动双精度路径的任何字节。峰值瞬态 = 该段 float + double, 用完即释放。
    torch::Tensor ampSegmentToTensor(ctComplex* base, int64_t offset_el,
        int64_t nEl, int gpu = 0) const
    {
        // P4：实现已搬到 ctpwa::ampSegmentToTensor（include/SLAmpsIO.h）
        return ctpwa::ampSegmentToTensor(base, offset_el, nEl, gpu, float_amps_);
    }

    // ---------------------------------------------------------------------
    // [可删 · 零消费的调试接口] 三个张量 getter 只差 offset，可合并成
    //   getAmpTensor(sample="data"|"phsp"|"bkg")；目前 example/tests 均不用。
    // ---------------------------------------------------------------------
    torch::Tensor getDataTensor() const
    {
        return ampSegmentToTensor(d_all_amplitudes_[0], amp_offsets_[0][1],
            (int64_t)events_[0][1] * n_polar_ * n_amplitudes_);
    }

    int getNDevices() const { return static_cast<int>(d_all_amplitudes_.size()); }

    torch::Tensor getPhspTensor() const { return getPhspTensor(0); }

    // 调试用：读取指定 GPU 的 phsp 振幅。默认无参版本仍只返回 GPU0，
    // 多卡独立重建必须由调用方遍历 [0,getNDevices()) 后拼接。
    torch::Tensor getPhspTensor(int gpu) const
    {
        TORCH_CHECK(!phsp_freed_,
            "Constraints.free_phsp_amplitudes=true 模式下 phsp 振幅不驻留（流式），"
            "getPhspTensor 不可用；请关闭该开关，或用 analysis.getNLL（C++ 路径）拟合");
        TORCH_CHECK(gpu >= 0 && gpu < static_cast<int>(d_all_amplitudes_.size()),
            "getPhspTensor: gpu=", gpu, " 超出设备数 ", d_all_amplitudes_.size());
        return ampSegmentToTensor(d_all_amplitudes_[gpu], 0,
            (int64_t)events_[gpu][0] * n_polar_ * n_amplitudes_, gpu);
    }

    // ---------------------------------------------------------------------
    // [可删 · 零消费] 见 getDataTensor 的说明。
    // ---------------------------------------------------------------------
    torch::Tensor getBkgTensor() const
    {
        if (events_[0].size() <= 2 || events_[0][2] == 0)
            return torch::empty({ 0 }, torch::TensorOptions().dtype(TORCH_COMPLEX).device(ctpwa_dev(0)));
        return ampSegmentToTensor(d_all_amplitudes_[0], amp_offsets_[0][2],
            (int64_t)events_[0][2] * n_polar_ * n_amplitudes_);
    }

    // ---------------------------------------------------------------------
    // [可删 · 零消费] 返回 float32，与全库 float64 口径也不一致。
    // ---------------------------------------------------------------------
    torch::Tensor getBkgWeightsTensor() const
    {
        // if (bkg_weights_ != nullptr && bkg_length > 0)
        if (events_[0].size() > 2 && events_[0][2] > 0 && !bkg_weights_.empty() && bkg_weights_[0] != nullptr)
        {
            torch::Tensor output = torch::from_blob(bkg_weights_[0], { events_[0][2] },
                torch::TensorOptions().dtype(torch::kFloat).device(ctpwa_dev(0))).clone();
            return output;
        }
        else
        {
            return torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kFloat).device(ctpwa_dev(0)));
        }
    }
    // ---------------------------------------------------------------------
    // [可删 · 零消费的调试接口] 与 getSLAmpsTensor 合并或直接删。
    // ---------------------------------------------------------------------
    void saveSLAmps(const std::string& filename) const
    {
        // P4：实现已搬到 ctpwa::saveSLAmpsToFile（include/SLAmpsIO.h）
        ctpwa::saveSLAmpsToFile(amp_calc_, filename, phsp_freed_);
    }

    // 返回所有链拼接的 SLAmps tensor，shape [totalSL, nEv, nPol]
    // ---------------------------------------------------------------------
    // [可删 · 零消费的调试接口] 见 saveSLAmps。
    // ---------------------------------------------------------------------
    torch::Tensor getSLAmpsTensor() const
    {
        // P4：实现已搬到 ctpwa::slAmpsTensor（include/SLAmpsIO.h）
        return ctpwa::slAmpsTensor(amp_calc_, phsp_freed_);
    }
    /////////////////////////////

    std::vector<std::vector<int>> getConstraintsIndex() const
    {
        return params_.constraintsIndex();
    }

    std::vector<std::vector<std::pair<double, double>>> getConstraintsValues() const
    {
        std::vector<std::vector<std::pair<double, double>>> output;
        for (const auto& vec : params_.constraintsValues())
        {
            std::vector<std::pair<double, double>> temp;
            for (const auto& val : vec)
            {
                temp.emplace_back(std::make_pair(real(val), imag(val)));
            }
            output.push_back(temp);
        }
        return output;
    }

    int getNPolarizations() const { return n_polar_; }

    // 当前 θ 的 host 副本（空 = 无自由参数或尚未设置 → 用配置初值）
    std::vector<double> currentThetaHost() const
    {
        std::vector<double> h;
        const int nfree = amp_calc_.nFreeResParams();
        if (nfree <= 0 || !theta_cur_.defined() || theta_cur_.numel() != nfree)
            return h;
        auto cpu = theta_cur_.to(torch::kCPU).contiguous();
        h.assign(cpu.data_ptr<double>(), cpu.data_ptr<double>() + nfree);
        return h;
    }

    // phsp Gram 的 double 上转/缩放块不能超过 512 MiB，且至多占当前空闲显存的
    // 1/8。大 nPolar×nAmp 模型若仍按固定 150k 事件分批，单块可达数 GB。
    int phspGramBatchEvents() const
    {
        const size_t per_event = (size_t)n_polar_ * n_amplitudes_ * sizeof(ctComplex);
        size_t free_b = 0, total_b = 0;
        cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
        TORCH_CHECK(err == cudaSuccess,
            "phspGramBatchEvents: cudaMemGetInfo 失败: ", cudaGetErrorString(err));
        const size_t cap = 512ULL * 1024ULL * 1024ULL;
        const size_t budget = std::min(cap, free_b / 8);
        size_t n = per_event > 0 ? budget / per_event : 150000;
        n = std::min<size_t>(150000, std::max<size_t>(1, n));
        return static_cast<int>(n);
    }

    // 从常驻振幅重算 phsp 的 Gram B̄ = Σ_e (w_e/W_total) A_e A_e^H。
    // 与 getHessian 里原来那段一致（分块 scalePhspAmpsKernel + CGEMM + 跨卡 axpy）；
    // 抽出来是为了让 reCalcAmp(θ) 之后 B̄ 也新鲜（此前只在 getHessian 里更新）。
    void rebuildPhspGram()
    {
        // P4：实现已搬到 ctpwa::rebuildPhspGram（include/PhspGram.h）
        ctpwa::rebuildPhspGram(d_all_amplitudes_, events_, phsp_weights_,
                               float_amps_, n_amplitudes_, n_polar_, primary_dev_,
                               &d_phsp_matrix_,
                               [this]() { return phspGramBatchEvents(); });
    }

    void reCalcAmp(torch::Tensor params);

    // ---- DIYMC: 任意四动量的逐事件强度 Σ_pol |Σ_i A_i v_i|² ----
    //
    // momenta: CPU float64 [N, n_final, 4] (E, px, py, pz)，末态顺序 = Data.order。
    // params:  当前后端设备上的统一参数 [Re(v), Im(v), θ]（与 getNLL 同布局）。
    //
    // 内部用局部 AmpCalc（不污染拟合用的 amp_calc_）：临时替换
    // events_offsets_/amp_offsets_ → calculateAmplitudes → reComputeAmps(θ)
    // → computeTotalIntensity → 释放 + 恢复。返回 CPU float64 [N]。
    // 归一化任意（只保证相对值），供舍选/诊断使用。
    torch::Tensor getIntensity(torch::Tensor momenta, torch::Tensor params);

    // ---- DIYMC: 用当前模型参数产生振幅加权 MC ----
    //
    // mode="unweighted": 平坦相空间 → I=getIntensity → u*Imax < I 舍选；
    //                    返回等权事件。
    // mode="weighted":   返回全部平坦事例 + I 权重（不做舍选）。
    // batch: 每次产生/计算的平坦事例数；max_weight=None 时用首批 pilot 自适应；
    // max_trials: 平坦事例数上限（None=不设）。
    // 返回 {"events": [N,n_final,4], "weights": [N] 或空, "stats": {...}}。
    pybind11::dict generateMC(int64_t n, torch::Tensor params,
        uint64_t seed = 0, std::string mode = "unweighted",
        int64_t batch = 100000, pybind11::object max_weight = pybind11::none(),
        pybind11::object max_trials = pybind11::none())
    {
        TORCH_CHECK(initialized_, "generateMC: analysis not initialized");
        TORCH_CHECK(n > 0, "generateMC: n must be positive");
        TORCH_CHECK(ctpwa_is_cuda_like(params),
            "generateMC: params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(params.dtype() == torch::kFloat64 && params.dim() == 1,
            "generateMC: params must be 1-D float64");
        TORCH_CHECK(params.numel() == getNParams(),
            "generateMC: params size mismatch: got ", params.numel(),
            ", expected ", getNParams());
        TORCH_CHECK(mode == "unweighted" || mode == "weighted",
            "generateMC: mode must be 'unweighted' or 'weighted'");
        if (batch < 1) batch = 1;

        const bool weighted = (mode == "weighted");
        auto opts = torch::TensorOptions().dtype(torch::kFloat64);

        PhaseSpaceGenerator ps(config_parser_);
        std::vector<torch::Tensor> ev_chunks, u_chunks, i_chunks;
        int64_t total_flat = 0;
        int64_t total_genbod = 0;
        int64_t accepted_total = 0;
        bool truncated = false;

        bool have_max = !max_weight.is_none();
        double Imax = have_max ? max_weight.cast<double>() : 0.0;
        TORCH_CHECK(!have_max || (Imax > 0.0 && std::isfinite(Imax)),
            "generateMC: max_weight must be positive and finite");

        const int64_t max_trials_val =
            max_trials.is_none() ? -1 : max_trials.cast<int64_t>();

        std::mt19937_64 urng(seed ^ 0xD1B54A32D192ED03ULL);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        int64_t iter = 0;
        while (true) {
            int64_t want = batch;
            if (max_trials_val >= 0) {
                const int64_t left = max_trials_val - total_flat;
                if (left <= 0) {
                    truncated = true;
                    break;
                }
                want = std::min(want, left);
            }
            if (weighted) want = std::min(want, n - total_flat);
            if (want <= 0) {
                truncated = true;
                break;
            }

            const uint64_t batch_seed =
                seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(iter + 1);
            torch::Tensor flat = ps.generate(want, batch_seed);
            total_genbod += ps.lastTrials();
            torch::Tensor I = getIntensity(flat, params);
            total_flat += flat.size(0);

            if (weighted) {
                TORCH_CHECK(torch::isfinite(I).all().item<bool>(),
                    "generateMC: non-finite intensity in weighted mode "
                    "(check params/model)");
                ev_chunks.push_back(flat);
                i_chunks.push_back(I);
                if (total_flat >= n) break;
                if (max_trials_val >= 0 && total_flat >= max_trials_val) {
                    truncated = true;
                    break;
                }
                ++iter;
                continue;
            }

            const double bmax = I.max().item<double>();
            TORCH_CHECK(std::isfinite(bmax) && bmax > 0.0,
                "generateMC: all/部分 intensity 非有限或为零 "
                "(bmax=", bmax, "); 检查 params/模型");
            if (!have_max || bmax * 1.05 > Imax) {
                const double new_max = std::max(bmax * 1.05, 1e-300);
                if (have_max && new_max > Imax) {
                    // Imax 抬高后，已接受事例按保存的 u 精确重判：
                    // 最终判据统一为 u*Imax_final < I，样本无偏。
                    for (size_t c = 0; c < ev_chunks.size(); ++c) {
                        auto keep = (u_chunks[c] * new_max < i_chunks[c]);
                        ev_chunks[c] = ev_chunks[c].index({ keep });
                        u_chunks[c] = u_chunks[c].index({ keep });
                        i_chunks[c] = i_chunks[c].index({ keep });
                    }
                    accepted_total = 0;
                    for (const auto& t : ev_chunks) accepted_total += t.size(0);
                }
                Imax = new_max;
                have_max = true;
            }

            std::vector<double> u(static_cast<size_t>(want));
            for (auto& x : u) x = unif(urng);
            torch::Tensor u_t =
                torch::from_blob(u.data(), { want }, opts).clone();
            auto keep = (u_t * Imax < I);
            torch::Tensor ev_keep = flat.index({ keep });
            ev_chunks.push_back(ev_keep);
            u_chunks.push_back(u_t.index({ keep }));
            i_chunks.push_back(I.index({ keep }));
            accepted_total += ev_keep.size(0);

            if (accepted_total >= n) break;
            if (max_trials_val >= 0 && total_flat >= max_trials_val) {
                truncated = true;
                break;
            }
            ++iter;
        }

        torch::Tensor events, weights;
        const int64_t nf = ps.nFinal();
        if (ev_chunks.empty()) {
            events = torch::empty({ 0, nf, 4 }, opts);
            weights = torch::empty({ 0 }, opts);
        } else if (weighted) {
            events = torch::cat(ev_chunks, 0);
            weights = torch::cat(i_chunks, 0);
            if (events.size(0) > n) {
                events = events.narrow(0, 0, n);
                weights = weights.narrow(0, 0, n);
            }
        } else {
            events = torch::cat(ev_chunks, 0);
            if (events.size(0) > n) events = events.narrow(0, 0, n);
            weights = torch::empty({ 0 }, opts);
        }

        double max_w = Imax;
        if (weighted && weights.numel() > 0)
            max_w = weights.max().item<double>();

        pybind11::dict stats;
        stats["mode"] = mode;
        stats["n_requested"] = n;
        stats["n_events"] = events.size(0);
        stats["n_flat"] = total_flat;
        stats["n_genbod_trials"] = total_genbod;
        stats["n_accepted"] = events.size(0);
        stats["acceptance"] = total_flat > 0
            ? static_cast<double>(events.size(0)) /
                static_cast<double>(total_flat)
            : 0.0;
        stats["flat_acceptance"] = total_genbod > 0
            ? static_cast<double>(total_flat) /
                static_cast<double>(total_genbod)
            : 0.0;
        stats["max_weight"] = max_w;
        stats["truncated"] = truncated;
        stats["seed"] = seed;

        pybind11::dict out;
        out["events"] = events;
        out["weights"] = weights;
        out["stats"] = stats;
        return out;
    }

    // 把 generateMC 的结果写成 DAT/ROOT（末态顺序 = Data.order；ROOT 分支 =
    // 粒子名，与 readMomentaFromDat 的约定一致）。weights 为空 = unweighted。
    void writeMC(torch::Tensor events, const std::string& filename,
        pybind11::object weights = pybind11::none()) const
    {
        PhaseSpaceGenerator ps(config_parser_);
        if (weights.is_none())
            ps.write(events, filename);
        else
            ps.write(events, filename, weights.cast<torch::Tensor>());
    }

    torch::Tensor getFreeResParams() const
    {
        const auto& slots = amp_calc_.slots();
        int n = static_cast<int>(slots.size());
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
        torch::Tensor result = torch::empty({ 3, n }, options);
        auto acc = result.accessor<double, 2>();
        for (int i = 0; i < n; ++i) {
            acc[0][i] = slots[i].init_value;
            acc[1][i] = slots[i].lower;
            acc[2][i] = slots[i].upper;
        }
        return result;
    }

    std::vector<std::string> getAmplitudeNames() const
    {
        return amplitude_names_;
    }

private:
    int n_gls_;
    int n_polar_ = 1;
    int n_coupling_names_ = 0;   // 耦合参数名段长度（fix_var/var_equal 后 theta 名重建用）
    int n_polar_total_ = 1;               // total tensor polarizations (before mask)
    std::vector<int> polarization_map_;    // output_idx -> tensor_idx for polarization mask
    std::vector<int> nSLvectors_;
    // 当前自由共振态参数（设备张量，最后调用者给的值）。真值积分
    // （getFitFractions/getEfficiency/矩阵/ writeResult 原始矩阵）从这里取 θ —— 那条
    // 路径重建振幅用的是配置粒子对象，不读驻留振幅，必须显式带上 θ。
    torch::Tensor theta_cur_;

    // 振幅数据，设备端
    std::vector<ctComplex*> d_all_amplitudes_;
    ctComplex* d_phsp_matrix_ = nullptr;
    // 双精度 phsp 矩阵（Constraints.free_phsp_amplitudes 流式模式用，主 GPU）：
    // phsp_sum = Re(v^H M_double v)，替代对原始 phsp 振幅的逐 forward 扫描
    cuDoubleComplex* d_phsp_matrix_double_ = nullptr;
    bool phsp_freed_ = false;   // true = phsp 振幅不驻留（流式模式）
    bool float_amps_ = false;   // true = 混合精度: 驻留振幅矩阵存 float（double .so 上 precision:float 或 auto 默认）
    PrecMode prec_mode_ = PrecMode::Double;  // 运行期档位（config precision）: Float=全 float;
                                             // Double=全 double（hybrid/auto 记 Double + float_amps_）
    std::vector<double*> data_weights_;
    std::vector<double*> phsp_weights_;
    std::vector<double*> bkg_weights_;
    double bkg_integral_ = 0.0;
    double data_total_weight_ = 0.0;   // Σ data 权重（无权重时 = data 事件数）

    // 设备管理（GPU 枚举、内存预检；预留 CPU/双精度扩展）
    DeviceManager device_mgr_;

    // 事件数量、振幅偏移等信息
    int n_gpus_ = 0;
    std::vector<std::vector<int>> events_;
    std::vector<std::vector<int>> events_offsets_;
    std::vector<std::vector<int>> amp_offsets_;

    // 四动量数据，主机端
    std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_all_;

    // amplitude 信息
    std::vector<std::string> amplitude_names_;
    std::vector<std::string> resonance_names_;
    std::vector<std::string> legends_;

    // 参数管理器（统一管理 vector / theta / 约束 / 耦合矩阵）
    Parameters params_;

    // config 初始化
    ConfigParser config_parser_;
    std::vector<Particle> particles_;
    std::unordered_map<std::string, Resonance> resonances_;
    int n_amplitudes_ = 0;
    std::vector<ChainInfo> chains_info_;
    AmpCalc amp_calc_;
    bool initialized_ = false;
    int primary_dev_ = 0;  // 主 GPU：构造时 torch 当前设备；d_phsp_matrix_ 等跨 GPU 缓冲所在

    // fit mode: 0 = FREEPARAMS (chain×step, default), 1 = VSPACE (direct amplitudes)
    int fit_mode_ = 0;
    // 构造期冻结的 fit mode（参数化实际生效的模式）；构造后 setFitMode 只改语义
    int frozen_fit_mode_ = 0;

    void initialize(std::string config_file = "config.yml");

    void initializeMultiGPUs(std::vector<int> init_events);

    void initializeParticles()
    {
        const auto& config_particles = config_parser_.getParticles();
        for (const auto& particle_config : config_particles)
        {
            particles_.push_back(particle_config);
        }
    }

    void initializePolarization()
    {
        // P4：组合数学在 ctpwa::initializePolarization（include/Polarization.h）
        ctpwa::PolarizationInit pol = ctpwa::initializePolarization(particles_);
        n_polar_total_ = pol.n_polar_total;
        n_polar_ = pol.n_polar;
        polarization_map_ = std::move(pol.map);
        std::cout << "polarization: " << n_polar_ << std::endl;
    }

    void initializeDecayChains();

    // 是否能在主振幅构造时按 JP 组提前释放固定 SL 表。
    // trans/var_equal 的自由参数归属要等所有 block 注册后才能确定，必须保留到
    // 原来的统一释放阶段；CTPWA_KEEP_SLAMP=1 时同样禁用提前释放。
    bool earlyStaticSLReleaseEnabled() const
    {
        if (std::getenv("CTPWA_KEEP_SLAMP") != nullptr) return false;
        for (const auto& c : config_parser_.getConstraints())
            if (c.type == "trans" || c.type == "var_equal") return false;
        return true;
    }

    // theta_override: 自由共振态参数（按 amp_calc slots 顺序；nullptr = 用配置初始值）。
    // 必须在这里生效：本函数用新构造的 AmpCasDecay 从**配置粒子对象**重建振幅，
    // 那批对象里的 m0/Γ0 是配置初值；reComputeAmps 只改驻留振幅，管不到这里。
    // 映射靠 AmpCalc::slots_ 的 (block_idx, res_idx, param_idx) —— block_idx 与
    // 本函数里 addBlock 的调用顺序一致（每 chain×组合×共振组合一个 block）。
    std::vector<ctComplex*> calculateAmplitudes(
        const std::vector<std::map<std::string, std::vector<LorentzVector>>>& Vp4,
        AmpCalc* amp_calc = nullptr, int n_gpus_used = -1,
        const double* theta_override = nullptr) const
    {
        int block_cursor = 0;   // 与 AmpCalc::blocks_ 顺序一致的 block 计数器（θ 覆盖用）
        // n_gpus_used < 0 = 现有行为（全部 GPU）；getIntensity 在 N < n_gpus_ 时
        // 只启用前 N 个 GPU，避免空分片让 computeSLAmps 解引用空 map。
        const int n_gpu_eff = (n_gpus_used > 0)
            ? std::min(n_gpus_used, n_gpus_) : n_gpus_;

        // 多GPU支持：为每个GPU分配振幅内存
        // int num_gpus = events_.size();
        std::vector<ctComplex*> d_all_amplitudes_vec(n_gpu_eff, nullptr);

        // 计算每个GPU的总事件数
        // std::vector<int> nEvents_per_gpu(n_gpus_, 0);
        // for (int gpu = 0; gpu < n_gpus_; ++gpu) {
        //     for (size_t file_type = 0; file_type < events_[gpu].size(); ++file_type) {
        //         nEvents_per_gpu[gpu] += events_[gpu][file_type];
        //     }
        // }

        // 为每个GPU分配设备内存（float_amps_: A 存 float2 → 字节减半；元素偏移
        // amp_offsets 语义不变，消费端按 float2* reinterpret 读取）
        size_t amp_elem_bytes = float_amps_ ? sizeof(float2) : sizeof(ctComplex);
        for (int gpu = 0; gpu < n_gpu_eff; ++gpu) {
            if (amp_offsets_[gpu].back() > 0) {
                cudaSetDevice(gpu);
                cudaMalloc(&d_all_amplitudes_vec[gpu], amp_offsets_[gpu].back() * amp_elem_bytes);
            }
        }
        // cudaSetDevice(0);

        // std::cout << "Calculating amplitudes for " << num_gpus << " GPUs..." << std::endl;

        auto chains = config_parser_.getDecayChains();

        // Build chain link map from trans constraints (substring match short→full names)
        //   chain_linked["decay1_R_Keta_0"] = {"decay1_R_Keta_1"}
        std::map<std::string, std::set<std::string>> chain_linked;
        for (const auto& c : config_parser_.getConstraints()) {
            if (c.type == "trans" && c.names.size() >= 2) {
                // Resolve constraint names to full chain names via substring
                std::vector<std::string> matched;
                for (const auto& cn : c.names) {
                    for (const auto& ch : chains) {
                        if (ch.name.find(cn) != std::string::npos) {
                            matched.push_back(ch.name);
                            break;
                        }
                    }
                }
                for (size_t i = 0; i < matched.size(); ++i)
                    for (size_t j = 0; j < matched.size(); ++j)
                        if (i != j) chain_linked[matched[i]].insert(matched[j]);
            }
        }

        // 每个 chain 有多个 JP 组合，每个组合生成一组 block。
        // chain_resonances: chain → [[comb0_res0, comb0_res1, ...], [comb1_res0, ...], ...]
        // 组合索引 ci 用于跨链 trans 匹配同一 JP 位置的共振态。
        std::map<std::string, std::vector<std::vector<std::string>>> chain_resonances;
        // 每个 chain 当前的组合索引
        std::map<std::string, int> chain_comb_index;

        int gls_index = 0;
        for (auto chain : chains)
        {
            ChainInfo chain_info;
            for (auto chaininfo : chains_info_)
            {
                if (chaininfo.name == chain.name)
                {
                    chain_info = chaininfo;
                    break;
                }
            }
            auto intermediate_resonance_map = chain_info.intermediate_resonance_map;
            auto intermediate_combs = chain_info.intermediate_combs;

            // 跨链全同粒子: 收集链中出现的全同组（≥2 成员）→ cas 生成置换拓扑。
            // 全同组由 Constraints.identical 声明，自动检测；无全同粒子时结果为空，等效关闭。
            // coset 生成（子树感知、稳定子、费米子 sgn）在 computeSLAmps 内部完成，
            // 这里只需把 (成员名, is_boson) 传给 cas。
            std::vector<std::pair<std::vector<std::string>, bool>> identical_groups;
            {
                std::set<std::string> chain_particles;
                for (const auto& step : chain.decay_steps) {
                    chain_particles.insert(step.mother);
                    for (const auto& d : step.daughters) chain_particles.insert(d);
                }
                std::map<std::string, std::vector<std::string>> groups_by_name;
                std::map<std::string, bool> group_boson;
                for (const auto& p : particles_) {
                    if (p.identical_group.empty() ||
                        chain_particles.find(p.name) == chain_particles.end())
                        continue;
                    groups_by_name[p.identical_group].push_back(p.name);
                    group_boson[p.identical_group] = !p.is_fermion();
                }
                for (const auto& [gname, members] : groups_by_name) {
                    if (members.size() >= 2)
                        identical_groups.push_back({ members, group_boson[gname] });
                }
            }

            for (auto comb : intermediate_combs)
            {
                auto cas = std::make_shared<AmpCasDecay>(particles_);
                cas->setChainName(chain.name);
                cas->setNPolarizations(n_polar_);
                cas->setNPolarizationsTotal(n_polar_total_);
                cas->setPolarizationMap(polarization_map_);
                cas->setIdenticalGroups(identical_groups);
                for (const auto& step : chain.decay_steps)
                {
                    std::array<int, 3> spins = { 0 };
                    std::array<int, 3> parities = { 0 };
                    for (auto particle : particles_)
                    {
                        if (particle.name == step.mother)
                        {
                            spins[0] = particle.spin;
                            parities[0] = particle.parity;
                        }

                        for (int i = 0; i < step.daughters.size(); i++)
                        {
                            if (particle.name == step.daughters[i])
                            {
                                spins[i + 1] = particle.spin;
                                parities[i + 1] = particle.parity;
                            }
                        }
                    }
                    for (auto res_jp : comb)
                    {
                        if (res_jp.name == step.mother)
                        {
                            spins[0] = res_jp.spin;
                            parities[0] = res_jp.parity;
                        }

                        for (int i = 0; i < step.daughters.size(); i++)
                        {
                            if (res_jp.name == step.daughters[i])
                            {
                                spins[i + 1] = res_jp.spin;
                                parities[i + 1] = res_jp.parity;
                            }
                        }
                    }
                    // 检查两个子粒子是否全同（Constraints.identical 自动检测）
                    bool identical_daughters2 = false;
                    bool is_boson2 = true;
                    {
                        const Particle* p_d1 = nullptr, * p_d2 = nullptr;
                        for (const auto& p : particles_) {
                            if (p.name == step.daughters[0]) p_d1 = &p;
                            if (p.name == step.daughters[1]) p_d2 = &p;
                        }
                        if (p_d1 && p_d2 &&
                            !p_d1->identical_group.empty() &&
                            p_d1->identical_group == p_d2->identical_group) {
                            identical_daughters2 = true;
                            is_boson2 = !p_d1->is_fermion();
                        }
                    }
                    // 分波白名单: [[S, L], ...] → set<pair<int,int>>
                    std::set<std::pair<int, int>> sl_filter;
                    for (const auto& sf : step.sl_filter)
                        if (sf.size() >= 2) sl_filter.insert({ sf[0], sf[1] });
                    int maxL2 = config_parser_.getGlobalMaxL();
                    Amp2BD amp2bd(spins, parities, identical_daughters2, is_boson2, maxL2,
                                  step.p_break, step.has_bf, step.bf_d, std::move(sl_filter));
                    // 零-SL 诊断：该步无任何合法 (S,L) 分波时振幅恒为零。
                    // 原因通常是 [J,P] 物理禁戒（角动量/宇称不守恒）、全同粒子
                    // 选择定则排除、L > maxL，或 sl 白名单过滤掉了全部波。
                    if (amp2bd.getSL().empty()) {
                        std::cerr << "Warning: chain \"" << chain.name << "\" step "
                                  << step.mother << "[" << jpLabel(spins[0], parities[0])
                                  << "] -> " << step.daughters[0] << "["
                                  << jpLabel(spins[1], parities[1]) << "] + "
                                  << step.daughters[1] << "["
                                  << jpLabel(spins[2], parities[2]) << "]"
                                  << " has no valid (L,S) waves"
                                  << " (maxL=" << maxL2
                                  << (step.sl_filter.empty() ? "" : ", sl whitelist active")
                                  << "): amplitude will be identically zero. "
                                  << "Remove this [J,P] group or relax the constraints."
                                  << std::endl;
                    }
                    cas->addDecay(std::move(amp2bd), step.mother, step.daughters[0], step.daughters[1]);
                }

                auto slcombs = cas->getSLCombinations();
                // 链级诊断：任一步零-SL 都会使整条链的 SL 组合（笛卡尔积）为空，
                // 链内所有共振态振幅恒为零——这是比单步更严重的静默失败。
                if (slcombs.empty()) {
                    std::cerr << "Warning: chain \"" << chain.name << "\" has NO valid (L,S) wave "
                                 "combinations across its decay steps: all amplitudes of this "
                                 "chain will be identically zero. Check the per-step warnings above."
                              << std::endl;
                }

                std::vector<std::vector<Resonance>> resonance_combinations = {
                    {} };
                for (const auto& particle : comb)
                {
                    std::pair<std::string, std::vector<int>> key = {
                        particle.name, {particle.spin, particle.parity} };
                    std::vector<Resonance>& resonance_list =
                        intermediate_resonance_map[key];

                    std::vector<std::vector<Resonance>> temp;
                    for (const auto& res_comb : resonance_combinations)
                    {
                        for (const auto& resonance : resonance_list)
                        {
                            std::vector<Resonance> new_res_comb = res_comb;
                            new_res_comb.push_back(resonance);
                            temp.push_back(std::move(new_res_comb));
                        }
                    }
                    resonance_combinations = std::move(temp);
                }

                // Constraints.fix_var: 覆盖命名参数值（fix_var \ free_var）。
                // 作用于组合副本，不影响其他 chain 的共振态对象。
                {
                    const auto& fix_var_map = config_parser_.getFixVar();
                    const auto& free_var_set = config_parser_.getFreeVar();
                    if (!fix_var_map.empty()) {
                        for (auto& res_combo : resonance_combinations) {
                            for (auto& res : res_combo) {
                                const auto& pnames = res.getOrderedParamNames();
                                for (size_t pi = 0; pi < pnames.size(); ++pi) {
                                    std::string nm = res.getName() + "_" + pnames[pi];
                                    auto it = fix_var_map.find(nm);
                                    if (it != fix_var_map.end() && !free_var_set.count(nm))
                                        res.setParam(static_cast<int>(pi), it->second);
                                }
                            }
                        }
                    }
                }

                // 计算SL振幅（多GPU版本）
                // θ 覆盖：把当前自由参数写进本组的 Resonance 副本（不改 chains_info_）。
                // block_idx 顺序 = addBlock 调用顺序 = 本循环 (chain×comb×resonance_combo)。
                if (theta_override != nullptr) {
                    const AmpCalc* slot_src = (amp_calc != nullptr) ? amp_calc : &amp_calc_;
                    const int n_combo = static_cast<int>(resonance_combinations.size());
                    for (int q = 0; q < n_combo; ++q) {
                        const int bi = block_cursor + q;
                        for (const auto& b : slot_src->paramBindings(bi)) {
                            const int ri = b[0];
                            if (ri < 0 || ri >= static_cast<int>(resonance_combinations[q].size()))
                                continue;
                            resonance_combinations[q][ri].setParam(b[1], theta_override[b[2]]);
                        }
                    }
                }
                cas->computeSLAmps(Vp4);
                // int nSLcombs = cas->getNSLCombs();
                // int nEvents = cas->getNEvents();

                // 调用多GPU版本的getAmps函数
                for (const auto resonance : resonance_combinations)
                {
                    cas->getAmps(d_all_amplitudes_vec, resonance, gls_index, n_amplitudes_, events_offsets_, amp_offsets_, float_amps_);

                    // 如果有 scan 配置且 amp_calc 非空，注册到 AmpCalc
                    if (amp_calc)
                    {
                        const auto& config_res = config_parser_.getResonances();
                        const auto& fix_var_map = config_parser_.getFixVar();
                        const auto& free_var_set = config_parser_.getFreeVar();
                        const auto& var_range_map = config_parser_.getVarRange();
                        std::vector<std::vector<int>> all_free;
                        std::vector<std::vector<std::vector<double>>> all_free_ranges;
                        std::set<std::string> skip_slots_for;
                        std::map<std::string, std::string> conjugate_name_map;

                        // 当前 chain 的组合索引（每个 JP 组合对应一个 ci）
                        int ci = chain_comb_index[chain.name]++;

                        for (size_t ri = 0; ri < resonance.size(); ++ri) {
                            const auto& res = resonance[ri];
                            auto it = config_res.find(res.getName());
                            // 参数自由度统一语义（所有模型一致）：resonance 未显式声明
                            // free 就不拟合；free: [-1] 表示全部拟合。Custom 模型不再特例。
                            if (it != config_res.end() && !it->second.free.empty())
                            {
                                std::vector<int> free_idx;
                                if (it->second.free.size() == 1 && it->second.free[0] == -1) {
                                    for (int pi = 0; pi < (int)res.getOrderedParamNames().size(); ++pi)
                                        free_idx.push_back(pi);
                                } else {
                                    free_idx = it->second.free;
                                }
                                // fix_var: 从自由列表中移除固定参数（free_var 取消 fix）
                                if (!fix_var_map.empty()) {
                                    const auto& pnames = res.getOrderedParamNames();
                                    for (auto itf = free_idx.begin(); itf != free_idx.end(); ) {
                                        std::string nm = ((int)*itf < (int)pnames.size())
                                            ? res.getName() + "_" + pnames[*itf]
                                            : res.getName() + "_p" + std::to_string(*itf);
                                        if (fix_var_map.count(nm) && !free_var_set.count(nm))
                                            itf = free_idx.erase(itf);
                                        else ++itf;
                                    }
                                }
                                all_free.push_back(std::move(free_idx));
                                all_free_ranges.push_back(it->second.free_range);
                            }
                            else
                            {
                                all_free.push_back({});
                                all_free_ranges.push_back({});
                            }
                            // 跨链 trans 匹配：同一组合索引 ci、同一共振态位置 ri
                            if (skip_slots_for.count(res.getName()) == 0) {
                                const auto& linked = chain_linked[chain.name];
                                for (const auto& linked_chain : linked) {
                                    auto lit = chain_resonances.find(linked_chain);
                                    if (lit != chain_resonances.end()
                                        && ci < (int)lit->second.size()
                                        && ri < lit->second[ci].size()) {
                                        // linked 链任意 comb 存在同名共振 → 不在此 skip：
                                        // 交由 addBlock 的同 rname 全局共享按名字精确配对
                                        // （顺序无关；位置对应在顺序打乱时会错配广播）
                                        bool has_same = false;
                                        for (const auto& per_ci : lit->second)
                                            for (const auto& n : per_ci)
                                                if (n == res.getName()) { has_same = true; break; }
                                        if (has_same) break;
                                        // 不同名（历史 trans 语义，如 N1720p/N1720m）：
                                        // 按位置对应（要求 trans 链接链结构镜像、顺序一致）
                                        const auto& owner_name = lit->second[ci][ri];
                                        skip_slots_for.insert(res.getName());
                                        if (res.getName() != owner_name)
                                            conjugate_name_map[res.getName()] = owner_name;
                                        break;
                                    }
                                }
                            }
                        }
                        // 按组合索引追加共振态名字（不再覆盖）
                        {
                            std::vector<std::string> names;
                            for (const auto& res : resonance)
                                names.push_back(res.getName());
                            if (ci >= (int)chain_resonances[chain.name].size())
                                chain_resonances[chain.name].resize(ci + 1);
                            chain_resonances[chain.name][ci] = std::move(names);
                        }
                        // Always add block — even without free params, its SL channels
                        // contribute to cross-block mixed Hessian (vθ).
                        amp_calc->addBlock(cas, resonance, gls_index, all_free, all_free_ranges,
                                           skip_slots_for, conjugate_name_map, var_range_map);
                    }

                    gls_index += cas->getNSLCombs();
                }
                block_cursor += static_cast<int>(resonance_combinations.size());
                // 当前 JP 的全部共振组合已物化到 A。无跨块共享约束时，立即释放
                // 无自由参数的 SL 表，避免所有 JP 的表一直叠加到构造末尾。
                if (amp_calc && earlyStaticSLReleaseEnabled())
                    amp_calc->releaseStaticCasData();
            }
        }

        // Constraints.var_equal: 所有 block 注册完毕后合并共享参数槽
        // （必须在此处——setNFreeTheta 之前——以便参数数一致）
        if (amp_calc) amp_calc->applyVarEqual(config_parser_.getVarEqual());

        return d_all_amplitudes_vec;
    }

};
