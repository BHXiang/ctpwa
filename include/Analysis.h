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


std::vector<double*> readWeightsFromFile(const std::vector<std::string>& fileinfo, const std::vector<int>& events_per_gpu)
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

std::vector<LorentzVector*> convertToLorentzVectors(
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


std::vector<std::map<std::string, std::vector<LorentzVector>>> mergeMaps(
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
CTPWA_KERNEL void castPhspBatchToDoubleKernel(
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
CTPWA_KERNEL void scalePhspAmpsDoubleKernel(
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
CTPWA_KERNEL void scalePhspSByWeightKernel(
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
CTPWA_KERNEL void axpyZComplexKernel(
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
    void release()
    {
        int saved_dev = 0;
        if (cudaGetDevice(&saved_dev) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            (void)cudaGetLastError();
            d_all_amplitudes_.clear();
            initialized_ = false;
            return;
        }

        for (size_t g = 0; g < d_all_amplitudes_.size(); ++g) {
            cudaSetDevice(static_cast<int>(g));
            if (d_all_amplitudes_[g]) {
                cudaFree(d_all_amplitudes_[g]);
                d_all_amplitudes_[g] = nullptr;
            }
        }
        d_all_amplitudes_.clear();

        ctpwa::ensure_device(primary_dev_);
        if (d_phsp_matrix_) {
            cudaFree(d_phsp_matrix_);
            d_phsp_matrix_ = nullptr;
        }
        if (d_phsp_matrix_double_) {
            cudaFree(d_phsp_matrix_double_);
            d_phsp_matrix_double_ = nullptr;
        }
        auto free_weights = [](std::vector<double*>& weights) {
            for (size_t g = 0; g < weights.size(); ++g) {
                cudaSetDevice(static_cast<int>(g));
                if (weights[g]) cudaFree(weights[g]);
            }
            weights.clear();
        };
        free_weights(data_weights_);
        free_weights(phsp_weights_);
        free_weights(bkg_weights_);

        // 这些表也按 primary_dev_ 分配，先切到主卡再释放。
        params_.freeCouplingData();
        // 释放各 ResBlock、d_dF/static/JIT、slot 及 cas_list_ 的持久化资源。
        // AmpCalc::release 内部会保存并恢复 current device。
        amp_calc_.release();
        initialized_ = false;
        // 析构/释放不能成为新的 current-device 污染源：恢复进入前的设备。
        cudaSetDevice(saved_dev);
    }

    // 析构函数，用于释放 CUDA 内存
    ~analysis() { release(); }

    bool isValid() const { return initialized_; }

    torch::Tensor getNLL(torch::Tensor params)
    {
        TORCH_CHECK(initialized_, "analysis not initialized: invalid or missing config file");

        TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64");
        TORCH_CHECK(ctpwa_on_primary(params, primary_dev_),
            "params 必须位于主设备 (primary=" + std::to_string(primary_dev_) + ")，"
            "当前在 " + params.device().str());

        // 记录当前 θ：真值/积分路径（getFitFractions/getEfficiency/矩阵）不读驻留
        // 振幅，必须显式带 θ；这里顺手记下，拟合循环里最后一次调用的 θ 就是最优点。
        if (params_.nFreeTheta() > 0) {
            const int n_cpl = getNVector();
            if (params.numel() == 2 * n_cpl + params_.nFreeTheta())
                theta_cur_ = params.slice(0, 2 * n_cpl, params.size(0)).detach().clone();
        }

        return NLLFunction::apply(params, &params_, d_all_amplitudes_, &amp_calc_,
            d_phsp_matrix_, d_phsp_matrix_double_, events_, events_offsets_, amp_offsets_,
            data_weights_, phsp_weights_, bkg_weights_, bkg_integral_, n_amplitudes_, n_polar_);
    }

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
                     torch::Tensor hessian_in = torch::Tensor())
    {
        ctpwa::ensure_device(primary_dev_);   // 入口恢复主设备
        TORCH_CHECK(ctpwa_is_cuda_like(params), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");

        // 流式 phsp 分支会临时改写 offsets；异常时也要还原。
        auto wr_ev0 = events_offsets_, wr_amp0 = amp_offsets_;
        auto wr_guard = makeScopeExit([this, wr_ev0, wr_amp0]() {
            events_offsets_ = wr_ev0;
            amp_offsets_ = wr_amp0;
            ctpwa::ensure_device(primary_dev_);
        });

        int npartials_all = (int)nSLvectors_.size();
        std::vector<int> wave_mask_host(npartials_all, 0);
        bool has_waves = !waves.empty();
        if (has_waves) {
            for (int w : waves) {
                TORCH_CHECK(w >= 0 && w < npartials_all,
                    "waves 下标越界: ", w, " (partials = ", npartials_all, ")");
                wave_mask_host[w] = 1;
            }
        }

        torch::Device dev = params.device();
        torch::Tensor extended_vector;

        if (params_.hasCouplingMatrix()) {
            TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64 with coupling matrix");
            const auto& cm = params_.couplingMatrix();
            int ncf = cm.n_free;
            int na  = cm.n_amps;
            int nt  = static_cast<int>(params.size(0)) - 2 * ncf;

            // Extend: [Re_p, Im_p, θ] → [Re_v, Im_v, θ]
            auto ext = torch::empty({2 * na + nt},
                torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            params_.extendCouplingParams(
                params.data_ptr<double>(), ext.data_ptr<double>(), ncf, nt);

            // Build complex vector from extended format
            extended_vector = torch::complex(
                ext.slice(0, 0, na).to(TORCH_FLOAT),
                ext.slice(0, na, 2 * na).to(TORCH_FLOAT));

            // Recompute amplitudes if theta params present
            if (nt > 0) {
                auto theta = params.slice(0, 2 * ncf, params.size(0));
                theta_cur_ = theta.detach().clone();
                amp_calc_.reComputeAmps(d_all_amplitudes_,
                    reinterpret_cast<const double*>(theta.data_ptr()),
                    n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                    dev.index());
                // θ 变了：常驻 B̄ 同步刷新（否则再算矩阵还是旧 θ 的）
                if (!phsp_freed_) rebuildPhspGram();
            }
        } else {
            TORCH_CHECK(params.dtype() == TORCH_COMPLEX, "params must be complex128 in legacy mode");
            extended_vector = params_.extendVector(params, dev);
        }

        // 将extended_vector分配到多个GPU
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            extended_vec_per_gpu.push_back(extended_vector.to(ctpwa_dev(i)));
        }

        const int target_dev = dev.index();

        int npartials = nSLvectors_.size();
        // 每个GPU在自己的设备上分配输出缓冲区
        std::vector<double*> d_final_result_vec;
        std::vector<double*> d_partial_result_vec;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            cudaSetDevice(i);
            double* d_final_result;
            cudaMalloc(&d_final_result, events_[i][0] * sizeof(double));
            d_final_result_vec.push_back(d_final_result);
            double* d_partial_result;
            cudaMalloc(&d_partial_result, events_[i][0] * npartials * sizeof(double));
            // ⚠️ 必须清零: kernel 对 partial 用 += 累加（总权重是 = 赋值），
            // 不清零 = 垃圾内存 → weight_<i> 分支一直是错的（守恒校验发掘）
            cudaMemset(d_partial_result, 0,
                events_[i][0] * npartials * sizeof(double));
            d_partial_result_vec.push_back(d_partial_result);
        }
        cudaSetDevice(target_dev);

        // 分配nSLvectors_的设备内存
        // int* d_nSLvectors;
        // cudaMalloc(&d_nSLvectors, nSLvectors_.size() * sizeof(int));
        // cudaMemcpy(d_nSLvectors, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);

        // 分配干涉矩阵和事件干涉项的设备内存
        // 干涉矩阵不再由 computeResults 累加（改由 computeFileBatchIntegrals 出
        // interference_rec/gen），这里传 nullptr 省掉每事例 n² 次原子操作。

        int N_phsp = 0;
        int N_data = 0;
        int N_bkg = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            cudaSetDevice(gpu);
            int phsp_events = events_[gpu][0];
            N_phsp += static_cast<int>(phsp_events);
            int data_events = events_[gpu][1];
            N_data += static_cast<int>(data_events);
            int bkg_events = (events_[gpu].size() > 2) ? events_[gpu][2] : 0;
            N_bkg += static_cast<int>(bkg_events);
        }

        double h_phsp_integral = 0.0;
        double* h_total_results = new double[N_phsp];
        double* h_partial_results = new double[N_phsp * npartials];
        // 中途抛异常（Hessian 维度/ROOT 写入失败等）时也不能漏掉这两块 host
        // 缓冲：大样本下 N_phsp×npartials×8B 可达 GB 级。
        auto h_guard = makeScopeExit([&]() {
            delete[] h_total_results;
            delete[] h_partial_results;
        });
        int ev_cumulative = 0;  // 多GPU累加偏移
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            cudaSetDevice(gpu);
            int* d_nSLvectors;
            cudaMalloc(&d_nSLvectors, nSLvectors_.size() * sizeof(int));
            cudaMemcpy(d_nSLvectors, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);
            // 单个GPU integral
            double* d_total_integral_gpu;
            cudaMalloc(&d_total_integral_gpu, sizeof(double));
            cudaMemset(d_total_integral_gpu, 0, sizeof(double));
            // 单个GPU event interference（本函数不用, 传 nullptr; writeInterfResult 使用）
            // 子集选择（waves 非空时）: mask + 子集积分缓冲
            int* d_wave_mask_gpu = nullptr;
            double* d_selected_integral_gpu = nullptr;
            if (has_waves) {
                cudaMalloc(&d_wave_mask_gpu, npartials * sizeof(int));
                cudaMemcpy(d_wave_mask_gpu, wave_mask_host.data(),
                    npartials * sizeof(int), cudaMemcpyHostToDevice);
                cudaMalloc(&d_selected_integral_gpu, sizeof(double));
                cudaMemset(d_selected_integral_gpu, 0, sizeof(double));
            }

            /////////////////////////
            /////////////////////////
            if (!phsp_freed_) {
                // phsp 驻留模式: 逐批上转 + computeResults 累加。原实现整表
                // upcast(double 副本) + computeResults 内部整段临时, 大 phsp 下
                // 峰值 = 2×PA×nEv×16B(可达 7GB+), A100-40GB 实测 OOM →
                // 改按批（峰值 ~批内 2×PA×16B, 默认 100k 事件 ≈ 0.3GB）。
                int ne = events_[gpu][0];
                if (ne > 0) {
                    long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
                    int batch_cap = 100000;
                    if (const char* e = getenv("CTPWA_WR_BATCH")) {
                        int v = atoi(e);
                        if (v > 0) batch_cap = v;   // 显式覆盖（回归测试/调优用）
                    } else if (per_ev_bytes > 0) {
                        long long cap = 200LL * 1024 * 1024 / per_ev_bytes;
                        if (cap < 2000) cap = 2000;
                        if (cap < batch_cap) batch_cap = (int)cap;
                    }
                    const ctComplex* d_amp_base = d_all_amplitudes_[gpu];
                    for (int c0 = 0; c0 < ne; c0 += batch_cap) {
                        int nch = std::min(batch_cap, ne - c0);
                        const ctComplex* d_amp_seg = d_amp_base;
                        ctComplex* d_amp_dbl = nullptr;
                        if (float_amps_ && !ctIsFloatComplex) {
                            // 注意: A 表为 float2(8B) 存储而 d_amp_base 是
                            // ctComplex*(16B 语义) —— 段偏移必须按 float2 算术,
                            // 否则 c0>0 的批指针翻倍错位（分批后 hfit 全 nan）
                            const float2* f2 =
                                reinterpret_cast<const float2*>(d_amp_base)
                                + (size_t)c0 * n_polar_ * n_amplitudes_;
                            d_amp_dbl = upcastAmpSegToDouble(
                                reinterpret_cast<const ctComplex*>(f2),
                                (size_t)nch * n_polar_ * n_amplitudes_);
                            d_amp_seg = d_amp_dbl;
                        } else {
                            d_amp_seg =
                                d_amp_base + (size_t)c0 * n_polar_ * n_amplitudes_;
                        }
                        double* d_ct = nullptr;
                        double* d_cp = nullptr;
                        if (cudaMalloc(&d_ct, (size_t)nch * sizeof(double))
                                != cudaSuccess
                            || cudaMalloc(&d_cp, (size_t)nch * npartials
                                                     * sizeof(double))
                                   != cudaSuccess) {
                            char buf[256];
                            snprintf(buf, sizeof(buf),
                                     "writeResult: 批缓冲分配失败 (nch=%d)",
                                     nch);
                            throw std::runtime_error(buf);
                        }
                        cudaMemset(d_cp, 0,
                                   (size_t)nch * npartials * sizeof(double));
                        computeResults(d_amp_seg,
                            reinterpret_cast<const ctComplex*>(
                                extended_vec_per_gpu[gpu].data_ptr()),
                            d_ct, d_total_integral_gpu, d_cp,
                            nullptr, nullptr,
                            d_wave_mask_gpu, d_selected_integral_gpu,
                            d_ct,
                            nullptr, 0,
                            d_nSLvectors, npartials, nch, n_amplitudes_,
                            n_polar_);
                        // 批结果拷回全尺寸缓冲偏移（total 段 = 赋值, partial
                        // 段 [p][ev] 主序）
                        cudaMemcpy(d_final_result_vec[gpu] + c0, d_ct,
                                   (size_t)nch * sizeof(double),
                                   cudaMemcpyDeviceToDevice);
                        for (int p = 0; p < npartials; ++p)
                            cudaMemcpy(d_partial_result_vec[gpu]
                                           + (size_t)p * ne + c0,
                                       d_cp + (size_t)p * nch,
                                       (size_t)nch * sizeof(double),
                                       cudaMemcpyDeviceToDevice);
                        cudaFree(d_ct);
                        cudaFree(d_cp);
                        if (d_amp_dbl) cudaFree(d_amp_dbl);
                    }
                }
            } else {
                // phsp 流式模式: 按批重算 phsp 振幅 → computeResults 逐批累加
                int ne = events_[gpu][0];
                if (ne > 0) {
                    const int PHSP_BATCH = 100000;
                    long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
                    int batch_cap = PHSP_BATCH;
                    if (per_ev_bytes > 0) {
                        long long cap = 200LL * 1024 * 1024 / per_ev_bytes;
                        if (cap < 2000) cap = 2000;
                        if (cap < batch_cap) batch_cap = (int)cap;
                    }
                    auto saved_ev = events_offsets_, saved_amp = amp_offsets_;
                    for (int c0 = 0; c0 < ne; c0 += batch_cap) {
                        int nch = std::min(batch_cap, ne - c0);
                        std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_chunk(n_gpus_);
                        for (int g = 0; g < n_gpus_; ++g) {
                            if (g == (int)gpu) {
                                for (const auto& [k, v] : Vp4_all_[gpu])
                                    Vp4_chunk[g][k].assign(v.begin() + c0, v.begin() + c0 + nch);
                                events_offsets_[g] = {0, nch};
                                amp_offsets_[g] = {0, nch * n_polar_ * n_amplitudes_};
                            } else {
                                for (const auto& [k, v] : Vp4_all_[g])
                                    Vp4_chunk[g][k].assign(v.begin(), v.begin() + 1);
                                events_offsets_[g] = {0, 1};
                                amp_offsets_[g] = {0, n_polar_ * n_amplitudes_};
                            }
                        }
                        std::vector<ctComplex*> d_chunk = calculateAmplitudes(Vp4_chunk);
                        cudaSetDevice(gpu);
                        double* d_ct;
                        cudaMalloc(&d_ct, nch * sizeof(double));
                        double* d_cp;
                        cudaMalloc(&d_cp, (size_t)nch * npartials * sizeof(double));
                        cudaMemset(d_cp, 0, (size_t)nch * npartials * sizeof(double));
                        {
                        const ctComplex* d_chunk_res = d_chunk[gpu];
                        ctComplex* d_chunk_dbl = nullptr;
                        if (float_amps_ && !ctIsFloatComplex) {
                            size_t nElem = (size_t)nch * n_polar_ * n_amplitudes_;
                            d_chunk_dbl = upcastAmpSegToDouble(d_chunk_res, nElem);
                            d_chunk_res = d_chunk_dbl;
                        }
                        computeResults(d_chunk_res,
                            reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[gpu].data_ptr()),
                            d_ct, d_total_integral_gpu, d_cp,
                            nullptr, nullptr,
                            d_wave_mask_gpu, d_selected_integral_gpu,
                            d_ct,
                            nullptr, 0,
                            d_nSLvectors, npartials, nch, n_amplitudes_, n_polar_);
                        if (d_chunk_dbl) cudaFree(d_chunk_dbl);
                        }
                        // 拷贝进全尺寸结果缓冲（chunk 结果 → 偏移 c0）
                        cudaMemcpy(d_final_result_vec[gpu] + c0, d_ct,
                                   nch * sizeof(double), cudaMemcpyDeviceToDevice);
                        for (int p = 0; p < npartials; ++p)
                            cudaMemcpy(d_partial_result_vec[gpu] + (size_t)p * ne + c0,
                                       d_cp + (size_t)p * nch, nch * sizeof(double),
                                       cudaMemcpyDeviceToDevice);
                        cudaFree(d_ct);
                        cudaFree(d_cp);
                        for (int g = 0; g < n_gpus_; ++g)
                            if (d_chunk[g]) { cudaSetDevice(g); cudaFree(d_chunk[g]); }
                        cudaSetDevice(gpu);
                    }
                    events_offsets_ = saved_ev;
                    amp_offsets_ = saved_amp;
                }
            }

            // 将单个GPU的结果累加到全局结果
            double h_total_integral_gpu;
            if (has_waves) {
                cudaMemcpy(&h_total_integral_gpu, d_selected_integral_gpu,
                    sizeof(double), cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpy(&h_total_integral_gpu, d_total_integral_gpu,
                    sizeof(double), cudaMemcpyDeviceToHost);
            }
            h_phsp_integral += h_total_integral_gpu;
            double* h_total_results_gpu = new double[events_[gpu][0]];
            cudaMemcpy(h_total_results_gpu, d_final_result_vec[gpu], events_[gpu][0] * sizeof(double), cudaMemcpyDeviceToHost);
            std::copy(h_total_results_gpu, h_total_results_gpu + events_[gpu][0], h_total_results + ev_cumulative);
            // 每 GPU 块是 [分量][事件]（p-major）→ 展开到全局 [分量][全事件] 布局
            for (int p = 0; p < npartials; ++p)
                cudaMemcpy(h_partial_results + (size_t)p * N_phsp + ev_cumulative,
                    d_partial_result_vec[gpu] + (size_t)p * events_[gpu][0],
                    events_[gpu][0] * sizeof(double), cudaMemcpyDeviceToHost);
            ev_cumulative += events_[gpu][0];
            cudaFree(d_total_integral_gpu);
            cudaFree(d_nSLvectors);
            if (d_wave_mask_gpu) cudaFree(d_wave_mask_gpu);
            if (d_selected_integral_gpu) cudaFree(d_selected_integral_gpu);
            delete[] h_total_results_gpu;

        }

        // bkg weights积分
        double h_bkg_integral = 0.0;
        // if (bkg_weights_ != nullptr && bkg_length > 0)
        if (!bkg_weights_.empty() && N_bkg > 0)
        {
            h_bkg_integral = bkg_integral_;
            // // thrust::device_ptr<double> d_ptr(bkg_weights_);
            // // std::cout << "Calculating background integral with " << events_[2] << " events..." << std::endl;
            // // h_bkg_integral = thrust::reduce(d_ptr, d_ptr + events_[2]);
            // // h_bkg_integral = thrust::reduce(d_ptr, d_ptr + bkg_length /
            // // n_polar_, 0.0, thrust::plus<double>());
            // for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            //     cudaSetDevice(gpu);
            //     if (events_[gpu][2] > 0) {
            //         // double* d_bkg_weights_gpu;
            //         // cudaMalloc(&d_bkg_weights_gpu, events_[gpu][2] * sizeof(double));
            //         // cudaMemcpy(d_bkg_weights_gpu, bkg_weights_[gpu], events_[gpu][2] * sizeof(double), cudaMemcpyDeviceToDevice);
            //         thrust::device_ptr<double> d_ptr_gpu(bkg_weights_[gpu]);
            //         h_bkg_integral += thrust::reduce(d_ptr_gpu, d_ptr_gpu + events_[gpu][2]);
            //         // cudaFree(d_bkg_weights_gpu);
            //     }
            // }
        }

        // int dataIntegral = data_length / n_polar_;
        // if (bkg_fix_ != nullptr && bkg_length > 0)
        // data 加权时用总权重（无权重时 = 事件数），与 NLL 的 N_eff 一致
        double dataIntegral = data_total_weight_;
        if (N_bkg > 0)
        {
            if (h_bkg_integral > 0)
            {
                dataIntegral -= h_bkg_integral;
            }
            else
            {
                dataIntegral -= N_bkg;
            }
        }
        // 空 phsp（N_phsp=0）时 h_phsp_integral=0；inf 会把所有 plot 直方图
        // scale 成 inf。没有 phsp 就没有可画的拟合分布，保持 1.0 即可。
        double normFactor = (h_phsp_integral > 0.0)
            ? static_cast<double>(dataIntegral) / h_phsp_integral : 1.0;

        TFile* rootFile = new TFile(filename.c_str(), "RECREATE");
        // 中途抛异常（Hessian 形状/维度、积分扫描失败等）时也必须关闭 ROOT
        // 文件，否则 TFile 缓冲区不落盘，后续再写同名文件会拿到半成品。
        auto root_guard = makeScopeExit([&]() {
            if (rootFile) { rootFile->Close(); delete rootFile; rootFile = nullptr; }
        });

        TTree* legend = new TTree("legends", "Amplitude Legends");
        legend->Branch("legend", &legends_);
        legend->Fill();
        legend->Write();
        delete legend;


        if (is_saved_weight == 1)
        {
            TTree* phspTree = new TTree("saved_weight", "fitting result weights");

            // 添加权重分支
            double total_weight;
            phspTree->Branch("totalweight", &total_weight);

            // 为每个部分波创建分支
            std::vector<double> partial_weights(npartials);
            for (int i = 0; i < npartials; ++i)
            {
                std::string branch_name = "weight_" + resonance_names_[i];
                phspTree->Branch(branch_name.c_str(), &partial_weights[i]);
            }

            // 末态粒子四动量分支（任意分布按需现算: M/θ/φ/cosβ/... 全部可由它导出）
            std::vector<std::string> mom_names;
            if (!Vp4_all_.empty())
                for (const auto& kv : Vp4_all_[0])
                    mom_names.push_back(kv.first);
            const int n_mom = (int)mom_names.size();
            std::vector<LorentzVector> mom_host((size_t)N_phsp * (size_t)std::max(1, n_mom));
            {
                int off = 0;
                for (size_t gpu = 0; gpu < Vp4_all_.size(); ++gpu) {
                    int ne = events_[gpu][0];
                    for (int m = 0; m < n_mom; ++m) {
                        const auto& vec = Vp4_all_[gpu].at(mom_names[m]);
                        for (int e = 0; e < ne; ++e)
                            mom_host[(size_t)(off + e) * n_mom + m] = vec[e];
                    }
                    off += ne;
                }
            }
            std::vector<double> mom_vals((size_t)std::max(1, n_mom) * 4, 0.0);
            for (int m = 0; m < n_mom; ++m)
            {
                const std::string& nm = mom_names[m];
                phspTree->Branch((nm + "_px").c_str(), &mom_vals[m * 4 + 0]);
                phspTree->Branch((nm + "_py").c_str(), &mom_vals[m * 4 + 1]);
                phspTree->Branch((nm + "_pz").c_str(), &mom_vals[m * 4 + 2]);
                phspTree->Branch((nm + "_E").c_str(),  &mom_vals[m * 4 + 3]);
            }

            // 填充 phsp tree
            // ⚠️ kernel 缓冲是 [分量][事件]（p-major: p*N+ev）, 读取必须同布局——
            // 旧代码按事件主序读 = 转置错位（weight_<i> 分支一直是错配的）
            for (int i = 0; i < N_phsp; ++i)
            {
                // 设置权重
                total_weight = h_total_results[i] / h_phsp_integral * static_cast<double>(dataIntegral);
                // std::cout << "Event " << i << ": Total Weight = " <<
                // total_weight << std::endl;
                for (int j = 0; j < npartials; ++j)
                {
                    partial_weights[j] = h_partial_results[(size_t)j * N_phsp + i] * normFactor;
                    // std::cout << "  Partial Weight " << j << " = " <<
                    // partial_weights[j] << std::endl;
                }
                for (int m = 0; m < n_mom; ++m)
                {
                    const LorentzVector& mv = mom_host[(size_t)i * n_mom + m];
                    mom_vals[m * 4 + 0] = mv.Px;
                    mom_vals[m * 4 + 1] = mv.Py;
                    mom_vals[m * 4 + 2] = mv.Pz;
                    mom_vals[m * 4 + 3] = mv.E;
                }

                phspTree->Fill();
            }

            phspTree->Write();
            delete phspTree;
        }

        auto plotconfig = config_parser_.getPlotConfigs();
        std::vector<MassHistConfig> masshist;
        std::vector<AngleHistConfig> anglehist;
        std::vector<DalitzHistConfig> dalitzhist;
        int mass_hist_count = 0;
        int angle_hist_count = 0;
        int dalitz_hist_count = 0;
        for (const auto& histConfig : plotconfig)
        {
            // 输出histConfig内容以进行调试
            // std::cout << "PlotConfig type: " << histConfig.type << std::endl;
            if (histConfig.type == "mass")
            {
                std::vector<std::string> particles = histConfig.particles[0];
                int bins = histConfig.bins[0];
                std::vector<double> range = histConfig.ranges[0];
                std::vector<std::string> display = histConfig.display;

                std::string hist_name =
                    "mass" + std::to_string(mass_hist_count++);
                for (const auto& p : particles)
                {
                    hist_name += "_" + p;
                }
                std::cout << "Creating mass histogram: " << hist_name << std::endl;
                masshist.emplace_back(hist_name, "", particles, bins, range, display);
            }
            else if (histConfig.type == "cosbeta")
            {
                // 处理角度直方图配置
                std::vector<std::vector<std::string>> particles = histConfig.particles;
                int bins = histConfig.bins[0];
                std::vector<double> range = histConfig.ranges[0];
                std::vector<std::string> display = histConfig.display;
                std::string hist_name = "cosbeta" + std::to_string(angle_hist_count++);
                for (const auto& pvec : particles)
                {
                    hist_name += "_";
                    for (const auto& p : pvec)
                    {
                        hist_name += p;
                    }
                }
                std::cout << "Creating angle histogram: " << hist_name << std::endl;
                anglehist.emplace_back(hist_name, "", particles, bins, range, display);
            }
            else if (histConfig.type == "dalitz")
            {
                std::vector<std::vector<std::string>> particles = histConfig.particles;
                std::vector<int> bins = histConfig.bins;
                std::vector<std::vector<double>> ranges = histConfig.ranges;
                std::vector<std::string> display = histConfig.display;
                std::string hist_name = "dalitz" + std::to_string(dalitz_hist_count++);
                for (const auto& pvec : particles)
                {
                    hist_name += "_";
                    for (const auto& p : pvec)
                    {
                        hist_name += p;
                    }
                }
                std::cout << "Creating dalitz histogram: " << hist_name << std::endl;
                dalitzhist.emplace_back(hist_name, "", particles, bins, ranges, display);
            }
        }

        for (const auto& histConfig : masshist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel", TObject::kOverwrite);
            ylabel_obj.Write("ylabel", TObject::kOverwrite);
        }

        for (const auto& histConfig : anglehist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel", TObject::kOverwrite);
            ylabel_obj.Write("ylabel", TObject::kOverwrite);
        }

        for (const auto& histConfig : dalitzhist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel", TObject::kOverwrite);
            ylabel_obj.Write("ylabel", TObject::kOverwrite);
        }

        // 四动量index
        std::map<std::string, int> particleToIndex;
        for (int i = 0; i < particles_.size(); ++i)
        {
            particleToIndex[particles_[i].name] = i;
        }

        std::vector<LorentzVector*> device_momenta_list = convertToLorentzVectors(Vp4_all_, particleToIndex);
        int n_particles = particleToIndex.size();

        // 计算并保存data直方图
        // data_weights_ 为空/该 GPU 无权重时返回 nullptr，填充 kernel 按权重 1 处理。
        auto data_weights_for_gpu = [this](size_t gpu) -> double* {
            return gpu < data_weights_.size() ? data_weights_[gpu] : nullptr;
        };
        if (N_data > 0)
        {
            std::vector<TH1F*> masshist_data;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_data").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                masshist_data.emplace_back(hist);
            }

            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_hists[i];
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateMassHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, masshist, data_weights_for_gpu(i), temp, events_[i][1], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < masshist.size(); ++j)
                {
                    masshist_data[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();
                masshist_data[i]->Write("hdata", TObject::kOverwrite);
                delete masshist_data[i];
            }

            std::vector<TH1F*> anglehist_data;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_data").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                anglehist_data.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_angle_hists[i];
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateAngleHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, anglehist, data_weights_for_gpu(i), temp, events_[i][1], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < anglehist.size(); ++j)
                {
                    anglehist_data[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                anglehist_data[i]->Write("hdata", TObject::kOverwrite);
                delete anglehist_data[i];
            }

            std::vector<TH2F*> dalitzhist_data;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F((histConfig.name + "_data").c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
                    histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_data.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH2F*>> temp_dalitz_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH2F*>& temp = temp_dalitz_hists[i];
                for (const auto& histConfig : dalitzhist)
                {
                    TH2F* hist = new TH2F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(),
                        histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
                        histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateDalitzHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, dalitzhist, data_weights_for_gpu(i), temp, events_[i][1], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < dalitzhist.size(); ++j)
                {
                    dalitzhist_data[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                dalitzhist_data[i]->Write("hdata", TObject::kOverwrite);
                delete dalitzhist_data[i];
            }

            // if (device_momenta != nullptr)
            // cudaFree(device_momenta); // device_momenta not allocated
        }

        // 计算并保存拟合结果直方图
        if (N_phsp > 0)
        {
            std::vector<TH1F*> masshist_fit;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_fit").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                masshist_fit.emplace_back(hist);
            }

            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_mass_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_mass_hists[i];
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图（phsp偏移量为0）
                CalculateMassHist(device_momenta_list[i], particleToIndex, masshist,
                    d_final_result_vec[i], temp, events_[i][0], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < masshist.size(); ++j)
                {
                    masshist_fit[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();

                TH1F* hist = masshist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit", TObject::kOverwrite);
                delete masshist_fit[i];
            }

            std::vector<TH1F*> anglehist_fit;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_fit").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                anglehist_fit.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_angle_hists[i];
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateAngleHist(device_momenta_list[i], particleToIndex, anglehist,
                    d_final_result_vec[i], temp, events_[i][0], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < anglehist.size(); ++j)
                {
                    anglehist_fit[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                TH1F* hist = anglehist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit", TObject::kOverwrite);
                delete anglehist_fit[i];
            }

            std::vector<TH2F*> dalitzhist_fit;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F((histConfig.name + "_fit").c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0],
                    histConfig.range[0][1], histConfig.bins[1],
                    histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_fit.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH2F*>> temp_dalitz_hists(device_momenta_list.size());
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH2F*>& temp = temp_dalitz_hists[i];
                for (const auto& histConfig : dalitzhist)
                {
                    TH2F* hist = new TH2F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(),
                        histConfig.bins[0], histConfig.range[0][0],
                        histConfig.range[0][1], histConfig.bins[1],
                        histConfig.range[1][0], histConfig.range[1][1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateDalitzHist(device_momenta_list[i], particleToIndex, dalitzhist,
                    d_final_result_vec[i], temp, events_[i][0], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < dalitzhist.size(); ++j)
                {
                    dalitzhist_fit[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;
            }
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                TH2F* hist = dalitzhist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit", TObject::kOverwrite);
                delete dalitzhist_fit[i];
            }
        }

        // if (N_bkg > 0 && !bkg_weights_.empty() && bkg_weights_[0] != nullptr)
        if (N_bkg > 0)
        {
            std::vector<TH1F*> masshist_bkg;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_bkg").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                masshist_bkg.emplace_back(hist);
            }

            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_mass_hists(device_momenta_list.size());
            // 计算权重偏移量
            int bkg_weight_offset = 0;
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                if (events_[i][2] == 0) continue;  // 该GPU无bkg事件

                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_mass_hists[i];
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                CalculateMassHist(device_momenta_list[i] + events_offsets_[i][2] * n_particles,
                    particleToIndex, masshist,
                    bkg_weights_[i], temp, events_[i][2], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < masshist.size(); ++j)
                {
                    masshist_bkg[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;

                // 更新权重偏移量
                bkg_weight_offset += events_[i][2];
            }
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();
                masshist_bkg[i]->Write("hbkg", TObject::kOverwrite);
                delete masshist_bkg[i];
            }

            std::vector<TH1F*> anglehist_bkg;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F((histConfig.name + "_bkg").c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                anglehist_bkg.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
            bkg_weight_offset = 0;
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                if (events_[i][2] == 0) continue;

                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH1F*>& temp = temp_angle_hists[i];
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                CalculateAngleHist(device_momenta_list[i] + events_offsets_[i][2] * n_particles,
                    particleToIndex, anglehist,
                    bkg_weights_[i], temp, events_[i][2], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < anglehist.size(); ++j)
                {
                    anglehist_bkg[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;

                bkg_weight_offset += events_[i][2];
            }
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                anglehist_bkg[i]->Write("hbkg", TObject::kOverwrite);
                delete anglehist_bkg[i];
            }

            std::vector<TH2F*> dalitzhist_bkg;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F((histConfig.name + "_bkg").c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0],
                    histConfig.range[0][1], histConfig.bins[1],
                    histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_bkg.emplace_back(hist);
            }
            // 临时直方图向量，用于每个GPU
            std::vector<std::vector<TH2F*>> temp_dalitz_hists(device_momenta_list.size());
            bkg_weight_offset = 0;
            for (size_t i = 0; i < device_momenta_list.size(); ++i)
            {
                if (events_[i][2] == 0) continue;

                cudaSetDevice(i);
                // 为当前GPU创建临时直方图
                std::vector<TH2F*>& temp = temp_dalitz_hists[i];
                for (const auto& histConfig : dalitzhist)
                {
                    TH2F* hist = new TH2F((histConfig.name + "_temp").c_str(), histConfig.title.c_str(),
                        histConfig.bins[0], histConfig.range[0][0],
                        histConfig.range[0][1], histConfig.bins[1],
                        histConfig.range[1][0], histConfig.range[1][1]);
                    temp.emplace_back(hist);
                }
                // 计算当前GPU的直方图
                // double* bkg_weight_ptr = nullptr;//bkg_weights_[i] + bkg_weight_offset;
                CalculateDalitzHist(device_momenta_list[i] + events_offsets_[i][2] * n_particles,
                    particleToIndex, dalitzhist,
                    bkg_weights_[i], temp, events_[i][2], n_particles);
                // 累加到总直方图
                for (size_t j = 0; j < dalitzhist.size(); ++j)
                {
                    dalitzhist_bkg[j]->Add(temp[j]);
                }
                // 清理临时直方图
                for (auto* hist : temp) delete hist;

                bkg_weight_offset += events_[i][2];
            }
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                dalitzhist_bkg[i]->Write("hbkg", TObject::kOverwrite);
                delete dalitzhist_bkg[i];
            }
        }

        // 计算并保存部分波直方图（多GPU版本）
        if (N_phsp > 0)
        {
            int npartials = nSLvectors_.size();
            for (int partial_idx = 0; partial_idx < npartials; ++partial_idx)
            {
                // 为当前部分波创建质量直方图
                std::vector<TH1F*> masshist_partial;
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_partial").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    masshist_partial.push_back(hist);
                }

                // 临时直方图向量，用于每个GPU
                std::vector<std::vector<TH1F*>> temp_mass_hists(device_momenta_list.size());
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu)
                {
                    if (events_[gpu][0] == 0) continue;  // 该GPU无phsp事件

                    cudaSetDevice(gpu);
                    // 为当前GPU创建临时直方图
                    std::vector<TH1F*>& temp = temp_mass_hists[gpu];
                    for (const auto& histConfig : masshist)
                    {
                        TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                            histConfig.title.c_str(), histConfig.bins,
                            histConfig.range[0], histConfig.range[1]);
                        temp.emplace_back(hist);
                    }
                    // 计算当前GPU的部分波直方图
                    // 权重指针：d_partial_result_vec[gpu][partial_idx * events_[gpu][0]]
                    double* partial_weight_ptr = d_partial_result_vec[gpu] + partial_idx * events_[gpu][0];
                    CalculateMassHist(device_momenta_list[gpu], particleToIndex, masshist,
                        partial_weight_ptr, temp, events_[gpu][0], n_particles);
                    // 累加到总直方图
                    for (size_t j = 0; j < masshist.size(); ++j)
                    {
                        masshist_partial[j]->Add(temp[j]);
                    }
                    // 清理临时直方图
                    for (auto* hist : temp) delete hist;
                }

                for (size_t j = 0; j < masshist_partial.size(); ++j)
                {
                    TDirectory* histDir = rootFile->GetDirectory(masshist[j].name.c_str());
                    histDir->cd();

                    std::string partial_dir_name = "h_" + sanitizeROOTName (resonance_names_[partial_idx]);

                    TH1F* hist = masshist_partial[j];
                    hist->Scale(normFactor);
                    hist->Write(partial_dir_name.c_str(), TObject::kOverwrite);
                    delete hist;
                }

                // 角度直方图
                std::vector<TH1F*> anglehist_partial;
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F((histConfig.name + "_partial").c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    anglehist_partial.push_back(hist);
                }

                // 临时直方图向量，用于每个GPU
                std::vector<std::vector<TH1F*>> temp_angle_hists(device_momenta_list.size());
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu)
                {
                    if (events_[gpu][0] == 0) continue;

                    cudaSetDevice(gpu);
                    std::vector<TH1F*>& temp = temp_angle_hists[gpu];
                    for (const auto& histConfig : anglehist)
                    {
                        TH1F* hist = new TH1F((histConfig.name + "_temp").c_str(),
                            histConfig.title.c_str(), histConfig.bins,
                            histConfig.range[0], histConfig.range[1]);
                        temp.emplace_back(hist);
                    }
                    double* partial_weight_ptr = d_partial_result_vec[gpu] + partial_idx * events_[gpu][0];
                    CalculateAngleHist(device_momenta_list[gpu], particleToIndex, anglehist,
                        partial_weight_ptr, temp, events_[gpu][0], n_particles);
                    for (size_t j = 0; j < anglehist.size(); ++j)
                    {
                        anglehist_partial[j]->Add(temp[j]);
                    }
                    for (auto* hist : temp) delete hist;
                }

                for (size_t j = 0; j < anglehist_partial.size(); ++j)
                {
                    TDirectory* histDir = rootFile->GetDirectory(anglehist[j].name.c_str());
                    histDir->cd();

                    std::string partial_dir_name = "h_" + sanitizeROOTName(resonance_names_[partial_idx]);

                    TH1F* hist = anglehist_partial[j];
                    hist->Scale(normFactor);
                    hist->Write(partial_dir_name.c_str(), TObject::kOverwrite);
                    delete anglehist_partial[j];
                }
            }
        }

        // =====================================================================
        // 统一观测直方图 (Plot 序列 form, type=="obs"): 1d/2d 自动区分
        //   数据 hdata: data_weights_（无权重时按 1）;  模型 hfit: phsp 权重
        //   d_final_result_vec (全模型或 waves 子集) × normFactor
        // =====================================================================
        {
            std::vector<PlotConfig> obshists;
            for (const auto& pc : plotconfig)
                if (pc.type == "obs") obshists.push_back(pc);
            if (!obshists.empty()) {
                // 防御: 过滤 bins/ranges 维度不足的项(正常解析不会发生; 兜底手写/旧 .so)
                std::vector<PlotConfig> valid;
                for (const auto& cfg : obshists) {
                    bool ok = (int)cfg.bins.size() >= (int)cfg.obs.size()
                           && (int)cfg.ranges.size() >= (int)cfg.obs.size();
                    for (int d = 0; ok && d < (int)cfg.obs.size(); ++d)
                        if (cfg.ranges[d].size() < 2) ok = false;
                    if (ok) valid.push_back(cfg);
                    else std::cerr << "Warning: 跳过格式不完整的 obs 项 '"
                                   << cfg.name << "' (bins/ranges 维度不足)" << std::endl;
                }
                obshists.swap(valid);
                // 顶层母粒子 index（Angle/CosAngle 的轴; 取第一个链的第一步母粒子）
                int motherIdx = -1;
                {
                    const auto& chains = config_parser_.getDecayChains();
                    if (!chains.empty() && !chains[0].decay_steps.empty()) {
                        auto it = particleToIndex.find(chains[0].decay_steps[0].mother);
                        if (it != particleToIndex.end()) motherIdx = it->second;
                    }
                }
                for (const auto& cfg : obshists) {
                    TDirectory* dir = rootFile->mkdir(cfg.name.c_str());
                    dir->cd();
                    if (cfg.display.size() >= 2) {
                        TObjString xl(cfg.display[0].c_str());
                        TObjString yl(cfg.display[1].c_str());
                        xl.Write("xlabel", TObject::kOverwrite);
                        yl.Write("ylabel", TObject::kOverwrite);
                    }
                }

                // ---- 数据直方图 ----
                std::vector<TH1F*> obs1d_data(obshists.size(), nullptr);
                std::vector<TH2F*> obs2d_data(obshists.size(), nullptr);
                for (size_t j = 0; j < obshists.size(); ++j) {
                    const auto& cfg = obshists[j];
                    if (cfg.obs.size() == 1)
                        obs1d_data[j] = new TH1F((cfg.name + "_data").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                    else
                        obs2d_data[j] = new TH2F((cfg.name + "_data").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                            cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                }
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                    if (events_[gpu][1] == 0) continue;
                    cudaSetDevice(gpu);
                    std::vector<TH1F*> t1(obshists.size(), nullptr);
                    std::vector<TH2F*> t2(obshists.size(), nullptr);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        const auto& cfg = obshists[j];
                        if (cfg.obs.size() == 1)
                            t1[j] = new TH1F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                        else
                            t2[j] = new TH2F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                    }
                    CalculateObsHist(
                        device_momenta_list[gpu] + events_offsets_[gpu][1] * n_particles,
                        particleToIndex, obshists, data_weights_for_gpu(gpu), t1, t2,
                        events_[gpu][1], n_particles, motherIdx);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        if (obs1d_data[j]) obs1d_data[j]->Add(t1[j]);
                        if (obs2d_data[j]) obs2d_data[j]->Add(t2[j]);
                    }
                    for (auto* h : t1) if (h) delete h;
                    for (auto* h : t2) if (h) delete h;
                }
                for (size_t j = 0; j < obshists.size(); ++j) {
                    TDirectory* dir = rootFile->GetDirectory(obshists[j].name.c_str());
                    dir->cd();
                    if (obs1d_data[j]) {
                        obs1d_data[j]->Write("hdata", TObject::kOverwrite);
                        delete obs1d_data[j];
                    }
                    if (obs2d_data[j]) {
                        obs2d_data[j]->Write("hdata", TObject::kOverwrite);
                        delete obs2d_data[j];
                    }
                }

                // ---- 本底直方图 hbkg (与旧格式一致: bkg 事件 × bkg_weights_, 不 Scale) ----
                if (N_bkg > 0) {
                    std::vector<TH1F*> obs1d_bkg(obshists.size(), nullptr);
                    std::vector<TH2F*> obs2d_bkg(obshists.size(), nullptr);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        const auto& cfg = obshists[j];
                        if (cfg.obs.size() == 1)
                            obs1d_bkg[j] = new TH1F((cfg.name + "_bkg").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                        else
                            obs2d_bkg[j] = new TH2F((cfg.name + "_bkg").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                    }
                    for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                        if (events_[gpu][2] == 0) continue;
                        cudaSetDevice(gpu);
                        std::vector<TH1F*> t1(obshists.size(), nullptr);
                        std::vector<TH2F*> t2(obshists.size(), nullptr);
                        for (size_t j = 0; j < obshists.size(); ++j) {
                            const auto& cfg = obshists[j];
                            if (cfg.obs.size() == 1)
                                t1[j] = new TH1F((cfg.name + "_t").c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                            else
                                t2[j] = new TH2F((cfg.name + "_t").c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                    cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                        }
                        CalculateObsHist(
                            device_momenta_list[gpu] + events_offsets_[gpu][2] * n_particles,
                            particleToIndex, obshists, bkg_weights_[gpu], t1, t2,
                            events_[gpu][2], n_particles, motherIdx);
                        for (size_t j = 0; j < obshists.size(); ++j) {
                            if (obs1d_bkg[j]) obs1d_bkg[j]->Add(t1[j]);
                            if (obs2d_bkg[j]) obs2d_bkg[j]->Add(t2[j]);
                        }
                        for (auto* h : t1) if (h) delete h;
                        for (auto* h : t2) if (h) delete h;
                    }
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        TDirectory* dir = rootFile->GetDirectory(obshists[j].name.c_str());
                        dir->cd();
                        if (obs1d_bkg[j]) {
                            obs1d_bkg[j]->Write("hbkg", TObject::kOverwrite);
                            delete obs1d_bkg[j];
                        }
                        if (obs2d_bkg[j]) {
                            obs2d_bkg[j]->Write("hbkg", TObject::kOverwrite);
                            delete obs2d_bkg[j];
                        }
                    }
                }

                // ---- 模型直方图 ----
                std::vector<TH1F*> obs1d_fit(obshists.size(), nullptr);
                std::vector<TH2F*> obs2d_fit(obshists.size(), nullptr);
                for (size_t j = 0; j < obshists.size(); ++j) {
                    const auto& cfg = obshists[j];
                    if (cfg.obs.size() == 1)
                        obs1d_fit[j] = new TH1F((cfg.name + "_fit").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                    else
                        obs2d_fit[j] = new TH2F((cfg.name + "_fit").c_str(), "",
                            cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                            cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                }
                for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                    if (events_[gpu][0] == 0) continue;
                    cudaSetDevice(gpu);
                    std::vector<TH1F*> t1(obshists.size(), nullptr);
                    std::vector<TH2F*> t2(obshists.size(), nullptr);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        const auto& cfg = obshists[j];
                        if (cfg.obs.size() == 1)
                            t1[j] = new TH1F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                        else
                            t2[j] = new TH2F((cfg.name + "_t").c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                    }
                    CalculateObsHist(device_momenta_list[gpu], particleToIndex,
                        obshists, d_final_result_vec[gpu], t1, t2,
                        events_[gpu][0], n_particles, motherIdx);
                    for (size_t j = 0; j < obshists.size(); ++j) {
                        if (obs1d_fit[j]) obs1d_fit[j]->Add(t1[j]);
                        if (obs2d_fit[j]) obs2d_fit[j]->Add(t2[j]);
                    }
                    for (auto* h : t1) if (h) delete h;
                    for (auto* h : t2) if (h) delete h;
                }
                for (size_t j = 0; j < obshists.size(); ++j) {
                    TDirectory* dir = rootFile->GetDirectory(obshists[j].name.c_str());
                    dir->cd();
                    if (obs1d_fit[j]) {
                        obs1d_fit[j]->Scale(normFactor);
                        obs1d_fit[j]->Write("hfit", TObject::kOverwrite);
                        delete obs1d_fit[j];
                    }
                    if (obs2d_fit[j]) {
                        obs2d_fit[j]->Scale(normFactor);
                        obs2d_fit[j]->Write("hfit", TObject::kOverwrite);
                        delete obs2d_fit[j];
                    }
                }

                // ---- 分波形状: 每个 obs 图 × 每波 |A_i|² 权重 → h_<波名> 子目录 ----
                // (与旧格式的 mass/angle 分波形状一致; 1d 用 TH1F, 2d 用 TH2F)
                for (size_t j = 0; j < obshists.size(); ++j) {
                    const auto& cfg = obshists[j];
                    const bool is2 = (cfg.obs.size() == 2);
                    std::vector<TH1F*> obs1d_part(npartials, nullptr);
                    std::vector<TH2F*> obs2d_part(npartials, nullptr);
                    for (int p = 0; p < npartials; ++p) {
                        std::string pn = cfg.name + "_p" + std::to_string(p);
                        if (is2)
                            obs2d_part[p] = new TH2F(pn.c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                        else
                            obs1d_part[p] = new TH1F(pn.c_str(), "",
                                cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                    }
                    for (size_t gpu = 0; gpu < device_momenta_list.size(); ++gpu) {
                        if (events_[gpu][0] == 0) continue;
                        cudaSetDevice(gpu);
                        for (int p = 0; p < npartials; ++p) {
                            std::vector<PlotConfig> one = {cfg};
                            std::vector<TH1F*> t1(is2 ? 0 : 1, nullptr);
                            std::vector<TH2F*> t2(is2 ? 1 : 0, nullptr);
                            std::string tn = cfg.name + "_t" + std::to_string(p);
                            if (is2)
                                t2[0] = new TH2F(tn.c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1],
                                    cfg.bins[1], cfg.ranges[1][0], cfg.ranges[1][1]);
                            else
                                t1[0] = new TH1F(tn.c_str(), "",
                                    cfg.bins[0], cfg.ranges[0][0], cfg.ranges[0][1]);
                            double* partial_weight_ptr =
                                d_partial_result_vec[gpu] + (size_t)p * events_[gpu][0];
                            CalculateObsHist(device_momenta_list[gpu], particleToIndex,
                                one, partial_weight_ptr, t1, t2,
                                events_[gpu][0], n_particles, motherIdx);
                            if (is2) obs2d_part[p]->Add(t2[0]);
                            else obs1d_part[p]->Add(t1[0]);
                            if (!t1.empty() && t1[0]) delete t1[0];
                            if (!t2.empty() && t2[0]) delete t2[0];
                        }
                    }
                    for (int p = 0; p < npartials; ++p) {
                        std::string partial_dir_name =
                            "h_" + sanitizeROOTName(resonance_names_[p]);
                        TDirectory* dir = rootFile->GetDirectory(cfg.name.c_str());
                        dir->cd();
                        if (is2) {
                            if (!obs2d_part[p]) continue;
                            obs2d_part[p]->Scale(normFactor);
                            obs2d_part[p]->Write(partial_dir_name.c_str(),
                                TObject::kOverwrite);
                            delete obs2d_part[p];
                        } else {
                            if (!obs1d_part[p]) continue;
                            obs1d_part[p]->Scale(normFactor);
                            obs1d_part[p]->Write(partial_dir_name.c_str(),
                                TObject::kOverwrite);
                            delete obs1d_part[p];
                        }
                    }
                }
            }
        }

        // 直方图都写完之后，先释放逐事例设备缓冲，再跑矩阵/误差的全样本扫描。
        // 大统计量下这些缓冲非常占显存（每卡）：
        //   d_partial_result_vec = N_phsp × npartials × 8 B   （15M×20 → 2.4 GB）
        //   d_final_result_vec   = N_phsp × 8 B
        //   device_momenta_list  = N_phsp × n_final × sizeof(LorentzVector)
        // 不先放掉，后面每张矩阵/每个误差方向都要再分配批 workspace → 多卡 OOM。
        for (size_t i = 0; i < d_final_result_vec.size(); ++i)
            if (d_final_result_vec[i]) {
                cudaSetDevice((int)i);
                cudaFree(d_final_result_vec[i]);
                d_final_result_vec[i] = nullptr;
            }
        for (size_t i = 0; i < d_partial_result_vec.size(); ++i)
            if (d_partial_result_vec[i]) {
                cudaSetDevice((int)i);
                cudaFree(d_partial_result_vec[i]);
                d_partial_result_vec[i] = nullptr;
            }
        for (size_t i = 0; i < device_momenta_list.size(); ++i)
            if (device_momenta_list[i]) {
                cudaSetDevice((int)i);
                cudaFree(device_momenta_list[i]);
                device_momenta_list[i] = nullptr;
            }
        cudaSetDevice(target_dev);


        // =====================================================================
        // 两张干涉矩阵（各自除以本样本的相干总额 W，因此元素是 O(1) 的份额）:
        //   interference_rec : P_ij / W_rec   （重建侧，带效率 MC）
        //   interference_gen : T_ij / W_gen   （产生侧 truth / 无效率 MC）
        // 定义（权重口径与拟合 NLL 一致，phsp_weights / phsp_truth_weights 生效）:
        //   M_ii = Σ_e w·|A_i|² ,  M_ij = Σ_e w·2Re(A_iA_j*)  (i≠j, 对称)
        //   W    = Σ_i M_ii + Σ_{i<j} M_ij  = Σ_e w·|Σ_i A_i|²
        // 于是（与 doc/branching_fraction_recipe.md 一致）:
        //   文献拟合分数 F_i = interference_gen[i][i]，干涉项 I_ij = interference_gen[i][j]
        //   非相干份额   FF_i = diag / Σ diag           （本库 getFitFractions 的口径）
        //   ε̄ = W_rec / W_gen（两样本同源产生时）;  ε_i = (P_ii/N_rec)/(T_ii/N_gen)
        //   —— 需要"原始积分"时把矩阵乘回对应的 W（integrals 树里有 W_rec/W_gen）。
        // integrals 树给出 W_rec, W_gen, N_rec, N_gen（Σw）, npartials。
        // 矩阵行列顺序 = 分波顺序（与 getFitFractions / writeResult 的 h_<n> 一致）。
        // CTPWA_WR_NO_INTEGRALS=1 可跳过这两趟全样本扫描（不写矩阵与 integrals）。
        // =====================================================================
        const auto& wr_data_files = config_parser_.getDataFiles();
        const bool want_raw = (std::getenv("CTPWA_WR_NO_INTEGRALS") == nullptr);
        double N_rec_w = 0.0, N_gen_w = 0.0, W_rec_raw = 0.0, W_gen_raw = 0.0;
        std::vector<double> P_diag, P_off, T_diag, T_off;
        auto gramTotal = [npartials](const std::vector<double>& diag,
                                     const std::vector<double>& off) {
            double w = 0.0;
            for (int i = 0; i < npartials; ++i) w += diag[i];
            for (int i = 0; i < npartials; ++i)
                for (int j = i + 1; j < npartials; ++j) w += off[i * npartials + j];
            return w;
        };
        auto runIntegrals = [&](const std::string& key, std::vector<double>& diag,
                                std::vector<double>& off) -> double {
            diag.assign(npartials, 0.0);
            off.assign((size_t)npartials * npartials, 0.0);
            auto t0 = std::chrono::high_resolution_clock::now();
            double wsum = computeFileBatchIntegrals(key, extended_vector,
                std::vector<torch::Tensor>{}, std::vector<torch::Tensor>{},
                diag.data(), off.data(), nullptr,
                nullptr, nullptr, nullptr, nullptr, npartials);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (std::getenv("CTPWA_VERBOSE"))
                std::printf("%s 积分矩阵: %.1f ms\n", key.c_str(),
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            return wsum;
        };
        if (want_raw && wr_data_files.count("phsp")) {
            {
                // 重建侧：常驻 B̄ 收缩（不读文件）；Mg 只有 n×n，取完 diag/off 立即析构
                double Wg = 0.0;
                torch::Tensor Mg = phspMatrixFromGram(extended_vector, &Wg, &N_rec_w);
                P_diag.assign(npartials, 0.0);
                P_off.assign((size_t)npartials * npartials, 0.0);
                for (int i = 0; i < npartials; ++i) {
                    P_diag[i] = Mg[i][i].item<double>();
                    for (int j = i + 1; j < npartials; ++j)
                        P_off[(size_t)i * npartials + j] = Mg[i][j].item<double>();
                }
            }

            W_rec_raw = gramTotal(P_diag, P_off);
        }
        bool have_truth = false;
        if (want_raw && wr_data_files.count("phsp_truth")) {
            N_gen_w = runIntegrals("phsp_truth", T_diag, T_off);
            W_gen_raw = gramTotal(T_diag, T_off);
            have_truth = true;
        }

        // 创建 ROOT 文件

        // 对称矩阵写出（对角用 partial，非对角用 2Re；两三角都填）
        auto writeSymmetric = [&](const char* name, const std::vector<double>& diag,
                                  const std::vector<double>& off, double scale) {
            if ((int)diag.size() != npartials) return;
            rootFile->cd();
            TMatrixD m(npartials, npartials);
            for (int i = 0; i < npartials; ++i) {
                m[i][i] = diag[i] / scale;
                for (int j = i + 1; j < npartials; ++j) {
                    double v = off[(size_t)i * npartials + j] / scale;
                    m[i][j] = v;
                    m[j][i] = v;
                }
            }
            m.Write(name);
        };

        if (!P_diag.empty() && W_rec_raw > 0.0)
            writeSymmetric("interference_rec", P_diag, P_off, W_rec_raw);
        if (have_truth && W_gen_raw > 0.0)
            writeSymmetric("interference_gen", T_diag, T_off, W_gen_raw);

        // ---- 参数误差矩阵（线性化传播；MC 统计误差不在这里，属系统项）----
        // 需要统一参数 [Re p, Im p, θ]（legacy complex 模式没有 θ 布局，跳过）。
        if (!P_diag.empty() && params_.hasCouplingMatrix()
            && params.dtype() == torch::kFloat64
            && std::getenv("CTPWA_WR_NO_ERR") == nullptr)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto errs = computeMatrixErrors(params, hessian_in);
            bool wrote_err = false;
            if (errs.size() == 2) {
                auto writeTensor = [&](const char* name, const torch::Tensor& E) {
                    rootFile->cd();
                    TMatrixD m(npartials, npartials);
                    for (int i = 0; i < npartials; ++i)
                        for (int j = 0; j < npartials; ++j)
                            m[i][j] = E[i][j].item<double>();
                    m.Write(name);
                };
                // rec/gen 独立写：空 phsp 时 rec_err 为空但 gen_err 仍有效。
                if (errs[0].numel() > 0) {
                    writeTensor("interference_rec_err", errs[0]);
                    wrote_err = true;
                }
                if (errs[1].numel() > 0) {
                    writeTensor("interference_gen_err", errs[1]);
                    wrote_err = true;
                }
            }
            if (wrote_err) {
                auto t1 = std::chrono::high_resolution_clock::now();
                if (std::getenv("CTPWA_VERBOSE"))
                    std::printf("参数误差矩阵: %.1f ms\n",
                        std::chrono::duration<double, std::milli>(t1 - t0).count());
            } else {
                std::cerr << "参数误差矩阵未写出（Hessian 非正定/缺失，或对应样本为空）"
                          << std::endl;
            }
        }

        // 归一化/换算要用的样本量（Σw）与相干总额（乘回去就是原始积分）
        if (!P_diag.empty() || have_truth) {
            rootFile->cd();
            TTree* t = new TTree("integrals",
                "sum of weights and coherent totals of the interference matrices");
            t->Branch("N_rec", &N_rec_w, "N_rec/D");
            t->Branch("W_rec", &W_rec_raw, "W_rec/D");
            t->Branch("N_gen", &N_gen_w, "N_gen/D");
            t->Branch("W_gen", &W_gen_raw, "W_gen/D");
            t->Branch("npartials", &npartials, "npartials/I");
            t->Fill();
            t->Write();
            delete t;
        }
        // 关闭 ROOT 文件（root_guard 兜底异常路径）
        rootFile->Close();
        delete rootFile;
        rootFile = nullptr;

        // std::cout << "Data written to ROOT file: " << filename << std::endl;

        // 释放设备内存
        // 释放每个GPU的final_result和partial_result
        for (size_t i = 0; i < d_final_result_vec.size(); ++i) {
            if (d_final_result_vec[i] != nullptr) {
                cudaSetDevice(i);
                cudaFree(d_final_result_vec[i]);
            }
        }
        for (size_t i = 0; i < d_partial_result_vec.size(); ++i) {
            if (d_partial_result_vec[i] != nullptr) {
                cudaSetDevice(i);
                cudaFree(d_partial_result_vec[i]);
            }
        }
        // 释放四动量内存
        for (size_t i = 0; i < device_momenta_list.size(); ++i) {
            if (device_momenta_list[i] != nullptr) {
                cudaSetDevice(i);
                cudaFree(device_momenta_list[i]);
            }
        }
        cudaSetDevice(target_dev);
        // h_total_results / h_partial_results 由 h_guard 统一释放（含异常路径）。
    }

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
                           const std::vector<std::vector<int>>& pairs)
    {
        TORCH_CHECK(ctpwa_is_cuda_like(params), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(!pairs.empty(), "pairs 不能为空");
        ctpwa::ensure_device(primary_dev_);
        auto wi_ev0 = events_offsets_, wi_amp0 = amp_offsets_;
        auto wi_guard = makeScopeExit([this, wi_ev0, wi_amp0]() {
            events_offsets_ = wi_ev0;
            amp_offsets_ = wi_amp0;
            ctpwa::ensure_device(primary_dev_);
        });
        int npartials = (int)nSLvectors_.size();
        std::vector<int> pair_list;
        std::vector<std::pair<int, int>> pair_named;
        for (const auto& pr : pairs) {
            TORCH_CHECK(pr.size() == 2, "每个 pair 需要 [i, j] 两元素");
            int i = pr[0], j = pr[1];
            TORCH_CHECK(i >= 0 && i < npartials && j >= 0 && j < npartials && i != j,
                "pair 下标越界或相同: ", i, ",", j);
            if (i > j) std::swap(i, j);
            pair_list.push_back(i * npartials - i * (i - 1) / 2 + (j - i));
            pair_named.emplace_back(i, j);
        }
        const int npairs = (int)pair_list.size();

        torch::Device dev = params.device();
        torch::Tensor extended_vector;
        int nt = 0;
        if (params_.hasCouplingMatrix()) {
            TORCH_CHECK(params.dtype() == torch::kFloat64,
                "params must be float64 with coupling matrix");
            const auto& cm = params_.couplingMatrix();
            int ncf = cm.n_free, na = cm.n_amps;
            nt = (int)params.size(0) - 2 * ncf;
            auto ext = torch::empty({2 * na + nt},
                torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            params_.extendCouplingParams(params.data_ptr<double>(),
                ext.data_ptr<double>(), ncf, nt);
            extended_vector = torch::complex(
                ext.slice(0, 0, na).to(TORCH_FLOAT),
                ext.slice(0, na, 2 * na).to(TORCH_FLOAT));
            if (nt > 0) {
                auto theta = params.slice(0, 2 * ncf, params.size(0));
                theta_cur_ = theta.detach().clone();
                amp_calc_.reComputeAmps(d_all_amplitudes_,
                    reinterpret_cast<const double*>(theta.data_ptr()),
                    n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                    dev.index());
            }
        } else {
            TORCH_CHECK(params.dtype() == TORCH_COMPLEX,
                "params must be complex128 in legacy mode");
            extended_vector = params_.extendVector(params, dev);
        }
        std::vector<torch::Tensor> ev_per_gpu;
        for (size_t i = 0; i < d_all_amplitudes_.size(); ++i)
            ev_per_gpu.push_back(extended_vector.to(ctpwa_dev(i)));

        int N_phsp = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu)
            N_phsp += events_[gpu][0];
        TORCH_CHECK(N_phsp > 0, "没有 phsp 事件");

        double* h_total = new double[N_phsp];
        double* h_partial = new double[(size_t)N_phsp * npartials];
        double* h_interf = new double[(size_t)N_phsp * npairs];
        auto h_guard = makeScopeExit([&]() {
            delete[] h_total;
            delete[] h_partial;
            delete[] h_interf;
        });

        int ev_cum = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            int ne = events_[gpu][0];
            if (ne == 0) continue;
            cudaSetDevice(gpu);
            // 每 GPU 自己的归约/索引缓冲：跨设备指针传给 computeResults 会非法访问。
            double* d_total_integral = nullptr;
            cudaMalloc(&d_total_integral, sizeof(double));
            int* d_pair_list = nullptr;
            cudaMalloc(&d_pair_list, npairs * sizeof(int));
            cudaMemcpy(d_pair_list, pair_list.data(), npairs * sizeof(int),
                cudaMemcpyHostToDevice);
            int* d_nsl = nullptr;
            cudaMalloc(&d_nsl, npartials * sizeof(int));
            cudaMemcpy(d_nsl, nSLvectors_.data(), npartials * sizeof(int),
                cudaMemcpyHostToDevice);
            double* d_total = nullptr;
            double* d_partial = nullptr;
            double* d_intf = nullptr;
            if (!phsp_freed_) {
                // phsp 驻留模式: 每 GPU 全尺寸结果缓冲
                cudaMalloc(&d_total, ne * sizeof(double));
                cudaMalloc(&d_partial, (size_t)ne * npartials * sizeof(double));
                cudaMemset(d_partial, 0, (size_t)ne * npartials * sizeof(double));
                cudaMalloc(&d_intf, (size_t)ne * npairs * sizeof(double));
            }
            // 本函数只用逐事例干涉项（d_intf）与 partial，矩阵传 nullptr
            cudaMemset(d_total_integral, 0, sizeof(double));

            if (!phsp_freed_) {
                const ctComplex* d_amp_res = d_all_amplitudes_[gpu];
                ctComplex* d_amp_dbl = nullptr;
                if (float_amps_ && !ctIsFloatComplex) {
                    size_t nElem = (size_t)ne * n_polar_ * n_amplitudes_;
                    d_amp_dbl = upcastAmpSegToDouble(d_amp_res, nElem);
                    d_amp_res = d_amp_dbl;
                }
                computeResults(d_amp_res,
                    reinterpret_cast<const ctComplex*>(ev_per_gpu[gpu].data_ptr()),
                    d_total, d_total_integral, d_partial, nullptr, d_intf,
                    nullptr, nullptr, nullptr,
                    d_pair_list, npairs,
                    d_nsl, npartials, ne, n_amplitudes_, n_polar_);
                if (d_amp_dbl) cudaFree(d_amp_dbl);
            } else {
                // phsp 流式模式: 按批重算 phsp 振幅 → computeResults 逐批累加
                const int PHSP_BATCH = 100000;
                long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
                int batch_cap = PHSP_BATCH;
                if (per_ev_bytes > 0) {
                    long long cap = 200LL * 1024 * 1024 / per_ev_bytes;
                    if (cap < 2000) cap = 2000;
                    if (cap < batch_cap) batch_cap = (int)cap;
                }
                auto saved_ev = events_offsets_, saved_amp = amp_offsets_;
                for (int c0 = 0; c0 < ne; c0 += batch_cap) {
                    int nch = std::min(batch_cap, ne - c0);
                    std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_chunk(n_gpus_);
                    for (int g = 0; g < n_gpus_; ++g) {
                        if (g == (int)gpu) {
                            for (const auto& [k, v] : Vp4_all_[gpu])
                                Vp4_chunk[g][k].assign(v.begin() + c0, v.begin() + c0 + nch);
                            events_offsets_[g] = {0, nch};
                            amp_offsets_[g] = {0, nch * n_polar_ * n_amplitudes_};
                        } else {
                            for (const auto& [k, v] : Vp4_all_[g])
                                Vp4_chunk[g][k].assign(v.begin(), v.begin() + 1);
                            events_offsets_[g] = {0, 1};
                            amp_offsets_[g] = {0, n_polar_ * n_amplitudes_};
                        }
                    }
                    std::vector<ctComplex*> d_chunk = calculateAmplitudes(Vp4_chunk);
                    cudaSetDevice(gpu);
                    double* d_ct;
                    cudaMalloc(&d_ct, nch * sizeof(double));
                    double* d_cp;
                    cudaMalloc(&d_cp, (size_t)nch * npartials * sizeof(double));
                    cudaMemset(d_cp, 0, (size_t)nch * npartials * sizeof(double));
                    double* d_cintf;
                    cudaMalloc(&d_cintf, (size_t)nch * npairs * sizeof(double));
                    {
                    const ctComplex* d_chunk_res = d_chunk[gpu];
                    ctComplex* d_chunk_dbl = nullptr;
                    if (float_amps_ && !ctIsFloatComplex) {
                        size_t nElem = (size_t)nch * n_polar_ * n_amplitudes_;
                        d_chunk_dbl = upcastAmpSegToDouble(d_chunk_res, nElem);
                        d_chunk_res = d_chunk_dbl;
                    }
                    computeResults(d_chunk_res,
                        reinterpret_cast<const ctComplex*>(ev_per_gpu[gpu].data_ptr()),
                        d_ct, d_total_integral, d_cp, nullptr, d_cintf,
                        nullptr, nullptr, nullptr,
                        d_pair_list, npairs,
                        d_nsl, npartials, nch, n_amplitudes_, n_polar_);
                    if (d_chunk_dbl) cudaFree(d_chunk_dbl);
                    }
                    cudaMemcpy(h_total + ev_cum + c0, d_ct, nch * sizeof(double),
                        cudaMemcpyDeviceToHost);
                    for (int p = 0; p < npartials; ++p)
                        cudaMemcpy(h_partial + (size_t)p * N_phsp + ev_cum + c0,
                            d_cp + (size_t)p * nch, nch * sizeof(double),
                            cudaMemcpyDeviceToHost);
                    for (int q = 0; q < npairs; ++q)
                        cudaMemcpy(h_interf + (size_t)q * N_phsp + ev_cum + c0,
                            d_cintf + (size_t)q * nch, nch * sizeof(double),
                            cudaMemcpyDeviceToHost);
                    cudaFree(d_ct); cudaFree(d_cp); cudaFree(d_cintf);
                    for (int g = 0; g < n_gpus_; ++g)
                        if (d_chunk[g]) { cudaSetDevice(g); cudaFree(d_chunk[g]); }
                    cudaSetDevice(gpu);
                }
                events_offsets_ = saved_ev;
                amp_offsets_ = saved_amp;
            }

            if (!phsp_freed_) {
                // phsp 驻留模式: 每 GPU 全尺寸结果缓冲 → 一次性拷到 host
                cudaMemcpy(h_total + ev_cum, d_total, ne * sizeof(double),
                    cudaMemcpyDeviceToHost);
                for (int p = 0; p < npartials; ++p)
                    cudaMemcpy(h_partial + (size_t)p * N_phsp + ev_cum,
                        d_partial + (size_t)p * ne, ne * sizeof(double),
                        cudaMemcpyDeviceToHost);
                for (int q = 0; q < npairs; ++q)
                    cudaMemcpy(h_interf + (size_t)q * N_phsp + ev_cum,
                        d_intf + (size_t)q * ne, ne * sizeof(double),
                        cudaMemcpyDeviceToHost);
            }
            // 流式模式: host 结果已在分块循环内逐批拷出
            ev_cum += ne;
            cudaFree(d_nsl);
            cudaFree(d_total_integral);
            cudaFree(d_pair_list);
            if (d_total) cudaFree(d_total);
            if (d_partial) cudaFree(d_partial);
            if (d_intf) cudaFree(d_intf);
        }
        cudaSetDevice(dev.index());

        double h_phsp_integral = 0.0;
        for (int i = 0; i < N_phsp; ++i) h_phsp_integral += h_total[i];
        double dataIntegral = data_total_weight_ - bkg_integral_;
        double normFactor = h_phsp_integral > 0
            ? dataIntegral / h_phsp_integral : 1.0;

        TFile* rootFile = new TFile(filename.c_str(), "RECREATE");
        auto wi_root_guard = makeScopeExit([&]() {
            if (rootFile) { rootFile->Close(); delete rootFile; rootFile = nullptr; }
        });
        TTree* tree = new TTree("saved_weight", "interference shapes per event");
        double total_weight = 0.0;
        tree->Branch("totalweight", &total_weight);
        std::vector<double> partial_weights(npartials, 0.0);
        for (int i = 0; i < npartials; ++i)
            tree->Branch(("weight_" + resonance_names_[i]).c_str(), &partial_weights[i]);
        std::vector<double> interf_vals(npairs, 0.0);
        for (int q = 0; q < npairs; ++q) {
            std::string bn = "interf_" + std::to_string(pair_named[q].first)
                + "_" + std::to_string(pair_named[q].second);
            tree->Branch(bn.c_str(), &interf_vals[q]);
        }
        // 末态四动量
        std::vector<std::string> mom_names;
        if (!Vp4_all_.empty())
            for (const auto& kv : Vp4_all_[0]) mom_names.push_back(kv.first);
        const int n_mom = (int)mom_names.size();
        std::vector<LorentzVector> mom_host((size_t)N_phsp * std::max(1, n_mom));
        {
            int off = 0;
            for (size_t gpu = 0; gpu < Vp4_all_.size(); ++gpu) {
                int ne = events_[gpu][0];
                for (int m = 0; m < n_mom; ++m) {
                    const auto& vec = Vp4_all_[gpu].at(mom_names[m]);
                    for (int e = 0; e < ne; ++e)
                        mom_host[(size_t)(off + e) * n_mom + m] = vec[e];
                }
                off += ne;
            }
        }
        std::vector<double> mom_vals(std::max(1, n_mom) * 4, 0.0);
        for (int m = 0; m < n_mom; ++m) {
            const std::string& nm = mom_names[m];
            tree->Branch((nm + "_px").c_str(), &mom_vals[m * 4 + 0]);
            tree->Branch((nm + "_py").c_str(), &mom_vals[m * 4 + 1]);
            tree->Branch((nm + "_pz").c_str(), &mom_vals[m * 4 + 2]);
            tree->Branch((nm + "_E").c_str(),  &mom_vals[m * 4 + 3]);
        }
        for (int i = 0; i < N_phsp; ++i) {
            total_weight = h_total[i] * normFactor;
            for (int j = 0; j < npartials; ++j)
                partial_weights[j] = h_partial[(size_t)j * N_phsp + i] * normFactor;
            for (int q = 0; q < npairs; ++q)
                interf_vals[q] = h_interf[(size_t)q * N_phsp + i] * normFactor;
            for (int m = 0; m < n_mom; ++m) {
                const LorentzVector& mv = mom_host[(size_t)i * n_mom + m];
                mom_vals[m * 4 + 0] = mv.Px;
                mom_vals[m * 4 + 1] = mv.Py;
                mom_vals[m * 4 + 2] = mv.Pz;
                mom_vals[m * 4 + 3] = mv.E;
            }
            tree->Fill();
        }
        tree->Write();
        rootFile->Close();
        delete rootFile;
        rootFile = nullptr;
        // h_* 由 h_guard 统一释放（含异常路径）。
        std::cout << "writeInterfResult: " << filename
                  << " (pairs=" << npairs << ", events=" << N_phsp << ")" << std::endl;
        return;
    }


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
    torch::Tensor computeCouplingHessian(torch::Tensor vector)
    {
        // P4：实现已搬到 ctpwa::computeCouplingHessian（include/CouplingHessian.h）
        ctpwa::CouplingHessianCtx ctx;
        ctx.d_all_amplitudes = &d_all_amplitudes_;
        ctx.events = &events_;
        ctx.amp_offsets = &amp_offsets_;
        ctx.data_weights = &data_weights_;
        ctx.bkg_weights = &bkg_weights_;
        ctx.d_phsp_matrix = d_phsp_matrix_;
        ctx.params = &params_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_polar = n_polar_;
        ctx.float_amps = float_amps_;
        ctx.primary_dev = primary_dev_;
        ctx.data_total_weight = data_total_weight_;
        ctx.bkg_integral = bkg_integral_;
        ctx.free_params_to_amplitudes =
            [this](const torch::Tensor& v) { return freeParamsToAmplitudes(v); };
        return ctpwa::computeCouplingHessian(ctx, vector);
    }

    // ========== 统一 params 接口 ==========

    // 完整 Hessian: 返回 (2n+P) × (2n+P)，params = [real(v), imag(v), θ] float64
    torch::Tensor getHessian(torch::Tensor params) {
        // P4：实现已搬到 ctpwa::getHessian（include/FullHessian.h）。
        // 十二轮的"先搬底层"到这里收口：它曾经的 12 个依赖只剩 1 个还需要回调。
        ctpwa::FullHessianCtx ctx;
        ctx.d_all_amplitudes = &d_all_amplitudes_;
        ctx.events = &events_;
        ctx.events_offsets = &events_offsets_;
        ctx.amp_offsets = &amp_offsets_;
        ctx.data_weights = &data_weights_;
        ctx.bkg_weights = &bkg_weights_;
        ctx.phsp_weights = &phsp_weights_;
        ctx.d_phsp_matrix = d_phsp_matrix_;
        ctx.params = &params_;
        ctx.amp_calc = &amp_calc_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_polar = n_polar_;
        ctx.primary_dev = primary_dev_;
        ctx.float_amps = float_amps_;
        ctx.phsp_freed = phsp_freed_;
        ctx.data_total_weight = data_total_weight_;
        ctx.bkg_integral = bkg_integral_;
        ctx.free_params_to_amplitudes =
            [this](const torch::Tensor& v) { return freeParamsToAmplitudes(v); };
        return ctpwa::getHessian(ctx, params);
    }

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
        int npartials)
    {
        // P4：实现已搬到 ctpwa::computeFileBatchIntegrals（include/BatchIntegrals.h）。
        // 状态显式化 + 两个回调（θ 读取、按显式 θ 算振幅）。
        ctpwa::BatchIntegralCtx ctx;
        ctx.data_files = &config_parser_.getDataFiles();
        ctx.data_order = &config_parser_.getDataOrder();
        ctx.particles = &particles_;
        ctx.events_offsets = &events_offsets_;
        ctx.amp_offsets = &amp_offsets_;
        ctx.n_slvectors = &nSLvectors_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_polar = n_polar_;
        ctx.float_amps = float_amps_;
        ctx.n_polar = n_polar_;
        ctx.n_gpus = n_gpus_;
        ctx.primary_dev = primary_dev_;
        ctx.float_amps = float_amps_;
        ctx.current_theta_host = [this]() { return currentThetaHost(); };
        ctx.calculate_amplitudes = [this](
            const std::vector<std::map<std::string, std::vector<LorentzVector>>>& Vp4,
            int active_gpus, const double* d_theta) {
            return calculateAmplitudes(Vp4, nullptr, active_gpus, d_theta);
        };
        return ctpwa::computeFileBatchIntegrals(
            ctx, file_key, ev_center, ev_perturbed_p, ev_perturbed_m,
            out_partial, out_scattering, out_square,
            out_p_partial, out_p_scattering, out_m_partial, out_m_scattering, npartials);
    }

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
        torch::Tensor params, torch::Tensor hessian_in)
    {
        // P4：计算在 ctpwa::computeMatrixErrors（include/MatrixErrors.h）。
        // 状态走 ctx 字段，算法走回调——本类只做装配。
        ctpwa::MatrixErrorCtx ctx;
        ctx.events = &events_;
        ctx.phsp_weights = &phsp_weights_;
        ctx.n_slvectors = &nSLvectors_;
        ctx.d_phsp_matrix = d_phsp_matrix_;
        ctx.d_phsp_matrix_double = d_phsp_matrix_double_;
        ctx.n_amplitudes = n_amplitudes_;
        ctx.n_theta = params_.nFreeTheta();
        ctx.primary_dev = primary_dev_;
        ctx.has_gen = config_parser_.getDataFiles().count("phsp_truth") > 0;
        ctx.n_vector = [this]() { return getNVector(); };
        ctx.free_params_to_amplitudes =
            [this](const torch::Tensor& cpl) { return freeParamsToAmplitudes(cpl); };
        ctx.hessian = [this](const torch::Tensor& p) { return getHessian(p); };
        ctx.set_theta = [this](const torch::Tensor& t) { theta_cur_ = t; };
        ctx.get_theta = [this]() { return theta_cur_; };
        ctx.batch_integrals_full =
            [this](const std::string& sample, const torch::Tensor& ev,
                   const std::vector<torch::Tensor>& ev_p,
                   const std::vector<torch::Tensor>& ev_m,
                   double* diag, double* off, double* sq,
                   double* pp, double* ps, double* mp, double* ms, int npart) {
                return computeFileBatchIntegrals(sample, ev, ev_p, ev_m,
                                                 diag, off, sq, pp, ps, mp, ms, npart);
            };
        return ctpwa::computeMatrixErrors(ctx, params, hessian_in);
    }

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

    void reCalcAmp(torch::Tensor params)
    {
        TORCH_CHECK(ctpwa_is_cuda_like(params), "params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64");
        TORCH_CHECK(params.dim() == 1, "params must be 1-dimensional");
        TORCH_CHECK(params.numel() == amp_calc_.nFreeResParams(),
            "params size mismatch: got ", params.numel(),
            ", expected ", amp_calc_.nFreeResParams());

        theta_cur_ = params.detach().clone();
        amp_calc_.reComputeAmps(d_all_amplitudes_,
            reinterpret_cast<const double*>(params.data_ptr()),
            n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
            params.device().index());
        // θ 变了就要把 phsp 的 Gram 一起更新，否则常驻 B̄ 还是旧 θ 的
        //（此前只有 getHessian 会更新，reCalcAmp 之后矩阵/份额会拿到过期值）。
        if (!phsp_freed_) rebuildPhspGram();
    }

    // ---- DIYMC: 任意四动量的逐事件强度 Σ_pol |Σ_i A_i v_i|² ----
    //
    // momenta: CPU float64 [N, n_final, 4] (E, px, py, pz)，末态顺序 = Data.order。
    // params:  当前后端设备上的统一参数 [Re(v), Im(v), θ]（与 getNLL 同布局）。
    //
    // 内部用局部 AmpCalc（不污染拟合用的 amp_calc_）：临时替换
    // events_offsets_/amp_offsets_ → calculateAmplitudes → reComputeAmps(θ)
    // → computeTotalIntensity → 释放 + 恢复。返回 CPU float64 [N]。
    // 归一化任意（只保证相对值），供舍选/诊断使用。
    torch::Tensor getIntensity(torch::Tensor momenta, torch::Tensor params)
    {
        TORCH_CHECK(initialized_, "analysis not initialized");
        TORCH_CHECK(momenta.defined() && momenta.dim() == 3 &&
                momenta.size(2) == 4,
            "getIntensity: momenta must have shape [N, n_final, 4] "
            "(E, px, py, pz)");
        TORCH_CHECK(momenta.scalar_type() == torch::kFloat64,
            "getIntensity: momenta must be float64");
        TORCH_CHECK(momenta.device().is_cpu(),
            "getIntensity: momenta must be a CPU tensor");
        TORCH_CHECK(ctpwa_is_cuda_like(params),
            "getIntensity: params 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(params.dtype() == torch::kFloat64 && params.dim() == 1,
            "getIntensity: params must be 1-D float64");
        TORCH_CHECK(ctpwa_on_primary(params, primary_dev_),
            "getIntensity: params 必须位于主设备 (primary=" +
                std::to_string(primary_dev_) + ")");
        TORCH_CHECK(params.numel() == getNParams(),
            "getIntensity: params size mismatch: got ", params.numel(),
            ", expected ", getNParams());

        const auto& data_order = config_parser_.getDataOrder();
        TORCH_CHECK(momenta.size(1) == static_cast<int64_t>(data_order.size()),
            "getIntensity: momenta final-state count ", momenta.size(1),
            " != Data.order size ", data_order.size());

        const int64_t N = momenta.size(0);
        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        if (N == 0) return torch::empty({ 0 }, opts);

        auto mom = momenta.contiguous();
        auto acc = mom.accessor<double, 3>();

        // N < n_gpus 时只启用前 N 个 GPU；每个启用 GPU 至少 1 个事例。
        const int n_gpu = static_cast<int>(
            std::min<int64_t>(static_cast<int64_t>(n_gpus_), N));
        std::vector<int> counts(static_cast<size_t>(n_gpu), 0);
        const int base = static_cast<int>(N / n_gpu);
        const int rem = static_cast<int>(N % n_gpu);
        for (int g = 0; g < n_gpu; ++g) counts[g] = base + (g < rem ? 1 : 0);

        std::vector<std::map<std::string, std::vector<LorentzVector>>>
            Vp4(static_cast<size_t>(n_gpu));
        int ev_off = 0;
        for (int g = 0; g < n_gpu; ++g) {
            const int ng = counts[g];
            for (size_t a = 0; a < data_order.size(); ++a) {
                auto& v = Vp4[g][data_order[a]];
                v.resize(static_cast<size_t>(ng));
                for (int e = 0; e < ng; ++e) {
                    const int64_t i = ev_off + e;
                    v[e] = LorentzVector(acc[i][a][0], acc[i][a][1],
                        acc[i][a][2], acc[i][a][3]);
                }
            }
            ev_off += ng;
        }

        auto saved_ev = events_offsets_;
        auto saved_amp = amp_offsets_;

        AmpCalc tmp;
        std::vector<ctComplex*> amps;
        std::vector<ctComplex*> upcast;
        std::vector<ctComplex*> d_vec(static_cast<size_t>(n_gpu), nullptr);
        std::vector<double*> d_out(static_cast<size_t>(n_gpu), nullptr);
        std::vector<torch::Tensor> v_hold;
        bool released = false;

        auto cleanup = [&]() {
            for (int g = 0; g < static_cast<int>(upcast.size()); ++g)
                if (upcast[g]) { cudaSetDevice(g); cudaFree(upcast[g]); }
            upcast.clear();
            for (int g = 0; g < static_cast<int>(amps.size()); ++g)
                if (amps[g]) { cudaSetDevice(g); cudaFree(amps[g]); }
            amps.clear();
            for (int g = 0; g < static_cast<int>(d_out.size()); ++g)
                if (d_out[g]) { cudaSetDevice(g); cudaFree(d_out[g]); }
            d_out.clear();
            d_vec.clear();
            if (!released) {
                tmp.release();
                released = true;
            }
            events_offsets_ = saved_ev;
            amp_offsets_ = saved_amp;
            ctpwa::ensure_device(primary_dev_);
        };

        try {
            events_offsets_.assign(static_cast<size_t>(n_gpu),
                std::vector<int>{});
            amp_offsets_.assign(static_cast<size_t>(n_gpu),
                std::vector<int>{});
            for (int g = 0; g < n_gpu; ++g) {
                events_offsets_[g] = { 0, counts[g] };
                amp_offsets_[g] = { 0, counts[g] * n_polar_ * n_amplitudes_ };
            }

            // 临时 AmpCalc 必须与 analysis 的存储档位一致：calculateAmplitudes
            // 按 float_amps_ 分配 A 表，reComputeAmps 按 tmp.float_out_/dF_float_
            // 选择写入布局；不一致会把 double 写进 float2 表（实测全 NaN）。
            tmp.setFloatOutput(float_amps_);
            tmp.setDFloat(prec_mode_ == PrecMode::Float);

            amps = calculateAmplitudes(Vp4, &tmp, n_gpu);
            TORCH_CHECK(static_cast<int>(amps.size()) == n_gpu,
                "getIntensity: calculateAmplitudes returned ", amps.size(),
                " amplitude tables, expected ", n_gpu);

            torch::Tensor vector, theta;
            std::tie(vector, theta) = params_.splitParams(params);
            torch::Tensor v_ext = freeParamsToAmplitudes(vector);
            TORCH_CHECK(v_ext.numel() == n_amplitudes_,
                "getIntensity: extended vector size ", v_ext.numel(),
                " != n_amplitudes ", n_amplitudes_);
            TORCH_CHECK(theta.numel() == tmp.nFreeResParams(),
                "getIntensity: theta size ", theta.numel(),
                " != free resonance params ", tmp.nFreeResParams());
            if (tmp.nFreeResParams() > 0) {
                tmp.reComputeAmps(amps, theta.data_ptr<double>(),
                    n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                    primary_dev_);
            }

            v_hold.reserve(static_cast<size_t>(n_gpu));
            for (int g = 0; g < n_gpu; ++g) {
                v_hold.push_back(v_ext.to(ctpwa_dev(g)));
                d_vec[g] =
                    reinterpret_cast<ctComplex*>(v_hold.back().data_ptr());
            }

            for (int g = 0; g < n_gpu; ++g) {
                cudaSetDevice(g);
                ctComplex* a = amps[g];
                if (float_amps_ && !ctIsFloatComplex && a) {
                    const size_t n_el = static_cast<size_t>(counts[g]) *
                        static_cast<size_t>(n_polar_) *
                        static_cast<size_t>(n_amplitudes_);
                    ctComplex* dbl = upcastAmpSegToDouble(a, n_el);
                    upcast.push_back(dbl);
                    a = dbl;
                }
                cudaError_t e = cudaMalloc(&d_out[g],
                    static_cast<size_t>(counts[g]) * sizeof(double));
                TORCH_CHECK(e == cudaSuccess,
                    "getIntensity: cudaMalloc(out) 失败: ",
                    cudaGetErrorString(e));
                computeTotalIntensity(a, d_vec[g], d_out[g], counts[g],
                    n_amplitudes_, n_polar_);
            }

            torch::Tensor out = torch::empty({ N }, opts);
            double* host = out.data_ptr<double>();
            int off = 0;
            for (int g = 0; g < n_gpu; ++g) {
                cudaSetDevice(g);
                cudaError_t e = cudaMemcpy(host + off, d_out[g],
                    static_cast<size_t>(counts[g]) * sizeof(double),
                    cudaMemcpyDeviceToHost);
                TORCH_CHECK(e == cudaSuccess,
                    "getIntensity: cudaMemcpy(D2H) 失败: ",
                    cudaGetErrorString(e));
                off += counts[g];
            }
            cleanup();
            return out;
        } catch (...) {
            cleanup();
            throw;
        }
    }

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

    void initialize(std::string config_file = "config.yml")
    {
        // 读取配置文件
        // std::string config_file = "config.yml";
        // config_parser_(config_file);
        std::cout << "Reading config file: " << config_file << std::endl;
        std::cout << "Particles: " << config_parser_.getParticles().size()
            << std::endl;
        std::cout << "Decay chains: " << config_parser_.getDecayChains().size()
            << std::endl;
        std::cout << "Resonances: " << config_parser_.getResonances().size()
            << std::endl;
        std::cout << "Constraints: " << config_parser_.getConstraints().size()
            << std::endl;

        // 主设备统一固定为 cuda:0。d_phsp_matrix_、耦合表和统一参数向量都
        // 先放在主卡，再显式复制到其他卡；不能继承前一个多卡调用残留的
        // current device=1，否则后续 make_params(cuda:0) 与主卡定义冲突。
        primary_dev_ = 0;
        ctpwa::ensure_device(primary_dev_);
        params_.setPrimaryDevice(primary_dev_);

        // 初始化粒子信息
        initializeParticles();
        // 初始化极化状态
        initializePolarization();
        // 初始化衰变链
        initializeDecayChains();

        // // 获取配置信息
        // const auto &config_parser = calculator.getConfigParser();
        const auto& data_files = config_parser_.getDataFiles();
        const auto& data_order = config_parser_.getDataOrder();

        legends_ = config_parser_.getLegends();
        n_gls_ = n_amplitudes_;

        std::vector<std::string> particles_names;
        for (const auto& particle : particles_)
        {
            particles_names.push_back(particle.name);
        }
        // events_.clear();
        std::vector<int> init_events;
        init_events.clear();

        std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_to_merge;

        // 计算相空间振幅
        std::cout << "Reading phase space samples..." << std::endl;
        auto Vp4_phsp = readMomentaFromDat(data_files.at("phsp"), data_order, particles_names);
        std::cout << "events: " << Vp4_phsp.begin()->second.size() << std::endl;
        init_events.push_back(Vp4_phsp.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_phsp);

        // 计算数据振幅
        std::cout << "Reading data samples..." << std::endl;
        auto Vp4_data = readMomentaFromDat(data_files.at("data"), data_order, particles_names);
        std::cout << "events: " << Vp4_data.begin()->second.size() << std::endl;
        init_events.push_back(Vp4_data.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_data);

        // 计算本底振幅
        if (data_files.count("bkg") > 0)
        {
            std::cout << "Reading background samples..." << std::endl;
            auto Vp4_bkg = readMomentaFromDat(data_files.at("bkg"), data_order, particles_names);
            std::cout << "events: " << Vp4_bkg.begin()->second.size() << std::endl;
            init_events.push_back(Vp4_bkg.begin()->second.size());
            Vp4_to_merge.push_back(Vp4_bkg);
        }

        // 设备检测：枚举 GPU 属性/显存，替换散落的 cudaGetDeviceCount
        device_mgr_.detect();
        n_gpus_ = device_mgr_.numDevices();

        // 精度解析（运行期档位; .so 编译 double 为默认产物）:
        //   precision:double → PrecMode::Double 全 double（16B 全表 + double 计算）
        //   precision:hybrid / auto（缺省）→ 旧 float-A 混合: A 存 float2(8B),
        //     核心计算仍 double（NLL 相对 double 漂移 ~1e-8）——大统计量默认省显存档
        //   precision:float   → PrecMode::Float 全 float 档（模板化双实例路线,
        //     存储与计算均 float; 逐批接线, 未接线入口显式报错）
        //   .so 编译 float:  precision:float/hybrid/auto → 原生 float 行为;
        //                    precision:double → 报错（float .so 无法提精度）
        std::string req = config_parser_.getPrecision();
        {
            if (req != "auto" && req != "hybrid" && req != "float" && req != "double") {
                std::cerr << "ERROR: 配置 precision=\"" << req
                          << "\" 无效（仅支持 auto | hybrid | float | double）" << std::endl;
                throw std::runtime_error("invalid precision in config");
            }
            const bool double_so = std::string(PRECISION_NAME) == "double";
            prec_mode_ = PrecMode::Double;
#if defined(CTPWA_BACKEND_CPU)
            // CPU 后端固定全 double：float2 档位的意义是省显存，RAM 上不需要，
            // 而且能直接砍掉 9 个 float2 专用 kernel（见 doc/cpu-backend-plan.md §1.3）。
            // 配置里的 precision 仍照收，只是全部映射到 double。
            if (req != "double") {
                std::cerr << "[ctpwa] CPU 后端：precision=" << req
                          << " 已按 double 运行（float 档位仅用于省显存）" << std::endl;
            }
            prec_mode_ = PrecMode::Double;
            float_amps_ = false;
#else
            if (req == "double") {
                if (!double_so) {
                    std::cerr << "ERROR: 配置文件请求 precision=double，但当前 .so 编译为 float；"
                                 "请用默认编译（double）重新 build_ext。float .so 仅供大统计量省显存。"
                              << std::endl;
                    throw std::runtime_error("double requested on float build");
                }
            } else if (!double_so) {
                // float .so: float/hybrid/auto 均原生 float（编译期已定），档位记 Float
                prec_mode_ = PrecMode::Float;
                float_amps_ = true;
            } else if (req == "float") {
                // double .so 上的"全 float 档"（模板化双实例；接线中）
                prec_mode_ = PrecMode::Float;
                float_amps_ = true;   // A 等大表先按 float2 存储（hybrid 同款布局）
            } else {
                // double .so 上 hybrid / auto: A float2 + double 核心（默认体验）
                prec_mode_ = PrecMode::Double;
                float_amps_ = true;
            }
#endif  // CTPWA_BACKEND_CPU
            // 精度档位 banner 注释掉（真实分析每次 ctor 刷屏）:
            // 精度信息由 DeviceManager.print() 的 "complex precision" 行提供。
            // if (prec_mode_ == PrecMode::Float) {
            //     std::cout << "[ctpwa] 全 float 档（precision:float）: 存储/计算 float;"
            //                  " 尚未接线的入口会显式报错，请先用 precision:hybrid/double" << std::endl;
            // } else if (float_amps_) {
            //     std::cout << "[ctpwa] 混合精度模式: A 存 float（省显存）, 核心计算 double";
            //     if (req == "auto" || req == "hybrid")
            //         std::cout << "（precision:" << req << " 默认; 需全 double 请在 config 写 precision: double）";
            //     std::cout << std::endl;
            // }
            amp_calc_.setFloatOutput(float_amps_);
            // d_dF 表存储档位: 仅全 float 档存 float2（hybrid 保持 double 语义）
            amp_calc_.setDFloat(prec_mode_ == PrecMode::Float);
            // 容量预检必须使用 A 的运行时存储档位：double .so + auto/hybrid
            // 实际是 float2(8B)，不能再按 sizeof(ctComplex)=16B 高估一倍。
            device_mgr_.setAmplitudeBytes(float_amps_ ? sizeof(float2) : sizeof(ctComplex));
        }
        if (n_gpus_ == 0) {
            std::cerr << "ERROR: 无可用 CUDA 设备。ctpwa 当前仅支持 GPU 计算"
                         "（CPU 后端尚未实现），无法继续。" << std::endl;
            throw std::runtime_error("no CUDA devices available");
        }
        // 运行时存储档位（打印在 DeviceManager 的 complex 编译精度行之后）:
        //   auto/hybrid: A float2(8B) + dF double2(16B) + double 核心
        //   float:       A/dF float2(8B) + float 核心（Hessian 走 double 上转）
        //   double:      全 double2(16B)
        {
            const char* a_st = float_amps_ ? "float2 (8B)" : "double2 (16B)";
            const char* d_st = (prec_mode_ == PrecMode::Float) ? "float2 (8B)"
                                                               : "double2 (16B)";
            const char* core =
                (prec_mode_ == PrecMode::Float) ? "float" : "double";
            std::string note = std::string("A ") + a_st + " | dF " + d_st +
                               " | 核心 " + core + " (precision: " + req + ")";
            device_mgr_.setStorageNote(note);
        }
        // DeviceManager 设备列表延迟到内存预检之后打印——那时 checkCapacity
        // 已把每 GPU 预计占用写入 estimated_usage_, memory 字段显示"预计/总量"
        // 而非"空闲/总量"。需设备信息本身时仍可调 ctpwa.DeviceManager().print()。

        initializeMultiGPUs(init_events);

        // ---- Constraints.free_phsp_amplitudes: phsp 流式（不驻留）模式判定 ----
        // 仅当所有共振态都没有 free 参数（质量/宽度全固定）时生效；
        // fix_var 只会移除、不会新增自由参数，故 config 层判定是保守且安全的。
        bool phsp_streamed = config_parser_.getFreePhspAmplitudes();
        if (phsp_streamed) {
            bool any_free_declared = false;
            for (const auto& [rname, rc] : config_parser_.getResonances()) {
                (void)rname;
                if (!rc.free.empty()) { any_free_declared = true; break; }
            }
            if (any_free_declared) {
                std::cerr << "WARNING: Constraints.free_phsp_amplitudes 需要所有共振态"
                             "质量/宽度固定（free 为空）；检测到 free 参数，开关被忽略"
                             "（phsp 保持驻留）。" << std::endl;
                phsp_streamed = false;
            }
        }

        // 内存预检：输入数据能否被设备集承载（含每 GPU 事件分布）
        {
            // 常驻振幅表按该 GPU 的 phsp+data+bkg 总事件数分配，不能用单个
            // 样本的最大值估；其他临时 buffer 用它作上界是可接受的保守估计。
            // phsp 流式模式下 phsp 不驻留，只统计 data/bkg。
            std::vector<int> resident_events(n_gpus_, 0);
            for (int gpu = 0; gpu < n_gpus_; ++gpu)
                for (size_t j = (phsp_streamed ? 1 : 0); j < events_[gpu].size(); ++j)
                    resident_events[gpu] += events_[gpu][j];
            bool has_bkg = (init_events.size() > 2 && init_events[2] > 0);
            // SL 表按独立的 chain×JP 级联组合分配，同一 JP 下的全部共振态组合
            // 共享一张表。无 trans/var_equal 时，固定 JP 的表在 A 物化后立即释放；
            // 此时构造期峰值 = max(此前自由 JP 的保留表 + 当前 JP 表)，而不是全表求和。
            const bool early_sl_release = earlyStaticSLReleaseEnabled();
            const auto& config_res = config_parser_.getResonances();
            int n_slcombs_total = 0;
            int n_slcombs_retained = 0;
            for (const auto& chain_info : chains_info_) {
                TORCH_CHECK(chain_info.slcombs_per_jp.size() == chain_info.intermediate_combs.size(),
                    "ChainInfo ", chain_info.name, " 的 SL 计数与 JP 组合数不一致");
                for (size_t ci = 0; ci < chain_info.intermediate_combs.size(); ++ci) {
                    const int nsl = chain_info.slcombs_per_jp[ci];
                    if (!early_sl_release) {
                        n_slcombs_total += nsl;
                        continue;
                    }
                    // 同一 cas 下只要有一个共振态组合含自由参数，整张 SL 表都要保留。
                    // fix_var 可能进一步把它固定，因此这里的保留估计只会偏保守。
                    bool keep_cas = false;
                    for (const auto& particle : chain_info.intermediate_combs[ci]) {
                        const auto key = std::make_pair(particle.name,
                            std::vector<int>{particle.spin, particle.parity});
                        const auto ri = chain_info.intermediate_resonance_map.find(key);
                        if (ri == chain_info.intermediate_resonance_map.end()) continue;
                        for (const auto& res : ri->second) {
                            const auto cfg = config_res.find(res.getName());
                            if (cfg != config_res.end() && !cfg->second.free.empty())
                                keep_cas = true;
                        }
                    }
                    n_slcombs_total = std::max(n_slcombs_total, n_slcombs_retained + nsl);
                    if (keep_cas) n_slcombs_retained += nsl;
                }
            }
            auto cap = device_mgr_.checkCapacity(
                resident_events, n_amplitudes_, n_polar_, n_slcombs_total, particles_.size(),
                has_bkg, params_.nFreeTheta());
            // 预检后打印设备列表（含每 GPU 预计占用）
            device_mgr_.print();
            if (cap.overall == DeviceManager::CapacityStatus::FAIL) {
                std::cerr << "ERROR: 内存预检失败——GPU " << cap.failing_device
                          << " 无法承载数据: " << cap.failing_buffer
                          << " 预计总计 " << cap.required_bytes / 1e6
                          << " MiB，可用 " << cap.available_bytes / 1e6
                          << " MiB（事件数 " << resident_events[cap.failing_device]
                          << "）。请减少事件数或使用多 GPU。" << std::endl;
                for (const auto& [name, bytes] : cap.breakdown) {
                    std::cerr << "    - " << name << ": "
                              << bytes / 1048576.0 << " MiB" << std::endl;
                }
                if (std::getenv("CTPWA_SKIP_MEM_CHECK") != nullptr) {
                    std::cerr << "WARNING: CTPWA_SKIP_MEM_CHECK 已设置，跳过预检；"
                                 "若实际显存不足会在 cudaMalloc 处报错。" << std::endl;
                } else {
                    throw std::runtime_error("input data exceeds device memory");
                }
            } else if (cap.overall == DeviceManager::CapacityStatus::WARN) {
                std::cerr << "WARNING: 内存占用超过可用显存的 80%，"
                             "建议减少事件数或增加 GPU。" << std::endl;
            } else {
                std::cout << "Memory check passed." << std::endl;
            }
        }

        std::cout << "Calculating amplitudes..." << std::endl;
        Vp4_all_ = mergeMaps(Vp4_to_merge, events_);
        events_offsets_.clear();
        amp_offsets_.clear();
        for (size_t i = 0; i < events_.size(); ++i)
        {
            std::vector<int> ev_offsets;
            std::vector<int> amp_offsets;
            int ev_sum = 0;
            int amp_sum = 0;
            ev_offsets.push_back(ev_sum);
            amp_offsets.push_back(amp_sum);
            for (size_t j = 0; j < events_[i].size(); ++j)
            {
                // std::cout << "Events in dataset " << i << ", type " << j << ": " << events_[i][j] << std::endl;
                ev_sum += events_[i][j];
                amp_sum += events_[i][j] * n_polar_ * n_amplitudes_;
                ev_offsets.push_back(ev_sum);
                amp_offsets.push_back(amp_sum);
                // std::cout << "  Cumulative events: " << ev_sum << ", Cumulative amplitudes: " << amp_sum << std::endl;
            }
            events_offsets_.push_back(ev_offsets);
            amp_offsets_.push_back(amp_offsets);
        }

        if (phsp_streamed) {
            // ---- phsp 流式模式：常驻表只含 data+bkg ----
            // 常驻表布局 [data | bkg]：data 在偏移 0，bkg 在 events_[gpu][1]·PA。
            // 消费端统一用 amp_offsets_[gpu][1]/[2] 取 data/bkg 基址：
            //   freed:  amp_offsets_ = {0, 0, N_data·PA, (N_data+N_bkg)·PA} → data=0, bkg=N_data·PA
            //   非 freed: amp_offsets_ = {0, N_phsp·PA, (N_phsp+N_data)·PA, ...}（虚拟，含 phsp）
            // 因此下面把成员 amp_offsets_ 改写为主表相对约定（data=0）。
            std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_db(n_gpus_);
            for (size_t g = 0; g < (size_t)n_gpus_; ++g) {
                int nP = events_[g][0];
                int nD = events_[g][1];
                int nB = (events_[g].size() > 2) ? events_[g][2] : 0;
                for (const auto& [k, v] : Vp4_all_[g]) {
                    auto& dst = Vp4_db[g][k];
                    dst.assign(v.begin() + nP, v.begin() + nP + nD + nB);
                }
                events_offsets_[g] = {0, nD, nD + nB};
                amp_offsets_[g] = {0, nD * n_polar_ * n_amplitudes_,
                                   (nD + nB) * n_polar_ * n_amplitudes_};
            }
            d_all_amplitudes_ = calculateAmplitudes(Vp4_db, &amp_calc_);
            // 防御: config 层判定（无 free 声明）应与实际自由参数数一致
            if (amp_calc_.nFreeResParams() > 0) {
                throw std::runtime_error(
                    "free_phsp_amplitudes 与自由共振态参数检测不一致"
                    "（不应发生：free 参数只来自 Resonances.free）");
            }
            // 恢复事件偏移（虚拟布局，四动量索引用）；振幅偏移改写为主表相对约定
            for (size_t g = 0; g < (size_t)n_gpus_; ++g) {
                std::vector<int> ev_off = {0};
                int cum = 0;
                for (size_t j = 0; j < events_[g].size(); ++j) {
                    cum += events_[g][j];
                    ev_off.push_back(cum);
                }
                events_offsets_[g] = ev_off;
                std::vector<int> amp_off = {0, 0};
                int amp_cum = 0;
                for (size_t j = 1; j < events_[g].size(); ++j) {
                    amp_cum += events_[g][j] * n_polar_ * n_amplitudes_;
                    amp_off.push_back(amp_cum);
                }
                amp_offsets_[g] = amp_off;   // {0, 0, N_data·PA, (N_data+N_bkg)·PA}
            }
            phsp_freed_ = true;
        } else {
            d_all_amplitudes_ = calculateAmplitudes(Vp4_all_, &amp_calc_);
        }

#if defined(CTPWA_BACKEND_CPU)
        // 诊断：CPU 后端振幅表填充检查（CTPWA_CPU_DEBUG=1）
        if (getenv("CTPWA_CPU_DEBUG") && !d_all_amplitudes_.empty()) {
            for (size_t g = 0; g < d_all_amplitudes_.size(); ++g) {
                const long long n = (long long)amp_offsets_[g].back();
                if (n <= 0 || d_all_amplitudes_[g] == nullptr) continue;
                std::vector<ctComplex> h((size_t)n);
                cudaMemcpy(h.data(), d_all_amplitudes_[g],
                           (size_t)n * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                long long n_nan = 0, n_zero = 0;
                double vmin = 1e300, vmax = -1e300, amax = 0.0;
                for (long long i = 0; i < n; ++i) {
                    const double re = (double)h[(size_t)i].x, im = (double)h[(size_t)i].y;
                    if (std::isnan(re) || std::isnan(im)) { ++n_nan; continue; }
                    if (re == 0.0 && im == 0.0) ++n_zero;
                    vmin = std::min(vmin, std::min(re, im));
                    vmax = std::max(vmax, std::max(re, im));
                    amax = std::max(amax, std::sqrt(re * re + im * im));
                }
                std::printf("[CPU-DBG] gpu %zu: n=%lld nan=%lld zero=%lld "
                            "min=%.6g max=%.6g |A|max=%.6g | first8:",
                            g, n, n_nan, n_zero, vmin, vmax, amax);
                for (int i = 0; i < 8 && i < n; ++i)
                    std::printf(" (%.4g,%.4g)", (double)h[(size_t)i].x,
                                (double)h[(size_t)i].y);
                std::printf("\n");
            }
        }
#endif

        // 设置共振态自由参数个数（calculateAmplitudes 内部已应用 var_equal 槽合并）
        if (!amp_calc_.empty()) {
            params_.setNFreeTheta(amp_calc_.nFreeResParams());
            // fix_var/var_equal 移除了部分槽: 用实际槽名重建 theta 参数名段，
            // 保证参数名与自由参数一一对应（getParamNames 长度不变量）
            const auto& slots = amp_calc_.slots();
            auto names = params_.paramNames();
            if (static_cast<int>(names.size()) >= n_coupling_names_) {
                names.resize(n_coupling_names_);
                for (const auto& s : slots) names.push_back(s.name);
                params_.setParamNames(names);
            }
        }

        // Constraints.gauss_constr: name → theta 下标映射（μ 默认取 slot 初值）
        if (!amp_calc_.empty() && !config_parser_.getGaussConstr().empty()) {
            const auto& gc = config_parser_.getGaussConstr();
            const auto& slots = amp_calc_.slots();
            std::map<std::string, int> name_to_idx;
            std::map<std::string, double> sigma, mu;
            for (size_t s = 0; s < slots.size(); ++s) {
                auto git = gc.find(slots[s].name);
                if (git != gc.end()) {
                    name_to_idx[slots[s].name] = static_cast<int>(s);
                    sigma[slots[s].name] = git->second;
                    mu[slots[s].name] = slots[s].init_value;
                }
            }
            params_.setGaussConstr(name_to_idx, sigma, mu);
        }

        // 输出d_all_amplitudes_[0]所有内容:
        // int Ntotal = 0;
        // for (size_t j = 0; j < events_[0].size(); ++j)
        // {
        //     Ntotal += events_[0][j];
        // }
        // ctComplex* h_amp = new ctComplex[Ntotal * n_polar_ * n_amplitudes_];
        // cudaMemcpy(h_amp, d_all_amplitudes_[0], Ntotal * n_polar_ * n_amplitudes_ * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < Ntotal * n_polar_ * n_amplitudes_; ++i)
        // {
        //     std::cout << "Amplitude[" << i << "] = " << h_amp[i].x << " + " << h_amp[i].y << "i" << std::endl;
        // }
        // delete[] h_amp;

        // data_weights_
        data_total_weight_ = 0.0;
        if (data_files.count("data_weights") > 0)
        {
            std::vector<int> data_events_per_gpu;
            for (size_t i = 0; i < events_.size(); ++i)
                data_events_per_gpu.push_back(events_[i][1]);
            data_weights_ = readWeightsFromFile(data_files.at("data_weights"), data_events_per_gpu);
            // Σ data 权重（phsp 归一化 N_eff 用；无权重 GPU 按事件数计）
            for (size_t gpu = 0; gpu < data_weights_.size(); ++gpu) {
                if (data_weights_[gpu] != nullptr && events_[gpu].size() > 1 && events_[gpu][1] > 0) {
                    cudaSetDevice(gpu);
                    thrust::device_ptr<double> dp(data_weights_[gpu]);
                    data_total_weight_ += thrust::reduce(dp, dp + events_[gpu][1]);
                }
                else if (events_[gpu].size() > 1) {
                    data_total_weight_ += events_[gpu][1];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < events_.size(); ++i) {
                data_weights_.push_back(nullptr);
                if (events_[i].size() > 1)
                    data_total_weight_ += events_[i][1];
            }
        }

        // phsp_weights_
        if (data_files.count("phsp_weights") > 0)
        {
            std::vector<int> phsp_events_per_gpu;
            for (size_t i = 0; i < events_.size(); ++i)
                phsp_events_per_gpu.push_back(events_[i][0]);
            phsp_weights_ = readWeightsFromFile(data_files.at("phsp_weights"), phsp_events_per_gpu);
        }
        else
        {
            for (size_t i = 0; i < events_.size(); ++i)
                phsp_weights_.push_back(nullptr);
        }

        // bkg_weights_
        if (data_files.count("bkg_weights") > 0 && data_files.count("bkg") > 0)
        {
            std::vector<int> bkg_events_per_gpu;
            for (size_t i = 0; i < events_.size(); ++i)
                bkg_events_per_gpu.push_back(events_[i][2]);
            bkg_weights_ = readWeightsFromFile(data_files.at("bkg_weights"), bkg_events_per_gpu);

            // bkg_weigths_求和
            double h_bkg_integral = 0.0;
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
                cudaSetDevice(gpu);
                if (events_[gpu][2] > 0) {
                    thrust::device_ptr<double> d_ptr_gpu(bkg_weights_[gpu]);
                    h_bkg_integral += thrust::reduce(d_ptr_gpu, d_ptr_gpu + events_[gpu][2]);
                }
            }
            bkg_integral_ = h_bkg_integral;
        }
        else
        {
            for (size_t i = 0; i < events_.size(); ++i)
            {
                bkg_weights_.push_back(nullptr);
                if (events_[i].size() > 2)
                    bkg_integral_ += events_[i][2];
            }
        }

        // // 打印d_all_amplitudes_[0]的所有元素，验证数据正确加载
        // ctComplex* h_amp = new ctComplex[n_amplitudes_ * n_polar_ * (init_events[0] + init_events[1])];
        // cudaMemcpy(h_amp, d_all_amplitudes_[0], n_amplitudes_ * n_polar_ * (init_events[0] + init_events[1]) * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // std::cout << "First amplitudes on GPU 0:" << std::endl;
        // for (int i = 0; i < n_amplitudes_ * n_polar_ * (init_events[0] + init_events[1]); ++i)
        // {
        //     std::cout << "  Amplitude[" << i << "] = " << h_amp[i].x << " + " << h_amp[i].y << "i" << std::endl;
        // }
        // delete[] h_amp;

        // phsp*phsp^T矩阵: A^H diag(w) A / (Σ w)，加权或均匀
        // Pre-compute total phsp weight for correct global normalization
        double W_total = 0.0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            int nP = events_[gpu][0];
            if (nP == 0) continue;
            if (gpu < phsp_weights_.size() && phsp_weights_[gpu] != nullptr) {
                cudaSetDevice(gpu);
                thrust::device_ptr<double> dp(phsp_weights_[gpu]);
                W_total += thrust::reduce(dp, dp + nP);
            } else {
                W_total += (double)nP;
            }
        }
        if (W_total <= 0.0) W_total = 1.0;

        ctpwa::ensure_device(primary_dev_);  // ensure d_phsp_matrix_ is allocated on primary GPU
        cudaMalloc(&d_phsp_matrix_, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
        cudaMemset(d_phsp_matrix_, 0, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));

        if (phsp_freed_) {
            // ---- phsp 流式（不驻留）: 分块算 phsp 振幅 → 累加 float32 + double 矩阵 ----
            // 每批: calculateAmplitudes(单批 Vp4) → 批振幅临时缓冲 → 累加矩阵 → 释放。
            // 峰值 = 常驻 data/bkg 表 + 一批振幅，与 phsp 总量无关。
            cudaMalloc(&d_phsp_matrix_double_,
                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
            cudaMemset(d_phsp_matrix_double_, 0,
                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
            const int PHSP_BATCH = 100000;
            long long per_ev_bytes = (long long)n_polar_ * n_amplitudes_ * 8;
            int batch_cap = PHSP_BATCH;
            if (per_ev_bytes > 0) {
                long long cap = 200LL * 1024 * 1024 / per_ev_bytes;  // ≤ ~200MB/批
                if (cap < 2000) cap = 2000;
                if (cap < batch_cap) batch_cap = (int)cap;
            }
            std::cout << "[ctpwa] phsp 流式模式: 分块构建 phsp 矩阵 (batch="
                      << batch_cap << " 事件/批)..." << std::endl;
            auto saved_ev = events_offsets_, saved_amp = amp_offsets_;
            for (int gpu = 0; gpu < n_gpus_; ++gpu) {
                int nPhsp = events_[gpu][0];
                if (nPhsp == 0) continue;
                cudaSetDevice(gpu);
                cublasHandle_t h;
                cublasCreate(&h);
                for (int c0 = 0; c0 < nPhsp; c0 += batch_cap) {
                    int nch = std::min(batch_cap, nPhsp - c0);
                    // 批四动量（本 GPU 取 [c0, c0+nch)，其他 GPU 放 1 事件占位防空 map）
                    std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_batch(n_gpus_);
                    for (int g = 0; g < n_gpus_; ++g) {
                        if (g == gpu) {
                            for (const auto& [k, v] : Vp4_all_[gpu])
                                Vp4_batch[g][k].assign(v.begin() + c0, v.begin() + c0 + nch);
                            events_offsets_[g] = {0, nch};
                            amp_offsets_[g] = {0, nch * n_polar_ * n_amplitudes_};
                        } else {
                            for (const auto& [k, v] : Vp4_all_[g])
                                Vp4_batch[g][k].assign(v.begin(), v.begin() + 1);
                            events_offsets_[g] = {0, 1};
                            amp_offsets_[g] = {0, n_polar_ * n_amplitudes_};
                        }
                    }
                    std::vector<ctComplex*> d_batch = calculateAmplitudes(Vp4_batch);
                    cudaSetDevice(gpu);
                    int nTot = nch * n_polar_ * n_amplitudes_;

                    // float 模式（double .so 混合精度）: 批振幅为 float2（8B/元素），
                    // 下方"原生 ctComplex 累加"与"double 累加"都按 ctComplex/cuDoubleComplex
                    // 读——先上转一份 double 段供两处共用（峰值 = 常驻表 + 一批 ×2，
                    // batch_cap ≤ ~200MB/批 → 有界，与 phsp 总量无关）。
                    ctComplex* d_batch_up = nullptr;   // 非空 = d_batch_r 指向它的 double 上转段
                    const ctComplex* d_batch_r = d_batch[gpu];
                    if (float_amps_ && !ctIsFloatComplex) {
                        d_batch_up = upcastAmpSegToDouble(d_batch[gpu], (size_t)nTot);
                        d_batch_r = d_batch_up;
                    }

                    // float32 累加（scale 后 CGEMM）
                    {
                        ctComplex* d_scaled;
                        cudaMalloc(&d_scaled, nTot * sizeof(ctComplex));
                        cudaMemcpy(d_scaled, d_batch_r, nTot * sizeof(ctComplex),
                                   cudaMemcpyDeviceToDevice);
                        const double* d_w = (gpu < (int)phsp_weights_.size())
                            ? phsp_weights_[gpu] : nullptr;
                        int grid = (nTot + 255) / 256;
                        CTPWA_LAUNCH(scalePhspAmpsKernel, grid, 256, 0, d_scaled,
                            d_w ? d_w + c0 : nullptr, nch, n_polar_, n_amplitudes_,
                            1.0 / W_total, 0);
                        ctComplex* d_m;
                        cudaMalloc(&d_m, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                        ctComplex alpha = ctMake(1.0f, 0.0f), beta = ctMake(0.0f, 0.0f);
                        CUBLAS_CGEMM(h, CUBLAS_OP_N, CUBLAS_OP_C,
                            n_amplitudes_, n_amplitudes_, nch * n_polar_,
                            &alpha, d_scaled, n_amplitudes_, d_scaled, n_amplitudes_,
                            &beta, d_m, n_amplitudes_);
                        if (gpu == primary_dev_) {
                            axpyComplex(d_phsp_matrix_, d_m, ctMake(1.0f, 0.0f),
                                        n_amplitudes_ * n_amplitudes_);
                        } else {
                            ctpwa::ensure_device(primary_dev_);
                            ctComplex* d_temp;
                            cudaMalloc(&d_temp, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                            cudaMemcpyPeer(d_temp, primary_dev_, d_m, gpu,
                                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                            axpyComplex(d_phsp_matrix_, d_temp, ctMake(1.0f, 0.0f),
                                        n_amplitudes_ * n_amplitudes_);
                            cudaFree(d_temp);
                            cudaSetDevice(gpu);
                        }
                        cudaFree(d_scaled);
                        cudaFree(d_m);
                    }

                    // double 累加（cast → ZGEMM）
                    {
                        cuDoubleComplex* d_batch_d;
                        cudaMalloc(&d_batch_d, (size_t)nTot * sizeof(cuDoubleComplex));
                        if (d_batch_up != nullptr) {
                            // double .so 上 cuDoubleComplex 与 ctComplex 布局相同:
                            // 直接 D2D 拷贝上转段（值=double 原生, 与 cast kernel 等价）
                            cudaMemcpy(d_batch_d, d_batch_up,
                                (size_t)nTot * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice);
                        } else {
                            int grid2 = (nTot + 255) / 256;
                            CTPWA_LAUNCH(castPhspBatchToDoubleKernel, grid2, 256, 0, d_batch[gpu], d_batch_d, nTot);
                        }
                        // 与 float32 分支**同一归一化**：A ← A·sqrt(w/W_total)。
                        // 之前漏了这一步，double 矩阵是 Σ A A^H（未加权/未除 W），
                        // 下游又乘回 W_total → W_rec 大 N 倍、有 phsp_weights 时整个
                        // interference_rec 变成无权重矩阵。
                        {
                            const double* d_w = (gpu < (int)phsp_weights_.size())
                                ? phsp_weights_[gpu] : nullptr;
                            scalePhspAmpsDouble(d_batch_d, d_w ? d_w + c0 : nullptr,
                                nch, n_polar_, n_amplitudes_, 1.0 / W_total);
                        }
                        cuDoubleComplex* d_md;
                        cudaMalloc(&d_md, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
                        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
                        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
                        cublasZgemm(h, CUBLAS_OP_N, CUBLAS_OP_C,
                            n_amplitudes_, n_amplitudes_, nch * n_polar_,
                            &alpha, d_batch_d, n_amplitudes_, d_batch_d, n_amplitudes_,
                            &beta, d_md, n_amplitudes_);
                        if (gpu == primary_dev_) {
                            axpyZComplex(d_phsp_matrix_double_, d_md,
                                         n_amplitudes_ * n_amplitudes_);
                        } else {
                            ctpwa::ensure_device(primary_dev_);
                            cuDoubleComplex* d_temp;
                            cudaMalloc(&d_temp, (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
                            cudaMemcpyPeer(d_temp, primary_dev_, d_md, gpu,
                                (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(cuDoubleComplex));
                            axpyZComplex(d_phsp_matrix_double_, d_temp,
                                         n_amplitudes_ * n_amplitudes_);
                            cudaFree(d_temp);
                            cudaSetDevice(gpu);
                        }
                        cudaFree(d_batch_d);
                        cudaFree(d_md);
                    }

                    // 释放上转段（本 GPU）与批振幅（含占位 GPU 的 1 事件缓冲）
                    if (d_batch_up != nullptr) cudaFree(d_batch_up);
                    for (int g = 0; g < n_gpus_; ++g) {
                        if (d_batch[g]) { cudaSetDevice(g); cudaFree(d_batch[g]); }
                    }
                    cudaSetDevice(gpu);
                }
                cublasDestroy(h);
            }
            events_offsets_ = saved_ev;
            amp_offsets_ = saved_amp;
            ctpwa::ensure_device(primary_dev_);
            std::cout << "[ctpwa] phsp 矩阵（float32+double）构建完成，phsp 振幅不驻留。"
                      << std::endl;
        } else {
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu)
            {
                int nPhsp = events_[gpu][0];
                if (nPhsp == 0) continue;
                cudaSetDevice(gpu);

                // resident 模式构造期 B̄ 构建。float 模式下 A 为 float2（8B/元素），
                // B̄ 保持 double（phsp_factor 恒走 double 轨道）→ 按块上转 double 再
                // 走原 scale+CGEMM（峰值 = 单块，与 getHessian 分块重建同模式）。
                const int kPhspBatch = phspGramBatchEvents();
                for (int c0 = 0; c0 < nPhsp; c0 += kPhspBatch) {
                    int nch = std::min(kPhspBatch, nPhsp - c0);
                    int nch_tot = nch * n_polar_ * n_amplitudes_;
                    const double* d_w = (gpu < phsp_weights_.size()) ? phsp_weights_[gpu] : nullptr;
                    ctComplex* d_phsp_scaled;
                    ctpwaCheckedMalloc((void**)&d_phsp_scaled,
                        (size_t)nch_tot * sizeof(ctComplex), "phsp Gram scaled batch");
                    if (float_amps_ && !ctIsFloatComplex) {
                        // float2 A 块 → double 块（A 元素偏移 = c0*nPolar*nAmp，单位元素数）
                        const float2* srcF = reinterpret_cast<const float2*>(d_all_amplitudes_[gpu])
                            + (size_t)c0 * n_polar_ * n_amplitudes_;
                        CTPWA_LAUNCH(castF2ToDouble2Kernel, (nch_tot + 255) / 256, 256, 0, 
                            srcF, reinterpret_cast<cuDoubleComplex*>(d_phsp_scaled), nch_tot);
                    } else {
                        cudaMemcpy(d_phsp_scaled,
                                   d_all_amplitudes_[gpu] + (size_t)c0 * n_polar_ * n_amplitudes_,
                                   nch_tot * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                    }
                    int grid = (nch_tot + 255) / 256;
                    CTPWA_LAUNCH(scalePhspAmpsKernel, grid, 256, 0, d_phsp_scaled,
                        d_w ? d_w + c0 : nullptr, nch, n_polar_, n_amplitudes_,
                        1.0 / W_total, 0);

                    ctComplex* d_phsp_gpu;
                    ctpwaCheckedMalloc((void**)&d_phsp_gpu,
                        (size_t)n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex),
                        "phsp Gram batch matrix");
                    cublasHandle_t h;
                    cublasCreate(&h);
                    ctComplex alpha = ctMake(1.0f, 0.0f);
                    ctComplex beta  = ctMake(0.0f, 0.0f);
                    CUBLAS_CGEMM(h, CUBLAS_OP_N, CUBLAS_OP_C, n_amplitudes_, n_amplitudes_, nch * n_polar_,
                        &alpha, d_phsp_scaled, n_amplitudes_, d_phsp_scaled, n_amplitudes_,
                        &beta, d_phsp_gpu, n_amplitudes_);
                    cublasDestroy(h);
                    cudaFree(d_phsp_scaled);

                    if (gpu == (size_t)primary_dev_) {
                        ctComplex one = ctMake(1.0f, 0.0f);
                        axpyComplex(d_phsp_matrix_, d_phsp_gpu, one, n_amplitudes_ * n_amplitudes_);
                        cudaFree(d_phsp_gpu);
                    } else {
                        ctpwa::ensure_device(primary_dev_);
                        ctComplex* d_temp;
                        cudaMalloc(&d_temp, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                        cudaMemcpyPeer(d_temp, primary_dev_, d_phsp_gpu, gpu,
                            n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                        cudaSetDevice(gpu);
                        cudaFree(d_phsp_gpu);
                        ctpwa::ensure_device(primary_dev_);
                        ctComplex one = ctMake(1.0f, 0.0f);
                        axpyComplex(d_phsp_matrix_, d_temp, one, n_amplitudes_ * n_amplitudes_);
                        cudaFree(d_temp);
                        cudaSetDevice(gpu);
                    }
                }
            }
        }
        // // cudaFree(d_phsp);
        // // 打印矩阵d_phsp
        // ctComplex* h_phsp = new ctComplex[n_amplitudes_ * n_amplitudes_];
        // cudaMemcpy(h_phsp, d_phsp_matrix_, n_amplitudes_* n_amplitudes_ * sizeof(ctComplex), cudaMemcpyDeviceToHost);
        // std::cout << "Phase space matrix (phsp*phsp^T):" << std::endl;
        // for (int i = 0; i < n_amplitudes_; ++i) {
        //     for (int j = 0; j < n_amplitudes_; ++j) {
        //         std::cout << "  [" << i << "][" << j << "] = " << h_phsp[i * n_amplitudes_ + j].x << " + " << h_phsp[i * n_amplitudes_ + j].y << "j" << std::endl;
        //     }
        // }
        // delete[] h_phsp;

        // 静态 SL 缩减：仅保留含自由参数（含 trans/var_equal 关联）链组合的 slamp，
        // 固定链组合的 A 已在构造期物化 → 释放其 slamp（大样本可省数 GB/卡）。
        // 逃生阀 CTPWA_KEEP_SLAMP=1 可关闭。
        if (!amp_calc_.empty()) amp_calc_.releaseStaticCasData();

        std::cout << "Number of GPUs available: " << n_gpus_ << std::endl;
        // n_gls_ 是 SL 振幅条数（= 进入 cublas 的列数），不是分波数：多 SL 的链上
        // 两者差别很大（如 51 个分波 → 105 条 SL 振幅），而 writeResult 的模板桶/
        // 局部内存是按**分波数** nSLvectors_.size() 选的，所以两个都打出来。
        std::cout << "Number of partial waves: " << nSLvectors_.size()
                  << " (SL amplitudes: " << n_gls_ << ")" << std::endl;
        std::cout << "Number of res parameters: " << params_.nFreeTheta() << std::endl;
        std::cout << "Initialization complete." << std::endl;

        // 恢复主设备为当前设备：构造期大量 per-GPU 循环（SL 表/振幅/权重上传/
        // resident B̄ 构建）会把 raw current device 留在最后一张卡。torch 的
        // current-device 缓存在其首次 CUDA 调用时从 raw device 懒初始化——
        // 漏掉此恢复会让随后的 torch.device('cuda')/tensor 落到非主 GPU
        // （getNLL/getHessian 的 params 位置检查触发，测试套件双卡实测必现）。
        ctpwa::ensure_device(primary_dev_);
    }

    void initializeMultiGPUs(std::vector<int> init_events)
    {
        // 根据GPU个数分配evets给events_，每个元素都均匀分配
        if (n_gpus_ != 0)
        {
            events_.clear();                         // 清空旧数据
            events_.resize(n_gpus_);                 // 为每个 GPU 预留一行

            // 为每个 GPU 的每一列（事件类型）预先分配空间
            for (int gpu = 0; gpu < n_gpus_; ++gpu) {
                events_[gpu].resize(init_events.size());
            }

            // 对每一种事件类型进行分配
            for (size_t j = 0; j < init_events.size(); ++j) {
                int val = init_events[j];            // 该类事件的总数
                int base = val / n_gpus_;            // 每个 GPU 至少获得的数量
                int remainder = val % n_gpus_;       // 余数，前 remainder 个 GPU 各多得 1

                for (int gpu = 0; gpu < n_gpus_; ++gpu) {
                    events_[gpu][j] = base + (gpu < remainder ? 1 : 0);
                }
            }
        }
        else
        {
            std::cerr << "No GPUs available. Running on CPU." << std::endl;
        }
    }

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

    void initializeDecayChains()
    {
        // Use DecayInfo to build coupling matrix from config
        DecayInfo info(config_parser_);
        if (!info.isValid()) return;

        // Copy data from DecayInfo
        amplitude_names_   = info.amplitudeNames();
        resonance_names_   = info.resonanceNames();
        chains_info_       = info.chainInfos();
        n_amplitudes_      = info.nAmplitudes();
        nSLvectors_        = info.nSLvectors();

        // Set coupling matrix + param names on Parameters
        if (info.hasCouplingMatrix()) {
            params_.setCouplingMatrix(info.couplingMatrix());
            auto all_names = info.paramNames();
            const auto& rnames = info.resonanceParamNames();
            all_names.insert(all_names.end(), rnames.begin(), rnames.end());
            params_.setParamNames(all_names);
            // 耦合参数名段长度（theta 段重建时保留）
            n_coupling_names_ = static_cast<int>(info.paramNames().size());

            // std::cout << "Coupling matrix: " << info.couplingMatrix().n_amps
            //           << " amplitudes → " << info.couplingMatrix().n_free
            //           << " free coupling params ("
            //           << info.couplingMatrix().n_chain_free << " chain + "
            //           << info.couplingMatrix().n_step_free << " step)" << std::endl;
        } else {
            // Legacy constraint mode
            auto constraints = config_parser_.getConstraints();
            std::vector<std::vector<int>> con_trans_id;
            std::vector<std::vector<std::complex<double>>> con_trans_values;
            for (const auto& constraint : constraints) {
                std::vector<std::vector<int>> amp_ids_con;
                for (const auto& amp_name : constraint.names) {
                    std::vector<int> amp_ids;
                    for (int i = 0; i < (int)amplitude_names_.size(); ++i)
                        if (amplitude_names_[i].find(amp_name) != std::string::npos)
                            amp_ids.push_back(i);
                    amp_ids_con.push_back(amp_ids);
                }
                if (amp_ids_con.empty() || amp_ids_con[0].empty()) continue;
                for (size_t i = 0; i < amp_ids_con[0].size(); ++i) {
                    std::vector<int> combination;
                    for (size_t j = 0; j < amp_ids_con.size(); ++j)
                        combination.push_back(amp_ids_con[j][i]);
                    con_trans_id.push_back(combination);
                    std::vector<std::complex<double>> values = {std::complex<double>(1.0, 1.0)};
                    for (const auto& val : constraint.values) values.push_back(val);
                    con_trans_values.push_back(values);
                }
            }
            params_.initialize(n_amplitudes_ - static_cast<int>(con_trans_id.size()),
                               con_trans_id, con_trans_values);
        }

        // vspace mode: direct amplitude mapping, preserving trans folding
        if (fit_mode_ == 1) {
            const auto& cm = info.couplingMatrix();

            CouplingMatrixResult id;
            id.n_amps = n_amplitudes_;
            id.n_step_free = 0;
            id.amp_step_params.resize(n_amplitudes_);
            id.amp_chain.resize(n_amplitudes_);
            // 保留 buildWithTrans 计算的 trans ratio（非折叠=1.0，折叠=-1.0 等）
            id.amp_chain_ratio = cm.amp_chain_ratio;

            // 判断是否需要"拆分"：检查是否有 active chain 包含多个 owner 振幅
            // 这发生在同一共振态有多个 SL 组合时
            std::map<int, std::vector<int>> chain_owners;
            for (int ai = 0; ai < n_amplitudes_; ++ai) {
                if (cm.amp_chain[ai] < 0) continue;   // 不按 ratio 过滤: trans 值可为 1.0
                chain_owners[cm.amp_chain[ai]].push_back(ai);
            }

            bool needs_split = false;
            for (const auto& [ci, owners] : chain_owners) {
                if (owners.size() > 1) { needs_split = true; break; }
            }

            // free param → 组代表链振幅名(折叠链共享参数时显示代表链)
            std::map<int, std::string> free_idx_to_name;

            if (!needs_split) {
                // 简单情况：每个 active chain 恰好 1 个 owner 振幅
                // buildWithTrans 的结果可直接用于 vspace
                id.amp_chain = cm.amp_chain;
                id.chain_names = cm.chain_names;
                id.n_chain_free = cm.n_chain_free;
                id.n_free = id.n_chain_free;
            } else {
                // 复杂情况：同一共振态有多个 SL 组合，需要拆分 chain param
                // 使用 amp_map 恢复原始 chain 归属和链内排序
                std::map<std::string, std::vector<int>> orig_chain_amps;
                for (const auto& am : cm.amp_map) {
                    orig_chain_amps[am.chain_key].push_back(am.amp_idx);
                }
                for (auto& [ck, amps] : orig_chain_amps) {
                    std::sort(amps.begin(), amps.end());
                }

                // 折叠判定: 映射到同一 active index 的多条原链共享参数。
                // 注意不能用 ratio ≠ 1 判定 —— trans 约束值本身可为 1.0,
                // 折叠链的 ratio(=约束值) 与「每条链一个参数」时的 ratio 都是 1.0。
                // 组内第一条原链(按字符串序, 即展开顺序)的振幅分配独立参数,
                // 同组其余原链按位置对齐到组代表的对应参数。
                int next_free = 0;
                std::map<int, std::map<std::string, std::vector<int>>> groups; // ac → ck → amps
                for (const auto& [ck, amps] : orig_chain_amps) {
                    if (amps.empty() || cm.amp_chain[amps[0]] < 0) continue;
                    groups[cm.amp_chain[amps[0]]][ck] = amps;
                }
                // active_chain → 组代表(第一条原链)各位置对应的 free param index
                std::map<int, std::vector<int>> ac_free_indices;

                for (const auto& [ac, ck_amps] : groups) {
                    for (const auto& [ck, amps] : ck_amps) {
                        bool is_owner = (ck == ck_amps.begin()->first);
                        const std::vector<int>& owner_amps = ck_amps.begin()->second;
                        if (is_owner) {
                            for (int ai : amps) {
                                id.amp_chain[ai] = next_free;
                                ac_free_indices[ac].push_back(next_free);
                                next_free++;
                            }
                        } else {
                            const auto& free_indices = ac_free_indices[ac];
                            for (size_t pos = 0; pos < amps.size()
                                 && pos < free_indices.size() && pos < owner_amps.size(); ++pos) {
                                id.amp_chain[amps[pos]] = free_indices[pos];
                            }
                        }
                    }
                }

                id.n_chain_free = next_free;
                id.n_free = next_free;
            }

            params_.setCouplingMatrix(id);

            // 参数名 = 最后一个映射到该 free param 的振幅名（与历史行为一致：
            // 折叠链共享参数时均指向同一个 index, 后写覆盖 = 展开序靠后者）
            for (int ai = 0; ai < n_amplitudes_; ++ai) {
                int f = id.amp_chain[ai];
                if (f < 0) continue;
                free_idx_to_name[f] = amplitude_names_[ai];
            }
            std::vector<std::string> vspace_names;
            for (int idx = 0; idx < id.n_chain_free; ++idx) {
                auto it = free_idx_to_name.find(idx);
                vspace_names.push_back(it != free_idx_to_name.end() ? it->second : "");
            }
            const auto& rnames = info.resonanceParamNames();
            vspace_names.insert(vspace_names.end(), rnames.begin(), rnames.end());
            params_.setParamNames(vspace_names);
        }
    }

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
