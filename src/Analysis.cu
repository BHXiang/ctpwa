// #include <pybind11/pybind11.h>
#include <chrono>
#include <cstdio>
#include "ComplexType.h"
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <random>
#include <torch/extension.h>
#include <unordered_set>
#include <vector>

#include <AmpGen.cuh>
#include <DeviceManager.cuh>
// #include <ComputeGrad.cuh>
#include <ComputeNLL.cuh>
#include <ComputeResults.cuh>
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

//////////////////////////////////////////////
// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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

std::map<std::string, std::vector<LorentzVector>> readMomentaFromDat(
    const std::vector<std::string>& fileinfo,
    const std::vector<std::string>& particleNames,
    const std::vector<std::string>& particlelists,
    int nEvents = -1,
    int offset = 0)
{
    std::map<std::string, std::vector<LorentzVector>> fullMomenta;
    for (const auto& name : particlelists)
    {
        fullMomenta[name] = std::vector<LorentzVector>();
    }

    // 检查输入参数
    if (fileinfo.size() < 2)
    {
        std::cerr
            << "Error: fileinfo must contain at least file type and filename"
            << std::endl;
        return fullMomenta;
    }

    std::string fileType = fileinfo[0];
    std::string filename = fileinfo[1];

    std::unordered_set<std::string> particleNameSet(particleNames.begin(),
        particleNames.end());
    std::string initialName;
    bool foundParticle = false;

    for (const auto& name : particlelists)
    {
        if (particleNameSet.find(name) == particleNameSet.end())
        {
            if (!foundParticle)
            {
                initialName = name;
                foundParticle = true;
            }
            else
            {
                std::cerr
                    << "Error: Found multiple particles in particlelists not "
                    "present in particleNames"
                    << std::endl;
                return fullMomenta;
            }
        }
    }

    // 处理DAT文件
    if (fileType == "dat")
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return fullMomenta;
        }

        std::string line;
        int eventCount = 0;
        int lineCount = 0;
        int skippedEvents = 0;
        int particlesPerEvent = particleNames.size();
        int skipLines = offset * particlesPerEvent;

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            std::istringstream iss(line);
            double E, px, py, pz;

            if (iss >> E >> px >> py >> pz)
            {
                // Skip lines for offset events
                if (lineCount < skipLines) {
                    lineCount++;
                    continue;
                }

                // 根据行号确定粒子类型
                int particleIndex = (lineCount - skipLines) % particlesPerEvent;
                const std::string& particleName = particleNames[particleIndex];

                fullMomenta[particleName].emplace_back(E, px, py, pz);
                lineCount++;

                // 每读完一组粒子表示完成一个事件
                if (particleIndex == particlesPerEvent - 1)
                {
                    LorentzVector initialMomentum(0, 0, 0, 0);
                    for (const auto& name : particleNames)
                    {
                        initialMomentum =
                            initialMomentum + fullMomenta[name].back();
                    }

                    fullMomenta[initialName].emplace_back(initialMomentum);

                    eventCount++;

                    // 如果指定了事件数并且已达到，则停止读取
                    if (nEvents > 0 && eventCount >= nEvents)
                    {
                        break;
                    }
                }
            }
            else
            {
                std::cerr << "Warning: Invalid line format: " << line
                    << std::endl;
            }
        }

        file.close();
    }
    // 处理ROOT文件
    else if (fileType == "ROOT" || fileType == "root")
    {
        // #ifdef USE_ROOT
        if (fileinfo.size() < 3)
        {
            std::cerr
                << "Error: For ROOT files, fileinfo must contain at least file "
                "type, filename and TTree name"
                << std::endl;
            return fullMomenta;
        }

        std::string treeName = fileinfo[2];

        // 打开ROOT文件
        TFile* file = TFile::Open(filename.c_str(), "READ");
        if (!file || file->IsZombie())
        {
            std::cerr << "Error: Cannot open ROOT file " << filename
                << std::endl;
            return fullMomenta;
        }

        // 获取TTree
        TTree* tree = (TTree*)file->Get(treeName.c_str());
        if (!tree)
        {
            std::cerr << "Error: Cannot find TTree " << treeName << " in file "
                << filename << std::endl;
            file->Close();
            delete file;
            return fullMomenta;
        }

        // 准备读取TLorentzVector的分支
        std::vector<TLorentzVector*> particleLV(particleNames.size());
        std::vector<std::string> branchNames;

        // 如果提供了分支名，使用提供的分支名
        if (fileinfo.size() >= 2 + particleNames.size())
        {
            for (size_t i = 0; i < particleNames.size(); ++i)
            {
                branchNames.push_back(fileinfo[3 + i]);
            }
        }
        // 否则使用粒子名作为分支名
        else
        {
            branchNames = particleNames;
        }

        // 设置分支地址
        for (size_t i = 0; i < particleNames.size(); ++i)
        {
            particleLV[i] = new TLorentzVector();
            tree->SetBranchAddress(branchNames[i].c_str(), &particleLV[i]);
        }

        // 读取事件
        Long64_t nEntries = tree->GetEntries();
        if (nEvents > 0 && nEvents < nEntries)
        {
            nEntries = nEvents;
        }

        for (Long64_t iEvent = 0; iEvent < nEntries; ++iEvent)
        {
            tree->GetEntry(iEvent);

            // 读取每个粒子的四动量
            for (size_t i = 0; i < particleNames.size(); ++i)
            {
                const std::string& particleName = particleNames[i];
                TLorentzVector* lv = particleLV[i];

                // 转换为你的LorentzVector类型
                // 假设你的LorentzVector构造函数接受(E, px, py, pz)
                fullMomenta[particleName].emplace_back(lv->E(), lv->Px(),
                    lv->Py(), lv->Pz());
            }

            // 计算初始粒子的四动量
            LorentzVector initialMomentum(0, 0, 0, 0);
            for (const auto& name : particleNames)
            {
                initialMomentum = initialMomentum + fullMomenta[name].back();
            }
            fullMomenta[initialName].emplace_back(initialMomentum);
        }

        // 清理内存
        for (auto lv : particleLV)
        {
            delete lv;
        }

        file->Close();
        delete file;
        // #else
        // std::cerr << "Error: ROOT support not compiled in. Please define
        // USE_ROOT and link with ROOT libraries." << std::endl; return
        // fullMomenta; #endif
    }
    else
    {
        std::cerr << "Error: Unknown file type: " << fileType << std::endl;
        return fullMomenta;
    }

    return fullMomenta;
}

std::vector<double*> readWeightsFromFile(const std::vector<std::string>& fileinfo, const std::vector<int>& events_per_gpu)
{
    // 读取所有权重到主机内存
    std::vector<double> weights;

    if (fileinfo.empty()) {
        std::cerr << "Error: fileinfo is empty" << std::endl;
        return {};
    }

    std::string fileType = fileinfo[0];
    std::string filename = fileinfo.size() > 1 ? fileinfo[1] : "";

    // 常量权重: 只传入一个值，所有bkg事例乘以这个weight
    if (fileinfo.size() == 1) {
        double const_weight = std::stod(fileType);
        int total_events = 0;
        for (int ev : events_per_gpu) total_events += ev;
        weights.resize(total_events, const_weight);
    }
    // 检查输入参数
    else if (fileinfo.size() < 2)
    {
        std::cerr << "Error: fileinfo must contain at least file type and filename" << std::endl;
        return {};
    }

    if (weights.empty())
    {

    // 处理DAT文件
    if (fileType == "dat")
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return {};
        }

        double weight;
        while (file >> weight)
        {
            weights.push_back(weight);
        }

        file.close();
    }
    // 处理ROOT文件
    else if (fileType == "ROOT" || fileType == "root")
    {
        // #ifdef USE_ROOT
        if (fileinfo.size() < 4)
        {
            std::cerr << "Error: For ROOT files, fileinfo must contain at least file "
                "type, filename, TTree name and weight branch name" << std::endl;
            return {};
        }

        std::string treeName = fileinfo[2];
        std::string branchName = fileinfo[3];

        // 打开ROOT文件
        TFile* file = TFile::Open(filename.c_str(), "READ");
        if (!file || file->IsZombie())
        {
            std::cerr << "Error: Cannot open ROOT file " << filename << std::endl;
            return {};
        }

        // 获取TTree
        TTree* tree = (TTree*)file->Get(treeName.c_str());
        if (!tree)
        {
            std::cerr << "Error: Cannot find TTree " << treeName << " in file " << filename << std::endl;
            file->Close();
            delete file;
            return {};
        }

        // 设置权重分支
        double weight = 0.0;
        tree->SetBranchAddress(branchName.c_str(), &weight);

        // 读取所有事件的权重
        Long64_t nEntries = tree->GetEntries();
        for (Long64_t iEvent = 0; iEvent < nEntries; ++iEvent)
        {
            tree->GetEntry(iEvent);
            weights.push_back(weight);
        }

        file->Close();
        delete file;
        // #else
        // std::cerr << "Error: ROOT support not compiled in. Please define
        // USE_ROOT and link with ROOT libraries." << std::endl; return {};
        // #endif
    }
    else
    {
        std::cerr << "Error: Unknown file type: " << fileType << std::endl;
        return {};
    }

    } // end if (weights.empty())

    // 计算总权重数
    int total_weights = 0;
    for (int ev : events_per_gpu)
    {
        total_weights += ev;
    }

    // 检查权重数量是否匹配
    if (weights.size() != static_cast<size_t>(total_weights))
    {
        std::cerr << "Error: Weights size " << weights.size() << " does not match total events " << total_weights << std::endl;
        // 可以根据需求决定是否返回空向量或调整大小
        // 这里选择调整大小，不足补零，多余截断
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
static inline bool hessianFastPathEnabled() {
    const char* e = getenv("CTPWA_HESS_FAST");
    return (e == nullptr) || (atoi(e) != 0);
}

// ----------------------------------------------------------------
// phsp 流式模式辅助 kernel（Constraints.free_phsp_amplitudes）
// ----------------------------------------------------------------
// 批振幅 float→double 转换（供 cublasZgemm 累加 double phsp 矩阵）
__global__ void castPhspBatchToDoubleKernel(
    const ctComplex* __restrict__ src, cuDoubleComplex* __restrict__ dst, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = make_cuDoubleComplex((double)src[i].x, (double)src[i].y);
}

// y += x（cuDoubleComplex 版 axpy）
__global__ void axpyZComplexKernel(
    cuDoubleComplex* __restrict__ y, const cuDoubleComplex* __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i].x += x[i].x;
    y[i].y += x[i].y;
}
static void axpyZComplex(cuDoubleComplex* y, const cuDoubleComplex* x, int n)
{
    int grid = (n + 255) / 256;
    axpyZComplexKernel<<<grid, 256>>>(y, x, n);
}
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
        // phsp 矩阵/梯度段的持久化缓冲（懒分配，避免每次 forward 的 cublasCreate/cudaMalloc）
        static std::vector<ctComplex*> s_d_phsp_scaled;
        static std::vector<ctComplex*> s_d_phsp_gpu;
        static std::vector<cublasHandle_t> s_phsp_handle;
        static std::vector<int> s_phsp_alloc_sz;   // 每 GPU 已分配的 nPhsp_total
        static std::vector<int> s_phsp_mat_sz;     // 已分配的矩阵大小（nA*nA）
        static std::vector<ctComplex*> s_d_S_bufs;
        static std::vector<int> s_dS_alloc_sz;

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
        TORCH_CHECK(vector.is_cuda(), "params must be on CUDA");

        // 1. 将 vector 扩展向量
        torch::Tensor extended_vector = params_mgr->extendVector(vector, vector.device());
        int extended_n_gls = extended_vector.numel();
        const int primary_dev = vector.get_device();

        // 2. 将extended_vector分发到每个GPU（实际拷贝）
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < num_gpus; ++i) {
            extended_vec_per_gpu.push_back(
                extended_vector.to(torch::Device(torch::kCUDA, i)));
        }

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

            // 预计算有效耦合 T（复用各 GPU 上的 extended_vector）
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(gpu);
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(
                    extended_vec_per_gpu[gpu].data_ptr());
                amp_calc->computeEffectiveCoupling(d_v_gpu, extended_n_gls);
                cudaDeviceSynchronize();
            }
            t2_reamp = std::chrono::high_resolution_clock::now();
            if (getenv("CTPWA_PROF")) printf("[PROF] reComputeAmps: %.2f ms, +T: %.2f ms\n",
                std::chrono::duration<double, std::milli>(t1_reamp - t0_reamp).count(),
                std::chrono::duration<double, std::milli>(t2_reamp - t1_reamp).count());

            // 更新 d_phsp_matrix_（振幅已变，phsp 矩阵需同步）
            cudaSetDevice(primary_dev);
            // Pre-compute total phsp weight for correct global normalization
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
            cudaSetDevice(primary_dev);
            tF2 = std::chrono::high_resolution_clock::now();  // W_total reduce 完成

            cudaMemset(d_phsp_matrix_, 0, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
            if ((int)s_phsp_handle.size() < num_gpus) {
                s_d_phsp_scaled.resize(num_gpus, nullptr);
                s_d_phsp_gpu.resize(num_gpus, nullptr);
                s_phsp_handle.resize(num_gpus, nullptr);
                s_phsp_alloc_sz.assign(num_gpus, 0);
                s_phsp_mat_sz.assign(num_gpus, 0);
            }
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                int nPhsp = events_list[gpu][0];
                if (nPhsp == 0) continue;
                cudaSetDevice(gpu);
                if (!s_phsp_handle[gpu]) cublasCreate(&s_phsp_handle[gpu]);

                int nPhsp_total = nPhsp * n_polar_ * n_amplitudes_;
                int nMat = n_amplitudes_ * n_amplitudes_;
                if (s_phsp_alloc_sz[gpu] < nPhsp_total) {
                    if (s_d_phsp_scaled[gpu]) cudaFree(s_d_phsp_scaled[gpu]);
                    cudaMalloc(&s_d_phsp_scaled[gpu], nPhsp_total * sizeof(ctComplex));
                    s_phsp_alloc_sz[gpu] = nPhsp_total;
                }
                if (s_phsp_mat_sz[gpu] < nMat) {
                    if (s_d_phsp_gpu[gpu]) cudaFree(s_d_phsp_gpu[gpu]);
                    cudaMalloc(&s_d_phsp_gpu[gpu], nMat * sizeof(ctComplex));
                    s_phsp_mat_sz[gpu] = nMat;
                }
                ctComplex* d_phsp_scaled = s_d_phsp_scaled[gpu];
                cudaMemcpy(d_phsp_scaled, d_all_amplitudes_list[gpu],
                           nPhsp_total * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                const double* d_w = (gpu < (int)d_phsp_weights_list.size()) ? d_phsp_weights_list[gpu] : nullptr;
                int grid = (nPhsp_total + 255) / 256;
                scalePhspAmpsKernel<<<grid, 256>>>(d_phsp_scaled, d_w,
                    nPhsp, n_polar_, n_amplitudes_, 1.0 / W_total, 0);

                ctComplex* d_phsp_gpu = s_d_phsp_gpu[gpu];
                ctComplex alpha = ctMake(1.0f, 0.0f);
                ctComplex beta  = ctMake(0.0f, 0.0f);
                CUBLAS_CGEMM(s_phsp_handle[gpu], CUBLAS_OP_N, CUBLAS_OP_C,
                    n_amplitudes_, n_amplitudes_, nPhsp * n_polar_,
                    &alpha, d_phsp_scaled, n_amplitudes_,
                    d_phsp_scaled, n_amplitudes_,
                    &beta, d_phsp_gpu, n_amplitudes_);

                if (gpu == primary_dev) {
                    ctComplex one = ctMake(1.0f, 0.0f);
                    axpyComplex(d_phsp_matrix_, d_phsp_gpu, one, nMat);
                } else {
                    cudaSetDevice(primary_dev);
                    ctComplex* d_temp;
                    cudaMalloc(&d_temp, nMat * sizeof(ctComplex));
                    cudaMemcpyPeer(d_temp, primary_dev, d_phsp_gpu, gpu,
                        nMat * sizeof(ctComplex));
                    cudaSetDevice(gpu);
                    cudaSetDevice(primary_dev);
                    ctComplex one = ctMake(1.0f, 0.0f);
                    axpyComplex(d_phsp_matrix_, d_temp, one, nMat);
                    cudaFree(d_temp);
                }
            }
        }

        tF3 = std::chrono::high_resolution_clock::now();  // phsp 矩阵 CGEMM 完成

        // 3. 全局量: d_P_vec和phsp_factor（小矩阵M<100，用自定义核替代cuBLAS更快）
        cudaSetDevice(primary_dev);
        ctComplex* d_P_vec;
        cudaMalloc(&d_P_vec, n_amplitudes_ * sizeof(ctComplex));
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
        // phsp_factor = mean f: Σ f_ev 用 double 累加（float32 phsp 矩阵求和在
        // 50 万事件上有 ~1e-4 相对系统偏差 → NLL 与 tf-pwa (float64) 差 ~2.4）
        double phsp_sum = 0.0;
        int total_phsp_evts = 0;
        if (d_phsp_matrix_double_ != nullptr) {
            // phsp 流式模式: 原始 phsp 振幅不驻留，phsp_sum = Re(v^H M_double v)（主 GPU）
            TORCH_CHECK(n_free_res == 0,
                "phsp 流式模式（free_phsp_amplitudes）要求无自由共振态参数");
            cudaSetDevice(primary_dev);
            const ctComplex* d_v_primary = reinterpret_cast<const ctComplex*>(
                extended_vector.data_ptr());
            phsp_sum = computeDoublePhspSum(d_phsp_matrix_double_, d_v_primary,
                                            n_amplitudes_);
            for (int gpu = 0; gpu < num_gpus; ++gpu) total_phsp_evts += events_list[gpu][0];
        } else {
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                int nP = events_list[gpu][0];
                if (nP == 0) continue;
                cudaSetDevice(gpu);
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(
                    extended_vec_per_gpu[gpu].data_ptr());
                phsp_sum += computePhspMeanSum(d_all_amplitudes_list[gpu], d_v_gpu,
                                               nP, n_polar_, n_amplitudes_);
                total_phsp_evts += nP;
            }
        }
        double phsp_factor = (total_phsp_evts > 0) ? phsp_sum / total_phsp_evts : 0.0;
        cudaSetDevice(primary_dev);
        tF4 = std::chrono::high_resolution_clock::now();  // phsp_factor D2H 完成

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

            cudaSetDevice(primary_dev);
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
        cudaSetDevice(primary_dev);
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
                double nll = computeFactorNLL(d_amp, d_vec_gpu,
                    d_grad, nData_gpu, n_polar_, n_amplitudes_, d_w_data, d_w_out);

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
                    cudaSetDevice(primary_dev);
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
                double nll = computeFactorNLL(d_amp, d_vec_gpu,
                    d_grad, nBkg_gpu, n_polar_, n_amplitudes_, d_w, d_w_bkg_out);

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
                    cudaSetDevice(primary_dev);
                    axpyComplex(d_grad_global, d_grad_buf, ctMake(-1.0f, 0.0f), extended_n_gls);
                }
            }
        }

        t6_nll = std::chrono::high_resolution_clock::now();  // computeFactorNLL data+bkg 完成

        // === 新增：计算共振态参数梯度 ===
        torch::Tensor grad_theta;
        if (amp_calc && n_free_res > 0 && theta.numel() > 0) {
            cudaSetDevice(primary_dev);
            // 持久化懒分配（n_free_res 固定，避免每次 forward 的 malloc/free）
            static double* s_d_grad_res = nullptr;
            static int s_grad_res_sz = 0;
            if (s_grad_res_sz < n_free_res) {
                if (s_d_grad_res) cudaFree(s_d_grad_res);
                cudaSetDevice(primary_dev);
                cudaMalloc(&s_d_grad_res, n_free_res * sizeof(double));
                s_grad_res_sz = n_free_res;
            }
            double* d_grad_res = s_d_grad_res;
            cudaSetDevice(primary_dev);
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
                amp_calc->computeResonanceGradient(s_d_w_bufs, n_data_events, d_grad_res,
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
                amp_calc->computeResonanceGradient(s_d_w_bkg_bufs, n_bkg_events, d_grad_res,
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
                    double phsp_sign = -effective_data / (phsp_factor * total_phsp);

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
                        ctComplex* d_amp = d_all_amplitudes_list[g];
                        const ctComplex* d_vg = reinterpret_cast<const ctComplex*>(
                            extended_vec_per_gpu[g].data_ptr());
                        ctComplex a = ctMake(1.0f, 0.0f);
                        ctComplex b = ctMake(0.0f, 0.0f);
                        CUBLAS_CGEMV(s_phsp_handle[g], CUBLAS_OP_T, n_amplitudes_, nTot,
                            &a, d_amp, n_amplitudes_, d_vg, 1, &b, d_S_bufs[g], 1);
                    }

                    cudaSetDevice(primary_dev);  // d_grad_res is on primary_dev
                    std::vector<ctComplex*> d_v_ptrs(num_gpus);
                    for (int g = 0; g < num_gpus; ++g)
                        d_v_ptrs[g] = const_cast<ctComplex*>(
                            reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[g].data_ptr()));
                    auto t0_rgp = std::chrono::high_resolution_clock::now();
                    amp_calc->computeResonanceGradient(d_S_bufs, n_phsp_evts, d_grad_res,
                        phsp_sign, {}, d_v_ptrs);
                    auto t1_rgp = std::chrono::high_resolution_clock::now();
                    if (getenv("CTPWA_PROF")) printf("[PROF] resonanceGradient(phsp): %.2f ms\n",
                        std::chrono::duration<double, std::milli>(t1_rgp - t0_rgp).count());

                    cudaSetDevice(primary_dev);
                }
            }
            t7_rg = std::chrono::high_resolution_clock::now();  // resonanceGradient 各段完成

            cudaSetDevice(primary_dev);
            grad_theta = torch::empty({ n_free_res },
                torch::TensorOptions().dtype(torch::kFloat64).device(torch::Device(torch::kCUDA, primary_dev)));
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
        cudaSetDevice(primary_dev);
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
            printf("[PROF] f.extend: %.2f | f.reAmp+T: %.2f | f.Wred: %.2f | f.phspM: %.2f | f.quad+D2H: %.2f | f.nll: %.2f | f.gradRes: %.2f | f.tail: %.2f\n",
                ms(tF0, tF1), ms(tF1, t2_reamp), ms(t2_reamp, tF2), ms(tF2, tF3),
                ms(tF3, tF4), ms(tF4, t6_nll), ms(t6_nll, t7_rg), ms(t7_rg, tFend));
        }
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
class analysis
{
public:
    analysis(const std::string& config_file = "config.yml")
        : config_parser_(config_file), n_amplitudes_(0), n_polar_(0), d_all_amplitudes_(), initialized_(false)
    {
        if (!config_parser_.isValid()) {
            std::cerr << "Warning: Config file \"" << config_file
                      << "\" is empty or not found. Analysis not initialized." << std::endl;
            return;
        }
        initialize();
        initialized_ = true;
    }

    // 析构函数，用于释放 CUDA 内存
    ~analysis()
    {
        // 同 AmpCalc::~AmpCalc：不在析构里释放 GPU 内存——pybind11 对象
        // 在进程退出 GC 阶段析构时 torch 已销毁 CUDA 上下文，裸 cudaFree
        // 会段错误（实测 rc=139）。CUDA 上下文销毁自动回收显存。
        d_all_amplitudes_.clear();
    }

    bool isValid() const { return initialized_; }

    torch::Tensor getNLL(torch::Tensor params)
    {
        TORCH_CHECK(initialized_, "analysis not initialized: invalid or missing config file");
        TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64");
        TORCH_CHECK(params.device().is_cuda() && params.device().index() == primary_dev_,
            "params 必须位于主 GPU (cuda:" + std::to_string(primary_dev_) + ")，"
            "当前在 cuda:" + std::to_string(params.device().index()) + "。"
            "跨 GPU 归一化缓冲 (d_phsp_matrix_) 固定在主 GPU 上");

        return NLLFunction::apply(params, &params_, d_all_amplitudes_, &amp_calc_,
            d_phsp_matrix_, d_phsp_matrix_double_, events_, events_offsets_, amp_offsets_,
            data_weights_, phsp_weights_, bkg_weights_, bkg_integral_, n_amplitudes_, n_polar_);
    }

    // ---- fit mode ----
    void setFitMode(int mode) { fit_mode_ = mode; }
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
        torch::Device dev(torch::kCUDA, 0);
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt).device(dev);
        return torch::tensor(nSLvectors_, options);
    }

    // waves: 可选, 只画选中分波子集 |Σ_{i∈S} A_i·v_i|² 的分布（空=全部=现状）。
    // 实现: kernel 用 mask 覆盖每事件总值 + 子集积分, 下游(直方图/TTree/归一化)不变。
    void writeResult(torch::Tensor params, const std::string& filename,
                     const int is_saved_weight = 0,
                     const std::vector<int>& waves = {})
    {
        TORCH_CHECK(params.is_cuda(), "params must be on CUDA");

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
                amp_calc_.reComputeAmps(d_all_amplitudes_,
                    reinterpret_cast<const double*>(theta.data_ptr()),
                    n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                    dev.index());
            }
        } else {
            TORCH_CHECK(params.dtype() == TORCH_COMPLEX, "params must be complex128 in legacy mode");
            extended_vector = params_.extendVector(params, dev);
        }

        // 将extended_vector分配到多个GPU
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            extended_vec_per_gpu.push_back(extended_vector.to(torch::Device(torch::kCUDA, i)));
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
        double* d_total_integral;
        cudaMalloc(&d_total_integral, sizeof(double));
        cudaMemset(d_total_integral, 0, sizeof(double));

        // 分配干涉矩阵和事件干涉项的设备内存
        double* d_interference_matrix;
        cudaMalloc(&d_interference_matrix, npartials * npartials * sizeof(double));
        cudaMemset(d_interference_matrix, 0, npartials * npartials * sizeof(double));

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
        double* h_interference_matrix = new double[nSLvectors_.size() * nSLvectors_.size()];
        double* h_total_results = new double[N_phsp];
        double* h_partial_results = new double[N_phsp * npartials];
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
            // 单个GPU interference matrix
            double* d_interference_matrix_gpu;
            cudaMalloc(&d_interference_matrix_gpu, npartials * npartials * sizeof(double));
            cudaMemset(d_interference_matrix_gpu, 0, npartials * npartials * sizeof(double));
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
                // phsp 驻留模式（现状）: 直接读常驻表 phsp 区
                computeResults(d_all_amplitudes_[gpu],
                    reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[gpu].data_ptr()),
                    d_final_result_vec[gpu], d_total_integral_gpu, d_partial_result_vec[gpu],
                    d_interference_matrix_gpu, nullptr,
                    d_wave_mask_gpu, d_selected_integral_gpu,
                    d_final_result_vec[gpu],
                    nullptr, 0,
                    d_nSLvectors, npartials, events_[gpu][0], n_amplitudes_, n_polar_);
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
                        computeResults(d_chunk[gpu],
                            reinterpret_cast<const ctComplex*>(extended_vec_per_gpu[gpu].data_ptr()),
                            d_ct, d_total_integral_gpu, d_cp,
                            d_interference_matrix_gpu, nullptr,
                            d_wave_mask_gpu, d_selected_integral_gpu,
                            d_ct,
                            nullptr, 0,
                            d_nSLvectors, npartials, nch, n_amplitudes_, n_polar_);
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
            double* h_interference_matrix_gpu = new double[npartials * npartials];
            cudaMemcpy(h_interference_matrix_gpu, d_interference_matrix_gpu, npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < npartials * npartials; ++i) {
                h_interference_matrix[i] += h_interference_matrix_gpu[i];
            }
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
            cudaFree(d_interference_matrix_gpu);
            if (d_wave_mask_gpu) cudaFree(d_wave_mask_gpu);
            if (d_selected_integral_gpu) cudaFree(d_selected_integral_gpu);
            delete[] h_interference_matrix_gpu;
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
        double normFactor = static_cast<double>(dataIntegral) / h_phsp_integral;

        // 创建 ROOT 文件
        TFile* rootFile = new TFile(filename.c_str(), "RECREATE");

        TTree* legend = new TTree("legends", "Amplitude Legends");
        legend->Branch("legend", &legends_);
        legend->Fill();
        legend->Write();
        delete legend;


        // 写入干涉矩阵（对称矩阵，需填充上下三角）
        if (h_interference_matrix != nullptr)
        {
            TMatrixD interferenceMatrix(npartials, npartials);
            for (int i = 0; i < npartials; ++i)
            {
                for (int j = i; j < npartials; ++j)
                {
                    int idx = i * npartials - i * (i - 1) / 2 + (j - i);
                    interferenceMatrix[i][j] = h_interference_matrix[idx];
                    interferenceMatrix[j][i] = h_interference_matrix[idx];  // 对称填充下三角
                }
            }

            interferenceMatrix.Write("interference");
        }

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
                CalculateMassHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, masshist, nullptr, temp, events_[i][1], n_particles);
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
                CalculateAngleHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, anglehist, nullptr, temp, events_[i][1], n_particles);
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
                CalculateDalitzHist(device_momenta_list[i] + events_offsets_[i][1] * n_particles, particleToIndex, dalitzhist, nullptr, temp, events_[i][1], n_particles);
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
        //   数据 hdata: 原始计数;  模型 hfit: phsp 权重 d_final_result_vec
        //   (全模型或 waves 子集) × normFactor
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
                        particleToIndex, obshists, nullptr, t1, t2,
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

        // 关闭 ROOT 文件
        rootFile->Close();
        delete rootFile;

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
        if (d_total_integral != nullptr) cudaFree(d_total_integral);
        if (d_interference_matrix != nullptr) cudaFree(d_interference_matrix);
        delete[] h_total_results;
        // h_partial_results 可能未分配，检查是否为空
        // if (h_partial_results != nullptr) delete[] h_partial_results;
    }

    // =====================================================================
    // writeInterfResult: 保存指定波对的逐事件干涉形状。
    //   输出 TTree saved_weight: totalweight / weight_<i>(全部分波) /
    //   interf_<i>_<j>(只含选中对) / 末态粒子四动量。
    //   容量 = len(pairs) × N_phsp（用户按需选择, 不会因全波数×大MC爆显存）。
    //   pairs: [[i,j], ...] 波对(partial 索引, 与 weight_<i> 分支序一致)。
    // =====================================================================
    void writeInterfResult(torch::Tensor params, const std::string& filename,
                           const std::vector<std::vector<int>>& pairs)
    {
        TORCH_CHECK(params.is_cuda(), "params must be on CUDA");
        TORCH_CHECK(!pairs.empty(), "pairs 不能为空");
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
            ev_per_gpu.push_back(extended_vector.to(torch::Device(torch::kCUDA, i)));

        int N_phsp = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu)
            N_phsp += events_[gpu][0];
        TORCH_CHECK(N_phsp > 0, "没有 phsp 事件");

        double* h_total = new double[N_phsp];
        double* h_partial = new double[(size_t)N_phsp * npartials];
        double* h_interf = new double[(size_t)N_phsp * npairs];
        int* d_pair_list = nullptr;
        cudaMalloc(&d_pair_list, npairs * sizeof(int));
        cudaMemcpy(d_pair_list, pair_list.data(), npairs * sizeof(int),
            cudaMemcpyHostToDevice);

        double* d_total_integral = nullptr;
        cudaMalloc(&d_total_integral, sizeof(double));
        int ev_cum = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            int ne = events_[gpu][0];
            if (ne == 0) continue;
            cudaSetDevice(gpu);
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
            double* d_imat = nullptr;
            cudaMalloc(&d_imat, npartials * npartials * sizeof(double));
            cudaMemset(d_imat, 0, npartials * npartials * sizeof(double));
            cudaMemset(d_total_integral, 0, sizeof(double));

            if (!phsp_freed_) {
                computeResults(d_all_amplitudes_[gpu],
                    reinterpret_cast<const ctComplex*>(ev_per_gpu[gpu].data_ptr()),
                    d_total, d_total_integral, d_partial, d_imat, d_intf,
                    nullptr, nullptr, nullptr,
                    d_pair_list, npairs,
                    d_nsl, npartials, ne, n_amplitudes_, n_polar_);
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
                    computeResults(d_chunk[gpu],
                        reinterpret_cast<const ctComplex*>(ev_per_gpu[gpu].data_ptr()),
                        d_ct, d_total_integral, d_cp, d_imat, d_cintf,
                        nullptr, nullptr, nullptr,
                        d_pair_list, npairs,
                        d_nsl, npartials, nch, n_amplitudes_, n_polar_);
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
            if (d_total) cudaFree(d_total);
            if (d_partial) cudaFree(d_partial);
            if (d_intf) cudaFree(d_intf);
            cudaFree(d_imat);
        }
        cudaFree(d_pair_list);
        cudaFree(d_total_integral);
        cudaSetDevice(dev.index());

        double h_phsp_integral = 0.0;
        for (int i = 0; i < N_phsp; ++i) h_phsp_integral += h_total[i];
        double dataIntegral = data_total_weight_ - bkg_integral_;
        double normFactor = h_phsp_integral > 0
            ? dataIntegral / h_phsp_integral : 1.0;

        TFile* rootFile = new TFile(filename.c_str(), "RECREATE");
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
        delete[] h_total;
        delete[] h_partial;
        delete[] h_interf;
        std::cout << "writeInterfResult: " << filename
                  << " (pairs=" << npairs << ", events=" << N_phsp << ")" << std::endl;
        return;
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
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision (double/float 编译)");
        TORCH_CHECK(vector.dim() == 1, "vector must be 1-dimensional");
        TORCH_CHECK(vector.device().index() == primary_dev_,
            "vector 必须位于主 GPU (cuda:" + std::to_string(primary_dev_) + ")，"
            "当前在 cuda:" + std::to_string(vector.device().index()) + "。"
            "跨 GPU 归一化缓冲 (d_phsp_matrix_) 固定在主 GPU 上");

        // 自由耦合 p → 振幅 v（乘法链映射；无耦合矩阵时恒等）
        torch::Tensor extended_vector = freeParamsToAmplitudes(vector);
        const int n = extended_vector.numel();  // = n_amps
        torch::Device dev = vector.device();
        torch::Tensor extended_vector_conj = extended_vector.conj();
        const ctComplex* d_vec = reinterpret_cast<const ctComplex*>(extended_vector.data_ptr());
        const ctComplex* d_vec_conj = reinterpret_cast<const ctComplex*>(extended_vector_conj.data_ptr());

        // 计算phsp量 (与forward pass一致)
        cudaSetDevice(dev.index());
        ctComplex* d_P_vec;
        cudaMalloc(&d_P_vec, n_amplitudes_ * sizeof(ctComplex));
        ctPhspReal* d_pr, * d_pi;
        cudaMalloc(&d_pr, sizeof(ctPhspReal));
        cudaMalloc(&d_pi, sizeof(ctPhspReal));
        // 用d_vec（非共轭）与phspHessianKernel的Pu/tildeP约定保持一致
        computeQuadraticForm(d_phsp_matrix_, d_vec, d_P_vec, d_pr, d_pi, n_amplitudes_);
        ctPhspReal phr, phi;
        cudaMemcpy(&phr, d_pr, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phi, d_pi, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
        double phsp_factor = (double)phr;
        cudaFree(d_pr); cudaFree(d_pi);

        // std::cout << "phsp_factor: " << phsp_factor << std::endl;

        // 分配扩展Hessian (2·n_ext × 2·n_ext, GPU 0)
        const int n_ext = extended_vector.numel();  // = n_amplitudes_
        torch::Tensor hessian = torch::zeros({ 2 * n_ext, 2 * n_ext }, torch::kDouble).to(dev);
        double* d_hessian = hessian.data_ptr<double>();

        // 多GPU: 累加data和bkg的Hessian贡献（每GPU独立分配，P2P累加到GPU 0）
        int hess_sz = (2 * n_ext) * (2 * n_ext);
        int totalDataEvents = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            cudaSetDevice(gpu);
            torch::Tensor vec_gpu = extended_vector.to(torch::Device(torch::kCUDA, gpu));
            const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(vec_gpu.data_ptr());

            // 在当前GPU上分配临时hessian并清零
            double* d_hess_gpu;
            cudaMalloc(&d_hess_gpu, hess_sz * sizeof(double));
            cudaMemset(d_hess_gpu, 0, hess_sz * sizeof(double));

            // --- data ---
            int nData = events_[gpu][1];
            if (nData > 0) {
                totalDataEvents += nData;
                ctComplex* d_amp = d_all_amplitudes_[gpu] + amp_offsets_[gpu][1];
                const double* d_w_data = (gpu < data_weights_.size()) ? data_weights_[gpu] : nullptr;
                (hessianFastPathEnabled()
                    ? computeDataHessianContribFast(d_amp, d_v_gpu, d_w_data, d_hess_gpu, nData, n_polar_, n_ext)
                    : computeDataHessianContrib(d_amp, d_v_gpu, d_w_data, d_hess_gpu, nData, n_polar_, n_ext));
            }

            // --- bkg ---
            int nBkg = (events_[gpu].size() > 2) ? events_[gpu][2] : 0;
            if (nBkg > 0) {
                double* d_w_bkg;
                cudaMalloc(&d_w_bkg, nBkg * sizeof(double));
                if (bkg_weights_[gpu] != nullptr) {
                    std::vector<double> h_w_neg(nBkg);
                    cudaMemcpy(h_w_neg.data(), bkg_weights_[gpu], nBkg * sizeof(double), cudaMemcpyDeviceToHost);
                    for (int i = 0; i < nBkg; ++i) h_w_neg[i] = -h_w_neg[i];
                    cudaMemcpy(d_w_bkg, h_w_neg.data(), nBkg * sizeof(double), cudaMemcpyHostToDevice);
                }
                else {
                    std::vector<double> h_w_neg(nBkg, -1.0);
                    cudaMemcpy(d_w_bkg, h_w_neg.data(), nBkg * sizeof(double), cudaMemcpyHostToDevice);
                }
                ctComplex* d_amp = d_all_amplitudes_[gpu] + amp_offsets_[gpu][2];
                (hessianFastPathEnabled()
                    ? computeDataHessianContribFast(d_amp, d_v_gpu, d_w_bkg, d_hess_gpu, nBkg, n_polar_, n_ext)
                    : computeDataHessianContrib(d_amp, d_v_gpu, d_w_bkg, d_hess_gpu, nBkg, n_polar_, n_ext));
                cudaFree(d_w_bkg);
            }

            // P2P累加到GPU 0的主hessian
            if (gpu == dev.index()) {
                double one = 1.0;
                cublasHandle_t h; cublasCreate(&h);
                cublasDaxpy(h, hess_sz, &one, d_hess_gpu, 1, d_hessian, 1);
                cublasDestroy(h);
            } else {
                double* d_hess_buf;
                cudaSetDevice(dev.index());
                cudaMalloc(&d_hess_buf, hess_sz * sizeof(double));
                cudaMemcpyPeer(d_hess_buf, dev.index(), d_hess_gpu, gpu, hess_sz * sizeof(double));
                double one = 1.0;
                cublasHandle_t h; cublasCreate(&h);
                cublasDaxpy(h, hess_sz, &one, d_hess_buf, 1, d_hessian, 1);
                cublasDestroy(h);
                cudaFree(d_hess_buf);
            }
            cudaFree(d_hess_gpu);
        }
        cudaSetDevice(dev.index());

        // phsp Hessian贡献
        double phsp_weight = data_total_weight_ - bkg_integral_;
        cudaSetDevice(dev.index());
        computePhspHessian(d_phsp_matrix_, d_vec, phsp_factor, phsp_weight, d_hessian, n_ext);

        // std::cout << "n_ext: " << n_ext << ", n: " << n << ", totalDataEvents: " << totalDataEvents << std::endl;

        // 按约束Jacobian投影: H_orig (2n×2n) = J^T * H_ext (2·n_ext×2·n_ext) * J
        // J是diagonal blocks: J[2*eid][2*oid]=real_ratio, J[2*eid+1][2*oid+1]=imag_ratio
        if (n_ext > n) {
            std::vector<int> h_oids, h_eids;
            std::vector<double> h_re, h_im;
            for (const auto& g : params_.constraintGroups()) {
                for (size_t j = 0; j < g.ext_indices.size(); ++j) {
                    if (static_cast<int>(j) == g.origin_idx_in_group) continue;
                    h_oids.push_back(g.origin_id);
                    h_eids.push_back(g.ext_indices[j]);
                    h_re.push_back(static_cast<double>(g.real_ratios[j]));
                    h_im.push_back(static_cast<double>(g.imag_ratios[j]));
                }
            }
            int ncons = h_oids.size();
            int* d_oids, * d_eids;
            double* d_re, * d_im;
            cudaMalloc(&d_oids, ncons * sizeof(int));
            cudaMalloc(&d_eids, ncons * sizeof(int));
            cudaMalloc(&d_re, ncons * sizeof(double));
            cudaMalloc(&d_im, ncons * sizeof(double));
            cudaMemcpy(d_oids, h_oids.data(), ncons * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_eids, h_eids.data(), ncons * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_re, h_re.data(), ncons * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_im, h_im.data(), ncons * sizeof(double), cudaMemcpyHostToDevice);

            // 分配输出Hessian (2n × 2n)，避免就地修改导致读写冲突
            torch::Tensor hessian_reduced = torch::zeros({ 2 * n, 2 * n }, torch::kDouble).to(dev);
            double* d_hessian_reduced = hessian_reduced.data_ptr<double>();

            reduceHessianWithConstraints(d_hessian, d_hessian_reduced, d_oids, d_eids, d_re, d_im, ncons, n, n_ext);

            cudaFree(d_oids); cudaFree(d_eids); cudaFree(d_re); cudaFree(d_im);
            cudaFree(d_P_vec);
            return hessian_reduced;
        }

        // 无约束：直接截取
        torch::Tensor hessian_reduced = hessian.slice(0, 0, 2 * n).slice(1, 0, 2 * n).clone();
        cudaFree(d_P_vec);
        return hessian_reduced;
    }

    // ========== 统一 params 接口 ==========

    // 完整 Hessian: 返回 (2n+P) × (2n+P)，params = [real(v), imag(v), θ] float64
    torch::Tensor getHessian(torch::Tensor params) {
        TORCH_CHECK(params.is_cuda() && params.dtype() == torch::kFloat64,
            "params must be float64 CUDA tensor");
        TORCH_CHECK(params.device().index() == primary_dev_,
            "params 必须位于主 GPU (cuda:" + std::to_string(primary_dev_) + ")，"
            "当前在 cuda:" + std::to_string(params.device().index()) + "。"
            "跨 GPU 归一化缓冲 (d_phsp_matrix_) 固定在主 GPU 上");

        // ---- 1. Setup: determine nv, vector, theta ----
        bool is_coupling = params_.hasCouplingMatrix();
        int ncf = 0;
        int nv, nt, n2, total;
        torch::Tensor vector, theta;

        if (is_coupling) {
            const auto& cm = params_.couplingMatrix();
            ncf = cm.n_free;
            nv  = cm.n_amps;
            nt  = params_.nFreeTheta();
            auto dev = params.device();
            auto v_ext = torch::empty({nv},
                torch::TensorOptions().dtype(TORCH_COMPLEX).device(dev));
            params_.applyCouplingMatrix(params.data_ptr<double>(),
                reinterpret_cast<ctComplex*>(v_ext.data_ptr()));
            vector = v_ext;
            theta = (nt > 0) ? params.slice(0, 2*ncf, params.size(0))
                             : torch::Tensor();
        } else {
            nv = params_.nFreeVector();
            nt = params_.nFreeTheta();
            std::tie(vector, theta) = params_.splitParams(params);
        }

        n2 = 2 * nv;
        total = n2 + nt;
        auto dev = params.device();
        torch::Tensor hessian = torch::zeros({total, total}, torch::kFloat64).to(dev);

        // 打印hessian全部元素
        // std::cout << "Hessian elements " << __LINE__ << ": \n" << hessian << std::endl;

        // ---- 2. recompute amplitudes + rebuild phsp matrix ----
        if (nt > 0 && theta.numel() > 0) {
            amp_calc_.reComputeAmps(d_all_amplitudes_,
                reinterpret_cast<const double*>(theta.data_ptr()),
                n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
                dev.index());
        }

        bool hprof = getenv("CTPWA_PROF") != nullptr;
        auto hT0 = std::chrono::high_resolution_clock::now();  // reComputeAmps 完成
        auto hT1 = hT0, hT2 = hT0, hT3 = hT0, hT4 = hT0, hT5 = hT0;

        // Update d_phsp_matrix_ to reflect current amplitudes
        // phsp 流式模式下不重建：质量/宽度固定 → 矩阵在构造期已就绪且不变
        //（常驻表不含 phsp，重建会读错数据）
        if (!phsp_freed_) {
        {
            int primary_dev = dev.index();
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

            cudaSetDevice(primary_dev);
            cudaMemset(d_phsp_matrix_, 0, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
                int nPhsp = events_[gpu][0];
                if (nPhsp == 0) continue;
                cudaSetDevice(gpu);

                int nPhsp_total = nPhsp * n_polar_ * n_amplitudes_;
                ctComplex* d_phsp_scaled;
                cudaMalloc(&d_phsp_scaled, nPhsp_total * sizeof(ctComplex));
                cudaMemcpy(d_phsp_scaled, d_all_amplitudes_[gpu],
                           nPhsp_total * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                const double* d_w = (gpu < phsp_weights_.size()) ? phsp_weights_[gpu] : nullptr;
                int grid = (nPhsp_total + 255) / 256;
                scalePhspAmpsKernel<<<grid, 256>>>(d_phsp_scaled, d_w,
                    nPhsp, n_polar_, n_amplitudes_, 1.0 / W_total, 0);

                ctComplex* d_phsp_gpu;
                cudaMalloc(&d_phsp_gpu, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                cublasHandle_t h;
                cublasCreate(&h);
                ctComplex alpha = ctMake(1.0f, 0.0f);
                ctComplex beta  = ctMake(0.0f, 0.0f);
                CUBLAS_CGEMM(h, CUBLAS_OP_N, CUBLAS_OP_C, n_amplitudes_, n_amplitudes_, nPhsp * n_polar_,
                    &alpha, d_phsp_scaled, n_amplitudes_, d_phsp_scaled, n_amplitudes_,
                    &beta, d_phsp_gpu, n_amplitudes_);
                cublasDestroy(h);
                cudaFree(d_phsp_scaled);

                if (gpu == (size_t)primary_dev) {
                    ctComplex one = ctMake(1.0f, 0.0f);
                    axpyComplex(d_phsp_matrix_, d_phsp_gpu, one, n_amplitudes_ * n_amplitudes_);
                    cudaFree(d_phsp_gpu);
                } else {
                    cudaSetDevice(primary_dev);
                    ctComplex* d_temp;
                    cudaMalloc(&d_temp, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                    cudaMemcpyPeer(d_temp, primary_dev, d_phsp_gpu, gpu,
                        n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                    cudaSetDevice(gpu);
                    cudaFree(d_phsp_gpu);
                    cudaSetDevice(primary_dev);
                    ctComplex one = ctMake(1.0f, 0.0f);
                    axpyComplex(d_phsp_matrix_, d_temp, one, n_amplitudes_ * n_amplitudes_);
                    cudaFree(d_temp);
                }
            }
        }
        }  // if (!phsp_freed_) (getHessian phsp 矩阵重建)

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        hT1 = std::chrono::high_resolution_clock::now();  // phsp 矩阵重建完成

        // ---- 3. vv block [0:n2, 0:n2] ----
        if (n2 > 0) {
            torch::Tensor extended_vector = is_coupling ? vector : params_.extendVector(vector, dev);
            const ctComplex* d_vec = reinterpret_cast<const ctComplex*>(extended_vector.data_ptr());
            const int n_ext = extended_vector.numel();

            // phsp factor
            cudaSetDevice(dev.index());
            ctComplex* d_P_vec;
            cudaMalloc(&d_P_vec, n_amplitudes_ * sizeof(ctComplex));
            ctPhspReal *d_pr, *d_pi;
            cudaMalloc(&d_pr, sizeof(ctPhspReal));
            cudaMalloc(&d_pi, sizeof(ctPhspReal));
            computeQuadraticForm(d_phsp_matrix_, d_vec, d_P_vec, d_pr, d_pi, n_amplitudes_);
            ctPhspReal phr, phi;
            cudaMemcpy(&phr, d_pr, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
            cudaMemcpy(&phi, d_pi, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
            double phsp_factor = (double)phr;
            cudaFree(d_pr); cudaFree(d_pi);
            cudaFree(d_P_vec);

            // Extended Hessian (2·n_ext × 2·n_ext)
            torch::Tensor hess_ext = torch::zeros({2 * n_ext, 2 * n_ext}, torch::kDouble).to(dev);
            double* d_hess_ext = hess_ext.data_ptr<double>();
            int hess_sz = (2 * n_ext) * (2 * n_ext);
            int totalDataEvents = 0;

            // Multi-GPU: data + bkg Hessian contributions
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
                cudaSetDevice(gpu);
                torch::Tensor vec_gpu = extended_vector.to(torch::Device(torch::kCUDA, gpu));
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(vec_gpu.data_ptr());

                double* d_hess_gpu;
                cudaMalloc(&d_hess_gpu, hess_sz * sizeof(double));
                cudaMemset(d_hess_gpu, 0, hess_sz * sizeof(double));

                int nData = events_[gpu][1];
                if (nData > 0) {
                    totalDataEvents += nData;
                    ctComplex* d_amp = d_all_amplitudes_[gpu] + amp_offsets_[gpu][1];
                    const double* d_w_data = (gpu < data_weights_.size()) ? data_weights_[gpu] : nullptr;
                    (hessianFastPathEnabled()
                        ? computeDataHessianContribFast(d_amp, d_v_gpu, d_w_data, d_hess_gpu, nData, n_polar_, n_ext)
                        : computeDataHessianContrib(d_amp, d_v_gpu, d_w_data, d_hess_gpu, nData, n_polar_, n_ext));
                }

                int nBkg = (events_[gpu].size() > 2) ? events_[gpu][2] : 0;
                if (nBkg > 0) {
                    double* d_w_bkg;
                    cudaMalloc(&d_w_bkg, nBkg * sizeof(double));
                    if (bkg_weights_[gpu] != nullptr) {
                        std::vector<double> h_w_neg(nBkg);
                        cudaMemcpy(h_w_neg.data(), bkg_weights_[gpu], nBkg * sizeof(double), cudaMemcpyDeviceToHost);
                        for (int i = 0; i < nBkg; ++i) h_w_neg[i] = -h_w_neg[i];
                        cudaMemcpy(d_w_bkg, h_w_neg.data(), nBkg * sizeof(double), cudaMemcpyHostToDevice);
                    } else {
                        std::vector<double> h_w_neg(nBkg, -1.0);
                        cudaMemcpy(d_w_bkg, h_w_neg.data(), nBkg * sizeof(double), cudaMemcpyHostToDevice);
                    }
                    ctComplex* d_amp = d_all_amplitudes_[gpu] + amp_offsets_[gpu][2];
                    (hessianFastPathEnabled()
                        ? computeDataHessianContribFast(d_amp, d_v_gpu, d_w_bkg, d_hess_gpu, nBkg, n_polar_, n_ext)
                        : computeDataHessianContrib(d_amp, d_v_gpu, d_w_bkg, d_hess_gpu, nBkg, n_polar_, n_ext));
                    cudaFree(d_w_bkg);
                }

                if (gpu == dev.index()) {
                    double one = 1.0;
                    cublasHandle_t h; cublasCreate(&h);
                    cublasDaxpy(h, hess_sz, &one, d_hess_gpu, 1, d_hess_ext, 1);
                    cublasDestroy(h);
                } else {
                    double* d_hess_buf;
                    cudaSetDevice(dev.index());
                    cudaMalloc(&d_hess_buf, hess_sz * sizeof(double));
                    cudaMemcpyPeer(d_hess_buf, dev.index(), d_hess_gpu, gpu, hess_sz * sizeof(double));
                    double one = 1.0;
                    cublasHandle_t h; cublasCreate(&h);
                    cublasDaxpy(h, hess_sz, &one, d_hess_buf, 1, d_hess_ext, 1);
                    cublasDestroy(h);
                    cudaFree(d_hess_buf);
                }
                cudaFree(d_hess_gpu);
            }
            cudaSetDevice(dev.index());

            // phsp Hessian contribution
            double phsp_weight = data_total_weight_ - bkg_integral_;
            computePhspHessian(d_phsp_matrix_, d_vec, phsp_factor, phsp_weight, d_hess_ext, n_ext);

            // Constraint projection (identity in coupling mode since nv=na=n_ext)
            if (is_coupling || n_ext == nv) {
                // Copy top-left n2×n2 (interleaved format from d_hess_ext)
                for (int i = 0; i < n2; ++i)
                    cudaMemcpy(hessian.data_ptr<double>() + i * total,
                               d_hess_ext + i * (2 * n_ext),
                               n2 * sizeof(double), cudaMemcpyDeviceToDevice);
            } else {
                // Legacy constraint projection
                int ncons = 0;
                for (const auto& g : params_.constraintGroups())
                    for (size_t c = 0; c < g.ext_indices.size(); ++c)
                        if ((int)c != g.origin_idx_in_group) ++ncons;
                std::vector<int> h_oids, h_eids;
                std::vector<double> h_re, h_im;
                for (const auto& g : params_.constraintGroups())
                    for (size_t c = 0; c < g.ext_indices.size(); ++c) {
                        if ((int)c == g.origin_idx_in_group) continue;
                        h_oids.push_back(g.origin_id);
                        h_eids.push_back(g.ext_indices[c]);
                        h_re.push_back((double)g.real_ratios[c]);
                        h_im.push_back((double)g.imag_ratios[c]);
                    }
                int *d_oids, *d_eids;
                double *d_re, *d_im;
                cudaMalloc(&d_oids, ncons*sizeof(int)); cudaMemcpy(d_oids, h_oids.data(), ncons*sizeof(int), cudaMemcpyHostToDevice);
                cudaMalloc(&d_eids, ncons*sizeof(int)); cudaMemcpy(d_eids, h_eids.data(), ncons*sizeof(int), cudaMemcpyHostToDevice);
                cudaMalloc(&d_re, ncons*sizeof(double)); cudaMemcpy(d_re, h_re.data(), ncons*sizeof(double), cudaMemcpyHostToDevice);
                cudaMalloc(&d_im, ncons*sizeof(double)); cudaMemcpy(d_im, h_im.data(), ncons*sizeof(double), cudaMemcpyHostToDevice);
                torch::Tensor hess_reduced = torch::zeros({n2, n2}, torch::kDouble).to(dev);
                reduceHessianWithConstraints(d_hess_ext, hess_reduced.data_ptr<double>(), d_oids, d_eids, d_re, d_im, ncons, nv, n_ext);
                cudaFree(d_oids); cudaFree(d_eids); cudaFree(d_re); cudaFree(d_im);
                hessian.slice(0, 0, n2).slice(1, 0, n2).copy_(hess_reduced);
            }
        }

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        // Reorder vv block to grouped [Re0..Re_n, Im0..Im_n] BEFORE vθ assembly
        // reorderVVBlockInterleavedToGrouped(hessian.data_ptr<double>(), nv, total);

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        hT2 = std::chrono::high_resolution_clock::now();  // vv block 完成

        // ---- 4. vθ/θθ block [n2:total, n2:total] ----
        if (nt > 0 && theta.numel() > 0) {
            int P = nt;
            int n_gpu = static_cast<int>(d_all_amplitudes_.size());
            int primary_dev = dev.index();

            torch::Tensor extended_v = is_coupling ? vector : params_.extendVector(vector, dev);
            int n_ext = extended_v.numel();

            // Build per-GPU interleaved v arrays (avoid cross-device access)
            std::vector<torch::Tensor> v_il_tensors;  // keep tensors alive
            std::vector<ctComplex*> d_v_per_gpu(n_gpu, nullptr);
            for (int gpu = 0; gpu < n_gpu; ++gpu) {
                cudaSetDevice(gpu);
                torch::Tensor v_gpu = extended_v.to(torch::Device(torch::kCUDA, gpu));
                amp_calc_.computeEffectiveCoupling(
                    reinterpret_cast<const ctComplex*>(v_gpu.data_ptr()), n_ext);
                // Build interleaved per-GPU（dtype 跟随编译精度：float32/float64）
                torch::Tensor vr = torch::real(v_gpu).to(TORCH_FLOAT);
                torch::Tensor vi = torch::imag(v_gpu).to(TORCH_FLOAT);
                torch::Tensor vil = torch::empty({n_ext * 2},
                    torch::TensorOptions().dtype(TORCH_FLOAT).device(torch::Device(torch::kCUDA, gpu)));
                vil.slice(0, 0, 2 * n_ext, 2).copy_(vr);
                vil.slice(0, 1, 2 * n_ext, 2).copy_(vi);
                d_v_per_gpu[gpu] = const_cast<ctComplex*>(reinterpret_cast<const ctComplex*>(vil.data_ptr()));
                v_il_tensors.push_back(vil);
            }
            cudaSetDevice(primary_dev);

            double* d_hess;
            cudaMalloc(&d_hess, P * P * sizeof(double));
            cudaMemset(d_hess, 0, P * P * sizeof(double));
            double* d_mixed = nullptr;
            cudaMalloc(&d_mixed, 2 * n_amplitudes_ * P * sizeof(double));
            cudaMemset(d_mixed, 0, 2 * n_amplitudes_ * P * sizeof(double));
            double* d_phsp_mixed_sum = nullptr;
            double* d_phsp_mixed_t3 = nullptr;
            cudaMalloc(&d_phsp_mixed_sum, 2 * n_amplitudes_ * P * sizeof(double));
            cudaMalloc(&d_phsp_mixed_t3, 2 * n_amplitudes_ * sizeof(double));
            cudaMemset(d_phsp_mixed_sum, 0, 2 * n_amplitudes_ * P * sizeof(double));
            cudaMemset(d_phsp_mixed_t3, 0, 2 * n_amplitudes_ * sizeof(double));

            // Data
            {
                std::vector<int> n_data_ev(n_gpu, 0), data_off(n_gpu, 0);
                for (int g = 0; g < n_gpu; ++g) { n_data_ev[g] = events_[g][1]; data_off[g] = events_[g][0]; }
                auto uT0 = std::chrono::high_resolution_clock::now();
                amp_calc_.computeUnifiedHessian(n_data_ev, d_hess, P, data_off, 1.0,
                    d_v_per_gpu, d_all_amplitudes_, n_amplitudes_, data_weights_,
                    nullptr, nullptr, nullptr, d_mixed, nullptr, nullptr);
                if (hprof) {
                    auto uT1 = std::chrono::high_resolution_clock::now();
                    printf("[PROF] UH-call data: %.2f ms\n",
                        std::chrono::duration<double, std::milli>(uT1 - uT0).count());
                    fflush(stdout);
                }
            }

            // Bkg
            {
                bool has_bkg = false;
                for (int g = 0; g < n_gpu; ++g) { if (events_[g].size() > 2 && events_[g][2] > 0) { has_bkg = true; break; } }
                std::vector<int> n_bkg_ev(n_gpu, 0), bkg_off(n_gpu, 0);
                if (has_bkg) {
                    for (int g = 0; g < n_gpu; ++g) { n_bkg_ev[g] = (events_[g].size() > 2) ? events_[g][2] : 0; bkg_off[g] = events_[g][0] + events_[g][1]; }
                }
                std::vector<double*> neg_bkg_weights;
                for (int g = 0; g < n_gpu; ++g) {
                    double* d_neg = nullptr;
                    if (g < (int)bkg_weights_.size() && bkg_weights_[g] != nullptr && n_bkg_ev[g] > 0) {
                        cudaSetDevice(g); cudaMalloc(&d_neg, n_bkg_ev[g] * sizeof(double));
                        int grid = (n_bkg_ev[g] + 255) / 256;
                        negateWeightsKernel<<<grid, 256>>>(d_neg, bkg_weights_[g], n_bkg_ev[g]);
                    }
                    neg_bkg_weights.push_back(d_neg);
                }
                cudaSetDevice(primary_dev);
                auto uT0 = std::chrono::high_resolution_clock::now();
                amp_calc_.computeUnifiedHessian(n_bkg_ev, d_hess, P, bkg_off, -1.0,
                    d_v_per_gpu, d_all_amplitudes_, n_amplitudes_, neg_bkg_weights,
                    nullptr, nullptr, nullptr, d_mixed, nullptr, nullptr);
                if (hprof) {
                    auto uT1 = std::chrono::high_resolution_clock::now();
                    printf("[PROF] UH-call bkg: %.2f ms\n",
                        std::chrono::duration<double, std::milli>(uT1 - uT0).count());
                    fflush(stdout);
                }
                for (int g = 0; g < n_gpu; ++g) { if (neg_bkg_weights[g]) { cudaSetDevice(g); cudaFree(neg_bkg_weights[g]); } }
                cudaSetDevice(primary_dev);
            }

            // Phsp
            double phsp_pf = 1.0, phsp_A = 0.0; int phsp_np = 1; double* phsp_h_pg = nullptr;
            {
                double *d_pI, *d_pg, *d_phA;
                cudaMalloc(&d_pI, sizeof(double)); cudaMalloc(&d_pg, P*sizeof(double)); cudaMalloc(&d_phA, P*P*sizeof(double));
                cudaMemset(d_pI, 0, sizeof(double)); cudaMemset(d_pg, 0, P*sizeof(double)); cudaMemset(d_phA, 0, P*P*sizeof(double));
                std::vector<int> n_phsp_ev(n_gpu,0), phsp_off(n_gpu,0);
                for (int g=0; g<n_gpu; ++g) { n_phsp_ev[g]=events_[g][0]; phsp_off[g]=0; }
                auto uT0 = std::chrono::high_resolution_clock::now();
                amp_calc_.computeUnifiedHessian(n_phsp_ev, d_hess, P, phsp_off, 0.0,
                    d_v_per_gpu, d_all_amplitudes_, n_amplitudes_, {},
                    d_pI, d_pg, d_phA, d_mixed, d_phsp_mixed_sum, d_phsp_mixed_t3);
                if (hprof) {
                    auto uT1 = std::chrono::high_resolution_clock::now();
                    printf("[PROF] UH-call phsp: %.2f ms\n",
                        std::chrono::duration<double, std::milli>(uT1 - uT0).count());
                    fflush(stdout);
                }
                double h_pI; cudaMemcpy(&h_pI, d_pI, sizeof(double), cudaMemcpyDeviceToHost);
                phsp_h_pg = new double[P];
                cudaMemcpy(phsp_h_pg, d_pg, P*sizeof(double), cudaMemcpyDeviceToHost);
                std::vector<double> h_ph(P*P), h_dh(P*P);
                cudaMemcpy(h_ph.data(), d_phA, P*P*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_dh.data(), d_hess, P*P*sizeof(double), cudaMemcpyDeviceToHost);
                cudaFree(d_pI); cudaFree(d_pg); cudaFree(d_phA);
                double A = data_total_weight_ - bkg_integral_;
                int npt=0; for(int g=0;g<n_gpu;++g) npt+=events_[g][0];
                // 所有块被跳过时（无自由参数块）h_pI=0 → pf=0 → c1/c2=inf，
                // 0·inf = NaN；此时 h_ph/pg 全 0，跳过该修正即可
                double pf = (npt > 0) ? h_pI/npt : 0.0;
                if (pf > 0.0) {
                    double c1=A/(pf*npt), c2=-A/(pf*pf*npt*npt);
                    for(int j=0;j<P;++j) for(int k=0;k<P;++k) h_dh[j*P+k] += c1*h_ph[j*P+k] + c2*phsp_h_pg[j]*phsp_h_pg[k];
                }
                cudaMemcpy(d_hess, h_dh.data(), P*P*sizeof(double), cudaMemcpyHostToDevice);
                phsp_pf=pf; phsp_A=A; phsp_np=npt;
            }

            // Mixed Hessian post-processing + constraint projection
            {
                std::vector<double> h_mixed(2*n_amplitudes_*P);
                cudaMemcpy(h_mixed.data(), d_mixed, 2*n_amplitudes_*P*sizeof(double), cudaMemcpyDeviceToHost);
                cudaFree(d_mixed);
                if (phsp_h_pg) {
                    std::vector<double> h_sum(2*n_amplitudes_*P), h_t3(2*n_amplitudes_);
                    cudaMemcpy(h_sum.data(), d_phsp_mixed_sum, 2*n_amplitudes_*P*sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_t3.data(), d_phsp_mixed_t3, 2*n_amplitudes_*sizeof(double), cudaMemcpyDeviceToHost);
                    cudaFree(d_phsp_mixed_sum); cudaFree(d_phsp_mixed_t3);
                    if (phsp_pf > 0.0) {
                        double c1m=2.0*phsp_A/(phsp_pf*phsp_np), c2m=2.0*phsp_A/(phsp_pf*phsp_pf*phsp_np*phsp_np);
                        for(int a=0;a<n_amplitudes_;++a) for(int j=0;j<P;++j) {
                            h_mixed[a*P+j] += c1m*h_sum[a*P+j] + c2m*h_t3[a]*phsp_h_pg[j];
                            h_mixed[(n_amplitudes_+a)*P+j] += -c1m*h_sum[(n_amplitudes_+a)*P+j] - c2m*h_t3[n_amplitudes_+a]*phsp_h_pg[j];
                        }
                    }
                    delete[] phsp_h_pg;
                }
                // Constraint projection on mixed Hessian
                if (!is_coupling) {
                    for (const auto& g : params_.constraintGroups()) {
                        int oid=g.origin_id;
                        for (size_t c=0;c<g.ext_indices.size();++c) {
                            if((int)c==g.origin_idx_in_group) continue;
                            int eid=g.ext_indices[c];
                            double rr=(double)g.real_ratios[c], ir=(double)g.imag_ratios[c];
                            for(int j=0;j<P;++j) {
                                h_mixed[oid*P+j] += rr*h_mixed[eid*P+j] + ir*h_mixed[(n_amplitudes_+eid)*P+j];
                                h_mixed[(n_amplitudes_+oid)*P+j] += -ir*h_mixed[eid*P+j] + rr*h_mixed[(n_amplitudes_+eid)*P+j];
                            }
                        }
                    }
                }
                std::vector<double> h_proj(2*nv*P, 0.0);
                for(int i=0;i<nv;++i) for(int j=0;j<P;++j) {
                    h_proj[i*P+j] = h_mixed[i*P+j];
                    h_proj[(nv+i)*P+j] = h_mixed[(n_amplitudes_+i)*P+j];
                }
                // Row-by-row copy to avoid sliced-tensor stride issues
                double* d_hess = hessian.data_ptr<double>();
                for (int i = 0; i < n2; ++i)
                    cudaMemcpy(d_hess + i * total + n2, h_proj.data() + i * P,
                               P * sizeof(double), cudaMemcpyHostToDevice);
                // θv block = vθ^T  (copy full block)
                std::vector<double> h_theta_v(n2 * P);
                for (int i = 0; i < n2; ++i)
                    for (int j = 0; j < P; ++j)
                        h_theta_v[j * n2 + i] = h_proj[i * P + j];
                for (int j = 0; j < P; ++j)
                    cudaMemcpy(d_hess + (n2 + j) * total, h_theta_v.data() + j * n2,
                               n2 * sizeof(double), cudaMemcpyHostToDevice);
        }

            // Copy theta-theta result
            torch::Tensor res_hess = torch::empty({P, P}, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            cudaMemcpy(res_hess.data_ptr(), d_hess, P*P*sizeof(double), cudaMemcpyDeviceToDevice);
            cudaFree(d_hess);
            hessian.slice(0,n2,total).slice(1,n2,total).copy_(res_hess);
        }
        hT3 = std::chrono::high_resolution_clock::now();  // vθ/θθ 段完成

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        // std::cout << "Coupling: " << is_coupling << std::endl;

        // ---- 5. Coupling transform ----
        if (is_coupling) {
            // Precompute Jacobian for Hessian transform (J^T · H_ext · J + Σ g_v · ∇²v)
            params_.precomputeJacobian(params.data_ptr<double>());

            // ---- compute gradient g_v = ∂NLL/∂v in extended amplitude space ----
            // Uses the same computeFactorNLL as the NLL forward pass for data/bkg.
            // Phsp: g_phsp = N_eff / phsp_factor * (M × v)
            double* d_g_v = nullptr;
            {
                int n_gpu = static_cast<int>(d_all_amplitudes_.size());
                int primary_dev = dev.index();
                const auto& cm = params_.couplingMatrix();
                int na = cm.n_amps;

                // Build extended vector v_ext
                auto v_ext_tensor = torch::empty({na}, torch::TensorOptions().dtype(TORCH_COMPLEX).device(dev));
                params_.applyCouplingMatrix(params.data_ptr<double>(),
                    reinterpret_cast<ctComplex*>(v_ext_tensor.data_ptr()));
                const ctComplex* d_v_ext = reinterpret_cast<const ctComplex*>(v_ext_tensor.data_ptr());

                // Allocate g_v on primary GPU
                cudaMalloc(&d_g_v, 2 * na * sizeof(double));
                cudaMemset(d_g_v, 0, 2 * na * sizeof(double));

                // --- Data gradient: g = ∂(-log I)/∂conj(v) via computeFactorNLL ---
                for (int gpu = 0; gpu < n_gpu; ++gpu) {
                    int nEv = events_[gpu][1];
                    if (nEv == 0) continue;
                    cudaSetDevice(gpu);
                    ctComplex* d_g_gpu;
                    cudaMalloc(&d_g_gpu, na * sizeof(ctComplex));
                    // d_v_ext 在主 GPU，需每 GPU 副本——跨设备传入 computeFactorNLL
                    // 的 cublas gemv 会非法访问（实测 2×A100 getHessian 必现）
                    ctComplex* d_v_gpu;
                    cudaMalloc(&d_v_gpu, na * sizeof(ctComplex));
                    cudaMemcpyPeer(d_v_gpu, gpu, d_v_ext, primary_dev, na * sizeof(ctComplex));
                    const double* d_w_data = (gpu < (int)data_weights_.size()) ? data_weights_[gpu] : nullptr;
                    computeFactorNLL(d_all_amplitudes_[gpu] + amp_offsets_[gpu][1],
                        d_v_gpu, d_g_gpu, nEv, n_polar_, na, d_w_data, nullptr);
                    cudaFree(d_v_gpu);
                    // Convert ctComplex [Re,Im,Re,Im,...] → double [Re...,Im...] layout
                    std::vector<ctComplex> h_g_tmp(na);
                    cudaMemcpy(h_g_tmp.data(), d_g_gpu, na * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                    std::vector<double> h_g(2 * na);
                    for (int a = 0; a < na; ++a) {
                        h_g[a]      = (double)h_g_tmp[a].x;
                        h_g[na + a] = (double)h_g_tmp[a].y;
                    }
                    cudaSetDevice(primary_dev);
                    double* d_tmp; cudaMalloc(&d_tmp, 2 * na * sizeof(double));
                    cudaMemcpy(d_tmp, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                    int grid = (2 * na + 255) / 256;
                    daxpy_kernel<<<grid, 256>>>(d_g_v, d_tmp, 1.0, 2 * na);
                    cudaFree(d_tmp); cudaFree(d_g_gpu);
                }

                // --- Bkg gradient: contribution to loss = data_nll - bkg_nll ---
                // computeFactorNLL with weight=-bkg_weight gives gradient for -(-log(I)*bkg_weight)
                // which adds to data gradient correctly in the loss formula.
                for (int gpu = 0; gpu < n_gpu; ++gpu) {
                    if (events_[gpu].size() <= 2 || events_[gpu][2] == 0) continue;
                    int nEv = events_[gpu][2];
                    cudaSetDevice(gpu);
                    ctComplex* d_g_gpu;
                    cudaMalloc(&d_g_gpu, na * sizeof(ctComplex));
                    // Negate bkg weights (same as in NLL forward)
                    const double* d_w = (gpu < (int)bkg_weights_.size() && bkg_weights_[gpu]) ? nullptr : nullptr;
                    // Use negated bkg weights
                    double* d_neg_w = nullptr;
                    if (gpu < (int)bkg_weights_.size() && bkg_weights_[gpu] != nullptr) {
                        cudaMalloc(&d_neg_w, nEv * sizeof(double));
                        int grid = (nEv + 255) / 256;
                        negateWeightsKernel<<<grid, 256>>>(d_neg_w, bkg_weights_[gpu], nEv);
                    }
                    // 同 data：v 每 GPU 副本（d_v_ext 在主 GPU，跨设备 gemv 非法访问）
                    ctComplex* d_v_gpu;
                    cudaMalloc(&d_v_gpu, na * sizeof(ctComplex));
                    cudaMemcpyPeer(d_v_gpu, gpu, d_v_ext, primary_dev, na * sizeof(ctComplex));
                    computeFactorNLL(d_all_amplitudes_[gpu] + amp_offsets_[gpu][2],
                        d_v_gpu, d_g_gpu, nEv, n_polar_, na, d_neg_w, nullptr);
                    cudaFree(d_v_gpu);
                    if (d_neg_w) cudaFree(d_neg_w);
                    // Convert ctComplex [Re,Im,Re,Im,...] → double [Re...,Im...] layout
                    std::vector<ctComplex> h_g_tmp(na);
                    cudaMemcpy(h_g_tmp.data(), d_g_gpu, na * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                    std::vector<double> h_g(2 * na);
                    for (int a = 0; a < na; ++a) {
                        h_g[a]      = (double)h_g_tmp[a].x;
                        h_g[na + a] = (double)h_g_tmp[a].y;
                    }
                    cudaSetDevice(primary_dev);
                    double* d_tmp; cudaMalloc(&d_tmp, 2 * na * sizeof(double));
                    cudaMemcpy(d_tmp, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                    int grid = (2 * na + 255) / 256;
                    daxpy_kernel<<<grid, 256>>>(d_g_v, d_tmp, -1.0, 2 * na);  // subtract: loss = data - bkg
                    cudaFree(d_tmp); cudaFree(d_g_gpu);
                }

                // --- Phsp gradient: g_phsp = N_eff / phsp_factor * (M × v) ---
                {
                    double N_eff = data_total_weight_ - bkg_integral_;
                    cudaSetDevice(primary_dev);

                    // Compute phsp_factor and P_vec = M × v
                    ctComplex* d_P_vec; cudaMalloc(&d_P_vec, na * sizeof(ctComplex));
                    {
                        cublasHandle_t h; cublasCreate(&h);
                        ctComplex alpha = ctMake(1,0), beta = ctMake(0,0);
                        CUBLAS_CGEMV(h, CUBLAS_OP_N, na, na, &alpha, d_phsp_matrix_, na,
                                    d_v_ext, 1, &beta, d_P_vec, 1);
                        cublasDestroy(h);
                    }
                    ctPhspReal phr;
                    {
                        auto v_conj = v_ext_tensor.conj();
                        const ctComplex* d_vc = reinterpret_cast<const ctComplex*>(v_conj.data_ptr());
                        ctPhspReal *d_pr, *d_pi;
                        cudaMalloc(&d_pr, sizeof(ctPhspReal)); cudaMalloc(&d_pi, sizeof(ctPhspReal));
                        computeQuadraticForm(d_phsp_matrix_, d_vc, d_P_vec, d_pr, d_pi, na);
                        cudaMemcpy(&phr, d_pr, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
                        cudaFree(d_pr); cudaFree(d_pi);
                    }
                    double phsp_factor = (double)phr;

                    if (phsp_factor > 1e-30) {
                        double scale = N_eff / phsp_factor;
                        std::vector<ctComplex> h_P(na);
                        cudaMemcpy(h_P.data(), d_P_vec, na * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                        std::vector<double> h_g(2 * na, 0.0);
                        for (int a = 0; a < na; ++a) {
                            h_g[a]      = scale * (double)h_P[a].x;
                            h_g[na + a] = scale * (double)h_P[a].y;
                        }
                        double* d_tmp; cudaMalloc(&d_tmp, 2 * na * sizeof(double));
                        cudaMemcpy(d_tmp, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                        int grid = (2 * na + 255) / 256;
                        daxpy_kernel<<<grid, 256>>>(d_g_v, d_tmp, 1.0, 2 * na);  // accumulate!
                        cudaFree(d_tmp);
                    }
                    cudaFree(d_P_vec);
                }

                // Convert d_g_v from ∂NLL/∂conj(v) (Wirtinger) to real gradient:
                //   ∂NLL/∂Re(v_a) = 2 * Re(∂NLL/∂conj(v_a))
                //   ∂NLL/∂Im(v_a) = 2 * Im(∂NLL/∂conj(v_a))
                // Both are simply ×2 (no sign flip — the conj flip cancels with
                // the minus in ∂/∂Im = -2 Im(∂/∂v)).
                {
                    std::vector<double> h_g(2 * na);
                    cudaMemcpy(h_g.data(), d_g_v, 2 * na * sizeof(double), cudaMemcpyDeviceToHost);
                    for (int a = 0; a < 2 * na; ++a) h_g[a] *= 2.0;
                    cudaMemcpy(d_g_v, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                }
            }

            int n_fit = 2 * ncf + nt;
            auto hess_fit = torch::zeros({n_fit, n_fit}, torch::kFloat64).to(dev);
            params_.transformExtendedHessian(
                hessian.data_ptr<double>(),
                params.data_ptr<double>(), d_g_v,
                hess_fit.data_ptr<double>(),
                nv, ncf, nt);
            cudaFree(d_g_v);
            // Constraints.gauss_constr: 罚项 Hessian 对角 1/σ²（θ 块起点 = 2*ncf；
            // transform 对 θθ 块是直接拷贝，变换前后追加等价）
            if (params_.nGaussConstr() > 0)
                params_.addGaussHessianDiag(hess_fit, 2 * ncf);

            if (hprof) {
                auto hTend = std::chrono::high_resolution_clock::now();
                auto hms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                printf("[PROF] H.reAmp: %.2f | H.phspM: %.2f | H.vv: %.2f | H.vth+thth+cpl: %.2f | H.total: %.2f\n",
                    hms(hT0, hT1), hms(hT1, hT2), hms(hT2, hT3), hms(hT3, hTend), hms(hT0, hTend));
                fflush(stdout);
            }
            return hess_fit;
        }

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        // Constraints.gauss_constr: 罚项 Hessian 对角 1/σ²（θ 块起点 = n2）
        if (params_.nGaussConstr() > 0)
            params_.addGaussHessianDiag(hessian, n2);

        // vv / θθ 对角块强制对称化：两块的数学形式必须对称（Gram 矩阵 / ∂²/∂θ²），
        // 但双卡路径是"每 GPU 先累积再归并"，与单卡顺序累积的浮点路径不同，
        // 会引入 ~1e-7~1e-6 的不对称（实测 9.9e-7，压测试容差 1e-6）。
        // (H+H^T)/2 消除浮点噪声，不改变物理。vθ/θv 块本来就是转置复制，无需处理。
        {
            if (n2 > 0) {
                std::vector<double> h_vv((size_t)n2 * n2);
                cudaMemcpy(h_vv.data(), hessian.data_ptr<double>(),
                           (size_t)n2 * n2 * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < n2; ++i)
                    for (int j = i + 1; j < n2; ++j) {
                        double m = 0.5 * (h_vv[(size_t)i * n2 + j] + h_vv[(size_t)j * n2 + i]);
                        h_vv[(size_t)i * n2 + j] = m; h_vv[(size_t)j * n2 + i] = m;
                    }
                cudaMemcpy(hessian.data_ptr<double>(), h_vv.data(),
                           (size_t)n2 * n2 * sizeof(double), cudaMemcpyHostToDevice);
            }
            int nt = total - n2;
            if (nt > 0) {
                std::vector<double> h_th((size_t)nt * nt);
                for (int i = 0; i < nt; ++i)
                    cudaMemcpy(h_th.data() + (size_t)i * nt,
                               hessian.data_ptr<double>() + (size_t)(n2 + i) * total + n2,
                               (size_t)nt * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < nt; ++i)
                    for (int j = i + 1; j < nt; ++j) {
                        double m = 0.5 * (h_th[(size_t)i * nt + j] + h_th[(size_t)j * nt + i]);
                        h_th[(size_t)i * nt + j] = m; h_th[(size_t)j * nt + i] = m;
                    }
                for (int i = 0; i < nt; ++i)
                    cudaMemcpy(hessian.data_ptr<double>() + (size_t)(n2 + i) * total + n2,
                               h_th.data() + (size_t)i * nt,
                               (size_t)nt * sizeof(double), cudaMemcpyHostToDevice);
            }
        }

        // 恢复主 GPU 为当前设备：内部 per-GPU 循环会把它留在最后一个设备，
        // 导致后续 torch.device('cuda')（index=None → 当前设备）的 tensor 落在
        // 非主 GPU，触发 getNLL/getHessian 的 params 位置检查（测试套件实测）。
        cudaSetDevice(primary_dev_);

        if (hprof) {
            auto hTend = std::chrono::high_resolution_clock::now();
            auto hms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            printf("[PROF] H.reAmp: %.2f | H.phspM: %.2f | H.vv: %.2f | H.vth+thth+cpl: %.2f | H.total: %.2f\n",
                hms(hT0, hT1), hms(hT1, hT2), hms(hT2, hT3), hms(hT3, hTend), hms(hT0, hTend));
            fflush(stdout);
        }
        return hessian;
    }

    // Helper: compute partial + scattering (+square) integrals from given amplitudes.
    // Accumulates into out_partial/out_scattering/out_square (caller must zero-init;
    // out_square may be nullptr when Σ|A_i|⁴ 不需要, e.g. 拟合分数).
    void computeTruthIntegrals(
        const torch::Tensor& extended_vector,
        const std::vector<ctComplex*>& d_amps,
        const std::vector<int>& ev_per_gpu,
        double* out_partial,
        double* out_scattering,
        double* out_square,
        int npartials) const
    {
        for (size_t gpu = 0; gpu < d_amps.size(); ++gpu) {
            int nt = ev_per_gpu[gpu];
            if (nt <= 0 || d_amps[gpu] == nullptr) continue;

            int* d_nsl;
            cudaSetDevice(gpu);
            cudaMalloc(&d_nsl, npartials * sizeof(int));
            cudaMemcpy(d_nsl, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);

            double* d_p; cudaMalloc(&d_p, npartials * sizeof(double));
            cudaMemset(d_p, 0, npartials * sizeof(double));
            double* d_s; cudaMalloc(&d_s, npartials * npartials * sizeof(double));
            cudaMemset(d_s, 0, npartials * npartials * sizeof(double));
            double* d_t; cudaMalloc(&d_t, sizeof(double)); cudaMemset(d_t, 0, sizeof(double));
            double* d_sq = nullptr;
            if (out_square != nullptr) {
                cudaMalloc(&d_sq, npartials * sizeof(double));
                cudaMemset(d_sq, 0, npartials * sizeof(double));
            }

            auto vg = extended_vector.to(torch::Device(torch::kCUDA, gpu));
            computeBranchingFractions(d_amps[gpu],
                reinterpret_cast<const ctComplex*>(vg.data_ptr()),
                d_p, d_s, d_t, d_sq, d_nsl, npartials, nt, n_amplitudes_, n_polar_);

            std::vector<double> hp(npartials), hs(npartials * npartials); double ht;
            cudaMemcpy(hp.data(), d_p, npartials * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(), d_s, npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&ht, d_t, sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < npartials; ++i) {
                out_partial[i] += hp[i];
                for (int j = 0; j < npartials; ++j)
                    out_scattering[i * npartials + j] += hs[i * npartials + j];
            }
            if (d_sq != nullptr) {
                std::vector<double> hsq(npartials);
                cudaMemcpy(hsq.data(), d_sq, npartials * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < npartials; ++i) out_square[i] += hsq[i];
                cudaFree(d_sq);
            }
            cudaFree(d_p); cudaFree(d_s); cudaFree(d_t); cudaFree(d_nsl);
        }
    }

    // =====================================================================
    // 通用: 在指定数据文件（phsp_truth/phsp 等）上分批积分
    //   out_partial [npartials]          Σ|A_i|²（中心）
    //   out_scattering [npartials²]      2Re(A_i A_j*)（中心）
    //   out_square [npartials]           Σ|A_i|⁴（中心, 可传 nullptr）
    //   out_p/m_partial/scattering        ±ε 扰动后的积分（可为 nullptr 跳过扰动）
    // 返回该文件的事件总数（供归一化）。
    // =====================================================================
    int computeFileBatchIntegrals(
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
        const auto& data_files = config_parser_.getDataFiles();
        TORCH_CHECK(data_files.count(file_key) > 0,
            std::string("No data file '") + file_key + "' in config");

        std::vector<std::string> particles_names;
        for (const auto& p : particles_) particles_names.push_back(p.name);

        std::string file = data_files.at(file_key)[1];
        int total_events = 0;
        {
            std::ifstream f(file);
            std::string line;
            int lines = 0;
            while (std::getline(f, line)) if (!line.empty()) ++lines;
            total_events = lines / (int)particles_.size();
        }
        std::cout << "[" << file_key << "] events: " << total_events << std::endl;

        // Batched accumulation
        const int batch_size = 100000;
        const auto& data_order = config_parser_.getDataOrder();
        auto saved_ev = events_offsets_, saved_amp = amp_offsets_;

        for (int start = 0; start < total_events; start += batch_size) {
            int n_batch = std::min(batch_size, total_events - start);

            auto Vp4_batch = readMomentaFromDat(data_files.at(file_key),
                data_order, particles_names, n_batch, start);

            // Split across GPUs
            std::vector<int> batch_ev_per_gpu(n_gpus_, 0);
            int base_b = n_batch / n_gpus_, rem_b = n_batch % n_gpus_;
            for (int g = 0; g < n_gpus_; ++g)
                batch_ev_per_gpu[g] = base_b + (g < rem_b ? 1 : 0);

            std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_tpg(n_gpus_);
            int ev_off = 0;
            for (int g = 0; g < n_gpus_; ++g) {
                int nev = batch_ev_per_gpu[g];
                if (nev > 0)
                    for (const auto& [k, v] : Vp4_batch)
                        Vp4_tpg[g][k].assign(v.begin() + ev_off, v.begin() + ev_off + nev);
                ev_off += nev;
            }

            std::vector<std::vector<int>> t_ev_off(n_gpus_), t_amp_off(n_gpus_);
            for (int g = 0; g < n_gpus_; ++g) {
                int nev = batch_ev_per_gpu[g];
                t_ev_off[g] = {0, nev};
                t_amp_off[g] = {0, nev * n_polar_ * n_amplitudes_};
            }
            events_offsets_ = t_ev_off; amp_offsets_ = t_amp_off;
            std::vector<ctComplex*> d_batch_amps = calculateAmplitudes(Vp4_tpg);

            // Center (+ square)
            computeTruthIntegrals(ev_center, d_batch_amps, batch_ev_per_gpu,
                out_partial, out_scattering, out_square, npartials);

            // Jacobian perturbations (reuse same batch amplitudes)
            if (out_p_partial != nullptr) {
                for (int j = 0; j < (int)ev_perturbed_p.size(); ++j) {
                    computeTruthIntegrals(ev_perturbed_p[j], d_batch_amps, batch_ev_per_gpu,
                        out_p_partial + j * npartials,
                        out_p_scattering + j * npartials * npartials, nullptr, npartials);
                    computeTruthIntegrals(ev_perturbed_m[j], d_batch_amps, batch_ev_per_gpu,
                        out_m_partial + j * npartials,
                        out_m_scattering + j * npartials * npartials, nullptr, npartials);
                }
            }

            // Free batch amplitudes
            for (size_t g = 0; g < d_batch_amps.size(); ++g)
                if (d_batch_amps[g]) { cudaSetDevice(static_cast<int>(g)); cudaFree(d_batch_amps[g]); }
        }
        events_offsets_ = saved_ev; amp_offsets_ = saved_amp;
        return total_events;
    }

    // =====================================================================
    // 拟合分数: FF_i = truth_partial[i] / Σ_j truth_partial[j]
    //           = ∫|A_i|² / Σ_j ∫|A_j|²（无干涉项的分母, Σ_i FF_i = 1）。
    // 只用 phsp_truth（无效率 MC）→ 不依赖效率 MC 与数据归一化, 跨实验可比。
    // 注意: tfpwa 标准 fit fraction 分母用 ∫|ΣA|²（含干涉, Σ_i f_i ≠ 1），
    // 强干涉时两者略有差别, 报告时建议注明定义。
    // hessian_in: 可选统一 Hessian。分波误差与拟合 getHessian 同源。
    // 返回 [npartials, 2]: [center, error]
    // =====================================================================
    torch::Tensor getFitFractions(torch::Tensor vector)
    {
        return getFitFractions(vector, torch::Tensor());
    }
    torch::Tensor getFitFractions(torch::Tensor vector,
        torch::Tensor hessian_in)
    {
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision (double/float 编译)");
        if (hessian_in.numel() > 0)
            TORCH_CHECK(hessian_in.dtype() == torch::kFloat64 && hessian_in.is_cuda(),
                "hessian_in must be float64 CUDA");

        // 未配置 phsp_truth(无效率相空间 MC)时天然返回空张量 [0,2]:
        // 早期拟合不带 mctruth, fit.py 据此自然跳过, 不抛异常、不触碰任何相关 kernel
        const auto& data_files = config_parser_.getDataFiles();
        if (data_files.count("phsp_truth") == 0) {
            return torch::empty({0, 2},
                torch::TensorOptions().dtype(torch::kFloat64));
        }

        // 自由耦合 p → 扩展振幅 v（乘法链映射），Jacobian/Hessian 在 p 空间
        torch::Tensor ev_center = freeParamsToAmplitudes(vector);
        const int n = vector.numel();
        const int n2 = 2 * n;
        const int npartials = nSLvectors_.size();
        torch::Device dev = vector.device();

        // Perturbed vectors for Jacobian (p 空间扰动 → v 空间积分)。
        // grouped 布局 [Re_p(0..n-1), Im_p(n..2n-1)]，与 getHessian 一致。
        // 相对步长: 绝对 5e-6 在 |p_j| ≳ 10 时低于
        // float32 ulp，差分退化为舍入噪声。
        auto vr = torch::view_as_real(vector);            // [n, 2] float32 interleaved
        torch::Tensor p_real = torch::cat({
            vr.slice(1, 0, 1).reshape({ -1 }),
            vr.slice(1, 1, 2).reshape({ -1 }),
            }).contiguous();                                   // [2n] grouped
        std::vector<double> eps_j(n2, 5e-6);
        std::vector<torch::Tensor> ev_perturbed_p(n2), ev_perturbed_m(n2);
        auto grouped_to_complex = [&](const torch::Tensor& g) {
            return torch::view_as_complex(
                torch::stack({ g.slice(0, 0, n), g.slice(0, n, 2 * n) }, 1).contiguous());
            };
        for (int j = 0; j < n2; ++j) {
            eps_j[j] = 5e-6 * std::max(std::abs((double)p_real[j].item<float>()), 1.0);
            auto vp = p_real.clone(); vp[j] += eps_j[j];
            auto vm = p_real.clone(); vm[j] -= eps_j[j];
            ev_perturbed_p[j] = freeParamsToAmplitudes(grouped_to_complex(vp));
            ev_perturbed_m[j] = freeParamsToAmplitudes(grouped_to_complex(vm));
        }

        // Truth integrals (center + Jacobian); only partial integrals needed
        std::vector<double> truth_c_partial(npartials, 0.0);
        std::vector<double> truth_c_scattering(npartials * npartials, 0.0);
        std::vector<double> truth_p_partial(n2 * npartials, 0.0);
        std::vector<double> truth_p_scattering(n2 * npartials * npartials, 0.0);
        std::vector<double> truth_m_partial(n2 * npartials, 0.0);
        std::vector<double> truth_m_scattering(n2 * npartials * npartials, 0.0);
        computeFileBatchIntegrals("phsp_truth", ev_center, ev_perturbed_p, ev_perturbed_m,
            truth_c_partial.data(), truth_c_scattering.data(), nullptr,
            truth_p_partial.data(), truth_p_scattering.data(),
            truth_m_partial.data(), truth_m_scattering.data(), npartials);

        // FF_i = truth_partial[i] / Σ_j truth_partial[j]
        double truth_total = 0.0;
        for (int i = 0; i < npartials; ++i) truth_total += truth_c_partial[i];
        if (truth_total <= 0.0) truth_total = 1e-30;

        std::vector<double> ff_center(npartials);
        for (int i = 0; i < npartials; ++i)
            ff_center[i] = truth_c_partial[i] / truth_total;

        // ===== Errors: FF_error = sqrt(diag(J_FF @ H^{-1} @ J_FF^T)) =====
        // J_FF[i][j] = (FF_p[i] - FF_m[i]) / (2·eps_j[j])
        // 固定参考参数方向同上: mask 掉再检查正定性/求逆。
        // Hessian 优先用传入的统一 Hessian（与拟合正定判定同源）。
        std::vector<double> ff_errors(npartials, 0.0);
        torch::Tensor hessian;
        if (hessian_in.numel() > 0)
            hessian = hessian_in.slice(0, 0, n2).slice(1, 0, n2);  // 取耦合块
        else
            hessian = computeCouplingHessian(vector);
        if (hessian.numel() > 0 && hessian.size(0) == n2) {
            std::vector<int64_t> free_idx;
            for (int j = 0; j < n2; ++j)
                if (j != 0 && j != n) free_idx.push_back(j);
            const int n_free = (int)free_idx.size();
            if (n_free == 0) {
                // 全部方向固定(如单链仅参考耦合): 无自由参数可传播, 参数误差为 0
            } else {
            auto idx = torch::tensor(free_idx, torch::kInt64).to(hessian.device());
            torch::Tensor H_free = hessian.index_select(0, idx).index_select(1, idx);
            auto eig = torch::linalg_eigvalsh(H_free);
            if (eig[0].item<double>() > 1e-8) {
                torch::Tensor cov = torch::linalg_inv(H_free).cpu();
                std::vector<double> h_cov(n_free * n_free);
                std::memcpy(h_cov.data(), cov.data_ptr<double>(), n_free * n_free * sizeof(double));

                std::vector<double> J(npartials * n_free, 0.0);
                for (int fj = 0; fj < n_free; ++fj) {
                    const int j = free_idx[fj];
                    double tot_p = 0.0, tot_m = 0.0;
                    for (int i = 0; i < npartials; ++i) {
                        tot_p += truth_p_partial[j * npartials + i];
                        tot_m += truth_m_partial[j * npartials + i];
                    }
                    if (tot_p <= 0.0) tot_p = 1e-30;
                    if (tot_m <= 0.0) tot_m = 1e-30;
                    for (int i = 0; i < npartials; ++i) {
                        double ff_p = truth_p_partial[j * npartials + i] / tot_p;
                        double ff_m = truth_m_partial[j * npartials + i] / tot_m;
                        J[i * n_free + fj] = (ff_p - ff_m) / (2.0 * eps_j[j]);
                    }
                }

                computeBFErrors(J.data(), h_cov.data(), ff_errors.data(), npartials, n_free);
            }
            }
        }
        // 返回 n×2: [center, error]
        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor result = torch::empty({ npartials, 2 }, opts);
        for (int i = 0; i < npartials; ++i) {
            result[i][0] = ff_center[i];
            result[i][1] = ff_errors[i];
        }
        return result;
    }

    // =====================================================================
    // 效率: ε_i = (Σ_{phsp}|A_i|² / N_phsp) / (Σ_{phsp_truth}|A_i|² / N_truth)
    //   phsp（如 cut 后 MC）= 带效率样本; phsp_truth = 无效率 MC truth。
    //   ⇒ 分波加权的探测/选择效率: ∫|A_i|²ε(x)dΦ / ∫|A_i|²dΦ。
    //   ε 是拟合依赖的（振幅形状含共振参数）, 用拟合后 v 计算。
    // 误差 = 参数误差（Jacobian @ H⁻¹ @ Jᵀ, 与 getFitFractions 同机制）⊕
    //        MC 统计误差（tf-pwa add_int_error 同款: σ²/ε² = S_eff/I²_eff + S_tr/I²_tr,
    //        S = Σ|A_i|⁴, I = Σ|A_i|²）。
    // 返回 [npartials, 2]: [center, error]
    // =====================================================================
    torch::Tensor getEfficiency(torch::Tensor vector)
    {
        return getEfficiency(vector, torch::Tensor());
    }
    torch::Tensor getEfficiency(torch::Tensor vector,
        torch::Tensor hessian_in)
    {
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision (double/float 编译)");
        if (hessian_in.numel() > 0)
            TORCH_CHECK(hessian_in.dtype() == torch::kFloat64 && hessian_in.is_cuda(),
                "hessian_in must be float64 CUDA");

        // 需要 phsp（带效率 MC）与 phsp_truth（无效率 MC）两个数据键;
        // 缺任一 → 天然返回空张量 [0,2]（同 getFitFractions 的缺数据行为）
        const auto& data_files = config_parser_.getDataFiles();
        if (data_files.count("phsp_truth") == 0 || data_files.count("phsp") == 0) {
            return torch::empty({0, 2},
                torch::TensorOptions().dtype(torch::kFloat64));
        }

        torch::Tensor ev_center = freeParamsToAmplitudes(vector);
        const int n = vector.numel();
        const int n2 = 2 * n;
        const int npartials = nSLvectors_.size();

        // Perturbed vectors for Jacobian（与 getFitFractions 相同约定）
        auto vr = torch::view_as_real(vector);
        torch::Tensor p_real = torch::cat({
            vr.slice(1, 0, 1).reshape({ -1 }),
            vr.slice(1, 1, 2).reshape({ -1 }),
            }).contiguous();
        std::vector<double> eps_j(n2, 5e-6);
        std::vector<torch::Tensor> ev_perturbed_p(n2), ev_perturbed_m(n2);
        auto grouped_to_complex = [&](const torch::Tensor& g) {
            return torch::view_as_complex(
                torch::stack({ g.slice(0, 0, n), g.slice(0, n, 2 * n) }, 1).contiguous());
            };
        for (int j = 0; j < n2; ++j) {
            eps_j[j] = 5e-6 * std::max(std::abs((double)p_real[j].item<float>()), 1.0);
            auto vp = p_real.clone(); vp[j] += eps_j[j];
            auto vm = p_real.clone(); vm[j] -= eps_j[j];
            ev_perturbed_p[j] = freeParamsToAmplitudes(grouped_to_complex(vp));
            ev_perturbed_m[j] = freeParamsToAmplitudes(grouped_to_complex(vm));
        }

        // phsp（带效率）与 phsp_truth（无效率）各做一次批积分（中心+扰动+平方）
        std::vector<double> eff_c_partial(npartials, 0.0);
        std::vector<double> eff_c_scattering(npartials * npartials, 0.0);
        std::vector<double> eff_c_square(npartials, 0.0);
        std::vector<double> eff_p_partial(n2 * npartials, 0.0);
        std::vector<double> eff_p_scattering(n2 * npartials * npartials, 0.0);
        std::vector<double> eff_m_partial(n2 * npartials, 0.0);
        std::vector<double> eff_m_scattering(n2 * npartials * npartials, 0.0);
        int N_eff = computeFileBatchIntegrals("phsp", ev_center,
            ev_perturbed_p, ev_perturbed_m,
            eff_c_partial.data(), eff_c_scattering.data(), eff_c_square.data(),
            eff_p_partial.data(), eff_p_scattering.data(),
            eff_m_partial.data(), eff_m_scattering.data(), npartials);

        std::vector<double> tr_c_partial(npartials, 0.0);
        std::vector<double> tr_c_scattering(npartials * npartials, 0.0);
        std::vector<double> tr_c_square(npartials, 0.0);
        std::vector<double> tr_p_partial(n2 * npartials, 0.0);
        std::vector<double> tr_p_scattering(n2 * npartials * npartials, 0.0);
        std::vector<double> tr_m_partial(n2 * npartials, 0.0);
        std::vector<double> tr_m_scattering(n2 * npartials * npartials, 0.0);
        int N_truth = computeFileBatchIntegrals("phsp_truth", ev_center,
            ev_perturbed_p, ev_perturbed_m,
            tr_c_partial.data(), tr_c_scattering.data(), tr_c_square.data(),
            tr_p_partial.data(), tr_p_scattering.data(),
            tr_m_partial.data(), tr_m_scattering.data(), npartials);

        if (N_eff <= 0 || N_truth <= 0) {
            return torch::empty({0, 2},
                torch::TensorOptions().dtype(torch::kFloat64));
        }

        const double scale_eff = 1.0 / N_eff, scale_tr = 1.0 / N_truth;
        auto effOf = [&](double num_eff, double num_tr) -> double {
            double den = num_tr * scale_tr;
            if (den <= 0.0) return 0.0;
            return (num_eff * scale_eff) / den;
        };
        std::vector<double> eff_center(npartials);
        for (int i = 0; i < npartials; ++i)
            eff_center[i] = effOf(eff_c_partial[i], tr_c_partial[i]);

        // ===== 误差: σ² = σ_param² + σ_stat² =====
        std::vector<double> eff_errors(npartials, 0.0);

        // 参数误差（同 getFitFractions: H⁻¹ 用 mask 掉固定方向的自由块）
        torch::Tensor hessian;
        if (hessian_in.numel() > 0)
            hessian = hessian_in.slice(0, 0, n2).slice(1, 0, n2);
        else
            hessian = computeCouplingHessian(vector);
        if (hessian.numel() > 0 && hessian.size(0) == n2) {
            std::vector<int64_t> free_idx;
            for (int j = 0; j < n2; ++j)
                if (j != 0 && j != n) free_idx.push_back(j);
            const int n_free = (int)free_idx.size();
            if (n_free == 0) {
                // 全部方向固定(如单链仅参考耦合): 无自由参数可传播, 参数误差为 0
            } else {
            auto idx = torch::tensor(free_idx, torch::kInt64).to(hessian.device());
            torch::Tensor H_free = hessian.index_select(0, idx).index_select(1, idx);
            auto eig = torch::linalg_eigvalsh(H_free);
            if (eig[0].item<double>() > 1e-8) {
                torch::Tensor cov = torch::linalg_inv(H_free).cpu();
                std::vector<double> h_cov(n_free * n_free);
                std::memcpy(h_cov.data(), cov.data_ptr<double>(), n_free * n_free * sizeof(double));

                std::vector<double> J(npartials * n_free, 0.0);
                for (int fj = 0; fj < n_free; ++fj) {
                    const int j = free_idx[fj];
                    for (int i = 0; i < npartials; ++i) {
                        double e_p = effOf(eff_p_partial[j * npartials + i],
                                           tr_p_partial[j * npartials + i]);
                        double e_m = effOf(eff_m_partial[j * npartials + i],
                                           tr_m_partial[j * npartials + i]);
                        J[i * n_free + fj] = (e_p - e_m) / (2.0 * eps_j[j]);
                    }
                }
                computeBFErrors(J.data(), h_cov.data(), eff_errors.data(), npartials, n_free);
            }
            }
        }

        // MC 统计误差（tf-pwa add_int_error 同款）:
        //   σ_stat_i² = ε_i² × ( S_eff / I_eff² + S_tr / I_tr² )
        //   S, I 均为未归一化总和 Σ|A_i|⁴ / Σ|A_i|²（每事件均值因子 N 自然相消,
        //   与 tf-pwa cached_square / int_total² 一致）
        for (int i = 0; i < npartials; ++i) {
            double Ie = eff_c_partial[i];   // Σ|A_i|² (未归一化)
            double It = tr_c_partial[i];
            double stat2 = 0.0;
            if (Ie > 0.0) stat2 += eff_c_square[i] / (Ie * Ie);
            if (It > 0.0) stat2 += tr_c_square[i] / (It * It);
            double e = eff_center[i];
            double total2 = eff_errors[i] * eff_errors[i] + e * e * stat2;
            eff_errors[i] = std::sqrt(std::max(0.0, total2));
        }

        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor result = torch::empty({ npartials, 2 }, opts);
        for (int i = 0; i < npartials; ++i) {
            result[i][0] = eff_center[i];
            result[i][1] = eff_errors[i];
        }
        return result;
    }

    ////////////////////////
    torch::Tensor getDataTensor() const
    {
        // torch::Tensor output = torch::from_blob(data_fix_, {data_length *
        // n_gls_},
        // torch::TensorOptions().dtype(TORCH_COMPLEX).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_[0] + amp_offsets_[0][1],
            { events_[0][1] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(TORCH_COMPLEX).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getPhspTensor() const
    {
        TORCH_CHECK(!phsp_freed_,
            "Constraints.free_phsp_amplitudes=true 模式下 phsp 振幅不驻留（流式），"
            "getPhspTensor 不可用；请关闭该开关，或用 analysis.getNLL（C++ 路径）拟合");
        // torch::Tensor output = torch::from_blob(phsp_fix_, {phsp_length *
        // n_gls_},
        // torch::TensorOptions().dtype(TORCH_COMPLEX).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_[0],
            { events_[0][0] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(TORCH_COMPLEX).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getBkgTensor() const
    {
        if (events_[0].size() <= 2 || events_[0][2] == 0)
            return torch::empty({ 0 }, torch::TensorOptions().dtype(TORCH_COMPLEX).device(torch::kCUDA));
        // torch::Tensor output = torch::from_blob(bkg_fix_, {bkg_length *
        // n_gls_},
        // torch::TensorOptions().dtype(TORCH_COMPLEX).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_[0] + amp_offsets_[0][2],
            { events_[0][2] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(TORCH_COMPLEX).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getBkgWeightsTensor() const
    {
        // if (bkg_weights_ != nullptr && bkg_length > 0)
        if (events_[0].size() > 2 && events_[0][2] > 0 && !bkg_weights_.empty() && bkg_weights_[0] != nullptr)
        {
            torch::Tensor output = torch::from_blob(bkg_weights_[0], { events_[0][2] },
                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)).clone();
            return output;
        }
        else
        {
            return torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        }
    }
    void saveSLAmps(const std::string& filename) const
    {
        TORCH_CHECK(!phsp_freed_,
            "phsp 流式模式（free_phsp_amplitudes）下 SL 振幅表不含 phsp 事件，"
            "saveSLAmps 不可用");
        auto& cas_list = amp_calc_.casList();
        if (cas_list.empty()) { printf("saveSLAmps: no cas available\n"); return; }
        auto& cas = cas_list[0];
        int nEv = static_cast<int>(cas->getNEventsVec()[0]);
        int nPol = static_cast<int>(cas->getNPolarizations());
        int nSL = static_cast<int>(cas->getNSLCombs());
        int total = nEv * nPol * nSL;
        thrust::complex<double>* h_buf = new thrust::complex<double>[total];
        cudaMemcpy(h_buf, cas->getSLAmps()[0], total * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
        FILE* f = fopen(filename.c_str(), "wb");
        fwrite(h_buf, sizeof(thrust::complex<double>), total, f);
        fclose(f);
        delete[] h_buf;
        printf("saveSLAmps: saved %d complex numbers to %s (nEv=%d nPol=%d nSL=%d)\n", total, filename.c_str(), nEv, nPol, nSL);
    }

    // 返回所有链拼接的 SLAmps tensor，shape [totalSL, nEv, nPol]
    torch::Tensor getSLAmpsTensor() const
    {
        TORCH_CHECK(!phsp_freed_,
            "phsp 流式模式（free_phsp_amplitudes）下 SL 振幅表不含 phsp 事件，"
            "getSLAmpsTensor 不可用");
        auto& cas_list = amp_calc_.casList();
        TORCH_CHECK(!cas_list.empty(), "No cas available");
        int nEv = static_cast<int>(cas_list[0]->getNEventsVec()[0]);
        int nPol = static_cast<int>(cas_list[0]->getNPolarizations());
        int totalSL = 0;
        for (auto& cas : cas_list) totalSL += static_cast<int>(cas->getNSLCombs());
        auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
        torch::Tensor result = torch::empty({totalSL, nEv, nPol}, options);
        auto acc = result.accessor<c10::complex<double>, 3>();
        int sl_off = 0;
        for (auto& cas : cas_list) {
            int nSL = static_cast<int>(cas->getNSLCombs());
            int n = nEv * nPol * nSL;
            thrust::complex<double>* h_buf = new thrust::complex<double>[n];
            cudaMemcpy(h_buf, cas->getSLAmps()[0], n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
            for (int s = 0; s < nSL; ++s)
                for (int e = 0; e < nEv; ++e)
                    for (int p = 0; p < nPol; ++p) {
                        auto v = h_buf[s * nEv * nPol + e * nPol + p];
                        acc[sl_off + s][e][p] = c10::complex<double>(v.real(), v.imag());
                    }
            delete[] h_buf;
            sl_off += nSL;
        }
        return result;
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

    void reCalcAmp(torch::Tensor params)
    {
        TORCH_CHECK(params.is_cuda(), "params must be on CUDA");
        TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64");
        TORCH_CHECK(params.dim() == 1, "params must be 1-dimensional");
        TORCH_CHECK(params.numel() == amp_calc_.nFreeResParams(),
            "params size mismatch: got ", params.numel(),
            ", expected ", amp_calc_.nFreeResParams());

        amp_calc_.reComputeAmps(d_all_amplitudes_,
            reinterpret_cast<const double*>(params.data_ptr()),
            n_amplitudes_, events_offsets_, amp_offsets_, n_polar_,
            params.device().index());
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

    // 振幅数据，设备端
    std::vector<ctComplex*> d_all_amplitudes_;
    ctComplex* d_phsp_matrix_ = nullptr;
    // 双精度 phsp 矩阵（Constraints.free_phsp_amplitudes 流式模式用，主 GPU）：
    // phsp_sum = Re(v^H M_double v)，替代对原始 phsp 振幅的逐 forward 扫描
    cuDoubleComplex* d_phsp_matrix_double_ = nullptr;
    bool phsp_freed_ = false;   // true = phsp 振幅不驻留（流式模式）
    bool float_amps_ = false;   // true = 混合精度: 驻留振幅矩阵存 float（double .so + config precision:float）
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

        // 捕获主 GPU（torch 当前设备）。d_phsp_matrix_ 等跨 GPU 缓冲分配在此设备，
        // 后续所有 params 张量必须位于该设备（getNLL/getHessian 入口有 TORCH_CHECK）
        cudaGetDevice(&primary_dev_);

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

        // 精度解析（config precision 为"请求精度"）:
        //   .so 编译 double（默认）: precision:double/auto → 全 double;
        //                            precision:float → float_amps_=true（内存大户 A 存 float, 核心计算仍 double）
        //   .so 编译 float:          precision:float/auto → 原生 float; precision:double → 报错（float .so 无法提精度）
        {
            const std::string& req = config_parser_.getPrecision();
            if (req != "auto" && req != "float" && req != "double") {
                std::cerr << "ERROR: 配置 precision=\"" << req
                          << "\" 无效（仅支持 auto | float | double）" << std::endl;
                throw std::runtime_error("invalid precision in config");
            }
            if (req == "float" && std::string(PRECISION_NAME) == "double")
                float_amps_ = true;   // 混合精度: A 存 float（D1 起生效）, 核心 double
            else if (req == "double" && std::string(PRECISION_NAME) == "float") {
                std::cerr << "ERROR: 配置文件请求 precision=double，但当前 .so 编译为 float；"
                             "请用默认编译（double）重新 build_ext。float .so 仅供大统计量省显存。"
                          << std::endl;
                throw std::runtime_error("double requested on float build");
            }
            if (float_amps_)
                std::cout << "[ctpwa] 混合精度模式: A 存 float（省显存）, 核心计算 double"
                          << std::endl;
        }
        if (n_gpus_ == 0) {
            std::cerr << "ERROR: 无可用 CUDA 设备。ctpwa 当前仅支持 GPU 计算"
                         "（CPU 后端尚未实现），无法继续。" << std::endl;
            throw std::runtime_error("no CUDA devices available");
        }
        device_mgr_.print();

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
            // 每个 GPU 的峰值事件数 = data/phsp/bkg 的最大者。
            // phsp 流式模式下 phsp 不驻留（构造期按批算完即弃），峰值只按 data/bkg 估
            std::vector<int> peak_events(n_gpus_, 0);
            for (int gpu = 0; gpu < n_gpus_; ++gpu)
                for (size_t j = (phsp_streamed ? 1 : 0); j < events_[gpu].size(); ++j)
                    peak_events[gpu] = std::max(peak_events[gpu], events_[gpu][j]);
            bool has_bkg = (init_events.size() > 2 && init_events[2] > 0);
            auto cap = device_mgr_.checkCapacity(
                peak_events, n_amplitudes_, n_polar_, n_gls_, particles_.size(),
                has_bkg, params_.nFreeTheta());
            if (cap.overall == DeviceManager::CapacityStatus::FAIL) {
                std::cerr << "ERROR: 内存预检失败——GPU " << cap.failing_device
                          << " 无法承载数据: " << cap.failing_buffer
                          << " 需要 " << cap.required_bytes / 1e6
                          << " MiB，可用 " << cap.available_bytes / 1e6
                          << " MiB。请减少事件数或使用多 GPU。" << std::endl;
                throw std::runtime_error("input data exceeds device memory");
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

        cudaSetDevice(primary_dev_);  // ensure d_phsp_matrix_ is allocated on primary GPU
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

                    // float32 累加（scale 后 CGEMM）
                    {
                        ctComplex* d_scaled;
                        cudaMalloc(&d_scaled, nTot * sizeof(ctComplex));
                        cudaMemcpy(d_scaled, d_batch[gpu], nTot * sizeof(ctComplex),
                                   cudaMemcpyDeviceToDevice);
                        const double* d_w = (gpu < (int)phsp_weights_.size())
                            ? phsp_weights_[gpu] : nullptr;
                        int grid = (nTot + 255) / 256;
                        scalePhspAmpsKernel<<<grid, 256>>>(d_scaled,
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
                            cudaSetDevice(primary_dev_);
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
                        int grid2 = (nTot + 255) / 256;
                        castPhspBatchToDoubleKernel<<<grid2, 256>>>(d_batch[gpu], d_batch_d, nTot);
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
                            cudaSetDevice(primary_dev_);
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

                    // 释放批振幅（含占位 GPU 的 1 事件缓冲）
                    for (int g = 0; g < n_gpus_; ++g) {
                        if (d_batch[g]) { cudaSetDevice(g); cudaFree(d_batch[g]); }
                    }
                    cudaSetDevice(gpu);
                }
                cublasDestroy(h);
            }
            events_offsets_ = saved_ev;
            amp_offsets_ = saved_amp;
            cudaSetDevice(primary_dev_);
            std::cout << "[ctpwa] phsp 矩阵（float32+double）构建完成，phsp 振幅不驻留。"
                      << std::endl;
        } else {
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu)
            {
                int nPhsp = events_[gpu][0];
                if (nPhsp == 0) continue;
                cudaSetDevice(gpu);

                int nPhsp_total = nPhsp * n_polar_ * n_amplitudes_;
                ctComplex* d_phsp_scaled;
                cudaMalloc(&d_phsp_scaled, nPhsp_total * sizeof(ctComplex));
                cudaMemcpy(d_phsp_scaled, d_all_amplitudes_[gpu],
                           nPhsp_total * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                const double* d_w = (gpu < phsp_weights_.size()) ? phsp_weights_[gpu] : nullptr;
                int grid = (nPhsp_total + 255) / 256;
                scalePhspAmpsKernel<<<grid, 256>>>(d_phsp_scaled, d_w,
                    nPhsp, n_polar_, n_amplitudes_, 1.0 / W_total, 0);

                ctComplex* d_phsp_gpu;
                cudaMalloc(&d_phsp_gpu, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                cublasHandle_t h;
                cublasCreate(&h);
                ctComplex alpha = ctMake(1.0f, 0.0f);
                ctComplex beta  = ctMake(0.0f, 0.0f);
                CUBLAS_CGEMM(h, CUBLAS_OP_N, CUBLAS_OP_C, n_amplitudes_, n_amplitudes_, nPhsp * n_polar_,
                    &alpha, d_phsp_scaled, n_amplitudes_, d_phsp_scaled, n_amplitudes_,
                    &beta, d_phsp_gpu, n_amplitudes_);
                cublasDestroy(h);
                cudaFree(d_phsp_scaled);

                if (gpu == (size_t)primary_dev_) {
                    ctComplex one = ctMake(1.0f, 0.0f);
                    axpyComplex(d_phsp_matrix_, d_phsp_gpu, one, n_amplitudes_ * n_amplitudes_);
                    cudaFree(d_phsp_gpu);
                } else {
                    cudaSetDevice(primary_dev_);
                    ctComplex* d_temp;
                    cudaMalloc(&d_temp, n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                    cudaMemcpyPeer(d_temp, primary_dev_, d_phsp_gpu, gpu,
                        n_amplitudes_ * n_amplitudes_ * sizeof(ctComplex));
                    cudaSetDevice(gpu);
                    cudaFree(d_phsp_gpu);
                    cudaSetDevice(primary_dev_);
                    ctComplex one = ctMake(1.0f, 0.0f);
                    axpyComplex(d_phsp_matrix_, d_temp, one, n_amplitudes_ * n_amplitudes_);
                    cudaFree(d_temp);
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

        std::cout << "Number of GPUs available: " << n_gpus_ << std::endl;
        std::cout << "Number of partial waves: " << n_gls_ << std::endl;
        std::cout << "Number of res parameters: " << params_.nFreeTheta() << std::endl;
        std::cout << "Initialization complete." << std::endl;
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
        int n = particles_.size();

        // 收集每粒子的维度 (2J+1) 和允许的张量索引列表
        std::vector<int> dims(n);
        std::vector<std::vector<int>> keep(n); // 空 = 保留全部
        bool any_masked = false;

        for (int i = 0; i < n; ++i) {
            dims[i] = particles_[i].spin;
            if (particles_[i].is_polarized()) {
                any_masked = true;
                int dim_j = dims[i];
                for (int two_m : particles_[i].polarization_2m) {
                    keep[i].push_back((dim_j - 1 - two_m) / 2);
                }
            }
        }

        // 原始 C-order strides: orig_strides[i] = prod_{j>i} dims[j]
        std::vector<int> orig_strides(n);
        orig_strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; --i)
            orig_strides[i] = orig_strides[i + 1] * dims[i + 1];

        // 总张量极化数
        n_polar_total_ = 1;
        for (int d : dims) n_polar_total_ *= d;

        // 有效维度（mask 后）和 masked strides
        std::vector<int> eff_dims(n);
        std::vector<int> masked_strides(n);
        for (int i = 0; i < n; ++i)
            eff_dims[i] = keep[i].empty() ? dims[i] : (int)keep[i].size();
        masked_strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; --i)
            masked_strides[i] = masked_strides[i + 1] * eff_dims[i + 1];

        // 有效极化态数
        n_polar_ = 1;
        for (int d : eff_dims) n_polar_ *= d;

        // 构建完整映射: output_idx → tensor_idx
        if (any_masked) {
            polarization_map_.resize(n_polar_);
            for (int out = 0; out < n_polar_; ++out) {
                int rem = out;
                int tensor_idx = 0;
                for (int p = 0; p < n; ++p) {
                    int j = rem / masked_strides[p];
                    rem %= masked_strides[p];
                    int i = keep[p].empty() ? j : keep[p][j];
                    tensor_idx += i * orig_strides[p];
                }
                polarization_map_[out] = tensor_idx;
            }
        }
        else {
            polarization_map_.clear();
        }

        std::cout << "polarization: " << n_polar_;
        // std::cout  << " (total tensor states: " << n_polar_total_ << ")";
        // if (!polarization_map_.empty()) {
        //     std::cout << ", map: [";
        //     for (size_t i = 0; i < polarization_map_.size(); ++i) {
        //         if (i > 0) std::cout << ", ";
        //         std::cout << polarization_map_[i];
        //     }
        //     std::cout << "]";
        // }
        std::cout << std::endl;
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

    std::vector<ctComplex*> calculateAmplitudes(const std::vector<std::map<std::string, std::vector<LorentzVector>>>& Vp4, AmpCalc* amp_calc = nullptr) const
    {
        // 多GPU支持：为每个GPU分配振幅内存
        // int num_gpus = events_.size();
        std::vector<ctComplex*> d_all_amplitudes_vec(n_gpus_, nullptr);

        // 计算每个GPU的总事件数
        // std::vector<int> nEvents_per_gpu(n_gpus_, 0);
        // for (int gpu = 0; gpu < n_gpus_; ++gpu) {
        //     for (size_t file_type = 0; file_type < events_[gpu].size(); ++file_type) {
        //         nEvents_per_gpu[gpu] += events_[gpu][file_type];
        //     }
        // }

        // 为每个GPU分配设备内存
        for (int gpu = 0; gpu < n_gpus_; ++gpu) {
            if (amp_offsets_[gpu].back() > 0) {
                cudaSetDevice(gpu);
                // std::cout << "Allocating memory for GPU " << gpu << ": " << events_offsets_[gpu].back() * n_polar_ * n_amplitudes_ << " " << amp_offsets_[gpu].back() << " amplitudes." << std::endl;
                cudaMalloc(&d_all_amplitudes_vec[gpu], amp_offsets_[gpu].back() * sizeof(ctComplex));
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
                cas->computeSLAmps(Vp4);
                // int nSLcombs = cas->getNSLCombs();
                // int nEvents = cas->getNEvents();

                // 调用多GPU版本的getAmps函数
                for (const auto resonance : resonance_combinations)
                {
                    cas->getAmps(d_all_amplitudes_vec, resonance, gls_index, n_amplitudes_, events_offsets_, amp_offsets_);

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
            }
        }

        // Constraints.var_equal: 所有 block 注册完毕后合并共享参数槽
        // （必须在此处——setNFreeTheta 之前——以便参数数一致）
        if (amp_calc) amp_calc->applyVarEqual(config_parser_.getVarEqual());

        return d_all_amplitudes_vec;
    }

};
