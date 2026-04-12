// #include <pybind11/pybind11.h>
#include <chrono>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <fstream>
#include <map>
#include <omp.h>
#include <random>
#include <torch/extension.h>
#include <unordered_set>
#include <vector>

#include <AmpGen.cuh>
#include <ComputeGrad.cuh>
#include <ComputeNLL.cuh>
#include <ComputeResults.cuh>
#include <Config.cuh>
#include <Figure.cuh>

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
struct ChainInfo
{
    std::string name;
    std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>>
        intermediate_resonance_map;
    std::vector<std::vector<Particle>> intermediate_combs;
};

struct NLLGPUParams {
    // cuComplex* d_all_amplitudes;
    std::vector<int> events;
    std::vector<int> events_offsets;
    std::vector<int> amp_offsets;
    // double* bkg_weights;  // 按需
};

////////////////////////////////////////
std::map<std::string, std::vector<LorentzVector>>
readMomentaFromDat(const std::vector<std::string>& fileinfo,
    const std::vector<std::string>& particleNames,
    const std::vector<std::string>& particlelists,
    int nEvents = -1)
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
        int particlesPerEvent = particleNames.size();

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            std::istringstream iss(line);
            double E, px, py, pz;

            if (iss >> E >> px >> py >> pz)
            {
                // 根据行号确定粒子类型
                int particleIndex = lineCount % particlesPerEvent;
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

double* readWeightsFromFile(const std::vector<std::string>& fileinfo,
    int length)
{
    // 检查输入参数
    if (fileinfo.size() < 2)
    {
        std::cerr
            << "Error: fileinfo must contain at least file type and filename"
            << std::endl;
        return nullptr;
    }

    std::string fileType = fileinfo[0];
    std::string filename = fileinfo[1];

    std::vector<double> weights;

    // 处理DAT文件
    if (fileType == "dat")
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return nullptr;
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
            std::cerr
                << "Error: For ROOT files, fileinfo must contain at least file "
                "type, filename, TTree name and weight branch name"
                << std::endl;
            return nullptr;
        }

        std::string treeName = fileinfo[2];
        std::string branchName = fileinfo[3];

        // 打开ROOT文件
        TFile* file = TFile::Open(filename.c_str(), "READ");
        if (!file || file->IsZombie())
        {
            std::cerr << "Error: Cannot open ROOT file " << filename
                << std::endl;
            return nullptr;
        }

        // 获取TTree
        TTree* tree = (TTree*)file->Get(treeName.c_str());
        if (!tree)
        {
            std::cerr << "Error: Cannot find TTree " << treeName << " in file "
                << filename << std::endl;
            file->Close();
            delete file;
            return nullptr;
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
        // USE_ROOT and link with ROOT libraries." << std::endl; return nullptr;
        // #endif
    }
    else
    {
        std::cerr << "Error: Unknown file type: " << fileType << std::endl;
        return nullptr;
    }

    // 检查权重数量
    if (length > 0 && weights.size() != static_cast<size_t>(length))
    {
        std::cerr << "Error: Weights size " << weights.size()
            << " does not match expected length " << length << std::endl;
        // 可以根据需求决定是否返回nullptr
        // return nullptr;
    }

    // 如果length为-1或0，使用实际读取的权重数量
    if (length <= 0)
    {
        length = weights.size();
    }

    // 分配设备内存并复制数据
    double* d_weights = nullptr;
    cudaError_t cudaStatus = cudaMalloc(&d_weights, length * sizeof(double));
    // 确保weights向量有足够的元素
    if (weights.size() < static_cast<size_t>(length))
    {
        std::cerr << "Warning: Not enough weights in file. Padding with zeros."
            << std::endl;
        weights.resize(length, 0.0);
    }

    cudaStatus = cudaMemcpy(d_weights, weights.data(), length * sizeof(double),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Error: cudaMemcpy failed for weights: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_weights);
        return nullptr;
    }

    return d_weights;
}

LorentzVector* convertToLorentzVector(
    const std::map<std::string, std::vector<LorentzVector>>& finalMomenta,
    const std::map<std::string, int>& particleToIndex)
{
    // 获取事件数量和粒子数量
    int n_events = finalMomenta.begin()->second.size();
    int n_particles = particleToIndex.size();

    // 在主机端分配所有粒子的四动量数组
    std::vector<LorentzVector> host_momenta(n_events * n_particles);
    std::fill(host_momenta.begin(), host_momenta.end(), LorentzVector());

    // 创建粒子计算状态标记
    std::vector<bool> particle_calculated(n_particles, false);

    // 第一步：将末态粒子的四动量复制到对应位置
    for (const auto& particle_momenta : finalMomenta)
    {
        const std::string& particle_name = particle_momenta.first;
        const std::vector<LorentzVector>& momenta_vec = particle_momenta.second;

        auto it = particleToIndex.find(particle_name);
        if (it != particleToIndex.end())
        {
            int particle_idx = it->second;
            for (int event_idx = 0; event_idx < n_events; ++event_idx)
            {
                host_momenta[event_idx * n_particles + particle_idx] =
                    momenta_vec[event_idx];
            }
            particle_calculated[particle_idx] = true;
        }
    }

    // 第二步：将数据复制到设备
    LorentzVector* d_momenta;
    cudaMalloc(&d_momenta, host_momenta.size() * sizeof(LorentzVector));
    cudaMemcpy(d_momenta, host_momenta.data(),
        host_momenta.size() * sizeof(LorentzVector),
        cudaMemcpyHostToDevice);

    return d_momenta;
}


std::vector<std::map<std::string, std::vector<LorentzVector>>> mergeMaps(
    const std::vector<std::map<std::string, std::vector<LorentzVector>>>& maps, std::vector<std::vector<int>>& events)
{
    if (maps.empty())
        return {};

    // 结果 map
    std::vector<std::map<std::string, std::vector<LorentzVector>>> result;

    for (size_t i = 0; i < events.size(); ++i)
    {
        std::map<std::string, std::vector<LorentzVector>> tmp;

        // 第一步：预留空间，避免后续插入时多次重新分配
        // 由于所有 map 的 key 集合相同，可以从第一个 map 获取所有 key
        const auto& first = maps[0];
        for (const auto& [key, vec] : first)
        {
            size_t total = vec.size(); // 第一个 map 的大小
            for (size_t i = 1; i < maps.size(); ++i)
            {
                auto it = maps[i].find(key);
                if (it != maps[i].end())
                    total += it->second.size();
            }
            tmp[key].reserve(total); // 一次性预留足够空间
        }

        // 第二步：将每个 map 中的 vector 追加到结果中
        for (const auto& m : maps)
        {
            for (const auto& [key, vec] : m)
            {
                auto& target = tmp[key];
                target.insert(target.end(), vec.begin(), vec.end()); // 批量插入
            }
        }

        result.push_back(tmp);
    }

    return result;
}

//////////////////////////////////////////////////////////////
/// NLLFunction 类定义
///////////////////////////////////////////////////////////////
class NLLFunction : public torch::autograd::Function<NLLFunction>
{
private:
    // 私有成员变量，存储约束信息
    static std::vector<std::vector<int>> con_trans_id_;
    static std::vector<std::vector<std::complex<double>>> con_trans_values_;
    static bool constraints_initialized_;

public:
    /*
        static torch::Tensor
            forward(torch::autograd::AutogradContext* ctx, torch::Tensor& vector,
                cuComplex* d_all_amplitudes_,
                const double* phsp_weights_,
                const double* bkg_weights_,
                const std::vector<int>& events_,
                const std::vector<int>& events_offsets_,
                const std::vector<int>& amp_offsets_,
                int n_amplitudes_,
                int n_polar_)
        {
            TORCH_CHECK(vector.is_cuda(), "[NLLForward] vector must be on CUDA");
            TORCH_CHECK(vector.dtype() == c10::kComplexFloat,
                "[NLLForward] vector must be complex64");

            // 检查约束是否已初始化
            TORCH_CHECK(constraints_initialized_,
                "[NLLForward] Constraints not initialized. Call "
                "setConstraints() first.");

            // 获取当前设备并设置
            const int target_dev = vector.get_device();
            torch::Device dev(torch::kCUDA, target_dev);

            // 延长vector以处理约束
            torch::Tensor extended_vector =
                extendVectorWithConstraints(vector, dev);
            const int extended_n_gls = extended_vector.numel();

            // 后续逻辑（MC因子计算等）
            cuComplex* d_B = nullptr;
            double* d_mc_amp = nullptr;
            cudaMalloc(&d_B, events_[0] * n_polar_ * sizeof(cuComplex));
            cudaMalloc(&d_mc_amp, sizeof(double));

            // 注意：这里使用了 c10::complex 类型
            computePHSPfactor(d_all_amplitudes_, reinterpret_cast<const cuComplex*>(extended_vector.data_ptr()), d_B, d_mc_amp, events_[0] * n_polar_, extended_n_gls);

            double h_phsp_factor;
            cudaMemcpy(&h_phsp_factor, d_mc_amp, sizeof(double), cudaMemcpyDeviceToHost);
            h_phsp_factor = h_phsp_factor / static_cast<double>(events_[0]);

            // std::cout << "[NLLForward] PHSP factor: " << h_phsp_factor << "
            // Nphsp: "
            // << phsp_length / n_polar_ << std::endl;

            // NLL计算
            cuComplex* d_S = nullptr;
            cuComplex* d_Q = nullptr;
            double* d_data_nll = nullptr;
            const int Q_numel = events_[1];
            cudaMalloc(&d_S, events_[1] * n_polar_ * sizeof(cuComplex));
            cudaMalloc(&d_Q, Q_numel * sizeof(cuComplex));
            cudaMalloc(&d_data_nll, sizeof(double));

            computeNll(d_all_amplitudes_ + amp_offsets_[1], reinterpret_cast<const cuComplex*>(extended_vector.data_ptr()), nullptr, d_S, d_Q, d_data_nll, events_[1] * n_polar_, extended_n_gls, n_polar_, h_phsp_factor);

            double h_data_nll = 0.0;
            cudaMemcpy(&h_data_nll, d_data_nll, sizeof(double), cudaMemcpyDeviceToHost);

            // bkg部分
            cuComplex* d_bkg_S = nullptr;
            cuComplex* d_bkg_Q = nullptr;
            double* d_bkg_nll = nullptr;
            const int bkg_Q_numel = events_[2];
            cudaMalloc(&d_bkg_S, events_[2] * n_polar_ * sizeof(cuComplex));
            cudaMalloc(&d_bkg_Q, bkg_Q_numel * sizeof(cuComplex));
            cudaMalloc(&d_bkg_nll, sizeof(double));
            double h_bkg_nll = 0.0;
            if (events_[2] > 0)
            {
                computeNll(
                    d_all_amplitudes_ + amp_offsets_[2],
                    reinterpret_cast<const cuComplex*>(extended_vector.data_ptr()),
                    bkg_weights_, d_bkg_S, d_bkg_Q, d_bkg_nll,
                    events_[2] * n_polar_, extended_n_gls, n_polar_, h_phsp_factor);

                cudaMemcpy(&h_bkg_nll, d_bkg_nll, sizeof(double), cudaMemcpyDeviceToHost);
            }

            // 保存反向传播变量
            ctx->saved_data["target_dev"] = target_dev;
            ctx->saved_data["n_polar"] = n_polar_;
            // ctx->saved_data["h_phsp_factor"] = h_phsp_factor *
            // static_cast<double>(phsp_length / n_polar_);
            ctx->saved_data["h_phsp_factor"] = h_phsp_factor * static_cast<double>(events_[0]);
            ctx->saved_data["n_gls"] = n_amplitudes_;
            ctx->saved_data["extended_n_gls"] = extended_n_gls;
            ctx->saved_data["data_length"] = events_[1] * n_polar_;
            // ctx->saved_data["phsp_length"] = phsp_length;
            ctx->saved_data["phsp_length"] = events_[0] * n_polar_;
            ctx->saved_data["bkg_length"] = events_[2] * n_polar_;

            // 保存显存指针
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            ctx->saved_data["data_fix_ptr"] = reinterpret_cast<int64_t>(d_all_amplitudes_ + amp_offsets_[1]);
            ctx->saved_data["phsp_fix_ptr"] = reinterpret_cast<int64_t>(d_all_amplitudes_ + amp_offsets_[0]);
            ctx->saved_data["bkg_fix_ptr"] = reinterpret_cast<int64_t>(d_all_amplitudes_ + amp_offsets_[2]);
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            ctx->saved_data["bkg_weights_ptr"] = reinterpret_cast<int64_t>(bkg_weights_);
            ctx->saved_data["d_B_ptr"] = reinterpret_cast<int64_t>(d_B);
            ctx->saved_data["d_S_ptr"] = reinterpret_cast<int64_t>(d_S);
            ctx->saved_data["d_Q_ptr"] = reinterpret_cast<int64_t>(d_Q);
            ctx->saved_data["d_bkg_S_ptr"] = reinterpret_cast<int64_t>(d_bkg_S);
            ctx->saved_data["d_bkg_Q_ptr"] = reinterpret_cast<int64_t>(d_bkg_Q);
            ctx->saved_data["d_bkg_nll_ptr"] = reinterpret_cast<int64_t>(d_bkg_nll);

            ctx->save_for_backward({ vector, extended_vector });

            // 释放临时内存
            cudaFree(d_mc_amp);

            return torch::tensor({ h_data_nll - h_bkg_nll }, torch::kDouble).to(dev);
        }

        static torch::autograd::tensor_list
            backward(torch::autograd::AutogradContext* ctx,
                const torch::autograd::tensor_list& grad_outputs)
        {
            const int target_dev = ctx->saved_data["target_dev"].toInt();

            // 从 saved_data 获取参数
            const int n_polar = ctx->saved_data["n_polar"].toInt();
            const double h_phsp_factor = ctx->saved_data["h_phsp_factor"].toDouble();
            const int n_gls = ctx->saved_data["n_gls"].toInt();
            const int extended_n_gls = ctx->saved_data["extended_n_gls"].toInt();
            const int data_length = ctx->saved_data["data_length"].toInt();
            const int phsp_length = ctx->saved_data["phsp_length"].toInt();
            const int bkg_length = ctx->saved_data["bkg_length"].toInt();

            // 从 saved_data 获取显存指针
            cuComplex* d_B = reinterpret_cast<cuComplex*>(ctx->saved_data["d_B_ptr"].toInt());
            cuComplex* data_fix = reinterpret_cast<cuComplex*>(ctx->saved_data["data_fix_ptr"].toInt());
            cuComplex* phsp_fix = reinterpret_cast<cuComplex*>(ctx->saved_data["phsp_fix_ptr"].toInt());
            cuComplex* d_S = reinterpret_cast<cuComplex*>(ctx->saved_data["d_S_ptr"].toInt());
            cuComplex* d_Q = reinterpret_cast<cuComplex*>(ctx->saved_data["d_Q_ptr"].toInt());

            cuComplex* bkg_fix = reinterpret_cast<cuComplex*>(ctx->saved_data["bkg_fix_ptr"].toInt());
            double* bkg_weights = reinterpret_cast<double*>(ctx->saved_data["bkg_weights_ptr"].toInt());
            cuComplex* d_bkg_S = reinterpret_cast<cuComplex*>(ctx->saved_data["d_bkg_S_ptr"].toInt());
            cuComplex* d_bkg_Q = reinterpret_cast<cuComplex*>(ctx->saved_data["d_bkg_Q_ptr"].toInt());

            // 获取保存的变量
            const auto saved = ctx->get_saved_variables();
            const auto& original_vector = saved[0];
            const auto& extended_vector = saved[1];

            // 计算扩展向量的梯度
            cuComplex* d_extended_grad = nullptr;
            cudaMalloc(&d_extended_grad, extended_n_gls * sizeof(cuComplex));

            cublasHandle_t cublas_handle;
            cublasCreate(&cublas_handle);
            compute_gradient(data_fix, phsp_fix, d_S, d_Q, d_B, nullptr,
                h_phsp_factor, extended_n_gls, data_length / n_polar,
                n_polar, phsp_length, d_extended_grad, cublas_handle);

            // 如果有背景数据，减去背景NLL的梯度
            if (bkg_fix != nullptr && bkg_length > 0)
            {
                cuComplex* d_bkg_extended_grad = nullptr;
                cudaMalloc(&d_bkg_extended_grad, extended_n_gls * sizeof(cuComplex));

                cudaMemset(d_bkg_extended_grad, 0, extended_n_gls * sizeof(cuComplex));

                compute_gradient(bkg_fix, phsp_fix, d_bkg_S, d_bkg_Q, d_B,
                    bkg_weights, h_phsp_factor, extended_n_gls,
                    bkg_length / n_polar, n_polar, phsp_length,
                    d_bkg_extended_grad, cublas_handle);

                const cuComplex minus_one = make_cuComplex(-1.0f, 0.0f);
                cublasCaxpy(cublas_handle, extended_n_gls, &minus_one, d_bkg_extended_grad, 1, d_extended_grad, 1);

                cudaFree(d_bkg_extended_grad);
                cudaFree(d_bkg_S);
                cudaFree(d_bkg_Q);
            }

            // 将扩展梯度复制到torch张量
            torch::Tensor extended_grad = torch::empty({ extended_n_gls }, torch::kComplexFloat).to(original_vector.device());
            cudaMemcpy(extended_grad.data_ptr(), d_extended_grad, extended_n_gls * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

            // 合并梯度（考虑约束关系）
            torch::Tensor grad_vector = mergeGradientsWithConstraints(extended_grad, original_vector.numel());

            // 清理内存
            cudaFree(d_extended_grad);
            cudaFree(d_B);
            cudaFree(d_S);
            cudaFree(d_Q);
            cublasDestroy(cublas_handle);

            return { grad_vector,
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor() };
        }
    */

    // 多 GPU 前向传播
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor vector,                      // 原始参数张量（位于主设备）
        const std::vector<cuComplex*>& d_all_amplitudes_list,   // 每个 GPU 一个指针
        const std::vector<std::vector<int>>& events_list,       // 每个 GPU 的 events[3]
        const std::vector<std::vector<int>>& events_offsets_list,
        const std::vector<std::vector<int>>& amp_offsets_list,
        int n_amplitudes_,
        int n_polar_)
    {
        int num_gpus = d_all_amplitudes_list.size();
        TORCH_CHECK(num_gpus > 0, "No GPUs provided");
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(constraints_initialized_, "Constraints not initialized");

        // 1. 将 vector 复制到每个 GPU 上
        std::vector<torch::Tensor> vec_per_gpu;
        for (int i = 0; i < num_gpus; ++i) {
            vec_per_gpu.push_back(vector.to(torch::Device(torch::kCUDA, i)));
        }

        // 2. 每个 GPU 独立扩展向量（约束相同，结果相同，但需要各自设备上的张量）
        std::vector<torch::Tensor> extended_vec_per_gpu;
        int extended_n_gls = 0;
        for (int i = 0; i < num_gpus; ++i) {
            auto ext = extendVectorWithConstraints(vec_per_gpu[i], torch::Device(torch::kCUDA, i));
            extended_vec_per_gpu.push_back(ext);
            if (i == 0) extended_n_gls = ext.numel();
        }

        // 3. 每个 GPU 计算局部 phsp_factor 的分子（振幅平方和）和局部事件数
        std::vector<double> local_mc_amp(num_gpus, 0.0);
        std::vector<int> local_phsp_events(num_gpus, 0);
        std::vector<cuComplex*> d_B_list(num_gpus, nullptr);
        std::vector<cuComplex*> d_S_list(num_gpus, nullptr);
        std::vector<cuComplex*> d_Q_list(num_gpus, nullptr);
        std::vector<cuComplex*> d_bkg_S_list(num_gpus, nullptr);
        std::vector<cuComplex*> d_bkg_Q_list(num_gpus, nullptr);
        std::vector<double*> d_bkg_nll_list(num_gpus, nullptr);

        // 临时存储每个 GPU 的 h_phsp_factor（局部，稍后会被全局因子覆盖）
        std::vector<double> local_h_phsp_factor(num_gpus, 0.0);

        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            const auto& events = events_list[i];
            local_phsp_events[i] = events[0];  // 相空间事件数
            int phsp_len = events[0] * n_polar_;

            // 分配临时缓冲区
            cuComplex* d_B = nullptr;
            double* d_mc_amp = nullptr;
            cudaMalloc(&d_B, phsp_len * sizeof(cuComplex));
            cudaMalloc(&d_mc_amp, sizeof(double));
            d_B_list[i] = d_B;

            // 调用内核计算局部振幅平方和
            computePHSPfactor(d_all_amplitudes_list[i],
                reinterpret_cast<const cuComplex*>(extended_vec_per_gpu[i].data_ptr()),
                d_B, d_mc_amp, phsp_len, extended_n_gls);

            cudaMemcpy(&local_mc_amp[i], d_mc_amp, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_mc_amp);

            // 分配后续 NLL 计算需要的缓冲区（稍后使用）
            cudaMalloc(&d_S_list[i], events[1] * n_polar_ * sizeof(cuComplex));
            cudaMalloc(&d_Q_list[i], events[1] * sizeof(cuComplex));
            if (events[2] > 0) {
                cudaMalloc(&d_bkg_S_list[i], events[2] * n_polar_ * sizeof(cuComplex));
                cudaMalloc(&d_bkg_Q_list[i], events[2] * sizeof(cuComplex));
                cudaMalloc(&d_bkg_nll_list[i], sizeof(double));
            }
        }

        // 4. 合并所有 GPU 的分子和事件数，计算全局 phsp_factor
        double total_mc_amp = 0.0;
        int total_phsp_events = 0;
        for (int i = 0; i < num_gpus; ++i) {
            total_mc_amp += local_mc_amp[i];
            total_phsp_events += local_phsp_events[i];
        }
        double global_phsp_factor = total_mc_amp / static_cast<double>(total_phsp_events);

        // 5. 每个 GPU 使用全局 phsp_factor 计算各自的 NLL
        std::vector<double> local_data_nll(num_gpus, 0.0);
        std::vector<double> local_bkg_nll(num_gpus, 0.0);

        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            const auto& events = events_list[i];
            const auto& amp_offsets = amp_offsets_list[i];
            const auto& events_offsets = events_offsets_list[i];  // 若需要

            // 数据部分
            cuComplex* data_fix = d_all_amplitudes_list[i] + amp_offsets[1];
            double* d_data_nll = nullptr;
            cudaMalloc(&d_data_nll, sizeof(double));
            computeNll(data_fix,
                reinterpret_cast<const cuComplex*>(extended_vec_per_gpu[i].data_ptr()),
                nullptr,           // 数据权重（无）
                d_S_list[i], d_Q_list[i], d_data_nll,
                events[1] * n_polar_, extended_n_gls, n_polar_,
                global_phsp_factor);
            cudaMemcpy(&local_data_nll[i], d_data_nll, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_data_nll);

            // 背景部分
            if (events[2] > 0) {
                cuComplex* bkg_fix = d_all_amplitudes_list[i] + amp_offsets[2];
                // 背景权重指针（假设全局传入，实际应每个 GPU 独立）
                // 这里简化：假设 bkg_weights 已按 GPU 分片，传入一个 std::vector<double*>
                double* bkg_weights = nullptr; // 实际应从外部传入每个 GPU 的权重指针
                computeNll(bkg_fix,
                    reinterpret_cast<const cuComplex*>(extended_vec_per_gpu[i].data_ptr()),
                    bkg_weights,
                    d_bkg_S_list[i], d_bkg_Q_list[i], d_bkg_nll_list[i],
                    events[2] * n_polar_, extended_n_gls, n_polar_,
                    global_phsp_factor);
                cudaMemcpy(&local_bkg_nll[i], d_bkg_nll_list[i], sizeof(double), cudaMemcpyDeviceToHost);
            }
        }

        // 6. 合并 NLL
        double total_data_nll = 0.0, total_bkg_nll = 0.0;
        for (int i = 0; i < num_gpus; ++i) {
            total_data_nll += local_data_nll[i];
            total_bkg_nll += local_bkg_nll[i];
        }
        double loss = total_data_nll - total_bkg_nll;

        ///////////////////
        /////// 计算梯度
        ///////////////////
        torch::Tensor global_extended_grad = torch::zeros({ extended_n_gls }, torch::kComplexFloat).to(vector.device());
        // // 每个 GPU 计算局部梯度并累加
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);

            // 获取该 GPU 的指针
            cuComplex* data_fix = d_all_amplitudes_list[i] + amp_offsets_list[i][1];
            cuComplex* phsp_fix = d_all_amplitudes_list[i] + amp_offsets_list[i][0];

            // 该 GPU 的事件数
            const auto& events = events_list[i];

            // 分配局部扩展梯度
            cuComplex* d_extended_grad = nullptr;
            cudaMalloc(&d_extended_grad, extended_n_gls * sizeof(cuComplex));
            cudaMemset(d_extended_grad, 0, extended_n_gls * sizeof(cuComplex));

            cublasHandle_t cublas_handle;
            cublasCreate(&cublas_handle);

            // 计算数据部分的梯度贡献
            compute_gradient(data_fix, phsp_fix, d_S_list[i], d_Q_list[i], d_B_list[i], nullptr,
                total_mc_amp, extended_n_gls,
                events[1], n_polar_, events[0] * n_polar_,
                d_extended_grad, cublas_handle);

            // 如果有背景，减去背景梯度
            if (events[2] > 0) {
                cuComplex* bkg_fix = d_all_amplitudes_list[i] + amp_offsets_list[i][2];
                cuComplex* d_bkg_extended_grad = nullptr;
                cudaMalloc(&d_bkg_extended_grad, extended_n_gls * sizeof(cuComplex));
                cudaMemset(d_bkg_extended_grad, 0, extended_n_gls * sizeof(cuComplex));

                // 背景权重指针（需从外部传入每个 GPU 的权重）
                double* bkg_weights = nullptr;  // 实际应存储并恢复
                compute_gradient(bkg_fix, phsp_fix, d_bkg_S_list[i], d_bkg_Q_list[i], d_B_list[i],
                    bkg_weights, total_mc_amp, extended_n_gls,
                    events[2], n_polar_, events[0] * n_polar_,
                    d_bkg_extended_grad, cublas_handle);

                const cuComplex minus_one = make_cuComplex(-1.0f, 0.0f);
                cublasCaxpy(cublas_handle, extended_n_gls, &minus_one, d_bkg_extended_grad, 1, d_extended_grad, 1);

                cudaFree(d_bkg_extended_grad);
            }

            // 将局部梯度拷贝到主机/主设备并累加到全局梯度
            torch::Tensor local_grad = torch::empty({ extended_n_gls }, torch::kComplexFloat).to(torch::Device(torch::kCUDA, i));
            cudaMemcpy(local_grad.data_ptr(), d_extended_grad, extended_n_gls * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
            // 将 local_grad 转移到 global_extended_grad 的设备并累加
            global_extended_grad += local_grad.to(global_extended_grad.device());

            cudaFree(d_extended_grad);
            cublasDestroy(cublas_handle);
        }

        // 释放每个 GPU 的临时缓冲区
        for (int i = 0; i < num_gpus; ++i) {
            cudaSetDevice(i);
            cudaFree(d_B_list[i]);
            cudaFree(d_S_list[i]);
            cudaFree(d_Q_list[i]);
            if (events_list[i][2] > 0) {
                cudaFree(d_bkg_S_list[i]);
                cudaFree(d_bkg_Q_list[i]);
                cudaFree(d_bkg_nll_list[i]);
            }
        }

        ctx->save_for_backward({ vector, extended_vec_per_gpu[0] });
        ctx->saved_data["global_extended_grad"] = global_extended_grad;  // 保存全局扩展梯度以供反向传播使用

        return torch::tensor(loss, torch::kDouble).to(vector.device());
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::tensor_list& grad_outputs)
    {
        const auto saved = ctx->get_saved_variables();
        const auto& original_vector = saved[0];
        const auto& extended_vector_template = saved[1];  // 用于获取设备信息等
        const auto& global_extended_grad = ctx->saved_data["global_extended_grad"].toTensor();

        // 合并梯度（考虑约束）
        torch::Tensor grad_vector = mergeGradientsWithConstraints(global_extended_grad, original_vector.numel());

        // 返回梯度，其余输入参数返回空张量
        return { grad_vector,
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor() };
    }

    // 设置约束的静态方法
    static void setConstraints(
        const std::vector<std::vector<int>>& con_trans_id,
        const std::vector<std::vector<std::complex<double>>>& con_trans_values)
    {
        con_trans_id_ = con_trans_id;
        con_trans_values_ = con_trans_values;
        constraints_initialized_ = true;
    }

    static torch::Tensor
        extendVectorWithConstraints(const torch::Tensor& vector,
            const torch::Device& device)
    {
        TORCH_CHECK(vector.is_complex(), "Input vector must be complex type");
        TORCH_CHECK(vector.dim() == 1, "Input vector must be 1-dimensional");

        const int original_size = vector.numel();
        int extended_size = original_size;

        // 找到最大ID以确定扩展大小
        for (const auto& vecid : con_trans_id_)
        {
            if (!vecid.empty())
            {
                auto max_it = std::max_element(vecid.begin(), vecid.end());
                extended_size = std::max(extended_size, *max_it + 1);
            }
        }

        if (extended_size == original_size)
        {
            return vector.clone();
        }

        // 创建扩展后的向量
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kComplexFloat).device(device);

        torch::Tensor extended_vector = torch::zeros({ extended_size }, options);

        // 方法1：使用 PyTorch 的索引操作（在 GPU 上）
        // 创建索引，选择原始部分
        torch::Tensor indices = torch::arange(0, original_size, torch::kLong).to(device);
        extended_vector.index_copy_(0, indices, vector);

        // 在 GPU 上处理约束
        for (size_t i = 0; i < con_trans_id_.size(); ++i)
        {
            const auto& vecid = con_trans_id_[i];
            const auto& values = con_trans_values_[i];

            if (vecid.empty() || values.empty() ||
                vecid.size() != values.size())
            {
                continue;
            }

            // 找到原始ID（最小值）
            auto min_it = std::min_element(vecid.begin(), vecid.end());
            int origin_idx = std::distance(vecid.begin(), min_it);
            int origin_id = vecid[origin_idx];

            // 确保原始ID有效
            if (origin_id < 0 || origin_id >= original_size)
            {
                continue;
            }

            // 获取原始ID对应的系数
            std::complex<double> origin_coeff = values[origin_idx];
            double origin_coeff_real = std::real(origin_coeff);
            double origin_coeff_imag = std::imag(origin_coeff);

            // 检查分母不为零
            if (std::abs(origin_coeff_real) < 1e-10 ||
                std::abs(origin_coeff_imag) < 1e-10)
            {
                std::cerr << "Warning: origin coefficient too small, skipping "
                    "constraint group "
                    << i << std::endl;
                continue;
            }

            // 为每个扩展ID设置值
            for (size_t j = 0; j < vecid.size(); ++j)
            {
                if (j == origin_idx)
                    continue; // 跳过原始ID

                int extended_id = vecid[j];

                // 确保扩展ID有效且不超过values数组的大小
                if (extended_id >= 0 && extended_id < extended_size &&
                    j < values.size())
                {
                    std::complex<double> ext_coeff = values[j];
                    double ext_coeff_real = std::real(ext_coeff);
                    double ext_coeff_imag = std::imag(ext_coeff);

                    // 计算系数比例（直接在 GPU 上）
                    float real_ratio =
                        static_cast<float>(ext_coeff_real / origin_coeff_real);
                    float imag_ratio =
                        static_cast<float>(ext_coeff_imag / origin_coeff_imag);

                    // 获取原始向量的值
                    torch::Tensor origin_value = vector[origin_id];

                    // 计算实部和虚部
                    torch::Tensor real_part = (origin_value + torch::conj(origin_value)) / 2.0f;
                    torch::Tensor imag_part = (origin_value - torch::conj(origin_value)) / (2.0f * c10::complex<float>(0, 1));

                    // 计算扩展值
                    torch::Tensor extended_real = real_ratio * real_part;
                    torch::Tensor extended_imag = imag_ratio * imag_part;

                    // 合并实部和虚部
                    torch::Tensor extended_value = extended_real + c10::complex<float>(0, 1) * extended_imag;

                    // 赋值
                    extended_vector[extended_id] = extended_value;
                }
            }
        }

        return extended_vector;
    }

private:
    static torch::Tensor
        mergeGradientsWithConstraints(const torch::Tensor& extended_grad,
            int original_size)
    {
        torch::Device device = extended_grad.device();
        torch::Tensor grad_vector =
            torch::zeros({ original_size }, torch::kComplexFloat).to(device);

        // 复制原始元素的梯度
        if (original_size > 0)
        {
            grad_vector.copy_(extended_grad.slice(0, 0, original_size));
        }

        // 如果没有约束，直接返回
        if (con_trans_id_.empty())
        {
            return grad_vector;
        }

        // 对于每个约束组
        for (size_t group_idx = 0; group_idx < con_trans_id_.size();
            ++group_idx)
        {
            const auto& vecid = con_trans_id_[group_idx];
            const auto& values = con_trans_values_[group_idx];

            if (vecid.empty() || values.empty() ||
                vecid.size() != values.size())
            {
                continue;
            }

            // 找到原始ID（最小值）
            auto min_it = std::min_element(vecid.begin(), vecid.end());
            int origin_idx = std::distance(vecid.begin(), min_it);
            int origin_id = vecid[origin_idx];

            if (origin_id < 0 || origin_id >= original_size)
            {
                continue;
            }

            std::complex<double> origin_coeff = values[origin_idx];
            double origin_coeff_real = std::real(origin_coeff);
            double origin_coeff_imag = std::imag(origin_coeff);

            if (std::abs(origin_coeff_real) < 1e-10 ||
                std::abs(origin_coeff_imag) < 1e-10)
            {
                continue;
            }

            // 收集该原始元素对应的所有扩展元素
            std::vector<int> ext_indices;
            std::vector<float> real_ratios, imag_ratios;

            for (size_t j = 0; j < vecid.size(); ++j)
            {
                if (j == origin_idx)
                    continue;

                int extended_id = vecid[j];
                if (extended_id < 0 || extended_id >= extended_grad.numel() ||
                    j >= values.size())
                {
                    continue;
                }

                std::complex<double> ext_coeff = values[j];
                real_ratios.push_back(static_cast<float>(std::real(ext_coeff) /
                    origin_coeff_real));
                imag_ratios.push_back(static_cast<float>(std::imag(ext_coeff) /
                    origin_coeff_imag));
                ext_indices.push_back(extended_id);
            }

            if (ext_indices.empty())
                continue;

            // 转换为张量
            torch::Tensor ext_idx_tensor = torch::tensor(ext_indices, torch::kLong).to(device);
            torch::Tensor real_ratio_tensor = torch::tensor(real_ratios, torch::kFloat).to(device);
            torch::Tensor imag_ratio_tensor = torch::tensor(imag_ratios, torch::kFloat).to(device);

            // 获取扩展元素的梯度
            torch::Tensor ext_grads = extended_grad.index_select(0, ext_idx_tensor);

            // 分离实部和虚部
            torch::Tensor ext_grad_real = (ext_grads + torch::conj(ext_grads)) / 2.0f;
            torch::Tensor ext_grad_imag = (ext_grads - torch::conj(ext_grads)) / (2.0f * c10::complex<float>(0, 1));

            // 计算总贡献
            torch::Tensor total_contrib = (real_ratio_tensor * ext_grad_real +
                c10::complex<float>(0, 1) * imag_ratio_tensor * ext_grad_imag).sum();

            // 累加到原始元素
            grad_vector[origin_id] = grad_vector[origin_id] + total_contrib;
        }

        return grad_vector;
    }
};

// 初始化静态成员变量
std::vector<std::vector<int>> NLLFunction::con_trans_id_;
std::vector<std::vector<std::complex<double>>> NLLFunction::con_trans_values_;
bool NLLFunction::constraints_initialized_ = false;

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
class analysis
{
public:
    analysis(const std::string& config_file = "config.yml")
        : config_parser_(config_file), n_amplitudes_(0), n_polar_(0), d_all_amplitudes_()
    {
        initialize();
    }

    // 析构函数，用于释放 CUDA 内存
    ~analysis()
    {
        // if (d_all_amplitudes_ != nullptr)
        // {
        //     cudaFree(d_all_amplitudes_);
        //     d_all_amplitudes_ = nullptr;
        // }
        if (!d_all_amplitudes_.empty())
        {
            // 释放每个GPU的振幅内存
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
                if (d_all_amplitudes_[gpu] != nullptr) {
                    cudaSetDevice(gpu);
                    cudaFree(d_all_amplitudes_[gpu]);
                }
            }
            cudaSetDevice(0);  // 重置默认设备
            d_all_amplitudes_.clear();
        }
    }

    torch::Tensor getNLL(torch::Tensor& vector)
    {
        // return NLLFunction::apply(vector, d_all_amplitudes_, events_tensor_, events_offsets_tensor_, amp_offsets_tensor_, n_amplitudes_, n_polar_);
        return NLLFunction::apply(vector, d_all_amplitudes_, events_, events_offsets_, amp_offsets_, n_amplitudes_, n_polar_);
    }

    int getNVector() const { return n_gls_ - con_trans_id_.size(); }

    torch::Tensor getSLVectors() const
    {
        torch::Device dev(torch::kCUDA, 0);
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt).device(dev);
        return torch::tensor(nSLvectors_, options);
    }

    /*
    void writeResult(torch::Tensor& vector, const std::string& filename, const int is_saved_weight = 0)
    {
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(vector.dtype() == torch::kComplexFloat, "vector must be complex128");

        const int original_size = vector.numel();
        int extended_size = original_size;

        // const int target_dev = vector.get_device();
        torch::Device dev(torch::kCUDA, vector.get_device());

        torch::Tensor extended_vector = NLLFunction::extendVectorWithConstraints(vector, dev);

        const int target_dev = vector.get_device();
        cudaSetDevice(target_dev);

        const int n_events = events_[0]; // 事件数量
        double* d_final_result;
        double* d_partial_result;
        // double *d_partial_sum;

        // 分配设备内存
        cudaMalloc(&d_final_result, n_events * sizeof(double));
        int npartials = nSLvectors_.size();
        cudaMalloc(&d_partial_result, n_events * npartials * sizeof(double));
        // cudaMalloc(&d_partial_sum, npartials * sizeof(double));

        // 分配nSLvectors_的设备内存
        int* d_nSLvectors;
        cudaMalloc(&d_nSLvectors, nSLvectors_.size() * sizeof(int));
        cudaMemcpy(d_nSLvectors, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);
        double* d_total_integral;
        cudaMalloc(&d_total_integral, sizeof(double));
        cudaMemset(d_total_integral, 0, sizeof(double));

        // 分配干涉矩阵和事件干涉项的设备内存
        double* d_interference_matrix;
        cudaMalloc(&d_interference_matrix, npartials * npartials * sizeof(double));
        cudaMemset(d_interference_matrix, 0, npartials * npartials * sizeof(double));
        // double *d_event_interference;
        // cudaMalloc(&d_event_interference, n_events * npartials * npartials *
        // sizeof(double)); cudaMemset(d_event_interference, 0, n_events *
        // npartials
        // * npartials * sizeof(double));

        // 计算权重
        // computeResults(phsp_fix_, reinterpret_cast<const cuComplex
        // *>(extended_vector.data_ptr()), d_final_result, d_total_integral,
        // d_partial_result, d_interference_matrix, nullptr, d_nSLvectors,
        // npartials, n_events, n_gls_, n_polar_); computeResults(phsp_fix_,
        // reinterpret_cast<const cuComplex *>(extended_vector.data_ptr()),
        // d_final_result, d_total_integral, d_partial_result,
        // d_interference_matrix, d_nSLvectors, npartials, n_events, n_gls_,
        // n_polar_);
        computeResults(
            d_all_amplitudes_,
            reinterpret_cast<const cuComplex*>(extended_vector.data_ptr()),
            d_final_result, d_total_integral, d_partial_result,
            d_interference_matrix, d_nSLvectors, npartials, n_events,
            n_amplitudes_, n_polar_);
        // computeWeightResult(phsp_fix_, reinterpret_cast<const cuComplex
        // *>(extended_vector.data_ptr()), d_final_result, d_total_integral,
        // d_partial_result, d_nSLvectors, npartials, n_events, n_gls_,
        // n_polar_);

        double* h_total_results = new double[n_events];
        cudaMemcpy(h_total_results, d_final_result, n_events * sizeof(double), cudaMemcpyDeviceToHost);
        double* h_partial_results = new double[n_events * npartials];
        cudaMemcpy(h_partial_results, d_partial_result, n_events * npartials * sizeof(double), cudaMemcpyDeviceToHost);
        // double *h_partial_sums = new double[npartials];
        // cudaMemcpy(h_partial_sums, d_partial_sum, npartials * sizeof(double),
        // cudaMemcpyDeviceToHost);
        double h_phsp_integral;
        cudaMemcpy(&h_phsp_integral, d_total_integral, sizeof(double), cudaMemcpyDeviceToHost);
        double* h_interference_matrix = new double[npartials * npartials];
        cudaMemcpy(h_interference_matrix, d_interference_matrix, npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);

        // bkg weights积分
        double h_bkg_integral = 0.0;
        // if (bkg_weights_ != nullptr && bkg_length > 0)
        if (bkg_weights_ != nullptr && events_[2] > 0)
        {
            thrust::device_ptr<double> d_ptr(bkg_weights_);
            std::cout << "Calculating background integral with " << events_[2] << " events..." << std::endl;
            h_bkg_integral = thrust::reduce(d_ptr, d_ptr + events_[2]);
            // h_bkg_integral = thrust::reduce(d_ptr, d_ptr + bkg_length /
            // n_polar_, 0.0, thrust::plus<double>());
        }

        // int dataIntegral = data_length / n_polar_;
        // if (bkg_fix_ != nullptr && bkg_length > 0)
        int dataIntegral = events_[1];
        if (events_[2] > 0)
        {
            if (h_bkg_integral > 0)
            {
                dataIntegral -= h_bkg_integral;
            }
            else
            {
                dataIntegral -= events_[2];
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

        // 写入干涉矩阵
        if (h_interference_matrix != nullptr)
        {
            TMatrixD interferenceMatrix(npartials, npartials);
            for (int i = 0; i < npartials; ++i)
            {
                for (int j = i; j < npartials; ++j)
                {
                    int idx = i * npartials - i * (i - 1) / 2 + (j - i);
                    interferenceMatrix[i][j] = h_interference_matrix[idx];
                }
            }

            interferenceMatrix.Write("interference");
        }

        if (is_saved_weight == 1)
        {
            TTree* phspTree =
                new TTree("saved_weight", "fitting result weights");

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

            // 填充 phsp tree
            for (int i = 0; i < n_events; ++i)
            {
                // 设置权重
                total_weight = h_total_results[i] / h_phsp_integral * static_cast<double>(dataIntegral);
                // std::cout << "Event " << i << ": Total Weight = " <<
                // total_weight << std::endl;
                for (int j = 0; j < npartials; ++j)
                {
                    partial_weights[j] =
                        h_partial_results[i * npartials + j] * normFactor;
                    // std::cout << "  Partial Weight " << j << " = " <<
                    // partial_weights[j] << std::endl;
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
            xlabel_obj.Write("xlabel");
            ylabel_obj.Write("ylabel");
        }

        for (const auto& histConfig : anglehist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel");
            ylabel_obj.Write("ylabel");
        }

        for (const auto& histConfig : dalitzhist)
        {
            TDirectory* histDir = rootFile->mkdir(histConfig.name.c_str());
            histDir->cd();

            TObjString xlabel_obj(histConfig.tex[0].c_str());
            TObjString ylabel_obj(histConfig.tex[1].c_str());
            xlabel_obj.Write("xlabel");
            ylabel_obj.Write("ylabel");
        }

        // 四动量index
        std::map<std::string, int> particleToIndex;
        for (int i = 0; i < particles_.size(); ++i)
        {
            particleToIndex[particles_[i].name] = i;
        }

        // 计算并保存data直方图
        if (!Vp4_data_.empty())
        {
            std::vector<TH1F*> masshist_data;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                masshist_data.emplace_back(hist);
            }

            LorentzVector* device_momenta = convertToLorentzVector(Vp4_data_, particleToIndex);
            CalculateMassHist(device_momenta, particleToIndex, masshist, nullptr, masshist_data, Vp4_data_.begin()->second.size(), particleToIndex.size());
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();
                masshist_data[i]->Write("hdata");
                delete masshist_data[i];
            }

            std::vector<TH1F*> anglehist_data;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F(histConfig.name.c_str(), histConfig.title.c_str(), histConfig.bins, histConfig.range[0], histConfig.range[1]);
                anglehist_data.emplace_back(hist);
            }
            CalculateAngleHist(device_momenta, particleToIndex, anglehist, nullptr, anglehist_data, Vp4_data_.begin()->second.size(), particleToIndex.size());
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                anglehist_data[i]->Write("hdata");
                delete anglehist_data[i];
            }

            std::vector<TH2F*> dalitzhist_data;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F(histConfig.name.c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0], histConfig.range[0][1],
                    histConfig.bins[1], histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_data.emplace_back(hist);
            }
            CalculateDalitzHist(device_momenta, particleToIndex, dalitzhist, nullptr, dalitzhist_data, Vp4_data_.begin()->second.size(), particleToIndex.size());
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                dalitzhist_data[i]->Write("hdata");
                delete dalitzhist_data[i];
            }

            // if (device_momenta != nullptr)
            cudaFree(device_momenta);
        }

        // 计算并保存拟合结果直方图
        if (!Vp4_phsp_.empty())
        {
            std::vector<TH1F*> masshist_fit;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F(histConfig.name.c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                masshist_fit.emplace_back(hist);
            }

            LorentzVector* phsp_momenta = convertToLorentzVector(Vp4_phsp_, particleToIndex);
            CalculateMassHist(phsp_momenta, particleToIndex, masshist,
                d_final_result, masshist_fit,
                Vp4_phsp_.begin()->second.size(),
                particleToIndex.size());
            // CalculateMassHist(phsp_momenta, particleToIndex, masshist,
            // h_total_results, masshist_fit, Vp4_phsp_.begin()->second.size(),
            // particleToIndex.size());
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();

                TH1F* hist = masshist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit");
                delete masshist_fit[i];
            }

            std::vector<TH1F*> anglehist_fit;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F(histConfig.name.c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                anglehist_fit.emplace_back(hist);
            }
            CalculateAngleHist(phsp_momenta, particleToIndex, anglehist,
                d_final_result, anglehist_fit,
                Vp4_phsp_.begin()->second.size(),
                particleToIndex.size());
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                TH1F* hist = anglehist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit");
                delete anglehist_fit[i];
            }

            std::vector<TH2F*> dalitzhist_fit;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F(histConfig.name.c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0],
                    histConfig.range[0][1], histConfig.bins[1],
                    histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_fit.emplace_back(hist);
            }
            CalculateDalitzHist(phsp_momenta, particleToIndex, dalitzhist,
                d_final_result, dalitzhist_fit,
                Vp4_phsp_.begin()->second.size(),
                particleToIndex.size());
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                TH2F* hist = dalitzhist_fit[i];
                hist->Scale(normFactor);
                hist->Write("hfit");
                delete dalitzhist_fit[i];
            }

            cudaFree(phsp_momenta);
        }

        // 计算并保存本底直方图
        if (!Vp4_bkg_.empty())
        {
            std::vector<TH1F*> masshist_bkg;
            for (const auto& histConfig : masshist)
            {
                TH1F* hist = new TH1F(histConfig.name.c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                masshist_bkg.emplace_back(hist);
            }

            LorentzVector* device_momenta = convertToLorentzVector(Vp4_bkg_, particleToIndex);
            CalculateMassHist(device_momenta, particleToIndex, masshist,
                bkg_weights_, masshist_bkg,
                Vp4_bkg_.begin()->second.size(),
                particleToIndex.size());
            for (size_t i = 0; i < masshist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(masshist[i].name.c_str());
                histDir->cd();
                masshist_bkg[i]->Write("hbkg");
                delete masshist_bkg[i];
            }

            std::vector<TH1F*> anglehist_bkg;
            for (const auto& histConfig : anglehist)
            {
                TH1F* hist = new TH1F(histConfig.name.c_str(),
                    histConfig.title.c_str(), histConfig.bins,
                    histConfig.range[0], histConfig.range[1]);
                anglehist_bkg.emplace_back(hist);
            }
            CalculateAngleHist(device_momenta, particleToIndex, anglehist,
                bkg_weights_, anglehist_bkg,
                Vp4_bkg_.begin()->second.size(),
                particleToIndex.size());
            for (size_t i = 0; i < anglehist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(anglehist[i].name.c_str());
                histDir->cd();
                anglehist_bkg[i]->Write("hbkg");
                delete anglehist_bkg[i];
            }

            std::vector<TH2F*> dalitzhist_bkg;
            for (const auto& histConfig : dalitzhist)
            {
                TH2F* hist = new TH2F(histConfig.name.c_str(), histConfig.title.c_str(),
                    histConfig.bins[0], histConfig.range[0][0],
                    histConfig.range[0][1], histConfig.bins[1],
                    histConfig.range[1][0], histConfig.range[1][1]);
                dalitzhist_bkg.emplace_back(hist);
            }
            CalculateDalitzHist(device_momenta, particleToIndex, dalitzhist,
                bkg_weights_, dalitzhist_bkg,
                Vp4_bkg_.begin()->second.size(),
                particleToIndex.size());
            for (size_t i = 0; i < dalitzhist.size(); ++i)
            {
                TDirectory* histDir = rootFile->GetDirectory(dalitzhist[i].name.c_str());
                histDir->cd();
                dalitzhist_bkg[i]->Write("hbkg");
                delete dalitzhist_bkg[i];
            }

            cudaFree(device_momenta);
        }

        //
        if (!Vp4_phsp_.empty())
        {
            LorentzVector* device_momenta = convertToLorentzVector(Vp4_phsp_, particleToIndex);
            for (int i = 0; i < npartials; ++i)
            {
                // 为当前部分创建直方图
                // std::vector<TH1F *> partialHists;
                std::vector<TH1F*> masshist_partial;
                for (const auto& histConfig : masshist)
                {
                    TH1F* hist = new TH1F(histConfig.name.c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    masshist_partial.push_back(hist);
                }

                CalculateMassHist(
                    device_momenta, particleToIndex, masshist,
                    &d_partial_result[i * n_events], masshist_partial,
                    Vp4_phsp_.begin()->second.size(), particleToIndex.size());
                for (size_t j = 0; j < masshist_partial.size(); ++j)
                {

                    TDirectory* histDir = rootFile->GetDirectory(masshist[j].name.c_str());
                    histDir->cd();

                    // std::string partial_dir_name = "h_" +
                    // amplitude_names_[i];
                    std::string partial_dir_name = "h_" + resonance_names_[i];

                    TH1F* hist = masshist_partial[j];
                    hist->Scale(normFactor);
                    hist->Write(partial_dir_name.c_str());

                    delete hist;
                }
                masshist_partial.clear();

                std::vector<TH1F*> anglehist_partial;
                for (const auto& histConfig : anglehist)
                {
                    TH1F* hist = new TH1F(histConfig.name.c_str(),
                        histConfig.title.c_str(), histConfig.bins,
                        histConfig.range[0], histConfig.range[1]);
                    anglehist_partial.push_back(hist);
                }
                CalculateAngleHist(
                    device_momenta, particleToIndex, anglehist,
                    &d_partial_result[i * n_events], anglehist_partial,
                    Vp4_phsp_.begin()->second.size(), particleToIndex.size());
                for (size_t j = 0; j < anglehist_partial.size(); ++j)
                {
                    TDirectory* histDir = rootFile->GetDirectory(anglehist[j].name.c_str());
                    histDir->cd();

                    std::string partial_dir_name = "h_" + resonance_names_[i];

                    TH1F* hist = anglehist_partial[j];
                    hist->Scale(normFactor);
                    hist->Write(partial_dir_name.c_str());
                    delete anglehist_partial[j];
                }
                anglehist_partial.clear();
            }
            cudaFree(device_momenta);
        }

        // 关闭 ROOT 文件
        rootFile->Close();
        delete rootFile;

        // std::cout << "Data written to ROOT file: " << filename << std::endl;

        // 释放设备内存
        // cudaFree(d_row_results);
        cudaFree(d_final_result);
        cudaFree(d_partial_result);
        cudaFree(d_nSLvectors);
        cudaFree(d_total_integral);
        cudaFree(d_interference_matrix);
        // cudaFree(d_event_interference);
        delete[] h_total_results;
        delete[] h_partial_results;
    }

    torch::Tensor getDataTensor() const
    {
        // torch::Tensor output = torch::from_blob(data_fix_, {data_length *
        // n_gls_},
        // torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_ + amp_offsets_[1],
            { events_[1] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getPhspTensor() const
    {
        // torch::Tensor output = torch::from_blob(phsp_fix_, {phsp_length *
        // n_gls_},
        // torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_,
            { events_[0] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getTruthTensor() const
    {
        const auto& data_files = config_parser_.getDataFiles();
        if (data_files.count("phsp_truth") > 0)
        {
            std::vector<std::string> particles_names;
            for (const auto& particle : particles_)
            {
                particles_names.push_back(particle.name);
            }

            // std::cout << "Reading phase space truth samples..." << std::endl;
            std::map<std::string, std::vector<LorentzVector>> Vp4_truth =
                readMomentaFromDat(data_files.at("phsp_truth"),
                    config_parser_.getDataOrder(),
                    particles_names);
            std::cout << "Phase space truth events: "
                << Vp4_truth.begin()->second.size() << std::endl;
            // std::cout << "Calculating phase space truth amplitudes..." <<
            // std::endl;
            cuComplex* truth_fix = calculateAmplitudes(Vp4_truth, { 0 }, { 0 });
            int truth_length = Vp4_truth.begin()->second.size() * n_polar_;

            torch::Tensor output = torch::from_blob(truth_fix,
                { truth_length * n_gls_ },
                torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

            cudaFree(truth_fix);

            return output;
        }
        else
        {
            std::cerr
                << "No phsp_truth data file specified in the configuration."
                << std::endl;
            return torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA));
        }
    }

    torch::Tensor getBkgTensor() const
    {
        // torch::Tensor output = torch::from_blob(bkg_fix_, {bkg_length *
        // n_gls_},
        // torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_ + amp_offsets_[2],
            { events_[2] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getBkgWeightsTensor() const
    {
        // if (bkg_weights_ != nullptr && bkg_length > 0)
        if (bkg_weights_ != nullptr && events_[2] > 0)
        {
            torch::Tensor output = torch::from_blob(bkg_weights_, { events_[2] },
                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)).clone();
            return output;
        }
        else
        {
            return torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        }
    }
    */

    std::vector<std::vector<int>> getConstraintsIndex() const
    {
        return con_trans_id_;
    }

    std::vector<std::vector<std::pair<double, double>>>
        getConstraintsValues() const
    {
        std::vector<std::vector<std::pair<double, double>>> output;
        for (const auto& vec : con_trans_values_)
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

    std::vector<std::string> getAmplitudeNames() const
    {
        return amplitude_names_;
    }

private:
    int n_gls_;
    int n_polar_ = 1;
    std::vector<int> nSLvectors_;

    // 振幅数据，设备端
    std::vector<cuComplex*> d_all_amplitudes_;
    std::vector<double*> phsp_weights_;// = nullptr;
    std::vector<double*> bkg_weights_;// = nullptr;

    // 事件数量、振幅偏移等信息
    int n_gpus_ = 0;
    std::vector<std::vector<int>> events_;
    std::vector<std::vector<int>> events_offsets_;
    std::vector<std::vector<int>> amp_offsets_;
    // torch::Tensor events_tensor_;
    // torch::Tensor events_offsets_tensor_;
    // torch::Tensor amp_offsets_tensor_;

    // 四动量数据，主机端
    std::map<std::string, std::vector<LorentzVector>> Vp4_data_;
    std::map<std::string, std::vector<LorentzVector>> Vp4_phsp_;
    std::map<std::string, std::vector<LorentzVector>> Vp4_bkg_;
    // std::map<std::string, std::vector<LorentzVector>> Vp4_truth_;

    // amplitude 信息
    std::vector<std::string> amplitude_names_;
    std::vector<std::string> resonance_names_;
    std::vector<std::string> legends_;

    // 约束信息
    std::vector<std::vector<int>> con_trans_id_;
    std::vector<std::vector<std::complex<double>>> con_trans_values_;

    // config 初始化
    ConfigParser config_parser_;
    std::vector<Particle> particles_;
    std::unordered_map<std::string, Resonance> resonances_;
    int n_amplitudes_ = 0;
    std::vector<ChainInfo> chains_info_;

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
        Vp4_phsp_ = readMomentaFromDat(data_files.at("phsp"), data_order,
            particles_names);
        std::cout << "Phase space events: " << Vp4_phsp_.begin()->second.size()
            << std::endl;
        // std::cout << "Calculating phase space amplitudes..." << std::endl;
        // phsp_fix_ = calculateAmplitudes(Vp4_phsp_, {0,
        // static_cast<int>(Vp4_phsp_.begin()->second.size())},
        // {0, static_cast<int>(Vp4_phsp_.begin()->second.size() * n_amplitudes_
        // * n_polar_)}); phsp_length = Vp4_phsp_.begin()->second.size() *
        // n_polar_;
        init_events.push_back(Vp4_phsp_.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_phsp_);

        // 计算数据振幅
        std::cout << "Reading data samples..." << std::endl;
        Vp4_data_ = readMomentaFromDat(data_files.at("data"), data_order, particles_names);
        std::cout << "data events: " << Vp4_data_.begin()->second.size() << std::endl;
        // std::cout << "Calculating data amplitudes..." << std::endl;
        // data_fix_ = calculateAmplitudes(Vp4_data_, {0,
        // static_cast<int>(Vp4_data_.begin()->second.size())},
        // {0, static_cast<int>(Vp4_data_.begin()->second.size() * n_amplitudes_
        // * n_polar_)}); data_length = Vp4_data_.begin()->second.size() *
        // n_polar_;
        init_events.push_back(Vp4_data_.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_data_);

        // 计算本底振幅
        if (data_files.count("bkg") > 0)
        {
            std::cout << "Reading background samples..." << std::endl;
            Vp4_bkg_ = readMomentaFromDat(data_files.at("bkg"), data_order, particles_names);
            std::cout << "Background events: " << Vp4_bkg_.begin()->second.size() << std::endl;
            // std::cout << "Calculating background amplitudes..." << std::endl;
            // bkg_fix_ = calculateAmplitudes(Vp4_bkg_, {0,
            // static_cast<int>(Vp4_bkg_.begin()->second.size())},
            // {0, static_cast<int>(Vp4_bkg_.begin()->second.size() *
            // n_amplitudes_ * n_polar_)}); bkg_length =
            // Vp4_bkg_.begin()->second.size() * n_polar_;
            init_events.push_back(Vp4_bkg_.begin()->second.size());
            Vp4_to_merge.push_back(Vp4_bkg_);

            // if (data_files.count("bkg_weights") > 0)
            // {
            //     bkg_weights_ = readWeightsFromFile(data_files.at("bkg_weights"), Vp4_bkg_.begin()->second.size());
            // }
        }

        CUDA_CHECK(cudaGetDeviceCount(&n_gpus_));
        initializeMultiGPUs(init_events);

        std::cout << "Calculating amplitudes..." << std::endl;
        auto Vp4_all = mergeMaps(Vp4_to_merge, events_);
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
        // for (size_t i = 0; i < events_offsets_.size(); ++i)
        // {
        //     std::cout << "Dataset " << i << " event offsets: ";
        //     for (const auto& offset : events_offsets_[i])
        //     {
        //         std::cout << offset << " ";
        //     }
        //     std::cout << std::endl;

        //     std::cout << "Dataset " << i << " amplitude offsets: ";
        //     for (const auto& offset : amp_offsets_[i])
        //     {
        //         std::cout << offset << " ";
        //     }
        //     std::cout << std::endl;
        // }
        d_all_amplitudes_ = calculateAmplitudes(Vp4_all);

        NLLFunction::setConstraints(con_trans_id_, con_trans_values_);
        // int n_vp4 = events_[0].size();
        // std::vector<int> flat;
        // for (const auto& row : events_) {
        //     flat.insert(flat.end(), row.begin(), row.end());
        // }
        // events_tensor_ = torch::from_blob(flat.data(), { n_gpus_, n_vp4 }, torch::kInt).clone();
        // std::vector<int> flat_offsets;
        // for (const auto& row : events_offsets_) {
        //     flat_offsets.insert(flat_offsets.end(), row.begin(), row.end());
        // }
        // events_offsets_tensor_ = torch::from_blob(flat_offsets.data(), { n_gpus_, n_vp4 + 1 }, torch::kInt).clone();
        // std::vector<int> flat_amp_offsets;
        // for (const auto& row : amp_offsets_) {
        //     flat_amp_offsets.insert(flat_amp_offsets.end(), row.begin(), row.end());
        // }
        // amp_offsets_tensor_ = torch::from_blob(flat_amp_offsets.data(), { n_gpus_, n_vp4 + 1 }, torch::kInt).clone();

        std::cout << "Number of GPUs available: " << n_gpus_ << std::endl;
        std::cout << "Number of partial waves: " << n_gls_ << std::endl;
        // std::cout << "Number of amplitude names: " << amplitude_names_.size() << std::endl;
        // std::cout << "Number of constraints: " << conjugate_pairs_.size() <<
        // std::endl;
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
        n_polar_ = 1;

        for (const auto& particle : particles_)
        {
            if (particle.spin > 0)
            {
                // n_polar_ *= (2 * particle.spin + 1);
                n_polar_ *= particle.spin;
            }
        }

        std::cout << "polarization: " << n_polar_ << std::endl;
    }

    void initializeDecayChains()
    {
        auto chains = config_parser_.getDecayChains();

        const auto& config_resonances = config_parser_.getResonances();

        // 获取总振幅长度
        for (const auto& chain : chains)
        {
            // std::cout << "Processing decay chain: " << chain.name <<
            // std::endl;

            ChainInfo chain_info;
            chain_info.name = chain.name;

            std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>>               intermediate_resonance_map;
            std::vector<std::vector<Particle>> intermediate_particles;
            for (const auto& res_chain : chain.resonance_chains)
            {
                // std::cout << "  Intermediate: " << res_chain.intermediate <<
                // std::endl;
                std::vector<Particle> particles;
                for (const auto& spin_chain : res_chain.spin_chains)
                {
                    // std::cout << "    Spin-Parity options: ";

                    Particle intermediate_particle = { res_chain.intermediate, static_cast<int>(spin_chain.spin_parity[0]),
                         static_cast<int>(spin_chain.spin_parity[1]), -1 };

                    std::vector<Resonance> resonance_list;
                    // std::cout << " Resonance: " << std::endl;
                    for (const auto& resonance_name : spin_chain.resonances)
                    {
                        // std::cout << resonance_name << std::endl;

                        for (const auto& [name, res_config] : config_resonances)
                        {
                            if (name == resonance_name)
                            {
                                if (res_config.J == spin_chain.spin_parity[0] &&
                                    spin_chain.spin_parity[1])
                                {
                                    resonance_list.emplace_back(
                                        name, res_chain.intermediate,
                                        intermediate_particle.spin,
                                        intermediate_particle.parity,
                                        res_config.type, res_config.parameters);
                                    // std::cout << "      Added resonance: " <<
                                    // name << " J: " << res_config.J << " P: "
                                    // << res_config.P << std::endl;
                                }
                                else
                                {
                                    std::cout << "      Skipped resonance (J,P "
                                        "mismatch): "
                                        << name << " J: " << res_config.J
                                        << " P: " << res_config.P
                                        << std::endl;
                                }
                            }
                        }
                    }
                    // std::cout << std::endl;

                    std::pair<std::string, std::vector<int>> key = {
                        res_chain.intermediate,
                        {spin_chain.spin_parity[0], spin_chain.spin_parity[1]} };
                    intermediate_resonance_map[key] = resonance_list;
                    particles.push_back(intermediate_particle);
                }
                intermediate_particles.push_back(particles);
            }

            chain_info.intermediate_resonance_map = intermediate_resonance_map;

            std::vector<std::vector<Particle>> intermediate_combs = { {} };
            for (const auto& particleList : intermediate_particles)
            {
                std::vector<std::vector<Particle>> temp;
                for (const auto& comb : intermediate_combs)
                {
                    for (const auto& particle : particleList)
                    {
                        std::vector<Particle> new_res = comb;
                        // std::cout << " Adding particle to combination: " <<
                        // particle.name
                        // << " J: " << particle.spin << " P: " <<
                        // particle.parity << std::endl;
                        new_res.push_back(particle);
                        temp.push_back(new_res);
                    }
                }
                intermediate_combs = std::move(temp);
            }

            chain_info.intermediate_combs = intermediate_combs;

            for (auto comb : intermediate_combs)
            {
                AmpCasDecay cas(particles_);
                for (const auto& step : chain.decay_steps)
                {
                    std::array<int, 3> spins = { 0 };
                    std::array<int, 3> parities = { 0 };
                    for (auto particle : particles_)
                    {
                        if (particle.name == step.mother)
                        {
                            // std::cout << "mother: " << particle.name << " "
                            // << particle.spin << " " << particle.parity <<
                            // std::endl;
                            spins[0] = particle.spin;
                            parities[0] = particle.parity;
                        }

                        for (int i = 0; i < step.daughters.size(); i++)
                        {
                            if (particle.name == step.daughters[i])
                            {
                                // std::cout << "daugters: " << particle.name <<
                                // " " << particle.spin << " " <<
                                // particle.parity << std::endl;
                                spins[i + 1] = particle.spin;
                                parities[i + 1] = particle.parity;
                            }
                        }
                    }
                    for (auto res_jp : comb)
                    {
                        if (res_jp.name == step.mother)
                        {
                            // std::cout << "mother: " << res_jp.name << " " <<
                            // res_jp.spin << std::endl;
                            spins[0] = res_jp.spin;
                            parities[0] = res_jp.parity;
                        }

                        for (int i = 0; i < step.daughters.size(); i++)
                        {
                            if (res_jp.name == step.daughters[i])
                            {
                                // std::cout << "daugters: " << res_jp.name << "
                                // " << res_jp.spin << " " << res_jp.parity <<
                                // std::endl;
                                spins[i + 1] = res_jp.spin;
                                parities[i + 1] = res_jp.parity;
                            }
                        }
                    }
                    cas.addDecay(Amp2BD(spins, parities), step.mother,
                        step.daughters[0], step.daughters[1]);

                    // 输出decay chain结构
                    std::cout << step.mother << "(";
                    if (spins[0] % 2 != 0)
                        std::cout << (spins[0] - 1) / 2;
                    else
                        std::cout << spins[0] - 1 << "/2";
                    if (parities[0] == 1)
                        std::cout << "+)";
                    else if (parities[0] == -1)
                        std::cout << "-)";
                    std::cout << "->";
                    for (int i = 0; i < step.daughters.size(); i++)
                    {
                        std::cout << step.daughters[i] << "(";
                        if (spins[i + 1] % 2 != 0)
                            std::cout << (spins[i + 1] - 1) / 2;
                        else
                            std::cout << spins[i + 1] - 1 << "/2";
                        if (parities[i + 1] == 1)
                            std::cout << "+)";
                        else if (parities[i + 1] == -1)
                            std::cout << "-)";
                    }
                    std::cout << ", ";
                }

                auto slcombs = cas.getSLCombinations();
                std::cout << "SL:";
                for (auto slcomb : slcombs)
                {
                    std::cout << "{";
                    for (auto sl : slcomb)
                    {
                        std::cout << "(" << (sl.S - 1) / 2.0 << ", " << sl.L << ")";
                    }
                    std::cout << "}";
                }
                std::cout << std::endl;

                std::vector<std::vector<std::pair<std::string, std::string>>>
                    resonance_combinations = { {} };
                for (const auto& particle : comb)
                {
                    std::pair<std::string, std::vector<int>> key = {
                        particle.name, {particle.spin, particle.parity} };
                    const auto& resonance_list =
                        intermediate_resonance_map[key];

                    std::vector<
                        std::vector<std::pair<std::string, std::string>>>
                        temp;
                    for (const auto& res_comb : resonance_combinations)
                    {
                        for (const auto& resonance : resonance_list)
                        {
                            std::vector<std::pair<std::string, std::string>>
                                new_res_comb = res_comb;
                            new_res_comb.push_back(
                                { resonance.getTag(), resonance.getName() });
                            temp.push_back(new_res_comb);
                        }
                    }
                    resonance_combinations = std::move(temp);
                }

                std::cout << "Resonance: ";
                for (size_t k = 0; k < resonance_combinations.size(); ++k)
                {
                    n_amplitudes_ += slcombs.size();
                    nSLvectors_.push_back(slcombs.size());

                    std::string res_name = chain.name;
                    std::cout << "{ ";
                    for (size_t j = 0; j < resonance_combinations[k].size();
                        ++j)
                    {
                        const auto& res_pair = resonance_combinations[k][j];
                        res_name += "-" + res_pair.first + "-" + res_pair.second;

                        std::cout << res_pair.second; // 共振态名称
                        if (j < resonance_combinations[k].size() - 1)
                            std::cout << ", ";
                    }
                    if (k < resonance_combinations.size() - 1)
                        std::cout << " }, ";
                    else
                        std::cout << "}";

                    for (const auto& slcomb : slcombs)
                    {
                        std::string full_name = res_name + "-" + "SL";
                        for (const auto& sl : slcomb)
                        {
                            full_name += "_" + std::to_string(sl.S) + std::to_string(sl.L);
                        }
                        amplitude_names_.push_back(full_name);
                    }
                    resonance_names_.push_back(res_name);
                }
                std::cout << std::endl;
            }

            chains_info_.push_back(chain_info);
        }

        // 设置约束条件
        auto constraints = config_parser_.getConstraints();

        for (const auto& constraint : constraints)
        {
            std::vector<std::vector<int>> amp_ids_con;

            for (const auto& amp_name : constraint.names)
            {
                std::vector<int> amp_ids;
                for (int i = 0; i < amplitude_names_.size(); ++i)
                {
                    if (amplitude_names_[i].find(amp_name) != std::string::npos)
                    {
                        amp_ids.push_back(i);
                    }
                }
                amp_ids_con.push_back(amp_ids);
            }

            // 生成所有组合
            int num_constraints = amp_ids_con.size();
            for (int i = 0; i < amp_ids_con[0].size(); ++i)
            {
                // // 先输出amp_ids_con内容和amplitude_names_对应关系，便于调试

                std::vector<int> combination;
                for (int j = 0; j < num_constraints; ++j)
                {
                    combination.push_back(amp_ids_con[j][i]);
                }
                // all_combinations.push_back(combination);
                con_trans_id_.push_back(combination);

                // 第一个是{1+1j}, 后面是constraint.values
                std::vector<std::complex<double>> values = {
                    std::complex<double>(1.0, 1.0) };
                for (const auto& val : constraint.values)
                {
                    values.push_back(val);
                }
                // con_values.push_back(values);
                con_trans_values_.push_back(values);
            }

        }
    }

    std::vector<cuComplex*> calculateAmplitudes(const std::vector<std::map<std::string, std::vector<LorentzVector>>>& Vp4) const
    {
        // 多GPU支持：为每个GPU分配振幅内存
        // int num_gpus = events_.size();
        std::vector<cuComplex*> d_all_amplitudes_vec(n_gpus_, nullptr);

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
                cudaMalloc(&d_all_amplitudes_vec[gpu], amp_offsets_[gpu].back() * sizeof(cuComplex));
            }
        }
        // cudaSetDevice(0);

        // std::cout << "Calculating amplitudes for " << num_gpus << " GPUs..." << std::endl;

        auto chains = config_parser_.getDecayChains();
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

            for (auto comb : intermediate_combs)
            {
                AmpCasDecay cas(particles_);
                cas.setNPolarizations(n_polar_);
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
                    cas.addDecay(Amp2BD(spins, parities), step.mother,
                        step.daughters[0], step.daughters[1]);
                }

                auto slcombs = cas.getSLCombinations();

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

                // 计算SL振幅（多GPU版本）
                cas.computeSLAmps(Vp4);
                // int nSLcombs = cas.getNSLCombs();
                // int nEvents = cas.getNEvents();

                // 调用多GPU版本的getAmps函数
                for (const auto resonance : resonance_combinations)
                {
                    cas.getAmps(d_all_amplitudes_vec, resonance, gls_index, events_offsets_, amp_offsets_);
                    gls_index += cas.getNSLCombs();
                }
            }
        }

        return d_all_amplitudes_vec;
    }

};

// 定义Python模块
PYBIND11_MODULE(ctpwa, m)
{
    m.doc() = "ctpwa";

    pybind11::class_<analysis>(m, "analysis")
        .def(pybind11::init<>())
        .def("getNLL", &analysis::getNLL)
        .def("getNVector", &analysis::getNVector)
        .def("getSLVectors", &analysis::getSLVectors)
        // .def("writeResult", &analysis::writeResult)
        // .def("getDataTensor", &analysis::getDataTensor)
        // .def("getPhspTensor", &analysis::getPhspTensor)
        // .def("getTruthTensor", &analysis::getTruthTensor)
        // .def("getBkgTensor", &analysis::getBkgTensor)
        // .def("getBkgWeightsTensor", &analysis::getBkgWeightsTensor)
        .def("getConstraintsIndex", &analysis::getConstraintsIndex)
        .def("getConstraintsValues", &analysis::getConstraintsValues)
        .def("getAmplitudeNames", &analysis::getAmplitudeNames)
        .def("getNPolarizations", &analysis::getNPolarizations);
}
