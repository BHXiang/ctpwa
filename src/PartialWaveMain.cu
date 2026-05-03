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
#include <AutoDiff.cuh>
#include <ComputeHessian.cuh>
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
struct ChainInfo
{
    std::string name;
    std::map<std::pair<std::string, std::vector<int>>, std::vector<Resonance>>
        intermediate_resonance_map;
    std::vector<std::vector<Particle>> intermediate_combs;
};

////////////////////////////////////////
std::map<std::string, std::vector<LorentzVector>> readMomentaFromDat(
    const std::vector<std::string>& fileinfo,
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

std::vector<double*> readWeightsFromFile(const std::vector<std::string>& fileinfo, const std::vector<int>& events_per_gpu)
{
    // 检查输入参数
    if (fileinfo.size() < 2)
    {
        std::cerr << "Error: fileinfo must contain at least file type and filename" << std::endl;
        return {};
    }

    // 读取所有权重到主机内存
    std::vector<double> weights;
    std::string fileType = fileinfo[0];
    std::string filename = fileinfo[1];

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
    // 多 GPU 前向传播
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor vector,
        std::vector<cuComplex*>& d_all_amplitudes_list,
        cuComplex* d_phsp_matrix_,
        const std::vector<std::vector<int>>& events_list,
        const std::vector<std::vector<int>>& events_offsets_list,
        const std::vector<std::vector<int>>& amp_offsets_list,
        std::vector<double*>& d_phsp_weights_list,
        std::vector<double*>& d_bkg_weights_list,
        double bkg_integral_,
        int n_amplitudes_,
        int n_polar_)
    {
        int num_gpus = d_all_amplitudes_list.size();
        TORCH_CHECK(num_gpus > 0, "No GPUs provided");
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(constraints_initialized_, "Constraints not initialized");

        // 1. 将 vector 扩展向量
        torch::Tensor extended_vector = extendVectorWithConstraints(vector, vector.device());
        int extended_n_gls = extended_vector.numel();
        const int primary_dev = vector.get_device();

        // 2. 将extended_vector分发到每个GPU（实际拷贝）
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < num_gpus; ++i) {
            extended_vec_per_gpu.push_back(
                extended_vector.to(torch::Device(torch::kCUDA, i)));
        }

        // 3. 全局量: d_P_vec和phsp_factor（小矩阵M<100，用自定义核替代cuBLAS更快）
        cudaSetDevice(primary_dev);
        cuComplex* d_P_vec;
        cudaMalloc(&d_P_vec, n_amplitudes_ * sizeof(cuComplex));
        float* d_phsp_r, * d_phsp_i;
        cudaMalloc(&d_phsp_r, sizeof(float));
        cudaMalloc(&d_phsp_i, sizeof(float));
        {
            torch::Tensor extended_vector_conj = extended_vector.conj();
            const cuComplex* d_vec_conj = reinterpret_cast<const cuComplex*>(extended_vector_conj.data_ptr());
            computeQuadraticForm(d_phsp_matrix_, d_vec_conj, d_P_vec,
                d_phsp_r, d_phsp_i, n_amplitudes_);
        }
        float phsp_r, phsp_i;
        cudaMemcpy(&phsp_r, d_phsp_r, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phsp_i, d_phsp_i, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_phsp_r);
        cudaFree(d_phsp_i);
        double phsp_factor = static_cast<double>(phsp_r);

        // std::cout << "PHSP factor: " << phsp_factor << std::endl;
        // //输出 d_P_vec 检查
        // std::vector<cuComplex> h_P_vec(n_amplitudes_);
        // cudaMemcpy(h_P_vec.data(), d_P_vec, n_amplitudes_ * sizeof(cuComplex),
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

        cuComplex* d_grad_global;
        cudaSetDevice(primary_dev);
        cudaMalloc(&d_grad_global, extended_n_gls * sizeof(cuComplex));
        cudaMemset(d_grad_global, 0, extended_n_gls * sizeof(cuComplex));

        // P2P临时缓冲区（复用）
        cuComplex* d_grad_buf;
        cudaSetDevice(primary_dev);
        cudaMalloc(&d_grad_buf, extended_n_gls * sizeof(cuComplex));

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);
            const cuComplex* d_vec_gpu = reinterpret_cast<const cuComplex*>(
                extended_vec_per_gpu[gpu].data_ptr());

            // --- data ---
            int nData_gpu = events_list[gpu][1];
            if (nData_gpu > 0) {
                cuComplex* d_amp = d_all_amplitudes_list[gpu] + amp_offsets_list[gpu][1];
                cuComplex* d_grad;
                cudaMalloc(&d_grad, extended_n_gls * sizeof(cuComplex));

                double nll = computeFactorNLL(d_amp, d_vec_gpu,
                    d_grad, nData_gpu, n_polar_, n_amplitudes_);

                total_data_nll += nll;
                totalDataEvents += nData_gpu;
                // P2P: d_grad → d_grad_global (正号)
                if (gpu == primary_dev) {
                    cuComplex one = make_cuComplex(1.0f, 0.0f);
                    cublasHandle_t h; cublasCreate(&h);
                    cublasCaxpy(h, n_amplitudes_, &one, d_grad, 1, d_grad_global, 1);
                    cublasDestroy(h);
                }
                else {
                    cudaMemcpyPeer(d_grad_buf, primary_dev, d_grad, gpu,
                        extended_n_gls * sizeof(cuComplex));
                    cudaSetDevice(primary_dev);
                    cuComplex one = make_cuComplex(1.0f, 0.0f);
                    cublasHandle_t h; cublasCreate(&h);
                    cublasCaxpy(h, n_amplitudes_, &one, d_grad_buf, 1, d_grad_global, 1);
                    cublasDestroy(h);
                }
                cudaFree(d_grad);
            }

            // --- bkg ---
            int nBkg_gpu = events_list[gpu][2];
            if (nBkg_gpu > 0) {
                cuComplex* d_amp = d_all_amplitudes_list[gpu] + amp_offsets_list[gpu][2];
                const double* d_w = d_bkg_weights_list[gpu];  // nullptr=无权重
                cuComplex* d_grad;
                cudaMalloc(&d_grad, extended_n_gls * sizeof(cuComplex));

                double nll = computeFactorNLL(d_amp, d_vec_gpu,
                    d_grad, nBkg_gpu, n_polar_, n_amplitudes_, d_w);

                total_bkg_nll += nll;
                // P2P: d_grad → d_grad_global (负号，loss = data - bkg)
                if (gpu == primary_dev) {
                    cuComplex neg_one = make_cuComplex(-1.0f, 0.0f);
                    cublasHandle_t h; cublasCreate(&h);
                    cublasCaxpy(h, n_amplitudes_, &neg_one, d_grad, 1, d_grad_global, 1);
                    cublasDestroy(h);
                }
                else {
                    cudaMemcpyPeer(d_grad_buf, primary_dev, d_grad, gpu,
                        extended_n_gls * sizeof(cuComplex));
                    cudaSetDevice(primary_dev);
                    cuComplex neg_one = make_cuComplex(-1.0f, 0.0f);
                    cublasHandle_t h; cublasCreate(&h);
                    cublasCaxpy(h, n_amplitudes_, &neg_one, d_grad_buf, 1, d_grad_global, 1);
                    cublasDestroy(h);
                }
                cudaFree(d_grad);
            }
        }
        cudaFree(d_grad_buf);

        // // 输出d_grad_global检查
        // std::vector<cuComplex> h_grad_global(extended_n_gls);
        // cudaMemcpy(h_grad_global.data(), d_grad_global, extended_n_gls * sizeof(cuComplex),
        //     cudaMemcpyDeviceToHost);
        // std::cout << "Global gradient: ";
        // for (int i = 0; i < extended_n_gls; ++i) {
        //     std::cout << "(" << h_grad_global[i].x << ", " << h_grad_global[i].y << ") ";
        // }
        // std::cout << std::endl;

        // // 输出d_vec_gpu检查
        // std::vector<cuComplex> h_vec_gpu(extended_n_gls);
        // cudaMemcpy(h_vec_gpu.data(), extended_vec_per_gpu[0].data_ptr(),
        //     extended_n_gls * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        // std::cout << "Extended vector (GPU 0): ";
        // for (int i = 0; i < extended_n_gls; ++i) {
        //     std::cout << "(" << h_vec_gpu[i].x << ", " << h_vec_gpu[i].y << ") ";
        // }
        // std::cout << std::endl;

        // 5. 合并data和bkg的NLL与梯度（loss = data_nll - bkg_nll）
        if (phsp_factor <= 1e-30 || isnan(phsp_factor) || isinf(phsp_factor)) {
            std::cerr << "WARNING: phsp_factor invalid (" << phsp_factor
                << "), clamping to 1e-30" << std::endl;
            phsp_factor = 1e-30;
        }
        double loss = total_data_nll - total_bkg_nll
            + (totalDataEvents - bkg_integral_) * log(phsp_factor);
        if (isnan(loss) || isinf(loss)) {
            std::cerr << "WARNING: loss is " << (isnan(loss) ? "NaN" : "Inf")
                << ", resetting to 1e30" << std::endl;
            loss = 1e30;
            cudaMemset(d_grad_global, 0, extended_n_gls * sizeof(cuComplex));
        }
        cuComplex scale_phsp = make_cuComplex(
            static_cast<float>(totalDataEvents - bkg_integral_) / static_cast<float>(phsp_factor), 0.0f);
        {
            cublasHandle_t h;
            cublasCreate(&h);
            cublasCaxpy(h, n_amplitudes_, &scale_phsp, d_P_vec, 1, d_grad_global, 1);
            cublasDestroy(h);
        }
        cudaFree(d_P_vec);

        // 6. 保存梯度和loss到ctx
        cudaSetDevice(primary_dev);
        torch::Tensor global_extended_grad = torch::empty({ extended_n_gls },
            torch::kComplexFloat).to(vector.device());
        cudaMemcpy(global_extended_grad.data_ptr(), d_grad_global,
            extended_n_gls * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
        cudaFree(d_grad_global);

        // // 输出d_grad_global检查
        // std::vector<cuComplex> h_grad_global(extended_n_gls);
        // cudaMemcpy(h_grad_global.data(), global_extended_grad.data_ptr(),
        //     extended_n_gls * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        // std::cout << "Global gradient (after copy to tensor): ";
        // for (int i = 0; i < extended_n_gls; ++i) {
        //     std::cout << "(" << h_grad_global[i].x << ", " << h_grad_global[i].y << ") ";
        // }
        // std::cout << std::endl;

        // global_extended_grad = global_extended_grad.conj(); // 转置共轭以匹配PyTorch的梯度定义

        ctx->save_for_backward({ vector, extended_vec_per_gpu[0] });
        ctx->saved_data["global_extended_grad"] = global_extended_grad;

        // std::cout << "Total data NLL: " << loss << std::endl;
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
                torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor() };
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
        if (!d_all_amplitudes_.empty())
        {
            // 释放每个GPU的振幅内存
            for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
                if (d_all_amplitudes_[gpu] != nullptr) {
                    cudaSetDevice(gpu);
                    cudaFree(d_all_amplitudes_[gpu]);
                }
            }
            cudaSetDevice(0);
            d_all_amplitudes_.clear();
        }
    }

    torch::Tensor getNLL(torch::Tensor& vector)
    {
        return NLLFunction::apply(vector, d_all_amplitudes_, d_phsp_matrix_, events_, events_offsets_, amp_offsets_, phsp_weights_, bkg_weights_, bkg_integral_, n_amplitudes_, n_polar_);
    }

    int getNVector() const { return n_gls_ - con_trans_id_.size(); }

    torch::Tensor getSLVectors() const
    {
        torch::Device dev(torch::kCUDA, 0);
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt).device(dev);
        return torch::tensor(nSLvectors_, options);
    }

    void writeResult(torch::Tensor& vector, const std::string& filename, const int is_saved_weight = 0)
    {
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(vector.dtype() == torch::kComplexFloat, "vector must be complex128");

        // const int original_size = vector.numel();
        // int extended_size = original_size;

        torch::Device dev(torch::kCUDA, vector.get_device());

        torch::Tensor extended_vector = NLLFunction::extendVectorWithConstraints(vector, dev);

        // 将extended_vector分配到多个GPU（如果需要，可以直接在GPU上处理约束）
        std::vector<torch::Tensor> extended_vec_per_gpu;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            extended_vec_per_gpu.push_back(extended_vector.to(torch::Device(torch::kCUDA, i)));
        }

        const int target_dev = vector.get_device();
        cudaSetDevice(target_dev);

        int npartials = nSLvectors_.size();
        // 变量多GPU分配和计算
        std::vector<double*> d_final_result_vec;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            double* d_final_result;
            cudaMalloc(&d_final_result, events_[i][0] * sizeof(double));
            d_final_result_vec.push_back(d_final_result);
        }
        std::vector<double*> d_partial_result_vec;
        for (int i = 0; i < d_all_amplitudes_.size(); ++i) {
            double* d_partial_result;
            cudaMalloc(&d_partial_result, events_[i][0] * npartials * sizeof(double));
            d_partial_result_vec.push_back(d_partial_result);
        }

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
            int bkg_events = events_[gpu][2];
            N_bkg += static_cast<int>(bkg_events);
        }

        double h_phsp_integral = 0.0;
        double* h_interference_matrix = new double[nSLvectors_.size() * nSLvectors_.size()];
        double* h_total_results = new double[N_phsp];
        double* h_partial_results = new double[N_phsp * npartials];
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

            /////////////////////////
            /////////////////////////
            computeResults(d_all_amplitudes_[gpu],
                reinterpret_cast<const cuComplex*>(extended_vec_per_gpu[gpu].data_ptr()),
                d_final_result_vec[gpu], d_total_integral_gpu, d_partial_result_vec[gpu],
                d_interference_matrix_gpu, d_nSLvectors, npartials, events_[gpu][0],
                n_amplitudes_, n_polar_);

            // 将单个GPU的结果累加到全局结果
            double h_total_integral_gpu;
            cudaMemcpy(&h_total_integral_gpu, d_total_integral_gpu, sizeof(double), cudaMemcpyDeviceToHost);
            h_phsp_integral += h_total_integral_gpu;
            double* h_interference_matrix_gpu = new double[npartials * npartials];
            cudaMemcpy(h_interference_matrix_gpu, d_interference_matrix_gpu, npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < npartials * npartials; ++i) {
                h_interference_matrix[i] += h_interference_matrix_gpu[i];
            }
            double* h_total_results_gpu = new double[events_[gpu][0]];
            cudaMemcpy(h_total_results_gpu, d_final_result_vec[gpu], events_[gpu][0] * sizeof(double), cudaMemcpyDeviceToHost);
            std::copy(h_total_results_gpu, h_total_results_gpu + events_[gpu][0], h_total_results + static_cast<int>(N_phsp - events_[gpu][0]));
            double* h_partial_results_gpu = new double[events_[gpu][0] * npartials];
            cudaMemcpy(h_partial_results_gpu, d_partial_result_vec[gpu], events_[gpu][0] * npartials * sizeof(double), cudaMemcpyDeviceToHost);
            std::copy(h_partial_results_gpu, h_partial_results_gpu + events_[gpu][0] * npartials, h_partial_results + static_cast<int>((N_phsp - events_[gpu][0]) * npartials));
            cudaFree(d_total_integral_gpu);
            cudaFree(d_interference_matrix_gpu);
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
        int dataIntegral = N_data;
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

            // 填充 phsp tree
            for (int i = 0; i < N_phsp; ++i)
            {
                // 设置权重
                total_weight = h_total_results[i] / h_phsp_integral * static_cast<double>(dataIntegral);
                // std::cout << "Event " << i << ": Total Weight = " <<
                // total_weight << std::endl;
                for (int j = 0; j < npartials; ++j)
                {
                    partial_weights[j] = h_partial_results[i * npartials + j] * normFactor;
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

                    std::string partial_dir_name = "h_" + resonance_names_[partial_idx];

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

                    std::string partial_dir_name = "h_" + resonance_names_[partial_idx];

                    TH1F* hist = anglehist_partial[j];
                    hist->Scale(normFactor);
                    hist->Write(partial_dir_name.c_str(), TObject::kOverwrite);
                    delete anglehist_partial[j];
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

    torch::Tensor getHessian(torch::Tensor& vector)
    {
        // 基于解析公式计算 Hessian，参考 best_hess.py
        // H = Σ_k [-2*tildeB_k/S_k + 4*Bu_k*Bu_k^T/S_k²]
        //     - Σ_k w_k*[-2*tildeB_bkg_k/S_bkg_k + 4*Bu_bkg_k*Bu_bkg_k^T/S_bkg_k²]
        //     + (N_data-W_bkg)*[2*tildeP/T - 4*Pu*Pu^T/T²]

        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(vector.dtype() == torch::kComplexFloat, "vector must be complex128");
        TORCH_CHECK(vector.dim() == 1, "vector must be 1-dimensional");

        const int n = vector.numel();
        const int n2 = 2 * n;
        torch::Device dev = vector.device();
        torch::Tensor extended_vector = NLLFunction::extendVectorWithConstraints(vector, dev);
        torch::Tensor extended_vector_conj = extended_vector.conj();
        const cuComplex* d_vec = reinterpret_cast<const cuComplex*>(extended_vector.data_ptr());
        const cuComplex* d_vec_conj = reinterpret_cast<const cuComplex*>(extended_vector_conj.data_ptr());

        // 计算phsp量 (与forward pass一致)
        cudaSetDevice(dev.index());
        cuComplex* d_P_vec;
        cudaMalloc(&d_P_vec, n_amplitudes_ * sizeof(cuComplex));
        float* d_pr, * d_pi;
        cudaMalloc(&d_pr, sizeof(float));
        cudaMalloc(&d_pi, sizeof(float));
        // 用d_vec（非共轭）与phspHessianKernel的Pu/tildeP约定保持一致
        computeQuadraticForm(d_phsp_matrix_, d_vec, d_P_vec, d_pr, d_pi, n_amplitudes_);
        float phr, phi;
        cudaMemcpy(&phr, d_pr, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phi, d_pi, sizeof(float), cudaMemcpyDeviceToHost);
        double phsp_factor = (double)phr;
        cudaFree(d_pr); cudaFree(d_pi);

        // std::cout << "phsp_factor: " << phsp_factor << std::endl;

        // 分配扩展Hessian (2·n_ext × 2·n_ext, GPU 0)
        const int n_ext = extended_vector.numel();  // = n_amplitudes_
        torch::Tensor hessian = torch::zeros({ 2 * n_ext, 2 * n_ext }, torch::kDouble).to(dev);
        double* d_hessian = hessian.data_ptr<double>();

        // 多GPU: 累加data和bkg的Hessian贡献
        int totalDataEvents = 0;
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            cudaSetDevice(gpu);
            torch::Tensor vec_gpu = extended_vector.to(torch::Device(torch::kCUDA, gpu));
            const cuComplex* d_v_gpu = reinterpret_cast<const cuComplex*>(vec_gpu.data_ptr());

            // --- data ---
            int nData = events_[gpu][1];
            if (nData > 0) {
                totalDataEvents += nData;
                cuComplex* d_amp = d_all_amplitudes_[gpu] + amp_offsets_[gpu][1];
                computeDataHessianContrib(d_amp, d_v_gpu, nullptr, d_hessian, nData, n_polar_, n_ext);
            }

            // --- bkg ---
            int nBkg = events_[gpu][2];
            if (nBkg > 0) {
                // NLL = data_NLL - bkg_NLL, 所以 Hessian = H_data - H_bkg
                // 传入负权重以实现减法
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
                cuComplex* d_amp = d_all_amplitudes_[gpu] + amp_offsets_[gpu][2];
                computeDataHessianContrib(d_amp, d_v_gpu, d_w_bkg, d_hessian, nBkg, n_polar_, n_ext);
                cudaFree(d_w_bkg);
            }
        }

        // phsp Hessian贡献
        double phsp_weight = (double)totalDataEvents - bkg_integral_;
        cudaSetDevice(dev.index());
        computePhspHessian(d_phsp_matrix_, d_vec, phsp_factor, phsp_weight, d_hessian, n_ext);

        // std::cout << "n_ext: " << n_ext << ", n: " << n << ", totalDataEvents: " << totalDataEvents << std::endl;

        // 按约束Jacobian投影: H_orig (2n×2n) = J^T * H_ext (2·n_ext×2·n_ext) * J
        // J是diagonal blocks: J[2*eid][2*oid]=real_ratio, J[2*eid+1][2*oid+1]=imag_ratio
        // 与extendVectorWithConstraints的约定一致
        if (n_ext > n) {
            std::vector<int> h_oids, h_eids;
            std::vector<double> h_re, h_im;
            for (size_t gi = 0; gi < con_trans_id_.size(); ++gi) {
                const auto& vecid = con_trans_id_[gi];
                const auto& vals = con_trans_values_[gi];
                if (vecid.empty()) continue;
                int oid = *std::min_element(vecid.begin(), vecid.end());
                auto oit = std::find(vecid.begin(), vecid.end(), oid);
                int oidx = std::distance(vecid.begin(), oit);
                std::complex<double> oc = vals[oidx];
                double oc_re = std::real(oc), oc_im = std::imag(oc);
                for (size_t j = 0; j < vecid.size(); ++j) {
                    if ((int)j == oidx) continue;
                    h_oids.push_back(oid);
                    h_eids.push_back(vecid[j]);
                    // 与extendVectorWithConstraints一致：分别计算实部和虚部的比例
                    h_re.push_back(std::real(vals[j]) / oc_re);
                    h_im.push_back(std::imag(vals[j]) / oc_im);
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

    // 内部：用已加载的truth振幅，对给定extended_vector计算phsp积分+BF
    void computePhspAndBF(
        const torch::Tensor& extended_vector,
        std::vector<double>& out_phsp,
        std::vector<double>& out_bf,
        int npartials,
        const std::vector<cuComplex*>& d_truth_amps,
        const std::vector<int>& truth_ev_per_gpu,
        double dataIntegral) const
    {
        // 每GPU分配d_nSLvectors (不能跨GPU共享cudaMalloc指针)
        auto alloc_nsl = [&](int gpu) {
            int* d_nsl;
            cudaSetDevice(gpu);
            cudaMalloc(&d_nsl, npartials * sizeof(int));
            cudaMemcpy(d_nsl, nSLvectors_.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);
            return d_nsl;
            };

        // PHSP
        std::fill(out_phsp.begin(), out_phsp.end(), 0.0);
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu) {
            int nPhsp = events_[gpu][0];
            if (nPhsp <= 0) continue;
            int* d_nsl = alloc_nsl(gpu);
            double* d_p; cudaMalloc(&d_p, npartials * sizeof(double));
            cudaMemset(d_p, 0, npartials * sizeof(double));
            double* d_t; cudaMalloc(&d_t, sizeof(double)); cudaMemset(d_t, 0, sizeof(double));

            auto vg = extended_vector.to(torch::Device(torch::kCUDA, gpu));
            computeBranchingFractions(d_all_amplitudes_[gpu],
                reinterpret_cast<const cuComplex*>(vg.data_ptr()),
                d_p, nullptr, d_t, d_nsl, npartials, nPhsp, n_amplitudes_, n_polar_);

            std::vector<double> hp(npartials); double ht;
            cudaMemcpy(hp.data(), d_p, npartials * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&ht, d_t, sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < npartials; ++i) out_phsp[i] += hp[i];
            cudaFree(d_p); cudaFree(d_t); cudaFree(d_nsl);
        }

        // Truth integrals
        std::vector<double> h_truth_partial(npartials, 0.0);
        std::vector<double> h_scattering(npartials * npartials, 0.0);

        for (size_t gpu = 0; gpu < d_truth_amps.size(); ++gpu) {
            int nt = truth_ev_per_gpu[gpu];
            if (nt <= 0 || d_truth_amps[gpu] == nullptr) continue;
            int* d_nsl = alloc_nsl(gpu);
            double* d_p; cudaMalloc(&d_p, npartials * sizeof(double)); cudaMemset(d_p, 0, npartials * sizeof(double));
            double* d_s; cudaMalloc(&d_s, npartials * npartials * sizeof(double)); cudaMemset(d_s, 0, npartials * npartials * sizeof(double));
            double* d_t; cudaMalloc(&d_t, sizeof(double)); cudaMemset(d_t, 0, sizeof(double));

            auto vg = extended_vector.to(torch::Device(torch::kCUDA, gpu));
            computeBranchingFractions(d_truth_amps[gpu],
                reinterpret_cast<const cuComplex*>(vg.data_ptr()),
                d_p, d_s, d_t, d_nsl, npartials, nt, n_amplitudes_, n_polar_);

            std::vector<double> hp(npartials), hs(npartials * npartials); double ht;
            cudaMemcpy(hp.data(), d_p, npartials * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(), d_s, npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&ht, d_t, sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < npartials; ++i) {
                h_truth_partial[i] += hp[i];
                for (int j = 0; j < npartials; ++j) h_scattering[i * npartials + j] += hs[i * npartials + j];
            }
            cudaFree(d_p); cudaFree(d_s); cudaFree(d_t); cudaFree(d_nsl);
        }

        computeBFfromIntegrals(out_phsp.data(), h_truth_partial.data(),
            h_scattering.data(), out_bf.data(), npartials, dataIntegral);
    }

    torch::Tensor getBranchFractions(torch::Tensor& vector)
    {
        TORCH_CHECK(vector.is_cuda(), "vector must be on CUDA");
        TORCH_CHECK(vector.dtype() == torch::kComplexFloat, "vector must be ComplexFloat");

        const auto& data_files = config_parser_.getDataFiles();
        TORCH_CHECK(data_files.count("phsp_truth") > 0, "No phsp_truth in config");

        const int n = vector.numel();
        const int n2 = 2 * n;
        const int npartials = nSLvectors_.size();
        torch::Device dev = vector.device();

        // dataIntegral
        int totalDataEvents = 0;
        for (size_t gpu = 0; gpu < events_.size(); ++gpu)
            totalDataEvents += events_[gpu][1];
        double dataIntegral = static_cast<double>(totalDataEvents) - bkg_integral_;

        // ===== 加载truth振幅(一次性) =====
        std::vector<std::string> particles_names;
        for (const auto& p : particles_) particles_names.push_back(p.name);

        std::cout << "Reading phase space truth samples..." << std::endl;
        auto Vp4_truth = readMomentaFromDat(data_files.at("phsp_truth"),
            config_parser_.getDataOrder(), particles_names);
        int total_truth = Vp4_truth.begin()->second.size();
        std::cout << "Phase space truth events: " << total_truth << std::endl;

        std::vector<int> truth_ev_per_gpu(n_gpus_, 0);
        int base_tr = total_truth / n_gpus_, rem_tr = total_truth % n_gpus_;
        for (int g = 0; g < n_gpus_; ++g)
            truth_ev_per_gpu[g] = base_tr + (g < rem_tr ? 1 : 0);

        std::vector<std::map<std::string, std::vector<LorentzVector>>> Vp4_tpg(n_gpus_);
        int ev_off = 0;
        for (int g = 0; g < n_gpus_; ++g) {
            int nev = truth_ev_per_gpu[g];
            if (nev > 0)
                for (const auto& [k, v] : Vp4_truth)
                    Vp4_tpg[g][k].assign(v.begin() + ev_off, v.begin() + ev_off + nev);
            ev_off += nev;
        }

        std::vector<std::vector<int>> t_ev_off(n_gpus_), t_amp_off(n_gpus_);
        for (int g = 0; g < n_gpus_; ++g) {
            int nev = truth_ev_per_gpu[g];
            t_ev_off[g] = { 0, nev };
            t_amp_off[g] = { 0, nev * n_polar_ * n_amplitudes_ };
        }
        auto saved_ev = events_offsets_, saved_amp = amp_offsets_;
        events_offsets_ = t_ev_off; amp_offsets_ = t_amp_off;
        std::vector<cuComplex*> d_truth_amps = calculateAmplitudes(Vp4_tpg);
        events_offsets_ = saved_ev; amp_offsets_ = saved_amp;

        // ===== 中心值 =====
        torch::Tensor ev_center = NLLFunction::extendVectorWithConstraints(vector, dev);
        std::vector<double> phsp_center(npartials), bf_center(npartials);
        computePhspAndBF(ev_center, phsp_center, bf_center,
            npartials, d_truth_amps, truth_ev_per_gpu, dataIntegral);

        // ===== 误差: BF_error = sqrt(diag(J @ H^{-1} @ J^T)) =====
        std::vector<double> bf_errors(npartials, 0.0);
        torch::Tensor hessian = getHessian(vector);
        if (hessian.numel() > 0 && hessian.size(0) == n2) {
            auto eig = torch::linalg_eigvalsh(hessian);
            if (eig[0].item<double>() > 1e-8) {
                torch::Tensor cov = torch::linalg_inv(hessian).cpu();
                std::vector<double> h_cov(n2 * n2);
                std::memcpy(h_cov.data(), cov.data_ptr<double>(), n2 * n2 * sizeof(double));

                // 数值Jacobian
                std::vector<double> J(npartials * n2, 0.0);
                torch::Tensor v_real = torch::view_as_real(vector).flatten().clone();
                const double eps = 5e-6;
                for (int j = 0; j < n2; ++j) {
                    auto vp = v_real.clone(); vp[j] += eps;
                    auto vm = v_real.clone(); vm[j] -= eps;
                    auto cvp = torch::view_as_complex(vp.view({ -1, 2 })).contiguous();
                    auto cvm = torch::view_as_complex(vm.view({ -1, 2 })).contiguous();
                    auto evp = NLLFunction::extendVectorWithConstraints(cvp, dev);
                    auto evm = NLLFunction::extendVectorWithConstraints(cvm, dev);

                    std::vector<double> phsp_p(npartials), bf_p(npartials);
                    std::vector<double> phsp_m(npartials), bf_m(npartials);
                    computePhspAndBF(evp, phsp_p, bf_p, npartials,
                        d_truth_amps, truth_ev_per_gpu, dataIntegral);
                    computePhspAndBF(evm, phsp_m, bf_m, npartials,
                        d_truth_amps, truth_ev_per_gpu, dataIntegral);

                    for (int i = 0; i < npartials; ++i)
                        J[i * n2 + j] = (bf_p[i] - bf_m[i]) / (2.0 * eps);
                }

                computeBFErrors(J.data(), h_cov.data(), bf_errors.data(), npartials, n2);
            }
        }

        // 释放
        for (size_t g = 0; g < d_truth_amps.size(); ++g)
            if (d_truth_amps[g]) { cudaSetDevice(g); cudaFree(d_truth_amps[g]); }
        // 返回 n×2: [center, error]
        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor result = torch::empty({ npartials, 2 }, opts);
        for (int i = 0; i < npartials; ++i) {
            result[i][0] = bf_center[i];
            result[i][1] = bf_errors[i];
        }
        return result;
    }

    ////////////////////////
    torch::Tensor getDataTensor() const
    {
        // torch::Tensor output = torch::from_blob(data_fix_, {data_length *
        // n_gls_},
        // torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_[0] + amp_offsets_[0][1],
            { events_[0][1] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getPhspTensor() const
    {
        // torch::Tensor output = torch::from_blob(phsp_fix_, {phsp_length *
        // n_gls_},
        // torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_[0],
            { events_[0][0] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

        return output;
    }

    // torch::Tensor getTruthTensor() const
    // {
    //     const auto& data_files = config_parser_.getDataFiles();
    //     if (data_files.count("phsp_truth") > 0)
    //     {
    //         std::vector<std::string> particles_names;
    //         for (const auto& particle : particles_)
    //         {
    //             particles_names.push_back(particle.name);
    //         }

    //         // std::cout << "Reading phase space truth samples..." << std::endl;
    //         std::map<std::string, std::vector<LorentzVector>> Vp4_truth =
    //             readMomentaFromDat(data_files.at("phsp_truth"),
    //                 config_parser_.getDataOrder(),
    //                 particles_names);
    //         std::cout << "Phase space truth events: "
    //             << Vp4_truth.begin()->second.size() << std::endl;
    //         // std::cout << "Calculating phase space truth amplitudes..." <<
    //         // std::endl;
    //         cuComplex* truth_fix = calculateAmplitudes(Vp4_truth, { 0 }, { 0 });
    //         int truth_length = Vp4_truth.begin()->second.size() * n_polar_;

    //         torch::Tensor output = torch::from_blob(truth_fix,
    //             { truth_length * n_gls_ },
    //             torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

    //         cudaFree(truth_fix);

    //         return output;
    //     }
    //     else
    //     {
    //         std::cerr
    //             << "No phsp_truth data file specified in the configuration."
    //             << std::endl;
    //         return torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA));
    //     }
    // }

    torch::Tensor getBkgTensor() const
    {
        // torch::Tensor output = torch::from_blob(bkg_fix_, {bkg_length *
        // n_gls_},
        // torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();
        torch::Tensor output = torch::from_blob(d_all_amplitudes_[0] + amp_offsets_[0][2],
            { events_[0][2] * n_polar_ * n_amplitudes_ },
            torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA)).clone();

        return output;
    }

    torch::Tensor getBkgWeightsTensor() const
    {
        // if (bkg_weights_ != nullptr && bkg_length > 0)
        if (bkg_weights_[0] != nullptr && events_[0][2] > 0)
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
    /////////////////////////////

    std::vector<std::vector<int>> getConstraintsIndex() const
    {
        return con_trans_id_;
    }

    std::vector<std::vector<std::pair<double, double>>> getConstraintsValues() const
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
    cuComplex* d_phsp_matrix_ = nullptr;
    std::vector<double*> phsp_weights_;
    std::vector<double*> bkg_weights_;
    double bkg_integral_ = 0.0;

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
        auto Vp4_phsp = readMomentaFromDat(data_files.at("phsp"), data_order, particles_names);
        std::cout << "Phase space events: " << Vp4_phsp.begin()->second.size() << std::endl;
        init_events.push_back(Vp4_phsp.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_phsp);

        // 计算数据振幅
        std::cout << "Reading data samples..." << std::endl;
        auto Vp4_data = readMomentaFromDat(data_files.at("data"), data_order, particles_names);
        std::cout << "data events: " << Vp4_data.begin()->second.size() << std::endl;
        init_events.push_back(Vp4_data.begin()->second.size());
        Vp4_to_merge.push_back(Vp4_data);

        // 计算本底振幅
        if (data_files.count("bkg") > 0)
        {
            std::cout << "Reading background samples..." << std::endl;
            auto Vp4_bkg = readMomentaFromDat(data_files.at("bkg"), data_order, particles_names);
            std::cout << "Background events: " << Vp4_bkg.begin()->second.size() << std::endl;
            init_events.push_back(Vp4_bkg.begin()->second.size());
            Vp4_to_merge.push_back(Vp4_bkg);
        }

        CUDA_CHECK(cudaGetDeviceCount(&n_gpus_));
        initializeMultiGPUs(init_events);

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
        d_all_amplitudes_ = calculateAmplitudes(Vp4_all_);

        // 输出d_all_amplitudes_[0]所有内容:
        // int Ntotal = 0;
        // for (size_t j = 0; j < events_[0].size(); ++j)
        // {
        //     Ntotal += events_[0][j];
        // }
        // cuComplex* h_amp = new cuComplex[Ntotal * n_polar_ * n_amplitudes_];
        // cudaMemcpy(h_amp, d_all_amplitudes_[0], Ntotal * n_polar_ * n_amplitudes_ * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < Ntotal * n_polar_ * n_amplitudes_; ++i)
        // {
        //     std::cout << "Amplitude[" << i << "] = " << h_amp[i].x << " + " << h_amp[i].y << "i" << std::endl;
        // }
        // delete[] h_amp;

        // bkg_weights_
        if (data_files.count("bkg_weights") > 0 && data_files.count("bkg_weights") > 0)
        {
            std::vector<int> bkg_events_per_gpu;
            for (size_t i = 0; i < events_.size(); ++i)
            {
                bkg_events_per_gpu.push_back(events_[i][2]);
            }
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
                bkg_integral_ += events_[i][2];
            }
        }

        // phsp*phsp^T矩阵，大小为 n_amplitudes_ * n_amplitudes_
        // cuComplex* d_phsp;
        cudaMalloc(&d_phsp_matrix_, n_amplitudes_ * n_amplitudes_ * sizeof(cuComplex));
        for (size_t gpu = 0; gpu < d_all_amplitudes_.size(); ++gpu)
        {
            cudaSetDevice(gpu);
            // std::cout << "GPU " << gpu << ": Events for type 0 (phsp) = " << events_[gpu][0] << std::endl;

            cuComplex* d_phsp_gpu;
            cudaMalloc(&d_phsp_gpu, n_amplitudes_ * n_amplitudes_ * sizeof(cuComplex));
            cublasHandle_t cublas_handle;
            cublasCreate(&cublas_handle);
            cuComplex alpha = make_cuComplex(1.0 / static_cast<float>(Vp4_phsp.begin()->second.size()), 0.0f);
            cuComplex beta = make_cuComplex(0.0f, 0.0f);
            cublasCgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C, n_amplitudes_, n_amplitudes_, events_[gpu][0] * n_polar_,
                &alpha, d_all_amplitudes_[gpu], n_amplitudes_, d_all_amplitudes_[gpu], n_amplitudes_,
                &beta, d_phsp_gpu, n_amplitudes_);
            // 将每个GPU计算的phsp矩阵累加到主GPU的d_phsp中
            if (gpu == 0) {
                cudaMemcpy(d_phsp_matrix_, d_phsp_gpu, n_amplitudes_ * n_amplitudes_ * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
            }
            else {
                cublasCaxpy(cublas_handle, n_amplitudes_ * n_amplitudes_, &alpha, d_phsp_gpu, 1, d_phsp_matrix_, 1);
            }
            cublasDestroy(cublas_handle);
            cudaFree(d_phsp_gpu);
        }
        // cudaFree(d_phsp);
        // 打印矩阵d_phsp
        // cuComplex* h_phsp = new cuComplex[n_amplitudes_ * n_amplitudes_];
        // cudaMemcpy(h_phsp, d_phsp_matrix_, n_amplitudes_* n_amplitudes_ * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        // std::cout << "Phase space matrix (phsp*phsp^T):" << std::endl;
        // for (int i = 0; i < n_amplitudes_; ++i) {
        //     for (int j = 0; j < n_amplitudes_; ++j) {
        //         std::cout << "  [" << i << "][" << j << "] = " << h_phsp[i * n_amplitudes_ + j].x << " + " << h_phsp[i * n_amplitudes_ + j].y << "j" << std::endl;
        //     }
        // }
        // delete[] h_phsp;

        NLLFunction::setConstraints(con_trans_id_, con_trans_values_);

        std::cout << "Number of GPUs available: " << n_gpus_ << std::endl;
        std::cout << "Number of partial waves: " << n_gls_ << std::endl;
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
                    cas.getAmps(d_all_amplitudes_vec, resonance, gls_index, n_amplitudes_, events_offsets_, amp_offsets_);
                    gls_index += cas.getNSLCombs();
                }
            }
        }

        return d_all_amplitudes_vec;
    }


    // double computeNLL(std::vector<std::complex<float>>& params)
    double computeNLL(std::vector<cuComplex*>& d_params)
    {
        int extended_n_gls = n_amplitudes_;

        std::vector<double> local_mc_amp(n_gpus_, 0.0);
        std::vector<int> local_phsp_events(n_gpus_, 0);
        std::vector<cuComplex*> d_B_list(n_gpus_, nullptr);
        std::vector<cuComplex*> d_S_list(n_gpus_, nullptr);
        std::vector<cuComplex*> d_Q_list(n_gpus_, nullptr);
        std::vector<cuComplex*> d_bkg_S_list(n_gpus_, nullptr);
        std::vector<cuComplex*> d_bkg_Q_list(n_gpus_, nullptr);
        std::vector<double*> d_bkg_nll_list(n_gpus_, nullptr);

        // 临时存储每个 GPU 的 h_phsp_factor（局部，稍后会被全局因子覆盖）
        std::vector<double> local_h_phsp_factor(n_gpus_, 0.0);

        for (int i = 0; i < n_gpus_; ++i) {
            cudaSetDevice(i);
            const auto& events = events_[i];
            local_phsp_events[i] = events[0];  // 相空间事件数
            int phsp_len = events[0] * n_polar_;

            // 分配临时缓冲区
            cuComplex* d_B = nullptr;
            double* d_mc_amp = nullptr;
            cudaMalloc(&d_B, phsp_len * sizeof(cuComplex));
            cudaMalloc(&d_mc_amp, sizeof(double));
            d_B_list[i] = d_B;

            // 调用内核计算局部振幅平方和
            computePHSPfactor(d_all_amplitudes_[i],
                d_params[i],
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
        for (int i = 0; i < n_gpus_; ++i) {
            total_mc_amp += local_mc_amp[i];
            total_phsp_events += local_phsp_events[i];
        }
        double global_phsp_factor = total_mc_amp / static_cast<double>(total_phsp_events);

        // 5. 每个 GPU 使用全局 phsp_factor 计算各自的 NLL
        std::vector<double> local_data_nll(n_gpus_, 0.0);
        std::vector<double> local_bkg_nll(n_gpus_, 0.0);

        for (int i = 0; i < n_gpus_; ++i) {
            cudaSetDevice(i);
            const auto& events = events_[i];
            const auto& amp_offsets = amp_offsets_[i];
            const auto& events_offsets = events_offsets_[i];  // 若需要

            // 数据部分
            cuComplex* data_fix = d_all_amplitudes_[i] + amp_offsets[1];
            double* d_data_nll = nullptr;
            cudaMalloc(&d_data_nll, sizeof(double));
            computeNll(data_fix,
                d_params[i],
                nullptr, d_S_list[i], d_Q_list[i], d_data_nll,
                events[1] * n_polar_, extended_n_gls, n_polar_,
                global_phsp_factor);
            cudaMemcpy(&local_data_nll[i], d_data_nll, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_data_nll);

            // 背景部分
            if (events[2] > 0) {
                cuComplex* bkg_fix = d_all_amplitudes_[i] + amp_offsets[2];
                // 背景权重指针（假设全局传入，实际应每个 GPU 独立）
                // 这里简化：假设 bkg_weights 已按 GPU 分片，传入一个 std::vector<double*>
                // double* bkg_weights = nullptr; // 实际应从外部传入每个 GPU 的权重指针
                computeNll(bkg_fix,
                    d_params[i],
                    bkg_weights_[i],
                    d_bkg_S_list[i], d_bkg_Q_list[i], d_bkg_nll_list[i],
                    events[2] * n_polar_, extended_n_gls, n_polar_,
                    global_phsp_factor);
                cudaMemcpy(&local_bkg_nll[i], d_bkg_nll_list[i], sizeof(double), cudaMemcpyDeviceToHost);
            }
        }

        // 6. 合并 NLL
        double total_data_nll = 0.0, total_bkg_nll = 0.0;
        for (int i = 0; i < n_gpus_; ++i) {
            total_data_nll += local_data_nll[i];
            total_bkg_nll += local_bkg_nll[i];
        }
        double loss = total_data_nll - total_bkg_nll;

        for (int i = 0; i < n_gpus_; ++i) {
            cudaSetDevice(i);
            cudaFree(d_B_list[i]);
            cudaFree(d_S_list[i]);
            cudaFree(d_Q_list[i]);
            if (events_[i][2] > 0) {
                cudaFree(d_bkg_S_list[i]);
                cudaFree(d_bkg_Q_list[i]);
                cudaFree(d_bkg_nll_list[i]);
            }
        }

        return loss;
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
        .def("writeResult", &analysis::writeResult)
        .def("getHessian", &analysis::getHessian)
        .def("getDataTensor", &analysis::getDataTensor)
        .def("getPhspTensor", &analysis::getPhspTensor)
        // .def("getTruthTensor", &analysis::getTruthTensor)
        .def("getBranchFractions", &analysis::getBranchFractions)
        .def("getBkgTensor", &analysis::getBkgTensor)
        .def("getBkgWeightsTensor", &analysis::getBkgWeightsTensor)
        .def("getConstraintsIndex", &analysis::getConstraintsIndex)
        .def("getConstraintsValues", &analysis::getConstraintsValues)
        .def("getAmplitudeNames", &analysis::getAmplitudeNames)
        .def("getNPolarizations", &analysis::getNPolarizations);
}
