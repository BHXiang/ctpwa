#include <AmpGen.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
// #include <cuComplex.h>
#include <cuda_runtime.h>
#include <iostream>
// #include <Resonance.cuh>

// Amp2BD 类实现
Amp2BD::Amp2BD(std::array<int, 3> jvalues, std::array<int, 3> parities)
    : jvalues_(jvalues), parities_(parities)
{
    spinOrbitCombinations_ = ComSL(jvalues, parities);
}

// std::vector<SL> Amp2BD::ComSL(const std::array<int, 3> &spins, const
// std::array<int, 3> &parities)
// {
//     std::vector<SL> combinations;
//     auto [s1, s2, s3] = spins;
//     auto [p1, p2, p3] = parities;

//     const int S_min = std::abs(s2 - s3);
//     const int S_max = s2 + s3;
//     for (int S = S_min; S <= S_max; ++S)
//     {
//         const int L_min = std::abs(s1 - S);
//         const int L_max = s1 + S;
//         for (int L = L_min; L <= L_max; ++L)
//         {
//             const int sign = (L % 2 == 0) ? 1 : -1;
//             if (p1 == p2 * p3 * sign)
//             {
//                 combinations.emplace_back(S, L);
//             }
//         }
//     }
//     return combinations;
// }

std::vector<SL> Amp2BD::ComSL(const std::array<int, 3>& spins, const std::array<int, 3>& parities)
{
    std::vector<SL> combinations;

    // 将 2J+1 转换回 J*2 的形式（避免浮点数）
    // spins[i] = 2*J_i + 1 => 2*J_i = spins[i] - 1
    auto [s1_2j, s2_2j, s3_2j] = spins;

    // 转换为两倍的自旋值，以便整数运算
    int two_j1 = s1_2j - 1; // 2 * J1
    int two_j2 = s2_2j - 1; // 2 * J2
    int two_j3 = s3_2j - 1; // 2 * J3

    auto [p1, p2, p3] = parities;

    // S（总自旋）的范围：|J2 - J3| ≤ S ≤ J2 + J3
    // 使用两倍的值进行计算
    const int two_S_min = std::abs(two_j2 - two_j3);
    const int two_S_max = two_j2 + two_j3;

    for (int two_S = two_S_min; two_S <= two_S_max; two_S += 2)
    {
        // L（轨道角动量）的范围：|J1 - S| ≤ L ≤ J1 + S
        const int two_L_min = std::abs(two_j1 - two_S);
        const int two_L_max = two_j1 + two_S;

        for (int two_L = two_L_min; two_L <= two_L_max; two_L += 2)
        {
            // 宇称条件：P1 = P2 * P3 * (-1)^L
            // 注意：L 是轨道角动量，不是两倍值
            int L = two_L / 2; // 实际的轨道角动量量子数
            const int sign = (L % 2 == 0) ? 1 : -1;

            if (p1 == p2 * p3 * sign)
            {

                // if (L > 3) continue;

                // 存储实际的自旋量子数（不是两倍值）
                // int S = two_S / 2; // 实际的总自旋量子数
                combinations.emplace_back(two_S + 1, L);
            }
        }
    }
    return combinations;
}

// DeviceMomenta 成员函数实现
__device__ LorentzVector DeviceMomenta::getMomentum(int event_idx, int particle_idx) const
{
    if (event_idx >= 0 && event_idx < n_events && particle_idx >= 0 &&
        particle_idx < n_particles_per_event)
    {
        return momenta[event_idx * n_particles_per_event + particle_idx];
    }
    return LorentzVector();
}

// AmpCasDecay 类实现
AmpCasDecay::AmpCasDecay(const std::vector<Particle>& particles)
{
    for (const auto& p : particles)
    {
        particleMap_[p.name] = { p.spin, p.parity, p.mass };
        particleNames_.push_back(p.name);
    }
    nSLCombs_ = 1;
}

AmpCasDecay::~AmpCasDecay()
{
    // cudaFree(d_slamps_);
    // cudaFree(d_momenta_);
    // cudaFree(d_decayNodes_);
    // cudaFree(d_slCombination_);
    for (size_t i = 0; i < d_slamps_.size(); ++i)
    {
        if (d_slamps_[i])
            cudaFree(d_slamps_[i]);
        if (d_momenta_[i])
            cudaFree(d_momenta_[i]);
        if (d_decayNodes_[i])
            cudaFree(d_decayNodes_[i]);
        if (d_slCombination_[i])
            cudaFree(d_slCombination_[i]);
    }
}

void AmpCasDecay::addDecay(const Amp2BD& amp, const std::string& mother, const std::string& daug1, const std::string& daug2)
{
    decayChain_.push_back({ amp, mother, daug1, daug2 });

    auto jvals = amp.getJValues();
    auto pars = amp.getParities();
    addParticleIfNotExists(mother, jvals[0], pars[0], -1.0);
    addParticleIfNotExists(daug1, jvals[1], pars[1], -1.0);
    addParticleIfNotExists(daug2, jvals[2], pars[2], -1.0);

    nSLCombs_ *= amp.getSL().size();
}

// AmpCasDecay 私有方法实现
void AmpCasDecay::addParticleIfNotExists(const std::string& name, int spin, int parity, double mass)
{
    if (particleMap_.find(name) == particleMap_.end())
    {
        particleMap_[name] = { spin, parity, mass };
        particleNames_.push_back(name);
    }
}

std::vector<std::vector<SL>> AmpCasDecay::getSLCombinations() const
{
    std::vector<std::vector<SL>> result = { {} };
    for (const auto& chain : decayChain_)
    {
        const auto& sls = chain.amp.getSL();
        std::vector<std::vector<SL>> temp;
        for (const auto& r : result)
        {
            for (const auto& sl : sls)
            {
                auto newComb = r;
                newComb.push_back(sl);
                temp.push_back(newComb);
            }
        }
        result = temp;
    }
    return result;
}

// int AmpCasDecay::computeNPolarizations(const std::map<std::string, std::vector<LorentzVector>>& finalMomenta)
// {
//     int nPolar = 1;
//     // 母粒子极化态数
//     // if (!decayChain_.empty())
//     // {
//     //     const std::string &motherName = decayChain_[0].mother;
//     //     int motherSpin = particleMap_.at(motherName).spin;
//     //     nPolar *= (2 * motherSpin + 1);
//     // }
//     // 末态粒子极化态数
//     for (const auto& [name, _] : finalMomenta)
//     {
//         int particleSpin = particleMap_.at(name).spin;
//         // nPolar *= (2 * particleSpin + 1);
//         nPolar *= particleSpin;
//     }

//     return nPolar;
// }

std::vector<DeviceMomenta*> AmpCasDecay::convertToDeviceMomenta(
    const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta,
    const std::map<std::string, int>& particleToIndex,
    const std::vector<DecayNodeHost>& decayChain)
{
    std::vector<DeviceMomenta*> d_momenta(finalMomenta.size(), nullptr);
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

        // 第二步：根据衰变链逐级计算中间态和初态粒子的四动量
        bool changed = true;
        int max_iterations = decayChain.size() + 1;
        int iteration = 0;

        while (changed && iteration < max_iterations)
        {
            changed = false;

            for (const auto& decay_node : decayChain)
            {
                const std::string& mother = decay_node.mother;
                const std::string& daug1 = decay_node.daug1;
                const std::string& daug2 = decay_node.daug2;

                auto mother_it = particleToIndex.find(mother);
                auto daug1_it = particleToIndex.find(daug1);
                auto daug2_it = particleToIndex.find(daug2);

                if (mother_it == particleToIndex.end() ||
                    daug1_it == particleToIndex.end() ||
                    daug2_it == particleToIndex.end())
                {
                    continue;
                }

                int mother_idx = mother_it->second;
                int daug1_idx = daug1_it->second;
                int daug2_idx = daug2_it->second;

                if (!particle_calculated[mother_idx] &&
                    particle_calculated[daug1_idx] &&
                    particle_calculated[daug2_idx])
                {
                    // 计算所有事件的母粒子四动量
                    for (int event_idx = 0; event_idx < n_events; ++event_idx)
                    {
                        const LorentzVector& daug1_momentum = host_momenta[event_idx * n_particles + daug1_idx];
                        const LorentzVector& daug2_momentum = host_momenta[event_idx * n_particles + daug2_idx];
                        LorentzVector mother_momentum = daug1_momentum + daug2_momentum;
                        host_momenta[event_idx * n_particles + mother_idx] = mother_momentum;
                    }

                    particle_calculated[mother_idx] = true;
                    changed = true;
                }
            }
            iteration++;
        }

        // 第三步：将数据复制到设备
        DeviceMomenta* d_momenta_i;
        cudaMalloc(&d_momenta_i, sizeof(DeviceMomenta));

        // 在设备端分配四动量数组
        LorentzVector* d_momenta_array;
        cudaMalloc(&d_momenta_array, host_momenta.size() * sizeof(LorentzVector));
        cudaMemcpy(d_momenta_array, host_momenta.data(), host_momenta.size() * sizeof(LorentzVector), cudaMemcpyHostToDevice);

        // 设置设备端结构体参数
        DeviceMomenta h_momenta;
        h_momenta.momenta = d_momenta_array;
        h_momenta.n_events = n_events;
        h_momenta.n_particles_per_event = n_particles;

        // 将结构体复制到设备
        cudaMemcpy(d_momenta_i, &h_momenta, sizeof(DeviceMomenta), cudaMemcpyHostToDevice);

        d_momenta[i] = d_momenta_i;
    }

    return d_momenta;
}

void AmpCasDecay::computeSLAmps(const std::vector<std::map<std::string, std::vector<LorentzVector>>>& finalMomenta)
{
    d_slamps_.resize(finalMomenta.size(), nullptr);
    d_decayNodes_.resize(finalMomenta.size(), nullptr);
    d_slCombination_.resize(finalMomenta.size(), nullptr);

    // 所有粒子及其索引
    std::set<std::string> allParticles;
    for (const auto& node : decayChain_)
    {
        allParticles.insert(node.mother);
        allParticles.insert(node.daug1);
        allParticles.insert(node.daug2);
    }
    for (const auto& pair : finalMomenta[0])
    {
        allParticles.insert(pair.first);
    }
    int index = 0;
    for (const auto& name : allParticles)
    {
        particleToIndex_[name] = index++;
    }
    // 所有四动量都入设备端
    d_momenta_ = convertToDeviceMomenta(finalMomenta, particleToIndex_, decayChain_);

    for (size_t i = 0; i < finalMomenta.size(); ++i)
    {
        cudaSetDevice(i);

        // 计算事件数和极化态数
        nEvents_.push_back(finalMomenta[i].begin()->second.size());

        const auto slCombinations = getSLCombinations();
        nSLCombs_ = slCombinations.size();

        // 准备使用索引的衰变节点
        std::vector<DecayNode> host_decayNodes;
        for (const auto& node : decayChain_)
        {
            DecayNode indexed_node;
            indexed_node.mother_idx = particleToIndex_[node.mother];
            indexed_node.daug1_idx = particleToIndex_[node.daug1];
            indexed_node.daug2_idx = particleToIndex_[node.daug2];

            indexed_node.mass[0] = particleMap_[node.mother].mass;
            indexed_node.mass[1] = particleMap_[node.daug1].mass;
            indexed_node.mass[2] = particleMap_[node.daug2].mass;

            host_decayNodes.push_back(indexed_node);
        }

        // 准备衰变链信息
        std::vector<int> host_dj, host_dj1, host_dj2;
        for (size_t i = 0; i < decayChain_.size(); ++i)
        {
            const auto& node = decayChain_[i];
            auto jvals = node.amp.getJValues();
            host_dj.push_back(std::get<0>(jvals));  // dj
            host_dj1.push_back(std::get<1>(jvals)); // dj1
            host_dj2.push_back(std::get<2>(jvals)); // dj2
        }

        // 分配设备内存
        cudaMalloc(&d_slamps_[i], nEvents_[i] * nPolarizations_ * nSLCombs_ * sizeof(thrust::complex<double>));

        // 准备设备端的衰变链信息
        int* d_dimj, * d_dimj1, * d_dimj2;
        cudaMalloc(&d_dimj, decayChain_.size() * sizeof(int));
        cudaMalloc(&d_dimj1, decayChain_.size() * sizeof(int));
        cudaMalloc(&d_dimj2, decayChain_.size() * sizeof(int));
        cudaMemcpy(d_dimj, host_dj.data(), decayChain_.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dimj1, host_dj1.data(), decayChain_.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dimj2, host_dj2.data(), decayChain_.size() * sizeof(int), cudaMemcpyHostToDevice);

        // 将衰变节点复制到设备
        cudaMalloc(&d_decayNodes_[i], decayChain_.size() * sizeof(DecayNode));
        cudaMemcpy(d_decayNodes_[i], host_decayNodes.data(), decayChain_.size() * sizeof(DecayNode), cudaMemcpyHostToDevice);

        // 准备设备端的SL组合
        std::vector<SL> host_slCombinations;
        for (size_t slIdx = 0; slIdx < slCombinations.size(); ++slIdx)
        {
            for (const auto& sl : slCombinations[slIdx])
            {
                host_slCombinations.push_back(sl);
            }
        }

        // 传递SL组合到设备
        cudaMalloc(&d_slCombination_[i], host_slCombinations.size() * sizeof(SL));
        cudaMemcpy(d_slCombination_[i], host_slCombinations.data(), host_slCombinations.size() * sizeof(SL), cudaMemcpyHostToDevice);

        // 预计算振幅偏移量
        int amp_size = 0;
        for (size_t i = 0; i < decayChain_.size(); ++i)
        {
            int dim_j = host_dj[i];
            int dim_j1 = host_dj1[i];
            int dim_j2 = host_dj2[i];

            int total_amp_size = dim_j * dim_j1 * dim_j2;
            int trans1_size = dim_j1 * dim_j1;
            int trans2_size = dim_j2 * dim_j2;
            int max_dim = max(dim_j1, dim_j2);
            int massive_shared_size = 2 * max_dim * max_dim;

            size_t total_size = 2 * total_amp_size + trans1_size + trans2_size + massive_shared_size;

            amp_size += 2 * dim_j * dim_j1 * dim_j2 + total_size;
        }

        int batch_size = 1000000;

        for (int start = 0; start < nEvents_[i]; start += batch_size)
        {
            int n_events = 0;
            if (start + batch_size <= nEvents_[i])
            {
                n_events = batch_size;
            }
            else
            {
                n_events = nEvents_[i] - start;
            }

            thrust::complex<double>* d_amp_buffer;
            cudaMalloc(&d_amp_buffer, n_events * nSLCombs_ * amp_size * sizeof(thrust::complex<double>));
            // cudaMalloc(&d_amp_buffer, 1 * sizeof(thrust::complex<double>));

            int sharedMemSize = 3000 * sizeof(thrust::complex<double>);
            // int sharedMemSize = 1 * sizeof(thrust::complex<double>);

            // 计算振幅
            int blockSize = 256;
            int numBlocks = (nEvents_[i] + blockSize - 1) / blockSize;
            // computeSLAmpKernel<<<gridDim, blockDim, sharedMemSize>>>(
            // computeSLAmpKernel<<<gridDim, blockDim>>>(
            computeSLAmpKernel << <numBlocks, blockSize, sharedMemSize >> > (
                d_slamps_[i], d_amp_buffer, d_momenta_[i], d_decayNodes_[i], d_dimj, d_dimj1,
                d_dimj2, d_slCombination_[i], nSLCombs_, nEvents_[i], nPolarizations_,
                decayChain_.size(), amp_size * nSLCombs_, n_events, start);

            cudaDeviceSynchronize();

            // 检查CUDA错误
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error in computeSLAmp: "
                    << cudaGetErrorString(err) << std::endl;
            }

            cudaFree(d_amp_buffer);
        }

        // 清理临时设备内存
        cudaFree(d_dimj);
        cudaFree(d_dimj1);
        cudaFree(d_dimj2);
    }
}

// 实现步长计算函数（逻辑不变，步长为int）
__host__ __device__ void compute_strides(const int* shape, int rank,
    int* strides)
{
    if (rank == 0)
        return;
    strides[rank - 1] = 1; // 最后一个维度步长恒为1
    for (int i = rank - 2; i >= 0; i--)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// 通用复数张量缩并函数（将float替换为Complex）
template <int MAX_RANK = 8>
__device__ static void
contract(const thrust::complex<double>* A, const int* shape_A, int rank_A,
    const thrust::complex<double>* B, const int* shape_B, int rank_B,
    thrust::complex<double>* C, const int* shape_C, int rank_C,
    const int* contract_dims_A, const int* contract_dims_B,
    int num_contract_dims)
{

    // 计算步长（步长类型仍为int）
    int strides_A[MAX_RANK], strides_B[MAX_RANK], strides_C[MAX_RANK];
    compute_strides(shape_A, rank_A, strides_A);
    compute_strides(shape_B, rank_B, strides_B);
    compute_strides(shape_C, rank_C, strides_C);

    // 计算输出总大小
    int total_C = 1;
    for (int i = 0; i < rank_C; i++)
        total_C *= shape_C[i];

    // 遍历所有输出元素（累加初始值为复数0）
    for (int idx_c = 0; idx_c < total_C; idx_c++)
    {
        thrust::complex<double> sum(0.0, 0.0); // 复数累加器，实部虚部初始为0

        // 线性索引转多维索引（逻辑不变）
        int indices_C[MAX_RANK];
        int temp = idx_c;
        for (int i = rank_C - 1; i >= 0; i--)
        {
            indices_C[i] = temp % shape_C[i];
            temp /= shape_C[i];
        }

        // 计算缩并维度总大小（逻辑不变）
        int contract_size = 1;
        for (int i = 0; i < num_contract_dims; i++)
        {
            contract_size *= shape_A[contract_dims_A[i]];
        }

        // 遍历缩并维度（复数乘法+加法）
        for (int contract_idx = 0; contract_idx < contract_size; contract_idx++)
        {
            int indices_A[MAX_RANK] = { 0 };
            int indices_B[MAX_RANK] = { 0 };

            // 设置非缩并维度（逻辑不变）
            int c_pos = 0;
            for (int i = 0; i < rank_A; i++)
            {
                bool is_contract = false;
                for (int j = 0; j < num_contract_dims; j++)
                {
                    if (i == contract_dims_A[j])
                    {
                        is_contract = true;
                        break;
                    }
                }
                if (!is_contract)
                    indices_A[i] = indices_C[c_pos++];
            }

            for (int i = 0; i < rank_B; i++)
            {
                bool is_contract = false;
                for (int j = 0; j < num_contract_dims; j++)
                {
                    if (i == contract_dims_B[j])
                    {
                        is_contract = true;
                        break;
                    }
                }
                if (!is_contract)
                    indices_B[i] = indices_C[c_pos++];
            }

            // 设置缩并维度（逻辑不变）
            int temp_idx = contract_idx;
            for (int i = num_contract_dims - 1; i >= 0; i--)
            {
                int dim = contract_dims_A[i];
                int size = shape_A[dim];
                indices_A[dim] = temp_idx % size;
                indices_B[contract_dims_B[i]] = temp_idx % size;
                temp_idx /= size;
            }

            // 计算线性索引并执行复数累加（A*B为复数乘法，sum+=为复数加法）
            int idx_a = 0, idx_b = 0;
            for (int i = 0; i < rank_A; i++)
                idx_a += indices_A[i] * strides_A[i];
            for (int i = 0; i < rank_B; i++)
                idx_b += indices_B[i] * strides_B[i];

            sum += A[idx_a] * B[idx_b]; // 复数核心运算：乘法+加法
        }

        // 计算C的线性索引并赋值（复数赋值）
        int idx_c_linear = 0;
        for (int i = 0; i < rank_C; i++)
        {
            idx_c_linear += indices_C[i] * strides_C[i];
        }
        C[idx_c_linear] = sum;
    }
}

__global__ void computeSLAmpKernel(
    thrust::complex<double>* d_amp, thrust::complex<double>* d_amp_buffer,
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const int* d_dimj, const int* d_dimj1, const int* d_dimj2,
    const SL* d_slCombination, int num_sl, int num_events, int num_polar,
    int decayChain_size, int buffer_size_per_event, int num_batchs,
    int start_events)
{
    int eventIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (eventIdx >= num_batchs)
    {
        return;
    }

    extern __shared__ thrust::complex<double> shared_buf[];

    // 为当前事件分配缓冲区
    thrust::complex<double>* event_buffer =
        &d_amp_buffer[eventIdx * buffer_size_per_event];
    int buffer_used = 0;

    for (int slIdx = 0; slIdx < num_sl; ++slIdx)
    {
        // 重置buffer
        // 存储当前振幅的标签（粒子index）
        int ampLabels[20]; // 假设最多20个粒子标签
        int ampLabelCount = 0;

        // 存储当前总振幅的数据、形状和秩
        thrust::complex<double>* currentAmp = nullptr;
        // thrust::complex<double> *currentAmp =
        // &event_buffer[buffer_size_per_event];
        int currentAmpShape[10]; // 假设最大秩为10
        int currentAmpRank = 0;

        // 按照节点顺序计算振幅
        for (int nodeIdx = 0; nodeIdx < decayChain_size; ++nodeIdx)
        {
            const DecayNode& node = d_decayNodes[nodeIdx];
            const SL& sl = d_slCombination[nodeIdx + slIdx * decayChain_size];

            int dim_j = d_dimj[nodeIdx];
            int dim_j1 = d_dimj1[nodeIdx];
            int dim_j2 = d_dimj2[nodeIdx];

            // 直接从设备内存获取四动量
            LorentzVector pDaug1 =
                d_momenta->getMomentum(start_events + eventIdx, node.daug1_idx);
            LorentzVector pDaug2 =
                d_momenta->getMomentum(start_events + eventIdx, node.daug2_idx);

            // 打印四动量
            // printf("Event %d, SL %d, Node %d: pDaug1 = (%f, %f, %f, %f),
            // pDaug2 =
            // (%f, %f, %f, %f)\n", eventIdx, slIdx, nodeIdx, pDaug1.E,
            // pDaug1.Px, pDaug1.Py, pDaug1.Pz, pDaug2.E, pDaug2.Px, pDaug2.Py,
            // pDaug2.Pz);

            // node振幅
            size_t amp_size = dim_j * dim_j1 * dim_j2;
            thrust::complex<double>* node_amp = &event_buffer[buffer_used];
            buffer_used += amp_size;

            // 计算振幅
            pwa_amp(node_amp, pDaug1, dim_j1, pDaug2, dim_j2, dim_j, sl.S, sl.L,
                &event_buffer[buffer_used]);
            // int max_dim = max(2 * dj + 1, 2 * dj2 + 1);
            // shared_used += (2 * dj1 + 1) * (2 * dj1 + 1) + (2 * dj2 + 1) * (2
            // * dj2
            // + 1) + 2 * (2 * max_dim + 1) * (2 * max_dim + 1) + 2 * (2 * dj +
            // 1) * (2 * dj1 + 1) * (2 * dj2 + 1); int dim_j = 2 * dj + 1; int
            // dim_j1 = 2 * dj1 + 1; int dim_j2 = 2 * dj2 + 1;

            int total_amp_size = dim_j * dim_j1 * dim_j2;
            int trans1_size = dim_j1 * dim_j1;
            int trans2_size = dim_j2 * dim_j2;
            int max_dim = max(dim_j1, dim_j2);
            int massive_shared_size = 2 * max_dim * max_dim;
            // int massive_shared_size = max(2 * trans1_size, 2 * trans2_size);

            buffer_used += 2 * total_amp_size + trans1_size + trans2_size +
                massive_shared_size;

            // for (int i = 0; i < dim_j; i++)
            // {
            //     for (int j = 0; j < dim_j1; j++)
            //     {
            //         for (int k = 0; k < dim_j2; k++)
            //         {
            //             int idx = i * dim_j1 * dim_j2 + j * dim_j2 + k;
            //             printf("Event %d, sl %d, Node %d: Amp[%d,%d,%d] =
            //             (%f, %f i)\n", eventIdx, slIdx, nodeIdx, i, j, k,
            //             node_amp[idx].real(), node_amp[idx].imag());
            //             // printf("Event %d, sl %d, Node %d: Amp[%d,%d,%d] =
            //             (%f, %f i)\n", eventIdx, slIdx, nodeIdx, i, j, k,
            //             d_amp_buffer[idx].real(), d_amp_buffer[idx].imag());
            //         }
            //     }
            // }

            // int nodeAmpShape[3] = {2 * dj + 1, 2 * dj1 + 1, 2 * dj2 + 1};
            int nodeAmpShape[3] = { dim_j, dim_j1, dim_j2 };
            int nodeAmpRank = 3;

            if (nodeIdx == 0)
            {
                // 第一个节点，直接作为总振幅
                currentAmp = node_amp;
                currentAmpRank = 3;
                currentAmpShape[0] = dim_j;
                currentAmpShape[1] = dim_j1;
                currentAmpShape[2] = dim_j2;

                // 设置标签 [mother, daug1, daug2]
                ampLabels[0] = node.mother_idx;
                ampLabels[1] = node.daug1_idx;
                ampLabels[2] = node.daug2_idx;
                ampLabelCount = 3;

                continue;
            }

            // 寻找缩并位置（在现有标签中查找母粒子）
            int contractIndex = -1;
            for (int j = 0; j < ampLabelCount; ++j)
            {
                if (ampLabels[j] == node.mother_idx)
                {
                    contractIndex = j;
                    break;
                }
            }

            if (contractIndex == -1)
            {
                // 错误处理：应该总是能找到母粒子
                printf("Error: Mother particle %d not found in labels\n",
                    node.mother_idx);
                return;
            }

            // 执行张量缩并
            // 总振幅形状: [shape0, shape1, ..., shape_{contractIndex}, ...,
            // shape_{currentAmpRank-1}] 节点振幅形状: [dj, dj1, dj2] 缩并:
            // 总振幅的第contractIndex维与节点振幅的第0维

            // 计算输出形状
            int outputRank =
                currentAmpRank + nodeAmpRank - 2; // 去掉两个缩并维度
            int outputShape[10];
            int outputSize = 1;

            // 构建输出形状
            int pos = 0;
            for (int i = 0; i < currentAmpRank; ++i)
            {
                if (i != contractIndex)
                {
                    outputShape[pos++] = currentAmpShape[i];
                    outputSize *= currentAmpShape[i];
                }
            }
            for (int i = 1; i < nodeAmpRank;
                ++i) // 跳过节点振幅的第0维（被缩并）
            {
                outputShape[pos++] = nodeAmpShape[i];
                outputSize *= nodeAmpShape[i];
            }

            // 分配输出空间
            // thrust::complex<double> *outputAmp = &d_amp_buffer[buffer_used];
            thrust::complex<double>* outputAmp = &event_buffer[buffer_used];
            buffer_used += outputSize;

            // 设置缩并维度
            int contractDimsA[1] = { contractIndex };
            int contractDimsB[1] = { 0 };

            // 执行缩并
            contract<10>(currentAmp, currentAmpShape, currentAmpRank, node_amp,
                nodeAmpShape, nodeAmpRank, outputAmp, outputShape,
                outputRank, contractDimsA, contractDimsB, 1);

            // 更新总振幅
            currentAmp = outputAmp;
            currentAmpRank = outputRank;
            for (int i = 0; i < outputRank; ++i)
            {
                currentAmpShape[i] = outputShape[i];
            }

            // 更新标签：删除被缩并的母粒子，添加新的子粒子
            for (int j = contractIndex; j < ampLabelCount - 1; ++j)
            {
                ampLabels[j] = ampLabels[j + 1];
            }
            ampLabels[ampLabelCount - 1] = node.daug1_idx;
            ampLabels[ampLabelCount] = node.daug2_idx;
            ampLabelCount += 1; // -1 + 2 = +1

            // printf("Event %d, Node %d: Contraction completed. New rank:
            // %d\n",eventIdx, nodeIdx, currentAmpRank);
        }

        // d_amp按
        // thrust::complex<double>* event_final_amp =
        //     &d_amp[slIdx * num_events * num_polar +
        //     (start_events + eventIdx) * num_polar];
        for (int i = 0; i < num_polar; ++i)
        {
            int idx = slIdx * num_events * num_polar + (start_events + eventIdx) * num_polar + i;
            // int idx = (start_events + eventIdx) * num_sl * num_polar + i * num_sl + slIdx;
            d_amp[idx] = currentAmp[i];
            // event_final_amp[i] = currentAmp[i];
            // printf("Event %d, sl %d, Final Amp[%d] = (%f, %f i)\n", eventIdx, slIdx, i, event_final_amp[i].real(), event_final_amp[i].imag());
            // printf("Event %d, sl %d, k %d, Final Amp[%d] = (%f, %f i)\n", eventIdx, slIdx, i, idx, currentAmp[i].real(), currentAmp[i].imag());
        }

        // printf("Event %d: Final amplitude size: %d\n", eventIdx,
        // finalAmpSize);
    }
}

void AmpCasDecay::getAmps(std::vector<cuComplex*>& d_amplitudes,
    const std::vector<Resonance>& resonances,
    const int site,
    const int n_amplitudes,
    const std::vector<std::vector<int>>& event_offsets,
    const std::vector<std::vector<int>>& amp_offsets)
{
    for (size_t i = 0; i < d_amplitudes.size(); ++i)
    {
        cudaSetDevice(i);

        // 分配设备内存用于共振态数组
        DeviceResonance* d_resonances;
        size_t resonance_count = resonances.size();
        cudaMalloc(&d_resonances, resonance_count * sizeof(DeviceResonance));

        // 将每个共振态转换为设备格式并复制到设备内存
        std::vector<DeviceResonance> host_resonances;
        for (auto& resonance : resonances)
        {
            DeviceResonance devRes;
            devRes.J = resonance.getJ();
            devRes.P = resonance.getP();
            devRes.particle_idx = particleToIndex_[resonance.getTag()];
            devRes.type = resonance.getModelType();

            const auto& params = resonance.getParams();
            devRes.param_count = std::min(8, (int)params.size());
            int i = 0;
            for (const auto& param : params)
            {
                if (i >= 8)
                    break;
                devRes.params[i] = param.second;
                i++;
            }
            host_resonances.push_back(devRes);
        }

        // 将共振态数组复制到设备
        cudaMemcpy(d_resonances, host_resonances.data(),
            resonance_count * sizeof(DeviceResonance),
            cudaMemcpyHostToDevice);

        // 分配设备内存用于输出振幅
        // cuComplex *d_amplitudes;
        // size_t total_amplitudes = nSLCombs_ * nEvents_ * nPolarizations_;
        // cudaMalloc(&d_amplitudes, total_amplitudes * sizeof(cuComplex));

        // std::cout << "SL Combinations: " << nSLCombs_ << ", Events: " << nEvents_
        // << ", Polarizations: " << nPolarizations_ << ", site: " << site <<
        // std::endl;
        int* d_amp_offsets;
        cudaMalloc(&d_amp_offsets, amp_offsets[i].size() * sizeof(int));
        cudaMemcpy(d_amp_offsets, amp_offsets[i].data(), amp_offsets[i].size() * sizeof(int), cudaMemcpyHostToDevice);
        int* d_event_offsets;
        cudaMalloc(&d_event_offsets, event_offsets[i].size() * sizeof(int));
        cudaMemcpy(d_event_offsets, event_offsets[i].data(), event_offsets[i].size() * sizeof(int), cudaMemcpyHostToDevice);
        int num_offsets = amp_offsets[i].size();

        // 设置核函数配置
        dim3 blockDim(256); // 每个块256个线程
        dim3 gridDim(nSLCombs_, (nEvents_[i] + blockDim.x - 1) / blockDim.x);
        // int blockSize = 256; // 每个块256个线程
        // int numBlocks = (nEvents_[i] + blockSize - 1) / blockSize;

        // 调用核函数计算振幅
        // computeAmpsKernel<<<numBlocks, blockSize>>>(
        computeAmpsKernel << <gridDim, blockDim >> >
            (d_amplitudes[i],       // 输出振幅
                d_momenta_[i],         // 四动量数据
                d_slCombination_[i],   // SL组合
                d_slamps_[i],          // SL振幅
                d_resonances,       // 共振态数组
                resonance_count,    // 共振态数量
                d_decayNodes_[i],      // 衰变链信息
                decayChain_.size(), // 衰变链长度
                nEvents_[i],           // 事件数
                nSLCombs_,          // SL组合数
                nPolarizations_,    // 极化态数
                d_amp_offsets,      // 振幅偏移量
                d_event_offsets,    // 事件偏移量
                num_offsets,        // 偏移量数量
                n_amplitudes,       // 振幅数量
                site                // 位置
                );

        cudaDeviceSynchronize();

        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error in computeAmps: " << cudaGetErrorString(err)
                << std::endl;
            cudaFree(d_resonances);
            // cudaFree(d_amplitudes);
            // return nullptr;
        }

        // 释放共振态数组设备内存
        cudaFree(d_resonances);
    }
}

__global__ void
computeAmpsKernel(cuComplex* amplitudes,                 // 输出振幅
    const DeviceMomenta* d_momenta,        // 所有事件的四动量数据
    const SL* slCombinations,              // SL组合数据
    const thrust::complex<double>* slamps, // SL振幅
    const DeviceResonance* resonances,     // 共振态数组
    int resonance_count,                   // 共振态数量
    const DecayNode* decayChain,           // 衰变链信息
    int decayChain_size, int nEvents, int nSLComb, int nPolar,
    const int* amp_offsets, const int* event_offsets,
    int num_offsets, int n_amplitudes, int site)
{
    // int event_idx = threadIdx.x * blockDim.x + threadIdx.x;
    int sl_idx = blockIdx.x;
    int event_idx = threadIdx.x + blockDim.x * blockIdx.y;

    if (sl_idx >= nSLComb || event_idx >= nEvents)
    {
        // if (event_idx >= nEvents)
        return;
    }

    // event_idx处于event_offsets哪个位置
    int offset_idx = 0;
    for (size_t i = 0; i < num_offsets - 1; ++i)
    {
        if (event_idx < event_offsets[i + 1])
        {
            offset_idx = i;
            break;
        }
    }
    // printf("Event %d, SL %d: Offset index = %d, event_offsets = %d,
    // amp_offsets = %d\n", event_idx, sl_idx, offset_idx,
    // event_offsets[offset_idx], amp_offsets[offset_idx]);

    thrust::complex<float> resAmp(1.0, 0.0);

    // 遍历衰变链中的每个节点
    for (int nodeIdx = 0; nodeIdx < decayChain_size; ++nodeIdx)
    {
        const DecayNode& node = decayChain[nodeIdx];
        const SL& sl = slCombinations[nodeIdx + sl_idx * decayChain_size];

        // 获取母粒子四动量
        LorentzVector pMother =
            d_momenta->getMomentum(event_idx, node.mother_idx);
        LorentzVector pDaug1 =
            d_momenta->getMomentum(event_idx, node.daug1_idx);
        LorentzVector pDaug2 =
            d_momenta->getMomentum(event_idx, node.daug2_idx);

        double mm = pMother.M();
        double qq =
            std::sqrt((mm * mm - std::pow(pDaug1.M() + pDaug2.M(), 2)) *
                (mm * mm - std::pow(pDaug1.M() - pDaug2.M(), 2))) /
            2 / mm;

        double mass_mother = decayChain[nodeIdx].mass[0];
        double mass_daug1 = decayChain[nodeIdx].mass[1];
        double mass_daug2 = decayChain[nodeIdx].mass[2];

        // 检查当前节点是否对应某个共振态
        bool is_resonance_node = false;
        DeviceResonance current_res;

        for (int i = 0; i < resonance_count; ++i)
        {
            if (decayChain[nodeIdx].mother_idx == resonances[i].particle_idx)
            {
                is_resonance_node = true;
                current_res = resonances[i];
                break;
            }
        }

        // 更新质量参数
        if (mass_mother == -1 && is_resonance_node)
        {
            mass_mother = current_res.params[0];
        }
        if (mass_daug1 == -1)
        {
            for (int i = 0; i < resonance_count; ++i)
            {
                if (decayChain[nodeIdx].daug1_idx == resonances[i].particle_idx)
                {
                    mass_daug1 = resonances[i].params[0];
                    break;
                }
            }
        }
        if (mass_daug2 == -1)
        {
            for (int i = 0; i < resonance_count; ++i)
            {
                if (decayChain[nodeIdx].daug2_idx == resonances[i].particle_idx)
                {
                    mass_daug2 = resonances[i].params[0];
                    break;
                }
            }
        }

        double q0 = std::sqrt((mass_mother * mass_mother -
            std::pow(mass_daug1 + mass_daug2, 2)) *
            (mass_mother * mass_mother -
                std::pow(mass_daug1 - mass_daug2, 2))) /
            2 / mass_mother;

        if (nodeIdx == 0)
        {
            // 第一个节点特殊处理
            // resAmp *= BlattWeisskopf(sl.L, qq, q0);
            resAmp *= Bf<double>(sl.L, qq, q0);

            // printf("Event %d, sl %d, First Node: L=%d, qq=%f, q0=%f, BW
            // Factor=(%f, %f i)\n", event_idx, sl_idx, sl.L, qq, q0,
            // resAmp.real(), resAmp.imag());
            continue;
        }

        // 如果是共振态节点，计算相应的振幅因子
        if (is_resonance_node)
        {
            if (current_res.type == ResModelType::BWR)
            {
                resAmp *= BWR<double>(mm, current_res.params[0],
                    current_res.params[1], sl.L, qq, q0);
                resAmp *= Bf<double>(sl.L, qq, q0);
            }
            else if (current_res.type == ResModelType::ONE)
            {
                resAmp *= Bf<double>(sl.L, qq, q0);
            }
            else if (current_res.type == ResModelType::Flatte)
            {
                resAmp *= Flatte(mm, current_res.params[0],
                    current_res.params[1], current_res.params[2]);
                resAmp *= Bf<double>(sl.L, qq, q0);
            }
        }
    }

    // 计算极化相关的振幅
    for (int k = 0; k < nPolar; ++k)
    {
        int idx = sl_idx * nPolar * nEvents + event_idx * nPolar + k;
        // int idx = event_idx * nPolar * n_amplitudes + k * n_amplitudes + sl_idx;
        // int amp_idx = site * nEvents * nPolar + idx;
        int amp_idx = 0;
        if (offset_idx < num_offsets)
        {
            int nEvents = event_offsets[offset_idx + 1] - event_offsets[offset_idx];
            // int nEvents = event_offsets[offset_idx + 1];
            // amp_idx = amp_offsets[offset_idx] + site * nEvents * nPolar + sl_idx * nPolar * nEvents + (event_idx - event_offsets[offset_idx]) * nPolar + k;
            amp_idx = amp_offsets[offset_idx] + (event_idx - event_offsets[offset_idx]) * n_amplitudes * nPolar + k * n_amplitudes + sl_idx + site;
        }
        else
        {
            return;
        }

        // printf("Event %d, SL %d, Polarization %d: amp_idx = %d\n", event_idx,
        // sl_idx, k, amp_idx);

        thrust::complex<float> temp = resAmp * slamps[idx] * 100.0f;
        amplitudes[amp_idx] = make_cuComplex(temp.real(), temp.imag());

        // 打印
        // printf("Event %d, sl %d, Amp[%d] = (%f, %f i)\n", event_idx, sl_idx, k, temp.real(), temp.imag());
        // printf("Event %d, sl %d, ResAmp[%d] = (% f, % f i)\n", event_idx, sl_idx, k, resAmp.real(), resAmp.imag());
        // printf("Event %d, sl %d, pol %d, slamp[%d] = (%f, %f i), amp[%d] = (%f, %f i)\n", event_idx, sl_idx, k, idx, slamps[idx].real(), slamps[idx].imag(), amp_idx, cuCrealf(amplitudes[amp_idx]), cuCimagf(amplitudes[amp_idx]));
    }
}
// }
