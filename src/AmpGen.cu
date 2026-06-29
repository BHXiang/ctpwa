#include <AmpGen.cuh>
#include <ComputeNLL.cuh>
#include <Parameters.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <iostream>

// Amp2BD 类实现
Amp2BD::Amp2BD(std::array<int, 3> jvalues, std::array<int, 3> parities,
    bool identical_daughters, bool is_boson, int maxL, bool p_break, bool is_bf)
    : jvalues_(jvalues), parities_(parities),
    identical_daughters_(identical_daughters), is_boson_(is_boson),
    maxL_(maxL), p_break_(p_break), is_bf_(is_bf)
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

            // 宇称条件（若宇称破缺则跳过该条件，如弱衰变）
            if (p_break_ || p1 == p2 * p3 * sign)
            {
                // 全同粒子选择定则: (-1)^{L+S} = +1 (Bose) 或 -1 (Fermi)
                if (identical_daughters_) {
                    int S = two_S / 2; // 实际的总自旋量子数
                    int parity_LS = ((L + S) % 2 == 0) ? 1 : -1;
                    int required = is_boson_ ? 1 : -1;
                    if (parity_LS != required) continue;
                }

                // 轨道角动量上限截断
                if (maxL_ > 0 && L > maxL_) continue;

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
    nPolarizations_total_ = 0;
}

AmpCasDecay::~AmpCasDecay()
{
    // cudaFree(d_slamps_);
    // cudaFree(d_momenta_);
    // cudaFree(d_decayNodes_);
    // cudaFree(d_slCombination_);
    for (size_t i = 0; i < d_slamps_.size(); ++i)
    {
        cudaSetDevice(static_cast<int>(i));
        if (d_slamps_[i])
            cudaFree(d_slamps_[i]);
        if (d_momenta_[i])
            cudaFree(d_momenta_[i]);
        if (d_decayNodes_[i])
            cudaFree(d_decayNodes_[i]);
        if (d_slCombination_[i])
            cudaFree(d_slCombination_[i]);
    }
    for (size_t i = 0; i < d_polarization_map_.size(); ++i)
    {
        cudaSetDevice(static_cast<int>(i));
        if (d_polarization_map_[i])
            cudaFree(d_polarization_map_[i]);
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

void AmpCasDecay::setPolarizationMap(const std::vector<int>& map)
{
    h_polarization_map_ = map; // stored on host, uploaded to GPU in computeSLAmps
}

void AmpCasDecay::setPermutedMappings(const std::vector<std::map<std::string, int>>& maps, bool is_boson)
{
    permuted_mappings_ = maps;
    identical_boson_ = is_boson;
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
    d_polarization_map_.resize(finalMomenta.size(), nullptr);

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

        // 上传极化 mask 到设备
        if (!h_polarization_map_.empty()) {
            cudaMalloc(&d_polarization_map_[i], h_polarization_map_.size() * sizeof(int));
            cudaMemcpy(d_polarization_map_[i], h_polarization_map_.data(),
                h_polarization_map_.size() * sizeof(int), cudaMemcpyHostToDevice);
        }

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
        int sharedMemSize = 3000 * sizeof(thrust::complex<double>);
        int blockSize = 256;
        int numBlocks = (nEvents_[i] + blockSize - 1) / blockSize;

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
            // computeSLAmpKernel<<<gridDim, blockDim, sharedMemSize>>>(
            // computeSLAmpKernel<<<gridDim, blockDim>>>(
            computeSLAmpKernel << <numBlocks, blockSize, sharedMemSize >> > (
                d_slamps_[i], d_amp_buffer, d_momenta_[i], d_decayNodes_[i], d_dimj, d_dimj1,
                d_dimj2, d_slCombination_[i], nSLCombs_, nEvents_[i], nPolarizations_,
                decayChain_.size(), amp_size* nSLCombs_, n_events, start,
                d_polarization_map_.empty() ? nullptr : d_polarization_map_[i],
                nPolarizations_total_);

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

        // 跨链全同粒子: 对每个交换拓扑计算 SL 振幅并就地累加
        for (size_t p = 0; p < permuted_mappings_.size(); ++p) {
            // 用交换后的映射重建 DeviceMomenta（动量位置被交换）
            auto d_mom_perm = convertToDeviceMomenta(
                finalMomenta, permuted_mappings_[p], decayChain_);
            cudaSetDevice(i);  // restore device after convertToDeviceMomenta

            double sign = identical_boson_ ? 1.0 : -1.0;

            // 分配全尺寸临时 buffer（kernel 按绝对事件索引写入）
            size_t full_size = nEvents_[i] * nPolarizations_ * nSLCombs_;
            thrust::complex<double>* d_temp;
            cudaMalloc(&d_temp, full_size * sizeof(thrust::complex<double>));

            for (int start = 0; start < nEvents_[i]; start += batch_size) {
                int n_events = (start + batch_size <= nEvents_[i])
                    ? batch_size : (nEvents_[i] - start);

                // scratch buffer 按 batch 分配
                thrust::complex<double>* d_scratch;
                cudaMalloc(&d_scratch, n_events * nSLCombs_ * amp_size * sizeof(thrust::complex<double>));

                // 用交换后的动量计算 SL 振幅到临时 buffer
                computeSLAmpKernel << <numBlocks, blockSize, sharedMemSize >> > (
                    d_temp, d_scratch, d_mom_perm[i], d_decayNodes_[i],
                    d_dimj, d_dimj1, d_dimj2, d_slCombination_[i],
                    nSLCombs_, nEvents_[i], nPolarizations_,
                    decayChain_.size(), amp_size* nSLCombs_, n_events, start,
                    d_polarization_map_.empty() ? nullptr : d_polarization_map_[i],
                    nPolarizations_total_);
                cudaDeviceSynchronize();

                cudaFree(d_scratch);
            }

            // 一次性累加: d_slamps_[i] += sign * d_temp
            int blk = 256;
            int grd = (full_size + blk - 1) / blk;
            addSLAmpsKernel << <grd, blk >> > (d_slamps_[i], d_temp, full_size, sign);
            cudaDeviceSynchronize();

            cudaFree(d_temp);

            // 释放所有GPU的交换链 DeviceMomenta
            for (size_t g = 0; g < d_mom_perm.size(); ++g) {
                cudaSetDevice(static_cast<int>(g));
                cudaFree(d_mom_perm[g]->momenta);
                cudaFree(d_mom_perm[g]);
            }
            cudaSetDevice(i);  // restore device
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
    int start_events,
    const int* d_polarization_map, int num_polar_total)
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
            LorentzVector pDaug1 = d_momenta->getMomentum(start_events + eventIdx, node.daug1_idx);
            LorentzVector pDaug2 = d_momenta->getMomentum(start_events + eventIdx, node.daug2_idx);

            // // 打印四动量
            // printf("Event %d, SL %d, Node %d: pDaug1 = (%f, %f, %f, %f), pDaug2 = (% f, % f, % f, % f)\n",
            //     eventIdx, slIdx, nodeIdx,
            //     pDaug1.E, pDaug1.Px, pDaug1.Py, pDaug1.Pz,
            //     pDaug2.E, pDaug2.Px, pDaug2.Py, pDaug2.Pz);

            // node振幅
            size_t amp_size = dim_j * dim_j1 * dim_j2;
            thrust::complex<double>* node_amp = &event_buffer[buffer_used];
            buffer_used += amp_size;

            // 计算振幅
            pwa_amp(node_amp, pDaug1, dim_j1, pDaug2, dim_j2, dim_j, sl.S, sl.L, &event_buffer[buffer_used]);

            // // if (eventIdx == 0 && slIdx == 0) {
            // for (int i = 0; i < dim_j; i++)
            //     for (int j = 0; j < dim_j1; j++)
            //         for (int k = 0; k < dim_j2; k++) {
            //             int idx = i * dim_j1 * dim_j2 + j * dim_j2 + k;
            //             printf("Node %d: Amp[%d,%d,%d] = (%f, %f)\n",
            //                 nodeIdx, i, j, k,
            //                 node_amp[idx].real(), node_amp[idx].imag());
            //         }
            // // }

            int total_amp_size = dim_j * dim_j1 * dim_j2;
            int trans1_size = dim_j1 * dim_j1;
            int trans2_size = dim_j2 * dim_j2;
            int max_dim = max(dim_j1, dim_j2);
            int massive_shared_size = 2 * max_dim * max_dim;
            // int massive_shared_size = max(2 * trans1_size, 2 * trans2_size);

            buffer_used += 2 * total_amp_size + trans1_size + trans2_size +
                massive_shared_size;

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
        // 按极化 mask 选取张量索引写入（Strategy B: kernel 内筛选）
        if (d_polarization_map != nullptr) {
            for (int i = 0; i < num_polar; ++i) {
                int tensor_idx = d_polarization_map[i];
                int idx = slIdx * num_events * num_polar + (start_events + eventIdx) * num_polar + i;
                d_amp[idx] = currentAmp[tensor_idx];
            }
        }
        else {
            for (int i = 0; i < num_polar; ++i) {
                int idx = slIdx * num_events * num_polar + (start_events + eventIdx) * num_polar + i;
                d_amp[idx] = currentAmp[i];
            }
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
    const std::vector<std::vector<int>>& amp_offsets,
    double bf_d)
{
    for (size_t i = 0; i < d_amplitudes.size(); ++i)
    {
        cudaSetDevice(i);

        // 分配设备内存用于共振态数组
        size_t resonance_count = resonances.size();
        DeviceResonance* d_resonances;
        cudaMalloc(&d_resonances, resonance_count * sizeof(DeviceResonance));

        // 构建 flat 参数数组和 channel 数组（主机端）
        std::vector<DeviceResonance> host_resonances;
        std::vector<double> h_all_params;
        std::vector<double> h_all_channels;

        for (auto& resonance : resonances)
        {
            DeviceResonance devRes;
            devRes.J = resonance.getJ();
            devRes.P = resonance.getP();
            devRes.particle_idx = particleToIndex_[resonance.getTag()];
            devRes.type = resonance.getModelType();

            // 自由参数 offset
            auto ordered_params = resonance.getOrderedParams();
            devRes.param_offset = static_cast<int>(h_all_params.size());
            devRes.param_count = static_cast<int>(ordered_params.size());
            h_all_params.insert(h_all_params.end(),
                ordered_params.begin(), ordered_params.end());

            // Flatte channel masses
            const auto& channels = resonance.getChannels();
            devRes.n_channels = static_cast<int>(channels.size());
            devRes.channel_offset = static_cast<int>(h_all_channels.size());
            for (const auto& ch : channels) {
                h_all_channels.push_back(ch.first);
                h_all_channels.push_back(ch.second);
            }

            host_resonances.push_back(devRes);
        }

        // 上传 DeviceResonance 数组
        cudaMemcpy(d_resonances, host_resonances.data(),
            resonance_count * sizeof(DeviceResonance),
            cudaMemcpyHostToDevice);

        // 上传 flat 参数数组
        double* d_all_params = nullptr;
        if (!h_all_params.empty()) {
            cudaMalloc(&d_all_params, h_all_params.size() * sizeof(double));
            cudaMemcpy(d_all_params, h_all_params.data(),
                h_all_params.size() * sizeof(double), cudaMemcpyHostToDevice);
        }

        // 上传 channel masses 数组
        double* d_all_channels = nullptr;
        if (!h_all_channels.empty()) {
            cudaMalloc(&d_all_channels, h_all_channels.size() * sizeof(double));
            cudaMemcpy(d_all_channels, h_all_channels.data(),
                h_all_channels.size() * sizeof(double), cudaMemcpyHostToDevice);
        }

        int* d_amp_offsets;
        cudaMalloc(&d_amp_offsets, amp_offsets[i].size() * sizeof(int));
        cudaMemcpy(d_amp_offsets, amp_offsets[i].data(), amp_offsets[i].size() * sizeof(int), cudaMemcpyHostToDevice);
        int* d_event_offsets;
        cudaMalloc(&d_event_offsets, event_offsets[i].size() * sizeof(int));
        cudaMemcpy(d_event_offsets, event_offsets[i].data(), event_offsets[i].size() * sizeof(int), cudaMemcpyHostToDevice);
        int num_offsets = amp_offsets[i].size();

        // 设置核函数配置
        dim3 blockDim(256);
        dim3 gridDim(nSLCombs_, (nEvents_[i] + blockDim.x - 1) / blockDim.x);

        // skip empty calls (no SL combos or no resonances for this combination)
        if (nEvents_[i] == 0 || nSLCombs_ == 0 || resonance_count == 0) {
            cudaFree(d_resonances);
            if (d_all_params) cudaFree(d_all_params);
            if (d_all_channels) cudaFree(d_all_channels);
            cudaFree(d_amp_offsets);
            cudaFree(d_event_offsets);
            continue;
        }

        // 调用核函数计算振幅
        computeAmpsKernel << <gridDim, blockDim >> >
            (d_amplitudes[i],       // 输出振幅
                d_momenta_[i],         // 四动量数据
                d_slCombination_[i],   // SL组合
                d_slamps_[i],          // SL振幅
                d_resonances,       // 共振态数组
                resonance_count,    // 共振态数量
                d_all_params,       // flat 自由参数
                d_all_channels,     // flat channel masses
                d_decayNodes_[i],      // 衰变链信息
                decayChain_.size(), // 衰变链长度
                nEvents_[i],           // 事件数
                nSLCombs_,          // SL组合数
                nPolarizations_,    // 极化态数
                d_amp_offsets,      // 振幅偏移量
                d_event_offsets,    // 事件偏移量
                num_offsets,        // 偏移量数量
                n_amplitudes,       // 振幅数量
                site,               // 位置
                bf_d                // 势垒因子 d
                );

        cudaDeviceSynchronize();

        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error in computeAmps: " << cudaGetErrorString(err)
                << std::endl;
        }

        // 释放临时设备内存
        cudaFree(d_resonances);
        if (d_all_params) cudaFree(d_all_params);
        if (d_all_channels) cudaFree(d_all_channels);
        cudaFree(d_amp_offsets);
        cudaFree(d_event_offsets);
    }
}

__global__ void
computeAmpsKernel(cuComplex* amplitudes,                 // 输出振幅
    const DeviceMomenta* d_momenta,        // 所有事件的四动量数据
    const SL* slCombinations,              // SL组合数据
    const thrust::complex<double>* slamps, // SL振幅
    const DeviceResonance* resonances,     // 共振态数组
    int resonance_count,                   // 共振态数量
    const double* d_all_params,            // 所有共振态的自由参数（flat）
    const double* d_all_channels,          // Flatte channel masses（flat）
    const DecayNode* decayChain,           // 衰变链信息
    int decayChain_size, int nEvents, int nSLComb, int nPolar,
    const int* amp_offsets, const int* event_offsets,
    int num_offsets, int n_amplitudes, int site,
    double bf_d)
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
        double qq = std::sqrt((mm * mm - std::pow(pDaug1.M() + pDaug2.M(), 2)) *
                (mm * mm - std::pow(pDaug1.M() - pDaug2.M(), 2))) / 2 / mm;

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
            mass_mother = d_all_params[current_res.param_offset];
        }
        if (mass_daug1 == -1)
        {
            for (int i = 0; i < resonance_count; ++i)
            {
                if (decayChain[nodeIdx].daug1_idx == resonances[i].particle_idx)
                {
                    mass_daug1 = d_all_params[resonances[i].param_offset];
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
                    mass_daug2 = d_all_params[resonances[i].param_offset];
                    break;
                }
            }
        }

        double q0 = std::sqrt((mass_mother * mass_mother - std::pow(mass_daug1 + mass_daug2, 2)) * (mass_mother * mass_mother - std::pow(mass_daug1 - mass_daug2, 2))) / 2 / mass_mother;
         
        // printf("mother mass = %f, daug1 mass = %f, daug2 mass = %f, q0 = %f\n",
        //     mass_mother, mass_daug1, mass_daug2, q0);
        // // 打印qq
        // printf("mm = %f, mdaug1 = %f, daug2 mass = %f, qq = %f\n", mm, pDaug1.M(), pDaug2.M(), qq);

        if (nodeIdx == 0)
        {
            // 第一个节点特殊处理
            // resAmp *= BlattWeisskopf(sl.L, qq, q0);
            resAmp *= Bf<double>(sl.L, qq, q0, bf_d);

            // printf("Event %d, SL %d, Node %d: First node, resAmp = (%f, %f)\n",
            //     event_idx, sl_idx, nodeIdx, resAmp.real(), resAmp.imag());

            continue;
        }

        // 如果是共振态节点，计算相应的振幅因子
        if (is_resonance_node)
        {
            const double* p = d_all_params + current_res.param_offset;
            const double* ch = (current_res.type == ResModelType::Flatte)
                ? d_all_channels + current_res.channel_offset : nullptr;
            resAmp *= computeNodeFactor<double>(sl.L, mm, qq, q0,
                p, current_res.param_count, current_res.type,
                ch, current_res.n_channels, bf_d);
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

        thrust::complex<float> temp = resAmp * slamps[idx]; // * 100.0f;
        amplitudes[amp_idx] = make_cuComplex(temp.real(), temp.imag());

        // printf("Event %d, SL %d, Polarization %d: resAmp = (%f, %f), slamp = (%f, %f), Final Amp = (%f, %f)\n",
        //     event_idx, sl_idx, k,
        //     resAmp.real(), resAmp.imag(),
        //     slamps[idx].real(), slamps[idx].imag(),
        //     temp.real(), temp.imag());

    }
}

// 跨链全同粒子: 将交换拓扑的 SL 振幅就地累加到原始振幅
// d_amp += sign * d_add,  sign = +1 (Bose) or -1 (Fermi)
__global__ void addSLAmpsKernel(
    thrust::complex<double>* d_amp,
    const thrust::complex<double>* d_add,
    int total_size, double sign)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    d_amp[idx] = thrust::complex<double>(
        d_amp[idx].real() + sign * d_add[idx].real(),
        d_amp[idx].imag() + sign * d_add[idx].imag());
}

// ============================================================
// AmpCalc 实现
// ============================================================

// 小 kernel：将自由参数写入 device 端的 d_all_params 数组
__global__ void updateResonanceParamsKernel(
    double* d_all_params,               // flat 自由参数数组
    const DeviceResonance* d_resonances,
    int resonance_count,
    const double* d_free_params,        // 全局自由参数数组
    const int* d_res_idx,               // 每个自由参数 → 共振态索引
    const int* d_param_idx,             // 每个自由参数 → params[] 下标
    const int* d_global_offset,         // 每个本block自由参数 → 全局slot下标
    int n_local_free)                   // 本 block 涉及的自由参数数
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local_free) return;
    int r = d_res_idx[i];
    int p = d_param_idx[i];
    if (r < resonance_count) {
        int offset = d_resonances[r].param_offset;
        d_all_params[offset + p] = d_free_params[d_global_offset[i]];
    }
}

AmpCalc::~AmpCalc()
{
    for (auto& block : blocks_) {
        for (size_t gpu = 0; gpu < block.d_resonances.size(); ++gpu) {
            cudaSetDevice(static_cast<int>(gpu));
            if (block.d_resonances[gpu]) cudaFree(block.d_resonances[gpu]);
            if (block.d_all_params[gpu]) cudaFree(block.d_all_params[gpu]);
            if (block.d_all_channels[gpu]) cudaFree(block.d_all_channels[gpu]);
            if (block.d_T.size() > gpu && block.d_T[gpu]) cudaFree(block.d_T[gpu]);
        }
    }
    // cas_list_ 由 shared_ptr 自动释放
}

// 辅助：根据 free_indices 将自由参数的 params[] 下标展开
// free_indices: {-1}=全扫, {0,1}=扫params[0]和[1], 空=不拟合
static std::vector<int> expandFreeIndices(
    const std::vector<double>& params,
    const std::vector<int>& free_indices)
{
    std::vector<int> result;
    if (free_indices.empty()) return result;
    if (free_indices.size() == 1 && free_indices[0] == -1) {
        // 全扫
        for (int i = 0; i < static_cast<int>(params.size()); ++i) {
            result.push_back(i);
        }
    } else {
        for (int idx : free_indices) {
            if (idx >= 0 && idx < static_cast<int>(params.size())) {
                result.push_back(idx);
            }
        }
    }
    return result;
}

void AmpCalc::addBlock(std::shared_ptr<AmpCasDecay> cas,
                       const std::vector<Resonance>& resonances,
                       int site,
                       const std::vector<std::vector<int>>& free_indices,
                       const std::vector<std::vector<std::vector<double>>>& free_ranges,
                       const std::set<std::string>& skip_slots_for,
                       const std::map<std::string, std::string>& conjugate_name_map)
{
    // 1. 查找或添加 cas 到 cas_list_
    int cas_idx = -1;
    for (size_t i = 0; i < cas_list_.size(); ++i) {
        if (cas_list_[i] == cas) {
            cas_idx = static_cast<int>(i);
            break;
        }
    }
    if (cas_idx < 0) {
        cas_idx = static_cast<int>(cas_list_.size());
        cas_list_.push_back(cas);
    }

    // 2. 为每个 GPU 分配持久化 DeviceResonance 和 flat 参数数组
    int n_gpu = static_cast<int>(cas->getSLAmps().size());
    ResBlock block;
    block.cas_idx = cas_idx;
    block.site = site;
    block.resonance_count = static_cast<int>(resonances.size());
    block.d_resonances.resize(n_gpu, nullptr);
    block.d_all_params.resize(n_gpu, nullptr);
    block.d_all_channels.resize(n_gpu, nullptr);
    block.d_T.resize(n_gpu, nullptr);

    // 构建主机端 DeviceResonance 数组 + flat params + flat channels
    std::vector<DeviceResonance> h_res;
    std::vector<double> h_all_params;
    std::vector<double> h_all_channels;

    for (const auto& res : resonances) {
        DeviceResonance dr;
        dr.J = res.getJ();
        dr.P = res.getP();
        dr.particle_idx = cas->getParticleIndex(res.getTag());
        dr.type = res.getModelType();

        auto ordered_params = res.getOrderedParams();
        dr.param_offset = static_cast<int>(h_all_params.size());
        dr.param_count = static_cast<int>(ordered_params.size());
        h_all_params.insert(h_all_params.end(),
            ordered_params.begin(), ordered_params.end());

        const auto& channels = res.getChannels();
        dr.n_channels = static_cast<int>(channels.size());
        dr.channel_offset = static_cast<int>(h_all_channels.size());
        for (const auto& ch : channels) {
            h_all_channels.push_back(ch.first);
            h_all_channels.push_back(ch.second);
        }

        h_res.push_back(dr);
    }
    h_templates_.push_back(h_res);
    h_param_templates_.push_back(h_all_params);
    h_channel_templates_.push_back(h_all_channels);

    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        // DeviceResonance 数组
        cudaMalloc(&block.d_resonances[gpu],
                   resonances.size() * sizeof(DeviceResonance));
        cudaMemcpy(block.d_resonances[gpu], h_res.data(),
                   resonances.size() * sizeof(DeviceResonance),
                   cudaMemcpyHostToDevice);
        // flat 自由参数
        if (!h_all_params.empty()) {
            cudaMalloc(&block.d_all_params[gpu],
                       h_all_params.size() * sizeof(double));
            cudaMemcpy(block.d_all_params[gpu], h_all_params.data(),
                       h_all_params.size() * sizeof(double), cudaMemcpyHostToDevice);
        }
        // flat channel masses
        if (!h_all_channels.empty()) {
            cudaMalloc(&block.d_all_channels[gpu],
                       h_all_channels.size() * sizeof(double));
            cudaMemcpy(block.d_all_channels[gpu], h_all_channels.data(),
                       h_all_channels.size() * sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    int block_idx = static_cast<int>(blocks_.size());
    blocks_.push_back(std::move(block));

    // 3. 记录参数槽映射
    for (size_t i = 0; i < resonances.size(); ++i) {
        // Skip slot creation for conjugate resonances (params shared via trans-linked chain)
        if (skip_slots_for.count(resonances[i].getName())) {
            // Determine owner name: use conjugate_name_map for name translation,
            // fall back to own name (for same-name matches)
            std::string owner_name = resonances[i].getName();
            auto cnm_it = conjugate_name_map.find(owner_name);
            if (cnm_it != conjugate_name_map.end())
                owner_name = cnm_it->second;
            auto owner_it = resonance_owners_.find(owner_name);
            if (owner_it != resonance_owners_.end()) {
                conjugate_broadcast_[{block_idx, (int)i}] = owner_it->second;
            }
            continue;
        }
        // Register as owner for this resonance name
        resonance_owners_[resonances[i].getName()] = {block_idx, (int)i};

        auto ordered_params = resonances[i].getOrderedParams();
        auto expanded = expandFreeIndices(ordered_params, free_indices[i]);

        // 获取该共振态的 free_ranges（可能为空）
        const auto& ranges = (i < free_ranges.size()) ? free_ranges[i]
                            : std::vector<std::vector<double>>{};

        for (size_t si = 0; si < expanded.size(); ++si) {
            int p_idx = expanded[si];
            double val = ordered_params[p_idx];
            double lower, upper;
            if (si < ranges.size() && ranges[si].size() >= 2) {
                lower = ranges[si][0];
                upper = ranges[si][1];
            } else {
                // 默认范围：val ± 50% * |val|，正参数下界不低于 0
                double half = std::abs(val) * 0.5;
                lower = val - half;
                upper = val + half;
            }
            slots_.push_back({block_idx, static_cast<int>(i), p_idx, val, lower, upper});
        }
    }
}

void AmpCalc::reComputeAmps(std::vector<cuComplex*>& d_amplitudes,
                            const double* d_params,
                            int n_amplitudes,
                            const std::vector<std::vector<int>>& event_offsets,
                            const std::vector<std::vector<int>>& amp_offsets,
                            size_t n_polar,
                            double bf_d)
{
    int n_gpu = static_cast<int>(d_amplitudes.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;

    // 1. 把 d_params 广播到所有 GPU
    std::vector<double*> d_params_per_gpu(n_gpu, nullptr);
    int primary_dev = 0;
    cudaGetDevice(&primary_dev);
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaMalloc(&d_params_per_gpu[gpu], n_free * sizeof(double));
    }
    // 从 primary GPU 拷贝到各 GPU
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == primary_dev) {
            cudaSetDevice(primary_dev);
            cudaMemcpy(d_params_per_gpu[primary_dev], d_params,
                       n_free * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyPeer(d_params_per_gpu[gpu], gpu,
                           d_params, primary_dev,
                           n_free * sizeof(double));
        }
    }

    // 2. 构建设备端的 slots 映射数组
    std::vector<int> h_slot_block(n_free), h_slot_res(n_free), h_slot_param(n_free);
    for (int i = 0; i < n_free; ++i) {
        h_slot_block[i] = slots_[i].block_idx;
        h_slot_res[i] = slots_[i].res_idx;
        h_slot_param[i] = slots_[i].param_idx;
    }

    // 3. 每个 GPU: 更新 DeviceResonance 参数 + 重跑 computeAmpsKernel
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        int blockSize = 256;

        // 为每个 block 更新 DeviceResonance 中的自由参数
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            auto& block = blocks_[bi];

            // 收集本 block 涉及的 slots（按原始 slots_ 顺序的子集）
            std::vector<int> local_res_idx, local_param_idx, local_global_offset;
            for (int s = 0; s < n_free; ++s) {
                if (slots_[s].block_idx == static_cast<int>(bi)) {
                    local_res_idx.push_back(slots_[s].res_idx);
                    local_param_idx.push_back(slots_[s].param_idx);
                    local_global_offset.push_back(s);
                }
            }
            int n_local = static_cast<int>(local_res_idx.size());
            if (n_local == 0) continue;

            int* d_res_idx, * d_param_idx, * d_global_offset;
            cudaMalloc(&d_res_idx, n_local * sizeof(int));
            cudaMalloc(&d_param_idx, n_local * sizeof(int));
            cudaMalloc(&d_global_offset, n_local * sizeof(int));
            cudaMemcpy(d_res_idx, local_res_idx.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_param_idx, local_param_idx.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_global_offset, local_global_offset.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);

            int grid = (n_local + blockSize - 1) / blockSize;
            updateResonanceParamsKernel<<<grid, blockSize>>>(
                block.d_all_params[gpu], block.d_resonances[gpu],
                block.resonance_count,
                d_params_per_gpu[gpu], d_res_idx, d_param_idx, d_global_offset, n_local);

            cudaDeviceSynchronize();
            cudaFree(d_res_idx);
            cudaFree(d_param_idx);
            cudaFree(d_global_offset);
        }

        // Broadcast theta params to conjugate blocks (cross-chain shared resonances)
        for (const auto& [cj_key, owner_key] : conjugate_broadcast_) {
            int cj_bi = cj_key.first, cj_ri = cj_key.second;
            int ow_bi = owner_key.first, ow_ri = owner_key.second;
            auto& cj_block = blocks_[cj_bi];
            auto& ow_block = blocks_[ow_bi];
            // Read owner param_offset and param_count from template
            const auto& h_template = h_templates_[ow_bi];
            int ow_offset = h_template[ow_ri].param_offset;
            int ow_count = h_template[ow_ri].param_count;
            const auto& cj_template = h_templates_[cj_bi];
            int cj_offset = cj_template[cj_ri].param_offset;
            // Copy param values from owner to conjugate
            double* d_ow_params = ow_block.d_all_params[gpu] + ow_offset;
            double* d_cj_params = cj_block.d_all_params[gpu] + cj_offset;
            cudaMemcpy(d_cj_params, d_ow_params,
                       ow_count * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();

        // 重跑 computeAmpsKernel
        for (auto& block : blocks_) {
            auto& cas = cas_list_[block.cas_idx];

            auto& d_slamps = cas->getSLAmps();
            auto& d_momenta = cas->getMomenta();
            auto& d_decayNodes = cas->getDecayNodes();
            auto& d_slComb = cas->getDeviceSLCombs();
            auto& nEvents = cas->getNEventsVec();

            // 准备 amp_offsets 和 event_offsets 的设备端数据
            int* d_amp_offsets;
            cudaMalloc(&d_amp_offsets, amp_offsets[gpu].size() * sizeof(int));
            cudaMemcpy(d_amp_offsets, amp_offsets[gpu].data(),
                       amp_offsets[gpu].size() * sizeof(int), cudaMemcpyHostToDevice);
            int* d_event_offsets;
            cudaMalloc(&d_event_offsets, event_offsets[gpu].size() * sizeof(int));
            cudaMemcpy(d_event_offsets, event_offsets[gpu].data(),
                       event_offsets[gpu].size() * sizeof(int), cudaMemcpyHostToDevice);
            int num_offsets = static_cast<int>(amp_offsets[gpu].size());

            dim3 gridDim(static_cast<unsigned int>(cas->getNSLCombs()),
                         (static_cast<unsigned int>(nEvents[gpu]) + 255) / 256);

            computeAmpsKernel<<<gridDim, blockSize>>>(
                d_amplitudes[gpu],
                d_momenta[gpu],
                d_slComb[gpu],
                d_slamps[gpu],
                block.d_resonances[gpu],
                block.resonance_count,
                block.d_all_params[gpu],
                block.d_all_channels[gpu],
                d_decayNodes[gpu],
                cas->getDecayChainSize(),
                static_cast<int>(nEvents[gpu]),
                static_cast<int>(cas->getNSLCombs()),
                static_cast<int>(n_polar),
                d_amp_offsets,
                d_event_offsets,
                num_offsets,
                n_amplitudes,
                block.site,
                bf_d);

            cudaDeviceSynchronize();
            cudaFree(d_amp_offsets);
            cudaFree(d_event_offsets);
        }
    }

    // 4. 释放临时的 d_params 副本
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (d_params_per_gpu[gpu]) {
            cudaSetDevice(gpu);
            cudaFree(d_params_per_gpu[gpu]);
        }
    }
    cudaSetDevice(primary_dev);
}

// ============================================================
// 共振态参数梯度
// ============================================================

// 辅助：破缺动量 q(m, m1, m2)
__device__ double breakup_momentum(double m, double m1, double m2) {
    double q_sq = (m*m - (m1+m2)*(m1+m2)) * (m*m - (m1-m2)*(m1-m2));
    if (q_sq <= 0.0) return 0.0;
    return sqrt(q_sq) / (2.0 * m);
}

// 梯度 kernel 模板实现 (完整衰变链 AutoDiff 版)
// R_total = Bf_0 * Propagator_1 * Bf_1 * Propagator_2 * Bf_2 * ...
// ∂R_total/∂θ 通过 AutoDiff 自动传播所有依赖关系（包括 Bf 对 mass 的依赖）
template <int Nfree>
__global__ void resonanceGradientKernel(
    const cuComplex* d_w, const cuComplex* d_T,
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const SL* d_slComb, const DeviceResonance* d_res,
    int res_idx_in_block, const double* d_all_params,
    const int* d_param_map, double* d_grad, const int* d_global_idx,
    int nEvents, int nPolar, int decayChain_size, double bf_d, double sign,
    int t_evt_offset,
    const thrust::complex<double>* d_slamps,
    const cuComplex* d_v, int site, int nSLComb)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int global_evt = evt + t_evt_offset;

    using AD = Var<double, Nfree, false>;
    using CV = ComplexVar<double, Nfree, false>;

    // 确定目标共振态的参数位置和自由参数映射
    const DeviceResonance& target_res = d_res[res_idx_in_block];
    const double* target_rp = d_all_params + target_res.param_offset;

    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nfree; ++j) { int pi = d_param_map[j]; if (pi>=0&&pi<8) ftg[pi] = j; }

    // 目标共振态参数 AD（只有目标共振态的参数有梯度）
    AD m0_ad(target_rp[0]);
    if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;

    AD gamma_ad; // width / couplings
    if (target_res.type == ResModelType::BWR || target_res.type == ResModelType::BW) {
        gamma_ad = AD(target_rp[1]);
        if (ftg[1] >= 0) gamma_ad.grad[ftg[1]] = 1.0;
    }

    AD other_g[4]; // Flatte couplings
    if (target_res.type == ResModelType::Flatte) {
        for (int k = 0; k < target_res.param_count - 1 && k < 4; ++k) {
            other_g[k] = AD(target_rp[1 + k]);
            if (ftg[1 + k] >= 0) other_g[k].grad[ftg[1 + k]] = 1.0;
        }
    }

    // 累加各 SL 贡献: c[pol] = Σ_sl ∂R_sl/∂θ · T_sl[evt,pol]
    // 然后 d log(I)/dθ = 2 Re(Σ_pol conj(w_pol) · c[pol])
    // 由于 ∂R_sl/∂θ 对 pol 独立, 等价于: 2 Re(Σ_sl ∂R_sl/∂θ · Σ_pol conj(w_pol) T_sl[evt,pol])
    int n_events_total = d_momenta->n_events;

    for (int sl_idx = 0; sl_idx < nSLComb; ++sl_idx) {
        // ----------------------------------------------------------------
        // 用 AD 计算 R_sl = ∏_ni node_factor(L_{sl,ni})
        // ----------------------------------------------------------------
        CV R_ad(1.0, 0.0);

        for (int ni = 0; ni < decayChain_size; ++ni) {
            const DecayNode& node = d_decayNodes[ni];
            const SL& sl = d_slComb[sl_idx * decayChain_size + ni];
            int L = sl.L;

            LorentzVector pM  = d_momenta->getMomentum(global_evt, node.mother_idx);
            LorentzVector pD1 = d_momenta->getMomentum(global_evt, node.daug1_idx);
            LorentzVector pD2 = d_momenta->getMomentum(global_evt, node.daug2_idx);

            double mm = pM.M();
            // qq 使用事件四动量的质量（与 computeAmpsKernel 保持一致）
            double qq = breakup_momentum(mm, pD1.M(), pD2.M());
            AD q_ad(qq);

            // q0 时回退到事件质量（仅在 node.mass < 0 且非目标共振态时使用）
            double md1 = pD1.M();
            double md2 = pD2.M();

            AD m0_q0, md1_q0, md2_q0;
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0)
                m0_q0 = m0_ad;
            else if (node.mass[0] > 0) m0_q0 = AD(node.mass[0]);
            else m0_q0 = AD(1.0);
            if (node.mass[1] <= 0 && node.daug1_idx == target_res.particle_idx)
                md1_q0 = m0_ad;
            else md1_q0 = AD(node.mass[1] > 0 ? node.mass[1] : md1);
            if (node.mass[2] <= 0 && node.daug2_idx == target_res.particle_idx)
                md2_q0 = m0_ad;
            else md2_q0 = AD(node.mass[2] > 0 ? node.mass[2] : md2);

            AD q0_ad = computeQ0AD(m0_q0, md1_q0, md2_q0);
            bool is_res = (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0);

            CV nf;
            if (is_res) {
                AD params_arr[2] = {m0_ad, gamma_ad};
                nf = computeNodeFactor<AD>(L, AD(mm), q_ad, q0_ad,
                                          params_arr, 2, target_res.type, nullptr, 0, bf_d);
            } else {
                AD bf_val = Bf<AD>(L, q_ad, q0_ad, bf_d);
                nf.real = bf_val; nf.imag = AD(0.0);
            }
            CV new_R;
            new_R.real = R_ad.real * nf.real - R_ad.imag * nf.imag;
            new_R.imag = R_ad.real * nf.imag + R_ad.imag * nf.real;
            R_ad = new_R;
        }

        // ----------------------------------------------------------------
        // 累加 sl 的贡献: 对 pol 求和 Σ_p conj(w_p) · v_sl · slamps[sl,e,p]
        // 然后乘以 ∂R_sl/∂θ 的 (实部, 虚部) 并提取实部
        // ----------------------------------------------------------------
        cuComplex v_sl = d_v[site + sl_idx];

        double s_re = 0.0, s_im = 0.0;
        for (int pol = 0; pol < nPolar; ++pol) {
            cuComplex w_val = d_w[evt * nPolar + pol];
            int amp_idx = sl_idx * n_events_total * nPolar + global_evt * nPolar + pol;
            auto sl_amp = d_slamps[amp_idx];
            double sl_re = sl_amp.real();
            double sl_im = sl_amp.imag();
            // T_sl = v_sl * slamps[sl,e,p]
            double T_re = (double)v_sl.x * sl_re - (double)v_sl.y * sl_im;
            double T_im = (double)v_sl.x * sl_im + (double)v_sl.y * sl_re;
            // conj(w) * T
            s_re += (double)w_val.x * T_re + (double)w_val.y * T_im;
            s_im += (double)w_val.x * T_im - (double)w_val.y * T_re;
        }

        // }

        for (int j = 0; j < Nfree; ++j) {
            double dRr = R_ad.real.grad[j];
            double dRi = R_ad.imag.grad[j];
            // 2 Re((s_re + i s_im) * (dRr + i dRi)) = 2 (s_re*dRr - s_im*dRi)
            double contrib = -2.0 * sign * (s_re * dRr - s_im * dRi);
            atomicAdd(&d_grad[d_global_idx[j]], contrib);
        }
    }
}
// ---------------------------------------------------------------------------
// T 计算 kernel: T[e,p] = Σ_sl v[site+sl] * slamps[sl, e, p]
// ---------------------------------------------------------------------------
__global__ void computeEffectiveCouplingKernel(
    cuComplex* d_T,
    const thrust::complex<double>* d_slamps,
    const cuComplex* d_v,
    int nSL, int nTotal,
    int sl_start, int sl_end)  // SL range: only sum over these SL channels
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTotal) return;

    double re = 0.0, im = 0.0;
    for (int sl = sl_start; sl < sl_end; ++sl) {
        auto sv = d_slamps[sl * nTotal + idx];
        auto vv = d_v[sl];
        re += (double)vv.x * sv.real() - (double)vv.y * sv.imag();
        im += (double)vv.x * sv.imag() + (double)vv.y * sv.real();
    }
    d_T[idx] = make_cuComplex((float)re, (float)im);
}

// ---------------------------------------------------------------------------
// AmpCalc::computeEffectiveCoupling
// ---------------------------------------------------------------------------
void AmpCalc::computeEffectiveCoupling(const cuComplex* d_v, int n_amplitudes)
{
    if (cas_list_.empty() || blocks_.empty()) return;
    // 仅在当前设备上计算 T（调用者负责 per-GPU 循环）
    int gpu = 0; cudaGetDevice(&gpu);

    constexpr int kBlockSize = 256;

    for (auto& block : blocks_) {
        auto& cas = cas_list_[block.cas_idx];
        int nSL = static_cast<int>(cas->getNSLCombs());
        int nEv  = static_cast<int>(cas->getNEventsVec()[gpu]);
        int nPol = static_cast<int>(cas->getNPolarizations());
        int nTotal = nEv * nPol;

        if (block.d_T[gpu] == nullptr) {
            cudaMalloc(&block.d_T[gpu], nTotal * sizeof(cuComplex));
        }

        int grid = (nTotal + kBlockSize - 1) / kBlockSize;
        computeEffectiveCouplingKernel<<<grid, kBlockSize>>>(
            block.d_T[gpu], cas->getSLAmps()[gpu],
            d_v + block.site, nSL, nTotal, 0, nSL);
        cudaDeviceSynchronize();
    }
}

// 小 kernel：双精度数组累加 y[i] += alpha * x[i]
__global__ void daxpy_kernel(double* y, const double* x, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

// 计算 w[e,p] = sign * S[e,p] / I[e]（用于梯度计算：g = A^H * w）
__global__ void computeGradWeightKernel(
    cuComplex* d_w, const cuComplex* d_S, double* d_I,
    int nEv, int nPol, double sign)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEv) return;
    double I_val = 0.0;
    for (int p = 0; p < nPol; ++p) {
        cuComplex s = d_S[e * nPol + p];
        I_val += (double)s.x * s.x + (double)s.y * s.y;
    }
    d_I[e] = I_val;
    double inv_I = I_val > 1e-30 ? sign / I_val : 0.0;
    for (int p = 0; p < nPol; ++p) {
        cuComplex s = d_S[e * nPol + p];
        d_w[e * nPol + p] = make_cuComplex((float)(inv_I * s.x), (float)(inv_I * s.y));  // sign * S/I
    }
}

// 对 bkg 事件乘上权重
__global__ void applyBkgWeightsKernel(
    cuComplex* d_w, const double* d_weights, int nEv, int nPol)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEv) return;
    double wgt = d_weights ? d_weights[e] : 1.0;
    for (int p = 0; p < nPol; ++p) {
        int idx = e * nPol + p;
        d_w[idx].x *= (float)wgt;
        d_w[idx].y *= (float)wgt;
    }
}

// ---------------------------------------------------------------------------
// AmpCalc::computeResonanceGradient
// ---------------------------------------------------------------------------
void AmpCalc::computeResonanceGradient(
    const std::vector<cuComplex*>& d_w,
    const std::vector<int>& n_events,
    double* d_grad_res,
    double sign,
    const std::vector<int>& t_offset,
    const std::vector<cuComplex*>& d_v_per_gpu)
{
    int n_gpu = static_cast<int>(d_w.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;
    bool has_offset = (t_offset.size() == d_w.size());
    bool has_dv = (d_v_per_gpu.size() == d_w.size());
    int primary_dev = 0; cudaGetDevice(&primary_dev);  // d_grad_res lives here

    constexpr int kBlockSize = 256;

    // 为每 GPU 分配临时梯度 buffer
    std::vector<double*> d_grad_per_gpu(n_gpu, nullptr);
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaMalloc(&d_grad_per_gpu[gpu], n_free * sizeof(double));
        cudaMemset(d_grad_per_gpu[gpu], 0, n_free * sizeof(double));
    }

    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        auto& block = blocks_[bi];
        auto& cas = cas_list_[block.cas_idx];

        for (int r = 0; r < block.resonance_count; ++r) {
            std::vector<int> local_map, global_idx;
            for (int s = 0; s < n_free; ++s) {
                if (slots_[s].block_idx == (int)bi && slots_[s].res_idx == r) {
                    local_map.push_back(slots_[s].param_idx);
                    global_idx.push_back(s);
                }
            }
            int Nlocal = static_cast<int>(local_map.size());
            if (Nlocal == 0) continue;

            for (int gpu = 0; gpu < n_gpu; ++gpu) {
                cudaSetDevice(gpu);

                int *d_param_map, *d_global_idx;
                cudaMalloc(&d_param_map, Nlocal * sizeof(int));
                cudaMalloc(&d_global_idx, Nlocal * sizeof(int));
                cudaMemcpy(d_param_map, local_map.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_global_idx, global_idx.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);

                int nEv = n_events[gpu];
                int nPol = static_cast<int>(cas->getNPolarizations());
                int grid = (nEv + kBlockSize - 1) / kBlockSize;
                int evt_off = has_offset ? t_offset[gpu] : 0;
                const cuComplex* d_v_gpu = has_dv ? d_v_per_gpu[gpu] : nullptr;
                switch (Nlocal) {
                case 1:
                    resonanceGradientKernel<1><<<grid, kBlockSize>>>(
                        d_w[gpu], block.d_T[gpu],
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], block.d_resonances[gpu],
                        r, block.d_all_params[gpu],
                        d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                        nEv, nPol, cas->getDecayChainSize(), 3.0, sign,
                        evt_off, cas->getSLAmps()[gpu], d_v_gpu, block.site,
                        static_cast<int>(cas->getNSLCombs()));
                    break;
                case 2:
                    resonanceGradientKernel<2><<<grid, kBlockSize>>>(
                        d_w[gpu], block.d_T[gpu],
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], block.d_resonances[gpu],
                        r, block.d_all_params[gpu],
                        d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                        nEv, nPol, cas->getDecayChainSize(), 3.0, sign,
                        evt_off, cas->getSLAmps()[gpu], d_v_gpu, block.site,
                        static_cast<int>(cas->getNSLCombs()));
                    break;
                case 3:
                    resonanceGradientKernel<3><<<grid, kBlockSize>>>(
                        d_w[gpu], block.d_T[gpu],
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], block.d_resonances[gpu],
                        r, block.d_all_params[gpu],
                        d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                        nEv, nPol, cas->getDecayChainSize(), 3.0, sign,
                        evt_off, cas->getSLAmps()[gpu], d_v_gpu, block.site,
                        static_cast<int>(cas->getNSLCombs()));
                    break;
                default: break;
                }
                cudaDeviceSynchronize();
                cudaFree(d_param_map);
                cudaFree(d_global_idx);
            }
        }
    }

    // ============================================================
    // 处理 conjugate 块：用 conjugate 块的数据计算梯度，
    // 但累加到 owner 的梯度槽位中。如果不处理，trans 约束下
    // 共享的共振态参数梯度会缺少 conjugate 链的贡献，
    // 导致放开质量/宽度参数后拟合结果不对称。
    // ============================================================
    for (const auto& [cj_key, owner_key] : conjugate_broadcast_) {
        int cj_bi = cj_key.first, cj_ri = cj_key.second;
        int ow_bi = owner_key.first, ow_ri = owner_key.second;

        // 查找 owner 的 slot 索引（conjugate 自己没有 ParamSlot）
        std::vector<int> local_map, global_idx;
        for (int s = 0; s < n_free; ++s) {
            if (slots_[s].block_idx == ow_bi && slots_[s].res_idx == ow_ri) {
                local_map.push_back(slots_[s].param_idx);
                global_idx.push_back(s);
            }
        }
        int Nlocal = static_cast<int>(local_map.size());
        if (Nlocal == 0) continue;

        auto& cj_block = blocks_[cj_bi];
        auto& cj_cas = cas_list_[cj_block.cas_idx];

        for (int gpu = 0; gpu < n_gpu; ++gpu) {
            cudaSetDevice(gpu);

            int *d_param_map, *d_global_idx;
            cudaMalloc(&d_param_map, Nlocal * sizeof(int));
            cudaMalloc(&d_global_idx, Nlocal * sizeof(int));
            cudaMemcpy(d_param_map, local_map.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_global_idx, global_idx.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);

            int nEv = n_events[gpu];
            int nPol = static_cast<int>(cj_cas->getNPolarizations());
            int grid = (nEv + kBlockSize - 1) / kBlockSize;
            int evt_off = has_offset ? t_offset[gpu] : 0;
            const cuComplex* d_v_gpu = has_dv ? d_v_per_gpu[gpu] : nullptr;
            switch (Nlocal) {
            case 1:
                resonanceGradientKernel<1><<<grid, kBlockSize>>>(
                    d_w[gpu], cj_block.d_T[gpu],
                    cj_cas->getMomenta()[gpu], cj_cas->getDecayNodes()[gpu],
                    cj_cas->getDeviceSLCombs()[gpu], cj_block.d_resonances[gpu],
                    cj_ri, cj_block.d_all_params[gpu],
                    d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                    nEv, nPol, cj_cas->getDecayChainSize(), 3.0, sign,
                    evt_off, cj_cas->getSLAmps()[gpu], d_v_gpu, cj_block.site,
                    static_cast<int>(cj_cas->getNSLCombs()));
                break;
            case 2:
                resonanceGradientKernel<2><<<grid, kBlockSize>>>(
                    d_w[gpu], cj_block.d_T[gpu],
                    cj_cas->getMomenta()[gpu], cj_cas->getDecayNodes()[gpu],
                    cj_cas->getDeviceSLCombs()[gpu], cj_block.d_resonances[gpu],
                    cj_ri, cj_block.d_all_params[gpu],
                    d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                    nEv, nPol, cj_cas->getDecayChainSize(), 3.0, sign,
                    evt_off, cj_cas->getSLAmps()[gpu], d_v_gpu, cj_block.site,
                    static_cast<int>(cj_cas->getNSLCombs()));
                break;
            case 3:
                resonanceGradientKernel<3><<<grid, kBlockSize>>>(
                    d_w[gpu], cj_block.d_T[gpu],
                    cj_cas->getMomenta()[gpu], cj_cas->getDecayNodes()[gpu],
                    cj_cas->getDeviceSLCombs()[gpu], cj_block.d_resonances[gpu],
                    cj_ri, cj_block.d_all_params[gpu],
                    d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                    nEv, nPol, cj_cas->getDecayChainSize(), 3.0, sign,
                    evt_off, cj_cas->getSLAmps()[gpu], d_v_gpu, cj_block.site,
                    static_cast<int>(cj_cas->getNSLCombs()));
                break;
            default: break;
            }
            cudaDeviceSynchronize();
            cudaFree(d_param_map);
            cudaFree(d_global_idx);
        }
    }

    // 将各 GPU 结果通过 daxpy 累加到 d_grad_res（在 primary_dev 上）
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == primary_dev) {
            cudaSetDevice(primary_dev);
            int grid = (n_free + 255) / 256;
            daxpy_kernel<<<grid, 256>>>(d_grad_res, d_grad_per_gpu[primary_dev], 1.0, n_free);
            cudaDeviceSynchronize();
        } else {
            std::vector<double> h_temp(n_free);
            cudaSetDevice(gpu);
            cudaMemcpy(h_temp.data(), d_grad_per_gpu[gpu],
                       n_free * sizeof(double), cudaMemcpyDeviceToHost);
            cudaSetDevice(primary_dev);
            double* d_temp;
            cudaMalloc(&d_temp, n_free * sizeof(double));
            cudaMemcpy(d_temp, h_temp.data(), n_free * sizeof(double), cudaMemcpyHostToDevice);
            int grid = (n_free + 255) / 256;
            daxpy_kernel<<<grid, 256>>>(d_grad_res, d_temp, 1.0, n_free);
            cudaDeviceSynchronize();
            cudaFree(d_temp);
        }
    }

    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaFree(d_grad_per_gpu[gpu]);
    }
    cudaSetDevice(primary_dev);
}


// ============================================================
// AmpCalc::computeUnifiedHessian
// Computes θθ Hessian block using hessianStage1Kernel + cross-block kernels.
// ============================================================
void AmpCalc::computeUnifiedHessian(
    const std::vector<int>& n_events,
    double* d_hess, int hess_ld,
    const std::vector<int>& t_offset,
    double default_weight,
    const std::vector<cuComplex*>& d_v_per_gpu,
    const std::vector<cuComplex*>& d_amp_per_gpu,
    int n_amp_total,
    const std::vector<double*>& d_event_weights,
    double* d_phsp_I,
    double* d_phsp_grad,
    double* d_phsp_hessA,
    double* d_mixed_out,
    double* d_phsp_mixed_sum,
    double* d_phsp_mixed_t3)
{
    int n_gpu = static_cast<int>(n_events.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;
    bool has_offset = (t_offset.size() == (size_t)n_gpu);
    constexpr int kBlockSize = 256;

    int primary_dev = 0; cudaGetDevice(&primary_dev);  // capture caller's device
    int hess_sz = hess_ld * hess_ld;

    // Temp buffer info per block (stored for stage 2 cross-block)
    struct BlockTemp {
        double* d_g = nullptr;
        double* d_dS_re = nullptr;
        double* d_dS_im = nullptr;
        double* d_dF_re = nullptr;
        double* d_dF_im = nullptr;
        int* d_gidx = nullptr;
        int NT;
        int nEv;
    };
    std::vector<std::vector<BlockTemp>> temps_per_gpu(n_gpu);

    // ===== Stage 1-4: per-GPU, with local output buffers for remote GPUs =====
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        int nEv = n_events[gpu];
        if (nEv <= 0) continue;
        int evt_off = has_offset ? t_offset[gpu] : 0;
        const double* d_w = (gpu < (int)d_event_weights.size()) ? d_event_weights[gpu] : nullptr;
        const cuComplex* d_v = d_v_per_gpu[gpu];
        const cuComplex* d_amp = d_amp_per_gpu[gpu];

        // --- For remote GPUs, allocate local copies of global output buffers ---
        bool is_remote = (gpu != primary_dev);
        double* d_hess_g = d_hess;
        double* d_phA_g = d_phsp_hessA;
        double* d_pg_g = d_phsp_grad;
        double* d_pI_g = d_phsp_I;
        double* d_mix_g = d_mixed_out;
        double* d_msum_g = d_phsp_mixed_sum;
        double* d_t3_g = d_phsp_mixed_t3;
        if (is_remote) {
            cudaMalloc(&d_hess_g, hess_sz * sizeof(double));
            cudaMemset(d_hess_g, 0, hess_sz * sizeof(double));
            if (d_phsp_hessA) {
                cudaMalloc(&d_phA_g, hess_sz * sizeof(double));
                cudaMemset(d_phA_g, 0, hess_sz * sizeof(double));
                cudaMalloc(&d_pg_g, n_free * sizeof(double));
                cudaMemset(d_pg_g, 0, n_free * sizeof(double));
                cudaMalloc(&d_pI_g, sizeof(double));
                cudaMemset(d_pI_g, 0, sizeof(double));
            }
            if (d_mixed_out) {
                int mix_sz = 2 * n_amp_total * n_free;
                cudaMalloc(&d_mix_g, mix_sz * sizeof(double));
                cudaMemset(d_mix_g, 0, mix_sz * sizeof(double));
            }
            if (d_phsp_mixed_sum) {
                int mix_sz = 2 * n_amp_total * n_free;
                cudaMalloc(&d_msum_g, mix_sz * sizeof(double));
                cudaMemset(d_msum_g, 0, mix_sz * sizeof(double));
            }
            if (d_phsp_mixed_t3) {
                cudaMalloc(&d_t3_g, 2 * n_amp_total * sizeof(double));
                cudaMemset(d_t3_g, 0, 2 * n_amp_total * sizeof(double));
            }
        }

        // Pre-pass: compute full S[p] = Σ_a v[a]·amp[a,e,p] and I[e] from raw amplitudes
        auto& cas0 = cas_list_[blocks_[0].cas_idx];
        int nPol = static_cast<int>(cas0->getNPolarizations());
        double *d_S_re, *d_S_im, *d_I_full;
        cudaMalloc(&d_S_re, nEv*nPol*sizeof(double));
        cudaMalloc(&d_S_im, nEv*nPol*sizeof(double));
        cudaMalloc(&d_I_full, nEv*sizeof(double));
        {
            int grid = (nEv + kBlockSize - 1) / kBlockSize;
            double *t3_re = nullptr, *t3_im = nullptr;
            if (d_t3_g) {
                t3_re = d_t3_g;
                t3_im = d_t3_g + n_amp_total;
            }
            computeSfromAmpsKernel<<<grid, kBlockSize>>>(
                d_S_re, d_S_im, d_I_full,
                d_amp + evt_off * nPol * n_amp_total,
                d_v, nEv, nPol, n_amp_total,
                t3_re, t3_im);
            cudaDeviceSynchronize();
        }

        temps_per_gpu[gpu].resize(blocks_.size());

        bool first_free_block = true;  // track first block with free params for phsp_I

        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            auto& blk = blocks_[bi];
            auto& cas = cas_list_[blk.cas_idx];
            int Nres = blk.resonance_count;
            if (Nres < 1 || Nres > 4) continue;

            int nSL = static_cast<int>(cas->getNSLCombs());
            int nPol = static_cast<int>(cas->getNPolarizations());
            int dsz = cas->getDecayChainSize();

            int Npr = 0;
            for (int s = 0; s < n_free; ++s) {
                if (slots_[s].block_idx == (int)bi && slots_[s].res_idx == 0) ++Npr;
            }

            // 检查是否为 conjugate 块：如果 Npr==0，查 conjugate_broadcast_
            //（该块的所有共振态通过 trans 约束与 owner 块共享参数）
            bool is_conjugate = false;
            int ow_bi = -1;
            if (Npr == 0) {
                for (const auto& [cj_key, owner_key] : conjugate_broadcast_) {
                    if (cj_key.first == (int)bi) {
                        is_conjugate = true;
                        ow_bi = owner_key.first;
                        break;
                    }
                }
                if (is_conjugate && ow_bi >= 0) {
                    for (int s = 0; s < n_free; ++s) {
                        if (slots_[s].block_idx == ow_bi && slots_[s].res_idx == 0) ++Npr;
                    }
                }
            }
            if (Npr < 1 || Npr > 3) continue;

            int NT = Npr * Nres;

            // 构建全局索引映射（conjugate 块映射到 owner 的 slot 索引）
            std::vector<int> global_idx(NT, -1);
            for (int r = 0; r < Nres; ++r) {
                int target_bi = is_conjugate ? ow_bi : (int)bi;
                int target_ri = r;
                if (is_conjugate) {
                    for (const auto& [cj_key, owner_key] : conjugate_broadcast_) {
                        if (cj_key.first == (int)bi && cj_key.second == r) {
                            target_ri = owner_key.second;
                            break;
                        }
                    }
                }
                int count = 0;
                for (int s = 0; s < n_free; ++s) {
                    if (slots_[s].block_idx == target_bi && slots_[s].res_idx == target_ri) {
                        if (count < Npr) { global_idx[r * Npr + count] = s; ++count; }
                    }
                }
            }

            // Allocate temp buffers for this block
            auto& bt = temps_per_gpu[gpu][bi];
            bt.NT = NT;
            bt.nEv = nEv;
            cudaMalloc(&bt.d_g, nEv * NT * sizeof(double));
            cudaMalloc(&bt.d_dS_re, nEv * NT * nPol * sizeof(double));
            cudaMalloc(&bt.d_dS_im, nEv * NT * nPol * sizeof(double));
            cudaMalloc(&bt.d_dF_re, nEv * nSL * Npr * sizeof(double));
            cudaMalloc(&bt.d_dF_im, nEv * nSL * Npr * sizeof(double));
            cudaMalloc(&bt.d_gidx, NT * sizeof(int));
            cudaMemcpy(bt.d_gidx, global_idx.data(), NT * sizeof(int), cudaMemcpyHostToDevice);

            int grid = (nEv + kBlockSize - 1) / kBlockSize;

            const cuComplex* d_v_blk = d_v + blk.site;
            // Conjugate 块：phsp 积分已由 owner 计算，跳过 d_pI_g。
            // phsp 梯度/Hessian（d_pg_g, d_phA_g）通过 owner slot 累加。
            double* d_pI_ptr = (first_free_block && !is_conjugate) ? d_pI_g : nullptr;
            if (Npr == 2 && Nres == 2) {
                hessianStage1Kernel<2,2><<<grid, kBlockSize>>>(
                    cas->getSLAmps()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off);
            } else if (Npr == 1 && Nres == 1) {
                hessianStage1Kernel<1,1><<<grid, kBlockSize>>>(
                    cas->getSLAmps()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off);
            } else if (Npr == 2 && Nres == 1) {
                hessianStage1Kernel<2,1><<<grid, kBlockSize>>>(
                    cas->getSLAmps()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off);
            } else if (Npr == 1 && Nres == 2) {
                hessianStage1Kernel<1,2><<<grid, kBlockSize>>>(
                    cas->getSLAmps()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off);
            } else {
                printf("computeUnifiedHessian: unsupported Npr=%d Nres=%d\n", Npr, Nres);
            }
            if (!is_conjugate) first_free_block = false;
            cudaDeviceSynchronize();
        }

        // ===== Stage 2: cross-block Hessian =====
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            auto& btA = temps_per_gpu[gpu][bi];
            if (!btA.d_g) continue;
            for (size_t bj = bi + 1; bj < blocks_.size(); ++bj) {
                auto& btB = temps_per_gpu[gpu][bj];
                if (!btB.d_g) continue;

                int grid = (nEv + kBlockSize - 1) / kBlockSize;
                hessianCrossBlockKernel<<<grid, kBlockSize>>>(
                    btA.d_g, btA.d_dS_re, btA.d_dS_im,
                    btB.d_g, btB.d_dS_re, btB.d_dS_im,
                    d_I_full, btA.d_gidx, btB.d_gidx,
                    btA.NT, btB.NT, nEv, nPol,
                    d_hess_g, hess_ld, default_weight, d_w,
                    (default_weight == 0.0) ? d_phA_g : nullptr,
                    (d_phA_g ? nFreeResParams() : 1));
                cudaDeviceSynchronize();
            }
        }

        // ===== Stage 3: Per-block mixed Hessian (vθ same-block) =====
        // ===== Stage 4: Cross-block mixed Hessian (vθ cross terms) =====
        if (d_mixed_out) {
            for (size_t bi = 0; bi < blocks_.size(); ++bi) {
                auto& bt = temps_per_gpu[gpu][bi];
                if (!bt.d_g || !bt.d_dF_re) continue;
                auto& blk = blocks_[bi];
                // 用当前 block 自己的 cas，而非 cas0（多链时不同链 SL 数不同）
                int nSL = static_cast<int>(cas_list_[blk.cas_idx]->getNSLCombs());
                int Npr = 0;
                for (int s = 0; s < n_free; ++s)
                    if (slots_[s].block_idx == (int)bi && slots_[s].res_idx == 0) ++Npr;
                // Conjugate 块：使用 owner 的 Npr（与 Stage 1 逻辑一致）
                if (Npr == 0) {
                    for (const auto& [cj_key, owner_key] : conjugate_broadcast_) {
                        if (cj_key.first == (int)bi) {
                            int ow_bi = owner_key.first;
                            for (int s = 0; s < n_free; ++s)
                                if (slots_[s].block_idx == ow_bi && slots_[s].res_idx == 0) ++Npr;
                            break;
                        }
                    }
                }
                if (Npr < 1) continue;
                int nTotal_slamp = static_cast<int>(cas_list_[blk.cas_idx]->getNEventsVec()[gpu]) * nPol;
                int grid = (nEv + kBlockSize - 1) / kBlockSize;
                hessianMixedBlockKernel<<<grid, kBlockSize>>>(
                    d_S_re, d_S_im, d_I_full,
                    d_amp + evt_off * nPol * n_amp_total,
                    cas_list_[blk.cas_idx]->getSLAmps()[gpu],
                    bt.d_g, bt.d_dS_re, bt.d_dS_im,
                    bt.d_dF_re, bt.d_dF_im, bt.d_gidx,
                    d_mix_g, nFreeResParams(),
                    nEv, nSL, Npr, nPol, n_amp_total, blk.site,
                    nTotal_slamp, default_weight, d_w, d_msum_g, evt_off);
                cudaDeviceSynchronize();
            }

            // Cross-block mixed
            for (size_t bi = 0; bi < blocks_.size(); ++bi) {
                auto& blkA = blocks_[bi];
                // 用 block bi 自己的 cas（多链时不同链 SL 数不同）
                int nSL_A = static_cast<int>(cas_list_[blkA.cas_idx]->getNSLCombs());
                for (size_t bj = 0; bj < blocks_.size(); ++bj) {
                    if (bi == bj) continue;
                    auto& btB = temps_per_gpu[gpu][bj];
                    if (!btB.d_g) continue;
                    // 跨链 vθ 项是必需的：去掉 cas_idx 过滤（kernel 只用 d_amp，与链无关）
                    int grid = (nEv + kBlockSize - 1) / kBlockSize;
                    hessianCrossMixedKernel<<<grid, kBlockSize>>>(
                        d_S_re, d_S_im, d_I_full,
                        d_amp + evt_off * nPol * n_amp_total,
                        btB.d_g, btB.d_dS_re, btB.d_dS_im, btB.d_gidx, btB.NT,
                        nSL_A, blkA.site,
                        nEv, nPol, n_amp_total,
                        d_mix_g, nFreeResParams(),
                        default_weight, d_w, d_msum_g, evt_off);
                    cudaDeviceSynchronize();
                }
            }
        }

        // Free temp buffers
        for (auto& bt : temps_per_gpu[gpu]) {
            if (bt.d_g) cudaFree(bt.d_g);
            if (bt.d_dS_re) cudaFree(bt.d_dS_re);
            if (bt.d_dS_im) cudaFree(bt.d_dS_im);
            if (bt.d_dF_re) cudaFree(bt.d_dF_re);
            if (bt.d_dF_im) cudaFree(bt.d_dF_im);
            if (bt.d_gidx) cudaFree(bt.d_gidx);
        }
        cudaFree(d_S_re); cudaFree(d_S_im); cudaFree(d_I_full);

        // --- Accumulate remote GPU results to global buffers on primary_dev ---
        if (is_remote) {
            double one = 1.0;
            // d_hess: copy peer → daxpy on primary_dev
            cudaSetDevice(primary_dev);
            double* d_tmp; cudaMalloc(&d_tmp, hess_sz * sizeof(double));
            cudaMemcpyPeer(d_tmp, primary_dev, d_hess_g, gpu, hess_sz * sizeof(double));
            cublasHandle_t h; cublasCreate(&h);
            cublasDaxpy(h, hess_sz, &one, d_tmp, 1, d_hess, 1);
            cublasDestroy(h); cudaFree(d_tmp);
            // d_phsp_hessA
            if (d_phsp_hessA) {
                cudaSetDevice(primary_dev); cudaMalloc(&d_tmp, hess_sz * sizeof(double));
                cudaMemcpyPeer(d_tmp, primary_dev, d_phA_g, gpu, hess_sz * sizeof(double));
                cublasCreate(&h); cublasDaxpy(h, hess_sz, &one, d_tmp, 1, d_phsp_hessA, 1);
                cublasDestroy(h); cudaFree(d_tmp);
                // d_phsp_I: copy single value
                double h_pI;
                cudaSetDevice(gpu); cudaMemcpy(&h_pI, d_pI_g, sizeof(double), cudaMemcpyDeviceToHost);
                cudaSetDevice(primary_dev); cudaMalloc(&d_tmp, sizeof(double));
                cudaMemcpy(d_tmp, &h_pI, sizeof(double), cudaMemcpyHostToDevice);
                cublasCreate(&h); cublasDaxpy(h, 1, &one, d_tmp, 1, d_phsp_I, 1);
                cublasDestroy(h); cudaFree(d_tmp);
                // d_phsp_grad
                cudaSetDevice(primary_dev); cudaMalloc(&d_tmp, n_free * sizeof(double));
                cudaMemcpyPeer(d_tmp, primary_dev, d_pg_g, gpu, n_free * sizeof(double));
                cublasCreate(&h); cublasDaxpy(h, n_free, &one, d_tmp, 1, d_phsp_grad, 1);
                cublasDestroy(h); cudaFree(d_tmp);
            }
            // d_mixed_out, d_phsp_mixed_sum
            int mix_sz = 2 * n_amp_total * n_free;
            if (d_mixed_out) {
                cudaSetDevice(primary_dev); cudaMalloc(&d_tmp, mix_sz * sizeof(double));
                cudaMemcpyPeer(d_tmp, primary_dev, d_mix_g, gpu, mix_sz * sizeof(double));
                cublasCreate(&h); cublasDaxpy(h, mix_sz, &one, d_tmp, 1, d_mixed_out, 1);
                cublasDestroy(h); cudaFree(d_tmp);
            }
            if (d_phsp_mixed_sum) {
                cudaSetDevice(primary_dev); cudaMalloc(&d_tmp, mix_sz * sizeof(double));
                cudaMemcpyPeer(d_tmp, primary_dev, d_msum_g, gpu, mix_sz * sizeof(double));
                cublasCreate(&h); cublasDaxpy(h, mix_sz, &one, d_tmp, 1, d_phsp_mixed_sum, 1);
                cublasDestroy(h); cudaFree(d_tmp);
            }
            if (d_phsp_mixed_t3) {
                int t3_sz = 2 * n_amp_total;
                cudaSetDevice(primary_dev); cudaMalloc(&d_tmp, t3_sz * sizeof(double));
                cudaMemcpyPeer(d_tmp, primary_dev, d_t3_g, gpu, t3_sz * sizeof(double));
                cublasCreate(&h); cublasDaxpy(h, t3_sz, &one, d_tmp, 1, d_phsp_mixed_t3, 1);
                cublasDestroy(h); cudaFree(d_tmp);
            }
            cudaSetDevice(gpu);
        }

        // Free remote GPU local buffers
        if (is_remote) {
            cudaFree(d_hess_g);
            if (d_phA_g) cudaFree(d_phA_g);
            if (d_pg_g) cudaFree(d_pg_g);
            if (d_pI_g) cudaFree(d_pI_g);
            if (d_mix_g) cudaFree(d_mix_g);
            if (d_msum_g) cudaFree(d_msum_g);
            if (d_t3_g) cudaFree(d_t3_g);
        }
    }
    cudaSetDevice(primary_dev);
}


void AmpCalc::testBWRHessian(double m, double m0, double g0, int L, double q, double q0, double d, double* out)
{
    using AD = Var<double, 2, true>;
    AD m_ad(m);
    AD m0_ad(m0);  m0_ad.grad[0] = 1.0;
    AD g0_ad(g0);  g0_ad.grad[1] = 1.0;
    AD q_ad(q);
    AD q0_ad(q0);
    auto R = BWR<AD>(m_ad, m0_ad, g0_ad, L, q_ad, q0_ad, d);
    out[0] = R.real.val;  out[1] = R.imag.val;
    out[2] = R.real.grad[0]; out[3] = R.real.grad[1];
    out[4] = R.imag.grad[0]; out[5] = R.imag.grad[1];
    out[6] = R.real.hess[0][0]; out[7] = R.real.hess[0][1]; out[8] = R.real.hess[1][1];
    out[9] = R.imag.hess[0][0]; out[10]= R.imag.hess[0][1]; out[11]= R.imag.hess[1][1];
}

// ============================================================
// Multiplicative coupling: v[a] = ratio[a] × chain[c_a] × Π step[k]
// d_params: [Re_0..Re_{n-1}, Im_0..Im_{n-1}] (all Re, all Im)
// ============================================================

__global__ void multiplicativeCouplingKernel(
    cuComplex* d_v, const double* d_params,
    const int* d_amp_chain,
    const int* d_step_offsets,
    const int* d_step_data,
    const double* d_amp_chain_ratio,
    int n_amps, int n_step_free, int n_free)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_amps) return;

    double ratio = d_amp_chain_ratio[a];
    int nch = n_free - n_step_free;
    int c_idx = d_amp_chain[a];                 // chains first in d_params
    double re = d_params[c_idx] * ratio;
    double im = d_params[n_free + c_idx] * ratio;

    for (int k = d_step_offsets[a]; k < d_step_offsets[a + 1]; ++k) {
        int s = nch + d_step_data[k];            // steps after chains
        double s_re = d_params[s];
        double s_im = d_params[n_free + s];
        double new_re = re * s_re - im * s_im;
        double new_im = re * s_im + im * s_re;
        re = new_re; im = new_im;
    }

    d_v[a].x = (float)re;
    d_v[a].y = (float)im;
}

// Gradient transform: ∂L/∂p using Wirtinger calculus
// ∂L/∂p_j* = (1/p_j) Σ_{a∈A_j} grad_v[a]* · v[a]
// d_params, d_grad_p: [Re_0..Re_{n-1}, Im_0..Im_{n-1}] (all Re, all Im)
__global__ void multiplicativeGradientKernel(
    double* d_grad_p,           // output [2·n_free] — [Re, Im] format
    const cuComplex* d_grad_v,  // ∂L/∂v [n_amps]
    const cuComplex* d_v,       // current v [n_amps]
    const double* d_params,     // [2·n_free] — [Re, Im] format
    const int* d_amp_chain,     // [n_amps]
    const int* d_step_offsets,  // [n_amps+1]
    const int* d_step_data,     // flat step param indices
    int n_amps, int n_step_free, int n_free)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_free) return;

    int nch = n_free - n_step_free;
    double sum_re = 0.0, sum_im = 0.0;

    if (j < nch) {
        // Chain param: d_amp_chain[a] == j (chains are first in d_params)
        for (int a = 0; a < n_amps; ++a) {
            if (d_amp_chain[a] == j) {
                double gv_re = (double)d_grad_v[a].x;
                double gv_im = (double)d_grad_v[a].y;
                double v_re  = (double)d_v[a].x;
                double v_im  = (double)d_v[a].y;
                sum_re += gv_re * v_re + gv_im * v_im;
                sum_im += gv_re * v_im - gv_im * v_re;
            }
        }
    } else {
        // Step param: index = j - nch (steps after chains)
        int sj = j - nch;
        for (int a = 0; a < n_amps; ++a) {
            bool uses_j = false;
            for (int k = d_step_offsets[a]; k < d_step_offsets[a + 1]; ++k) {
                if (d_step_data[k] == sj) { uses_j = true; break; }
            }
            if (uses_j) {
                double gv_re = (double)d_grad_v[a].x;
                double gv_im = (double)d_grad_v[a].y;
                double v_re  = (double)d_v[a].x;
                double v_im  = (double)d_v[a].y;
                sum_re += gv_re * v_re + gv_im * v_im;
                sum_im += gv_re * v_im - gv_im * v_re;
            }
        }
    }

    double p_re = d_params[j];
    double p_im = d_params[n_free + j];
    double p_sq = p_re * p_re + p_im * p_im;
    if (p_sq < 1e-30) p_sq = 1e-30;
    double dL_re = (sum_re * p_re + sum_im * p_im) / p_sq;
    double dL_im = (sum_im * p_re - sum_re * p_im) / p_sq;

    d_grad_p[j]            = 2.0 * dL_re;
    d_grad_p[n_free + j]   = -2.0 * dL_im;
}
