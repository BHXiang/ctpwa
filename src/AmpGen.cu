#include <AmpGen.cuh>
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

            // 释放交换链的 DeviceMomenta
            cudaFree(d_mom_perm[i]->momenta);
            cudaFree(d_mom_perm[i]);
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

        double q0 = std::sqrt((mass_mother * mass_mother -
            std::pow(mass_daug1 + mass_daug2, 2)) *
            (mass_mother * mass_mother -
                std::pow(mass_daug1 - mass_daug2, 2))) /
            2 / mass_mother;

        if (nodeIdx == 0)
        {
            // 第一个节点特殊处理
            // resAmp *= BlattWeisskopf(sl.L, qq, q0);
            resAmp *= Bf<double>(sl.L, qq, q0, bf_d);

            // printf("Event %d, sl %d, First Node: L=%d, qq=%f, q0=%f, BW
            // Factor=(%f, %f i)\n", event_idx, sl_idx, sl.L, qq, q0,
            // resAmp.real(), resAmp.imag());
            continue;
        }

        // 如果是共振态节点，计算相应的振幅因子
        if (is_resonance_node)
        {
            const double* p = d_all_params + current_res.param_offset;
            double res_mass = p[0];

            if (current_res.type == ResModelType::BWR)
            {
                double res_width = p[1];
                resAmp *= BWR<double>(mm, res_mass, res_width, sl.L, qq, q0, bf_d);
                resAmp *= Bf<double>(sl.L, qq, q0, bf_d);
            }
            else if (current_res.type == ResModelType::BW)
            {
                double res_width = p[1];
                resAmp *= BW<double>(mm, res_mass, res_width);
            }
            else if (current_res.type == ResModelType::ONE)
            {
                resAmp *= Bf<double>(sl.L, qq, q0, bf_d);
            }
            else if (current_res.type == ResModelType::Flatte)
            {
                // p[0] = mass, p[1..] = couplings; channels in d_all_channels
                const double* ch = d_all_channels + current_res.channel_offset;
                resAmp *= Flatte(mm, res_mass, current_res.n_channels, &p[1], ch);
                resAmp *= Bf<double>(sl.L, qq, q0, bf_d);
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
    int n_local_free)                   // 本 block 涉及的自由参数数
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local_free) return;
    int r = d_res_idx[i];
    int p = d_param_idx[i];
    if (r < resonance_count) {
        int offset = d_resonances[r].param_offset;
        d_all_params[offset + p] = d_free_params[i];
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
                       const std::vector<std::vector<std::vector<double>>>& free_ranges)
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

            int* d_res_idx, * d_param_idx;
            cudaMalloc(&d_res_idx, n_local * sizeof(int));
            cudaMalloc(&d_param_idx, n_local * sizeof(int));
            cudaMemcpy(d_res_idx, local_res_idx.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_param_idx, local_param_idx.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);

            int grid = (n_local + blockSize - 1) / blockSize;
            updateResonanceParamsKernel<<<grid, blockSize>>>(
                block.d_all_params[gpu], block.d_resonances[gpu],
                block.resonance_count,
                d_params_per_gpu[gpu], d_res_idx, d_param_idx, n_local);

            cudaDeviceSynchronize();
            cudaFree(d_res_idx);
            cudaFree(d_param_idx);
        }

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

// 梯度 kernel 模板实现 (q0 AD + per-SL Bf修正 + event offset)
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
    const cuComplex* d_v, int site)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int global_evt = evt + t_evt_offset;
    int nTotalEvt = d_momenta->n_events;

    const DeviceResonance& res = d_res[res_idx_in_block];
    const double* rp = d_all_params + res.param_offset;

    // Step 1: 找共振态节点
    double mm = 0.0, qq = 0.0, node_md1 = 1.0, node_md2 = 1.0;
    int res_L = 0; bool found = false;
    for (int ni = 0; ni < decayChain_size; ++ni) {
        const DecayNode& node = d_decayNodes[ni];
        if (node.mother_idx != res.particle_idx) continue;
        LorentzVector pM = d_momenta->getMomentum(global_evt, node.mother_idx);
        LorentzVector pD1 = d_momenta->getMomentum(global_evt, node.daug1_idx);
        LorentzVector pD2 = d_momenta->getMomentum(global_evt, node.daug2_idx);
        mm = pM.M();
        double md1 = node.mass[1], md2 = node.mass[2];
        if (md1 <= 0) md1 = pD1.M();
        if (md2 <= 0) md2 = pD2.M();
        node_md1 = md1; node_md2 = md2;
        qq = breakup_momentum(mm, md1, md2);
        res_L = d_slComb[ni].L;
        found = true; break;
    }
    if (!found) return;

    // Step 2: AutoDiff
    using AD = Var<double, Nfree, false>;
    AD m_ad(mm);
    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nfree; ++j) { int pi = d_param_map[j]; if (pi>=0&&pi<8) ftg[pi] = j; }

    AD m0_ad(rp[0]); if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;
    AD gamma0_ad;
    if (res.type == ResModelType::BWR || res.type == ResModelType::BW) {
        gamma0_ad = AD(rp[1]); if (ftg[1] >= 0) gamma0_ad.grad[ftg[1]] = 1.0;
    }
    AD q_ad(qq);
    // q0 AD (preserves derivative through breakup_momentum)
    AD md1a(node_md1), md2a(node_md2);
    AD sm = md1a+md2a, dm = md1a-md2a, msq = m0_ad*m0_ad;
    AD q0sq = (msq - sm*sm)*(msq - dm*dm)/(AD(4.0)*msq);
    q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
    AD q0_ad = sqrt(q0sq);
    double q0_val = q0_ad.val;

    ComplexVar<double, Nfree, false> R_ad;
    if (res.type == ResModelType::BWR)
        R_ad = BWR<AD>(m_ad, m0_ad, gamma0_ad, res_L, q_ad, q0_ad, bf_d);
    else if (res.type == ResModelType::BW)
        R_ad = BW<AD>(m_ad, m0_ad, gamma0_ad);
    else return;

    double bf_val = 1.0;
    if (res.type == ResModelType::BWR || res.type == ResModelType::ONE)
        bf_val = Bf<double>(res_L, qq, q0_val, bf_d);

    // Step 3: per-SL Bf 修正
    double bf0_sl0 = 1.0, bf0_sl1 = 1.0;
    AD bf_ad_sl1;
    if (d_slamps && d_v) {
        const DecayNode& nd0 = d_decayNodes[0];
        LorentzVector pM0 = d_momenta->getMomentum(global_evt, nd0.mother_idx);
        double mm0 = pM0.M(), md1_0 = nd0.mass[1], md2_0 = nd0.mass[2];
        LorentzVector pD1 = d_momenta->getMomentum(global_evt, nd0.daug1_idx);
        LorentzVector pD2 = d_momenta->getMomentum(global_evt, nd0.daug2_idx);
        if (md1_0 <= 0) md1_0 = pD1.M();
        if (md2_0 <= 0) md2_0 = pD2.M();
        double qq0 = breakup_momentum(mm0, md1_0, md2_0);
        double md2q0 = nd0.mass[2] <= 0 ? rp[0] : md2_0;
        double q00 = breakup_momentum(nd0.mass[0], md1_0, md2q0);
        bf0_sl0 = Bf<double>(d_slComb[0].L, qq0, q00, bf_d);
        int L1 = d_slComb[decayChain_size].L;
        bf0_sl1 = Bf<double>(L1, qq0, q00, bf_d);
        if (L1 > 0 && Nfree > 0) {
            AD qq0a(qq0), m1a(md1_0), m2a = m0_ad, m0a(nd0.mass[0]);
            AD s2=m1a+m2a, d2=m1a-m2a, msq2=m0a*m0a;
            AD q0s2 = (msq2-s2*s2)*(msq2-d2*d2)/(AD(4.0)*msq2);
            q0s2.val = q0s2.val < 0.0 ? 0.0 : q0s2.val;
            bf_ad_sl1 = Bf<AD>(L1, qq0a, sqrt(q0s2), bf_d);
        }
    }

    for (int pol = 0; pol < nPolar; ++pol) {
        cuComplex w_val = d_w[evt * nPolar + pol];
        cuComplex T_eff, T1_BWR;
        if (d_slamps && d_v) {
            int i0 = 0*nTotalEvt*nPolar + global_evt*nPolar + pol;
            int i1 = 1*nTotalEvt*nPolar + global_evt*nPolar + pol;
            auto s0 = d_slamps[i0], s1 = d_slamps[i1];
            cuComplex v0 = d_v[site+0], v1 = d_v[site+1];
            cuComplex T0 = make_cuComplex((float)(v0.x*s0.real()-v0.y*s0.imag()),(float)(v0.x*s0.imag()+v0.y*s0.real()));
            cuComplex T1 = make_cuComplex((float)(v1.x*s1.real()-v1.y*s1.imag()),(float)(v1.x*s1.imag()+v1.y*s1.real()));
            T_eff = make_cuComplex((float)(T0.x*bf0_sl0+T1.x*bf0_sl1),(float)(T0.y*bf0_sl0+T1.y*bf0_sl1));
            double br=R_ad.real.val, bi=R_ad.imag.val;
            T1_BWR = make_cuComplex((float)(T1.x*br-T1.y*bi),(float)(T1.x*bi+T1.y*br));
        } else {
            T_eff = d_T[global_evt * nPolar + pol];
            T1_BWR = make_cuComplex(0.0f,0.0f);
        }
        double cr = (double)w_val.x*(double)T_eff.x + (double)w_val.y*(double)T_eff.y;
        double ci = (double)w_val.x*(double)T_eff.y - (double)w_val.y*(double)T_eff.x;
        double cbr=0.0, cbi=0.0;
        if (bf0_sl1 != 1.0) {
            cbr = (double)w_val.x*(double)T1_BWR.x + (double)w_val.y*(double)T1_BWR.y;
            cbi = (double)w_val.x*(double)T1_BWR.y - (double)w_val.y*(double)T1_BWR.x;
        }
        for (int j = 0; j < Nfree; ++j) {
            double dRr = R_ad.real.grad[j], dRi = R_ad.imag.grad[j];
            double c = -2.0 * sign * (cr*dRr - ci*dRi) * bf_val;
            if (bf0_sl1 != 1.0 && Nfree > 0) c += -2.0 * sign * (cbr * bf_ad_sl1.grad[j]) * bf_val;
            atomicAdd(&d_grad[d_global_idx[j]], c);
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
    int nSL, int nTotal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTotal) return;

    double re = 0.0, im = 0.0;
    for (int sl = 0; sl < nSL; ++sl) {
        auto sv = d_slamps[sl * nTotal + idx];
        auto vv = d_v[sl];
        // (vv.x + i*vv.y) * (sv.real + i*sv.imag)
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
            d_v + block.site, nSL, nTotal);
        cudaDeviceSynchronize();
    }
}

// 小 kernel：双精度数组累加 y[i] += alpha * x[i]
__global__ void daxpy_kernel(double* y, const double* x, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
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
    const cuComplex* d_v)
{
    int n_gpu = static_cast<int>(d_w.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;
    bool has_offset = (t_offset.size() == d_w.size());

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
                switch (Nlocal) {
                case 1:
                    resonanceGradientKernel<1><<<grid, kBlockSize>>>(
                        d_w[gpu], block.d_T[gpu],
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], block.d_resonances[gpu],
                        r, block.d_all_params[gpu],
                        d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                        nEv, nPol, cas->getDecayChainSize(), 3.0, sign,
                        evt_off, cas->getSLAmps()[gpu], d_v, block.site);
                    break;
                case 2:
                    resonanceGradientKernel<2><<<grid, kBlockSize>>>(
                        d_w[gpu], block.d_T[gpu],
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], block.d_resonances[gpu],
                        r, block.d_all_params[gpu],
                        d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                        nEv, nPol, cas->getDecayChainSize(), 3.0, sign,
                        evt_off, cas->getSLAmps()[gpu], d_v, block.site);
                    break;
                case 3:
                    resonanceGradientKernel<3><<<grid, kBlockSize>>>(
                        d_w[gpu], block.d_T[gpu],
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], block.d_resonances[gpu],
                        r, block.d_all_params[gpu],
                        d_param_map, d_grad_per_gpu[gpu], d_global_idx,
                        nEv, nPol, cas->getDecayChainSize(), 3.0, sign,
                        evt_off, cas->getSLAmps()[gpu], d_v, block.site);
                    break;
                default: break;
                }
                cudaDeviceSynchronize();
                cudaFree(d_param_map);
                cudaFree(d_global_idx);
            }
        }
    }

    // 累加所有 GPU 到 d_grad_res（GPU 0）
    cudaSetDevice(0);
    cudaMemset(d_grad_res, 0, n_free * sizeof(double));
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == 0) {
            cudaMemcpy(d_grad_res, d_grad_per_gpu[0],
                       n_free * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            // 通过 host 中转累加（n_free 很小，~几个 double）
            std::vector<double> h_temp(n_free);
            cudaSetDevice(gpu);
            cudaMemcpy(h_temp.data(), d_grad_per_gpu[gpu],
                       n_free * sizeof(double), cudaMemcpyDeviceToHost);
            cudaSetDevice(0);
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
    cudaSetDevice(0);
}

// ============================================================
// 共振态 Hessian kernel 实现
// ============================================================

// 辅助：从 Var 数组构建 param AD 对象（设置独立变量）
template <int Nfree>
__device__ void make_var_params(
    const double* p,                   // 参数值数组
    const int* param_map,              // [Nfree] → params 下标
    Var<double, Nfree, false>* m0_ad,
    Var<double, Nfree, false>* g_ad)
{
    *m0_ad = Var<double, Nfree, false>(p[0]);
    *g_ad  = Var<double, Nfree, false>(p[1]);
    for (int j = 0; j < Nfree; ++j) {
        int pi = param_map[j];
        if (pi == 0) m0_ad->grad[j] = 1.0;
        if (pi == 1) g_ad->grad[j]  = 1.0;
    }
}

template <int Nfree>
__device__ void make_var_params_hess(
    const double* p, const int* param_map,
    Var<double, Nfree, true>* m0_ad,
    Var<double, Nfree, true>* g_ad)
{
    *m0_ad = Var<double, Nfree, true>(p[0]);
    *g_ad  = Var<double, Nfree, true>(p[1]);
    for (int j = 0; j < Nfree; ++j) {
        int pi = param_map[j];
        if (pi == 0) m0_ad->grad[j] = 1.0;
        if (pi == 1) g_ad->grad[j]  = 1.0;
    }
}

template <int Pr, int Ps, bool SameRes>
__global__ void resonanceHessianBlockKernel(
    const cuComplex* d_w, const cuComplex* d_T_r, const cuComplex* d_T_s,
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const SL* d_slComb, const DeviceResonance* d_res, const double* d_all_params,
    int res_r_idx, int res_s_idx,
    const int* d_param_map_r, const int* d_param_map_s,
    double* d_hess, int hess_ld, int offset_r, int offset_s,
    int nEvents, int nPolar, int decayChain_size, double bf_d, double sign)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;

    const DeviceResonance& res_r = d_res[res_r_idx];
    const double* p_r = d_all_params + res_r.param_offset;

    // Step 1: 找共振态 r 节点，算不变质量
    double mm = 0.0, qq = 0.0, q0_r = 0.0;
    int res_L_r = 0;
    bool found = false;
    for (int ni = 0; ni < decayChain_size; ++ni) {
        const DecayNode& node = d_decayNodes[ni];
        if (node.mother_idx != res_r.particle_idx) continue;
        LorentzVector pM = d_momenta->getMomentum(evt, node.mother_idx);
        LorentzVector pD1 = d_momenta->getMomentum(evt, node.daug1_idx);
        LorentzVector pD2 = d_momenta->getMomentum(evt, node.daug2_idx);
        mm = pM.M();
        double md1 = node.mass[1], md2 = node.mass[2];
        if (md1 <= 0) md1 = pD1.M();
        if (md2 <= 0) md2 = pD2.M();
        qq = breakup_momentum(mm, md1, md2);
        q0_r = breakup_momentum(p_r[0], md1, md2);
        res_L_r = d_slComb[ni].L;
        found = true;
        break;
    }
    if (!found) return;

    // Step 2: D_r = ∂R_r/∂θ（Pr 个值），用 WithHess=false
    using AD1 = Var<double, Pr, false>;
    AD1 m_ad(mm);
    AD1 m0_r_ad(p_r[0]), g_r_ad(p_r[1]);
    for (int j = 0; j < Pr; ++j) {
        int pi = d_param_map_r[j];
        if (pi == 0) m0_r_ad.grad[j] = 1.0;
        if (pi == 1) g_r_ad.grad[j]  = 1.0;
    }
    AD1 q_ad(qq), q0_ad(q0_r);
    ComplexVar<double, Pr, false> R_r_ad;
    if (res_r.type == ResModelType::BWR)
        R_r_ad = BWR<AD1>(m_ad, m0_r_ad, g_r_ad, res_L_r, q_ad, q0_ad, bf_d);
    else if (res_r.type == ResModelType::BW)
        R_r_ad = BW<AD1>(m_ad, m0_r_ad, g_r_ad);
    else return;
    double bf_val = Bf<double>(res_L_r, qq, q0_r, bf_d);

    // Step 3: D_s 和（如果 SameRes）D²_r
    const DeviceResonance* res_s_ptr = &res_r;
    const double* p_s = p_r;
    int res_L_s = res_L_r;
    double q0_s = q0_r;

    if constexpr (!SameRes) {
        res_s_ptr = &d_res[res_s_idx];
        p_s = d_all_params + res_s_ptr->param_offset;
        // 找共振态 s 节点
        bool found_s = false;
        for (int ni = 0; ni < decayChain_size; ++ni) {
            if (d_decayNodes[ni].mother_idx == res_s_ptr->particle_idx) {
                LorentzVector pM = d_momenta->getMomentum(evt, d_decayNodes[ni].mother_idx);
                mm = pM.M();
                LorentzVector pD1 = d_momenta->getMomentum(evt, d_decayNodes[ni].daug1_idx);
                LorentzVector pD2 = d_momenta->getMomentum(evt, d_decayNodes[ni].daug2_idx);
                double md1 = d_decayNodes[ni].mass[1], md2 = d_decayNodes[ni].mass[2];
                if (md1 <= 0) md1 = pD1.M();
                if (md2 <= 0) md2 = pD2.M();
                qq = breakup_momentum(mm, md1, md2);
                q0_s = breakup_momentum(p_s[0], md1, md2);
                res_L_s = d_slComb[ni].L;
                found_s = true;
                break;
            }
        }
        if (!found_s) return;
    }

    using AD2 = Var<double, Ps, false>;
    AD2 m_ad_s(mm);
    AD2 m0_s_ad(p_s[0]), g_s_ad(p_s[1]);
    for (int k = 0; k < Ps; ++k) {
        int pi = d_param_map_s[k];
        if (pi == 0) m0_s_ad.grad[k] = 1.0;
        if (pi == 1) g_s_ad.grad[k]  = 1.0;
    }
    AD2 q_ad_s(qq), q0_ad_s(q0_s);
    ComplexVar<double, Ps, false> R_s_ad;
    if (res_s_ptr->type == ResModelType::BWR)
        R_s_ad = BWR<AD2>(m_ad_s, m0_s_ad, g_s_ad, res_L_s, q_ad_s, q0_ad_s, bf_d);
    else if (res_s_ptr->type == ResModelType::BW)
        R_s_ad = BW<AD2>(m_ad_s, m0_s_ad, g_s_ad);
    else return;
    double bf_val_s = Bf<double>(res_L_s, qq, q0_s, bf_d);

    // Step 4: 如果需要 Hessian（SameRes），用 WithHess=true 再算一次
    ComplexVar<double, Pr, true> R_r_hess;
    if constexpr (SameRes) {
        using ADH = Var<double, Pr, true>;
        ADH m_ad_h(mm);
        ADH m0_h(p_r[0]), g_h(p_r[1]);
        for (int j = 0; j < Pr; ++j) {
            int pi = d_param_map_r[j];
            if (pi == 0) m0_h.grad[j] = 1.0;
            if (pi == 1) g_h.grad[j]  = 1.0;
        }
        ADH q_ad_h(qq), q0_ad_h(q0_r);
        if (res_r.type == ResModelType::BWR)
            R_r_hess = BWR<ADH>(m_ad_h, m0_h, g_h, res_L_r, q_ad_h, q0_ad_h, bf_d);
        else
            R_r_hess = BW<ADH>(m_ad_h, m0_h, g_h);
    }

    // Step 5: 对每个极化累加
    for (int p = 0; p < nPolar; ++p) {
        cuComplex w_val = d_w[evt * nPolar + p];
        cuComplex Tr_val = d_T_r[evt * nPolar + p];
        cuComplex Ts_val = SameRes ? Tr_val : d_T_s[evt * nPolar + p];

        double w2 = (double)w_val.x * w_val.x + (double)w_val.y * w_val.y;
        if (w2 < 1e-30) continue;
        double inv_w2 = 1.0 / w2;
        double I_inv = w2;  // I = 1/|w|² → 1/I = |w|²

        // conj(w)
        double cw_re = (double)w_val.x;
        double cw_im = -(double)w_val.y;

        // c_T_r = conj(w) * T_r
        double cTr_re = cw_re * Tr_val.x - cw_im * Tr_val.y;
        double cTr_im = cw_re * Tr_val.y + cw_im * Tr_val.x;

        // c_T_s = conj(w) * T_s
        double cTs_re, cTs_im;
        if constexpr (SameRes) {
            cTs_re = cTr_re; cTs_im = cTr_im;
        } else {
            cTs_re = cw_re * Ts_val.x - cw_im * Ts_val.y;
            cTs_im = cw_re * Ts_val.y + cw_im * Ts_val.x;
        }

        // G_r[j] = 2 * inv_w2 * Re(c_T_r * D_r[j])
        double G_r[8], G_s[8];
        for (int j = 0; j < Pr; ++j) {
            double dRr_re = R_r_ad.real.grad[j];
            double dRr_im = R_r_ad.imag.grad[j];
            G_r[j] = 2.0 * inv_w2 * (cTr_re * dRr_re - cTr_im * dRr_im) * bf_val;
        }
        for (int k = 0; k < Ps; ++k) {
            double dRs_re = R_s_ad.real.grad[k];
            double dRs_im = R_s_ad.imag.grad[k];
            G_s[k] = 2.0 * inv_w2 * (cTs_re * dRs_re - cTs_im * dRs_im) * bf_val_s;
        }

        // 一阶乘积项: H += |w|⁴ * G[j] * G[k] (即 I^{-2} * G*G)
        double I2_inv = w2 * w2;  // |w|⁴ = 1/I²
        for (int j = 0; j < Pr; ++j) {
            for (int k = 0; k < Ps; ++k) {
                double prod = sign * I2_inv * G_r[j] * G_s[k];
                atomicAdd(&d_hess[(offset_s + k) * hess_ld + (offset_r + j)], prod);
            }
        }

        // 二阶导数项: H -= 2*|w|² * Re(... )  (= -2/I * Re(...))
        double factor = sign * 2.0 * I_inv;
        if constexpr (SameRes) {
            // SameRes: -2|w|² * Re(|T_r|² * conj(D_k)*D_j + conj(w)/|w|² * T_r * D²_jk)
            double T2 = (double)Tr_val.x * Tr_val.x + (double)Tr_val.y * Tr_val.y;
            double ST_re = cw_re * inv_w2 * Tr_val.x - cw_im * inv_w2 * Tr_val.y;
            double ST_im = cw_re * inv_w2 * Tr_val.y + cw_im * inv_w2 * Tr_val.x;
            for (int j = 0; j < Pr; ++j) {
                for (int k = 0; k < Pr; ++k) {
                    double dR_re_j = R_r_ad.real.grad[j];
                    double dR_im_j = R_r_ad.imag.grad[j];
                    double dR_re_k = R_r_ad.real.grad[k];
                    double dR_im_k = R_r_ad.imag.grad[k];
                    // |T|² * conj(D_k) * D_j
                    double d2_re = T2 * (dR_re_k * dR_re_j + dR_im_k * dR_im_j);
                    double d2_im = T2 * (dR_re_k * dR_im_j - dR_im_k * dR_re_j);
                    // conj(w)/|w|² * T * D²_jk
                    double D2_re = R_r_hess.real.hess[j][k];
                    double D2_im = R_r_hess.imag.hess[j][k];
                    d2_re += ST_re * D2_re - ST_im * D2_im;
                    double term = factor * d2_re * bf_val * bf_val;
                    atomicAdd(&d_hess[(offset_s + k) * hess_ld + (offset_r + j)], -term);
                }
            }
        } else {
            // !SameRes: -2|w|² * Re(conj(T_s)*T_r * conj(D_sk)*D_rj)
            double TsTr_re = Ts_val.x * Tr_val.x + Ts_val.y * Tr_val.y;
            double TsTr_im = Ts_val.x * Tr_val.y - Ts_val.y * Tr_val.x;
            for (int j = 0; j < Pr; ++j) {
                for (int k = 0; k < Ps; ++k) {
                    double dRr_re = R_r_ad.real.grad[j];
                    double dRr_im = R_r_ad.imag.grad[j];
                    double dRs_re = R_s_ad.real.grad[k];
                    double dRs_im = R_s_ad.imag.grad[k];
                    // conj(D_sk) * D_rj
                    double cd_re = dRs_re * dRr_re + dRs_im * dRr_im;
                    double cd_im = dRs_re * dRr_im - dRs_im * dRr_re;
                    // conj(T_s)*T_r * conj(D_sk)*D_rj
                    double d2_re = TsTr_re * cd_re - TsTr_im * cd_im;
                    double term = factor * d2_re * bf_val * bf_val_s;
                    atomicAdd(&d_hess[(offset_s + k) * hess_ld + (offset_r + j)], -term);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AmpCalc::computeResonanceHessian
// ---------------------------------------------------------------------------
void AmpCalc::computeResonanceHessian(
    const std::vector<cuComplex*>& d_w,
    const std::vector<int>& n_events,
    double* d_hess, int hess_ld, double sign)
{
    int n_gpu = static_cast<int>(d_w.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;

    // 计算每个 block 的全局参数偏移
    std::vector<int> block_offset(blocks_.size(), 0);
    // 简化：直接用 slots_ 遍历所有共振态及其自由参数

    constexpr int kBlockSize = 256;

    // 为每 GPU 分配临时 Hessian buffer
    std::vector<double*> d_hess_per_gpu(n_gpu, nullptr);
    int hess_total = n_free * n_free;
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaMalloc(&d_hess_per_gpu[gpu], hess_total * sizeof(double));
        cudaMemset(d_hess_per_gpu[gpu], 0, hess_total * sizeof(double));
    }

    // 遍历所有 block 对
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        auto& blk_i = blocks_[bi];
        auto& cas_i = cas_list_[blk_i.cas_idx];

        for (size_t bj = bi; bj < blocks_.size(); ++bj) {
            auto& blk_j = blocks_[bj];
            auto& cas_j = cas_list_[blk_j.cas_idx];

            // cas 必须相同（共享 SL 数据）
            if (cas_i != cas_j) continue;

            for (int ri = 0; ri < blk_i.resonance_count; ++ri) {
                // 收集共振态 ri 的自由参数
                std::vector<int> map_i, global_i;
                for (int s = 0; s < n_free; ++s) {
                    if (slots_[s].block_idx == (int)bi && slots_[s].res_idx == ri) {
                        map_i.push_back(slots_[s].param_idx);
                        global_i.push_back(s);
                    }
                }
                int Pr = static_cast<int>(map_i.size());
                if (Pr == 0) continue;

                for (int rj = ((bi == bj) ? ri : 0); rj < blk_j.resonance_count; ++rj) {
                    if (bi == bj && rj < ri) continue;  // 只算上三角

                    std::vector<int> map_j, global_j;
                    for (int s = 0; s < n_free; ++s) {
                        if (slots_[s].block_idx == (int)bj && slots_[s].res_idx == rj) {
                            map_j.push_back(slots_[s].param_idx);
                            global_j.push_back(s);
                        }
                    }
                    int Ps = static_cast<int>(map_j.size());
                    if (Ps == 0) continue;

                    bool same_res = (bi == bj && ri == rj);

                    // 每 GPU 启动 kernel
                    for (int gpu = 0; gpu < n_gpu; ++gpu) {
                        cudaSetDevice(gpu);

                        int *d_map_i, *d_map_j;
                        cudaMalloc(&d_map_i, Pr * sizeof(int));
                        cudaMalloc(&d_map_j, Ps * sizeof(int));
                        cudaMemcpy(d_map_i, map_i.data(), Pr * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_map_j, map_j.data(), Ps * sizeof(int), cudaMemcpyHostToDevice);

                        int nEv = n_events[gpu];
                        int nPol = static_cast<int>(cas_i->getNPolarizations());
                        int grid = (nEv + kBlockSize - 1) / kBlockSize;

                        // 按 Pr, Ps 分发
                        if (same_res) {
                            if (Pr == 1) {
                                resonanceHessianBlockKernel<1,1,true><<<grid,kBlockSize>>>(
                                    d_w[gpu], blk_i.d_T[gpu], blk_i.d_T[gpu],
                                    cas_i->getMomenta()[gpu], cas_i->getDecayNodes()[gpu],
                                    cas_i->getDeviceSLCombs()[gpu], blk_i.d_resonances[gpu],
                                    blk_i.d_all_params[gpu],
                                    ri, ri, d_map_i, d_map_i,
                                    d_hess_per_gpu[gpu], n_free,
                                    global_i[0], global_i[0],
                                    nEv, nPol, cas_i->getDecayChainSize(), 3.0, sign);
                            } else if (Pr == 2) {
                                resonanceHessianBlockKernel<2,2,true><<<grid,kBlockSize>>>(
                                    d_w[gpu], blk_i.d_T[gpu], blk_i.d_T[gpu],
                                    cas_i->getMomenta()[gpu], cas_i->getDecayNodes()[gpu],
                                    cas_i->getDeviceSLCombs()[gpu], blk_i.d_resonances[gpu],
                                    blk_i.d_all_params[gpu],
                                    ri, ri, d_map_i, d_map_i,
                                    d_hess_per_gpu[gpu], n_free,
                                    global_i[0], global_i[0],
                                    nEv, nPol, cas_i->getDecayChainSize(), 3.0, sign);
                            } else if (Pr == 3) {
                                resonanceHessianBlockKernel<3,3,true><<<grid,kBlockSize>>>(
                                    d_w[gpu], blk_i.d_T[gpu], blk_i.d_T[gpu],
                                    cas_i->getMomenta()[gpu], cas_i->getDecayNodes()[gpu],
                                    cas_i->getDeviceSLCombs()[gpu], blk_i.d_resonances[gpu],
                                    blk_i.d_all_params[gpu],
                                    ri, ri, d_map_i, d_map_i,
                                    d_hess_per_gpu[gpu], n_free,
                                    global_i[0], global_i[0],
                                    nEv, nPol, cas_i->getDecayChainSize(), 3.0, sign);
                            }
                        } else {
                            if (Pr == 1 && Ps == 1) {
                                resonanceHessianBlockKernel<1,1,false><<<grid,kBlockSize>>>(
                                    d_w[gpu], blk_i.d_T[gpu], blk_j.d_T[gpu],
                                    cas_i->getMomenta()[gpu], cas_i->getDecayNodes()[gpu],
                                    cas_i->getDeviceSLCombs()[gpu],
                                    // 同cas，用同一个res数组和params
                                    blk_i.d_resonances[gpu],
                                    blk_i.d_all_params[gpu],
                                    ri, rj, d_map_i, d_map_j,
                                    d_hess_per_gpu[gpu], n_free,
                                    global_i[0], global_j[0],
                                    nEv, nPol, cas_i->getDecayChainSize(), 3.0, sign);
                            }
                            // 更多 Pr×Ps 组合...
                        }

                        cudaDeviceSynchronize();
                        cudaFree(d_map_i);
                        cudaFree(d_map_j);
                    }
                }
            }
        }
    }

    // 累加所有 GPU 到 d_hess
    cudaSetDevice(0);
    cudaMemset(d_hess, 0, hess_total * sizeof(double));
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == 0) {
            cudaMemcpy(d_hess, d_hess_per_gpu[0],
                       hess_total * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            std::vector<double> h_temp(hess_total);
            cudaSetDevice(gpu);
            cudaMemcpy(h_temp.data(), d_hess_per_gpu[gpu],
                       hess_total * sizeof(double), cudaMemcpyDeviceToHost);
            cudaSetDevice(0);
            double* d_temp;
            cudaMalloc(&d_temp, hess_total * sizeof(double));
            cudaMemcpy(d_temp, h_temp.data(), hess_total * sizeof(double), cudaMemcpyHostToDevice);
            int grid = (hess_total + 255) / 256;
            daxpy_kernel<<<grid, 256>>>(d_hess, d_temp, 1.0, hess_total);
            cudaDeviceSynchronize();
            cudaFree(d_temp);
        }
    }

    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaFree(d_hess_per_gpu[gpu]);
    }
    cudaSetDevice(0);
}
