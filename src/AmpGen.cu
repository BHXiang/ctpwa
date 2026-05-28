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

            AD m0_q0_ad, md1_q0_ad, md2_q0_ad;
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) {
                m0_q0_ad = m0_ad;
            } else if (node.mass[0] > 0) {
                m0_q0_ad = AD(node.mass[0]);
            } else {
                m0_q0_ad = AD(1.0);
            }
            if (node.mass[1] <= 0 && node.daug1_idx == target_res.particle_idx)
                md1_q0_ad = m0_ad;
            else
                md1_q0_ad = AD(node.mass[1] > 0 ? node.mass[1] : md1);
            if (node.mass[2] <= 0 && node.daug2_idx == target_res.particle_idx)
                md2_q0_ad = m0_ad;
            else
                md2_q0_ad = AD(node.mass[2] > 0 ? node.mass[2] : md2);

            AD s_md = md1_q0_ad + md2_q0_ad;
            AD d_md = md1_q0_ad - md2_q0_ad;
            AD m0sq = m0_q0_ad * m0_q0_ad;
            AD q0sq = (m0sq - s_md*s_md)*(m0sq - d_md*d_md)/(AD(4.0)*m0sq);
            q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
            AD q0_ad = sqrt(q0sq);

            CV node_factor(1.0, 0.0);
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) {
                AD m_ad(mm);
                AD res_m0 = m0_ad;
                if (target_res.type == ResModelType::BWR) {
                    node_factor = BWR<AD>(m_ad, res_m0, gamma_ad, L, q_ad, q0_ad, bf_d);
                } else if (target_res.type == ResModelType::BW) {
                    node_factor = BW<AD>(m_ad, res_m0, gamma_ad);
                } else {
                    node_factor = CV(1.0, 0.0);
                }
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = node_factor.real * bf_ad;
                node_factor.imag = node_factor.imag * bf_ad;
            } else {
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = bf_ad;
                node_factor.imag = AD(0.0);
            }

            CV new_R;
            new_R.real = R_ad.real * node_factor.real - R_ad.imag * node_factor.imag;
            new_R.imag = R_ad.real * node_factor.imag + R_ad.imag * node_factor.real;
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
                        evt_off, cas->getSLAmps()[gpu], d_v, block.site,
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
                        evt_off, cas->getSLAmps()[gpu], d_v, block.site,
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
                        evt_off, cas->getSLAmps()[gpu], d_v, block.site,
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

    // 将各 GPU 结果通过 daxpy 累加到 d_grad_res
    cudaSetDevice(0);
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == 0) {
            daxpy_kernel<<<1, 64>>>(d_grad_res, d_grad_per_gpu[0], 1.0, n_free);
            cudaDeviceSynchronize();
        } else {
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

// 完整链路 AutoDiff Hessian kernel (单共振态版)
// 计算 ∂²(w·log I)/∂θ_j∂θ_k 每事件贡献并累加到 d_hess
//   I = Σ_p |S_p|², S_p = Σ_sl v_sl · slamp_{sl,e,p} · F_sl(θ)
//   d²(log I)/dθ² = 2 Re(Σ_p [dS_k·conj(dS_j) + conj(S_p)·d²S_jk]) / I  -  (dI/dθ_j)(dI/dθ_k) / I²
// w_evt: 每事件权重 (data: -1, bkg: +w_bkg) — 通过 d_event_weights 或 default_weight 提供
template <int Nfree>
__global__ void resonanceHessianFullKernel(
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const SL* d_slComb, const DeviceResonance* d_res,
    int res_idx_in_block, const double* d_all_params,
    const int* d_param_map, double* d_hess, const int* d_global_idx,
    int nEvents, int nPolar, int decayChain_size, int nSLComb, double bf_d,
    int t_evt_offset,
    const thrust::complex<double>* d_slamps,
    const cuComplex* d_v, int site, int hess_ld,
    const double* d_event_weights, double default_weight,
    const cuComplex* d_amp, int n_ext)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int global_evt = evt + t_evt_offset;
    double w_evt = d_event_weights ? d_event_weights[evt] : default_weight;
    if (w_evt == 0.0) return;

    using AD = Var<double, Nfree, true>;
    using CV = ComplexVar<double, Nfree, true>;

    const DeviceResonance& target_res = d_res[res_idx_in_block];
    const double* target_rp = d_all_params + target_res.param_offset;

    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nfree; ++j) {
        int pi = d_param_map[j];
        if (pi >= 0 && pi < 8) ftg[pi] = j;
    }

    AD m0_ad(target_rp[0]);
    if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;
    AD gamma_ad;
    if (target_res.type == ResModelType::BWR || target_res.type == ResModelType::BW) {
        gamma_ad = AD(target_rp[1]);
        if (ftg[1] >= 0) gamma_ad.grad[ftg[1]] = 1.0;
    }

    int n_events_total = d_momenta->n_events;
    constexpr int MAX_POL = 32;

    double S_re[MAX_POL], S_im[MAX_POL];
    double dS_re[MAX_POL][Nfree], dS_im[MAX_POL][Nfree];
    double d2S_re[MAX_POL][Nfree][Nfree], d2S_im[MAX_POL][Nfree][Nfree];
    for (int p = 0; p < nPolar; ++p) {
        S_re[p] = 0.0; S_im[p] = 0.0;
        for (int j = 0; j < Nfree; ++j) {
            dS_re[p][j] = 0.0; dS_im[p][j] = 0.0;
            for (int k = 0; k < Nfree; ++k) {
                d2S_re[p][j][k] = 0.0; d2S_im[p][j][k] = 0.0;
            }
        }
    }

    // Full S over all partial waves (from precomputed d_amp)
    for (int p = 0; p < nPolar; ++p) {
        for (int a = 0; a < n_ext; ++a) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
            cuComplex v_a = d_v[a];
            S_re[p] += (double)v_a.x * amp_ap.x - (double)v_a.y * amp_ap.y;
            S_im[p] += (double)v_a.x * amp_ap.y + (double)v_a.y * amp_ap.x;
        }
    }

    for (int sl_idx = 0; sl_idx < nSLComb; ++sl_idx) {
        CV R_ad(1.0, 0.0);
        for (int ni = 0; ni < decayChain_size; ++ni) {
            const DecayNode& node = d_decayNodes[ni];
            const SL& sl = d_slComb[sl_idx * decayChain_size + ni];
            int L = sl.L;

            LorentzVector pM  = d_momenta->getMomentum(global_evt, node.mother_idx);
            LorentzVector pD1 = d_momenta->getMomentum(global_evt, node.daug1_idx);
            LorentzVector pD2 = d_momenta->getMomentum(global_evt, node.daug2_idx);

            double mm = pM.M();
            double qq = breakup_momentum(mm, pD1.M(), pD2.M());
            AD q_ad(qq);
            double md1 = pD1.M(), md2 = pD2.M();

            AD m0_q0_ad, md1_q0_ad, md2_q0_ad;
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) {
                m0_q0_ad = m0_ad;
            } else if (node.mass[0] > 0) {
                m0_q0_ad = AD(node.mass[0]);
            } else {
                m0_q0_ad = AD(1.0);
            }
            if (node.mass[1] <= 0 && node.daug1_idx == target_res.particle_idx)
                md1_q0_ad = m0_ad;
            else
                md1_q0_ad = AD(node.mass[1] > 0 ? node.mass[1] : md1);
            if (node.mass[2] <= 0 && node.daug2_idx == target_res.particle_idx)
                md2_q0_ad = m0_ad;
            else
                md2_q0_ad = AD(node.mass[2] > 0 ? node.mass[2] : md2);

            AD s_md = md1_q0_ad + md2_q0_ad;
            AD d_md = md1_q0_ad - md2_q0_ad;
            AD m0sq = m0_q0_ad * m0_q0_ad;
            AD q0sq = (m0sq - s_md*s_md)*(m0sq - d_md*d_md)/(AD(4.0)*m0sq);
            q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
            AD q0_ad = sqrt(q0sq);

            CV node_factor(1.0, 0.0);
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) {
                AD m_ad(mm);
                if (target_res.type == ResModelType::BWR) {
                    node_factor = BWR<AD>(m_ad, m0_ad, gamma_ad, L, q_ad, q0_ad, bf_d);
                } else if (target_res.type == ResModelType::BW) {
                    node_factor = BW<AD>(m_ad, m0_ad, gamma_ad);
                }
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = node_factor.real * bf_ad;
                node_factor.imag = node_factor.imag * bf_ad;
            } else {
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = bf_ad;
                node_factor.imag = AD(0.0);
            }

            CV new_R;
            new_R.real = R_ad.real * node_factor.real - R_ad.imag * node_factor.imag;
            new_R.imag = R_ad.real * node_factor.imag + R_ad.imag * node_factor.real;
            R_ad = new_R;
        }

        cuComplex v_sl = d_v[site + sl_idx];
        for (int p = 0; p < nPolar; ++p) {
            int amp_idx = sl_idx * n_events_total * nPolar + global_evt * nPolar + p;
            auto sl_amp = d_slamps[amp_idx];
            double sl_re = sl_amp.real();
            double sl_im = sl_amp.imag();
            double c_re = (double)v_sl.x * sl_re - (double)v_sl.y * sl_im;
            double c_im = (double)v_sl.x * sl_im + (double)v_sl.y * sl_re;

            for (int j = 0; j < Nfree; ++j) {
                dS_re[p][j] += c_re * R_ad.real.grad[j] - c_im * R_ad.imag.grad[j];
                dS_im[p][j] += c_re * R_ad.imag.grad[j] + c_im * R_ad.real.grad[j];
                for (int k = 0; k < Nfree; ++k) {
                    d2S_re[p][j][k] += c_re * R_ad.real.hess[j][k] - c_im * R_ad.imag.hess[j][k];
                    d2S_im[p][j][k] += c_re * R_ad.imag.hess[j][k] + c_im * R_ad.real.hess[j][k];
                }
            }
        }
    }

    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) I_val += S_re[p]*S_re[p] + S_im[p]*S_im[p];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;
    double inv_I2 = inv_I * inv_I;

    double G[8] = {0};
    for (int j = 0; j < Nfree; ++j) {
        double gv = 0.0;
        for (int p = 0; p < nPolar; ++p) gv += S_re[p]*dS_re[p][j] + S_im[p]*dS_im[p][j];
        G[j] = 2.0 * gv;  // = dI/dθ_j
    }

    for (int j = 0; j < Nfree; ++j) {
        for (int k = 0; k < Nfree; ++k) {
            double R2 = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                R2 += dS_re[p][k]*dS_re[p][j] + dS_im[p][k]*dS_im[p][j]
                    + S_re[p]*d2S_re[p][j][k] + S_im[p]*d2S_im[p][j][k];
            }
            // d²(log I)/dθ_j∂θ_k = 2·R2/I − G_j·G_k/I²
            double term = w_evt * (2.0 * R2 * inv_I - G[j] * G[k] * inv_I2);
            atomicAdd(&d_hess[d_global_idx[k]*hess_ld + d_global_idx[j]], term);
        }
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
    int nEvents, int nPolar, int decayChain_size, double bf_d, double sign,
    int evt_offset = 0)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int evt_abs = evt + evt_offset;  // absolute event index in momenta array

    const DeviceResonance& res_r = d_res[res_r_idx];
    const double* p_r = d_all_params + res_r.param_offset;

    // Step 1: 找共振态 r 节点，算不变质量
    double mm = 0.0, qq = 0.0, q0_r = 0.0;
    int res_L_r = 0;
    bool found = false;
    for (int ni = 0; ni < decayChain_size; ++ni) {
        const DecayNode& node = d_decayNodes[ni];
        if (node.mother_idx != res_r.particle_idx) continue;
        LorentzVector pM = d_momenta->getMomentum(evt_abs, node.mother_idx);
        LorentzVector pD1 = d_momenta->getMomentum(evt_abs, node.daug1_idx);
        LorentzVector pD2 = d_momenta->getMomentum(evt_abs, node.daug2_idx);
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
                LorentzVector pM = d_momenta->getMomentum(evt_abs, d_decayNodes[ni].mother_idx);
                mm = pM.M();
                LorentzVector pD1 = d_momenta->getMomentum(evt_abs, d_decayNodes[ni].daug1_idx);
                LorentzVector pD2 = d_momenta->getMomentum(evt_abs, d_decayNodes[ni].daug2_idx);
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

// ============================================================
// Unified Hessian over global free-parameter dimension
// 3-pass scheme:
//   Pass A: compute S_re/S_im per (evt,p) from d_amp and d_v.
//   Pass B: per (block,res) launch: AD with local Nlocal slots → accumulate
//           dS_re/dS_im/d2S_re/d2S_im into per-event arrays indexed by GLOBAL slot.
//           Also stash dF_re/dF_im per block-event-sl for the mixed step.
//   Pass C: per-event, evaluate H_jk = w·(2·R2/I − G_j·G_k/I²) → atomicAdd.
//   Pass D (mixed): per (block,res) launch: use stashed dF and full S/dS
//           to write mixed Hessian d_mixed[2·n_ext × P_total].
// ============================================================

__global__ void computeSperEventKernel(
    double* d_S_re, double* d_S_im,                 // [nEvents * nPolar]
    const cuComplex* d_amp,                         // [nEvents * nPolar * n_ext]
    const cuComplex* d_v,                           // [n_ext]
    int nEvents, int nPolar, int n_ext)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    for (int p = 0; p < nPolar; ++p) {
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_ext; ++a) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
            cuComplex v_a = d_v[a];
            sre += (double)v_a.x * amp_ap.x - (double)v_a.y * amp_ap.y;
            sim += (double)v_a.x * amp_ap.y + (double)v_a.y * amp_ap.x;
        }
        d_S_re[evt * nPolar + p] = sre;
        d_S_im[evt * nPolar + p] = sim;
    }
}

// Accumulate dS/d2S in GLOBAL-Nfree indexing for this (block,res).
// Also write dF per (sl,j_global) for this block-event window into d_dF arrays.
template <int Nlocal>
__global__ void accumDSperEventKernel(
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const SL* d_slComb, const DeviceResonance* d_res,
    int res_idx_in_block, const double* d_all_params,
    const int* d_param_map, const int* d_global_idx,
    int nEvents, int nPolar, int decayChain_size, int nSLComb, double bf_d,
    int t_evt_offset,
    const thrust::complex<double>* d_slamps,
    const cuComplex* d_v, int site,
    double* d_dS_re, double* d_dS_im,            // [nEvents * nPolar * Nglobal]
    double* d_d2S_re, double* d_d2S_im,          // [nEvents * nPolar * Nglobal * Nglobal]
    double* d_dF_re, double* d_dF_im,            // [nEvents * nSLComb * Nglobal] for this block (per-block buffer)
    int Nglobal)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int global_evt = evt + t_evt_offset;
    int n_events_total = d_momenta->n_events;

    using AD = Var<double, Nlocal, true>;
    using CV = ComplexVar<double, Nlocal, true>;

    const DeviceResonance& target_res = d_res[res_idx_in_block];
    const double* target_rp = d_all_params + target_res.param_offset;

    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nlocal; ++j) {
        int pi = d_param_map[j];
        if (pi >= 0 && pi < 8) ftg[pi] = j;
    }

    AD m0_ad(target_rp[0]);
    if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;
    AD gamma_ad;
    if (target_res.type == ResModelType::BWR || target_res.type == ResModelType::BW) {
        gamma_ad = AD(target_rp[1]);
        if (ftg[1] >= 0) gamma_ad.grad[ftg[1]] = 1.0;
    }

    for (int sl_idx = 0; sl_idx < nSLComb; ++sl_idx) {
        CV R_ad(1.0, 0.0);
        for (int ni = 0; ni < decayChain_size; ++ni) {
            const DecayNode& node = d_decayNodes[ni];
            const SL& sl = d_slComb[sl_idx * decayChain_size + ni];
            int L = sl.L;

            LorentzVector pM  = d_momenta->getMomentum(global_evt, node.mother_idx);
            LorentzVector pD1 = d_momenta->getMomentum(global_evt, node.daug1_idx);
            LorentzVector pD2 = d_momenta->getMomentum(global_evt, node.daug2_idx);

            double mm = pM.M();
            double qq = breakup_momentum(mm, pD1.M(), pD2.M());
            AD q_ad(qq);
            double md1 = pD1.M(), md2 = pD2.M();

            AD m0_q0_ad, md1_q0_ad, md2_q0_ad;
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0)
                m0_q0_ad = m0_ad;
            else if (node.mass[0] > 0)
                m0_q0_ad = AD(node.mass[0]);
            else
                m0_q0_ad = AD(1.0);
            if (node.mass[1] <= 0 && node.daug1_idx == target_res.particle_idx)
                md1_q0_ad = m0_ad;
            else
                md1_q0_ad = AD(node.mass[1] > 0 ? node.mass[1] : md1);
            if (node.mass[2] <= 0 && node.daug2_idx == target_res.particle_idx)
                md2_q0_ad = m0_ad;
            else
                md2_q0_ad = AD(node.mass[2] > 0 ? node.mass[2] : md2);

            AD s_md = md1_q0_ad + md2_q0_ad;
            AD d_md = md1_q0_ad - md2_q0_ad;
            AD m0sq = m0_q0_ad * m0_q0_ad;
            AD q0sq = (m0sq - s_md*s_md)*(m0sq - d_md*d_md)/(AD(4.0)*m0sq);
            q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
            AD q0_ad = sqrt(q0sq);

            CV node_factor(1.0, 0.0);
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) {
                AD m_ad(mm);
                if (target_res.type == ResModelType::BWR)
                    node_factor = BWR<AD>(m_ad, m0_ad, gamma_ad, L, q_ad, q0_ad, bf_d);
                else if (target_res.type == ResModelType::BW)
                    node_factor = BW<AD>(m_ad, m0_ad, gamma_ad);
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = node_factor.real * bf_ad;
                node_factor.imag = node_factor.imag * bf_ad;
            } else {
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = bf_ad;
                node_factor.imag = AD(0.0);
            }

            CV new_R;
            new_R.real = R_ad.real * node_factor.real - R_ad.imag * node_factor.imag;
            new_R.imag = R_ad.real * node_factor.imag + R_ad.imag * node_factor.real;
            R_ad = new_R;
        }

        // stash dF for this block (for the mixed-hess step)
        for (int j = 0; j < Nlocal; ++j) {
            int jg = d_global_idx[j];
            int dF_idx = ((evt * nSLComb) + sl_idx) * Nglobal + jg;
            d_dF_re[dF_idx] = R_ad.real.grad[j];
            d_dF_im[dF_idx] = R_ad.imag.grad[j];
        }

        cuComplex v_sl = d_v[site + sl_idx];
        double vr = (double)v_sl.x, vi = (double)v_sl.y;

        for (int p = 0; p < nPolar; ++p) {
            int slamp_idx = sl_idx * n_events_total * nPolar + global_evt * nPolar + p;
            auto sl_amp = d_slamps[slamp_idx];
            double sl_re = sl_amp.real();
            double sl_im = sl_amp.imag();
            double c_re = vr * sl_re - vi * sl_im;
            double c_im = vr * sl_im + vi * sl_re;

            int base_p = (evt * nPolar + p);
            for (int j = 0; j < Nlocal; ++j) {
                int jg = d_global_idx[j];
                double dSre = c_re * R_ad.real.grad[j] - c_im * R_ad.imag.grad[j];
                double dSim = c_re * R_ad.imag.grad[j] + c_im * R_ad.real.grad[j];
                d_dS_re[base_p * Nglobal + jg] += dSre;
                d_dS_im[base_p * Nglobal + jg] += dSim;
                for (int k = 0; k < Nlocal; ++k) {
                    int kg = d_global_idx[k];
                    double d2re = c_re * R_ad.real.hess[j][k] - c_im * R_ad.imag.hess[j][k];
                    double d2im = c_re * R_ad.imag.hess[j][k] + c_im * R_ad.real.hess[j][k];
                    d_d2S_re[(base_p * Nglobal + jg) * Nglobal + kg] += d2re;
                    d_d2S_im[(base_p * Nglobal + jg) * Nglobal + kg] += d2im;
                }
            }
        }
    }
}

// Assemble per-event θθ Hessian contribution.
__global__ void assembleResHessKernel(
    const double* d_S_re, const double* d_S_im,
    const double* d_dS_re, const double* d_dS_im,
    const double* d_d2S_re, const double* d_d2S_im,
    double* d_hess, int hess_ld,
    const double* d_event_weights, double default_weight,
    int nEvents, int nPolar, int Nglobal)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    double w_evt = d_event_weights ? d_event_weights[evt] : default_weight;
    if (w_evt == 0.0) return;

    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) {
        double sr = d_S_re[evt * nPolar + p];
        double si = d_S_im[evt * nPolar + p];
        I_val += sr*sr + si*si;
    }
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;
    double inv_I2 = inv_I * inv_I;

    // G_j = 2·Σ_p (S_re·dS_re + S_im·dS_im)
    // We can't store G in registers (Nglobal could be up to ~8 here). Recompute per j.
    for (int j = 0; j < Nglobal; ++j) {
        double Gj = 0.0;
        for (int p = 0; p < nPolar; ++p) {
            double sr = d_S_re[evt * nPolar + p];
            double si = d_S_im[evt * nPolar + p];
            int idx = (evt * nPolar + p) * Nglobal + j;
            Gj += sr * d_dS_re[idx] + si * d_dS_im[idx];
        }
        Gj *= 2.0;
        for (int k = 0; k < Nglobal; ++k) {
            double Gk = 0.0;
            double R2 = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                double sr = d_S_re[evt * nPolar + p];
                double si = d_S_im[evt * nPolar + p];
                int idxj = (evt * nPolar + p) * Nglobal + j;
                int idxk = (evt * nPolar + p) * Nglobal + k;
                Gk += sr * d_dS_re[idxk] + si * d_dS_im[idxk];
                int idx2 = ((evt * nPolar + p) * Nglobal + j) * Nglobal + k;
                R2 += d_dS_re[idxk] * d_dS_re[idxj] + d_dS_im[idxk] * d_dS_im[idxj]
                    + sr * d_d2S_re[idx2] + si * d_d2S_im[idx2];
            }
            Gk *= 2.0;
            double term = w_evt * (2.0 * R2 * inv_I - Gj * Gk * inv_I2);
            atomicAdd(&d_hess[k * hess_ld + j], term);
        }
    }
}

// Mixed Hessian assembly using full S, full dS, and per-block dF.
// For each block: a ∈ [site, site+nSLComb); compute K, M; only that block's
// sl rows contribute the N correction.
__global__ void assembleMixedHessKernel(
    const double* d_S_re, const double* d_S_im,
    const double* d_dS_re, const double* d_dS_im,
    const double* d_dF_re, const double* d_dF_im,   // [nEvents * nSLComb * Nglobal] (this block)
    const cuComplex* d_v,
    const cuComplex* d_amp,                         // [nEvents * nPolar * n_ext]
    const thrust::complex<double>* d_slamps,
    int t_evt_offset,
    double* d_mixed, int P_total,
    const double* d_event_weights, double default_weight,
    int nEvents, int nPolar, int n_ext, int nSLComb, int site, int Nglobal,
    int n_events_total)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int global_evt = evt + t_evt_offset;
    double w_evt = d_event_weights ? d_event_weights[evt] : default_weight;
    if (w_evt == 0.0) return;

    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) {
        double sr = d_S_re[evt * nPolar + p];
        double si = d_S_im[evt * nPolar + p];
        I_val += sr*sr + si*si;
    }
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;
    double inv_I2 = inv_I * inv_I;

    // For each a in [site, site+nSLComb): write H[(Re or Im)v_a, θ_j]
    for (int sl_in_block = 0; sl_in_block < nSLComb; ++sl_in_block) {
        int a = site + sl_in_block;

        // K_a = Σ_p S_p · conj(amp_ap)  → real K_re, imag K_im
        double K_re = 0.0, K_im = 0.0;
        for (int p = 0; p < nPolar; ++p) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
            double ar = amp_ap.x, ai = amp_ap.y;
            double sr = d_S_re[evt * nPolar + p];
            double si = d_S_im[evt * nPolar + p];
            K_re += sr * ar + si * ai;
            K_im += sr * ai - si * ar;
        }
        double dI_xa = 2.0 * K_re;
        double dI_ya = -2.0 * K_im;

        for (int j = 0; j < Nglobal; ++j) {
            // G_j = dI/dθ_j
            double Gj = 0.0;
            // M_j = Σ_p dS_pj · conj(amp_ap)
            double M_re = 0.0, M_im = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
                double ar = amp_ap.x, ai = amp_ap.y;
                double sr = d_S_re[evt * nPolar + p];
                double si = d_S_im[evt * nPolar + p];
                int idx = (evt * nPolar + p) * Nglobal + j;
                double dSr = d_dS_re[idx], dSi = d_dS_im[idx];
                Gj += sr * dSr + si * dSi;
                M_re += dSr * ar + dSi * ai;
                M_im += dSi * ar - dSr * ai;
            }
            Gj *= 2.0;

            // N correction: only nonzero if (a corresponds to this block's sl)
            // dF for this (block,sl_in_block, j_global)
            int dF_idx = ((evt * nSLComb) + sl_in_block) * Nglobal + j;
            double dFr = d_dF_re[dF_idx];
            double dFi = d_dF_im[dF_idx];
            double N_re = 0.0, N_im = 0.0;
            if (dFr != 0.0 || dFi != 0.0) {
                for (int p = 0; p < nPolar; ++p) {
                    int slamp_idx = sl_in_block * n_events_total * nPolar + global_evt * nPolar + p;
                    auto sl_amp = d_slamps[slamp_idx];
                    double sl_re = sl_amp.real(), sl_im = sl_amp.imag();
                    double sr = d_S_re[evt * nPolar + p];
                    double si = d_S_im[evt * nPolar + p];
                    double cSs_re = sr * sl_re + si * sl_im;
                    double cSs_im = sr * sl_im - si * sl_re;
                    N_re += dFr * cSs_re - dFi * cSs_im;
                    N_im += dFr * cSs_im + dFi * cSs_re;
                }
            }
            double d2I_xa_th = 2.0 * M_re + 2.0 * N_re;
            double d2I_ya_th = 2.0 * M_im - 2.0 * N_im;
            double H_xa = w_evt * (d2I_xa_th * inv_I - dI_xa * Gj * inv_I2);
            double H_ya = w_evt * (d2I_ya_th * inv_I - dI_ya * Gj * inv_I2);
            atomicAdd(&d_mixed[a * P_total + j], H_xa);
            atomicAdd(&d_mixed[(n_ext + a) * P_total + j], H_ya);
        }
    }
}

// ---------------------------------------------------------------------------
// AmpCalc::computeResonanceHessian
// 累加 ∂² (Σ_e w_e log I_e) / ∂θ_j ∂θ_k 到 d_hess (P×P) — 调用方负责累加 data/bkg/phsp
// ---------------------------------------------------------------------------
void AmpCalc::computeResonanceHessian(
    const std::vector<int>& n_events,
    double* d_hess, int hess_ld,
    const std::vector<int>& t_offset,
    const std::vector<double*>& d_event_weights,
    double default_weight,
    const cuComplex* d_v,
    int n_ext,
    const std::vector<cuComplex*>& d_amp_batches)
{
    int n_gpu = static_cast<int>(n_events.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;
    bool has_offset = (t_offset.size() == (size_t)n_gpu);

    constexpr int kBlockSize = 256;
    const int Nglobal = n_free;

    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        int nEv = n_events[gpu];
        if (nEv <= 0) continue;
        int evt_off = has_offset ? t_offset[gpu] : 0;
        double* dw = (d_event_weights.size() == (size_t)n_gpu) ? d_event_weights[gpu] : nullptr;
        cuComplex* d_amp_gpu = (d_amp_batches.size() == (size_t)n_gpu) ? d_amp_batches[gpu] : nullptr;
        if (d_amp_gpu == nullptr) continue;

        int nPol = static_cast<int>(cas_list_[0]->getNPolarizations());
        int grid = (nEv + kBlockSize - 1) / kBlockSize;

        // Allocate per-event buffers (global Nfree dim)
        size_t n_S = (size_t)nEv * nPol;
        size_t n_dS = n_S * Nglobal;
        size_t n_d2S = n_dS * Nglobal;
        double *d_S_re, *d_S_im, *d_dS_re, *d_dS_im, *d_d2S_re, *d_d2S_im;
        cudaMalloc(&d_S_re, n_S * sizeof(double));
        cudaMalloc(&d_S_im, n_S * sizeof(double));
        cudaMalloc(&d_dS_re, n_dS * sizeof(double));
        cudaMalloc(&d_dS_im, n_dS * sizeof(double));
        cudaMalloc(&d_d2S_re, n_d2S * sizeof(double));
        cudaMalloc(&d_d2S_im, n_d2S * sizeof(double));
        cudaMemset(d_dS_re, 0, n_dS * sizeof(double));
        cudaMemset(d_dS_im, 0, n_dS * sizeof(double));
        cudaMemset(d_d2S_re, 0, n_d2S * sizeof(double));
        cudaMemset(d_d2S_im, 0, n_d2S * sizeof(double));

        // Pass A: compute S per (evt,p)
        computeSperEventKernel<<<grid, kBlockSize>>>(
            d_S_re, d_S_im, d_amp_gpu, d_v, nEv, nPol, n_ext);

        // Pass B: per (block,res), accumulate dS/d2S into global buffers.
        // (dF buffer is per-block-launch but we don't keep it here.)
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            auto& blk = blocks_[bi];
            auto& cas = cas_list_[blk.cas_idx];
            int nSLComb = static_cast<int>(cas->getNSLCombs());
            int nPol_b = static_cast<int>(cas->getNPolarizations());
            (void)nPol_b;

            for (int r = 0; r < blk.resonance_count; ++r) {
                std::vector<int> local_map, global_idx;
                for (int s = 0; s < n_free; ++s) {
                    if (slots_[s].block_idx == (int)bi && slots_[s].res_idx == r) {
                        local_map.push_back(slots_[s].param_idx);
                        global_idx.push_back(s);
                    }
                }
                int Nlocal = static_cast<int>(local_map.size());
                if (Nlocal == 0) continue;

                int *d_map, *d_gidx;
                cudaMalloc(&d_map, Nlocal * sizeof(int));
                cudaMalloc(&d_gidx, Nlocal * sizeof(int));
                cudaMemcpy(d_map, local_map.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_gidx, global_idx.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);

                // throw-away dF buffer for this (block,res)
                size_t n_dF = (size_t)nEv * nSLComb * Nglobal;
                double *d_dF_re, *d_dF_im;
                cudaMalloc(&d_dF_re, n_dF * sizeof(double));
                cudaMalloc(&d_dF_im, n_dF * sizeof(double));
                cudaMemset(d_dF_re, 0, n_dF * sizeof(double));
                cudaMemset(d_dF_im, 0, n_dF * sizeof(double));

                switch (Nlocal) {
                case 1:
                    accumDSperEventKernel<1><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_gidx,
                        nEv, nPol, cas->getDecayChainSize(), nSLComb, 3.0,
                        evt_off, cas->getSLAmps()[gpu], d_v, blk.site,
                        d_dS_re, d_dS_im, d_d2S_re, d_d2S_im,
                        d_dF_re, d_dF_im, Nglobal);
                    break;
                case 2:
                    accumDSperEventKernel<2><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_gidx,
                        nEv, nPol, cas->getDecayChainSize(), nSLComb, 3.0,
                        evt_off, cas->getSLAmps()[gpu], d_v, blk.site,
                        d_dS_re, d_dS_im, d_d2S_re, d_d2S_im,
                        d_dF_re, d_dF_im, Nglobal);
                    break;
                case 3:
                    accumDSperEventKernel<3><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_gidx,
                        nEv, nPol, cas->getDecayChainSize(), nSLComb, 3.0,
                        evt_off, cas->getSLAmps()[gpu], d_v, blk.site,
                        d_dS_re, d_dS_im, d_d2S_re, d_d2S_im,
                        d_dF_re, d_dF_im, Nglobal);
                    break;
                default: break;
                }
                cudaDeviceSynchronize();
                cudaFree(d_map);
                cudaFree(d_gidx);
                cudaFree(d_dF_re);
                cudaFree(d_dF_im);
            }
        }

        // Pass C: assemble Hessian using full S, full dS, full d²S
        assembleResHessKernel<<<grid, kBlockSize>>>(
            d_S_re, d_S_im, d_dS_re, d_dS_im, d_d2S_re, d_d2S_im,
            d_hess, hess_ld, dw, default_weight, nEv, nPol, Nglobal);
        cudaDeviceSynchronize();

        cudaFree(d_S_re); cudaFree(d_S_im);
        cudaFree(d_dS_re); cudaFree(d_dS_im);
        cudaFree(d_d2S_re); cudaFree(d_d2S_im);
    }
    cudaSetDevice(0);
}

// ============================================================
// 混合 Hessian kernel: ∂²(w·log I)/∂v_a∂θ_j
// 布局: d_mixed[a*P_total + j_global] = ∂²L/∂Re(v_a)∂θ_j  for a∈[0,n_ext)
//       d_mixed[(n_ext+a)*P_total + j_global] = ∂²L/∂Im(v_a)∂θ_j
// 即 PyTorch tensor [2*n_ext, P_total] 行主序
// ============================================================
template <int Nfree>
__global__ void resonanceMixedHessianKernel(
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const SL* d_slComb, const DeviceResonance* d_res,
    int res_idx_in_block, const double* d_all_params,
    const int* d_param_map, double* d_mixed, const int* d_global_idx,
    int nEvents, int nPolar, int decayChain_size, int nSLComb,
    double bf_d, int t_evt_offset,
    const thrust::complex<double>* d_slamps,
    const cuComplex* d_v, int site, int n_ext,
    const cuComplex* d_amp, int P_total,
    const double* d_event_weights, double default_weight)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int global_evt = evt + t_evt_offset;
    double w_evt = d_event_weights ? d_event_weights[evt] : default_weight;
    if (w_evt == 0.0) return;
    int n_events_total = d_momenta->n_events;

    using AD = Var<double, Nfree, false>;
    using CV = ComplexVar<double, Nfree, false>;

    const DeviceResonance& target_res = d_res[res_idx_in_block];
    const double* target_rp = d_all_params + target_res.param_offset;
    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nfree; ++j) {
        int pi = d_param_map[j];
        if (pi >= 0 && pi < 8) ftg[pi] = j;
    }
    AD m0_ad(target_rp[0]);
    if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;
    AD gamma_ad;
    if (target_res.type == ResModelType::BWR || target_res.type == ResModelType::BW) {
        gamma_ad = AD(target_rp[1]);
        if (ftg[1] >= 0) gamma_ad.grad[ftg[1]] = 1.0;
    }

    constexpr int MAX_POL = 32;
    constexpr int MAX_SL = 16;

    double S_re[MAX_POL], S_im[MAX_POL];
    double D_re[MAX_POL][Nfree], D_im[MAX_POL][Nfree];
    double dF_re[MAX_SL][Nfree], dF_im[MAX_SL][Nfree];

    for (int p = 0; p < nPolar; ++p) {
        S_re[p] = 0.0; S_im[p] = 0.0;
        for (int j = 0; j < Nfree; ++j) { D_re[p][j] = 0.0; D_im[p][j] = 0.0; }
    }

    // S_p = Σ_a v_a · d_amp[a,e,p]
    for (int p = 0; p < nPolar; ++p) {
        for (int a = 0; a < n_ext; ++a) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
            cuComplex v_a = d_v[a];
            S_re[p] += (double)v_a.x * amp_ap.x - (double)v_a.y * amp_ap.y;
            S_im[p] += (double)v_a.x * amp_ap.y + (double)v_a.y * amp_ap.x;
        }
    }

    // AD pass over block's sl: compute ∂F_sl/∂θ_j, accumulate D_j(p)
    for (int sl_idx = 0; sl_idx < nSLComb; ++sl_idx) {
        CV R_ad(1.0, 0.0);
        for (int ni = 0; ni < decayChain_size; ++ni) {
            const DecayNode& node = d_decayNodes[ni];
            const SL& sl = d_slComb[sl_idx * decayChain_size + ni];
            int L = sl.L;
            LorentzVector pM  = d_momenta->getMomentum(global_evt, node.mother_idx);
            LorentzVector pD1 = d_momenta->getMomentum(global_evt, node.daug1_idx);
            LorentzVector pD2 = d_momenta->getMomentum(global_evt, node.daug2_idx);
            double mm = pM.M();
            double qq = breakup_momentum(mm, pD1.M(), pD2.M());
            AD q_ad(qq);
            double md1 = pD1.M(), md2 = pD2.M();
            AD m0_q0_ad, md1_q0_ad, md2_q0_ad;
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) m0_q0_ad = m0_ad;
            else if (node.mass[0] > 0) m0_q0_ad = AD(node.mass[0]);
            else m0_q0_ad = AD(1.0);
            if (node.mass[1] <= 0 && node.daug1_idx == target_res.particle_idx) md1_q0_ad = m0_ad;
            else md1_q0_ad = AD(node.mass[1] > 0 ? node.mass[1] : md1);
            if (node.mass[2] <= 0 && node.daug2_idx == target_res.particle_idx) md2_q0_ad = m0_ad;
            else md2_q0_ad = AD(node.mass[2] > 0 ? node.mass[2] : md2);
            AD s_md = md1_q0_ad + md2_q0_ad;
            AD d_md = md1_q0_ad - md2_q0_ad;
            AD m0sq = m0_q0_ad * m0_q0_ad;
            AD q0sq = (m0sq - s_md*s_md)*(m0sq - d_md*d_md)/(AD(4.0)*m0sq);
            q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
            AD q0_ad = sqrt(q0sq);
            CV node_factor(1.0, 0.0);
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) {
                AD m_ad(mm);
                if (target_res.type == ResModelType::BWR)
                    node_factor = BWR<AD>(m_ad, m0_ad, gamma_ad, L, q_ad, q0_ad, bf_d);
                else if (target_res.type == ResModelType::BW)
                    node_factor = BW<AD>(m_ad, m0_ad, gamma_ad);
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = node_factor.real * bf_ad;
                node_factor.imag = node_factor.imag * bf_ad;
            } else {
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = bf_ad;
                node_factor.imag = AD(0.0);
            }
            CV new_R;
            new_R.real = R_ad.real * node_factor.real - R_ad.imag * node_factor.imag;
            new_R.imag = R_ad.real * node_factor.imag + R_ad.imag * node_factor.real;
            R_ad = new_R;
        }
        for (int j = 0; j < Nfree; ++j) {
            dF_re[sl_idx][j] = R_ad.real.grad[j];
            dF_im[sl_idx][j] = R_ad.imag.grad[j];
        }
        cuComplex v_sl = d_v[site + sl_idx];
        double vr = (double)v_sl.x, vi = (double)v_sl.y;
        for (int p = 0; p < nPolar; ++p) {
            int slamp_idx = sl_idx * n_events_total * nPolar + global_evt * nPolar + p;
            auto slamp = d_slamps[slamp_idx];
            double a_re = slamp.real(), a_im = slamp.imag();
            for (int j = 0; j < Nfree; ++j) {
                double dF_a_re = R_ad.real.grad[j] * a_re - R_ad.imag.grad[j] * a_im;
                double dF_a_im = R_ad.real.grad[j] * a_im + R_ad.imag.grad[j] * a_re;
                D_re[p][j] += vr * dF_a_re - vi * dF_a_im;
                D_im[p][j] += vr * dF_a_im + vi * dF_a_re;
            }
        }
    }

    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) I_val += S_re[p]*S_re[p] + S_im[p]*S_im[p];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;
    double inv_I2 = inv_I * inv_I;

    double G_theta[8] = {0};
    for (int j = 0; j < Nfree; ++j) {
        double g = 0.0;
        for (int p = 0; p < nPolar; ++p) g += S_re[p]*D_re[p][j] + S_im[p]*D_im[p][j];
        G_theta[j] = 2.0 * g;
    }

    for (int a = 0; a < n_ext; ++a) {
        double K_re = 0.0, K_im = 0.0;
        double M_re[8] = {0}, M_im[8] = {0};
        for (int p = 0; p < nPolar; ++p) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
            double a_re = (double)amp_ap.x, a_im = (double)amp_ap.y;
            K_re += S_re[p] * a_re + S_im[p] * a_im;
            K_im += S_re[p] * a_im - S_im[p] * a_re;
            for (int j = 0; j < Nfree; ++j) {
                M_re[j] += D_re[p][j] * a_re + D_im[p][j] * a_im;
                M_im[j] += D_im[p][j] * a_re - D_re[p][j] * a_im;
            }
        }
        double N_re[8] = {0}, N_im[8] = {0};
        int sl_in_block = a - site;
        if (sl_in_block >= 0 && sl_in_block < nSLComb) {
            for (int p = 0; p < nPolar; ++p) {
                int slamp_idx = sl_in_block * n_events_total * nPolar + global_evt * nPolar + p;
                auto slamp = d_slamps[slamp_idx];
                double a_re = slamp.real(), a_im = slamp.imag();
                double cSs_re = S_re[p] * a_re + S_im[p] * a_im;
                double cSs_im = S_re[p] * a_im - S_im[p] * a_re;
                for (int j = 0; j < Nfree; ++j) {
                    double dFr = dF_re[sl_in_block][j];
                    double dFi = dF_im[sl_in_block][j];
                    N_re[j] += dFr * cSs_re - dFi * cSs_im;
                    N_im[j] += dFr * cSs_im + dFi * cSs_re;
                }
            }
        }
        double dI_xa = 2.0 * K_re;
        double dI_ya = -2.0 * K_im;
        for (int j = 0; j < Nfree; ++j) {
            double dI_th = G_theta[j];
            double d2I_xa_th = 2.0 * M_re[j] + 2.0 * N_re[j];
            double d2I_ya_th = 2.0 * M_im[j] - 2.0 * N_im[j];
            double H_xa = w_evt * (d2I_xa_th * inv_I - dI_xa * dI_th * inv_I2);
            double H_ya = w_evt * (d2I_ya_th * inv_I - dI_ya * dI_th * inv_I2);
            int j_global = d_global_idx[j];
            atomicAdd(&d_mixed[a * P_total + j_global], H_xa);
            atomicAdd(&d_mixed[(n_ext + a) * P_total + j_global], H_ya);
        }
    }
}

// ---------------------------------------------------------------------------
// AmpCalc::computeMixedHessian
// 累加 ∂²(Σ_e w_e log I_e) / ∂v_a ∂θ_j 到 d_mixed [2*n_ext × P_total] 行主序
// ---------------------------------------------------------------------------
void AmpCalc::computeMixedHessian(
    const std::vector<int>& n_events,
    int n_ext,
    double* d_mixed, int P_total,
    const std::vector<int>& t_offset,
    const std::vector<double*>& d_event_weights,
    double default_weight,
    const cuComplex* d_v,
    const std::vector<cuComplex*>& d_amp_batches)
{
    int n_gpu = static_cast<int>(n_events.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;
    bool has_offset = (t_offset.size() == (size_t)n_gpu);

    constexpr int kBlockSize = 256;

    int mixed_total = 2 * n_ext * P_total;
    std::vector<double*> d_mixed_per_gpu(n_gpu, nullptr);
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaMalloc(&d_mixed_per_gpu[gpu], mixed_total * sizeof(double));
        cudaMemset(d_mixed_per_gpu[gpu], 0, mixed_total * sizeof(double));
    }

    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        auto& blk = blocks_[bi];
        auto& cas = cas_list_[blk.cas_idx];

        for (int r = 0; r < blk.resonance_count; ++r) {
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
                int nEv = n_events[gpu];
                if (nEv <= 0) continue;

                int *d_map, *d_gidx;
                cudaMalloc(&d_map, Nlocal * sizeof(int));
                cudaMalloc(&d_gidx, Nlocal * sizeof(int));
                cudaMemcpy(d_map, local_map.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_gidx, global_idx.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);

                int nPol = static_cast<int>(cas->getNPolarizations());
                int grid = (nEv + kBlockSize - 1) / kBlockSize;
                int evt_off = has_offset ? t_offset[gpu] : 0;
                double* dw = (d_event_weights.size() == (size_t)n_gpu) ? d_event_weights[gpu] : nullptr;
                cuComplex* d_amp_batch = d_amp_batches[gpu];

                switch (Nlocal) {
                case 1:
                    resonanceMixedHessianKernel<1><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_mixed_per_gpu[gpu], d_gidx,
                        nEv, nPol, cas->getDecayChainSize(),
                        static_cast<int>(cas->getNSLCombs()), 3.0, evt_off,
                        cas->getSLAmps()[gpu], d_v, blk.site, n_ext, d_amp_batch, P_total,
                        dw, default_weight);
                    break;
                case 2:
                    resonanceMixedHessianKernel<2><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_mixed_per_gpu[gpu], d_gidx,
                        nEv, nPol, cas->getDecayChainSize(),
                        static_cast<int>(cas->getNSLCombs()), 3.0, evt_off,
                        cas->getSLAmps()[gpu], d_v, blk.site, n_ext, d_amp_batch, P_total,
                        dw, default_weight);
                    break;
                case 3:
                    resonanceMixedHessianKernel<3><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_mixed_per_gpu[gpu], d_gidx,
                        nEv, nPol, cas->getDecayChainSize(),
                        static_cast<int>(cas->getNSLCombs()), 3.0, evt_off,
                        cas->getSLAmps()[gpu], d_v, blk.site, n_ext, d_amp_batch, P_total,
                        dw, default_weight);
                    break;
                default: break;
                }
                cudaDeviceSynchronize();
                cudaFree(d_map);
                cudaFree(d_gidx);
            }
        }
    }

    cudaSetDevice(0);
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == 0) {
            int grid = (mixed_total + 255) / 256;
            daxpy_kernel<<<grid, 256>>>(d_mixed, d_mixed_per_gpu[0], 1.0, mixed_total);
            cudaDeviceSynchronize();
        } else {
            std::vector<double> h_temp(mixed_total);
            cudaSetDevice(gpu);
            cudaMemcpy(h_temp.data(), d_mixed_per_gpu[gpu],
                       mixed_total * sizeof(double), cudaMemcpyDeviceToHost);
            cudaSetDevice(0);
            double* d_temp;
            cudaMalloc(&d_temp, mixed_total * sizeof(double));
            cudaMemcpy(d_temp, h_temp.data(), mixed_total * sizeof(double), cudaMemcpyHostToDevice);
            int grid = (mixed_total + 255) / 256;
            daxpy_kernel<<<grid, 256>>>(d_mixed, d_temp, 1.0, mixed_total);
            cudaDeviceSynchronize();
            cudaFree(d_temp);
        }
    }
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaFree(d_mixed_per_gpu[gpu]);
    }
    cudaSetDevice(0);
}

// ============================================================
// phsp 贡献 kernel: 累加 per-event ∂I/∂θ, ∂²I/∂θ∂θ, ∂I/∂v, ∂²I/∂v∂θ, 以及 I_e
// 对于 phsp: L_phsp = c · log(F),  F = (1/N) Σ I_e
//   ∂²L/∂θ_j∂θ_k = c · (B_jk/S - A_j·A_k/S²)
//   ∂²L/∂v_a∂θ_j = c · (B_aj/S - A_va·A_thj/S²)
// ============================================================
template <int Nfree>
__global__ void phspContributionSumsKernel(
    const DeviceMomenta* d_momenta, const DecayNode* d_decayNodes,
    const SL* d_slComb, const DeviceResonance* d_res,
    int res_idx_in_block, const double* d_all_params,
    const int* d_param_map, const int* d_global_idx,
    int nEvents, int nPolar, int decayChain_size, int nSLComb,
    double bf_d, int t_evt_offset,
    const thrust::complex<double>* d_slamps,
    const cuComplex* d_v, int site, int n_ext,
    const cuComplex* d_amp, int P_total,
    double* d_S, double* d_A_th, double* d_B_thth,
    double* d_A_v, double* d_B_vth)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int global_evt = evt + t_evt_offset;
    int n_events_total = d_momenta->n_events;

    using AD = Var<double, Nfree, true>;
    using CV = ComplexVar<double, Nfree, true>;

    const DeviceResonance& target_res = d_res[res_idx_in_block];
    const double* target_rp = d_all_params + target_res.param_offset;
    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nfree; ++j) {
        int pi = d_param_map[j];
        if (pi >= 0 && pi < 8) ftg[pi] = j;
    }
    AD m0_ad(target_rp[0]);
    if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;
    AD gamma_ad;
    if (target_res.type == ResModelType::BWR || target_res.type == ResModelType::BW) {
        gamma_ad = AD(target_rp[1]);
        if (ftg[1] >= 0) gamma_ad.grad[ftg[1]] = 1.0;
    }

    constexpr int MAX_POL = 32;
    constexpr int MAX_SL = 16;

    double S_re[MAX_POL], S_im[MAX_POL];
    double dS_re[MAX_POL][Nfree], dS_im[MAX_POL][Nfree];
    double d2S_re[MAX_POL][Nfree][Nfree], d2S_im[MAX_POL][Nfree][Nfree];
    double dF_re_arr[MAX_SL][Nfree], dF_im_arr[MAX_SL][Nfree];
    for (int p = 0; p < nPolar; ++p) {
        S_re[p] = 0.0; S_im[p] = 0.0;
        for (int j = 0; j < Nfree; ++j) {
            dS_re[p][j] = 0.0; dS_im[p][j] = 0.0;
            for (int k = 0; k < Nfree; ++k) {
                d2S_re[p][j][k] = 0.0; d2S_im[p][j][k] = 0.0;
            }
        }
    }

    // S_p from full d_amp · d_v
    for (int p = 0; p < nPolar; ++p) {
        for (int a = 0; a < n_ext; ++a) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
            cuComplex v_a = d_v[a];
            S_re[p] += (double)v_a.x * amp_ap.x - (double)v_a.y * amp_ap.y;
            S_im[p] += (double)v_a.x * amp_ap.y + (double)v_a.y * amp_ap.x;
        }
    }

    // 对 block 中每个 sl: 计算 ∂F/∂θ 和 ∂²F/∂θ∂θ，累加 dS, d2S
    for (int sl_idx = 0; sl_idx < nSLComb; ++sl_idx) {
        CV R_ad(1.0, 0.0);
        for (int ni = 0; ni < decayChain_size; ++ni) {
            const DecayNode& node = d_decayNodes[ni];
            const SL& sl = d_slComb[sl_idx * decayChain_size + ni];
            int L = sl.L;
            LorentzVector pM  = d_momenta->getMomentum(global_evt, node.mother_idx);
            LorentzVector pD1 = d_momenta->getMomentum(global_evt, node.daug1_idx);
            LorentzVector pD2 = d_momenta->getMomentum(global_evt, node.daug2_idx);
            double mm = pM.M();
            double qq = breakup_momentum(mm, pD1.M(), pD2.M());
            AD q_ad(qq);
            double md1 = pD1.M(), md2 = pD2.M();
            AD m0_q0_ad, md1_q0_ad, md2_q0_ad;
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) m0_q0_ad = m0_ad;
            else if (node.mass[0] > 0) m0_q0_ad = AD(node.mass[0]);
            else m0_q0_ad = AD(1.0);
            if (node.mass[1] <= 0 && node.daug1_idx == target_res.particle_idx) md1_q0_ad = m0_ad;
            else md1_q0_ad = AD(node.mass[1] > 0 ? node.mass[1] : md1);
            if (node.mass[2] <= 0 && node.daug2_idx == target_res.particle_idx) md2_q0_ad = m0_ad;
            else md2_q0_ad = AD(node.mass[2] > 0 ? node.mass[2] : md2);
            AD s_md = md1_q0_ad + md2_q0_ad;
            AD d_md = md1_q0_ad - md2_q0_ad;
            AD m0sq = m0_q0_ad * m0_q0_ad;
            AD q0sq = (m0sq - s_md*s_md)*(m0sq - d_md*d_md)/(AD(4.0)*m0sq);
            q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
            AD q0_ad = sqrt(q0sq);
            CV node_factor(1.0, 0.0);
            if (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0) {
                AD m_ad(mm);
                if (target_res.type == ResModelType::BWR)
                    node_factor = BWR<AD>(m_ad, m0_ad, gamma_ad, L, q_ad, q0_ad, bf_d);
                else if (target_res.type == ResModelType::BW)
                    node_factor = BW<AD>(m_ad, m0_ad, gamma_ad);
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = node_factor.real * bf_ad;
                node_factor.imag = node_factor.imag * bf_ad;
            } else {
                AD bf_ad = Bf<AD>(L, q_ad, q0_ad, bf_d);
                node_factor.real = bf_ad;
                node_factor.imag = AD(0.0);
            }
            CV new_R;
            new_R.real = R_ad.real * node_factor.real - R_ad.imag * node_factor.imag;
            new_R.imag = R_ad.real * node_factor.imag + R_ad.imag * node_factor.real;
            R_ad = new_R;
        }
        for (int j = 0; j < Nfree; ++j) {
            dF_re_arr[sl_idx][j] = R_ad.real.grad[j];
            dF_im_arr[sl_idx][j] = R_ad.imag.grad[j];
        }
        cuComplex v_sl = d_v[site + sl_idx];
        double vr = (double)v_sl.x, vi = (double)v_sl.y;
        for (int p = 0; p < nPolar; ++p) {
            int slamp_idx = sl_idx * n_events_total * nPolar + global_evt * nPolar + p;
            auto slamp = d_slamps[slamp_idx];
            double a_re = slamp.real(), a_im = slamp.imag();
            double c_re = vr * a_re - vi * a_im;
            double c_im = vr * a_im + vi * a_re;
            for (int j = 0; j < Nfree; ++j) {
                dS_re[p][j] += c_re * R_ad.real.grad[j] - c_im * R_ad.imag.grad[j];
                dS_im[p][j] += c_re * R_ad.imag.grad[j] + c_im * R_ad.real.grad[j];
                for (int k = 0; k < Nfree; ++k) {
                    d2S_re[p][j][k] += c_re * R_ad.real.hess[j][k] - c_im * R_ad.imag.hess[j][k];
                    d2S_im[p][j][k] += c_re * R_ad.imag.hess[j][k] + c_im * R_ad.real.hess[j][k];
                }
            }
        }
    }

    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) I_val += S_re[p]*S_re[p] + S_im[p]*S_im[p];
    if (I_val < 1e-30) return;

    atomicAdd(d_S, I_val);

    // A_th[j] = Σ_e ∂I/∂θ_j = Σ_e 2·Σ_p (S_re·dS_re + S_im·dS_im)[j]
    for (int j = 0; j < Nfree; ++j) {
        double dI_j = 0.0;
        for (int p = 0; p < nPolar; ++p) dI_j += S_re[p]*dS_re[p][j] + S_im[p]*dS_im[p][j];
        dI_j *= 2.0;
        atomicAdd(&d_A_th[d_global_idx[j]], dI_j);
        for (int k = 0; k < Nfree; ++k) {
            double d2I = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                d2I += dS_re[p][k]*dS_re[p][j] + dS_im[p][k]*dS_im[p][j]
                     + S_re[p]*d2S_re[p][j][k] + S_im[p]*d2S_im[p][j][k];
            }
            d2I *= 2.0;
            atomicAdd(&d_B_thth[d_global_idx[j]*P_total + d_global_idx[k]], d2I);
        }
    }

    // 累加 v_a 相关 (A_v 和 B_vth)
    for (int a = 0; a < n_ext; ++a) {
        double K_re = 0.0, K_im = 0.0;
        double M_re[8] = {0}, M_im[8] = {0};
        for (int p = 0; p < nPolar; ++p) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_ext + p * n_ext + a];
            double a_re = (double)amp_ap.x, a_im = (double)amp_ap.y;
            K_re += S_re[p] * a_re + S_im[p] * a_im;
            K_im += S_re[p] * a_im - S_im[p] * a_re;
            for (int j = 0; j < Nfree; ++j) {
                M_re[j] += dS_re[p][j] * a_re + dS_im[p][j] * a_im;
                M_im[j] += dS_im[p][j] * a_re - dS_re[p][j] * a_im;
            }
        }
        double N_re[8] = {0}, N_im[8] = {0};
        int sl_in_block = a - site;
        if (sl_in_block >= 0 && sl_in_block < nSLComb) {
            for (int p = 0; p < nPolar; ++p) {
                int slamp_idx = sl_in_block * n_events_total * nPolar + global_evt * nPolar + p;
                auto slamp = d_slamps[slamp_idx];
                double a_re = slamp.real(), a_im = slamp.imag();
                double cSs_re = S_re[p] * a_re + S_im[p] * a_im;
                double cSs_im = S_re[p] * a_im - S_im[p] * a_re;
                for (int j = 0; j < Nfree; ++j) {
                    double dFr = dF_re_arr[sl_in_block][j];
                    double dFi = dF_im_arr[sl_in_block][j];
                    N_re[j] += dFr * cSs_re - dFi * cSs_im;
                    N_im[j] += dFr * cSs_im + dFi * cSs_re;
                }
            }
        }
        double dI_xa = 2.0 * K_re;
        double dI_ya = -2.0 * K_im;
        atomicAdd(&d_A_v[a], dI_xa);
        atomicAdd(&d_A_v[n_ext + a], dI_ya);
        for (int j = 0; j < Nfree; ++j) {
            double d2I_xa_th = 2.0 * M_re[j] + 2.0 * N_re[j];
            double d2I_ya_th = 2.0 * M_im[j] - 2.0 * N_im[j];
            int j_global = d_global_idx[j];
            atomicAdd(&d_B_vth[a * P_total + j_global], d2I_xa_th);
            atomicAdd(&d_B_vth[(n_ext + a) * P_total + j_global], d2I_ya_th);
        }
    }
}

// phsp 最终组装: d_hess_th[j,k] += c·(B/S - A·A/S²); d_mixed[a,j] += c·(B_vth/S - A_v·A_th/S²)
__global__ void phspFinalizeKernel(
    double* d_hess_th, int P_total,
    double* d_mixed, int n_ext,
    const double* d_A_th, const double* d_B_thth,
    const double* d_A_v, const double* d_B_vth,
    double c_over_S, double c_over_S2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_th = P_total * P_total;
    int total_v = 2 * n_ext * P_total;
    if (idx < total_th) {
        int j = idx / P_total;
        int k = idx % P_total;
        double add = c_over_S * d_B_thth[idx] - c_over_S2 * d_A_th[j] * d_A_th[k];
        d_hess_th[idx] += add;
    } else if (idx < total_th + total_v) {
        int i = idx - total_th;
        int a = i / P_total;
        int j = i % P_total;
        double add = c_over_S * d_B_vth[i] - c_over_S2 * d_A_v[a] * d_A_th[j];
        d_mixed[i] += add;
    }
}

void AmpCalc::computePhspContribution(
    const std::vector<int>& n_phsp_events,
    double c,
    int n_ext,
    double* d_hess_th, int P_total,
    double* d_mixed,
    const std::vector<int>& t_offset,
    const cuComplex* d_v,
    const std::vector<cuComplex*>& d_amp_phsp)
{
    int n_gpu = static_cast<int>(n_phsp_events.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty() || P_total == 0) return;
    bool has_offset = (t_offset.size() == (size_t)n_gpu);

    constexpr int kBlockSize = 256;

    std::vector<double*> d_S_per_gpu(n_gpu, nullptr);
    std::vector<double*> d_A_th_per_gpu(n_gpu, nullptr);
    std::vector<double*> d_B_thth_per_gpu(n_gpu, nullptr);
    std::vector<double*> d_A_v_per_gpu(n_gpu, nullptr);
    std::vector<double*> d_B_vth_per_gpu(n_gpu, nullptr);

    int sz_B_thth = P_total * P_total;
    int sz_A_v = 2 * n_ext;
    int sz_B_vth = 2 * n_ext * P_total;

    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaMalloc(&d_S_per_gpu[gpu], sizeof(double));
        cudaMemset(d_S_per_gpu[gpu], 0, sizeof(double));
        cudaMalloc(&d_A_th_per_gpu[gpu], P_total * sizeof(double));
        cudaMemset(d_A_th_per_gpu[gpu], 0, P_total * sizeof(double));
        cudaMalloc(&d_B_thth_per_gpu[gpu], sz_B_thth * sizeof(double));
        cudaMemset(d_B_thth_per_gpu[gpu], 0, sz_B_thth * sizeof(double));
        cudaMalloc(&d_A_v_per_gpu[gpu], sz_A_v * sizeof(double));
        cudaMemset(d_A_v_per_gpu[gpu], 0, sz_A_v * sizeof(double));
        cudaMalloc(&d_B_vth_per_gpu[gpu], sz_B_vth * sizeof(double));
        cudaMemset(d_B_vth_per_gpu[gpu], 0, sz_B_vth * sizeof(double));
    }

    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        auto& blk = blocks_[bi];
        auto& cas = cas_list_[blk.cas_idx];
        for (int r = 0; r < blk.resonance_count; ++r) {
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
                int nEv = n_phsp_events[gpu];
                if (nEv <= 0) continue;
                int *d_map, *d_gidx;
                cudaMalloc(&d_map, Nlocal * sizeof(int));
                cudaMalloc(&d_gidx, Nlocal * sizeof(int));
                cudaMemcpy(d_map, local_map.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_gidx, global_idx.data(), Nlocal * sizeof(int), cudaMemcpyHostToDevice);

                int nPol = static_cast<int>(cas->getNPolarizations());
                int grid = (nEv + kBlockSize - 1) / kBlockSize;
                int evt_off = has_offset ? t_offset[gpu] : 0;

                switch (Nlocal) {
                case 1:
                    phspContributionSumsKernel<1><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_gidx,
                        nEv, nPol, cas->getDecayChainSize(),
                        static_cast<int>(cas->getNSLCombs()), 3.0, evt_off,
                        cas->getSLAmps()[gpu], d_v, blk.site, n_ext,
                        d_amp_phsp[gpu], P_total,
                        d_S_per_gpu[gpu], d_A_th_per_gpu[gpu], d_B_thth_per_gpu[gpu],
                        d_A_v_per_gpu[gpu], d_B_vth_per_gpu[gpu]);
                    break;
                case 2:
                    phspContributionSumsKernel<2><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_gidx,
                        nEv, nPol, cas->getDecayChainSize(),
                        static_cast<int>(cas->getNSLCombs()), 3.0, evt_off,
                        cas->getSLAmps()[gpu], d_v, blk.site, n_ext,
                        d_amp_phsp[gpu], P_total,
                        d_S_per_gpu[gpu], d_A_th_per_gpu[gpu], d_B_thth_per_gpu[gpu],
                        d_A_v_per_gpu[gpu], d_B_vth_per_gpu[gpu]);
                    break;
                case 3:
                    phspContributionSumsKernel<3><<<grid, kBlockSize>>>(
                        cas->getMomenta()[gpu], cas->getDecayNodes()[gpu],
                        cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                        r, blk.d_all_params[gpu], d_map, d_gidx,
                        nEv, nPol, cas->getDecayChainSize(),
                        static_cast<int>(cas->getNSLCombs()), 3.0, evt_off,
                        cas->getSLAmps()[gpu], d_v, blk.site, n_ext,
                        d_amp_phsp[gpu], P_total,
                        d_S_per_gpu[gpu], d_A_th_per_gpu[gpu], d_B_thth_per_gpu[gpu],
                        d_A_v_per_gpu[gpu], d_B_vth_per_gpu[gpu]);
                    break;
                default: break;
                }
                cudaDeviceSynchronize();
                cudaFree(d_map);
                cudaFree(d_gidx);
            }
        }
    }

    // 跨 GPU 累加到 GPU 0
    cudaSetDevice(0);
    double* d_S_tot;     cudaMalloc(&d_S_tot, sizeof(double));     cudaMemset(d_S_tot, 0, sizeof(double));
    double* d_A_th_tot;  cudaMalloc(&d_A_th_tot, P_total * sizeof(double));  cudaMemset(d_A_th_tot, 0, P_total * sizeof(double));
    double* d_B_thth_tot; cudaMalloc(&d_B_thth_tot, sz_B_thth * sizeof(double)); cudaMemset(d_B_thth_tot, 0, sz_B_thth * sizeof(double));
    double* d_A_v_tot;   cudaMalloc(&d_A_v_tot, sz_A_v * sizeof(double));    cudaMemset(d_A_v_tot, 0, sz_A_v * sizeof(double));
    double* d_B_vth_tot; cudaMalloc(&d_B_vth_tot, sz_B_vth * sizeof(double)); cudaMemset(d_B_vth_tot, 0, sz_B_vth * sizeof(double));

    auto reduce_to_primary = [&](double** d_per, double* d_tot, int n) {
        for (int gpu = 0; gpu < n_gpu; ++gpu) {
            if (gpu == 0) {
                int grid = (n + 255) / 256;
                daxpy_kernel<<<grid, 256>>>(d_tot, d_per[gpu], 1.0, n);
                cudaDeviceSynchronize();
            } else {
                std::vector<double> h(n);
                cudaSetDevice(gpu);
                cudaMemcpy(h.data(), d_per[gpu], n * sizeof(double), cudaMemcpyDeviceToHost);
                cudaSetDevice(0);
                double* d_tmp; cudaMalloc(&d_tmp, n * sizeof(double));
                cudaMemcpy(d_tmp, h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
                int grid = (n + 255) / 256;
                daxpy_kernel<<<grid, 256>>>(d_tot, d_tmp, 1.0, n);
                cudaDeviceSynchronize();
                cudaFree(d_tmp);
            }
        }
    };
    reduce_to_primary(d_S_per_gpu.data(), d_S_tot, 1);
    reduce_to_primary(d_A_th_per_gpu.data(), d_A_th_tot, P_total);
    reduce_to_primary(d_B_thth_per_gpu.data(), d_B_thth_tot, sz_B_thth);
    reduce_to_primary(d_A_v_per_gpu.data(), d_A_v_tot, sz_A_v);
    reduce_to_primary(d_B_vth_per_gpu.data(), d_B_vth_tot, sz_B_vth);

    // 读取 S
    double S_host = 0.0;
    cudaMemcpy(&S_host, d_S_tot, sizeof(double), cudaMemcpyDeviceToHost);
    if (S_host > 1e-30) {
        double c_over_S = c / S_host;
        double c_over_S2 = c / (S_host * S_host);
        int total = P_total * P_total + sz_B_vth;
        int grid = (total + 255) / 256;
        phspFinalizeKernel<<<grid, 256>>>(
            d_hess_th, P_total, d_mixed, n_ext,
            d_A_th_tot, d_B_thth_tot, d_A_v_tot, d_B_vth_tot,
            c_over_S, c_over_S2);
        cudaDeviceSynchronize();
    }

    cudaFree(d_S_tot); cudaFree(d_A_th_tot); cudaFree(d_B_thth_tot);
    cudaFree(d_A_v_tot); cudaFree(d_B_vth_tot);
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaFree(d_S_per_gpu[gpu]);
        cudaFree(d_A_th_per_gpu[gpu]);
        cudaFree(d_B_thth_per_gpu[gpu]);
        cudaFree(d_A_v_per_gpu[gpu]);
        cudaFree(d_B_vth_per_gpu[gpu]);
    }
    cudaSetDevice(0);
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
