#include <AmpGen.cuh>
#include <ComputeHessian.cuh>
#include <HessianStage1Kernel.cuh>
#include <CustomExpr.cuh>
#include <SymbolicDiff.cuh>  // Q0MassDep + buildModelAST（addBlock 构建符号微分 aux）
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
    for (size_t i = 0; i < d_slamp_tab_.size(); ++i)
    {
        cudaSetDevice(static_cast<int>(i));
        if (d_slamp_tab_[i])
            cudaFree(d_slamp_tab_[i]);
        if (d_sign_tab_[i])
            cudaFree(d_sign_tab_[i]);
        if (d_mom_tab_[i])
            cudaFree(d_mom_tab_[i]);
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
    // σ≥1 的重建动量（σ=0 即 d_momenta_，上面已释放）
    for (size_t m = 1; m < d_mom_sigma_.size(); ++m)
    {
        for (size_t i = 0; i < d_mom_sigma_[m].size(); ++i)
        {
            cudaSetDevice(static_cast<int>(i));
            if (d_mom_sigma_[m][i])
            {
                DeviceMomenta h;
                cudaMemcpy(&h, d_mom_sigma_[m][i], sizeof(DeviceMomenta),
                           cudaMemcpyDeviceToHost);
                cudaFree(h.momenta);
                cudaFree(d_mom_sigma_[m][i]);
            }
        }
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

void AmpCasDecay::setIdenticalGroups(const std::vector<std::pair<std::vector<std::string>, bool>>& groups)
{
    identical_groups_ = groups;
}

// ============================================================================
// 全同粒子置换拓扑生成（coset of S_n / 稳定子）
//
// 对应文章中 "permutation amplitudes" PA = Σ_σ sgn(σ) A(σ)：
//   - σ 作用于链中同组全同粒子的槽位；同一子衰变内的交换由选择定则
//     (-1)^{L+S}=±1 自动满足（稳定子），不进求和；
//   - 若成员是中间态（有子衰变），置换必须整棵子树一起换（子结构相同）；
//   - 费米子带 sgn(σ)（换位乘积奇偶性），玻色子恒 +1；
//   - 多组 = 各组 coset 的笛卡尔积（槽位不相交，直接叠加）。
// ============================================================================
void AmpCasDecay::buildPermTopologies()
{
    h_perm_maps_.clear();
    h_signs_.clear();
    h_signs_.push_back(1.0);  // σ=0：恒等

    if (identical_groups_.empty()) return;

    std::string root = decayChain_[0].mother;

    // 链中粒子集合（校验用）
    std::set<std::string> in_chain;
    for (const auto& node : decayChain_) {
        in_chain.insert(node.mother);
        in_chain.insert(node.daug1);
        in_chain.insert(node.daug2);
    }

    // 一个成员的子树：path（相对该成员的"左/右"走向序列）→ 粒子名 + 逆映射
    struct Subtree {
        std::string name;
        std::map<std::vector<int>, std::string> paths;      // path → name
        std::map<std::string, std::vector<int>> name_path;  // name → path
    };

    // 每组生成 coset（每个成员 i 的槽位接收成员 σ(i) 的内容）
    for (const auto& [members, is_boson] : identical_groups_) {
        std::vector<Subtree> subs;
        for (const auto& m : members) {
            if (m == root || in_chain.find(m) == in_chain.end()) continue;
            Subtree s;
            s.name = m;
            subs.push_back(s);
        }
        if (subs.size() < 2) continue;  // 单成员组无置换

        // 子树路径传播（父→子，循环直至不动点）
        for (auto& s : subs) {
            std::vector<int> empty;
            s.paths[empty] = s.name;
            s.name_path[s.name] = empty;
            bool changed = true;
            while (changed) {
                changed = false;
                for (const auto& node : decayChain_) {
                    auto it = s.name_path.find(node.mother);
                    if (it == s.name_path.end()) continue;
                    std::vector<int> p1 = it->second, p2 = it->second;
                    p1.push_back(1);
                    p2.push_back(2);
                    if (s.paths.find(p1) == s.paths.end()) {
                        s.paths[p1] = node.daug1;
                        s.name_path[node.daug1] = p1;
                        changed = true;
                    }
                    if (s.paths.find(p2) == s.paths.end()) {
                        s.paths[p2] = node.daug2;
                        s.name_path[node.daug2] = p2;
                        changed = true;
                    }
                }
            }
        }

        // 一致性：所有成员子树结构必须相同（全同粒子必须有相同的衰变结构）
        const auto& ref_paths = subs[0].paths;
        for (const auto& s : subs) {
            if (s.paths.size() != ref_paths.size()) {
                std::cerr << "Error: identical group members have different decay structures ("
                          << subs[0].name << " vs " << s.name << ")" << std::endl;
                return;  // 安全失败：不做对称化
            }
            for (const auto& [p, n] : ref_paths)
                if (s.paths.find(p) == s.paths.end()) {
                    std::cerr << "Error: identical group members have different decay structures ("
                              << subs[0].name << " vs " << s.name << ")" << std::endl;
                    return;
                }
        }

        // 嵌套检查：一个成员的子树内不能有同组另一成员
        for (size_t i = 0; i < subs.size(); ++i)
            for (size_t j = 0; j < subs.size(); ++j)
                if (i != j && subs[i].name_path.count(subs[j].name)) {
                    std::cerr << "Error: identical group members nested inside each other ("
                              << subs[i].name << " contains " << subs[j].name << ")" << std::endl;
                    return;
                }

        // 稳定子：同一步的两个子粒子互为同组成员 → 该换位是稳定子
        std::vector<std::pair<int, int>> stab_pairs;
        for (const auto& node : decayChain_) {
            int i1 = -1, i2 = -1;
            for (size_t i = 0; i < subs.size(); ++i) {
                if (subs[i].name == node.daug1) i1 = static_cast<int>(i);
                if (subs[i].name == node.daug2) i2 = static_cast<int>(i);
            }
            if (i1 >= 0 && i2 >= 0) stab_pairs.emplace_back(i1, i2);
        }
        // 稳定子元素 = 各换位的任意乘积（不相交 → 直接枚举子集）
        int n = static_cast<int>(subs.size());
        std::vector<int> id_perm(n);
        for (int i = 0; i < n; ++i) id_perm[i] = i;
        std::vector<std::vector<int>> stab_elts = { id_perm };
        for (const auto& [a, b] : stab_pairs) {
            size_t cur = stab_elts.size();
            for (size_t k = 0; k < cur; ++k) {
                std::vector<int> t = stab_elts[k];
                std::swap(t[a], t[b]);
                stab_elts.push_back(t);
            }
        }

        // 枚举 S_n 全部置换，按稳定子取陪集代表元（canonical = 轨道内字典序最小）
        auto canonical = [&](const std::vector<int>& p) {
            std::vector<int> best = p;
            for (const auto& t : stab_elts) {
                std::vector<int> q(n);
                for (int i = 0; i < n; ++i) q[i] = p[t[i]];
                if (q < best) best = q;
            }
            return best;
        };

        std::vector<std::vector<int>> reps;
        std::set<std::vector<int>> seen;
        std::vector<int> perm = id_perm;
        do {
            auto c = canonical(perm);
            if (seen.insert(c).second) reps.push_back(c);
        } while (std::next_permutation(perm.begin(), perm.end()));
        for (const auto& sb : subs) std::cerr << " " << sb.name;
        std::cerr << "  stab_pairs=" << stab_pairs.size() << "  reps:";
        for (const auto& r : reps) {
            std::cerr << " [";
            for (int x : r) std::cerr << x;
            std::cerr << "]";
        }
        std::cerr << std::endl;

        // 每组生成自己的 σ≥1 重建动量映射 + 符号
        std::vector<std::map<std::string, int>> group_maps;
        std::vector<double> group_signs;
        for (const auto& rep : reps) {
            if (rep == id_perm) continue;  // 恒等（含纯稳定子项）
            // 符号：费米子 sgn(rep)（逆序数奇偶），玻色子 +1
            double sg = 1.0;
            if (!is_boson) {
                int inv = 0;
                for (int i = 0; i < n; ++i)
                    for (int j = i + 1; j < n; ++j)
                        if (rep[i] > rep[j]) ++inv;
                sg = (inv % 2 == 0) ? 1.0 : -1.0;
            }
            // 重建动量映射：位置 i 的子树内容换成成员 rep[i] 的子树（路径对应）。
            // 映射语义（与 convertToDeviceMomenta 一致）：mapping[name] = 动量新槽位。
            // 必须覆盖全部粒子（恒等底 + 组内交换）——convertToDeviceMomenta 用 map 尺寸分配数组
            std::map<std::string, int> pmap = particleToIndex_;
            for (int i = 0; i < n; ++i) {
                int j = rep[i];  // 成员 subs[j] 的内容移到位置 i
                for (const auto& [path, name] : subs[i].paths) {
                    auto it = subs[j].paths.find(path);
                    if (it == subs[j].paths.end()) continue;  // 已校验一致，不会发生
                    pmap[it->second] = particleToIndex_[name];
                }
            }
            group_maps.push_back(pmap);
            group_signs.push_back(sg);
        }

        // 与已有组叠加（多组笛卡尔积；槽位不相交）
        if (group_maps.empty()) continue;
        if (h_perm_maps_.empty()) {
            h_perm_maps_ = group_maps;
            h_signs_ = { 1.0 };
            for (auto sg : group_signs) h_signs_.push_back(sg);
        } else {
            std::vector<std::map<std::string, int>> new_maps;
            std::vector<double> new_signs;
            for (size_t a = 0; a < h_perm_maps_.size(); ++a)
                for (size_t b = 0; b < group_maps.size(); ++b) {
                    std::map<std::string, int> merged = group_maps[b];
                    for (const auto& [k, v] : h_perm_maps_[a]) merged[k] = v;
                    new_maps.push_back(merged);
                    new_signs.push_back(h_signs_[a + 1] * group_signs[b]);
                }
            h_perm_maps_ = std::move(new_maps);
            h_signs_.clear();
            h_signs_.push_back(1.0);
            for (auto sg : new_signs) h_signs_.push_back(sg);
        }

        std::cout << "  Identical group (" << (is_boson ? "boson" : "fermion")
                  << "): " << n << " members (" << group_maps.size()
                  << " permutation topologies)" << std::endl;
    }

    if (h_signs_.size() > 33)
        std::cerr << "Warning: " << h_signs_.size() - 1
                  << " permutation topologies — memory usage scales with this!" << std::endl;
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

bool AmpCasDecay::getDaughterMassDep(int mother_idx, int target_idx,
    Q0MassDep& md1_dep, double& md1_fixed,
    Q0MassDep& md2_dep, double& md2_fixed) const
{
    if (mother_idx < 0 || mother_idx >= (int)particleNames_.size()) return false;
    const std::string& mname = particleNames_[mother_idx];
    for (const auto& node : decayChain_) {
        if (node.mother != mname) continue;
        // 质量依赖规则与 AD 版 kernel 的 q0 回退一致:
        //   固定质量(config 粒子表 mass>0) → FixedMass
        //   子粒子 = target（共振态，无固定质量）→ M0Param（= m0 参数）
        //   否则（事件质量）→ EventMass
        auto dep = [&](const std::string& dname, int d_idx) {
            auto it = particleMap_.find(dname);
            double mass = (it != particleMap_.end()) ? it->second.mass : -1.0;
            if (mass > 0) return std::make_pair(Q0MassDep::FixedMass, mass);
            if (d_idx == target_idx) return std::make_pair(Q0MassDep::M0Param, 0.0);
            return std::make_pair(Q0MassDep::EventMass, 0.0);
        };
        int d1 = getParticleIndex(node.daug1);
        int d2 = getParticleIndex(node.daug2);
        auto p1 = dep(node.daug1, d1);
        auto p2 = dep(node.daug2, d2);
        md1_dep = p1.first; md1_fixed = p1.second;
        md2_dep = p2.first; md2_fixed = p2.second;
        return true;
    }
    return false;
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
        // 注意：重建索引必须用【原始】索引空间（成员 particleToIndex_）——
        // σ 拓扑时传入的 mapping 是"动量→槽位"的放置映射，而 kernel 按节点
        // 原始索引读动量；中间态动量 = 其子树内（已按 σ 放置的）末态动量之和。
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

                auto mother_it = particleToIndex_.find(mother);
                auto daug1_it = particleToIndex_.find(daug1);
                auto daug2_it = particleToIndex_.find(daug2);

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
    d_slamp_tab_.resize(finalMomenta.size(), nullptr);
    d_mom_tab_.resize(finalMomenta.size(), nullptr);
    d_sign_tab_.resize(finalMomenta.size(), nullptr);
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
    {
        const auto& f0 = finalMomenta[0];
        auto it1 = f0.find("pi01");
        if (it1 != f0.end() && it1->second.size() > 8000) {
            const auto& p = it1->second[8000];
        }
    }
    // 全同粒子置换拓扑（在权威 index 空间内生成）
    buildPermTopologies();
    int nSigma = static_cast<int>(h_signs_.size());
    // 所有四动量都入设备端
    d_momenta_ = convertToDeviceMomenta(finalMomenta, particleToIndex_, decayChain_);
    // 每 σ 拓扑的重建四动量（σ=0 复用 d_momenta_；σ≥1 用交换映射重建——
    // 中间态动量随末态交换重新求和，这是置换正确性的关键）
    d_mom_sigma_.clear();
    d_mom_sigma_.push_back(d_momenta_);
    for (size_t m = 0; m < h_perm_maps_.size(); ++m)
        d_mom_sigma_.push_back(
            convertToDeviceMomenta(finalMomenta, h_perm_maps_[m], decayChain_));

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

        // 分配 SL 振幅 tab（nSigma 行：σ=0 恒等 + 置换拓扑）
        size_t slamp_row = (size_t)nEvents_[i] * nPolarizations_ * nSLCombs_;
        cudaMalloc(&d_slamp_tab_[i], (size_t)nSigma * slamp_row * sizeof(thrust::complex<double>));
        if (nSigma > 1) {
            cudaMalloc(&d_sign_tab_[i], (size_t)nSigma * sizeof(double));
            cudaMemcpy(d_sign_tab_[i], h_signs_.data(),
                       (size_t)nSigma * sizeof(double), cudaMemcpyHostToDevice);
            // σ 动量数组（DeviceMomenta 值拷贝，每项含各自的 momenta 指针）
            // σ 动量数组：DeviceMomenta 结构体在设备端，须 D2H 拷贝取值
            std::vector<DeviceMomenta> h_mom_arr(nSigma);
            for (int s = 0; s < nSigma; ++s)
                cudaMemcpy(&h_mom_arr[s], d_mom_sigma_[s][i],
                           sizeof(DeviceMomenta), cudaMemcpyDeviceToHost);
            cudaMalloc(&d_mom_tab_[i], (size_t)nSigma * sizeof(DeviceMomenta));
            cudaMemcpy(d_mom_tab_[i], h_mom_arr.data(),
                       (size_t)nSigma * sizeof(DeviceMomenta), cudaMemcpyHostToDevice);
        }

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

        // 逐 σ 计算 SL 振幅（σ=0 恒等；σ≥1 用重建动量）
        for (int s = 0; s < nSigma; ++s)
        {
            thrust::complex<double>* target = d_slamp_tab_[i] + (size_t)s * slamp_row;
            const DeviceMomenta* dm = d_mom_sigma_[s][i];
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
                computeSLAmpKernel << <numBlocks, blockSize, sharedMemSize >> > (
                    target, d_amp_buffer, dm, d_decayNodes_[i], d_dimj, d_dimj1,
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

            // 直接从设备内存获取四动量（σ 拓扑传入已重建的 d_momenta）
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

void AmpCasDecay::getAmps(std::vector<ctComplex*>& d_amplitudes,
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

            // 模型辅助数据（Hist 形状表）→ 同一辅助段
            const auto& aux = resonance.getAuxData();
            devRes.aux_offset = static_cast<int>(h_all_channels.size());
            devRes.aux_size = static_cast<int>(aux.size());
            h_all_channels.insert(h_all_channels.end(), aux.begin(), aux.end());

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
                d_slamp_tab_[i],       // SL振幅 tab（σ=0 行=恒等）
                getNSigma(),           // 置换拓扑数
                d_mom_tab_[i],         // σ 动量数组
                d_sign_tab_[i],        // 符号
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
computeAmpsKernel(ctComplex* amplitudes,                 // 输出振幅
    const DeviceMomenta* d_momenta,        // 所有事件的四动量数据
    const SL* slCombinations,              // SL组合数据
    const thrust::complex<double>* slamp_tab, // SL振幅 [nSigma × nSL×nPol×nEv]
    int nSigma,                            // 置换拓扑数（含恒等）
    const DeviceMomenta* d_mom_tab,        // [nSigma] 重建动量数组
    const double* d_sign_tab,              // [nSigma]
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

    // 全同粒子置换求和：A = Σ_σ sgn(σ) · resAmp(σ) · slamp(σ)
    // （resAmp 随 σ 变化：置换改变了中间态不变质量 → F 因子必须参与求和）
    size_t slamp_row = (size_t)nSLComb * nPolar * nEvents;
    const int MAX_POL = 32;
    if (nPolar > MAX_POL) { printf("computeAmpsKernel: nPolar=%d > %d\n", nPolar, MAX_POL); return; }
    ctResAmp A_tot[MAX_POL];
    for (int k = 0; k < nPolar; ++k) A_tot[k] = ctResAmp(0.0, 0.0);

    for (int s = 0; s < nSigma; ++s)
    {
        const DeviceMomenta* dm = (s == 0 || !d_mom_tab) ? d_momenta : &d_mom_tab[s];
        double sg = (s == 0) ? 1.0 : d_sign_tab[s];
        const thrust::complex<double>* slamps = slamp_tab + (size_t)s * slamp_row;

        ctResAmp resAmp(1.0, 0.0);

    // 遍历衰变链中的每个节点
    for (int nodeIdx = 0; nodeIdx < decayChain_size; ++nodeIdx)
    {
        const DecayNode& node = decayChain[nodeIdx];
        const SL& sl = slCombinations[nodeIdx + sl_idx * decayChain_size];

        // 获取母粒子四动量（σ 拓扑用重建动量）
        LorentzVector pMother = dm->getMomentum(event_idx, node.mother_idx);
        LorentzVector pDaug1 = dm->getMomentum(event_idx, node.daug1_idx);
        LorentzVector pDaug2 = dm->getMomentum(event_idx, node.daug2_idx);

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
            // 无参数模型（Hist）用默认质量，避免读空的 d_all_params
            mass_mother = (current_res.param_count > 0)
                ? d_all_params[current_res.param_offset] : 1.0;
        }
        if (mass_daug1 == -1)
        {
            for (int i = 0; i < resonance_count; ++i)
            {
                if (decayChain[nodeIdx].daug1_idx == resonances[i].particle_idx)
                {
                    mass_daug1 = (resonances[i].param_count > 0)
                        ? d_all_params[resonances[i].param_offset] : 1.0;
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
                    mass_daug2 = (resonances[i].param_count > 0)
                        ? d_all_params[resonances[i].param_offset] : 1.0;
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
                ch, current_res.n_channels,
                d_all_channels, current_res.aux_offset, bf_d);
        }
    }

    // 计算极化相关的振幅（σ 内累加；amp_idx 在循环外统一计算）
    for (int k = 0; k < nPolar; ++k)
    {
        int idx = sl_idx * nPolar * nEvents + event_idx * nPolar + k;

        ctResAmp temp = resAmp * slamps[idx];
        A_tot[k] += sg * temp;
    }
    }  // σ 循环结束

    // 输出对称化振幅
    for (int k = 0; k < nPolar; ++k)
    {
        int idx = sl_idx * nPolar * nEvents + event_idx * nPolar + k;
        int amp_idx = 0;
        if (offset_idx < num_offsets)
        {
            amp_idx = amp_offsets[offset_idx] + (event_idx - event_offsets[offset_idx]) * n_amplitudes * nPolar + k * n_amplitudes + sl_idx + site;
        }
        else
        {
            return;
        }
        amplitudes[amp_idx] = ctMake(A_tot[k].real(), A_tot[k].imag());
    }
}

// ============================================================================
// 合并非 AD kernel：一次启动处理多个无自由参数的 block
// （与 computeAmpsMergedKernel 共用 ADBlockDesc，忽略 AD 字段）
// ============================================================================
__global__ void computeAmpsMergedPlainKernel(
    ctComplex* amplitudes,
    const ADBlockDesc* desc, int nblocks, int nSL_total,
    const int* amp_offsets, const int* event_offsets, int num_offsets,
    int n_amplitudes, double bf_d)
{
    int slg = blockIdx.x;
    int event_idx = threadIdx.x + blockDim.x * blockIdx.y;

    int bi = 0;
    while (bi + 1 < nblocks && desc[bi + 1].sl_start <= slg) ++bi;
    const ADBlockDesc& B = desc[bi];
    int sl_idx = slg - B.sl_start;
    if (sl_idx >= B.nSL || event_idx >= B.nEvents) return;

    int offset_idx = 0;
    for (size_t i = 0; i < (size_t)num_offsets - 1; ++i) {
        if (event_idx < event_offsets[i + 1]) { offset_idx = i; break; }
    }

    ctResAmp resAmp(1.0, 0.0);

    // 遍历衰变链中的每个节点（逻辑与 computeAmpsKernel 相同）
    for (int nodeIdx = 0; nodeIdx < B.decayChain_size; ++nodeIdx)
    {
        const DecayNode& node = B.d_decayNodes[nodeIdx];
        const SL& sl = B.d_slComb[nodeIdx + sl_idx * B.decayChain_size];

        LorentzVector pMother = B.d_momenta->getMomentum(event_idx, node.mother_idx);
        LorentzVector pDaug1 = B.d_momenta->getMomentum(event_idx, node.daug1_idx);
        LorentzVector pDaug2 = B.d_momenta->getMomentum(event_idx, node.daug2_idx);

        double mm = pMother.M();
        double qq = std::sqrt((mm * mm - std::pow(pDaug1.M() + pDaug2.M(), 2)) *
                (mm * mm - std::pow(pDaug1.M() - pDaug2.M(), 2))) / 2 / mm;

        double mass_mother = node.mass[0];
        double mass_daug1 = node.mass[1];
        double mass_daug2 = node.mass[2];

        bool is_resonance_node = false;
        DeviceResonance current_res;
        for (int i = 0; i < B.resonance_count; ++i)
        {
            if (node.mother_idx == B.d_res[i].particle_idx)
            {
                is_resonance_node = true;
                current_res = B.d_res[i];
                break;
            }
        }

        if (mass_mother == -1 && is_resonance_node)
            mass_mother = (current_res.param_count > 0)
                ? B.d_all_params[current_res.param_offset] : 1.0;
        if (mass_daug1 == -1)
        {
            for (int i = 0; i < B.resonance_count; ++i)
            {
                if (node.daug1_idx == B.d_res[i].particle_idx)
                {
                    mass_daug1 = (B.d_res[i].param_count > 0)
                        ? B.d_all_params[B.d_res[i].param_offset] : 1.0;
                    break;
                }
            }
        }
        if (mass_daug2 == -1)
        {
            for (int i = 0; i < B.resonance_count; ++i)
            {
                if (node.daug2_idx == B.d_res[i].particle_idx)
                {
                    mass_daug2 = (B.d_res[i].param_count > 0)
                        ? B.d_all_params[B.d_res[i].param_offset] : 1.0;
                    break;
                }
            }
        }

        double q0 = std::sqrt((mass_mother * mass_mother - std::pow(mass_daug1 + mass_daug2, 2)) *
                (mass_mother * mass_mother - std::pow(mass_daug1 - mass_daug2, 2))) / 2 / mass_mother;

        if (nodeIdx == 0)
        {
            resAmp *= Bf<double>(sl.L, qq, q0, bf_d);
            continue;
        }

        if (is_resonance_node)
        {
            const double* p = B.d_all_params + current_res.param_offset;
            const double* ch = (current_res.type == ResModelType::Flatte)
                ? B.d_all_channels + current_res.channel_offset : nullptr;
            resAmp *= computeNodeFactor<double>(sl.L, mm, qq, q0,
                p, current_res.param_count, current_res.type,
                ch, current_res.n_channels,
                B.d_all_channels, current_res.aux_offset, bf_d);
        }
    }

    for (int k = 0; k < B.nPolar; ++k)
    {
        int idx = sl_idx * B.nPolar * B.nEvents + event_idx * B.nPolar + k;
        int amp_idx = 0;
        if (offset_idx < num_offsets)
        {
            int nEv_seg = event_offsets[offset_idx + 1] - event_offsets[offset_idx];
            amp_idx = amp_offsets[offset_idx]
                    + (event_idx - event_offsets[offset_idx]) * n_amplitudes * B.nPolar
                    + k * n_amplitudes + sl_idx + B.site;
        }
        else
        {
            return;
        }
        ctResAmp temp = resAmp * B.d_slamp_tab[idx];  // σ=0 行（恒等；本 kernel 未启用置换求和）
        amplitudes[amp_idx] = ctMake(temp.real(), temp.imag());
    }
}

// ============================================================================
// 合并 AD kernel：一次启动处理多个同 Nfree 的 block（减少启动开销，
// 提升 GPU 占用——多个小 grid 合并成一个大 grid）
// ============================================================================
template <int Nfree>
__global__ void computeAmpsMergedKernel(
    ctComplex* amplitudes,
    const ADBlockDesc* desc, int nblocks, int nSL_total,
    const int* amp_offsets, const int* event_offsets, int num_offsets,
    int n_amplitudes, double bf_d)
{
    int slg = blockIdx.x;
    int event_idx = threadIdx.x + blockDim.x * blockIdx.y;

    // 定位 block（顺序扫描，nblocks 通常 1-3）
    int bi = 0;
    while (bi + 1 < nblocks && desc[bi + 1].sl_start <= slg) ++bi;
    const ADBlockDesc& B = desc[bi];
    int sl_idx = slg - B.sl_start;
    if (sl_idx >= B.nSL || event_idx >= B.nEvents) return;

    int offset_idx = 0;
    for (size_t i = 0; i < (size_t)num_offsets - 1; ++i) {
        if (event_idx < event_offsets[i + 1]) { offset_idx = i; break; }
    }

    using AD = Var<double, Nfree, false>;
    using CV = ComplexVar<double, Nfree, false>;

    // 目标共振态 = block 的第一个共振态（架构保证每 block 一个）
    const DeviceResonance& target_res = B.d_res[0];
    const double* target_rp = B.d_all_params + target_res.param_offset;

    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nfree; ++j) {
        int pi = B.d_param_map[j];
        if (pi >= 0 && pi < 8) ftg[pi] = j;
    }

    AD m0_ad((target_res.param_count > 0) ? target_rp[0] : 1.0);
    if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;
    AD gamma_ad;
    if (target_res.type == ResModelType::BWR || target_res.type == ResModelType::BW ||
        target_res.type == ResModelType::Custom) {
        gamma_ad = AD((target_res.param_count > 1) ? target_rp[1] : 1.0);
        if (ftg[1] >= 0) gamma_ad.grad[ftg[1]] = 1.0;
    }
    AD other_g[4];
    if (target_res.type == ResModelType::Flatte) {
        for (int k = 0; k < target_res.param_count - 1 && k < 4; ++k) {
            other_g[k] = AD((target_res.param_count > 1 + k) ? target_rp[1 + k] : 1.0);
            if (ftg[1 + k] >= 0) other_g[k].grad[ftg[1 + k]] = 1.0;
        }
    }

    // ---- 全同粒子置换求和：A = Σ_σ sgn(σ)·R(σ)·slamp(σ) ----
    // R(σ) 为 σ 拓扑的节点因子链（AD，θ 相关）；slamp(σ) 为预缓存的 SL 振幅（θ 无关）。
    // per-σ 输出 ∂F(σ)/∂θ 到 d_dF_tab 供梯度/Hessian 使用。
    size_t slamp_row = (size_t)B.nSL * B.nPolar * B.nEvents;
    size_t dF_row = (size_t)B.nEvents * B.nSL * Nfree;
    // 每个极化独立累加（k 之间不能共享——各极化 slamp 不同）
    const int MAX_POL = 32;
    if (B.nPolar > MAX_POL) { printf("computeAmpsMergedKernel: nPolar=%d > %d\n", B.nPolar, MAX_POL); return; }
    CV R_tot_pol[MAX_POL];
    for (int k = 0; k < B.nPolar; ++k) R_tot_pol[k] = CV(0.0, 0.0);

    for (int s = 0; s < B.nSigma; ++s) {
        const DeviceMomenta* dm = (s == 0 || !B.d_mom_tab) ? B.d_momenta : &B.d_mom_tab[s];
        double sg = (s == 0) ? 1.0 : B.d_sign_tab[s];
        const thrust::complex<double>* slam = B.d_slamp_tab + (size_t)s * slamp_row;

        // ---- 节点循环：q0 质量回退规则与 computeAmpsKernelAD 一致 ----
        CV R_ad(1.0, 0.0);
        for (int nodeIdx = 0; nodeIdx < B.decayChain_size; ++nodeIdx) {
            const DecayNode& node = B.d_decayNodes[nodeIdx];
            const SL& sl = B.d_slComb[nodeIdx + sl_idx * B.decayChain_size];
            int L = sl.L;

            // σ 拓扑用重建动量（中间态动量已随交换重新求和）
            LorentzVector pM  = dm->getMomentum(event_idx, node.mother_idx);
            LorentzVector pD1 = dm->getMomentum(event_idx, node.daug1_idx);
            LorentzVector pD2 = dm->getMomentum(event_idx, node.daug2_idx);
            double mm = pM.M();
            double qq = breakup_momentum(mm, pD1.M(), pD2.M());
            AD q_ad(qq);
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
                AD params_arr[4] = {m0_ad, gamma_ad, AD(0.0), AD(0.0)};
                if (target_res.type == ResModelType::Custom && target_res.param_count > 2) {
                    params_arr[2] = AD(target_rp[2]);
                    if (ftg[2] >= 0) params_arr[2].grad[ftg[2]] = 1.0;
                }
                nf = computeNodeFactor<AD>(L, AD(mm), q_ad, q0_ad,
                                          params_arr, target_res.param_count, target_res.type,
                                          B.d_all_channels, target_res.n_channels,
                                          B.d_all_channels, target_res.aux_offset, bf_d);
            } else {
                AD bf_val = Bf<AD>(L, q_ad, q0_ad, bf_d);
                nf.real = bf_val; nf.imag = AD(0.0);
            }
            CV new_R;
            new_R.real = R_ad.real * nf.real - R_ad.imag * nf.imag;
            new_R.imag = R_ad.real * nf.imag + R_ad.imag * nf.real;
            R_ad = new_R;
        }

        // ---- per-σ 输出导数 ∂F(σ)/∂θ_j（复数）----
        int base = (event_idx * B.nSL + sl_idx) * Nfree + (int)((size_t)s * dF_row);
        for (int j = 0; j < Nfree; ++j) {
            B.d_dF_tab[base + j] = ctMake(R_ad.real.grad[j], R_ad.imag.grad[j]);
        }

        // ---- 振幅累加（val only，省 AD 标量传播）----
        for (int k = 0; k < B.nPolar; ++k) {
            size_t idx = (size_t)sl_idx * B.nPolar * B.nEvents + event_idx * B.nPolar + k;
            auto sl_amp = slam[idx];
            R_tot_pol[k].real.val += sg * (R_ad.real.val * sl_amp.real() - R_ad.imag.val * sl_amp.imag());
            R_tot_pol[k].imag.val += sg * (R_ad.real.val * sl_amp.imag() + R_ad.imag.val * sl_amp.real());
        }
    }

    // ---- 输出振幅（布局与 computeAmpsKernel 相同）----
    for (int k = 0; k < B.nPolar; ++k) {
        int idx = sl_idx * B.nPolar * B.nEvents + event_idx * B.nPolar + k;
        int amp_idx = 0;
        if (offset_idx < num_offsets) {
            int nEv_seg = event_offsets[offset_idx + 1] - event_offsets[offset_idx];
            amp_idx = amp_offsets[offset_idx]
                    + (event_idx - event_offsets[offset_idx]) * n_amplitudes * B.nPolar
                    + k * n_amplitudes + sl_idx + B.site;
        } else {
            return;
        }
        (void)idx;
        amplitudes[amp_idx] = ctMake(R_tot_pol[k].real.val, R_tot_pol[k].imag.val);
    }
}

// ============================================================================
// AD 版振幅 kernel：一次计算同时输出振幅 A 和共振态因子导数 ∂F/∂θ
// （reComputeAmps 对含自由参数的 block 使用；resonanceGradientKernel 读取 d_dF）
// 架构保证每个 block 恰有一个共振态（resonance_combinations 笛卡尔积展开）。
// ============================================================================
template <int Nfree>
__global__ void computeAmpsKernelAD(
    ctComplex* amplitudes,                 // 输出振幅（与 computeAmpsKernel 相同布局）
    ctComplex* d_dF,                       // 输出 ∂F/∂θ [nEvents*nSLComb*Nfree]
    const DeviceMomenta* d_momenta,
    const SL* slCombinations,
    const thrust::complex<double>* slamps,
    const DeviceResonance* resonances,
    int resonance_count,
    const double* d_all_params,
    const double* d_all_channels,
    const DecayNode* decayChain,
    int decayChain_size, int nEvents, int nSLComb, int nPolar,
    const int* amp_offsets, const int* event_offsets,
    int num_offsets, int n_amplitudes, int site,
    const int* d_param_map,               // [Nfree]: 自由参数下标
    double bf_d)
{
    int sl_idx = blockIdx.x;
    int event_idx = threadIdx.x + blockDim.x * blockIdx.y;
    if (sl_idx >= nSLComb || event_idx >= nEvents) return;

    int offset_idx = 0;
    for (size_t i = 0; i < (size_t)num_offsets - 1; ++i) {
        if (event_idx < event_offsets[i + 1]) { offset_idx = i; break; }
    }

    using AD = Var<double, Nfree, false>;
    using CV = ComplexVar<double, Nfree, false>;

    // 目标共振态 = block 的第一个共振态（架构保证每 block 一个）
    const DeviceResonance& target_res = resonances[0];
    const double* target_rp = d_all_params + target_res.param_offset;

    int ftg[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    for (int j = 0; j < Nfree; ++j) {
        int pi = d_param_map[j];
        if (pi >= 0 && pi < 8) ftg[pi] = j;
    }

    AD m0_ad((target_res.param_count > 0) ? target_rp[0] : 1.0);
    if (ftg[0] >= 0) m0_ad.grad[ftg[0]] = 1.0;
    AD gamma_ad;
    if (target_res.type == ResModelType::BWR || target_res.type == ResModelType::BW ||
        target_res.type == ResModelType::Custom) {
        gamma_ad = AD((target_res.param_count > 1) ? target_rp[1] : 1.0);
        if (ftg[1] >= 0) gamma_ad.grad[ftg[1]] = 1.0;
    }
    AD other_g[4];
    if (target_res.type == ResModelType::Flatte) {
        for (int k = 0; k < target_res.param_count - 1 && k < 4; ++k) {
            other_g[k] = AD((target_res.param_count > 1 + k) ? target_rp[1 + k] : 1.0);
            if (ftg[1 + k] >= 0) other_g[k].grad[ftg[1 + k]] = 1.0;
        }
    }

    // ---- 节点循环：q0 质量回退规则与 resonanceGradientKernel 一致 ----
    CV R_ad(1.0, 0.0);
    for (int nodeIdx = 0; nodeIdx < decayChain_size; ++nodeIdx) {
        const DecayNode& node = decayChain[nodeIdx];
        const SL& sl = slCombinations[nodeIdx + sl_idx * decayChain_size];
        int L = sl.L;

        LorentzVector pM  = d_momenta->getMomentum(event_idx, node.mother_idx);
        LorentzVector pD1 = d_momenta->getMomentum(event_idx, node.daug1_idx);
        LorentzVector pD2 = d_momenta->getMomentum(event_idx, node.daug2_idx);
        double mm = pM.M();
        double qq = breakup_momentum(mm, pD1.M(), pD2.M());
        AD q_ad(qq);
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
            AD params_arr[4] = {m0_ad, gamma_ad, AD(0.0), AD(0.0)};
            if (target_res.type == ResModelType::Custom && target_res.param_count > 2) {
                params_arr[2] = AD(target_rp[2]);
                if (ftg[2] >= 0) params_arr[2].grad[ftg[2]] = 1.0;
            }
            nf = computeNodeFactor<AD>(L, AD(mm), q_ad, q0_ad,
                                      params_arr, target_res.param_count, target_res.type,
                                      d_all_channels, target_res.n_channels,
                                      d_all_channels, target_res.aux_offset, bf_d);
        } else {
            AD bf_val = Bf<AD>(L, q_ad, q0_ad, bf_d);
            nf.real = bf_val; nf.imag = AD(0.0);
        }
        CV new_R;
        new_R.real = R_ad.real * nf.real - R_ad.imag * nf.imag;
        new_R.imag = R_ad.real * nf.imag + R_ad.imag * nf.real;
        R_ad = new_R;
    }

    // ---- 输出振幅（与 computeAmpsKernel 相同索引）----
    for (int k = 0; k < nPolar; ++k) {
        int idx = sl_idx * nPolar * nEvents + event_idx * nPolar + k;
        int amp_idx = 0;
        if (offset_idx < num_offsets) {
            int nEv_seg = event_offsets[offset_idx + 1] - event_offsets[offset_idx];
            amp_idx = amp_offsets[offset_idx]
                    + (event_idx - event_offsets[offset_idx]) * n_amplitudes * nPolar
                    + k * n_amplitudes + sl_idx + site;
        } else {
            return;
        }
        auto sl_amp = slamps[idx];
        CV sl_cv(sl_amp.real(), sl_amp.imag());
        CV temp = R_ad * sl_cv;
        amplitudes[amp_idx] = ctMake(temp.real.val, temp.imag.val);
    }

    // ---- 输出导数 ∂F/∂θ_j（复数）----
    if (d_dF) {
        int base = (event_idx * nSLComb + sl_idx) * Nfree;
        for (int j = 0; j < Nfree; ++j) {
            d_dF[base + j] = ctMake(R_ad.real.grad[j], R_ad.imag.grad[j]);
        }
    }
}

// ============================================================
// O 因子（非目标节点 Bf 乘积）对质量参数的导数辅助函数
// O = Π_i Bf(L_i, qq_i, q0_i), q0_i = breakup(m0_i, m1_i, m2_i)
// 当某子质量经回退等于 target_rp[0]（m0 参数）时，q0 依赖 m0 →
// dO/dm0 = O · Σ_i ∂lnBf_i/∂q0_i · ∂q0_i/∂m0
// ============================================================

// ∂q(m,m1,m2)/∂m_which (which=1: m1, which=2: m2)
// qsq ≤ 0 时 q 被钳位为 0（常数）→ 导数为 0
__device__ double breakup_dq_dm(int which, double m, double m1, double m2)
{
    double s = m1 + m2, d = m1 - m2;
    double A = m * m - s * s, B = m * m - d * d;
    double qsq = A * B;
    if (qsq <= 0.0) return 0.0;
    double dqsq = (which == 1) ? (-2.0 * s * B - 2.0 * d * A)
                               : (-2.0 * s * B + 2.0 * d * A);
    return dqsq / (4.0 * m * sqrt(qsq));
}

// ∂ln Bf(L,q,q0,d)/∂q0 = 0.5·N0'(z0)/N0(z0)·d, z0 = q0·d
// N0(z0) 多项式系数与 ResModel.cuh 的 Bf 完全一致
__device__ double dlnBf_dq0(int L, double q0, double bf_d)
{
    double z0 = q0 * bf_d;
    double z2 = z0 * z0;
    double n0, dn0;
    switch (L) {
    case 0: n0 = 1.0; dn0 = 0.0; break;
    case 1: n0 = 1.0 + z2; dn0 = 2.0 * z0; break;
    case 2: n0 = 9.0 + z2 * (3.0 + z2); dn0 = z0 * (6.0 + 4.0 * z2); break;
    case 3: n0 = 225.0 + z2 * (45.0 + z2 * (6.0 + z2));
            dn0 = z0 * (90.0 + z2 * (24.0 + 6.0 * z2)); break;
    case 4: n0 = 11025.0 + z2 * (1575.0 + z2 * (135.0 + z2 * (10.0 + z2)));
            dn0 = z0 * (3150.0 + z2 * (540.0 + z2 * (60.0 + 8.0 * z2))); break;
    case 5: n0 = 893025.0 + z2 * (99225.0 + z2 * (6300.0 + z2 * (315.0 + z2 * (15.0 + z2))));
            dn0 = z0 * (198450.0 + z2 * (25200.0 + z2 * (1890.0 + z2 * (120.0 + 10.0 * z2)))); break;
    case 6: n0 = 540326025.0 + z2 * (6185025.0 + z2 * (363825.0 + z2 * (17325.0 + z2 * (630.0 + z2 * (21.0 + z2)))));
            dn0 = z0 * (12370050.0 + z2 * (1455300.0 + z2 * (103950.0 + z2 * (5040.0 + z2 * (210.0 + 12.0 * z2))))); break;
    default: return 0.0;
    }
    return 0.5 * dn0 / n0 * bf_d;
}

__global__ void computeCustomAmpsKernel(
    ctComplex* amplitudes,
    const ADBlockDesc* desc, int nblocks, int nSL_total,
    const int* amp_offsets, const int* event_offsets, int num_offsets,
    int n_amplitudes, double bf_d)
{
    int slg = blockIdx.x;
    int event_idx = threadIdx.x + blockDim.x * blockIdx.y;

    int bi = 0;
    while (bi + 1 < nblocks && desc[bi + 1].sl_start <= slg) ++bi;
    const ADBlockDesc& B = desc[bi];
    int sl_idx = slg - B.sl_start;
    if (sl_idx >= B.nSL || event_idx >= B.nEvents) return;

    int offset_idx = 0;
    for (size_t i = 0; i < (size_t)num_offsets - 1; ++i)
        if (event_idx < event_offsets[i + 1]) { offset_idx = i; break; }

    const DeviceResonance& target_res = B.d_res[0];
    int P = target_res.param_count;
    if (P < 1) P = 1;
    if (P > 16) P = 16;   // 上限保护（实际 K-matrix 等 < 16）
    const double* target_rp = B.d_all_params + target_res.param_offset;
    const double* aux = B.d_all_channels;
    int aux_offset = target_res.aux_offset;

    double dFr[16], dFi[16], d2Fr[16 * 16], d2Fi[16 * 16];

    size_t slamp_row = (size_t)B.nSL * B.nPolar * B.nEvents;
    const int MAX_POL = 32;
    double R_re[MAX_POL], R_im[MAX_POL];
    for (int k = 0; k < B.nPolar; ++k) { R_re[k] = 0.0; R_im[k] = 0.0; }

    for (int s = 0; s < B.nSigma; ++s) {
        const DeviceMomenta* dm = (s == 0 || !B.d_mom_tab) ? B.d_momenta : &B.d_mom_tab[s];
        double sg = (s == 0) ? 1.0 : B.d_sign_tab[s];
        const thrust::complex<double>* slam = B.d_slamp_tab + (size_t)s * slamp_row;

        // 节点循环（标量）：非目标节点 × Bf；目标节点 = Custom 标量求值
        double Or = 1.0, Oi = 0.0;   // 非目标节点因子积
        double dlnO_dm0 = 0.0;       // Σ_i ∂lnBf_i/∂m0（经 q0 回退进入 O 的质量导数）
        double Fr = 1.0, Fi = 0.0;   // 目标 Custom F
        bool custom_eval = false;
        for (int nodeIdx = 0; nodeIdx < B.decayChain_size; ++nodeIdx) {
            const DecayNode& node = B.d_decayNodes[nodeIdx];
            const SL& sl = B.d_slComb[nodeIdx + sl_idx * B.decayChain_size];
            int L = sl.L;

            LorentzVector pM  = dm->getMomentum(event_idx, node.mother_idx);
            LorentzVector pD1 = dm->getMomentum(event_idx, node.daug1_idx);
            LorentzVector pD2 = dm->getMomentum(event_idx, node.daug2_idx);
            double mm = pM.M();
            double qq = breakup_momentum(mm, pD1.M(), pD2.M());
            double md1 = pD1.M(), md2 = pD2.M();
            bool is_target = (node.mother_idx == target_res.particle_idx && node.mass[0] <= 0);
            // q0 质量回退（与 computeAmpsMergedKernel 一致）:
            //   目标节点 → 参数 m0；固定质量 → node.mass；其它 → 1.0
            double m0_q0 = is_target
                ? ((target_res.param_count > 0) ? target_rp[0] : 1.0)
                : ((node.mass[0] > 0) ? node.mass[0] : 1.0);
            // 子粒子是目标共振态（无固定质量）→ 用其 m0 参数
            // （与 computeAmpsMergedKernel 一致；否则回退到事件质量）
            double md1_q0 = (node.mass[1] > 0) ? node.mass[1]
                : ((node.daug1_idx == target_res.particle_idx && target_res.param_count > 0)
                       ? target_rp[0] : md1);
            double md2_q0 = (node.mass[2] > 0) ? node.mass[2]
                : ((node.daug2_idx == target_res.particle_idx && target_res.param_count > 0)
                       ? target_rp[0] : md2);
            double q0 = breakup_momentum(m0_q0, md1_q0, md2_q0);

            if (is_target) {
                if (!custom_eval) {
                    evalCustomAll(aux, aux_offset, mm, qq, q0, L, bf_d,
                        md1_q0, md2_q0,
                        target_rp, P, Fr, Fi, dFr, dFi, d2Fr, d2Fi);
                    custom_eval = true;
                }
            } else {
                double bf = Bf<double>(L, qq, q0, bf_d);
                Or *= bf; Oi *= bf;
                // q0 回退到 target_rp[0]（m0 参数）→ 累积 d ln Bf / d m0
                double dq0_dm = 0.0;
                if (md1_q0 == target_rp[0]) dq0_dm += breakup_dq_dm(1, m0_q0, md1_q0, md2_q0);
                if (md2_q0 == target_rp[0]) dq0_dm += breakup_dq_dm(2, m0_q0, md1_q0, md2_q0);
                dlnO_dm0 += dlnBf_dq0(L, q0, bf_d) * dq0_dm;
            }
        }

        // F_total = O × F_custom; dF_total[j] = O × dF_custom[j] + (dO/dm0)·F_custom·δ_{j0}
        // （O 经 q0 回退依赖质量参数 m0，宽度参数不进入 O）
        double Ftr = Or * Fr - Oi * Fi, Fti = Or * Fi + Oi * Fr;
        double dO_dm0 = Or * dlnO_dm0;

        if (B.d_dF_tab) {
            int base = (event_idx * B.nSL + sl_idx) * P
                     + (int)((size_t)s * ((size_t)B.nEvents * B.nSL * P));
            for (int j = 0; j < P; ++j) {
                double dtr = Or * dFr[j] - Oi * dFi[j];
                double dti = Or * dFi[j] + Oi * dFr[j];
                if (j == 0) { dtr += dO_dm0 * Fr; dti += dO_dm0 * Fi; }
                B.d_dF_tab[base + j] = ctMake(dtr, dti);
            }
        }

        for (int k = 0; k < B.nPolar; ++k) {
            size_t idx = (size_t)sl_idx * B.nPolar * B.nEvents + event_idx * B.nPolar + k;
            auto sl_amp = slam[idx];
            R_re[k] += sg * (Ftr * sl_amp.real() - Fti * sl_amp.imag());
            R_im[k] += sg * (Ftr * sl_amp.imag() + Fti * sl_amp.real());
        }
    }

    // 输出振幅
    for (int k = 0; k < B.nPolar; ++k) {
        int idx = sl_idx * B.nPolar * B.nEvents + event_idx * B.nPolar + k;
        if (offset_idx < num_offsets) {
            int nEv_seg = event_offsets[offset_idx + 1] - event_offsets[offset_idx];
            int amp_idx = amp_offsets[offset_idx]
                        + (event_idx - event_offsets[offset_idx]) * n_amplitudes * B.nPolar
                        + k * n_amplitudes + sl_idx + B.site;
            amplitudes[amp_idx] = ctMake(R_re[k], R_im[k]);
        }
        (void)idx;
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
            if (block.d_dF.size() > gpu && block.d_dF[gpu]) cudaFree(block.d_dF[gpu]);
            if (block.d_res_idx_.size() > gpu && block.d_res_idx_[gpu]) cudaFree(block.d_res_idx_[gpu]);
            if (block.d_param_idx_.size() > gpu && block.d_param_idx_[gpu]) cudaFree(block.d_param_idx_[gpu]);
            if (block.d_global_offset_.size() > gpu && block.d_global_offset_[gpu]) cudaFree(block.d_global_offset_[gpu]);
            if (block.d_param_map_.size() > gpu && block.d_param_map_[gpu]) cudaFree(block.d_param_map_[gpu]);
            if (block.d_global_idx_.size() > gpu && block.d_global_idx_[gpu]) cudaFree(block.d_global_idx_[gpu]);
        }
    }
    // 跨块持久化缓冲（每 GPU 一份）
    size_t n_gpu_persist = d_params_per_gpu_.size();
    for (size_t gpu = 0; gpu < n_gpu_persist; ++gpu) {
        cudaSetDevice(static_cast<int>(gpu));
        if (d_params_per_gpu_[gpu]) cudaFree(d_params_per_gpu_[gpu]);
        if (d_grad_per_gpu_[gpu]) cudaFree(d_grad_per_gpu_[gpu]);
        if (d_ev_off_cache_[gpu]) cudaFree(d_ev_off_cache_[gpu]);
        if (d_amp_off_cache_[gpu]) cudaFree(d_amp_off_cache_[gpu]);
        if (d_ad_desc_[gpu]) cudaFree(d_ad_desc_[gpu]);
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
    block.d_dF.resize(n_gpu, nullptr);

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

        // 模型辅助数据 → 同一辅助段
        // Hist/Custom: 构造时已构建（getAuxData）；内置模型（BWR/BW/ONE/Flatte）:
        // 符号微分 aux 需要 decay 结构（q0 链的子粒子质量依赖）→ 此处构建。
        std::vector<double> aux = res.getAuxData();
        if (aux.empty() &&
            (dr.type == ResModelType::BWR || dr.type == ResModelType::BW ||
             dr.type == ResModelType::ONE || dr.type == ResModelType::Flatte)) {
            const auto& opts = res.getOptions();
            int L = (dr.type == ResModelType::BWR) ? 1 : 2;
            auto it = opts.find("L");
            if (it != opts.end()) L = std::stoi(it->second);
            double dd = 3.0;
            it = opts.find("d");
            if (it != opts.end()) dd = std::stod(it->second);
            std::vector<double> cf;
            for (const auto& ch : res.getChannels()) {
                cf.push_back(ch.first);
                cf.push_back(ch.second);
            }
            Q0MassDep m1d, m2d;
            double m1f = 0.0, m2f = 0.0;
            if (!cas->getDaughterMassDep(dr.particle_idx, dr.particle_idx,
                                         m1d, m1f, m2d, m2f)) {
                m1d = Q0MassDep::EventMass; m2d = Q0MassDep::EventMass;
            }
            aux = buildModelAST(dr.type, L, dd, dr.param_count, dr.n_channels, cf,
                                m1d, m1f, m2d, m2f);
        }
        dr.aux_offset = static_cast<int>(h_all_channels.size());
        dr.aux_size = static_cast<int>(aux.size());
        h_all_channels.insert(h_all_channels.end(), aux.begin(), aux.end());

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
            const auto& pnames = resonances[i].getOrderedParamNames();
            std::string pname = (p_idx < (int)pnames.size()) ? pnames[p_idx]
                                                            : ("p" + std::to_string(p_idx));
            slots_.push_back({block_idx, static_cast<int>(i), p_idx, val, lower,
                              upper, resonances[i].getName() + "_" + pname});
        }
    }

    // 记录该块的自由参数信息（AD kernel 用）
    blocks_.back().free_global_idx.clear();
    blocks_.back().free_param_idx.clear();
    blocks_.back().nFree = 0;
    for (int s = 0; s < (int)slots_.size(); ++s) {
        if (slots_[s].block_idx == block_idx) {
            blocks_.back().free_global_idx.push_back(s);
            blocks_.back().free_param_idx.push_back(slots_[s].param_idx);
            blocks_.back().nFree++;
        }
    }
}

void AmpCalc::reComputeAmps(std::vector<ctComplex*>& d_amplitudes,
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

    // 1. 把 d_params 广播到所有 GPU（缓冲持久化，仅首次分配）
    if (d_params_per_gpu_.size() != (size_t)n_gpu) {
        d_params_per_gpu_.assign(n_gpu, nullptr);
        for (int gpu = 0; gpu < n_gpu; ++gpu) {
            cudaSetDevice(gpu);
            cudaMalloc(&d_params_per_gpu_[gpu], n_free * sizeof(double));
        }
    }
    int primary_dev = 0;
    cudaGetDevice(&primary_dev);
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == primary_dev) {
            cudaSetDevice(primary_dev);
            cudaMemcpy(d_params_per_gpu_[primary_dev], d_params,
                       n_free * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyPeer(d_params_per_gpu_[gpu], gpu,
                           d_params, primary_dev,
                           n_free * sizeof(double));
        }
    }

    // 2. amp/event offsets 设备缓冲：内容不变则复用（fit 循环中内容固定）
    if (cached_ev_off_.size() != (size_t)n_gpu) {
        cached_ev_off_.resize(n_gpu);
        cached_amp_off_.resize(n_gpu);
        d_ev_off_cache_.assign(n_gpu, nullptr);
        d_amp_off_cache_.assign(n_gpu, nullptr);
    }
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (cached_ev_off_[gpu] != event_offsets[gpu]) {
            cudaSetDevice(gpu);
            if (!d_ev_off_cache_[gpu])
                cudaMalloc(&d_ev_off_cache_[gpu],
                           std::max((size_t)1, event_offsets[gpu].size()) * sizeof(int));
            cudaMemcpy(d_ev_off_cache_[gpu], event_offsets[gpu].data(),
                       event_offsets[gpu].size() * sizeof(int), cudaMemcpyHostToDevice);
            cached_ev_off_[gpu] = event_offsets[gpu];
        }
        if (cached_amp_off_[gpu] != amp_offsets[gpu]) {
            cudaSetDevice(gpu);
            if (!d_amp_off_cache_[gpu])
                cudaMalloc(&d_amp_off_cache_[gpu],
                           std::max((size_t)1, amp_offsets[gpu].size()) * sizeof(int));
            cudaMemcpy(d_amp_off_cache_[gpu], amp_offsets[gpu].data(),
                       amp_offsets[gpu].size() * sizeof(int), cudaMemcpyHostToDevice);
            cached_amp_off_[gpu] = amp_offsets[gpu];
        }
    }

    // 3. 每个 GPU: 更新 DeviceResonance 参数 + 重跑 computeAmpsKernel
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        int blockSize = 256;

        // 为每个 block 更新 DeviceResonance 中的自由参数
        // （slot → (res_idx, param_idx, global_idx) 映射在 addBlock 后固定，持久化懒分配）
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            auto& block = blocks_[bi];

            if (!block.d_res_idx_.size()) {
                block.d_res_idx_.assign(n_gpu, nullptr);
                block.d_param_idx_.assign(n_gpu, nullptr);
                block.d_global_offset_.assign(n_gpu, nullptr);
            }
            if (block.d_res_idx_[gpu]) continue;  // 映射已上传过（内容固定）

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

            cudaMalloc(&block.d_res_idx_[gpu], std::max(1, n_local) * sizeof(int));
            cudaMalloc(&block.d_param_idx_[gpu], std::max(1, n_local) * sizeof(int));
            cudaMalloc(&block.d_global_offset_[gpu], std::max(1, n_local) * sizeof(int));
            cudaMemcpy(block.d_res_idx_[gpu], local_res_idx.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(block.d_param_idx_[gpu], local_param_idx.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(block.d_global_offset_[gpu], local_global_offset.data(), n_local * sizeof(int), cudaMemcpyHostToDevice);
        }

        // 启动 updateResonanceParams（每个 GPU 只需更新一次全局 slot 映射——见下方统一 kernel）
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            auto& block = blocks_[bi];
            int n_local = static_cast<int>(block.free_global_idx.size());
            if (n_local == 0 || !block.d_res_idx_[gpu]) continue;
            int grid = (n_local + blockSize - 1) / blockSize;
            updateResonanceParamsKernel<<<grid, blockSize>>>(
                block.d_all_params[gpu], block.d_resonances[gpu],
                block.resonance_count,
                d_params_per_gpu_[gpu], block.d_res_idx_[gpu],
                block.d_param_idx_[gpu], block.d_global_offset_[gpu], n_local);
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

        // 为 conjugate 块填充自由参数信息（从 owner 复制；参数值已广播）
        for (const auto& [cj_key, owner_key] : conjugate_broadcast_) {
            auto& cj = blocks_[cj_key.first];
            auto& ow = blocks_[owner_key.first];
            cj.nFree = ow.nFree;
            cj.free_global_idx = ow.free_global_idx;
            cj.free_param_idx = ow.free_param_idx;
        }

        // 重跑 computeAmpsKernel：AD block 按 nFree 分组，每组一次合并启动
        // （同 stream 隐式顺序依赖，无需 per-block sync）
        int* d_amp_offsets = d_amp_off_cache_[gpu];
        int* d_event_offsets = d_ev_off_cache_[gpu];
        int num_offsets = static_cast<int>(amp_offsets[gpu].size());

        // 分组：nFree → block 列表（host；块结构固定，开销可忽略）
        // Custom 模型走独立标量 kernel（参数数 P 运行时，不经过 Var 模板）
        std::map<int, std::vector<int>> ad_groups;
        std::vector<int> custom_blocks;
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            if (blocks_[bi].nFree <= 0) continue;
            // 用 h_templates_ 判断类型（host 端）
            const auto& tmpl = h_templates_[bi];
            bool is_custom = !tmpl.empty() && tmpl[0].type == ResModelType::Custom;
            // 内置模型（BWR/BW/Flatte 等）有符号微分 aux[] 且 Nres=1 → 同样走统一标量
            // kernel（P4：符号微分替代 AutoDiff；BWR/ONE 的 L 依赖在 aux 段内以
            // CVAR_L 运行时解析，无需 per-block L 路由）
            bool has_sym_aux = !tmpl.empty() && tmpl[0].aux_size > 0
                            && blocks_[bi].resonance_count == 1;
            if (is_custom || has_sym_aux) custom_blocks.push_back(static_cast<int>(bi));
            else ad_groups[blocks_[bi].nFree].push_back(static_cast<int>(bi));
        }

        // 确保所有 AD block 的 d_dF / d_param_map 已分配（懒分配持久化）
        // （custom_blocks 的 d_dF 同样需要）
        auto ensure_ddF = [&](int bi) {
            auto& block = blocks_[bi];
            auto& cas = cas_list_[block.cas_idx];
            int nSL = static_cast<int>(cas->getNSLCombs());
            int nEv = static_cast<int>(cas->getNEventsVec()[gpu]);
            int nSigma = cas->getNSigma();
            size_t dF_size = (size_t)nEv * nSL * block.nFree * nSigma;
            if (!block.d_dF[gpu])
                cudaMalloc(&block.d_dF[gpu], dF_size * sizeof(ctComplex));
            if (!block.d_param_map_.size()) block.d_param_map_.assign(n_gpu, nullptr);
            if (!block.d_param_map_[gpu]) {
                cudaMalloc(&block.d_param_map_[gpu], block.nFree * sizeof(int));
                cudaMemcpy(block.d_param_map_[gpu], block.free_param_idx.data(),
                           block.nFree * sizeof(int), cudaMemcpyHostToDevice);
            }
        };
        for (int bi : custom_blocks) ensure_ddF(bi);
        for (const auto& [nfree, blist] : ad_groups) {
            (void)nfree;
            for (int bi : blist) ensure_ddF(bi);
        }

        // 按 nFree 分组构建 desc 并启动（每组一次 kernel）
        int n_ad_blocks = 0;
        for (const auto& [nfree, blist] : ad_groups) n_ad_blocks += (int)blist.size();
        n_ad_blocks += (int)custom_blocks.size();   // Custom block 复用同一 desc 缓冲
        if (d_ad_desc_.size() != (size_t)n_gpu) d_ad_desc_.assign(n_gpu, nullptr);
        if (n_ad_blocks > d_ad_desc_cap_) {
            for (int g = 0; g < n_gpu; ++g) {
                cudaSetDevice(g);
                if (d_ad_desc_[g]) cudaFree(d_ad_desc_[g]);
                cudaMalloc(&d_ad_desc_[g], n_ad_blocks * sizeof(ADBlockDesc));
            }
            d_ad_desc_cap_ = n_ad_blocks;
        }
        std::vector<ADBlockDesc> h_desc;
        for (const auto& [nfree, blist] : ad_groups) {
            h_desc.clear();
            int sl_start = 0, max_evt = 0;
            for (int bi : blist) {
                auto& block = blocks_[bi];
                auto& cas = cas_list_[block.cas_idx];
                ADBlockDesc d;
                d.d_momenta = cas->getMomenta()[gpu];
                d.d_slComb = cas->getDeviceSLCombs()[gpu];
                d.d_slamp_tab = cas->getSLAmpsTab()[gpu];
                d.d_res = block.d_resonances[gpu];
                d.d_all_params = block.d_all_params[gpu];
                d.d_all_channels = block.d_all_channels[gpu];
                d.d_decayNodes = cas->getDecayNodes()[gpu];
                d.d_param_map = block.d_param_map_[gpu];
                d.d_dF_tab = block.d_dF[gpu];
                d.nSigma = cas->getNSigma();
                d.d_mom_tab = cas->getMomentaTab()[gpu];
                d.d_sign_tab = cas->getSignsTab()[gpu];
                d.resonance_count = block.resonance_count;
                d.decayChain_size = cas->getDecayChainSize();
                d.nEvents = static_cast<int>(cas->getNEventsVec()[gpu]);
                d.nSL = static_cast<int>(cas->getNSLCombs());
                d.nPolar = static_cast<int>(n_polar);
                d.sl_start = sl_start;
                d.site = block.site;
                h_desc.push_back(d);
                sl_start += d.nSL;
                max_evt = std::max(max_evt, d.nEvents);
            }
            int nblocks = static_cast<int>(h_desc.size());
            cudaMemcpy(d_ad_desc_[gpu], h_desc.data(), nblocks * sizeof(ADBlockDesc),
                       cudaMemcpyHostToDevice);
            dim3 gridM(static_cast<unsigned int>(sl_start),
                       (static_cast<unsigned int>(max_evt) + 255) / 256);
            switch (nfree) {
            case 1:
                computeAmpsMergedKernel<1><<<gridM, blockSize>>>(
                    d_amplitudes[gpu], d_ad_desc_[gpu], nblocks, sl_start,
                    d_amp_offsets, d_event_offsets, num_offsets, n_amplitudes, bf_d);
                break;
            case 2:
                computeAmpsMergedKernel<2><<<gridM, blockSize>>>(
                    d_amplitudes[gpu], d_ad_desc_[gpu], nblocks, sl_start,
                    d_amp_offsets, d_event_offsets, num_offsets, n_amplitudes, bf_d);
                break;
            case 3:
                computeAmpsMergedKernel<3><<<gridM, blockSize>>>(
                    d_amplitudes[gpu], d_ad_desc_[gpu], nblocks, sl_start,
                    d_amp_offsets, d_event_offsets, num_offsets, n_amplitudes, bf_d);
                break;
            default:
                // 4+ 自由参数（Flatte 多通道）暂不支持 AD 融合，回退逐个非 AD
                for (int bi : blist) {
                    auto& block = blocks_[bi];
                    auto& cas = cas_list_[block.cas_idx];
                    dim3 gd(static_cast<unsigned int>(cas->getNSLCombs()),
                            (static_cast<unsigned int>(cas->getNEventsVec()[gpu]) + 255) / 256);
                    computeAmpsKernel<<<gd, blockSize>>>(
                        d_amplitudes[gpu], cas->getMomenta()[gpu], cas->getDeviceSLCombs()[gpu],
                        cas->getSLAmpsTab()[gpu], cas->getNSigma(),
                        cas->getMomentaTab()[gpu], cas->getSignsTab()[gpu],
                        block.d_resonances[gpu],
                        block.resonance_count, block.d_all_params[gpu],
                        block.d_all_channels[gpu], cas->getDecayNodes()[gpu],
                        cas->getDecayChainSize(),
                        static_cast<int>(cas->getNEventsVec()[gpu]),
                        static_cast<int>(cas->getNSLCombs()),
                        static_cast<int>(n_polar),
                        d_amp_offsets, d_event_offsets, num_offsets,
                        n_amplitudes, block.site, bf_d);
                }
                break;
            }
        }

        // Custom block：独立标量 kernel（参数数 P 运行时，dF 直接输出）
        if (!custom_blocks.empty()) {
            h_desc.clear();
            int sl_start = 0, max_evt = 0;
            for (int bi : custom_blocks) {
                auto& block = blocks_[bi];
                auto& cas = cas_list_[block.cas_idx];
                ADBlockDesc d;
                d.d_momenta = cas->getMomenta()[gpu];
                d.d_slComb = cas->getDeviceSLCombs()[gpu];
                d.d_slamp_tab = cas->getSLAmpsTab()[gpu];
                d.d_res = block.d_resonances[gpu];
                d.d_all_params = block.d_all_params[gpu];
                d.d_all_channels = block.d_all_channels[gpu];
                d.d_decayNodes = cas->getDecayNodes()[gpu];
                d.d_param_map = block.d_param_map_[gpu];
                d.d_dF_tab = block.d_dF[gpu];
                d.nSigma = cas->getNSigma();
                d.d_mom_tab = cas->getMomentaTab()[gpu];
                d.d_sign_tab = cas->getSignsTab()[gpu];
                d.resonance_count = block.resonance_count;
                d.decayChain_size = cas->getDecayChainSize();
                d.nEvents = static_cast<int>(cas->getNEventsVec()[gpu]);
                d.nSL = static_cast<int>(cas->getNSLCombs());
                d.nPolar = static_cast<int>(n_polar);
                d.sl_start = sl_start;
                d.site = block.site;
                h_desc.push_back(d);
                sl_start += d.nSL;
                max_evt = std::max(max_evt, d.nEvents);
            }
            int nblocks = static_cast<int>(h_desc.size());
            cudaMemcpy(d_ad_desc_[gpu], h_desc.data(), nblocks * sizeof(ADBlockDesc),
                       cudaMemcpyHostToDevice);
            dim3 gridC(static_cast<unsigned int>(sl_start),
                       (static_cast<unsigned int>(max_evt) + 255) / 256);
            computeCustomAmpsKernel<<<gridC, blockSize>>>(
                d_amplitudes[gpu], d_ad_desc_[gpu], nblocks, sl_start,
                d_amp_offsets, d_event_offsets, num_offsets, n_amplitudes, bf_d);
        }

        // 非 AD block（无自由参数）：振幅 A 只依赖四动量，与 θ 无关，
        // 初始化时 getAmps 已计算，fit 循环中跳过重算（纯浪费）
    }

    // 恢复 primary device
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

// 梯度 kernel（纯读取 d_dF 版）：∂F/∂θ 由 computeAmpsKernelAD 预计算，
// 本 kernel 只做: -2·sign·Re(conj(w)·v·slamp·∂F/∂θ) 累加
// 优化：per-block shared 部分和 → 每 block 只做 Nfree 次全局 atomicAdd
// （50 万事件时全局 atomicAdd 从 200 万次降到 ~8000 次，消除竞争）
// 注意：Custom 模型（参数数 4-16，运行时）走 resonanceGradientKernelRuntime
//（下方），不经过本模板 —— computeCustomAmpsKernel 的 d_dF_tab 布局相同。
template <int Nfree>
__global__ void resonanceGradientKernel(
    const ctComplex* d_w,
    const thrust::complex<double>* d_slamp_tab,  // [nSigma × nSL×nPol×nEv_total]
    const ctComplex* d_v,
    const ctComplex* d_dF_tab,       // [nSigma × nEv_total×nSL×Nfree] 复数导数
    const int* d_global_idx,         // [Nfree]
    double* d_grad,
    int nEvents, int nPolar, int nSLComb, double sign,
    int evt_off, int site, int n_events_total,
    int nSigma, const double* d_sign_tab)
{
    __shared__ double shm[Nfree];
    for (int j = 0; j < Nfree; ++j) shm[j] = 0.0;
    __syncthreads();

    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt < nEvents) {
        int global_evt = evt + evt_off;
        double acc[Nfree];
        for (int j = 0; j < Nfree; ++j) acc[j] = 0.0;

        size_t slamp_row = (size_t)nSLComb * nPolar * n_events_total;
        size_t dF_row = (size_t)n_events_total * nSLComb * Nfree;

        for (int sl_idx = 0; sl_idx < nSLComb; ++sl_idx) {
            ctComplex v_sl = d_v[site + sl_idx];

            // 全同粒子：∂A/∂θ_j = Σ_σ sgn(σ)·v·slamp(σ)·∂F(σ)/∂θ_j
            for (int s = 0; s < nSigma; ++s) {
                double sg = (s == 0) ? 1.0 : d_sign_tab[s];
                const thrust::complex<double>* slam = d_slamp_tab + (size_t)s * slamp_row;
                const ctComplex* dF_s = d_dF_tab + (size_t)s * dF_row;

                double s_re = 0.0, s_im = 0.0;
                for (int pol = 0; pol < nPolar; ++pol) {
                    ctComplex w_val = d_w[evt * nPolar + pol];
                    int amp_idx = sl_idx * n_events_total * nPolar + global_evt * nPolar + pol;
                    auto sl_amp = slam[amp_idx];
                    double sl_re = sl_amp.real();
                    double sl_im = sl_amp.imag();
                    // T_sl = v_sl * slamps[sl,e,p]
                    double T_re = (double)v_sl.x * sl_re - (double)v_sl.y * sl_im;
                    double T_im = (double)v_sl.x * sl_im + (double)v_sl.y * sl_re;
                    // conj(w) * T
                    s_re += (double)w_val.x * T_re + (double)w_val.y * T_im;
                    s_im += (double)w_val.x * T_im - (double)w_val.y * T_re;
                }

                // d_dF 由 reComputeAmps 按全局事件索引写入（含 phsp/data/bkg 全段），
                // 此处必须用 global_evt（段内 evt 会错位）
                int dF_base = (global_evt * nSLComb + sl_idx) * Nfree;
                for (int j = 0; j < Nfree; ++j) {
                    ctComplex dF = dF_s[dF_base + j];
                    // -2 Re((s_re + i s_im) * (dF_re + i dF_im)) = -2 (s_re*dF_re - s_im*dF_im)
                    acc[j] += -2.0 * sign * sg * (s_re * (double)dF.x - s_im * (double)dF.y);
                }
            }
        }
        for (int j = 0; j < Nfree; ++j)
            atomicAdd(&shm[j], acc[j]);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int j = 0; j < Nfree; ++j)
            atomicAdd(&d_grad[d_global_idx[j]], shm[j]);
}

// ---------------------------------------------------------------------------
// T 计算 kernel: T[e,p] = Σ_sl v[site+sl] * slamps[sl, e, p]
// ---------------------------------------------------------------------------
__global__ void computeEffectiveCouplingKernel(
    ctComplex* d_T,
    const thrust::complex<double>* d_slamps,
    const ctComplex* d_v,
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
    d_T[idx] = ctMake(ctCastFloat(re), ctCastFloat(im));
}

// ---------------------------------------------------------------------------
__global__ void resonanceGradientKernelRuntime(
    const ctComplex* d_w,
    const thrust::complex<double>* d_slamp_tab,  // [nSigma × nSL×nPol×nEv_total]
    const ctComplex* d_v,
    const ctComplex* d_dF_tab,       // [nSigma × nEv_total×nSL×Nfree] 复数导数
    const int* d_global_idx,         // [Nfree]
    double* d_grad,
    int nEvents, int nPolar, int nSLComb, double sign,
    int evt_off, int site, int n_events_total,
    int nSigma, const double* d_sign_tab, int Nfree)
{
    __shared__ double shm[16];
    for (int j = 0; j < Nfree; ++j) shm[j] = 0.0;
    __syncthreads();

    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt < nEvents) {
        int global_evt = evt + evt_off;
        double acc[16];
        for (int j = 0; j < Nfree; ++j) acc[j] = 0.0;

        size_t slamp_row = (size_t)nSLComb * nPolar * n_events_total;
        size_t dF_row = (size_t)n_events_total * nSLComb * Nfree;

        for (int sl_idx = 0; sl_idx < nSLComb; ++sl_idx) {
            ctComplex v_sl = d_v[site + sl_idx];

            for (int s = 0; s < nSigma; ++s) {
                double sg = (s == 0) ? 1.0 : d_sign_tab[s];
                const thrust::complex<double>* slam = d_slamp_tab + (size_t)s * slamp_row;
                const ctComplex* dF_s = d_dF_tab + (size_t)s * dF_row;

                double s_re = 0.0, s_im = 0.0;
                for (int pol = 0; pol < nPolar; ++pol) {
                    ctComplex w_val = d_w[evt * nPolar + pol];
                    int amp_idx = sl_idx * n_events_total * nPolar + global_evt * nPolar + pol;
                    auto sl_amp = slam[amp_idx];
                    double sl_re = sl_amp.real();
                    double sl_im = sl_amp.imag();
                    double T_re = (double)v_sl.x * sl_re - (double)v_sl.y * sl_im;
                    double T_im = (double)v_sl.x * sl_im + (double)v_sl.y * sl_re;
                    s_re += (double)w_val.x * T_re + (double)w_val.y * T_im;
                    s_im += (double)w_val.x * T_im - (double)w_val.y * T_re;
                }

                // d_dF 由 reComputeAmps 按全局事件索引写入（含 phsp/data/bkg 全段）
                int dF_base = (global_evt * nSLComb + sl_idx) * Nfree;
                for (int j = 0; j < Nfree; ++j) {
                    ctComplex dF = dF_s[dF_base + j];
                    acc[j] += -2.0 * sign * sg * (s_re * (double)dF.x - s_im * (double)dF.y);
                }
            }
        }
        for (int j = 0; j < Nfree; ++j)
            atomicAdd(&shm[j], acc[j]);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int j = 0; j < Nfree; ++j)
            atomicAdd(&d_grad[d_global_idx[j]], shm[j]);
}

// ---------------------------------------------------------------------------
// AmpCalc::computeEffectiveCoupling
// ---------------------------------------------------------------------------
void AmpCalc::computeEffectiveCoupling(const ctComplex* d_v, int n_amplitudes)
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
            cudaMalloc(&block.d_T[gpu], nTotal * sizeof(ctComplex));
        }

        int grid = (nTotal + kBlockSize - 1) / kBlockSize;
        computeEffectiveCouplingKernel<<<grid, kBlockSize>>>(
            block.d_T[gpu], cas->getSLAmps()[gpu],
            d_v + block.site, nSL, nTotal, 0, nSL);
        // 无 sync：同 stream 隐式顺序；调用方负责最终同步
    }
}

// 小 kernel：双精度数组累加 y[i] += alpha * x[i]
__global__ void daxpy_kernel(double* y, const double* x, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

// 计算 w[e,p] = sign * S[e,p] / I[e]（用于梯度计算：g = A^H * w）
__global__ void computeGradWeightKernel(
    ctComplex* d_w, const ctComplex* d_S, double* d_I,
    int nEv, int nPol, double sign)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEv) return;
    double I_val = 0.0;
    for (int p = 0; p < nPol; ++p) {
        ctComplex s = d_S[e * nPol + p];
        I_val += (double)s.x * s.x + (double)s.y * s.y;
    }
    d_I[e] = I_val;
    double inv_I = I_val > 1e-30 ? sign / I_val : 0.0;
    for (int p = 0; p < nPol; ++p) {
        ctComplex s = d_S[e * nPol + p];
        d_w[e * nPol + p] = ctMake((float)(inv_I * s.x), (float)(inv_I * s.y));  // sign * S/I
    }
}

// 对 bkg 事件乘上权重
__global__ void applyBkgWeightsKernel(
    ctComplex* d_w, const double* d_weights, int nEv, int nPol)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nEv) return;
    double wgt = d_weights ? d_weights[e] : 1.0;
    for (int p = 0; p < nPol; ++p) {
        int idx = e * nPol + p;
        d_w[idx].x *= ctCastFloat(wgt);
        d_w[idx].y *= ctCastFloat(wgt);
    }
}

// ---------------------------------------------------------------------------
// AmpCalc::computeResonanceGradient
// ---------------------------------------------------------------------------
void AmpCalc::computeResonanceGradient(
    const std::vector<ctComplex*>& d_w,
    const std::vector<int>& n_events,
    double* d_grad_res,
    double sign,
    const std::vector<int>& t_offset,
    const std::vector<ctComplex*>& d_v_per_gpu)
{
    int n_gpu = static_cast<int>(d_w.size());
    int n_free = nFreeResParams();
    if (n_free == 0 || blocks_.empty()) return;
    bool has_offset = (t_offset.size() == d_w.size());
    bool has_dv = (d_v_per_gpu.size() == d_w.size());
    int primary_dev = 0; cudaGetDevice(&primary_dev);  // d_grad_res lives here

    constexpr int kBlockSize = 256;

    // 每 GPU 的临时梯度 buffer：持久化懒分配，每次清零（避免每次 malloc/free 隐含同步）
    if (d_grad_per_gpu_.size() != (size_t)n_gpu) {
        d_grad_per_gpu_.assign(n_gpu, nullptr);
        for (int gpu = 0; gpu < n_gpu; ++gpu) {
            cudaSetDevice(gpu);
            cudaMalloc(&d_grad_per_gpu_[gpu], n_free * sizeof(double));
        }
    }
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        cudaSetDevice(gpu);
        cudaMemset(d_grad_per_gpu_[gpu], 0, n_free * sizeof(double));
    }

    // 启动梯度 kernel 的公共 lambda（每 block 一份）
    auto launch_block = [&](const ResBlock& block, const std::vector<int>& global_idx) {
        if (block.nFree == 0) return;
        auto& cas = cas_list_[block.cas_idx];
        int Nlocal = block.nFree;
        int nSL = static_cast<int>(cas->getNSLCombs());

        for (int gpu = 0; gpu < n_gpu; ++gpu) {
            cudaSetDevice(gpu);
            int nEv = n_events[gpu];
            if (nEv == 0) continue;
            int nPol = static_cast<int>(cas->getNPolarizations());
            int grid = (nEv + kBlockSize - 1) / kBlockSize;
            int evt_off = has_offset ? t_offset[gpu] : 0;
            int n_events_total = static_cast<int>(cas->getNEventsVec()[gpu]);
            const ctComplex* d_v_gpu = has_dv ? d_v_per_gpu[gpu] : nullptr;

            // free_global_idx 固定 → 持久化懒分配（conjugate 复用 owner 的指针）
            if (!block.d_global_idx_.size()) {
                // 从 owner 复制（reComputeAmps 已填充 host 侧 free_global_idx）
                if (block.free_global_idx.empty() && !global_idx.empty()) {
                    const_cast<ResBlock&>(block).free_global_idx = global_idx;
                }
                const_cast<ResBlock&>(block).d_global_idx_.assign(n_gpu, nullptr);
            }
            int* d_global_idx = block.d_global_idx_[gpu];
            if (!d_global_idx) {
                cudaMalloc(&d_global_idx, std::max(1, Nlocal) * sizeof(int));
                cudaMemcpy(d_global_idx, block.free_global_idx.data(),
                           Nlocal * sizeof(int), cudaMemcpyHostToDevice);
                const_cast<ResBlock&>(block).d_global_idx_[gpu] = d_global_idx;
            }

            int nSigma = cas->getNSigma();
            const double* d_sign_tab = cas->getSignsTab()[gpu];
            // P4 统一：所有 block 的 d_dF 由 computeCustomAmpsKernel（符号微分 aux）
            // 写入 [nSigma × nEv×nSL×Nfree] 布局，梯度统一走运行时 Nfree kernel
            resonanceGradientKernelRuntime<<<grid, kBlockSize>>>(
                d_w[gpu], cas->getSLAmpsTab()[gpu], d_v_gpu, block.d_dF[gpu],
                d_global_idx, d_grad_per_gpu_[gpu],
                nEv, nPol, nSL, sign, evt_off, block.site, n_events_total,
                cas->getNSigma(), d_sign_tab, Nlocal);
        }
    };

    // 主块：自由参数信息在 block.free_global_idx（addBlock 记录）。
    // conjugate 块跳过——它们在下面的 conjugate_broadcast_ 循环中处理
    // （否则贡献会被算两次：主循环 + conjugate 循环）
    std::set<int> cj_block_set;
    for (const auto& [cj_key, ow_key] : conjugate_broadcast_)
        cj_block_set.insert(cj_key.first);
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        auto& block = blocks_[bi];
        if (block.nFree == 0) continue;
        if (cj_block_set.count((int)bi)) continue;
        launch_block(block, block.free_global_idx);
    }

    // conjugate 块：用 cj_block 的 d_dF（参数已广播）+ owner 的全局索引
    for (const auto& [cj_key, owner_key] : conjugate_broadcast_) {
        auto& cj_block = blocks_[cj_key.first];
        auto& ow_block = blocks_[owner_key.first];
        if (ow_block.nFree == 0) continue;
        if (!cj_block.d_dF[0]) continue;  // 该块从未 AD 计算（无事件）
        launch_block(cj_block, ow_block.free_global_idx);
    }

    // 将各 GPU 结果通过 daxpy 累加到 d_grad_res（在 primary_dev 上）
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        if (gpu == primary_dev) {
            cudaSetDevice(primary_dev);
            int grid = (n_free + 255) / 256;
            daxpy_kernel<<<grid, 256>>>(d_grad_res, d_grad_per_gpu_[primary_dev], 1.0, n_free);
            cudaDeviceSynchronize();
        } else {
            std::vector<double> h_temp(n_free);
            cudaSetDevice(gpu);
            cudaMemcpy(h_temp.data(), d_grad_per_gpu_[gpu],
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

    cudaSetDevice(primary_dev);
}

void AmpCalc::computeUnifiedHessian(
    const std::vector<int>& n_events,
    double* d_hess, int hess_ld,
    const std::vector<int>& t_offset,
    double default_weight,
    const std::vector<ctComplex*>& d_v_per_gpu,
    const std::vector<ctComplex*>& d_amp_per_gpu,
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
        const ctComplex* d_v = d_v_per_gpu[gpu];
        const ctComplex* d_amp = d_amp_per_gpu[gpu];

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
            // Custom 模型（参数数运行时）→ 标量 Hessian kernel，无模板上限
            bool is_custom_block = !h_templates_[bi].empty() &&
                h_templates_[bi][0].type == ResModelType::Custom;
            // 内置模型有符号微分 aux（Nres=1）→ 同样走标量路径（P4 统一）
            bool has_sym_aux = !h_templates_[bi].empty() &&
                h_templates_[bi][0].aux_size > 0 && Nres == 1;
            if (is_custom_block || has_sym_aux) {
                // 仍需要 temp buffer（stage 2 交叉项）— 下方统一分配
            } else if (Npr < 1 || Npr > 3) {
                continue;
            }

            int NT = (is_custom_block || has_sym_aux) ? Npr : Npr * Nres;

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
            // dF 按 σ 行存储（全同粒子置换拓扑；σ=0 恒等）
            int nSigma = cas->getNSigma();
            cudaMalloc(&bt.d_dF_re, (size_t)nEv * nSL * Npr * nSigma * sizeof(double));
            cudaMalloc(&bt.d_dF_im, (size_t)nEv * nSL * Npr * nSigma * sizeof(double));
            cudaMalloc(&bt.d_gidx, NT * sizeof(int));
            cudaMemcpy(bt.d_gidx, global_idx.data(), NT * sizeof(int), cudaMemcpyHostToDevice);

            int grid = (nEv + kBlockSize - 1) / kBlockSize;

            const ctComplex* d_v_blk = d_v + blk.site;
            // Conjugate 块：phsp 积分已由 owner 计算，跳过 d_pI_g。
            // phsp 梯度/Hessian（d_pg_g, d_phA_g）通过 owner slot 累加。
            double* d_pI_ptr = (first_free_block && !is_conjugate) ? d_pI_g : nullptr;

            if (is_custom_block || has_sym_aux) {
                // Custom / 符号微分标量路径（P 运行时无上限）
                computeCustomHessianKernel<<<grid, kBlockSize>>>(
                    cas->getSLAmpsTab()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], blk.d_all_channels[gpu], bt.d_gidx,
                    d_hess_g, hess_ld, nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im,
                    bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off,
                    nSigma, cas->getMomentaTab()[gpu], cas->getSignsTab()[gpu]);
                cudaDeviceSynchronize();
                if (!is_conjugate) first_free_block = false;
                continue;   // 跳过模板分派
            }
            if (Npr == 2 && Nres == 2) {
                hessianStage1Kernel<2,2><<<grid, kBlockSize>>>(
                    cas->getSLAmpsTab()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], blk.d_all_channels[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off,
                    nSigma, cas->getMomentaTab()[gpu], cas->getSignsTab()[gpu]);
            } else if (Npr == 1 && Nres == 1) {
                hessianStage1Kernel<1,1><<<grid, kBlockSize>>>(
                    cas->getSLAmpsTab()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], blk.d_all_channels[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off,
                    nSigma, cas->getMomentaTab()[gpu], cas->getSignsTab()[gpu]);
            } else if (Npr == 2 && Nres == 1) {
                hessianStage1Kernel<2,1><<<grid, kBlockSize>>>(
                    cas->getSLAmpsTab()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], blk.d_all_channels[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off,
                    nSigma, cas->getMomentaTab()[gpu], cas->getSignsTab()[gpu]);
            } else if (Npr == 1 && Nres == 2) {
                hessianStage1Kernel<1,2><<<grid, kBlockSize>>>(
                    cas->getSLAmpsTab()[gpu], d_v_blk,
                    cas->getMomenta()[gpu], cas->getDecayNodes()[gpu], dsz,
                    cas->getDeviceSLCombs()[gpu], blk.d_resonances[gpu],
                    blk.d_all_params[gpu], blk.d_all_channels[gpu], bt.d_gidx, d_hess_g, hess_ld,
                    nEv, nSL, nPol, 3.0, default_weight, d_w,
                    d_S_re, d_S_im, bt.d_g, bt.d_dS_re, bt.d_dS_im, bt.d_dF_re, bt.d_dF_im,
                    d_pI_ptr, d_pg_g, d_phA_g, evt_off,
                    nSigma, cas->getMomentaTab()[gpu], cas->getSignsTab()[gpu]);
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
                    cas_list_[blk.cas_idx]->getSLAmpsTab()[gpu],
                    bt.d_g, bt.d_dS_re, bt.d_dS_im,
                    bt.d_dF_re, bt.d_dF_im, bt.d_gidx,
                    d_mix_g, nFreeResParams(),
                    nEv, nSL, Npr, nPol, n_amp_total, blk.site,
                    nTotal_slamp, default_weight, d_w, d_msum_g, evt_off,
                    cas_list_[blk.cas_idx]->getNSigma(),
                    cas_list_[blk.cas_idx]->getSignsTab()[gpu]);
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
    ctComplex* d_v, const double* d_params,
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

    d_v[a].x = ctCastFloat(re);
    d_v[a].y = ctCastFloat(im);
}

// Gradient transform: ∂L/∂p using Wirtinger calculus
// ∂L/∂p_j* = (1/p_j) Σ_{a∈A_j} grad_v[a]* · v[a]
// d_params, d_grad_p: [Re_0..Re_{n-1}, Im_0..Im_{n-1}] (all Re, all Im)
__global__ void multiplicativeGradientKernel(
    double* d_grad_p,           // output [2·n_free] — [Re, Im] format
    const ctComplex* d_grad_v,  // ∂L/∂v [n_amps]
    const ctComplex* d_v,       // current v [n_amps]
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
