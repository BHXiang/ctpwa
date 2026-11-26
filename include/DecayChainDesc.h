// include/DecayChainDesc.h — 一条衰变链的描述（纯 host 值类型，无设备依赖）
//
// modularization-plan P3 的第二步：`AmpCasDecay` 里"这条链长什么样"与"SL 表怎么算/"
// "显存怎么管"原本混在一起。把描述部分抽成这个类之后：
//
//   * 构造一条链只需要 Amp2BD 与粒子信息，**不需要 GPU、不需要 config 文件**；
//   * 描述可以被拷贝/比较/单测（例如"同一对粒子重复出现时只登记一次"）；
//   * `AmpCasDecay` 从此是 `DecayChainDesc` + SL 表（has-a，而非自己维护 4 个容器）。
//
// 判据：本头不出现任何设备 API / 设备类型，也不 include AmpGen.cuh。
#pragma once

#include <DecayDesc.h>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace ctpwa {

// 一个粒子的静态描述（与 config 的 Particle 解耦：只留链需要的信息）
struct ParticleDesc {
    int spin = 1;      // 2J+1
    int parity = 1;
    double mass = -1.0;
};

class DecayChainDesc {
public:
    // 登记一个粒子（同名只登记一次，保持首次出现顺序）
    void addParticle(const std::string& name, int spin, int parity, double mass)
    {
        if (particleMap_.find(name) != particleMap_.end()) return;
        particleMap_[name] = ParticleDesc{spin, parity, mass};
        particleNames_.push_back(name);
    }

    // 追加一步两体衰变；n_sl = 该步允许的 (S,L) 组合数（决定总波数乘积）
    void addNode(DecayNodeHost node, std::size_t n_sl)
    {
        decayChain_.push_back(std::move(node));
        nSLCombs_ *= (n_sl == 0 ? 1 : n_sl);
    }

    // 表侧算出的实际波数（可能因过滤/去重与乘积不同）
    void setNSLCombs(std::size_t n) { nSLCombs_ = n; }

    const std::vector<DecayNodeHost>& nodes() const { return decayChain_; }
    const std::map<std::string, ParticleDesc>& particles() const { return particleMap_; }
    const std::vector<std::string>& particleNames() const { return particleNames_; }
    std::size_t nSLCombs() const { return nSLCombs_; }
    std::size_t size() const { return decayChain_.size(); }
    bool empty() const { return decayChain_.empty(); }

private:
    std::vector<DecayNodeHost> decayChain_;
    std::map<std::string, ParticleDesc> particleMap_;
    std::vector<std::string> particleNames_;
    std::size_t nSLCombs_ = 1;
};

}  // namespace ctpwa
