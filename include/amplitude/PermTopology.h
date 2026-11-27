// include/PermTopology.h — 全同粒子置换拓扑（coset）：纯 host，无设备依赖
//
// 为什么单独成文件（modularization-plan P3）
// ------------------------------------------
// 全同粒子置换拓扑是**衰变描述**的性质，与 SL 振幅表、GPU 缓冲无关。它原来住在
// `AmpCasDecay::buildPermTopologies()` 里，只有构造一个持有设备表的对象才能算——
// 于是这段纯组合逻辑既不能脱离 GPU 单测，也让"衰变描述"和"设备生命周期"更难分开。
//
// 现在：输入只有衰变链 + 粒子索引 + 全同组，输出是每 σ 的重建映射与符号。
#pragma once

#include <DecayDesc.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ctpwa {

// σ=0（恒等）**不在** maps 里：maps 的第 k 项对应 signs 的第 k+1 项。
struct PermTopology {
    std::vector<std::map<std::string, int>> maps;  // [nSigma-1] 每 σ 的 name→idx 重建映射
    std::vector<double> signs;                     // [nSigma]，signs[0] 恒为 +1

    int nSigma() const { return static_cast<int>(signs.size()); }
    int nPerm() const { return static_cast<int>(maps.size()); }
    bool empty() const { return signs.empty(); }
};

struct PermTopologyInputs {
    const std::vector<DecayNodeHost>& decayChain;
    const std::map<std::string, int>& particleToIndex;
    const std::vector<std::pair<std::vector<std::string>, bool>>& identicalGroups;
    const std::string& chainName;
};

PermTopology buildPermTopology(const PermTopologyInputs& in);

}  // namespace ctpwa
