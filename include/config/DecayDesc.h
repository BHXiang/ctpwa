// include/DecayDesc.h — 衰变描述类型（纯 host 语义，与设备/SL 表无关）
//
// modularization-plan P3 的第一步：把 `SL`、`Amp2BD`、`DecayNodeHost` 从
// `AmpGen.cuh`（那个头里塞着 SL 振幅表、设备结构体、JIT 状态）里拆出来。
//
// 判据：本头只依赖 C++ 标准库；不出现 cuda/thrust/torch 类型。
// 这样"一条衰变链长什么样"可以在没有 GPU、没有 SL 表的上下文里被构造和检查，
// 后续 `DecayChainDesc`（P3 的下一步）才有地方落脚。
#pragma once

#include "CudaCompat.h"   // SL 需要 __host__ __device__ 属性

#include <array>
#include <set>
#include <string>
#include <utility>
#include <vector>

// 自旋-轨道组合结构体
struct SL {
    int S; // 2s+1
    int L;
    __host__ __device__ SL(int s = 0, int l = 0) : S(s), L(l) {}
};

// 两体衰变振幅类
class Amp2BD {
public:
    Amp2BD(std::array<int, 3> jvalues, std::array<int, 3> parities,
           bool identical_daughters = false, bool is_boson = true,
           int maxL = -1, bool p_break = false, bool has_bf = true,
           double bf_d = 3.0,
           std::set<std::pair<int, int>> sl_filter = {});
    const std::vector<SL>& getSL() const { return spinOrbitCombinations_; }
    const std::array<int, 3>& getJValues() const { return jvalues_; }
    const std::array<int, 3>& getParities() const { return parities_; }
    bool getPbreak() const { return p_break_; }
    bool getHasBf() const { return has_bf_; }
    double getBfD() const { return bf_d_; }

private:
    std::vector<SL> ComSL(const std::array<int, 3>& spins,
        const std::array<int, 3>& parities);
    std::array<int, 3> jvalues_;
    std::array<int, 3> parities_;
    bool identical_daughters_;
    bool is_boson_; // true=boson(symmetric, +), false=fermion(antisymmetric, -)
    int maxL_;      // 轨道角动量上限; -1 = 无限制
    bool p_break_;  // 宇称是否破缺（弱衰变）
    bool has_bf_;   // 是否有势垒因子（三级作用域决议后的最终值）
    double bf_d_;   // 势垒半径 d
    std::set<std::pair<int, int>> sl_filter_; // 允许的 (S, L) 分波白名单; 空 = 全允许
    std::vector<SL> spinOrbitCombinations_;
};

// 一条衰变链上的一步（host 侧描述；设备端对应 DecayNode）
struct DecayNodeHost {
    Amp2BD amp;
    std::string mother;
    std::string daug1;
    std::string daug2;
};
