// include/Polarization.h — 初态极化张量的索引映射（纯 host 组合数学）
//
// P4：`analysis::initializePolarization` 只依赖 `particles_`（每个粒子的 2J+1 与
// 允许的 2m 列表），却写在 god class 里。抽出来之后它成了纯函数——给定粒子表，
// 就能算出张量维数、有效极化数与 output→tensor 的索引映射，可脱离 GPU 单测。
#pragma once

#include <Config.cuh>   // Particle

#include <vector>

namespace ctpwa {

struct PolarizationInit {
    int n_polar_total = 1;   // 未做 mask 的张量极化总数
    int n_polar = 1;         // mask 后的有效极化数
    std::vector<int> map;    // output_idx → tensor_idx；空 = 未做 mask（恒等）
};

PolarizationInit initializePolarization(const std::vector<Particle>& particles);

}  // namespace ctpwa
