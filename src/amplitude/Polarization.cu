// src/Polarization.cu — 见 include/Polarization.h
#include <Polarization.h>

#include <iostream>

namespace ctpwa {

PolarizationInit initializePolarization(const std::vector<Particle>& particles)
{
    PolarizationInit res;

        int n = particles.size();

        // 收集每粒子的维度 (2J+1) 和允许的张量索引列表
        std::vector<int> dims(n);
        std::vector<std::vector<int>> keep(n); // 空 = 保留全部
        bool any_masked = false;

        for (int i = 0; i < n; ++i) {
            dims[i] = particles[i].spin;
            if (particles[i].is_polarized()) {
                any_masked = true;
                int dim_j = dims[i];
                for (int two_m : particles[i].polarization_2m) {
                    keep[i].push_back((dim_j - 1 - two_m) / 2);
                }
            }
        }

        // 原始 C-order strides: orig_strides[i] = prod_{j>i} dims[j]
        std::vector<int> orig_strides(n);
        orig_strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; --i)
            orig_strides[i] = orig_strides[i + 1] * dims[i + 1];

        // 总张量极化数
        res.n_polar_total = 1;
        for (int d : dims) res.n_polar_total *= d;

        // 有效维度（mask 后）和 masked strides
        std::vector<int> eff_dims(n);
        std::vector<int> masked_strides(n);
        for (int i = 0; i < n; ++i)
            eff_dims[i] = keep[i].empty() ? dims[i] : (int)keep[i].size();
        masked_strides[n - 1] = 1;
        for (int i = n - 2; i >= 0; --i)
            masked_strides[i] = masked_strides[i + 1] * eff_dims[i + 1];

        // 有效极化态数
        res.n_polar = 1;
        for (int d : eff_dims) res.n_polar *= d;

        // 构建完整映射: output_idx → tensor_idx
        if (any_masked) {
            res.map.resize(res.n_polar);
            for (int out = 0; out < res.n_polar; ++out) {
                int rem = out;
                int tensor_idx = 0;
                for (int p = 0; p < n; ++p) {
                    int j = rem / masked_strides[p];
                    rem %= masked_strides[p];
                    int i = keep[p].empty() ? j : keep[p][j];
                    tensor_idx += i * orig_strides[p];
                }
                res.map[out] = tensor_idx;
            }
        }
        else {
            res.map.clear();
        }

        std::cout << "polarization: " << res.n_polar;
        // std::cout  << " (total tensor states: " << res.n_polar_total << ")";
        // if (!res.map.empty()) {
        //     std::cout << ", map: [";
        //     for (size_t i = 0; i < res.map.size(); ++i) {
        //         if (i > 0) std::cout << ", ";
        //         std::cout << res.map[i];
        //     }
        //     std::cout << "]";
        // }
        std::cout << std::endl;
    
    return res;
}

}  // namespace ctpwa
