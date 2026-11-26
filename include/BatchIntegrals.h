// include/BatchIntegrals.h — 分批全样本积分（分波/散射矩阵 + 雅可比扰动）
//
// P4：`analysis::computeFileBatchIntegrals` 有 247 行，是份额/效率/误差矩阵/写结果
// 共同的底层。它读 9 个成员、调 2 个成员函数，并且会**临时改写**
// `events_offsets_/amp_offsets_`（批扫描后还原）。按 P4 模板：
//   读的状态 → 只读字段；要改的状态 → 指针；算法 → 回调。
#pragma once

#include "ComplexType.h"

#include <Amplitude.cuh>   // LorentzVector
#include <Config.cuh>      // Particle

#include <torch/extension.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ctpwa {

struct BatchIntegralCtx {
    // ---- 只读配置 ----
    const std::map<std::string, std::vector<std::string>>* data_files = nullptr;
    const std::vector<std::string>* data_order = nullptr;
    const std::vector<Particle>* particles = nullptr;
    // ---- 布局/设备 ----
    std::vector<std::vector<int>>* events_offsets = nullptr;  // 批扫描会临时改写后还原
    std::vector<std::vector<int>>* amp_offsets = nullptr;
    const std::vector<int>* n_slvectors = nullptr;
    int n_amplitudes = 0;
    int n_polar = 0;
    int n_gpus = 0;
    int primary_dev = 0;
    bool float_amps = false;
    // ---- 算法回调 ----
    std::function<std::vector<double>()> current_theta_host;
    // 用显式 θ 覆盖算振幅（Vp4 → 每 GPU 的振幅缓冲）
    std::function<std::vector<ctComplex*>(
        const std::vector<std::map<std::string, std::vector<LorentzVector>>>&,
        int, const double*)> calculate_amplitudes;
};

// 返回该文件 key 的总权重（Σw）；积分结果写进 out_*（见调用点的块布局说明）。
double computeFileBatchIntegrals(const BatchIntegralCtx& ctx,
                                 const std::string& file_key,
                                 const torch::Tensor& ev_center,
                                 const std::vector<torch::Tensor>& ev_perturbed_p,
                                 const std::vector<torch::Tensor>& ev_perturbed_m,
                                 double* out_partial,
                                 double* out_scattering,
                                 double* out_square,
                                 double* out_p_partial,
                                 double* out_p_scattering,
                                 double* out_m_partial,
                                 double* out_m_scattering,
                                 int npartials);

}  // namespace ctpwa
