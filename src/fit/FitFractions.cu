// src/FitFractions.cu — 见 include/FitFractions.h
#include <FitFractions.h>

#include "BackendHelpers.h"
#include "DeviceGuard.h"
#include "MatrixStats.h"

#include <ComputeBF.cuh>   // computeBFErrors

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace ctpwa {

torch::Tensor fitFractions(const FitFractionCtx& ctx, const torch::Tensor& vector,
                           const torch::Tensor& hessian_in)
{
        ctpwa::ensure_device(ctx.primary_dev);   // 入口恢复主设备（调用方可能刚从多 GPU 段返回）

        // 未配置 phsp_truth(无效率相空间 MC)时天然返回空张量 [0,2]:
        // 早期拟合不带 mctruth, fit.py 据此自然跳过, 不抛异常、不触碰任何相关 kernel。
        // 早退放在 Tensor 检查**之前**：调用方探测"有没有份额"时不该被 dtype/device
        // 挡住（以前 CPU 构建传 CPU Hessian 会被下面的 is_cuda 检查误伤）。
        const auto& data_files = (*ctx.data_files);
        if (data_files.count("phsp_truth") == 0) {
            return torch::empty({0, 2},
                torch::TensorOptions().dtype(torch::kFloat64));
        }

        // computeTruthIntegrals 内已做 float2→double 上转: 两种精度共用双精度积分,
        // 分支比/效率及 Jacobian 结果与 precision:double 数值一致。
        TORCH_CHECK(ctpwa_is_cuda_like(vector), "vector 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision (double/float 编译)");
        if (hessian_in.numel() > 0)
            TORCH_CHECK(hessian_in.dtype() == torch::kFloat64 && ctpwa_is_cuda_like(hessian_in),
                "hessian_in must be float64 on the backend device (CPU build: CPU tensor)");

        // 自由耦合 p → 扩展振幅 v（乘法链映射），Jacobian/Hessian 在 p 空间
        torch::Tensor ev_center = ctx.free_params_to_amplitudes(vector);
        const int n = vector.numel();
        const int n2 = 2 * n;
        const int npartials = (*ctx.n_slvectors).size();
        torch::Device dev = vector.device();

        // Perturbed vectors for Jacobian (p 空间扰动 → v 空间积分)。
        // grouped 布局 [Re_p(0..n-1), Im_p(n..2n-1)]，与 getHessian 一致。
        // 相对步长: 绝对 5e-6 在 |p_j| ≳ 10 时低于
        // float32 ulp，差分退化为舍入噪声。
        auto vr = torch::view_as_real(vector);            // [n, 2] float32 interleaved
        torch::Tensor p_real = torch::cat({
            vr.slice(1, 0, 1).reshape({ -1 }),
            vr.slice(1, 1, 2).reshape({ -1 }),
            }).contiguous();                                   // [2n] grouped
        std::vector<double> eps_j(n2, 5e-6);
        std::vector<torch::Tensor> ev_perturbed_p(n2), ev_perturbed_m(n2);
        auto grouped_to_complex = [&](const torch::Tensor& g) {
            return torch::view_as_complex(
                torch::stack({ g.slice(0, 0, n), g.slice(0, n, 2 * n) }, 1).contiguous());
            };
        for (int j = 0; j < n2; ++j) {
            eps_j[j] = 5e-6 * std::max(std::abs((double)p_real[j].item<float>()), 1.0);
            auto vp = p_real.clone(); vp[j] += eps_j[j];
            auto vm = p_real.clone(); vm[j] -= eps_j[j];
            ev_perturbed_p[j] = ctx.free_params_to_amplitudes(grouped_to_complex(vp));
            ev_perturbed_m[j] = ctx.free_params_to_amplitudes(grouped_to_complex(vm));
        }

        // Truth integrals (center + Jacobian); only partial integrals needed
        std::vector<double> truth_c_partial(npartials, 0.0);
        std::vector<double> truth_c_scattering(npartials * npartials, 0.0);
        std::vector<double> truth_p_partial(n2 * npartials, 0.0);
        std::vector<double> truth_p_scattering(n2 * npartials * npartials, 0.0);
        std::vector<double> truth_m_partial(n2 * npartials, 0.0);
        std::vector<double> truth_m_scattering(n2 * npartials * npartials, 0.0);
        // 返回值 Σw（无权重时 = 事件数）这里用不到: FF 是同一权重样本内的比值。
        ctx.batch_integrals("phsp_truth", ev_center,
            ev_perturbed_p, ev_perturbed_m,
            truth_c_partial.data(), truth_c_scattering.data(), nullptr,
            truth_p_partial.data(), truth_p_scattering.data(),
            truth_m_partial.data(), truth_m_scattering.data(), npartials);

        // FF_i = truth_partial[i] / Σ_j truth_partial[j]
        double truth_total = 0.0;
        for (int i = 0; i < npartials; ++i) truth_total += truth_c_partial[i];
        if (!(truth_total > 0.0) || !std::isfinite(truth_total)) {
            // phsp_truth 文件为空（或振幅退化）时，truth 积分全 0。
            // 与"缺 phsp_truth"一致返回空，不能把 0/0 伪装成 FF=0。
            return torch::empty({0, 2},
                torch::TensorOptions().dtype(torch::kFloat64));
        }

        std::vector<double> ff_center(npartials);
        for (int i = 0; i < npartials; ++i)
            ff_center[i] = truth_c_partial[i] / truth_total;

        // ===== Errors: FF_error = sqrt(diag(J_FF @ H^{-1} @ J_FF^T)) =====
        // J_FF[i][j] = (FF_p[i] - FF_m[i]) / (2·eps_j[j])
        // 固定参考参数方向同上: mask 掉再检查正定性/求逆。
        // Hessian 优先用传入的统一 Hessian（与拟合正定判定同源）。
        std::vector<double> ff_errors(npartials, 0.0);
        // hessian_in 是统一 Hessian（含 θ）时，couplingCovariance 会给边际协方差；
        // 只有 [2n,2n] 的耦合 Hessian 时才退化为条件协方差（无 θ 信息可用）。
        torch::Tensor hessian;
        if (hessian_in.numel() > 0)
            hessian = hessian_in;
        else
            hessian = ctx.coupling_hessian(vector);
        int n_free = 0;
        bool cov_ok = false;
        torch::Tensor cov = ctpwa::couplingCovariance(hessian, n, n_free, cov_ok);
        if (n_free > 0 && cov_ok) {
            std::vector<int64_t> free_idx;
            for (int j = 0; j < n2; ++j)
                if (j != 0 && j != n) free_idx.push_back(j);
            TORCH_CHECK((int)free_idx.size() == n_free,
                "getFitFractions: 自由方向计数不一致");
            std::vector<double> h_cov((size_t)n_free * n_free);
            std::memcpy(h_cov.data(), cov.data_ptr<double>(),
                        h_cov.size() * sizeof(double));

            std::vector<double> J(npartials * n_free, 0.0);
            for (int fj = 0; fj < n_free; ++fj) {
                const int j = free_idx[fj];
                double tot_p = 0.0, tot_m = 0.0;
                for (int i = 0; i < npartials; ++i) {
                    tot_p += truth_p_partial[j * npartials + i];
                    tot_m += truth_m_partial[j * npartials + i];
                }
                if (tot_p <= 0.0) tot_p = 1e-30;
                if (tot_m <= 0.0) tot_m = 1e-30;
                for (int i = 0; i < npartials; ++i) {
                    double ff_p = truth_p_partial[j * npartials + i] / tot_p;
                    double ff_m = truth_m_partial[j * npartials + i] / tot_m;
                    J[i * n_free + fj] = (ff_p - ff_m) / (2.0 * eps_j[j]);
                }
            }
            computeBFErrors(J.data(), h_cov.data(), ff_errors.data(),
                            npartials, n_free);
        } else if (n_free > 0) {
            std::cerr << "[ctpwa] getFitFractions: Hessian 自由块非正定/求逆失败，"
                         "跳过参数误差" << std::endl;
        }
        // 返回 n×2: [center, error]
        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor result = torch::empty({ npartials, 2 }, opts);
        for (int i = 0; i < npartials; ++i) {
            result[i][0] = ff_center[i];
            result[i][1] = ff_errors[i];
        }
        return result;
    
}

torch::Tensor efficiency(const FitFractionCtx& ctx, const torch::Tensor& vector,
                         const torch::Tensor& hessian_in)
{
        ctpwa::ensure_device(ctx.primary_dev);   // 入口恢复主设备

        // 需要 phsp（带效率 MC）与 phsp_truth（无效率 MC）两个数据键;
        // 缺任一 → 天然返回空张量 [0,2]（同 getFitFractions 的缺数据行为）。
        // 早退放在 Tensor 检查之前：探测"有没有效率"时不该被 dtype/device 挡住。
        const auto& data_files = (*ctx.data_files);
        if (data_files.count("phsp_truth") == 0 || data_files.count("phsp") == 0) {
            return torch::empty({0, 2},
                torch::TensorOptions().dtype(torch::kFloat64));
        }

        TORCH_CHECK(ctpwa_is_cuda_like(vector), "vector 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision (double/float 编译)");
        if (hessian_in.numel() > 0)
            TORCH_CHECK(hessian_in.dtype() == torch::kFloat64 && ctpwa_is_cuda_like(hessian_in),
                "hessian_in must be float64 on the backend device (CPU build: CPU tensor)");

        torch::Tensor ev_center = ctx.free_params_to_amplitudes(vector);
        const int n = vector.numel();
        const int n2 = 2 * n;
        const int npartials = (*ctx.n_slvectors).size();

        // Perturbed vectors for Jacobian（与 getFitFractions 相同约定）
        auto vr = torch::view_as_real(vector);
        torch::Tensor p_real = torch::cat({
            vr.slice(1, 0, 1).reshape({ -1 }),
            vr.slice(1, 1, 2).reshape({ -1 }),
            }).contiguous();
        std::vector<double> eps_j(n2, 5e-6);
        std::vector<torch::Tensor> ev_perturbed_p(n2), ev_perturbed_m(n2);
        auto grouped_to_complex = [&](const torch::Tensor& g) {
            return torch::view_as_complex(
                torch::stack({ g.slice(0, 0, n), g.slice(0, n, 2 * n) }, 1).contiguous());
            };
        for (int j = 0; j < n2; ++j) {
            eps_j[j] = 5e-6 * std::max(std::abs((double)p_real[j].item<float>()), 1.0);
            auto vp = p_real.clone(); vp[j] += eps_j[j];
            auto vm = p_real.clone(); vm[j] -= eps_j[j];
            ev_perturbed_p[j] = ctx.free_params_to_amplitudes(grouped_to_complex(vp));
            ev_perturbed_m[j] = ctx.free_params_to_amplitudes(grouped_to_complex(vm));
        }

        // phsp（带效率）与 phsp_truth（无效率）各做一次批积分（中心+扰动+平方）
        std::vector<double> eff_c_partial(npartials, 0.0);
        std::vector<double> eff_c_scattering(npartials * npartials, 0.0);
        std::vector<double> eff_c_square(npartials, 0.0);
        std::vector<double> eff_p_partial(n2 * npartials, 0.0);
        std::vector<double> eff_p_scattering(n2 * npartials * npartials, 0.0);
        std::vector<double> eff_m_partial(n2 * npartials, 0.0);
        std::vector<double> eff_m_scattering(n2 * npartials * npartials, 0.0);
        double W_eff = ctx.batch_integrals("phsp", ev_center,
            ev_perturbed_p, ev_perturbed_m,
            eff_c_partial.data(), eff_c_scattering.data(), eff_c_square.data(),
            eff_p_partial.data(), eff_p_scattering.data(),
            eff_m_partial.data(), eff_m_scattering.data(), npartials);

        std::vector<double> tr_c_partial(npartials, 0.0);
        std::vector<double> tr_c_scattering(npartials * npartials, 0.0);
        std::vector<double> tr_c_square(npartials, 0.0);
        std::vector<double> tr_p_partial(n2 * npartials, 0.0);
        std::vector<double> tr_p_scattering(n2 * npartials * npartials, 0.0);
        std::vector<double> tr_m_partial(n2 * npartials, 0.0);
        std::vector<double> tr_m_scattering(n2 * npartials * npartials, 0.0);
        double W_truth = ctx.batch_integrals("phsp_truth", ev_center,
            ev_perturbed_p, ev_perturbed_m,
            tr_c_partial.data(), tr_c_scattering.data(), tr_c_square.data(),
            tr_p_partial.data(), tr_p_scattering.data(),
            tr_m_partial.data(), tr_m_scattering.data(), npartials);

        if (W_eff <= 0.0 || W_truth <= 0.0) {
            return torch::empty({0, 2},
                torch::TensorOptions().dtype(torch::kFloat64));
        }

        // 归一化分母 = Σw（无权重时 = 事件数），与 NLL 的 Σw·f/Σw 同口径
        const double scale_eff = 1.0 / W_eff, scale_tr = 1.0 / W_truth;
        auto effOf = [&](double num_eff, double num_tr) -> double {
            double den = num_tr * scale_tr;
            if (den <= 0.0) return 0.0;
            return (num_eff * scale_eff) / den;
        };
        std::vector<double> eff_center(npartials);
        for (int i = 0; i < npartials; ++i)
            eff_center[i] = effOf(eff_c_partial[i], tr_c_partial[i]);

        // ===== 误差: σ² = σ_param² + σ_stat² =====
        std::vector<double> eff_errors(npartials, 0.0);

        // 参数误差（同 getFitFractions: 联合 Hessian 时对 θ 取边际协方差）
        torch::Tensor hessian;
        if (hessian_in.numel() > 0)
            hessian = hessian_in;
        else
            hessian = ctx.coupling_hessian(vector);
        int n_free = 0;
        bool cov_ok = false;
        torch::Tensor cov = ctpwa::couplingCovariance(hessian, n, n_free, cov_ok);
        if (n_free > 0 && cov_ok) {
            std::vector<int64_t> free_idx;
            for (int j = 0; j < n2; ++j)
                if (j != 0 && j != n) free_idx.push_back(j);
            TORCH_CHECK((int)free_idx.size() == n_free,
                "getEfficiency: 自由方向计数不一致");
            std::vector<double> h_cov((size_t)n_free * n_free);
            std::memcpy(h_cov.data(), cov.data_ptr<double>(),
                        h_cov.size() * sizeof(double));

            std::vector<double> J(npartials * n_free, 0.0);
            for (int fj = 0; fj < n_free; ++fj) {
                const int j = free_idx[fj];
                for (int i = 0; i < npartials; ++i) {
                    double e_p = effOf(eff_p_partial[j * npartials + i],
                                       tr_p_partial[j * npartials + i]);
                    double e_m = effOf(eff_m_partial[j * npartials + i],
                                       tr_m_partial[j * npartials + i]);
                    J[i * n_free + fj] = (e_p - e_m) / (2.0 * eps_j[j]);
                }
            }
            computeBFErrors(J.data(), h_cov.data(), eff_errors.data(),
                            npartials, n_free);
        } else if (n_free > 0) {
            std::cerr << "[ctpwa] getEfficiency: Hessian 自由块非正定/求逆失败，"
                         "跳过参数误差" << std::endl;
        }

        // MC 统计误差（tf-pwa add_int_error 同款）:
        //   σ_stat_i² = ε_i² × ( S_eff / I_eff² + S_tr / I_tr² )
        //   S = Σ(w|A_i|²)², I = Σ w|A_i|²（未归一化；每样本的 Σw 均值因子相消,
        //   与 tf-pwa cached_square（_eval_integral 里也是 Σ(w·f)²）/ int_total² 一致）
        for (int i = 0; i < npartials; ++i) {
            double Ie = eff_c_partial[i];   // Σ w|A_i|² (未归一化)
            double It = tr_c_partial[i];
            double stat2 = 0.0;
            if (Ie > 0.0) stat2 += eff_c_square[i] / (Ie * Ie);
            if (It > 0.0) stat2 += tr_c_square[i] / (It * It);
            double e = eff_center[i];
            double total2 = eff_errors[i] * eff_errors[i] + e * e * stat2;
            eff_errors[i] = std::sqrt(std::max(0.0, total2));
        }

        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor result = torch::empty({ npartials, 2 }, opts);
        for (int i = 0; i < npartials; ++i) {
            result[i][0] = eff_center[i];
            result[i][1] = eff_errors[i];
        }
        return result;
    
}

torch::Tensor matrixForSample(const FitFractionCtx& ctx, const torch::Tensor& vector,
                              std::string sample)
{
        ctpwa::ensure_device(ctx.primary_dev);
        if (sample == "truth") sample = "phsp_truth";
        if (sample == "phsp") {
            // 重建侧不再读文件：常驻 B̄ 收缩（B̄ 在构造 / reCalcAmp / writeResult 时刷新）
            TORCH_CHECK(ctpwa_is_cuda_like(vector), "vector 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
            TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision");
            return ctx.phsp_matrix_from_gram(ctx.free_params_to_amplitudes(vector));
        }

        TORCH_CHECK(sample == "phsp_truth" || sample == "phsp",
            "sample 只能是 'truth' (phsp_truth) 或 'phsp'，收到 '", sample, "'");
        const auto& data_files = (*ctx.data_files);
        if (data_files.count(sample) == 0) {
            return torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kFloat64));
        }
        TORCH_CHECK(ctpwa_is_cuda_like(vector), "vector 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision (double/float 编译)");

        torch::Tensor ev_center = ctx.free_params_to_amplitudes(vector);
        const int npartials = (*ctx.n_slvectors).size();
        std::vector<double> partial(npartials, 0.0);
        std::vector<double> scattering(npartials * npartials, 0.0);

        // 无扰动方向（nullptr）: 只算中心值，省掉 2·n_free 遍积分
        ctx.batch_integrals(sample, ev_center,
            std::vector<torch::Tensor>{}, std::vector<torch::Tensor>{},
            partial.data(), scattering.data(), nullptr,
            nullptr, nullptr, nullptr, nullptr, npartials);

        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor result = torch::zeros({ npartials, npartials }, opts);
        // 散射矩阵的对角是 2|A_i|²（与 computeBFKernel 的约定），Gram 对角要用 partial
        for (int i = 0; i < npartials; ++i) {
            result[i][i] = partial[i];
            for (int j = i + 1; j < npartials; ++j) {
                result[i][j] = scattering[i * npartials + j];
                result[j][i] = scattering[i * npartials + j];
            }
        }
        return result;
    
}

}  // namespace ctpwa
