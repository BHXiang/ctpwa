// src/MatrixErrors.cu — 见 include/MatrixErrors.h
#include <MatrixErrors.h>

#include "ComplexType.h"
#include "BackendHelpers.h"
#include "MatrixStats.h"
#include "PrecCpx.h"

#include <cuda_runtime.h>
#include "DeviceGuard.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace ctpwa {

std::vector<torch::Tensor> computeMatrixErrors(const MatrixErrorCtx& ctx,
                                               const torch::Tensor& params,
                                               const torch::Tensor& hessian_in)
{
        ctpwa::ensure_device(ctx.primary_dev);
        TORCH_CHECK(ctpwa_is_cuda_like(params) && params.dtype() == torch::kFloat64,
            "params 必须是当前后端设备上的 float64 统一参数 [Re p, Im p, θ]");
        const int n = ctx.n_vector();
        const int ntheta = ctx.n_theta;
        TORCH_CHECK(params.numel() == 2 * n + ntheta,
            "params 长度 ", params.numel(), " != 2*n_coupling + n_theta = ",
            2 * n + ntheta);
        const int npartials = (*ctx.n_slvectors).size();
        auto opts2d = torch::TensorOptions().dtype(torch::kFloat64);
        // 缺 phsp_truth（只有重建侧样本）时仍给 rec 侧误差，gen 侧返回空。
        // 空样本/零相干总额在中心扫描后也会把 has_gen 降为 false。
        bool has_gen = ctx.has_gen;
        // 误差扫描会在内部反复改 theta_cur_（每个 θ 方向的 ±ε）。出函数必须还原，
        // 否则后续 getFitFractionMatrix / getFitFractions / writeResult 会拿扰动过的
        // θ 算振幅（实测残留会让份额第 5 位就偏）。
        const torch::Tensor theta_saved = ctx.get_theta();
        auto restoreTheta = [&]() { ctx.set_theta(theta_saved); };
        // 异常路径也必须还原 θ：runMatrices/逐方向扫描会改 theta_cur_。
        auto theta_guard = makeScopeExit([&]() { ctx.set_theta(theta_saved); });
        // 读 B̄ 权重时按 GPU 循环会把 current device 留在最后一张卡。
        auto dev_guard = makeScopeExit([&]() { ctpwa::ensure_device(ctx.primary_dev); });

        // ---- 统一 Hessian → Cov（耦合自由方向的边际协方差）----
        torch::Tensor H = hessian_in;
        if (H.numel() == 0) H = ctx.hessian(params);
        TORCH_CHECK(H.dtype() == torch::kFloat64, "hessian 必须是 float64");
        TORCH_CHECK(H.size(0) == 2 * n + ntheta, "hessian 维度与 params 不一致: ",
            H.size(0), " vs ", 2 * n + ntheta);

        // J 的列只放耦合方向（θ 的 FD 在 M~1e4 时被舍入噪声主导），
        // 但协方差必须来自**联合** Hessian 的耦合块：Cov_pp = [inv(H_joint)]_pp。
        // 旧实现用 inv(H_coupling)（= θ 固定的条件协方差），系统性偏小。
        std::vector<int64_t> free_idx;
        for (int j = 0; j < 2 * n; ++j)
            if (j != 0 && j != n) free_idx.push_back(j);      // 参考方向固定
        int nf = 0;
        bool cov_ok = false;
        torch::Tensor cov = couplingCovariance(H, n, nf, cov_ok);
        TORCH_CHECK((int)free_idx.size() == nf,
            "computeMatrixErrors: 自由方向计数不一致");
        if (nf == 0) {
            // 全固定配置（常见：只画图不拟合、或参数全被 fix_var 锁死）——
            // 没有参数误差可言，返回空让调用方跳过，绝不能把 writeResult 弄挂。
            std::cout << "[ctpwa] 参数误差矩阵: 无自由耦合方向，跳过" << std::endl;
            restoreTheta();
            return { torch::empty({0, 0}, opts2d), torch::empty({0, 0}, opts2d) };
        }
        if (!cov_ok) {
            std::cerr << "[ctpwa] 参数误差矩阵: 联合 Hessian 自由块非正定或求逆失败"
                         "（min eig / 数值问题），不写误差矩阵" << std::endl;
            restoreTheta();
            return { torch::empty({0, 0}, opts2d), torch::empty({0, 0}, opts2d) };
        }
        const double* h_cov = cov.data_ptr<double>();

        // ---- 中心值：两张归一化矩阵 ----
        auto runMatrices = [&](const torch::Tensor& p_unified,
                               torch::Tensor& out_rec, torch::Tensor& out_gen,
                               double& W_rec, double& W_gen) {
            const torch::Tensor cpl = torch::complex(
                p_unified.slice(0, 0, n).to(TORCH_FLOAT),
                p_unified.slice(0, n, 2 * n).to(TORCH_FLOAT)).contiguous();
            if (ntheta > 0)
                ctx.set_theta(p_unified.slice(0, 2 * n, p_unified.size(0)).detach().clone());
            torch::Tensor ev_center = ctx.free_params_to_amplitudes(cpl);
            std::vector<double> dP(npartials), sP((size_t)npartials * npartials);
            std::vector<double> dT(npartials), sT((size_t)npartials * npartials);
            ctx.batch_integrals_full("phsp", ev_center, {}, {}, dP.data(), sP.data(), nullptr, nullptr, nullptr, nullptr, nullptr, npartials);
            if (has_gen)
                ctx.batch_integrals_full("phsp_truth", ev_center, {}, {}, dT.data(), sT.data(), nullptr, nullptr, nullptr, nullptr, nullptr, npartials);
            auto toMat = [&](const std::vector<double>& d, const std::vector<double>& s,
                             torch::Tensor& out) {
                out = torch::zeros({ npartials, npartials }, opts2d);
                for (int i = 0; i < npartials; ++i) {
                    out[i][i] = d[i];
                    for (int j = i + 1; j < npartials; ++j) {
                        out[i][j] = s[(size_t)i * npartials + j];
                        out[j][i] = s[(size_t)i * npartials + j];
                    }
                }
            };
            toMat(dP, sP, out_rec);
            if (has_gen) toMat(dT, sT, out_gen);
            else out_gen = torch::zeros({ npartials, npartials }, opts2d);
            auto tot = [&](const torch::Tensor& M) {
                double w = 0.0;
                for (int i = 0; i < npartials; ++i) {
                    w += M[i][i].item<double>();
                    for (int j = i + 1; j < npartials; ++j) w += M[i][j].item<double>();
                }
                return w;
            };
            W_rec = tot(out_rec);
            W_gen = has_gen ? tot(out_gen) : 0.0;
        };

        // ---- 耦合方向：解析（rec 由常驻 B̄、gen 一趟扫描）；θ 方向本轮不传播 ----
        // 成本：rec = 0 趟（n×n 收缩），gen = 1 趟（复用同一批振幅算所有方向的 ±）。
        // 对比旧实现：4·n_free 趟全样本扫描（每趟都读文件、重算振幅）。
        auto p_cpu = params.detach().to(torch::kCPU).contiguous();
        const double* h_p = p_cpu.data_ptr<double>();
        const int np = npartials;          // 分波数（矩阵维度）
        const int namp = ctx.n_amplitudes;    // 振幅数（B̄ 的维度；多 LS 时 > np）

        // 自由耦合方向（与 free_idx 的前段一致；参考方向 Re v0 / Im v0 固定）
        std::vector<int64_t> dir_idx;
        for (int j = 0; j < 2 * n; ++j)
            if (j != 0 && j != n) dir_idx.push_back(j);
        const int nd = (int)dir_idx.size();

        auto cpl_from = [&](const torch::Tensor& pu) {
            return torch::complex(pu.slice(0, 0, n).to(TORCH_FLOAT),
                                  pu.slice(0, n, 2 * n).to(TORCH_FLOAT)).contiguous();
        };
        const torch::Tensor ev_center = ctx.free_params_to_amplitudes(cpl_from(params));
        std::vector<double> eps_d(nd, 1e-6);
        std::vector<torch::Tensor> ev_p(nd), ev_m(nd);
        for (int d = 0; d < nd; ++d) {
            const int64_t g = dir_idx[d];
            eps_d[d] = 5e-6 * std::max(std::abs(h_p[g]), 1.0);
            torch::Tensor pp = params.clone(), pm = params.clone();
            pp[g] += eps_d[d];
            pm[g] -= eps_d[d];
            ev_p[d] = ctx.free_params_to_amplitudes(cpl_from(pp));
            ev_m[d] = ctx.free_params_to_amplitudes(cpl_from(pm));
        }

        // 产生侧：一趟扫描，同时给出中心与所有方向的 ±
        std::vector<double> C_diag(np, 0.0), C_off((size_t)np * np, 0.0);
        std::vector<double> P_diag((size_t)nd * np, 0.0), P_off((size_t)nd * np * np, 0.0);
        std::vector<double> M_diag((size_t)nd * np, 0.0), M_off((size_t)nd * np * np, 0.0);
        std::vector<double> Wg(1, 0.0);
        if (has_gen) {
            ctx.batch_integrals_full("phsp_truth", ev_center, ev_p, ev_m,
                C_diag.data(), C_off.data(), nullptr,
                P_diag.data(), P_off.data(), M_diag.data(), M_off.data(), np);
            Wg[0] = 0.0;
            for (int i = 0; i < np; ++i) {
                Wg[0] += C_diag[i];
                for (int j = i + 1; j < np; ++j) Wg[0] += C_off[(size_t)i * np + j];
            }
            if (!(Wg[0] > 0.0) || !std::isfinite(Wg[0])) {
                // 空 phsp_truth（或退化权重/耦合）不应抹掉有效的 rec 结果。
                // 中心 writeResult 本来就不会写 W=0 的 gen 表；这里同样只
                // 跳过 gen_err，不抛出异常打断整个文件。
                std::cerr << "[ctpwa] phsp_truth 相干总额非正/非有限，"
                             "跳过 interference_gen_err" << std::endl;
                has_gen = false;
            }
        }

        // 重建侧：常驻 B̄ 拷一次 host，之后每个方向只是 n×n 收缩（0 趟）
        std::vector<double> Br, Bi;
        double W_total_rec = 0.0;
        {
            ctpwa::ensure_device(ctx.primary_dev);
            Br.resize((size_t)namp * namp);
            Bi.resize((size_t)namp * namp);
            if (ctx.d_phsp_matrix_double != nullptr) {
                std::vector<cuDoubleComplex> tmp((size_t)namp * namp);
                cudaMemcpy(tmp.data(), ctx.d_phsp_matrix_double,
                           tmp.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
                for (size_t k = 0; k < tmp.size(); ++k) { Br[k] = tmp[k].x; Bi[k] = tmp[k].y; }
            } else {
                std::vector<ctComplex> tmp((size_t)namp * namp);
                cudaMemcpy(tmp.data(), ctx.d_phsp_matrix, tmp.size() * sizeof(ctComplex),
                           cudaMemcpyDeviceToHost);
                for (size_t k = 0; k < tmp.size(); ++k) { Br[k] = (double)tmp[k].x; Bi[k] = (double)tmp[k].y; }
            }
            for (size_t g = 0; g < (*ctx.events).size(); ++g) {
                int nP = (*ctx.events)[g][0];
                if (nP == 0) continue;
                if (g < (*ctx.phsp_weights).size() && (*ctx.phsp_weights)[g] != nullptr) {
                    cudaSetDevice((int)g);
                    std::vector<double> hw(nP);
                    cudaMemcpy(hw.data(), (*ctx.phsp_weights)[g], nP * sizeof(double),
                               cudaMemcpyDeviceToHost);
                    for (double x : hw) W_total_rec += x;
                } else {
                    W_total_rec += (double)nP;
                }
            }
        }
        // 空 phsp / 权重和为 0：没有重建侧矩阵，也就不该写 rec_err。
        const bool has_rec = W_total_rec > 0.0;
        auto contractRec = [&](const torch::Tensor& vec, std::vector<double>& Mout,
                               double& Wsum) {
            auto vc = vec.detach().to(torch::kCPU).contiguous();
            TORCH_CHECK(vc.scalar_type() == TORCH_COMPLEX,
                "contractRec: vector dtype 与 .so 复数精度不一致");
            const ctC10Complex* v =
                reinterpret_cast<const ctC10Complex*>(vc.data_ptr());
            // 与 phspMatrixFromGram 同一口径：行/列 = 分波（链 × 共振态组合），
            // 分波内多个 LS 相干求和（(*ctx.n_slvectors) 给出每段长度）。
            const int npart = (int)(*ctx.n_slvectors).size();
            std::vector<int> pst(npart + 1, 0);
            for (int p = 0; p < npart; ++p) pst[p + 1] = pst[p] + (*ctx.n_slvectors)[p];
            Mout.assign((size_t)npart * npart, 0.0);
            Wsum = 0.0;
            for (int p = 0; p < npart; ++p)
                for (int q = 0; q < npart; ++q) {
                    std::complex<double> acc(0.0, 0.0);
                    for (int k = pst[p]; k < pst[p + 1]; ++k)
                        for (int l = pst[q]; l < pst[q + 1]; ++l) {
                            std::complex<double> b(Br[(size_t)k * namp + l], Bi[(size_t)k * namp + l]);
                            const std::complex<double> vk(v[k].real(), v[k].imag());
                            const std::complex<double> vl(v[l].real(), v[l].imag());
                            acc += vk * std::conj(vl) * std::conj(b);
                        }
                    double val = (p == q ? 1.0 : 2.0) * acc.real() * W_total_rec;
                    Mout[(size_t)p * npart + q] = val;
                    if (p == q || q > p) Wsum += val;
                }
        };

        // ---- Jacobian：[nf][np][np]（θ 列保持 0）----
        std::vector<double> Jrec((size_t)nf * np * np, 0.0);
        std::vector<double> Jgen((size_t)nf * np * np, 0.0);
        std::vector<double> Mp, Mm;
        double Wp = 0.0, Wm = 0.0;
        for (int d = 0; d < nd; ++d) {
            const size_t f = (size_t)(&dir_idx[d] - &dir_idx[0]);
            // rec：B̄ 收缩（空 phsp 没有重建侧矩阵，跳过）
            if (has_rec) {
                contractRec(ev_p[d], Mp, Wp);
                contractRec(ev_m[d], Mm, Wm);
            }
            for (int i = 0; i < np; ++i)
                for (int j = 0; j < np; ++j) {
                    const size_t o = (f * np + i) * np + j;
                    if (has_rec) {
                        const double wp = (Wp > 0.0) ? Wp : 1e-30;
                        const double wm = (Wm > 0.0) ? Wm : 1e-30;
                        Jrec[o] = (Mp[(size_t)i * np + j] / wp
                                   - Mm[(size_t)i * np + j] / wm)
                                  / (2.0 * eps_d[d]);
                    }
                    if (has_gen) {
                        const double cp = C_off[(size_t)i * np + j];
                        const double cm = C_off[(size_t)i * np + j];
                        (void)cp; (void)cm;
                        auto at = [&](const std::vector<double>& a, int which) {
                            return a[(size_t)which * np * np + (size_t)i * np + j];
                        };
                        const double Pij = (i == j) ? P_diag[(size_t)d * np + j] : at(P_off, d);
                        const double Mij = (i == j) ? M_diag[(size_t)d * np + j] : at(M_off, d);
                        double Wp_g = 0.0, Wm_g = 0.0;
                        for (int a = 0; a < np; ++a) {
                            Wp_g += P_diag[(size_t)d * np + a];
                            Wm_g += M_diag[(size_t)d * np + a];
                            for (int b = a + 1; b < np; ++b) {
                                Wp_g += P_off[(size_t)d * np * np + (size_t)a * np + b];
                                Wm_g += M_off[(size_t)d * np * np + (size_t)a * np + b];
                            }
                        }
                        if (Wp_g <= 0.0) Wp_g = 1e-30;
                        if (Wm_g <= 0.0) Wm_g = 1e-30;
                        Jgen[o] = (Pij / Wp_g - Mij / Wm_g) / (2.0 * eps_d[d]);
                    }
                }
        }

        // ---- σ = sqrt(J^T cov J)（逐元素）----
        auto sigma = [&](const std::vector<double>& J, torch::Tensor& out) {
            out = torch::zeros({ npartials, npartials }, opts2d);
            std::vector<double> v(nf);
            for (int i = 0; i < npartials; ++i)
                for (int j = 0; j < npartials; ++j) {
                    for (int a = 0; a < nf; ++a)
                        v[a] = J[((size_t)a * npartials + i) * npartials + j];
                    double acc = 0.0;
                    for (int a = 0; a < nf; ++a) {
                        if (v[a] == 0.0) continue;
                        double inner = 0.0;
                        for (int b = 0; b < nf; ++b) inner += h_cov[(size_t)a * nf + b] * v[b];
                        acc += v[a] * inner;
                    }
                    out[i][j] = std::sqrt(std::max(0.0, acc));
                    out[j][i] = out[i][j];
                }
        };
        torch::Tensor err_rec, err_gen;
        if (has_rec) sigma(Jrec, err_rec);
        else err_rec = torch::empty({0, 0}, opts2d);
        if (has_gen) sigma(Jgen, err_gen);
        else err_gen = torch::empty({0, 0}, opts2d);
        restoreTheta();
        return { err_rec, err_gen };
    
}

}  // namespace ctpwa
