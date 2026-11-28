// src/FullHessian.cu — 见 include/FullHessian.h
#include <FullHessian.h>

#include "BackendHelpers.h"
#include "CouplingHessian.h"
#include "DeviceGuard.h"
#include "HessianContrib.h"

#include <PhspGram.h>
#include "MixedPrecision.h"

#include <ComputeHessian.cuh>
#include <ComputeNLL.cuh>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ctpwa {

torch::Tensor getHessian(FullHessianCtx& ctx, const torch::Tensor& params)
{

        // Float 档 getHessian 走 hybrid 同款路径（A float2 存储 + double 计算 +
        // 窗口上转）：数值 ≈ hybrid（~1e-8 对 double）；真 float 计算的 UH
        // 模板化属后续批次。hessian 链不读常驻 d_dF（其 float2 化不影响此处）。
        TORCH_CHECK(ctpwa_is_cuda_like(params) && params.dtype() == torch::kFloat64,
            "params must be float64 CUDA tensor");
        TORCH_CHECK(ctpwa_on_primary(params, ctx.primary_dev),
            "params 必须位于主设备 (primary=" + std::to_string(ctx.primary_dev) +
            ")，当前在 " + params.device().str());

        // ---- 1. Setup: determine nv, vector, theta ----
        bool is_coupling = (*ctx.params).hasCouplingMatrix();
        int ncf = 0;
        int nv, nt, n2, total;
        torch::Tensor vector, theta;

        if (is_coupling) {
            const auto& cm = (*ctx.params).couplingMatrix();
            ncf = cm.n_free;
            nv  = cm.n_amps;
            nt  = (*ctx.params).nFreeTheta();
            auto dev = params.device();
            auto v_ext = torch::empty({nv},
                torch::TensorOptions().dtype(TORCH_COMPLEX).device(dev));
            (*ctx.params).applyCouplingMatrix(params.data_ptr<double>(),
                reinterpret_cast<ctComplex*>(v_ext.data_ptr()));
            vector = v_ext;
            theta = (nt > 0) ? params.slice(0, 2*ncf, params.size(0))
                             : torch::Tensor();
        } else {
            nv = (*ctx.params).nFreeVector();
            nt = (*ctx.params).nFreeTheta();
            std::tie(vector, theta) = (*ctx.params).splitParams(params);
        }

        n2 = 2 * nv;
        total = n2 + nt;
        auto dev = params.device();
        torch::Tensor hessian = torch::zeros({total, total}, torch::kFloat64).to(dev);

        // 打印hessian全部元素
        // std::cout << "Hessian elements " << __LINE__ << ": \n" << hessian << std::endl;

        // ---- 2. recompute amplitudes + rebuild phsp matrix ----
        if (nt > 0 && theta.numel() > 0) {
            ctx.amp_calc->reComputeAmps((*ctx.d_all_amplitudes),
                reinterpret_cast<const double*>(theta.data_ptr()),
                ctx.n_amplitudes, (*ctx.events_offsets), (*ctx.amp_offsets), ctx.n_polar,
                dev.index());
        }

        bool hprof = getenv("CTPWA_PROF") != nullptr;
        auto hT0 = std::chrono::high_resolution_clock::now();  // reComputeAmps 完成
        // ---- Hessian 分段显存快照（CTPWA_PROF 下打印; 主卡视角）----
        size_t hmem[5] = {0, 0, 0, 0, 0};
        int hmem_n = 0;
        auto hmem_snap = [&]() {
            if (!hprof || hmem_n >= 5) return;
            ctpwa::ensure_device(dev.index());
            size_t free_b = 0, tot_b = 0;
            cudaMemGetInfo(&free_b, &tot_b);
            hmem[hmem_n++] = (tot_b - free_b) / 1048576ULL;
        };
        hmem_snap();  // 基线（reComputeAmps 前）
        auto hT1 = hT0, hT2 = hT0, hT3 = hT0, hT4 = hT0, hT5 = hT0;

        // Update ctx.d_phsp_matrix to reflect current amplitudes
        // phsp 流式模式下不重建：质量/宽度固定 → 矩阵在构造期已就绪且不变
        //（常驻表不含 phsp，重建会读错数据）
        if (!ctx.phsp_freed)
            ctpwa::rebuildPhspGram((*ctx.d_all_amplitudes), (*ctx.events),
                                   (*ctx.phsp_weights), ctx.float_amps, ctx.n_amplitudes,
                                   ctx.n_polar, ctx.primary_dev, ctx.d_phsp_matrix,
                                   [] { return 150000; });
        // （原内联重建代码已抽成 rebuildPhspGram()）

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        hT1 = std::chrono::high_resolution_clock::now();  // phsp 矩阵重建完成
        hmem_snap();

        // ---- 3. vv block [0:n2, 0:n2] ----
        if (n2 > 0) {
            torch::Tensor extended_vector = is_coupling ? vector : (*ctx.params).extendVector(vector, dev);
            const ctComplex* d_vec = reinterpret_cast<const ctComplex*>(extended_vector.data_ptr());
            const int n_ext = extended_vector.numel();

            // phsp factor
            ctpwa::ensure_device(dev.index());
            ctComplex* d_P_vec;
            cudaMalloc(&d_P_vec, ctx.n_amplitudes * sizeof(ctComplex));
            ctPhspReal *d_pr, *d_pi;
            cudaMalloc(&d_pr, sizeof(ctPhspReal));
            cudaMalloc(&d_pi, sizeof(ctPhspReal));
            computeQuadraticForm(ctx.d_phsp_matrix, d_vec, d_P_vec, d_pr, d_pi, ctx.n_amplitudes);
            ctPhspReal phr, phi;
            cudaMemcpy(&phr, d_pr, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
            cudaMemcpy(&phi, d_pi, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
            double phsp_factor = (double)phr;
            cudaFree(d_pr); cudaFree(d_pi);
            cudaFree(d_P_vec);

            // Extended Hessian (2·n_ext × 2·n_ext)
            torch::Tensor hess_ext = torch::zeros({2 * n_ext, 2 * n_ext}, torch::kDouble).to(dev);
            double* d_hess_ext = hess_ext.data_ptr<double>();
            int hess_sz = (2 * n_ext) * (2 * n_ext);
            int totalDataEvents = 0;

            // Multi-GPU: data + bkg Hessian contributions
            for (size_t gpu = 0; gpu < (*ctx.d_all_amplitudes).size(); ++gpu) {
                ctpwa::ensure_device(gpu);
                torch::Tensor vec_gpu = extended_vector.to(ctpwa_dev(gpu));
                const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(vec_gpu.data_ptr());

                double* d_hess_gpu;
                ctpwaCheckedMalloc((void**)&d_hess_gpu, (size_t)hess_sz * sizeof(double), "d_hess_gpu");
                cudaMemset(d_hess_gpu, 0, hess_sz * sizeof(double));

                int nData = (*ctx.events)[gpu][1];
                if (nData > 0) {
                    totalDataEvents += nData;
                    auto vvt0 = std::chrono::high_resolution_clock::now();
                    ctpwa::hessianSegmentContrib(
                        (*ctx.d_all_amplitudes), (*ctx.events), (*ctx.amp_offsets),
                        (*ctx.data_weights), (*ctx.bkg_weights), ctx.float_amps,
                        ctx.n_amplitudes, ctx.n_polar, (int)gpu, 1,
                        d_v_gpu, d_hess_gpu, n_ext, ctpwa::hessianFastPathEnabled());
                    if (hprof) {
                        auto vvt1 = std::chrono::high_resolution_clock::now();
                        printf("[PROF] H.vv-data(gpu%zu): %.2f ms (nData=%d)\n", gpu,
                            std::chrono::duration<double, std::milli>(vvt1 - vvt0).count(), nData);
                        fflush(stdout);
                    }
                }

                // bkg 段（无本底事件则内部直接返回）
                {
                    auto vvt0 = std::chrono::high_resolution_clock::now();
                    ctpwa::hessianSegmentContrib(
                        (*ctx.d_all_amplitudes), (*ctx.events), (*ctx.amp_offsets),
                        (*ctx.data_weights), (*ctx.bkg_weights), ctx.float_amps,
                        ctx.n_amplitudes, ctx.n_polar, (int)gpu, 2,
                        d_v_gpu, d_hess_gpu, n_ext, ctpwa::hessianFastPathEnabled());
                    if (hprof) {
                        auto vvt1 = std::chrono::high_resolution_clock::now();
                        printf("[PROF] H.vv-bkg(gpu%zu): %.2f ms\n", gpu,
                            std::chrono::duration<double, std::milli>(vvt1 - vvt0).count());
                        fflush(stdout);
                    }
                }

                if (gpu == dev.index()) {
                    double one = 1.0;
                    cublasHandle_t h; cublasCreate(&h);
                    cublasDaxpy(h, hess_sz, &one, d_hess_gpu, 1, d_hess_ext, 1);
                    cublasDestroy(h);
                } else {
                    double* d_hess_buf;
                    ctpwa::ensure_device(dev.index());
                    ctpwaCheckedMalloc((void**)&d_hess_buf, (size_t)hess_sz * sizeof(double), "d_hess_buf");
                    cudaMemcpyPeer(d_hess_buf, dev.index(), d_hess_gpu, gpu, hess_sz * sizeof(double));
                    double one = 1.0;
                    cublasHandle_t h; cublasCreate(&h);
                    cublasDaxpy(h, hess_sz, &one, d_hess_buf, 1, d_hess_ext, 1);
                    cublasDestroy(h);
                    cudaFree(d_hess_buf);
                }
                cudaFree(d_hess_gpu);
            }
            ctpwa::ensure_device(dev.index());

            // phsp Hessian contribution
            {
                auto vvt0 = std::chrono::high_resolution_clock::now();
                double phsp_weight = ctx.data_total_weight - ctx.bkg_integral;
                computePhspHessian(ctx.d_phsp_matrix, d_vec, phsp_factor, phsp_weight, d_hess_ext, n_ext);
                if (hprof) {
                    auto vvt1 = std::chrono::high_resolution_clock::now();
                    printf("[PROF] H.vv-phspHessian: %.2f ms\n",
                        std::chrono::duration<double, std::milli>(vvt1 - vvt0).count());
                    fflush(stdout);
                }
            }

            // Constraint projection (identity in coupling mode since nv=na=n_ext)
            if (is_coupling || n_ext == nv) {
                // Copy top-left n2×n2 (interleaved format from d_hess_ext)
                for (int i = 0; i < n2; ++i)
                    cudaMemcpy(hessian.data_ptr<double>() + i * total,
                               d_hess_ext + i * (2 * n_ext),
                               n2 * sizeof(double), cudaMemcpyDeviceToDevice);
            } else {
                // Legacy constraint projection
                int ncons = 0;
                for (const auto& g : (*ctx.params).constraintGroups())
                    for (size_t c = 0; c < g.ext_indices.size(); ++c)
                        if ((int)c != g.origin_idx_in_group) ++ncons;
                std::vector<int> h_oids, h_eids;
                std::vector<double> h_re, h_im;
                for (const auto& g : (*ctx.params).constraintGroups())
                    for (size_t c = 0; c < g.ext_indices.size(); ++c) {
                        if ((int)c == g.origin_idx_in_group) continue;
                        h_oids.push_back(g.origin_id);
                        h_eids.push_back(g.ext_indices[c]);
                        h_re.push_back((double)g.real_ratios[c]);
                        h_im.push_back((double)g.imag_ratios[c]);
                    }
                int *d_oids, *d_eids;
                double *d_re, *d_im;
                cudaMalloc(&d_oids, ncons*sizeof(int)); cudaMemcpy(d_oids, h_oids.data(), ncons*sizeof(int), cudaMemcpyHostToDevice);
                cudaMalloc(&d_eids, ncons*sizeof(int)); cudaMemcpy(d_eids, h_eids.data(), ncons*sizeof(int), cudaMemcpyHostToDevice);
                cudaMalloc(&d_re, ncons*sizeof(double)); cudaMemcpy(d_re, h_re.data(), ncons*sizeof(double), cudaMemcpyHostToDevice);
                cudaMalloc(&d_im, ncons*sizeof(double)); cudaMemcpy(d_im, h_im.data(), ncons*sizeof(double), cudaMemcpyHostToDevice);
                torch::Tensor hess_reduced = torch::zeros({n2, n2}, torch::kDouble).to(dev);
                reduceHessianWithConstraints(d_hess_ext, hess_reduced.data_ptr<double>(), d_oids, d_eids, d_re, d_im, ncons, nv, n_ext);
                cudaFree(d_oids); cudaFree(d_eids); cudaFree(d_re); cudaFree(d_im);
                hessian.slice(0, 0, n2).slice(1, 0, n2).copy_(hess_reduced);
            }
        }

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        // Reorder vv block to grouped [Re0..Re_n, Im0..Im_n] BEFORE vθ assembly
        // reorderVVBlockInterleavedToGrouped(hessian.data_ptr<double>(), nv, total);

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        hT2 = std::chrono::high_resolution_clock::now();  // vv block 完成
        hmem_snap();

        // ---- 4. vθ/θθ block [n2:total, n2:total] ----
        if (nt > 0 && theta.numel() > 0) {
            int P = nt;
            int n_gpu = static_cast<int>((*ctx.d_all_amplitudes).size());
            // CPU 张量的 dev.index() 是 -1；归一化到 0（该值当设备号用）
            int primary_dev = dev.is_cuda() ? dev.index() : 0;

            torch::Tensor extended_v = is_coupling ? vector : (*ctx.params).extendVector(vector, dev);
            int n_ext = extended_v.numel();

            // Build per-GPU interleaved v arrays (avoid cross-device access)
            std::vector<torch::Tensor> v_il_tensors;  // keep tensors alive
            std::vector<ctComplex*> d_v_per_gpu(n_gpu, nullptr);
            for (int gpu = 0; gpu < n_gpu; ++gpu) {
                ctpwa::ensure_device(gpu);
                torch::Tensor v_gpu = extended_v.to(ctpwa_dev(gpu));
                // computeEffectiveCoupling 已停用（d_T 无消费者，见 forward 处注释）
                // ctx.amp_calc->computeEffectiveCoupling(
                //     reinterpret_cast<const ctComplex*>(v_gpu.data_ptr()), n_ext);
                // Build interleaved per-GPU（dtype 跟随编译精度：float32/float64）
                torch::Tensor vr = torch::real(v_gpu).to(TORCH_FLOAT);
                torch::Tensor vi = torch::imag(v_gpu).to(TORCH_FLOAT);
                torch::Tensor vil = torch::empty({n_ext * 2},
                    torch::TensorOptions().dtype(TORCH_FLOAT).device(ctpwa_dev(gpu)));
                vil.slice(0, 0, 2 * n_ext, 2).copy_(vr);
                vil.slice(0, 1, 2 * n_ext, 2).copy_(vi);
                d_v_per_gpu[gpu] = const_cast<ctComplex*>(reinterpret_cast<const ctComplex*>(vil.data_ptr()));
                v_il_tensors.push_back(vil);
            }
            ctpwa::ensure_device(primary_dev);

            double* d_hess;
            ctpwaCheckedMalloc((void**)&d_hess, (size_t)P * P * sizeof(double), "d_hess(2n+P 全 Hessian)");
            cudaMemset(d_hess, 0, P * P * sizeof(double));
            double* d_mixed = nullptr;
            ctpwaCheckedMalloc((void**)&d_mixed, (size_t)2 * ctx.n_amplitudes * P * sizeof(double), "d_mixed");
            cudaMemset(d_mixed, 0, 2 * ctx.n_amplitudes * P * sizeof(double));
            double* d_phsp_mixed_sum = nullptr;
            double* d_phsp_mixed_t3 = nullptr;
            ctpwaCheckedMalloc((void**)&d_phsp_mixed_sum, (size_t)2 * ctx.n_amplitudes * P * sizeof(double), "d_phsp_mixed_sum");
            ctpwaCheckedMalloc((void**)&d_phsp_mixed_t3, (size_t)2 * ctx.n_amplitudes * sizeof(double), "d_phsp_mixed_t3");
            cudaMemset(d_phsp_mixed_sum, 0, 2 * ctx.n_amplitudes * P * sizeof(double));
            cudaMemset(d_phsp_mixed_t3, 0, 2 * ctx.n_amplitudes * sizeof(double));

            // Data
            {
                std::vector<int> n_data_ev(n_gpu, 0), data_off(n_gpu, 0);
                for (int g = 0; g < n_gpu; ++g) { n_data_ev[g] = (*ctx.events)[g][1]; data_off[g] = (*ctx.events)[g][0]; }
                auto uT0 = std::chrono::high_resolution_clock::now();
                ctx.amp_calc->computeUnifiedHessian(ctx.primary_dev, n_data_ev, d_hess, P, data_off, 1.0,
                    d_v_per_gpu, (*ctx.d_all_amplitudes), ctx.n_amplitudes, (*ctx.data_weights),
                    nullptr, nullptr, nullptr, d_mixed, nullptr, nullptr);
                if (hprof) {
                    auto uT1 = std::chrono::high_resolution_clock::now();
                    printf("[PROF] UH-call data: %.2f ms\n",
                        std::chrono::duration<double, std::milli>(uT1 - uT0).count());
                    fflush(stdout);
                }
            }

            // Bkg
            {
                bool has_bkg = false;
                for (int g = 0; g < n_gpu; ++g) { if ((*ctx.events)[g].size() > 2 && (*ctx.events)[g][2] > 0) { has_bkg = true; break; } }
                std::vector<int> n_bkg_ev(n_gpu, 0), bkg_off(n_gpu, 0);
                if (has_bkg) {
                    for (int g = 0; g < n_gpu; ++g) { n_bkg_ev[g] = ((*ctx.events)[g].size() > 2) ? (*ctx.events)[g][2] : 0; bkg_off[g] = (*ctx.events)[g][0] + (*ctx.events)[g][1]; }
                }
                std::vector<double*> neg_bkg_weights;
                for (int g = 0; g < n_gpu; ++g) {
                    double* d_neg = nullptr;
                    if (g < (int)(*ctx.bkg_weights).size() && (*ctx.bkg_weights)[g] != nullptr && n_bkg_ev[g] > 0) {
                        ctpwa::ensure_device(g); cudaMalloc(&d_neg, n_bkg_ev[g] * sizeof(double));
                        int grid = (n_bkg_ev[g] + 255) / 256;
                        CTPWA_LAUNCH(negateWeightsKernel, grid, 256, 0, d_neg, (*ctx.bkg_weights)[g], n_bkg_ev[g]);
                    }
                    neg_bkg_weights.push_back(d_neg);
                }
                ctpwa::ensure_device(primary_dev);
                auto uT0 = std::chrono::high_resolution_clock::now();
                ctx.amp_calc->computeUnifiedHessian(ctx.primary_dev, n_bkg_ev, d_hess, P, bkg_off, -1.0,
                    d_v_per_gpu, (*ctx.d_all_amplitudes), ctx.n_amplitudes, neg_bkg_weights,
                    nullptr, nullptr, nullptr, d_mixed, nullptr, nullptr);
                if (hprof) {
                    auto uT1 = std::chrono::high_resolution_clock::now();
                    printf("[PROF] UH-call bkg: %.2f ms\n",
                        std::chrono::duration<double, std::milli>(uT1 - uT0).count());
                    fflush(stdout);
                }
                for (int g = 0; g < n_gpu; ++g) { if (neg_bkg_weights[g]) { ctpwa::ensure_device(g); cudaFree(neg_bkg_weights[g]); } }
                ctpwa::ensure_device(primary_dev);
            }

            // Phsp
            double phsp_pf = 1.0, phsp_A = 0.0; int phsp_np = 1; double* phsp_h_pg = nullptr;
            {
                double *d_pI, *d_pg, *d_phA;
                cudaMalloc(&d_pI, sizeof(double)); cudaMalloc(&d_pg, P*sizeof(double)); cudaMalloc(&d_phA, P*P*sizeof(double));
                cudaMemset(d_pI, 0, sizeof(double)); cudaMemset(d_pg, 0, P*sizeof(double)); cudaMemset(d_phA, 0, P*P*sizeof(double));
                std::vector<int> n_phsp_ev(n_gpu,0), phsp_off(n_gpu,0);
                for (int g=0; g<n_gpu; ++g) { n_phsp_ev[g]=(*ctx.events)[g][0]; phsp_off[g]=0; }
                auto uT0 = std::chrono::high_resolution_clock::now();
                ctx.amp_calc->computeUnifiedHessian(ctx.primary_dev, n_phsp_ev, d_hess, P, phsp_off, 0.0,
                    d_v_per_gpu, (*ctx.d_all_amplitudes), ctx.n_amplitudes, {},
                    d_pI, d_pg, d_phA, d_mixed, d_phsp_mixed_sum, d_phsp_mixed_t3);
                if (hprof) {
                    auto uT1 = std::chrono::high_resolution_clock::now();
                    printf("[PROF] UH-call phsp: %.2f ms\n",
                        std::chrono::duration<double, std::milli>(uT1 - uT0).count());
                    fflush(stdout);
                }
                double h_pI; cudaMemcpy(&h_pI, d_pI, sizeof(double), cudaMemcpyDeviceToHost);
                phsp_h_pg = new double[P];
                cudaMemcpy(phsp_h_pg, d_pg, P*sizeof(double), cudaMemcpyDeviceToHost);
                std::vector<double> h_ph(P*P), h_dh(P*P);
                cudaMemcpy(h_ph.data(), d_phA, P*P*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_dh.data(), d_hess, P*P*sizeof(double), cudaMemcpyDeviceToHost);
                cudaFree(d_pI); cudaFree(d_pg); cudaFree(d_phA);
                double A = ctx.data_total_weight - ctx.bkg_integral;
                int npt=0; for(int g=0;g<n_gpu;++g) npt+=(*ctx.events)[g][0];
                // 所有块被跳过时（无自由参数块）h_pI=0 → pf=0 → c1/c2=inf，
                // 0·inf = NaN；此时 h_ph/pg 全 0，跳过该修正即可
                double pf = (npt > 0) ? h_pI/npt : 0.0;
                if (pf > 0.0) {
                    double c1=A/(pf*npt), c2=-A/(pf*pf*npt*npt);
                    for(int j=0;j<P;++j) for(int k=0;k<P;++k) h_dh[j*P+k] += c1*h_ph[j*P+k] + c2*phsp_h_pg[j]*phsp_h_pg[k];
                }
                cudaMemcpy(d_hess, h_dh.data(), P*P*sizeof(double), cudaMemcpyHostToDevice);
                phsp_pf=pf; phsp_A=A; phsp_np=npt;
            }

            // Mixed Hessian post-processing + constraint projection
            {
                std::vector<double> h_mixed(2*ctx.n_amplitudes*P);
                cudaMemcpy(h_mixed.data(), d_mixed, 2*ctx.n_amplitudes*P*sizeof(double), cudaMemcpyDeviceToHost);
                cudaFree(d_mixed);
                if (phsp_h_pg) {
                    std::vector<double> h_sum(2*ctx.n_amplitudes*P), h_t3(2*ctx.n_amplitudes);
                    cudaMemcpy(h_sum.data(), d_phsp_mixed_sum, 2*ctx.n_amplitudes*P*sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_t3.data(), d_phsp_mixed_t3, 2*ctx.n_amplitudes*sizeof(double), cudaMemcpyDeviceToHost);
                    cudaFree(d_phsp_mixed_sum); cudaFree(d_phsp_mixed_t3);
                    if (phsp_pf > 0.0) {
                        double c1m=2.0*phsp_A/(phsp_pf*phsp_np), c2m=2.0*phsp_A/(phsp_pf*phsp_pf*phsp_np*phsp_np);
                        for(int a=0;a<ctx.n_amplitudes;++a) for(int j=0;j<P;++j) {
                            h_mixed[a*P+j] += c1m*h_sum[a*P+j] + c2m*h_t3[a]*phsp_h_pg[j];
                            h_mixed[(ctx.n_amplitudes+a)*P+j] += -c1m*h_sum[(ctx.n_amplitudes+a)*P+j] - c2m*h_t3[ctx.n_amplitudes+a]*phsp_h_pg[j];
                        }
                    }
                    delete[] phsp_h_pg;
                }
                // Constraint projection on mixed Hessian
                if (!is_coupling) {
                    for (const auto& g : (*ctx.params).constraintGroups()) {
                        int oid=g.origin_id;
                        for (size_t c=0;c<g.ext_indices.size();++c) {
                            if((int)c==g.origin_idx_in_group) continue;
                            int eid=g.ext_indices[c];
                            double rr=(double)g.real_ratios[c], ir=(double)g.imag_ratios[c];
                            for(int j=0;j<P;++j) {
                                h_mixed[oid*P+j] += rr*h_mixed[eid*P+j] + ir*h_mixed[(ctx.n_amplitudes+eid)*P+j];
                                h_mixed[(ctx.n_amplitudes+oid)*P+j] += -ir*h_mixed[eid*P+j] + rr*h_mixed[(ctx.n_amplitudes+eid)*P+j];
                            }
                        }
                    }
                }
                std::vector<double> h_proj(2*nv*P, 0.0);
                for(int i=0;i<nv;++i) for(int j=0;j<P;++j) {
                    h_proj[i*P+j] = h_mixed[i*P+j];
                    h_proj[(nv+i)*P+j] = h_mixed[(ctx.n_amplitudes+i)*P+j];
                }
                // Row-by-row copy to avoid sliced-tensor stride issues
                double* d_hess = hessian.data_ptr<double>();
                for (int i = 0; i < n2; ++i)
                    cudaMemcpy(d_hess + i * total + n2, h_proj.data() + i * P,
                               P * sizeof(double), cudaMemcpyHostToDevice);
                // θv block = vθ^T  (copy full block)
                std::vector<double> h_theta_v(n2 * P);
                for (int i = 0; i < n2; ++i)
                    for (int j = 0; j < P; ++j)
                        h_theta_v[j * n2 + i] = h_proj[i * P + j];
                for (int j = 0; j < P; ++j)
                    cudaMemcpy(d_hess + (n2 + j) * total, h_theta_v.data() + j * n2,
                               n2 * sizeof(double), cudaMemcpyHostToDevice);
        }

            // Copy theta-theta result
            torch::Tensor res_hess = torch::empty({P, P}, torch::TensorOptions().dtype(torch::kFloat64).device(dev));
            cudaMemcpy(res_hess.data_ptr(), d_hess, P*P*sizeof(double), cudaMemcpyDeviceToDevice);
            cudaFree(d_hess);
            hessian.slice(0,n2,total).slice(1,n2,total).copy_(res_hess);
        }
        hT3 = std::chrono::high_resolution_clock::now();  // vθ/θθ 段完成
        hmem_snap();

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        // std::cout << "Coupling: " << is_coupling << std::endl;

        // ---- 5. Coupling transform ----
        if (is_coupling) {
            // Precompute Jacobian for Hessian transform (J^T · H_ext · J + Σ g_v · ∇²v)
            (*ctx.params).precomputeJacobian(params.data_ptr<double>());

            // ---- compute gradient g_v = ∂NLL/∂v in extended amplitude space ----
            // Uses the same computeFactorNLL as the NLL forward pass for data/bkg.
            // Phsp: g_phsp = N_eff / phsp_factor * (M × v)
            double* d_g_v = nullptr;
            {
                int n_gpu = static_cast<int>((*ctx.d_all_amplitudes).size());
                // CPU 张量的 dev.index() 是 -1；归一化到 0（该值当设备号用）
            int primary_dev = dev.is_cuda() ? dev.index() : 0;
                const auto& cm = (*ctx.params).couplingMatrix();
                int na = cm.n_amps;

                // Build extended vector v_ext
                auto v_ext_tensor = torch::empty({na}, torch::TensorOptions().dtype(TORCH_COMPLEX).device(dev));
                (*ctx.params).applyCouplingMatrix(params.data_ptr<double>(),
                    reinterpret_cast<ctComplex*>(v_ext_tensor.data_ptr()));
                const ctComplex* d_v_ext = reinterpret_cast<const ctComplex*>(v_ext_tensor.data_ptr());

                // Allocate g_v on primary GPU
                cudaMalloc(&d_g_v, 2 * na * sizeof(double));
                cudaMemset(d_g_v, 0, 2 * na * sizeof(double));

                // --- Data gradient: g = ∂(-log I)/∂conj(v) via computeFactorNLL ---
                for (int gpu = 0; gpu < n_gpu; ++gpu) {
                    int nEv = (*ctx.events)[gpu][1];
                    if (nEv == 0) continue;
                    ctpwa::ensure_device(gpu);
                    ctComplex* d_g_gpu;
                    cudaMalloc(&d_g_gpu, na * sizeof(ctComplex));
                    // d_v_ext 在主 GPU，需每 GPU 副本——跨设备传入 computeFactorNLL
                    // 的 cublas gemv 会非法访问（实测 2×A100 getHessian 必现）
                    ctComplex* d_v_gpu;
                    cudaMalloc(&d_v_gpu, na * sizeof(ctComplex));
                    cudaMemcpyPeer(d_v_gpu, gpu, d_v_ext, primary_dev, na * sizeof(ctComplex));
                    const double* d_w_data = (gpu < (int)(*ctx.data_weights).size()) ? (*ctx.data_weights)[gpu] : nullptr;
                    if (ctx.float_amps) {
                        const float2* d_ampF = reinterpret_cast<const float2*>((*ctx.d_all_amplitudes)[gpu])
                            + (*ctx.amp_offsets)[gpu][1];
                        computeFactorNLLF(d_ampF, d_v_gpu, d_g_gpu, nEv, ctx.n_polar, na, d_w_data, nullptr);
                    } else {
                        computeFactorNLL((*ctx.d_all_amplitudes)[gpu] + (*ctx.amp_offsets)[gpu][1],
                            d_v_gpu, d_g_gpu, nEv, ctx.n_polar, na, d_w_data, nullptr);
                    }
                    cudaFree(d_v_gpu);
                    // Convert ctComplex [Re,Im,Re,Im,...] → double [Re...,Im...] layout
                    std::vector<ctComplex> h_g_tmp(na);
                    cudaMemcpy(h_g_tmp.data(), d_g_gpu, na * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                    std::vector<double> h_g(2 * na);
                    for (int a = 0; a < na; ++a) {
                        h_g[a]      = (double)h_g_tmp[a].x;
                        h_g[na + a] = (double)h_g_tmp[a].y;
                    }
                    ctpwa::ensure_device(primary_dev);
                    double* d_tmp; cudaMalloc(&d_tmp, 2 * na * sizeof(double));
                    cudaMemcpy(d_tmp, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                    int grid = (2 * na + 255) / 256;
                    CTPWA_LAUNCH(daxpy_kernel, grid, 256, 0, d_g_v, d_tmp, 1.0, 2 * na);
                    cudaFree(d_tmp);
                    // d_g_gpu 在副卡分配，切回所属设备后释放。
                    ctpwa::ensure_device(gpu);
                    cudaFree(d_g_gpu);
                }

                // --- Bkg gradient: contribution to loss = data_nll - bkg_nll ---
                // computeFactorNLL with weight=-bkg_weight gives gradient for -(-log(I)*bkg_weight)
                // which adds to data gradient correctly in the loss formula.
                for (int gpu = 0; gpu < n_gpu; ++gpu) {
                    if ((*ctx.events)[gpu].size() <= 2 || (*ctx.events)[gpu][2] == 0) continue;
                    int nEv = (*ctx.events)[gpu][2];
                    ctpwa::ensure_device(gpu);
                    ctComplex* d_g_gpu;
                    cudaMalloc(&d_g_gpu, na * sizeof(ctComplex));
                    // Negate bkg weights (same as in NLL forward)
                    const double* d_w = (gpu < (int)(*ctx.bkg_weights).size() && (*ctx.bkg_weights)[gpu]) ? nullptr : nullptr;
                    // Use negated bkg weights
                    double* d_neg_w = nullptr;
                    if (gpu < (int)(*ctx.bkg_weights).size() && (*ctx.bkg_weights)[gpu] != nullptr) {
                        cudaMalloc(&d_neg_w, nEv * sizeof(double));
                        int grid = (nEv + 255) / 256;
                        CTPWA_LAUNCH(negateWeightsKernel, grid, 256, 0, d_neg_w, (*ctx.bkg_weights)[gpu], nEv);
                    }
                    // 同 data：v 每 GPU 副本（d_v_ext 在主 GPU，跨设备 gemv 非法访问）
                    ctComplex* d_v_gpu;
                    cudaMalloc(&d_v_gpu, na * sizeof(ctComplex));
                    cudaMemcpyPeer(d_v_gpu, gpu, d_v_ext, primary_dev, na * sizeof(ctComplex));
                    if (ctx.float_amps) {
                        const float2* d_ampF = reinterpret_cast<const float2*>((*ctx.d_all_amplitudes)[gpu])
                            + (*ctx.amp_offsets)[gpu][2];
                        computeFactorNLLF(d_ampF, d_v_gpu, d_g_gpu, nEv, ctx.n_polar, na, d_neg_w, nullptr);
                    } else {
                        computeFactorNLL((*ctx.d_all_amplitudes)[gpu] + (*ctx.amp_offsets)[gpu][2],
                            d_v_gpu, d_g_gpu, nEv, ctx.n_polar, na, d_neg_w, nullptr);
                    }
                    cudaFree(d_v_gpu);
                    if (d_neg_w) cudaFree(d_neg_w);
                    // Convert ctComplex [Re,Im,Re,Im,...] → double [Re...,Im...] layout
                    std::vector<ctComplex> h_g_tmp(na);
                    cudaMemcpy(h_g_tmp.data(), d_g_gpu, na * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                    std::vector<double> h_g(2 * na);
                    for (int a = 0; a < na; ++a) {
                        h_g[a]      = (double)h_g_tmp[a].x;
                        h_g[na + a] = (double)h_g_tmp[a].y;
                    }
                    ctpwa::ensure_device(primary_dev);
                    double* d_tmp; cudaMalloc(&d_tmp, 2 * na * sizeof(double));
                    cudaMemcpy(d_tmp, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                    int grid = (2 * na + 255) / 256;
                    // bkg 的 g_v 符号分两种情况（很容易翻错）:
                    //  * 有 bkg_weights: d_neg_w = -w，computeFactorNLL 对权重
                    //    线性 → 已翻一次；这里 +1。
                    //  * 无 bkg_weights: d_neg_w=nullptr，kernel 用单位正权重 →
                    //    没翻；这里 -1（loss = data - bkg）。
                    // 旧代码无条件 -1：无权重时正确，带权重时把二阶项
                    // Σ g_v·∂²v 的 bkg 贡献加反（J/psi→ωηη' 69 振幅案例）。
                    const double bkg_axpy = (d_neg_w != nullptr) ? 1.0 : -1.0;
                    CTPWA_LAUNCH(daxpy_kernel, grid, 256, 0, d_g_v, d_tmp, bkg_axpy, 2 * na);
                    cudaFree(d_tmp);
                    // d_g_gpu 在副卡分配，切回所属设备后释放。
                    ctpwa::ensure_device(gpu);
                    cudaFree(d_g_gpu);
                }

                // --- Phsp gradient: g_phsp = N_eff / phsp_factor * (M × v) ---
                {
                    double N_eff = ctx.data_total_weight - ctx.bkg_integral;
                    ctpwa::ensure_device(primary_dev);

                    // Compute phsp_factor and P_vec = M × v
                    ctComplex* d_P_vec; cudaMalloc(&d_P_vec, na * sizeof(ctComplex));
                    {
                        cublasHandle_t h; cublasCreate(&h);
                        ctComplex alpha = ctMake(1,0), beta = ctMake(0,0);
                        CUBLAS_CGEMV(h, CUBLAS_OP_N, na, na, &alpha, ctx.d_phsp_matrix, na,
                                    d_v_ext, 1, &beta, d_P_vec, 1);
                        cublasDestroy(h);
                    }
                    ctPhspReal phr;
                    {
                        auto v_conj = v_ext_tensor.conj();
                        const ctComplex* d_vc = reinterpret_cast<const ctComplex*>(v_conj.data_ptr());
                        ctPhspReal *d_pr, *d_pi;
                        cudaMalloc(&d_pr, sizeof(ctPhspReal)); cudaMalloc(&d_pi, sizeof(ctPhspReal));
                        computeQuadraticForm(ctx.d_phsp_matrix, d_vc, d_P_vec, d_pr, d_pi, na);
                        cudaMemcpy(&phr, d_pr, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
                        cudaFree(d_pr); cudaFree(d_pi);
                    }
                    double phsp_factor = (double)phr;

                    if (phsp_factor > 1e-30) {
                        double scale = N_eff / phsp_factor;
                        std::vector<ctComplex> h_P(na);
                        cudaMemcpy(h_P.data(), d_P_vec, na * sizeof(ctComplex), cudaMemcpyDeviceToHost);
                        std::vector<double> h_g(2 * na, 0.0);
                        for (int a = 0; a < na; ++a) {
                            h_g[a]      = scale * (double)h_P[a].x;
                            h_g[na + a] = scale * (double)h_P[a].y;
                        }
                        double* d_tmp; cudaMalloc(&d_tmp, 2 * na * sizeof(double));
                        cudaMemcpy(d_tmp, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                        int grid = (2 * na + 255) / 256;
                        CTPWA_LAUNCH(daxpy_kernel, grid, 256, 0, d_g_v, d_tmp, 1.0, 2 * na);  // accumulate!
                        cudaFree(d_tmp);
                    }
                    cudaFree(d_P_vec);
                }

                // Convert d_g_v from ∂NLL/∂conj(v) (Wirtinger) to real gradient:
                //   ∂NLL/∂Re(v_a) = 2 * Re(∂NLL/∂conj(v_a))
                //   ∂NLL/∂Im(v_a) = 2 * Im(∂NLL/∂conj(v_a))
                // Both are simply ×2 (no sign flip — the conj flip cancels with
                // the minus in ∂/∂Im = -2 Im(∂/∂v)).
                {
                    std::vector<double> h_g(2 * na);
                    cudaMemcpy(h_g.data(), d_g_v, 2 * na * sizeof(double), cudaMemcpyDeviceToHost);
                    for (int a = 0; a < 2 * na; ++a) h_g[a] *= 2.0;
                    cudaMemcpy(d_g_v, h_g.data(), 2 * na * sizeof(double), cudaMemcpyHostToDevice);
                }
            }

            int n_fit = 2 * ncf + nt;
            auto hess_fit = torch::zeros({n_fit, n_fit}, torch::kFloat64).to(dev);
            (*ctx.params).transformExtendedHessian(
                hessian.data_ptr<double>(),
                params.data_ptr<double>(), d_g_v,
                hess_fit.data_ptr<double>(),
                nv, ncf, nt);
            cudaFree(d_g_v);
            // Constraints.gauss_constr: 罚项 Hessian 对角 1/σ²（θ 块起点 = 2*ncf；
            // transform 对 θθ 块是直接拷贝，变换前后追加等价）
            if ((*ctx.params).nGaussConstr() > 0)
                (*ctx.params).addGaussHessianDiag(hess_fit, 2 * ncf);

            if (hprof) {
                auto hTend = std::chrono::high_resolution_clock::now();
                hmem_snap();
                auto hms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                printf("[PROF] H.reAmp: %.2f | H.phspM: %.2f | H.vv: %.2f | H.vth+thth+cpl: %.2f | H.total: %.2f\n",
                    hms(hT0, hT1), hms(hT1, hT2), hms(hT2, hT3), hms(hT3, hTend), hms(hT0, hTend));
                printf("[PROF] H.mem(MiB): 基=%zu | phspM后=%zu | vv后=%zu | theta后=%zu | 返回前=%zu\n",
                    hmem[0], hmem[1], hmem[2], hmem[3], hmem[4]);
                fflush(stdout);
            }
            return hess_fit;
        }

        // std::cout << "Hessian elements in line." << __LINE__ << ": \n" << hessian << std::endl;

        // Constraints.gauss_constr: 罚项 Hessian 对角 1/σ²（θ 块起点 = n2）
        if ((*ctx.params).nGaussConstr() > 0)
            (*ctx.params).addGaussHessianDiag(hessian, n2);

        // vv / θθ 对角块强制对称化：两块的数学形式必须对称（Gram 矩阵 / ∂²/∂θ²），
        // 但双卡路径是"每 GPU 先累积再归并"，与单卡顺序累积的浮点路径不同，
        // 会引入 ~1e-7~1e-6 的不对称（实测 9.9e-7，压测试容差 1e-6）。
        // (H+H^T)/2 消除浮点噪声，不改变物理。vθ/θv 块本来就是转置复制，无需处理。
        {
            if (n2 > 0) {
                std::vector<double> h_vv((size_t)n2 * n2);
                cudaMemcpy(h_vv.data(), hessian.data_ptr<double>(),
                           (size_t)n2 * n2 * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < n2; ++i)
                    for (int j = i + 1; j < n2; ++j) {
                        double m = 0.5 * (h_vv[(size_t)i * n2 + j] + h_vv[(size_t)j * n2 + i]);
                        h_vv[(size_t)i * n2 + j] = m; h_vv[(size_t)j * n2 + i] = m;
                    }
                cudaMemcpy(hessian.data_ptr<double>(), h_vv.data(),
                           (size_t)n2 * n2 * sizeof(double), cudaMemcpyHostToDevice);
            }
            int nt = total - n2;
            if (nt > 0) {
                std::vector<double> h_th((size_t)nt * nt);
                for (int i = 0; i < nt; ++i)
                    cudaMemcpy(h_th.data() + (size_t)i * nt,
                               hessian.data_ptr<double>() + (size_t)(n2 + i) * total + n2,
                               (size_t)nt * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < nt; ++i)
                    for (int j = i + 1; j < nt; ++j) {
                        double m = 0.5 * (h_th[(size_t)i * nt + j] + h_th[(size_t)j * nt + i]);
                        h_th[(size_t)i * nt + j] = m; h_th[(size_t)j * nt + i] = m;
                    }
                for (int i = 0; i < nt; ++i)
                    cudaMemcpy(hessian.data_ptr<double>() + (size_t)(n2 + i) * total + n2,
                               h_th.data() + (size_t)i * nt,
                               (size_t)nt * sizeof(double), cudaMemcpyHostToDevice);
            }
        }

        // 恢复主 GPU 为当前设备：内部 per-GPU 循环会把它留在最后一个设备，
        // 导致后续 torch.device('cuda')（index=None → 当前设备）的 tensor 落在
        // 非主 GPU，触发 getNLL/getHessian 的 params 位置检查（测试套件实测）。
        ctpwa::ensure_device(ctx.primary_dev);

        if (hprof) {
            auto hTend = std::chrono::high_resolution_clock::now();
            hmem_snap();
            auto hms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            printf("[PROF] H.reAmp: %.2f | H.phspM: %.2f | H.vv: %.2f | H.vth+thth+cpl: %.2f | H.total: %.2f\n",
                hms(hT0, hT1), hms(hT1, hT2), hms(hT2, hT3), hms(hT3, hTend), hms(hT0, hTend));
            printf("[PROF] H.mem(MiB): 基=%zu | phspM后=%zu | vv后=%zu | theta后=%zu | 返回前=%zu\n",
                hmem[0], hmem[1], hmem[2], hmem[3], hmem[4]);
            fflush(stdout);
        }
        return hessian;
    
}

}  // namespace ctpwa
