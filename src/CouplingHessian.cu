// src/CouplingHessian.cu — 见 include/CouplingHessian.h
#include <CouplingHessian.h>

#include "BackendHelpers.h"
#include "DeviceGuard.h"
#include "HessianContrib.h"

#include <ComputeHessian.cuh>
#include <ComputeNLL.cuh>

#include <cublas_v2.h>

#include <cstdio>
#include <vector>

namespace ctpwa {

torch::Tensor computeCouplingHessian(const CouplingHessianCtx& ctx,
                                     const torch::Tensor& vector)
{

        TORCH_CHECK(ctpwa_is_cuda_like(vector),
            "vector 必须位于当前后端的设备上（CPU 构建用 CPU 张量）");
        TORCH_CHECK(vector.dtype() == TORCH_COMPLEX, "vector dtype must match .so complex precision (double/float 编译)");
        TORCH_CHECK(vector.dim() == 1, "vector must be 1-dimensional");
        TORCH_CHECK(ctpwa_on_primary(vector, ctx.primary_dev),
            "vector 必须位于主设备 (primary=" + std::to_string(ctx.primary_dev) +
            ")，当前在 " + vector.device().str());

        // 自由耦合 p → 振幅 v（乘法链映射；无耦合矩阵时恒等）
        torch::Tensor extended_vector = ctx.free_params_to_amplitudes(vector);
        const int n = extended_vector.numel();  // = n_amps
        torch::Device dev = vector.device();
        torch::Tensor extended_vector_conj = extended_vector.conj();
        const ctComplex* d_vec = reinterpret_cast<const ctComplex*>(extended_vector.data_ptr());
        const ctComplex* d_vec_conj = reinterpret_cast<const ctComplex*>(extended_vector_conj.data_ptr());

        // 计算phsp量 (与forward pass一致)
        ctpwa::ensure_device(dev.index());
        ctComplex* d_P_vec;
        cudaMalloc(&d_P_vec, ctx.n_amplitudes * sizeof(ctComplex));
        ctPhspReal* d_pr, * d_pi;
        cudaMalloc(&d_pr, sizeof(ctPhspReal));
        cudaMalloc(&d_pi, sizeof(ctPhspReal));
        // 用d_vec（非共轭）与phspHessianKernel的Pu/tildeP约定保持一致
        computeQuadraticForm(ctx.d_phsp_matrix, d_vec, d_P_vec, d_pr, d_pi, ctx.n_amplitudes);
        ctPhspReal phr, phi;
        cudaMemcpy(&phr, d_pr, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phi, d_pi, sizeof(ctPhspReal), cudaMemcpyDeviceToHost);
        double phsp_factor = (double)phr;
        cudaFree(d_pr); cudaFree(d_pi);

        // std::cout << "phsp_factor: " << phsp_factor << std::endl;

        // 分配扩展Hessian (2·n_ext × 2·n_ext, GPU 0)
        const int n_ext = extended_vector.numel();  // = ctx.n_amplitudes
        torch::Tensor hessian = torch::zeros({ 2 * n_ext, 2 * n_ext }, torch::kDouble).to(dev);
        double* d_hessian = hessian.data_ptr<double>();

        // 多GPU: 累加data和bkg的Hessian贡献（每GPU独立分配，P2P累加到GPU 0）
        int hess_sz = (2 * n_ext) * (2 * n_ext);
        int totalDataEvents = 0;
        for (size_t gpu = 0; gpu < (*ctx.d_all_amplitudes).size(); ++gpu) {
            ctpwa::ensure_device(gpu);
            torch::Tensor vec_gpu = extended_vector.to(ctpwa_dev(gpu));
            const ctComplex* d_v_gpu = reinterpret_cast<const ctComplex*>(vec_gpu.data_ptr());

            // 在当前GPU上分配临时hessian并清零
            double* d_hess_gpu;
            ctpwaCheckedMalloc((void**)&d_hess_gpu, (size_t)hess_sz * sizeof(double), "d_hess_gpu");
            cudaMemset(d_hess_gpu, 0, hess_sz * sizeof(double));

            // --- data ---
            if ((*ctx.events)[gpu][1] > 0) {
                totalDataEvents += (*ctx.events)[gpu][1];
                ctpwa::hessianSegmentContrib(
                    (*ctx.d_all_amplitudes), (*ctx.events), (*ctx.amp_offsets),
                    (*ctx.data_weights), (*ctx.bkg_weights), ctx.float_amps,
                    ctx.n_amplitudes, ctx.n_polar, (int)gpu, 1,
                    d_v_gpu, d_hess_gpu, n_ext, ctpwa::hessianFastPathEnabled());
            }

            // --- bkg ---
            ctpwa::hessianSegmentContrib(
                (*ctx.d_all_amplitudes), (*ctx.events), (*ctx.amp_offsets),
                (*ctx.data_weights), (*ctx.bkg_weights), ctx.float_amps,
                ctx.n_amplitudes, ctx.n_polar, (int)gpu, 2,
                d_v_gpu, d_hess_gpu, n_ext, ctpwa::hessianFastPathEnabled());

            // P2P累加到GPU 0的主hessian
            if (gpu == dev.index()) {
                double one = 1.0;
                cublasHandle_t h; cublasCreate(&h);
                cublasDaxpy(h, hess_sz, &one, d_hess_gpu, 1, d_hessian, 1);
                cublasDestroy(h);
            } else {
                double* d_hess_buf;
                ctpwa::ensure_device(dev.index());
                ctpwaCheckedMalloc((void**)&d_hess_buf, (size_t)hess_sz * sizeof(double), "d_hess_buf");
                cudaMemcpyPeer(d_hess_buf, dev.index(), d_hess_gpu, gpu, hess_sz * sizeof(double));
                double one = 1.0;
                cublasHandle_t h; cublasCreate(&h);
                cublasDaxpy(h, hess_sz, &one, d_hess_buf, 1, d_hessian, 1);
                cublasDestroy(h);
                cudaFree(d_hess_buf);
                // d_hess_gpu 在副卡分配；P2P 后当前设备是主卡，
                // 必须先切回其所属设备再释放。
                ctpwa::ensure_device(gpu);
            }
            cudaFree(d_hess_gpu);
        }
        ctpwa::ensure_device(dev.index());

        // phsp Hessian贡献
        double phsp_weight = ctx.data_total_weight - ctx.bkg_integral;
        ctpwa::ensure_device(dev.index());
        computePhspHessian(ctx.d_phsp_matrix, d_vec, phsp_factor, phsp_weight, d_hessian, n_ext);

        // std::cout << "n_ext: " << n_ext << ", n: " << n << ", totalDataEvents: " << totalDataEvents << std::endl;

        // 按约束Jacobian投影: H_orig (2n×2n) = J^T * H_ext (2·n_ext×2·n_ext) * J
        // J是diagonal blocks: J[2*eid][2*oid]=real_ratio, J[2*eid+1][2*oid+1]=imag_ratio
        if (n_ext > n) {
            std::vector<int> h_oids, h_eids;
            std::vector<double> h_re, h_im;
            for (const auto& g : (*ctx.params).constraintGroups()) {
                for (size_t j = 0; j < g.ext_indices.size(); ++j) {
                    if (static_cast<int>(j) == g.origin_idx_in_group) continue;
                    h_oids.push_back(g.origin_id);
                    h_eids.push_back(g.ext_indices[j]);
                    h_re.push_back(static_cast<double>(g.real_ratios[j]));
                    h_im.push_back(static_cast<double>(g.imag_ratios[j]));
                }
            }
            int ncons = h_oids.size();
            int* d_oids, * d_eids;
            double* d_re, * d_im;
            cudaMalloc(&d_oids, ncons * sizeof(int));
            cudaMalloc(&d_eids, ncons * sizeof(int));
            cudaMalloc(&d_re, ncons * sizeof(double));
            cudaMalloc(&d_im, ncons * sizeof(double));
            cudaMemcpy(d_oids, h_oids.data(), ncons * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_eids, h_eids.data(), ncons * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_re, h_re.data(), ncons * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_im, h_im.data(), ncons * sizeof(double), cudaMemcpyHostToDevice);

            // 分配输出Hessian (2n × 2n)，避免就地修改导致读写冲突
            torch::Tensor hessian_reduced = torch::zeros({ 2 * n, 2 * n }, torch::kDouble).to(dev);
            double* d_hessian_reduced = hessian_reduced.data_ptr<double>();

            reduceHessianWithConstraints(d_hessian, d_hessian_reduced, d_oids, d_eids, d_re, d_im, ncons, n, n_ext);

            cudaFree(d_oids); cudaFree(d_eids); cudaFree(d_re); cudaFree(d_im);
            cudaFree(d_P_vec);
            return hessian_reduced;
        }

        // 无约束：直接截取
        torch::Tensor hessian_reduced = hessian.slice(0, 0, 2 * n).slice(1, 0, 2 * n).clone();
        cudaFree(d_P_vec);
        return hessian_reduced;
    
}

}  // namespace ctpwa
