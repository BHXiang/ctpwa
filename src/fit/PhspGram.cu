// src/PhspGram.cu — 见 include/PhspGram.h
#include <PhspGram.h>

#include "ComplexType.h"
#include "PrecCpx.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "BackendHelpers.h"
#include "DeviceGuard.h"
#include "MixedPrecision.h"

#include <ComputeNLL.cuh>   // axpyComplex

#include <cmath>
#include <complex>
#include <vector>

namespace ctpwa {

void rebuildPhspGram(const std::vector<ctComplex*>& d_all_amplitudes,
                     const std::vector<std::vector<int>>& events,
                     const std::vector<double*>& phsp_weights,
                     bool float_amps, int n_amplitudes, int n_polar, int primary_dev,
                     ctComplex* d_matrix_io,
                     std::function<int()> batch_events)
{
            // 每次调用只恢复一次（不是逐批/逐事件操作），避免 reCalcAmp 后
            // raw current device 停在最后一张卡；异常路径也同样恢复。
            // 入口设备固定为 primary_dev；退出时还原（等价于原来的 makeScopeExit）
            auto device_guard = makeScopeExit([primary_dev]() {
                ctpwa::ensure_device(primary_dev);
            });
            // Pre-compute total phsp weight for correct global normalization
            double W_total = 0.0;
            for (size_t gpu = 0; gpu < d_all_amplitudes.size(); ++gpu) {
                int nP = events[gpu][0];
                if (nP == 0) continue;
                if (gpu < phsp_weights.size() && phsp_weights[gpu] != nullptr) {
                    ctpwa::ensure_device(gpu);
                    thrust::device_ptr<double> dp(phsp_weights[gpu]);
                    W_total += thrust::reduce(dp, dp + nP);
                } else {
                    W_total += (double)nP;
                }
            }
            if (W_total <= 0.0) W_total = 1.0;

            ctpwa::ensure_device(primary_dev);
            cudaMemset(d_matrix_io, 0, n_amplitudes * n_amplitudes * sizeof(ctComplex));
            // B̄ = Σ_ev,p (w/W)·A·A^H：分块构建（峰值 = 单块 scaled，而非整段 6.7GB 拷贝）。
            // 与构造期流式模式同法：scalePhspAmpsKernel 的 evt 从 0 计 → 权重指针偏移 d_w+c0。
            for (size_t gpu = 0; gpu < d_all_amplitudes.size(); ++gpu) {
                int nPhsp = events[gpu][0];
                if (nPhsp == 0) continue;
                ctpwa::ensure_device(gpu);
                const int kPhspBatch = batch_events();
                const double* d_w = (gpu < phsp_weights.size()) ? phsp_weights[gpu] : nullptr;
                cublasHandle_t h;
                cublasCreate(&h);
                ctComplex alpha = ctMake(1.0f, 0.0f);
                for (int c0 = 0; c0 < nPhsp; c0 += kPhspBatch) {
                    int nch = std::min(kPhspBatch, nPhsp - c0);
                    int nTot = nch * n_polar * n_amplitudes;
                    ctComplex* d_phsp_scaled;
                    ctpwaCheckedMalloc((void**)&d_phsp_scaled,
                        (size_t)nTot * sizeof(ctComplex), "phsp Gram scaled batch");
                    // phsp 段在 d_all_amplitudes[gpu] 基址，块起点 = c0 事件 × nPolar×nA
                    if (float_amps && !ctIsFloatComplex) {
                        const float2* srcF = reinterpret_cast<const float2*>(d_all_amplitudes[gpu])
                            + (size_t)c0 * n_polar * n_amplitudes;
                        CTPWA_LAUNCH(castF2ToDouble2Kernel, (nTot + 255) / 256, 256, 0, 
                            srcF, reinterpret_cast<cuDoubleComplex*>(d_phsp_scaled), nTot);
                    } else {
                        cudaMemcpy(d_phsp_scaled,
                            d_all_amplitudes[gpu] + (size_t)c0 * n_polar * n_amplitudes,
                            nTot * sizeof(ctComplex), cudaMemcpyDeviceToDevice);
                    }
                    int grid = (nTot + 255) / 256;
                    CTPWA_LAUNCH(scalePhspAmpsKernel, grid, 256, 0, d_phsp_scaled,
                        d_w ? d_w + c0 : nullptr, nch, n_polar, n_amplitudes,
                        1.0 / W_total, 0);

                    ctComplex* d_phsp_gpu;
                    ctpwaCheckedMalloc((void**)&d_phsp_gpu,
                        (size_t)n_amplitudes * n_amplitudes * sizeof(ctComplex),
                        "phsp Gram batch matrix");
                    ctComplex beta = ctMake(0.0f, 0.0f);
                    CUBLAS_CGEMM(h, CUBLAS_OP_N, CUBLAS_OP_C, n_amplitudes, n_amplitudes, nch * n_polar,
                        &alpha, d_phsp_scaled, n_amplitudes, d_phsp_scaled, n_amplitudes,
                        &beta, d_phsp_gpu, n_amplitudes);
                    cudaFree(d_phsp_scaled);

                    if (gpu == (size_t)primary_dev) {
                        ctComplex one = ctMake(1.0f, 0.0f);
                        axpyComplex(d_matrix_io, d_phsp_gpu, one, n_amplitudes * n_amplitudes);
                        cudaFree(d_phsp_gpu);
                    } else {
                        ctpwa::ensure_device(primary_dev);
                        ctComplex* d_temp;
                        cudaMalloc(&d_temp, n_amplitudes * n_amplitudes * sizeof(ctComplex));
                        cudaMemcpyPeer(d_temp, primary_dev, d_phsp_gpu, gpu,
                            n_amplitudes * n_amplitudes * sizeof(ctComplex));
                        ctpwa::ensure_device(gpu);
                        cudaFree(d_phsp_gpu);
                        ctpwa::ensure_device(primary_dev);
                        ctComplex one = ctMake(1.0f, 0.0f);
                        axpyComplex(d_matrix_io, d_temp, one, n_amplitudes * n_amplitudes);
                        cudaFree(d_temp);
                        ctpwa::ensure_device(gpu);
                    }
                }
                cublasDestroy(h);
            }
            // 多卡循环会把 current device 留在最后一张卡；调用方（reCalcAmp/
            // writeResult/getHessian）随后用 torch 默认 device 建张量，必须回来。
            ctpwa::ensure_device(primary_dev);
    
}

torch::Tensor phspMatrixFromGram(const PhspGramInputs& in,
                                 const torch::Tensor& vector,
                                 double* W_out, double* Wtotal_out)
{
        ctpwa::ensure_device(in.primary_dev);
        const int np = in.n_amplitudes;      // B̄ 是 in.n_amplitudes × in.n_amplitudes
        auto opts2d = torch::TensorOptions().dtype(torch::kFloat64);
        TORCH_CHECK((int)vector.numel() == np,
            "phspMatrixFromGram 需要扩展后的振幅向量（in.n_amplitudes=", np,
            "），收到 ", vector.numel());
        TORCH_CHECK(in.d_matrix != nullptr || in.d_matrix_double != nullptr,
            "phsp Gram 未就绪（无 phsp 样本或尚未构建）");

        double W_total = 0.0;
        for (size_t g = 0; g < (*in.events).size(); ++g) {
            int nP = (*in.events)[g][0];
            if (nP == 0) continue;
            if (g < (*in.phsp_weights).size() && (*in.phsp_weights)[g] != nullptr) {
                ctpwa::DeviceGuard dev_guard((int)g);
                std::vector<double> hw(nP);
                cudaMemcpy(hw.data(), (*in.phsp_weights)[g], nP * sizeof(double),
                           cudaMemcpyDeviceToHost);
                for (double x : hw) W_total += x;
            } else {
                W_total += (double)nP;
            }
        }
        if (W_total <= 0.0) W_total = 1.0;

        std::vector<double> Br((size_t)np * np), Bi((size_t)np * np);
        if (in.d_matrix_double != nullptr) {
            std::vector<cuDoubleComplex> tmp((size_t)np * np);
            cudaMemcpy(tmp.data(), in.d_matrix_double,
                       tmp.size() * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
            for (size_t k = 0; k < tmp.size(); ++k) { Br[k] = tmp[k].x; Bi[k] = tmp[k].y; }
        } else {
            std::vector<ctComplex> tmp((size_t)np * np);
            cudaMemcpy(tmp.data(), in.d_matrix, tmp.size() * sizeof(ctComplex),
                       cudaMemcpyDeviceToHost);
            for (size_t k = 0; k < tmp.size(); ++k) { Br[k] = (double)tmp[k].x; Bi[k] = (double)tmp[k].y; }
        }

        auto vcpu = vector.detach().to(torch::kCPU).contiguous();
        // vector 的复数精度随 .so：double 构建是 complex128，原生 float 构建是
        // complex64。这里不能用 std::complex<double>* 硬读（会按 16B 跨步读 8B
        // 数据，得到错误的 v，W_rec=0 → interference_rec 不写）。
        TORCH_CHECK(vcpu.scalar_type() == TORCH_COMPLEX,
            "phspMatrixFromGram: vector dtype 与 .so 复数精度不一致");
        const ctC10Complex* v =
            reinterpret_cast<const ctC10Complex*>(vcpu.data_ptr());

        // 行/列 = 分波 = (链 × 共振态组合)，同一条分波内的多个 LS 组合**相干求和**：
        //   S_p = Σ_{k∈p} v_k A_k
        //   M_pq = W_total · [p==q ? 1 : 2] · Re Σ_{k∈p,l∈q} v_k conj(v_l) conj(B̄_kl)
        // (*in.nSLvectors) 给出的正是每条分波包含几个 SL 振幅（ls_filter: [4] → 1 行）。
        const int npart = (int)(*in.nSLvectors).size();
        std::vector<int> pstart(npart + 1, 0);
        for (int p = 0; p < npart; ++p) pstart[p + 1] = pstart[p] + (*in.nSLvectors)[p];
        TORCH_CHECK(pstart[npart] == np,
            "(*in.nSLvectors) 之和 (", pstart[npart], ") 与 in.n_amplitudes (", np, ") 不一致");

        torch::Tensor M = torch::zeros({ npart, npart }, opts2d);
        double W = 0.0;
        for (int p = 0; p < npart; ++p)
            for (int q = 0; q < npart; ++q) {
                std::complex<double> acc(0.0, 0.0);
                for (int k = pstart[p]; k < pstart[p + 1]; ++k)
                    for (int l = pstart[q]; l < pstart[q + 1]; ++l) {
                        std::complex<double> b(Br[(size_t)k * np + l], Bi[(size_t)k * np + l]);
                        const std::complex<double> vk(v[k].real(), v[k].imag());
                        const std::complex<double> vl(v[l].real(), v[l].imag());
                        acc += vk * std::conj(vl) * std::conj(b);
                    }
                double val = (p == q ? 1.0 : 2.0) * acc.real() * W_total;
                M[p][q] = val;
                if (p == q || q > p) W += val;
            }
        if (W_out) *W_out = W;
        if (Wtotal_out) *Wtotal_out = W_total;
        return M;
    }

}  // namespace ctpwa
