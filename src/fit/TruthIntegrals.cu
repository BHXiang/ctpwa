// src/TruthIntegrals.cu — 见 include/TruthIntegrals.h
#include <TruthIntegrals.h>

#include "BackendHelpers.h"
#include "DeviceGuard.h"
#include "MixedPrecision.h"

#include <ComputeBF.cuh>   // computeBranchingFractions

#include <cuda_runtime.h>

#include <cstdio>

namespace ctpwa {

void computeTruthIntegrals(
    const torch::Tensor& extended_vector,
    const std::vector<ctComplex*>& d_amps,
    const std::vector<int>& ev_per_gpu,
    double* out_partial,
    double* out_scattering,
    double* out_square,
    int npartials,
    const std::vector<ctComplex*>& ws_complex,
    const std::vector<ctComplex*>& ws_matrix,
    const std::vector<ctComplex*>& ws_upcast,
    const std::vector<double*>& d_weights_per_gpu,
    const std::vector<int>& n_slvectors,
    bool float_amps,
    int n_polar,
    int n_amplitudes,
    int primary_dev)
{
        for (size_t gpu = 0; gpu < d_amps.size(); ++gpu) {
            int nt = ev_per_gpu[gpu];
            if (nt <= 0 || d_amps[gpu] == nullptr) continue;

            int* d_nsl = nullptr;
            ctpwa::DeviceGuard dev_guard(gpu);   // 循环体结束即还原
            TORCH_CHECK(cudaMalloc(&d_nsl, npartials * sizeof(int)) == cudaSuccess,
                        "computeTruthIntegrals: cudaMalloc(d_nsl) 失败");
            cudaMemcpy(d_nsl, n_slvectors.data(), npartials * sizeof(int), cudaMemcpyHostToDevice);

            double* d_p = nullptr;
            TORCH_CHECK(cudaMalloc(&d_p, npartials * sizeof(double)) == cudaSuccess,
                        "computeTruthIntegrals: cudaMalloc(d_p) 失败");
            cudaMemset(d_p, 0, npartials * sizeof(double));
            double* d_s = nullptr;
            TORCH_CHECK(cudaMalloc(&d_s, npartials * npartials * sizeof(double)) == cudaSuccess,
                        "computeTruthIntegrals: cudaMalloc(d_s) 失败");
            cudaMemset(d_s, 0, npartials * npartials * sizeof(double));
            double* d_t = nullptr;
            TORCH_CHECK(cudaMalloc(&d_t, sizeof(double)) == cudaSuccess,
                        "computeTruthIntegrals: cudaMalloc(d_t) 失败");
            cudaMemset(d_t, 0, sizeof(double));
            double* d_sq = nullptr;
            if (out_square != nullptr) {
                TORCH_CHECK(cudaMalloc(&d_sq, npartials * sizeof(double)) == cudaSuccess,
                            "computeTruthIntegrals: cudaMalloc(d_sq) 失败");
                cudaMemset(d_sq, 0, npartials * sizeof(double));
            }
            // computeBranchingFractions 可能抛异常；每卡的临时归约缓冲必须释放。
            auto tmp_guard = makeScopeExit([&]() {
                if (d_sq) cudaFree(d_sq);
                cudaFree(d_p);
                cudaFree(d_s);
                cudaFree(d_t);
                cudaFree(d_nsl);
            });

            auto vg = extended_vector.to(ctpwa_dev(gpu));
            // float 模式: 该批振幅缓冲为 float2 → 上转 double 段后再进双精度
            // computeBranchingFractions（批 ≤ 100k 事件, 临时翻倍有界, 与驻留 A 无关）
            const ctComplex* d_mat = d_amps[gpu];
            ctComplex* d_amp_dbl = nullptr;
            if (float_amps && !ctIsFloatComplex) {
                // 复用调用方的 upcast workspace（旧实现每次都 cudaMalloc/cudaFree）
                d_amp_dbl = upcastAmpSegToDouble(d_amps[gpu],
                    (size_t)nt * n_polar * n_amplitudes,
                    (gpu < ws_upcast.size()) ? ws_upcast[gpu] : nullptr);
                d_mat = d_amp_dbl;
            }
            const double* d_w = (gpu < d_weights_per_gpu.size())
                ? d_weights_per_gpu[gpu] : nullptr;
            computeBranchingFractions(d_mat,
                reinterpret_cast<const ctComplex*>(vg.data_ptr()),
                d_p, d_s, d_t, d_sq, d_nsl, npartials, nt, n_amplitudes, n_polar,
                ws_complex[gpu], ws_matrix[gpu], d_w);
            if (d_amp_dbl != nullptr
                && (gpu >= ws_upcast.size() || d_amp_dbl != ws_upcast[gpu]))
                cudaFree(d_amp_dbl);

            std::vector<double> hp(npartials), hs(npartials * npartials); double ht;
            cudaMemcpy(hp.data(), d_p, npartials * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(), d_s, npartials * npartials * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(&ht, d_t, sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < npartials; ++i) {
                out_partial[i] += hp[i];
                for (int j = 0; j < npartials; ++j)
                    out_scattering[i * npartials + j] += hs[i * npartials + j];
            }
            if (d_sq != nullptr) {
                std::vector<double> hsq(npartials);
                cudaMemcpy(hsq.data(), d_sq, npartials * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < npartials; ++i) out_square[i] += hsq[i];
            }
            // 临时缓冲由 tmp_guard 在本次 GPU 迭代结束时释放（含异常路径）。
        }
        // 多 GPU 循环后恢复主设备（FF/EFF/writeResult 等随后会做 torch/耦合运算）
        ctpwa::ensure_device(primary_dev);
    
}

}  // namespace ctpwa
