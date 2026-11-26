// src/HessianContrib.cu — 见 include/HessianContrib.h
#include <HessianContrib.h>

#include "BackendHelpers.h"
#include "DeviceGuard.h"

#include <ComputeHessian.cuh>   // computeDataHessianContrib(Fast) / castF2ToDouble2Kernel

#include <cstdlib>
#include <vector>

namespace ctpwa {

void hessianSegmentContrib(
    const std::vector<ctComplex*>& d_all_amplitudes,
    const std::vector<std::vector<int>>& events,
    const std::vector<std::vector<int>>& amp_offsets,
    const std::vector<double*>& data_weights,
    const std::vector<double*>& bkg_weights,
    bool float_amps,
    int n_amplitudes,
    int n_polar,
    int gpu, int seg,
    const ctComplex* d_v, double* d_hess,
    int n_amp, bool fast)
{

        int nEv = (seg == 1) ? events[gpu][1]
                             : ((events[gpu].size() > 2) ? events[gpu][2] : 0);
        if (nEv <= 0) return;
        const double* d_w = nullptr;
        double* d_w_own = nullptr;
        if (seg == 1) {
            d_w = (gpu < (int)data_weights.size()) ? data_weights[gpu] : nullptr;
        } else {
            cudaMalloc(&d_w_own, nEv * sizeof(double));
            if (gpu < (int)bkg_weights.size() && bkg_weights[gpu] != nullptr) {
                std::vector<double> h_w_neg(nEv);
                cudaMemcpy(h_w_neg.data(), bkg_weights[gpu],
                           nEv * sizeof(double), cudaMemcpyDeviceToHost);
                for (int i = 0; i < nEv; ++i) h_w_neg[i] = -h_w_neg[i];
                cudaMemcpy(d_w_own, h_w_neg.data(),
                           nEv * sizeof(double), cudaMemcpyHostToDevice);
            } else {
                std::vector<double> h_w_neg(nEv, -1.0);
                cudaMemcpy(d_w_own, h_w_neg.data(),
                           nEv * sizeof(double), cudaMemcpyHostToDevice);
            }
            d_w = d_w_own;
        }
        auto run = [&](const ctComplex* d_amp_seg, int n_ev, const double* d_w_c) {
            if (fast) computeDataHessianContribFast(
                d_amp_seg, d_v, d_w_c, d_hess, n_ev, n_polar, n_amp);
            else computeDataHessianContrib(
                d_amp_seg, d_v, d_w_c, d_hess, n_ev, n_polar, n_amp);
        };
        if (!float_amps || ctIsFloatComplex) {
            // 原生 float .so：A 本身就是 ctComplex（cuComplex），kernel 直接读；
            // 全 double .so：A 是 ctComplex（cuDoubleComplex），也直接读。
            run(d_all_amplitudes[gpu] + amp_offsets[gpu][seg], nEv, d_w);
        } else {
            // 窗口化上转（默认 50k 事件/窗口，与 computeDataHessianContribFast 内部 chunk
            // 同量级；CTPWA_HESS_CHUNK 可调——调大=更少 kernel 启动、临时峰值更高）
            int kHChunk = 50000;
            if (const char* e = getenv("CTPWA_HESS_CHUNK")) { int v = atoi(e); if (v > 0) kHChunk = v; }
            const float2* fseg = reinterpret_cast<const float2*>(d_all_amplitudes[gpu])
                                + amp_offsets[gpu][seg];
            const int segStride = n_polar * n_amplitudes;
            for (int off = 0; off < nEv; off += kHChunk) {
                int nch = std::min(kHChunk, nEv - off);
                int nTot = nch * segStride;
                ctComplex* dbl = nullptr;
                cudaMalloc(&dbl, (size_t)nTot * sizeof(ctComplex));
                CTPWA_LAUNCH(castF2ToDouble2Kernel, (nTot + 255) / 256, 256, 0, 
                    fseg + (size_t)off * segStride,
                    reinterpret_cast<cuDoubleComplex*>(dbl), nTot);
                run(dbl, nch, d_w ? d_w + off : nullptr);
                cudaFree(dbl);
            }
        }
        if (d_w_own) cudaFree(d_w_own);
    
}

}  // namespace ctpwa
