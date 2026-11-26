// src/SLAmpsIO.cu — 见 include/SLAmpsIO.h
#include <SLAmpsIO.h>

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <cstdio>

namespace ctpwa {

torch::Tensor slAmpsTensor(const AmpCalc& amp_calc, bool phsp_freed)
{
    TORCH_CHECK(!phsp_freed,
        "phsp 流式模式（free_phsp_amplitudes）下 SL 振幅表不含 phsp 事件，"
        "getSLAmpsTensor 不可用");
    auto& cas_list = const_cast<AmpCalc&>(amp_calc).casList();
    TORCH_CHECK(!cas_list.empty(), "No cas available");
    int nEv = static_cast<int>(cas_list[0]->getNEventsVec()[0]);
    int nPol = static_cast<int>(cas_list[0]->getNPolarizations());
    int totalSL = 0;
    for (auto& cas : cas_list) totalSL += static_cast<int>(cas->getNSLCombs());
    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor result = torch::empty({totalSL, nEv, nPol}, options);
    auto acc = result.accessor<c10::complex<double>, 3>();
    int sl_off = 0;
    for (auto& cas : cas_list) {
        TORCH_CHECK(cas->hasSLAmps(),
            "getSLAmpsTensor: 该链组合无自由参数，静态 SL 表已按最小保存释放"
            "（A 已物化；如需 SL 数据请以 CTPWA_KEEP_SLAMP=1 重建）");
        int nSL = static_cast<int>(cas->getNSLCombs());
        int n = nEv * nPol * nSL;
        thrust::complex<double>* h_buf = new thrust::complex<double>[n];
        cudaMemcpy(h_buf, cas->getSLAmps(0), n * sizeof(thrust::complex<double>),
                   cudaMemcpyDeviceToHost);
        for (int s = 0; s < nSL; ++s)
            for (int e = 0; e < nEv; ++e)
                for (int p = 0; p < nPol; ++p) {
                    auto v = h_buf[s * nEv * nPol + e * nPol + p];
                    acc[sl_off + s][e][p] = c10::complex<double>(v.real(), v.imag());
                }
        delete[] h_buf;
        sl_off += nSL;
    }
    return result;
}

void saveSLAmpsToFile(const AmpCalc& amp_calc, const std::string& filename, bool phsp_freed)
{
    TORCH_CHECK(!phsp_freed,
        "phsp 流式模式（free_phsp_amplitudes）下 SL 振幅表不含 phsp 事件，"
        "saveSLAmps 不可用");
    auto& cas_list = const_cast<AmpCalc&>(amp_calc).casList();
    if (cas_list.empty()) { printf("saveSLAmps: no cas available\n"); return; }
    auto& cas = cas_list[0];
    TORCH_CHECK(cas->hasSLAmps(),
        "saveSLAmps: 该链组合无自由参数，静态 SL 表已按最小保存释放"
        "（A 已物化；如需 SL 数据请以 CTPWA_KEEP_SLAMP=1 重建）");
    int nEv = static_cast<int>(cas->getNEventsVec()[0]);
    int nPol = static_cast<int>(cas->getNPolarizations());
    int nSL = static_cast<int>(cas->getNSLCombs());
    int total = nEv * nPol * nSL;
    thrust::complex<double>* h_buf = new thrust::complex<double>[total];
    cudaMemcpy(h_buf, cas->getSLAmps(0), total * sizeof(thrust::complex<double>),
               cudaMemcpyDeviceToHost);
    FILE* f = fopen(filename.c_str(), "wb");
    fwrite(h_buf, sizeof(thrust::complex<double>), total, f);
    fclose(f);
    delete[] h_buf;
    printf("saveSLAmps: saved %d complex numbers to %s (nEv=%d nPol=%d nSL=%d)\n",
           total, filename.c_str(), nEv, nPol, nSL);
}

}  // namespace ctpwa
