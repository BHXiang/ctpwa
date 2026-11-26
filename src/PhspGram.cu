// src/PhspGram.cu — 见 include/PhspGram.h
#include <PhspGram.h>

#include "ComplexType.h"
#include "PrecCpx.h"

#include <cuda_runtime.h>

#include "DeviceGuard.h"

#include <cmath>
#include <complex>
#include <vector>

namespace ctpwa {

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
