// src/MatrixStats.cu — 矩阵统计实现（纯张量数学，见 include/MatrixStats.h）
#include <MatrixStats.h>

#include <cmath>
#include <vector>

namespace ctpwa {

torch::Tensor couplingCovariance(const torch::Tensor& H, int n,
                                 int& nf_out, bool& ok)
{
    ok = false;
    nf_out = 0;
    const int n2 = 2 * n;
    std::vector<int64_t> free_c;
    for (int j = 0; j < n2; ++j)
        if (j != 0 && j != n) free_c.push_back(j);
    nf_out = (int)free_c.size();
    auto empty = torch::empty({0, 0},
        torch::TensorOptions().dtype(torch::kFloat64));
    if (H.numel() == 0) return empty;
    TORCH_CHECK(H.dim() == 2 && H.size(0) == H.size(1),
        "hessian 必须是方阵，收到 dim=", H.dim(),
        " size0=", H.dim() > 0 ? H.size(0) : -1);
    TORCH_CHECK(H.size(0) >= n2,
        "hessian 维度 ", H.size(0), " 小于耦合参数维度 ", n2);
    if (nf_out == 0) { ok = true; return empty; }

    std::vector<int64_t> free_all = free_c;
    for (int t = n2; t < (int)H.size(0); ++t) free_all.push_back(t);
    auto idx = torch::tensor(free_all, torch::kInt64).to(H.device());
    torch::Tensor Hf = H.index_select(0, idx).index_select(1, idx)
                           .to(torch::kCPU).contiguous();
    Hf = 0.5 * (Hf + Hf.transpose(0, 1));

    double scale = Hf.diagonal().abs().max().item<double>();
    if (!(scale > 0.0) || !std::isfinite(scale)) return empty;
    double emin = 0.0;
    try {
        auto eigs = torch::linalg_eigvalsh(Hf);
        emin = eigs[0].item<double>();
    } catch (const std::exception& e) {
        std::cerr << "[ctpwa] Hessian 特征分解失败（" << e.what()
                  << "），按非正定处理" << std::endl;
        return empty;
    }
    // 联合自由块（耦合 + θ）必须正定；只用耦合块判正定会漏掉 θ 方向的
    // 负曲率，也会把条件协方差误当成边际协方差。
    // 阈值必须按**联合块**的量级缩放：耦合曲率可能比 θ 块大 1e8 倍，
    // 用绝对值 1e-8 会把完全正常的 θ 单位阵误判成非正定。
    if (!(emin > 1e-12 * scale)) return empty;
    torch::Tensor cov_full;
    try {
        cov_full = torch::linalg_inv(Hf);
    } catch (const std::exception& e) {
        std::cerr << "[ctpwa] Hessian 求逆失败（" << e.what()
                  << "），跳过参数误差" << std::endl;
        return empty;
    }
    ok = true;
    return cov_full.slice(0, 0, nf_out).slice(1, 0, nf_out).contiguous();
}

}  // namespace ctpwa
