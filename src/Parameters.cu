#include <Parameters.cuh>
#include <AmpGen.cuh>

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

// ============================================================
// Parameters: free ↔ extended vector transforms
// ============================================================

void Parameters::initialize(
    int n_free_vector,
    const std::vector<std::vector<int>>& con_ids,
    const std::vector<std::vector<std::complex<double>>>& con_values)
{
    n_free_vector_ = n_free_vector;
    n_extended_ = n_free_vector;
    con_indices_ = con_ids;
    con_values_ = con_values;

    for (const auto& vecid : con_ids) {
        if (!vecid.empty()) {
            auto max_it = std::max_element(vecid.begin(), vecid.end());
            n_extended_ = std::max(n_extended_, *max_it + 1);
        }
    }

    groups_.clear();
    for (size_t i = 0; i < con_ids.size(); ++i) {
        const auto& vecid = con_ids[i];
        const auto& values = con_values[i];

        if (vecid.empty() || values.empty() || vecid.size() != values.size())
            continue;

        auto min_it = std::min_element(vecid.begin(), vecid.end());
        int origin_idx = static_cast<int>(std::distance(vecid.begin(), min_it));
        int origin_id = vecid[origin_idx];

        if (origin_id < 0 || origin_id >= n_free_vector)
            continue;

        std::complex<double> origin_coeff = values[origin_idx];
        double oc_re = std::real(origin_coeff);
        double oc_im = std::imag(origin_coeff);

        if (std::abs(oc_re) < 1e-10 || std::abs(oc_im) < 1e-10)
            continue;

        ConstraintGroup group;
        group.origin_id = origin_id;
        group.origin_idx_in_group = origin_idx;

        for (size_t j = 0; j < vecid.size(); ++j) {
            int ext_id = vecid[j];
            if (ext_id < 0 || ext_id >= n_extended_ || j >= values.size())
                continue;

            group.ext_indices.push_back(ext_id);

            if (static_cast<int>(j) == origin_idx) {
                group.real_ratios.push_back(1.0f);
                group.imag_ratios.push_back(1.0f);
            } else {
                std::complex<double> ec = values[j];
                group.real_ratios.push_back(static_cast<float>(std::real(ec) / oc_re));
                group.imag_ratios.push_back(static_cast<float>(std::imag(ec) / oc_im));
            }
        }

        groups_.push_back(group);
    }

    initialized_ = true;
}

torch::Tensor Parameters::extendVector(
    const torch::Tensor& free_vector,
    const torch::Device& device) const
{
    TORCH_CHECK(free_vector.is_complex(), "Input vector must be complex");
    TORCH_CHECK(free_vector.dim() == 1, "Input vector must be 1-dimensional");

    const int original_size = free_vector.numel();

    int ext_size = n_extended_;
    if (ext_size < original_size) ext_size = original_size;

    if (ext_size == original_size) {
        return free_vector.to(device);
    }

    auto options = torch::TensorOptions().dtype(TORCH_COMPLEX).device(device);
    torch::Tensor ext = torch::zeros({ext_size}, options);

    torch::Tensor indices = torch::arange(0, original_size, torch::kLong).to(device);
    ext.index_copy_(0, indices, free_vector.to(device));

    for (const auto& g : groups_) {
        torch::Tensor origin_val = ext[g.origin_id];

        torch::Tensor real_part = (origin_val + torch::conj(origin_val)) / 2.0f;
        torch::Tensor imag_part =
            (origin_val - torch::conj(origin_val)) / (2.0f * c10::complex<float>(0, 1));

        for (size_t j = 0; j < g.ext_indices.size(); ++j) {
            if (static_cast<int>(j) == g.origin_idx_in_group) continue;

            int ext_id = g.ext_indices[j];
            torch::Tensor ext_val =
                g.real_ratios[j] * real_part + c10::complex<float>(0, 1) * g.imag_ratios[j] * imag_part;
            ext[ext_id] = ext_val;
        }
    }

    return ext;
}

torch::Tensor Parameters::collapseVectorGrad(
    const torch::Tensor& extended_grad,
    int original_size) const
{
    torch::Device device = extended_grad.device();
    auto options = torch::TensorOptions().dtype(TORCH_COMPLEX).device(device);
    torch::Tensor grad = torch::zeros({original_size}, options);

    if (original_size > 0) {
        grad.copy_(extended_grad.slice(0, 0, original_size));
    }

    if (groups_.empty()) return grad;

    for (const auto& g : groups_) {
        std::vector<int> ext_idx_vec;
        std::vector<float> rr_vec, ir_vec;
        for (size_t j = 0; j < g.ext_indices.size(); ++j) {
            if (static_cast<int>(j) == g.origin_idx_in_group) continue;
            int eid = g.ext_indices[j];
            if (eid < 0 || eid >= extended_grad.numel()) continue;
            ext_idx_vec.push_back(eid);
            rr_vec.push_back(g.real_ratios[j]);
            ir_vec.push_back(g.imag_ratios[j]);
        }

        if (ext_idx_vec.empty()) continue;

        torch::Tensor ext_idx_t = torch::tensor(ext_idx_vec, torch::kLong).to(device);
        torch::Tensor rr_t = torch::tensor(rr_vec, TORCH_FLOAT).to(device);
        torch::Tensor ir_t = torch::tensor(ir_vec, TORCH_FLOAT).to(device);

        torch::Tensor ext_grads = extended_grad.index_select(0, ext_idx_t);

        // 全部用同 dtype 张量运算（torch 的 complex 标量运算符在 complex128 下缺失）
        torch::Tensor one_i = torch::complex(
            torch::tensor(0.0, torch::TensorOptions().dtype(TORCH_FLOAT)),
            torch::tensor(1.0, torch::TensorOptions().dtype(TORCH_FLOAT)));
        torch::Tensor ext_re = (ext_grads + torch::conj(ext_grads)) / 2.0f;
        torch::Tensor ext_im = (ext_grads - torch::conj(ext_grads)) / (2.0f * one_i);

        torch::Tensor rr_c = rr_t.to(TORCH_COMPLEX);
        torch::Tensor ir_c = ir_t.to(TORCH_COMPLEX);
        torch::Tensor contrib =
            (rr_c * ext_re + ir_c * ext_im * one_i).sum();

        grad[g.origin_id] = grad[g.origin_id] + contrib;
    }

    return grad;
}

std::pair<torch::Tensor, torch::Tensor> Parameters::splitParams(
    const torch::Tensor& params) const
{
    TORCH_CHECK(params.dim() == 1, "params must be 1-dimensional");
    TORCH_CHECK(params.dtype() == torch::kFloat64, "params must be float64");
    TORCH_CHECK(params.numel() == nParams(),
        "params size mismatch: got ", params.numel(),
        ", expected ", nParams());

    int nv = n_free_vector_;
    int nt = n_free_theta_;

    torch::Tensor real_part = params.slice(0, 0, nv);
    torch::Tensor imag_part = params.slice(0, nv, 2 * nv);
    torch::Tensor vector = torch::complex(
        real_part.to(TORCH_FLOAT), imag_part.to(TORCH_FLOAT));

    torch::Tensor theta;
    if (nt > 0) {
        theta = params.slice(0, 2 * nv, 2 * nv + nt);
    } else {
        theta = torch::empty({0},
            torch::TensorOptions().dtype(torch::kFloat64).device(params.device()));
    }

    return {vector, theta};
}

void Parameters::setCouplingMatrix(const CouplingMatrixResult& cm)
{
    coupling_matrix_ = cm;
    has_coupling_matrix_ = true;
    initialize(cm.n_free, {}, {});
    uploadCouplingData();
}

void Parameters::freeCouplingData()
{
    if (d_amp_chain_)       { cudaFree(d_amp_chain_);       d_amp_chain_       = nullptr; }
    if (d_step_offsets_)    { cudaFree(d_step_offsets_);    d_step_offsets_    = nullptr; }
    if (d_step_data_)       { cudaFree(d_step_data_);       d_step_data_       = nullptr; }
    if (d_amp_chain_ratio_) { cudaFree(d_amp_chain_ratio_); d_amp_chain_ratio_ = nullptr; }
    if (d_jac_re_)   { cudaFree(d_jac_re_);   d_jac_re_   = nullptr; }
    if (d_jac_im_)   { cudaFree(d_jac_im_);   d_jac_im_   = nullptr; }
    if (d_jac_p_re_) { cudaFree(d_jac_p_re_); d_jac_p_re_ = nullptr; }
    if (d_jac_p_im_) { cudaFree(d_jac_p_im_); d_jac_p_im_ = nullptr; }
    if (d_v_re_)     { cudaFree(d_v_re_);     d_v_re_     = nullptr; }
    if (d_v_im_)     { cudaFree(d_v_im_);     d_v_im_     = nullptr; }
    step_data_len_ = 0;
}

void Parameters::uploadCouplingData()
{
    if (!has_coupling_matrix_) return;
    if (primary_dev_ >= 0) cudaSetDevice(primary_dev_);   // 表固定上传到主设备
    freeCouplingData();
    const auto& r = coupling_matrix_;

    cudaMalloc(&d_amp_chain_, r.n_amps * sizeof(int));
    cudaMemcpy(d_amp_chain_, r.amp_chain.data(), r.n_amps * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_amp_chain_ratio_, r.n_amps * sizeof(double));
    cudaMemcpy(d_amp_chain_ratio_, r.amp_chain_ratio.data(), r.n_amps * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<int> h_offsets(r.n_amps + 1, 0);
    int total = 0;
    for (int i = 0; i < r.n_amps; ++i) {
        h_offsets[i] = total;
        total += (int)r.amp_step_params[i].size();
    }
    h_offsets[r.n_amps] = total;
    cudaMalloc(&d_step_offsets_, (r.n_amps + 1) * sizeof(int));
    cudaMemcpy(d_step_offsets_, h_offsets.data(), (r.n_amps + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_step_data_, total * sizeof(int));
    std::vector<int> h_data(total);
    int pos = 0;
    for (int i = 0; i < r.n_amps; ++i)
        for (int p : r.amp_step_params[i]) h_data[pos++] = p;
    cudaMemcpy(d_step_data_, h_data.data(), total * sizeof(int), cudaMemcpyHostToDevice);
    step_data_len_ = total;
}

// ============================================================
// Fused extend kernel: params [Re_p, Im_p, θ] → [Re_v, Im_v, θ]
//   d_in:  [Re(p_0)..Re(p_{ncf-1}), Im(p_0)..Im(p_{ncf-1}), θ_0..θ_{P-1}]
//   d_out: [Re(v_0)..Re(v_{na-1}),   Im(v_0)..Im(v_{na-1}),   θ_0..θ_{P-1}]
// ============================================================

__global__ void extendCouplingParamsKernel(
    double* d_out, const double* d_in,
    const int* d_amp_chain,
    const int* d_step_offsets,
    const int* d_step_data,
    const double* d_amp_chain_ratio,
    int n_amps, int n_step_free, int ncf, int nt)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_amps) {
        int t = a - n_amps;
        if (t < nt)
            d_out[2 * n_amps + t] = d_in[2 * ncf + t];
        return;
    }

    // v_ext[a] = ratio[a] × chain[c_a] × Π step[k]
    // d_in layout: [Re_chain[0..nch-1], Re_step[0..nst-1], Im_chain[...], Im_step[...]]
    // nch = ncf - n_step_free
    double ratio = d_amp_chain_ratio[a];
    int nch = ncf - n_step_free;
    int c_idx = d_amp_chain[a];                 // chains first
    double re = d_in[c_idx] * ratio;
    double im = d_in[ncf + c_idx] * ratio;

    for (int k = d_step_offsets[a]; k < d_step_offsets[a + 1]; ++k) {
        int s = nch + d_step_data[k];            // steps after chains
        double s_re = d_in[s];
        double s_im = d_in[ncf + s];
        double new_re = re * s_re - im * s_im;
        double new_im = re * s_im + im * s_re;
        re = new_re; im = new_im;
    }

    // Output: [Re(v_all), Im(v_all), θ]
    d_out[a] = re;
    d_out[n_amps + a] = im;
}

void Parameters::extendCouplingParams(
    const double* d_in, double* d_out, int ncf, int nt) const
{
    if (!has_coupling_matrix_) return;
    const auto& cm = coupling_matrix_;
    int total_threads = cm.n_amps + nt;
    int grid = (total_threads + 255) / 256;
    if (primary_dev_ >= 0) cudaSetDevice(primary_dev_);
    extendCouplingParamsKernel<<<grid, 256>>>(
        d_out, d_in,
        d_amp_chain_, d_step_offsets_, d_step_data_, d_amp_chain_ratio_,
        cm.n_amps, cm.n_step_free, ncf, nt);
    cudaDeviceSynchronize();
}

void Parameters::applyCouplingMatrix(const double* d_params, ctComplex* d_v_out) const
{
    if (!has_coupling_matrix_) return;
    const auto& cm = coupling_matrix_;
    int grid = (cm.n_amps + 255) / 256;
    if (primary_dev_ >= 0) cudaSetDevice(primary_dev_);   // 耦合 kernel 固定主设备（防多卡错卡）
    multiplicativeCouplingKernel<<<grid, 256>>>(
        d_v_out, d_params,
        d_amp_chain_, d_step_offsets_, d_step_data_, d_amp_chain_ratio_,
        cm.n_amps, cm.n_step_free, cm.n_free);
    cudaDeviceSynchronize();
}

void Parameters::transformCouplingGradient(
    const ctComplex* d_grad_v, const ctComplex* d_v,
    const double* d_params, double* d_grad_p) const
{
    if (!has_coupling_matrix_) return;
    const auto& cm = coupling_matrix_;
    int grid = (cm.n_free + 255) / 256;
    if (primary_dev_ >= 0) cudaSetDevice(primary_dev_);
    multiplicativeGradientKernel<<<grid, 256>>>(
        d_grad_p, d_grad_v, d_v, d_params,
        d_amp_chain_, d_step_offsets_, d_step_data_,
        cm.n_amps, cm.n_step_free,
        cm.n_free);
    cudaDeviceSynchronize();
}

// ============================================================
// Hessian: precomputeJacobian + full transform (single kernel)
// ============================================================

// Precompute w[a][j] = v[a] / p[j], plus v[a] and p[j] values
__global__ void precomputeJacobianKernel(
    double* d_jac_re, double* d_jac_im,
    double* d_p_re, double* d_p_im,
    double* d_v_re, double* d_v_im,
    const double* d_params,
    const int* d_amp_chain,
    const int* d_step_offsets, const int* d_step_data,
    const double* d_amp_chain_ratio,
    int n_amps, int n_step_free, int n_free, int ncf)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_amps) return;

    int nch = n_free - n_step_free;
    double ratio = d_amp_chain_ratio[a];
    int c_idx = d_amp_chain[a];                 // chains first
    double v_re = d_params[c_idx] * ratio;
    double v_im = d_params[n_free + c_idx] * ratio;
    for (int k = d_step_offsets[a]; k < d_step_offsets[a + 1]; ++k) {
        int s = nch + d_step_data[k];            // steps after chains
        double s_re = d_params[s], s_im = d_params[n_free + s];
        double nre = v_re * s_re - v_im * s_im;
        double nim = v_re * s_im + v_im * s_re;
        v_re = nre; v_im = nim;
    }
    d_v_re[a] = v_re; d_v_im[a] = v_im;

    double pc_re = d_params[c_idx], pc_im = d_params[n_free + c_idx];
    double pc_sq = pc_re * pc_re + pc_im * pc_im;
    if (pc_sq < 1e-30) pc_sq = 1e-30;
    int chain_pos = d_step_offsets[n_amps] + a;
    d_jac_re[chain_pos] = (v_re * pc_re + v_im * pc_im) / pc_sq;
    d_jac_im[chain_pos] = (v_im * pc_re - v_re * pc_im) / pc_sq;

    for (int k = d_step_offsets[a]; k < d_step_offsets[a + 1]; ++k) {
        int s = nch + d_step_data[k];            // steps after chains
        double ps_re = d_params[s], ps_im = d_params[n_free + s];
        double ps_sq = ps_re * ps_re + ps_im * ps_im;
        if (ps_sq < 1e-30) ps_sq = 1e-30;
        d_jac_re[k] = (v_re * ps_re + v_im * ps_im) / ps_sq;
        d_jac_im[k] = (v_im * ps_re - v_re * ps_im) / ps_sq;
    }

    if (a == 0) {
        for (int j = 0; j < ncf; ++j) {
            d_p_re[j] = d_params[j];
            d_p_im[j] = d_params[n_free + j];
        }
    }
}

void Parameters::precomputeJacobian(const double* d_params)
{
    if (!has_coupling_matrix_) return;
    const auto& cm = coupling_matrix_;
    int total = step_data_len_ + cm.n_amps;
    int ncf = cm.n_free;

    if (d_jac_re_) cudaFree(d_jac_re_);
    if (d_jac_im_) cudaFree(d_jac_im_);
    if (d_jac_p_re_) cudaFree(d_jac_p_re_);
    if (d_jac_p_im_) cudaFree(d_jac_p_im_);
    if (d_v_re_) cudaFree(d_v_re_);
    if (d_v_im_) cudaFree(d_v_im_);

    cudaMalloc(&d_jac_re_, total * sizeof(double));
    cudaMalloc(&d_jac_im_, total * sizeof(double));
    cudaMalloc(&d_jac_p_re_, ncf * sizeof(double));
    cudaMalloc(&d_jac_p_im_, ncf * sizeof(double));
    cudaMalloc(&d_v_re_, cm.n_amps * sizeof(double));
    cudaMalloc(&d_v_im_, cm.n_amps * sizeof(double));

    int grid = (cm.n_amps + 255) / 256;
    precomputeJacobianKernel<<<grid, 256>>>(
        d_jac_re_, d_jac_im_, d_jac_p_re_, d_jac_p_im_,
        d_v_re_, d_v_im_, d_params,
        d_amp_chain_, d_step_offsets_, d_step_data_, d_amp_chain_ratio_,
        cm.n_amps, cm.n_step_free, cm.n_free, ncf);
    cudaDeviceSynchronize();
}

// Single kernel: H_fit = J_full^T · H_ext · J_full + sum g·grad^2 v
__global__ void hessianFullTransformKernel(
    double* H_out, const double* H_ext,
    const double* d_jac_re, const double* d_jac_im,
    const double* d_p_re, const double* d_p_im,
    const double* d_v_re, const double* d_v_im,
    const double* d_g_v,
    const int* d_amp_chain, const int* d_step_offsets, const int* d_step_data,
    int na, int ncf, int n_step_free, int jac_chain_base, int nt)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int n_out = 2 * ncf + nt;
    if (row >= n_out || col >= n_out) return;

    int n_ext = 2 * na + nt;
    bool rowV = (row < 2 * ncf), colV = (col < 2 * ncf);

    // Helper: extract (param_idx, Re/Im) from grouped row/col index
    // Grouped: rows 0..n-1 = Re_0..Re_{n-1}, rows n..2n-1 = Im_0..Im_{n-1}
    auto groupedIdx = [](int idx, int n) {
        return (idx < n) ? std::make_pair(idx, 0) : std::make_pair(idx - n, 1);
    };

    int nch = ncf - n_step_free;  // chain params come first in free-param ordering

    if (rowV && colV) {
        // ---- vv block: J_v^T·H_vv·J_v + second-order ----
        auto [j, cr] = groupedIdx(row, ncf);
        auto [k, cc] = groupedIdx(col, ncf);
        bool jCh = (j < nch), kCh = (k < nch);
        int jc = jCh ? j : j - nch;
        int kc = kCh ? k : k - nch;
        double sum = 0.0;

        for (int a = 0; a < na; ++a) {
            double Jjr = 0, Jji = 0; bool ji = false;
            if (jCh) { if (d_amp_chain[a]==jc) { int p=jac_chain_base+a; Jjr=d_jac_re[p]; Jji=d_jac_im[p]; ji=true; } }
            else { for (int s=d_step_offsets[a];s<d_step_offsets[a+1];++s) if (d_step_data[s]==jc) { Jjr=d_jac_re[s]; Jji=d_jac_im[s]; ji=true; break; } }
            if (!ji) continue;
            double JT_R = (cr==0)?Jjr:-Jji, JT_I = (cr==0)?Jji:Jjr;

            for (int b = 0; b < na; ++b) {
                double Jkr=0, Jki=0; bool ki=false;
                if (kCh) { if (d_amp_chain[b]==kc) { int p=jac_chain_base+b; Jkr=d_jac_re[p]; Jki=d_jac_im[p]; ki=true; } }
                else { for (int s=d_step_offsets[b];s<d_step_offsets[b+1];++s) if (d_step_data[s]==kc) { Jkr=d_jac_re[s]; Jki=d_jac_im[s]; ki=true; break; } }
                if (!ki) continue;
                double J_C = (cc==0)?Jkr:-Jki, J_Ic = (cc==0)?Jki:Jkr;

                // Grouped: Re_a at row a, Im_a at na+a
                double Hrr=H_ext[a*n_ext+b],        Hri=H_ext[a*n_ext+na+b];
                double Hir=H_ext[(na+a)*n_ext+b],    Hii=H_ext[(na+a)*n_ext+na+b];
                sum += (JT_R*Hrr+JT_I*Hir)*J_C + (JT_R*Hri+JT_I*Hii)*J_Ic;
            }
        }

        // ---- 二阶项: Σ_a g_v[a] · ∂²v_a/(∂p_j ∂p_k) ----
        // v_a = ratio_a · Π_{f ∈ factors(a)} p_f   (每个因子幂次 1) ⇒
        //   · j == k, 或 j/k 中有一个不是 a 的因子  →  0;
        //   · j ≠ k 且都是 a 的因子  →  ∂²v_a/∂p_j∂p_k = v_a / (p_j · p_k)
        //     (复数; trans 的 ratio 已经含在 v_a 里, 不需要额外处理)
        // 统一处理 chain×step **与 step×step**; 旧版只做前者、把后者整类跳过,
        // 并把 z = v_a/(p_j p_k) 硬编码成 1 —— 只在"每个振幅恰 2 个因子且 ratio=1"
        // 时才正确, 这就是 >=3 因子 / ratio≠1 时 Hessian 错的原因。
        // (VSPACE 下每个振幅只有 1 个因子 → 这里自动恒为 0, 与真值一致。)
        if (d_g_v && j != k) {
            double gt = 0.0;
            for (int a = 0; a < na; ++a) {
                auto is_factor = [&](int pidx) -> bool {
                    if (pidx < nch) return d_amp_chain[a] == pidx;
                    int sc = pidx - nch;            // step_data 存的是去掉 nch 的偏移
                    for (int ss = d_step_offsets[a]; ss < d_step_offsets[a + 1]; ++ss)
                        if (d_step_data[ss] == sc) return true;
                    return false;
                };
                if (!is_factor(j) || !is_factor(k)) continue;
                // z = v_a / (p_j · p_k)
                double pr = d_p_re[j], pi = d_p_im[j];
                double qr = d_p_re[k], qi = d_p_im[k];
                double dr = pr * qr - pi * qi;
                double di = pr * qi + pi * qr;
                double dsq = dr * dr + di * di;
                if (dsq < 1e-300) dsq = 1e-300;
                double vr = d_v_re[a], vi = d_v_im[a];
                double zr = (vr * dr + vi * di) / dsq;
                double zi = (vi * dr - vr * di) / dsq;
                // 实坐标 8 块, 系数 = Re/Im(z) (j==k 已排除)
                //   ∂²V_r: (Re,Re)=+zr (Re,Im)=-zi (Im,Re)=-zi (Im,Im)=-zr
                //   ∂²V_i: (Re,Re)=+zi (Re,Im)=+zr (Im,Re)=+zr (Im,Im)=-zi
                double d2Vr, d2Vi;
                if (cr == 0 && cc == 0)      { d2Vr =  zr; d2Vi =  zi; }
                else if (cr == 0 && cc == 1) { d2Vr = -zi; d2Vi =  zr; }
                else if (cr == 1 && cc == 0) { d2Vr = -zi; d2Vi =  zr; }
                else                         { d2Vr = -zr; d2Vi = -zi; }
                gt += d_g_v[a] * d2Vr + d_g_v[na + a] * d2Vi;
            }
            sum += gt;
        }
        H_out[row*n_out+col] = sum;
        return;
    }

    if (rowV) {
        // ---- vtheta: J_v^T · H_vtheta ----
        auto [j, comp] = groupedIdx(row, ncf);
        int t = col-2*ncf;
        bool jCh = (j < nch); int jc = jCh ? j : j - nch;
        double sum = 0.0;
        for (int a=0;a<na;++a) {
            double Jr=0,Ji=0; bool ji=false;
            if (jCh) { if (d_amp_chain[a]==jc) { int p=jac_chain_base+a; Jr=d_jac_re[p]; Ji=d_jac_im[p]; ji=true; } }
            else { for (int s=d_step_offsets[a];s<d_step_offsets[a+1];++s) if (d_step_data[s]==jc) { Jr=d_jac_re[s]; Ji=d_jac_im[s]; ji=true; break; } }
            if (!ji) continue;
            double Hr=H_ext[a*n_ext+2*na+t], Hi=H_ext[(na+a)*n_ext+2*na+t];
            sum += (comp==0) ? (Jr*Hr+Ji*Hi) : (-Ji*Hr+Jr*Hi);
        }
        H_out[row*n_out+col] = sum;
        return;
    }

    if (colV) {
        // ---- thetav: H_thetav · J_v ----
        int t = row-2*ncf;
        auto [j, comp] = groupedIdx(col, ncf);
        bool jCh = (j < nch); int jc = jCh ? j : j - nch;
        double sum = 0.0;
        for (int a=0;a<na;++a) {
            double Jr=0,Ji=0; bool ji=false;
            if (jCh) { if (d_amp_chain[a]==jc) { int p=jac_chain_base+a; Jr=d_jac_re[p]; Ji=d_jac_im[p]; ji=true; } }
            else { for (int s=d_step_offsets[a];s<d_step_offsets[a+1];++s) if (d_step_data[s]==jc) { Jr=d_jac_re[s]; Ji=d_jac_im[s]; ji=true; break; } }
            if (!ji) continue;
            double Hr=H_ext[(2*na+t)*n_ext+a], Hi=H_ext[(2*na+t)*n_ext+na+a];
            sum += (comp==0) ? (Hr*Jr+Hi*Ji) : (Hr*(-Ji)+Hi*Jr);
        }
        H_out[row*n_out+col] = sum;
        return;
    }

    // ---- thetatheta: direct copy ----
    int ti=row-2*ncf, tj=col-2*ncf;
    H_out[row*n_out+col] = H_ext[(2*na+ti)*n_ext + (2*na+tj)];
}

void Parameters::transformExtendedHessian(
    const double* d_H_ext,
    const double* d_params, const double* d_g_v,
    double* d_H_fitting, int na, int ncf, int nt) const
{
    if (!has_coupling_matrix_) return;
    int jcb = step_data_len_;
    int n_out = 2*ncf + nt;
    dim3 block(16, 16);
    dim3 grid((n_out+15)/16, (n_out+15)/16);
    hessianFullTransformKernel<<<grid, block>>>(
        d_H_fitting, d_H_ext,
        d_jac_re_, d_jac_im_,
        d_jac_p_re_, d_jac_p_im_,
        d_v_re_, d_v_im_, d_g_v,
        d_amp_chain_, d_step_offsets_, d_step_data_,
        na, ncf, coupling_matrix_.n_step_free, jcb, nt);
    cudaDeviceSynchronize();
}
// ============================================================
// CouplingMatrixBuilder: buildWithTrans implementation
// ============================================================

namespace {

// chain_key 结构: <chain.name> + ("_" + <中间态名> + "[" + <共振态名> + "]")*
struct TransKeyParts {
    std::string group;                                       // chain.name
    std::vector<std::pair<std::string, std::string>> elems;  // (中间态名, 共振态名)
};

// 删除 name 在 s 中的所有出现并规整 '_' 分隔（归一化 chain.name 用）
std::string transStripName(const std::string& s, const std::string& name)
{
    if (name.empty()) return s;
    std::string raw;
    size_t pos = 0;
    while (true) {
        size_t f = s.find(name, pos);
        if (f == std::string::npos) { raw += s.substr(pos); break; }
        raw += s.substr(pos, f - pos);
        pos = f + name.size();
    }
    std::string out;
    for (char c : raw) {
        if (c == '_' && (out.empty() || out.back() == '_')) continue;
        out += c;
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

}  // namespace

CouplingMatrixResult CouplingMatrixBuilder::buildWithTrans(
    const std::vector<std::vector<std::string>>& trans_names,
    const std::vector<std::vector<double>>& trans_ratios) const
{
    CouplingMatrixResult r;
    r.steps = steps_;
    r.amp_map = amp_map_;
    r.n_amps = static_cast<int>(amp_map_.size());

    // --- 收集链级参数 ---
    std::vector<std::string> chain_keys;
    for (const auto& am : amp_map_) {
        auto it = std::find(chain_keys.begin(), chain_keys.end(), am.chain_key);
        if (it == chain_keys.end()) chain_keys.push_back(am.chain_key);
    }

    // chain_key → (chain.name, 元素表)。chain.name 自身可能含 '_'（如 R_Keta），
    // 不能从 '[' 反推, 用已知链名（chain_step_order_ 的键）做最长前缀匹配。
    auto parseKey = [&](const std::string& key) {
        TransKeyParts p;
        for (const auto& kv : chain_step_order_) {
            const std::string& nm = kv.first;
            if (key.size() >= nm.size() && key.compare(0, nm.size(), nm) == 0 &&
                nm.size() > p.group.size())
                p.group = nm;
        }
        size_t pos = p.group.size();
        while (pos < key.size() && key[pos] == '_') {
            size_t lb = key.find('[', pos);
            if (lb == std::string::npos) break;
            size_t rb = key.find(']', lb);
            if (rb == std::string::npos) break;
            p.elems.emplace_back(key.substr(pos + 1, lb - pos - 1),
                                 key.substr(lb + 1, rb - lb - 1));
            pos = rb + 1;
        }
        return p;
    };

    // --- 应用 trans 约束 ---
    std::map<int, std::pair<int, double>> step_fold_map;
    std::map<int, std::pair<int, double>> chain_fold_map;  // cB → {cA, ratio}

    // 折叠一对链: chain_fold_map + 两组 step 按注册顺序逐位折叠
    auto foldPair = [&](int cB, int cA, double ratio) -> bool {
        if (cB == cA || chain_fold_map.count(cB)) return false;
        chain_fold_map[cB] = {cA, ratio};
        for (const auto& kvB : chain_step_order_) {
            if (chain_keys[cB].compare(0, kvB.first.size(), kvB.first) != 0) continue;
            for (const auto& kvA : chain_step_order_) {
                if (chain_keys[cA].compare(0, kvA.first.size(), kvA.first) != 0) continue;
                if (kvA.first == kvB.first) break;
                const auto& sB_list = kvB.second;
                const auto& sA_list = kvA.second;
                for (size_t pi = 0; pi < sB_list.size() && pi < sA_list.size(); ++pi)
                    if (!step_fold_map.count(sB_list[pi]))
                        step_fold_map[sB_list[pi]] = {sA_list[pi], ratio};
                break;
            }
        }
        return true;
    };

    for (size_t ti = 0; ti < trans_names.size(); ++ti) {
        const std::vector<std::string>& names = trans_names[ti];
        if (names.size() < 2) {
            if (!names.empty())
                fprintf(stderr, "[ctpwa] warn: trans 约束至少需要 2 个链名, 跳过 [%s]\n",
                        names[0].c_str());
            continue;
        }
        // 比值: names[1..] 依次对应 trans_ratios[ti][0..]（names[0] 为基准, 比值 1）
        const std::vector<double> no_ratio;
        const std::vector<double>& vals =
            (ti < trans_ratios.size()) ? trans_ratios[ti] : no_ratio;
        std::vector<double> ratios(names.size(), 1.0);
        for (size_t j = 1; j < names.size(); ++j)
            if (j - 1 < vals.size()) ratios[j] = vals[j - 1];
        if (vals.size() > names.size() - 1)
            fprintf(stderr,
                "[ctpwa] warn: trans 约束 [%s, ...] 有 %zu 个链名但给了 %zu 个比值; 多余的忽略\n",
                names[0].c_str(), names.size(), vals.size());

        // 每条匹配链: group（chain.name）/ groupKey（去掉被约束点名部分后的归一化
        // 链名, 作为"同一拓扑"的配对键）/ elems（各层 (中间态, 共振态)）
        struct Hit {
            int ci;
            std::string groupKey;
            std::vector<std::pair<std::string, std::string>> elems;
        };
        std::vector<std::vector<Hit>> hits(names.size());

        // 去掉链名的 "_<序号>" 实例后缀（深层子中间态的实例语义用）
        auto stripInstanceSuffix = [](std::string s) {
            auto us = s.rfind('_');
            if (us == std::string::npos || us + 1 >= s.size()) return s;
            std::string num = s.substr(us + 1);
            if (num.find_first_not_of("0123456789") != std::string::npos) return s;
            s.erase(us);
            return s;
        };

        for (size_t j = 0; j < names.size(); ++j) {
            const std::string& nm = names[j];
            std::vector<int> matched;
            for (size_t ci = 0; ci < chain_keys.size(); ++ci)
                if (chain_keys[ci].find(nm) != std::string::npos)
                    matched.push_back(static_cast<int>(ci));
            if (matched.empty()) {
                // 实例语义回退: "<中间态名>_<出现序号>" —— 该中间态是深层子中间态
                // （链名不含此串）时, 按包含它的链组出现次序定位。
                std::string base = nm;
                int inst = -1;
                auto us = nm.rfind('_');
                if (us != std::string::npos) {
                    std::string num = nm.substr(us + 1);
                    if (!num.empty() && num.find_first_not_of("0123456789") == std::string::npos) {
                        base = nm.substr(0, us);
                        inst = atoi(num.c_str());
                    }
                }
                if (inst >= 0) {
                    std::string tag = "_" + base + "[";
                    std::map<int, std::vector<int>> grp_chains;  // 组序号 → [链 id…]
                    std::map<std::string, int> grp_of;           // 链前缀 → 组序号
                    for (size_t ci = 0; ci < chain_keys.size(); ++ci) {
                        if (chain_keys[ci].find(tag) == std::string::npos) continue;
                        auto first_br = chain_keys[ci].find('[');
                        if (first_br == std::string::npos) continue;
                        std::string prefix = chain_keys[ci].substr(0, first_br);
                        auto git = grp_of.find(prefix);
                        int gi;
                        if (git == grp_of.end()) { gi = (int)grp_chains.size(); grp_of[prefix] = gi; }
                        else gi = git->second;
                        grp_chains[gi].push_back(static_cast<int>(ci));
                    }
                    if (inst < (int)grp_chains.size()) matched = grp_chains[inst];
                }
            }
            if (matched.empty()) {
                fprintf(stderr,
                    "[ctpwa] warn: trans 约束 [%s, ...] 的名字 \"%s\" 未匹配到任何链; 忽略该名字\n",
                    names[0].c_str(), nm.c_str());
                continue;
            }
            for (int ci : matched) {
                TransKeyParts p = parseKey(chain_keys[ci]);
                Hit h;
                h.ci = ci;
                // 配对键 = 链名去掉被约束点名的部分（约束名可能是中间态名, 也可能是
                // "omega_Rx_rho0" 这类链名/子链片段）。名字没出现在链名里（深层子中
                // 间态的实例语义）时, 再去掉 "_<序号>" 实例后缀。
                h.groupKey = transStripName(p.group, nm);
                if (p.group.find(nm) == std::string::npos)
                    h.groupKey = stripInstanceSuffix(h.groupKey);
                h.elems = p.elems;
                hits[j].push_back(std::move(h));
            }
        }
        if (hits[0].empty()) continue;

        // groupKey → names[0] 的下标（保持 chain_keys 顺序）
        std::map<std::string, std::vector<int>> gA;
        for (size_t a = 0; a < hits[0].size(); ++a) gA[hits[0][a].groupKey].push_back((int)a);

        // 逐位比较两条链的共振组合差异（长度不同每缺一位记 2, 更贵）
        auto elemDiff = [](const std::vector<std::pair<std::string, std::string>>& a,
                           const std::vector<std::pair<std::string, std::string>>& b) {
            int d = 0;
            size_t n = std::max(a.size(), b.size());
            for (size_t i = 0; i < n; ++i) {
                if (i >= a.size() || i >= b.size()) { d += 2; continue; }
                if (a[i].first != b[i].first || a[i].second != b[i].second) ++d;
            }
            return d;
        };
        struct Cand { int diff; size_t ai, bi; };
        auto candLess = [](const Cand& x, const Cand& y) {
            if (x.diff != y.diff) return x.diff < y.diff;
            if (x.ai != y.ai) return x.ai < y.ai;
            return x.bi < y.bi;
        };

        int paired = 0, folded = 0;
        std::vector<bool> noted_diff(names.size(), false);  // 每个名字只提示一次
        for (const auto& kv : gA) {
            const std::string& gk = kv.first;
            const std::vector<int>& a_idx = kv.second;
            for (size_t j = 1; j < names.size(); ++j) {
                std::vector<int> b_idx;
                for (size_t b = 0; b < hits[j].size(); ++b)
                    if (hits[j][b].groupKey == gk) b_idx.push_back((int)b);
                if (b_idx.empty()) continue;
                // 最小差异贪心配对: 同共振组合（差异 0/1, 差异处就是被约束点名的
                // 那一层）优先一一对应, 不会把 f0_2020 配到 f2_2340 上。
                std::vector<Cand> cand;
                for (size_t ai = 0; ai < a_idx.size(); ++ai)
                    for (size_t bi = 0; bi < b_idx.size(); ++bi)
                        cand.push_back({elemDiff(hits[0][a_idx[ai]].elems,
                                                 hits[j][b_idx[bi]].elems), ai, bi});
                std::sort(cand.begin(), cand.end(), candLess);
                std::vector<bool> usedA(a_idx.size(), false), usedB(b_idx.size(), false);
                for (const auto& c : cand) {
                    if (usedA[c.ai] || usedB[c.bi]) continue;
                    usedA[c.ai] = usedB[c.bi] = true;
                    const Hit& ha = hits[0][a_idx[c.ai]];
                    const Hit& hb = hits[j][b_idx[c.bi]];
                    ++paired;
                    if (c.diff > 1)
                        fprintf(stderr,
                            "[ctpwa] warn: trans [%s, %s] 配对的两条链差异较大 (%d 处不同): "
                            "%s ↔ %s\n",
                            names[0].c_str(), names[j].c_str(), c.diff,
                            chain_keys[ha.ci].c_str(), chain_keys[hb.ci].c_str());
                    if (foldPair(hb.ci, ha.ci, ratios[j])) {
                        ++folded;
                        if (!noted_diff[j] && c.diff > 0) {
                            std::string da, db;
                            size_t n = std::max(ha.elems.size(), hb.elems.size());
                            for (size_t k = 0; k < n; ++k) {
                                const std::string ea = k < ha.elems.size()
                                    ? ha.elems[k].first + "[" + ha.elems[k].second + "]" : "-";
                                const std::string eb = k < hb.elems.size()
                                    ? hb.elems[k].first + "[" + hb.elems[k].second + "]" : "-";
                                if (ea == eb) continue;
                                if (!da.empty()) { da += ", "; db += ", "; }
                                da += ea; db += eb;
                            }
                            noted_diff[j] = true;
                            fprintf(stderr,
                                "[ctpwa] note: trans [%s, %s] = %g: 配对链在 %s vs %s 处不同 "
                                "（只共享耦合/步参数, 该层线形各自独立）\n",
                                names[0].c_str(), names[j].c_str(), ratios[j],
                                da.c_str(), db.c_str());
                        }
                    }
                }
                size_t leftA = 0, leftB = 0;
                for (bool u : usedA) if (!u) ++leftA;
                for (bool u : usedB) if (!u) ++leftB;
                if (leftA || leftB)
                    fprintf(stderr,
                        "[ctpwa] warn: trans [%s, %s] 链组 \"%s\" 两侧组合数不同 (剩 %zu vs %zu); "
                        "未配对的链保持独立\n",
                        names[0].c_str(), names[j].c_str(), gk.c_str(), leftA, leftB);
            }
        }
        if (paired == 0)
            fprintf(stderr,
                "[ctpwa] warn: trans 约束 [%s, ...] 匹配到链但未配成任何一对 "
                "(检查链名/中间态名与共振组合是否对应); 忽略\n",
                names[0].c_str());
    }

    // --- 跨约束链式折叠: [A,B] + [B,C] → C 折到 A（比值相乘）---
    // 成环（约束自相矛盾）时断环: 环内最小下标保留为 origin, 环内其它条目删除,
    // 保证解析后的 origin 一定不是被折叠链（否则 amp_chain 会拿到无效下标）。
    auto resolveFolds = [](const std::map<int, std::pair<int, double>>& in)
        -> std::map<int, std::pair<int, double>> {
        std::map<int, std::pair<int, double>> out;
        std::set<int> drop;
        for (const auto& kv : in) {
            int cur = kv.second.first;
            double ratio = kv.second.second;
            std::set<int> seen{kv.first};
            while (in.count(cur)) {
                if (!seen.insert(cur).second) {
                    fprintf(stderr, "[ctpwa] warn: trans 折叠成环 (含下标 %d), 已断环\n", cur);
                    for (int m : seen) if (in.count(m)) drop.insert(m);
                    cur = *seen.begin();
                    break;
                }
                ratio *= in.at(cur).second;
                cur = in.at(cur).first;
            }
            out[kv.first] = {cur, ratio};
        }
        for (int m : drop) out.erase(m);
        return out;
    };
    chain_fold_map = resolveFolds(chain_fold_map);
    step_fold_map = resolveFolds(step_fold_map);

    // --- 分配步级参数 (跳过被折叠的步) ---
    int sp_idx = 0;
    for (size_t si = 0; si < steps_.size(); ++si) {
        auto& s = r.steps[si];
        s.first_free_idx = -1;
        if (step_fold_map.find((int)si) != step_fold_map.end()) continue;
        if (s.n_sl() > 1) { s.first_free_idx = sp_idx; sp_idx += s.n_sl() - 1; }
    }

    // --- 清理无振幅引用的步 ---
    {
        std::set<int> ref_steps;
        for (const auto& am : amp_map_) {
            for (const auto& [step_idx, sl_idx] : am.step_sl) {
                if (sl_idx > 0) {
                    int origin = step_idx;
                    auto sf = step_fold_map.find(step_idx);
                    if (sf != step_fold_map.end()) origin = sf->second.first;
                    ref_steps.insert(origin);
                }
            }
        }
        int new_sp = 0;
        for (size_t si = 0; si < r.steps.size(); ++si) {
            if (r.steps[si].first_free_idx < 0) continue;
            if (ref_steps.find((int)si) == ref_steps.end()) {
                r.steps[si].first_free_idx = -1;
            } else {
                int n = r.steps[si].n_sl() - 1;
                r.steps[si].first_free_idx = new_sp;
                new_sp += n;
            }
        }
        sp_idx = new_sp;
    }
    r.n_step_free = sp_idx;

    // --- 收集折叠后的链参数 ---
    std::vector<std::string> active_chains;
    for (size_t ci = 0; ci < chain_keys.size(); ++ci)
        if (chain_fold_map.find((int)ci) == chain_fold_map.end())
            active_chains.push_back(chain_keys[ci]);
    r.chain_names = active_chains;
    r.n_chain_free = static_cast<int>(active_chains.size());
    r.n_free = r.n_step_free + r.n_chain_free;

    // --- 振幅映射到折叠后的参数 ---
    r.amp_chain.assign(r.n_amps, -1);
    r.amp_step_params.resize(r.n_amps);
    r.amp_chain_ratio.assign(r.n_amps, 1.0);
    for (const auto& am : amp_map_) {
        int ai = am.amp_idx;
        if (ai < 0 || ai >= r.n_amps) continue;

        // chain param (with trans ratio)
        auto cit = std::find(chain_keys.begin(), chain_keys.end(), am.chain_key);
        int oc = static_cast<int>(cit - chain_keys.begin());
        double chain_ratio = 1.0;
        int fc = oc;
        if (chain_fold_map.count(oc)) {
            fc = chain_fold_map[oc].first;
            chain_ratio = chain_fold_map[oc].second;
        }
        auto acit = std::find(active_chains.begin(), active_chains.end(), chain_keys[fc]);
        r.amp_chain[ai] = static_cast<int>(acit - active_chains.begin());
        r.amp_chain_ratio[ai] = chain_ratio;

        // step params
        for (const auto& [step_idx, sl_idx] : am.step_sl) {
            if (step_idx < 0 || step_idx >= (int)r.steps.size() || sl_idx <= 0) continue;
            int origin = step_idx;
            auto sf = step_fold_map.find(step_idx);
            if (sf != step_fold_map.end()) origin = sf->second.first;
            if (r.steps[origin].first_free_idx >= 0)
                r.amp_step_params[ai].push_back(r.steps[origin].first_free_idx + sl_idx - 1);
        }
    }

    return r;
}

// ============================================================
// gauss_constr（高斯罚项约束）— 仅作用于 theta 参数
// ============================================================

void Parameters::setGaussConstr(const std::map<std::string, int>& name_to_idx,
                                const std::map<std::string, double>& sigma,
                                const std::map<std::string, double>& mu)
{
    theta_name_to_idx_ = name_to_idx;
    gauss_constr_sigma_ = sigma;
    gauss_constr_mu_ = mu;
}

// 辅助：把约束表展开为 (indices, mu, sigma) 设备张量
static void buildGaussConstrTensors(
    const std::map<std::string, int>& name_to_idx,
    const std::map<std::string, double>& sigma,
    const std::map<std::string, double>& mu,
    const torch::Device& dev,
    torch::Tensor& idx_t, torch::Tensor& mu_t, torch::Tensor& sig_t)
{
    std::vector<int64_t> idxs;
    std::vector<double> mus, sigmas;
    for (const auto& [name, idx] : name_to_idx) {
        idxs.push_back(idx);
        mus.push_back(mu.at(name));
        sigmas.push_back(sigma.at(name));
    }
    idx_t = torch::tensor(idxs,
        torch::TensorOptions().dtype(torch::kLong).device(dev));
    mu_t = torch::tensor(mus,
        torch::TensorOptions().dtype(torch::kFloat64).device(dev));
    sig_t = torch::tensor(sigmas,
        torch::TensorOptions().dtype(torch::kFloat64).device(dev));
}

double Parameters::gaussPenalty(const torch::Tensor& theta) const
{
    if (theta_name_to_idx_.empty() || theta.numel() == 0) return 0.0;
    torch::Tensor idx_t, mu_t, sig_t;
    buildGaussConstrTensors(theta_name_to_idx_, gauss_constr_sigma_,
                            gauss_constr_mu_, theta.device(), idx_t, mu_t, sig_t);
    auto diff = theta.index_select(0, idx_t) - mu_t;
    return (diff * diff / (2.0 * sig_t * sig_t)).sum().item<double>();
}

void Parameters::gaussPenaltyGrad(const torch::Tensor& theta,
                                  torch::Tensor& grad_theta) const
{
    if (theta_name_to_idx_.empty() || theta.numel() == 0) return;
    torch::Tensor idx_t, mu_t, sig_t;
    buildGaussConstrTensors(theta_name_to_idx_, gauss_constr_sigma_,
                            gauss_constr_mu_, theta.device(), idx_t, mu_t, sig_t);
    auto diff = theta.index_select(0, idx_t) - mu_t;
    grad_theta.index_add_(0, idx_t, diff / (sig_t * sig_t));
}

void Parameters::addGaussHessianDiag(torch::Tensor& hess, int theta_offset) const
{
    if (theta_name_to_idx_.empty()) return;
    auto dev = hess.device();
    std::vector<int64_t> idxs;
    std::vector<double> vals;
    for (const auto& [name, idx] : theta_name_to_idx_) {
        double sig = gauss_constr_sigma_.at(name);
        idxs.push_back(theta_offset + idx);
        vals.push_back(1.0 / (sig * sig));
    }
    auto idx_t = torch::tensor(idxs,
        torch::TensorOptions().dtype(torch::kLong).device(dev));
    auto val_t = torch::tensor(vals,
        torch::TensorOptions().dtype(torch::kFloat64).device(dev));
    hess.diagonal(0).index_add_(0, idx_t, val_t);
}

void CouplingMatrixResult::print(std::ostream& os) const {
    os << "\n=== Coupling params ===" << std::endl;
    os << "  total: " << n_free << " free (" << n_chain_free
       << " chain + " << n_step_free << " step)" << std::endl;

    os << "\n  [Chain-level params]" << std::endl;
    for (size_t ci = 0; ci < chain_names.size(); ++ci) {
        os << "    chain[" << ci << "] " << chain_names[ci];
        os << "  → amps: [";
        bool first = true;
        for (int ai = 0; ai < n_amps; ++ai)
            if (amp_chain[ai] == (int)ci) {
                if (!first) os << ","; first = false; os << ai;
            }
        os << "]" << std::endl;
    }

    os << "\n  [Step-level SL params]" << std::endl;
    for (size_t si = 0; si < steps.size(); ++si) {
        const auto& s = steps[si];
        if (s.first_free_idx < 0 && s.n_sl() <= 1) continue;
        os << "    step[" << si << "] " << s.label;
        if (s.n_sl() <= 1) os << "  [all fixed 1.0]";
        os << std::endl;
        for (int sli = 0; sli < s.n_sl(); ++sli) {
            os << "      S=" << s.sl_list[sli].S << ",L=" << s.sl_list[sli].L;
            if (sli == 0 || s.first_free_idx < 0) os << "  → 1.0 (fixed)";
            else os << "  → step_param[" << (s.first_free_idx + sli - 1) << "]";
            os << std::endl;
        }
    }

    os << "\n  [Per-amplitude] coupling = chain × Π step_params" << std::endl;
    for (int i = 0; i < n_amps && i < 10; ++i) {
        os << "    amp[" << i << "] = chain[" << amp_chain[i] << "]";
        if (amp_step_params[i].empty()) os << " × 1.0";
        else for (int p : amp_step_params[i]) os << " × step[" << p << "]";
        os << std::endl;
    }
    if (n_amps > 10) os << "    ... (" << (n_amps - 10) << " more)" << std::endl;
}
