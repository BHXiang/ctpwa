#include <Parameters.cuh>

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

        // ---- second-order gradient term: Σ_a g_v[a] × ∂²v_a/(∂p_j ∂p_k) ----
        // Only non-zero for mixed chain×step derivatives
        if (d_g_v && jCh != kCh) {
            double gt = 0.0;
            int chain_j = jCh ? jc : kc;     // which is the chain
            int step_j  = jCh ? kc : jc;     // which is the step
            bool row_is_chain = jCh;
            for (int a = 0; a < na; ++a) {
                if (d_amp_chain[a] != chain_j) continue;
                bool uses_step = false;
                for (int ss = d_step_offsets[a]; ss < d_step_offsets[a + 1]; ++ss)
                    if (d_step_data[ss] == step_j) { uses_step = true; break; }
                if (!uses_step) continue;
                double g_re = d_g_v[a], g_im = d_g_v[na + a];
                // ∂²v/∂p_j∂p_k in Re/Im grouped format:
                // ∂²Re/∂Re(c)∂Re(s)=1, ∂²Im/∂Re(c)∂Im(s)=1,
                // ∂²Im/∂Im(c)∂Re(s)=1, ∂²Re/∂Im(c)∂Im(s)=-1
                // All other combos = 0
                if (row_is_chain) {
                    if (cr == 0 && cc == 0) gt += g_re;        // Re_c×Re_s → +g_re
                    else if (cr == 0 && cc == 1) gt += g_im;   // Re_c×Im_s → +g_im
                    else if (cr == 1 && cc == 0) gt += g_im;   // Im_c×Re_s → +g_im
                    else if (cr == 1 && cc == 1) gt -= g_re;   // Im_c×Im_s → -g_re
                } else {
                    // row is step, col is chain — symmetric
                    if (cr == 0 && cc == 0) gt += g_re;
                    else if (cr == 1 && cc == 0) gt += g_im;
                    else if (cr == 0 && cc == 1) gt += g_im;
                    else if (cr == 1 && cc == 1) gt -= g_re;
                }
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

CouplingMatrixResult CouplingMatrixBuilder::buildWithTrans(
    const std::vector<std::vector<std::string>>& trans_names,
    const std::vector<std::complex<double>>& trans_values) const
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

    // --- 应用 trans 约束: 逐对折叠 chain + 匹配步 ---
    std::map<int, std::pair<int, double>> step_fold_map;
    std::map<int, std::pair<int, double>> chain_fold_map;  // cB → {cA, ratio}

    for (size_t ti = 0; ti < trans_names.size(); ++ti) {
        if (trans_names[ti].size() < 2) continue;
        double ratio = 1.0;
        if (ti < trans_values.size()) ratio = std::real(trans_values[ti]);
        const std::string& nameA = trans_names[ti][0];
        const std::string& nameB = trans_names[ti][1];

        // Helper: extract resonance name (after last '[')
        auto resName = [](const std::string& key) -> std::string {
            auto pos = key.rfind('[');
            return (pos != std::string::npos) ? key.substr(pos) : key;
        };

        // Collect all chains matching A and B, indexed by resonance name
        std::map<std::string, int> chainsA, chainsB;
        for (size_t ci = 0; ci < chain_keys.size(); ++ci) {
            if (chain_keys[ci].find(nameA) != std::string::npos)
                chainsA[resName(chain_keys[ci])] = (int)ci;
            if (chain_keys[ci].find(nameB) != std::string::npos)
                chainsB[resName(chain_keys[ci])] = (int)ci;
        }

        // Pairwise fold: same resonance → B folds into A
        for (const auto& [res, cB] : chainsB) {
            auto itA = chainsA.find(res);
            if (itA == chainsA.end()) continue;
            int cA = itA->second;
            chain_fold_map[cB] = {cA, ratio};

            // Fold steps by position for this pair
            for (auto& [chainB_name, cB_steps] : chain_step_order_) {
                if (chain_keys[cB].find(chainB_name) != 0) continue;
                for (auto& [chainA_name, cA_steps] : chain_step_order_) {
                    if (chain_keys[cA].find(chainA_name) != 0) continue;
                    if (chainA_name == chainB_name) continue;
                    for (size_t pi = 0; pi < cB_steps.size() && pi < cA_steps.size(); ++pi) {
                        int sB = cB_steps[pi], sA = cA_steps[pi];
                        if (step_fold_map.find(sB) == step_fold_map.end())
                            step_fold_map[sB] = {sA, ratio};
                    }
                    break;
                }
            }
        }
    }

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
