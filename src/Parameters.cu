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

    auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(device);
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
    auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(device);
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
        torch::Tensor rr_t = torch::tensor(rr_vec, torch::kFloat).to(device);
        torch::Tensor ir_t = torch::tensor(ir_vec, torch::kFloat).to(device);

        torch::Tensor ext_grads = extended_grad.index_select(0, ext_idx_t);

        torch::Tensor ext_re = (ext_grads + torch::conj(ext_grads)) / 2.0f;
        torch::Tensor ext_im =
            (ext_grads - torch::conj(ext_grads)) / (2.0f * c10::complex<float>(0, 1));

        torch::Tensor contrib =
            (rr_t * ext_re + c10::complex<float>(0, 1) * ir_t * ext_im).sum();

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
        real_part.to(torch::kFloat), imag_part.to(torch::kFloat));

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
    // Format: [Re_all, Im_all] → Re at [idx], Im at [ncf + idx]
    double ratio = d_amp_chain_ratio[a];
    int c_idx = n_step_free + d_amp_chain[a];
    double re = d_in[c_idx] * ratio;
    double im = d_in[ncf + c_idx] * ratio;

    for (int k = d_step_offsets[a]; k < d_step_offsets[a + 1]; ++k) {
        int s = d_step_data[k];
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

void Parameters::applyCouplingMatrix(const double* d_params, cuComplex* d_v_out) const
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
    const cuComplex* d_grad_v, const cuComplex* d_v,
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
