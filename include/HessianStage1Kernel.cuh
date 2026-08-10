#ifndef HESSIAN_STAGE1_KERNEL_CUH
#define HESSIAN_STAGE1_KERNEL_CUH

// Template __global__ kernel defined here for cross-TU instantiation.
// Included by ComputeHessian.cu (definition TU) and AmpGen.cu (call TU).
// Requires -rdc=true for device linking.

template<int Npr, int Nres>
__global__ void hessianStage1Kernel(
    const thrust::complex<double>* d_slamp_tab,  // [nSigma × nSL×nPol×nEv_total]
    const ctComplex* d_v,
    const DeviceMomenta* d_momenta,
    const DecayNode* d_decayNodes, int decayChain_size,
    const SL* d_slComb,
    const DeviceResonance* d_resonances,
    const double* d_all_params,
    const double* d_all_channels,       // flat 辅助段（Flatte channels + Hist 表）
    const int* d_global_idx,
    double* d_hess, int hess_ld,
    int nEvents, int nSL, int nPolar, double bf_d, double default_weight,
    const double* d_event_weights,
    // Temp output for cross-block stage 2
    const double* d_S_re_full,   // [nEv * nPolar] pre-computed full S
    const double* d_S_im_full,   // [nEv * nPolar]
    // Temp output for cross-block stage 2
    double* d_g_out,       // [nEv * NT]
    double* d_dS_re_out,   // [nEv * NT * nPolar]
    double* d_dS_im_out,   // [nEv * NT * nPolar]
    double* d_dF_re_out = nullptr,  // [nSigma × nEv * nSL * Npr] ∂F/∂θ for mixed Hessian
    double* d_dF_im_out = nullptr,
    // Phsp accumulators
    double* d_phsp_I = nullptr,
    double* d_phsp_grad = nullptr,
    double* d_phsp_hessA = nullptr,
    int evt_offset = 0,
    // 全同粒子置换拓扑（σ=0 恒等 +1）
    int nSigma = 1,
    const DeviceMomenta* d_mom_tab = nullptr,
    const double* d_sign_tab = nullptr)
{
    static_assert(Npr >= 1 && Npr <= 3, "Npr must be 1-3");
    static_assert(Nres >= 1 && Nres <= 4, "Nres must be 1-4");

    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    int evt_abs = evt + evt_offset;
    int nTotal = d_momenta->n_events * nPolar;
    double weight = d_event_weights ? d_event_weights[evt] : default_weight;

    constexpr int NT = Npr * Nres;
    int sl_per_res = nSL / Nres;

    // ===== AD variables per resonance =====
    using AD = Var<double, Npr, true>;
    AD m0_ad[Nres], g_ad[Nres];
    int ftg[Nres][Npr];
    for (int r = 0; r < Nres; ++r) {
        int po = d_resonances[r].param_offset;
        m0_ad[r] = AD((d_resonances[r].param_count > 0) ? d_all_params[po] : 1.0);
        m0_ad[r].grad[0] = 1.0;
        g_ad[r] = AD((d_resonances[r].param_count > 1) ? d_all_params[po + 1] : 1.0);
        g_ad[r].grad[1] = 1.0;
        for (int j = 0; j < Npr; ++j)
            ftg[r][j] = d_global_idx[r * Npr + j];
    }

    // ===== Read full S and I from pre-pass kernel =====
    double Sr[32] = { 0 }, Si[32] = { 0 };
    for (int p = 0; p < nPolar; ++p) {
        Sr[p] = d_S_re_full[evt * nPolar + p];
        Si[p] = d_S_im_full[evt * nPolar + p];
    }
    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) I_val += Sr[p] * Sr[p] + Si[p] * Si[p];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;

    // ===== Pass 2: dS[j][p] and d²S[j][k][p] =====
    double dS_re[NT][32] = { {0} }, dS_im[NT][32] = { {0} };
    double d2S_re[NT][NT][32] = { {{0}} }, d2S_im[NT][NT][32] = { {{0}} };

    size_t dF_row = (size_t)nEvents * nSL * Npr;         // per-σ dF 行距
    size_t slamp_row = (size_t)nSL * nTotal;             // per-σ slamp 行距（nTotal = nEv_total×nPol）

    for (int sl_idx = 0; sl_idx < nSL; ++sl_idx) {
        int res = sl_idx / sl_per_res;
        if (res >= Nres) continue;
        ctComplex vv = d_v[sl_idx];

        // 全同粒子：S、∂S/∂θ、∂²S/∂θ² = Σ_σ sgn(σ)·v·slamp(σ)·{R(σ), ∂R(σ)/∂θ, ∂²R(σ)/∂θ²}
        for (int s = 0; s < nSigma; ++s) {
            const DeviceMomenta* dm = (s == 0 || !d_mom_tab) ? d_momenta : &d_mom_tab[s];
            double sg = (s == 0) ? 1.0 : d_sign_tab[s];
            const thrust::complex<double>* slam = d_slamp_tab + (size_t)s * slamp_row;
            const double* dF_re_s = d_dF_re_out ? d_dF_re_out + (size_t)s * dF_row : nullptr;
            const double* dF_im_s = d_dF_im_out ? d_dF_im_out + (size_t)s * dF_row : nullptr;

            using CV = ComplexVar<double, Npr, true>;
            CV R_ad(1.0, 0.0);
            {
                const DeviceResonance& target = d_resonances[res];
                AD* m0p = &m0_ad[res];
                AD* gp = &g_ad[res];

                for (int ni = 0; ni < decayChain_size; ++ni) {
                    const DecayNode& node = d_decayNodes[ni];
                    const SL& sl = d_slComb[sl_idx * decayChain_size + ni];
                    int L = sl.L;
                    LorentzVector pM = dm->getMomentum(evt_abs, node.mother_idx);
                    LorentzVector pD1 = dm->getMomentum(evt_abs, node.daug1_idx);
                    LorentzVector pD2 = dm->getMomentum(evt_abs, node.daug2_idx);
                    double mm = pM.M();
                    double md1 = pD1.M();
                    double md2 = pD2.M();
                    double qq = breakup_momentum(mm, md1, md2);

                    AD m0_q0, md1_q0, md2_q0;
                    if (node.mother_idx == target.particle_idx && node.mass[0] <= 0)
                        m0_q0 = *m0p;
                    else if (node.mass[0] > 0) m0_q0 = AD(node.mass[0]);
                    else m0_q0 = AD(1.0);
                    if (node.mass[1] <= 0 && node.daug1_idx == target.particle_idx)
                        md1_q0 = *m0p;
                    else md1_q0 = AD(node.mass[1] > 0 ? node.mass[1] : md1);
                    if (node.mass[2] <= 0 && node.daug2_idx == target.particle_idx)
                        md2_q0 = *m0p;
                    else md2_q0 = AD(node.mass[2] > 0 ? node.mass[2] : md2);

                    AD q0_ad = computeQ0AD(m0_q0, md1_q0, md2_q0);
                    AD q_ad(qq);
                    bool is_res = (node.mother_idx == target.particle_idx && node.mass[0] <= 0);

                    CV nf;
                    if (is_res) {
                        AD params_arr[4] = {*m0p, *gp, AD(0.0), AD(0.0)};
                        if (target.type == ResModelType::Custom && target.param_count > 2) {
                            params_arr[2] = AD(d_all_params[target.param_offset + 2]);
                            if (d_global_idx[2] >= 0) params_arr[2].grad[d_global_idx[2]] = 1.0;
                        }
                        nf = computeNodeFactor<AD>(L, AD(mm), q_ad, q0_ad,
                                                  params_arr, target.param_count, target.type,
                                                  d_all_channels, target.n_channels,
                                                  d_all_channels, target.aux_offset, bf_d);
                    } else {
                        auto bf = Bf<AD>(L, q_ad, q0_ad, bf_d);
                        nf.real = bf; nf.imag = AD(0.0);
                    }
                    CV new_R;
                    new_R.real = R_ad.real * nf.real - R_ad.imag * nf.imag;
                    new_R.imag = R_ad.real * nf.imag + R_ad.imag * nf.real;
                    R_ad = new_R;
                }
            }

            // Output ∂F(σ)/∂θ for mixed Hessian（per-σ 行）
            if (d_dF_re_out) {
                for (int j = 0; j < Npr; ++j) {
                    int fidx = (int)((size_t)s * dF_row) + evt * nSL * Npr + sl_idx * Npr + j;
                    d_dF_re_out[fidx] = R_ad.real.grad[j];
                    d_dF_im_out[fidx] = R_ad.imag.grad[j];
                }
            }

            int j0 = res * Npr;
            for (int p = 0; p < nPolar; ++p) {
                auto sl_amp = slam[sl_idx * nTotal + evt_abs * nPolar + p];
                double sl_re = sl_amp.real(), sl_im = sl_amp.imag();
                double t_re = (double)vv.x * sl_re - (double)vv.y * sl_im;
                double t_im = (double)vv.x * sl_im + (double)vv.y * sl_re;

                for (int j = 0; j < Npr; ++j) {
                    double dFr = R_ad.real.grad[j], dFi = R_ad.imag.grad[j];
                    dS_re[j0 + j][p] += sg * (dFr * t_re - dFi * t_im);
                    dS_im[j0 + j][p] += sg * (dFr * t_im + dFi * t_re);
                }
                for (int j = 0; j < Npr; ++j) {
                    for (int k = j; k < Npr; ++k) {
                        double d2Fr = R_ad.real.hess[j][k], d2Fi = R_ad.imag.hess[j][k];
                        d2S_re[j0 + j][j0 + k][p] += sg * (d2Fr * t_re - d2Fi * t_im);
                        d2S_im[j0 + j][j0 + k][p] += sg * (d2Fr * t_im + d2Fi * t_re);
                    }
                }
            }
        }
    }

    // ===== Gradient g[j] =====
    double g[NT] = { 0 };
    for (int p = 0; p < nPolar; ++p) {
        double cwr = Sr[p] * inv_I, cwi = -Si[p] * inv_I;
        for (int j = 0; j < NT; ++j)
            g[j] += cwr * dS_re[j][p] - cwi * dS_im[j][p];
    }
    for (int j = 0; j < NT; ++j) g[j] *= -2.0;

    // ===== Output g, dS to temp buffers =====
    for (int j = 0; j < NT; ++j)
        d_g_out[evt * NT + j] = g[j];
    for (int j = 0; j < NT; ++j)
        for (int p = 0; p < nPolar; ++p) {
            int idx = evt * NT * nPolar + j * nPolar + p;
            d_dS_re_out[idx] = dS_re[j][p];
            d_dS_im_out[idx] = dS_im[j][p];
        }

    // ===== Same-resonance Hessian → d_hess =====
    double H_loc[NT][NT] = { {0} };
    for (int j = 0; j < NT; ++j) {
        for (int k = j; k < NT; ++k) {
            double hjk = g[j] * g[k];  // Term A

            // Term B
            double termB = 0.0;
            for (int p = 0; p < nPolar; ++p)
                termB += dS_re[k][p] * dS_re[j][p] + dS_im[k][p] * dS_im[j][p];
            termB *= -2.0 * inv_I;
            hjk += termB;

            // Term C (same-resonance only)
            int rj = j / Npr, rk = k / Npr;
            if (rj == rk) {
                double termC = 0.0;
                for (int p = 0; p < nPolar; ++p) {
                    double cwr = Sr[p] * inv_I, cwi = -Si[p] * inv_I;
                    termC += cwr * d2S_re[j][k][p] - cwi * d2S_im[j][k][p];
                }
                termC *= -2.0;
                hjk += termC;
            }

            H_loc[j][k] = hjk;
            if (j != k) H_loc[k][j] = hjk;

            int gj = ftg[rj][j % Npr];
            int gk = ftg[rk][k % Npr];
            if (gj >= 0 && gk >= 0) {
                double contrib = weight * hjk;
                atomicAdd(&d_hess[gk * hess_ld + gj], contrib);
                if (j != k) atomicAdd(&d_hess[gj * hess_ld + gk], contrib);
            }
            // Phsp: I * (g·g^T - H) for same-resonance (use hess_ld, global stride)
            if (default_weight == 0.0 && d_phsp_hessA != nullptr && gj >= 0 && gk >= 0) {
                atomicAdd(&d_phsp_hessA[gk * hess_ld + gj], I_val * (g[j] * g[k] - hjk));
                if (j != k) atomicAdd(&d_phsp_hessA[gj * hess_ld + gk], I_val * (g[j] * g[k] - hjk));
            }
        }
    }

    // Phsp: accumulate I and I*g
    if (default_weight == 0.0 && d_phsp_I != nullptr)
        atomicAdd(d_phsp_I, I_val);
    if (default_weight == 0.0 && d_phsp_grad != nullptr) {
        for (int j = 0; j < NT; ++j) {
            int gj = ftg[j / Npr][j % Npr];
            if (gj >= 0) atomicAdd(&d_phsp_grad[gj], I_val * g[j]);
        }
    }
}

// ============================================================
// Custom 模型 Hessian（标量路径，参数数 P 运行时无上限）
// 与 hessianStage1Kernel 相同的输出契约（g/dS/d2S/H/dF/phsp），
// 但节点因子用标量 evalCustomAll（不经 Var<double,N> 模板）。
// Nres=1（每 block 一个 Custom 共振态），NT = P。

#endif // HESSIAN_STAGE1_KERNEL_CUH
