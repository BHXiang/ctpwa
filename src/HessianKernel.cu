/**
 * Two-stage Hessian kernels — verified formula from lab_hessian/test_hessian.cu
 *
 * Stage 1 (hessianStage1Kernel):
 *   Per-block: computes g_e, dS_e, I_e, same-resonance Hessian (Terms A+B+C).
 *   Outputs g_e and dS_e to temp buffers for cross-block stage.
 *
 * Stage 2 (hessianCrossBlockKernel):
 *   Cross-block: reads two blocks' temp buffers, computes cross-resonance
 *   Terms A (g·g^T) and B (-2/I·Re(conj(dS)·dS)), atomicAdd to d_hess.
 *
 * Formula (per event e):
 *   S_{e,p} = Σ_sl v_sl · slamps_{sl,e,p} · F_sl(θ)
 *   I_e = Σ_p |S_{e,p}|²
 *   g_ej = -2/I_e · Σ_p Re(conj(S_{e,p}) · dS_{e,j,p})
 *   H_ejk = g_j·g_k - 2/I_e·Re(conj(dS_k)·dS_j) - 2/I_e·Re(conj(S)·d²S_jk)
 *         = Term A        + Term B                    + Term C (same-res only)
 */
#include "AmpGen.cuh"

 // ============================================================
 // Pre-pass: compute full S[p]=Σ_a v[a]·amp[a,e,p] from raw amplitudes
 // amp: [nEv * nPol * n_amp] layout, v: [n_amp] interleaved
 // Outputs S_re[nEv*nPol], S_im[nEv*nPol], I[nEv]
 // ============================================================
__global__ void computeSfromAmpsKernel(
    double* d_S_re, double* d_S_im, double* d_I,
    const cuComplex* d_amp,   // [nEv * nPol * n_amp]
    const cuComplex* d_v,     // [n_amp]
    int nEvents, int nPolar, int n_amp)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;

    double Sr[32] = { 0 }, Si[32] = { 0 };
    for (int p = 0; p < nPolar; ++p) {
        double sre = 0.0, sim = 0.0;
        for (int a = 0; a < n_amp; ++a) {
            cuComplex amp_ap = d_amp[evt * nPolar * n_amp + p * n_amp + a];
            cuComplex v_a = d_v[a];
            sre += (double)v_a.x * (double)amp_ap.x - (double)v_a.y * (double)amp_ap.y;
            sim += (double)v_a.x * (double)amp_ap.y + (double)v_a.y * (double)amp_ap.x;
        }
        Sr[p] = sre;
        Si[p] = sim;
        d_S_re[evt * nPolar + p] = sre;
        d_S_im[evt * nPolar + p] = sim;
    }
    double I_val = 0.0;
    for (int p = 0; p < nPolar; ++p) I_val += Sr[p] * Sr[p] + Si[p] * Si[p];
    d_I[evt] = I_val;
}

// ============================================================
// Stage 1: per-block kernel — same-resonance Hessian + output g,dS
// Template: Npr = params per resonance, Nres = resonances in this block
// ============================================================
template<int Npr, int Nres>
__global__ void hessianStage1Kernel(
    const thrust::complex<double>* d_slamps,
    const cuComplex* d_v,
    const DeviceMomenta* d_momenta,
    const DecayNode* d_decayNodes, int decayChain_size,
    const SL* d_slComb,
    const DeviceResonance* d_resonances,
    const double* d_all_params,
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
    // Phsp accumulators
    double* d_phsp_I = nullptr,
    double* d_phsp_grad = nullptr,
    double* d_phsp_hessA = nullptr,
    int evt_offset = 0)
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
        m0_ad[r] = AD(d_all_params[po]);
        m0_ad[r].grad[0] = 1.0;
        g_ad[r] = AD(d_all_params[po + 1]);
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

    for (int sl_idx = 0; sl_idx < nSL; ++sl_idx) {
        int res = sl_idx / sl_per_res;
        if (res >= Nres) continue;
        cuComplex vv = d_v[sl_idx];

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
                LorentzVector pM = d_momenta->getMomentum(evt_abs, node.mother_idx);
                LorentzVector pD1 = d_momenta->getMomentum(evt_abs, node.daug1_idx);
                LorentzVector pD2 = d_momenta->getMomentum(evt_abs, node.daug2_idx);
                double mm = pM.M();
                // double mm = pM.M();
                double md1 = pD1.M();
                double md2 = pD2.M();
                double qq = breakup_momentum(mm, md1, md2);

                AD m0_q0_ad, md1_q0_ad, md2_q0_ad;
                if (node.mother_idx == target.particle_idx && node.mass[0] <= 0)
                    m0_q0_ad = *m0p;
                else if (node.mass[0] > 0) m0_q0_ad = AD(node.mass[0]);
                else m0_q0_ad = AD(1.0);
                if (node.mass[1] <= 0 && node.daug1_idx == target.particle_idx)
                    md1_q0_ad = *m0p;
                else md1_q0_ad = AD(node.mass[1] > 0 ? node.mass[1] : md1);
                if (node.mass[2] <= 0 && node.daug2_idx == target.particle_idx)
                    md2_q0_ad = *m0p;
                else md2_q0_ad = AD(node.mass[2] > 0 ? node.mass[2] : md2);

                AD s_md = md1_q0_ad + md2_q0_ad;
                AD d_md = md1_q0_ad - md2_q0_ad;
                AD m0sq = m0_q0_ad * m0_q0_ad;
                AD q0sq = (m0sq - s_md * s_md) * (m0sq - d_md * d_md) / (AD(4.0) * m0sq);
                q0sq.val = q0sq.val < 0.0 ? 0.0 : q0sq.val;
                AD q0_ad = sqrt(q0sq);
                AD q_ad(qq);

                CV node_factor(1.0, 0.0);
                if (node.mother_idx == target.particle_idx && node.mass[0] <= 0) {
                    AD m_ad(mm);
                    if (target.type == ResModelType::BWR) {
                        auto bw = BWR<AD>(m_ad, *m0p, *gp, L, q_ad, q0_ad, bf_d);
                        auto bf = Bf<AD>(L, q_ad, q0_ad, bf_d);
                        node_factor.real = bw.real * bf;
                        node_factor.imag = bw.imag * bf;
                    }
                    else { node_factor = BW<AD>(m_ad, *m0p, *gp); }
                }
                else {
                    auto bf = Bf<AD>(L, q_ad, q0_ad, bf_d);
                    node_factor.real = bf;
                    node_factor.imag = AD(0.0);
                }
                CV new_R;
                new_R.real = R_ad.real * node_factor.real - R_ad.imag * node_factor.imag;
                new_R.imag = R_ad.real * node_factor.imag + R_ad.imag * node_factor.real;
                R_ad = new_R;
            }
        }

        int j0 = res * Npr;
        for (int p = 0; p < nPolar; ++p) {
            auto sl_amp = d_slamps[sl_idx * nTotal + evt_abs * nPolar + p];
            double sl_re = sl_amp.real(), sl_im = sl_amp.imag();
            double t_re = (double)vv.x * sl_re - (double)vv.y * sl_im;
            double t_im = (double)vv.x * sl_im + (double)vv.y * sl_re;

            for (int j = 0; j < Npr; ++j) {
                double dFr = R_ad.real.grad[j], dFi = R_ad.imag.grad[j];
                dS_re[j0 + j][p] += dFr * t_re - dFi * t_im;
                dS_im[j0 + j][p] += dFr * t_im + dFi * t_re;
            }
            for (int j = 0; j < Npr; ++j) {
                for (int k = j; k < Npr; ++k) {
                    double d2Fr = R_ad.real.hess[j][k], d2Fi = R_ad.imag.hess[j][k];
                    d2S_re[j0 + j][j0 + k][p] += d2Fr * t_re - d2Fi * t_im;
                    d2S_im[j0 + j][j0 + k][p] += d2Fr * t_im + d2Fi * t_re;
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
// Stage 2: cross-block kernel — Term A + Term B between blocks
// Reconstructs I from d_I buffer; d_I[evt] holds I_e for each event.
// ============================================================
__global__ void hessianCrossBlockKernel(
    const double* d_g_A,        // [nEv * NTA]
    const double* d_dS_re_A,    // [nEv * NTA * nPolar]
    const double* d_dS_im_A,
    const double* d_g_B,
    const double* d_dS_re_B,
    const double* d_dS_im_B,
    const double* d_I,           // [nEv] intensities for inv_I reconstruction
    const int* d_global_idx_A,   // [NTA]
    const int* d_global_idx_B,   // [NTB]
    int NTA, int NTB, int nEvents, int nPolar,
    double* d_hess, int hess_ld,
    double default_weight,
    const double* d_event_weights,
    // Phsp cross-block: accumulates 2*Re(conj(dS_B)·dS_A)
    double* d_phsp_hessA = nullptr,
    int phsp_ld = 1)
{
    int evt = blockIdx.x * blockDim.x + threadIdx.x;
    if (evt >= nEvents) return;
    double weight = d_event_weights ? d_event_weights[evt] : default_weight;

    double I_val = d_I[evt];
    if (I_val < 1e-30) return;
    double inv_I = 1.0 / I_val;

    const double* gA = d_g_A + evt * NTA;
    const double* gB = d_g_B + evt * NTB;
    const double* dsReA = d_dS_re_A + evt * NTA * nPolar;
    const double* dsImA = d_dS_im_A + evt * NTA * nPolar;
    const double* dsReB = d_dS_re_B + evt * NTB * nPolar;
    const double* dsImB = d_dS_im_B + evt * NTB * nPolar;

    for (int ja = 0; ja < NTA; ++ja) {
        int gja = d_global_idx_A[ja];
        if (gja < 0) continue;
        for (int kb = 0; kb < NTB; ++kb) {
            int gkb = d_global_idx_B[kb];
            if (gkb < 0) continue;

            // Term A: g[j] * g[k]
            double hjk = gA[ja] * gB[kb];

            // Term B: -2/I * Σ_p Re(conj(dS_B)·dS_A)
            double termB = 0.0;
            for (int p = 0; p < nPolar; ++p) {
                int ia = ja * nPolar + p;
                int ib = kb * nPolar + p;
                termB += dsReB[ib] * dsReA[ia] + dsImB[ib] * dsImA[ia];
            }
            termB *= -2.0 * inv_I;
            hjk += termB;

            double contrib = weight * hjk;
            atomicAdd(&d_hess[gkb * hess_ld + gja], contrib);
            // Symmetric: also fill [gja * hess_ld + gkb] (valid even if A≠B since
            // the total Hessian must be symmetric; cross term pair A→B and B→A
            // both run, so this is safe.)
            atomicAdd(&d_hess[gja * hess_ld + gkb], contrib);

            // Phsp cross: I * (g·g^T - H) = 2 * Re(conj(dS_B)·dS_A)
            if (d_phsp_hessA) {
                double phsp_val = 2.0 * termB / (-2.0 * inv_I); // = -termB * I
                // Actually: I*(g*g^T - H) = I*(g*g - g*g + 2/I*Re) = 2*Re
                // = I * 2/I * Re = 2 * Re
                // The raw Re sum is termB_Re = Σ_p Re(conj(dS_B)·dS_A)
                // So phsp_val = 2 * termB_Re
                // And termB = -2/I * termB_Re → termB_Re = -termB * I / 2
                double termB_Re = -termB * I_val * 0.5;
                // Actually let me recompute: termB = -2/I * Σ Re, so Σ Re = -termB * I / 2
                // Phsp contribution: 2 * Σ Re = -termB * I
                // Simpler: compute Σ Re directly
                double sumRe = 0.0;
                for (int p = 0; p < nPolar; ++p) {
                    int ia = ja * nPolar + p;
                    int ib = kb * nPolar + p;
                    sumRe += dsReB[ib] * dsReA[ia] + dsImB[ib] * dsImA[ia];
                }
                phsp_val = 2.0 * sumRe;
                atomicAdd(&d_phsp_hessA[gkb * phsp_ld + gja], phsp_val);
                atomicAdd(&d_phsp_hessA[gja * phsp_ld + gkb], phsp_val);
            }
        }
    }
}
