# Resonance Parameter Hessian Formula

## 1. NLL Definition

For a single decay chain with coupling parameters `v` and resonance parameters `θ`:

```
NLL = -∑_data log(I_e) + n_data · log(phsp_factor)

where:
  I_e = ∑_p |S_{e,p}|²                          (intensity summed over polarizations)
  S_{e,p} = ∑_a A_{e,p}^a · v_a                (amplitude sum)
  A_{e,p}^a = SL_{e,p}^a · R_e(θ) · bf         (full amplitude)
  T_r(e,p) = ∑_{a∈block} SL_{e,p}^a · v_a      (effective coupling, no R, no bf)
  phsp_factor = (1/N_phsp) · ∑_phsp |S|²
```

Key relation: `S_{e,p} = R_e(θ) · bf · T_r(e,p)` for single-chain (all channels share same R).

## 2. Per-event Gradient ∂NLL/∂θ

```
g_θ[j] = -(1/I) · ∂I/∂θ_j

∂I/∂θ_j = 2 · bf · ∑_p Re(conj(S_p) · T_p · D^j)

where D^j = ∂R/∂θ_j  (complex, from AutoDiff Var<double,N,true>)

Therefore:
  g_θ[j] = -(2·bf/I) · ∑_p Re(conj(S_p) · T_p · D^j)
         = -(2·bf/I) · Re( D^j · ∑_p conj(S_p)·T_p )
         = -2·bf · Re( D^j · ∑_p conj(w_p)·T_p )      [since conj(S)=I·conj(w)]
```

The last form uses `w = S/I`, which is what ctpwa's `computeFactorNLL` outputs.

## 3. Per-event Hessian ∂²NLL/∂θ∂θ

```
H = g·g^T - (1/I) · ∂²I/∂θ∂θ

where ∂²I/∂θ_j∂θ_k = 
    2·bf² · ∑_p |T_p|² · Re(conj(D^k) · D^j)           [|T|² term]
  + 2·bf   · ∑_p Re(conj(S_p) · T_p · D²_jk)           [D² term]
```

**Note**: `D²_jk = ∂²R/∂θ_j∂θ_k` from AutoDiff `Var<double,N,true>::hess[j][k]`.

## 4. CUDA-compatible per-event formula

All quantities aggregated in ONE pass over polarizations:

```cpp
// Per-event aggregates:
double I_inv = 0.0;           // 1/I = Σ|w_p|²
double sum_T2 = 0.0;          // Σ|T_p|²
double sum_cwT_re = 0.0;      // Σ Re(conj(w_p)·T_p)
double sum_cwT_im = 0.0;      // Σ Im(conj(w_p)·T_p)

for (int p = 0; p < nPolar; ++p) {
    w = d_w[evt * nPolar + p];
    T = d_T[evt * nPolar + p];
    I_inv += |w|²;
    sum_T2 += |T|²;
    sum_cwT += conj(w) * T;   // complex accumulation
}

// Gradient:
g[j] = -2 · bf · (D_re[j] · sum_cwT_re - D_im[j] · sum_cwT_im)

// Hessian:
H[j][k] = g[j] · g[k]                                    // outer product
        - 2·I_inv · bf² · sum_T2 · Re(conj(D_k)·D_j)     // |T|² correction
        - 2 · bf · Re(sum_cwT · D²_jk)                    // D² correction
```

Where `Re(conj(D_k)·D_j) = D_re[k]·D_re[j] + D_im[k]·D_im[j]`.

## 5. Key Simplification

The `I` cancels in the D² correction:
```
-(2/I)·bf·∑ Re(conj(S)·T·D²) = -2·bf·∑ Re(conj(w)·T·D²)
```
So the D² correction uses `sum_cwT = Σ conj(w)·T` directly, WITHOUT `I_inv` factor.

## 6. Verified Against PyTorch

Python verification (matching exact ctpwa formulas, q0 fixed):

```
PyTorch autograd θ-θ:
  [[-21.052335, -5.967517],
   [-5.967517,  2.008298]]

Manual formula θ-θ:
  [[-21.052335, -5.967517],
   [-5.967517,  2.008298]]
  → MATCH (diff < 1e-10)
```

AutoDiff D² values also verified:
```
PyTorch BWR Hessian:
  d²Re/dm0²=18.4789  d²Re/dm0dg0=13.0991  d²Re/dg0²=-0.5906
  d²Im/dm0²=41.1079  d²Im/dm0dg0=-2.8514  d²Im/dg0²=-5.2698

ctpwa testBWRHessian (Var<double,2,true>):
  d²Re/dm0²=18.4789  d²Re/dm0dg0=13.0991  d²Re/dg0²=-0.5906
  d²Im/dm0²=41.1079  d²Im/dm0dg0=-2.8514  d²Im/dg0²=-5.2698
  → MATCH (diff < 1e-15)
```

## 7. Found Bugs

### Bug 1: Missing event offset in resonanceHessianBlockKernel
- **Symptom**: D values ~4x too small, wrong kinematics (m=1.80 vs expected m=1.11)
- **Root cause**: Kernel accesses `d_momenta->getMomentum(evt, ...)` where `evt` is data-relative index (0), but momenta array contains ALL events. Data event is at absolute index 3.
- **Fix**: Add `evt_offset` parameter. Use `evt_abs = evt + evt_offset` for momenta access.

### Bug 2: Per-polarization |w|² weighting (in original code)
- **Symptom**: Hessian values ~10x too small
- **Root cause**: Correction term uses `|w_p|² = |S_p|²/I²` per-polarization instead of per-event `1/I = Σ|w|²`. Also, outer product does `Σ_p X_p·Y_p` instead of `(Σ_p X_p)·(Σ_q Y_q)`.
- **Fix**: Restructure Step 5 to per-event aggregates (see Section 4).

## 8. Missing Feature: PHSP Contribution

The phsp contributes to θ-θ Hessian through:
```
n_data · ∂²log(phsp_factor)/∂θ²

where phsp_factor = Σ_phsp |S|² / n_phsp

∂²log(phsp_factor)/∂θ² = (Σ∂²I/∂θ²)/(ΣI) - (Σ∂I/∂θ)(Σ∂I/∂θ)^T/(ΣI)²
```

Currently `computeResonanceHessian` only handles data+bkg events.

## 9. Mixed Block ∂²NLL/∂v∂θ

For single-chain models where all channels share the same resonance:
```
g_v[a] = -2 · Re(conj(T)·SL_a / Σ|T|²)
```
This is **independent of θ** (R and bf cancel). Therefore `∂²NLL/∂v∂θ = 0` identically.

For multi-chain models, the mixed block is non-zero and needs `computeMixedHessian`.
