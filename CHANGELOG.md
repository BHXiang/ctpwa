# Changelog

## [Unreleased]

## [v0.3.24] - 2026-11-24

### Fixed

- **干涉矩阵维度错了**：常驻 `B̄` 的收缩原来按"逐 SL 振幅"做，导致同一分波的多个
  L/S 组合各占一行。现在按 `nSLvectors_` **合并 LS**（与
  `ComputeResults.cu::computeModWithInterference` 的口径一致：分波 = 链 × 共振态组合，
  内部 LS 相干求和）：
  `ls_filter` 4×4 → **1×1**、`eff2` 2×2、`with_trans` 6×6。

### Removed

- **`getFitFractionMatrix` / `getFitFractionMatrixErrors` 公共 API 删除**。矩阵与
  误差统一由 `writeResult` 写进 `weight_best.root`（`interference_rec/gen`、
  `interference_*_err`、`integrals`）。C++ 侧改为内部函数
  `matrixForSample()` / `computeMatrixErrors()`。
- 测试同步迁移：`tests/conftest.py` 提供 `fit_fraction_matrix()` /
  `matrix_errors()`，通过"写一次 ROOT 再读回"复现原 API 的返回值，
  5 个测试文件共 30 余处调用全部改为读 ROOT。

### Known limitation

- `interference_*_err` 目前**只含耦合方向**的参数误差。θ 方向的有限差分在
  `M ~ 1e4`、宽度 4.5 MeV 时被舍入噪声主导（实测与参考差 300 倍），必须走
  逐事例 `d_dF` 的解析路线（表已在 `reComputeAmps` 里生成，待接进扫描）。

## [v0.3.23] - 2026-11-23

### Changed（性能）

- **重建侧干涉矩阵改用常驻 `B̄`**：`writeResult` 与
  `getFitFractionMatrix("phsp")` 不再读 phsp 文件、不再重算振幅，而是
  `M_ij = 2Re(v_i v_j* B̄_ij)`（`B̄ = Σ_e (w_e/W_total) A_e A_e^H`，NLL 的 phsp
  分母，本来就在显存里）。1M 事例下 `writeResult` 47.4 s → 24.3 s，剩下的
  24 s 全是 `phsp_truth` 那一趟（truth 没有常驻表，省不掉）。
  文件路径已删除（`getFitFractionMatrix("phsp")` 现在走常驻 Gram）。

- **误差矩阵改为解析传播（耦合方向）**：`getFitFractionMatrixErrors` 不再对
  每个自由方向跑两趟全样本扫描（`4·n_free` 趟），改成
  - 重建侧：`B̄` 收缩（0 趟）；
  - 产生侧：一趟 `computeFileBatchIntegrals`，通过已有的
    `ev_perturbed_p/m` 通道复用同一批振幅算出所有方向的导数；
  - `Cov = H[coupling, coupling]⁻¹`，自由块只取耦合。
  200k 事例（2 个自由耦合）实测 40.05 s → 1.89 s（**21×**）；趟数从
  `4·n_free` 降到 `1`，与耦合数无关。
  **θ 方向暂不传播**（等 `Hm = Σ w(∂_θA A^H + A ∂_θA^H)` 接进同一趟）。

### Fixed

- **θ 变了 `B̄` 不重建**：重建逻辑从 `getHessian` 抽成 `rebuildPhspGram()`，
  `reCalcAmp(θ)`、`writeResult(θ)` 与 `getHessian` 都调用它。此前只有
  `getHessian` 会刷新，`reCalcAmp` 之后份额/矩阵会拿到旧 θ 的 `B̄`。

### Changed（日志）

- 去掉 `[writeResult] ...`、`[phsp] events: ...` 这类带方括号的进度打印；
  计时行只在 `CTPWA_VERBOSE=1` 时输出，事件数只在真正读文件时打印。

## [v0.3.22] - 2026-11-22

### Deprecated / 待删（只是标记，行为不变）

为后续"程序减负"先在源码里标出来（`src/Analysis.cu` 函数前 + `src/main.cu` 的
pybind docstring 都有 `[可删·v0.3.22 标记]`）：

- `getFitFractions(v[, H])` — 中心值 = `diag(interference_gen)/Σdiag`，已经由
  `writeResult` 写进 `weight_best.root`（实测两条路径一致到 2e-15）；它的误差按
  "非相干分母"传播，与 ROOT 里相干口径的 `interference_gen_err` 不是同一个量
  （0.0476 vs 0.2797），而该口径本身不是文献定义。删除前先把 `example/fit.py`
  的调用改成读 ROOT 或走矩阵 API。
- `writeInterfResult` — 逐事例干涉树，且树名 `saved_weight` 与 `writeResult` 的
  重名；建议合并成 `writeResult(..., pairs=[[i,j],...])`。当前无调用方。
- `saveSLAmps` / `getSLAmpsTensor` / `getDataTensor` / `getBkgTensor` /
  `getBkgWeightsTensor` — 零消费的调试接口（后三个只差一个 offset，可合并成
  `getAmpTensor(sample=...)`；`getBkgWeightsTensor` 还返回 float32，与全库口径不一致）。
- 暂留：`getEfficiency` 的中心值可由 ROOT 直接算，但 **MC 统计误差只在那里**；
  把 MC 误差也写进 ROOT 之后即可与 `getFitFractions` 一起删。

### Fixed

- **Fitted resonance parameters (θ) now reach the truth / efficiency integrals.**
  `getFitFractions`, `getEfficiency`, `getFitFractionMatrix` and the raw matrices
  written by `writeResult` rebuild the amplitudes from the configured particle
  objects, whose m0/Γ0 are the *config* values — `reCalcAmp(θ)` only updated the
  resident amplitudes. With free resonance parameters every derived quantity was
  therefore evaluated at the initial θ (a 10% shift of φ(1020)'s mass moved the
  integrals by <1e-14), and the θ derivative was identically zero, so θ
  parameter errors could not propagate either. `calculateAmplitudes` now takes
  an explicit θ and applies it to the freshly built `Resonance` copies through
  `AmpCalc::paramBindings(block)` (free slots + `var_equal` ghosts +
  conjugate/same-name broadcast), and `analysis` tracks the current θ from
  `reCalcAmp`, `getNLL`, `getHessian` and `writeResult`. Validated against an
  independent reconstruction from the resident (device) amplitudes — agreement
  1e-14 on single-chain, two-chain+`trans` and `var_equal` configs — see
  `tests/test_theta_propagation.py`. Note: θ still does not enter the *parameter
  error* propagation (Jacobian is coupling-only); that is the next step.

- `computeResults` no longer accumulates an interference matrix for `writeResult` /
  `writeInterfResult` (neither consumed it): the matrix argument now accepts `nullptr`
  and the per-event `n(n+1)/2` atomics are skipped. The matrices come from
  `computeFileBatchIntegrals` instead.

### Added

- **Two symmetric interference matrices in `weight_best.root`**, each normalised by
  its own coherent total (entries are O(1) fractions, `trace + Σ_{i<j} = 1`):
  `interference_rec` (reconstructed, efficiency-weighted MC) and `interference_gen`
  (generated, `phsp_truth`). `diag = Σw|A_i|²/W`, `off = Σw·2Re(A_iA_j*)/W`, with the
  same weights as the fit (`phsp_weights` / `phsp_truth_weights` now apply — the old
  `interference` matrix ignored them). The companion tree `integrals` stores
  `N_rec`/`N_gen` (Σw) and `W_rec`/`W_gen`, so the raw integrals are one multiplication
  away; the diagonal of `interference_gen` is the literature fit fraction `F_i` and the
  upper triangle its interference term `I_ij`. The old `interference` object and the
  redundant `amplitude_names` tree were dropped. `CTPWA_WR_NO_INTEGRALS=1` skips both
  passes (and then writes neither the matrices nor the tree).
- **`interference_rec_err` / `interference_gen_err`** — parameter-error matrices for the
  two tables, obtained by linearised propagation `σ(F_ij) = sqrt(J·Cov·Jᵀ)` with
  `Cov = H[free,free]⁻¹` over the free couplings (reference directions masked) plus the
  free θ directions (`∂/∂m₀`, `∂/∂Γ₀` now reach the integrals after the θ fix above).
  Cost is `2·n_free` integral passes; `writeResult(..., hessian=H)` reuses the fit
  Hessian (omitting it computes one internally), `CTPWA_WR_NO_ERR=1` skips them. A
  non-positive-definite Hessian yields no matrices plus a warning instead of a bogus
  number. New API `getFitFractionMatrixErrors(params, hessian)` returns the same two
  matrices for scripting; see `tests/test_matrix_errors.py` (validated against an
  independent finite-difference implementation).
- **`getFitFractionMatrix(vector, sample="truth"|"phsp")`** returns the raw integral
  matrix of either sample (truth = `phsp_truth`, phsp = the efficiency-weighted MC;
  divided by its coherent total it reproduces the `interference_gen` / `interference_rec`
  objects written by `writeResult`, verified element by element). See
  `doc/branching_fraction_recipe.md` for the literature formulas
  (Laura++ Eq. 22-23, LHCb Eq. 4-5, Eq. 21-22/25) and for which matrix is needed
  to turn fit fractions into branching fractions.
- **`getFitFractionMatrix(vector)`** returns the raw truth-sample integral matrix
  behind `getFitFractions`: `M[i][i] = Σw|A_i|²` (the quantity the built-in fit
  fractions divide) and `M[i][j] = Σw·2Re(A_iA_j*)` (the interference integral the
  kernel already computed and then discarded). With `W = trace(M) + Σ_{i<j} M[i][j]`
  both conventions come out of one matrix — ctpwa's `FF_i = M_ii/Σ_j M_jj`
  (incoherent denominator, `Σ FF_i = 1`) and tf-pwa's
  `FF_i = M_ii/W`, `FF_ij = M_ij/W` (coherent denominator, table closes to 1).
  See `doc/fitfraction_vs_tfpwa.md` for the full comparison with
  `/home/whitewash/pkgs/tf-pwa`.
- **Per-event weights for the truth sample** (`phsp_truth_weights`). `phsp_truth`
  is what fit fractions and per-wave efficiencies integrate over; it previously
  always ran equal-weight, so a generator-level sample produced with a (small)
  model could not stand in for flat phase space. The new key takes the same
  forms as `data_weights` / `phsp_weights` (constant `[2.0]`, `[dat, file]` with
  one weight per line, or `[ROOT, file, tree, branch]`) and changes the truth
  integrals to `Σ w|A|²`, the normalization to `Σ w`, and the MC statistical
  error of the efficiency to `Σ(w|A|²)²` — matching tf-pwa's
  `_eval_integral` (`Σ w·f`, `Σ(w·f)²`). A model-generated MC with
  `w ∝ 1/|M_gen|²` therefore behaves as flat phase-space MC.
  `phsp_weights` now also enters the *numerator* of `getEfficiency` (it used to
  apply to the fit NLL only, so the efficiency ratio mixed weighted and
  unweighted samples); a constant weight leaves every result unchanged.
  Weighting is batched per GPU and reuses one workspace, so a 15M-event truth
  sample costs one extra `double` buffer per GPU.
- `phsp_truth` / `phsp` may now be ROOT ntuples in the batched integrals: the
  entry count comes from the `TTree` and `readMomentaFromDat` honours the
  `offset` argument on the ROOT path (it used to ignore it, silently re-reading
  the first batch for every batch beyond the first). DIYMC output
  (`[ROOT, file, "mc"]` plus `[ROOT, file, "mc", "weight"]`) can therefore be
  fed straight back as a weighted truth sample.

### Improved

- **Fixed-node budget (static chain factors).** A decay node whose parameters are
  all fixed *and* whose daughters are not free resonances contributes a
  θ-independent factor, so the product over all such nodes is materialised once
  per (σ, event, SL) into a table (`d_static`) at initialisation. The amplitude
  kernel then only loops over the remaining free nodes and multiplies the chain
  amplitude by that table — `dF = F_total × dln` stays exact because the static
  nodes contribute nothing to `dln`. Node classification is conservative: nodes
  whose daughter is a free resonance stay in the hot loop (their `q0` moves with
  θ), and only BWR(2-parameter)/`Bf`/unity factors are foldable — Custom, Interp,
  GS and `ONE` nodes stay in the loop. `CTPWA_NO_STATIC=1` forces the old
  all-nodes path (bitwise A/B, used in the review). On the 5-body benchmark with
  one free resonance per chain: 193 → 113 ms (1.7×, RTX 5060); the all-free
  benchmark is unchanged (no table is built).
  A/B parity vs `CTPWA_NO_STATIC=1`: NLL identical to 1e-16 relative, gradients
  2.6e-15 maximum relative deviation (CUDA and CPU backends).
- The contraction is now generic: `buildLiftedAux()` takes the model's "shell"
  expression (lifted quantities referenced as placeholder parameters) plus the
  list of lifted sub-expressions, and assembles every derivative segment from
  the multivariate chain rule. Each model only declares *what* to lift
  (BWR/GS are ~30 lines each now) and lift-to-lift dependencies are resolved
  automatically (BWR's `gamma` is built from the `Bf` table).
- Symbolic differentiation can now be *contracted*: expensive model factors
  (`Bf`, the `(q/q0)^{2L+1}` factor, the BWR width `gamma`, and the GS
  coefficients `d/m0`, `f/g0`, `(q/q0)^3 m0/m`) are lifted into a per-node
  derivative table that is evaluated once per event/SL, and the node's
  derivative segments are assembled from it by the chain rule instead of
  expanding every factor again. BWR `d2F` drops from 21k to 4.3k bytecode
  instructions, GS from 264k to 35k.
  The contracted variant is only used by the second-order (`Hessian`) path;
  amplitude/gradient paths keep the original `aux`. GS Hessian is ~4.8x faster
  (RTX 5060, `tests/configs/gs.yml`: 588 -> 123 ms); BWR keeps its analytic
  specialisation and is unchanged. Disable with `CTPWA_NO_LIFT=1`.

### Fixed

- The BWR contraction used a single `Bf` table for both the width (`Lmin`) and the
  vertex (runtime `L`); they only coincide when `L == Lmin`. The vertex barrier
  factor is now its own lifted quantity.
- `evalCustomAll` passed `lifts = nullptr` while evaluating the lift tables, so
  a lifted quantity that referenced an earlier one (BWR `gamma` using `B`) read
  zero. This silently zeroed the imaginary part of the amplitude and made the
  Hessian's q0 finite-difference terms explode to ~1e11.
- `addModelQ0HessianTerms` now evaluates its base value from the same `aux` as
  the perturbed ones; previously the base came from the caller (possibly a
  different path) and the mismatch was amplified by `1/h^2`.

## [v0.3.21] - 2026-11-16

### Fixed

- Stop `writeResult` / `writeInterfResult` from exhausting device memory on
  analyses with slightly more than 50 partial waves. `computeModWithInterference`
  kept a per-thread `npartials*(npartials+1)/2` accumulator in local memory even
  when no per-event interference was requested, and CUDA reserves local memory for
  the maximum resident thread count (V100: 2048 x 80 threads). The 200-slot bucket
  therefore demanded ~25 GiB of local memory at kernel launch while the amplitude
  tables were resident, so the fit ran fine and only the result-writing step died
  with `computeModWithInterference failed: out of memory`. The accumulator is now
  compiled out when no per-event export is requested (164 KB -> 3.2 KB per thread)
  and otherwise bounded by the number of exported pairs (32/128/512/2048 buckets,
  at most 2048 pairs per call).
- Warn when a result-writing kernel's local-memory pool would exceed a quarter of
  the free device memory, so the failure points at the partial-wave count instead
  of the phase-space sample size.

### Improved

- Report the partial-wave count in the initialization banner and print the SL
  amplitude count next to it. The line previously printed only the SL amplitude
  count under the label "Number of partial waves", which hid which
  `computeModWithInterference` bucket a configuration lands in.

## [v0.3.20] - 2026-11-13

### Fixed

- Apply `data_weights` when filling data histograms in `writeResult`. Previously the
  fitted model and background histograms used the weighted event yield, but `hdata`
  used raw event counts, so weighted datasets produced inconsistent plots.
- Keep unweighted data behavior unchanged: a missing per-GPU data-weight pointer is
  treated as weight 1.

## [v0.3.19] - 2026-11-08

### Fixed

- Initialize the host interference-matrix accumulator in `writeResult`. Previously,
  uninitialized memory could contaminate arbitrary matrix elements, producing NaN or
  values above `1e170` even when the fitted amplitudes were finite.
- Keep off-diagonal interference calculations in double precision instead of temporarily
  narrowing the partial-wave amplitudes to `float`. This affects both the saved matrix and
  event-by-event interference output.
- Complete model-`q0` derivative and Hessian terms for nested resonances with free masses,
  while avoiding double counting of vertex barrier-factor contributions.
- Enable JIT compilation for built-in BWR/GS derivative segments and make JIT activation
  explicit.
- Remove a double free in `releaseStaticData` for identical-particle permutation momenta.
- Correct CPU-side Hessian checks in `getFitFractions`/`getEfficiency` and handle missing
  `phsp_truth` earlier.
- Follow the extension complex dtype in the example fit and apply a scale-aware rank threshold.

### Improved

- Evaluate BWR through native functions instead of the bytecode interpreter, reducing the
  amplitude-path runtime by about 32% in the development benchmark.
- Add AD kernel timing breakdowns and skip unnecessary polar-angle work for built-in models.
- Add a benchmark script separating NLL and NLL-plus-gradient costs for free parameters.
- Improve documentation of device-dependent random parameter initialization in tests.

### Documentation

- Reorganize the README around the project purpose, features, installation, backend selection,
  quick start, and common APIs.
- Add a separate English README and cross-links between the Chinese and English versions.
