# Changelog

## [Unreleased]

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
