# Changelog

## [Unreleased]

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
