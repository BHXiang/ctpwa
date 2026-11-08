# Changelog

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
