# CTPWA

**English** | [简体中文](README.md)

**CTPWA (CUDA-Torch Partial Wave Analysis) is a covariant-tensor partial-wave analysis toolkit for high-energy physics.**

It defines particles, decay chains, and resonances through YAML configuration files,
builds amplitudes from experimental data and phase-space samples, and evaluates the
negative log-likelihood (NLL), gradients, and Hessians. It also produces fit projections,
partial-wave weights, interference matrices, and Monte Carlo samples. The core calculations
are implemented in C++/CUDA and exposed to Python through PyTorch.

Each wheel contains both CUDA and CPU backends. CTPWA uses a GPU when one is available;
otherwise, it falls back to the CPU and prints the reason. The same analysis configuration
can therefore be used on local machines, computing clusters, and CI systems.

## Main features

- **Covariant-tensor amplitudes**: two-body and cascade decays, spin-orbit couplings,
  barrier factors, and identical-particle handling.
- **Resonance models**: BWR, BW, Flatté, Gounaris–Sakurai, ONE, Interp, and Custom.
- **Fit calculations**: NLL, PyTorch autograd gradients, Hessians, fit fractions,
  and efficiency corrections.
- **CUDA and multi-GPU support**: GPU acceleration for amplitudes, likelihoods,
  and result projections.
- **CPU backend**: OpenMP-based execution without a CUDA GPU, suitable for debugging,
  small samples, and machines without GPUs.
- **Result output**: ROOT histograms, event-by-event weights, interference terms for
  selected wave pairs, and final-state four-momenta.
- **Monte Carlo generation**: flat multibody phase space and MC samples weighted by
  the current amplitude.
- **Large-sample support**: streaming phase-space mode and multiple precision modes
  to reduce memory consumption.

## Requirements

| Component | Requirement | Notes |
|---|---|---|
| Operating system | Linux x86_64, glibc ≥ 2.34 | AlmaLinux/RHEL 9, Ubuntu 22.04, and compatible systems |
| Python | 3.12 | Build environment of the current PyPI wheel |
| PyTorch | 2.12+ | The release process uses `2.12.0+cu126` |
| ROOT | 6.26+ | Data input and result output; load `thisroot.sh` before use |
| NVIDIA GPU | Optional, sm_70 or newer | The CPU backend works without a GPU |
| CUDA Toolkit | Required when building the CUDA backend from source | Release builds use CUDA 12.9 |

> PyPI wheels statically link yaml-cpp, so wheel users do not need to install it separately.
> Source builds require a C++17 compiler and yaml-cpp. Building the CUDA backend also
> requires the CUDA Toolkit.

## Installation

### Option 1: Install from PyPI

Install ROOT and load its environment, then install the PyTorch version used by the
current wheel and CTPWA itself:

```bash
source /path/to/root/bin/thisroot.sh

python -m pip install torch==2.12.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install ctpwa
```

GPU users also need an NVIDIA driver and CUDA runtime compatible with their GPU architecture.
The CPU backend does not require an available NVIDIA GPU.

Verify the installation and the selected backend:

```python
import ctpwa

print(ctpwa.__version__)
print(ctpwa.backend())     # "cuda" or "cpu"
print(ctpwa.build_info())  # Precision, complex type, CUDA support, and other build information
```

### Option 2: Build from source

```bash
git clone https://github.com/BHXiang/CTPWA.git
cd CTPWA

# Example: use conda to provide ROOT and yaml-cpp
conda install -c conda-forge root yaml-cpp=0.8
```

Build the extension in the repository for development:

```bash
# Build only the CPU backend; neither CUDA Toolkit nor a GPU is required
CTPWA_BACKEND=cpu MAX_JOBS=4 python setup.py build_ext --inplace

# Build only the CUDA backend
CTPWA_BACKEND=cuda MAX_JOBS=4 python setup.py build_ext --inplace

# Build both backends, as done for release wheels
CTPWA_BACKEND=both MAX_JOBS=4 python setup.py build_ext --inplace
```

`MAX_JOBS` limits parallel C++/CUDA compilation. Set it to `1` if the build runs out of memory.

## Backend selection

```bash
# Default: probe CUDA and prefer it; fall back to CPU with a printed reason if unavailable
export CTPWA_DEVICE=auto

# Require CUDA; fail immediately if it is unavailable
export CTPWA_DEVICE=cuda

# Require CPU
export CTPWA_DEVICE=cpu
```

`CTPWA_DEVICE` is a **runtime** option, while `CTPWA_BACKEND` is a **build-time** option.
The CPU backend is useful for debugging and small samples; GPUs are recommended for
production-scale fits.

## Quick start

### 1. Write an analysis configuration

The following minimal configuration describes `J/ψ → π⁺π⁻π⁰` through
`ρ(770) → π⁺π⁻`:

```yaml
Particles:
  Jpsi: {J: 1, P: -1, mass: 3.0969}
  pip:  {J: 0, P: -1, mass: 0.13975}
  pim:  {J: 0, P: -1, mass: 0.13975}
  pi0:  {J: 0, P: -1, mass: 0.13495}

Data:
  order: [pip, pim, pi0]
  data: [dat, "data.dat"]
  phsp: [dat, "phsp.dat"]

DecayChains:
  chain1:
    Jpsi: [[pi0, R_pipi]]
    R_pipi: [pip, pim]
    intermediates:
      R_pipi:
        - [J: 1, P: -1]: [rho770]

Resonances:
  rho770:
    J: 1
    P: -1
    model: BWR
    parameters: [0.775, 0.149]
    free: [0, 1]
```

Here, `data` contains experimental or simulated events, while `phsp` contains the
phase-space sample used for fit normalization. The final-state ordering in the data files
must match `Data.order`. See [example](example/) for complete configurations and fit scripts.

All three samples accept optional per-event weights (same syntax for each):

```yaml
Data:
  data_weights:        [dat, "data_w.dat"]          # one weight per line
  phsp_weights:        [ROOT, "phsp.root", "tree", "weight"]
  phsp_truth_weights:  [2.0]                        # constant weight
```

With weights the normalization uses `Σw` instead of the event count and
`NLL = -Σ w·log f`. For `phsp_truth` (the efficiency-free MC truth behind fit
fractions and per-wave efficiencies) the integrals become `Σ w|A|²` and the MC
statistical error uses `Σ(w|A|²)²`. A model-generated sample
(e.g. `generateMC(mode="weighted")`) therefore becomes a usable flat phase-space
sample simply by supplying `w ∝ 1/|M_gen|²`.

### 2. Evaluate the NLL and its gradient

```python
import torch
import ctpwa

ana = ctpwa.analysis("config.yml")
device = torch.device(ctpwa.backend())

n_vec = ana.getNVector()
n_theta = ana.getNFreeTheta()
params = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64, device=device)

# Fix one reference coupling to remove the overall phase and magnitude gauge freedom
params[0] = 1.0

# Initialize free resonance parameters to their nominal configured values
if n_theta:
    params[2 * n_vec:] = ana.getFreeResParams()[0].to(
        device=device, dtype=torch.float64
    )

params.requires_grad_(True)
nll = ana.getNLL(params)
grad = torch.autograd.grad(nll, params)[0]

print("NLL:", nll.item())
print("gradient:", grad)
```

The parameter layout is `[Re(couplings), Im(couplings), resonance_parameters]`. Fits can
use PyTorch optimizers; see [example/fit.py](example/fit.py) for a complete workflow.

### 3. Save the fit result

```python
ana.writeResult(params.detach(), "weight_best.root", 0)
```

The output file contains model/data projections, partial-wave weights, legends, and **two
raw (unnormalised) integral matrices**:

| Object | Content |
|---|---|
| `interference_rec` | reconstructed (efficiency-weighted MC) interference matrix ÷ `W_rec` |
| `interference_gen` | generated (`phsp_truth`, efficiency-free MC) interference matrix ÷ `W_gen` |
| `interference_rec_err` / `interference_gen_err` | same-shape **parameter-error** matrices (`sqrt(J·H⁻¹·Jᵀ)`, MC statistics excluded) |
| `integrals` | tree: `N_rec`/`N_gen` (Σw), `W_rec`/`W_gen` (coherent total = `trace+Σ_{i<j}`), `npartials` |

Both tables are normalised (`trace + Σ_{i<j} = 1`, entries are O(1)) and use the same
event weights as the fit. Therefore:

```
literature fit fractions F_i = interference_gen[i][i]   interference I_ij = interference_gen[i][j]
incoherent fractions     FF_i = diag / Σ diag           (what getFitFractions returns)
average efficiency       ε̄ = W_rec / W_gen             (same generated sample)
per-wave efficiency      ε_i = (P_ii/N_rec)/(T_ii/N_gen)
raw integrals            M_ij = interference_*[i][j] × W_*
```

See [doc/branching_fraction_recipe.md](doc/branching_fraction_recipe.md).
Read the two tables with `python3 example/read_interference.py weight_best.root`
(per-wave F_i±σ, full matrix, closure check, ε̄/ε_i, and `--bf-total` for BF_i).

The error matrices are propagated from the `hessian=H` argument of
`writeResult(params, file, flag, waves, hessian=H)` (computed internally when omitted); cost is
`2·n_free` integral passes. `CTPWA_WR_NO_INTEGRALS=1` skips the matrices and `integrals`,
`CTPWA_WR_NO_ERR=1` skips only the error matrices.

## Common Python API

| Interface | Purpose |
|---|---|
| `ctpwa.analysis(config)` | Load the configuration and initialize amplitudes, data, and device buffers |
| `getNLL(params)` | Evaluate the negative log-likelihood with PyTorch autograd support |
| `getHessian(params)` | Evaluate the Hessian for couplings and free resonance parameters |
| `getFitFractions(...)` | Calculate partial-wave fit fractions (incoherent denominator, `Σ_i FF_i = 1`) |
| `getFitFractionMatrix(vector, sample="truth"\|"phsp")` | Raw integrals `M[i][i]=Σw\|A_i\|²`, `M[i][j]=Σw·2Re(A_iA_j*)`; normalising by `W=trace+Σ_{i<j}M_ij` gives the literature fit fractions `FF_i`/`FF_ij`, and `W_phsp/W_truth` is the model-averaged efficiency |
| `getEfficiency(...)` | Calculate efficiency corrections |
| `writeResult(...)` | Write ROOT fit results, projections, and the interference matrix |
| `writeInterfResult(...)` | Write event-by-event interference terms for selected wave pairs |
| `PhaseSpaceGenerator(config)` | Generate flat multibody phase space |
| `getIntensity(momenta, params)` | Evaluate event-by-event intensity for arbitrary four-momenta |
| `generateMC(...)` / `writeMC(...)` | Generate and save amplitude-weighted MC samples |

## Large samples and precision

For large phase-space samples, enable streaming mode to avoid keeping all phase-space
amplitudes in device memory:

```yaml
Constraints:
  free_phsp_amplitudes: true
```

This mode requires all resonance parameters to be fixed. The
`precision: auto | hybrid | float | double` setting controls amplitude storage and
calculation precision. Mixed precision can save memory during an initial fit; use double
precision for final validation.

## Monte Carlo samples

```python
# Flat phase space
mcgen = ctpwa.PhaseSpaceGenerator("config.yml")
events = mcgen.generate(100_000, seed=42)
mcgen.write(events, "flat.root")

# Flat events + GENBOD weights (weighted output goes to ROOT only: branch "weight" of tree "mc")
flat, w = mcgen.generateWeighted(100_000, seed=42)
mcgen.write(flat, "flat_weighted.root", weights=w)

# Rejection sampling with the current amplitude, producing an equal-weight MC sample
result = ana.generateMC(10_000, params.detach(), seed=42, mode="unweighted")
ana.writeMC(result["events"], "mc.root")

# Amplitude-weighted MC: feed the weights back through `phsp_truth_weights`
weighted = ana.generateMC(10_000, params.detach(), seed=42, mode="weighted")
ana.writeMC(weighted["events"], "mc_weighted.root", weighted["weights"])
```

The phase-space sample must resolve the narrowest resonance in the model. Otherwise,
statistical fluctuations in the normalization can affect the fit and the Hessian.

## Development and testing

```bash
# Build and run the complete test gate
bash scripts/run_tests.sh

# Run tests without rebuilding
bash scripts/run_tests.sh --no-build
```

See the source code and tests for development conventions, dual-backend implementation,
and debugging procedures. Run the appropriate tests on both CUDA and CPU backends before
a release. Release notes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

MIT License
