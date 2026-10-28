# CTPWA

**CUDA + PyTorch 协变张量分波分析（Partial Wave Analysis）工具包**

用于高能物理分波分析：基于协变张量理论构建衰变振幅，
用 CUDA 在 GPU 上并行计算。

同一个 wheel 同时带 **GPU 后端**与**纯 CPU 后端**：`import ctpwa` 时自动探测，
有可用 GPU 就用 GPU，没有（或驱动/架构不匹配）就回退 CPU 并明确告警。

---

## 特性

- **协变张量分波**：两体衰变振幅（自旋-轨道耦合、势垒因子、宇称约束），支持级联衰变链
- **GPU 并行**：振幅、SL 振幅、共振态因子全部 CUDA kernel 实现，多 GPU 支持
- **CPU 后端**：同一份源码可编出纯 CPU 版（OpenMP + 自实现 BLAS，不依赖 OpenBLAS/MKL），
  与 GPU 版数值一致（NLL ~1e-11、梯度 ~1e-10、Hessian ~1e-12 相对差）
- **拟合**：NLL + 梯度（解析/AutoDiff）+ Hessian，接入 PyTorch autograd 优化器

## 系统要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Linux x86_64 | glibc ≥ 2.34 | AlmaLinux 9.8 / RHEL 9 / Ubuntu 22.04+ |
| NVIDIA GPU | sm_70 及以上 | **可选**：没有 GPU 时自动使用 CPU 后端 |
| CUDA Toolkit | 12.9 | 运行时需对应驱动 |
| PyTorch | 2.12+（cu126） | 官方 legacy 构建（仅 download.pytorch.org） |
| ROOT | 6.26+ | 读取 .root 数据（`source thisroot.sh`） |
| yaml-cpp | 0.8+ | config 解析（conda-forge 最新 0.8.0；0.9.0 仅 defaults 源） |

## 安装

### 从 PyPI 安装（免编译）

```bash
# 系统依赖
conda install -c conda-forge yaml-cpp=0.8
# ROOT：下载/编译后 source thisroot.sh
# CUDA：安装与 GPU 匹配的驱动 + CUDA 12.9 runtime

# PyTorch
pip install torch==2.12.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# CTPWA（manylinux_2_34 wheel，内置 sm_70~sm_120 与 CPU 后端）
pip install ctpwa
```

### 选哪个后端

```python
import ctpwa
ctpwa.backend()      # 'cuda' 或 'cpu'（实际使用的）
ctpwa.build_info()   # {'backend': ..., 'precision': ..., 'capabilities': ...}
```

| 环境变量 | 作用 |
|---|---|
| `CTPWA_DEVICE=auto`（默认） | 有可用 GPU 用 GPU，否则回退 CPU（会打印原因与耗时提醒） |
| `CTPWA_DEVICE=cuda` | 强制 GPU；不可用时**直接报错**，不静默降级 |
| `CTPWA_DEVICE=cpu` | 强制 CPU |
| `CTPWA_QUIET=1` | 抑制导入期提示 |

回退判定不只看 `torch.cuda.is_available()`，还会真启动一次 kernel ——
"驱动看得见卡但 kernel 起不来"（编译架构与卡不匹配、卡被 MIG 切分或占满）
只有真跑才能发现。显存不足（OOM）**不会**回退：那是配置问题，应减小 phsp
事例数或开 `free_phsp_amplitudes`。

> CPU 后端比 GPU 慢 20~60 倍（事例越多差距越大），适合无卡机器、CI 与调试；
> 生产规模拟合仍建议用 GPU。

### 源码编译（开发 / 老系统）

```bash
git clone git@github.com:BHXiang/CTPWA.git
cd CTPWA
pip install -e . --no-build-isolation
```


## 快速开始

```python
import torch
import ctpwa

ana = ctpwa.analysis("config.yml")          # 解析 config + 初始化振幅
n_vec = ana.getNVector()                    # 自由耦合数（trans 折叠后）
n_theta = ana.getNFreeTheta()               # 自由共振态参数数

params = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64, device="cuda")
params[0] = 1.0                             # 参考振幅 v0 = 1

nll = ana.getNLL(params)                    # 负对数似然（可 autograd）
grad = torch.autograd.grad(nll, params)[0]  # 梯度
H = ana.getHessian(params)                  # 全 Hessian（含耦合+共振态参数块）
```

config.yml 最小示例：

```yaml
# 经典道 J/ψ → π⁺π⁻π⁰：中间态 ρ(770) → π⁺π⁻（π⁰ 为旁观粒子）
Particles:
  Jpsi: {J: 1, P: -1, mass: 3.0969}
  pip:  {J: 0, P: -1, mass: 0.13975}
  pim:  {J: 0, P: -1, mass: 0.13975}
  pi0:  {J: 0, P: -1, mass: 0.13495}

Data:
  order: [pip, pim, pi0]
  data: [ROOT, "file/data.root", TTree, TBranch1, TBranch2, ...]
  phsp: [ROOT, "file/phsp.root", TTree, TBranch1, TBranch2, ...]
  bkg:  [ROOT, "file/bkg.root", TTree, TBranch1, TBranch2, ...]
  bkg_weights: [0.8] # [dat, "*.dat"]

DecayChains:
  chain1:
    Jpsi: [[pi0, R_pipi]]
    R_pipi: [pip, pim]
    intermediates:
      R_pipi: [[J: 1, P: -1]: [rho770]]

Resonances:
  rho770: {J: 1, P: -1, model: BWR, parameters: [0.775, 0.149], free: [0, 1]}
```

## Python API

| 方法 | 说明 |
|---|---|
| `analysis(config)` | 加载 config，初始化粒子/衰变链/振幅 |
| `getNLL(params)` | NLL（float64 CUDA tensor，支持 autograd） |
| `getHessian(params)` | 全 Hessian `(2n+P)×(2n+P)` |
| `getNVector()` / `getNFreeTheta()` | 自由耦合数 / 自由共振态参数数 |
| `getFreeResParams()` | 共振态参数初值、下界、上界 |
| `getParamNames()` | 参数名（trans 折叠后唯一） |
| `writeResult(params, file)` | 保存拟合结果 |
| `DeviceManager` | GPU 检测 / 显存容量检查 / 精度查询 |
| `PhaseSpaceGenerator(config)` | DIYMC：产生平坦 n 体相空间（CPU，不需 data/GPU） |
| `getIntensity(momenta, params)` | 任意四动量逐事件强度（CPU 输入，CPU 返回） |
| `generateMC(n, params, ...)` | 振幅加权 MC：unweighted 舍选 / weighted |
| `writeMC(events, file, weights=None)` | 写 DIYMC 样本（DAT/ROOT） |

### Hessian 正定性与盒约束（重要）

`getHessian` 返回的是**无约束** NLL 的精确 Hessian，布局
`[Re(p), Im(p), θ]`。其中固定参考耦合 `Re_0=1` / `Im_0=0` 对应两行、两列
恒为 0（规范方向），求特征值前必须剔除，否则 `eigvalsh(...).min()` 必然
看到零/负的数值噪声。

拟合参数带 `free_range` / `v_max` 时，最优点可能落在边界上。此时沿
**朝外方向** Hessian 为负是正常的边界效应，不代表拟合点不是约束极小。
正确判据（KKT 二阶条件）是：先剔除"贴边且梯度朝外"的活跃参数，再在
**可行子空间**判正定。

```python
from ctpwa.diagnostics import analyze_hessian

rep = analyze_hessian(H, params, grad, lower, upper, fixed_mask)
print(rep.summary(param_names))
rep.reduced_pd        # ← 应该看的那个（可行子空间）
rep.free_pd           # 旧口径（含贴边参数，会假报警）
rep.active_idx        # 被边界钉住的参数
```

`example/fit.py` 已按这套判据输出 `is_positive_definite`（可行子空间）
与 `is_positive_definite_full`（旧口径），并列出 `active_params`。
对 `J/ψ→K⁺K⁻η` 这类单 LS 过程，`getHessian` 本身是精确的：随机点/真极小点
的解析-FD 一致性由 `tests/test_cpu_numerics.py` / `tests/test_numerical.py`
覆盖；本次还在生产拟合点上沿最负的 5 个特征方向做了二阶差分校验
（相对差 < 0.5%）。多 LS + 大 bkg 的符号 bug 见下。

**DIYMC 闭包注意**：phsp 样本必须能分辨最窄的共振。实测 4.5 MeV 宽的
φ(1020) + 仅 10k phsp 时，toy 真值不是 NLL 极小（拟合把宽度推到边界），
真值处 Hessian 出现负本征值；同一 toy 数据只把 phsp 增到 200k，BFGS 极小
（ΔNLL=4.3）处 H 已正定、θ 回到真值。这是归一化 MC 的统计分辨率不足，
不是 getHessian 写错。

**data_weights / bkg_weights**：已用 FD 对照（data 0.5/1.0、bkg 0.8，
含/不含 phsp_weights），梯度相对差 ~1e-7、Hessian PASS；它们同时进入
`computeFactorNLL`、data Hessian 和 `N_eff`，没有发现 phsp 权重那种
NLL/导数不同源的问题。

**phsp_weights 归一化**：旧版 forward 的 `phsp_factor` 用无权重 `Σf/N`，
但 `d_phsp_matrix_` / 梯度 / Hessian 用 `Σw/Σw`，NLL 与导数不是同一条函数
（`J/ψ→γK_SK_Sπ⁰π⁰` 上 H vs FD 差 15%、梯度差 30%）。现已改为
`Σw f / Σw`，回归见 `tests/test_cpu_numerics.py::test_phsp_weights_nll_grad_hessian`。
注意：流式 phsp（`free_phsp_amplitudes`）仍用旧的 `/N`，且自由 θ 的 θθ
Hessian phsp 项尚未接权重（NLL/梯度已接）。

**多 LS + 大 bkg 的专门回归**：`J/ψ→ωηη′`（9 链、69 振幅、21 耦合，
3074 bkg / 2024 data）在旧版上 H 有 9 个负本征值（λmin=-533），FD 却是
PSD。根因是 `getHessian` 耦合变换二阶项里 bkg 的 `g_v` 被翻了两次符号
（有权重时 `d_neg_w` 一次 + `axpy(-1)` 一次；无权重时 kernel 用单位正权重），
已在 `src/Analysis.cu` 按是否有 `bkg_weights` 选择 `axpy(+1/-1)`。修复后 H 与 FD 逐元一致（max rel 0.48%），λmin=+0.002，
0 个负本征值。回归见
`tests/test_cpu_numerics.py::test_bkg_sign_in_coupling_hessian`（把
`hess_mult` 的 bkg 权重放大到 20；旧版 FAIL max_rel=1.28，新版 PASS
max_rel=9e-3）。GPU（RTX 5060）复测：该过程 H 从 9 个负本征值变为 0，
λmin=+0.00244，FD 0 负；全量 `pytest tests/` 为 **208 passed, 50 skipped,
1 xfailed**。**有 bkg_weights 的用户需要重新编译 `_cuda.so` / `_cpu.so`：**

```bash
MAX_JOBS=1 CTPWA_BACKEND=cuda python3 setup.py build_ext --inplace
MAX_JOBS=4 CTPWA_BACKEND=cpu  python3 setup.py build_ext --inplace
```


## DIYMC：平坦相空间 + 振幅加权 MC

### 平坦相空间（P1）

`PhaseSpaceGenerator` 只解析 config 的 `Particles` / `Data.order` / `DecayChains`，
**不读 data/phsp 文件，也不需要 GPU**：

```python
import torch, ctpwa

gen = ctpwa.PhaseSpaceGenerator("config.yml")   # 或 mother="Jpsi"
flat = gen.generate(100_000, seed=42)            # [N, n_final, 4]，(E, px, py, pz)
gen.write(flat, "flat.dat")                      # 或 "flat.root"
```

- 产生的是**平坦相空间**：内部用 GENBOD 的 `wt` 做舍选（`u < wt`），不是直接
  收下所有 `Generate()` 事例。
- 末态顺序 = `Data.order`；母粒子默认静止系，四动量守恒/质量壳由测试保证。
- `gen.generateWeighted(n, seed)` 返回 `(events, weights)`。

### 振幅加权 MC（P2/P3）

在已构造的 `analysis` 上用拟合参数产生满足 |M|² 的样本：

```python
res = ana.generateMC(
    10_000, params, seed=42,
    mode="unweighted",   # 或 "weighted"
    batch=50_000,        # 每批平坦事例数
)
print(res["stats"])
ana.writeMC(res["events"], "mc.root")   # weighted 时传 weights=res["weights"]
```

- `unweighted`：平坦相空间 → `I=|M|²` → `u*Imax < I` 舍选，返回等权样本。
- `weighted`：返回全部平坦事例 + `I` 权重（不做舍选）。
- `max_weight` 可给 `Imax` 初值；`None` 时用首批 pilot 自适应（抬高时按保存的
  `u` 精确重判已接受事例）。`max_trials` 限制平坦事例数（安全检查）。
- `getIntensity(momenta, params)` 是低层原语：`momenta` 为 CPU
  `float64 [N, n_final, 4]`，返回 CPU `float64 [N]`；归一化任意，只保证相对值。
- 两者都使用**局部 `AmpCalc`**，不改变拟合状态（调用前后 `getNLL` 不变），
  GPU / CPU 后端均可用。

## 大 phsp 低显存模式（`Constraints.free_phsp_amplitudes: true`）

当 phsp 样本远大于 data（如千万级 phsp）时，默认把 phsp 振幅整表驻留显存会爆。
config 的 `Constraints` 段加一行开关即可切换为 **phsp 全程流式（不驻留）**：

```yaml
Constraints:
  free_phsp_amplitudes: true   # 仅当所有共振态质量/宽度固定（无 free 参数）时生效
```

效果：
- 构造期：phsp 振幅按批（默认 ≤10 万事件/批，按振幅规模自适应）从四动量算完即弃，
  分块累加 phsp 矩阵（float32 给梯度/Hessian + double 给归一化因子）；
- 拟合期：phsp 零驻留，NLL/梯度/Hessian 只依赖 data/bkg 表 + n×n 矩阵；
- `writeResult` / `writeInterfResult`：按批重算 phsp 振幅出图，峰值与 phsp 总量无关。

注意：
- 有任一 `free` 参数时开关被忽略（打印警告，保持驻留）；
- 流式模式下 `getPhspTensor()` / `saveSLAmps()` / `getSLAmpsTensor()` 不可用（明确报错）；
- 数值上 phsp 归一化因子改走 double 矩阵，与驻留模式一致（精度只升不降）。

## 性能（RTX 5060, 50 万事件）

- fwd+bwd（NLL+梯度）：约 18 ms
- Hessian：约 0.5 s


---

# CTPWA (English)

**Covariant-tensor Partial Wave Analysis with CUDA and PyTorch**

A partial-wave analysis toolkit for high-energy physics: decay amplitudes built on
covariant-tensor formalism, computed in parallel on GPUs with CUDA.

## Features

- Covariant-tensor partial waves (spin-orbit couplings, barrier factors, parity
  constraints), cascade decay chains
- GPU-accelerated amplitudes / SL amplitudes / resonance factors, multi-GPU support
- NLL + analytic/AutoDiff gradients + Hessian, plug into PyTorch optimizers

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Linux x86_64 | glibc ≥ 2.34 | AlmaLinux 9.8 / RHEL 9 / Ubuntu 22.04+ |
| NVIDIA GPU | sm_70+ | V100, A100, RTX 20-50, H100 |
| CUDA Toolkit | 12.9 | needed to compile; matching driver at runtime (sm_120 needs driver ≥ 570) |
| PyTorch | 2.12+ (cu126) | official legacy build (download.pytorch.org only) |
| ROOT | 6.26+ | for reading .root data (`source thisroot.sh`) |
| yaml-cpp | 0.8+ | config parsing (conda-forge latest 0.8.0; 0.9.0 is defaults-only) |

## Installation

### From PyPI (prebuilt wheel)

```bash
# system deps
conda install -c conda-forge yaml-cpp=0.8
# ROOT: install and `source thisroot.sh`
# CUDA: driver + CUDA 12.9 runtime

pip install torch==2.12.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install ctpwa
```

### From source

```bash
git clone git@github.com:BHXiang/CTPWA.git
cd CTPWA
pip install -e . --no-build-isolation
```

## Quick start

```python
import torch
import ctpwa

ana = ctpwa.analysis("config.yml")
n_vec = ana.getNVector()
n_theta = ana.getNFreeTheta()

params = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64, device="cuda")
params[0] = 1.0

nll = ana.getNLL(params)
grad = torch.autograd.grad(nll, params)[0]
H = ana.getHessian(params)
```

Minimal config.yml:

```yaml
# Classic J/psi -> pi+ pi- pi0: intermediate rho(770) -> pi+ pi- (pi0 as bachelor)
Particles:
  Jpsi: {J: 1, P: -1, mass: 3.0969}
  pip:  {J: 0, P: -1, mass: 0.13975}
  pim:  {J: 0, P: -1, mass: 0.13975}
  pi0:  {J: 0, P: -1, mass: 0.13495}

Data:
  order: [pip, pim, pi0]
  data: [ROOT, "file/data.root", TTree, TBranch1, TBranch2, ...]
  phsp: [ROOT, "file/phsp.root", TTree, TBranch1, TBranch2, ...]
  bkg:  [ROOT, "file/bkg.root", TTree, TBranch1, TBranch2, ...]
  bkg_weights: [0.8] # [dat, "*.dat"]

DecayChains:
  chain1:
    Jpsi: [[pi0, R_pipi]]
    R_pipi: [pip, pim]
    intermediates:
      R_pipi: [[J: 1, P: -1]: [rho770]]

Resonances:
  rho770: {J: 1, P: -1, model: BWR, parameters: [0.775, 0.149], free: [0, 1]}
```

## Python API

| Method | Description |
|---|---|
| `analysis(config)` | load config, initialize particles/chains/amplitudes |
| `getNLL(params)` | NLL (float64 CUDA tensor, autograd-friendly) |
| `getHessian(params)` | full Hessian `(2n+P)×(2n+P)` |
| `getNVector()` / `getNFreeTheta()` | #free couplings / #free resonance params |
| `getFreeResParams()` | resonance param initial values, lower/upper bounds |
| `getParamNames()` | parameter names (unique after trans folding) |
| `writeResult(params, file)` | save fit results |
| `DeviceManager` | GPU detection / memory capacity check / precision query |
| `PhaseSpaceGenerator(config)` | DIYMC: flat n-body phase space (CPU, no data/GPU) |
| `getIntensity(momenta, params)` | per-event intensity for arbitrary momenta |
| `generateMC(n, params, ...)` | amplitude-weighted MC (unweighted / weighted) |
| `writeMC(events, file, weights=None)` | write a DIYMC sample (DAT/ROOT) |

### Hessian positive-definiteness with box constraints (important)

`getHessian` is the **unconstrained** NLL Hessian in layout
`[Re(p), Im(p), θ]`. The fixed reference couplings `Re_0=1` / `Im_0=0`
give two exactly-zero rows/columns (gauge directions); drop them before
calling `eigvalsh`, or numerical zeros will look like "not PD".

When parameters have `free_range` / `v_max`, the optimum may sit on a
bound. Negative curvature along the **outward** direction is then a
boundary effect, not an unstable fit. The correct test (KKT second-order
condition) removes "at bound AND gradient points outward" parameters and
checks the **feasible subspace**:

```python
from ctpwa.diagnostics import analyze_hessian

rep = analyze_hessian(H, params, grad, lower, upper, fixed_mask)
print(rep.summary(param_names))
rep.reduced_pd        # <- the meaningful one
rep.free_pd           # legacy check (includes bound-active params)
rep.active_idx        # parameters pinned by bounds
```

`example/fit.py` now reports `is_positive_definite` (feasible subspace)
and `is_positive_definite_full` (legacy), plus `active_params`.
For single-LS processes such as J/psi→K+K-eta, `getHessian` is exact:
analytic-vs-FD agreement at random points and true minima is covered by
`tests/test_cpu_numerics.py` / `tests/test_numerical.py`; we additionally
checked the five most negative eigen-directions at a production fit point
(relative difference < 0.5%). The multi-LS + large-bkg sign bug is below.

**DIYMC closure caveat**: the phsp sample must resolve the narrowest
resonance. With a 4.5 MeV φ(1020) and only 10k phsp events, the toy truth
was not the NLL minimum (the fit pushed widths to the bounds) and the
truth-point Hessian had negative eigenvalues; using the same toy data
with 200k phsp, the BFGS minimum (ΔNLL=4.3) had a PD Hessian and θ back
at truth. That is MC normalization resolution, not a `getHessian` bug.

## DIYMC: flat phase space + amplitude-weighted MC

### Flat phase space (P1)

`PhaseSpaceGenerator` only reads `Particles` / `Data.order` / `DecayChains` from
config; it **does not open data/phsp files and does not need a GPU**:

```python
import torch, ctpwa

gen = ctpwa.PhaseSpaceGenerator("config.yml")   # or mother="Jpsi"
flat = gen.generate(100_000, seed=42)            # [N, n_final, 4], (E, px, py, pz)
gen.write(flat, "flat.dat")                      # or "flat.root"
```

- Flat phase space: GENBOD events are accepted with probability `wt` (`u < wt`),
  not simply all `Generate()` events.
- Final-state order = `Data.order`; the mother is at rest by default.
- `gen.generateWeighted(n, seed)` returns `(events, weights)`.

### Amplitude-weighted MC (P2/P3)

From an initialized `analysis` and fitted parameters:

```python
res = ana.generateMC(
    10_000, params, seed=42,
    mode="unweighted",   # or "weighted"
    batch=50_000,        # flat events per batch
)
print(res["stats"])
ana.writeMC(res["events"], "mc.root")   # pass weights=res["weights"] if weighted
```

- `unweighted`: flat phase space → `I=|M|²` → `u*Imax < I` (equal-weight sample).
- `weighted`: all flat events plus their `I` weights (no rejection).
- `max_weight` optionally seeds `Imax`; `None` uses a pilot batch and raises it
  adaptively, re-checking previously accepted events with the saved `u`.
  `max_trials` bounds the number of flat events (safety).
- `getIntensity(momenta, params)` is the low-level primitive: `momenta` is CPU
  `float64 [N, n_final, 4]`; returns CPU `float64 [N]` (arbitrary normalization,
  only ratios are meaningful).
- Both use a **local `AmpCalc`**, so the fit state is untouched (`getNLL` is
  unchanged before/after); works on both CUDA and CPU backends.

## Performance (RTX 5060, 500k events)

- fwd+bwd (NLL+gradient): ~18 ms
- Hessian: ~0.5 s

## License

MIT License
## 运行期精度档位（precision: auto | hybrid | float | double）

业务复数类型：`include/PrecCpx.h` 的 PrecMode + ComplexType.h（模板化双实例路线）。

| `precision:` | 档位 | A 表 | d_dF 表 | 计算 | 相对 double 的偏差 | hessian |
|---|---|---|---|---|---|---|
| `double` | 全 double | 16B | 16B | double | 基准 | double |
| `auto`(缺省)/`hybrid` | float-A 混合 | 8B | 16B | double | NLL ~1e-8 | double(A 上转) |
| `float` | 全 float 档 | 8B | 8B | 共振梯度等 float（F 求值仍 double，见注） | NLL ~1e-9、θ梯度 ~1e-5 | hybrid 同款 ~1e-9 |

注：共振态传播子 F(θ) 符号求值当前为 double（解释器/JIT）；float 档以 double 求值后窄化写出 A/dF（float2）。float 构建（CTPWA_COMPLEX=float）下 float/hybrid/auto 为原生 float，double 报错。

**推荐工作流（初解 float → 精测 double）**：同一 config 只改一行 `precision:`；params 恒 float64，脚本无需分支；float 阶段解出的参数直接作 double 阶段初值（偏差 ~1e-5，double 会快速收敛）。Float 档能力全景（Batch1-3，61da042 起）：NLL+梯度（free-θ/fixed-θ）、getHessian、writeResult、FitFractions/Efficiency 全部可用。

**逃生阀/对照**：CTPWA_KEEP_SLAMP=1（关 slamp 最小保存）、CTPWA_HESS_CHUNK（vv 上转窗口）、CTPWA_SL_BATCH_MB（SL scratch 批上限）、CTPWA_NO_JIT、CTPWA_PROF。**测试**：tests/test_float_mode.py（Float 档行为，需手动同步集群）、test_precision.py 的 hybrid = 原 precision:float 混合语义。
