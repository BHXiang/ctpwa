# CTPWA

**CUDA + PyTorch 协变张量分波分析（Partial Wave Analysis）工具包**

用于高能物理分波分析：基于协变张量理论构建衰变振幅，
用 CUDA 在 GPU 上并行计算。

---

## 特性

- **协变张量分波**：两体衰变振幅（自旋-轨道耦合、势垒因子、宇称约束），支持级联衰变链
- **GPU 并行**：振幅、SL 振幅、共振态因子全部 CUDA kernel 实现，多 GPU 支持
- **拟合**：NLL + 梯度（解析/AutoDiff）+ Hessian，接入 PyTorch autograd 优化器

## 系统要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Linux x86_64 | glibc ≥ 2.34 | Ubuntu 22.04+ / Debian 12+ |
| NVIDIA GPU | sm_75 及以上 | RTX 20/30/40/50 系、A100/H100 |
| CUDA Toolkit | 13.2 | 运行时需对应驱动 |
| PyTorch | 2.12+（cu132） | 与系统 CUDA 版本匹配 |
| ROOT | 6.26+ | 读取 .root 数据（`source thisroot.sh`） |
| yaml-cpp | 0.9 | config 解析 |

## 安装

### 从 PyPI 安装（免编译）

```bash
# 系统依赖
conda install yaml-cpp 
# ROOT：下载/编译后 source thisroot.sh
# CUDA：安装与 GPU 匹配的驱动 + CUDA 13.2 runtime

# PyTorch（必须与 CUDA 匹配）
pip install torch==2.12.0+cu132 --index-url https://download.pytorch.org/whl/cu132

# CTPWA（manylinux wheel，内置 sm_75~sm_120）
pip install ctpwa
```

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
Particles:
  Jpsi: {J: 1, P: -1, mass: 3.0969}
  Kp:   {J: 0, P: -1, mass: 0.4937}
  Km:   {J: 0, P: -1, mass: 0.4937}
  eta:  {J: 0, P: -1, mass: 0.5478}

Data:
  order: [Kp, Km, eta]
  data: [ROOT, "file/data.root", TTree, TBranch1, TBranch2, ...]
  phsp: [ROOT, "file/phsp.root", TTree, TBranch1, TBranch2, ...]
  bkg:  [ROOT, "file/bkg.root", TTree, TBranch1, TBranch2, ...]
  bkg_weights: [0.8]

DecayChains:
  chain1:
    Jpsi: [[eta, R_KK], [Kp, R_Keta], [Km, R_Keta]]
    R_KK: [Kp, Km]
    intermediates:
      R_KK:  [[J: 1, P: -1]: [phi1680]]
      R_Keta:[[J: 1, P: -1]: [K1_1410]]

Resonances:
  phi1680: {J: 1, P: -1, model: BWR, parameters: [1.66, 0.125], free: [0, 1]}
  K1_1410: {J: 1, P: -1, model: BWR, parameters: [1.402, 0.149], free: [0, 1]}

Constraints:
  trans: [[R_Keta_0, R_Keta_1]: -1]
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
| Linux x86_64 | glibc ≥ 2.34 | Ubuntu 22.04+ / Debian 12+ |
| NVIDIA GPU | sm_75+ | RTX 20/30/40/50, A100/H100 |
| CUDA Toolkit | 13.2 | needed to compile; matching driver at runtime |
| PyTorch | 2.12+ (cu132) | must match system CUDA |
| ROOT | 6.26+ | for reading .root data (`source thisroot.sh`) |
| yaml-cpp | 0.9 | config parsing |

## Installation

### From PyPI (prebuilt wheel)

```bash
# system deps
conda install yaml-cpp
# ROOT: install and `source thisroot.sh`
# CUDA: driver + CUDA 13.2 runtime

pip install torch==2.12.0+cu132 --index-url https://download.pytorch.org/whl/cu132
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
Particles:
  Jpsi: {J: 1, P: -1, mass: 3.0969}
  Kp:   {J: 0, P: -1, mass: 0.4937}
  Km:   {J: 0, P: -1, mass: 0.4937}
  eta:  {J: 0, P: -1, mass: 0.5478}

Data:
  order: [Kp, Km, eta]
  data: [ROOT, "file/data.root", TTree, TBranch1, TBranch2, ...]
  phsp: [ROOT, "file/phsp.root", TTree, TBranch1, TBranch2, ...]
  bkg:  [ROOT, "file/bkg.root", TTree, TBranch1, TBranch2, ...]
  bkg_weights: [0.8]

DecayChains:
  chain1:
    Jpsi: [[eta, R_KK], [Kp, R_Keta], [Km, R_Keta]]
    R_KK: [Kp, Km]
    intermediates:
      R_KK:  [[J: 1, P: -1]: [phi1680]]
      R_Keta:[[J: 1, P: -1]: [K1_1410]]

Resonances:
  phi1680: {J: 1, P: -1, model: BWR, parameters: [1.66, 0.125], free: [0, 1]}
  K1_1410: {J: 1, P: -1, model: BWR, parameters: [1.402, 0.149], free: [0, 1]}

Constraints:
  trans: [[R_Keta_0, R_Keta_1]: -1]
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

## Performance (RTX 5060, 500k events)

- fwd+bwd (NLL+gradient): ~18 ms
- Hessian: ~0.5 s

## License

MIT License
