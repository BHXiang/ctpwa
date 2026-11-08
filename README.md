# CTPWA

[English](README.en.md) | **简体中文**

**CTPWA（CUDA-Torch Partial Wave Analysis）是一个面向高能物理的协变张量分波分析工具包。**

它通过 YAML 配置定义粒子、衰变链和共振态，根据实验数据与相空间样本构建振幅，
计算负对数似然（NLL）、梯度和 Hessian，并输出拟合投影、分波权重、干涉矩阵及
Monte Carlo 样本。核心计算由 C++/CUDA 实现，通过 PyTorch 提供 Python 接口。

同一个 wheel 同时包含 CUDA 和 CPU 后端：有可用 GPU 时使用 GPU，否则自动回退到
CPU 并打印原因，便于在本地、集群和 CI 环境中使用同一套分析配置。

## 主要功能

- **协变张量振幅**：两体衰变、级联衰变链、自旋-轨道耦合、势垒因子及全同粒子处理。
- **共振态模型**：支持 BWR、BW、Flatté、Gounaris–Sakurai、ONE、Interp 和 Custom。
- **拟合计算**：NLL、PyTorch autograd 梯度、Hessian、fit fraction 和效率修正。
- **CUDA 与多 GPU**：振幅、似然及结果投影的核心计算可在 GPU 上并行执行。
- **CPU 后端**：使用 OpenMP，不依赖 CUDA GPU，适合调试、小样本和无卡环境。
- **结果输出**：保存 ROOT 直方图、逐事件权重、指定波对的干涉项和末态四动量。
- **Monte Carlo**：生成平坦多体相空间，以及按当前振幅加权的 MC 样本。
- **大样本支持**：提供 phsp 流式模式和多种精度档位，降低显存占用。

## 环境要求

| 组件 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Linux x86_64，glibc ≥ 2.34 | 适用于 AlmaLinux/RHEL 9、Ubuntu 22.04 等 |
| Python | 3.12 | 当前 PyPI wheel 的构建环境 |
| PyTorch | 2.12+ | 发布流程使用 `2.12.0+cu126` |
| ROOT | 6.26+ | 数据读取和结果输出；使用前加载 `thisroot.sh` |
| NVIDIA GPU | 可选，sm_70 及以上 | 无 GPU 时可使用 CPU 后端 |
| CUDA Toolkit | 源码构建 CUDA 后端时需要 | 发布构建使用 CUDA 12.9 |

> PyPI wheel 已静态链接 yaml-cpp，使用 wheel 时不需要单独安装 yaml-cpp。
> 源码编译则需要 C++17 编译器、yaml-cpp；构建 CUDA 后端还需要 CUDA Toolkit。

## 安装

### 方式一：从 PyPI 安装

先安装 ROOT 并加载环境，然后安装与当前 wheel 匹配的 PyTorch 和 CTPWA：

```bash
source /path/to/root/bin/thisroot.sh

python -m pip install torch==2.12.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install ctpwa
```

GPU 用户还需要与 GPU 架构匹配的 NVIDIA 驱动和 CUDA 运行环境。仅使用 CPU 后端时，
不需要可用的 NVIDIA GPU。

检查安装和实际选择的后端：

```python
import ctpwa

print(ctpwa.__version__)
print(ctpwa.backend())     # "cuda" 或 "cpu"
print(ctpwa.build_info())  # 精度、复数类型和 CUDA 支持等构建信息
```

### 方式二：从源码构建

```bash
git clone https://github.com/BHXiang/CTPWA.git
cd CTPWA

# 示例：使用 conda 提供 ROOT 和 yaml-cpp
conda install -c conda-forge root yaml-cpp=0.8
```

在仓库中编译开发用扩展：

```bash
# 只构建 CPU 后端，不需要 CUDA Toolkit 或 GPU
CTPWA_BACKEND=cpu MAX_JOBS=4 python setup.py build_ext --inplace

# 只构建 CUDA 后端
CTPWA_BACKEND=cuda MAX_JOBS=4 python setup.py build_ext --inplace

# 同时构建两个后端（发布 wheel 时使用）
CTPWA_BACKEND=both MAX_JOBS=4 python setup.py build_ext --inplace
```

`MAX_JOBS` 用于限制 C++/CUDA 并行编译数；内存不足时可改为 `1`。

## 后端选择

```bash
# 默认：探测并优先使用 CUDA，不可用时回退 CPU 并打印原因
export CTPWA_DEVICE=auto

# 强制 CUDA；不可用时直接报错
export CTPWA_DEVICE=cuda

# 强制 CPU
export CTPWA_DEVICE=cpu
```

`CTPWA_DEVICE` 是**运行期**选项，`CTPWA_BACKEND` 是**构建期**选项。CPU 后端适合调试
和小样本；生产规模拟合通常仍建议使用 GPU。

## 快速开始

### 1. 编写分析配置

下面是 `J/ψ → π⁺π⁻π⁰`，经 `ρ(770) → π⁺π⁻` 衰变的最小配置示例：

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

其中 `data` 是实验或模拟数据，`phsp` 是用于拟合归一化的相空间样本；数据文件中的
末态粒子顺序必须与 `Data.order` 一致。完整配置和拟合脚本见 [example](example/)。

### 2. 计算 NLL 和梯度

```python
import torch
import ctpwa

ana = ctpwa.analysis("config.yml")
device = torch.device(ctpwa.backend())

n_vec = ana.getNVector()
n_theta = ana.getNFreeTheta()
params = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64, device=device)

# 固定一个参考耦合以消除整体相位和模长规范自由度
params[0] = 1.0

# 自由共振态参数使用配置中的标称值
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

参数布局为 `[Re(couplings), Im(couplings), resonance_parameters]`。实际拟合可使用
PyTorch 优化器，或参考 [example/fit.py](example/fit.py) 中的完整流程。

### 3. 保存拟合结果

```python
ana.writeResult(params.detach(), "weight_best.root", 0)
```

输出文件包含模型与数据的投影、各分波权重、图例和干涉矩阵，可用于后续绘图与检查。

## 常用 Python 接口

| 接口 | 用途 |
|---|---|
| `ctpwa.analysis(config)` | 读取配置并初始化振幅、数据和设备缓冲区 |
| `getNLL(params)` | 计算负对数似然，支持 PyTorch autograd |
| `getHessian(params)` | 计算耦合及自由共振态参数的 Hessian |
| `getFitFractions(...)` | 计算分波份额 |
| `getEfficiency(...)` | 计算效率修正 |
| `writeResult(...)` | 输出 ROOT 拟合结果、投影和干涉矩阵 |
| `writeInterfResult(...)` | 输出指定波对的逐事件干涉项 |
| `PhaseSpaceGenerator(config)` | 生成平坦多体相空间 |
| `getIntensity(momenta, params)` | 计算任意四动量样本的逐事件强度 |
| `generateMC(...)` / `writeMC(...)` | 生成并保存振幅加权 MC 样本 |

## 大样本与精度

对于大 phsp 样本，可在配置中启用流式模式，避免全部 phsp 振幅驻留显存：

```yaml
Constraints:
  free_phsp_amplitudes: true
```

该模式要求所有共振态参数固定。`precision: auto | hybrid | float | double` 用于控制
振幅存储和计算精度；需要节省显存时可先使用混合精度，最终再使用 double 检查。

## Monte Carlo 样本

```python
# 平坦相空间
mcgen = ctpwa.PhaseSpaceGenerator("config.yml")
events = mcgen.generate(100_000, seed=42)
mcgen.write(events, "flat.root")

# 按当前振幅舍选，生成等权 MC
result = ana.generateMC(10_000, params.detach(), seed=42, mode="unweighted")
ana.writeMC(result["events"], "mc.root")
```

相空间样本必须足以分辨模型中最窄的共振态，否则归一化统计涨落会影响拟合和 Hessian。

## 开发与测试

```bash
# 编译并运行完整测试门禁
bash scripts/run_tests.sh

# 已编译时只运行测试
bash scripts/run_tests.sh --no-build
```

开发约定、双后端实现和调试方法见仓库中的源码及测试。提交发布前应在 CUDA 和 CPU
后端上分别完成相应测试。版本变化记录见 [CHANGELOG.md](CHANGELOG.md)。

## License

MIT License
