# `example` 使用说明（`fit.py` 为主）

`example/fit.py` 是 ctpwa 的拟合驱动脚本：读取 `config.yml` → 构建振幅 → 多次优化 →
Hessian / 参数误差 → 保存结果与摘要。

本文以 **`fit.py` 为主**：§1–§7 讲运行、参数、输出，§8 讲参数削减/误差截断/边界诊断/
`--fit-mode`，§9 讲后端选择，§11 讲干涉与幅度控制三参数。**不属于 `fit.py` 用法的脚本**——
画图 `plot.py`、两个 MC 生成脚本（`gen_phsp_mc.py` / `gen_toy_mc.py`）和读干涉矩阵的
`read_interference.py`——统一放在 **§10「其它脚本」**（§10.1–§10.4）。
各文件的变更历史见 git 提交记录。

---

## 目录

- [0. TL;DR](#0-tldr)
- [1. 运行前提](#1-运行前提)
- [2. 快速开始](#2-快速开始)
- [3. 优化器区别（重点）](#3-优化器区别重点)
  - [3.1 `reparam`（默认，推荐）](#31-reparam默认推荐)
  - [3.2 `projected`](#32-projected)
  - [3.3 `lbfgs`（legacy，仅 A/B 对照）](#33-lbfgslegacy仅-ab-对照)
  - [3.4 对比（示例数据，`--runs`/`--niter` 见备注）](#34-对比示例数据--runs--niter-见备注)
  - [3.5 推荐工作流（随机多起点 → 取最优 → polish）](#35-推荐工作流随机多起点--取最优--polish)
- [4. 参数总表](#4-参数总表)
  - [配置 / 日志](#配置--日志)
  - [核心运行参数](#核心运行参数)
  - [约束 / 数值稳定](#约束--数值稳定)
  - [误差估计（Hessian 非正定回退）](#误差估计hessian-非正定回退)
  - [Warm start](#warm-start)
  - [Polish（二阶抛光）](#polish二阶抛光)
  - [权重文件](#权重文件)
  - [Checkpoint / Resume](#checkpoint--resume)
  - [输出](#输出)
  - [参数化 / 参数削减 / 边界诊断（见 §8）](#参数化--参数削减--边界诊断见-8)
- [5. 环境变量](#5-环境变量)
- [6. 输出文件](#6-输出文件)
- [7. 已知问题与备注](#7-已知问题与备注)
- [8. 参数化 / 参数削减 / 误差截断 / 边界诊断](#8-参数化--参数削减--误差截断--边界诊断)
  - [8.1 `--fit-mode {0,1}`：参数化](#81---fit-mode-01参数化)
  - [8.2 `--reduce-params`：有效秩降维 + 降维重拟合](#82---reduce-params有效秩降维--降维重拟合)
  - [8.3 `--no-errors-truncate`：非正定误差口径](#83---no-errors-truncate非正定误差口径)
  - [8.4 `--bound-margin`：贴边参数诊断](#84---bound-margin贴边参数诊断)
- [9. 后端选择（CPU / GPU）与 `CTPWA_DEVICE`（ctpwa ≥ 0.3.15）](#9-后端选择cpu--gpu与-ctpwa_devicectpwa--0315)
- [10. 其它脚本（不属于 `fit.py` 用法）](#10-其它脚本不属于-fitpy-用法)
  - [10.1 `plot.py` 使用说明](#101-plotpy-使用说明)
  - [10.2 `gen_phsp_mc.py`：平坦相空间（P1，**不需要 GPU**）](#102-gen_phsp_mcpy平坦相空间p1不需要-gpu)
  - [10.3 `gen_toy_mc.py`：振幅加权 toy → pseudo-data（P2/P3，**需要 backend**）](#103-gen_toy_mcpy振幅加权-toy--pseudo-datap2p3需要-backend)
  - [10.4 `read_interference.py`：读取 `weight_best.root` 里的干涉矩阵](#104-read_interferencepy读取-weight_bestroot-里的干涉矩阵)
- [11. 干涉与幅度控制三参数](#11-干涉与幅度控制三参数)
  - [11.1 问题：干涉简并方向](#111-问题干涉简并方向)
  - [11.2 三个参数的分工](#112-三个参数的分工)
  - [11.3 由预期干涉强度反推 `--amp-lambda`](#113-由预期干涉强度反推---amp-lambda)
  - [11.4 推荐流程](#114-推荐流程)
  - [11.5 复现旧结果（`--amp-lambda`）](#115-复现旧结果--amp-lambda)

---

## 0. TL;DR

```bash
# 在放有 config.yml 的目录里运行
python -u fit.py --config config.yml --runs 10 --niter 3000            # 默认优化器 reparam
python -u fit.py --optimizer projected --runs 2 --niter 3000           # 对照（慢）
python -u fit.py --optimizer reparam --amp-max 1000 --amp-lambda 1e-3  # 可选：开启幅度罚项（默认 0=关闭，见 §11）
```

---

## 1. 运行前提

- 需要 **ctpwa 包**与 **NVIDIA GPU**（默认）。本版本要求 **ctpwa ≥ 0.3.26**，各功能对应的最低版本：
  - `≥ 0.3.10`：`ctpwa.analysis(config, fit_mode)`（`--fit-mode`）；
  - `≥ 0.3.15`：新门面 `import ctpwa`（单 wheel 双后端）+ `build_info()`/`backend()`；
  - `≥ 0.3.17`：`ctpwa.diagnostics`（可行子空间 PD 判据，见 §7/§8）；
  - `≥ 0.3.22`：`writeResult` 往 `weight_best.root` 写 `interference_rec`/`interference_gen`/
    `*_err`/`integrals`，配 `read_interference.py`（§10.4）；
  - `≥ 0.3.24`：干涉矩阵按分波**合并 LS**（维度 = 分波数）；
  - `≥ 0.3.25`：多 LS 分波（`nSL > npart`）时 `interference_rec_err` 不再是 0；
  - `≥ 0.3.26`：Hessian 病态时 `writeResult` 不被 `eigvalsh` 异常中断；耦合误差按**完整自由
    Hessian** 的耦合块（θ 边际化）；流式 `phsp` 的 Gram 归一化修正。
  具体安装版本用 `python -c "import ctpwa; print(ctpwa.__version__)"` 查看；`≥ 0.3.27` 另附带
  双卡内存预检、大 n Hessian、大模型 SL 峰值与 BWR `q0=0` 的稳定性修复，推荐使用。
- **从源码编译**：`setup.py` 与上游一致（`-std=c++17` + 可选的 `CTPWA_GLIBCXX_ASSERTIONS=1`）。
  ⚠ 若用 torch ≥ 2.13 编译，其 C++ 头文件需要 C++20——此时需**在本机编译目录**临时把三处
  `-std=c++17` 改成 `-std=c++20`（不要提交）。
- **CPU 回退（`--cpu`）**：`fit.py` **默认要求 GPU**（检测不到可用 CUDA 就直接退出）；
  如确需在 CPU 上跑（慢 20–60 倍，仅适合小样本/调试），加 `--cpu` 强制 CPU 后端——
  它会在 `import ctpwa` **之前**把 `CTPWA_DEVICE` 置为 `cpu`（覆盖已有 env 并提示）。
  只设 `CTPWA_DEVICE=cpu`（不加 `--cpu`）也算显式要 CPU、同样放行。详见 §9。
- **后端自动选择**（ctpwa ≥ 0.3.15）：`import ctpwa` 时会真跑一次 kernel 探测，
  GPU 可用则用 `_cuda`，否则**回退 CPU**（慢 20–60 倍，会打印警告）。
  用 `CTPWA_DEVICE=auto|cuda|cpu` 控制：**显式 `cuda` 时探测失败直接报错**（推荐在
  slurm 作业里设，避免静默退 CPU）。详见 §9。当前后端在启动 dump 里打印
  （`backend=… build_info=…`）。
- **配置由 `--config` 指定**（默认 `./config.yml`）。`fit.py` 在 **import 时**用
  `_config_path_from_argv()` 从 `sys.argv` 取出 `--config`（转绝对路径），**并在构造
  `ctpwa.analysis()` 之前 `chdir` 到 config 目录**，因此 `--config /path/to/other.yml`
  **真正生效**（无需 cd 到配置目录）。
- **路径口径（两条规则）**：
  - **config 内**的相对路径（`data` / `phsp` / `phsp_truth` / `bkg` / `bkg_weights`，以及
    `Resonances` / `Constraints.chains(_exact)` 引用的外部文件）= **相对 config 文件所在目录**；
  - **命令行**给的路径（`--config`、`--output-dir`、`--warm-start`、`--resume` 指向的
    `checkpoint.pt`）= **相对你调用 `fit.py` 的目录**。
  启动 dump 会打印 `output_dir=` / `warm_start=` 以及 config 里各数据文件的**解析结果**
  （绝对路径 + 是否存在，缺失会标 `← 不存在!`），便于一眼核对。
- 建议用 `python -u`（无缓冲）以便实时看日志。
- ⚠ **不要在仓库根目录 `import ctpwa`**：仓库根自带上游的 `ctpwa/` 门面包（源码目录，
  没有 `_cuda.so`/`_cpu.so`），在 `python -m pytest`/`python -c`（cwd 进 `sys.path`）时会
  **遮蔽**已安装的 wheel → 报“没有可用的 ctpwa 后端”。**不在仓库根跑**的作业目录不受影响。

---

## 2. 快速开始

```bash
# 正式拟合（默认优化器 reparam）
python -u fit.py --runs 10 --niter 3000

# 早期快速调试
python -u fit.py --runs 3 --niter 100 --no-polish

# 只画/导出部分波
python -u fit.py --runs 1 --niter 1000 --waves 6,7

# warm start 继续优化
python -u fit.py --runs 5 --niter 1000 --warm-start results/best_params.pt

# 断点续跑
python -u fit.py --runs 20 --niter 3000 --resume
```

---

## 3. 优化器区别（重点）

用 `--optimizer {reparam,projected,lbfgs}` 选择，默认 **`reparam`**。
三者都在**同一个物理参数空间**上优化；下游的 Hessian / polish / 误差 / 输出与优化器无关。

### 3.1 `reparam`（默认，推荐）

**做法**：把有界/半有界参数映射到无约束空间 `u`，用 `torch.optim.LBFGS(strong_wolfe)` 优化：

- 耦合：极坐标 `amp = amp_max · sigmoid(u_amp)`（>0，小幅度区 ≈ 对数幅度，接近
  `amp_max` 时 `damp/du→0` 自限幅）、相位 `phi = u_phi` 自由无界。
- 共振态参数：`theta = lo + (hi-lo)·sigmoid(u)`，严格落在 `(lo, hi)`。
- 固定参考 `re_0=1, im_0=0` 不进入 `u`。
- 优化期可选**幅度罚项** `loss = NLL + λ·Σ|A_i|²`（S2，只进优化 loss；`nll_history`、
  报告 NLL、Hessian **仍用真实 NLL**），抑制“同共振不同 LS 两波相互抵消”的简并方向把
  幅度吹大。**默认关闭（`λ=0`）**；是否开启、取多大见 **§11**。
- polish 阶段按“当前 |A|”限制耦合步长、并对候选点做 `amp_max` 硬帽（S4）。

**优点**：收敛快（典型 ~1000–1800 次求值、~300–500 s/run）、NLL 好、振幅可控。
**代价**：开启罚项（`λ>0`）时 `λ` 是对耦合幅度的弱先验，会轻微改变优化落点
（`λ=1e-4` 影响可忽略，量级见 §11）。

相关参数：`--amp-max`、`--amp-lambda`、`--lr`、`--tol-grad`、`--tol-change`、`--history-size`。

### 3.2 `projected`

**做法**：自研的盒约束 L-BFGS——投影梯度 KKT 停机 + 活跃集剔除 + 可行 Armijo 线搜索，
在物理参数空间直接优化，边界由算法处理（`honest=True`，不 clamp、不清零梯度）。

**注意**：它用绝对投影梯度 `‖P(x-g)-x‖∞ ≤ tol_grad(默认 1e-5)` 作唯一停机判据；在
本模型上该条件**长期达不到**，于是每次优化都会跑到 `niter`（`max-iter`），实测比 reparam
慢约 2.6×，且 best NLL 更差。

相关参数：`--vmax`（耦合盒约束上界）、`--tol-grad`（即 gtol）、`--opt-verbose`
（每轮打印 `|pg|/active/ΔNLL`）。

### 3.3 `lbfgs`（legacy，仅 A/B 对照）

**做法**：`torch.optim.LBFGS(strong_wolfe)` + 每步把参数 `clamp` 回盒内 + 把“指向界外”
的边界梯度清零（`--no-project` 可关）。因为在边界处会**伪收敛**（参数贴边、梯度被清零
导致收敛判据被满足），已不作为默认，仅用于对照。

相关参数：`--vmax`、`--no-project`、`--lr`、`--tol-grad`、`--tol-change`、`--history-size`。

### 3.4 对比（示例数据，`--runs`/`--niter` 见备注）

| 维度 | `reparam`（默认） | `projected` | `lbfgs`（legacy） |
|---|---|---|---|
| 边界处理 | 重参数化软墙 | 投影梯度 + 活跃集（盒约束） | clamp + 边界梯度清零 |
| 停机判据 | 无约束梯度/变化量（u 空间） | 投影梯度 KKT | 无约束梯度/变化量 |
| 典型停机状态 | `reparam-max-iter` / `reparam-tol-stop`※ | 多为 `max-iter` | 伪收敛 |
| 每次求值次数（中位）* | ~1400 | ~3300 | ~2600 |
| 每次耗时（中位）* | ~400 s | ~900 s | ~780 s |
| 20-run best NLL* | **-59190.17** | -58940.34 | — |

\* 实测自 `--niter 3000`：reparam/projected 数据来自 20/5 个 run，lbfgs 来自 2 个 run；
具体数据集为一次 3 体强子末态分析，仅供相对比较。

※ `reparam-tol-stop` = torch LBFGS 自身的容差（`--tol-grad/--tol-change`）停机，
**不等于**物理空间 KKT 收敛：实测约半数随机起点会停在坏盆地（NLL 可比正常解差
100~700）。多起点流程（best-of-N）下这没问题；**单次结果不要只看 status 判定成功**。

**如何选**：默认用 `reparam`。做 A/B 或复现历史行为时用 `--optimizer projected` 或
`--optimizer lbfgs`。若 reparam 偶发振幅偏大（贴到 `--amp-max` 软墙帽），可开启幅度罚项
`--amp-lambda`（建议从 `1e-3` 起，取值方法见 **§11**）或调小 `--amp-max`。

### 3.5 推荐工作流（随机多起点 → 取最优 → polish）

> 本节数值来自上游示例配置 `Jpsi2KKeta`（其 polish 后 `PD=True`）；另一配置（3 体强子末态）
> 的同类对比见 §3.4（该配置恒 `PD=False`）。不同配置/数据不可混用。

本分析场景是"随机初值跑很多次、取最优、再对最优做 polish"，此时判据是
**单位时间内的 best-of-N**，不是单起点可靠性：

```bash
# 推荐：12 个随机起点（默认优化器已是 reparam、默认 niter=500）
# main() 会自动对 best 做 polish_damped_newton → 给出 PD 与参数误差
python fit.py --runs 12

# 兜底/复核：参数最优普遍贴 free_range 边界、或要与历史结果对齐时
python fit.py --optimizer projected --runs 5
```

**默认 niter 保持 500 不动**——不同分波分析的收敛需求不同（本仓库不止一个分波）。
对本分析（`Jpsi2KKeta`）实测 `--niter 300` 已足够，可作为加速选项：`polish_damped_newton`
（最多 200 步缩放阻尼 LM + 精确 Hessian）是很强的局部精修器，L-BFGS 只需把起点送进
正确盆地，收尾交给 polish。两组独立 6-run 系综中，`--niter 300` 的 pre-polish best 只有
−3443 / −3413，**polish 后都收敛到 −3463.9027（四位小数一致）、PD=True**，单组拟合
190~204 s（+36 s polish + 33 s init ≈ **4 分钟**）。对照：

| 工作流 | 总耗时 | polish 后 best | PD |
|---|---|---|---|
| **reparam `--niter 300 --runs 6`** | **≈4 分钟** | **−3463.90** | **True** |
| reparam `--niter 1000 --runs 6` | ≈7 分钟 | −3463.81 | True |
| projected `--niter 1000 --runs 6` | ≈30 分钟 | −3464.87 | False |

后两者与第一行的差异（0.09 / 0.97 NLL）都远小于系综混沌（σ≈9）与显著性阈值 3，
即 **4 分钟的 reparam 短跑 + polish 与 30 分钟的 projected 结果统计等价**。

实测（`Jpsi2KKeta` 29 分波 / 19 耦合 / 10 自由共振参数，同 6 起点、`--niter 1000`）：

| | reparam | projected |
|---|---|---|
| 单起点耗时（中位） | **69 s** | 281 s |
| 单起点求值数（中位） | **578** | 3193 |
| 好起点产出率（距最好值 <10 NLL） | **0.0082 /s** | 0.0022 /s |
| 每个好起点平均耗时 | **122 s** | 452 s |
| best-of-6 | −3463.81 | −3464.87（差 1.06，在系综混沌 σ≈9 内） |
| best polish 后 | **PD=True**（可出参数误差） | PD=False |

要点：
1. **多起点会饱和**：约 5~8 个起点就摸到最好的盆地，`--runs 12` 足够，不需要几百次。
2. **不要把 reparam 的最优解再交给 projected 精修**：实测无效（`no-descent`，0 改进）
   或更差（中段交接在 3/3 起点上差 12~373 NLL）——两者停在不同局部极小。
   polish 本来就是两条路共用的同一个 `polish_damped_newton`。
3. **`--amp-max`（默认 1000）**要确认大于该分析的最大 `|A|`，否则会撞软墙；
   `--amp-lambda` 是只进优化目标的幅度罚项，报告 NLL 仍为真实值，**默认 0（关闭）**，
   需要保险丝时再开（取值方法见 §11）。
4. **只抛 best 还是逐 run 都抛**：默认只对 best 做一次 polish（省时间）；若想在多个极小之间
   比较"抛光后的 NLL 与正定性"（例如宁可要一个 PD 解、不只认 NLL 最低），用 **`--polish-all`**
   对每个 run 各抛一次，并写 `polish_history.json` + `all_params/run_XX.pt`，最后按
   **正定优先、NLL 次之**重排 best（见 §4 Polish 表、§6 输出文件）。

---

## 4. 参数总表

优先级：**CLI > `FIT_*` 环境变量 > 默认值**。默认值以 `resolve_args()` 为准。

### 配置 / 日志
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--config` | `config.yml` 路径（见 §1 注意事项） | `config.yml` | — |
| `--cpu` | 强制 CPU 后端拟合（慢 20–60 倍，见 §9）；不加则**默认要求 GPU**，检测不到就退出 | False | （看 `CTPWA_DEVICE`） |
| `-v`, `-vv` | 提高日志详细度（info / debug） | 0 | — |
| `-q` | 安静模式（只输出最终结果） | False | — |

### 核心运行参数
| 参数 | 说明 | 默认 | 环境变量 | 适用优化器 |
|---|---|---|---|---|
| `--runs` | 优化运行次数（多起点取最优） | 10 | `FIT_RUNS` | 全部 |
| `--niter` | 最大迭代次数 | 500 | `FIT_NITER` | 全部 |
| `--lr` | LBFGS 初始步长 | 0.3 | `FIT_LR` | reparam / lbfgs |
| `--tol-grad` | 梯度收敛阈值 | 1e-5 | `FIT_TOL_GRAD` | 全部 |
| `--tol-change` | 参数变化收敛阈值 | 1e-5 | `FIT_TOL_CHANGE` | reparam / lbfgs |
| `--history-size` | LBFGS history 大小 | 200 | `FIT_HISTORY_SIZE` | 全部 |
| `--seed` | 随机初值 base seed（run i 用 `seed+i`）。**默认取当前 Unix 时间戳**→每次随机；要复现就用日志 `[seed]` 行打印的值 | 当前时间 | `FIT_SEED` | 全部 |

### 约束 / 数值稳定
| 参数 | 说明 | 默认 | 环境变量 | 适用优化器 |
|---|---|---|---|---|
| `--vmax` | 耦合幅度硬上界（`vmax`，防发散的保险丝，见 §11） | 10000 | `FIT_VMAX` | projected / lbfgs |
| `--amp-max` | reparam 幅度软墙上限 `amp = amp_max·sigmoid(u)`（见 §11） | 1000 | `FIT_AMP_MAX` | reparam |
| `--amp-lambda` | reparam 幅度罚项 `λ·ΣA²`（只进优化 loss；**0=关闭，默认关**；取值见 §11） | 0 | `FIT_AMP_LAMBDA` | reparam |
| `--no-project` | 关闭 legacy 路径的边界梯度清零 | False | `FIT_PROJECT=0` | lbfgs |
| `--optimizer` | 优化器：`reparam`/`projected`/`lbfgs` | `reparam` | `FIT_OPTIMIZER` | — |
| `--opt-verbose` | 每轮打印 projected 的 `pg/active/ΔNLL` | False | `FIT_OPT_VERBOSE=1` | projected |
| `--ff-only` | 只算拟合分数：跳过拟合与 polish，用最佳参数（`--warm-start` 或 `<output-dir>/best_params.pt`） | False | `FIT_FF_ONLY=1` | — |
| `--cal-ff` | 是否输出拟合分数（`True`/`False` 带值；默认开）。**从 `weight_best.root` 的干涉矩阵读**（相干分母 `F_i`，与 ROOT 一致）；读不到矩阵时**只提示、不输出**（不再回退计算） | `True` | — | — |
| `--cal-eff` | 是否计算分波效率（默认关；大 `phsp`/`phsp_truth` 上可能 C++ 崩溃，建议配小样本 config + `--ff-only`） | `False` | — | — |

### 误差估计（Hessian 非正定回退）
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--err-mode` | `auto`=PD 直接求逆，否则**有效秩截断**（见下）；`strict`=原行为(非PD不给)；`pinv`=伪逆(丢 `λ<τλmax` 方向)；`psd`=把 `λ` 截到 `τλmax` 后求逆(更保守) | `reparam`→`auto`，其它→`strict` | `FIT_ERR_MODE` |
| `--err-tau` | 相对阈值 `τ`（`λ<τ·λmax` 视为平坦/不可测） | `1e-6` | `FIT_ERR_TAU` |
| `--no-errors-truncate` | 关闭“非正定时按有效秩截断传播误差”，退回 `pinv` | 开启截断 | `FIT_ERRORS_TRUNCATE=0` |

> **默认策略**：显式 `--err-mode` > `FIT_ERR_MODE` > **优化器相关**（`reparam`→`auto`，
> 其它→`strict`）。即：只有选 `reparam` 时默认才自动回退给误差；`projected`/`lbfgs`
> 默认与上游一致（非正定就不给误差）。
> 当精确 Hessian 近奇异/非正定（本模型常见，存在 ~20–30 维近平坦子空间）时，`auto`
> 保留 `λ_i > τ·λmax` 的方向做误差传播 `σ²_m = Σ_{i∈keep} E[m,i]²/λ_i`：
> **每个参数仍有有限误差**（只有被冻结的线性组合没有），日志打印保留维数/λmin/cond 与
> 冻结方向数（`--reduce-params` 可看这些组合的物理内容）；
> 输出会标注这是**秩亏下的可行估计，非严格统计误差**。
> `--no-errors-truncate` 退回伪逆（`σ=0` 的“无误差”口径）；
> `--err-mode strict` 恢复“非正定就不给误差”的行为。

### Warm start
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--warm-start [路径]` | 用已收敛解作为 run 0 初值；不带路径时自动用 `results/best_params.pt` | None | `FIT_WARM=1` |

### Polish（二阶抛光）
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--polish` / `--no-polish` | 是否在最佳点做 damped-Newton 抛光 | 开启 | `FIT_POLISH` |
| `--polish-all` | 对**每个** run 各做一次抛光（不是只抛 best）；抛光后按「正定优先、NLL 次之」重排 best，并写 `polish_history.json` 与 `all_params/run_XX.pt`。配 `--no-polish` 时该开关无效（会 warning） | 关闭 | `FIT_POLISH_ALL` |

### 权重文件
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--waves` | 导出权重的分波下标子集，逗号分隔（如 `6,7`） | 全部 | `FIT_WAVES` |
| `--event-data` | 权重 TTree 额外包含末态四动量 | False | `FIT_EVENT_DATA=1` |

### Checkpoint / Resume
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--checkpoint-interval` | 每 N 轮保存一次 checkpoint | 1 | `FIT_CHECKPOINT_INTERVAL` |
| `--resume` | 从 `output-dir/checkpoint.pt` 续跑 | False | — |

### 输出
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--output-dir` | 结果输出目录 | `results` | — |

### 参数化 / 参数削减 / 边界诊断（见 §8）
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--fit-mode` | `0`=FREEPARAMS chain×step；`1`=VSPACE 逐振幅（Hessian 二阶项恒为 0 → 精确）。**只能在构造期决定**，故由 import 期读取 | 0 | `FIT_MODE` |
| `--reduce-params` | 拟合后用 Hessian 有效秩降维并**降维重拟合**，输出冻结组合与全参数误差 | False | `FIT_REDUCE_PARAMS=1` |
| `--rank-tau` | 有效秩门槛 `λ_i > tau·λmax` | `1e-3` | `FIT_RANK_TAU` |
| `--rank-iters` | 降维重拟合的 LBFGS 迭代上限 | 300 | `FIT_RANK_ITERS` |
| `--rank-fixed-point` | 降维基重算次数（基依赖拟合点） | 1 | `FIT_RANK_FIXED_POINT` |
| `--rank-scale` | 有效秩尺度基准：`max`/`median`/`p90`（病态/跑飞解用 `median`） | `max` | `FIT_RANK_SCALE` |
| `--bound-margin` | θ 贴边告警阈值（占 `free_range` 宽度比例） | 0.02 | `FIT_BOUND_MARGIN` |

---

## 5. 环境变量

每个 CLI 参数都有同名 `FIT_*` fallback，优先级 `CLI > 环境变量 > 默认值`：

```
FIT_RUNS  FIT_NITER  FIT_LR  FIT_TOL_GRAD  FIT_TOL_CHANGE  FIT_HISTORY_SIZE
FIT_VMAX  FIT_AMP_MAX  FIT_AMP_LAMBDA  FIT_PROJECT  FIT_OPTIMIZER  FIT_OPT_VERBOSE
FIT_ERR_MODE  FIT_ERR_TAU  FIT_POLISH  FIT_POLISH_ALL  FIT_WARM  FIT_WAVES  FIT_EVENT_DATA
FIT_CHECKPOINT_INTERVAL  FIT_SEED
FIT_MODE  FIT_REDUCE_PARAMS  FIT_RANK_TAU  FIT_RANK_ITERS  FIT_RANK_FIXED_POINT
FIT_RANK_SCALE  FIT_ERRORS_TRUNCATE  FIT_BOUND_MARGIN
# 注意：CTPWA_DEVICE / CTPWA_QUIET 属于 ctpwa 包（不是 fit.py 的参数），见 §9：
CTPWA_DEVICE=auto|cuda|cpu   CTPWA_QUIET=1
# 诊断 / 复现实验 / projected A/B 开关（默认=当前行为，一般不用设）：
FIT_OPT_PROF  FIT_DETERMINISTIC  FIT_OPT_M  FIT_OPT_TWOLOOP  FIT_OPT_HONEST
FIT_OPT_LEGACY_SIGN  FIT_OPT_POLISH_STEPS
```

例：`FIT_OPTIMIZER=reparam FIT_AMP_LAMBDA=1e-3 python -u fit.py --runs 20 --niter 3000`
（`FIT_AMP_LAMBDA` 默认 `0`=关闭，取值方法见 §11）

- `FIT_OPT_PROF=1`：打印**每次 run 的求值次数与耗时占比**（`[pLBFGS-prof]`/`[reparam-prof]`，
  含 `ms/eval`、`n_iter`、停机状态），用于“求值次数 × 单次成本”的归因。
- `FIT_DETERMINISTIC=1`：启用确定性设置（`torch.use_deterministic_algorithms(warn_only)` +
  cudnn 标志 + seed），降低不确定性；**不保证前向 NLL 逐位可复现**，仅用于对比实验。
- `FIT_OPT_M` / `FIT_OPT_TWOLOOP` / `FIT_OPT_HONEST` / `FIT_OPT_LEGACY_SIGN` /
  `FIT_OPT_POLISH_STEPS`：`projected` 路径的曲率历史/实现/停机 A/B 开关；默认即当前行为。

---

## 6. 输出文件

`--output-dir`（默认 `results/`）下：

| 文件 | 内容 |
|---|---|
| `parameters.txt` | 每个 run 的耦合（实/虚部、幅度、相位）与共振态参数 |
| `nll_history.txt` | 每个 run 的逐次求值 NLL 曲线 |
| `optimization_summary.txt` | 各 run 的 NLL / 迭代 / **eval数 / n_iter** / 耗时 / 正定性 / **误差模式 / 平坦维** / 优化器状态，及最佳解 |
| `best_params.pt` | 最佳参数（供 warm start） |
| `checkpoint.pt` | 续跑状态（`--resume` 用；含 base seed） |
| `weight_best.root` | 最佳解的权重文件（供 `plot.py` 画图）；另含 `interference_rec` / `interference_gen` / `*_err` / `integrals`（见 §10.4） |
| `param_covariance.csv` | **最佳点**的参数协方差矩阵（完整 N×N，`--err-mode` 回退口径） |
| `param_correlation.csv` | **最佳点**的参数关联系数矩阵 `ρ_ij=C_ij/(σ_iσ_j)`（完整 N×N；`σ=0` 处置 `nan`） |
| `polish_history.json` | 每次抛光一条记录（默认只有 best 一条；`--polish-all` 时逐 run 各一条）：before/trial/final 的 NLL 与正定、步数、Hessian 次数、是否采纳 |
| `all_params/run_XX.pt` | 仅 `--polish-all`：每个 run 的双精度最终参数（避免 `parameters.txt` 的 8 位小数在奇异点改变 NLL/Hessian） |
| `param_reduction.txt` | 仅 `--reduce-params`：参数削减报告（保留维度 k / 条件数 / 被冻结线性组合 / 全参数 σ） |
| `best_params_reduced.pt` | 仅 `--reduce-params` 且采用降维解时：降维重拟合后的参数 |

> 参数协方差/相关矩阵只在**正常拟合**时输出（`--ff-only` 不输出），且只对 **best** 点；
> log 里有一行 `[param-err] 参数协方差/相关矩阵已保存: ...` 提醒。矩阵覆盖**全部非固定参数**
> （排除固定参考 `idx 0/nc`，含贴边 `active`）。因本模型常见 `PD=False`、`err_mode=auto`
> 走有效秩截断（`--no-errors-truncate` 则退回伪逆），这是**秩亏下的可行估计、非严格统计误差**
> （CSV 头部亦有注明）。

> `plot.py` 另外生成 `results_plot.pdf` 等图形，不属于 `fit.py`（见 §10.1）。

> 多起点（`--runs N`）结束时终端会打印一行 `[ensemble]`：`n / best / median / IQR / std /
> worst`。前向 NLL 有数值混沌，**比较不同优化器/配置时请看中位数/IQR 而非单次 NLL**
> （差异需大于系综 std 才可信）；详见 §3.5 备注。

> **随机初值 seed**：启动时打印一行 `[seed] base seed=...; run i 的 seed = base+i; 复现本作业请加 --seed ...`。
> 默认 `base seed = 当前 Unix 时间戳`（每次作业不同）；要复现/要同 seed 系综对比，用日志里的值
> 加 `--seed <base>`（或用 `FIT_SEED`）；run i 的 seed = `base+i`。`--resume`
> 以 `checkpoint.pt` 内记录的 seed 为准。

---

## 7. 已知问题与备注

1. **配置由 `--config` 指定**（见 §1）：`--config /path/to/other.yml` 现在**真正生效**
   （import 期解析成绝对路径，并在构造 analysis 前 chdir 到 config 目录）。
   路径口径见 §1 的两条规则（config 内相对路径 → config 目录；CLI 路径 → 调用目录）。
2. `reparam` 的 `--amp-lambda` 是作用在**优化 loss** 上的弱先验；报告/保存的 NLL、
   Hessian、误差全部基于**真实 NLL**，不受罚项影响。**默认 `0`（关闭）**，开启方法与
   取值见 §11。
3. 结果普遍 `正定性=False`：精确 Hessian 存在 ~20–30 维近平坦/简并子空间（诊断显示
   沿负特征值方向 NLL 两侧都不下降，说明是数值/秩亏而非真鞍点）。
   **正定判据是“可行子空间”（KKT 二阶条件）**：先剔固定参考
   （Re_0/Im_0）、再剔**活跃**参数（贴边且梯度朝外——该方向根本不可行），才判正定；
   `is_positive_definite` = `reduced_pd`；`is_positive_definite_full` 是全自由块口径。
   因此贴边参数的“朝外负曲率”不再假报警，`--bound-margin` 可列出贴边 θ 作为对照。
   **用 `reparam` 时**参数误差默认由 `--err-mode auto` **按有效秩截断传播**
   （保留 `λ>τλmax` 方向，`σ²_m=Σ E[m,i]²/λ_i`，被冻结组合单独列出）；`--no-errors-truncate`
   退回伪逆口径；`projected`/`lbfgs` 默认 `strict`（非正定不给误差）。
   **拟合分数**现在直接读 `weight_best.root` 的干涉矩阵（相干分母，与 ROOT 逐位一致）：
   中心值恒有；`σ` 需要 C++ 写出 `interference_*_err`（耦合自由块正定）。
   **效率**仍由 C++ `getEfficiency` 计算，不以真 PD 为门控：非正定时用与 `--err-mode` 一致的回退曲率传播误差
   （中心值恒输出）。
4. `--lr` 只对 `reparam`/`lbfgs` 有意义；`projected` 不用 `lr`，其停机由
   `--tol-grad`（gtol）控制。
5. 若在集群提交作业，注意 `fit.py` 会读 cwd 的 `config.yml` 并需要 GPU；建议
   一个方案一个目录（或配合 `--output-dir` 隔离结果）。
6. **拟合分数默认输出，分波效率默认关闭**：`--cal-ff True`（默认）**从 `weight_best.root`
   读** FF（`writeResult` 已算好，不再额外扫 `phsp_truth`）；效率需显式 `--cal-eff True`，
   它仍要扫 `phsp`/`phsp_truth`。`writeResult` 的 truth 一趟与效率计算在过大样本上可能触发
   C++ `computeBFKernel` illegal memory access / glibc abort（本机实测：拟合期显存已近满，
   约 3.5M 事件即可崩）——该崩溃致命、`try/except` 无法捕获。因此 `fit.py` **先写核心摘要**
   再算 FF/效率，并在计算前打印提示。要算效率时建议：**减小 `phsp` 与 `phsp_truth` 样本量**
   后用 `--ff-only --cal-eff True` 跳过拟合、直接以 `best_params.pt` 重算（两个样本都要减，
   效率同时用到它们）。
7. **`--warm-start auto` / `FIT_WARM` 按 `--output-dir` 找 `best_params.pt`**（不是固定 `results/`）。
8. **输出约定**：`print` 输出结果（配置/进度/结果表/落盘提示/可复现 tag），
   `log`（含 `-v/-q` 级别控制）输出内部诊断（`[pLBFGS]`/`[polish]` 迭代与统计）。

---

## 8. 参数化 / 参数削减 / 误差截断 / 边界诊断

四组开关，**默认都保持既有行为**，按需开启。

### 8.1 `--fit-mode {0,1}`：参数化

| 值 | 含义 |
|---|---|
| `0`（默认） | `FREEPARAMS`：chain×step 参数化（与历史结果一致） |
| `1` | `VSPACE`：逐振幅（每个振幅只有 1 个因子 → Hessian 二阶项恒为 0 → 精确） |

```bash
python -u fit.py --fit-mode 1 --runs 5 --niter 500
```

⚠ `fit_mode` **只能在 `ctpwa.analysis()` 构造期决定**（构造后 `setFitMode` 改不了参数化）。
`fit.py` 因此在 **import 期**用 `_fit_mode_from_argv()` 从 argv / `FIT_MODE` 取出该值与
`--config` 一起传给 `ctpwa.analysis()`；需要 `ctpwa ≥ 0.3.10`。

### 8.2 `--reduce-params`：有效秩降维 + 降维重拟合

动机：乘积参数化下数据往往只约束住 Hessian 的一部分方向，其余方向近平坦甚至微负 →
直接求逆给出天文数字误差。把弱方向**冻结**、只保留 stiff 子空间：

```bash
python -u fit.py --reduce-params                  # 默认 tau=1e-3、iters=300
python -u fit.py --reduce-params --rank-scale median   # 谱病态/跑飞解
```

- 用精确 Hessian 的有效秩构造降维基 `p = p0 + E·y`，在 `y` 空间（Hessian 按构造正定、
  条件数从 ~1e7 降到 ~1e3）重新拟合；
- 输出 `param_reduction.txt`：保留维度 `k`、条件数、**被冻结的线性组合**（≈本数据未测定
  的方向，给出前 3 个主要分量）与每个参数由保留子空间传播的 σ；
- 若降维解 NLL 不差于原解（`≤ +1e-6`），**采用降维解作为最终结果**并另存
  `best_params_reduced.pt`，同时 `best_params.pt`/`weight_best.root` 用新解；
- 采用降维解时对角误差改为降维基传播，协方差/相关矩阵 CSV 在**最终点**重算。
- 相关旋钮：`--rank-tau`（门槛 `λ_i > tau·λmax`，默认 `1e-3`）、`--rank-iters`、
  `--rank-fixed-point`（降维基重算次数）、`--rank-scale {max,median,p90}`。

> 默认**不开**（会改变最终解与 `best_params.pt`）；用于诊断“误差给不出来”的原因。

### 8.3 `--no-errors-truncate`：非正定误差口径

`--err-mode auto`（`reparam` 默认）下 Hessian 不定时，默认改为**有效秩截断**：
只保留 `λ_i > τ·λmax` 的方向做误差传播 `σ²_m = Σ_{i∈keep} E[m,i]²/λ_i` →
**每个参数仍有有限误差**，被冻结的只是那些线性组合。加 `--no-errors-truncate` 退回
伪逆口径（未测定方向 σ=0）。`--err-mode strict/pinv/psd` 显式指定时不受影响。

### 8.4 `--bound-margin`：贴边参数诊断

```bash
python -u fit.py --bound-margin 0.02 --runs 1 --niter 200
# [bounds] ⚠ 1 个参数贴住边界 (边距 < 2% 宽度):
# [bounds]   phi1680_width: 1.5 ∈ [0.1, 1.5] 边距=  0.0% |∂NLL/∂θ|=3527
```

θ 顶到 `free_range` 边界后 reparam 的 sigmoid 梯度→0、优化器按容差停机，但物理空间梯度
很大 → Hessian 是在**边界/非驻点**上评估的，容易得到“假非正定”。该诊断会算出
`|∂NLL/∂θ|` 提示应放宽范围或加先验。

---

## 9. 后端选择（CPU / GPU）与 `CTPWA_DEVICE`（ctpwa ≥ 0.3.15）

`ctpwa ≥ 0.3.15` 起是**单 wheel 双后端**：wheel 里同时带 `_cuda` 与 `_cpu` 两个扩展，
`import ctpwa` 时先探测、再加载**一个**（两个后端互斥加载）。

探测不是只看 `torch.cuda.is_available()`（那只回答“驱动看得见卡吗”），而是**真跑一个
torch kernel** —— 编译架构与卡不匹配 / 卡被 MIG 切分或占满时，前者为 True 但 kernel
起不来。探测失败会**吵着回退 CPU**（慢 20–60 倍）。

| `CTPWA_DEVICE` | 行为 |
|---|---|
| `auto`（默认） | GPU 可用则用 CUDA，否则回退 CPU（打印警告） |
| `cuda` | GPU 不可用（或已 dlopen `_cuda` 后自检失败）**直接报错**，绝不偷偷降级 |
| `cpu` | 强制 CPU 后端（调试/无卡机器；大统计量不可用） |

```bash
export CTPWA_DEVICE=cuda        # slurm 作业推荐：GPU 拿不到就失败，别静默跑 CPU
python -u fit.py --config config.yml --runs 10
```

**`fit.py` 的策略**：**默认要求 GPU**——后端不是 `cuda` 就直接报错退出
（提示可以加 `--cpu`）；显式要 CPU 时才继续：

```bash
python -u fit.py --config config.yml --cpu --runs 1 --niter 100   # 小样本/调试：CPU 上跑（慢 20–60×）
```

- `--cpu` 会在 `import ctpwa` **之前**把 `CTPWA_DEVICE` 置为 `cpu`（覆盖已有 env，并打印一行提示）；
  之所以必须在 import 前做，是因为后端在 import 期就**互斥加载**好了，之后再换来不及。
- **只设** `CTPWA_DEVICE=cpu`（不加 `--cpu`）也算"显式要 CPU"，同样放行。
- 想"拿不到 GPU 就失败、绝不静默降级"的作业，仍然照旧 `export CTPWA_DEVICE=cuda`。

其它相关接口/环境变量：

- `ctpwa.backend()` → `'cuda'` / `'cpu'`；`ctpwa.build_info()` →
  `{'backend','precision','complex_bytes','has_cuda'}`；`ctpwa.BACKEND`、
  `ctpwa.FALLBACK_REASON`（auto 模式下的回退原因）。
- `CTPWA_QUIET=1` 抑制导入期的提示（不影响 `ctpwa.backend()` 的返回值）。
- `fit.py` 启动 dump 会打印 `backend=… precision=… complex_bytes=… has_cuda=…`；
  非 `cuda` 时额外 `log.warning`，方便在 `.log` 里一眼看出是否跑错后端。
- **dtype**：`getFitFractions`/`getEfficiency` 的 C++ 入口要求 vector 的复数 dtype 与
  `.so` 编译精度一致，`fit.py` 按 `build_info()['complex_bytes']`（16→complex128 /
  8→complex64）选，探测不到再用 `DeviceManager().compiledPrecision()` / params 精度兜底。
- ⚠ 仓库根有上游自带的 `ctpwa/` 门面**源码**目录（没有 `.so`），在 `python -m pytest` /
  `python -c`（cwd 进 `sys.path`）时会**遮蔽**已安装的 wheel → 报“没有可用的 ctpwa 后端”。
  跑在各自作业目录不受影响。
- 本仓库本地单测 `example/tests` 自己往 `sys.modules` 放 stub，并按**文件路径**加载真实的
  `ctpwa.diagnostics`（纯 torch，CPU 可跑），因此不受上面这条影响。

---

## 10. 其它脚本（不属于 `fit.py` 用法）

本章四个脚本都**自包含**（不 import 本仓库其它文件，可单独拷到任何目录运行）：§10.1 画图、
§10.2–§10.3 生成 MC、§10.4 读 `weight_best.root` 里的干涉矩阵。后两个 MC 生成
脚本基于上游 DIYMC（`ctpwa ≥ 0.3.15`）的 `PhaseSpaceGenerator` / `generateMC` / `getIntensity` /
`writeMC`，产物与 `fit.py` 的读取约定一致：`.dat` **每事件 n_final 行**、每行 `E px py pz`
（12 位小数、空格分隔），末态顺序 = config 的 `Data.order`；也支持 `.root`
（`TTree "mc"`，TLorentzVector 分支）与 `.npy`（`(n, n_final, 4)`）。

> ⚠ **`.dat` 每行固定 4 个数，没有权重列**（C++ 读取端按「每事件 4×n_final 个数」解析，多一列会错位）。
> 要带权重：写 `root`（权重分支）/`npy`（另存 `<path>.weights.npy`），或按 config 的
> `data_weights` / `bkg_weights` / `phsp_weights` 约定**另存一个每事件 1 行的独立权重文件**。

> ⚠ **`--seed` 默认取当前 Unix 时间戳**（与 `fit.py` 一致）→ **每次运行都不同**，避免"忘了改
> seed"导致重复样本；要复现就把启动打印的 `[seed]`（或 `--stats-json` 里的 `seed`）用
> `--seed N` 传回来。

### 10.1 `plot.py` 使用说明

从 `<output-dir>/weight_best.root`（或指定 run 的参数）读取直方图并生成 PDF 图集。

#### 运行

```bash
python plot.py                # 读 results/weight_best.root → results/results_plot.pdf
python plot.py --params 3     # 用 parameters.txt 第 3 个 run 参数经 GPU 重新生成权重 → results/results_plot_run_3.pdf
python plot.py --nll 3        # 画第 3 个 run 的 NLL 历史
```

#### 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--params N` | 用 `parameters.txt` 第 N 个 run 的参数经 GPU 重新生成权重再画图（需 `ctpwa`） | 无（直接读 `weight_best.root`） |
| `--nll N` | 画 `nll_history.txt` 第 N 个 run 的 NLL 历史 | 无 |

#### 输入直方图（由 fit.py / C++ 写入 `weight_best.root`）

| 对象 | 含义 |
|---|---|
| `hdata` | 原始数据计数 |
| `hfit` | 信号模型（振幅²×phsp）× `normFactor`（即信号产额） |
| `hbkg` | 预期本底计数（本底事件 × `bkg_weights`，不 Scale） |
| `h_<分波名>` | 各分波形状（`\|A_i\|²` 加权） |
| `interference_rec` / `interference_gen` | 干涉矩阵（`TMatrixD`，完整对称矩阵）；`plot.py` 不读它，读法见 §10.4 |
| `legends` | 分波图例 |

#### 输出内容
- **1D 页**：每个观测（质量 / cosβ / obs-1d）一组合图——上方：数据点 + 各分波形状 +
  **Fit+Bkg** + 本底填充；下方：**pull** `(data-(hfit+hbkg))/√(hfit+hbkg)`，并标注 `χ²/ndf`。
- **2D Dalitz 页**：每个 2D 观测（dalitz / obs-2d）三列 **Data / Fit+Bkg / Pull**，
  pull 口径同上（含本底）。

> 口径说明：`data ≈ hfit+hbkg`（`hfit` 已按“数据−本底”归一，`hbkg` 为预期本底）。
> 因此 Fit 面板与 Pull 都必须用 `hfit+hbkg`；无 `hbkg` 时退化为只看 `hfit`。

#### 依赖
- `uproot` 读取 ROOT 文件；`--params` 需要 GPU + `ctpwa`。
- 复数 dtype 按 `ctpwa` `.so` 编译精度自动选择（`complex128`/`complex64`）。

### 10.2 `gen_phsp_mc.py`：平坦相空间（P1，**不需要 GPU**）

```bash
python gen_phsp_mc.py --config config.yml -n 100000 --seed 42 -o phsp_flat.dat
python gen_phsp_mc.py --config config.yml -n 5000000 --chunk 500000 -o phsp_flat.dat  # 分块追加写
python gen_phsp_mc.py --config config.yml -n 200000 --format npy -o phsp_flat          # 给探测器模拟
python gen_phsp_mc.py --config config.yml -n 200000 --format root -o phsp_flat.root
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--config` | `config.yml` 路径 | `config.yml` |
| `-n`, `--nevents` | 目标事件数（**必填**） | — |
| `-o`, `--output` | 输出文件（dat/root 带扩展名；npy 可省） | — |
| `--seed` | 随机种子；**默认取当前 Unix 时间戳（每次运行不同）**。分块时第 k 块用 `seed+k`（块间不重复）；复现请传回启动打印的 seed | 当前时间 |
| `--format {dat,root,npy}` | 输出格式；默认按 `-o` 的扩展名，无扩展名时用 `dat` | 按扩展名 |
| `--mother` | 母粒子名 | 从 config 的 `DecayChains` 推 |
| `--mother-mass` | 覆盖母粒子质量 (GeV) | config 值 |
| `--weighted` | 导出 GENBOD 权重；`.dat` 每行固定 4 个数、无权重列，故仅 `root`/`npy` 有意义（见章首 ⚠） | False |
| `--chunk` | 分块生成的事件数（>0 逐块生成并追加写；仅 dat/npy，用于千万级样本） | `0`（不分块） |
| `--force` | 末态顺序与 `Data.order` 不一致时仍继续 | False |
| `--no-check` | 跳过我方物理自检（守恒/质量壳） | False |
| `--check-events` | 物理自检抽样事件数 | `2000` |

要点：

- 只解析 config 的 `Particles` / `Data.order` / `DecayChains`，**不读数据文件、不需 GPU**；
- 内置自检并打印：母粒子与质量、末态顺序 vs `Data.order`、`weightMax`、四动量守恒、
  各末态质量壳、`dat` 行数 = `n × n_final`；不一致时**退出**（`--force` 可放行）。
- `--weighted`（GENBOD 权重）只写 `root`/`npy`；`--weighted` + `--format dat` **直接报错退出**。

> ⚠ 两个语义提醒：
> 1. **它不等于带效率的 `phsp_truth`**：后者是同一批探测器 MC 的 truth 级版本，可能带 truth 级
>    fiducial 选择；用平坦 PS 顶替它会改变"效率"的定义（∫|A|²ε/∫|A|²）。它适合
>    ① closure/toy 研究里当理想 truth 样本 ② 作为**外部探测器模拟的输入**（重建后才得到 `phsp`）。
> 2. **用于 closure 时样本量要能分辨最窄的共振**：上游实测 4.5 MeV 宽的 φ(1020) 只给 10k
>    phsp 时，toy 真值不是 NLL 极小（拟合把宽度顶到边界），加到 200k 才恢复。本仓库共振宽度
>    是 MeV 级 → 建议**百万级**。

### 10.3 `gen_toy_mc.py`：振幅加权 toy → pseudo-data（P2/P3，**需要 backend**）

```bash
# 5 万信号 toy + Poisson + 真本底 → pseudo-data（并派生 config、写 stats）
python gen_toy_mc.py --config config.yml --params-from results/best_params.pt \
    -n 50000 --seed 42 --poisson --nobs-from-data --bkg \
    -o toy_data.dat --write-config config_toy.yml --stats-json toy.json

# 纯信号等权 / 带权（快）样本
python gen_toy_mc.py --config config.yml --params-from results/best_params.pt -n 100000 -o toy.dat
python gen_toy_mc.py --config config.yml --params-from results/best_params.pt \
    -n 500000 --mode weighted --format root -o toy_w.root

# 自洽性检查：生成分布 vs |M|² 加权平坦样本（多生成一份加权样本）
python gen_toy_mc.py --config config.yml --params-from results/best_params.pt \
    -n 50000 --check-distribution -o toy.dat
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `--config` | `config.yml` 路径 | `config.yml` |
| `--params-from` | 拟合参数 `best_params.pt`（`[Re.., Im.., θ]` float64；长度按 config 校验；**会自动搬到当前后端设备**——CUDA 后端要求 CUDA 张量，而 `best_params.pt` 是 CPU 张量） | `results/best_params.pt` |
| `-n`, `--nevents` | 目标事例数（`unweighted` 时为输出事件数；`--poisson` 且未给 `-n` 时默认 = `nobs`） | None |
| `-o`, `--output` | 输出文件（dat/root/npy） | — |
| `--seed` | 随机种子；**默认取当前 Unix 时间戳（每次运行不同）**。同时喂给 `generateMC`（平坦相空间 + 舍选）和本地 RNG（Poisson 抽样 / 重抽样 / 本底 / 打乱，内部用 `seed+20260101`）；`--check-distribution` 的参照样本用 `seed+9999`；复现请传回启动打印的 seed | 当前时间 |
| `--mode {unweighted,weighted}` | `unweighted`=平坦相空间 + `\|M\|²` **舍选（AR）** → **等权**；`weighted`=**全收** + 每事件权重 `I` → **快**（详见下方要点） | `unweighted` |
| `--batch` | 每批平坦事例数（传给 `generateMC`） | `100000` |
| `--max-trials` | 平坦事例数上限（安全检查） | None（不限） |
| `--format {dat,root,npy}` | 输出格式；默认按 `-o` 的扩展名 | 按扩展名 |
| `--poisson` | 抽成等权 pseudo-data：`nobs ~ Poisson(目标事件数)`（等权无放回 / 带权按 w 有放回） | False |
| `--nobs N` | `--poisson` 的目标事件数（信号部分） | None |
| `--nobs-from-data` | `--poisson` 的目标事件数取 config 的 `data` 文件事件数 | 该行为默认 |
| `--allow-short` | 可用信号 < nobs 时取 min 并警告（默认报错） | False |
| `--bkg` / `--no-bkg` | 叠加 config 的 `bkg`（按 `bkg_weights` 加权抽样）；默认“有 bkg 就叠” | 有则叠 |
| `--bkg-scale` | 本底产额缩放：`Poisson(scale·Σw)` | `1.0` |
| `--no-shuffle` | 信号与本底拼接后不打乱（默认打乱） | False |
| `--write-config` | 派生 config（把 `data` 指向 toy），可直接 `fit.py --config config_toy.yml` | None |
| `--stats-json` | seed / 参数 md5 / 事件数 / bkg 产额 / `generateMC` 的 `stats` | None |
| `--check-distribution` | 自洽性检查：把**生成样本**与「`\|M\|²` 加权平坦样本」投影到 `m²(i,j)` 分布比对（详见下方要点） | False |
| `--check-events` | 自洽性检查用的事件数 | `20000` |
| `--check-pair` | 自洽性检查投影的两个末态下标 | `0,1` |
| `--check-bins` | 自洽性检查直方图 bin 数 | `25` |

要点：

- **`--mode unweighted`（默认）**：平坦相空间 → 按 `|M|²` **舍选（AR）** 得**等权**样本，可直接写
  `dat`。日志 `stats` 里的 `acceptance` = **接受率**（接受数 / 平坦事例数）、`max_weight` = 舍选上界
  `Imax`（判据 `u·Imax < I`；未给 `--max-weight` 时用首批 pilot 自适应）——**接受率越低越费时**
  （本模型实测 ≈2%）。同批还会打印 `flat_acceptance`（平坦 / GENBOD 试探比）、`n_genbod_trials`、
  `truncated` 等。
- **`--mode weighted`**：平坦事例**全收** + 每事件权重 `I`（不做舍选）→ **快**（≈`1/acceptance` 倍）；
  但 `.dat` 每行固定 4 个数、无权重列，故**只能写 `root`/`npy`**。要拿到等权的 `dat`，配 `--poisson`
  （按权重 `p=w/Σw` 有放回抽成等权 pseudo-data）。
- **`--check-distribution` 在做什么（自洽性检查）**：它额外生成一份**独立**的参照样本——
  **加权平坦样本**（`mode=weighted`、`--check-events` 个事件、seed = `seed+9999`），后者天然
  ∝ `|M|²`（平坦相空间 × 权重 `I`）。然后把**本次生成的信号样本**和这份参照样本都投影到末态对
  `--check-pair`（默认 `0,1`）的**不变质量平方** `m²(i,j)` 上，用同一组 `--check-bins`（默认 25）
  直方图分别归一化到 Σ=1，取**逐 bin 最大绝对差** `max|Δ|`：
  - 判据：`max|Δ| < 0.05` → 打印 `OK`，否则 `偏大，检查样本量/参数`。含义是「**两条独立路径
    （舍选 vs 加权）给出的 `|M|²` 形状一致**」，用来抓生成实现里的错（判据写反、权重/顺序错…）。
  - 它是**统计**比较：有限样本有涨落，bin 越多、样本越小越容易超阈值 → 报大时先加大
    `--check-events`（或减小 `--check-bins`）再判断，**不是**"代码一定错了"的结论。
  - 前提：被检查的信号样本本身要是**等权、∝`|M|²`**。即用 `--mode unweighted`（默认，示例用法）
    或 `--mode weighted --poisson`（重抽样后已变等权）。若用 `--mode weighted` 又**不加** `--poisson`，
    生成的样本是"平坦 + 权重"，其**未加权**直方图不是 `|M|²` → 该检查会（正确地）报偏大，但那属于
    **用法问题**。对比只用**纯信号**（叠加 `--bkg` 之前），本底不参与。
  - 它**只**验证"生成分布 = `|M|²`"这条自洽性，**不**验证"toy 真值是否 NLL 极小"（闭包）——
    后者要用 `--write-config` 派生 config 去跑 `fit.py`。

> ⚠ 所有便宜的错误路径（参数文件/长度、`weighted`+`dat` 组合）都在构造 `analysis` **之前**报错。
> ⚠ 本脚本**不做拟合**。想验证"toy 真值是否 NLL 极小"（闭包），用 `--write-config` 派生 config
> 后跑 `fit.py`。样本量要求同上（MeV 级宽度 → 百万级）。
> ⚠ `fit.py` **默认要求 CUDA**（检测不到可用 GPU 直接退出）；确需 CPU 时加 `--cpu`
> （或 `CTPWA_DEVICE=cpu`），见 §1 / §9。

### 10.4 `read_interference.py`：读取 `weight_best.root` 里的干涉矩阵

`writeResult` 会把两张**按本样本相干总额 `W` 归一**的对称矩阵（`interference_rec` /
`interference_gen`）连同参数误差矩阵和 `integrals` 树写进 `weight_best.root`；
`read_interference.py` 直接把它们打成表。本节讲这些对象是什么、口径怎么算、怎么读。

> 重建侧直接由常驻 `B̄ = Σ_e (w_e/W)A_e A_e^H` 收缩得到（不读 `phsp` 文件），产生侧需要一趟
> `phsp_truth` 扫描；误差矩阵是解析传播（只含耦合方向）。实现细节的版本变更见 git 提交记录。

#### 10.4.1 `weight_best.root` 里多了什么

| 对象 | 类型 | 归一化 | 含义 |
|---|---|---|---|
| `interference_rec` | `TMatrixD` n×n | `P_ij / W_rec` | **重建侧**：`phsp` 样本（带效率 MC）。`P_ii = Σ_e w·\|A_i\|²`，`P_ij = Σ_e w·2Re(A_iA_j*)` |
| `interference_gen` | `TMatrixD` n×n | `T_ij / W_gen` | **产生侧**：`phsp_truth` 样本（无效率）。元素定义同上 |
| `interference_rec_err` | `TMatrixD` n×n | σ | 参数误差传播 `σ(F_ij) = √(J H⁻¹ Jᵀ)`（线性化；**不含 MC 统计误差**；只含**耦合**方向，θ 方向暂未传播） |
| `interference_gen_err` | `TMatrixD` n×n | σ | 同上，truth 侧 |
| `integrals` | `TTree`（1 entry） | – | `N_rec` / `W_rec` / `N_gen` / `W_gen`（`Double_t`）、`npartials`（`Int_t`） |
| `legends` | `TTree` | – | 分波 LaTeX 标签；长度 = `npartials` 时 `read_interference.py` 拿它当行标签 |

**行列顺序** = **partial wave**（分波）顺序，与 `getFitFractions` 的返回顺序、`h_<分波名>` 直方图、
`legends` 的图例顺序一致。⚠ 矩阵维度是**分波数**（启动日志里的
`Number of partial waves: N (SL amplitudes: M)`，通常 `N < M`），**不是** `parameters.txt` 里那
`M` 个 SL 振幅耦合的下标；两者不能按下标一一对应。

什么时候**没有**这些对象：

| 缺什么 | 原因 |
|---|---|
| `interference_rec` | config 没有 `phsp`，或设了 `CTPWA_WR_NO_INTEGRALS=1` |
| `interference_gen` | config 没有 `phsp_truth`，或设了 `CTPWA_WR_NO_INTEGRALS=1` |
| `interference_*_err` | 耦合自由块的 Hessian 非正定 / 没有自由耦合方向 / 参数不是 float64（legacy complex 精度）/ `CTPWA_WR_NO_ERR=1`。C++ 端会在生成 ROOT 时的 stderr/stdout 打一行原因 |

> `CTPWA_WR_NO_INTEGRALS=1` 跳过矩阵中心值这一步（重建侧 0 趟，直接收缩常驻 `B̄`；
> 产生侧 1 趟 `phsp_truth` 全样本扫描，大样本上这是最贵的部分）。跳过时矩阵与 `integrals`
> 都不写，直方图照写。
> `save_weight_file` 会先把拟合缓存里的 Hessian 递给 `writeResult`，所以正常情况下误差矩阵不需要额外算 Hessian。
> 误差矩阵的额外成本：重建侧 0 趟（`B̄` 代数）、产生侧复用同一趟扫描的 ± 方向，与耦合数无关。

#### 10.4.2 矩阵口径与三个「份额」的区别

```
M_ii = Σ_e w·|A_i|² / W                    （对角：无因子 2）
M_ij = Σ_e w·2Re(A_i A_j*) / W   (i ≠ j)   （非对角：有因子 2，对称）
W    = Σ_i M_ii + Σ_{i<j} M_ij = Σ_e w·|Σ_i A_i|²
⇒ Σ_i F_i + Σ_{i<j} I_ij = 1               （闭合，可直接当自检）
```

| 量 | 定义 | 从 ROOT 怎么取 | 性质 / 用途 |
|---|---|---|---|
| 文献 fit fraction `F_i` | `M_ii` | `diag(interference_gen)` | `ΣF + ΣI = 1`，**推荐对外报告这个** |
| 干涉项 `I_ij` | `M_ij` | 上三角 | 可正可负；强干涉时单对可达 ±0.1 量级 |
| 非相干份额 `FF_i` | `F_i / ΣF` | `diag / Σdiag` | `ΣFF = 1` 但把干涉**摊回**各波（`getFitFractions` 口径，已被标为待删） |
| 平均效率 `ε̄` | `W_rec / W_gen` | `integrals` | 只在 rec/gen 两样本同源产生时有意义 |
| 每波效率 `ε_i` | `(P_ii/N_rec) / (T_ii/N_gen)`，其中 `P_ii = diag_rec × W_rec`、`T_ii = diag_gen × W_gen` | 两张矩阵 + `integrals` | 相对值（cut 后样本分布 ∝ `εdΦ/∫ε`）；矩阵里存的是归一化值，**必须乘回 W** 才是原始积分 |
| 分支比 `BF_i` | `BF_total × F_i` | `--bf-total` | `ΣBF_i ≠ BF_total`，差额正是干涉项 |
| 原始积分 `P_ij` / `T_ij` | `M_ij × W` | 矩阵乘回 `integrals` 的 `W` | 想自己做归一化（换分母、按 run 合并）时用 |

> 对象名是 `interference_rec`（重建侧，带效率）与 `interference_gen`（产生侧），**没有**单个
> `interference` 矩阵——读旧名字会 `KeyError`。`plot.py` 里跳过 `"interference"` 目录的名单是
> 历史残留，无影响。

#### 10.4.3 `read_interference.py` 用法与参数

```bash
# 基本：两张表 + 派生量全打
python3 read_interference.py results/weight_best.root

# 只看产生侧（对外报 fit fraction 用这个）
python3 read_interference.py results/weight_best.root --sample gen

# 折算绝对分支比（给总 BF），并把矩阵导成 CSV
python3 read_interference.py results/weight_best.root --bf-total 3.7e-4 --csv ff.csv
```

| 参数 | 说明 | 默认 |
|---|---|---|
| `root`（位置参数） | `weight_best.root` 路径 | 必填 |
| `--sample {rec,gen,both}` | 打哪一侧 | `both` |
| `--bf-total X` | 给总分支比，逐波打印 `BF_i = X × F_i`（用 gen 侧对角；无 gen 时退到 rec） | 无 |
| `--csv FILE` | 把两张矩阵的全部元素 + 对角误差写成**一行** CSV | 无 |

输出块顺序：每侧一块（`== interference_xxx ==` 标题 + 闭合检验 + 逐波 `F_i ± σ` 表 + 全矩阵 +
误差矩阵）→ 效率块（`both` 且两样本都在时）→ 分支比块（给了 `--bf-total` 才有）→ CSV 提示。

依赖 `uproot`（`pip install uproot`；`envs/ctpwa` 已装）。它只读 ROOT，和拟合本身用哪个环境无关。

#### 10.4.4 输出块怎么读

跑一次就能看到真实数值（每个分析的数当然不同）。脚本输出按固定顺序分块，各块的含义：

| 输出块 | 含义 |
|---|---|
| `== interference_rec/gen ==  W=… N=…` | 本侧样本的相干总额 `W = Σw|ΣA|²` 与 `N = Σw`（来自 `integrals`；把矩阵元素乘回 `W` 就是原始积分） |
| `Σ_i F_i = …` / `Σ_{i<j} I_ij = …` / `闭合 (ΣF+ΣI) = …` | 对角和、干涉项之和、闭合自检。前两者相加**必须 = 1**；不等（如 0.99998）说明矩阵读错或被截断。`ΣF` 远大于 1 表示各波强相消（360° 相位差），不是错误 |
| `非相干份额 F_i/ΣF = [...]` | `diag/Σdiag`（`getFitFractions` 口径，`Σ=1`）。`fit.py` 不打印它，只在来源行提示"与 `F_i` 相差 `1/ΣF`" |
| 逐波表 `# / 分波 / F_i / σF_i` | 文献口径 fit fraction 与误差；`σF_i = -` 表示 ROOT 里没有 `interference_*_err`（见 §10.4.1 的写入条件）。**`fit.py` 末尾打印的就是这张表**（同一份 `weight_best.root`，逐位一致） |
| `矩阵 (行= i, 列= j ...)` | 对角 = `F_i`、上三角 = `I_ij`（对称，两边都填）。看"谁和谁相消"就看或或这里最负/最正的元素 |
| `误差矩阵 σ` | 仅当 ROOT 里有 `interference_*_err` 时打印 |
| `== 效率（由两张表直接得到）==` | `ε̄ = W_rec/W_gen` 与逐波 `ε_i = (P_ii/N_rec)/(T_ii/N_gen)`（`P_ii`/`T_ii` = 表里的对角乘回 `W`）；只在两样本同源产生时可解释（相对效率，`ε_i^abs/ε_all`） |
| `== 分支比（BF_total = …）==` | 给了 `--bf-total` 才有：`BF_i = BF_total × F_i`；`ΣBF_i ≠ BF_total` 的差额正是干涉项 |
| `已写 xxx.csv` | 给了 `--csv` 才有：两张矩阵的全部元素 + 对角误差，**两行**（表头 + 一行数据） |

> 自动环境里看到 `wave0/wave1…` 这种行标签，说明 `legends` 长度和 `npartials` 对不上，
> 按 `idx` 对 `getFitFractions` 的返回顺序 / `h_<分波名>` 直方图即可（**不是** `parameters.txt` 的 SL 耦合）。

#### 10.4.5 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `... 里没有 interference_rec/gen`（脚本返回 1） | `CTPWA_WR_NO_INTEGRALS=1`；或 config 没有 `phsp` / `phsp_truth` | 去掉该环境变量；补样本 |
| 有矩阵但没 `*_err` | 耦合自由块 Hessian 非正定 / 无自由方向 / 参数非 float64 / `CTPWA_WR_NO_ERR=1`（θ 方向本来就不传播） | 看生成 ROOT 时 stdout 那行原因；`fit.py` 的 `--err-mode` 只管它自己的误差，不影响这里 |
| 行标签是 `wave0/wave1…` | `legends` 长度与 `npartials` 对不上（config 的 `tex` 展开规则见教程 §4.0） | 正常；按下标对 `getFitFractions` 的返回顺序 / `h_<分波名>` 直方图（**不是** `parameters.txt` 的 SL 耦合） |
| `--bf-total` 的 `ΣBF_i ≠ BF_total` | 差额 = 干涉项 `ΣI`（同一张表里有） | 物理上正常；要"非相干"合计就用 `ΣF` |
| 读的是 rec 还是 gen？ | rec 带效率（`phsp`），gen 是产生侧（`phsp_truth`） | 报 fit fraction / 与文献比 → gen；与数据、效率相关 → rec |
| 跑得很慢 / 占显存 | 矩阵本身很小；贵的是 `phsp_truth` 那一趟全样本扫描（重建侧已不读文件） | 大规模扫描可临时 `CTPWA_WR_NO_INTEGRALS=1`；要计时加 `CTPWA_VERBOSE=1`（只打 `phsp_truth 积分矩阵: … ms`） |

---

## 11. 干涉与幅度控制三参数

本节讲 `--amp-lambda` / `--amp-max` / `--vmax` 这三个“幅度控制”参数，以及**如何按预期的干涉强度
反推 `--amp-lambda`**。三者都只影响**优化路径**，不改变报告的 NLL / Hessian / FF 的定义。

### 11.1 问题：干涉简并方向

同一个共振态可以有多条 LS 波。若两条波相位差 π，它们会**互相抵消**：总强度 `|Σ A_i v_i|²` 几乎
不变，但各自幅度可以任意放大 —— 这条方向的 NLL **近乎平坦**。后果：

- 拟合会沿该方向“漂移”，把 `|v|` 推到软墙帽（`--amp-max`）附近；
- 表达干涉靠“两个大数相减”，数值精度差、误差估计不稳；
- 不同起点落到这条平坦方向的不同位置 → best-of-N 的 NLL 看着差不多，但参数/FF 会发散。

```
抵消前:  |1|² = 1
抵消后:  |1 + a·e^{iπ}|² = (1−a)² ，但 Σ|v_i|² = 1 + a² 随 a 一起变大
```

### 11.2 三个参数的分工

| 参数 | 层级 | 数学形式 | 标度（单位） | 生效优化器 | 默认 |
|---|---|---|---|---|---|
| `--vmax` | **硬**盒约束 | `\|v\| ≤ vmax`（KKT / 活跃集；不做 clamp） | 直接是耦合幅度 | `projected` / `lbfgs` | `10000` |
| `--amp-max` | **软**墙 | `amp = amp_max·sigmoid(u)`，逼近上限时 `damp/du → 0` 自限幅；polish 阶段另按 `amp_max` 做硬帽 | 极坐标幅度 | `reparam` | `1000` |
| `--amp-lambda` | **罚项**（弱先验） | `loss = NLL + λ·Σ_{i≥1}\|v_i\|²`（参考波 `idx0` 不计） | 等价高斯先验 `σ = 1/√(2λ)`，以参考波 `\|v_0\| = 1` 为单位 | `reparam` | **`0`（关闭）** |

要点：

- **互斥**：`reparam` 下 `--vmax` 不生效；`projected` / `lbfgs` 下 `--amp-max` 不生效
  （内部 `amp_cap = amp_max if reparam else v_max`）。
- **只进优化 loss**：`λ` 这一项不进 `nll_history`、不进报告 NLL、不进 Hessian —— 它像一个弱先验，
  改变的是**落点**，不是物理量的定义。
- `λ` 与“先验尺度”的换算（`σ = 1/√(2λ)`，相对参考波 `|v_0| = 1`）：

  | `λ` | `1e-4` | `1e-3` | `1e-2` | `1e-1` | `1` |
  |---|---|---|---|---|---|
  | `σ` | 70.7 | 22.4 | 7.07 | 2.24 | 0.71 |

- 启动 dump 会打印 `lambda=0(off)` 或 `lambda=0.001`，便于确认本作业到底开没开。

### 11.3 由预期干涉强度反推 `--amp-lambda`

#### Step 1：量出“干涉深度” `d`

```
D = Σ_i ∫|A_i|²          （对角和）
T = ∫|Σ_i A_i|²          （总强度，含干涉）
d = 1 − T/D              （d = 0 无干涉；d → 1 完全抵消）
```

**两种取法**（⚠ 不能用 `--cal-ff` 输出的那个 FF 列表：它的分母就是 `D`，恒 `Σ FF = 1`，读不出 `d`；
要用**按相干总额归一**的矩阵，见 §10.4）：

| 来源 | 公式 |
|---|---|
| tf-pwa 输出 | `d = 1 − sum_diag`（`sum_diag = Σ_i FF_i`，那边 FF 用相干分母） |
| 本仓 `weight_best.root` 的 `interference_gen` | `d = 1 − 1/Σ_i diag_i`，见下面的片段 |

```python
import uproot, numpy as np
m = uproot.open("weight_best.root")["interference_gen"]   # 产生侧（重建侧用 interference_rec）
M = np.asarray(m.member("fElements"), dtype=float).reshape(
    m.member("fNrows"), m.member("fNcols"))
D, T = float(np.trace(M)), float(M.sum())     # T ≡ 1（已按相干总额归一）
d = 1.0 - T / D                               # 等价于 1 − 1/ΣF
# 自检：T == 1（ΣF + ΣI = 1，见 §10.4）
```

> 该矩阵口径：`M_ii = Σw|A_i|²/W`、`M_ij = Σw·2Re(A_iA_j*)/W`，
> `W = Σ_i M_ii + Σ_{i<j} M_ij`，因此 `Σ_i M_ii + Σ_{i<j} M_ij = 1`。
> ⚠ 这与 `getFitFractions` 内部用的 `scattering` 矩阵（**对角含因子 2**）**不是**同一套，别混用。
> 没有单个 `interference` 对象；按样本用 `interference_gen`（产生侧）或 `interference_rec`
> （重建侧，带效率），两者的定义相同。

#### Step 2：反解“需要的伙伴幅度” `a`

2 波模型（参考 `v_0 = 1` + 伙伴 `a`、相位差 π）可严格求解：

```
d = 2a / (1 + a²)        ⇒        a(d) = (1 − √(1 − d²)) / d
```

- 若干涉由 `m` 对平摊：每对取 `d_i = d/m`，再算 `a = a(d/m)`；
- 若物理上本来就需要较大的 `|v|`（窄共振、相空间压低等），取
  `a_need = max( a(d), max_i|v_i| )`，后者从一次 `λ = 0` 拟合的 `parameters.txt` 里读。

#### Step 3：定 `λ`

```
弱先验判据（推荐）：  λ ≤ 1 / (2·k²·a_need²)              k = 5（推荐）/ 3（中等）/ 1（明显压低）
严格判据（有误差时）：λ ≤ ε·H_flat / 2 = ε / (2·σ_flat²)   ε = 允许的相对偏移（如 0.1）
```

第二式的来历：罚项把该方向的最小点按 `H/(H + 2λ)` 收缩（`H` 为该方向的 NLL 曲率），相对偏移
`≈ 2λ/H`；而 `H` 与参数误差满足 `σ_flat = 1/√H`。

代入 `a(d)`（单对，取 `k = 5` 与 `k = 1` 两档）：

| 干涉深度 `d` | 0.05 | 0.10 | 0.30 | 0.50 | 0.70 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| 需要幅度 `a` | 0.025 | 0.050 | 0.154 | 0.268 | 0.408 | 0.627 | 0.724 | 0.868 |
| `λ_max`（`k=5`，弱先验） | 88.8 | 22.1 | 2.36 | 0.279 | 0.120 | 5.1e-2 | 3.8e-2 | 2.7e-2 |
| `λ_max`（`k=1`，明显压低） | 799 | 199 | 21.2 | 6.96 | 3.00 | 1.27 | 0.95 | 0.66 |

多对平摊时（总深度 `d = 0.9`）：`m=1 → λ_max ≈ 5.1e-2`、`m=2 → 0.35`、`m=4 → 1.5`、`m=8 → 6.3`。

**这张表最重要的结论**：对任何现实干涉深度（`d = 0.1–0.95`），“先验不咬人”的上限是
`λ ≲ 3e-2 ~ 2e1`；而实用区间 `λ = 1e-4 … 1e-3` 比它小 1–5 个数量级。所以

- **默认关掉（`λ = 0`）与原来的 `1e-4` 在物理上几乎没有差别**，`1e-4` 原本只起“防跑飞”；
- 反过来，`λ ≳ 3e-2` 才真正开始压缩干涉 —— 想主动压干涉时才用这个量级。

### 11.4 推荐流程

1. **默认 `λ = 0` 直接跑**（当前默认已如此）；
2. 看 `parameters.txt` 里各 `|v_i|` 有没有贴 `--amp-max`：贴了说明在沿简并方向漂移；
3. 需要保险丝时按 §11.3 的表取 `λ`（一般 `1e-4` 或 `1e-3` 就够，且不改变结论）；
4. 想**主动**压掉“两波抵消”的虚假解 → 用 `k = 1` 档（`λ ~ 1e-2`）并把 `--amp-max` 调小，
   再用多起点 best-of-N 交叉检查；
5. 无论开不开，`--amp-max` 都要确认大于该分析真实的最大 `|A|`。

### 11.5 复现旧结果（`--amp-lambda`）

> 当前默认 `--amp-lambda 0`（关闭）。要复现历史上用 `1e-4` 跑出的结果，显式加
> `--amp-lambda 1e-4` 或 `FIT_AMP_LAMBDA=1e-4`；`λ` 只影响落点，两者 NLL 差异通常远小于
> 多起点的散布，但若要与历史逐位对齐，请显式指定。

---
