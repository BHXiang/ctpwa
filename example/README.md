# `fit.py` 使用说明

`example/fit.py` 是 ctpwa 的拟合驱动脚本：读取 `config.yml` → 构建振幅 → 多次优化 →
Hessian / 参数误差 → 保存结果与摘要。

本文主要说明**可选参数**和**三种优化器的区别**。

---

## 0. TL;DR

```bash
# 在放有 config.yml 的目录里运行
python -u fit.py --config config.yml --runs 10 --niter 3000            # 默认 reparam
python -u fit.py --optimizer projected --runs 2 --niter 3000           # 对照（慢）
python -u fit.py --optimizer reparam --amp-max 1000 --amp-lambda 1e-4  # 显式指定 reparam 参数
```

---

## 1. 运行前提

- 需要 **ctpwa 包**（含 CUDA 扩展）与 **NVIDIA GPU**；`torch.cuda.is_available()` 必须为 True。
- ⚠️ **必须在包含 `config.yml` 的目录下运行**。`fit.py` 在 **import 时**就用
  **当前工作目录的 `./config.yml`** 构造 `ctpwa.analysis()`；`--config` 参数只用于
  路径检查与相对路径解析，**不会**改变已经建好的 `ana`。因此换配置的正确做法是换
  目录（或在 cwd 放对应 `config.yml`），而不是只传 `--config`。
- 建议用 `python -u`（无缓冲）以便实时看日志。

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
  幅度吹大。
- polish 阶段按“当前 |A|”限制耦合步长、并对候选点做 `amp_max` 硬帽（S4）。

**优点**：收敛快（典型 ~1000–1800 次求值、~300–500 s/run）、NLL 好、振幅可控。
**代价**：`λ` 是对耦合幅度的弱先验，会轻微改变优化落点（默认 1e-4，影响很小）。

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
具体数据集为 `jlsp` 的 `single` 配置，仅供相对比较。

※ `reparam-tol-stop` = torch LBFGS 自身的容差（`--tol-grad/--tol-change`）停机，
**不等于**物理空间 KKT 收敛：实测约半数随机起点会停在坏盆地（NLL 可比正常解差
100~700）。多起点流程（best-of-N）下这没问题；**单次结果不要只看 status 判定成功**。

**如何选**：默认用 `reparam`。做 A/B 或复现历史行为时用 `--optimizer projected` 或
`--optimizer lbfgs`。若 reparam 偶发振幅偏大，可调大 `--amp-lambda`（如 1e-3）或调小
`--amp-max`。

### 3.5 推荐工作流（随机多起点 → 取最优 → polish）

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
3. **`--amp-max`（默认 1000）**要确认大于该分析的最大 `\|A\|`，否则会撞软墙；
   `--amp-lambda`（默认 1e-4）是只进优化目标的幅度罚项，报告 NLL 仍为真实值，
   设 0 可关闭。

---

## 4. 参数总表

优先级：**CLI > `FIT_*` 环境变量 > 默认值**。默认值以 `resolve_args()` 为准。

### 配置 / 日志
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--config` | `config.yml` 路径（见 §1 注意事项） | `config.yml` | — |
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

### 约束 / 数值稳定
| 参数 | 说明 | 默认 | 环境变量 | 适用优化器 |
|---|---|---|---|---|
| `--vmax` | 耦合幅度硬上界（`vmax`，防发散的保险丝） | 10000 | `FIT_VMAX` | projected / lbfgs |
| `--amp-max` | reparam 幅度软墙上限 `amp = amp_max·sigmoid(u)` | 1000 | `FIT_AMP_MAX` | reparam |
| `--amp-lambda` | reparam 幅度罚项 `λ·ΣA²`（只进优化 loss；0=关闭） | 1e-4 | `FIT_AMP_LAMBDA` | reparam |
| `--no-project` | 关闭 legacy 路径的边界梯度清零 | False | `FIT_PROJECT=0` | lbfgs |
| `--optimizer` | 优化器：`reparam`/`projected`/`lbfgs` | `reparam` | `FIT_OPTIMIZER` | — |
| `--opt-verbose` | 每轮打印 projected 的 `pg/active/ΔNLL` | False | `FIT_OPT_VERBOSE=1` | projected |

### 误差估计（Hessian 非正定回退）
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--err-mode` | `auto`=PD 直接求逆、否则伪逆；`strict`=原行为(非PD不给)；`pinv`=伪逆(丢 `λ<τλmax` 方向)；`psd`=把 `λ` 截到 `τλmax` 后求逆(更保守) | `auto` | `FIT_ERR_MODE` |
| `--err-tau` | 相对阈值 `τ`（`λ<τ·λmax` 视为平坦/不可测） | `1e-6` | `FIT_ERR_TAU` |

> 当精确 Hessian 近奇异/非正定（本模型常见，存在 ~20–30 维近平坦子空间）时，
> `auto` 自动退化为伪逆并给出参数误差，同时在日志打印 `mode=... / λmin / λmax /
> flat(λ<τλmax)=N`；输出会标注这是**秩亏下的可行估计，非严格统计误差**。
> `--err-mode strict` 可恢复“非正定就不给误差”的旧行为。

### Warm start
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--warm-start [路径]` | 用已收敛解作为 run 0 初值；不带路径时自动用 `results/best_params.pt` | None | `FIT_WARM=1` |

### Polish（二阶抛光）
| 参数 | 说明 | 默认 | 环境变量 |
|---|---|---|---|
| `--polish` / `--no-polish` | 是否在最佳点做 damped-Newton 抛光 | 开启 | `FIT_POLISH` |

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

---

## 5. 环境变量

每个 CLI 参数都有同名 `FIT_*` fallback，优先级 `CLI > 环境变量 > 默认值`：

```
FIT_RUNS  FIT_NITER  FIT_LR  FIT_TOL_GRAD  FIT_TOL_CHANGE  FIT_HISTORY_SIZE
FIT_VMAX  FIT_AMP_MAX  FIT_AMP_LAMBDA  FIT_PROJECT  FIT_OPTIMIZER  FIT_OPT_VERBOSE
FIT_ERR_MODE  FIT_ERR_TAU  FIT_POLISH  FIT_WARM  FIT_WAVES  FIT_EVENT_DATA
FIT_CHECKPOINT_INTERVAL
```

例：`FIT_OPTIMIZER=reparam FIT_AMP_LAMBDA=1e-4 python -u fit.py --runs 20 --niter 3000`

---

## 6. 输出文件

`--output-dir`（默认 `results/`）下：

| 文件 | 内容 |
|---|---|
| `parameters.txt` | 每个 run 的耦合（实/虚部、幅度、相位）与共振态参数 |
| `nll_history.txt` | 每个 run 的逐次求值 NLL 曲线 |
| `optimization_summary.txt` | 各 run 的 NLL / 迭代 / 耗时 / 正定性 / 优化器状态，及最佳解 |
| `best_params.pt` | 最佳参数（供 warm start） |
| `checkpoint.pt` | 续跑状态（`--resume` 用） |
| `weight_best.root` | 最佳解的权重文件（供 `plot.py` 画图） |

> `plot.py` 另外生成 `results_plot.pdf` 等图形，不属于 `fit.py`。

---

## 7. 已知问题与备注

1. **必须在含 `config.yml` 的目录运行**（见 §1）。
2. `reparam` 的 `--amp-lambda` 是作用在**优化 loss** 上的弱先验；报告/保存的 NLL、
   Hessian、误差全部基于**真实 NLL**，不受罚项影响。
3. 结果普遍 `正定性=False`：精确 Hessian 存在 ~20–30 维近平坦/简并子空间（诊断显示
   沿负特征值方向 NLL 两侧都不下降，说明是数值/秩亏而非真鞍点）。参数误差默认由
   **`--err-mode auto` 自动回退到伪逆**给出，并标注为“可行估计、非严格统计误差”；
   `--err-mode strict` 可恢复“非正定就不给误差”的旧行为。拟合分数/效率仍以真 PD 为门控。
4. `--lr` 只对 `reparam`/`lbfgs` 有意义；`projected` 不用 `lr`，其停机由
   `--tol-grad`（gtol）控制。
5. 若在集群提交作业，注意 `fit.py` 会读 cwd 的 `config.yml` 并需要 GPU；建议
   一个方案一个目录（或配合 `--output-dir` 隔离结果）。
