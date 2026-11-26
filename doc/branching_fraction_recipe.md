# 从拟合分数到分支比：文献公式 + 需要哪几个矩阵

结论：**算 BF_i 只需要"一侧"矩阵的完整表（含干涉），但要把重建侧和 truth 侧接起来（效率、ε̄），
两个矩阵都得存，而且要连"总积分 W"和"样本 Σw"一起存**。只有对角线不够——连文献定义的
拟合分数都算不出来，因为分母是相干的。

---

## 1. 文献怎么定义、怎么算

### 1.1 拟合分数（含干涉）

Laura++（arXiv:1711.09854）Eq. (22)-(23)：

```
FF_j  = ∫|c_j F_j|² dΦ / ∫|Σ_i c_i F_i|² dΦ
FF_ij = ∫ 2Re[c_i c_j* F_i F_j*] dΦ / ∫|Σ_i c_i F_i|² dΦ        (i < j)
```

原文："The sum of these fit fractions is not necessarily unity due to the potential presence
of net constructive or destructive interference."（[Laura++ §2.3](https://arxiv.org/abs/1711.09854)）

LHCb（arXiv:1902.07955）Eq. (4) 同款，并明确"fit fractions need not sum to unity"；
LHCb 最新（arXiv:2512.05102）Eq. (21)-(22) 还给出 CP 平均版本，并写明
"the sum over all fit and interference fractions is unity by definition"。

### 1.2 分支比怎么从拟合分数来

LHCb（arXiv:1902.07955）Eq. (5)：

```
B(B_s → K*K; K* → Kπ) = F_j × B(B_s → K⁰K±π∓)
```

即 **BF_i = FF_i × BF_total**，`FF_i` 用 1.1 的相干分母定义；干涉项不进任何单个 BF，
它解释"Σ BF_i ≠ BF_total"的那部分。同一篇文章里 LASS 的例子写得很直白：
"K0*(1430) 78%，有效程项 46%，两者破坏性干涉贡献 −24%"。

total 怎么来？LHCb（arXiv:2512.05102）Eq. (25)：用**效率修正后的产额比**

```
B(B → X) / B(B → norm) = (N_X / N_norm) · (ε_norm / ε_X) · 1/B(sub)
```

其中 ε 是**整样本的总效率**；而且原文特别说明模拟样本"reweighted by the amplitude
results of this analysis"——即效率要按**拟合出来的振幅形状**加权，这正是
`ε̄ = ∫ε|ΣA|²/∫|ΣA|²`。

### 1.3 效率只进似然，不进拟合分数

Laura++ Eq. (24)：信号 PDF 用 `|A|²ε / ∫|A|²ε dΦ`；而拟合分数对**平坦**相空间积分。
原因是效率只进归一化 → 拟合出的振幅已经是物理振幅 → FF 用无效率样本积分即可。
这也是 tf-pwa 用 `phsp_noeff`、我们用 `phsp_truth` 的道理。

---

## 2. 两个矩阵各自是什么

记（`M` 的第 i 行 j 列，i≠j 时为 2Re，i=j 时为 |A_i|²）：

| | truth 侧（无效率 MC） | 重建侧（带效率 MC） |
|---|---|---|
| 内容 | `T_ij = Σ_truth w·(2Re A_iA_j* 或 \|A_i\|²)` | `P_ij = Σ_phsp w·(…)` |
| 相干总额 | `W_T = Σ_i T_ii + Σ_{i<j} T_ij` = ∫\|ΣA\|² | `W_P` = ∫ε\|ΣA\|² |
| 归一化表 | `F_i = T_ii/W_T`, `I_ij = T_ij/W_T` | `F_i^rec = P_ii/W_P`, `I_ij^rec = P_ij/W_P` |
| 物理含义 | **文献里的 FF**（Eq. 21/22） | 探测器里的份额（模型期望值） |

两个样本的"同一事例形状"关系（都要除各自的样本量 `Σw`，否则只是积分之比）：

```
r_i = (P_ii/Σ_phsp w) / (T_ii/Σ_truth w)      ← 注意！这是 ε_i/ε_all，不是绝对 ε_i
ε̄   = W_P / W_T                              ← 当 phsp 是同一批产生事例被 cut 后的子集
     = (W_P/Σ_phsp w)/(W_T/Σ_truth w) × ε_all  ← 一般情况（ε_all = N_phsp/N_generated）
```

**效率的绝对归一化是个坑**（实测）：把 truth 随机切一半当 phsp（等价恒定 ε=0.5），
`getEfficiency` 返回 ~1 而不是 0.5 —— 因为它算的正是上面的 `r_i = ε_i/ε_all`。
绝对每波效率要用 `ε_i^abs = r_i × ε_all`，ε_all = 该样本的总接受率（cut 后/产生数）。
物理上这是对的：cut 后的样本分布 ∝ εdΦ/∫ε，任何"样本内平均"天然带 1/ε_all。
如果只做 `BF_i = BF_total × F_i`（下一节路线 A），压根用不到 ε_i，这个因子不影响结果。

**只有 truth 侧的对角线 × 2 是不够的**：`W_T` 依赖非对角项，否则你算的是
"非相干分母"的份额（我们现在的 `getFitFractions`），和文献的 FF 是两个量。

---

## 3. 算 BF 到底需要什么

### 3.0 先分清两种"道"

| 说法 | 需要什么 | 分波程序的作用 |
|---|---|---|
| **整个衰变道** `J/ψ→γX` 的 BF | `BF = N_sig/(ε̄·N_J/ψ·B_sub)` | 只提供 `ε̄` 里的模型权重（以及证明信号纯度）；矩阵本身不用 |
| **某个子道/共振态** `J/ψ→γR` 的 BF | `BF_i = BF_total × F_i` | 提供 `F_i`（truth 矩阵，**含非对角**） |

下面说的都是第二种。

### 3.1 路线 A：BF_i = BF_total × F_i（推荐，和文献口径一致）

需要：

1. truth 侧**完整矩阵**（含非对角）→ `F_i`；
2. `BF_total`：`BF_total = N_sig/(N_J/ψ · ε̄)`，或从归一化道比值来；
3. `ε̄`：需要两侧的 `W` 和 `Σw`（若两个样本是"同一批产生事例，cut 前/cut 后"，
   则 `ε̄ = W_P/W_T`，因为 `Σw_p/Σw_t = ε_all` 会自动相消）。

`F_i` 里的干涉项与单波 BF 无关；但报"总和"时要一起给 `I_ij`，否则读者会以为你漏了 20%。

### 3.2 路线 B：从数据权重直接算（等效，用得上重建矩阵）

```
N_i^rec = Σ_data w_i          w_i = |A_i|²/|ΣA|²      （数据侧逐事例权重）
BF_i    = N_i^rec / (ε_i · N_J/ψ)
```

`E[Σ_data w_i] = N_sig · P_ii/W_P`，所以它和路线 A 数学上等价（前提：拟合正确、
数据权重是扣过背景的信号权重）。重建侧矩阵在这里的作用是给 `ε_i` 和做数据-模型对比。

### 3.2b 结论：算"一个道"的 BF，truth 矩阵就够

```
BF_i = BF_total × F_i ,      F_i = T_ii / W_T ,   W_T = Σ_i T_ii + Σ_{i<j} T_ij
BF_total = N_sig / (ε̄ · N_J/ψ · B_sub)
```

- **重建矩阵不参与这个数**：它只用来①自洽地反推 `ε̄ = W_P/W_T`、`ε_i`；
  ②走 3.2 的数据权重路线；③做数据-模型对比。
- 唯一"和分波程序有关"的外部量是 `ε̄`：探测器效率本身与模型无关，
  但**按振幅形状平均**这一步是模型相关的（`ε̄ = ∫ε|ΣA|²/∫|ΣA|²`）。
  如果你的总效率是用平坦/生成器模型算的，就要么换成 `W_P/W_T`，
  要么把它当系统误差（两波量级差得多时能到百分量级）。
- 一个好消息：Laura++ §2.3 指出"同一两体组合里不同分波之间的干涉，
  在对螺旋角积分后相互抵消"。所以全角度、效率平坦时 `Σ_i F_i` 很接近 1，
  真正剩下的干涉项只是同分波/不同两体组合的那些——单看一个道不受影响。

### 3.3 "两个矩阵够不够"

| 你想得到的量 | truth 矩阵 | 重建矩阵 | 还需要 |
|---|---|---|---|
| `F_i`、`I_ij`（文献口径） | ✅ 完整表 | — | — |
| 单波效率 `ε_i` | 对角 | 对角 | 两侧 `Σw` |
| `ε̄` | `W_T` | `W_P` | 两侧 `Σw`（或同源样本） |
| `BF_i` | ✅ | 可选（路线 B） | `BF_total`/`N_sig`、`N_J/ψ`、子衰变 B |
| 数据-模型对比 | — | ✅ | 数据侧 `Σ_data w_i`（背景已扣） |

所以：**是，这两个矩阵（各含完整表）+ 两个 `Σw` 就够所有"相对量"**；绝对 BF 还需要产额和
`N_J/ψ`，这本来就不该从矩阵里来。

---

## 4. ctpwa 现状（本仓库）

| 量 | 接口 | 说明 |
|---|---|---|
| 非相干份额（Σ=1） | `getFitFractions(v)` | 分母 `Σ_i ∫|A_i|²`，**不是**文献 FF |
| truth 侧完整矩阵 | `getFitFractionMatrix(v)` / `sample="truth"`（默认） | 原始积分 `T_ij` |
| 重建侧完整矩阵 | `getFitFractionMatrix(v, sample="phsp")` | 原始积分 `P_ij` |
| **ROOT 里的两张归一化表** | `writeResult(...)` 的 `interference_rec` / `interference_gen` | **各自除以本样本 W**（元素 O(1)，`trace+Σ_{i<j}=1`），吃各自权重；乘回 `W` 即原始积分 |
| 归一化元数据 | `writeResult(...)` 的 `integrals` 树 | `N_rec`/`N_gen`（Σw）、`W_rec`/`W_gen`（相干总额）、`npartials` |
| 矩阵的参数误差 | `writeResult(..., hessian=H)` 的 `interference_rec_err` / `interference_gen_err` | `σ(F_ij)=sqrt(J·H⁻¹·Jᵀ)`，F 为归一化元素；**不含** MC 统计与模型系统 |
| 原始积分 | 两张表 × 对应 `W` | `M_ij = interference_*[i][j] × W_*` |
| 每波效率 | `getEfficiency(v)` | 内部就是 `(P_ii/N_p)/(T_ii/N_t)`，带误差（注意是相对值，§2） |
| 重建侧逐事例权重 | `writeResult(..., is_saved_weight=1)` 的 `saved_weight` 树 | 逐 phsp 事例的 `weight_<i>`/`totalweight`（**不是**数据侧） |
| 数据侧逐事例权重 | 暂无输出 | 需要 `Σ_data w_i` 的话得自己加；走路线 A 用不到 |

已验证（`tests/configs/eff2.yml`，`phsp == phsp_truth`，随机耦合）：

```
T 矩阵 == P 矩阵（同一批事例）            max|Δ| = 4e-10
writeResult:interference == P/W_P         逐元素完全相等
F + I = 1                                 （0.7725 + 0.2925 − 0.0650 = 1）
```

复现：

```python
import numpy as np, ctpwa
ana = ctpwa.analysis("configs/eff2.yml")
# v: 自由耦合向量（无 trans 约束时就是 extended vector）
T = ana.getFitFractionMatrix(v).numpy()              # truth
P = ana.getFitFractionMatrix(v, sample="phsp").numpy()  # 重建
W = lambda M: M.diagonal().sum() + M[np.triu_indices(len(M), 1)].sum()
print("FF(文献) =", (T.diagonal() / W(T)).tolist())
print("I_ij     =", (T - np.diag(T.diagonal()))[np.triu_indices(len(T), 1)] / W(T))
print("ε_i      =", (P.diagonal()/N_p) / (T.diagonal()/N_t))
```

---

## 5. 还剩下的坑（按重要性）

1. ~~矩阵 / W / Σw 没进 ROOT~~ **已完成**：`writeResult` 现在写
   `interference_rec`、`interference_gen`（各自除以本样本 W）+ `integrals`
   （`N_rec`/`N_gen`、`W_rec`/`W_gen`、`npartials`）；`CTPWA_WR_NO_INTEGRALS=1` 可跳过。
2. **θ（共振态质量/宽度）现在会进中心值，但不进误差**。`writeResult` /
   `getFitFractionMatrix` 的矩阵会按当前拟合的 θ 重算（`reCalcAmp` / `getNLL` /
   `getHessian` 都会记录 θ）；但参数误差的 Jacobian 仍只对耦合方向求导，
   θ 的贡献还没加进去（见 `doc` 末尾 TODO）。
3. **效率样本的"产生数"要自己留档**。`ε_all = N_phsp/N_generated` 只能从 MC 记录里拿
   —— 矩阵里的 `N_phsp` 是 cut 后的 Σw。走 `BF_total × F_i` 时用不到它；走数据权重
   路线才需要（见 §2 的坑）。
4. **积分区域必须和数据一致**。LHCb Eq. (21) 把积分限制在分析区（`m_Kπ ∈ [633,1042]`）。
   我们的 `phsp_truth` 如果是全相空间产生样本，而数据有质量窗/PID 等 cut，
   那 `F_i` 的定义区域就不是数据区；要么给 truth 样本加同样的 cut，要么在矩阵里做区域限制。
5. **CP/电荷共轭平均**：LHCb Eq. (21) 是对 B 与 B̄ 求和。我们多链 + `identical` 的配置里，
   partial 已经按链区分，做 CP 平均要显式把对应 partial 的振幅相加后再算，不能加 FF。
6. **背景**：路线 B 的 `Σ_data w_i` 必须是信号权重（sWeight 或边带扣除）；
   我们 `bkg_weights` + `bkg` 走的是 NLL，不会自动进 `weight_<i>`。
7. **误差矩阵只含参数误差**。`interference_*_err` 是线性化传播（Jacobian 对自由耦合 +
   自由 θ 做 ±ε 差分，`Cov = H[free,free]⁻¹`）。Hessian 非正定时**不写**并告警（不会给
   假误差）。MC 统计误差按你的口径算作系统项，不在里面。
8. **MC 统计误差**：`I_ij` 是两个大数之差量级的东西，truth 样本不够时 `Σ F + Σ I = 1`
   会漂；报告前用 `getEfficiency` 同款的 `Σ(w f)²` 估一下。

---

## 6. 怎么验证矩阵和误差（含真实数据）

**矩阵本身**（都写在 `tests/test_matrix_validation.py`，可直接对任何配置跑）：

| 检验 | 判据 | 能抓到的错 |
|---|---|---|
| 闭合 | `trace + Σ_{i<j} − 1` | 分母错、漏项 |
| Cauchy–Schwarz | `\|I_ij\| ≤ 2√(F_iF_j)` | 标度/归一化错、干涉项符号 |
| 半正定 | `eigvalsh(F) ≥ 0`（F = 2Re Gram） | 索引错位、per-event 权重错 |
| 规范不变 | 耦合整体 ×c、×e^{iφ} → F 不变 | 归一化里混进未抵消的 |v|² |
| 同一文件 | `phsp == phsp_truth` ⇒ rec ≡ gen | rec/gen 弄反/权重口径不一致 |
| 独立 kernel | rec 对角 = `saved_weight` 树的 `Σweight_i/Σtotalweight` | 两条实现（computeResults vs 分支比 kernel）任一算错 |
| batching | `CTPWA_TRUTH_BATCH=700` 与默认结果一致 | 分批/offset 错 |

**误差**：

| 检验 | 判据 | 说明 |
|---|---|---|
| 协方差代数 | `H → 4H` ⇒ `σ → σ/2` | H⁻¹ 与方向映射 |
| FD 稳定性 | ε 减半，σ 变化 < 1e-3 | 差分截断误差没污染 |
| toy-MC | 参数按 Cov 抖动，F 的经验 std = σ | 线性化传播；实测 eff2 上中位偏差 1.7% |
| 非正定 | 不写矩阵 + 告警 | 不给假误差 |

真实数据上唯一"物理"的对照是**数据侧**：`Σ_data w_i / N_data`（sWeight 后的信号权重）应等于
`interference_rec` 的对角。这个还没输出（见 §5 第 6 条），需要时再加。

---

## 参考文献

- J. Back et al., *Laura++: a Dalitz plot fitter*, [arXiv:1711.09854](https://arxiv.org/abs/1711.09854)
  §2.3 Eq. (22)-(23)（FF 与干涉项定义）、Eq. (24)（效率只进 PDF 归一化）。
- LHCb, *Amplitude analysis of B_s⁰→K_S⁰K±π∓*, [arXiv:1902.07955](https://arxiv.org/abs/1902.07955)
  Eq. (4)（FF 定义，"need not sum to unity"）、Eq. (5)（**BF = FF × BF_total**）。
- LHCb, *B⁰_(s)→K\*⁰K̄\*⁰ branching fractions and polarisations*,
  [arXiv:2512.05102](https://arxiv.org/abs/2512.05102) Eq. (21)-(22)（CP 平均 FF + 干涉，
  "sum ... is unity by definition"）、Eq. (25)（效率修正产额比；模拟按振幅结果重加权）。
- tf-pwa `tf_pwa/fitfractions.py`（同样的定义，`FF_ij` 用 `|A_i+A_j|²-|A_i|²-|A_j|²` 实现）。
