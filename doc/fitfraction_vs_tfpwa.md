# 拟合分数 / 分波效率：ctpwa 与 tf-pwa 口径对照

对照对象：

- ctpwa：本仓库（`src/Analysis.cu::getFitFractions` / `getEfficiency`）
- tf-pwa：`/home/whitewash/pkgs/tf-pwa`（`tf_pwa/fitfractions.py`、`tf_pwa/applications.py`）

结论先行：

| | ctpwa | tf-pwa |
|---|---|---|
| 拟合分数分母 | **非相干和** `Σ_i ∫|A_i|²` | **相干和** `∫|Σ_i A_i|²` |
| 干涉项 | 不进分母（内部有，未参与） | 单列 `FF_ij`，`Σ_i FF_i + Σ_{i<j} FF_ij = 1` |
| `Σ_i FF_i` | 恒等于 1 | `= 1 - Σ_{i<j} FF_ij`，可 >1 或 <1 |
| "分波"粒度 | (链 × 共振态组合) | 共振态（跨链合并） |
| 效率 | 内置 `getEfficiency` | **没有**（效率以逐事例权重进 NLL） |

两者差一个可观测量——干涉份额 `f_int`（见 §1.2），不是因子 2/符号之类的记账差别。

---

## 1. 拟合分数

### 1.1 定义

tf-pwa（`tf_pwa/fitfractions.py:200-210` 的 docstring 就是定义）：

```
FF_i    = ∫|A_i|² dΦ / ∫|Σ_j A_j|² dΦ
FF_ij   = ∫2Re(A_i A_j*) dΦ / ∫|Σ_j A_j|² dΦ
          = [∫|A_i+A_j|² - ∫|A_i|² - ∫|A_j|²] / ∫|Σ A|²
```

注意 `A_i` 是"第 i 个共振态在所有链上的振幅之和"。表闭合：
`Σ_i FF_i + Σ_{i<j} FF_ij = 1`。

ctpwa（`src/Analysis.cu:4685`）：

```
FF_i = Σ_e w|A_i(e)|² / Σ_j Σ_e w|A_j(e)|²         （e 跑 truth 样本，w 为权重）
```

`Σ_i FF_i = 1` 是构造出来的，不是物理闭合。

### 1.2 两者的换算（只差一个数）

记

```
I_i  = ∫ w|A_i|²          （ctpwa 的分子）
I_ij = ∫ w·2Re(A_iA_j*)   （ctpwa 内核已经算了，见 §1.3）
f_int = Σ_{i<j} I_ij / Σ_i I_i
```

则

```
FF_i^tf-pwa = FF_i^ctpwa / (1 + f_int)
Σ_i FF_i^tf = 1 / (1 + f_int)
```

`f_int` 就是"干涉占非相干总和的比例"。`f_int → 0`（单波主导、或相位抵消）时两套数字一致；
两波量级相当且相位对齐时，差别可以到 10~20% 甚至更大。

### 1.3 ctpwa 其实已经算了全部干涉项

`computeBFKernel`（`src/ComputeBF.cu:28-95`）一次扫描同时产出

- `partial[i]  = Σ w|A_i|²`
- `scattering[i][j] = Σ w·2Re(A_iA_j*)`（对称，对角线是 `2|A_i|²`）

但 `getFitFractions` 只用了 `partial`，`scattering` 填完就丢。为此新增了
`getFitFractionMatrix(vector)`（`src/Analysis.cu:4762`），把原始矩阵导出来：

```python
M = ana.getFitFractionMatrix(coupling)      # [npartials, npartials] float64
# M[i][i] = I_i,  M[i][j] = I_ij (i≠j, 2Re)
W  = M.diagonal().sum() + M.triu(1).sum()   # = ∫|ΣA|²（不要用 M.sum()，上三角会重复计入）
ff_ours = M.diagonal() / M.diagonal().sum() # ctpwa 口径
ff_tf   = M.diagonal() / W                  # tf-pwa 口径（对角）
ff_ij   = M.triu(1) / W                     # tf-pwa 口径（干涉项）
```

### 1.4 数值示例：`tests/configs/eff2.yml`（单链，φ(1020)+φ(1680) 两个 [1⁻] 波）

相位扫描（耦合取 `v = (1, r·e^{iφ})`；`f_int`、两套 FF 用上面的矩阵算）：

| r | φ | I₁ | I₂ | I₁₂ | f_int | FF₁(ctpwa) | FF₂(ctpwa) | FF₁(tf) | FF₂(tf) | Σ FF(tf) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0° | 53906 | 344262 | -32538 | -0.0817 | 0.1354 | 0.8646 | 0.1474 | 0.9416 | 1.0890 |
| 1.0 | 30° | 53906 | 344262 | -298 | -0.0007 | 0.1354 | 0.8646 | 0.1355 | 0.8653 | 1.0007 |
| 1.0 | 60° | 53906 | 344262 | +32023 | +0.0804 | 0.1354 | 0.8646 | 0.1253 | 0.8003 | 0.9256 |
| 1.0 | 90° | 53906 | 344262 | +55763 | +0.1400 | 0.1354 | 0.8646 | 0.1188 | 0.7584 | 0.8772 |
| 1.0 | 120° | 53906 | 344262 | +64561 | +0.1621 | 0.1354 | 0.8646 | 0.1165 | 0.7440 | 0.8605 |
| 0.5 | 0° | 53906 | 86065 | -16269 | -0.1162 | 0.3851 | 0.6149 | 0.4358 | 0.6957 | 1.1315 |
| 0.5 | 120° | 53906 | 86065 | +32281 | +0.2306 | 0.3851 | 0.6149 | 0.3129 | 0.4996 | 0.8126 |

读法：同样一组耦合，两套定义下单个波的份额最多差 **~25% 相对**；`Σ FF(tf)` 从 0.81 到 1.13。
真实分析里干涉通常只有百分之几，差异同量级（`f_int ≈ 2%` → 每个 FF 偏 2% 相对）。

### 1.5 "分波"的粒度不同（多链时会撞上）

- ctpwa：partial = **(链, 共振态组合)**，见 `src/Info.cu:317`（每个 chain×res_combo 一个）。
- tf-pwa：`set_used_res([res])` 把**所有含该共振态的链**打开
  （`tf_pwa/amp/core.py:2103-2126`），所以是**按共振态**合并。

实例 `tests/configs/with_trans.yml`（两条链，R_KK 与 R_Keta）：

```
ctpwa 6 个 partial:
  0 Jpsi→eta+phi1020     1 Jpsi→eta+phi1680
  2 Jpsi→Kp+K1_1410      3 Jpsi→Kp+K1_1680
  4 Jpsi→Km+K1_1410      5 Jpsi→Km+K1_1680        ← K1± 两个电荷指配合计 4 个
tf-pwa 4 个共振态: phi1020, phi1680, K1_1410(2 条链合并), K1_1680(2 条链合并)
```

同一组耦合下（`f_int = -0.0938`）：

| ctpwa partial | FF(ctpwa) | FF(tf) |
|---|---|---|
| phi1020 | 0.5953 | 0.6569 |
| phi1680 | 0.2127 | 0.2347 |
| K1_1410 (Kp 侧) | 0.0063 | 0.0069 |
| K1_1680 (Kp 侧) | 0.0898 | 0.0991 |
| K1_1410 (Km 侧) | 0.0063 | 0.0070 |
| K1_1680 (Km 侧) | 0.0895 | 0.0988 |
| Σ | 1.0 | 1.1035 |

要和 tf-pwa 的"共振态表"对齐，还要把 ctpwa 的 K1 侧两行**先相加**（振幅相干相加，不能只加
FF——严格做法是把这些 partial 的振幅先合并成一个"共振态振幅"再算积分）。

### 1.6 样本与权重

| | ctpwa | tf-pwa |
|---|---|---|
| 样本 | `phsp_truth`；缺 → 返回空 `[0,2]` | `phsp_noeff`；缺 → **退化为第一个 `phsp`** 并告警（`config_loader/data.py:366`） |
| 权重 | `phsp_truth_weights`（常数 / dat / ROOT） | 数据字典里的 `weight`（`phsp_noeff_weight`） |
| 积分 | `Σ w·f`，平方项 `Σ(w·f)²` | `_eval_integral`: `Σ w·f`、`Σ(w·f)²`（同款） |
| 归一化 | 比值，`Σw` 相消 | 比值，同样相消 |

两边权重语义一致（都是"事件多重数"式的加权积分，`w≡1` 时退化为等权）。

### 1.7 误差传播

| | ctpwa | tf-pwa |
|---|---|---|
| 微分方式 | 对耦合做 ±ε 数值差分（`ε=5e-6·max(|p|,1)`） | TF 自动微分 `GradientTape` |
| 变量集 | 只有自由耦合 `2n`，且 mask 掉参考方向（`Re v_0`、`Im v_0`） | `amp.trainable_variables` 全部（默认只有耦合；把质量/宽度设成 free 时也含它们） |
| 协方差 | 耦合块 `H⁻¹`（`computeCouplingHessian` 或调用方传入的统一 Hessian 的左上块） | 调用方给的 `inv_he`（`error_matrix.npy`，`pinv`/`force_pos_def` 处理过） |
| 非正定 | `min eig > 1e-8` 不满足 → **静默返回误差 0** | 用 `inv_he`，没有额外判定（`force_pos_def` 在算误差矩阵时告警+正则） |
| MC 统计项 | FF 不加（只有效率加） | `add_int_error=True` 时加 `sqrt(Σ(w f)²)/I_tot²`，默认 **关** |
| 公式 | `sqrt(diag(J H⁻¹ Jᵀ))` | `sqrt(gᵀ · inv_he · g)` |

两个需要留意的点：

1. **ctpwa 的 FF 误差不含 θ 方向**（共振态质量/宽度）。`getFitFractions` 只把 Hessian 的
   左上 `2n×2n` 耦合块取出来用，Jacobian 也只对耦合求。θ free 时这是低估；
   tf-pwa 只要 θ 在 `trainable_variables` 里就自然包含。
2. ctpwa 在 Hessian 非正定时**不报错、直接给 0 误差**，容易被误读成"误差很小"。

### 1.8 计算量

- tf-pwa：`set_used_res` 切链后对每个 `(i, i)` 和 `(i, j)` **重新算一遍全样本振幅**
  → `n(n+1)/2` 次（n=20 波时 210 次；`batch=25000` 再乘批数）。
- ctpwa：`phsp_truth` 一次扫描，`S = A·v` 后同时累加 partial 与 scattering
  （一次 `gemv` + 一次 `dgmm` + 一个 kernel），干涉项是免费副产品。

---

## 2. 分波效率

### 2.1 tf-pwa 没有这个函数

在 `/home/whitewash/pkgs/tf-pwa` 里搜不到 `cal_eff` / `get_efficiency`。效率有两条路进拟合：

- **默认（`model: default`）**：`phsp` 直接给"带效率的样本"（配合 `phsp_weight`
  逐事例权重），归一化积分自然带 ε；
- **`model: cfit`**：`phsp` 逐事例带 `eff_value`（`model/cfit.py:24` 的 `f_eff`），
  积分用 `sig = eff·Amp`（`model/cfit.py:48-90`）。

两条路都只影响**归一化积分**，拟合本身是效率修正过的；`phsp_noeff` 只用来算拟合分数
（`config_loader/data.py:366`）。每个分波的效率要报的话，是用户脚本自己按
`Σ_phsp ε|A_i|² / Σ_phsp_noeff |A_i|²` 现算的。

所以 ctpwa 的 `getEfficiency` 不是"复刻 tf-pwa 的某个函数"，而是把 tf-pwa 用户脚本里
那段手算**内置**了，并补了误差。

### 2.2 ctpwa 的定义（`src/Analysis.cu:4808`）

```
ε_i = (Σ_phsp w|A_i|² / Σ_phsp w) / (Σ_phsp_truth w|A_i|² / Σ_phsp_truth w)
```

⚠ 这个比值等于 `ε_i^abs / ε_all`（ε_all = 样本总接受率），不是绝对每波效率：
cut 后的 phsp 样本分布 ∝ εdΦ/∫ε，样本内平均天然带 1/ε_all。实测把 truth 随机切一半
当 phsp（恒定 ε=0.5）时，返回 ≈1 而不是 0.5。绝对效率 = 返回值 × ε_all。
只做 `BF_i = BF_total × F_i` 时用不到它，所以不影响分支比；详见
[branching_fraction_recipe.md](branching_fraction_recipe.md)。

- 依赖拟合出的耦合（`A_i` 含共振参数），必须用拟合后的 vector；
- 权重：`phsp_weights` / `phsp_truth_weights`，缺省 w≡1；
- 误差 = **参数误差**（耦合块 Jacobian@H⁻¹@Jᵀ，同 FF 机制）⊕ **MC 统计误差**
  （`σ²/ε² = S_eff/I²_eff + S_tr/I²_tr`，`S=Σ(w|A_i|²)²`，tf-pwa `add_int_error` 同款）。

### 2.3 样本角色对应

| ctpwa | tf-pwa 对应物 | 说明 |
|---|---|---|
| `phsp`（cut 后 MC） | `phsp` + `eff_value`（或 `phsp_weight`） | 拟合归一化用的带效率样本 |
| `phsp_truth`（无效率 MC） | `phsp_noeff` | 只用于 FF / 效率分母 |
| `phsp_weights` | `phsp_weight` | 归一化分子权重 |
| `phsp_truth_weights` | `phsp_noeff_weight` | FF/效率权重 |

归一化积分三家其实是同一个量：ctpwa 用 `Σw|A|²/Σw`，tf-pwa 默认模型用
`mean(ε|A|²) = Σε|A|²/N`；等权时逐位一致，带权重时只差一个与参数无关的常数因子
（对 NLL 只差常数、对梯度/Hessian 无影响）。

### 2.4 什么时候会不一样

1. **分母归一化**：ctpwa 用 `Σw`；tf-pwa 手动算的时候常写 `/N`（等权时相同，带权重时不同）。
2. **误差**：tf-pwa 那套没有效率误差的现成实现；ctpwa 的 MC 统计项用 `Σ(w|A|²)²`，
   对重要性权重是正确的一阶；对"多重数"权重会偏大/偏小（见 CHANGELOG 的说明）。
3. **口径**：ctpwa 的 ε_i 是"分波加权效率"，分母是**非相干** `Σ_i∫|A_i|²`；
   若要报"某个共振态的探测效率"，这和 tf-pwa 里按链积分再相除的做法在强干涉时会差一点
   （同样是 `f_int` 量级）。

---

## 3. 复现

### 3.1 ctpwa 侧

```python
import torch, ctpwa
ana = ctpwa.analysis("tests/configs/eff2.yml")   # 双波，phsp == phsp_truth
n = ana.getNVector()
c = torch.zeros(n, dtype=torch.complex128, device=ana.getPhspTensor().device)
c[0] = 1.0
c[1] = 0.5j                                       # 两波耦合 (1, 0.5i)

M = ana.getFitFractionMatrix(c)                   # 原始积分矩阵
W = M.diagonal().sum() + M.triu(1).sum()          # ∫|ΣA|²
print("ff ctpwa :", (M.diagonal() / M.diagonal().sum()).tolist())
print("ff tf-pwa:", (M.diagonal() / W).tolist())
print("interf   :", (M.triu(1) / W).tolist())
print("f_int    :", float(M.triu(1).sum() / M.diagonal().sum()))
```

注意 `getFitFractionMatrix` 的口径：
`M[i][i] = Σw|A_i|²`、`M[i][j] = Σw·2Re(A_iA_j*)`（i≠j，对称）。
**不要用 `M.sum()`**——上三角的 `2Re` 会被重复计入；总数是 `trace + Σ_{i<j} M_ij`。

### 3.2 tf-pwa 侧（本机 `~/miniconda3/envs/tfpwa/bin/python`，TF 2.21 可跑）

等价的 toy 配置（`J/ψ → η R_KK`, `R_KK → K⁺K⁻`，两个 BWR）：

```yaml
data:
  dat_order: [Kp, Km, eta]
  data: [".../tests/data/test_data.dat"]
  phsp: [".../tests/data/test_phsp.dat"]
  phsp_noeff: [".../tests/data/test_phsp.dat"]
decay:
  J/psi: [[eta, R_KK]]
  R_KK: [Kp, Km]
particle:
  $top: {J/psi: {J: 1, P: -1, mass: 3.0969}}
  $finals:
    eta: {J: 0, P: -1, mass: 0.5478}
    Kp:  {J: 0, P: -1, mass: 0.4937}
    Km:  {J: 0, P: -1, mass: 0.4937}
  R_KK: [phi1020, phi1680]
  phi1020: {J: 1, P: -1, mass: 1.0195, width: 0.0045}
  phi1680: {J: 1, P: -1, mass: 1.66, width: 0.125}
```

```python
from tf_pwa.config_loader import ConfigLoader
c = ConfigLoader("cfg.yml")
frac, err = c.cal_fitfractions({}, c.get_phsp_noeff())
# {'phi1020': 0.9886, 'phi1680': 0.0078, ('phi1680','phi1020'): 0.0036}  → Σ = 1
```

（默认耦合是 `config_loader` 里 `fix_chain_val` 的随机数，见
`config_loader.py::add_decay_constraints`；要固定就写 `constrains: decay: fix_chain_val: ...`。
两边的耦合参数化不同，**绝对数值不能直接对比**，能对比的是定义与上面的换算关系。）

---

## 4. 建议

1. **对外报数**：如果目标是和 tf-pwa / 实验组的口径一致，报 `FF_i^tf = I_i/∫|ΣA|²` 和
   `FF_ij`，并注明 `Σ_i FF_i ≠ 1`。ctpwa 现在默认给的是非相干口径（`Σ=1`），
   换算只需 `f_int` 一个数（§1.2）。
2. **BF 转换**：`BF_i = BF_total × FF_i` 在 tf-pwa 口径下才有"第 i 个共振态占总衰变率的比例"
   的含义；ctpwa 口径下 `Σ_i BF_i = BF_total` 但单项含义不同。
3. **多链模型**：ctpwa 的 partial 是 chain-specific，和 tf-pwa 的共振态表对齐前要先合并
   （振幅相干合并后再积分）。
4. **误差**：θ free 时 ctpwa 的 FF 误差偏小（只走耦合块）；如果要报严格误差，
   要么固定 θ，要么等我们把 θ 方向接进 `getFitFractions` 的 Jacobian。
5. 需要的话可以再加：`getFitFractions` 直接给 tf-pwa 口径（多一个 `denominator="coherent"`
   开关），以及把 `FF_ij` 表导出成 `writeResult` 那样的 ROOT/文本。
