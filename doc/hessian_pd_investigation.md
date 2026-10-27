# Hessian 非正定排查（2026-10-27）

> 环境：`CTPWA_DEVICE=cpu`（double）与 `CTPWA_DEVICE=cuda`（RTX 5060 8GB，
> double）双后端复测。生产 config 的数据/phsp 完整可读。

## 0. 结论

**结论要分两类。**

* 对 `J/ψ→K⁺K⁻η` 这类每个顶点只有一个 LS 的过程：当前 `getHessian`
  没有写错；随机点、闭包极小点、生产拟合点上解析 Hessian 与有限差分
  逐元/逐方向一致。用户看到的“不正定”主要是**判据问题**（下面 1~4）。
* 对 `J/ψ→ωηη′` 这类多 LS、多链、带大 bkg 的过程：`getHessian` 确实有
  一个 **bkg 符号 bug**（下面 5），已修复；修复后 H 与 FD 一致且正定。

判据问题/优化器问题：

1. `getHessian` 返回的是**无约束** NLL 的 Hessian。拟合参数带 `free_range`
   时，最优点可能贴边；沿“朝外”方向的负曲率是边界效应，不是拟合点不稳定。
   旧 `fit.py` 对**全自由块**直接 `all(eig > 0)`，把这类合法的约束极小误报
   成“不正定”。
2. `getHessian` 的固定参考耦合 `Re_0=1`、`Im_0=0` 对应两行/列恒为 0；
   不剔除就求特征值，必然看到零/负的数值噪声。
3. 另一部分“不正定”是**真的没收敛**：例如 `--optimizer projected` 跑满
   max-iter 时梯度仍非零，H 在非驻点处当然可能不定；或者 `best_params.pt`
   来自旧 config，当前 config 下 NLL 已不是极小。
4. 旧 `fit.py` 在 `optimize_single_run` 里把**贴边参数**也纳入求逆，并且按
   交错 `[Re,Im,Re,Im]` 取耦合误差；实际布局是 grouped
   `[Re..., Im..., θ]`。这会把 Re/Im 误差错位（已在本次修复）。
5. **`getHessian` 本身还有一个 bkg 符号 bug**：耦合变换的二阶项
   `Σ g_v·∂²v` 里，bkg 的 `g_v` 被翻了两次符号。bkg 小时 FD 测试看不出来；
   `J/ψ→ωηη′`（69 振幅、3074 bkg vs 2024 data）上会直接把 H 从 PSD
   变成 9 个负本征值。详见 §4.5。

## 1. 方法

### 1.1 解析 Hessian vs 有限差分

- 对每个配置取 `tests/conftest.py::make_params` 的合法参数点；
- 用 `torch.autograd.grad(getNLL)` 做中心差分得到 `H_fd`；
- 用 `tests/fd_utils.compare_hessian` 过滤 FD 噪声，只判定“显著且两个步长
  自洽”的矩阵元。步长 `h=1e-3`（更小步长受 NLL 浮点噪声限制）。

### 1.2 DIYMC 闭包

- 用 `ctpwa.PhaseSpaceGenerator` 产生平坦 phsp，`analysis.generateMC(...,
  mode="unweighted")` 从已知真值产生 toy 数据；
- 用 `analysis.writeMC` 写 DAT，替换 config 的 `Data.data/phsp`；
- 用 scipy / `example/fit.py` 的优化器拟合，在拟合点重算 H。

## 2. 解析 Hessian 校验

| config | 模型/结构 | verdict | max rel (h=1e-3) |
|---|---|---|---|
| simple | BWR | PASS | 6e-5 (h=1e-4) |
| with_trans | 3 链 + trans + BWR | PASS | 5.3e-4 |
| hess_mult | 3 因子乘法链 | PASS | 4.2e-3 |
| flatte | Flatté | PASS | 2.8e-6 |
| gs | Gounaris-Sakurai | PASS | 5.2e-6 |
| constraints | 2 个自由 BWR | PASS | 1.9e-3 |
| ident2 | 全同粒子 | PASS | 2.9e-5 |
| custom | Custom DSL | PASS | 8.0e-5 |

另外 BWR 解析特化 vs 字节码（`CTPWA_NO_BWR_SPEC=1`）逐元相对差
`5e-16`（simple），Flatte/GS 完全一致——二阶闭式没有写错。

## 3. 生产点复现

生产目录：`~/pwa/Jpsi2KKeta/solve2`（config 中 19 个自由耦合 + 10 个 θ）。

### 3.1 旧 `best_params.pt` 已与当前 config 不一致

在**当前** config 下：

```
NLL(best_params) = -2475.33     # 而 results/optimization_summary.txt 是 -3464.98
max|grad|         = 1.05e4
H 全自由块         ：10 个负本征值，λmin = -2.16e4
```

把 `best_params.pt` 当作“已收敛点”检查 H，当然会得到不正定——这是
**config 改过 / 结果目录过期**，不是 Hessian 错。

### 3.2 从该点重新优化（reparam，80 iter）

> 这一段是 **bkg 符号修复前**的 J/ψ→K⁺K⁻η 复跑。修复后同一附近点的
> active 数在 1~2 之间（bkg 只占 ~2%，对结论无影响），可行子空间仍正定。

```
NLL = -3464.9816       （与历史最佳 -3464.983 一致）
全自由块 λmin = -0.132，1 个负本征值
可行子空间 λmin = +0.409，0 个负本征值
active = 2：θ(K1_1410_mass) 在下界 1.300，
            θ(K1_1410_width) 在上界 1.000
```

负本征向量最大分量正是 `K1_1410_width`（贴在上界）——这是**朝外方向**的
负曲率。旧判据报 False；按 KKT 二阶条件（剔除贴边且梯度朝外的方向）报 True。

### 3.3 同一生产点沿最负 5 个方向的 FD 校验

| 方向 k | λ_k | vᵀHv | FD 曲率 (h=1e-3) | 相对差 |
|---|---|---|---|---|
| 0 | -21620.9 | -37334.6 | -37496.1 | 0.43% |
| 1 | -7942.6 | -26833.0 | -26989.5 | 0.58% |
| 2 | -2953.2 | -8943.2 | -8941.5 | 0.02% |
| 5 | -480.9 | -1701.2 | -1703.9 | 0.16% |
| 10 | +12.6 | +30.6 | +30.5 | 0.43% |

结论：负本征值不是 `getHessian` 的数值错误，是真实的无约束二阶曲率。

### 3.4 projected 路径

`--optimizer projected` 跑 80 iter 只到 NLL=-3463.77，`max|grad|=20.4`，
无活跃边界，H 仍有 2 个负本征值。这不是边界效应，是**没收敛到驻点**；
需要更多迭代 / polish，或换 reparam。

## 4. DIYMC 闭包：phsp 分辨率也会造成“假不正定”

用一个含 4.5 MeV 宽 φ(1020) 的 3 链模型，**同一份 5k toy 数据**：

- **10k phsp**：toy 真值处 NLL=-18189，拟合却降到 -21801，宽度被推到边界；
  真值处 H 有 3 个大的负本征值。生成样本的 Dalitz 分布与 10k phsp 加权
  分布明显不符（窄峰没被采到）→ 归一化 MC 分辨率不足。
- **200k phsp（同一窄模型，只换 phsp）**：NLL(truth)=-14302.26；
  BFGS 极小 NLL=-14306.57（Δ=4.3），MLE 处 H 本征值
  `[0, 0, +87, +447, ...]`（两个零对应固定参考 Re0/Im0 的规范方向），
  θ=(1.6571, 0.1403, 1.3991, 0.1448) 与真值一致。
  （L-BFGS-B 在真值处直接“收敛”停住，是量纲/步长问题；BFGS 才找到 MLE。）

所以：**toy/闭包检查时 phsp 必须能分辨最窄共振**；否则真值不是 NLL 极小，
H 在真值处为负是正常的统计/归一化效应，不是 Hessian 写错。

## 4.5 `J/ψ→ωηη′`：多 LS + 大 bkg 暴露的 `getHessian` 符号 bug

目录：`~/pwa/Jpsi2omegaetaetap`。config 有 9 条链、69 个振幅、21 个自由耦合、
0 个自由共振态参数；bkg 3074 事件、data 2024 事件。

在旧 `.so` 上取 `fit/results/fit.json` 的 best 点（NLL=-465.9173，`max|grad|=0.077`）：

| 版本 | H 全自由块本征值 | n_neg |
|---|---|---|
| 旧 `.so` | λmin=-532.8, -76.9, -53.2, ... | 9 |
| 同一 `.so` 的 FD | λmin=+0.0024, +0.0041, ... | 0 |
| 修复后 | λmin=+0.0024, +0.0045, ... | 0 |
| 修复后 FD | λmin=+0.0024, +0.0041, ... | 0 |

修复后解析-FD 逐元最大相对差 **0.48%**（旧版 2.27%，但足以把近零方向翻成
-533；该 H 的条件数 ~2.5e7）。

**根因**：`getHessian` 为耦合变换准备 `d_g_v` 时，bkg 段先用
`d_neg_w = -bkg_weight` 调 `computeFactorNLL`（权重线性 → 已翻一次符号），
随后又无条件 `axpy(..., -1.0)`（再翻一次）→ **有** `bkg_weights` 时 bkg 的
`g_v` 符号反了。无 bkg 权重时 kernel 用单位正权重，`-1` 反而是对的。

**修复**：`src/Analysis.cu` bkg 段按 `d_neg_w` 是否存在选择
`axpy(+1)`（有权重）或 `axpy(-1)`（无权重的单位权重）。
CPU 后端已重编；GPU 用户需重编 `_cuda.so`。

**回归**：`tests/test_cpu_numerics.py::test_bkg_sign_in_coupling_hessian`
用 `hess_mult` + bkg 权重 20 放大 bkg。旧 `.so` FAIL（max rel 1.28），
新 `.so` PASS（max rel 9.3e-3）。

### DIYMC 输入/输出闭包

用该 config 的 best 耦合作真值，`generateMC` 产生 3000 个 unweighted 事例；
phsp 用 `PhaseSpaceGenerator` 重新产生 300k **平坦**样本（不用带效率的
`totmc.dat`），去掉 bkg 后做 toy 拟合：

| 量 | 结果 |
|---|---|
| NLL(truth) | -8806.17 |
| NLL(BFGS fit) | -8848.25（Δ=-42.1，40 个自由实参数，属有限样本正常改善） |
| fit 处 H | 最小 +0.0107，最大 1.60e4，**0 个负本征值** |
| intensity 相关 | 0.9915 |
| intensity 比值中位数 | 1.004 |
| 单振幅 `v` 最大相对差 | 4.63（比值 std 1.55） |

结论：**observable（强度/形状）能对上，H 能正定；但 21 个复耦合在
3000 事例下并不逐个可辨识**——H 条件数 ~1.5e6，存在很强的近平坦方向。
要“输入输出逐参数对上”，需要更多事例、更强约束或正则化，不能只看 H 正定。

## 4.6 GPU 复测（RTX 5060 8GB）

修复后重编 `_cuda.so`，在 GPU 上复测：

| 检查 | 结果 |
|---|---|
| `J/ψ→ωηη′` best 点 H | 旧 `.so`: 9 个负本征值，λmin=-532.76；新: **0 负**，λmin=+0.00244 |
| 同点 H vs FD | 新: 0 负，max rel **0.48%**（FD λmin=+0.00236） |
| `tests/test_numerical.py` | **35 passed, 1 xfailed** |
| `test_hessian_bwr_spec + constraints + identical + custom_q0` | **27 passed** |
| `test_mc_amplitude.py`（DIYMC） | **9 passed** |
| `hess_mult` + 权重 20 的 bkg 回归（GPU） | PASS，max rel 1.6e-2 |
| **全量 `pytest tests/`（GPU）** | **208 passed, 50 skipped, 1 xfailed**（2m24s） |
| `J/ψ→K⁺K⁻η` 生产点 | feasible PD=True，active=2，reduced λmin=+0.367（full λmin=-0.79，边界）|
| `J/ψ→ωηη′` toy fit 点 H | 0 负，λmin=+0.0107（CPU）/ +0.0104（GPU），λmax≈1.60e4 |
| toy closure GPU 重拟合 | NLL=-8848.2497（CPU -8848.2498），H 0 负 |

CPU 与 GPU 的 H 本征值在 1e-6 量级一致；生产点 full λmin 在
-0.13（CPU）~-0.79（GPU）之间摆动，是贴边方向的数值噪声，但 active
子空间稳定为正。

## 5. 本次修复

- `src/Analysis.cu`：修正 `getHessian` 里 bkg `g_v` 的符号（`-1 → +1`）。
- 新增 `ctpwa/diagnostics.py`：
  - `analyze_hessian(...)`：固定参考 → 活跃边界 → 可行子空间，返回
    `reduced_pd`（应看的判据）和 `free_pd`（旧口径）；
  - `map_errors_to_params(...)`：grouped 布局的误差映射。
- `example/fit.py`：
  - `is_positive_definite` 改为可行子空间判据；新增
    `is_positive_definite_full`、`full_min_eigenvalue`、`n_active`、
    `active_params`、`kkt_grad_max`；
  - 参数误差只在可行子空间求逆，修正 Re/Im 交错错位；
  - `final_nll` 用 `final_params` 重算，不再用 `nll_history[-1]`；
  - polish 后即使 NLL 只差 1e-6 量级，也采纳 PD 的新点。
- `tests/test_hessian_diagnostics.py`：9 个纯数值回归。
- `tests/test_cpu_numerics.py::test_bkg_sign_in_coupling_hessian`：bkg 符号回归。
- README 增加“Hessian 正定性与盒约束”说明。

## 6. 给用户的检查清单

1. 先确认 `best_params.pt` 与当前 config 同源：重算 NLL/grad，`|grad|` 应
   接近 0；若 NLL 与 summary 差很多，结果目录过期。
2. 用 `ctpwa.diagnostics.analyze_hessian` 看 `reduced_pd`，不要直接对全 H
   做 `all(eig>0)`；固定参考 `Re_0/Im_0` 必须剔除。
3. 看 `active_params`：贴边参数没有统计误差，应报单侧限制或放宽
   `free_range`，不要把它们当测量值。
4. `kkt_grad_max` 大 = 没收敛；先 polish / 增加迭代，再谈 H 正定。
5. 用 DIYMC 做闭包时，phsp 数要覆盖最窄共振；窄峰先用大样本验证
   `truth` 是否 NLL 极小。
6. 有 bkg 的过程先确认 `getHessian` 是带本底符号修复的版本：旧版在
   bkg 占比大、耦合变换二阶项显著时会给出假负本征值。
