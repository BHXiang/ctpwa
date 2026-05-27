# 共振态参数 Hessian 公式

## 1. NLL 定义

单衰变链，耦合参数 `v`，共振态参数 `θ`：

$$
\text{NLL} = -\sum_{\text{data}} \log I_e + n_{\text{data}} \cdot \log(\text{phsp\_factor})
$$

其中：
- $I_e = \sum_p |S_{e,p}|^2$ （所有极化求和）
- $S_{e,p} = \sum_a A_{e,p}^a \cdot v_a$
- $A_{e,p}^a = SL_{e,p}^a \cdot R_e(\theta) \cdot bf$ （完整振幅）
- $T_r(e,p) = \sum_{a \in \text{block}} SL_{e,p}^a \cdot v_a$ （有效耦合，不含 R 和 bf）
- $\text{phsp\_factor} = \frac{1}{N_{\text{phsp}}} \sum_{\text{phsp}} |S|^2$

关键关系（单链，所有channel共享同一R）：$S_{e,p} = R_e(\theta) \cdot bf \cdot T_r(e,p)$

## 2. 梯度 $\partial\text{NLL}/\partial\theta$ （每个事件）

$$
g_\theta[j] = -\frac{1}{I} \frac{\partial I}{\partial\theta_j}
$$

$$
\frac{\partial I}{\partial\theta_j} = 2 \cdot bf \cdot \sum_p \text{Re}\big(\text{conj}(S_p) \cdot T_p \cdot D^j\big)
$$

其中 $D^j = \partial R/\partial\theta_j$ （复数，由 AutoDiff `Var<double,N,true>` 计算）。

因此：
$$
g_\theta[j] = -\frac{2 \cdot bf}{I} \sum_p \text{Re}\big(\text{conj}(S_p) \cdot T_p \cdot D^j\big)
            = -\frac{2 \cdot bf}{I} \cdot \text{Re}\Big(D^j \cdot \sum_p \text{conj}(S_p) \cdot T_p\Big)
            = -2 \cdot bf \cdot \text{Re}\Big(D^j \cdot \sum_p \text{conj}(w_p) \cdot T_p\Big)
$$

最后一步用了 $\text{conj}(S) = I \cdot \text{conj}(w)$，$w = S/I$ 正是 ctpwa 的 `computeFactorNLL` 输出。

## 3. Hessian $\partial^2\text{NLL}/\partial\theta\partial\theta$ （每个事件）

$$
H = g \cdot g^T - \frac{1}{I} \cdot \frac{\partial^2 I}{\partial\theta\partial\theta}
$$

其中：
$$
\frac{\partial^2 I}{\partial\theta_j \partial\theta_k} =
    \underbrace{2 \cdot bf^2 \cdot \sum_p |T_p|^2 \cdot \text{Re}\big(\text{conj}(D^k) \cdot D^j\big)}_{|T|^2\text{ 项}}
  + \underbrace{2 \cdot bf \cdot \sum_p \text{Re}\big(\text{conj}(S_p) \cdot T_p \cdot D^2_{jk}\big)}_{D^2\text{ 项}}
$$

其中 $D^2_{jk} = \partial^2 R/\partial\theta_j\partial\theta_k$ 由 AutoDiff `Var<double,N,true>::hess[j][k]` 提供。

**已验证**：AutoDiff 的 $D^j$ 和 $D^2_{jk}$ 与 PyTorch autograd 完全一致（差异 < 1e-15）。

## 4. CUDA 实现：per-event 聚合公式

一次循环遍历所有极化，聚合所有量，然后算一次外积+修正：

```
// 第一步：遍历极化，聚合 per-event 量
I_inv = Σ|w_p|²          // = 1/I
sum_T2 = Σ|T_p|²
sum_cwT = Σ conj(w_p)·T_p   // 复数

// 第二步：梯度
g[j] = -2·bf · Re(D^j · sum_cwT)

// 第三步：Hessian
H[j][k] = g[j]·g[k]                                          // 外积
        - 2·I_inv·bf²·sum_T2·Re(conj(D_k)·D_j)               // |T|² 修正
        - 2·bf·Re(sum_cwT · D²_jk)                            // D² 修正
```

其中 $\text{Re}(\text{conj}(D_k) \cdot D_j) = D_{re}[k] \cdot D_{re}[j] + D_{im}[k] \cdot D_{im}[j]$.

## 5. 关键简化

D² 修正项中的 $I$ 会约掉：

$$
-\frac{2}{I} \cdot bf \cdot \sum \text{Re}(\text{conj}(S) \cdot T \cdot D^2)
= -2 \cdot bf \cdot \sum \text{Re}(\text{conj}(w) \cdot T \cdot D^2)
$$

所以 D² 修正直接用 `sum_cwT = Σ conj(w)·T`，**不需要乘 `I_inv`**。

## 6. 与 PyTorch 的对比验证

Python 验证（使用与 ctpwa 完全相同的 BWR 公式，q0 固定近似）：

```
PyTorch autograd θ-θ 块:
  [[-21.052335, -5.967517],
   [-5.967517,  2.008298]]

手写公式 θ-θ 块:
  [[-21.052335, -5.967517],
   [-5.967517,  2.008298]]
  → 完全一致 (diff < 1e-10)
```

AutoDiff 计算的 $D^2$ 值也验证通过：
```
PyTorch BWR Hessian:
  d²Re/dm0²=18.4789  d²Re/dm0dg0=13.0991  d²Re/dg0²=-0.5906
  d²Im/dm0²=41.1079  d²Im/dm0dg0=-2.8514  d²Im/dg0²=-5.2698

ctpwa testBWRHessian (Var<double,2,true>):
  d²Re/dm0²=18.4789  d²Re/dm0dg0=13.0991  d²Re/dg0²=-0.5906
  d²Im/dm0²=41.1079  d²Im/dm0dg0=-2.8514  d²Im/dg0²=-5.2698
  → 完全一致 (diff < 1e-15)
```

## 7. 已发现的 Bug

### Bug 1：Hessian kernel 缺少事件偏移量
- **现象**：D 值偏小约4倍，运动学不对（m=1.80 vs 正确值 m=1.11）
- **根因**：`resonanceHessianBlockKernel` 访问 `d_momenta->getMomentum(evt, ...)` 时，`evt` 是 data 子集内的相对索引（0），但 momenta 数组包含所有事件。数据事件的绝对索引是 3（= 3个phsp事件）。
- **修复**：增加 `evt_offset` 参数，用 `evt_abs = evt + evt_offset` 访问 momenta。在 `getHessian` 中传入 `events_[gpu][0]`（phsp数量）作为偏移。

### Bug 2：逐个极化的 |w|² 加权（原始代码）
- **现象**：Hessian 值偏小约10倍
- **根因**：修正项用 `|w_p|² = |S_p|²/I²` 逐极化加权，应改用 per-event `1/I = Σ|w|²`。外积项做的是 `Σ_p X_p·Y_p` 而非 `(Σ_p X_p)·(Σ_q Y_q)`。
- **修复**：Step 5 重构为 per-event 聚合（见第4节公式）。

## 8. 缺失功能：PHSP 对 θ-θ 的贡献

phsp_factor 也依赖 θ（通过 R），其 Hessian 贡献为：

$$
n_{\text{data}} \cdot \frac{\partial^2 \log(\text{phsp\_factor})}{\partial\theta^2}
$$

$$
\frac{\partial^2 \log(\Sigma I / n)}{\partial\theta^2}
= \frac{\Sigma \partial^2 I/\partial\theta^2}{\Sigma I}
- \frac{(\Sigma \partial I/\partial\theta)(\Sigma \partial I/\partial\theta)^T}{(\Sigma I)^2}
$$

当前 `computeResonanceHessian` 只算了 data+bkg 的贡献，缺少 phsp 部分。

## 9. 混合块 $\partial^2\text{NLL}/\partial v\partial\theta$

对于所有 channel 共享同一共振态的单链模型：

$$
g_v[a] = -2 \cdot \text{Re}\left(\frac{\text{conj}(T) \cdot SL_a}{\Sigma|T|^2}\right)
$$

**与 θ 无关**（R 和 bf 在分子分母中约掉）。因此 $\partial^2\text{NLL}/\partial v\partial\theta = 0$。

多链模型（不同channel对应不同共振态）混合块非零，需要 `computeMixedHessian`。
