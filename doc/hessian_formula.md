# Hessian 公式备忘录

> 已验证精度：vv, vθ, θv, θθ 四个 block 与 PyTorch autograd 差异 < 1.2×10⁻⁶

## 1. NLL 定义

$$
\text{NLL} = -\sum_{\text{data}} \log I_e + w_{\text{bkg}} \sum_{\text{bkg}} \log I_e + A \cdot \log(\text{pf})
$$

其中：

- $A = N_{\text{data}} - \int_{\text{bkg}}$ （phsp 项的系数）
- $\text{pf} = \frac{1}{N_{\text{phsp}}} \sum_{e \in \text{phsp}} I_e$
- $I_e = \sum_p |S_{e,p}|^2$ （所有极化求和）
- $S_{e,p} = \sum_a v_a \cdot \text{amp}_{a,e,p}$ （所有 channel 求和）
- $\text{amp}_{a,e,p} = SL_{a,e,p} \cdot R_{\text{res}(a),e}(\theta) \cdot bf$ （完整振幅）

## 2. Per-event 量定义

### 2.1 基本量

| 符号 | 定义 | 含义 |
|------|------|------|
| $S[p]$ | $\sum_a v_a \cdot \text{amp}_a[p]$ | 每个极化的复振幅 |
| $I$ | $\sum_p (S_{\text{re}}[p]^2 + S_{\text{im}}[p]^2)$ | 每个事件的强度 |
| $1/I$ | | 逆强度 |

### 2.2 一阶导数

| 符号 | 公式 | 含义 |
|------|------|------|
| $\partial S/\partial v_a^{\text{re}}$ | $\text{amp}_a$ | S 对耦合实部的导数 |
| $\partial S/\partial v_a^{\text{im}}$ | $i \cdot \text{amp}_a$ | S 对耦合虚部的导数 |
| $dS_j[p]$ | $\partial S[p]/\partial\theta_j$ | S 对共振态参数的导数 |
| $\partial I/\partial v_a^{\text{re}}$ | $2 \cdot \text{Re}(\text{conj}(S) \cdot \text{amp}_a)$ | I 对耦合实部的导数 |
| $\partial I/\partial v_a^{\text{im}}$ | $-2 \cdot \text{Im}(\text{conj}(S) \cdot \text{amp}_a)$ | I 对耦合虚部的导数 |
| $\partial I/\partial\theta_j$ | $2 \cdot \text{Re}(\text{conj}(S) \cdot dS_j)$ | I 对共振态参数的导数 |

### 2.3 二阶导数

| 符号 | 公式 | 含义 |
|------|------|------|
| $\partial^2 I/\partial v_a^{\text{re}}\partial\theta_j$ | $2 \cdot \text{Re}(\text{conj}(dS_j) \cdot \text{amp}_a + \text{conj}(S) \cdot \partial\text{amp}_a/\partial\theta_j)$ | 混合二阶导数（实部） |
| $\partial^2 I/\partial v_a^{\text{im}}\partial\theta_j$ | $-2 \cdot \text{Im}(\text{conj}(dS_j) \cdot \text{amp}_a + \text{conj}(S) \cdot \partial\text{amp}_a/\partial\theta_j)$ | 混合二阶导数（虚部） |
| $\partial\text{amp}_a/\partial\theta_j$ | $SL_a \cdot \partial R_{\text{res}(a)}/\partial\theta_j$ | 振幅对 θ 的导数（同块非零，跨块为零） |

### 2.4 Per-event 梯度

| 符号 | 公式 | 含义 |
|------|------|------|
| $g_j$ | $-\frac{2}{I} \cdot \text{Re}(\text{conj}(S) \cdot dS_j)$ | $\partial(-\log I)/\partial\theta_j$（per-event，无权重） |
| $g_a^{(v,\text{re})}$ | $-\frac{2}{I} \cdot \text{Re}(\text{conj}(S) \cdot \text{amp}_a)$ | $\partial(-\log I)/\partial v_a^{\text{re}}$ |
| $g_a^{(v,\text{im})}$ | $+\frac{2}{I} \cdot \text{Im}(\text{conj}(S) \cdot \text{amp}_a)$ | $\partial(-\log I)/\partial v_a^{\text{im}}$ |

## 3. Hessian 公式（data + bkg）

每事件带权重 $w_e$（data=+1, bkg=-w_b）：

### 3.1 vv Block（由 PyTorch autograd 计算）

$H_{vv} = \partial^2\text{NLL}/\partial v\partial v$，由 `NLLFunction` 的 autograd 自动计算。

### 3.2 θθ Block

对每个事件 $e$，per-event Hessian：

$$
h_{jk} = g_j \cdot g_k - \frac{2}{I} \cdot \text{Re}(\text{conj}(dS_k) \cdot dS_j) - \frac{2}{I} \cdot \text{Re}(\text{conj}(S) \cdot d^2S_{jk})
$$

三项分别称为 **Term A**（外积）、**Term B**（$|dS|^2$ 修正）、**Term C**（$d^2S$ 修正，仅同块非零）。

总 θθ Hessian：

$$
H_{\theta\theta}[j,k] = \sum_e w_e \cdot h_{jk}^{(e)}
$$

### 3.3 vθ / θv Block（混合 Hessian）

对每个振幅 $a$ 和共振态参数 $j$（同块，即 $a$ 和 $\theta_j$ 属于同一 resonance）：

$$
\begin{aligned}
H_{v_a^{\text{re}}, \theta_j} &= -\frac{2w_e}{I} \cdot \Big[\text{Re}(\text{conj}(dS_j) \cdot \text{amp}_a) + \text{Re}(\text{conj}(S) \cdot \partial\text{amp}_a/\partial\theta_j) + g_j \cdot \text{Re}(\text{conj}(S) \cdot \text{amp}_a)\Big] \\
&= -\frac{2w_e}{I} \cdot \big[\text{Term1}_{\text{re}} + \text{Term2}_{\text{re}} + g_j \cdot \text{Term3}_{\text{re}}\big]
\end{aligned}
$$

$$
H_{v_a^{\text{im}}, \theta_j} = +\frac{2w_e}{I} \cdot \big[\text{Term1}_{\text{im}} + \text{Term2}_{\text{im}} + g_j \cdot \text{Term3}_{\text{im}}\big]
$$

对跨块（$a$ 和 $\theta_j$ 属于不同 resonance）：Term2 = 0（因为 $\partial\text{amp}_a/\partial\theta_j = 0$），只用 Term1 + Term3。

---

**Term1, Term2, Term3 定义：**

| 项 | 实部公式 | 虚部公式 |
|----|---------|---------|
| Term1 | $\text{Re}(\text{conj}(dS_j) \cdot \text{amp}_a)$ | $\text{Im}(\text{conj}(dS_j) \cdot \text{amp}_a)$ |
| Term2 | $\text{Re}(\text{conj}(S) \cdot SL_a \cdot \partial R/\partial\theta_j)$ | $\text{Im}(\text{conj}(S) \cdot SL_a \cdot \partial R/\partial\theta_j)$ |
| Term3 | $\text{Re}(\text{conj}(S) \cdot \text{amp}_a)$ | $\text{Im}(\text{conj}(S) \cdot \text{amp}_a)$ |

---

## 4. Phsp 贡献

### 4.1 θθ Block

Phsp 项 $L_{\text{phsp}} = A \cdot \log(\text{pf})$，其中 $\text{pf} = \frac{1}{N_{\text{phsp}}} \sum_{e \in \text{phsp}} I_e$。

$$
\frac{\partial^2 L_{\text{phsp}}}{\partial\theta_j\partial\theta_k} = c_1 \cdot \Sigma_{\text{phsp}}[I \cdot (g_j g_k - h_{jk})] + c_2 \cdot (\Sigma I g_j) \cdot (\Sigma I g_k)
$$

其中：
- $c_1 = \dfrac{A}{\text{pf} \cdot N_{\text{phsp}}}$
- $c_2 = -\dfrac{A}{\text{pf}^2 \cdot N_{\text{phsp}}^2}$

### 4.2 vθ Block

Phsp 对混合 Hessian 的贡献：

$$
\frac{\partial^2 L_{\text{phsp}}}{\partial v_a^{\text{re}}\partial\theta_j} = c_{1m} \cdot \Sigma_{\text{phsp}}(\text{Term1}_{\text{re}} + \text{Term2}_{\text{re}}) + c_{2m} \cdot \big(\Sigma_{\text{phsp}} \text{Term3}_{\text{re}}^{(a)}\big) \cdot \big(\Sigma_{\text{phsp}} I \cdot g_j\big)
$$

$$
\frac{\partial^2 L_{\text{phsp}}}{\partial v_a^{\text{im}}\partial\theta_j} = -\Big[c_{1m} \cdot \Sigma_{\text{phsp}}(\text{Term1}_{\text{im}} + \text{Term2}_{\text{im}}) + c_{2m} \cdot \big(\Sigma_{\text{phsp}} \text{Term3}_{\text{im}}^{(a)}\big) \cdot \big(\Sigma_{\text{phsp}} I \cdot g_j\big)\Big]
$$

其中：
- $c_{1m} = \dfrac{2A}{\text{pf} \cdot N_{\text{phsp}}}$
- $c_{2m} = \dfrac{2A}{\text{pf}^2 \cdot N_{\text{phsp}}^2}$

> **注意**：$c_{1m}$ 和 $c_{2m}$ 都有因子 2（与 θθ 的 $c_1, c_2$ 不同），这是因为 $\partial I/\partial v = 2 \cdot \text{Term3}$，而 $\partial I/\partial\theta = -I \cdot g$。

---

## 5. CUDA 实现结构

### 5.1 Kernel 流水线

```
Pre-pass: computeSfromAmpsKernel
  → 计算 S[p], I, 以及 Term3（用于 phsp vθ）

Stage 1: hessianStage1Kernel<Npr, Nres>  (per-block, per-event)
  → 输出 per-event g_j, dS_j, ∂F/∂θ_j
  → 累加同块 θθ Hessian
  → 累加 phsp: I·g, I·(g⊗g - h)

Stage 2: hessianCrossBlockKernel
  → 跨块 θθ Hessian（仅 Term A + Term B）

Stage 3: hessianMixedBlockKernel  (per-block)
  → 同块 vθ Hessian（Term 1 + 2 + 3）
  → phsp path: 写 Term1+Term2 到 h_sum，Term3 由 pre-pass 计算

Stage 4: hessianCrossMixedKernel
  → 跨块 vθ Hessian（仅 Term 1 + 3，Term2=0）
  → phsp path: 写 Term1 到 h_sum（Term3 已由 pre-pass 计算，不重复）
```

### 5.2 关键注意点

- **Pre-pass 计算完整 S**：Stage 1 不自己算 S，而是从 pre-pass 读入。确保 S 包含所有 channel 的贡献（不仅是本块的）。
- **Term3（t3）在 pre-pass 统一计算**：Term3 是 per-amplitude 的量，与具体哪个 θ block 无关。在 pre-pass 中一次性计算所有 SL channel 的 Term3，避免 stage 3 和 stage 4 重复累加。
- **跨块 Term2 = 0**：对于 $a \in \text{block A}, \theta_j \in \text{block B}$，$\partial\text{amp}_a/\partial\theta_j = 0$，因此 Term2 不参与跨块贡献。
- **Phsp 的 is_phsp 判断**：`default_weight == 0.0 && d_phsp_sum != nullptr` 时走 phsp 路径，写入独立 buffer（非 d_mixed）。
- **所有 resonance 必须注册到 blocks_**：即使没有 free params，其 SL channels 也需要参与跨块 vθ 计算（stage 4）。

### 5.3 符号约定

| ctpwa | 数学含义 |
|-------|---------|
| `g[j]` | $-\partial(\log I)/\partial\theta_j$（无权重） |
| `d_dS_re[j][p]` | $\text{Re}(\partial S[p]/\partial\theta_j)$ |
| `d_dS_im[j][p]` | $\text{Im}(\partial S[p]/\partial\theta_j)$ |
| `d_dF_re[a][j]` | $\text{Re}(\partial R_{\text{res}(a)}/\partial\theta_j)$ |
| `d_dF_im[a][j]` | $\text{Im}(\partial R_{\text{res}(a)}/\partial\theta_j)$ |
| `d_phsp_I` | $\Sigma_{\text{phsp}} I_e$ |
| `d_phsp_grad[j]` | $\Sigma_{\text{phsp}} I_e \cdot g_j^{(e)}$ |
| `d_phsp_hessA[j][k]` | $\Sigma_{\text{phsp}} I_e \cdot (g_j g_k - h_{jk})$ |
| `d_phsp_mixed_sum[a][j]` | $\Sigma_{\text{phsp}}(\text{Term1} + \text{Term2})_{a,j}$ |
| `d_phsp_mixed_t3[a]` | $\Sigma_{\text{phsp}} \text{Term3}_a$（由 pre-pass 计算） |

### 5.4 约束投影

在 vv 和 vθ block 计算完成后，需要将 extended parameter space 的结果投影到 free parameter space：

对于实部约束：$v_{\text{ext}}[e] = rr \cdot v_{\text{free}}[\text{oid}] + \cdots$

$$
H_{\text{free}}[\text{oid}, :] \mathrel{+}= rr \cdot H_{\text{ext}}[e, :] + ir \cdot H_{\text{ext}}[n_{\text{ext}} + e, :]
$$
$$
H_{\text{free}}[n_{\text{free}} + \text{oid}, :] \mathrel{+}= -ir \cdot H_{\text{ext}}[e, :] + rr \cdot H_{\text{ext}}[n_{\text{ext}} + e, :]
$$

---

## 6. 验证结果

| Block | 1-res（1 个自由共振态） | 2-res（2 个自由共振态） |
|-------|------------------------|------------------------|
| vv | 7.73×10⁻⁷ | 7.73×10⁻⁷ |
| vθ | 2.89×10⁻⁷ | 3.18×10⁻⁷ |
| θv | 2.89×10⁻⁷ | 3.18×10⁻⁷ |
| θθ | 8.95×10⁻⁷ | 1.18×10⁻⁶ |

测试条件：1 data event, 3 phsp events, 1 bkg event, 9 polarizations, BWR×2 resonances。
