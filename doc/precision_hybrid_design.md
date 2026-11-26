# CTPWA 设计：config 驱动精度（混合：内存大户 float，核心计算 double）

> 状态：评审稿 → **rev2：已按评审定稿 4 项决策（见文件头），待 D0 开工**
>
> **rev2 决策记录**
> 1. A 写入：kernel **模板化 float2/double2 输出**（先试，出问题再换 double 临时+downcast）
> 2. 耦合 v：**恒 double**（~100 元素不占空间）；NLL matvec 前 downcast 一份 float v 副本（代价可忽略）
> 3. SL 表：**不动**（保持恒 double）
> 4. vθ 混合块：**event 相关**（读 per-event A，`computeUnifiedHessian` 以 `d_all_amplitudes_` 为输入）
>    → float 模式下读 float A、**double 累加**、dF/SL 恒 double；误差界 ~√N·1e-7（300k→5e-5 相对，可忽略）；
>    唯一风险 1/I² 近零强度事件，D3 数值验证
> 目标用户：最终用户 pip 安装后，config.yml 写 `precision: float|double` 即可选精度，无需重编。
> 本设计只把**内存大户**降到 float；模型/导数核心（F/dF、共振梯度、Hessian）保持 double。

---

## 1. 目标与边界

### 1.1 用户视角

```yaml
# config.yml
precision: double   # 默认: 全 double（高精度）
# precision: float  # 大统计量: 内存大户走 float, 省显存; 核心计算仍 double
```

同一 wheel（默认 double 编译），config 一处改、零重编。

### 1.2 边界（明确不做）

- ❌ 不做"全 float"（连 F/dF/θ 梯度都 float）——那需要全路径双实例化（方案 C 全量），范围大、精度风险高。
- ✅ 只做：**驻留振幅矩阵等"内存大户"可存 float**，其消费路径（NLL 的纯 matvec/归约）提供 float 变体；**凡与导数/二阶导相关的计算恒 double**。
- ✅ **耦合 v 恒 double**（~100 元素，不占空间）；float 模式仅在 NLL matvec 前 downcast 一份 float v 副本。
- 精度语义：`precision: float` ≈ "存储 float、归约 double、导数计算 double"。

---

## 2. 现状与内存账单

### 2.1 精度类型（现状）

- `ComplexType.h`：`ctComplex` 编译期 ifdef（`CTPWA_DOUBLE_COMPLEX`）。默认 .so = double（最近已切）。
- config `precision:` 只做一致性检查（Analysis.cu:4613 `req != PRECISION_NAME` 报错），无运行期开关。
- 全库单一类型路径：kernel、cublas 宏（`CUBLAS_CGEMV`→S/Z）、缓冲区、torch dtype 宏全部绑 `ctComplex`。

### 2.2 内存大户清单（按驻留大小）

| 量 | 现状精度 | 大小（~） | 用途 |
|---|---|---|---|
| **A：驻留振幅矩阵** `d_all_amplitudes_` | ctComplex（double .so = double） | n_events×n_polar×n_amps×16B（5M×8×44×16 ≈ 28GB 单卡） | NLL 的 S=A·v、phsp B̄、耦合梯度 |
| **SL 表** `d_slamp_tab_`（每 cas） | **恒 double**（thrust::complex\<double\>，硬编码） | n_events×nSL×nPol×nSigma×16B | 模型核 F(θ)·A_SL 的 kinematic 输入（θ 无关） |
| phsp 重建瞬态 `d_phsp_scaled` | ctComplex | nPhsp×n_polar×n_amps×16B（瞬态） | B̄ 重建（free-θ 时每步） |
| 每事件 S/w 缓冲 | ctComplex | n_events×n_polar×16B | NLL/梯度 |

**本设计把第 1 行（A）与第 3 行（phsp 瞬态）作为 float 目标**——它们是"随 n_amplitudes_×事件数"最大的两块。SL 表维持现状（见 §6 风险：它本身也大，是否后续 float 化单列）。

---

## 3. 精度策略（核心设计）

### 3.1 每类量的精度（float 模式）

| 量 | float 模式 | double 模式 | 说明 |
|---|---|---|---|
| 共振态模型求值 F/dF（per-event over SL） | **double** | double | 符号微分核恒 double（精度/梯度来源） |
| 驻留振幅矩阵 **A**（= F·A_SL 逐波） | **float**（写入前 downcast） | double | 内存主目标 |
| 耦合向量 **v** | **double（恒）**；NLL matvec 前 downcast float 副本（~100 元素） | double | 源真理恒 double；float 副本仅喂 Sgemv |
| NLL data/bkg 的 S=A·v、w=S/factor | float（Sgemv + float S） | double | 逐事件；**-log 累加用 double** |
| phsp 矩阵 **B̄** 累加 | float 矩阵 + **double 矩阵并行**（复用 `d_phsp_matrix_double_`） | double | phsp_factor 恒走 double 轨道（现状已有） |
| 耦合梯度 ∂NLL/∂v | float 反向 matvec + **double 归约** | double | 梯度值可 double |
| **共振梯度 ∂NLL/∂θ、Hessian 的 rr 块** | **double**（F/dF double × 权重，double 累加） | double | 核心导数不降精度 |
| **Hessian 的 vθ 混合块** | **读 float A（float 模式）** + **double 累加**；dF/SL 侧恒 double | double | event 相关（读 per-event A）；累加 double → 误差 ~√N·1e-7 可忽略；`computeUnifiedHessian` 需 float-A 输入变体 |
| 标量（loss、phsp_factor、χ²…） | **double** | double | |

**不变量**：float 模式只在"A 存储 + 纯 matvec 的中间量"用 float；任何跨事件**累加**与任何**对 θ 的导数**都是 double。

### 3.2 为何这个切法成立

- A 是 **F·A_SL 的逐波值**：double 核算出 double 结果 → downcast float 存储（每值 ~1e-16→1e-7 相对误差，事件平均后对 NLL/梯度影响 ~√N·1e-7，可忽略，见 §6 数值界）。
- NLL = -Σ_data log|S|² + N_eff·log(phsp_factor)：逐事件 |S|² 用 float 算、**double 累加**；phsp_factor 恒走 double（现状 `computePhspMeanSum`/`d_phsp_matrix_double_`）。历史上 NLL 漂移 ~2.4 来自 **float32 跨 50 万事件的矩阵累加**，不是逐事件 float——本设计把跨事件累加全部 double，漂移消除。
- 共振导数：θ 梯度/Hessian 的输入是 double 的 F/dF 与 SL，仅乘上 float 的 w（=S/factor 的权重）——w 是纯权重，float 误差进入 double 累加后平均掉。

---

## 4. 需要"float 变体"的最小 kernel/路径集

### 4.1 必须新增/改造的（按必要性排序）

1. **A 的写入路径**（double 核产物 → float 驻留）：
   - `AmpGen.cu` `computeCustomAmpsKernelT`：当前按 ctComplex 写 `d_all_amplitudes_`。**方案（rev2 已定）**：kernel 输出类型**模板化 float2/double2 双实例**（先试；若实现出问题改「double 临时+downcast」）。
2. **NLL data/bkg 路径**（`ComputeNLL.cu` `computeFactorNLL`）：
   - float 模式：`CUBLAS_CGEMV`(S) 读 float A × float v → float S → float w；`-log` 累加 double（现 kernel 已 double 累加）。需要 S 级 cublas 调用分派 + float w 缓冲。
3. **耦合梯度**（`computeFactorNLL` 的 `d_grad_out`）与 backward 的 v 变换：float 反向 matvec + double 归约。
4. **phsp B̄ 重建**（`Analysis.cu` forward/getHessian 的 `s_d_phsp_scaled` 段）：float 模式用 float scaled + S 级 cublas，double 轨道并行（现状流式已建 `d_phsp_matrix_double_`，驻留模式补 double 并行轨道）。
5. **每事件 S/w 缓冲**与共振梯度入口（`Analysis.cu` NLLFunction::forward 的 `s_d_w_bufs` 等）：float 模式为 float2。

### 4.2 明确保持 double、不改

- `ResModel.cu`/`CustomExpr.cu` 的 F/dF 符号微分求值与字节码解释器。
- `AmpGen.cu` 共振梯度/Hessian kernel（读 double SL/dF）。
- `ComputeHessian.cu`。
- phsp_factor/标量轨道（`d_phsp_matrix_double_`、`computePhspMeanSum`）。

> 注：SL 表 `d_slamp_tab_` 现状恒 double——它也是内存大户（nEvents×nSL×nPol×nSigma）。本设计**不改它**（float 化 SL 会影响 dF 求值的输入精度，需单独评估）；作为后续项单列（§6）。

---

## 5. 运行时分派（精度绑定）

### 5.1 绑定时机

`analysis` 构造（`initialize()`）读 config `precision`：
- `double`/`auto` → 全 double（现状路径，零改动分支）；
- `float` → 进入"float 存储"模式：A/v/NLL/phsp 缓冲按 float2 分配，kernel/cublas 走 float 实例。

### 5.2 dispatch 实现（两选一，实现时定）

- **A. 类型层分派**：把热路径 kernel 模板化 `template<Cplx>`，`Cplx` 提供 re/im 与 cublas 句柄（S/Z）选择；init 选定后在 `analysis` 内持有一个 `enum Precis { Float, Double }`，各处按它选实例/缓冲。
- **B. 双缓冲成员**：`analysis` 同时持有 float 版 A（`d_all_amplitudes_float_`）与 double 版（现状），float 模式下 double 版不分配。kernel 调用点按精度选 float/double 实例。

**rev2 已定 A**：模板化热路径（单一调用点、少 if）；代价是热 kernel 双实例编译时间。

### 5.3 与现有机制的关系

- config `precision:` 从"一致性检查"改为"**请求精度**"：double 版 .so 下请求 float → 启用 float 存储路径（不再报错）；请求 double → 默认路径。float 版 .so（`CTPWA_COMPLEX=float` 编译）仍保留为"极致省显存"选项（未来若全 float 化）。

---

## 6. 风险、不变量与开放问题

### 6.1 数值不变量（测试必须守）

- float 模式下 NLL 与 double 模式的差：|ΔNLL| < 0.05（小样本）/ < 每百万事件 0.1（用现有 test_endtoend/test_numerical 的同一 config 双跑对比，golden 用 double 生成）。
- 梯度 vs 有限差分（`test_gradient_vs_fd`）在 float 模式下 rel 容差放宽一个量级（1e-4→1e-3？）——需实测定界。
- phsp_factor 恒 double → NLL 值不重现 float32 漂移（回归 2.4 那类偏差）。

### 6.2 风险

- kernel 双实例 → 编译时间↑（double 模板本就慢），CI 时间↑。
- float A + double v 混合处容易出类型 bug → 用编译期强类型 + 少量 `TORCH_CHECK` 类型断言兜住。
- w/S 缓冲跨精度复用（静态持久缓冲）要按精度分开，避免复用错类型（2×A100 那种跨设备指针错配的同类风险）。

### 6.3 开放问题（rev2 已定 / 遗留）

1. ~~A 写入~~ → **已定**：kernel 模板化 float2/double2（出问题再换 downcast）
2. ~~v 精度~~ → **已定**：恒 double，float 副本仅喂 Sgemv
3. ~~SL 表~~ → **已定**：不动（恒 double）；若大统计量仍吃紧，单开"SL float 化"评估
4. ~~vθ 混合块~~ → **已定**：float 模式读 float A、double 累加（event 相关但误差界可忽略）；实现上 `computeUnifiedHessian` 需 float-A 输入变体；**D3 用数值界验证 1/I² 近零事件风险**
5. （遗留）float 模式下 `getHessian` 的 data 侧 rr/混合块是否也读 float A —— 与 4 同理走 float-A 变体，D3 一并验证

---

## 7. 分阶段实施 + 每阶段验证

| 阶段 | 内容 | GPU 验证 |
|---|---|---|
| **D0** | 精度枚举 + init 绑定 + float A 分配路径（不跑，先让 `precision: float` 能进 float 分支不崩） | 构造 analysis(precision:float) 通过 |
| **D1** | A 写入 float（kernel float 输出/模板）+ NLL data/bkg float 路径（Sgemv+float w+double 累加） | `precision:float` 跑 simple 拟合，NLL 与 double 差 < 界 |
| **D2** | 耦合梯度/backward float 变换 + phsp B̄ 双轨道（float scaled + double 矩阵） | endtoend/identical 数值对拍 |
| **D3** | getHessian float 模式（vθ/rr 混合块读 float A） | Hessian 对称 + golden 对拍 |
| **D4** | 全量 pytest 双跑（float/double 各一遍）+ CI 时间评估 | 111+111 pass |

> 每阶段独立可测、可回退；D1 是"内存大户 float"的主体（A 存储 + NLL 路径），也是显存收益的大头。

---

## 8. 结论

- 混合方案可行：**A 存储 + NLL 纯 matvec 路径 float（省显存），F/dF/θ 导数/Hessian/所有跨事件累加 double（保精度）**。
- 是"最小双实例化"（比全 float 少一半范围），用户 config 一处改即可。
- 建议按 D0→D4 实施，D1 先落地显存收益，每阶段 GPU pytest 把关。
- 先评审本稿（尤其 §4 kernel 清单、§6.3 开放问题）再动代码。
