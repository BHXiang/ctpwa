# auto-pwa 能力评估：config.yml 检查与量子数计算

> 评估对象：`/home/whitewash/pkgs/auto-pwa`（DSH 插件 + 纯物理核心）
> 参考实现：`/home/whitewash/pkgs/ctpwa`（CUDA 分波引擎）
> 日期：2025-08
> 结论摘要：**三项能力 auto-pwa 都需要**。其中"衰变顶点 J^P 检查 + C 宇称守恒检查"（对应需求 2b/2c）是**高优先级物理缺口**——现有工具会放行物理禁戒的共振态；"SL 枚举"（需求 2a）和"Constraints 段解析"（需求 1）是中优先级。
>
> **状态更新（2025-08）：** 后续开发工作文档已移交
> `/home/whitewash/pkgs/auto-pwa/docs/capability-eval-and-plan.md`（本文件保留为原始评估）。
> ctpwa 侧配套改动——零-SL 加载诊断 warning——已实现于 `src/Analysis.cu`（见本文 §4.5）。

---

## 1. 项目结构速览

### 1.1 ctpwa（拟合引擎，C++/CUDA）
- `src/Config.cu`：解析 config.yml——`Particles`（J/P/mass）、`Data`、`DecayChains`（紧凑/展开两种格式、per-step `sl` 白名单、`p_break`、`is_bf`）、`Constraints`（`trans`、`identical`、`maxL`、`bf_d`、`has_bf`、`fix_var`、`free_var`、`var_range`、`var_equal`、`gauss_constr`、`chains` 过滤器）、`Resonances`（BWR/BW/GS/Flatte/ONE/Interp/Custom）、`Plot`。
- `include/AmpGen.cuh` + `src/AmpGen.cu`：**`Amp2BD::ComSL` 是引擎内部的 SL 枚举器**——对每个两体衰变步：
  - S ∈ [|J₂−J₃|, J₂+J₃]；L ∈ [|J₁−S|, J₁+S]
  - 宇称条件 P₁ = P₂·P₃·(−1)^L（`p_break` 时跳过）
  - **全同粒子选择定则**：(−1)^{L+S} = +1（玻色子）/ −1（费米子），仅对 `Constraints.identical` 组内成员生效
  - L ≤ 全局 `maxL`；`sl` 白名单按 **(2S+1, L)** 记号过滤
- `include/Config.cuh`：`is_fermion()` = 半整数自旋。

### 1.2 auto-pwa（AI 工具层，TypeScript）
- `src/` 纯物理核心（无 DSH 依赖）：`lookup.ts`（PDG 表查询）、`decay-check.ts`（产生顶点 J^P 可达性）、`resonance-validate.ts`（9 条硬规则 + 软规则）、`config-edit.ts`（parse/apply/dump config.yml）、`float-policy.ts`、`fit-runner.ts`、`fit-summary.ts`、`iteration-log.ts`、`report.ts`、`types.ts`。
- `plugin/auto-pwa.ts`：12 个 `auto_pwa_*` 工具（lookup / decay_check / validate_add / edit_config / iterate / round / run_fit / fit_status / evaluate / iter_start / note / history）。
- `data/pdg.json`：200 个条目，字段只有 `{id, aliases, jp:{j,p}, mass, width, decayModes, ...}`——**只有 J^P，没有 C 宇称、没有电荷/同位旋**。
- skill `auto-pwa-analysis`：分波物理规则 + 迭代流程（J^P 可达性、PDG 依据、阈值、float 策略）。

---

## 2. 能力现状盘点

### 2.1 能力 1：检查 config.yml —— **部分具备，缺口在 Constraints/Data/衰变步**

已有（`config-edit.ts` + `resonance-validate.ts`）：
- `parseConfig`：解析 `Particles`、`Resonances`（含 free/free_range/channels/tex）、`DecayChains` 的 intermediates 组 + **产生顶点运动学**（A→R+B 阈值，跨链取最紧）。
- `validateResonanceAddition`：模型合法、PDG 依据、J^P 一致、质量吻合、阈值、重复、组存在、参数结构、free 结构；软规则：近阈值、宽度偏差、拥挤组、质量重叠。
- `crossReferenceErrors`：intermediates ↔ Resonances 交叉引用（ctpwa 会静默跳过未定义共振态，工具拒绝）。
- `applyResonanceAddition` + `dumpConfig`：结构化编辑、原子写、`.bak` 备份、组只能追加（保住 `Constraints.trans` 振幅索引）。

**缺失（与本次需求直接相关）**：
1. **`Constraints.identical` 完全不解析**——工具不知道哪些粒子是全同的。ctpwa 引擎用它做 Bose/Fermi 对称化和 SL 选择定则，工具层没有对应视图。
2. `Constraints.trans` 不校验（引用的振幅名是否存在、组顺序变化是否破坏索引）。
3. `Constraints.maxL / bf_d / has_bf / fix_var / free_var / var_range / var_equal / gauss_constr / chains` 均未解析校验。
4. `Data.order` 与 `Particles` 的一致性、数据文件存在性不校验。
5. **衰变步（R→a+b）本身没进视图**：只提取了产生顶点运动学，两体衰变顶点（中间态的子粒子）未记录——这是后面"衰变顶点检查"的输入依赖。
6. per-step `sl` 白名单 / `p_break` / `is_bf` opts 未解析。

### 2.2 能力 2a：两个粒子 J^P → 可能的 SL —— **部分具备（只有 L，没有 S，没有 SL 对）**

已有：`allowedIntermediateJP(mother, daughter, maxL)`（`decay-check.ts`）——A→R+B 产生顶点的角动量+宇称枚举，输出 `{jp, L[]}`。

缺失/不完整：
- **S 没有算**：输出只有 L 列表；而 ctpwa 的波是 (S, L) 对（config 里 `sl: [[1,0],[3,2]]` 是 **2S+1 记号**）。模型写 `sl` 白名单或判断波成分时无工具可用。
- **不区分全同粒子**：ctpwa 在 ComSL 里对全同组施加 (−1)^{L+S} 选择定则，工具层没有。
- **没有衰变顶点侧**：R→a+b 的 (S,L) 完全没查（见 2.3 案例 2）。
- 与 ctpwa `ComSL` 的规则集不共享实现，存在漂移风险（parity 条件相同，但 identical/maxL/sl_filter 语义只在 ctpwa 里）。

### 2.3 能力 2b/2c：三体 J^PC → 中间共振态 J^PC；读 config 全同/正反粒子信息 —— **完全缺失**

现状：
- **C 宇称在整条工具链里不存在**：types.ts 无 C、pdg.json 无 C、计算无 C、工具输出无 J^PC。
- **正反粒子对/全同粒子识别不存在**：没有共轭粒子对表（pip↔pim、Kp↔Km、K0↔K0bar、p↔pbar…），`normalizeName` 恰好保留了电荷（`Kp→kp`、`Km→km`），是现成的识别起点，但没被利用。
- **fermion/boson 判定不存在**（ctpwa 用自旋半整数判定，工具层未建模）。
- 三体 isobar 枚举（每个中间态 R_ab 同时满足"衰变顶点 a+b"和"产生顶点 A→R+c"）不存在。

**两个可复现的物理缺口案例**（脚本验证过）：

**案例 1（C 宇称）**：链 J/ψ(1⁻⁻) → η(0⁻⁺) + R_KK，R_KK → K⁺K⁻。
- 产生顶点（现有工具能算）：J^P ∈ {0⁻,1±,2±,3±,4±,5⁺}。
- C 守恒：C(J/ψ)=−1，C(η)=+1 → **C(R_KK)=−1**；K⁺K⁻ 是玻色子-反玻色子对（S=0），C=(−1)^L、P=(−1)^L → J^PC ∈ {0⁺⁺,1⁻⁻,2⁺⁺,3⁻⁻,4⁺⁺}。
- 交集：**J^PC(R_KK) = {1⁻⁻, 3⁻⁻}**。
- **现有工具会放行 f2(1270)（2⁺⁺）**：产生顶点 2⁺ 可达 ✓、PDG 有 ✓、J^P 一致 ✓、质量 1.275 < 2.549 ✓——全部通过，但 J/ψ→η f2(1270) 违反 C 守恒（−1 → +1·+1）。同理 f2'(1525)、f0(1500) 等。
- 注：example/config.yml 里 f0_980（0⁺⁺）挂在 R_KK 是双重复合错误（产生顶点宇称也禁戒），现有工具恰好能拦（jp-not-allowed），但拦的原因是"产生顶点"，不是 C——换个 2⁺⁺ 就漏了。

**案例 2（衰变顶点）**：链 J/ψ → K + R_Keta，R_Keta → K⁺η。
- K⁺η 不是正反粒子对 → **无 C**，只查 J^P。
- 衰变顶点：两个 0⁻ → J_R = L，P_R = (−1)^L → J^P ∈ {0⁺,1⁻,2⁺,3⁻,4⁺}（自然宇称）。
- 产生顶点：J^P ∈ {0⁻,1±,2±,3±,4±,5⁺}。
- 交集：{1⁻,2⁺,3⁻,4⁺}。
- **现有工具会放行 K1(1410)（1⁺）**：产生顶点 1⁺ 可达 ✓、PDG J^P 一致 ✓、阈值 ✓——但 1⁺→Kη 宇称禁戒（1⁺ 需要 L=0 或 2，P=(−1)^L=+1 ≠ −1 要求）。example/config.yml 里 K1_1410 被注释掉正是这个原因，但工具无从知道。

**案例 3（全同粒子）**：π⁰π⁰（全同玻色子）：(−1)^L=+1 → 只 L 偶 → J^PC ∈ {0⁺⁺,2⁺⁺,4⁺⁺}；ΛΛ（全同费米子）：(−1)^{L+S}=−1；pp̄（正反费米子）：C=(−1)^{L+S}。这些现在都没有工具能回答。

---

## 3. 需求评估结论

| 需求 | 是否需要 | 优先级 | 理由 |
|---|---|---|---|
| 1. 检查 config.yml（完整） | **需要** | 中 | 已有基础扎实，但 Constraints 段（尤其 `identical`）不解析，`trans`/`maxL`/`sl` 无校验。模型改 config 时对 ctpwa 引擎语义是"盲改"；`identical` 同时是需求 2c 的输入依赖。 |
| 2a. 两粒子 J^P → SL | **需要** | 中 | ctpwa 波本质是 (S,L) 对（2S+1 记号），写 `sl` 白名单、理解波成分、对齐 ComSL 规则（含全同选择定则、maxL、p_break）都需要。实现是纯函数、工作量小、可逐点对拍 ComSL。 |
| 2b. 三体 J^PC → 中间共振态 J^PC | **需要** | **高** | 最大物理缺口：C 宇称守恒在 J/ψ 强衰变中是硬约束，现有 validate 会放行 C 禁戒波（案例 1），浪费拟合轮次、产生错误物理结论。三体枚举直接给出每个中间态 J^PC 全集+交集，是加波前第一性检查。 |
| 2c. 读 config（全同/正反费米子/玻色子）直接出 J^PC | **需要** | **高** | 与 2b 合并实现：config.yml 已含全部输入（Particles + Constraints.identical + DecayChains），无需用户额外提供信息。全同玻色子 (−1)^L、正反费米子对 C=(−1)^{L+S}、正反玻色子对 C=(−1)^L 的规则集明确，ctpwa 引擎内部已有 identical 部分，工具层补齐并前移为**写入前校验**。 |

**总体判断**：auto-pwa 的现有架构（纯函数核心 + 工具薄层 + PDG 表 + 强约束校验链）非常适合承接这三项能力，且缺口是真实的（两个放行禁戒态的案例可复现）。不做的话，模型在 J/ψ 三体分析里会持续提出/写入 C 禁戒或衰变顶点禁戒的波。

---

## 4. 开发方案

### Phase A：纯函数核心（无 DSH 依赖，与现有 src/ 风格一致，可单测）

1. **types.ts 扩展**
   - `JP` 增加 `c?: 1 | -1`（C 宇称，未定义时缺省）→ 新增 `JPC`。
   - 新增 `SL = { s: number; l: number }`（S 用 2S+1 记号存储，与 config `sl` 一致）。
   - `Particle` 增加可选 `conjugate?: string`（反粒子名）与 `fermion?: boolean` 派生标记。
   - `PwaConfig` 增加 `constraints` 视图（identical 组、trans、maxL、bf_d、has_bf、var_* 等）。

2. **新增 `src/jpc.ts`（或 `spin-statistics.ts`）**
   - `isFermion(j)`：半整数自旋（对齐 ctpwa `is_fermion`）。
   - `conjugatePairs` 约定表 + `pairKind(d1, d2, identicalGroups)`：返回 `{ kind: 'identical-boson' | 'identical-fermion' | 'pair-anti' | 'distinct', cDefined: boolean }`。共轭对识别基于 normalizeName 电荷约定（`kp↔km`、`pip↔pim`、`k0↔k0bar`…），**保守策略：表外名字一律 `cDefined=false`，只查 J^P**，避免错误映射。
   - `enumerateSL(motherJP, d1, d2, {maxL, identical, isBoson, pBreak, slFilter})`：**逐点复刻 `Amp2BD::ComSL`**（S 范围、L 范围、宇称、(−1)^{L+S} 选择定则、maxL、2S+1 白名单），保证工具输出与引擎实际波表一致。
   - `pairJPC(d1, d2, opts)`：衰变顶点侧 J^PC 全集：S 列表 → J=L⊗S → P=P₁P₂(−1)^L → C（仅 cDefined：玻色子对 (−1)^L、费米子对 (−1)^{L+S}、全同玻色子 (−1)^L 且 L 偶、全同费米子 (−1)^{L+S}=−1）。
   - `allowedIsobarJPC(mother, d1, d2, d3, {maxL})`：对每个 pair (i,j)：衰变顶点 J^PC ∩ 产生顶点（J^P + C 守恒，A→R+spectator）→ 输出每个中间态 J^PC 全集与交集。

3. **`config-edit.ts` 扩展**
   - `parseConfig` 记录**衰变步**（intermediate → [d1, d2]，含 per-step `sl`/`p_break` opts），并解析 `Constraints`（identical/trans/maxL/bf_d/has_bf/var_*）进 `PwaConfig.constraints`。
   - 新增 `validateConfig(config)`：`identical` 组内自旋一致（ctpwa 有同样校验）、`trans` 引用的振幅名存在、`maxL` 正整数、`Data.order` ⊆ Particles、`sl` 白名单条目 (2S+1,L) 合法。

4. **`resonance-validate.ts` 增加硬规则**
   - 规则 10（**衰变顶点**）：提案 J^P 必须 ∈ pairJPC(d1,d2)（需要中间态的子粒子，Phase A.3 提供）。→ 拦截案例 2（K1(1410)）。
   - 规则 11（**C 守恒**）：产生顶点 A→R+B 中若母子均有 C，则 C(A)=C(R)·C(B)；R 的 C 来自 pairJPC（若 cDefined）。→ 拦截案例 1（f2(1270)）。
   - 规则 12（**全同选择定则**）：R→a+b 中 a、b 属同一 `Constraints.identical` 组时施加 (−1)^{L+S}。
   - 注意：这些是**工具层写入前闸门**，ctpwa 引擎本身"能跑"C 禁戒振幅（只会在 SL 枚举里正常生成波），所以校验不会与引擎冲突，而是把物理错误挡在写 config 之前。

5. **单元测试**（对照 ComSL 数值 + 手工物理案例）
   - J/ψ→ηK⁺K⁻：R_KK J^PC 交集 = {1⁻⁻,3⁻⁻}；f2(1270) 被规则 11 拒。
   - R_Keta→K⁺η：1⁺ 被规则 10 拒；{1⁻,2⁺,3⁻,4⁺} 通过。
   - π⁰π⁰（identical 组）：只 {0⁺⁺,2⁺⁺,4⁺⁺}；ΛΛ：(−1)^{L+S}=−1。
   - `enumerateSL` 与 ctpwa ComSL 在随机 (J,P) 组合上逐点一致（含 maxL、2S+1 白名单、p_break）。
   - K⁺η / π⁺π⁰：cDefined=false，回退 J^P 模式。

### Phase B：工具层接入

6. 新增工具 **`auto_pwa_jpc_check`**：输入（链名或中间态名，从 config 读）→ 输出每个中间态：两顶点 J^PC 全集、交集、PDG 候选（复用 db，按 J^PC 过滤）。替代/增强现有 `auto_pwa_decay_check`（后者保留 J^P 查询语义，或合并为带 C 的完整输出）。
7. `auto_pwa_validate_add` 接入规则 10–12（errors 非空禁止写入，与现有门禁一致）。
8. `plugin/auto-pwa.ts` 注册新工具 + `vendor/dsh/` stubs 同步 + plugin 测试。

### Phase C：数据与知识增强（可选）

9. `data/pdg.json` 增加 `c` 字段（仅自共轭态，如 π⁰、f0、ρ、φ…），`fetch_pdg.py` 扩展；`lookup` 支持按 J^PC 查询。
10. skill `auto-pwa-analysis` 更新：JPC 规则章节（C 守恒、正反粒子对/全同粒子选择定则、两顶点检查），把案例 1/2 作为教学示例；文档标注 `sl` 白名单的 2S+1 记号。

### 风险与注意事项

- **共轭对映射表**是唯一容易出错的输入：采用"表内才启用 C、表外退化为 J^P"的保守策略；映射表（pip/pim、Kp/Km、K0/K0bar、p/pbar、n/nbar、Λ/Λ̄、Σ⁺/Σ⁻…）放一个可测的纯数据文件。
- **与 ctpwa 行为对齐**：identical 选择定则只作用于 `Constraints.identical` 组内成员（ctpwa 实现如此）；若两个全同粒子未声明 identical，工具不能替引擎假设对称化，需在规则 12 里以"组内"为准并给出 warning 提示未声明。
- **C 只对自共轭系统定义**：K⁺η、π⁺π⁰、K⁺K⁺ 等一律回退 J^P；不要在文档/输出中给这些系统"伪造" C。
- maxL 默认 4 与 skill 一致；`sl` 的 2S+1 记号在输出中必须显式标注，避免与物理 S 混淆。
- 现有 57 个测试 + golden 测试需保持通过；新规则只新增 error 码，不改已有输出格式（向后兼容）。

### 工作量估计

- Phase A：~1 天（jpc.ts ~300 行 + config-edit 扩展 ~150 行 + validate 规则 ~100 行 + 测试 ~200 行）
- Phase B：~0.5 天（新工具 + 插件注册 + stubs）
- Phase C：~0.5 天（pdg 扩展 + skill/文档）

---

## 附录：量子数规则速查（工具将实现）

| 系统 | 判定条件 | 规则 |
|---|---|---|
| 两体衰变 R→a+b | 任意 | S ∈ [|J_a−J_b|, J_a+J_b]，L ∈ [|J_R−S|, J_R+S]，P_R = P_a·P_b·(−1)^L，J_R ∈ L⊗S |
| 正反玻色子对（π⁺π⁻、K⁺K⁻） | 名字互为共轭且非全同 | C = (−1)^L（S=0 时），J^PC = J⁺⁺/J⁻⁻ 交替（自然宇称） |
| 正反费米子对（pp̄、ΛΛ̄） | 名字互为共轭且非全同 | C = (−1)^{L+S} |
| 全同玻色子（π⁰π⁰、KsKs） | `Constraints.identical` 组内 | (−1)^L = +1 → L 偶 |
| 全同费米子（ΛΛ） | `Constraints.identical` 组内 | (−1)^{L+S} = −1 |
| 非自共轭（K⁺η、π⁺π⁰） | 其余 | C 未定义，只查 J^P |
| 产生顶点 A→R+B | 任意 | 角动量三角 + P_A = P_R·P_B·(−1)^{L′}；若 A、R、B 均有 C：C_A = C_R·C_B |
