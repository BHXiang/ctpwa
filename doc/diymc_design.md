# DIYMC：用振幅模型产生相空间 / 信号 MC 的调研与实施计划

状态：**P1/P2/P3 已实现并测试**（GPU 数值验证 + CPU 后端 smoke）
日期：2026-10-23
适用版本：ctpwa v0.3.14

> **决策记录（2026-10-23 评审）**：首版采用 **native-first**（自写 GENBOD），
> ROOT `TGenPhaseSpace` 保留为可选对照/后备后端。首版范围：平坦相空间 +
> unweighted 舍选 + DAT/ROOT 输出；weighted、boosted 母粒子、探测器 smearing 后置。

---

## 0. 结论先行

1. **可行，且不需要新算法。** ROOT `TGenPhaseSpace` 的 `Generate()` 就是 CERNLIB
   GENBOD（Raubold–Lynch）算法：先抽有序的嵌套不变质量，再逐级做两体衰变 +
   随机旋转 + boost。它返回的 `weight` 不是“事件权重”这么简单——**必须按该
   weight 舍选，才能得到均匀相空间**；直接把 `Generate()` 的事例全部收下会得到
   有偏分布。这一点已用本机 ROOT 6.40.02 + PyROOT 数值验证（PyROOT 只用
   `/usr/bin/python3.14` 做参考验证；ctpwa 本身仍是 3.12 的扩展模块）。
2. **评审决定首版走 native-first**：自写 GENBOD（约 100 行，`std::mt19937_64`
   局部 RNG），按已发表的 GENBOD 公式实现（§2.4 的 Python 原型已对 ROOT 验证），
   不逐行复制 ROOT 源码（许可证，见 §2.6）。ROOT `TGenPhaseSpace` 封装保留为
   **可选对照/后备后端**：3 体 ~10.6M ev/s、4 体 ~6.1M ev/s（本机，标量单线程），
   适合做交叉验证；若 native 移植出问题可立即切回。
3. **振幅加权不需要重写振幅引擎。** ctpwa 已有“对任意四动量批量算振幅”的
   内部路径（`AmpCasDecay::computeSLAmps` → `analysis::calculateAmplitudes` →
   `AmpCalc::reComputeAmps`），`getFitFractions/getEfficiency` 的
   `computeFileBatchIntegrals` 就是先例。新增一个**任意事例的逐事件强度 API**
   即可：临时换 `events_offsets_/amp_offsets_` + 临时 `AmpCalc` + 现有的
   cublas 求 |A·v|²。
4. **舍选流程**：平坦相空间（GENBOD 按 wt 舍选）→ 用拟合参数算逐事件
   `I=|ΣA·v|²`（对极化求和）→ 按 `u·Imax < I` 舍选。`Imax` 用 pilot 估计并
   自适应抬高；若抬高，用保存的 `u` 对已接受事例重新判定，保证无偏。
5. **建议的 MVP（约 4.5–5.5 人日，含测试/文档 6–7 人日）**：
   `ctpwa.PhaseSpaceGenerator`（ROOT backend，平坦相空间，DAT/ROOT 输出）+
   `analysis.getIntensity` + `analysis.generateMC`（unweighted 舍选）+ 测试/文档。
   后续再补 native backend、weighted 模式、链感知重要性抽样、探测器 smearing。

---

## 1. 现状与缺口

### 1.1 ctpwa 的事件模型

- `Particles` 定义所有粒子（含母粒子）的 `J/P/mass`。
- `Data.order` 只列**末态粒子**；`readMomentaFromDat` 按 `order` 读 DAT/ROOT，
  并把“不在 order 里的粒子”当作初态，其四动量由末态求和重建
  （`src/Analysis.cu:86-160`）。所以：

  > 母粒子通常是 `Particles` 中不在 `Data.order` 里的那个粒子（多候选取 §5.2
  > 的链推断/显式指定）；产生 MC 时只需输出末态，母粒子由分析程序自动重建。

- 文件格式：
  - DAT：每行 `E px py pz`，按 `Data.order` 循环；
  - ROOT：`TTree` + 每个末态一个 `TLorentzVector` 分支（分支名默认=粒子名）。

### 1.2 振幅计算链路（可复用的部分）

```
config.yml
  → ConfigParser（Particles/Data/DecayChains）
  → DecayInfo（链×[J,P]×共振态 → 振幅列表/耦合矩阵）
  → AmpCasDecay::computeSLAmps(finalMomenta)   // 任意四动量 → 角向 SL 因子
  → AmpCasDecay::getAmps(...)                  // 共振态线形 × SL → A[e,p,i]
  → AmpCalc::reComputeAmps(theta)              // 用拟合的 mass/width 重算 A
  → computeResults / computeTotalIntensity     // |Σ_i A_i v_i|²，对极化求和
```

关键事实：

- `analysis::calculateAmplitudes(Vp4, AmpCalc*)` 已经支持**传入任意
  `Vp4`（粒子名 → 四动量 vector）**；`computeFileBatchIntegrals`
  （`src/Analysis.cu:4270-4400`）就是按文件分批调用它，临时改写
  `events_offsets_/amp_offsets_` 后再恢复。
- 因此“给任意相空间事例算振幅”只差一个公开入口，不需要新的物理代码。
- `NLLFunction` 的强度来自 `d_all_amplitudes_`（已物化的 A 表）与扩展耦合向量
  `v_ext`；`computeResults` 的 `computeModTotalWeight` 核会输出**逐事件**
  `Σ_pol |Σ_i A_i v_i|²`（`src/ComputeResults.cu:30-53`）。
- 精度：double 构建下 A 可能是 `float2` 存储（`precision: auto/float`），消费前
  需 `upcastAmpSegToDouble`（`src/Analysis.cu:1382`）。

### 1.3 现有 MC 来源与问题

- 生产输入完全依赖外部 `data`/`phsp` 文件；ctpwa 内部**没有**相空间产生器。
- `tests/generate_test_data.py`（3 体）用 Dalitz 平面均匀抽样 + 边界拒绝，
  对 3 体是正确平坦相空间；`tests/generate_4body_data.py` 等对中间质量**均匀
  抽样**，没有 GENBOD 的 `PDK` 权重，**不是**平坦 n 体相空间（脚本自己也声明
  “只要求物理允许”）。用它做效率/归一化测试会有系统偏差。
- 因此 DIYMC 同时补两个缺口：**通用平坦相空间** + **按振幅加权的信号 MC**。

---

## 2. ROOT `TGenPhaseSpace` 研究

源码：`math/physics/src/TGenPhaseSpace.cxx`（ROOT）
<https://github.com/root-project/root/blob/master/math/physics/src/TGenPhaseSpace.cxx>
算法：CERNLIB W515 GENBOD，Raubold–Lynch 方法，F. James, *Monte Carlo Phase
Space*, CERN 68-15 (1968)。

### 2.1 `SetDecay(P, n, masses)`

```
M      = P.M()                     # 母粒子不变质量
fTeCmTm = M - Σ m_i                # CM 中的“多余能量”
# 最大权重（只用于把 wt 归一到 ≤1）
emmax = fTeCmTm + m_0; emmin = 0; wtmax = 1
for i = 1..n-1:
    emmin += m_{i-1}
    emmax += m_i
    wtmax *= PDK(emmax, emmin, m_i)
fWtMax = 1 / wtmax
fBeta  = P 的 boost 参数（母粒子非静止时最后统一 boost）
```

其中

```
PDK(a,b,c) = sqrt((a-b-c)(a+b+c)(a-b+c)(a+b-c)) / (2a)
```

### 2.2 `Generate()`：产生一个事例

```
rno[0] = 0
rno[1..n-2] ~ U(0,1)，排序
rno[n-1] = 1

# 嵌套子系统的“不变质量”：invMas[0]=m_0，invMas[n-1]=M
invMas[i] = rno[i] * fTeCmTm + Σ_{j≤i} m_j

# 每一级两体衰变的动量
pd[i] = PDK(invMas[i+1], invMas[i], m_{i+1})     # i=0..n-2

wt = fWtMax * Π_i pd[i]                          # 返回的权重

# 构造四动量（Raubold–Lynch）
particle 0: (E0, 0, +pd[0], 0)                    # 沿 y 轴
for i = 1..n-1:
    particle i: (Ei, 0, -pd[i-1], 0)
    对 j≤i 的所有粒子做随机旋转（先绕 z，再绕 y）
    if i < n-1:
        β = pd[i] / sqrt(pd[i]² + invMas[i]²)
        对 j≤i 的所有粒子沿 y boost β
# 最后按母粒子的 β 把全部粒子 boost 到实验室系
```

### 2.3 权重与“均匀相空间”的关系（关键）

对 n 体相空间

```
dΦ_n(P; p_1..p_n) = (2π)^4 δ⁴(P-Σp_i) Π_i d³p_i / ((2π)³ 2E_i)
```

GENBOD 的抽样密度并不是 dΦ 的均匀分布；`wt` 正是“抽样密度 → dΦ”的
Radon–Nikodym 因子（归一化到 ≤1）。因此：

- **要平坦相空间**：产生后以概率 `wt` 接受（`u < wt`，`u~U(0,1)`）。
- **要加权事例**：保留 `wt`，后续按权重使用。
- **直接收下所有 `Generate()` 事例**：分布有偏，不能当 flat phsp 用。

> 这是调研中最重要的结论。很多用法示例只调用 `Generate()` 而不看返回值；
> 3 体也会有可见偏差（上例 wt 从 0.0024 变到 0.44），4/5 体更明显。

### 2.4 本地数值验证

用 ROOT 6.40.02 + `/usr/bin/python3.14`（PyROOT）验证：

| 检验 | 设置 | 结果 |
|---|---|---|
| 四动量守恒 | J/ψ(3.0969)→K⁺K⁻η，2 万事例 | max\|Σp\| ≈ 8.9e-16；max\|ΣE−M\| ≈ 1.3e-15 |
| wt 范围 | 同上，4 万事例 | wt ∈ [0.0024, 0.4448]，均值 0.3412，WtMax=0.6393 |
| wt 舍选接受率 | 同上 | 34.1% |
| wt 加权 vs wt 舍选 | 20 万事例，m(K⁺K⁻) CDF | 最大 CDF 差 0.0026（统计量级） |
| native 复现 vs ROOT | 10 万事例，m²(K⁺K⁻) | 均值 3.3325 vs 3.3313；最大 bin 差 0.0014 |
| 4 体平坦接受率 | ψ(3686)(3.686)→K⁺K⁻ηπ⁰ | ≈ 6.8% |
| 5 体平坦接受率 | ψ(3686)→4π+η | ≈ 1.2% |

结论：native 复现与 ROOT 一致；n≥4 的“平坦化”接受率下降很快，这是
**相空间产生本身**的效率，与振幅无关。产生器必须足够快（C++ 标量即可），
且应支持分批流式产生，不能把所有 trial 都存下来。

### 2.5 移植注意

- ROOT 源码里初始动量沿 **y**、先绕 z 再绕 y、再沿 **y** boost；照抄时轴不要混。
- `wt` 理论上 ≤1；数值上做 `min(max(wt,0),1)` 保护。
- ROOT 上限 18 体（`kMAXP`）；native 实现不设该限制，但实际分析很少 >6 体。
- ROOT 用全局 `gRandom`；native 用局部 `std::mt19937_64`，可复现且线程安全。
- 母粒子非静止时最后统一 boost；ctpwa 的角度在各自静止系中计算，默认产生
  母粒子静止系即可（极化轴约定见 §3.4）。

### 2.6 直接调用 ROOT vs 自写 native：对比与建议

| 维度 | 直接调用 `TGenPhaseSpace` | 自写 native GENBOD |
|---|---|---|
| 正确性 | 高：ROOT 实现、长期使用；我们只负责 wt 舍选封装 | 取决于移植质量；已用 Python 原型对 ROOT 验证（10 万事例，最大 bin 差 0.0014），C++ 移植仍需测试 |
| 依赖/构建 | ctpwa 已经链 ROOT（`-lPhysics`）；只需 include 头文件，**不用改 setup.py** | 纯 host C++，无 ROOT 依赖；但 `Config.cu` 是 inline TU，需沿用 main.cu include 模式 |
| 性能 | 3 体 10.6M ev/s、4 体 6.1M ev/s（标量单线程，本机） | 同量级或略快；可批量化/SIMD，但生成不是瓶颈，振幅计算才是 |
| 随机数/复现 | 全局 `gRandom`：`SetSeed` 可复现；但会被同进程其它 ROOT RNG 调用干扰，跨 ROOT 版本不保证逐位一致 | 局部 `std::mt19937_64`：每个 generator 独立、可复现、跨版本稳定 |
| 线程安全 | `gRandom` 非线程安全；多线程生成需全局 mutex 或串行化 | 天然线程安全（每个实例自己的 RNG） |
| 末态数上限 | 硬上限 18（`kMAXP`） | 无上限 |
| 可控性/可维护性 | 行为由 ROOT 决定；出问题可查上游 | 我们 own 算法；可加链感知重要性抽样、诊断、优化 |
| 许可证 | 动态链接 LGPL 的 ROOT 不改变 ctpwa 的 MIT；干净 | 若逐行复制 ROOT 源码会有 LGPL 衍生风险；应按 CERN 68-15/GENBOD 公式 clean-room 实现并注明出处 |
| 开发成本 | 0.5–1 天（封装 + wt 舍选 + 测试） | 1–1.5 天（实现 + 2..5 体验证 + 边界） |
| 适用场景 | 3–6 体、单线程、常规 PWA | >18 体、并行/多 generator、严格复现、ROOT-free |

**评审决定**：首版 **native-first**（`backend="native"`，默认），原因是用户
更看重局部 RNG/线程安全/无 18 体上限，并接受多 0.5–1 天的实现 + 验证成本。
ROOT 后端保留为 `backend="root"`（可选），用于交叉验证/后备；两者的
`PhaseSpaceBackend` 接口和上层 API 完全一致。

> 若 native 移植在测试中暴露问题，可以立即把默认切回 `root`，不影响 P2/P3。

---

## 3. 在 ctpwa 中做振幅加权的可行性

### 3.1 任意四动量的逐事件强度

新增（`analysis` 的公开方法，内部实现）：

```cpp
torch::Tensor getIntensity(torch::Tensor momenta,   // [N, n_final, 4] (E,px,py,pz), CPU f64
                           torch::Tensor params);   // [2*n_free + n_theta] f64
```

内部步骤：

1. `momenta` → `std::map<particle, vector<LorentzVector>>`（按 `Data.order`），
   母粒子四动量 = 末态和。
2. 保存 `events_offsets_ / amp_offsets_`；按 GPU 数切分，设置本批 offsets。
3. `AmpCalc tmp;`（**局部实例**，不污染拟合用的 `amp_calc_`）
   `auto amps = calculateAmplitudes(Vp4_per_gpu, &tmp);`
4. `params` → `(vector, theta)`（`Parameters::splitParams`）→
   `v_ext = freeParamsToAmplitudes(vector)`；`theta` 用于
   `tmp.reComputeAmps(amps, theta, ...)`。
5. 若 `float_amps_`，把 A 段 upcast 成 double。
6. 每个 GPU 上 `computeTotalIntensity(amps, v_ext, out, ...)`：
   `cgemv` 得 `S[e,p]=ΣA·v`，再 `Σ_p |S|²`。
7. 拷回 host、释放 A 表、`tmp.release()`、恢复 offsets。

> `computeResults` 也能算逐事件总值，但它还会分配一个与 A 同尺寸的
> `d_result_matrix` 并计算 partial/干涉矩阵，生成 MC 时纯属浪费。建议在
> `src/ComputeResults.cu` 新增一个只做 cgemv + `computeModTotalWeight` 的
> `computeTotalIntensity`（约 40 行）。

### 3.2 舍选抽样

目标分布 `dP/dΦ ∝ |M(x)|²`。流程：

```
flat = GENBOD 按 wt 舍选得到的平坦事例
I    = getIntensity(flat, params)            # 对极化求和
接受  ⇔  u * Imax < I,   u ~ U(0,1)
```

- `Imax` 初值：pilot（例如 min(batch, max(1000, N/10)) 个平坦事例）的
  `max(I) * 1.05`。
- 后续 batch 若 `max(I) > Imax`：抬高 `Imax`，并对**已接受**事例用保存的
  `(u, I)` 重新判定 `u*Imax_new < I` —— 这是精确的无偏修正；比“全部重来”
  省事，比 tf-pwa 的 `old/new` 近似更严格。
- 统计量：`n_trials_genbod`、`n_flat`、`n_intensity_eval`、`n_accepted`、
  `acceptance`、`max_weight`、耗时；接受率过低时给出警告。
- 可选 `mode="weighted"`：不做舍选，返回全部平坦事例 + `I` 权重；适合
  快速 toy 或需要保留统计量的场景。

### 3.3 归一化与闭包检验

- 舍选用的是 `I` 的**比值**，整体归一化常数无关。
- 产生的样本写成 `data` 后，拟合仍用原 `phsp` 做归一化；只要 `phsp` 是平坦
  相空间（或与产生器一致），闭包检验成立。
- 推荐回归测试：拟合 → 用拟合参数产生 N 个事例 → 以产生样本为 `data` 重拟合
  → 参数/份额在误差内一致。
- 若用户把 `phsp` 换成自产平坦样本，注意 `Constraints.free_phsp_amplitudes`
  等开关仍按原逻辑工作。

### 3.4 自旋、极化、全同粒子

- `n_polar_` = 所有粒子 `(2J+1)` 的乘积；`computeResults` 对**所有**极化态
  求和 `|ΣA·v|²`。这与拟合的强度定义一致：
  - 初态未极化 → 对初态极化求和；
  - 末态自旋未测量 → 对末态自旋求和；
  - `Particles.polarization` 掩码 → 只对允许态求和。
- 全同粒子：产生器按 `Data.order` 给粒子不同标签；振幅侧已由
  `Constraints.identical` + `computeSLAmps` 的置换拓扑处理，无需在产生器里
  对称化。
- 母粒子默认静止；若以后支持 boosted 产生，需要明确极化轴（ctpwa 以 z 轴为
  参考），否则 boosted 与 rest 的物理不完全等价。

---

## 4. API 设计

### 4.1 Python 层（建议）

```python
import ctpwa

# 1) 平坦相空间：只解析 config，不需要 data/phsp 文件，不占 GPU
gen = ctpwa.PhaseSpaceGenerator("config.yml")      # 或 mother="Jpsi"
print(gen.mother(), gen.motherMass(), gen.finalStates())
flat = gen.generate(100_000, seed=42)              # torch.float64 [N, n_final, 4] CPU
gen.write(flat, "flat.dat")                        # DAT（E px py pz，按 Data.order）
gen.write(flat, "flat.root")                       # ROOT TTree（TLorentzVector 分支）

# 2) 逐事件强度（高级用法）
ana = ctpwa.analysis("config.yml")
w = ana.getIntensity(flat, params)                 # [N] float64, Σ_pol |M|²

# 3) 振幅加权信号 MC（舍选）
res = ana.generateMC(
    n=10_000, params=params, seed=42,
    mode="unweighted",       # "unweighted" | "weighted"
    batch=100_000,           # 每批平坦事例数（内部按显存自动收缩）
    max_weight=None,         # None=自适应；也可给固定 Imax
    max_trials=None,         # 安全上限；None=不设
)
# res = {"events": [N,n,4], "weights": [N], "stats": {...}}
ana.writeMC(res["events"], "mc.root", weights=res.get("weights"))

# 4) 做 toy/闭包：把产生的样本写成新的 data 文件，改 config 的 Data.data 后
#    重新构造 analysis；phsp 仍用原来的平坦样本（或自产 flat）。
```

设计理由：

- `PhaseSpaceGenerator` 独立于 `analysis`：只依赖 `ConfigParser`（`isValid()`
  只要求 `Particles` 非空），所以可以在没有数据文件、没有 GPU 的环境下产生
  平坦 MC，也方便做单元测试。
- `analysis.generateMC` 复用已有模型/参数，避免用户自己拼 `A` 表。
- `getIntensity` 是低层原语，便于高级用户做自定义重要性抽样/诊断。
- 注意：`analysis` 构造期会读 `data`/`phsp`（`data_files.at(...)`），所以
  `generateMC` 仍需要一个“可构造”的 analysis。若只想做纯 toy、不碰真实数据，
  可以先用 §4.1 的 `PhaseSpaceGenerator` 产生一个小 flat 文件，或在后续版本
  增加 `analysis(config, load_data=False)` 的 model-only 模式（首版不做）。

### 4.2 C++ 层（建议签名）

```cpp
// include/PhaseSpace.cuh
class PhaseSpaceGenerator {
public:
    explicit PhaseSpaceGenerator(const std::string& config_file,
                                 const std::string& mother = "",
                                 const std::string& backend = "root");
    explicit PhaseSpaceGenerator(const ConfigParser& parser,
                                 const std::string& mother = "",
                                 const std::string& backend = "root");
    const std::string& mother() const;
    double motherMass() const;
    void setMotherMass(double m);                     // 覆盖（线形/离壳分析用）
    const std::vector<std::string>& finalStates() const;
    // 平坦相空间（内部按 wt 舍选）；返回 [N, n, 4] CPU f64
    torch::Tensor generate(int64_t n, uint64_t seed = 0);
    // 加权相空间：返回 (events, wt)（首版可不实现）
    std::pair<torch::Tensor, torch::Tensor> generateWeighted(int64_t n, uint64_t seed = 0);
    void write(const torch::Tensor& events, const std::string& filename,
               const torch::Tensor& weights = {});
private:
    ConfigParser parser_;
    std::string mother_;
    double mother_mass_;
    std::vector<std::string> final_states_;
    std::vector<double> masses_;
    std::string backend_;
    // ROOT backend: TGenPhaseSpace + 全局 gRandom 串行锁
    // native backend: std::mt19937_64 + GENBOD（可选）
};

// analysis 新增
torch::Tensor getIntensity(torch::Tensor momenta, torch::Tensor params);
pybind11::dict generateMC(int64_t n, torch::Tensor params,
                          uint64_t seed = 0, std::string mode = "unweighted",
                          int64_t batch = 100000,
                          pybind11::object max_weight = py::none(),
                          pybind11::object max_trials = py::none());
void writeMC(torch::Tensor events, const std::string& filename,
             pybind11::object weights = py::none());

// src/ComputeResults.cu 新增
void computeTotalIntensity(const ctComplex* d_matrix, const ctComplex* d_vector,
                           double* d_total_result, int nEvents, int ngls, int npolar);
```

### 4.3 输出格式

- **DAT**：每行 `E px py pz`，精确保留（`std::setprecision(17)`），按
  `Data.order` 循环；可直接作为 `data`/`phsp` 输入。
- **ROOT**：`TTree("mc", ...)`，每个末态一个 `TLorentzVector*` 分支（与
  `readMomentaFromDat` 的 `SetBranchAddress(&TLorentzVector*)` 约定一致）；
  可选 `weight/D`。这样现有 reader 不用改。
- DAT 没有逐事件权重列：weighted 模式请写 ROOT，或另存一个权重文本文件
  （`readWeightsFromFile` 已支持）。unweighted 模式的 DAT 可直接当 `data`/`phsp`。

---

## 5. 实现方案

### 5.1 新增/修改文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `include/PhaseSpace.cuh` | 新增 | `PhaseSpaceGenerator` 声明 + `PhaseSpaceBackend` 接口（ROOT/native） |
| `src/PhaseSpace.cu` | 新增 | ROOT backend 实现（首版）；native backend 可选；**由 `main.cu` `#include`（同 `Config.cu`/`Analysis.cu` 的 inline 模式）**，不改 `setup.py` |
| `src/Analysis.cu` | 改 | `getIntensity` / `generateMC` / `writeMC` |
| `src/ComputeResults.cu` | 改 | `computeTotalIntensity`（cgemv + 逐事件模方） |
| `include/AmpGen.cuh`, `src/AmpGen.cu` | 改 | `AmpCalc::release()`：释放 ResBlock/JIT 设备缓冲并 clear |
| `src/main.cu` | 改 | pybind 绑定 |
| `tests/test_mc.py` | 新增 | 相空间/强度/舍选/闭包测试 |
| `doc/diymc_design.md`, `README.md` | 文档 | 本文件 + 用法 |

### 5.2 相空间产生器

- 母粒子识别：
  1. 显式 `mother=` 优先；
  2. 否则从 `DecayChains` 找“从未作为 daughter 出现”的顶层 mother；
  3. 多个候选 → 报错并提示显式指定；
  4. 校验 `mother` 不在 `Data.order`，且 `Data.order` 中每个名字都在 `Particles`。
- 末态 = `Data.order`（为空时从链的末态推断，但建议强制要求）。
- 2 体不需要特判；GENBOD 在 n=2 时 `wt=1`，自然退化为两体。
- 平坦产生：`while (out < n) { genbod → (mom, wt); if (u < wt) accept; }`。
- 分批流式：每批产生 `batch` 个平坦事例后就交给强度计算，不缓存全部 trial。
- seed：
  - ROOT backend：`gRandom->SetSeed(seed)`；全局 RNG，生成段用**全局 mutex**
    串行化（同进程多 generator 共享一把锁）；
  - native backend（可选）：`std::mt19937_64`，每实例独立。
- 统计：`n_trials`、`n_accepted`、`wtmax`；ROOT backend 的 `wtmax` 来自
  `TGenPhaseSpace::GetWtMax()`。

### 5.3 强度计算

- 输入张量校验：`dim==3`、`size(1)==n_final`、`size(2)==4`、float64；CPU 或
  CUDA 均可（内部转 host）。
- 按 GPU 切分时保证每个 GPU 至少 1 个事例；若 `N < n_gpus`，只用一个 GPU
  （需要给 `calculateAmplitudes` 一个“使用几个 GPU”的参数，或把 N 补齐到
  `n_gpus` 后丢弃重复——推荐前者，改动很小）。
- `v_ext` 在每个 GPU 上各一份（从主 GPU 拷贝，避免跨卡读）。
- `float_amps_`：先 `upcastAmpSegToDouble` 再调 `computeTotalIntensity`。
- 异常安全：用 RAII guard 保存/恢复 `events_offsets_`/`amp_offsets_`，并保证
  `tmp.release()` + `cudaFree` 在抛异常时也执行。

### 5.4 舍选循环

```cpp
std::vector<Event> accepted; std::vector<double> u_acc, I_acc;
double Imax = 0;
int64_t trials = 0;
while (accepted.size() < n) {
    auto flat = ps_gen.generate(std::min(batch, remaining_guess), seed_stream++);
    auto I = getIntensity(flat, params);
    double m = I.max().item<double>();
    if (m > Imax) {
        Imax = m * 1.05;
        // 对已接受事例重新判定（精确）
        filter accepted by (u_acc[k] * Imax < I_acc[k]);
    }
    for each event with u*Imax < I: accept;
    if (max_trials && trials > max_trials) break/throw;
}
```

- `remaining_guess` 可以按当前接受率动态估计（避免最后一批过大）。
- `mode="weighted"`：跳过 while，直接把 `flat` 与 `I` 返回/写出。
- 进度/诊断：每批打印或通过 stats 返回；Python 侧可加 `verbose`。

### 5.5 `AmpCalc` 生命周期（重要）

- 生成时每个强度 batch 用一个**局部** `AmpCalc`，通过
  `calculateAmplitudes(Vp4, &tmp)` 注册，`reComputeAmps(theta)` 后算强度，
  最后 `tmp.release()`。
- 不能复用 `analysis::amp_calc_`：`calculateAmplitudes` 每次都会
  `addBlock`，会追加 block/slot，污染拟合参数映射；也不能只析构不释放
  （`AmpCalc::~AmpCalc` 有意不 free，避免退出期段错误）。
- `release()` 需要释放：每个 `ResBlock` 的 `d_resonances`、`d_all_params`、
  `d_all_channels`、`d_T`、`d_dF`、`d_res_idx_`、`d_param_idx_`、
  `d_global_offset_`、`d_param_map_`、`d_global_idx_`、JIT 的
  `grad_buf/full_buf/grad_slice/res_off_dev` 等；然后 `clear()` 三个 vector。
  `cas_list_` 里的 `AmpCasDecay` 析构会释放 SL/momenta 表。
- 若实测“每 batch 重新注册”太慢，后续优化：固定 batch 大小 + 复用已注册的
  `cas`，只调 `computeSLAmps` + `getAmps`；这需要把 `calculateAmplitudes`
  的“建链/注册”和“算表”拆开，属于第二阶段优化。

> **构建注意**：`ConfigParser` 的方法定义在 `src/Config.cu`，而它只被
> `src/main.cu` 以 `#include "Config.cu"` 的方式编进同一个 TU，`setup.py`
> 并没有单独编译它。因此 `PhaseSpace.cu` 不能作为独立源文件直接加进
> `setup.py`（会缺 `ConfigParser` 符号）；应沿用现有模式：在 `main.cu` 里
> `#include "PhaseSpace.cu"`（放在 `Config.cu` 之后、`Analysis.cu` 之前），
> 或把 GENBOD/配置解析写成 header-only 的 `include/PhaseSpace.cuh`。

### 5.6 多 GPU / 精度 / 边界

- 多 GPU：沿用 `computeFileBatchIntegrals` 的切分方式；每个 GPU 的 `A` 表、
  `v_ext`、输出分别处理。
- 精度：`precision: auto/hybrid/float` 下 A 可能是 float2；强度计算前 upcast。
  测试要覆盖 double 与 float 构建（至少 float 档的 `getIntensity`）。
- 边界：
  - `M <= Σm_i` → 明确报错（含数值）；
  - 零质量粒子（γ）→ PDK 正常；
  - `N=0` / `n=1` → 报错；
  - 母粒子在 `Data.order` 中 / 多个候选 mother → 明确报错；
  - 事例数小于 GPU 数 → 单 GPU 路径；
  - `max_trials` 用尽 → 返回现有样本 + 警告（或抛错，由参数控制）。

---

## 6. 测试计划

### 6.1 相空间（`tests/test_mc.py`）

| 测试 | 判据 |
|---|---|
| 2 体 | 动量大小=PDK，背对背，wt=1 |
| n=3,4,5 守恒 | `max|Σp| < 1e-10`，`max|ΣE−M| < 1e-10` |
| 质量壳 | 每个末态 `|E²−p²−m²| < 1e-10` |
| 种子复现 | 同 seed 逐位相同；不同 seed 不同 |
| 3 体平坦性 | Dalitz 平面内部小区间 2D 直方图 `chi2/ndf ≈ 1` |
| wt 加权 vs 舍选 | 同一批 GENBOD 事例，加权分布与舍选后分布 KS/chi2 一致 |
| native vs ROOT | 仅 native backend：10 万事例 m² 分布最大 bin 差 < 0.01 |
| 阈值错误 | `M < Σm` 抛异常且信息可读 |
| DAT 往返 | 写 → `analysis` 读 → 四动量差 < 1e-9 |
| 18 体上限 | ROOT backend 下清晰报错（native 不限） |

### 6.2 强度

| 测试 | 判据 |
|---|---|
| vs `writeResult` | 同一 `phsp` 上 `getIntensity / totalweight` 逐事件为常数（相对偏差 < 1e-6） |
| 多 GPU | 单卡与多卡结果逐事件一致（< 1e-6） |
| 精度 | double / float 档都在容差内 |
| 多次调用 | 连续 5 次 `generateMC` 后 `cudaMemGetInfo` 空闲显存稳定（无泄漏） |
| 状态隔离 | 调用后 `getNLL`/`reCalcAmp` 仍与调用前一致（局部 AmpCalc 不污染） |

### 6.3 舍选 / 闭包

| 测试 | 判据 |
|---|---|
| 分布一致性 | AR 样本与“平坦样本 × I 权重”的 1D/2D 分布 KS/chi2 一致 |
| 自适应 Imax | 故意给低初值，最终样本仍与高初值一致 |
| 闭包 | 拟合 → 产生 N=5000 → 重拟合，参数/份额在 ~2σ 内 |
| flat 复用 | 自产 flat 作为 `phsp` → 拟合正常、FF 和为 1 |
| 输出往返 | `writeMC` ROOT → `analysis` 读取 → 事件一致 |

---

## 7. 里程碑

| 阶段 | 内容 | 交付物 | 估计 |
|---|---|---|---|
| P1 | `PhaseSpaceGenerator` + **native backend** + wt 舍选 + DAT/ROOT 输出 | flat MC 可用，单元测试 | 1–1.5 d |
| P1b（可选） | ROOT backend + 与 native 交叉验证 | 可切换后端/后备 | 0.5 d |
| P2 | `computeTotalIntensity` + `analysis.getIntensity` + `AmpCalc::release` | 任意事例强度，vs `writeResult` 测试 | 2 d |
| P3 | `analysis.generateMC`（unweighted AR、自适应 Imax、多 GPU、stats） | 信号 MC 可用，分布/闭包测试 | 2–2.5 d |
| P4 | 文档/示例/性能调优（batch、注册复用） | README + `example/generate_mc.py` | 1 d |

**MVP = P1+P2+P3 的 unweighted 路径**，约 4.5–5.5 天；含测试/文档/调优约 6–7 天。
（若首版就做 native，P1 换成 native + 交叉验证，约多 0.5–1 天。）

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| `Imax` 低估 | 样本有偏 | pilot + 自适应 + 保存 `u` 精确重判；可选用已有 `phsp` 的 max |
| n≥4 平坦化效率低（6.8%/1.2%） | 产生慢 | 流式产生；只对**平坦化后**的事例算振幅；后续链感知重要性抽样 |
| 每 batch 重新注册 `AmpCalc` 慢 | 生成慢 | 先测量；必要时拆“建链/算表”，固定 batch 复用 cas |
| 临时 A 表爆显存 | OOM | batch 按 `n_amp*n_pol` 自动收缩；环境变量覆盖；分块 |
| `AmpCalc` 泄漏/状态污染 | 长进程崩溃/拟合错 | 局部实例 + `release()` + 泄漏回归测试 + 状态隔离测试 |
| float 精度路径漏 upcast | 强度错 | 统一在 `computeTotalIntensity` 入口 upcast；float 档测试 |
| 多 GPU 空分片 | 崩溃 | 保证每卡 ≥1 事例；`N < n_gpus` 走单卡 |
| 母粒子/末态识别歧义 | 产生错粒子 | 显式 `mother=` + 链推断 + 清晰报错 |
| ROOT `gRandom` 全局状态 | 复现受同进程其它 ROOT RNG 影响；多线程不安全 | 生成段串行化（全局 mutex）；文档说明；需要严格隔离时切 native |
| ROOT 18 体上限 | 无法产生 >18 体 | 首版明确报错；需要时切 native |
| native 移植 bug | 分布/权重错 | 2..5 体守恒/质量壳/平坦性测试 + 与 ROOT 交叉验证；按 GENBOD 公式 clean-room |
| native 复制 ROOT 源码 | MIT 许可证风险 | 只按 CERN 68-15/GENBOD 公式实现，注明出处，不复用 ROOT 代码/注释 |

---

## 9. 待确认问题

1. **后端选择**：已决定 native-first；ROOT backend 作为可选对照/后备。
2. **API 形态**：`analysis.generateMC` + 独立 `PhaseSpaceGenerator` 是否符合预期？
   还是想要一个 `ctpwa.DIYMC(config)` 一站式类？
3. **输出优先级**：DAT + ROOT 都做（你已勾选），还是先 DAT、ROOT 后置？
4. **是否需要 weighted 模式**（不做舍选，直接给权重）？首版你只勾了 unweighted，
   可以后置。
5. **母粒子**：默认静止即可，还是需要支持 boosted（给定母粒子四动量）？
6. **最大末态数**：典型 3–5 体，还是可能 >18 体（决定是否必须 native）？
7. **探测器效应**：这一版只做 generator-level，还是要把效率/分辨 smearing
   一并纳入（会显著增加复杂度）？
8. **性能目标**：典型 N 和 `n_amplitudes` 是多少？这决定 batch 默认值和是否
   需要做“注册复用”优化。
9. **闭包检验**是否作为首版验收标准（需要改测试数据/拟合流程）？

---

## 10. 实现进度（2026-10-23）

### P1（已完成）

- 新增 `include/PhaseSpace.cuh` + `src/PhaseSpace.cu`（native GENBOD），由
  `src/main.cu` inline include；未改 `setup.py`。
- 绑定 `ctpwa.PhaseSpaceGenerator`：
  - `mother()/motherMass()/setMotherMass()/finalStates()/nFinal()/weightMax()/lastTrials()`
  - `generate(n, seed=0)` → `[n, n_final, 4]` CPU float64（wt 舍选后的平坦 dΦ）
  - `generateWeighted(n, seed=0)` → `(events, weights)`
  - `write(events, filename, weights=None)` → `.dat` / `.root`
- `main.cu` 的模块初始化加了 `pybind11::module_::import("torch")`，修掉
  `import ctpwa`（未先 `import torch`）时返回 `at::Tensor` 的崩溃。
- 测试 `tests/test_mc.py`（13 项，全部通过）：
  - `wtmax` 与 ROOT 的 `GetWtMax()` 一致（0.6393028545698538）
  - 2/3/4/5 体四动量守恒、质量壳、seed 复现
  - 2 体 `wt=1`
  - 3 体中心 Dalitz 区 `chi2/ndf` 平坦性
  - `setMotherMass`、DAT 往返、ROOT 输出、阈值/母粒子错误
- 构建命令（本机无 GPU，需显式 arch）：
  `TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 python3 setup.py build_ext --inplace`
- ROOT 输出已用独立 C++ 程序按 `SetBranchAddress(&TLorentzVector*)` 读回验证，
  并在 `tests/test_mc.py` 中用 uproot 校验分支名与 `fE/fP` 数值；与
  `readMomentaFromDat` 的约定一致。

### P2/P3（已完成，2026-10-25）

- `include/ComputeResults.cuh` + `src/ComputeResults.cu`：新增
  `computeTotalIntensity`（cgemv + `computeModTotalWeight`，不物化
  result/partial/干涉矩阵）。
- `include/AmpGen.cuh` + `src/AmpGen.cu`：新增 `AmpCalc::release()`，释放
  ResBlock/JIT/offset cache 等持久化缓冲并清空 block/slot/cas（析构函数仍保持
  进程退出期不 free 的约定）。
- `src/Analysis.cu`：
  - `calculateAmplitudes(..., n_gpus_used)`：N < n_gpus 时只启用前 N 个 GPU。
  - `getIntensity(momenta, params)`：临时 offsets + 局部 `AmpCalc` +
    `calculateAmplitudes` + `reComputeAmps(theta)` + `computeTotalIntensity`，
    RAII 恢复 offsets、释放临时表；返回 CPU `[N]`。
  - `generateMC(n, params, seed, mode, batch, max_weight, max_trials)`：
    平坦相空间 → I → `u*Imax<I`；Imax 自适应抬高时用保存的 `u` 精确重判；
    `weighted` 模式返回平坦事件 + I 权重；stats 含 acceptance/flat_acceptance/
    n_genbod_trials/max_weight/truncated。
  - `writeMC`：复用 `PhaseSpaceGenerator::write` 写 DAT/ROOT。
- **关键坑**：临时 `AmpCalc` 必须 `setFloatOutput(float_amps_)` +
  `setDFloat(prec_mode_ == PrecMode::Float)`，否则 `reComputeAmps` 会把 double
  写进 float2 A 表（实测全 NaN）。
- 测试 `tests/test_mc_amplitude.py`（9 项，全部通过）：
  - `getIntensity` vs `writeResult(totalweight)`：比值相对展宽 ~4e-16
    （nominal + 扰动 theta 两种）；
  - 状态隔离：调用前后 `getNLL` 完全一致；连续 4 次调用空闲显存差 0 MiB；
  - unweighted：形状/守恒/质量壳/`I<=Imax`/seed 复现；
  - weighted：权重有限且 `max_weight=weights.max()`；
  - `max_trials` 截断、AR vs 加权 m²(K+K-) 分布、DAT/ROOT 往返。
- CPU 后端 smoke（`CTPWA_DEVICE=cpu`）：P1 13 项测试通过；
  `getIntensity` + `generateMC` + `writeMC` 在 200 个事例上正常。
- 构建：GPU 环境可直接 `MAX_JOBS=4 python3 setup.py build_ext --inplace`
  （同时产出 `ctpwa/_cuda.so` 与 `ctpwa/_cpu.so`）。

---

## 附：与 tf-pwa 的对照

- tf-pwa 的 `PhaseSpaceGenerator` 也是“平坦质量抽样 + 权重 flatten +
  `ARGenerator` 舍选”，思路一致；`multi_sampling` 在 max weight 变化时用
  `old/new` 近似重加权，本方案用保存的 `u` 精确重判。
- tf-pwa 的 `gen_mc` 只产生平坦相空间，振幅加权在 `ARGenerator` 里做；
  本方案把两者都收进 ctpwa，并复用现有的 GPU 振幅表，避免 Python 侧
  逐事件拼振幅。
