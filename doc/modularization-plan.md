# CTPWA 模块化改造：评估与施工计划

> 起点问题：感觉"改程序的功能会影响其它地方"，是否应按
> config/DecayInfo、设备管理、SL 振幅、共振态振幅（符号微分）、振幅与参数管理、
> 拟合、结果输出、数据 IO 拆成模块？
>
> **结论：目标分解对，但"再按功能拆一遍文件"本身不解决问题。**
> 现状已有 20 个按功能命名的源文件（18 个 `.cu` + 2 个 `.cpp`），用户列的 8 个轴里
> **7 个在名字上都已存在**。
> 真正的问题是四条边界"名义存在、实际不生效"，以及同一份状态有 2~3 个所有者。
> 模块化的收益只能来自把这两件事变成**编译器/断言能强制**的东西。
>
> 盘点基准 `56d500e`（v0.3.26），下列数字均为实测。

---

## 0. 一句话结论

| 说法 | 判断 |
|---|---|
| "需要模块化" | ✅ 对，但指的是**可强制的边界 + 单一所有权**，不是更细的文件划分 |
| "强设备管理函数能避免 setCudaDevice 改错" | ✅ **完全正确，且应排第一优先级**（性价比最高） |
| "拆了 god class 就少出 bug" | ⚠️ 不会。`analysis` 6300 行读起来痛苦，但 multi-GPU bug 的根因是"3 份 primary device + ambient current device"，不是"类太大" |
| "模块化能修数值 bug（梯度漏项）" | ❌ 不能。那是测试问题，与结构无关 |

---

## 1. 现状：8 个轴在代码里对应什么

| 设想模块 | 今天的位置 | 独立性 |
|---|---|---|
| config / DecayInfo | `Config.cu` 1859 + `Info.cu` 612 + **`AmpGen.cuh` 里的 `SL`/`DecayNode`/`AmpCasDecay`** | ❌ 衰变链描述对象 `AmpCasDecay` 直接持有 GPU 表（`d_slamp_tab_`/`d_mom_sigma_`/`slamp_released_`）与释放逻辑 → 看懂 config 必须看懂 SL kernel |
| 设备管理 | `DeviceManager.cu` 339 | ❌ **`setDevice()` 调用者 0 个**；真实设备逻辑散在 4 个文件、219 处裸调用 |
| SL 振幅计算 | `AmpGen.cu` 4764（`AmpCasDecay` + SL kernel） | ⚠️ 部分：kernel 在本文件，但 `AmpCasDecay` 的生命周期也在这里 |
| 共振态振幅 + 符号微分 | `ResModel.cu` 894 / `Resonance.cu` 448 / `CustomExpr.cu` 1365 / `SymbolicDiff.cuh` | ✅ **唯一今天已经相对干净的轴**（模板在头里，host 剩余在 .cu） |
| 振幅与参数管理 | `AmpGen.cu` 的 `AmpCalc`（147 个成员声明）+ `Parameters.cu` 1116 | ❌ 两个参数管理者：`Parameters` 管 coupling/theta，`AmpCalc::slots_` 管共振态自由参数，`var_equal`/conjugate 各实现一遍 |
| 拟合计算 | `ComputeNLL.cu` 1312 + `analysis::getNLL` 等 | ❌ 编排在 6300 行 / 57 个方法的 `analysis` 里 |
| 结果输出 | 真正的 writer 只有 `analysis::writeResult`(1553 行) / `writeInterfResult` / `writeMC`；同名的另三个**归属不同**：`ComputeResults.cu` 是 kernel（→`fit/`）、`Figure.cu` 是直方图 kernel、`Info.cu` 是 `DecayInfo`（→`config/`，见 §7.2） | ❌ writer 全在 god class 里，外面没有独立模块 |
| 数据文件 IO | `analysis` 的 const 成员里直接 `TFile`/`TTree`（40 处 ROOT 符号） | ❌ 没有独立模块，IO 与拟合共享成员状态 |

**读法**：8 个轴作为**目标分解是对的**——它准确指出了应该存在的接缝。
但今天只有 1 个轴真正独立。所以"再拆一遍"没有增量，得先把接缝变成能拦人的东西。

---

## 2. 四条"名义存在、实际不生效"的边界

这是"改一处影响一片"的四个物理原因。每一条都有实测证据。

### 2.1 编译边界：7 个 `.cu` 不是翻译单元

`src/main.cu` 用文本包含把 7 个 `.cu` 拼成**一个** TU：

```cpp
#include "DeviceManager.cu"   // 339
#include "Config.cu"          // 1859
#include "PhaseSpace.cu"      // 429
#include "Analysis.cu"        // 7878
#include "Info.cu"            // 612
#include "ResModel.cu"        // 894
#include "Resonance.cu"       // 448
```

→ 单个 TU **12,843 行**。`Analysis.cu` 里的 `static` 辅助函数对 `Config.cu`、`Info.cu`
全部可见；`ScopeExit`、`CUDA_CHECK` 这类名字在同一作用域里共享；改 `Config.cu`
一行 → 重编 12.8k 行。

另外 11 个文件是真 TU，但头文件扇入很宽：

| 头 | 被几个 TU 包含 |
|---|---|
| `CustomExpr.cuh` | 7 |
| `AmpGen.cuh` | 6 |
| `SymbolicDiff.cuh` | 6 |
| `Parameters.cuh` / `ComputeNLL.cuh` / `ComputeHessian.cuh` | 4 |

→ 改 `AmpGen.cuh` 里 `ResBlock` 的字段顺序 = 重编大半个项目。
`AmpGen.cuh` 还把 `AmpCalc` 的 147 个成员声明（含全部 `private:`）暴露给 6 个 TU。

**关键**：`-rdc=true` 已开（`setup.py:259`），跨 TU 的 `__device__` 调用本来就能工作
（`evalCustomAll` 就是这么用的）。而且我用静态符号扫描过：**7 个被包含文件之间
跨文件 static 依赖 = 0 处**。也就是说这个 amalgamation 是**白拿的耦合，不承载任何
必要功能**，拆开没有隐藏地雷。

### 2.2 设备所有权边界：模块加了，纪律退了

```
DeviceManager::setDevice() 调用者                    0
裸 cudaSetDevice                                   219
  ├─ 只是"还原主卡"的（cudaSetDevice(primary_dev)）    75   ← 34%
  └─ 多卡循环里的（i/g/gpu/dev.index()）             113
```

**34% 的设备调用只是在撤销另一次设备调用。** 这是 ambient state 的典型症状。

时间线更说明问题：

| 指标 | DeviceManager 诞生时 `08b3c7e`（2026-08-02） | 现在 `56d500e` | 变化 |
|---|---|---|---|
| 裸 `cudaSetDevice` | 132 | 219 | **+87（+66%）** |
| 裸 `cudaMalloc` | 167 | 255 | +88 |
| `DeviceManager::setDevice` 调用者 | 0 | 0 | — |

**加了模块，裸调用反而涨了 66%。** 这是本次评估最重要的一条经验：
没有强制手段的模块化 = 又一个可以被绕过的装饰。

后果已经发生过至少 4 次（`git log`）：

| commit | 症状 |
|---|---|
| `0414afd` | analysis 资源释放与设备归属错误 |
| `628e131` | 参数/coupling/peer copy 的主设备不一致；`cudaMemGetInfo` 失败被当成 0 空闲 → 假 OOM |
| `bc22b52` | 逐 GPU 权重归约把 current device 留在最后一张卡 → 副卡 kernel 直接读主卡指针（compute-sanitizer 定位） |
| `5f6a41e` | 干涉输出跨后端不一致 |

`bc22b52` 的修复注释就是问题定义：
> "The per-GPU weight reduction leaves the last GPU current."

**为什么靠测试也抓不到**：CPU 后端 `set_device` 是
`return index == 0;`（`Backend.cpp:186`），越界只是静默 `false`；
CI 是无卡机器；多卡一致性回归在 `scripts/cluster/multigpu_parity.py`，
是**需要双卡机手动跑的集群脚本**，不在 pytest 门禁里
（`tests/conftest.py` 只有感知多卡的 helper，没有"双卡逐值比对"用例）。
**设备 bug 在日常测试里基本隐形。**

### 2.3 后端抽象边界：`ctpwa::alloc` 只被用了 1 次

`include/Backend.h` 提供了 `ctpwa::alloc/copy/launch` 可移植层。实际使用：

| API | 调用 |
|---|---|
| `ctpwa::alloc` | **1** |
| `ctpwa::copy` | **0** |
| `ctpwa::launch` | **0** |
| 裸 `cudaMalloc` / `cudaFree` / `cudaMemcpy` | **255 / 288 / 360** |

真正让 CPU 后端工作的是 `include/cpu_shims/cuda_runtime.h` 的整体替换，不是这层 API。
**一个无人使用的抽象层比没有更糟**：它让人以为边界存在。

### 2.4 状态所有权：一份东西 2~3 个所有者

**（a）"主设备"有 3 份**

| 所有者 | 位置 |
|---|---|
| `analysis::primary_dev_` | `Analysis.cu:6479` |
| `Parameters::primary_dev_` | `Parameters.cuh:128`（由 `Analysis.cu:6506` 注入） |
| `AmpCalc`：每次现查 `cudaGetDevice(&primary_dev)` | `AmpGen.cu:3940`、`AmpGen.cu:4086` |

第三份尤其危险：它把"主设备"定义为**调用那一刻的 ambient 设备**。
所以 `AmpCalc` 的正确性依赖一个不成文约定："调用者进来前已经切回主卡"。
`bc22b52` 就是这个约定被打破。

**（b）设备指针有两套簿记 → double free**

`AmpGen.cu:1520-1526` 注释记录的 bug：同一批指针同时登记在 `d_mom_sigma_` 与
`d_momenta_structs_/d_momenta_arrays_`，只 free 一处 → 析构二次 free
（CPU 后端直接 `free(): double free detected`）。修法是"按指针匹配把它从另一个
登记表里摘掉"——**用更复杂的簿记去补两套簿记**。这是模块化失败的原型：
不是函数没拆开，是一个指针有两个所有者。

**（c）参数管理有两套**

`var_equal` 在 `AmpCalc`（`var_equal_ghosts_`）和 `analysis`（`Analysis.cu:6816-6819`
按实际槽名重建 theta 名段）各处理一遍；conjugate/同名跨链共享又是一套
（`conjugate_broadcast_`）。

---

## 3. 模块化能修什么、不能修什么

别指望一次重构把所有痛点带走。按失败类分开看：

| 失败类 | 实例 | 模块化/结构改造能修吗 | 真正需要的东西 |
|---|---|---|---|
| 环境状态泄漏 | `bc22b52` | ✅ 能根治 | **显式设备参数 + RAII guard + 断言**（不是"封装一个函数"） |
| 同一状态多份 | 3 份 primary device | ✅ 能根治 | 单一所有者 |
| 同一资源多所有者 | `d_mom_sigma_` vs `d_momenta_structs_` | ✅ 能根治 | owning RAII buffer，禁裸指针 |
| 改动爆炸半径 | 改 `Config.cu` 重编 12.8k 行 | ✅ 部分 | 真 TU |
| 编排不可测 | `analysis` 6300 行 | ✅ 但要**先有 owner** | 抽接口（顺序重要，见 §5） |
| 数值 bug | nested free 共振态 `q0` 漏项 | ❌ 不能 | FD 对照/回归测试，与模块化无关 |

结论：**先把"隐式状态"变成"显式参数"，再谈拆类。** 顺序反了，就会用 4 个模块去访问
同一份状态，bug 率反而上升。

---

## 4. 目标形态

> 本节给**分层与不变量**；**目录细节、文件归属、分类判据以 §7.2 / §7.3 为准**。

```
src/
  config/       Config.cu + Info.cu(DecayInfo)                    纯 host
      │                              ↓ 不可变描述
  model/        ResModel + Resonance + CustomExpr + SymbolicDiff  今天已基本干净
      │
  device/       DeviceManager + DeviceGuard + DevBuffer   ← 全项目唯一 cuda* 出口
      │
  amplitude/    SlampTable(原 AmpCasDecay 的设备半边) + AmpCalc
      │
  parameters/   coupling + theta + 约束（唯一 name→slot）
      │
  fit/          ComputeNLL + ComputeHessian + ComputeResults(kernel) 不碰 IO/ROOT
      │
  io/           ROOT/TTree/.dat 读取 + PhaseSpace(MC 产生)        纯 host
      │
  output/       result_root.cpp / matrix.cpp / figure.cpp        ← 只吃 FitResult
      │
  app/          analysis 编排 + main.cpp(pybind)
```

两个"看名字会归错"的（§7.2 已定，别按本节旧版猜）：

| 东西 | 看名字像 | 归属 | 依据 |
|---|---|---|---|
| `ComputeResults.cu` | output（结果） | **`fit/`** | 是 device kernel（`computeResults` / `computeTotalIntensity`），不是 writer |
| `Info.cu` | output（信息） | **`config/`** | 是 `DecayInfo`，`ChainInfo` 由它构建、被 `calculateAmplitudes` 消费 |

每个模块必须能写出一条**违反立刻失败**的不变量，否则等于没拆：

| 模块 | 不变量 | 强制手段 |
|---|---|---|
| device/ | `cudaSetDevice`/`cudaMalloc`/`cudaFree` 只出现在本模块 | `ForbiddenApi.h` 把裸名字 `#define` 成编译错误 + CI grep |
| device/ | 任何 kernel launch 的指针 `dev == 当前设备` | `CTPWA_STRICT_DEVICE=1` 时 launch 前 `cudaPointerGetAttributes` 断言 |
| device/ | CPU 后端越界 device index 立即 abort | 修掉 `Backend.cpp:186` 的静默 `false` |
| amplitude/ | 一个设备指针只有一处 free 路径 | `DevBuffer` 唯一所有权，禁拷贝 |
| parameters/ | 全项目只有一处 name→slot 映射 | 两个参数管理者合并 |
| io/ | 不出现任何 cuda 符号 | 纯 host，可脱离 GPU 单测（`PhaseSpace.cu` 实测无 kernel/无 `cudaMalloc`） |
| output/ | 不出现 `analysis`，只吃 `FitResult` | 目录 grep 门禁（§7.3） |

---

## 5. 路线（按 收益/风险 排序，不是按目录顺序）

### P0 设备纪律 —— 1~2 天，风险低，**如果只做一件事就做这个**

1. `DeviceGuard`（RAII：构造保存 current，析构还原）+ `dev::set(i)`。
   75 处"还原主卡"的调用直接消失。
2. 219 处裸 `cudaSetDevice` 机械替换；`AmpCalc` 里两处 `cudaGetDevice` 改成
   **显式形参**（`primary_dev` 已经在 `reComputeAmps` 里传了，把
   `computeResonanceGradient`/`computeUnifiedHessian` 补齐）。
3. `ForbiddenApi.h`：非 device 模块里 `#define cudaSetDevice CTPWA_USE_DeviceGuard`，
   裸调用直接编译失败（比 grep 门禁硬，且错误信息自带修法）。
4. CPU 后端 `set_device` 严格化：`index != 0` → 明确报错，不再静默。

**验收（缺一不可）**：
- `bash scripts/run_tests.sh` 全绿；
- 双卡机器上跑通现有 multigpu 用例；
- **把 `bc22b52` 那个 bug 注回去，必须被断言抓住**——这条是防线的有效性证明，
  没有它，P0 只是"改得更整齐"。

### P1 真 TU —— 2~3 天，风险低

7 个被包含的 `.cu` 各自成为 TU，去掉 `main.cu` 的 `#include`；
`analysis` 抽一个 `Analysis.cuh`（pimpl），`main.cu` 只依赖 1 个头。

- 静态扫描已确认跨文件 static 依赖 0 处，预期只会暴露少量遗漏声明，不会挖出架构地雷。
- 验收：`touch src/Config.cu` 后 ninja **只编 1 个文件**；`nm` 无新增全局符号泄漏。

> 收益是**编译防火墙**（编译器第一次会告诉你谁在越界用谁），不是编译速度。
> 12.8k → 分片后总编译量下降有限，别把它当性能项目卖。

### P2 DevBuffer 所有权 —— ~1 周，风险中

`DevBuffer<T>`（owning，带 dev 标签）替换关键路径的裸 `cudaMalloc/Free`，
先从 `AmpCasDecay` 下手（顺手删掉 §2.4(b) 那种"按指针匹配摘簿记"）。

- 分两步：先去重登记表（不改类型）→ 再换类型。
- 验收：`releaseStaticCasData` 不再需要 `std::set<DeviceMomenta*> drop` 这类补偿逻辑。

### P3 config/DecayInfo 与 SL 表解耦 —— ~1 周，风险中

`AmpCasDecay` 拆成 `DecayChainDesc`（纯 host 值类型）+ `SlampTable`（设备 + 生命周期）。

- 验收：`DecayChainDesc` 能在没有 GPU 的单测里构造与比较；`releaseStaticCasData`
  只出现在 `SlampTable`。

### P4 god class 拆分 —— 1~2 周，风险中高

顺序必须是 **io → output → fit 编排 → tensor API**（先拆依赖最少的）。
`analysis` 目标 < 800 行，每个抽出的模块带独立单测。

- 验收：`analysis.cpp` 行数 + 新模块各自可单测；数值回归对比
  `tests/golden/*`。

---

## 6. 不做什么

| 不做 | 原因 |
|---|---|
| 一次性大重构 / 重写 | 261 个用例是唯一护栏，diff 过大时护栏失效 |
| 为拆而拆的文件搬移 | 纯搬移 diff 巨大、收益 0；P1 例外（它是编译边界，不是逻辑重排） |
| 引入依赖注入/多层 wrapper 框架 | god class 的病因是所有权，不是缺抽象层 |
| 保留无人使用的抽象层 | `ctpwa::alloc` 用 1 次、`DeviceManager::setDevice` 用 0 次；要么强制，要么删 |
| 在无卡机器上验收设备改造 | CPU 后端把 device 概念抹平，会给出假的"通过" |
| 用重构去追数值 bug | 梯度漏项靠 FD 对照，与结构无关 |

---

## 7. 目录布局与测试分区（为"拓展"服务）

### 7.1 加一个新功能，今天要改几处

判断"模块化是否让拓展变方便"，只看一件事：**加功能时要不要碰已有文件、要碰几个**。

| 要加的东西 | 今天要动的地方 | 评价 |
|---|---|---|
| 新线形（BWR/GS/…） | `Resonance.cuh` 枚举 + `ResModel.cu` `buildModelAST` + `modelDeriv` + `Resonance.cu` 注册表 | 🟡 3~4 处，但**路径清晰、局部**（`ModelRegistry::registerModel` 是数据驱动的）；P1 之后还能顺带少重编 |
| 新观测量 / 画图 | `Config.cuh` `ObsFunc` 枚举 + `Config.cu` 名字解析 + `Figure.cu:570` switch | 🟡 典型"枚举+解析+switch"三处同步，漏一处就是静默错 |
| 新输出产物（JSON/CSV/新 TTree） | 只能在 `analysis::writeResult`（**1555 行、触及 24 个成员**）里加分支；ROOT 符号 92 处写在 god class 里 | 🔴 最难 |
| 新约束类型 | `Config.cu` 解析 + `Parameters.cu` + `AmpCalc::slots_` + `Analysis.cu:6816` 名字重建 | 🔴 最危险，四处必须同步 |
| 新设备 / 多机 | 219 处裸 `cudaSetDevice` | 🔴 不可行 |

**规律**：拓展难度 ≈ 该功能是否已有模块边界，与总行数无关。
所以 P0–P4 真正买到的是"新增输出/新增约束"从 🔴 变 🟢，而不是"新增线形"变快多少。

### 7.2 目标目录

> **状态（2026-11）：已按下面这张表实施。**`src/<分区>/*.cu` 与 `include/<分区>/*.h`
> 都已就位，`setup.py` 为每个分区加了一条 `-I`，include 写法不变。文件归属与两条
> "容易分错"的判据见 `src/README.md`；那里同时列出了**今天还没满足**的三条硬判据
> （output/ 里仍有 `analysis`、device/ 只统一了 set-device、分区之间没有反向引用检查），
> 不要把"目录建好了"当成"分层完成了"。

```
src/
  config/       Config.cu + Info.cu(DecayInfo)        纯 host
  model/        ResModel + Resonance + CustomExpr + SymbolicDiff
  device/       DeviceManager + DeviceGuard + DevBuffer   唯一 cuda* 出口
  amplitude/    SlampTable(原 AmpCasDecay 的设备半边) + AmpCalc
  parameters/   coupling + theta + 约束（唯一 name→slot）
  fit/          ComputeNLL + ComputeHessian + ComputeResults(是 kernel，不是 writer)
  io/           ROOT/TTree/.dat 读取 + PhaseSpace(MC 产生)
  output/       result_root.cpp / matrix.cpp / figure.cpp   ← 只吃 FitResult
  app/          analysis 编排 + main.cpp(pybind)
```

两个容易分类错的地方（今天已经存在）：

- `ComputeResults.cu` 名字像 output，实际是 **device kernel**（`computeResults` / `computeTotalIntensity`）→ 归 `fit/`。
- `Info.cu` 是 **`DecayInfo`（config 层）**，`ChainInfo` 的注释就写着"Built by DecayInfo, consumed by Analysis::calculateAmplitudes"→ 归 `config/`，**放进 output 是分类错误**。

### 7.3 output 拆分的硬判据

今天"输出"不是一个模块，是几个巨型函数挂在 god class 上（行数为实测）：

| 函数 | 行数 | 直读成员 |
|---|---|---|
| `analysis::writeResult` | **1553**（1732–3285） | **24 个**，含 `d_all_amplitudes_`/`events_offsets_`/`params_` 等 |
| `analysis::writeInterfResult` | 283（3299–3582） | 同一批成员 |
| `analysis::writeMC` | —（同文件 6393 起） | 同一批成员 |

所以拆 3 个文件夹的顺序必须是：

1. 先定义 `FitResult`（纯 host POD：`n_amplitudes / n_gpus / 份额 / 干涉矩阵 / 协方差 / θ 表 / 样本量 / legends`）；
2. `analysis` 只负责**填**它；
3. writers 只吃 `FitResult const&`：`output/result_root.cpp`、`output/matrix.cpp`、`output/figure.cpp`。

**硬判据：`output/` 里不允许出现 `analysis` 这个词。** 出现就说明切漏了，只是把 1553 行分成 3 段，
共享的仍是那 24 个成员 → 互相关联的错误一点没少，反而多两个跨文件依赖。

按"产物形态"切而不是按"数量"切，好处是三个 writer 之间**零引用**：加 JSON = 新增一个
`output/summary.cpp`，不碰另外两个；这就是"拓展方便"的实质。

### 7.4 测试分区

现状：41 个 test 文件 / 257 个用例，**已经按失败类分好了**（`test_numerical` 17 个梯度/Hessian、
`test_structure` 21 个 config+设备探测、`test_constraints` 11、`test_matrix_errors` 7、
`test_cpu_*`、`test_nested_resonances`、`test_write_result_robustness`），
这个组织其实比源码好。**别推倒重来，加目录 + 加映射表即可。**

按"能否脱离 GPU/config"做硬划分（这个判据可机械检查）：

```
tests/
  unit/      纯 host：config 解析、DecayInfo 链串、DSL 编译、AST 求导   （不建 analysis）
  contract/  跨模块不变量，永不 skip：设备契约、双后端 parity、golden 数值、内存预算
  physics/   JPC/阈值/identical/barrier/constraints 语义
  pipeline/  端到端：NLL/梯度/Hessian/份额/writeResult
```

规则：

| 规则 | 理由 |
|---|---|
| 改 X → 至少跑 X 所在区 + `contract/` | contract 是唯一不能被"按需跳过"的层 |
| 提交前跑全量（保留 `scripts/run_tests.sh` 门禁） | 数值路径贯穿所有区，只跑一格会给假安全感 |
| 用 marker 提供 `--fast` 子集（`-m "not slow and not multigpu"`）给改完即测的 30 秒反馈 | 反馈速度 ≠ 门禁 |

为什么 `contract/` 必须独立存在：历史上 5 个 multi-GPU bug
（`0414afd` `628e131` `bc22b52` `5f6a41e` `02c36ac`）**没有一个**会被 `output/test_output.py` 抓到，
它们全是跨模块不变量失守。分区若按"源码目录 1:1 镜像测试目录"，这一层就会消失。

### 7.5 反模式

- **目录 ≠ 隔离。** 只有当跨目录引用会被编译器/断言拦住时，目录才减少错误；否则 3 个文件 = 3 个入口。
- **测试分区 ≠ 可以只跑一格。** 分区的收益是可发现性 + 契约归属，不是少跑测试。
- **为凑数量而切。** "3 个文件夹"本身没有意义，有意义的是"3 个互不引用的 writer + 1 个 POD"。

---

## 8. 证据索引

| 结论 | 位置 |
|---|---|
| 7 个 `.cu` 被文本包含成一 TU | `src/main.cu:11,22,25,28,31,36,40`；`setup.py` `SOURCES_CORE` |
| `DeviceManager::setDevice` 0 调用者 | `include/DeviceManager.cuh:74` 定义，全仓 0 处 `setDevice(` 使用 |
| 219 处裸 `cudaSetDevice` | `Analysis.cu` 162 / `AmpGen.cu` 45 / `DeviceManager.cu` 7 / `Parameters.cu` 4 / `Backend.cpp` 1 |
| 75 处仅用于还原主卡 | `grep -c 'cudaSetDevice(primary_dev_)'` |
| `ctpwa::alloc` 1 次 vs `cudaMalloc` 255 次 | `include/Backend.h` vs `src/` |
| 3 份"主设备" | `Analysis.cu:6479`、`Parameters.cuh:128`、`AmpGen.cu:3940/4086` |
| 双簿记 → double free | `AmpGen.cu:1520-1526` |
| `var_equal` 两套 | `AmpGen.cu:2647,3029-3064`、`Analysis.cu:6816-6819` |
| CPU 后端 set_device 静默 | `src/Backend.cpp:186` |
| 8 个轴里只有共振态/符号微分不含编排耦合 | `ResModel.cu` / `Resonance.cu` / `CustomExpr.cu` 内无 fit、IO、参数编排代码（只被上层调用） |
| 多卡同类 bug 复发 4 次 | `0414afd` `628e131` `bc22b52` `5f6a41e` |
