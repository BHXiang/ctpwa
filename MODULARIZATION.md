# CTPWA 模块化设计与迁移方案

> 状态：设计草案 v0.1
> 日期：2026-11-26
> 目标读者：CTPWA 后续设计、开发与评审人员
> 基本原则：**不大重写、不改变公开物理语义，先固化契约，再逐步迁移实现。**

## 1. 文档目的

CTPWA 已覆盖衰变链展开、振幅生成、NLL/梯度/Hessian、CPU/CUDA 后端、多 GPU、
干涉矩阵、ROOT 输出和 Python 拟合接口。随着功能增加，`analysis`、`AmpCalc`、
`AmpCasDecay` 和 `writeResult` 同时承担了过多职责，使设备状态、显存估算、资源
生命周期和数值输出之间产生隐式耦合。

本文用于指导后续模块化设计与开发，重点回答：

1. 模块应如何划分，依赖方向是什么；
2. 哪些接口和不变量必须先固化；
3. 如何在不破坏现有用户接口的前提下渐进迁移；
4. 如何防止设备错配、假 OOM、重复显存估算和 CUDA 级联错误再次出现；
5. 每个阶段用什么测试证明迁移没有改变物理结果。

本文不是一次大规模重写计划，也不要求为了“文件更小”而拆碎性能关键 kernel。

## 2. 当前问题与设计经验

### 2.1 主要矛盾不是代码行数，而是隐式契约过多

当前 [Analysis.cu](src/Analysis.cu) 同时处理：

- 配置、链信息与参数初始化；
- 多 GPU 事件划分和 offset；
- 振幅、SL、动量和权重等常驻设备资源；
- NLL/autograd/Hessian；
- phsp/truth 积分与拟合分数；
- ROOT 文件、直方图和误差矩阵输出；
- 设备切换、错误处理和资源释放。

[AmpGen.cu](src/AmpGen.cu) 同时处理 SL 组合、设备表、全同粒子拓扑、共振态
重算、梯度和 Hessian。任何一个局部修改都可能影响多个隐含调用方。

### 2.2 本次暴露的问题及对应设计教训

| 现象 | 根因类型 | 必须固化的设计规则 |
|---|---|---|
| `primary=1`，参数却在 `cuda:0` | 当前设备是隐式全局状态 | 公开入口显式选择设备，退出统一恢复 |
| 副卡 kernel 读取主卡 `d_phsp_matrix_` | buffer 所属设备未形成统一约束 | buffer 记录设备与所有者，跨卡访问必须显式 |
| `cudaMemGetInfo` 失败后被报成“仅剩 0 MiB” | 错误值被当成正常查询结果 | 所有 CUDA 查询先检查返回码，禁止二次猜测 |
| auto/hybrid 的 A 被按 double 估算 | 预检使用编译期大小，分配使用运行期大小 | 精度与字节数只能来自同一布局对象 |
| SL 表按总振幅数重复估算 | 预检与真实链/共振组合结构脱节 | 预检和分配共享同一份 MemoryLayout |
| 多 GPU 矩阵参考只使用 GPU0 | 测试辅助接口的样本范围不明确 | API 明确返回单卡还是全样本，测试逐卡汇总 |
| 首次 CUDA 错误后出现大量无关失败 | 同进程继续使用已损坏上下文 | 首个 CUDA 致命错误后停止，本组诊断独立进程运行 |
| 作业运行了旧后端 | 构建路径、加载路径和运行路径未核验 | 作业打印模块路径、二进制和源码版本信息 |

结论：模块化首先要减少隐式知识，而不是仅减少单个文件的行数。

## 3. 设计目标与非目标

### 3.1 目标

1. 保持 `ctpwa.analysis(...)` 等公开 Python API 和 YAML 语义兼容。
2. 同一项物理、布局、精度和设备信息只在一个地方定义。
3. 设备状态、资源所有权和异常恢复由统一基础设施保证。
4. 模型、显存布局、CPU/CUDA kernel 和 ROOT I/O 能独立测试。
5. 单卡与多卡、CPU 与 CUDA 的数值一致性可自动验证。
6. 允许逐步替换内部实现，不要求一次性完成全部迁移。

### 3.2 非目标

- 不为了模块化改变拟合结果、振幅定义或默认精度策略。
- 不把所有物理公式拆成大量小文件。
- 不在第一阶段重写出新的拟合框架或新的配置格式。
- 不引入新的全局设备管理器来替代旧的隐式全局状态。
- 不用放宽测试容差代替根因分析。

## 4. 总体架构

### 4.1 推荐依赖方向

```text
Config/YAML
    ↓
DecayModel（只读物理模型）
    ↓
MemoryLayout / EventLayout（唯一布局真相）
    ↓
AmplitudeStorage + Backend Workspace
    ↓
Fit / Integrals / ResultBuilder
    ↓
RootResultWriter / Python Bindings

DeviceContext、DeviceGuard、Error Handling 为横向基础设施，
只能被上述模块调用，不反向依赖物理模型或 ROOT 输出。
```

依赖规则：

- kernel 不读取 Python、不直接写 ROOT；
- 显存预检不重新解释衰变链；
- ROOT 输出不改变拟合参数和设备状态；
- `DecayModel` 不依赖拟合结果；
- backend 不依赖具体共振态名称或用户配置文本；
- Python 门面不实现底层显存和设备算法。

### 4.2 推荐目录结构

```text
include/ctpwa/
├── model/          # ConfigParser、DecayInfo、ChainInfo、Resonance、Constraints
├── layout/         # EventLayout、AmplitudeLayout、MemoryPlanner
├── device/         # DeviceContext、DeviceGuard、DeviceBuffer、CUDA 错误
├── amplitude/      # Amp2BD、AmpCasDecay、AmpCalc、振幅存储
├── fit/            # Parameters、NLLFunction、Hessian、FitWorkspace
├── integrals/      # phsp/truth 积分、效率、干涉矩阵
├── io/             # dat/ROOT 输入、RootResultWriter
├── backend/
│   ├── cuda/       # CUDA kernel 与 CUDA workspace
│   └── cpu/        # CPU 实现与 OpenMP workspace
└── bindings/       # pybind11 接口

src/                 # 与 include/ctpwa 对应的实现
python/              # 高级 Python 拟合与诊断接口
tests/
├── model/
├── layout/
├── backend/
├── multigpu/
├── integration/
└── regression/
```

迁移期间可保留现有文件位置，先建立逻辑边界，再逐步移动物理文件。

## 5. 核心模块与职责

| 模块 | 职责 | 不应负责 |
|---|---|---|
| `DecayModel` | 粒子、链、JP、SL、共振组合、约束的只读展开 | 设备分配、拟合、ROOT |
| `EventLayout` | 每卡 data/phsp/bkg 数量、事件和振幅 offset | CUDA kernel、物理线形 |
| `MemoryPlanner` | 根据模型、精度和后端生成显存账本 | 直接执行 cudaMalloc |
| `DeviceContext` | 主卡、流、设备集合和错误上下文 | 解释共振态或拟合公式 |
| `AmplitudeStorage` | A、SL、动量、置换表及其生命周期 | NLL、ROOT 输出 |
| `FitWorkspace` | NLL/Hessian 临时缓冲与 autograd 状态 | 长期保存用户结果文件 |
| `MatrixIntegrator` | rec/gen 积分、效率、拟合分数 | 管理 ROOT 文件句柄 |
| `ResultBuilder` | 组合直方图、矩阵、权重和元数据 | 直接操作全局设备状态 |
| `RootResultWriter` | 将 ResultBuilder 的结果写入 ROOT | 重算振幅、改变 theta |
| `AnalysisContext` | 组合上述模块，提供公开 C++ 入口 | 重复实现各模块内部算法 |
| `bindings` | 保持 Python API 与参数转换 | 显存估算、设备循环 |

`analysis` 最终应成为 `AnalysisContext` 的薄封装，而不是所有资源的直接所有者。

## 6. 必须优先固化的核心契约

### 6.1 DecayModel：只读模型契约

构造完成后，以下信息只读：

- 粒子和有效极化映射；
- chain×JP 组合；
- 每个 JP 组合的 SL 组合数；
- 共振态组合及其与 JP 组合的共享关系；
- 振幅名、参数名和约束映射；
- 全同粒子拓扑及其 `nSigma`。

本次加入的 `slcombs_per_jp` 应成为该模型的一部分，而不是显存模块临时推导的补丁。

### 6.2 MemoryLayout：唯一显存真相

建议接口：

```cpp
struct MemoryLayout {
    int n_amplitudes = 0;
    int n_polar = 1;
    std::size_t amplitude_element_bytes = 0;
    std::size_t derivative_element_bytes = 0;

    // 每个 GPU 的常驻事件数；phsp 流式时不包含 phsp。
    std::vector<int> resident_events;

    // 每个独立 chain×JP 组合的 SL 数；共振态组合不重复展开。
    std::vector<int> slcombs_per_jp;

    // 与 slcombs_per_jp 对应的置换拓扑数，缺省为 1。
    std::vector<int> nsigma_per_jp;

    std::size_t amplitudeBytes(int gpu) const;
    std::size_t slTableBytes(int gpu) const;
    std::size_t momentaBytes(int gpu) const;
    std::size_t residentBytes(int gpu) const;
    std::size_t workspaceBytes(int gpu) const;
};
```

不变量：

1. auto/hybrid 的 A 使用运行期 `float2` 大小，不能使用编译期 `ctComplex` 猜测。
2. 常驻 A 按该卡 `phsp + data + bkg` 计算；流式 phsp 不计入常驻 A。
3. SL 表按独立 cas/JP 计算，同一 JP 下多个共振态组合不得重复收费。
4. SL 表应包含相应 `nSigma`，不能默认所有模型都是 1。
5. 常驻峰值和临时 workspace 峰值分列，不能把不同生命周期的表简单相加后称为常驻。
6. 预检、实际分配、offset、日志和测试均调用同一布局计算。
7. 字节计算使用 `std::size_t`；显示时区分 B、KiB/MiB 与 KB/MB。

推荐让实际分配函数接收 `MemoryLayout`，并记录每项分配的字节数，以便在测试中断言：

```text
capacity.estimate(buffer) == allocation.recordedBytes(buffer)
```

### 6.3 DeviceContext 与 DeviceGuard

每个 `AnalysisContext` 持有自己的 `DeviceContext`，至少包含：

- `primary_device`；
- 可用设备集合；
- 每卡 stream；
- 当前后端和精度；
- 设备错误上下文。

所有公开 C++ 入口采用 RAII：

```cpp
DeviceGuard guard(context.primary_device());
// 函数实现；正常返回和异常退出都恢复进入前的设备。
```

设备不变量：

1. 输入张量不在主卡时，入口立即给出明确错误。
2. kernel 启动前，当前设备必须与指针所属设备一致。
3. 跨卡复制只能使用明确的 peer copy 接口，不能依赖裸指针碰巧可访问。
4. 公开函数返回后，调用者看到的 current device 与进入前一致。
5. 多卡循环结束不是恢复设备的唯一位置，异常路径也必须恢复。
6. 不在析构阶段盲目使用可能已经销毁的 CUDA 上下文。

### 6.4 DeviceBuffer：资源所有权

建议用轻量描述包装裸指针：

```cpp
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    int device = -1;
    std::size_t bytes = 0;
    BufferOwner owner = BufferOwner::Analysis;
};
```

每块 buffer 必须能回答：在哪张卡、多少字节、谁分配、谁释放、哪个布局项对应。

跨 `analysis` 对象的静态缓存应逐步取消；确需共享的缓存必须按设备、布局和资源版本
建立明确 key，并定义并发与销毁语义。

### 6.5 错误处理契约

分三层处理：

1. **kernel 封装层**：检查 launch error 和必要的同步错误。
2. **公开 API 层**：对输入、设备和布局做前置检查，在合适边界捕获异步错误。
3. **诊断层**：报告第一条真实错误及调用上下文，不把查询失败转换成假 OOM。

规定：

- `cudaMemGetInfo`、`cudaMalloc`、`cudaMemcpy` 等调用必须检查返回码。
- 致命 CUDA 错误后，同一进程后续测试结果不再作为诊断依据。
- sanitizer 目标独立进程运行，不包裹整套 pytest。
- 错误信息包含模块、设备、操作和布局项，避免只打印“显存不足”。

## 7. 后端接口与公开 API

### 7.1 内部后端接口

CPU/CUDA 后端共享行为接口，而不是共享设备假设：

```text
allocate(layout)
computeNLL(params, workspace)
computeGradient(params, workspace)
computeHessian(params, workspace)
computeIntegrals(sample, layout, workspace)
release(workspace)
```

CPU 后端的“device 0”只是执行上下文标识，不复用 CUDA 裸指针语义。

### 7.2 外部兼容性

默认保持：

- `ctpwa.analysis(config, fit_mode)`；
- `getNLL/getHessian/writeResult/reCalcAmp`；
- YAML 字段、参数顺序、振幅名和 ROOT 对象名；
- CPU/CUDA 后端选择方式。

需要删除或改变的接口必须经历：

1. 标记 deprecated 并说明替代方案；
2. 至少一个过渡版本；
3. 更新示例和迁移说明；
4. 在 CHANGELOG 中明确记录。

## 8. 分阶段迁移计划

### 阶段 0：建立行为基线

交付物：

- 固定现有公开 API、参数顺序和 ROOT 对象清单；
- 保留当前 pytest、1 卡 vs N 卡一致性、精度档位和派生量测试；
- 作业输出 `ctpwa.__file__`、后端路径、构建时间和 git revision；
- 构建与 GPU 测试分离，GPU 作业不编译。

验收：能够明确区分“旧二进制问题”和“当前源码问题”。

### 阶段 1：设备基础设施

交付物：

- `DeviceContext`、`DeviceGuard`、`DeviceBuffer`；
- 公开 API 的入口/出口设备检查；
- CUDA 查询与分配统一错误处理；
- 多卡 current-device 不变量测试。

验收：所有公开调用结束后 current device 不变；设备错配在 kernel 前被拦截。

### 阶段 2：MemoryLayout 与预检统一

交付物：

- `EventLayout`、`MemoryLayout`、`MemoryPlanner`；
- 使用实际 JP/SL 共享关系和运行期精度；
- 预检与分配账本对照测试；
- 多链、多共振态、大 phsp、双 V100 的回归场景。

验收：不再用 `n_gls_` 代替独立 SL 表计数；多共振态配置不因重复 SL 估算被误拒。

### 阶段 3：只读 DecayModel

交付物：

- 将配置和链展开结果整理为不可变模型；
- `AnalysisContext` 只持有模型引用，不复制修改；
- 模型单元测试覆盖 JP、SL、共振组合和全同拓扑。

验收：同一模型可重复生成相同布局，拟合调用不改变模型内容。

### 阶段 4：积分与 ROOT 输出拆分

交付物：

- `MatrixIntegrator`、`ResultBuilder`、`RootResultWriter`；
- `writeResult` 变成编排入口；
- theta 只通过显式参数或受控上下文传递；
- 测试参考端遍历所有 GPU。

验收：输出前后拟合状态与设备状态不变；单卡/多卡矩阵语义一致。

### 阶段 5：拟合与振幅资源拆分

交付物：

- `AmplitudeStorage`、`FitWorkspace`；
- NLL forward/backward 与长期资源分离；
- Hessian vv/vθ/θθ 的输入输出和工作区独立；
- 清理无明确所有者的静态设备缓存。

验收：多个 analysis 对象在同一进程交替使用不会互相修改资源或设备状态。

### 阶段 6：后端与构建目标整理

交付物：

- `ctpwa_model/core/io/cpu/cuda` 等逻辑目标；
- `main.cu` 仅保留绑定和必要的编译编排；
- 控制 Torch 头文件传播，避免模块化后编译时间失控；
- 保留并验证 CUDA `-rdc/nvlink` 构建链。

验收：CPU、CUDA、wheel 和开发构建均通过；模块依赖不存在反向引用。

## 9. 测试与发布门禁

### 9.1 单元测试

- DecayModel：JP、SL、共振组合、约束、nSigma；
- MemoryLayout：每个 buffer 的字节数、offset、共享关系；
- DeviceGuard：正常返回和异常抛出的设备恢复；
- 数值 kernel：NLL、梯度、Hessian、干涉矩阵的小输入参考值。

### 9.2 后端与多卡测试

- CPU vs CUDA；
- 1 卡 vs N 卡 NLL、梯度和派生量；
- auto/hybrid/double 三档；
- `n_batch < n_gpus`、空分片、非均匀权重；
- 参考端必须明确使用单卡还是全样本，禁止默认只取 GPU0。

### 9.3 回归测试

必须保留以下场景：

1. 固定 theta 的常驻 phsp 二次型前恢复主卡；
2. auto/hybrid 的 A 按 float2 估算；
3. 多共振态共享 SL 表，不按总振幅数重复收费；
4. `cudaMemGetInfo` 失败时报告真实 CUDA 错误；
5. 多卡矩阵参考汇总所有设备；
6. 首个 CUDA 致命错误后停止该诊断进程。

### 9.4 Sanitizer 与故障注入

- `compute-sanitizer` 只运行最小复现用例；
- 默认直接使用已编译后端，不在 GPU 作业中强制编译；
- 需要源码行号时在登录节点单独构建 `-lineinfo` 版本；
- 可注入内存不足、非法设备和分配失败，验证错误路径与资源恢复。

### 9.5 发布门槛

不得仅依据“pytest 大部分通过”发布。至少要求：

- 全量 pytest 通过或每项例外均有明确 issue；
- 多卡 NLL/梯度及派生量门禁通过；
- 无未解释的 CUDA 错误；
- 显存账本与真实分配一致；
- 构建产物路径与测试加载路径一致；
- 物理结果和公开输出格式无未声明变化。

## 10. 后续功能开发检查单

### 新增设备 buffer 时

- [ ] 是否已加入 MemoryLayout？
- [ ] 是否记录所属设备、字节数和所有者？
- [ ] 是否区分常驻与临时峰值？
- [ ] 是否同时更新预检、分配、日志和测试？

### 新增公开 C++ 入口时

- [ ] 是否使用 DeviceGuard？
- [ ] 是否检查输入张量设备、dtype 和尺寸？
- [ ] 正常与异常路径是否都恢复状态？
- [ ] 是否定义多 analysis 对象交替调用行为？

### 新增 kernel 时

- [ ] 是否检查 launch error？
- [ ] 是否明确指针所属设备和布局？
- [ ] 是否有小输入参考值及 CPU/CUDA 对照？
- [ ] 是否有单卡最小 sanitizer 复现方式？

### 修改积分或输出时

- [ ] 是否保持 theta、耦合和振幅版本一致？
- [ ] 是否明确 rec/gen、单卡/全样本和归一化口径？
- [ ] 是否保持 ROOT 对象名与参数顺序兼容？
- [ ] 输出完成后是否恢复设备和内部 offset？

## 11. 建议的 PR 拆分

1. **PR-1：设备不变量与错误处理**——DeviceGuard、CUDA 查询错误、回归测试。
2. **PR-2：EventLayout/MemoryLayout**——统一 A/SL/事件 offset 和预检。
3. **PR-3：只读 DecayModel**——固化 JP/SL/共振组合信息。
4. **PR-4：MatrixIntegrator**——抽离 rec/gen 积分，不多改物理实现。
5. **PR-5：ResultBuilder/RootResultWriter**——拆出 ROOT I/O。
6. **PR-6：AmplitudeStorage/FitWorkspace**——明确资源生命周期。
7. **PR-7：后端与构建目标**——整理 CPU/CUDA 和 pybind 边界。

每个 PR 都应保持可编译、可测试、可回退，避免长期分支积累无法归因的大差异。

## 12. 大型项目的可借鉴经验

- **PyTorch**：以 DeviceGuard、dispatcher 和后端注册隔离设备细节；核心、autograd、
  CPU/CUDA kernel 和 Python 绑定分层。
- **ROOT**：Core、RIO、Tree、Hist、MathCore 分库；ROOT I/O 不侵入所有数值模块。
- **LLVM**：以统一中间表示连接前端、优化和目标后端；CTPWA 可用
  `DecayModel + MemoryLayout` 作为自己的共同中间表示。
- **Kokkos/RAA**：通过统一执行与内存抽象支持多后端，而不是在每个算法里散落设备分支。

共同点不是“文件很多”，而是：核心契约稳定、依赖方向明确、后端可替换、模块可独立验证。

## 13. 推荐优先级

近期只优先推进以下三项：

1. `DeviceGuard + DeviceBuffer`：消除隐式设备状态和所有权不清；
2. `MemoryLayout + MemoryPlanner`：统一预检与真实分配；
3. `AnalysisContext`：作为组合根，逐步接管当前 `analysis` 的编排职责。

这三项稳定后，再拆 NLL/Hessian、AmpGen 和 ROOT 输出。不要从物理公式大重写开始。

最终目标不是让代码看起来“更面向对象”，而是做到：

> **设备状态可证明、显存账本可复核、模块边界可测试、物理结果可回归。**
