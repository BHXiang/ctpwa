# CTPWA 多 GPU、显存生命周期与 analysis 解构方案

> 状态：设计建议 v0.1
> 日期：2026-11-27
> 适用范围：`refactor/modular` 后续开发
> 核心目标：**多卡资源归属明确、显存峰值可复核、`analysis` 只做编排，不再成为连锁 bug 的来源。**

## 1. 为什么只做这两条主线

当前 `analysis` 仍同时掌握多类可变状态：事件 offset、振幅和 SL 表、当前 theta、
多卡临时 buffer、NLL/Hessian、积分输出、ROOT 文件以及设备状态。结果是一次局部修改
可能影响不相关的路径：

- 改显存释放，后续 Hessian 继续使用旧指针；
- 改 `writeResult`，改变 theta、offset 或当前设备，随后 NLL 出错；
- 改多卡归约，积分或拟合分数使用另一套样本口径；
- 预检和实际分配使用不同的显存计算方式；
- 首个 CUDA 错误被后续的 OOM、非法参数或段错误掩盖。

本方案只优先处理两个问题：

1. **多 GPU 分配、复制、同步和释放必须有明确所有权；**
2. **`analysis` 必须解构为薄的组合入口。**

不追求把所有 CUDA 基础函数包装成 manager，也不重写成另一套拟合框架。

## 2. 设计原则

### 2.1 基础 CUDA 调用不需要抽象

`cudaMalloc`、`cudaMemcpy`、`cudaFree`、`cudaSetDevice` 可以保留。需要抽象的是：

- buffer 在哪张卡；
- 分配多少字节；
- 谁负责释放；
- 何时允许释放；
- kernel 是否仍在使用；
- 多卡如何分片、广播和归并。

普通的 `CTPWA_CHECK(...)` 检查返回码已经足够，不需要 `CudaManager::malloc()` 这类空包装。

### 2.2 先明确所有权，再拆模块

每块设备内存必须能回答：

```text
在哪张卡？多少字节？谁分配？谁释放？哪个布局项对应？何时失效？
```

优先用 RAII 和 workspace 管理生命周期，不靠调用者记住在某个函数末尾手动 free。

### 2.3 显存布局只有一个真相来源

预检、实际分配、offset、日志和测试必须读取同一份 `MemoryLayout`。禁止：

- 预检按总振幅数估算 SL，实际分配却按独立 JP/cas；
- 预检按编译期 complex 大小估算 A，实际按运行期 float2 存储；
- 一个函数按单卡事件数计算，另一个函数按全样本计算；
- 构造期、拟合期和 Hessian 期的临时空间混为一个总数。

### 2.4 `analysis` 只组合，不实现所有细节

`analysis` 是用户 API 和内部模块的组合根。它不应直接管理裸指针、逐卡循环、ROOT 文件或
每个临时 buffer。方法名和公开签名保持不变，但内部应尽量只做参数检查和委派。

### 2.5 显式状态，减少隐藏副作用

耦合、theta、振幅版本、事件 offset 和设备状态必须显式传递。特别要避免：

```text
getNLL 改变隐藏 theta → writeResult 又依赖该 theta → 调整调用顺序后结果改变
```

如果保留缓存，必须用参数版本说明缓存何时有效。

### 2.6 渐进迁移，保持物理语义

不改 YAML、Python API、参数顺序、振幅名、ROOT 对象名和默认数值约定。每一阶段都应可
编译、可回归、可回退，不做一次性大重写。

## 3. 目标结构

```text
用户 Python API
      ↓
analysis（薄编排层）
      ↓
AnalysisContext
  ├── DecayModel          只读物理模型
  ├── DeviceContext       主卡、设备集合、stream、错误上下文
  ├── MemoryLayout        每卡布局与显存账本
  ├── AmplitudeStorage    A、SL、动量、置换表等常驻资源
  └── FitWorkspace        NLL/Hessian 临时工作区
      ↓
NLL / Hessian / Integrals / ResultBuilder（自由函数或小型算法模块）
      ↓
FitResult（普通数据对象）
      ↓
RootResultWriter（只负责 ROOT I/O）
```

依赖规则：

- 物理算法不直接操作 ROOT；
- 输出模块不修改拟合状态；
- `MemoryLayout` 不启动 kernel；
- backend 不解释共振态或衰变链；
- `analysis` 不拥有无归属的静态设备缓冲。

## 4. 多 GPU 与显存生命周期设计

### 4.1 `DeviceBuffer`：单块设备内存的所有权

建议最小接口：

```cpp
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(int device, std::size_t count);
    ~DeviceBuffer();                    // 正常生命周期内释放

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;

    T* get() const { return ptr_; }
    int device() const { return device_; }
    std::size_t bytes() const { return bytes_; }

    void release();                     // 显式提前释放，幂等
    void disarm();                      // 进程退出期仅放弃所有权，不再调用 CUDA
private:
    T* ptr_ = nullptr;
    int device_ = -1;
    std::size_t bytes_ = 0;
};
```

要求：

- 禁止拷贝，防止同一指针被释放两次；
- 移动后只保留一个所有者；
- `release()` 在分配它的设备上执行，且可重复调用；
- 部分多卡分配失败时，已分配部分必须全部回收；
- 析构函数不抛异常，退出阶段可选择 `disarm()`；
- 不把 `cudaSetDevice` 隐藏在析构函数里，必要时用 `DeviceGuard` 或 `ensure_device`。

`DeviceBuffer` 只负责所有权，不负责 SL、振幅或 Hessian 公式。

### 4.2 `MultiGpuStorage`：多卡分片和归并

```cpp
template <typename T>
class MultiGpuStorage {
public:
    void allocate(const BufferLayout& layout);
    void broadcast(int primary, const T* source);
    void reduceTo(int primary, ReduceOp op);
    void clear();

    std::size_t bytes(int gpu) const;
    DeviceBuffer& at(int gpu);
    const DeviceBuffer& at(int gpu) const;
private:
    std::vector<DeviceBuffer> per_gpu_;
};
```

职责：

- 每卡分片分配；
- 主卡广播；
- 跨卡复制和归并；
- 统一释放；
- 记录每卡实际分配字节数。

它不应决定物理分波和事件定义，这些由 `MemoryLayout` 输入。

### 4.3 `MemoryLayout`：唯一布局与显存账本

建议至少包含：

```cpp
struct MemoryLayout {
    int n_amplitudes = 0;
    int n_polar = 1;

    std::size_t amplitude_element_bytes = 0;
    std::size_t derivative_element_bytes = 0;

    std::vector<int> resident_events;   // 每卡 phsp+data+bkg；流式时不含 phsp
    std::vector<int> slcombs_per_jp;   // 独立 JP/cas，不按共振组合重复
    std::vector<int> nsigma_per_jp;    // 全同粒子置换项数

    std::size_t amplitudeBytes(int gpu) const;
    std::size_t slTableBytes(int gpu) const;
    std::size_t momentaBytes(int gpu) const;
    std::size_t derivativeBytes(int gpu) const;
    std::size_t residentBytes(int gpu) const;
};
```

另设阶段账本：

```text
constructor_peak  构造期 A + 当前 JP/SL + 已保留自由 SL + 临时上转
fit_resident      A + 自由 SL + 常驻权重/动量/dF
nll_workspace     NLL/梯度临时 buffer
hessian_peak      vv/vθ/θθ + 上转窗口 + 多卡归并
output_workspace  writeResult 上转、直方图、矩阵和 ROOT 临时空间
```

要求：

- `estimate()` 与 `allocationRecord()` 可以逐项比较；
- 常驻和临时峰值分开，不把不同生命周期的表简单相加；
- `q0`、JP 和共振组合只能来自 `DecayModel`，不由 DeviceManager 重新推导。

### 4.4 `FitWorkspace`：临时资源的边界

按算法划分工作区：

```text
NllWorkspace       NLL forward/backward 复用缓冲
HessianWorkspace   Hessian vv/vθ/θθ、上转窗口、归并缓冲
OutputWorkspace    writeResult 临时上转和结果缓冲
```

要求：

- 工作区随 `AnalysisContext` 或一次算法调用存在；
- 不再使用没有明确所有者的进程级静态设备缓冲；
- 临时上转窗口根据当前空闲显存和 `nPolar×nAmp` 限制；
- 多次调用可复用，但不能跨 analysis 对象共享未定义版本的缓存；
- 每个窗口大小和当前设备在日志中可诊断。

## 5. `analysis` 解构方案

### 5.1 保留的公开接口

以下接口不变：

```python
ana = ctpwa.analysis(config, fit_mode)
ana.getNLL(params)
ana.getHessian(params)
ana.reCalcAmp(theta)
ana.writeResult(...)
```

内部只保留少量组合状态：

```cpp
class analysis {
    std::shared_ptr<const DecayModel> model_;
    DeviceContext device_;
    MemoryLayout layout_;
    AmplitudeStorage amplitudes_;
    FitWorkspace fit_workspace_;
    FitState fit_state_;
};
```

`FitState` 至少包含耦合、theta 和振幅版本。若 offset 也影响缓存，必须纳入版本判断。

### 5.2 方法只做编排

示例：

```cpp
torch::Tensor analysis::getHessian(torch::Tensor params)
{
    checkParams(params, device_.primary());
    return computeFullHessian(
        params, model_->desc(), amplitudes_, fit_workspace_, fit_state_, device_);
}
```

```cpp
void analysis::writeResult(torch::Tensor params, const std::string& filename, ...)
{
    FitResult result = buildFitResult(
        params, model_->desc(), amplitudes_, fit_workspace_, fit_state_, device_);
    RootResultWriter writer(filename);
    writer.write(result);
}
```

公开方法只负责：

- 参数设备、dtype、维度检查；
- 选择正确的 `FitState`；
- 调用自由函数或小型算法模块；
- 返回结果。

### 5.3 `analysis` 不再承担的职责

| 职责 | 新归属 |
|---|---|
| 事件划分和 offset | `EventLayout` |
| 显存估算和分配记录 | `MemoryLayout` |
| A/SL/动量等常驻表 | `AmplitudeStorage` |
| NLL/Hessian 临时表 | `FitWorkspace` |
| 振幅重新计算 | `AmplitudeBuilder` |
| 积分、效率和拟合分数 | `MatrixIntegrator` |
| 直方图和结果收集 | `ResultBuilder` |
| ROOT 文件操作 | `RootResultWriter` |
| 参数名、slot、约束 | `Parameters` |

建议硬约束：

1. `analysis` 内不直接出现 `cudaMalloc/cudaFree`；
2. 多卡循环只出现在 storage/backend/workspace；
3. 输出函数不改变 NLL 所需的 theta、offset 或设备状态；
4. 不允许无归属的静态设备缓冲；
5. `analysis` 不读取 SL 表的内部字节布局；
6. ROOT 不出现在拟合计算路径。

## 6. ToyMC：从 `analysis` 中独立

### 6.1 为什么 ToyMC 不需要完整 analysis

ToyMC 的物理流程是：

```text
平坦 phsp MC → 计算每个候选事件的振幅强度 → 舍选抽样 → 保留事件和统计信息
```

它真正依赖的只有：

- 只读 `DecayModel`：粒子、衰变链、JP、SL、共振组合和极化；
- 明确的 `FitState`：拟合后的耦合与 theta；
- `DeviceContext + Workspace`：批量计算和后端资源；
- 随机种子、批次大小、舍选上限和输出配置。

它不需要事件划分、NLL、Hessian、拟合分数、ROOT 拟合结果树，也不应读取或修改
`analysis` 的隐藏状态。因此 ToyMC 应与拟合入口解耦，而不是继续放在 `analysis` 内。

### 6.2 推荐模块边界

```text
DecayModel（只读）
   ↓
PhaseSpaceGenerator（只负责平坦相空间运动学与候选事件）
   ↓
IntensityEvaluator（只负责 I(x)=Σ_pol|Σ_i A_i(x)·v_i|²）
   ↓
RejectionSampler（舍选、随机数、max/truncation 和统计）
   ↓
ToyMCResult（四动量、权重、接受率、max、截断率、seed）
```

各模块职责：

| 模块 | 负责 | 不负责 |
|---|---|---|
| `PhaseSpaceGenerator` | 平坦 phsp、母粒子/末态运动学、候选四动量 | 拟合参数和振幅强度 |
| `IntensityEvaluator` | 按 DecayModel+FitState 批量计算强度 | 舍选、ROOT 输出、NLL |
| `RejectionSampler` | 随机数、保留/拒绝、max/truncation 和统计 | 振幅公式和设备内存分配 |
| `ToyMCResult` | 接受事件、权重和诊断元数据 | 拟合状态或设备指针 |

`ToyMCResult` 同样可以携带生成时的 `FitState` 摘要，便于追踪样本对应的物理参数。

### 6.3 推荐接口

```cpp
struct ToyMCRequest {
    std::size_t n_candidates = 0;
    std::size_t batch_size = 100000;
    std::uint64_t seed = 0;
    double max_intensity = 0.0;   // <=0 时先估计；必须记录估计来源和有效性
};

class ToyMCGenerator {
public:
    ToyMCGenerator(std::shared_ptr<const DecayModel> model,
                   const DeviceContext& device);

    ToyMCResult generate(const FitState& state,
                         const ToyMCRequest& request);
};
```

`IntensityEvaluator` 只接收模型、显式 `FitState` 和一批候选四动量，不接收 `analysis`。
现有 `analysis.getIntensity()` 可以迁移为该模块的薄封装，避免改变底层振幅计算语义。

### 6.4 舍选抽样的必要约束

1. **max 必须有效**：估计值必须不小于候选批次真实强度上界；无法保证时只能扩大样本
   或采用安全的上界，不能把未知 max 当作 1。
2. **拒绝概率**：`accept = (u * max_intensity <= I(x))`。
3. **全零或 NaN 强度**：必须显式报错，不能无限循环或把 NaN 事件保留下来。
4. **截断可见**：候选数、接受数、接受率、最大强度和截断率必须记录。
5. **显存分批**：phsp 候选、强度上转和舍选缓冲均按批次分配，不按总样本一次性驻留。
6. **拟合状态不可变**：生成过程中不得改变 NLL/Hessian 后续使用的 theta、振幅或 offset。
7. **可复现**：同一模型、FitState、seed、批次和后端配置应产生相同结果；多卡只改变分片，
   不改变舍选逻辑。

### 6.5 与 `analysis` 的关系

保留一个兼容包装即可：

```cpp
ToyMCResult analysis::generateMC(const ToyMCRequest& request)
{
    return ToyMCGenerator(model_, device_).generate(fit_state_, request);
}
```

这样老用户无需改接口，但 `analysis` 不再实现相空间生成、逐事件强度和舍选循环。
如果已有 `FitResult`，也允许直接：

```cpp
ToyMCGenerator(model, device).generate(fit_result.fitState(), request);
```

从拟合结果生成 ToyMC，而不构造 `analysis`。

### 6.6 ToyMC 独立测试

- 常数强度模型：接受率与理论值一致，候选分布与平坦 phsp 一致；
- 全零强度或 NaN：立即给出明确错误，不死循环；
- max 检查：所有接受事件的强度均不超过记录的 max；
- seed 可复现：同 seed 相同结果，不同 seed 可区分；
- CPU/CUDA 与单卡/多卡一致；
- 生成前后 NLL、theta、offset 和 current device 不变；
- 大样本分批不会超显存，截断和接受统计完整。

## 7. `FitResult` 与 ROOT 输出解耦

先由 `ResultBuilder` 生成普通数据对象：

```cpp
struct FitResult {
    std::vector<double> parameters;
    Matrix interference_rec;
    Matrix interference_gen;
    Matrix interference_rec_err;
    Matrix interference_gen_err;
    HistogramSet histograms;
    IntegralMeta meta;
};
```

`RootResultWriter` 只负责：

- 创建/关闭文件；
- 写矩阵、树和直方图；
- 保持既有对象名和布局；
- 报告 I/O 错误。

收益：

- `output/` 不再需要 `analysis`；
- 输出可以用固定样本独立测试；
- 修改 ROOT 格式不会影响 NLL/Hessian；
- 结果构建和文件写出可以分别优化。

## 8. 错误处理与设备纪律

基础 CUDA 调用可以直接使用，但必须遵守：

1. 分配、复制、查询和 kernel launch 检查返回码；
2. 错误信息包含操作、设备、布局项和调用位置；
3. 首个 CUDA 致命错误后，不继续用同一进程诊断后续错误；
4. `cudaMemGetInfo` 失败不能翻译成“剩余 0 MiB”；
5. 正常返回和异常路径都恢复调用者设备状态；
6. 不在析构阶段假定 CUDA 上下文一定存在。

`compute-sanitizer` 只运行最小复现，不用整套拟合日志代替首个错误。

## 9. 分阶段实施顺序

### 阶段 0：建立基线

- 保存当前 CPU/CUDA/多卡数值回归；
- 固定公开 API、参数顺序和 ROOT 对象名；
- 记录典型模型的显存阶段账本。

验收：后续每次迁移都能比较 NLL、梯度、Hessian 和输出矩阵。

### 阶段 1：`DeviceBuffer` 与释放责任

- 从 `AmpCasDecay`、phsp matrix 等边界清晰的资源开始；
- 改为 move-only 所有权；
- 处理部分分配失败、提前释放和退出期 disarm。

验收：无 double free、跨卡释放、释放后使用；当前设备不变。

### 阶段 2：`MemoryLayout` 与 `MultiGpuStorage`

- 统一 A/SL/事件 offset 和预检；
- 将多卡分配和复制集中；
- 预检与实际分配逐项对照。

验收：多链、多共振态、多卡配置不再因重复估算被误拒。

### 阶段 3：`FitWorkspace`

- 清理 NLL/Hessian/输出的静态工作区；
- 上转窗口按空闲显存动态限制；
- 工作区按 analysis 或调用作用域隔离。

验收：多个 analysis 交替调用不互相影响；大样本 NLL/Hessian 不再因无预算窗口失败。

### 阶段 4：`analysis` 薄编排化

- 将方法改成参数检查 + 委派；
- 引入显式 `FitState` 和振幅版本；
- 隐藏 theta/offset 副作用移出公开方法。

验收：公开 API 不变，改变输出或调用顺序不改变 NLL/Hessian 结果。

### 阶段 5：`FitResult` 与 `RootResultWriter`

- 抽出结果数据对象；
- ROOT 输出不再依赖 analysis；
- 保持 ROOT 对象名和内容兼容。

验收：`output/` 无 `analysis` 依赖；固定结果可独立写出并读回比较。

### 阶段 6：ToyMC 独立化

- 将 `getIntensity` 抽成 `IntensityEvaluator(DecayModel + FitState)`；
- `PhaseSpaceGenerator` 只负责平坦候选事件；
- 用独立 `RejectionSampler` 实现舍选和统计；
- `analysis::generateMC` 仅保留兼容包装。

验收：不构造 analysis 也能由模型和拟合参数生成 ToyMC；生成前后拟合状态不变。

## 10. 测试与验收矩阵

### 所有权与设备

- 单块 buffer 释放、移动、重复 release；
- 多卡部分分配失败后的回收；
- 提前释放后不再使用；
- 正常/异常返回时 current device 不变；
- 多 analysis 交替调用无资源串用。

### 数值与多卡

- CPU vs CUDA；
- 1 卡 vs N 卡 NLL、梯度和派生量；
- auto/hybrid/double 三档；
- 多共振态共享 SL 表；
- `n_batch < n_gpus`、空分片和非均匀权重；
- 大振幅数 Hessian 和 off-shell BWR。

### 显存与诊断

- 预检与分配记录逐项一致；
- 构造、拟合、Hessian、输出分别记录峰值；
- OOM 时报告实际分配项和剩余显存；
- 最小 sanitizer 用例可复现首个设备错误。

### 行为与输出

- 公开 Python API 无签名变化；
- 参数顺序、振幅名、ROOT 对象名不变；
- 输出前后 NLL、theta、offset 和 current device 不变；
- 相同 seed 下重构前后 NLL/Hessian 在既定容差内一致。

## 11. 开发检查单

### 新增长驻设备资源

- [ ] 是否由 DeviceBuffer/MultiGpuStorage 持有？
- [ ] 是否记录设备、字节数、owner 和布局项？
- [ ] 是否明确释放时机和部分失败回收？
- [ ] 是否加入 MemoryLayout？

### 新增临时工作区

- [ ] 是否有明确作用域？
- [ ] 是否限制窗口大小？
- [ ] 是否避免跨 analysis 的未命名静态缓存？
- [ ] 是否纳入对应阶段峰值？

### 修改 analysis 公开方法

- [ ] 是否只做参数检查和委派？
- [ ] 是否引入新的隐藏状态或设备循环？
- [ ] 是否改变输出顺序或 theta/offset 的隐式副作用？
- [ ] 是否有独立回归测试？

### 修改输出

- [ ] 是否只改变 FitResult 或 RootResultWriter？
- [ ] 是否保持 NLL/Hessian 和设备状态不变？
- [ ] 是否保持 ROOT 对象名和归一化口径？

## 12. 反模式

- 为 `cudaMalloc/cudaMemcpy` 再包一层无意义的 manager；
- 用另一个 `AnalysisManager` 取代当前 god class；
- 所有临时 buffer 继续放在进程级 static；
- 为了通过测试只放宽数值容差；
- 先拆物理公式，再补资源所有权；
- 把“目录建好了”当成“生命周期和模块契约完成了”。

## 13. 完成定义

这两条主线完成时，应满足：

1. 每块设备内存都有明确 owner 和释放路径；
2. 显存预检与真实分配使用同一布局；
3. `analysis` 公开方法只负责编排，没有裸设备和 ROOT 细节；
4. 输出顺序和多个 analysis 的交替调用不改变数值结果；
5. CPU/CUDA/多卡回归通过，首个 CUDA 错误可直接定位。

最终目标不是文件数量或类数量，而是：

> **设备状态可证明、显存峰值可复核、资源释放可预测、修改一个功能不再连锁破坏其他模块。**
