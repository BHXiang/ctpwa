# 把 CTPWA 底座换成 TileLang 以适配国产 GPU —— 可行性研究

> 研究对象：CTPWA（本仓库，CUDA + PyTorch 协变张量分波分析）
> 问题：把计算底座换成 TileLang，以适配国产 GPU，是否可行？
> 结论口径：区分「TileLang 能不能跑国产卡」和「CTPWA 能不能换成 TileLang」——这是两个完全不同的问题，答案前者是"能"，后者是"不该"。
> 日期基准：TileLang v0.1.15（2026-09-30 发布）。

---

## 0. 结论（TL;DR）

**一句话：`TileLang 支持国产 GPU` 是真的；但 `把 CTPWA 的底座换成 TileLang` 既不必要、也不划算，而且它解决不了真正的阻塞点。**

三条判定：

1. **TileLang 的国产后端确实存在**，且是厂商/社区共建的：华为 **Ascend 950** 是主仓库内正式支持的后端（`USE_ASCEND=ON` 源码构建，需要 CANN + torch_npu）；**HYGON `hcu`（海光）/ MetaX `maca`（沐曦）/ Moore Threads `musa`（摩尔线程）/ Sunrise `tang`** 是 Ecosystem 适配器（独立仓库、**不进 release wheel**）。Cambricon / Iluvatar / OpenCL / SYCL **没有** TileLang 后端。

2. **但 CTPWA 的瓶颈不在"用哪个 DSL 写 kernel"**。真正的门槛按杀伤力排序是：
   - **NVRTC 运行时 JIT**（`src/JITCustom.cu`：运行时拼 CUDA C 源码 → nvrtc → `cuModuleLoadData` → 启动）。全库只有 CUDA NVRTC / Hygon hipRTC / 摩尔线程 MTRTC 有文档化的对应物；Ascend、Cambricon 没有。
   - **cuBLAS（118 处符号出现，其中 31 处为实质计算调用）**、**thrust/cub**、**torch 的 device 后端**、**pybind11 + ROOT 的构建链**。
   TileLang 一个都不替换——它只生成 kernel。

3. **CTPWA 最热的 kernel 恰好是 TileLang 最表达不了的那类**。`computeCustomAmpsKernelT` 是一个跑在 (SL × event) 网格上的**栈式字节码解释器**（`switch(opcode)` + 数据相关动态控制流 + CSE 槽）；`computeCustomHessianKernel` 同理。TileLang 没有 `switch`、没有函数/类、没有 device 函数调用、**没有复数类型**，官方也没有任何字节码解释器的先例。

**推荐路线：不要换底座。** 要适配国产卡，按性价比排序：
1. **海光 DCU（ROCm/HIP）** —— 唯一在 NVRTC 这一最硬门槛上有成熟、社区验证答案（hipRTC + jitify）的平台，且有 hipBLAS / rocThrust / hipCUB、`torch.utils.cpp_extension` 有真实维护的 hipify 路径。**首选。**
2. **摩尔线程 MUSA** —— 唯一一个国产栈在文档上同时覆盖六个门槛（mcc + MUSA Mapping + **MTRTC** + muBLAS + torch_musa）。**国产非 ROCm 首选。**
3. MetaX / Iluvatar / Biren / Enflame —— 先逼厂商书面回答一个问题：**"在你平台上怎么运行时编译生成的 device 源码？"** 这一问决定是"移植"还是"重写"。
4. Ascend / Cambricon —— 非 SIMT 架构，kernel 层必须重写。TileLang-Ascend 只能帮 GEMM 类，帮不了解释器。

如果一定要在某处用 TileLang，它的正确位置是：**新写的、tile 形的 kernel**（NLL 的 matvec、Hessian 的 GEMM/规约），并且要接受它与现有 .cu 无法同编译单元、只能通过 .so/Python 层拼接。而这一部分其实有更便宜的可移植化手段（见 §8）。

---

## 1. 先拆解"底座换成 TileLang"到底指什么

这句话至少有三种解读，可行性差别巨大：

| 解读 | 含义 | 可行性 |
|---|---|---|
| **A. 全量替换** | 61 个 `__global__` 内核全部重写成 TileLang，TileLang 成为唯一 codegen 路径 | ❌ 不建议。核心 kernel 表达不了 + 依赖墙不动 + 性能大概率回退 |
| **B. 热点局部替换** | 只把 tile 形热点（matvec/GEMM/归约/elementwise）用 TileLang 写，其余保留手写 | ⚠️ 技术可行，但收益有限（见 §8，有更便宜的等价方案） |
| **C. 替换 JIT codegen 层** | 把 `JITCustom.cu` 的 "字节码 → CUDA C → NVRTC" 换成 "字节码 → TileLang → 目标后端" | ❌ 架构上不成立。TileLang 是 Python 侧 TVM 编译流程，产出自己的 .so，无法在 C++ 运行时内当 drop-in 用（§4.3） |

下面的分析主要针对 A，并给出 B/C 的边界。

---

## 2. CTPWA 可移植性体检（本地代码事实）

### 2.1 规模

| 项 | 数值 |
|---|---|
| `src/*.cu` | 22,600 行（17 个文件） |
| `include/*.cuh` | 3,598 行（19 个文件） |
| `__global__` 内核 | **61 个不同内核**（`src/*.cu` 中 66 处定义；连头文件声明共 85 处 `__global__` 出现） |
| `__device__` 函数 | 148 |
| cuBLAS | 118 处符号出现 / 115 行；其中**实质计算 31 处**：`Daxpy` 11、`Cgemv` 7、`Zgemm` 3、`Zgemv` 2、`Zdgmm` 2、`Cdgmm` 2、`ZgemmStridedBatched` 1、`CgemmStridedBatched` 1、`Cgemm` 1、`Cdotc` 1；其余为 `Create/Destroy/Handle/Status/Operation` |
| `cudaMemcpy` / `cudaMalloc` | 319 / 253 |
| `atomicAdd` / `__syncthreads` / `__shfl_down_sync` | 51 / 32 / 6 |
| kernel 内 `printf` | 82（调试用） |

### 2.2 依赖面（这是真正的墙）

```
CUDAExtension(torch.utils.cpp_extension)  ← 构建路径绑定 torch 的 CUDA 分支
 ├─ cudart / cublas / nvrtc / cuda(driver API)   ← setup.py 显式链接
 ├─ thrust (host_vector/device_vector/complex/sort/reduce/binary_search)
 ├─ cub (cub.cuh)
 ├─ ROOT (libCore…libPhysics)、yaml-cpp         ← 主机侧，可移植
 ├─ pybind11                                     ← 主机侧，可移植
 └─ nvcc flags: -rdc=true + nvcc_dlink -dlink    ← 可重定位 device 代码 + device link
                    --extended-lambda, --expt-relaxed-constexpr
```

### 2.3 对移植**有利**的一面（好消息）

CTPWA 没有用那些最难搬的 CUDA 特性：

- ❌ 无 texture / surface
- ❌ 无动态并行（device 内 `<<< >>>`）
- ❌ 无 cooperative groups
- ❌ 无 device lambda（`--extended-lambda` 是个空转的 flag）
- ✅ 只有 6 处 `__shfl_down_sync`（warp 语义敏感面很小）
- ✅ 主机侧 `Analysis.cu`(6,243 行) / `Config.cu`(1,855) / `Info.cu`(593) 基本是纯 C++

这解释了为什么"Shader/shim 式重编译"（MUSA Mapping、hipify）对 CTPWA 是**有可能**的——难点集中在 NVRTC 与 cublas/thrust/cub，而不在 kernel 语法。

### 2.4 计算画像（为什么 TileLang 不匹配）

来自 `doc/optimizer-attribution.md` 的实测：

| 项 | 耗时 | 说明 |
|---|---|---|
| 前向 `getNLL` | ~86 ms/eval | **占绝对主导**，内存带宽型 |
| 反向 `autograd.grad` | 0.81 ms | 0.9% |
| 一次完整 `getHessian` | ~403 ms | ≈ 4.7 次求值 |
| 驻留振幅矩阵 A | n_ev × n_pol × n_amp × 16B | 单卡可达 28 GB 量级 |

关键：**一次求值的成本 ≈ 9.8 ms（固定）+ b·N_phsp**，是典型的"大数组流式读写 + 归约"，不是 GEMM 密集。TileLang 的强项（tile/流水/tensor core）打不到这个点上。

---

## 3. 逐 kernel 可行性分级

把 61 个内核按"TileLang 是否能表达、是否值得"分四档：

### 档 A —— tile 形，TileLang 可直译（约 15~20 个）

`kern_fused_fwd`（A^T·v matvec + |S|² + log + warp 归约 + atomicAdd，14 个 NP 模板实例）、`perEventHessianKernel`、`reduceHessianKernel`、`gnBlockReduceKernel`、`computeSfromAmpsKernel`、`computeSFromFloatAmpsKernel`、`computeFactorsAndWeightsKernel`、各类 cast/axpy/conjugate/negate/reorder/downcast kernel。

TileLang 对此有原生设施：`T.Parallel`、`T.atomic_add`、`T.alloc_shared`、`while`/动态上界循环、per-thread 分支、`T.print`。

### 档 B —— 可重写，但要手写大量语义（约 10~15 个）

`computeSLAmpKernel`（SL 振幅表构建）、`computeBFKernel`（势垒因子/breakup momentum）、`resonanceGradientKernelRuntime`、`multiplicativeCouplingKernel`、`computeCustomAmpsKernelT` 中 `R==1` 的直算分支。

这些是标量数学，但涉及 `pow`/`log`/复数与势垒因子，需要 (re, im) 手动展开（§4.2 无复数类型）。

### 档 C —— TileLang 表达不了（核心阻碍）

| kernel | 为什么不行 |
|---|---|
| `CustomExpr.cu: evalCustomSeg / evalCustomAll` | **栈式字节码解释器**：`for(i<n_instr){ op=pc[0]; switch(op){...} }`，32 深数据栈 + 64 个 CSE 槽 + 数据相关控制流。TileLang 无 `switch`、无函数/类、无 device 函数调用、"Python list 不是 device buffer"，官方无任何解释器先例。 |
| `ComputeHessian.cu: computeCustomHessianKernel` | 同上，且是二阶导版本，函数体量更大 |
| `JITCustom.cu: ctpwa_jit_grad / ctpwa_jit_full` | **运行时生成**的 CUDA C + NVRTC + driver API。不是"写一个 kernel"，是"在进程里造 kernel"。 |

### 档 D —— 冷路径，不值得移植

`Figure.cu` 的直方图/观测量 kernel（ROOT 出图用，6 个）、`ComputeResults.cu` 的部分 kernel。

**结论：档 C 是硬伤。** 它不是"写起来麻烦"，而是"目标语言的表达能力不覆盖"。而档 C 里的 `computeCustomAmpsKernelT` 是需要移植的那条路径上的**最热 kernel**。

---

## 4. TileLang 侧的事实核查

### 4.1 后端支持矩阵（来源：TileLang README + docs，v0.1.15）

TileLang 官方定义四个等级：**Primary**（核心）、**Supported / Experimental**（主仓库内、进 wheel）、**Ecosystem**（独立仓库、**不进 release wheel**、"可能遵循独立的兼容节奏"）。

| 硬件 | target | 等级 | 进 wheel？ |
|---|---|---|---|
| NVIDIA CUDA | `cuda` | **Primary** | 是（SM70–SM120，CI 覆盖） |
| AMD ROCm/HIP | `hip` | **Supported** | 是（CI 只跑 gfx942/MI300X；gfx950 未覆盖） |
| Apple Metal | `metal` | **Supported** | 是 |
| **华为 Ascend 950** | `ascend` | **Supported（在树内）** | **否**，`USE_ASCEND=ON` 源码构建 + CANN + torch_npu |
| LLVM CPU | `llvm` | Experimental | 否 |
| CuTe DSL / WebGPU | `cutedsl` / `webgpu` | Experimental | — |
| **华为 Ascend A2/A3** | `ascendc`/`pto`/`npuir` | **Ecosystem** | 否（tilelang-ascend / tilelang-mlir-ascend） |
| **沐曦 MetaX** | `maca` | **Ecosystem** | 否（tilelang-metax） |
| **摩尔线程 MUSA** | `musa` | **Ecosystem** | 否（tilelang-musa） |
| **海光 HYGON** | `hcu` | **Ecosystem** | 否（tilelang-hygon，需 DTK） |
| Sunrise-AI TANG | `tang` | **Ecosystem** | 否 |
| **寒武纪 Cambricon / 天数智芯 Iluvatar** | — | **不存在** | — |
| 通用 OpenCL / SYCL | — | **不存在** | — |

要点：**国产卡都在 Ecosystem 档**——独立仓库、不进 wheel、不受上游 CI 保障、版本可能落后（例如 tilelang-metax 的 README 还停留在没有 Ascend 行的旧版）。Ascend 950 是唯一进了主仓库的国产后端，且由 9 名 DeepSeek 工程师主导、华为协作开发。

### 4.2 语言能力对 CTPWA 的匹配度

| 需要的 | TileLang | 影响 |
|---|---|---|
| flat array 逐元素标量运算 | ✅ `T.Parallel` / `T.serial` | 档 A 可以 |
| 归约 + atomic 累加 | ✅ `T.atomic_add/max/min`，含 tile-region 向量化 | 档 A 可以 |
| 数据相关动态上界 / 提前退出 | ✅ `while`(TIR 条件) + `break`/`continue` | 档 A/B 可以 |
| 发散 per-thread 分支 | ✅ `if/elif/else` + `T.get_thread_binding()` | 可以，但发散语义自己负责 |
| printf 调试 | ✅ `T.print`（`print()` 不推荐） | 82 处 CUDA printf 需改写 |
| **复数** | ❌ **无原生复数类型** | 全库 `ctComplex`/`thrust::complex<double>` 需手工拆成 (re,im) 对。Tile-AI 自家 FFT 就是这么做的，可行但工作量大 |
| **struct / class / 函数** | ❌ 不支持；只有 `@T.macro`（**编译期内联**） | 须把所有辅助结构摊平成裸数组 |
| **外部 `__device__` 函数** | ⚠️ 只能用 `T.call_extern` + `T.CUDASourceCodeKernel` 注入；无 `import_source` | 现有 148 个 `__device__` 无法复用 |
| **switch / 字节码解释器** | ⚠️ 只能写成 if/elif 链 + device buffer 当 dispatch 表；**无官方先例** | 档 C 阻塞 |

### 4.3 互操作（这条最要命）

- ✅ 能拿到生成的 CUDA C 源码（`kernel.get_kernel_source()`），也有 `tilelang_callback_cuda_postproc` 钩子。
- ❌ **它独占编译单元**：`tilelang/jit/adapter/libgen.py` 只写**一个**生成的源文件，然后 `nvcc --shared <那一个文件>`，只隐式链接 `-lcuda`，再 `ctypes.CDLL` 一个自己的 .so。**不存在把你的 .cu 一起编进去的路径。**
- ❌ **不支持从生成代码里调 cuBLAS/cuSOLVER**（没有链接、`T.gemm` 是它自己的 MMA 路径）。
- ⚠️ 因此"混用"只能发生在 .so / Python 层——而 CTPWA 是 pybind11 的 C++ 扩展，这意味着要么把逻辑上移到 Python，要么通过进程内 FFI 调 TileLang 的 .so。

**对解读 C（替换 JIT codegen）的致命一击**：CTPWA 的 JIT 是 C++ 进程内、运行时、按 config 现编译（还带超时看门狗、进程级二进制缓存、与主库逐位一致的数学约定）。TileLang 的编译是 Python/TVM 侧的重量级流程——无法在 C++ 运行时里当 drop-in 使用。

### 4.4 成熟度风险

- v0.1.15，**仍是 0.x，PyPI 分类 "Development Status :: 4 - Beta"**；月更节奏；438 open issues。
- **破坏性变更频繁**：0.1.13「移除若干 legacy API」；0.1.15 移除 `T.Pipelined(sync/group)`、`T.gemm_sp(k_pack)`，弃用 `T.symbolic`。必须 pin 版本。
- Ecosystem 适配器明确"不在 release wheel 内、可能独立兼容节奏"——**没有上游 CI 保障**。
- 唯一 Primary 是 CUDA；ROCm CI 只覆盖 gfx942。

---

## 5. 逐子系统可行性判定

| 子系统 | 行数（约） | TileLang 能否覆盖 | 说明 |
|---|---|---|---|
| `AmpGen.cu`（振幅生成/切片/JIT 物化） | 3,953 | ❌ 核心不行 | 含最热的解释器 kernel 与 JIT 物化读取 |
| `ComputeHessian.cu`（Hessian 20 kernel） | 1,843 | ⚠️ 一半 | GEMM/归约部分可以，per-event 二阶导的解释器部分不行 |
| `ComputeNLL.cu`（NLL/融合前向/梯度） | 1,094 | ✅ 大部分 | 档 A 的主场 |
| `CustomExpr.cu`（字节码解释器） | 1,306 | ❌ | 档 C |
| `JITCustom.cu`（NVRTC 运行时 JIT） | 1,209 | ❌ | 档 C，且架构上不成立 |
| `ResModel.cu`（模型 AST + 符号微分规则） | 591 | ➖ | 主机侧 codegen，与 TileLang 无关 |
| `Figure.cu`（ROOT 出图 kernel） | 738 | ➖ | 冷路径 |
| `ComputeBF.cu` / `ComputeResults.cu` | 516 | ✅ | 档 A/B |
| `Parameters.cu` / `Analysis.cu`（主机编排） | 7,315 | ➖ | 不是"移植 kernel"，是"换 cuda* API 名" |
| `Config.cu` / `Info.cu` / `Resonance.cu` / `main.cu` | 3,103 | ➖ | 纯主机，可移植 |

粗算：**需要移植的 device 代码 ~15k 行；其中 TileLang 真正适配的 ~4~5k 行（档 A/B），TileLang 表达不了的 ~4k 行（档 C），其余是主机侧或冷路径。**

---

## 6. 真正的六道门槛（与 DSL 选择无关）

这是本报告最重要的部分。即使你把 kernel 全部换成某种可移植 DSL，下面六道门仍然要逐道过：

| # | 门槛 | CTPWA 现状 | 国产栈对应物 |
|---|---|---|---|
| 1 | **运行时 JIT** | NVRTC + `cuModuleLoadData`（`JITCustom.cu`，带超时/缓存/逐位一致约定） | ✅ hipRTC(+jitify)（海光/ROCm）<br>✅ **MTRTC**（摩尔线程，有"NVRTC 接口对应关系"专章）<br>✅ 有 RTC（燧原）<br>❌ Ascend / Cambricon / MetaX / Iluvatar / Biren：**未找到文档化对应物** |
| 2 | cuBLAS（31 处实质计算） | cublas Cgemv/Zgemm/... | hipBLAS/rocBLAS（海光）、muBLAS（摩尔）、mcBLAS（沐曦）、ixBLAS（天数）、suBLAS（壁仞）、ACLBLAS（昇腾，API 不同） |
| 3 | thrust / cub | sort/reduce/binary_search/complex | rocThrust/hipCUB（海光）、**mcThrust/mcCUB（沐曦，文档明确）**、摩尔未确认 |
| 4 | torch device 后端 | `torch::Tensor`/`.cuda()`/autograd（199 + 14 处） | torch_musa、torch_npu、mcPyTorch、torch 2.7.1+corex、torch_mlu… **必须存在，且 pybind11 也要能用** |
| 5 | 构建链 | CUDAExtension + `-rdc=true` + `dlink` + ROOT + yaml-cpp | ROCm 有真实维护的 hipify 路径；其余多为 "换编译器/开插件" |
| 6 | 精度模型 | `CTPWA_DOUBLE_COMPLEX` 编译期 switch，double 为主 | 天数：**warpSize=64 + double 仅部分支持**；HIP：warpSize 64 |

**第 1 条是分水岭。** 建议对任何候选厂商只问一句话：

> **"在你们平台上，如何在运行时编译动态生成的 device 源码？"**

答案决定这是"移植"还是"重写"。

---

## 7. 三条现实路线对比

| 路线 | 覆盖厂商 | 本质 | 对 CTPWA 的代价 | 推荐度 |
|---|---|---|---|---|
| **HIP / ROCm** | 海光 DCU（+ AMD） | 源码级 CUDA 兼容语言 + hipify | 重编译 + 有界修补：warpSize 64、`__shfl_sync` 签名、hipBLASLt 覆盖、NVRTC→hipRTC 近乎逐行 | ⭐⭐⭐⭐⭐ 首选 |
| **MUSA Mapping + mcc** | 摩尔线程 | Clang 前端插件，语义级改写，**源文件保持 CUDA 写法**；`torch-include.json` 明确映射 `ATen/cuda`→`ATen/musa` | 重编译 + 有界修补。风险：插件是**闭源二进制且与 mcc/Clang 版本锁死**；mapping 表有缺口需改源码；host `.cpp` 也需插件；FP8 需 `-DCCCL_DISABLE_NVFP8_SUPPORT` | ⭐⭐⭐⭐ 国产非 ROCm 首选 |
| **TileLang 重写** | 昇腾950 / 海光 / 沐曦 / 摩尔（Ecosystem） | 用新 DSL **重写** kernel | 档 A/B 重写 ~5k 行，档 C（~4k 行，含最热 kernel）**写不出来**；且不碰门槛 1–6 | ⭐ 仅作为新 kernel 的备选 |

补充：生态里还存在一个**跨厂商 Triton 分支 FlagTree**（Triton 3.6 基座），文档化后端覆盖 nvidia/amd/iluvatar/mthreads/metax/ascend/cambricon/hcu/enflame/sunrise 等，并提供 TLE-Raw "inline CUDA" 逃生舱。它和 TileLang 是同一类工具（**重写**而非**搬运**），覆盖面更广，同样帮不了档 C。若目标只是"新 kernel 一次编写多卡运行"，TileLang 与 FlagTree 都应纳入候选，而不是拿它们去替换既有的 26k 行。

---

## 8. 如果一定要提升国产卡适配性，应该做什么

按"收益/成本"排序，这比换 TileLang 划算得多：

### 8.1 （最高性价比）把档 A 从手写 kernel 换成 torch 算子

NLL 的 `S = A^T·v`、`|S|²`、phsp 均值、Hessian 的 GEMM/归约，本来就是"tile 形"的。用 `torch.matmul` / `torch.sum` / `torch.log` 表达后，**自动获得 torch 已有的全部 device 后端**（torch_musa / torch_npu / torch_corex / …）——不需要 TileLang，不需要新 DSL。

代价：损失当前的融合优化（`kern_fused_fwd` 把 matvec+log+权重+归约融成一个 kernel，还有 autotune 的 chunk 划分）。所以正确做法是**保留 CUDA 快路径 + 增加一个 torch 回退路径**，由构建开关或运行时探测选择。

### 8.2 引入薄 Backend 抽象层，而不是换 DSL

在主机侧抽出 5 个原语：`alloc/free`、`memcpy`、`stream`、`gemm`、`launch`，加上 `jit_compile`。当前的 `cudaMalloc`(253)/`cudaMemcpy`(319)/`cublas`(118) 全部收敛到这层。这样：
- CUDA-like 厂商（摩尔/沐曦/天数/壁仞）→ 换实现 + shim 头
- 海光 → 换 HIP 实现
- 昇腾 → 换 CANN 实现（但 kernel 层仍需重写）

**一个 DSL 不能替代这个抽象层；TileLang 也不是这个抽象层。**

### 8.3 让 JIT 变成可关可插拔

`JITCustom.cu` 的注释已说明"任何失败 → 整块回退解释器"。**这意味着 JIT 是可选的优化，不是硬需求。** 在没有 NVRTC 对应物的平台上，关掉 JIT 走解释器即可先跑起来（会变慢），把"能不能用"和"快不快"解耦。这是最低成本的破局点。

### 8.4 三层拆分：哪些能转 torch 算子，哪些必须按卡开发

这是"能力拆分"的精确边界。关键是把两件不同的事分开：

- **按 GPU 特性开发** = 因为硬件执行模型不同而重写算法/并行分解（真·重写）
- **按厂商软件栈适配** = 换编译器、改 API 名、换 BLAS/JIT 库（每栈一次，工具链问题）

| 层 | 内容（CTPWA 具体对象） | 覆盖范围 | 工作量性质 |
|---|---|---|---|
| **T1 算子层**<br>torch 可完整表达 | `kern_fused_fwd`（S=Aᵀv + \|S\|² + log + 加权归约）<br>`computeFactorsAndWeightsKernel`<br>`computePhspMeanKernel` / `doublePhspSumKernel` / `quadratic_form_small` / `quadratic_form_large`（都是 Re(v^H M v)）<br>`perEventHessianKernel` + `reduceHessianKernel`<br>`computeSfromAmpsKernel` / `computeSFromFloatAmpsKernel` / `gnBlockReduceKernel`<br>全部 cast / axpy / conjugate / negate / downcast / reorder kernel<br>`Figure.cu` 直方图 | **零 per-GPU 开发**：写成 `torch.matmul` / `einsum` / `log` / `sum` 后，自动跑在任何 torch device 后端（torch_musa / torch_npu / torch_corex / …） | 新写一条回退路径 |
| **T2 标量/发散层**<br>源码级可移植 | `computeCustomAmpsKernelT`（含字节码解释器）<br>`evalCustomSeg` / `evalCustomAll`<br>`computeCustomHessianKernel`<br>`computeSLAmpKernel` / `computeBFKernel`<br>`resonanceGradientKernelRuntime` / `multiplicativeCoupling*`<br>`JITCustom.cu` 的 NVRTC 路径 | CUDA-like 家族（摩尔 mcc / 沐曦 mxcc / 天数 / 壁仞）→ **重编译**；海光 → hipify | **换栈，不重写**：纯标量数学，对 warp 语义不敏感 |
| **T3 主机/栈层** | `cudaMalloc`(253) / `cudaMemcpy`(319) / `cublas`(118) 收敛到 Backend 抽象<br>torch device 后端、ROOT、yaml-cpp、pybind11 | 每厂商一次 | 抽象层 + 适配器 |

**两个来自代码的事实，让 T1 比预想便宜得多：**

1. **warp 级代码 100% 集中在 T1**。全库的 `__shfl_down_sync` 与 `ctShflDown` 只出现在 `src/ComputeNLL.cu`（17 个 kernel / 1,094 行；即 `kern_fused_fwd` 与 `quadratic_form_*`/`doublePhspSumKernel` 的手写归约）。这意味着 **T1 一旦转成 torch 算子，`warpSize=32 vs 64` 这个最经典的移植坑基本消失**——T2 剩下的是纯标量数学，不碰 warp 语义。
2. **unfused 回退路径已经存在**。`src/ComputeNLL.cu:186` 的 `default:` 分支（`nP` 不在模板分发列表时）已经在做"cublas CGEMV + `computeFactorsAndWeightsKernel`"的非融合计算。把这两步换成 `torch.matmul` + `torch.log/sum`，就构成 T1 的最小可移植路径——不是从零重构。现状全库几乎不用 torch 算子（只有 `complex`/`cat`/`stack`/`real`/`imag` 共 16 处，且都在主机侧打包）。

**代价与边界（要诚实）：** T1 转 torch 会**丢掉融合**。`kern_fused_fwd` 之所以存在，就是因为 matvec+log+加权+归约融成一个 kernel 比 cublas + 分离 kernel 快。所以正确形态是**双路径：保留 CUDA 融合快路径 + 增加可移植回退路径**，由构建开关或运行时探测选择，并用 `tests/golden/` 做 A/B 数值验证。

**转不走、必须按卡决策的两件事（属"特性"，非"算子"）：**
- **double 精度**：CTPWA 默认 `CTPWA_DOUBLE_COMPLEX`（全 double）。天数智芯文档明确 double 仅部分支持，消费级卡 double 吞吐也低。这需要按卡决定是否走 `precision: float`（该开关在 `doc/precision_hybrid_design.md` 中已设计）。
- **显存容量 → 多卡切分策略**：单卡驻留振幅矩阵可达 28 GB 量级，不同卡容量直接改变切分/流式方案。这是配置与调度问题，不是 kernel 开发。

**结论：真正需要"按 GPU 特性开发"的只有非 SIMT 家族（昇腾/寒武纪），而那是 T2 的重写，与 T1 无关。** 只要优先覆盖 CUDA-like + HIP 家族，T2 的代价就是"每栈一次重编译 + 有界修补"，T1 是零。

### 8.5 若仍想在局部用 TileLang

最小可行范围：只做档 A 里 **1~2 个** 最大热点（例如 `kern_fused_fwd` 的 matvec+|S|²，以及 `perEventHessianKernel`），并且接受：
- 它产出独立 .so，通过 Python/FFI 层与主扩展拼接；
- 需自己实现复数（(re,im) 成对）；
- 需 pin 版本、自己承担 Ecosystem 后端的兼容风险；
- 需要 A/B 验证数值（CTPWA 有 `tests/golden/`，逐位/统计一致性门槛高）。

先做一个 spike 验证可行性再谈推广。

---

## 9. 工作量粗估

| 方案 | 设备代码工作 | 其他工作 | 风险 |
|---|---|---|---|
| 海光 DCU（HIP） | hipify + 手工修 61 个内核的 warp 语义（**无 texture/动态并行/cooperative group，只有 6 处 shfl**） | hipRTC 改写 JIT、hipBLAS、rocThrust/hipCUB、torch+DTK 版本匹配、数值回归 | 中。DTK 跟随某个 ROCm 版本，需锁定厂商版 torch |
| 摩尔线程 MUSA | MUSA Mapping 重编译 + mapping 缺口修补 | 验证 MTRTC 与 libmtrtc、确认 muThrust/muCUB、CMake `-x cu` 冲突处理 | 中。插件闭源、版本锁死 |
| TileLang 全量重写 | **重写 ~15k 行**，其中 ~4k 行（含最热 kernel）**无解** | 仍需自己解决 NVRTC/cuBLAS/torch 三墙 | **极高，且核心路径走不通** |
| TileLang 局部（档 A 的 1~2 个热点） | 数百行 + 复数手工展开 | 独立 .so 拼接、数值 A/B | 中 |

参考量级：把 15k 行手写 CUDA 重写成新 DSL，保守 40~75 人日以上，且**验证成本高于编码成本**（`tests/` 有 32 个测试文件、55 个 config、12K golden，物理/数值一致性门槛高）。

---

## 10. 建议的下一步

1. **先做门槛问询，不写代码。** 把 §6 的六道门做成一张问题清单，发给目标厂商（尤其第 1 条）。**未经书面确认前不要承诺移植。**
2. **如果目标是"尽快在国产卡上跑起来"** → 走海光 DCU/HIP 或摩尔线程 MUSA 的 shim 路线；先跑通 `tests/test_numerical.py::test_gradient_vs_fd` 这类最严的数值门禁。
3. **如果目标是"长期跨厂商可移植"** → 做 §8.1 + 8.2 + 8.3（torch 回退路径 + Backend 抽象 + JIT 可关），这比换 DSL 更根本，也不锁死任何厂商。
4. **TileLang 只在一个场景下值得投入**：未来**新写**的、结构规整的 kernel 需要一次编写多卡运行。此时把它和 FlagTree 一起评估，而不是拿它去改造既有代码。

---

## 附录：证据来源

**本地代码事实**：本仓库 `src/`、`include/`、`setup.py`、`doc/optimizer-attribution.md`、`doc/precision_hybrid_design.md`（行数/内核计数/依赖/性能画像均由本次会话直接统计）。

**TileLang**：
- [README › Platform and Backend Support](https://github.com/tile-ai/tilelang#platform-and-backend-support)（后端四级矩阵）
- [Understanding Targets](https://tilelang.com/get_started/targets.html) · [Type System](https://tilelang.com/programming_guides/type_system.html) · [Control Flow](https://tilelang.com/programming_guides/control_flow.html) · [Python 兼容性](https://tilelang.com/programming_guides/python_compatibility.html)
- [`tilelang/language/atomic.py`](https://github.com/tile-ai/tilelang/blob/main/tilelang/language/atomic.py) · [`tilelang/jit/adapter/libgen.py`](https://github.com/tile-ai/tilelang/blob/main/tilelang/jit/adapter/libgen.py)（独占编译单元）· [backend/README.md](https://github.com/tile-ai/tilelang/blob/main/tilelang/backend/README.md)
- [Ascend README](https://github.com/tile-ai/tilelang/blob/main/tilelang/ascend/README.md) · [v0.1.15 release](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15) · [PyPI](https://pypi.org/project/tilelang/)
- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) · [tilelang-metax](https://github.com/tile-ai/tilelang-metax) · [tilelang-musa](https://github.com/tile-ai/tilelang-musa) · [tilelang-hygon](https://github.com/tile-ai/tilelang-hygon)

**国产栈**：
- 海光/ROCm：[hipRTC](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html) · [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/hip_porting_guide.html) · [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/docs-7.14.1/index.html) · [jitify](https://github.com/ROCm/jitify)
- 摩尔线程：[MUSA Mapping](https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/musa_mapping/) · [**MTRTC**](https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/mtrtc_runtime_compilation/) · [musify](https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/musify/) · [torch_musa](https://github.com/moorethreads/torch_musa)
- 昇腾：[Ascend C AI-Core-SIMD 编程模型](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/programug/Ascendcopdevg/docs/zh/guide/编程指南/编程模型/AI-Core-SIMD编程/基于指针的C语言编程/C语言编程概述.md) · [torch_npu 自定义算子/插件](https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/adaptation_compile_opplugin.md)
- 沐曦：[MACA 编程接口](https://developer.metax-tech.com/api/client/document/preview/1192/split_files/编程接口.html) · [llm.c 移植报告（第三方）](https://developer.metax-tech.com/forum/t/jiu-ban-lun-tan-tui-jian-tie-yong-hu-inkstonegua-pei-karpathy-de-llmc-xiang-mu-dao-mu-xi-metaxgpu-ping-tai/27/)
- 天数：[PaddlePaddle iluvatar_gpu 支持文档](https://raw.githubusercontent.com/PaddlePaddle/docs/refs/heads/release/3.3/docs/hardware_support/iluvatar_gpu/install_cn.md)（含 warpSize=64 警告）
- 燧原：[TopsCC 编程指南](https://support.enflame-tech.com/onlinedoc_dev_2.5.115/4-develop/program_guide/topscc/topscc/content/source/topscc_programming_guide.html)（block dim 上限 6/12、GCU 2.0 禁用直接 L2/L3 访问）
- 寒武纪：[mlu-ops](https://github.com/Cambricon/mlu-ops) · [torch_mlu](https://github.com/Cambricon/torch_mlu)
- 跨厂商：[FlagTree 0.5.0 release notes](https://raw.githubusercontent.com/flagos-ai/FlagTree/main/reports/v0.5.0/release_notes_v0.5.0.md)

**高能物理侧**：IHEP「软硬协同：国产异构算力助力高能物理研究」（[indico 198682](https://indico.ihep.ac.cn/event/26373/contributions/198682/)）——存在但附件未能解析，**未找到公开的定量 CUDA→国产卡 HEP 移植成本研究**。

**未验证项（明确列出，不作否定结论）**：Ascend/Cambricon/MetaX/Iluvatar/Biren 是否绝对没有运行时 JIT（本次仅"未找到文档"）；MUSA 的 thrust/cub 对应物名称；MetaX 的 `<<<>>>`/threadIdx 语法确认；Biren SUPA 编程指南；寒武纪 CNNL API；SYCL/Kokkos/RAJA/ZLUDA/AKG 对国产卡的支持。
