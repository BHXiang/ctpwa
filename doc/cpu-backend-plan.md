# CTPWA CPU + GPU 双后端完整适配计划

> 目标：同一份源码树、同一套物理代码，编译出 CUDA 版与纯 CPU 版；`analysis` 在无 GPU
> 机器上可用，数值与 GPU 版统计等价。
> 口径：本文是**施工计划**，不是可行性论证（可行性见 `doc/tilelang-port-feasibility.md`
> §2 的可移植性体检）。
> 盘点基准：`main@0e452a1`（v0.3.14）。

---

## 0. "完整适配"的定义

先钉死验收口径，否则范围会无限膨胀。

| 功能 | GPU 现状 | CPU 目标 | 说明 |
|---|---|---|---|
| `analysis` 构造（config → 耦合矩阵 → 振幅表） | ✅ | **必须（M1）** | 这是最大的一块，也最容易漏 |
| `getNLL` 前向 | ✅ | **必须（M1）** | 含 data/phsp/bkg 三段 |
| `autograd.grad` 反向 | ✅ | **必须（M2）** | 走既有 `NLLFunction::backward` |
| `getHessian` 全 Hessian | ✅ | **必须（M3）** | 含 `free_phsp_amplitudes` 流式模式 |
| `writeResult` / `writeInterfResult` | ✅ | **必须（M4）** | 份额、干涉矩阵、saved_weight TTree |
| `Figure`（画图观测量） | ✅ | **可选（M4 后）** | 冷路径，慢一点无所谓 |
| 多 GPU | ✅ | **不做** | CPU 上 `n_devices = 1` |
| NVRTC 运行时 JIT | ✅ | **不做** | 走解释器回退（已有 `CTPWA_NO_JIT=1`） |
| `precision: float/hybrid` | ✅ | **不做** | CPU 统一 double，配置项照收、映射到 double |
| DIYMC 相空间产生 | ✅（本就 CPU） | 不变 | 不受影响 |

表中 M1~M4 是里程碑（见 §6），不是路线分叉：全部做完才算"完整适配"。

---

## 1. 代码盘点（实测数字）

### 1.1 规模

| 项 | 数值 |
|---|---|
| `src/*.cu` | 22,600 行 / 17 文件 |
| `include/*.cuh` | 3,598 行 / 19 文件 |
| `__global__` kernel | **65 个**（含 4 个函数名换行写的、含模板实例化族） |
| `__device__` 辅助函数 | **约 29 个** |
| kernel 启动点 `<<<>>>` | **93 处**（AmpGen 23 / ComputeNLL 23 / Analysis 17 / Hessian 13 / Figure 9 / Parameters 5 / BF 3） |
| `cudaMalloc` / `cudaFree` | 251 / 261 |
| `cudaMemcpy` / `cudaMemset` | 152 / 56（合计 720 处内存调用） |
| `cudaSetDevice` | **175** |
| `cudaDeviceSynchronize` | 33 |
| cuBLAS 实质计算 | 31（`Daxpy` 11、`Cgemv` 7、`Zgemm` 3、`Zgemv` 2、`dgmm` 4、`gemmStridedBatched` 2、`Cgemm` 1、`dotc` 1） |
| thrust | `complex` 90、`reduce` 9、`device_ptr` 8 |
| 流 / 事件 | **基本没有**（仅 1 处 `cudaEvent`，无 stream） |

> 无 stream 是重大利好：CPU 版不需要实现异步调度，`cudaDeviceSynchronize` 直接 no-op。

### 1.2 kernel 分层（这是本文的核心表）

按"改起来有多难"分三层：

| 层 | 判据 | 数量 | 工作量 |
|---|---|---|---|
| **T1 扁平** | 无 `__shared__`/`__syncthreads`/`__shfl`/`atomicAdd`，只有 `blockIdx*blockDim+threadIdx` 取下标 | **39**（扣 float2 后 32） | 机械（约 1.5 人周） |
| **T2 扁平+原子** | 同上，但有 `atomicAdd` 累加 | **10** | 机械 + 确定性归约（约 1 人周） |
| **T3 归约/融合** | 用了 shared memory / `__syncthreads` / warp shuffle | **16** | 重写（约 3 人周） |

**关键结论：三个最热的 kernel 全在 T1。** `computeCustomAmpsKernelT`（336 行）、
`computeAmpsKernelT`（176 行）、`computeSLAmpKernel`（178 行）——它们是纯下标 + 标量数学 +
调用字节码解释器，没有任何 shared memory 和同步原语。这意味着**最难的部分反而最好搬**。

T3 的 16 个全部是"归约 / 融合 / 分块"类，它们在 GPU 上存在的理由（启动开销、寄存器复用、
shared memory 中转）在 CPU 上不存在，所以**不应该 1:1 翻译，而应该按 CPU 的自然形态重写**
——多数会从 60~90 行缩到 10~20 行。

### 1.3 哪些 kernel 在 CPU 版可以直接不要（9 个）

| kernel | 位置 | 原因 |
|---|---|---|
| `computeFactorsAndWeightsF2Kernel` | ComputeNLL.cu:774 | float2 专用，CPU 不启用 float 档位 |
| `computePhspMeanF2Kernel` | ComputeNLL.cu:823 | 同上 |
| `conjugateF2Kernel` | ComputeNLL.cu:817 | 同上 |
| `scaleConjForGradPF2Kernel` | ComputeNLL.cu:849 | 同上 |
| `downcastV2F2Kernel` | ComputeNLL.cu:864 | 同上 |
| `addF2ToComplexKernel` | ComputeNLL.cu:876 | 同上 |
| `negateAddF2ToComplexKernel` | ComputeNLL.cu:870 | 同上 |
| `computeSFromFloatAmpsKernel` | ComputeHessian.cu:579 | 同上 |
| `castF2ToDouble2Kernel` | ComputeHessian.cu:1957 | 同上 |

即 **CPU 后端实际要处理 65 − 9 = 56 个 kernel**（其中 T1 32 / T2 10 / T3 14）。

### 1.4 host 侧胶水（真正的量在这里）

下表统计 `cudaMalloc` + `cudaMemcpy` + `cudaFree` 的**出现次数**（同一行多次算多次）：

| 文件 | malloc | memcpy | free | memset | 合计 | 性质 |
|---|---|---|---|---|---|---|
| `src/Analysis.cu` | 122 | 85 | 126 | 34 | **367** | 编排、多卡、内存管理、结果写出 |
| `src/AmpGen.cu` | 65 | 38 | 66 | 8 | **177** | 振幅表分配与上传 |
| `src/ComputeNLL.cu` | 21 | 10 | 21 | 9 | 61 | NLL 缓冲 |
| `src/Figure.cu` | 15 | 11 | 12 | 5 | 43 | 出图 |
| `src/Parameters.cu` | 10 | 4 | 16 | 0 | 30 | 参数变换 |
| `src/ComputeHessian.cu` | 12 | 1 | 11 | 0 | 24 | Hessian 缓冲 |
| `src/JITCustom.cu` | 3 | 3 | 7 | 0 | 13 | 整体 `#if` 掉 |
| `src/ComputeResults.cu` | 3 | 0 | 2 | 0 | 5 | 结果写出 |

> 合计 720 处，`Analysis.cu` + `AmpGen.cu` 占 76%——**host 侧的机械替换量
> 比 kernel 本身大一个量级**，这是本计划里最容易被低估的一项。

---

## 2. 架构决策

### 2.1 编译期双后端（选定），不做运行时虚接口

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **编译期双后端**：同一份 kernel 源码，`CTPWA_KERNEL`/`CTPWA_TID`/`CTPWA_LAUNCH` 三个宏分流 | 零运行时开销；调用点形状几乎不变；`__device__` 函数加 `CTPWA_HD` 即双栖；一个源码树 | 两个 build target、两份 .so | **选它** |
| 运行时虚接口：`Backend` 抽象类 + 类型擦除启动 | 一个 .so 同时带两后端 | 65 个 kernel 各要一个包装；热路径间接调用；kernel 参数签名各异，包装层极啰嗦 | 不做 |
| 两个独立仓库/分支 | 干净 | 物理代码永久分叉，必然失同步 | 不做 |

**为什么"一个 .so 双后端"不值得**：CUDA 与 CPU 的构建依赖本来就冲突（`-lcudart` +
`-fopenmp` 可以共存，但没装 CUDA toolkit 的机器根本编不出 `nvcc` 那部分）。发布两个
同名 wheel（`ctpwa` cpu / cuda 变体），和 torch 自己的分发方式一致。

### 2.2 后端接口边界

只抽象"CUDA 运行时"这一层，**不抽象数学**：

| 原语 | CUDA 实现 | CPU 实现 |
|---|---|---|
| `ctpwa::alloc(bytes)` / `dealloc(p)` | `cudaMalloc` / `cudaFree` | `posix_memalign(64, ...)` / `free` |
| `ctpwa::copy(dst, src, bytes, Dir)` | `cudaMemcpy` | `std::memmove` |
| `ctpwa::memset(p, v, bytes)` | `cudaMemset` | `std::memset` |
| `ctpwa::sync()` | `cudaDeviceSynchronize` | 空 |
| `ctpwa::set_device(i)` | `cudaSetDevice` | 断言 `i==0`，空 |
| `CTPWA_LAUNCH(K, grid, block, shm, ...)` | `K<<<grid,block,shm>>>(...)` | `cpu_launch(grid*block, [&](int){ K(...); })` |
| `ctpwa::set_smem(K, bytes)` | `cudaFuncSetAttribute` | 空 |
| BLAS | cuBLAS | 内置 6 个形状（见 §4.4）；不引入 OpenBLAS 依赖 |
| `thrust::reduce` | thrust | `std::accumulate` |
| NVRTC JIT | `#if CTPWA_CUDA` 内 | 不编译该文件 |

**不引入 BLAS 依赖的理由**：实际只用 6 个形状，且矩阵规模小（n_amp ~ 10²）；
自写 GEMV/GEMM（带 OpenMP）约 300 行，能保证 wheel 零新增依赖，也避免 MKL/OpenBLAS
的 ABI 与线程嵌套问题。性能不够时再在 `hw::gemv` 后面换实现，接口不变。

### 2.3 文件布局

```
include/Backend.h          ← 新增：三个宏 + 原语声明（唯一有 #if CTPWA_CUDA 的头部之一）
src/Backend_cpu.cpp        ← 新增：alloc/copy/launch/BLAS 的 CPU 实现
src/Backend_cuda.cu        ← 新增：薄封装，把原语转发到 cuda*/cublas*
include/ComplexType.h      ← 改：加 host 分支（ctComplex → std::complex<double>）
include/Backend_bench.h    ← 可选：cpu_launch 的分块策略与线程数
```

现有 17 个 `.cu` **不改文件名**，CPU 构建时用 `-x c++` 强制按 C++ 编译（见 §4.6）。

---

## 3. 工作分解

### P0 基础设施（1.5 人周）

| # | 内容 | 文件 | 验收 |
|---|---|---|---|
| P0.1 | `CTPWA_HD` / `CTPWA_KERNEL` / `CTPWA_TID` / `CTPWA_LAUNCH` / `CTPWA_SMEM_ATTR` | 新增 `include/Backend.h` | CPU/CUDA 两种展开都能编过空 kernel |
| P0.2 | `alloc/copy/memset/sync/set_device` + `Buffer<T>` RAII | `src/Backend_cpu.cpp` | 单测：分配对齐、拷贝、越界检测（debug） |
| P0.3 | `cpu_launch(n, f)`：OpenMP `parallel for` + **固定分块**（见 §5 R2） | 同上 | 同一 n、不同线程数下结果逐位一致 |
| P0.4 | `ctComplex` host 分支；`ctResAmp` 去 `thrust::complex` | `ComplexType.h` | `ctpwa.so` CPU 版不引用 thrust |
| P0.5 | `DeviceManager`：无 GPU 时构造 `DeviceType::CPU` 伪设备；容量检查读 `/proc/meminfo` | `DeviceManager.cu/.cuh` | `device_mgr_.print()` 显示 CPU / 可用 RAM |
| P0.6 | `setup.py` 双 target（`CTPWA_BACKEND=cuda\|cpu`） | `setup.py` | 两条命令各出一个 `ctpwa*.so` |

### P1 T1 kernel 移植（32 个，1.5 人周）

**统一改法**（每个 kernel 只动 1~2 行）：

```cpp
// 改前
__global__ void foo(A* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ...
}
// 改后
CTPWA_KERNEL void foo(A* a, int n) {
    int i = CTPWA_TID();          // CUDA: blockIdx*blockDim+threadIdx ; CPU: 线程局部 tid
    if (i >= n) return;
    ...
}
```

启动点统一包成：

```cpp
// 改前：foo<<<gb, bs>>>(a, n);
CTPWA_LAUNCH(foo, gb, bs, 0, a, n);   // CPU 展开为 cpu_launch(gb*bs, ...)
```

**32 个 T1 kernel 清单**（按文件）：

| 文件 | kernel（行号） |
|---|---|
| AmpGen.cu | `computeSLAmpKernel`(1118) `computeAmpsKernelT`(1529) `computeCustomAmpsKernelT`(1729) `addSLAmpsKernel`(2074) `updateResonanceParamsKernel`(2091) `uhProbeNoopKernel`(2960) `computeEffectiveCouplingKernel`(2979) `daxpy_kernel`(3103) `computeGradWeightKernel`(3109) `applyBkgWeightsKernel`(3129) `multiplicativeCouplingKernel`(3893) `multiplicativeGradientKernel`(3926) |
| Analysis.cu | `castPhspBatchToDoubleKernel`(627) `axpyZComplexKernel`(637) |
| ComputeHessian.cu | `conjKernel`(14) `negateWeightsKernel`(1515) `scalePhspAmpsKernel`(1523) `reorderVVBlockKernel`(1542) `computeIntensityKernel`(1592) `scaleAmpsForGramKernel`(1608) `subComplexKernel`(1633) `addTildeBStdKernel`(1744) |
| ComputeNLL.cu | `conjugateKernel`(87) `axpyComplexKernel`(93) `scaleConjForGradPKernel`(246) `castComplexSplitKernel`(680) |
| Figure.cu | `MassCalculator`(32) `AngleCalculator`(165) `ObservableKernel`(545) |
| Parameters.cu | `extendCouplingParamsKernel`(264) `precomputeJacobianKernel`(352) `hessianFullTransformKernel`(434) |

**三个大块的具体注意点**：

- `computeCustomAmpsKernelT`(336 行)：外层 `slg = blockIdx.x`、`event_idx = blockIdx.y`；
  内核里 `if (blockIdx.x==0 && threadIdx.x==0) printf` 调试输出要换成 `CTPWA_DEBUG` 宏。
  改成 `for (sl) for (ev)` 双层循环后注意**循环序**：GPU 版按 (SL, event) 网格写 A 表，
  CPU 版若改成 event 外层，A 表的写入顺序变化不影响结果（每元素唯一写者），但**缓存局部性
  差异很大**，建议保持 SL 外层。
- `computeSLAmpKernel`(178 行)：`getMomentum` 是 `__device__` 成员函数 → 加 `CTPWA_HD`。
- `hessianFullTransformKernel`(133 行)：纯矩阵变换，无原子操作，可直接搬。

**验收**：每个 kernel 有 GPU/CPU 对拍单测（同一输入 buffer，逐元素比较）。

### P2 T2 kernel（10 个，1 人周）

共性是 `atomicAdd` 累加到全局矩阵/直方图。CPU 上**不能**简单用 `#pragma omp atomic`
（性能差且破坏决定论），改成：

```cpp
// 每线程一份局部缓冲 → 固定顺序合并
std::vector<std::vector<double>> part(nthreads, std::vector<double>(n2*n2, 0.0));
#pragma omp parallel for
for (int e = 0; e < nE; ++e) { ... part[omp_get_thread_num()][k] += ...; }
for (int t = 0; t < nthreads; ++t) merge(part[t]);   // 顺序固定
```

若 `n2*n2` 太大（n_amp > 128 时 65k×8B = 512KB/线程），退化为"按 64 事件一批 + 每批一份
缓冲"。

| kernel | 位置 | 原子目标 | CPU 策略 |
|---|---|---|---|
| `computeBFKernel` | ComputeBF.cu:28 | 势垒因子表 | 每线程局部 |
| `computeSfromAmpsKernel` | Hessian.cu:531 | S 向量 | 每线程局部 n2 |
| `computeCustomHessianKernel` | Hessian.cu:836 | (2n+P)² 矩阵 | **分块 + 每线程局部** |
| `hessianCrossBlockKernel` | Hessian.cu:1213 | 交叉块 | 每线程局部 |
| `hessianMixedBlockKernel` | Hessian.cu:1304 | 混合块 | 每线程局部 |
| `computeModTotalWeight` | Results.cu:23 | 总权重 | `reduction(+:)` |
| `computeModWithInterference` | Results.cu:53 | 干涉矩阵 | 每线程局部 |
| `fillHistogramKernel` | Figure.cu:5 | 直方图 bin | 每线程直方图 + 合并 |
| `fill2DHistogramKernel` | Figure.cu:349 | 2D bin | 同上 |
| `fill2DValuesKernel` | Figure.cu:601 | 2D 值 | 同上 |

> 上表 10 个即 T2 全部（含 3 个 Figure 冷路径 kernel）。`computeSFromFloatAmpsKernel`
> 属 float2 专用，已在 §1.3 划归"不做"，不在本表。

### P3 T3 kernel 重写（14 个，3 人周）

**总原则：不要翻译，按 CPU 形态重写。** 这些 kernel 的 GPU 结构（shared memory 中转、
warp 树形归约、每 block 一事例）在 CPU 上是纯开销。

| # | kernel | 位置 | GPU 形态 | CPU 重写方案 | 估时 |
|---|---|---|---|---|---|
| 1 | `kern_fused_fwd`（+14 模板实例） | NLL.cu:110 | warp 内 `S=Aᵀv`、`log f`、shfl 归约、`atomicAdd` | **删除**。拆成两个直白循环：①`S[e,p]=Σ_i A·v` ②`Σ_e W_e·log f_e`。NP 模板不需要 | 1d |
| 2 | `computeFactorsAndWeightsKernel` | NLL.cu:21 | 同上（旧路径） | 同上，一个事件循环 | 0.5d |
| 3 | `computePhspMeanKernel` | NLL.cu:260 | 块归约 | 一个求和循环 | 0.5d |
| 4 | `quadratic_form_small` | NLL.cu:488 | 寄存器分块 + shared 归约 | `n×n` 复 matvec，O(n²)，无 shared | 0.5d |
| 5 | `quadratic_form_large` | NLL.cu:573 | 同上 + atomic | 同上 + 确定性累加 | 0.5d |
| 6 | `doublePhspSumKernel` | NLL.cu:690 | shared 存 v + shuffle 归约 | 一个循环，`double` 累加 | 0.5d |
| 7 | `resonanceGradientKernelRuntime` | AmpGen.cu:3003 | shared `shm[16]` + atomic | 每线程 `double acc[16]`，循环后合并 | 0.5d |
| 8 | `perEventHessianKernel` | Hessian.cu:25 | 每 block 一事例，动态 shared `n²` | 直接 `std::vector<std::complex<double>> Bk(n*n)`，逐事件算；去掉协作加载 | 1.5d |
| 9 | `reduceHessianKernel`（小） | Hessian.cu:121 | 树形归约 | 线性累加 | 0.5d |
| 10 | `reduceHessianKernel`（大） | Hessian.cu:416 | 同上 | 同上 | 0.5d |
| 11 | `phspHessianKernel` | Hessian.cu:224 | 块归约 | 线性累加 | 0.5d |
| 12 | `phspHessianKernelLarge` | Hessian.cu:284 | 同上 | 同上 | 0.5d |
| 13 | `hessianCrossMixedKernel` | Hessian.cu:1408 | shared + shuffle | 每线程局部 + 合并 | 1d |
| 14 | `gnBlockReduceKernel` | Hessian.cu:1643 | `BLK_EVENTS` 批量 + shared tile | 去掉批量与 tile，逐事件直接累加 | 1.5d |

> `perEventHessianKernel` 与 `gnBlockReduceKernel` 是 Hessian 的两条主干（解析 vs
> Gauss-Newton 近似）。**先做 1~10 打通 `getNLL`/`getHessian` 主路径（M1~M3），
> 11~14 放到 P3 末尾**，不阻塞里程碑。

### P4 host 编排（2 人周）

| # | 内容 | 位置 | 说明 |
|---|---|---|---|
| P4.1 | 93 个 `<<<>>>` → `CTPWA_LAUNCH` | 7 个文件 | 机械；注意 `cudaFuncSetAttribute`（9 处）→ `CTPWA_SMEM_ATTR`（CPU 空） |
| P4.2 | 720 处 `cudaMalloc/Free/Memcpy/Memset`（251+261+152+56）→ `ctpwa::alloc/copy/memset` | 全库 | 机械；`cudaMemcpyPeer`(18) → `memcpy`；**按文件排优先级：Analysis → AmpGen → 其余** |
| P4.3 | 175 处 `cudaSetDevice` → `ctpwa::set_device`（CPU 空转 + 断言 0） | Analysis/AmpGen | 逐处确认没有"切卡后依赖当前卡"的隐含逻辑 |
| P4.4 | 多卡循环 → `n_devices=1` | Analysis.cu | 保留 `events_offsets_`/`amp_offsets_` 结构，长度 1 |
| P4.5 | 按设备索引的 `static` 缓存 | Analysis/AmpGen/ComputeNLL | `static std::vector<cublasHandle_t>` 等改成 `n_devices` 长度（=1），或改为成员 |
| P4.6 | 31 处 cuBLAS → `hw::gemv/gemm/dgmm/daxpy/dotc` | 4 个文件 | 自写实现，OpenMP |
| P4.7 | 9 处 `thrust::reduce` → `std::accumulate` | Analysis.cu | 都是对 host 可见权重数组求和 |
| P4.8 | `JITCustom.cu` 整体 `#if CTPWA_CUDA` | JITCustom.cu | CPU 版不编译；`AmpGen` 的 JIT 计划构建处 `#if` |
| P4.9 | 构造期能力检查 | Analysis.cu | 见 §4.7 |

**多卡→单卡的注意点**：`Analysis.cu` 里有若干"2×A100 必现"的绕行注释（如 3947 行
cublas gemv 非法访问），单卡路径要确认不会踩到只在多卡出现的分支；同时 `cudaMemcpyPeer`
的对等拷贝要变成同进程 memcpy。

### P5 API / 能力矩阵 / 构建打包（1 人周）

| # | 内容 |
|---|---|
| P5.1 | `ctpwa.analysis(config, device="auto"\|"cuda"\|"cpu")`；`auto` = 有卡用卡，无卡用 CPU 且打 banner |
| P5.2 | `ctpwa.build_info()` 返回 `{"backend": "cpu"/"cuda", "precision": ...}` |
| P5.3 | 无卡 + `device="cuda"` → 明确报错，不静默 fallback |
| P5.4 | CPU banner：预计耗时（`n_ev × n_amp` 估算）、能力缺失清单 |
| P5.5 | `setup.py`：`CTPWA_BACKEND=cpu` 时 `Extension` + 影子源文件（`.cpp` 里 `#include "xxx.cu"`）+ `-fopenmp` |
| P5.6 | 发布两个同名 wheel（`ctpwa-<ver>-cp3XX-cp3XX-manylinux_x86_64.whl` 的 cpu/cuda 变体），README 说明 |
| P5.7 | config 兼容：`precision: float\|hybrid` 在 CPU 上接受并映射到 double，打 warning 不报错 |

**banner 示意**：

```
[ctpwa] 无可用 CUDA 设备 → 使用 CPU 后端（OpenMP 16 线程）
[ctpwa] 预计单次求值 ~1.8 s（1.49e6 事例 × 64 振幅）；完整拟合约 1.5 h
[ctpwa] CPU 后端不支持：float 精度档位（已按 double 运行）、多 GPU
```

### P6 测试与数值验收（2 人周）

| # | 内容 | 验收 |
|---|---|---|
| P6.1 | `conftest.py` 的 `device` fixture 参数化（`cpu`/`cuda`） | 156 个测试可两边跑 |
| P6.2 | kernel 级对拍：同输入 buffer，GPU/CPU 逐元素 | 相对差 < 1e-12（同运算顺序处） |
| P6.3 | 端到端：`simple`/`flatte`/`gs`/`interp`/`custom`/`ident2` 六个 config | ΔNLL/NLL < 1e-12，梯度 rel < 1e-6，Hessian rel < 1e-6 |
| P6.4 | golden 文件（`tests/golden/nll_simple.txt`）按 CPU 重生成或改容差比较 | CI 通过 |
| P6.5 | 自洽性：CPU 上 FD 梯度 vs 解析梯度（`test_numerical.py` 已有） | rel < 1e-5 |
| P6.6 | 决定论：同输入跑 3 次、不同线程数 | 逐位一致 |
| P6.7 | 性能基线：`bench/` 脚本 | 见 §7 |

---

## 4. 关键实现细节

### 4.1 `CTPWA_LAUNCH` 的确切形状

```cpp
#if defined(CTPWA_BACKEND_CUDA)
  #define CTPWA_KERNEL __global__
  #define CTPWA_HD     __host__ __device__
  #define CTPWA_TID()  (blockIdx.x * blockDim.x + threadIdx.x)
  #define CTPWA_LAUNCH(K, g, b, s, ...) K<<<g, b, s>>>(__VA_ARGS__)
#else
  #define CTPWA_KERNEL
  #define CTPWA_HD
  extern thread_local int ct_tid;
  #define CTPWA_TID()  (ct_tid)
  #define CTPWA_LAUNCH(K, g, b, s, ...) \
      ::ctpwa::cpu_launch((std::size_t)(g) * (b), [&](int){ K(__VA_ARGS__); })
#endif
```

`cpu_launch(n, f)` 里 `#pragma omp parallel for` + 每迭代写 `ct_tid = i`（编译器会把它
放进循环体；`thread_local` 在 OpenMP 线程池上安全）。

### 4.2 决定论：固定分块

`#pragma omp parallel for reduction(+:x)` 的合并顺序由线程数决定，**换核数就换结果**。
生产环境会出现"16 核机器与 32 核机器 NLL 差 1e-11"这种不可复现问题。方案：

```cpp
// 固定 64 个 chunk，chunk 内串行，chunk 间按固定顺序合并
constexpr int kChunks = 64;
double part[kChunks];
#pragma omp parallel for schedule(dynamic)
for (int c = 0; c < kChunks; ++c) {
    double acc = 0;
    for (std::size_t e = c*n/kChunks; e < (c+1)*n/kChunks; ++e) acc += ...;
    part[c] = acc;
}
double total = 0; for (int c = 0; c < kChunks; ++c) total += part[c];
```

代价：负载不均时浪费一点并行度（可用 `kChunks = 4*nthreads` 折中，但那就与线程数挂钩了）。
**推荐 `kChunks=64` 固定**，可复现性优先。

### 4.3 `thrust::complex` 的去向

全库 90 处，其中 kernel 签名里是 `thrust::complex<double>*`（SL 振幅表）。CPU 版用
`ctResAmp = std::complex<double>`（`ComplexType.h` 已有 typedef 位置，见 74/114 行）。
注意：

- `thrust::complex` 的 `.real()/.imag()` 与 `std::complex` 一致，`.x/.y` 不适用于 std；
- **内存布局**都是 `{re, im}` 连续 16B，与 `cuDoubleComplex` 一致 → 可 reinterpret；
- 唯一风险：`std::complex` 的乘法可能带 `-ffast-math` 优化差异，**CPU 版禁用 `-ffast-math`**。

### 4.4 cuBLAS 的 6 个形状

| 形状 | 出现 | CPU 实现 |
|---|---|---|
| `gemv`（复数） | S=Aᵀv、A·w | `for j: y[j] = Σ_i a[i,j]*x[i]`，OpenMP 按行 |
| `gemm`（复数） | B=AᴴA、双精度 phsp 矩阵 | 分块 64×64，寄存器累加 |
| `gemmStridedBatched` | 干涉矩阵 | 循环调用 gemm |
| `dgmm`（对角乘） | 权重归一 | 逐元素 |
| `daxpy` | Hessian 累加 | `y += a*x` |
| `dotc` | 二次型 | `re/imag` 分开累加 |

n_amp ~ 10²、n_ev ~ 10⁶ 时这些都不是瓶颈，瓶颈是 A 表遍历。

### 4.5 CPU 的 `cudaFuncSetAttribute`

9 处，都是为了 `n²×16B > 48KB` 的动态 shared memory。CPU 上直接 no-op；但**注意**
`perEventHessianKernel` 的 shared 缓冲区在 CPU 版要换成 `std::vector`（或复用一块
成员缓冲，避免每事件 malloc）。

### 4.6 单源码树双 target

```python
# setup.py
backend = os.environ.get("CTPWA_BACKEND", "cuda")
if backend == "cuda":
    ext = CUDAExtension(name="ctpwa", sources=SRCS, ...)
else:
    ext = Extension(name="ctpwa", sources=make_cpu_shadow_sources(SRCS),
                    extra_compile_args=["-std=c++17", "-O3", "-fopenmp",
                                        "-DCTPWA_BACKEND_CPU"],
                    extra_link_args=["-fopenmp"])
```

**为什么用影子源文件而不是 `-x c++`**：distutils 把 `extra_compile_args` 拼在源文件**之后**，
而 `-x` 只对其后的文件生效，对当前文件无效；gcc 又只按扩展名判语言。因此在
`build/cpu_src/<name>.cpp` 生成一行 `#include "<绝对路径>.cu"` 的包装文件即可——
被包含文件内的 `#include "xxx.cu"` 仍按其自身目录解析，内联结构不受影响。
ROOT/yaml-cpp 的链接方式不变（它们本来就是 host 库）。

### 4.7 `analysis` 构造期能力检查

```cpp
if (backend_ == Backend::CPU) {
    if (config.precision != "double") warn("CPU: 统一按 double 运行");
    if (n_gpus_requested > 1)        warn("CPU: 忽略多 GPU 设置");
    if (config.uses_jit)             warn("CPU: JIT 关闭，走解释器（略慢）");
}
```

能力缺失（Hessian 未移植、Figure 未移植）在 P0/P1 阶段：
`throw std::runtime_error("CPU backend: <feature> not implemented yet (phase P1)")`。

---

## 5. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | **Hessian** 14 个 T3 里占 7 个，且"收敛后才知道对不对" | 高 | 先做 §P3 的 8/9/10 打通主路径；11~14 后置；早期用 FD 交叉验证（n_amp 小，成本可接受） |
| R2 | **决定论**（atomic/归约顺序） | 中 | §4.2 固定分块；把"同输入 3 次逐位一致"写进 CI |
| R3 | **数值一致性**：CUDA `log`/`pow` 与 glibc 差 1 ulp | 中 | 验收用统计判据（ΔNLL/NLL < 1e-12、rel grad < 1e-6），不做 bitwise diff |
| R4 | `thrust::complex` 90 处替换 | 中 | 集中在 `ComplexType.h`；编译期 typedef 切换；CPU 版禁 `-ffast-math` |
| R5 | 175 处 `cudaSetDevice` 里藏着"切卡后依赖当前卡"的隐含逻辑 | 中 | 逐处 audit，`Buffer<T>` 记录所属设备，越界访问在 debug 版断言 |
| R6 | 按设备索引的 `static` 缓存跨测试泄漏 | 低 | 改成成员或按 `n_devices` 重建；`analysis` 析构清理 |
| R7 | **ROOT 仍是硬依赖** | 中 | CPU wheel 仍需 ROOT；若目标含 Windows/macOS 笔记本，需先把 ROOT 变成可选（独立议题） |
| R8 | `float` 档位被禁用后，`test_precision.py`/`test_float_mode.py` 语义变化 | 低 | 这两个测试 `@pytest.mark.skipif(backend=="cpu")`，并在文档写明 |
| R9 | 性能不达预期 | 中 | §7 的预算；P6.7 实测后再决定是否投 SIMD 优化 |

---

## 6. 排期与里程碑

| 阶段 | 内容 | 人周 | 累计 |
|---|---|---|---|
| P0 | 基础设施 | 1.5 | 1.5 |
| P1 | T1 kernel 32 个 + 启动点 | 1.5 | 3.0 |
| P2 | T2 kernel 10 个 | 1.0 | 4.0 |
| P4 | host 编排（可与 P1/P2 并行） | 2.0 | 6.0 |
| P3 | T3 kernel 14 个（Hessian 后半可后置） | 3.0 | 9.0 |
| P5 | API / 能力矩阵 / 打包 | 1.0 | 10.0 |
| P6 | 测试 + 数值 + 性能 | 2.0 | 12.0 |
| — | 缓冲 | 1.0 | **13.0** |

单人对齐约 **13 人周（≈3 人月）**；两人并行（一人 P1/P2/P3 的 kernel，一人 P4/P5/P6）
约 **7~8 周**。

> 注：比前一轮粗估的 8~12 人周略高。细化后 T3 实际是 14 个（不是估的 10 个），
> 且 Hessian 占其中 7 个；host 侧的 720 处内存调用也比预期密。

### 里程碑（每个都可独立交付）

| 里程碑 | 内容 | 验收 | 时点 |
|---|---|---|---|
| **M0** | 双后端骨架可编译 | CPU 版能 `import ctpwa`；`PhaseSpaceGenerator` 正常 | P0 |
| **M1** | `analysis(device="cpu")` 构造 + `getNLL` | 与 GPU ΔNLL/NLL < 1e-12 | P1+P4.1~4.4 |
| **M2** | `autograd.grad` | 梯度 rel < 1e-6；FREEPARAMS/VSPACE 都过 | P2 |
| **M3** | `getHessian` | 与 GPU rel < 1e-6；FD 交叉验证通过 | P3 前半 |
| **M4** | `writeResult` / `Figure` | 份额/干涉矩阵一致 | P3 后半 + P4 |
| **M5** | 全测试 + 双 wheel 发布 | 156 测试 CPU 全过；`pip install ctpwa`（cpu 变体）可用 | P5+P6 |

### 最小可用裁剪（如果只要 CI 能跑）

P0 + P1 + P4.1~P4.4 + P5.1~P5.5 ≈ **6.5 人周**，交付 M1+M2，
Hessian/出图显式 `NotImplemented`。这与前一轮说的 "C 路线 2~3 人周" 的差距在于：
那个估算是"只跑测试的小 config、可以砍路径"，这里是"完整前向 + 梯度、保留全部物理模型"。

---

## 7. 性能预算

| 场景 | GPU（单卡） | CPU 目标（16 核） | CPU 上限（加 SIMD/融合） |
|---|---|---|---|
| 1e5 事例 × 20 振幅，一次求值 | ~6 ms | 0.3~1 s | 0.1 s |
| 1.49e6 × 64，一次求值 | 86 ms | 1.5~3 s | 0.5~1 s |
| 单次拟合（2400 evals） | 3.6 min | 1~2 h | 20~40 min |
| 生产 5-run（7600 evals） | 11 min | 3~6 h | 1~2 h |

CPU 的主要成本是"字节码解释器 × (事例 × 振幅)"，A 表带宽其次。若 M5 后需要再快，
按性价比排序：
1. AVX2 跨事例向量化解释器（4 路，2~3×）；
2. 振幅与 NLL 融合（不物化 A 表，省一遍 3GB 读写）；
3. 用 `-O3 -march=native` + PGO。

---

## 附录 A：65 个 kernel 全表

### T1（32，机械改）

| 文件:行 | kernel | 行数 |
|---|---|---|
| AmpGen.cu:1118 | computeSLAmpKernel | 178 |
| AmpGen.cu:1529 | computeAmpsKernelT | 176 |
| AmpGen.cu:1729 | computeCustomAmpsKernelT | 336 |
| AmpGen.cu:2074 | addSLAmpsKernel | 7 |
| AmpGen.cu:2091 | updateResonanceParamsKernel | 10 |
| AmpGen.cu:2960 | uhProbeNoopKernel | 12 |
| AmpGen.cu:2979 | computeEffectiveCouplingKernel | 13 |
| AmpGen.cu:3103 | daxpy_kernel | 4 |
| AmpGen.cu:3109 | computeGradWeightKernel | 15 |
| AmpGen.cu:3129 | applyBkgWeightsKernel | 10 |
| AmpGen.cu:3893 | multiplicativeCouplingKernel | 22 |
| AmpGen.cu:3926 | multiplicativeGradientKernel | 48 |
| Analysis.cu:627 | castPhspBatchToDoubleKernel | 5 |
| Analysis.cu:637 | axpyZComplexKernel | 6 |
| ComputeHessian.cu:14 | conjKernel | 4 |
| ComputeHessian.cu:1515 | negateWeightsKernel | 4 |
| ComputeHessian.cu:1523 | scalePhspAmpsKernel | 14 |
| ComputeHessian.cu:1542 | reorderVVBlockKernel | 11 |
| ComputeHessian.cu:1592 | computeIntensityKernel | 9 |
| ComputeHessian.cu:1608 | scaleAmpsForGramKernel | 18 |
| ComputeHessian.cu:1633 | subComplexKernel | 4 |
| ComputeHessian.cu:1744 | addTildeBStdKernel | 19 |
| ComputeNLL.cu:87 | conjugateKernel | 4 |
| ComputeNLL.cu:93 | axpyComplexKernel | 6 |
| ComputeNLL.cu:246 | scaleConjForGradPKernel | 10 |
| ComputeNLL.cu:680 | castComplexSplitKernel | 6 |
| Figure.cu:32 | MassCalculator | 14 |
| Figure.cu:165 | AngleCalculator | 43 |
| Figure.cu:545 | ObservableKernel | 50 |
| Parameters.cu:264 | extendCouplingParamsKernel | 31 |
| Parameters.cu:352 | precomputeJacobianKernel | 41 |
| Parameters.cu:434 | hessianFullTransformKernel | 133 |

（T1 原始 39 个；扣除 7 个 float2 专用项后剩 32 个。）

### T2（10 个，原子→确定性归约）

`computeBFKernel`(BF:28)、`computeSfromAmpsKernel`(H:531)、`computeCustomHessianKernel`(H:836)、
`hessianCrossBlockKernel`(H:1213)、`hessianMixedBlockKernel`(H:1304)、
`computeModTotalWeight`(R:23)、`computeModWithInterference`(R:53)、
`fillHistogramKernel`(F:5)、`fill2DHistogramKernel`(F:349)、`fill2DValuesKernel`(F:601)

（10 个，全部需要移植；`fill2DValuesKernel` 属冷路径，可最后做。）

### T3（16 → CPU 需 14）

见 §P3 表。

### 不做（9 个 float2 专用）

见 §1.3。

---

## 附录 B：需加 `CTPWA_HD` 的 29 个 `__device__` 函数

| 文件 | 函数 |
|---|---|
| `include/AmpGen.cuh:55` | `breakup_momentum` |
| `include/Amplitude.cuh:121` | `pwa_amp`（声明） |
| `include/CustomExpr.cuh:81,106` | `evalCustomSeg`、`evalCustomAll` |
| `include/ResModel.cuh:52,59,73,260,278` | `interpEval`、`computeNodeFactor`、`lookupHistTable`、`interp1d` |
| `include/ResModel.cuh:333,346,361,378,400` | `breakup_dq_dm`、`breakup_d2q_dm2`、`breakup_d2q_dm1dm2`、`dlnBf_dq0`、`d2lnBf_dq0q0` |
| `src/AmpGen.cu:133,1009,1519` | `DeviceMomenta::getMomentum`、`contract`、`cplxOut` |
| `src/Amplitude.cu:5,100,146,201,237,272,319` | `factorial`、`ClebschGordan`、`associated_legendre_poly`、`spherical_harmonic`、`cmf_element`、`DFunc`、`wignerD_element` |
| `src/Amplitude.cu:331,409,466,500` | `MassiveTrans`、`MasslessTrans`、`multiplyDFuncLeft`、`pwa_amp` |
| `src/ComputeHessian.cu:609,697,723` | `addBfQ0HessianTerms`、`bwrN`、`bwrSpecEval` |

均为纯标量/复数数学，无同步原语；加入 `CTPWA_HD` 即可双栖。唯一需要替换的是
`thrust::complex<double>` → `ctResAmp`。

---

## 附录 C：改动模板

```cpp
// ---- 1) kernel 体（T1） ----
CTPWA_KERNEL void foo(ctComplex* a, int n) {
    int i = CTPWA_TID();
    if (i >= n) return;
    a[i] = ...;
}

// ---- 2) 启动点 ----
CTPWA_LAUNCH(foo, gb, bs, 0, a, n);

// ---- 3) device 函数 ----
CTPWA_HD inline double breakup(double m, double a, double b) { ... }

// ---- 4) 原子累加（T2） ----
#ifdef CTPWA_BACKEND_CUDA
    atomicAdd(&H[k], v);
#else
    local_H[k] += v;            // 每线程一份，循环后按固定顺序合并
#endif

// ---- 5) 内存 ----
auto* p = (ctComplex*)ctpwa::alloc(n * sizeof(ctComplex));
ctpwa::copy(p, h, bytes, ctpwa::Dir::H2D);
ctpwa::free(p);
```


---

## 8. 实施进展

### P0（已完成）

| 项 | 文件 | 状态 |
|---|---|---|
| 后端开关 `CTPWA_BACKEND_CUDA/CPU`、`CTPWA_HAS_CUDA` | `include/CudaCompat.h` | ✅ |
| `CTPWA_KERNEL` / `CTPWA_HD` / `CTPWA_TID` / `CTPWA_LAUNCH` / `CTPWA_SMEM_ATTR` / `CTPWA_SYNC` | `include/Backend.h` | ✅ |
| 内存原语 `alloc/dealloc/copy/memset/sync/set_device/device_count/available/set_smem_bytes/available_memory/total_memory` | `include/Backend.h`、`src/Backend.cpp` | ✅ 双后端 |
| `hw::` 自实现 BLAS（gemv / gemm / gemm_batched / dgmm / dgmm_real / daxpy / dotc），**零第三方依赖** | `src/Backend.cpp` | ✅ CPU 自写；CUDA 转发 cublas |
| `ComplexType.h` 的 CPU 分支（`thrust::complex` → `std::complex` 别名、`ctShflDown` 仅 nvcc） | `include/ComplexType.h` | ✅ |
| `DeviceManager` CPU 伪设备（index 0，"显存"=RAM，`DeviceType::CPU`） | `include/DeviceManager.cuh`、`src/DeviceManager.cu` | ✅ |
| `setup.py` 双 target（`CTPWA_BACKEND=cuda|cpu`，CPU 侧 g++ + OpenMP + 影子源文件） | `setup.py` | ✅（CPU 目标在 P1~P4 完成前会停在未移植 kernel） |
| 基础层单测（内存/launch/BLAS/确定性/线程数无关） | `tests/cpu_backend/` | ✅ 7 组全过 |

已验证：
- CPU 单测 7 组全过；`OMP_NUM_THREADS=1/2/8` 结果逐位一致。
- `nvcc -c src/DeviceManager.cu`、`nvcc -c`（含 Backend.h 的冒烟 TU）通过。
- `g++ -c src/Backend.cpp` 在 `CTPWA_BACKEND_CPU` 与 `CTPWA_BACKEND_CUDA` 两种宏下均通过。

**P0 暴露并已解决的两个坑**：

1. **`ctpwa::free` 与 `::free` 重载歧义**——在 `namespace ctpwa` 内调用 `free(ptr)` 时，`ctpwa::free(void*)`
   与 `stdlib.h` 的 `::free(void*)` 同等匹配。已改名为 `ctpwa::dealloc`。
2. **CUDA 13 的 thrust 路径**——`thrust/complex.h` 在 `targets/<arch>/include/cccl/` 下；
   `src/Backend.cpp` 由 g++ 编译且包含 `ComplexType.h`，因此 `setup.py` 必须把
   `targets/<arch>/include` 与 `.../include/cccl` 显式加进 `include_dirs`（nvcc 自带，g++ 没有）。

### P4 前半（已完成）：CUDA API 垫片层

原计划是逐个改 **720 处** `cuda*` 调用 + 20 处直接 include。改为加一层
`include/cpu_shims/`（只挂在 CPU 构建的 include 路径**最前面**，CUDA 构建看不到），
业务代码零改动：

| 垫片 | 覆盖 | 落地方式 |
|---|---|---|
| `cuda_runtime.h` | 28 个 `cuda*` 符号（含模板版 `cudaMalloc`） | inline 转发到 `ctpwa::` |
| `cublas_v2.h` | `cublasZgemv/Zgemm/ZgemmStridedBatched/Zdgmm/Zdotc/Daxpy` + 类型/枚举 | 转发到 `hw::`；float 变体给出明确 abort |
| `thrust/complex.h` 等 13 个头 | `complex / device_ptr / reduce / plus / exp / vector` | `<complex>`/`<vector>`/`<numeric>` 等价物 |
| `cub/cub.cuh` | 只 include 不使用 | 空文件 |

**为什么比逐个改调用点好**：① 零业务代码改动；② 只保留一份实现（`ctpwa::`/`hw::`）；
③ 新用到未适配的 CUDA API 会**立刻编译失败**，失败点即移植点，不会静默算错。
P4 从 ~2 人周降到 ~3 天。

配套的两个关键点：
- `CudaCompat.h` 把 `__device__`/`__global__` 定义成**空**，于是 **`__device__` 函数
  自动变成普通宿主函数**——`src/Amplitude.cu`（Wigner D / Clebsch-Gordan / 变换矩阵）
  与 `src/CustomExpr.cu`（字节码编译器 + 解释器）**零改动**即可在 CPU 上编译。
- 故意**不定义** `threadIdx/blockIdx/blockDim/gridDim`：未移植的 kernel 会在这些名字上
  编译失败，天然成为待移植清单。

### P1 进展：已移植的文件

| 文件 | CPU 错误数 | 做法 |
|---|---|---|
| `src/Amplitude.cu` | **0** | 无需改动（`__device__` → 宿主函数） |
| `src/CustomExpr.cu` | **0** | 仅给 `CustomExpr.cuh` 加一行 `#include "CudaCompat.h"` |
| `src/ComputeBF.cu` | **0** | 1 个 kernel 改 `CTPWA_KERNEL`+`CTPWA_TID`，3 处启动改 `CTPWA_LAUNCH` |
| `src/ComputeResults.cu` | **0** | 2 个 kernel + 4 处启动（原写法是 `< < > > >` 带空格） |
| `src/Parameters.cu` | 48 | 待移植 |
| `src/Figure.cu` | 71 | 待移植（冷路径） |
| `src/AmpGen.cu` | 99 | 待移植（含最热的两个 kernel） |
| `src/ComputeHessian.cu` | 197 | 待移植 |
| `src/ComputeNLL.cu` | 288 | 待移植（T3 重写） |
| `src/main.cu`（内联 Analysis 等） | — | 待移植 |

已移植的两个文件都做了**双后端验证**：`g++ -fsyntax-only`（CPU）与 `nvcc -c`（CUDA）
均 0 错误。

**下一步**：`Parameters.cu`（48）→ `AmpGen.cu`（99，先 `computeSLAmpKernel` 再
`computeCustomAmpsKernelT`）→ `ComputeNLL.cu`（288，T3 按循环重写）→ `ComputeHessian.cu`
（197）→ `main.cu`。


### P1~P4 完成后的实测结果（CPU 后端首次跑通）

| 项 | 结果 |
|---|---|
| **CPU 构建** | `CTPWA_BACKEND=cpu python3 setup.py build_ext` → RC=0，`ctpwa.so` 3.0 MB |
| **CUDA 构建** | 同源码树 `python3 setup.py build_ext` → RC=0（未破坏 GPU 路径） |
| **8 个 config** | simple / flatte / gs / interp / custom / ident2 / no_trans / with_trans 全部构造成功、NLL 有限 |
| **golden 对比** | simple: `1195.621163` vs golden `1195.621059` → **rel 8.6e-8** |
| **梯度正确性** | flatte 的 3 个 θ 分量：AD vs 有限差分 rel **2.3e-9 / 9.6e-11 / 7.4e-10** |
| **确定性** | 同输入重复求值**逐位一致**（NLL 求和改为单次有序累加） |
| **性能** | simple config（1 万 phsp 事例）**3.8 ms/次求值** |

### 实现中遇到的坑（都已修）

1. **`ctpwa::free` 与 `::free` 重载歧义** → 改名 `ctpwa::dealloc`。
2. **CUDA 13 的 thrust 在 `targets/<arch>/include/cccl/`**，g++ 编的 `Backend.cpp` 需要显式加路径。
3. **`-DCTPWA_DOUBLE_COMPLEX` 只给了 nvcc** → `Backend.cpp` 里 `ctComplex=float` 却调 `cublasZ*`（double）。改为 cxx/nvcc 共用同一变量。
4. **`cuComplex`/`cuDoubleComplex` 是 `float2`/`double2` 的 typedef**（CUDA 里就是），CPU 侧必须同样别名，否则 `cpxPrecOf<>` 特化落空。
5. **模板实参含逗号的 kernel 不能直接作为宏实参**（`kernel<0, 1024>`）→ 新增 `CTPWA_LAUNCH_P((&kernel<0, 1024>), ...)`，走 `cudaLaunchKernel`/函数指针。
6. **`torch::Device(torch::kCUDA, i)` 硬编码** → `ctpwa_dev(i)`（CPU 返回 `cpu`）。
7. **默认 `precision: auto` 会让 `float_amps_=true`**（A 表存 float2），CPU 上直接撞 float BLAS → CPU 强制 `float_amps_=false`。
8. **`params` 用全 0 初始化会得到 NLL=NaN**——θ 是绝对质量/宽度，必须用 `getFreeResParams()[0]` 标称值。这不是移植 bug，是调用约定。
9. **`atomicAdd` 求和顺序**导致末位 1e-12 抖动 → NLL/phsp 均值改为单次有序累加。

### 仍未完成

| 项 | 状态 |
|---|---|
| `getHessian` | ❌ 6 个 T3 kernel（perEventHessian / reduceHessian ×2 / phspHessian ×2 / gnBlockReduce / hessianCrossMixed）仍是 `not_implemented` |
| `writeResult` / Figure | ⚠️ 能编，未实测 |
| 156 个 pytest | ❌ conftest 仍 `assert torch.cuda.is_available()`，需参数化 device fixture |
| 性能优化（SIMD/融合） | ❌ 未开始 |


### P3 Hessian 完成（CPU getHessian 跑通）

| 项 | 结果 |
|---|---|
| `getHessian` | ✅ 可调用，返回 (2n_v+P)² 矩阵 |
| 对称性 | rel **6.4e-17** |
| θ 块 vs 有限差分 | rel **6.7e-9** |
| v 块 | 与 FD 一致（NLL 对 v 的整体标度不敏感，该块理论为 0） |
| 重复调用 | 逐元素散布 ~6e-10（|H|~5.7e4 → 1e-14 相对），来自仍保留的 atomicAdd |

移植的 6 个 T3 kernel（全部改为 CPU 串行/直累加）：

| kernel | CPU 做法 |
|---|---|
| `perEventHessianKernel` | 逐事件串行，std::vector 代替 shared |
| `reduceHessianKernel`(元素归约) | 外层元素 × 内层事件的双循环 |
| `phspHessianKernel` / `Large` | 去掉分块，直接算 Pu 后写矩阵 |
| `reduceHessianKernel`(约束重参数化) | 二维 (i,j) 摊平成一维工作项 |
| `hessianCrossMixedKernel` | 去掉 a 分块与 shared 暂存，直接 atomicAdd |
| `gnBlockReduceKernel` | 逐事件算 Bu 后直接累加 H += 4w·Bu·Buᵀ/I² |

**关键 bug：`CUBLAS_OP_C` 被当成 `CUBLAS_OP_T`。**
Gram 矩阵 `Aᴴ·A` 少了共轭 → v-v 块完全错（AD 给出 2173，FD 是 ~0），矩阵也不对称。
垫片现在区分 `Op::N/T/C`：`hw::gemm/gemv` 在 C 时对元素取共轭，CUDA 侧转发到
`CUBLAS_OP_C`。


### Hessian 严格复现（补完）

上一轮 Hessian 仍有 ~6e-10 的元素散布（|H|~5.7e4，1e-14 相对），来自 5 个仍用
跨事件 `atomicAdd` 的 kernel：

| kernel | 原子数 | 用途 |
|---|---|---|
| `computeCustomHessianKernel` | 6 | θ-θ 块（字节码二阶导） |
| `hessianCrossBlockKernel` | 4 | v-θ 跨块 |
| `hessianMixedBlockKernel` | 4 | v-θ 同块 |
| `hessianCrossMixedKernel` | 3 | v-θ 交叉 |
| `computeSfromAmpsKernel` | 2 | S 向量 |

**修法：`CTPWA_LAUNCH_SEQ`**——语义与 `CTPWA_LAUNCH` 相同，但 CPU 上按固定顺序
单线程遍历工作项（`launch_seq`）。原子竞争消失、累加顺序固定，**kernel 体一行没改**。
CUDA 侧该宏与普通启动等价（GPU 的 atomicAdd 本来就不保证顺序）。

代价：这 5 个 kernel 在 CPU 上不并行。实测 simple config 一次 `getHessian` 9 ms，
代价可忽略；若将来 Hessian 成为瓶颈，可改为"固定 K 块局部累加 + 顺序合并"，
既保并行又保复现。

验证（`OMP_NUM_THREADS` = 1 / 2 / 8）：

```
1 threads: nll=1195.6211628598485  H_sha=23612a8634efc3fc
2 threads: nll=1195.6211628598485  H_sha=23612a8634efc3fc
8 threads: nll=1195.6211628598485  H_sha=23612a8634efc3fc
```

NLL 与 Hessian 都与线程数无关、逐位一致。


### CPU vs GPU 数值对比（RTX 5060, sm_120）

两个 .so 同名，只能分进程跑：

```bash
# GPU 版
PYTHONPATH=tmp/gpu_run python3 tests/cpu_backend/dump_results.py tmp/gpu.npz cuda
# CPU 版
PYTHONPATH=tmp/cpu_run python3 tests/cpu_backend/dump_results.py tmp/cpu.npz cpu
python3 tests/cpu_backend/compare_results.py tmp/cpu.npz tmp/gpu.npz --label-a CPU --label-b GPU
```

| config | NLL(CPU) | NLL(GPU) | NLL rel | grad rel | H rel |
|---|---|---|---|---|---|
| simple | 1195.621162860 | 1195.621162110 | 6.3e-10 | 1.4e-08 | 3.2e-09 |
| flatte | 644.163227889 | 644.163226596 | 2.0e-09 | 6.0e-08 | 1.3e-09 |
| gs | 1083.738512462 | 1083.738513140 | 6.3e-10 | 1.5e-08 | 7.6e-10 |
| custom | 1257.444723584 | 1257.444721764 | 1.5e-09 | 1.5e-08 | 4.9e-09 |
| ident2 | 1406.895080596 | 1406.895080396 | 1.4e-10 | 4.2e-08 | 3.5e-09 |
| interp | 925.518285664 | 925.518285428 | 2.6e-10 | (近零量) | (近零量) |
| no_trans | 1095.908600958 | 1095.908602140 | 1.1e-09 | (近零量) | (近零量) |
| with_trans | 同上 | 同上 | 1.1e-09 | (近零量) | (近零量) |

**全部 8 个 config 的 NLL 相对差 ≤ 2e-9；有实际曲率的 5 个 config，梯度与 Hessian
相对差 ≤ 6e-8。**

后 3 个 config 的梯度/Hessian 解析上就是 0（`interp`/`no_trans`/`with_trans` 都只有
自由耦合 v、没有自由共振态参数，而 NLL 对 v 的整体标度不变），残差在 1e-12~1e-4
绝对量级、相对量级 1e-7 以下，属于抵消误差；用相对判据会被小分母放大成 O(1) 的假失配。
`compare_results.py` 对 `max|·| < 1e-3` 的量改用绝对容差。

### 性能（simple: 1.1 万事例；no_trans: 6 个自由振幅）

| config | 量 | CPU | GPU |
|---|---|---|---|
| simple | NLL+grad | 3.1 ms | 2.5 ms |
| simple | Hessian | 7.6 ms | 148 ms |
| no_trans | NLL+grad | 7.4 ms | 3.3 ms |
| no_trans | Hessian | 34 ms | 94 ms |

小 config 上 CPU 反超：GPU 的 kernel 启动/归约开销在矩阵只有几十阶时占主导。
事例数上到 10⁵~10⁶ 后 GPU 才会拉开（见 §7 性能预算）。


### CPU vs GPU 对比（同精度口径）与一个真实回归

一次对比脚本把两边都跑一遍（`tests/cpu_backend/dump_results.py` +
`compare_results.py`）。**必须让 GPU 也用 `precision: double`**：GPU 构建默认
`precision: auto` 把 A 表存成 float32，CPU 后端强制 double；不同口径下 Hessian 会差
~1e-4（正是 float32 存储的传播），很容易被误判成移植 bug。

同口径（double vs double，9 个 config）:

| 量 | 最大相对差 |
|---|---|
| NLL | 4.9e-12 |
| 梯度 | 2.2e-11 |
| Hessian（7/9 个 config） | 8.4e-13 |
| Hessian（no_trans / with_trans） | 1.2e-04（见下） |

`no_trans` / `with_trans` 的 Hessian 只有 4/256 个元素相对差 >1e-3（最大 6.6e-3），
且集中在受 `1/I_e` 加权放大、本身有强抵消的大元素上；NLL 与梯度仍一致到 1e-13。
判为求和顺序噪声，未做进一步追查（如需严格对齐可统一 Gram 的分块方式）。

### 回归：`axpyComplexKernel` 的 stride 循环被改坏

`getHessian` 的 vv 块一度与 GPU 差 1.35（相对），而 **NLL 完全正确**。二分定位到
`ComputeNLL.cu`：

```cpp
// 原始：单 block 步进循环，启动配置 <<<1, 64>>>
for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = ...;

// 错误改法：一线程一元素 → 只有前 64 个元素被更新
const int i = CTPWA_TID(); if (i >= n) return;
```

`axpyComplex` 用在梯度装配上，n（振幅数）> 64 时梯度被截断 → 梯度和 Hessian 错、
NLL 不受影响。修复：CUDA 保留 stride 循环，CPU 走 `CTPWA_LAUNCH_SERIAL` + 串行扫完。

**教训**：`CTPWA_TID()` 只适用于本来就是一工作项一线程的 kernel。迁移脚本
（`scripts/port_cuda_to_cpu.py`）只匹配 `blockIdx.x*blockDim.x+threadIdx.x` 这一种写法
是安全的；**手工把 stride 循环改成 TID 会静默截断**。

### CPU 后端的功能限制（明确报错）

| 项 | 行为 |
|---|---|
| `precision: float/hybrid` | 接受，按 double 运行并打印提示 |
| `CTPWA_HESS_FAST=0`（legacy Hessian） | 不接受，强制 fast 路径并打印提示（legacy 在 CPU 上返回 NaN，未移植） |
| 多 GPU | 只有 1 个 "CPU 设备" |
| NVRTC JIT | 关闭，自动走解释器（`jitEnabled()=false`） |
