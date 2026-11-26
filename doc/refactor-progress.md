# 重构进度（refactor/modular）

> 主线文档：`doc/modularization-plan.md`（为什么这么拆）、
> `doc/refactor-release-workflow.md`（怎么改、怎么测、怎么发）。
> 本文只记**实际做到哪一步、怎么验证的**。

分支：`refactor/modular`（从 `main@030baf5`，v0.3.27 起）

| 阶段 | 状态 | 提交 |
|---|---|---|
| 基线 + 设备原语 | ✅ | `8671609` |
| P1 真 TU | ✅ | `dfc6d5c` |
| P0-A 还原主卡 → `ensure_device` | ✅ | `5e3ab26` |
| P0-C `primary_dev` 显式化 | ✅ | `d0eab60` |
| P0-B 多卡循环内的 set（128 处） | ⏸ **等 GPU** | — |
| P2 DevBuffer 所有权 | ✅ 第一批（AmpCasDecay 四动量） | `9e91283` |
| P2 第二批（`AmpCasDecay` 六张设备表） | ✅ | `7c808a8` |
| P3 第一步（衰变描述类型 + 置换拓扑抽出） | ✅ | `d42f203` |
| P3 第二步（`DecayChainDesc` 组合进 `AmpCasDecay`） | ✅ | `ee9178b` |
| P4 第一步（数据读取模块） | ✅ | `7cab902` |
| P2 第三批（`AmpCalc` 四个大缓冲） | ✅ | `a9dc6db` |
| 门禁补强：数值用例上 CPU | ✅ | `d1b27b0` |
| P3 DecayInfo / SL 表解耦 | ⬜ | — |
| P4 god class 拆分 | ⬜ | — |

---

## P1：6 个内联 `.cu` 变真翻译单元

**改动**：`setup.py` 的 `SOURCES_CORE` 加入 `DeviceManager/Config/PhaseSpace/Info/
ResModel/Resonance`；`main.cu` 去掉 6 条 `#include "*.cu"`（`Analysis.cu` 仍内联，
它没有头文件）。另外 `make_cpu_shadow_sources` 改为"内容不变则不重写"——它原本
每次构建都更新 shadow `.cpp` 的 mtime，导致 ninja 每次都全量重编，是增量编译的
唯一天花板。

**实测**（CPU 后端，16 核）：

| 项 | 改前 | 改后 |
|---|---|---|
| 翻译单元数 | 11（main 内含 12.8k 行） | **17** |
| 全量构建 | ~15 min | ~15 min（总工作量不变） |
| `touch src/Config.cu` 后重建 | 全量（~15 min） | **只编 Config.cpp，9.3 s** |
| 逐文件测试 | 183 passed / 8 failed / 26 skipped | **同上，逐文件完全一致** |

判据达成：改一个 host 侧文件不再重编 12.8k 行，且编译器第一次能指出越界依赖
（`main.cu` 若再引用某个已拆出 `.cu` 的私有符号会直接链接失败）。

---

## P0-A：75 处「还原主设备」→ `ctpwa::ensure_device`

`cudaSetDevice(primary_dev_)` / `(primary_dev)` 共 75 处（Analysis 57 / AmpGen 14 /
Parameters 4）。`ensure_device` 语义完全等价：**已经是主卡则不调用 `set_device`**，
只切不还原。

门禁变化：`scripts/check_device_calls.sh` 违规数 **208 → 133**。

---

## P0-C：`primary_dev` 变显式形参

`AmpCalc::computeResonanceGradient` / `computeUnifiedHessian` 里各有一处
`cudaGetDevice(&primary_dev)`——它把"主设备"定义为**调用那一刻的 ambient 设备**，
`torch.to()` 或多卡循环一改就会错卡（`bc22b52` 同类根因）。现在改成第一个形参，
Analysis.cu 的 6 个调用点显式传 `primary_dev_`（局部作用域里是 `primary_dev`）。

---

## P2（第一批）：`AmpCasDecay` 的四动量缓冲改为单一所有权

**问题**（`modularization-plan §2.4b`）：同一批设备指针同时登记在
`d_mom_sigma_` 与 `d_momenta_structs_/d_momenta_arrays_` 两处。`releaseStaticData`
只 free 一处，析构会对同一指针再 free 一次（CPU 后端直接
`free(): double free detected`），补偿手段是"按指针匹配把条目从另一张登记表里摘掉"
——用更复杂的簿记去补两套簿记。

**改法**：

| 改动 | 说明 |
|---|---|
| 新增 `include/DevBuffer.h` | move-only 所有者：构造即分配、析构在**自己的设备**上释放、禁拷贝；`disarm()` 供进程退出期（上下文已销毁）只丢指针不 free |
| `d_momenta_structs_/arrays_`（`vector<pair<int,ptr>>`） | → `vector<ctpwa::DevBuffer<...>>`，成为唯一所有者 |
| `d_momenta_` / `d_mom_sigma_` | 降级为**视图**（裸指针，无所有权） |
| `releaseStaticData` | 删掉 `std::set<DeviceMomenta*> drop` + 重建登记表那 30 行，改为 `releasePermMomenta()`：所有者向量 `resize(n_gpu)`，尾部析构自动释放 |
| 析构 | `d_mom_structs_.clear()` 即可；上下文已销毁的分支先 `disarmMomentaOwners()` |

**验证**：全量逐文件扫描 251 passed / 7 failed / 30 skipped，与重构前基线逐文件
对比**只有 7 个 devify 过的文件变化（全部是增加通过）**，无一处回退；新增
`tests/test_ownership.py` 3 个用例（ident2/ident3 反复构造→NLL→release→销毁，
CPU 后端 double free 会直接 abort）。CUDA 侧只编译验证：EXIT=0、0 错误。

**顺带发现的一个真问题**：CPU 后端 `Tensor::get_device()` 返回 **-1**，而
`getNLL` 里 `const int primary_dev = vector.get_device()` 的这个值被当设备号用。
以前 `cudaSetDevice(-1)` 静默失败（不切设备），P0 的 `ensure_device` 把它变成
可见的报错。已归一化：`ctpwa_is_cuda_like(vector) ? get_device() : 0`。
（这类"CPU 上恰好不出事"的隐式越界，正是把裸调用换成检查型入口的额外收益。）

---

## P2（第二批）：`AmpCasDecay` 六张设备表改为单一所有权

同一类问题、更大量的一处：`d_slamp_tab_` / `d_mom_tab_` / `d_sign_tab_` /
`d_decayNodes_` / `d_slCombination_` / `d_polarization_map_` 原来有**三条释放路径**
（构造期清理块、`releaseStaticData`、析构函数），每条都要自己 `cudaSetDevice(g)` 再
`cudaFree`，且必须保证"谁先放、放完置空"，否则二次释放。

| 改动 | 说明 |
|---|---|
| 六个成员 | `std::vector<T*>` → `std::vector<ctpwa::DevBuffer<T>>`（唯一所有者） |
| 三条释放路径 | 统一成 `clear()` / `reset()`，**不需要** `cudaSetDevice`（DevBuffer 在自己的设备上释放） |
| 访问器 | `const std::vector<T*>& getX()` → `T* getX(int gpu)`（带边界检查），新增 `nGpuTables()` |
| 调用点 | 30 处 `cas->getX()[gpu]` → `cas->getX(gpu)`；39 处 `d_X_[i]` → `d_X_[i].data()` |
| 析构的失效上下文分支 | 六个表逐个 `disarm()`，与四动量一致 |

**验收**：

| 项 | 结果 |
|---|---|
| `grep 'cudaFree(d_slamp_tab_\|cudaMalloc(&d_slamp_tab_' src/` | 空（分配/释放只剩 DevBuffer 一条路） |
| 全量逐文件扫描 | 254 passed / 7 failed / 30 skipped，与 baseline2 对比**只多出 test_ownership 的 3 个**，其余逐行相同 |
| 裸 `cudaSetDevice` 违规 | 127 → **125** |
| CUDA 只编译 | EXIT=0，0 错误 |

---

## P3（第一步）：衰变描述类型与置换拓扑从 SL 表类里抽出来

**问题**：`AmpGen.cuh` 一个头里塞着三类东西——衰变描述（`SL`/`Amp2BD`/
`DecayNodeHost`）、设备结构体（`DeviceMomenta`）、SL 振幅表类（`AmpCasDecay`，
含 6 张设备表与 JIT 状态）。后果是任何"只想描述一条衰变链"的代码都必须先能
编译整个设备层；而全同粒子置换拓扑这种**纯组合逻辑**只能通过构造一个持有设备表
的对象才能算——于是它无法脱离 GPU 单测。

**改动**：

| 新增 | 内容 | 依赖 |
|---|---|---|
| `include/DecayDesc.h` | `SL`、`Amp2BD`（声明）、`DecayNodeHost` | 只依赖 C++ 标准库 + `CudaCompat.h`（SL 的 `__host__ __device__`） |
| `include/PermTopology.h` | `PermTopology`（每 σ 的重建映射 + 符号）、`buildPermTopology()` | 只依赖 `DecayDesc.h` 与标准库 |
| `src/PermTopology.cu` | 200 行置换拓扑算法（原 `AmpCasDecay::buildPermTopologies` 的 210 行） | 新 TU，纯 host |

`AmpCasDecay::buildPermTopologies()` 缩成 6 行接线；`AmpGen.cuh` 里
`DecayNodeHost` 保留 `using DecayNodeHost = ::DecayNodeHost;` 以兼容既有写法。

**判据（可机械检查）**：

```bash
# 设备 API / 设备类型：都为 0
grep -cE 'cudaMalloc|cudaMemcpy|cudaSetDevice|cudaFree|thrust::|torch::' \
     include/DecayDesc.h include/PermTopology.h src/PermTopology.cu
# 原类的成员残留：为 0
grep -c 'h_perm_maps_\|h_signs_' src/PermTopology.cu
```

两项实测都是 0。

**验证**：全量逐文件扫描 **254 passed / 7 failed / 30 skipped**，与 baseline3
**逐行相同**（置换拓扑的敏感用例 `test_identical` 11 passed、`test_barrier` 7
passed 都在其中）；CUDA 侧只编译 EXIT=0、0 错误。

---

## P3（第二步）：`DecayChainDesc` —— 衰变链成为可脱离 GPU 的值类型

第一步把"描述类型"搬出了 `AmpGen.cuh`，但它们仍散在 `AmpCasDecay` 的四个容器里
（`decayChain_`/`particleMap_`/`particleNames_`/`nSLCombs_`），想构造一条链依然
要先 new 一个持有 6 张设备表 + JIT 状态的对象。

**改动**：

| 新增/移动 | 内容 |
|---|---|
| `include/DecayChainDesc.h` | `ParticleDesc` + `DecayChainDesc`（header-only）：`addParticle` / `addNode` / `setNSLCombs` + 只读访问器 |
| `src/DecayDesc.cu`（新 TU） | `Amp2BD` 的构造与 `ComSL`（~110 行）从 `AmpGen.cu` 搬出 |
| `AmpCasDecay` | 四个容器 → 一个 `ctpwa::DecayChainDesc desc_;`（has-a），52 处引用改为 `desc_.nodes()` / `desc_.particles()` / … |

**验收（第一次有真正脱离 GPU 的单测）**：

```bash
bash tests/cpu_backend/build_and_run.sh
# [test_decay_desc] 纯 host，无 GPU/torch/ROOT 依赖
# [test_decay_desc] all checks passed
```

新增 `tests/cpu_backend/test_decay_desc.cpp`，编译链只有
`src/DecayDesc.cu + 测试文件 + 标准库`——**不链接 `Backend.cpp`、不需要 torch/ROOT**。
它检查：粒子去重与顺序、节点追加与波数乘积、`setNSLCombs` 覆盖、
`Amp2BD` 的 (S,L) 结构不变量（`|J2-J3| ≤ S ≤ J2+J3`、`L ≤ maxL`）。

全量 pytest 门禁：254 passed / 7 failed / 56 skipped / 4 known errors
（`test_reparam` 桩 bug），与 baseline3 逐文件 PASS/FAIL 一致；CUDA 只编译 EXIT=0。

---

## P2（第三批）：`AmpCalc::ResBlock` 的四个大缓冲

`d_T`（有效耦合 `nEvents×nPolar`）、`d_dF`（导数 `nEv×nSL×Nfree`）、`d_prop6`
（BWR 传播子复用表）、`d_static`（固定节点预算表）——全是每 GPU 一份、在 fit 循环里
懒分配的大块，原来只在 `release()` 里靠 `free_at` + `cudaSetDevice(g)` 释放。

| 改动 | 说明 |
|---|---|
| `DevBuffer::allocate_bytes(bytes)` + `bytes()` | `d_dF` 的元素宽度随 precision 档位变（float2 / double2），只能按字节算；`size()` 取整会错 |
| 四个成员 `vector<T*>` → `vector<DevBuffer<T>>` | 单一所有者 |
| `release()` | 这四个从 `free_at` 名单移除，改为 `clear()`（DevBuffer 在各自设备上释放） |
| 分配/使用点 | `allocate`/`allocate_bytes`；取 `.data()` |

**验证**：门禁 254 passed / 7 failed / 56 skipped / 4 known errors（与基线一致）；
CUDA 只编译 EXIT=0、0 错误。

---

## P4（第一步）：数据读取从 `analysis` 抽出

`Analysis.cu` 7624 行（原 7878）里，最先能独立的是**输入读取**：两个自由函数
（`readMomentaFromDat` 支持 .dat/ROOT、`readWeightValuesFromFile`），共 331 行，
逻辑上早就与设备无关，只是住在 god class 的 TU 里。

| 新增 | 内容 |
|---|---|
| `include/DataReader.h` | 两个纯 host 读取函数声明（返回 STL 容器） |
| `src/DataReader.cu` | 实现（ROOT `TFile`/`TTree` + 文本解析），新 TU |

留在原处：`readWeightsFromFile` 返回 `std::vector<double*>`（设备指针，需要
cudaMalloc），属 loader/device 层，不是纯 io——这条边界写进了 `DataReader.h` 的注释。

**这轮踩到的坑（已变成门禁的一部分）**：头文件里把参数写成 `int`、实现是 `long`，
编译照过、链接照过（shared library 允许未定义符号），到 `import ctpwa` 才炸：

```
RuntimeError: 没有可用的 ctpwa 后端 ... ImportError: _cpu.so: undefined symbol:
  readWeightValuesFromFile(... int)
```

而 pytest 对这种加载失败会给 **41 个 "usage error"**，噪音完全盖住真因。
`scripts/run_tests_parallel.sh` 现在先跑一次 `python3 -c "import ctpwa"` 预检，
失败就直接把真实 import 错误打出来并退出（exit 2），不再让 41 行噪音淹没它。

**验证**：门禁 254 passed / 7 failed / 56 skipped / 4 known errors（与 baseline3 一致）；
CUDA 只编译 EXIT=0、0 错误。

---

## P0-B 为什么停在这里（128 处多卡循环内的 set）

剩下的 133 处里，128 处是 `for (gpu) { cudaSetDevice(gpu); … }` 形态。改成
`DeviceGuard` 会让**每次迭代结束还原设备**，而今天是"循环结束后停在最后一张卡"
（sticky）。逐点核对发现只有 6 处能靠静态缩进判定成对，"循环之后再访存/释放、
依赖 current 仍是最后一张卡"的块无法用 grep 判安全。

这类改动的正确性只有在**双卡**上才能验证（单卡下任何设备号都退化到 0，测不出）。
因此按"无 GPU 不动设备语义"的原则停在这里，等 GPU 空闲后：
1. 单卡先跑（确认无回归）；
2. 双卡跑 `scripts/cluster/multigpu_parity.py` 与现有 multigpu 回归；
3. 再按文件分批（先 AmpGen 30 处，再 Analysis 103 处）转换，每批一次双卡验证。

替代做法（已评估、暂不采用）：把所有循环体改成 `ctpwa::with_device(g, [&]{…})`
需要重排花括号，机械 diff 面远大于 1 行替换，风险更高。

---

## CUDA 构建验证（只编译，不占用 GPU）

P0/P1 之后 CUDA 路径必须仍然能编（用户明确：不能把 CUDA 改成 CPU）。验证方式是
**只编译、不运行**——nvcc 编译不需要 GPU 设备，只有运行期才需要：

```bash
CTPWA_BACKEND=cuda TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=4 python3 setup.py build_ext --inplace
```

实测（2026-11-27）：

| 项 | 结果 |
|---|---|
| 退出码 | 0（20/20 步，含 `-dlink` RDC 链接） |
| 编译错误 | 0 |
| 产物 | `ctpwa/_cuda.so` 14.2 MB（单架构 sm_86；8 架构时 17.8 MB） |
| `ldd` 缺失库 | 0 |
| 未解析的 `ctpwa::` 符号 | 0 |

也就是说 17 个 TU + `-rdc=true` 的组合在 CUDA 侧同样成立，`DeviceGuard.h` 里的
`ctpwa::set_device/current_device` 两后端都能解析。

**没有做**的：CUDA 的运行时行为（多卡设备归属、性能）——那需要真 GPU，等空闲。

---

## 门禁现状（P2 后）

| 指标 | 重构前 | 现在 |
|---|---|---|
| 门禁 passed | 183 | **251 + 3**（新增 test_ownership） |
| 其中数值用例 | 0（全 skip） | **37**（含 golden NLL / golden 梯度 / 梯度 vs FD） |
| failed | 8 | 7（均已定位为 CPU 构建固有） |
| 裸 `cudaSetDevice` 违规 | 208 | **127** |

---

## 当前门禁口径

```bash
CTPWA_BACKEND=cpu MAX_JOBS=4 python3 setup.py build_ext --inplace
bash scripts/check_device_calls.sh          # 违规数只许下降
bash tmp/run_baseline.sh                    # 逐文件跑（整目录单次跑会被慢用例拖住）
diff 基线表 当前表                            # 判据：passed/failed/skipped 逐文件一致
```

基线：`doc/baseline-cpu-v0.3.27.md`（183 / 8 / 26，8 个失败都是 CPU 构建固有的）。
