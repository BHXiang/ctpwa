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
| P4 第二步（`analysis` 变成可包含的头 + 首个方法迁出） | ✅ | `0cbad61` |
| P4 第三步（`PhspGram` 显式输入模块） | ✅ | `b5b7975` |
| P4 第四步（`MatrixErrors` + 共享 helper 头） | ✅ | `8c0e86a` |
| P4 第五步（`SLAmpsIO`） | ✅ | `cd05b49` |
| P4 第六步（helper 上移 + `TruthIntegrals`） | ✅ | `fa8f7fd` |
| G3 CI 门禁（CPU + CUDA 只编译） | ✅ | `f383d62` |
| **A/B 数值对照（vs v0.3.27 基线）** | ✅ | `c95eddb` |
| P4 第七步（`BatchIntegrals` 247 行） | ✅ | `b94ebbc` |
| P4 第八步（`FitFractions` 约 330 行） | ✅ | `f73483a` |
| P4 第九步（`rebuildPhspGram` + helper 上移） | ✅ | `9021595` |
| P4 第十步（`Polarization` 纯函数） | ✅ | `8353f4d` |
| **门禁全绿（258/0/63）** | ✅ | `9409335` |
| P4 第十一步（`HessianContrib`） | ✅ | `1b503b1` |
| P4 第十二步（`CouplingHessian` 137 行） | ✅ | `bb95a4b` |
| **P4 第十三步（`FullHessian` 685 行，最大单体）** | ✅ | `4641bff` |
| **P4 第十四步（写入器搬出 header，−1830 行）** | ✅ | `5a58c7c` |
| **P4 第十五步（`initialize` 三件套，−1101 行）** | ✅ | `c4dcb3c` |
| **P4 第十六步（剩余成员搬出 + 删 `Analysis.cu` 中转）** | ✅ | `73bb08b` |
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

## P4（第二步）：解除 god class 的结构性阻塞

**阻塞点**：`class analysis` 的 57 个方法全部是**类内内联定义**，而且没有头文件 ——
`main.cu` 只能靠 `#include "Analysis.cu"` 把整个 god class 文本内联进来。在这种形态下，
"把某个方法搬到新模块"在语法上根本做不到：类定义不出现在任何头里，别的 TU 无法写
`analysis::method`。

**做法**（同一个 TU、同一份代码，行为不变）：

| 文件 | 变化 |
|---|---|
| `include/Analysis.h`（新，7594 行） | 原 `Analysis.cu` 全文（工具函数 + 类定义）原样搬入 |
| `src/Analysis.cu` | 7624 行 → **5 行**（只剩 `#include <Analysis.h>`） |
| `include/MatrixStats.h` + `src/MatrixStats.cu`（新） | `ctpwa::couplingCovariance` |

**解锁演示**：`couplingCovariance` 一个成员变量都不读（纯张量数学：取自由方向、
对称化、eigvalsh 判正定、求逆），把它抽成自由函数后，成员只剩一行委派：

```cpp
torch::Tensor couplingCovariance(const torch::Tensor& H, int n,
                                 int& nf_out, bool& ok) const
{
    return ctpwa::couplingCovariance(H, n, nf_out, ok);
}
```

这是 P4 第一个真正搬出 god class 的函数。**注意它的意义在"解锁"而不是"减重"**：
`Analysis.h` 仍有 ~6300 行类定义，但从此每轮可以按方法逐个改成
"类内声明 + 新 TU 定义"，而不用一次性重写 6000 行。

**验证**：门禁 254 passed / 7 failed / 56 skipped / 4 known errors（与基线一致）；
CUDA 只编译 EXIT=0、0 错误。

---

## P4（第三步）：`PhspGram` —— "先显式化输入，再搬函数" 的可复用模板

`analysis::phspMatrixFromGram` 只读 7 个成员（`events_` / `phsp_weights_` / 两个 B̄
指针 / `n_amplitudes_` / `nSLvectors_` / `primary_dev_`），但因为住在 god class 里，
既不能单测也不能复用。

| 新增 | 内容 |
|---|---|
| `include/PhspGram.h` | `PhspGramInputs`（只读结构体，7 个字段）+ 自由函数声明 |
| `src/PhspGram.cu` | 76 行实现（B̄ 收缩成 npartials×npartials 干涉矩阵 + 权重归一） |
| 成员 | 8 行委派：显式填 `in` 后调用 |

**这轮定下的模板**（后续每次搬家都照这个走）：

1. 先写一个**只读输入结构体**，字段就是函数真正读的那些成员；
2. 成员函数改成委派，把 `in` 一个个填上——**编译器会逐个字段告诉你漏了什么**；
3. 通过门禁后，函数体搬到新 TU。

比"直接搬函数、再补参数"安全：直接搬的话，漏一个字段往往编译能过（默认值/隐式转换），
错要到运行时才现。`Analysis.h` 7594 → 7529 行。

**验证**：门禁 254 passed / 7 failed / 56 skipped / 4 known errors；CUDA 只编译
EXIT=0、0 错误。

---

## P4（第四步）：`MatrixErrors` —— 第一个"状态 + 回调"混合切片

`analysis::computeMatrixErrors` 292 行，读 9 个成员、调 4 个成员函数、还会临时改写
`theta_cur_` 做中心差分。它是 P4 里第一个**不能只靠只读结构体**搬走的函数：θ 扫描
要改状态，积分要调算法。所以模板升级成两段式：

| 依赖类型 | 处理 |
|---|---|
| 它**读**的状态（events/phsp_weights/B̄ 指针/n_amplitudes/n_theta/…) | `MatrixErrorCtx` 的只读字段（9 个） |
| 它**调用**的算法（`freeParamsToAmplitudes` / `getHessian` / `computeFileBatchIntegrals` / θ 读写） | `std::function` 回调（6 个） |

回调签名直接照抄 `computeFileBatchIntegrals` 的 12 个参数，成员装配时原样转发 ——
**不包一层自定义协议**，避免"抽出来之后行为悄悄变了"。

顺带抽出 `include/BackendHelpers.h`（`ctpwa_dev` / `ctpwa_is_cuda_like` /
`makeScopeExit` / `CUDA_CHECK` 等）：它们原来只存在于 `Analysis.h` 内部，任何搬出去的
模块都得复制一遍，这正是"god class 的隐性成本"。

**踩到的坑**：`d_phsp_matrix_double_` → `ctx.d_phsp_matrix_double` 替换后，随后的
`d_phsp_matrix_` 规则又把它当子串匹配了一次，产生 `ctx.ctx.d_phsp_matrixdouble`。
**按前缀替换必须先长后短**，且替换完用 `grep -c 'ctx.ctx'` 验一遍。

`Analysis.h` 7529 → **7266 行**。门禁 254 passed / 7 failed / 56 skipped / 4 known
errors（`test_matrix_errors` 7 个用例正好覆盖这条路径）。

---

## P4（第五步）：SL 振幅导出 + 一次失败的尝试与它的结论

**成功**：`getSLAmpsTensor` / `saveSLAmps` 两个 const 成员只用到 `amp_calc_` 与
`phsp_freed_`，抽成 `ctpwa::slAmpsTensor` / `saveSLAmpsToFile`（`include/SLAmpsIO.h`
+ `src/SLAmpsIO.cu`），`Analysis.h` 7266 → **7219 行**。

**失败并回退**：本想顺手搬 `computeTruthIntegrals`（77 行，看似只依赖 `nSLvectors_`）。
实际编译暴露出一串隐藏依赖：

```
nSLvectors_ →  ok
float_amps_ / n_polar_ / n_amplitudes_ →  再加 3 个参数
upcastAmpSegToDouble（static 函数）→ 要上移到共享头
  └→ castF2ToDouble2Kernel（__global__）→ **不能**放进共享头
     （多 TU 包含会多重定义；放 .cu 又需要一个新 TU 与声明）
computeBranchingFractions（别处的自由函数）→ 还要把头也搬
```

**结论（写进 P4 的方法论）**：god class 的隐性成本不只在类本身，还在它**顺带定义**
的 static helper 与 kernel。批量搬函数之前要先做一轮"共享 helper 上移"：

| 要上移的东西 | 去处 | 注意 |
|---|---|---|
| `ctpwa_dev` / `ctpwa_is_cuda_like` / `makeScopeExit` / `CUDA_CHECK` | `BackendHelpers.h`（已完成） | 纯 host，可直接放头 |
| `upcastAmpSegToDouble` 等上转工具 | `MixedPrecision.h` + `.cu` | 依赖 `__global__` kernel，必须头声明 + 单独 TU 定义 |
| `castF2ToDouble2Kernel` 等 kernel | 新 TU 或 `ComputeBF` 已有 TU | `__global__` 不能进头（CPU 后端会多重定义） |

所以下一轮的第一个任务不是搬函数，而是**把这批 helper 归位**——之后
`computeTruthIntegrals`、`ampSegmentToTensor` 这类函数才能干净搬走。

---

## P4（第六步）：先做 helper 上移，再搬函数

上一轮发现"搬函数会牵出 static helper 与 kernel"。这轮先把它们归位：

| 上移的东西 | 去处 | 结果 |
|---|---|---|
| `upcastAmpSegToDouble`（原 `static`） | `include/MixedPrecision.h`（inline） | 它依赖的 `castF2ToDouble2Kernel` 定义在 `ComputeHessian.cu`、声明在 `.cuh`，所以放头安全 |
| `ctpwa_dev` / `ctpwa_is_cuda_like` / `ctpwa_on_primary` / `ScopeExit` / `makeScopeExit` / `CUDA_CHECK` | `BackendHelpers.h`（已存在，本轮把 Analysis.h 里的重复定义删掉） | 上轮为 MatrixErrors 新建时，Analysis.h 还留着自己的那份 → 重定义 |

归位之后，两个函数一次就搬干净了：

- `computeTruthIntegrals`（77 行）→ `include/TruthIntegrals.h` + `src/TruthIntegrals.cu`，
  新增 5 个显式参数（`n_slvectors` / `float_amps` / `n_polar` / `n_amplitudes` / `primary_dev`）；
- `ampSegmentToTensor`（15 行）→ 并入 `SLAmpsIO`，直接复用上移后的上转工具。

`Analysis.h` 7220 → **7079 行**。门禁 254 passed / 7 failed / 56 skipped / 4 known
errors；CUDA 只编译 EXIT=0、0 错误。

**方法论沉淀**：搬函数前先问三件事——它读哪些成员（→ 参数/字段）、它调哪些算法
（→ 回调）、它用到哪些 static helper / kernel（→ 先上移，kernel 只能进 .cu）。

---

## G3：CI 门禁落地（补齐发布流程里缺的那一环）

`doc/refactor-release-workflow.md` 里 G3 一直是空的：仓库只有 tag 触发的
`build-wheel.yml`，push/PR 什么都不跑。本轮补上 `.github/workflows/ci.yml`：

| job | 内容 |
|---|---|
| `gate-cpu` | conda-forge ROOT + torch(cpu) → CPU 构建 → **import 预检** → **设备棘轮** → 全量 pytest（无卡机器上 GPU 用例自动 skip） |
| `compile-cuda` | 只在 PR 动到 `src/` 时跑：nvcc 单架构**只编译**（不需要卡）+ 检查 `_cuda.so` **无未解析 `ctpwa::` 符号** |

两个细节来自踩过的坑：

1. **import 预检**：`.so` 缺符号时 pytest 给出 41 个 "usage error"，噪音完全盖住真因
   （第 6 轮实测）；预检失败直接把真实 import 错误打出来。
2. **未解析符号检查**：shared library 允许未定义符号，"链接成功"不等于"能 import"。
   `nm -C --undefined-only | grep ctpwa::` 把这条检查搬进 CI。

**设备门禁改成棘轮**：`scripts/check_device_calls.sh` 默认读
`scripts/device_call_budget.txt`（当前 **120**），违规数超过预算才失败。理由：P0-B 需要
真双卡才能验证，在完成前把门槛设成 0 只会让门禁长期红灯、然后被人绕过；棘轮的意思是
**只许下降**，每完成一批就手动调小预算（P0-A 的 208 → 133 → 125 → 120 就是这么走的）。

本轮顺手把自己新抽出模块里的 5 处裸 `cudaSetDevice` 改成 `ctpwa::DeviceGuard`
（PhspGram/TruthIntegrals/MatrixErrors/SLAmpsIO），它们本来就在每 GPU 循环体内，
guard 的作用域正好是循环体 —— 这也是"新模块应该一步到位用 DeviceGuard"的示范。

---

## P4 的进度评估（诚实版）

`Analysis.h` 仍有 **7529 行**，其中类定义约 6300 行。已迁出：io（331 行）、
矩阵统计（`couplingCovariance`）、`PhspGram`（76 行）。

按目前每轮 60~300 行的速度，把类压到计划里写的"<800 行"还需要 **十几轮**，
而且必须靠更大的切片：

| 切片 | 规模 | 依赖 |
|---|---|---|
| 写入器 `writeResult`/`writeInterfResult`/`writeMC` | ~1900 行 | 需要 `FitResult` POD |
| 误差矩阵 `computeMatrixErrors` + `runMatrices` | ~400 行 | 需要 θ 重算回调 |
| 份额/效率 `getFitFractions`/`getEfficiency`/`computeFileBatchIntegrals` | ~860 行 | 需要振幅/偏移上下文 |
| 初始化 `initialize`/`initializeDecayChains` | ~930 行 | 与设备分配强耦合 |

结论：**P4 的"启动"已完成**（类可包含 + 3 个模块边界 + 模板），"完成"仍是多轮工程。
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

## P4（第七步）：`computeFileBatchIntegrals` → `BatchIntegrals` 模块

份额/效率/误差矩阵/写结果共同的底层（247 行）。依赖形态最复杂的一类：9 个成员 +
2 个成员函数 + **会临时改写** `events_offsets_`/`amp_offsets_`。按 P4 模板拆成三类：

| 依赖 | 形态 |
|---|---|
| 读的状态（data_files / data_order / particles / n_slvectors / n_amplitudes / …） | 只读字段 |
| 要改的状态（`events_offsets` / `amp_offsets`） | **指针**（函数内保存-改写-还原） |
| 算法（`currentThetaHost` / `calculateAmplitudes`） | 2 个 `std::function` 回调 |

顺带把 `computeTruthIntegrals` 的 3 个调用点补上它上轮新增的 5 个显式参数。

`Analysis.h` 7079 → **6859 行**；门禁 254/7/56/4；A/B 指纹与 v0.3.27 基线**逐位一致**。

**同一个坑第二次踩**：`particles_` → `(*ctx.particles)` 的前缀替换把局部变量
`particles_names` 也改了（`(*ctx.particles)names`）。第 10 轮记过这条，这次仍中招。
现在把检查固化进流程：批量替换后立刻 `grep` 局部变量名是否被污染。

---

## P4（第八步）：份额/效率/按样本矩阵 → `FitFractions`

这三个函数（137+151+43 行）依赖形态完全一致，且它们调的底层
（`computeFileBatchIntegrals` / `phspMatrixFromGram` / `computeTruthIntegrals`）
**在前两轮已经变成自由函数**——所以这轮能一次搬走约 330 行，而不是又踩一轮级联。
这就是"先搬底层"的回报。

编译期仍牵出两个漏网的依赖，正好补上：

| 依赖 | 处理 |
|---|---|
| `computeCouplingHessian`（成员） | 回调接入（它自身还很耦合） |
| `couplingCovariance` | 直接换成**自由函数** `ctpwa::couplingCovariance`（第 8 轮抽出的） |
| `computeBFErrors` | `ComputeBF.cuh` 里本来就有声明，include 即可 |

`Analysis.h` 6859 → **6567 行**。门禁 254/7/56/4；A/B 指纹逐位一致；CUDA 编译 0 错误。

---

## P4（第九步）：`rebuildPhspGram` 并入 `PhspGram`，helper 再上移

`rebuildPhspGram`（90 行）是 B̄ 的分块重建，和已抽出的 `phspMatrixFromGram` 是同一个
模块的事，所以并进 `PhspGram`，参数显式化：

| 参数 | 说明 |
|---|---|
| 7 个只读（振幅/事件/权重/精度标志/维度/设备） | 值或 const 引用 |
| `ctComplex** d_matrix_io` | **读写**输出（先 memset 再逐块 axpy 累加） |
| `std::function<int()> batch_events` | 每块事件数依赖当前空闲显存，只能用回调 |

顺带再把 `ctpwaCheckedMalloc`（OOM 诊断用的分配包装）上移到 `BackendHelpers.h`——
又一条"god class 顺带定义的 static helper"，搬出去的模块都要用。

`Analysis.h` 6556 → **6471 行**。门禁 254/7/56/4；A/B 指纹逐位一致；CUDA 编译 0 错误。

**流程改进生效**：这轮两次 prefix 替换（`n_polar_` / `primary_dev_`）都先 `assert` 检查
再落盘，没有再出现"改到局部变量名"的事故。

---

## P4（第十步）：初态极化索引映射 → `Polarization`（纯函数）

`initializePolarization`（73 行）只依赖 `particles_`——张量维数、mask 后的有效极化数、
output→tensor 索引映射，全是组合数学。抽成 `ctpwa::initializePolarization(particles)`
后是纯 host 函数，可脱离 GPU 单测。

`Analysis.h` 6471 → **6408 行**。门禁 254/7/56/4；A/B 指纹逐位一致。

**踩坑（两个都值得记）**：

1. 结构体变量叫 `out`，撞上函数体里的 `for (int out = 0; ...)` → 编译错。搬代码时
   给汇总结构体起名要避开原函数里的局部名（`res` 比 `out` 安全）。
2. 前缀替换把 `n_polar_total_` 变成了 `res.n_polar` + `total`——**又**是前缀冲突，
   即使我先替换长的。根因是替换进去的新文本 `res.n_polar_total` 里又含
   `n_polar_`…… 现在的做法是：**用正则 `\b…\b` 并且一次生成、不从半成品再替换**。

---

## A/B 数值对照（第一次真有"基线 vs 重构"的逐值证据）

objective 要求"记录每阶段基线与 A/B 数值对照"。此前只有"测试是否通过"和
"逐文件 PASS/FAIL 是否与基线一致"——那证明的是**测试断言**没变，不是**数值**没变。

本轮补上：

1. `scripts/ab_fingerprint.py`：固定 seed（**参数在 CPU 上生成再搬**，避免
   `torch.Generator(device=cuda)` 的随机流差异）算 NLL / 梯度 / Hessian，dump JSON，
   `--compare` 逐值比。
2. `git worktree add baseline_wt main` 重建 **v0.3.27 的原始代码**（不是靠记忆，
   是同一台机器上现编的一份），两套构建跑同一脚本。

结果（`doc/parity/compare-baseline-vs-refactor.txt`）：

| config | nll max\|Δ\| | hess max\|Δ\| | grad 最大相对差 |
|---|---|---|---|
| simple | 0 | 0 | 1.5e-15 |
| with_trans | 0 | 0 | 7.2e-16 |
| flatte | 0 | 0 | 4.0e-15 |
| gs | 0 | 0 | 4.2e-15 |
| interp | 0 | 0 | 0 |
| ident2 | 0 | 0 | 7.2e-15 |
| ident3 | 0 | 0 | 1.9e-15 |

**NLL 与 Hessian 逐位相同**；梯度的 1e-15 级差异是 autograd 累加顺序，不是行为变化。

这条对照现在是可复跑的：基线指纹存在 `doc/parity/fingerprint-baseline-v0.3.27.json`，
每个阶段只要重跑一次 `--compare` 就能确认"这一阶段仍然没动物理"。

顺带修掉一个潜伏问题：`getHessian` 里两处 `primary_dev = dev.index()`，在 CPU 张量上
`dev.index()` 是 **-1**，却被当设备号用（`ensure_device(-1)` 会打错误日志）。
归一化为 0，与 `getNLL` 那处一致。

---

## 门禁全绿（2026-11，第一次 failed=0）

此前的验收一直是"与已知失败表对比"：`254 passed / 7 failed / 56 skipped / 4 errors`。
那条判据能用，但每次都要人肉确认"失败集合没变"，CI 也没法直接吃。

这轮把失败归零（**没有删测试**）：

| 原问题 | 真实原因 | 处理 |
|---|---|---|
| `test_reparam` 4 errors | `example/fit.py` 新增 `from ctpwa.diagnostics import …`，测试桩只塞了 `types.ModuleType("ctpwa")` | 补子模块桩 → **4 passed** |
| `test_float_mode` 4 failed | `precision:float` 是 GPU 混合精度档，CPU 构建固定 double | 标记"仅 CUDA 构建覆盖" + 写明覆盖来源 |
| `test_hessian_bwr_spec` 2 failed | BWR 解析特化只接了 CUDA kernel | 同上 |
| `test_memory_monitor` 1 failed | `cudaMemGetInfo` 显存监控，CPU 语义不同 | 同上 |

`scripts/run_tests_parallel.sh` 现在**用退出码表达判据**：`failed>0 || errors>0` → exit 1，
skipped 不计（GPU-only 用例在 CPU 构建上就该跳过）。CI 直接吃这个退出码。

当前：**258 passed / 0 failed / 63 skipped**。门禁第一次变成"红即回归"。

---

## P4（第十一步）：`hessianSegmentContrib` → `HessianContrib`

`getHessian` 那一簇里最先能搬的一块（58 行，8 个字段）。它能搬干净的原因是：
调用的 `computeDataHessianContrib(Fast)` 本来就是 `ComputeHessian.cu` 的自由函数，
所以**不需要任何回调**，纯参数化就够。

`Analysis.h` 6408 → **6359 行**。门禁 258/0/63（全绿）；A/B 指纹逐位一致；
CUDA 只编译 0 错误。

---

## P4（第十二步）：`computeCouplingHessian` → `CouplingHessian`

这轮验证了"底层自由化"的复利：它的 6 个依赖里
`computeQuadraticForm` / `computePhspHessian` / `reduceHessianWithConstraints` /
`hessianSegmentContrib` **都已经是自由函数**，`hessianFastPathEnabled` 本轮顺手
上移到 `HessianContrib.h`，于是只剩 `freeParamsToAmplitudes` 一个回调。

`Analysis.h` 6343 → **6223 行**。门禁 258/0/63；A/B 逐位一致；CUDA 编译 0 错误。

**踩坑（重要）**：给 ctx 补字段时用全文 replace，命中了**另一个** delegate
（`MatrixErrorCtx` 里也有 `ctx.events = &events_;`），把不存在的字段塞了进去。
教训：一个文件里同时存在多个 `ctx` 装配块时，**必须先按块定位（用各自的
`ctpwa::XxxCtx ctx;` 起止），再做行级修改**，不能全局 replace。

---

## P4（第十三步）：`getHessian` 685 行 → `FullHessian`（P4 最大单体）

这是"先搬底层"策略的收口。十二轮前它的 12 个依赖全是成员函数；现在
`computePhspHessian` / `computeQuadraticForm` / `reduceHessianWithConstraints` /
`computeFactorNLL(F)` / `hessianSegmentContrib` / `computeCouplingHessian` 全是对外
自由函数，**只剩 `freeParamsToAmplitudes` 一个回调**。接口：17 个字段 + 1 个回调。

`Analysis.h` 6223 → **5568 行**（单轮 −655）。

**过程中抓出两个真问题**（都靠门禁/测试抓的，不是靠 review）：

1. 委派里漏了 `ctx.amp_calc = &amp_calc_` → 空指针，整层 segfault。
2. 我图省事给 `rebuildPhspGram` 传了空 `phsp_weights` → 加权 NLL 的 Hessian 错了
   （`test_cpu_numerics::test_phsp_weights_nll_grad_hessian` 抓到，max_rel 0.85）。
   这个用例的注释写得很明白："phsp_weights 必须同时进入 NLL 归一化、梯度和 Hessian"
   ——正是为这类错误写的回归。

**门禁补洞**：段错误时 pytest 不打印 `N failed`，只看 failed/errors 会把整层信号错误
当成全绿（本轮实测：segfault 后 passed 258→122 却报"通过"）。现在
`rc ∉ {0,1,5}` 一律算失败并列出文件名。这是"门禁自己也可能骗人"的一次实例。

---

## P4（第十四步）：写入器搬出 god header —— -1830 行

第 8 轮把 `class analysis` 搬进头文件之后，出现了一条**零风险**的迁出路径：

    「类内内联定义」 → 「类内声明 + 新 TU 里 analysis::method 的定义」

不改一行方法体、不动任何接口、不需要 ctx/回调。这比"改成自由函数 + ctx"
省事得多——后者只在真要把方法变成**模块接口**时才需要。

| 项 | 数值 |
|---|---|
| `writeResult` | 1548 行 body |
| `writeInterfResult` | 280 行 body |
| `Analysis.h` | 5568 → **3738** 行 |
| `src/ResultWriter.cu` | 1852 行（新 TU） |

**头部卫生问题（第一次暴露）**：`Analysis.h` 里还有 4 个 `CTPWA_KERNEL` 定义和 3 个
非 inline 自由函数。第二个 TU include 这个头时立刻 `multiple definition`（实测）。
处理：kernel 加 `static`（CUDA 下 `static __global__` 合法，内部链接），自由函数加
`inline`。更彻底的做法是把 kernel 挪进 `.cu` + 头里只留声明——留给后续。

门禁 258/0/63；A/B 指纹逐位一致；CUDA 只编译 0 错误。

**这条路径对剩下的方法同样适用**：`initialize`（770）、`computePhspHessian` 等都可以
先"无脑搬出 header"，再决定要不要变成模块接口。

---

## P4（第十五步）：初始化三件套搬出 header（−1101 行）

同一路径第二批：`initialize`（770）、`initializeDecayChains`（159）、`getIntensity`（172）
→ `src/Initialize.cu`（1119 行）。方法体一字未改。

`Analysis.h` 3738 → **2637 行**。门禁 258/0/63；A/B 逐位一致；CUDA 编译 0 错误。

**累计**：`Analysis.h` 从重构前的 **7624 行**（Analysis.cu）降到 **2637 行**，
其中类定义部分已从 ~6300 行降到约 1600 行——剩下的主要是成员声明和小方法。

---

## P4（第十六步）：god class 拆解收尾

- `src/AnalysisMethods.cu`（新 TU）：`release` / `computeFileBatchIntegrals` /
  `computeMatrixErrors` / `initializeMultiGPUs` / `getHessian` / `getNLL` /
  `computeCouplingHessian` / `reCalcAmp` 的实现（266 行）。
- `src/main.cu` 改成 `#include <Analysis.h>`，**删掉 `src/Analysis.cu`**（最后的文本内联中转）。
  `analysis` 现在是"头文件声明 + 各 TU 定义"，与普通 C++ 类没有区别。

`include/Analysis.h` 2637 → **2423 行**。

### 这一阶段的账（P4 全程）

| 时点 | 形态 | 行数 |
|---|---|---|
| 重构前 | `src/Analysis.cu`，6300 行类 + 内联定义，无头文件 | 7624 |
| 第 8 轮 | 类搬进 `Analysis.h`，仍是内联定义 | 7594 |
| 第 23 轮 | 写入器搬出（`ResultWriter.cu`） | 3738 |
| 第 24 轮 | 初始化三件套搬出（`Initialize.cu`） | 2637 |
| **第 25 轮** | 剩余成员搬出 + 删中转（`AnalysisMethods.cu`） | **2423** |

2423 行里已几乎没有方法体：剩下的是 **成员声明 + 物理/格式约定注释**。要再降就得把
数据成员分组进子结构（如 `SampleState` / `PhspState`），那是另一种（更有值的）重构，
不属于"拆 god class"这一步。

---

## 当前门禁口径

```bash
CTPWA_BACKEND=cpu MAX_JOBS=4 python3 setup.py build_ext --inplace
bash scripts/check_device_calls.sh          # 违规数只许下降
bash tmp/run_baseline.sh                    # 逐文件跑（整目录单次跑会被慢用例拖住）
diff 基线表 当前表                            # 判据：passed/failed/skipped 逐文件一致
```

基线：`doc/baseline-cpu-v0.3.27.md`（183 / 8 / 26，8 个失败都是 CPU 构建固有的）。
