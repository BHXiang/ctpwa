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
| P2 第二批（`AmpCasDecay` 六张设备表） | ✅ | 见下 |
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

**验收**：`grep 'cudaFree(d_slamp_tab_\|cudaMalloc(&d_slamp_tab_' src/` 已为空
——六张表的分配/释放只剩 DevBuffer 一条路。

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
