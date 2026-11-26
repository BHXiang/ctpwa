# CTPWA 模块化改造：交付报告

> 分支 `refactor/modular`（从 `main@030baf5` = v0.3.27 起），61 个提交。
> 计划见 `doc/modularization-plan.md`，过程见 `doc/refactor-progress.md`，
> 发布流程见 `doc/refactor-release-workflow.md`。

---

## 0. 一句话结论

| 阶段 | 目标 | 状态 |
|---|---|---|
| **P0** 设备纪律 | `DeviceGuard` + 显式 `primary_dev` + 单一出口 | 🟡 **A/C/D 完成，B 待双卡** |
| **P1** 真翻译单元 | 拆掉文本内联的 12.8k 行单 TU | ✅ 17 TU，改一个文件 9.3 s（原 15 min） |
| **P2** DevBuffer 所有权 | 一个设备指针只有一处 free 路径 | ✅ `AmpCasDecay` 全部 + `AmpCalc` 四大缓冲 |
| **P3** DecayInfo 与 SL 表解耦 | 衰变描述可脱离 GPU 构造/检查 | ✅ `DecayChainDesc` + host 单测 |
| **P4** god class 拆分 | `analysis` 从 7624 行拆开 | ✅ 类定义 6300 → ~1600 行，18 个模块 |

**验证强度**（本次改造的最强证据，逐条可复跑）：
- CPU 门禁 **258 passed / 0 failed / 63 skipped**（全绿，红即回归）
- CPU A/B vs v0.3.27：**NLL 与 Hessian 逐位相同**
- **GPU A/B vs v0.3.27：NLL 逐位相同**（同一张 RTX 5060，sm_120）
- GPU 全套 **约 263 passed / 0 failed**
- 公开 Python API **0 增 0 删 0 签名变化**
- CUDA 侧始终能编译（0 error）

---

## 1. 现在的模块图

```
src/
  main.cu                 378   pybind 绑定 + 后端探针（只 include <Analysis.h>）
  AnalysisMethods.cu      266   analysis 的装配层成员（ctx 填充 + 调自由函数）
  Initialize.cu          1119   initialize / initializeDecayChains / getIntensity
  ResultWriter.cu        1852   writeResult / writeInterfResult
include/Analysis.h       2423   analysis 的声明 + 成员 + 约定注释（无方法体）

  —— 以下为本次抽出的领域模块（每个都有显式输入结构体/自由函数）——
  DataReader.cu           343   .dat / ROOT TTree 读取（纯 host）
  DecayDesc.cu            110   Amp2BD 构造 + (S,L) 枚举（纯 host）
  PermTopology.cu         224   全同粒子置换拓扑（纯 host，有 C++ 单测）
  Polarization.cu          87   初态极化索引映射（纯 host）
  MatrixStats.cu           64   couplingCovariance（纯张量数学）
  PhspGram.cu             204   phsp Gram → 干涉矩阵 / 重建 B̄
  MatrixErrors.cu         315   干涉矩阵的统计误差传播
  BatchIntegrals.cu       294   分批全样本积分（份额/效率/误差/写结果的底层）
  TruthIntegrals.cu       114   产生侧分波积分
  FitFractions.cu         356   份额 / 效率 / 按样本取矩阵
  HessianContrib.cu        85   单段 Hessian 贡献
  CouplingHessian.cu      165   耦合方向 Hessian
  FullHessian.cu          725   统一 Hessian（原 685 行的成员函数）
  SLAmpsIO.cu              94   SL 振幅表导出
  MixedPrecision.h              float2↔double 上转工具（原 Analysis.h 的 static）
  BackendHelpers.h              ctpwa_dev / makeScopeExit / CUDA_CHECK / ctpwaCheckedMalloc
```

`src/` 36 个源文件、`include/` 42 个头（含 `cpu_shims/` 与 kernel 头）。每个抽出的模块都配一个**只读输入结构体**
（`PhspGramInputs` / `BatchIntegralCtx` / `FullHessianCtx` …），字段就是该函数真正读的
成员，`std::function` 只留真正属于 `analysis` 的算法。

---

## 2. 三条可复用的手法（本次沉淀）

### 手法 A：显式输入 + 少量回调（把成员函数变成模块函数）

1. 写只读输入结构体，字段 = 函数读的成员；
2. 成员改成委派（填 `ctx` 再调用）——**编译器会逐个字段告诉你漏了什么**；
3. 通过门禁后，把函数体搬到新 TU。

用于：`computeMatrixErrors`、`computeFileBatchIntegrals`、`getHessian`、
`computeCouplingHessian`、`fitFractions`/`efficiency`、`phspMatrixFromGram` 等。

### 手法 B：header 化 + out-of-line（零风险搬家）

第 8 轮把 `class analysis` 搬进头文件之后：

```
「类内内联定义」 → 「类内声明 + 新 TU 里 analysis::method 的定义」
```

不改一行方法体、不动接口、不需要 ctx。`ResultWriter.cu`（1852 行）、
`Initialize.cu`（1119 行）都是这么来的。**先做 B 再做 A** 是效率最高的顺序。

### 手法 C：DevBuffer（把"谁负责 free"变成类型属性）

move-only 设备缓冲：构造即分配、析构在**自己的设备**上释放、禁拷贝、
`disarm()` 供进程退出期只丢指针不 free。消灭了"同一批指针两套簿记 →
double free"的整类问题。

**顺序判断法则**（本轮最有用的一条经验）：
> 一个函数能不能干净搬走，取决于**它调用的东西是否已经是自由函数**。
> 所以先搬底层（`computeTruthIntegrals` → `computeFileBatchIntegrals` → 上层），
> 最后搬最大的那个（`getHessian`）反而接口最小（17 字段 + 1 回调）。

---

## 3. 数值与契约的证据

### 3.1 A/B 指纹（`scripts/ab_fingerprint.py`）

7 个 config（simple / with_trans / flatte / gs / interp / ident2 / ident3），
参数**在 CPU 上固定 seed 生成再搬**（避免跨后端随机流差异），对比 NLL / 梯度 / Hessian。

| 对比 | NLL | 梯度 | Hessian |
|---|---|---|---|
| CPU 重构版 **vs** CPU v0.3.27 | **0（逐位）** | ≤7e-15 相对 | **0（逐位）** |
| GPU 重构版 **vs** GPU v0.3.27 | **0（逐位）** | ≤1.3e-13 | ≤1.7e-10 |
| CPU vs GPU（同一份重构代码） | ≤1e-9 | 1e-8 | **0.5%**（`with_trans`） |

最后一行不是回归：**v0.3.27 原版在 CPU/GPU 之间就有同样的 0.5%**，属固有跨后端差异
（见 `doc/parity/compare-gpu-baseline-vs-refactor.txt`）。

### 3.2 契约冻结（`tmp/api_dump.py` 实测）

`dir(ctpwa)` 与 `dir(ctpwa.analysis)` 及全部方法签名，与 v0.3.27 **完全一致**：

```
module:   移除 0, 新增 0
analysis: 移除 0, 新增 0
签名变化: 0
```

### 3.3 测试

| 环境 | 结果 | 说明 |
|---|---|---|
| CPU（`scripts/run_tests_parallel.sh`） | **258 / 0 / 63** | 全绿；`failed>0 \|\| errors>0 \|\| 异常退出` 都算失败 |
| GPU（RTX 5060, sm_120） | **约 263 passed / 0 failed** | 分五批跑；含 CPU 上 skip 的 float 档、BWR 特化 |

未跑的用例及原因：`test_memory_monitor`（针对 A100-40GB 的 400 万事件 phsp，
8GB 卡上无意义）、`test_cpu_*`（CPU 构建专属）。

---

## 4. 遗留项（明确未完成）

| 项 | 状态 | 说明 |
|---|---|---|
| **P0-B：120 处多卡循环内的 `cudaSetDevice`** | ⏸ **待双卡** | 改动语义（循环结束是否还依赖 current 停在最后一张卡），**只有双卡能验证**。已就绪：`scripts/check_device_calls.sh` 棘轮（预算 120，只许降）+ 每个新模块已改用 `DeviceGuard` |
| `Analysis.h` 里 4 个 `static CTPWA_KERNEL` | 未挪 | 已改成 `static`（内部链接）保证多 TU 安全；彻底做法是挪进 `.cu` + 头里留声明 |
| `analysis` 的 40+ 数据成员 | 未分组 | P4 的自然延伸：分成 `SampleState` / `PhspState` 等子结构，头文件可再降一档 |
| `test_reparam` 的桩 | 已修 | 顺手修掉的一个真 bug（`example/fit.py` 新增 `ctpwa.diagnostics` 导入后桩没跟上） |

### P0-B 怎么接着做（给下一次的自己/别人）

```bash
# 1. 单卡先确认无回归
bash scripts/run_tests_parallel.sh
# 2. 双卡机器上，按文件分批把循环内的 set 换成 DeviceGuard，每批一次：
python3 scripts/cluster/multigpu_parity.py           # 多卡一致性
bash scripts/check_device_calls.sh                   # 棘轮必须下降
# 3. 每批完成后把 scripts/device_call_budget.txt 调小锁住进度
```

判据：`multigpu_parity` 通过 + 门禁全绿 + A/B 指纹（GPU）逐位不变。

---

## 5. 怎么复跑全部证据

```bash
# CPU 门禁（约 8 分钟）
CTPWA_BACKEND=cpu MAX_JOBS=4 python3 setup.py build_ext --inplace
bash scripts/check_device_calls.sh          # 设备棘轮
bash scripts/run_tests_parallel.sh          # 258/0/63

# CUDA 只编译（不需要卡）
CTPWA_BACKEND=cuda TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=4 python3 setup.py build_ext --inplace

# A/B 指纹（CPU 与 GPU 各跑一次，再 --compare）
python3 scripts/ab_fingerprint.py --out doc/parity/fingerprint-xxx.json
python3 scripts/ab_fingerprint.py --compare doc/parity/fingerprint-baseline-v0.3.27.json \
                                          doc/parity/fingerprint-xxx.json

# host-only C++ 单测（衰变描述 / 置换拓扑）
bash tests/cpu_backend/build_and_run.sh
```

---

## 6. 参考

| 文件 | 内容 |
|---|---|
| `doc/modularization-plan.md` | 为什么这么拆、四条"名义存在的边界"、P0–P4 路线 |
| `doc/refactor-progress.md` | 每一步做了什么、踩了什么坑、验证数据 |
| `doc/refactor-release-workflow.md` | 分支模型、六级门禁、A/B oracle、GitLab 迁移 |
| `doc/baseline-cpu-v0.3.27.md` | CPU 基线与门禁口径的由来 |
| `doc/parity/` | A/B 指纹 JSON 与逐值对比文本 |
