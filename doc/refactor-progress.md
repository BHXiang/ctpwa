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
| P2 DevBuffer | ⬜ | — |
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

## 当前门禁口径

```bash
CTPWA_BACKEND=cpu MAX_JOBS=4 python3 setup.py build_ext --inplace
bash scripts/check_device_calls.sh          # 违规数只许下降
bash tmp/run_baseline.sh                    # 逐文件跑（整目录单次跑会被慢用例拖住）
diff 基线表 当前表                            # 判据：passed/failed/skipped 逐文件一致
```

基线：`doc/baseline-cpu-v0.3.27.md`（183 / 8 / 26，8 个失败都是 CPU 构建固有的）。
