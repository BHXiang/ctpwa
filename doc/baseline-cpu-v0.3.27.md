# CPU 构建基线（v0.3.27，重构前）

> 用途：`refactor/modular` 分支每个阶段的验收基准。判据是**不新增失败**，
> 而不是"全绿"——因为本基线本身就不是全绿的（见 §3）。
>
> 采集时间 2026-11-27 03:xx，commit `030baf5`（v0.3.27）+ 分支内 conftest 修正。

## 1. 环境

| 项 | 值 |
|---|---|
| 构建 | `CTPWA_BACKEND=cpu MAX_JOBS=3 CTPWA_GLIBCXX_ASSERTIONS=1 python3 setup.py build_ext --inplace` |
| 产物 | `ctpwa/_cpu.so` 3.6 MB（`build_info()`: `backend=cpu, precision=double, complex_bytes=16`） |
| 机器 | 16 核 / 15 GB，无 GPU（`torch.cuda.is_available() = False`） |
| torch | 2.12.0+cu132（本机无驱动，走 CPU） |
| ROOT | conda env `ctpwa`（`root-config` 可见） |
| 跑法 | 逐文件 `pytest <file> -q --tb=no`，每文件 240 s 上限（整目录单次跑会被一个慢用例拖住） |

## 2. 总计

| passed | failed | skipped | 文件数 |
|---|---|---|---|
| **183** | **8** → 修 P0-D 后 **7** | 26（另有 7 个文件整文件 skip，解析器未计入） | 41 |

耗时：全套约 6 分钟（逐文件串行，OMP_NUM_THREADS=4）。

## 3. 已知失败（重构期间必须保持**不变**）

| 文件 | 数量 | 现象 | 判断 |
|---|---|---|---|
| `test_float_mode.py` | 4 failed | `precision: float` 相关 | 预期：float 档是 GPU/混合精度路径，CPU 构建（double）不具备 |
| `test_hessian_bwr_spec.py` | 2 failed | BWR 解析特化 vs 字节码对照 | 预期：`bwrSpecEval` 闭式特化只接了 CUDA 路径，CPU 走 `not_implemented` 或退化解 |
| `test_memory_monitor.py` | 1 failed | 显存监控 | 预期：CPU 构建无显存概念 |
| ~~`test_structure.py`~~ | ~~1 failed~~ | ~~结构/设备探测~~ | ✅ 已修（见 P0-D：CPU 伪设备 cc=0 不该判失败），现 24 passed |
| `test_reparam.py` | 4 **errors** | `ModuleNotFoundError: ctpwa.diagnostics; 'ctpwa' is not a package` | **测试自身的 bug**：fixture 用 `types.ModuleType("ctpwa")` 打桩后加载 `example/fit.py`，但 `example/fit.py:28` 新增了 `from ctpwa.diagnostics import ...`，桩没跟上 |

## 4. 关键发现：CPU 门禁目前**不验证数值**

`test_numerical.py` 的 38 个用例（NLL golden、梯度 vs FD、Hessian 对称/FD）
**全部 skip**——它们都请求 `device` fixture，而无卡时 skip：

```
SKIPPED [2] test_numerical.py:333: 需要 CUDA GPU（当前无可用设备）；CPU 构建下跳过
...
38 skipped in 0.04s
```

这意味着一份"CPU 全绿"的 CI **对重构的数值正确性零保障**。要让它有意义，
必须先把 `test_numerical` 改成后端无关（用 `dev` fixture 而不是 `device`），
并确认 golden 值在 CPU（double）上与 GPU 一致到既有判据（NLL 1e-11）。
这是重构之前该做的一件独立小活，见 §6。

## 5. 重构期间的验收口径

每个阶段（P0/P1/…）完成后：

```bash
CTPWA_BACKEND=cpu MAX_JOBS=3 python3 setup.py build_ext --inplace
bash scripts/run_tests.sh --no-build        # 或逐文件循环（本文件的跑法）
```

| 判据 | 要求 |
|---|---|
| passed | ≥ 183（只许增，不许减） |
| failed | 恰好仍是上面那 8 个（不多一个） |
| 新增 error | 0 |
| `scripts/check_device_calls.sh` | 违规数只许下降（P0 目标 → 0） |

## 6. 建议的补强（不阻塞重构，但影响门禁价值）

1. `test_reparam` 的桩补上 `ctpwa.diagnostics`（4 个 error 立刻消失）。
2. `test_numerical` 从 `device` 换成后端无关的 `dev`，让 38 个数值用例在 CPU 上跑起来。
3. §3 里四个"预期失败"逐个加 `pytest.mark.skipif(CPU_BUILD)` 并注明原因，
   使 CPU 门禁变成**真全绿**——这样"新增失败"就等价于"红灯"，不需要人肉比对这张表。
