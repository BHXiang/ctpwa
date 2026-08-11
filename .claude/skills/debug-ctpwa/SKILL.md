---
name: debug-ctpwa
description: |
  Debug CTPWA partial-wave analysis. Use for gradient/Hessian issues, model
  development, compilation errors, or fitting problems. Covers the symbolic
  differentiation architecture, model zoo, build system, and pytest suite.
metadata:
  type: project
---

# Debug CTPWA — 分波分析程序

## 架构概览

```
config.yml → ConfigParser → DecayInfo → CouplingMatrixBuilder
                                    ↓
                          CouplingMatrixResult
                                    ↓
                          AmpCalc::addBlock()
                            ├─ buildModelAST → deriv → simplify → compileNode → aux[] (主机)
                            └─ 存入 DeviceResonance
                                    ↓
NLLFunction::forward(params)
  ├─ reComputeAmps(theta)
  │   └─ computeCustomAmpsKernel → evalCustomAll(aux, ...)  (GPU, 逐事件)
  │       或 interpEval(aux, ...)  (Interp 模型)
  ├─ computeFactorNLL(v, A) → NLL
  └─ computeResonanceGradient → dL/dθ
```

**核心**: 所有模型统一走符号微分。`buildModelAST` 在主机端构建模型 AST → `deriv` 符号微分 → `compileNode` 编译为字节码 `aux[]`。GPU kernel 逐事件执行字节码求值 F, dF, d²F。

## 关键文件

| 文件 | 职责 |
|---|---|
| `include/SymbolicDiff.cuh` | Node/deriv/simplify/buildModelAST 声明, CompositeId 枚举 |
| `src/ResModel.cu` | buildModelAST (各模型 AST 构建) + modelDeriv (导数规则) |
| `src/CustomExpr.cu` | evalCustomSeg/evalCustomAll (字节码解释器), Custom DSL 编译 |
| `src/AmpGen.cu` | computeCustomAmpsKernel, interpEval, reComputeAmps, Hessian |
| `include/Resonance.cuh` | ResModelType 枚举, DeviceResonance, ResonanceModel 基类 |
| `src/Resonance.cu` | 模型注册表 (BWR/BW/Flatte/GS/ONE/Interp/Hist/Custom) |

## 模型总览

| 模型 | config 名 | 参数 | 导数来源 | 状态 |
|---|---|---|---|---|
| Relativistic BW | `BWR` | [m0, w0] | modelDeriv(MODEL_BWR) + MODEL_BF | ✅ |
| Non-relativistic BW | `BW` | [m0, w0] | modelDeriv(MODEL_BW) | ✅ |
| Flatté | `Flatte` | [m0, g1..gn] | 分解为 RHO_RE/IM + 算术 | ✅ |
| Gounaris-Sakurai | `GS` | [m0, Γ0] | 分解为 BREAKUP_Q0 + 算术 + log | ✅ |
| Blatt-Weisskopf only | `ONE` | [ ] | MODEL_BF (无自由参数) | ✅ |
| 统一插值 | `Interp` | [ ] | interpEval (直接轨) | ✅ |
| 直方图 | `Hist` | [ ] | lookupHistTable (直接轨) | ⚠️ |
| 自定义表达式 | `Custom` | 用户定义 | deriv(用户 AST) | ✅ |

Interp 支持 `method: hist | linear | spline`。Custom DSL 可用变量: `m, q, q0, L, d, md1, md2, p1p, p1e, p1costheta, p1phi, p2p, p2e, p2costheta, p2phi`。

## 编译

```bash
# 本地编译 (MAX_JOBS 控制并行度，避免内存不足)
MAX_JOBS=2 python3 setup.py build_ext --inplace

# 内存不足时减少 MAX_JOBS (默认 4 路 ninja + nvcc，峰值 ~20GB)
MAX_JOBS=1 python3 setup.py build_ext --inplace

# 只改 .cu 文件时，增量编译很快 (~30s)
# 改 .cuh 头文件会触发全量重编
```

**关键**: `MAX_JOBS` 控制 ninja 并行编译数。`nvcc` 单文件编译峰值内存 ~4-6GB（模板展开 + ptxas）。`MAX_JOBS=4` 时峰值 ~20GB。内存不足时设 `MAX_JOBS=1`，nvidia-smi 监控 GPU 内存。

**注意**: `ForceBuildExtension` 会 `rm -rf build/`。日常开发建议保留 `build/` 目录做增量编译。

## 测试

```bash
cd tests && python3 -m pytest . -v --tb=short    # 全量 (~50s)
python3 -m pytest . -k "flatte or gs" -v         # 只跑特定模型
python3 -m pytest test_numerical.py::test_gradient_vs_fd -v
```

测试配置: `simple, no_trans, with_trans, flatte, gs, interp, custom, custom_bw, ident2/3`

**新增模型验证模板**:
1. 加 `tests/configs/xxx.yml`
2. 在 `test_numerical.py` 的 `test_gradient_vs_fd` parametrize 中加名字
3. 跑梯度 vs FD: `pytest test_numerical.py::test_gradient_vs_fd -k xxx`

## 调试

### 梯度/导数调试

```python
import torch, ctpwa, pathlib
ana = ctpwa.analysis("configs/xxx.yml")
params = make_params(ana, 'cuda').requires_grad_(True)
nll = ana.getNLL(params)
grad_ad = torch.autograd.grad(nll, params)[0]

# FD 验证
p = params.detach().cpu().double()
for i in range(len(p)):
    h = max(abs(p[i])*1e-3, 5e-4)
    pp, pm = p.clone(), p.clone()
    pp[i] += h; pm[i] -= h
    fd = (ana.getNLL(pp.cuda())-ana.getNLL(pm.cuda()))/(2*h)
    rel = abs(grad_ad[i]-fd)/max(abs(grad_ad[i]),1e-6)
    print(f"[{i}] AD={grad_ad[i]:.4e} FD={fd:.4e} rel={rel:.2e}")
```

### Kernel 内部调试

```cpp
// printf 在 kernel 中可用 (sm_120+)
if (event_idx == 0 && sl_idx == 0)
    printf("[DBG] mm=%.4f q0=%.4f F=(%.6f,%.6f)\n", mm, q0, Fr, Fi);
```

### aux 字节码 dump

添加临时 printf 到 `evalCustomSeg` 或 `buildModelAST` 末尾。

### 常见陷阱

- **改 .cuh 没触发重编**: `ForceBuildExtension` 每次 `rm -rf build`，改 .cuh → touch 所有 .cu → 全量重编
- **.so 没更新**: 检查 `ls -l ctpwa.so` 时间戳。`pip install -e .` 可能失败但不报错
- **GPU kernel 不报错但结果错**: 用 `cuda-memcheck python3 -c "..."` 检查越界
- **参数不生效**: reComputeAmps 是按需调用的。确认 `getNLL` 内部调了 `reComputeAmps`
- **Hessian 不对称**: 先验证梯度正确，再查 Hessian。常见原因: Stage 2/3/4 用错 cas

## 发布

```bash
git tag vX.Y.Z && git push origin vX.Y.Z   # 触发 CI → PyPI
```

本地不发 wheel (glibc 版本问题)。发布前门禁: `bash scripts/run_tests.sh`
