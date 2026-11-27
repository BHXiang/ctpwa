# src/ 分区

按 `doc/modularization-plan.md` §7.2 分。**分区是给"以后改哪里"用的导航，不是编译期隔离**——
今天跨区引用不会被编译器拦住（见 plan §7.5 的警告：目录 ≠ 隔离）。

```
config/       Config.cu         Info.cu(DecayInfo)    DecayDesc.cu(Amp2BD)      纯 host
model/        ResModel.cu  Resonance.cu  CustomExpr.cu  JITCustom.cu
device/       DeviceManager.cu  Backend.cpp(BLAS/设备)  JitStubs_cpu.cpp
amplitude/    AmpGen.cu(AmpCalc)  Amplitude.cu  PermTopology.cu  Polarization.cu
parameters/   Parameters.cu     耦合/θ/约束（唯一 name→slot）
fit/          ComputeNLL/ComputeHessian/ComputeResults(device kernel)/ComputeBF
              + HessianContrib/CouplingHessian/FullHessian
              + FitFractions/BatchIntegrals/TruthIntegrals
              + MatrixStats/PhspGram/MatrixErrors
io/           DataReader.cu(.dat/ROOT 读)  PhaseSpace.cu(MC 产生)
output/       ResultWriter.cu(writeResult/writeInterfResult)  Figure.cu  SLAmpsIO.cu
app/          main.cu(pybind)  Initialize.cu  AnalysisMethods.cu
```

头文件按同一分区放在 `include/<分区>/`，`setup.py` 为每个分区加了一条 `-I`。

## 两个容易分错的地方（plan §7.2 原话）

* `ComputeResults.cu` 名字像 output，实际是 **device kernel**（`computeResults` /
  `computeTotalIntensity`）→ `fit/`。
* `Info.cu` 是 **`DecayInfo`（config 层）**（`ChainInfo` 的注释就写着 "Built by DecayInfo,
  consumed by Analysis::calculateAmplitudes"）→ `config/`，**放进 output/ 是分类错误**。

## 今天**还没**满足的硬判据（别把分区当成了完成度）

| 判据 | 现状 |
|---|---|
| `output/` 里不允许出现 `analysis`（plan §7.3） | ❌ 未满足：`output/ResultWriter.cu` 里是 `analysis::writeResult` 的成员定义。要满足必须先有 `FitResult` POD，见 plan §7.3 |
| `device/` 是**唯一** `cuda*` 出口 | 🟡 部分：`cudaSetDevice` 已归零（只有 `include/device/DeviceGuard.h` 与 `Backend.cpp`/`DeviceManager.cu` 允许出现，`scripts/check_device_calls.sh --strict` 守着）；但 `cudaMalloc`/`cudaMemcpy` 等仍散在 `fit/`、`app/` 里 |
| 分区之间不许反向引用 | ❌ 未做检查。`app/` 引用所有分区是**允许**的（它是编排层），`fit/` 引用 `app/` 则不允许——目前没有自动检查 |

把上面三条当成**下一步**，而不是"已经实现"。
