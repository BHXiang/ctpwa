# include/ 分区

与 `src/` 一一对应（`config/ model/ device/ amplitude/ parameters/ fit/ io/ output/ app/`），
`setup.py` 为每个分区加了一条 `-I`，所以源码里的 `#include <Foo.cuh>` 不需要写路径。

两个例外：

* `include/cpu_shims/` —— CPU 后端对 `cuda_runtime.h` / `cublas_v2.h` / thrust / cub 的替身，
  **必须排在所有 -I 之前**（否则真 CUDA 头会被先找到）。
* `include/device/` —— `DeviceGuard.h`（设备切换唯一入口）、`DevBuffer.h`（设备缓冲所有权）、
  `Backend.h`、`ComplexType.h`/`PrecCpx.h`/`MixedPrecision.h`（精度档位胶水）。
  其中 `DeviceGuard.h` 是**唯一**允许出现 `cudaSetDevice` 的文件（由
  `scripts/check_device_calls.sh --strict` 强制）。
