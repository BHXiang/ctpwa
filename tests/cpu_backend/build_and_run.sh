#!/usr/bin/env bash
# build_and_run.sh — 构建并运行 CPU 后端基础层单测
#
# 只依赖 g++ + OpenMP + torch 的头文件（不需要 CUDA、不需要 GPU、不需要 BLAS）。
# 用法:
#   bash tests/cpu_backend/build_and_run.sh
#   OMP_NUM_THREADS=1 bash tests/cpu_backend/build_and_run.sh   # 验证与线程数无关

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${CXX:-g++}"
OUT="${ROOT}/build/cpu_backend"

# torch 头文件路径（ComplexType.h 用 torch::kComplexDouble 等常量）
TORCH_INC="$(python3 -c 'import torch.utils.cpp_extension as c; print(" ".join("-I"+p for p in c.include_paths()))' 2>/dev/null)"

mkdir -p "${OUT}"

echo "[build] ${CXX} -> ${OUT}/test_backend"
# shellcheck disable=SC2086
"${CXX}" -std=c++17 -O2 -fopenmp -DCTPWA_BACKEND_CPU \
    -Wall -Wextra -Wno-unused-parameter \
    -DCTPWA_DOUBLE_COMPLEX \
    -I"${ROOT}/include" ${TORCH_INC} \
    "${ROOT}/src/Backend.cpp" \
    "${ROOT}/tests/cpu_backend/test_backend.cpp" \
    -o "${OUT}/test_backend"

echo "[run] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<default>}"
"${OUT}/test_backend"

# ---- 衰变描述类型（P3）：纯 host 单测，不链接 Backend.cpp、不需要 torch ----
# 用 -x c++ 让 g++ 认 .cu 扩展名（与 setup.py 的 CPU 影子源文件同一手法）。
echo "[build] ${CXX} -> ${OUT}/test_decay_desc"
"${CXX}" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter \
    -I"${ROOT}/include" \
    -x c++ "${ROOT}/src/config/DecayDesc.cu" -x none \
    "${ROOT}/tests/cpu_backend/test_decay_desc.cpp" \
    -o "${OUT}/test_decay_desc"

echo "[run] test_decay_desc"
"${OUT}/test_decay_desc"
