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
