#!/usr/bin/env bash
# 原生 CTPWA_COMPLEX=float 构建的回归门禁（默认 run_tests.sh 只编 double）。
#
#   bash scripts/run_tests_float.sh
#
# 注意：脚本会把 ctpwa/_cuda.so 重新编成 float 版（覆盖现有 double 产物）。
# 跑完若要回到 double：CTPWA_BACKEND=cuda MAX_JOBS=4 python3 setup.py build_ext --inplace
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v nvcc >/dev/null 2>&1; then
    echo "SKIP: 没有 nvcc，无法构建 float CUDA 后端" >&2
    exit 0
fi
if ! python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "SKIP: 没有可用 GPU（float .so 只能在 CUDA 后端运行）" >&2
    exit 0
fi

echo "=== 构建 CTPWA_COMPLEX=float / CUDA 后端 ==="
CTPWA_BACKEND=cuda CTPWA_COMPLEX=float CTPWA_GLIBCXX_ASSERTIONS=1 \
    MAX_JOBS="${MAX_JOBS:-4}" python3 setup.py build_ext --inplace

echo "=== float 回归（原生 float 上转/越界 + free_phsp 矩阵）==="
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
cd tests
CTPWA_DEVICE=cuda python3 -m pytest \
    test_native_float.py test_free_phsp_matrix.py test_free_phsp.py \
    test_matrix_errors.py test_write_result_robustness.py \
    test_read_interference_example.py -v --tb=short
