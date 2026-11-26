#!/bin/bash
# 对单个 pytest 用例跑 CUDA memcheck；避免全量 pytest 的首个非法访问污染后续进程。
# 默认目标来自 log_validate_all.out 中当前最早的 CUDA 非法访问。
# 用法：
#   sbatch scripts/cluster/job_sanitize.sh
#   sbatch --export=ALL,SANITIZE_TEST='test_xxx.py::test_yyy' scripts/cluster/job_sanitize.sh
# 可选：SANITIZE_BUILD=1 时带 -lineinfo 强制重编，以获得 CUDA 源码行号。
# 输出：提交时当前目录的 log_sanitize.out。
#SBATCH --partition=gpupwa
#SBATCH --qos=pwadebug
#SBATCH --account=gpupwa
#SBATCH --job-name=ctpwa-sanitize
#SBATCH --output=./log_sanitize.out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:a100:2

set -uo pipefail

CTPWA_ROOT=/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa
ENV=/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/ctpwa
PY="$ENV/bin/python3"
SAN=${COMPUTE_SANITIZER:-/usr/local/cuda-12.9/bin/compute-sanitizer}
TEST_TARGET=${SANITIZE_TEST:-test_free_phsp_matrix.py::test_free_phsp_weighted_matrix_matches_resident}

export LD_LIBRARY_PATH="$ENV/lib:$ENV/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-12.9/lib64:/usr/local/cuda-13.2/lib64:/usr/lib64/root:/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$CTPWA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1

cd "$CTPWA_ROOT" || exit 2
echo "=== sanitizer 开始: $(date) ==="
echo "目标: $TEST_TARGET"
if [ "${SANITIZE_BUILD:-0}" = "1" ]; then
    echo "===== 可选构建（-lineinfo，不用 -G）====="
    CTPWA_BACKEND=cuda CTPWA_CUDA_LINEINFO=1 MAX_JOBS=4 \
        "$PY" setup.py build_ext --inplace --force
    BUILD_RC=$?
    if [ "$BUILD_RC" -ne 0 ]; then
        echo "!!! CUDA 构建失败 (rc=$BUILD_RC)"
        exit "$BUILD_RC"
    fi
else
    echo "===== 跳过构建，直接使用当前 ctpwa/_cuda.so ====="
fi

cd "$CTPWA_ROOT/tests" || exit 2
"$PY" -c 'import ctpwa; print("backend:", ctpwa.__file__, ctpwa.build_info())'
nvidia-smi --query-gpu=index,name --format=csv

echo "===== compute-sanitizer memcheck ====="
# 新进程只跑一个用例；CUDA_LAUNCH_BLOCKING 帮助把宿主栈靠近触发点，
# memcheck 报告中的 kernel/地址/源码行才是主要定位依据。
CUDA_LAUNCH_BLOCKING=1 "$SAN" \
    --tool memcheck \
    --error-exitcode 99 \
    --report-api-errors all \
    --print-limit 20 \
    "$PY" -m pytest "$TEST_TARGET" -q -s --tb=long 2>&1
SAN_RC=$?
echo "===== sanitizer 结束: $(date), rc=$SAN_RC ====="
echo "99 = sanitizer 检出设备内存/API 错误；其他非零见上方 pytest/构建输出。"
exit "$SAN_RC"
