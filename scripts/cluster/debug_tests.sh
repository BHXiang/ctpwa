#!/bin/bash
# 单卡调试作业：把 pytest 参数原样转发（默认跑 test_precision.py）。
# 用于定位特定测试失败（--tb=long 完整 traceback）或临时跑子集。
# 用法（集群上，先 git pull）:
#   sbatch scripts/cluster/debug_tests.sh test_precision.py::test_fit_fractions_parity --tb=long
#   sbatch scripts/cluster/debug_tests.sh -k "float"          # 任意 pytest 参数均可
# 输出 ./log_debug_tests.out（sbatch 时 cwd）
#SBATCH --partition=gpu
#SBATCH --qos=pwanormal
#SBATCH --account=gpupwa
#SBATCH --job-name=ctpwa-dbg
#SBATCH --output=./log_debug_tests.out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:v100:1

set -uo pipefail

CTPWA_ROOT=/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa
PY=/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/test2/bin/python3
export LD_LIBRARY_PATH=/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/test2/lib:/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/test2/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-12.9/lib64:/usr/local/cuda-13.2/lib64:/usr/lib64/root:/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/lib
export PYTHONPATH="$CTPWA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$CTPWA_ROOT/tests"
echo "=== 开始: $(date) | pytest args: ${*:-（无，默认 test_precision.py）} ==="
ls -l configs/eff2.yml data/test_data.dat data/test_phsp.dat data/test_sideband.dat 2>&1

if [ $# -eq 0 ]; then
    set -- test_precision.py
fi
$PY -m pytest "$@" -q --tb=long 2>&1 | tail -80
echo "=== 结束: $(date) ==="
