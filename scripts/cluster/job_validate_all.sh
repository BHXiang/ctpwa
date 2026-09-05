#!/bin/bash
# 统一验证作业（精度档位 + slamp + 多卡，集群专用）
# 前置: 集群 CTPWA_ROOT 为 ctpwa git 仓库（git pull 同步代码/tests/本脚本）。
# 用法（集群上）:
#   cd $CTPWA_ROOT && git pull          # 同步
#   export CONFIG=<真实分析 config.yml> # 可选: 追加真实分析单步 NLL
#   sbatch scripts/cluster/job_validate_all.sh
# 输出 ./log_validate_all.out（sbatch 时 cwd）
#SBATCH --partition=gpu
#SBATCH --qos=pwanormal
#SBATCH --account=gpupwa
#SBATCH --job-name=ctpwa-valall
#SBATCH --output=./log_validate_all.out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:v100:2

set -uo pipefail

CTPWA_ROOT=/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa
PY=/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/test2/bin/python3
export LD_LIBRARY_PATH=/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/test2/lib:/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/test2/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-12.9/lib64:/usr/local/cuda-13.2/lib64:/usr/lib64/root:/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/lib
export PYTHONPATH="$CTPWA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# probe 脚本在 repo scripts/cluster/；cd tests 后须显式指回。
# 注意: SLURM 会把 sbatch 脚本复制到 /var/spool/slurm/... 执行，$0 指向 spool
# 副本，不能 dirname $0——一律从硬编码 CTPWA_ROOT 推导。
HERE="$CTPWA_ROOT/scripts/cluster"

cd "$CTPWA_ROOT/tests"
echo "=== 开始: $(date) ==="
echo "--- .so: $(ls -l --time-style=+%F\ %T $CTPWA_ROOT/ctpwa.so | awk '{print $6, $7}')"
nvidia-smi --query-gpu=index,name --format=csv

echo; echo "===== 1) 全量 pytest ====="
$PY -m pytest . -q --tb=short 2>&1 | tail -5

echo; echo "===== 2) 三档数值一致性（simple free-θ + interp fixed-θ）====="
$PY -m pytest test_float_mode.py -q --tb=short 2>&1 | tail -3

echo; echo "===== 3) 显存: 三档 mem_probe（auto/hybrid 缺省、float、double）====="
# simple: 每档一次（Float 档跳过 hessian 段; hybrid/double 跑全流程）
for tag in auto float double; do
  cfg="configs/simple.yml"
  [ "$tag" = "auto" ] || { printf "precision: %s\n" "$tag" > _gen_mem_$tag.yml; cat configs/simple.yml >> _gen_mem_$tag.yml; cfg="_gen_mem_$tag.yml"; }
  if [ "$tag" = "float" ]; then
    MEM_NO_HESSIAN=1 $PY "$HERE/mem_probe.py" "$cfg" "mem_$tag"
  else
    $PY "$HERE/mem_probe.py" "$cfg" "mem_$tag"
  fi
done

echo; echo "===== 3.5) 三档性能基准（simple; float/hybrid/double 各段中位耗时+显存）====="
PERF_N_NLL=10 PERF_N_HESS=3 $PY "$HERE/perf_probe.py" configs/simple.yml perf_simple

if [ -n "${CONFIG:-}" ]; then
  echo; echo "===== 4) 真实分析三档单步 NLL（CONFIG=$CONFIG）====="
  for tag in float double; do
    printf "precision: %s\n" "$tag" > _gen_real_$tag.yml
    cat "$CONFIG" >> _gen_real_$tag.yml
    $PY - <<EOF
import torch, sys
sys.path.insert(0, ".")
from conftest import make_params
import ctpwa
ana = ctpwa.analysis("_gen_real_$tag.yml")
p = make_params(ana, torch.device("cuda:0"))
nll = float(ana.getNLL(p))
print(f"[$tag] 真实分析 NLL = {nll:.6f}")
EOF
  done
fi

echo "=== 结束: $(date) ==="
