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
PY=/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/ctpwa/bin/python3
export LD_LIBRARY_PATH=/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/ctpwa/lib:/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/envs/ctpwa/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-12.9/lib64:/usr/local/cuda-13.2/lib64:/usr/lib64/root:/besfs10/groups/jpsi/jpsigroup/user/bhxiang/miniconda3/lib
export PYTHONPATH="$CTPWA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# probe 脚本在 repo scripts/cluster/；cd tests 后须显式指回。
# 注意: SLURM 会把 sbatch 脚本复制到 /var/spool/slurm/... 执行，$0 指向 spool
# 副本，不能 dirname $0——一律从硬编码 CTPWA_ROOT 推导。
HERE="$CTPWA_ROOT/scripts/cluster"

echo "=== 开始: $(date) ==="
# git pull 只更新源码，不会更新已加载的二进制；统一作业必须先重编，
# 避免拿几个月前的 _cuda.so 验证新修复。
cd "$CTPWA_ROOT"
echo "===== 0) 构建 CUDA 后端 ====="
CTPWA_BACKEND=cuda MAX_JOBS=4 "$PY" setup.py build_ext --inplace
BUILD_RC=$?
if [ $BUILD_RC -ne 0 ]; then
  echo "!!! CUDA 构建失败 (rc=$BUILD_RC)"
  exit $BUILD_RC
fi

cd "$CTPWA_ROOT/tests"
echo "--- package _cuda.so: $(ls -l --time-style=+%F\ %T $CTPWA_ROOT/ctpwa/_cuda.so | awk '{print $6, $7}')"
$PY -c 'import ctpwa; print("--- loaded backend:", ctpwa.__file__, ctpwa.build_info())'
nvidia-smi --query-gpu=index,name --format=csv

echo; echo "===== 1) 全量 pytest ====="
$PY -m pytest . -q --tb=short 2>&1

echo; echo "===== 2) 三档数值一致性（simple free-θ + interp fixed-θ）====="
$PY -m pytest test_float_mode.py -q --tb=short 2>&1

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

# 多卡一致性必须跨进程：pytest 在进程内跑，改不了 CUDA_VISIBLE_DEVICES。
# 覆盖 per-GPU 分支：事件切分后的 d_static / d_prop6 / d_dF 行数、σ 置换表副本。
# 注: 上面 `pytest .` 已在 2 卡上跑过，所以 per-GPU 代码路径本身有覆盖；
#     这一步补的是"1 卡 与 N 卡给出同一 NLL/梯度"的**断言**。
#     (只有 configs/static_nodes.yml 会真的建静态表，其余 config 的 static=0)
echo; echo "===== 5) 多卡一致性（1 卡 vs N 卡，NLL + 全梯度）====="
$PY "$HERE/multigpu_parity.py" 2>&1
MGP_RC=$?
if [ $MGP_RC -ne 0 ]; then
  echo "!!! 多卡一致性未通过 (rc=$MGP_RC) —— 不要发布"
else
  echo "多卡一致性 ✓"
fi

# NLL/梯度一致并不能发现派生量错误：加权 Gram、truth 分批、空 GPU 分片、
# writeResult 的 N/W/矩阵/误差都要单独做 1卡 vs N卡断言。
echo; echo "===== 5.5) 多卡派生量（加权 rec/gen + err、streaming、n_batch<n_gpus、内部 Hessian）====="
$PY "$HERE/multigpu_results.py" 2>&1
MGR_RC=$?
if [ $MGR_RC -ne 0 ]; then
  echo "!!! 多卡派生量未通过 (rc=$MGR_RC) —— 不要发布"
  exit $MGR_RC
fi
echo "多卡派生量一致性 ✓"

# 两段都跑完后仍要保留 MGP 的失败状态，不能被派生量通过所掩盖。
if [ $MGP_RC -ne 0 ]; then
  exit $MGP_RC
fi

echo "=== 结束: $(date) ==="
