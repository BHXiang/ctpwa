#!/usr/bin/env bash
# GPU 测试门禁（单卡可跑；双卡时额外跑 multigpu parity）
#
# 为什么单独一个脚本：CPU 门禁（scripts/run_tests_parallel.sh）在无卡机器上跑，
# GPU-only 用例全部 skip。真正验证 CUDA 路径需要：
#   1. 用当前卡的架构重编 _cuda.so（sm_86 的产物跑不了 sm_120 的卡）；
#   2. 分批跑，避免单进程占满显存；
#   3. 跳过针对 40GB 卡的用例（test_memory_monitor 需要 ~400 万事件 phsp）。
#
# 用法：
#   bash scripts/run_tests_gpu.sh                 # 自动探测架构 + 全部分批
#   TORCH_CUDA_ARCH_LIST=12.0 bash scripts/run_tests_gpu.sh   # 指定架构
#   SKIP_BUILD=1 bash scripts/run_tests_gpu.sh    # 已有 _cuda.so 时跳过重编
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

ARCH="${TORCH_CUDA_ARCH_LIST:-}"
if [[ -z "$ARCH" ]]; then
    ARCH="$(python3 - <<'PY'
import torch
try:
    cap = torch.cuda.get_device_capability(0)
    print(f"{cap[0]}.{cap[1]}")
except Exception:
    print("8.6")
PY
)"
fi
echo "== GPU 门禁（arch=$ARCH） =="

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    CTPWA_BACKEND=cuda TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-4}" \
        python3 setup.py build_ext --inplace > /tmp/ctpwa_gpu_build.log 2>&1 \
        || { echo "❌ CUDA 构建失败（见 /tmp/ctpwa_gpu_build.log）"; exit 2; }
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CTPWA_DEVICE=cuda CTPWA_QUIET=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd tests

python3 -c "import ctpwa; assert ctpwa.backend()=='cuda', ctpwa.FALLBACK_REASON" || {
    echo "❌ 后端不是 cuda（回退原因见上）"; exit 2; }
echo "✅ backend=cuda"

# 分批：每批控制在单卡显存与 15 分钟内
BATCHES=(
  "数值核心:test_numerical.py test_hessian_diagnostics.py test_hessian_bwr_spec.py"
  "精度档:test_float_mode.py test_precision.py test_native_float.py"
  "份额/矩阵/写出:test_efficiency.py test_identical.py test_matrix_errors.py test_matrix_validation.py test_write_result_robustness.py test_data_weights_write.py"
  "模型/约束:test_constraints.py test_barrier.py test_custom.py test_custom_models.py test_custom_q0.py test_interp.py test_nested_resonances.py"
  "phsp/干涉:test_free_phsp.py test_free_phsp_matrix.py test_static_budget.py test_interference.py test_legends.py test_plot_obs.py"
  "端到端/其它:test_endtoend.py test_ownership.py test_structure.py test_reparam.py test_fit_mode.py test_theta_propagation.py test_fd_utils.py test_mod_kernel_local_mem.py test_trans_host.py test_deep_chains.py test_mc.py test_mc_amplitude.py test_read_interference_example.py"
)

fails=0
for entry in "${BATCHES[@]}"; do
    name="${entry%%:*}"
    files="${entry#*:}"
    printf '%-16s ' "$name"
    out=$(timeout 1800 python3 -m pytest $files -q --no-header -p no:cacheprovider --tb=line 2>&1)
    rc=$?
    line=$(echo "$out" | grep -E "passed|failed|error" | tail -1)
    echo "$line"
    if [[ $rc -ne 0 ]]; then
        fails=$((fails+1))
        echo "$out" | tail -20
    fi
done

# 已知跳过：test_memory_monitor 针对 A100-40GB 的 400 万事件 phsp，小卡上无意义
echo
echo "未跑：test_memory_monitor（需 40GB 级显存）；test_cpu_*（CPU 构建专属）"
if [[ $fails -ne 0 ]]; then
    echo "❌ GPU 门禁失败：$fails 批有问题"
    exit 1
fi
echo "✅ GPU 门禁通过"
