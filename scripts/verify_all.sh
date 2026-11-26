#!/usr/bin/env bash
# 一次跑完全部验收（CPU 门禁 + 设备纪律 + A/B 数值对照）
#
# 这是 doc/refactor-report.md 里那些证据的一键复现。默认只跑不需要 GPU 的部分；
# 加 --with-gpu 时在最后追加 GPU 门禁与 GPU A/B（需要真卡）。
#
# 用法：
#   bash scripts/verify_all.sh              # CPU：门禁 + 设备 + A/B
#   bash scripts/verify_all.sh --with-gpu   # 再加 GPU 门禁与 GPU A/B
#   SKIP_BUILD=1 bash scripts/verify_all.sh # 复用已有 .so
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WITH_GPU=0
[[ "${1:-}" == "--with-gpu" ]] && WITH_GPU=1

BASELINE_CPU="doc/parity/fingerprint-baseline-v0.3.27.json"
BASELINE_GPU="doc/parity/fingerprint-baseline-gpu-sm120.json"
OUTDIR="${OUTDIR:-$(mktemp -d)}"
rc_total=0

step() { printf '\n=== %s ===\n' "$1"; }

# ---------- 1. CPU 构建 ----------
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    step "CPU 构建"
    CTPWA_BACKEND=cpu MAX_JOBS="${MAX_JOBS:-4}" python3 setup.py build_ext --inplace \
        > /tmp/ctpwa_cpu_build.log 2>&1 \
        || { echo "❌ CPU 构建失败（/tmp/ctpwa_cpu_build.log）"; exit 2; }
    echo "ok"
fi

# ---------- 2. 设备纪律（必须 0 违规）----------
step "设备纪律（scripts/check_device_calls.sh --strict）"
if bash scripts/check_device_calls.sh --strict > /tmp/ctpwa_dev.log 2>&1; then
    tail -2 /tmp/ctpwa_dev.log
else
    cat /tmp/ctpwa_dev.log; rc_total=1
fi

# ---------- 3. CPU 门禁 ----------
step "CPU 测试门禁"
if bash scripts/run_tests_parallel.sh > /tmp/ctpwa_cpu_tests.log 2>&1; then
    grep -E "^合计|门禁通过" /tmp/ctpwa_cpu_tests.log | tail -2
else
    tail -30 /tmp/ctpwa_cpu_tests.log; rc_total=1
fi

# ---------- 4. CPU A/B ----------
step "CPU A/B 指纹 vs v0.3.27"
( cd tests && python3 ../scripts/ab_fingerprint.py --out "${OUTDIR}/fp_cpu.json" ) >/dev/null 2>&1
if python3 scripts/ab_fingerprint.py --compare "$BASELINE_CPU" "${OUTDIR}/fp_cpu.json" \
        > "${OUTDIR}/ab_cpu.txt" 2>&1; then
    tail -2 "${OUTDIR}/ab_cpu.txt"
else
    cat "${OUTDIR}/ab_cpu.txt"; rc_total=1
fi

# ---------- 5. host-only C++ 单测 ----------
step "host-only C++ 单测（衰变描述 / 置换拓扑）"
if bash tests/cpu_backend/build_and_run.sh > /tmp/ctpwa_host_tests.log 2>&1; then
    grep -E "ALL PASSED|all checks passed" /tmp/ctpwa_host_tests.log | tail -2
else
    tail -20 /tmp/ctpwa_host_tests.log; rc_total=1
fi

# ---------- 6. GPU（可选）----------
if [[ $WITH_GPU -eq 1 ]]; then
    step "GPU 门禁"
    if bash scripts/run_tests_gpu.sh > /tmp/ctpwa_gpu_tests.log 2>&1; then
        grep -E "✅" /tmp/ctpwa_gpu_tests.log | tail -3
    else
        tail -30 /tmp/ctpwa_gpu_tests.log; rc_total=1
    fi

    step "GPU A/B 指纹 vs v0.3.27（同架构基线）"
    ( cd tests && CTPWA_DEVICE=cuda python3 ../scripts/ab_fingerprint.py \
          --out "${OUTDIR}/fp_gpu.json" ) >/dev/null 2>&1
    if [[ -f "$BASELINE_GPU" ]] && python3 scripts/ab_fingerprint.py \
            --compare "$BASELINE_GPU" "${OUTDIR}/fp_gpu.json" > "${OUTDIR}/ab_gpu.txt" 2>&1; then
        tail -2 "${OUTDIR}/ab_gpu.txt"
    else
        echo "（无 GPU 基线或对比失败：先在同一张卡上生成 baseline 指纹）"
        [[ -f "${OUTDIR}/ab_gpu.txt" ]] && cat "${OUTDIR}/ab_gpu.txt"
        rc_total=1
    fi
fi

echo
echo "产物目录: ${OUTDIR}"
if [[ $rc_total -ne 0 ]]; then
    echo "❌ 有检查未通过"
    exit 1
fi
echo "✅ 全部检查通过"
