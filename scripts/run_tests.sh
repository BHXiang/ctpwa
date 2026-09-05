#!/usr/bin/env bash
# ctpwa 测试门禁（发布前必须通过）。
#   bash scripts/run_tests.sh              # 编译 + 全量测试
#   bash scripts/run_tests.sh --no-build   # 跳过编译，只跑测试
#   bash scripts/run_tests.sh -k "L0 or L1"  # 其余参数透传 pytest
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD=true
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-build) BUILD=false ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

echo "=========================================="
echo " ctpwa 测试门禁"
echo "=========================================="

if $BUILD; then
    echo "[1/3] 编译..."
    MAX_JOBS=4 python3 setup.py build_ext --inplace 2>&1 | tail -3
else
    echo "[1/3] 跳过编译 (--no-build)"
fi

echo "[2/3] 运行测试 (pytest tests/ ${EXTRA_ARGS[*]:-})..."
# build_ext --inplace 的 ctpwa.so 在仓库根；cd tests 后需 PYTHONPATH 指回仓库根
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
cd tests
python3 -m pytest . "${EXTRA_ARGS[@]}" -v --tb=short

echo "[3/3] 测试全部通过 ✅"
