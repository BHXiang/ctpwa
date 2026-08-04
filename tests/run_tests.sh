#!/usr/bin/env bash
# ctpwa 测试一键脚本：编译 + 全部测试
# 用法:
#   bash tests/run_tests.sh                # 编译 + 全量测试
#   bash tests/run_tests.sh --no-build     # 跳过编译，直接跑测试（已编译过时）
#   bash tests/run_tests.sh -k L0          # 只跑 L0（结构测试）
#   bash tests/run_tests.sh -k "L0 or L1"  # 只跑 L0+L1
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

if $BUILD; then
    echo "=========================================="
    echo "[1/2] 编译 ctpwa..."
    echo "=========================================="
    pip install -e . --no-build-isolation 2>&1 | tail -3
fi

echo "=========================================="
echo "[2/2] 运行测试 (pytest tests/ ${EXTRA_ARGS[*]:-})"
echo "=========================================="
cd tests
python3 -m pytest . "${EXTRA_ARGS[@]}" -v --tb=short
