#!/usr/bin/env bash
# 发布门禁：ctpwa 全量测试（编译 + L0 + L1 + L2）
# 发布前必须通过。用法:
#   bash scripts/run_tests.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo " ctpwa 测试门禁"
echo "=========================================="

echo "[1/3] 编译..."
pip install -e . --no-build-isolation 2>&1 | tail -3

echo "[2/3] 运行测试..."
cd tests
python3 -m pytest . -v --tb=short

echo "[3/3] 测试全部通过 ✅"
