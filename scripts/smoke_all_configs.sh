#!/usr/bin/env bash
# 深检查：把 tests/configs/*.yml **全部**构造一遍（不进主门禁，约 10 分钟）
#
# 为什么单独一个脚本：门禁里加"遍历全部 config"会让每轮多 9.5 分钟，太重。
# 但只挑几个 config 又会漏——第 33 轮那个 `deep_4pi0` 回归就是因为"没有测试用它"，
# 全量门禁绿着而配置构造不出来。
#
# 用法：
#   bash scripts/smoke_all_configs.sh              # 当前构建
#   PYTHONPATH=.../baseline_wt bash scripts/smoke_all_configs.sh   # 基线对比
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CTPWA_QUIET=1
cd tests

python3 - "$@" <<'PY'
import io, sys, contextlib, traceback
from pathlib import Path
import ctpwa

cfgs = sorted(Path("configs").glob("*.yml"))
bad = []
for f in cfgs:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ana = ctpwa.analysis(str(f))
            _ = ana.getNVector(), ana.getAmplitudeNames(), ana.getParamNames()
    except Exception as e:
        bad.append((f.stem, f"{type(e).__name__}: {e}"[:160]))

print(f"后端 {ctpwa.backend()}：{len(cfgs)} 个 config，失败 {len(bad)}")
for name, err in bad:
    print(f"  ❌ {name}: {err}")
sys.exit(1 if bad else 0)
PY
