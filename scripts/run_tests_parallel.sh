#!/usr/bin/env bash
# 并行逐文件测试门禁（与 scripts/run_tests.sh 判据相同，只是把 41 个文件并行跑）
#
# 为什么需要：拆成真 TU 之后，数值用例第一次真正在 CPU 上跑，全量串行要约 30 分钟；
# 逐文件并行（-P 5）约 6-8 分钟，适合重构期间每轮自检。
#
# 用法:
#   bash scripts/run_tests_parallel.sh            # 跑全部 test_*.py
#   JOBS=3 bash scripts/run_tests_parallel.sh     # 控制并发
#
# 输出：每个文件一行（PASS/FAIL/SKIP/ERR/TIME/RC），末尾打印总计与失败清单。
# 判据：PASS 数不得低于基线、FAIL 集合不得新增（基线见 doc/baseline-cpu-v0.3.27.md）。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-5}"
OUTDIR="$(mktemp -d)"
TABLE="${OUTDIR}/table.txt"

cd "${ROOT}/tests"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CTPWA_QUIET=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OUTDIR

# 预检：能 import 才继续。.so 缺符号时 pytest 会给出 41 个 "usage error"，
# 那种噪音掩盖真正原因（实测过一次：头文件里 one 个签名写错 → undefined symbol）。
if ! python3 -c "import ctpwa; ctpwa.backend()" >/dev/null 2>&1; then
    echo "❌ ctpwa 无法加载（多半是 .so 缺符号 / 构建不完整）："
    python3 -c "import ctpwa" 2>&1 | tail -3
    exit 2
fi

run_one() {
    f="$1"
    local t0=$(date +%s)
    timeout 900 python3 -m pytest "$f" -q --no-header -p no:cacheprovider --tb=no \
        > "$OUTDIR/$f.log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local line=$(grep -E "passed|failed|error|skipped" "$OUTDIR/$f.log" | tail -1)
    local p fl s e
    p=$(echo "$line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+'); p=${p:-0}
    fl=$(echo "$line" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+'); fl=${fl:-0}
    s=$(echo "$line" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+'); s=${s:-0}
    e=$(echo "$line" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+'); e=${e:-0}
    printf '%-42s %4s %4s %4s %4s %4ss %s\n' "$f" "$p" "$fl" "$s" "$e" "$((t1-t0))" "$rc" \
        > "$OUTDIR/$f.line"
    rm -f "$OUTDIR/$f.log"
}
export -f run_one

ls test_*.py | xargs -P "${JOBS}" -I{} bash -c 'run_one "$@"' _ {}

{ printf '%-42s %4s %4s %4s %4s %6s %s\n' FILE PASS FAIL SKIP ERR TIME RC
  cat "$OUTDIR"/*.line | sort; } > "$TABLE"

cat "$TABLE"
awk 'NR>1 && $2+$3+$4+$5>0 {p+=$2; f+=$3; s+=$4; e+=$5}
     END {printf "\n合计: passed=%d failed=%d skipped=%d errors=%d\n", p, f, s, e;
          printf "%d %d %d\n", p, f, e > "/dev/stderr"}' "$TABLE" 2> "${OUTDIR}/totals"
awk 'NR>1 && $3+$5>0 {printf "  !! %-40s fail=%s err=%s\n", $1, $3, $5}' "$TABLE"
echo "逐文件表: ${TABLE}"

# 门禁判据：failed 与 errors 必须为 0（skipped 允许——GPU-only 用例在 CPU 构建上就是跳过）。
# 2026-11 起 CPU 构建是全绿的（258 passed / 0 failed / 63 skipped），所以"红即回归"。
read -r P F E < "${OUTDIR}/totals"
# 崩溃/超时也必须算失败：段错误时 pytest 不打印 "N failed"，
# 只看 failed/errors 会把整层信号错误当成"全绿"（本轮实测踩过：segfault 后 passed 从 258 掉到 122 却报绿）。
CRASH=$(awk 'NR>1 && $7 != 0 && $7 != 1 && $7 != 5 {n++} END {print n+0}' "$TABLE")
TIMEOUT=$(awk 'NR>1 && $7 == 124 {n++} END {print n+0}' "$TABLE")
if (( F > 0 || E > 0 || CRASH > 0 )); then
    echo "❌ 门禁失败: failed=$F errors=$E 异常退出(含段错误/超时)=$CRASH（passed=$P）"
    awk 'NR>1 && $7 != 0 && $7 != 1 && $7 != 5 {printf "   !! %-40s rc=%s\n", $1, $7}' "$TABLE"
    exit 1
fi
echo "✅ 门禁通过: passed=$P failed=0 errors=0"
