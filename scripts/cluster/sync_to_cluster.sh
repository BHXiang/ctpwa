#!/bin/bash
# 集群 tests/ 同步工具（repo tests/ 为唯一权威）
#
# 模式:
#   ./sync_to_cluster.sh pack     # (默认) 生成 _to_cluster/ 待同步内容（自行 scp 到集群 tests/）
#   ./sync_to_cluster.sh push     # 直接 rsync 到集群（需配置 CLUSTER_TESTS 且 ssh 可用）
#   ./sync_to_cluster.sh pull     # 从集群拉回 log_*.out 到 ./logs/
#
# 生成/推送内容 = repo tests/ 全套（排除缓存/生成物）+ 本目录 4 个工具。
set -eu

REPO_TESTS="$(cd "$(dirname "$0")/../tests" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
CLUSTER_TESTS="${CLUSTER_TESTS:-bhxiang@lxlogin.ihep.ac.cn:/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa/tests}"
TOOLS="job_validate_all.sh mem_probe.py perf_probe.py ff_eff_probe.py"
EXCLUDES="--exclude=_gen_* --exclude=.pytest_cache --exclude=__pycache__ --exclude=outputs --exclude=*.root"

MODE="${1:-pack}"

pack() {
    rm -rf "$HERE/_to_cluster"
    mkdir -p "$HERE/_to_cluster"
    # repo tests/ 内容（文件直接放 _to_cluster/ 下，便于 scp 到集群 tests/）
    (cd "$REPO_TESTS" && tar cf - --exclude=_gen_* --exclude=.pytest_cache \
        --exclude=__pycache__ --exclude=outputs --exclude='*.root' .) \
        | (cd "$HERE/_to_cluster" && tar xf -)
    for t in $TOOLS; do cp "$HERE/$t" "$HERE/_to_cluster/"; done
    echo "已生成 $HERE/_to_cluster/（含 repo tests/ 全套 + 工具）"
    HOST="${CLUSTER_TESTS%%:*}"; DEST="${CLUSTER_TESTS#*:}"
    echo "上传示例:"
    echo "  scp -r $HERE/_to_cluster/* $HOST:$DEST/"
    echo "  rsync -av $HERE/_to_cluster/ $HOST:$DEST/   (可先 ssh 确认可达)"
}

push() {
    echo "rsync repo tests/ -> $CLUSTER_TESTS"
    rsync -av --delete $EXCLUDES "$REPO_TESTS/" "$CLUSTER_TESTS/"
    for t in $TOOLS; do rsync -av "$HERE/$t" "$CLUSTER_TESTS/"; done
}

pull() {
    mkdir -p "$HERE/logs"
    # 目标写法: CLUSTER_TESTS=user@host:path 时取 host:path
    SRC="${CLUSTER_TESTS%:*}"; D="${CLUSTER_TESTS#*:}"
    rsync -av "$SRC:$D/log_*.out" "$HERE/logs/" 2>/dev/null \
        || echo "拉取失败（检查 CLUSTER_TESTS 与 ssh）；日志也可手动拷到 ./logs/"
}

case "$MODE" in
    pack) pack ;;
    push) push ;;
    pull) pull ;;
    *) echo "用法: $0 [pack|push|pull]"; exit 1 ;;
esac
