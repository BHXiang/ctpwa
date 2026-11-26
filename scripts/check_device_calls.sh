#!/usr/bin/env bash
# 设备切换出口检查（P0 设备纪律的机械门禁）
#
# 规则：src/ 与 include/ 里**只有**下面这几个文件允许出现裸 cudaSetDevice。
# 其余一律改用 include/DeviceGuard.h 的三个原语：
#   ensure_device(dev)      粘性设置（= 原来的 cudaSetDevice(primary_dev_)）
#   DeviceGuard g(dev)      作用域（进入切、退出还原）
#   with_device(dev, fn)    多卡循环
#
# 用法:
#   bash scripts/check_device_calls.sh          # 检查，有违规 → exit 1
#   bash scripts/check_device_calls.sh --stats  # 只打印计数（用于跟踪改造进度）
set -uo pipefail
cd "$(dirname "$0")/.."

STATS_ONLY=false
[[ "${1:-}" == "--stats" ]] && STATS_ONLY=true

# 允许清单（每行一个前缀）
ALLOW_FILES=(
    "include/DeviceGuard.h"    # 唯一入口；注释里会提到 cudaSetDevice 这个名字
    "include/CudaCompat.h"     # 非 CUDA 构建的类型替身
    "src/Backend.cpp"          # 后端实现：cudaSetDevice 的唯一定义/调用处
    "src/DeviceManager.cu"     # 设备模块（后续 P0b 也应迁移到 DeviceGuard）
)
ALLOW_PREFIX=(
    "include/cpu_shims/"       # CPU 后端的假实现
)

is_allowed() {
    local f="$1"
    for p in "${ALLOW_PREFIX[@]}"; do [[ "$f" == "$p"* ]] && return 0; done
    for a in "${ALLOW_FILES[@]}"; do [[ "$f" == "$a" ]] && return 0; done
    return 1
}

# 收集所有含裸调用的文件（排除注释行里的提及：行首 // 或 * 的跳过）
mapfile -t HITS < <(
    grep -rn "cudaSetDevice" src include 2>/dev/null \
        | grep -v '^\s*$' \
        | awk -F: '
            {
                file=$1; line=$2; rest=$0
                sub(/^[^:]*:[^:]*:/, "", rest)
                gsub(/^[ \t]+/, "", rest)
                if (rest ~ /^\/\// || rest ~ /^\*/) next   # 纯注释行
                print file ":" line ":" rest
            }'
)

total=0
viol=0
declare -A per_file=()
declare -A viol_file=()

for h in "${HITS[@]:-}"; do
    [[ -z "$h" ]] && continue
    f="${h%%:*}"
    total=$((total + 1))
    per_file["$f"]=$(( ${per_file["$f"]:-0} + 1 ))
    if ! is_allowed "$f"; then
        viol=$((viol + 1))
        viol_file["$f"]=$(( ${viol_file["$f"]:-0} + 1 ))
        $STATS_ONLY || echo "  违规 $h"
    fi
done

echo "------------------------------------------------------------"
echo "裸 cudaSetDevice 总数: $total（允许清单内 $((total - viol))，违规 $viol）"
for f in "${!per_file[@]}"; do
    mark=""
    is_allowed "$f" || mark="  ← 违规"
    printf '  %-34s %3d%s\n' "$f" "${per_file[$f]}" "$mark"
done | sort -k2 -rn
echo "------------------------------------------------------------"

if $STATS_ONLY; then
    exit 0
fi

if (( viol > 0 )); then
    echo "失败: 请改用 include/DeviceGuard.h 的 ensure_device / DeviceGuard / with_device。"
    exit 1
fi
echo "通过: 设备切换只有单一出口。"
