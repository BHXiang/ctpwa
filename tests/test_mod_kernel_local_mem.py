"""静态门禁 (L0, 不需要 GPU): computeModWithInterference 的每线程局部内存不许失控。

背景 (v0.3.21 修的真实故障): 该 kernel 原来在栈上放
    double partial_real[N], partial_imag[N], interference_accumulator[N(N+1)/2]
且模板桶只有 50 / 200 / 1000 三档。CUDA 的局部内存是按"每线程字节 × 最大常驻
线程数"整块预留的(**与 grid 大小无关**)，V100 上 2048×80 = 163 840 线程，于是：
    <50>  : ~11 KB/线程 → ~1.7 GiB
    <200> : ~164 KB/线程 → ~25 GiB
一个 51 分波的分析（51 只比 50 大一）就落进 <200> 桶，writeResult 一启动就要
25 GiB 局部内存；节点上还驻留着振幅表 → `computeModWithInterference 失败:
out of memory`。而 NLL 拟合根本不走这个 kernel，所以"拟合正常、一写权重就 OOM"。

修法: 累加器只在需要逐事件干涉项时才存在(N_PAIRS>0)，且按导出对数分桶。
本测试用 cuobjdump 读 ptxas 的 STACK(局部帧) 字节数钉住这个性质，防止回退。
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SO = REPO / "ctpwa" / "_cuda.so"
CUOBJDUMP = shutil.which("cuobjdump")

pytestmark = pytest.mark.skipif(
    CUOBJDUMP is None or not SO.exists(),
    reason="需要 cuobjdump 与 ctpwa/_cuda.so（纯 CPU 构建跳过）",
)

# 每线程局部内存上限(字节)。N_PAIRS=0 的 <200> 桶实测 ~3.2 KB；桶跨度或累加器
# 一旦退回旧写法会立刻 >100 KB。
MAX_LOCAL_BYTES_NO_EXPORT = 32 * 1024
MAX_LOCAL_BYTES_N200_EXPORT = 32 * 1024


def _stack_bytes():
    """{ (N_PARTIALS, N_PAIRS): stack_bytes }，来自 ptxas 的资源报告。"""
    out = subprocess.run([CUOBJDUMP, "-res-usage", str(SO)],
                         capture_output=True, text=True, check=True).stdout
    usage = {}
    cur = None
    for line in out.splitlines():
        m = re.search(r"computeModWithInterferenceILi(\d+)ELi(\d+)E", line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.search(r"STACK:(\d+)", line)
        if m and cur is not None:
            usage[cur] = int(m.group(1))
            cur = None
    return usage


def test_no_export_buckets_have_small_local_frame():
    usage = _stack_bytes()
    assert usage, "没有从 _cuda.so 里解析到 computeModWithInterference 实例"
    no_export = {k: v for k, v in usage.items() if k[1] == 0}
    assert no_export, "缺少 N_PAIRS=0（writeResult 路径）的实例"
    for (npart, _), stack in sorted(no_export.items()):
        assert stack <= MAX_LOCAL_BYTES_NO_EXPORT, (
            f"N_PARTIALS={npart} 的局部帧 {stack} B 超过 "
            f"{MAX_LOCAL_BYTES_NO_EXPORT} B；常驻线程数一乘就是 GiB 级显存"
        )
    # 最常见的 51~200 分波区间必须远低于旧实现的 164 KB
    assert usage[(200, 0)] <= 8 * 1024


def test_export_bucket_is_bounded():
    usage = _stack_bytes()
    if (200, 2048) not in usage:
        pytest.skip("未编出导出桶实例")
    assert usage[(200, 2048)] <= MAX_LOCAL_BYTES_N200_EXPORT
    # 导出桶必须单调增长且远小于旧的 ninterference 累加器(200 波 ≈ 160 KB)
    assert usage[(200, 0)] < usage[(200, 32)] < usage[(200, 2048)] < 64 * 1024


def test_pair_export_limit_documented():
    """超过 2048 对必须显式报错（而不是悄悄吃掉 ninterference 个槽位的显存）。"""
    src = (REPO / "src" / "ComputeResults.cu").read_text()
    assert "2048" in src
    assert "一次最多导出" in src
