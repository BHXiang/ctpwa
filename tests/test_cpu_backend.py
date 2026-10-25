"""CPU 后端冒烟测试：能构造、NLL 有限、对 golden、可复现。

只在**当前进程选中 CPU 后端**时运行（GPU 机器上请用 ``CTPWA_DEVICE=cpu`` 跑）；
否则整体 skip。

运行::

    # 无卡机器
    python3 -m pytest tests/test_cpu_backend.py -v
    # 有卡机器上专门测 CPU 后端
    CTPWA_DEVICE=cpu python3 -m pytest tests/test_cpu_backend.py -v

说明：本文件刻意**不使用** conftest 的 ``device`` fixture（那个 fixture 断言有 CUDA），
所有张量都显式建在 CPU 上。
"""

import math
import os
import subprocess
import sys

import pytest
import torch

import ctpwa
from conftest import CONFIGS_DIR, cpu_only, load_golden, make_params

pytestmark = cpu_only

CPU = torch.device("cpu")

# 覆盖不同线形（BWR/Flatte/GS/Interp/Custom）、全同粒子（ident2）与两种参数化（trans）
CONFIGS = ["simple", "flatte", "gs", "interp", "custom", "ident2",
           "no_trans", "with_trans"]


# ---------------------------------------------------------------- 基本功能
@pytest.mark.parametrize("key", CONFIGS)
def test_construct_and_nll_finite(make_analysis, key):
    """每个 config 都能构造，且 NLL 是有限值（不是 NaN / 1e30 哨兵）。"""
    ana = make_analysis(key)
    p = make_params(ana, CPU)
    nll = ana.getNLL(p).item()
    assert math.isfinite(nll), f"{key}: NLL={nll!r} 不是有限值"
    assert abs(nll) < 1e11, f"{key}: NLL={nll!r} 落到 NaN 哨兵 (1e30)"


@pytest.mark.parametrize("key", CONFIGS)
def test_nll_is_negative_free(make_analysis, key):
    """NLL 对 v 的整体标度不变：加倍 v 不改变结果（物理自洽性的最小检查）。"""
    ana = make_analysis(key)
    p = make_params(ana, CPU)
    base = ana.getNLL(p).item()
    scaled = p.clone()
    nv = ana.getNVector()
    scaled[: 2 * nv] *= 2.0
    assert ana.getNLL(scaled).item() == pytest.approx(base, abs=1e-6)


def test_golden_nll_simple(make_analysis):
    """simple 的 v[1:] 为空 → 参数与 RNG 无关，可与 GPU 生成的 golden 直接比。"""
    ana = make_analysis("simple")
    want = load_golden("nll_simple.txt")
    got = ana.getNLL(make_params(ana, CPU)).item()
    assert abs(got - want) / abs(want) < 1e-6, f"golden={want:.12f} cpu={got:.12f}"


# ---------------------------------------------------------------- 可复现性
@pytest.mark.parametrize("key", ["simple", "flatte"])
def test_repeatable_bitwise(make_analysis, key):
    """同输入重复求值必须逐位一致（NLL 累加已改为单次有序）。"""
    ana = make_analysis(key)
    p = make_params(ana, CPU)
    a = ana.getNLL(p).item()
    b = ana.getNLL(p).item()
    assert a == b, f"{key}: {a!r} != {b!r}"


def test_independent_of_omp_threads():
    """换 OMP 线程数结果不变（子进程里跑，避免污染当前进程的线程池）。"""
    code = (
        "import ctpwa,torch,hashlib;"
        "ana=ctpwa.analysis('configs/simple.yml');"
        "nv,nt=ana.getNVector(),ana.getNFreeTheta();"
        "p=torch.zeros(2*nv+nt,dtype=torch.float64);p[0]=1.0;"
        "p[2*nv:]=ana.getFreeResParams()[0];"
        "H=ana.getHessian(p);"
        "print(round(ana.getNLL(p).item(),12),"
        "hashlib.sha256(H.numpy().tobytes()).hexdigest())"
    )
    outs = []
    for nt in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=nt)
        r = subprocess.run([sys.executable, "-c", code], cwd=str(CONFIGS_DIR.parent),
                           env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-800:]
        outs.append(r.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1], f"OMP_NUM_THREADS 1 vs 4 不一致: {outs}"


def test_hessian_repeatable_bitwise(make_analysis):
    ana = make_analysis("simple")
    p = make_params(ana, CPU)
    H1 = ana.getHessian(p)
    H2 = ana.getHessian(p)
    assert torch.equal(H1, H2), "Hessian 重复调用不是逐位一致"


# ---------------------------------------------------------------- 构建标识
def test_backend_is_cpu():
    assert ctpwa.backend() == "cpu"


def test_build_info():
    info = ctpwa.build_info()
    assert info["backend"] == "cpu"
    assert info["has_cuda"] is False
    assert info["precision"] == "double"      # CPU 后端固定 double
    assert info["complex_bytes"] == 16
