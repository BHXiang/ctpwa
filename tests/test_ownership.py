"""P2 回归：设备缓冲的所有权必须是单一的。

背景（doc/modularization-plan.md §2.4b）：`AmpCasDecay` 曾把同一批设备指针同时
登记在 `d_mom_sigma_` 与 `d_momenta_structs_/d_momenta_arrays_` 两处。
带全同粒子的链（nSigma>1）在"静态 SL 表缩减"时只 free 其中一处，析构再 free
一次 —— CPU 后端直接 `free(): double free detected` abort，CUDA 上是 cudaFree
二次释放。当时的补偿是"按指针匹配把条目从另一张登记表里摘掉"。

现在所有权唯一（`ctpwa::DevBuffer`），因此这里断言的行为很朴素：
**构造 → 算一次 NLL → 释放 → 销毁，反复多轮都不崩**。

CPU 后端用 malloc/free，double free 会立刻 abort，所以这个用例在无卡机器上
也能真正守住这条路径（这正是它在 CUDA 上容易被漏掉的原因）。
"""
import gc

import pytest
import torch

import ctpwa
from conftest import CONFIGS_DIR

# σ>1 的配置：ident2 = 2 个 π⁰ 跨子衰变，ident3 = 3 个
IDENT_CONFIGS = ["ident2", "ident3"]


def _make_params(ana):
    n_vec = ana.getNVector()
    n_theta = ana.getNFreeTheta()
    p = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64)
    p[0] = 1.0
    for k in range(1, n_vec):
        p[k] = 0.1 * k
        p[n_vec + k] = 0.05 * k
    if n_theta:
        p[2 * n_vec:] = ana.getFreeResParams()[0]
    return p


@pytest.mark.parametrize("name", IDENT_CONFIGS)
def test_release_and_destroy_repeatedly(name):
    """多轮构造/析构：任何一处双重释放都会在 CPU 后端 abort。"""
    cfg = CONFIGS_DIR / f"{name}.yml"
    assert cfg.exists(), f"缺少测试配置 {cfg}"

    nlls = []
    for _ in range(3):
        ana = ctpwa.analysis(str(cfg))
        p = _make_params(ana)
        nlls.append(ana.getNLL(p).item())
        ana.release()          # 显式释放（析构也会再走一遍清理）
        del ana
        gc.collect()

    assert all(n == n for n in nlls), f"NLL 出现 NaN: {nlls}"
    # 同一配置、同一参数 → 必须完全可复现（顺带守住释放顺序没污染状态）
    assert nlls[0] == nlls[1] == nlls[2], f"{name}: NLL 不可复现 {nlls}"


def test_destructor_without_explicit_release():
    """只靠析构（不调 release()）也必须干净：走的是 disarming 分支。"""
    cfg = CONFIGS_DIR / "ident2.yml"
    for _ in range(2):
        ana = ctpwa.analysis(str(cfg))
        ana.getNLL(_make_params(ana))
        del ana
        gc.collect()
