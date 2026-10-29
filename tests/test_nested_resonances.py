"""嵌套自由共振态的 θ 梯度 / Hessian 回归。

背景（bug）
----------
一条链里若出现 "共振态 A → 共振态 B"，且 A、B 的质量都自由，则父节点 A 的
BWR 线形

    γ = g0·(q/q0)^(2L+1)·(m0/m)·Bf²,   q0 = breakup(m0, md1, md2)

里 ``q0`` 依赖子粒子 B 的质量参数 ``m_B``。但建 aux 时 ``getDaughterMassDep``
把"子粒子是另一个自由共振态"判成 ``Q0MassDep::EventMass``（AST 里是
``CVAR_MD1/2`` 变量），于是 ``∂F_A/∂m_B ≡ 0`` —— 一阶、二阶都整项丢失。
已有的 ``dlnO_dm0`` / ``addBfQ0HessianTerms`` 只补了**顶点 Blatt-Weisskopf**
的 q0 链，没补模型线形内部的 q0。

修复（两个 kernel 各一处，均只在"本节点确有自由共振态子粒子"时触发）：
* 梯度：``reComputeAmps`` 的 R>1 路径按数值扰动补 ``∂F/∂m_child``；
* Hessian：``computeCustomHessianKernel`` 的 ``addModelQ0HessianTerms`` 补
  ``∂²lnF/∂m²``、``∂²lnF/∂m∂θ``、两个子质量的混合项。

数据在 fixture 里用 DIYMC 的 ``PhaseSpaceGenerator`` 现场生成（GENBOD 平坦
5 体相空间），避免把二进制样本入库，也避免 ``data/5body.dat`` 那种伪事例
（不满足四动量守恒）在阈值附近造成的折点污染二阶差分。

触发结构（``configs/nested_free.yml``）:
J/ψ → γ + R1(1+) → π01 + R2(1+) → π02 + R3(0+) → π03 π04，三个 BWR 全 free。
"""

import numpy as np
import pytest
import torch

import ctpwa
from conftest import CONFIGS_DIR, make_params

DEVICE = torch.device("cpu") if ctpwa.backend() == "cpu" else torch.device("cuda:0")
FD_H = 1e-5
GRAD_RTOL = 1e-4          # 按 max|g| 归一；GPU FD 噪声 ~3e-6，修复前 ~0.25


@pytest.fixture(scope="module")
def nested_ana(tmp_path_factory):
    """用 GENBOD 现场生成 5 体样本 → 构造嵌套链 analysis（无需入库数据）。"""
    d = tmp_path_factory.mktemp("nested_free")
    cfg_text = (CONFIGS_DIR / "nested_free.yml").read_text(encoding="utf-8")
    cfg_text = cfg_text.replace('data: [dat, "./data/5body.dat"]',
                                f'data: [dat, "{d}/data.dat"]')
    cfg_text = cfg_text.replace('phsp: [dat, "./data/5body.dat"]',
                                f'phsp: [dat, "{d}/phsp.dat"]')
    cfg_path = d / "config.yml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    # PhaseSpaceGenerator 只解析 Particles / Data.order / DecayChains，不读 data 文件
    gen = ctpwa.PhaseSpaceGenerator(str(cfg_path))
    gen.write(gen.generate(300, seed=11), str(d / "data.dat"))
    gen.write(gen.generate(3000, seed=12), str(d / "phsp.dat"))
    return ctpwa.analysis(str(cfg_path))


def _grad(ana, p):
    q = p.clone().requires_grad_(True)
    return torch.autograd.grad(ana.getNLL(q), q)[0].detach()


def _fd_scalar_grad(ana, p0, i, h=FD_H):
    pp, pm = p0.clone(), p0.clone()
    pp[i] += h
    pm[i] -= h
    return (float(ana.getNLL(pp)) - float(ana.getNLL(pm))) / (2 * h)


def test_theta_gradient_matches_fd(nested_ana):
    """θ 分量（尤其嵌套子共振态的质量）解析梯度对中心差分。"""
    ana = nested_ana
    p0 = make_params(ana, DEVICE)
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    assert nt >= 4, "nested_free 至少要有两个自由共振态"

    g = _grad(ana, p0)
    # 按 max|g| 归一（同 test_identical）：小分量在 GPU 上被 FD 噪声主导，
    # 逐分量相对误差会假报警；归一后对"整块梯度算错"依然灵敏（修复前 ~0.25）
    scale = float(g.abs().max().clamp_min(1e-6))
    worst, worst_i = 0.0, -1
    for i in range(2 * nv, 2 * nv + nt):
        fd = _fd_scalar_grad(ana, p0, i)
        d = abs(float(g[i]) - fd)
        if d > worst:
            worst, worst_i = d, i
    rel = worst / scale
    assert rel < GRAD_RTOL, (
        f"θ[{worst_i - 2 * nv}] AD={float(g[worst_i]):.8g} "
        f"FD={_fd_scalar_grad(ana, p0, worst_i):.8g} 归一化误差={rel:.2e}")


def test_hessian_theta_block_matches_fd(nested_ana):
    """θ-θ 块（拟合真正用到的部分）对有限差分。

    用仓库自带的 ``fd_utils.compare_hessian``（带 FD 可靠性过滤，与
    test_cpu_numerics 同口径）：修复前元素级相对差 ~90%（verdict=FAIL），
    修复后全部 judged 元 < 1e-2（verdict=PASS）。
    """
    import fd_utils

    ana = nested_ana
    p0 = make_params(ana, DEVICE)
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    th0 = 2 * nv
    H = ana.getHessian(p0).detach().double().cpu().numpy()
    idxs = list(range(th0, th0 + nt))
    rep = fd_utils.compare_hessian(H[np.ix_(idxs, idxs)],
                                   lambda q: _grad(ana, q), p0, idxs)
    assert rep.verdict != "FAIL", rep.summary()
