"""嵌套自由共振态的 θ 梯度 / Hessian 回归。

背景（bug）
----------
一条链里若出现 "共振态 A → 共振态 B"，且 A、B 的质量都自由，则父节点 A 的
BWR 线形

    γ = g0·(q/q0)^(2L+1)·(m0/m)·Bf²,   q0 = breakup(m0, md1, md2)

里 ``q0`` 依赖子粒子 B 的质量参数 ``m_B``。但建 aux 时 ``getDaughterMassDep``
把"子粒子是另一个自由共振态"判成 ``Q0MassDep::EventMass``（AST 里是
``CVAR_MD1/2`` 变量），于是 ``∂F_A/∂m_B ≡ 0`` —— 整项丢失。表现为:

* θ 质量梯度：AD 与中心差分差 0.7%（本例）/ 100%（近阈值时）；
* Hessian θθ 块：显著元错到 ~90%。

已有的 ``dlnO_dm0`` / ``addBfQ0HessianTerms`` 只补了**顶点 Blatt-Weisskopf**
的 q0 链，没补模型线形内部的 q0。

触发条件（本例 ``configs/nested_free.yml``）:
J/ψ → γ + R1(1+) → π01 + R2(1+) → π02 + R3(0+) → π03 π04，三个 BWR 全 free。

修复后把 ``xfail(strict=True)`` 去掉即可（XPASS 会主动报错提醒）。
"""

import numpy as np
import pytest
import torch

import ctpwa
from conftest import make_params

XFAIL_HESSIAN = pytest.mark.xfail(
    strict=True,
    reason="已知遗留：θθ Hessian 的 q0 跨共振二阶项尚未补（梯度已修，见模块 docstring）",
)

DEVICE = torch.device("cpu") if ctpwa.backend() == "cpu" else torch.device("cuda:0")
FD_H = 1e-5
GRAD_RTOL = 1e-5


def _grad(ana, p):
    q = p.clone().requires_grad_(True)
    return torch.autograd.grad(ana.getNLL(q), q)[0].detach()


def _fd_scalar_grad(ana, p0, i, h=FD_H):
    pp, pm = p0.clone(), p0.clone()
    pp[i] += h
    pm[i] -= h
    return (float(ana.getNLL(pp)) - float(ana.getNLL(pm))) / (2 * h)


def test_theta_gradient_matches_fd(make_analysis):
    """θ 分量（尤其嵌套子共振态的质量）解析梯度对中心差分。"""
    ana = make_analysis("nested_free")
    p0 = make_params(ana, DEVICE)
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    assert nt >= 4, "nested_free 至少要有两个自由共振态"

    g = _grad(ana, p0)
    for i in range(2 * nv, 2 * nv + nt):
        fd = _fd_scalar_grad(ana, p0, i)
        rel = abs(float(g[i]) - fd) / max(abs(fd), 1e-8)
        assert rel < GRAD_RTOL, (
            f"θ[{i - 2 * nv}] AD={float(g[i]):.8g} FD={fd:.8g} rel={rel:.2e}")


@XFAIL_HESSIAN
def test_hessian_theta_block_matches_fd(make_analysis):
    """θ-θ 块（拟合真正用到的部分）对有限差分。"""
    ana = make_analysis("nested_free")
    p0 = make_params(ana, DEVICE)
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    th0 = 2 * nv

    H = ana.getHessian(p0).detach().double()
    n = len(p0)
    Hfd = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        pp, pm = p0.clone(), p0.clone()
        pp[i] += FD_H
        pm[i] -= FD_H
        Hfd[i] = (_grad(ana, pp) - _grad(ana, pm)) / (2 * FD_H)

    ad = H[th0:, th0:].numpy()
    fd = Hfd[th0:, th0:].numpy()
    scale = float(np.abs(fd).max())
    rel = float(np.abs(ad - fd).max()) / max(scale, 1e-30)
    assert rel < 1e-5, f"Hessian θ 块 AD vs FD rel={rel:.2e}（|H|max={scale:.4g}）"
