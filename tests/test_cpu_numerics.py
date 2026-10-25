"""CPU 后端数值正确性：梯度/Hessian 对有限差分、对称性、重参数化等价。

只在**当前进程选中 CPU 后端**时运行（GPU 机器上请用 ``CTPWA_DEVICE=cpu``）。

判据（与 doc/cpu-backend-plan.md 的验收口径一致）:
- 梯度 vs FD：θ 分量相对差 < 1e-6
- Hessian θ 块 vs FD：相对差 < 1e-6；矩阵对称到 1e-10
- no_trans 与 with_trans：同一物理，NLL 必须逐位相同
"""

import numpy as np
import pytest
import torch

from conftest import cpu_only, make_params

pytestmark = cpu_only

CPU = torch.device("cpu")
FD_H = 1e-5


def _grad(ana, p):
    pp = p.clone().requires_grad_(True)
    return torch.autograd.grad(ana.getNLL(pp), pp)[0].detach()


def _fd_gradient_row(ana, p0, i, h=FD_H):
    """Hessian 的第 i 行：对解析梯度做中心差分 d(g)/dp_i。"""
    pp, pm = p0.clone(), p0.clone()
    pp[i] += h
    pm[i] -= h
    return (_grad(ana, pp) - _grad(ana, pm)) / (2.0 * h)


def _fd_scalar_grad(ana, p0, i, h=FD_H):
    """标量 NLL 对 p_i 的中心差分（与上面的 Hessian 行不同，别混用）。"""
    pp, pm = p0.clone(), p0.clone()
    pp[i] += h
    pm[i] -= h
    return (ana.getNLL(pp).item() - ana.getNLL(pm).item()) / (2.0 * h)


# ---------------------------------------------------------------- 梯度
@pytest.mark.parametrize("key", ["simple", "flatte", "gs"])
def test_gradient_matches_fd(make_analysis, key):
    """θ 分量的解析梯度对中心差分（v 分量信号低于 FD 噪声，不参与判定）。"""
    ana = make_analysis(key)
    p0 = make_params(ana, CPU)
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    if nt == 0:
        pytest.skip(f"{key} 无自由共振态参数")
    g = _grad(ana, p0)
    for i in range(2 * nv, 2 * nv + nt):
        fd = float(_fd_scalar_grad(ana, p0, i))
        ad = float(g[i])
        rel = abs(ad - fd) / max(abs(fd), 1e-8)
        assert rel < 1e-6, f"{key}[{i}]: AD={ad:.8g} FD={fd:.8g} rel={rel:.2e}"


# ---------------------------------------------------------------- Hessian
@pytest.mark.parametrize("key", ["simple", "flatte"])
def test_hessian_symmetric(make_analysis, key):
    ana = make_analysis(key)
    p = make_params(ana, CPU)
    H = ana.getHessian(p).double()
    scale = float(H.abs().max())
    assert float((H - H.T).abs().max()) <= 1e-10 * scale, f"{key}: Hessian 不对称"


@pytest.mark.parametrize("key", ["simple", "flatte", "hess_mult"])
def test_hessian_full_matches_fd(make_analysis, key):
    """整块 Hessian 对有限差分（用仓库自带的显著性过滤，见 tests/fd_utils.py）。

    为什么不能直接取 max|Δ|/max|H|：v 方向存在近乎零的模式（NLL 对 v 的整体
    标度不敏感），FD 在这些元上是纯噪声，直接比会被小分母放大成假失配。
    ``compare_hessian`` 只判定"显著且 FD 自洽"的元。

    hess_mult 是回归用例：它 n_amps 大且带耦合矩阵，曾因 ``axpyComplexKernel``
    被从 stride 循环改成"一线程一元素"而只更新前 64 个元素 → 梯度截断、
    Hessian 错，而 NLL 完全正常（只有这个用例抓得住）。
    """
    import fd_utils

    ana = make_analysis(key)
    p0 = make_params(ana, CPU)
    H = ana.getHessian(p0).double()
    grad_fn = lambda q: _grad(ana, q)      # noqa: E731
    rep = fd_utils.compare_hessian(H.numpy(), grad_fn, p0)
    assert rep.verdict != "FAIL", f"{key}: {rep.summary()}"
    if rep.verdict == "INCONCLUSIVE":
        pytest.skip(f"{key}: FD 无法判定（{rep.note}）")


@pytest.mark.parametrize("key", ["simple", "flatte"])
def test_hessian_theta_block_matches_fd(make_analysis, key):
    """θ-θ 块（拟合真正用到的部分）对有限差分。"""
    ana = make_analysis(key)
    p0 = make_params(ana, CPU)
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    if nt == 0:
        pytest.skip(f"{key} 无自由共振态参数")
    H = ana.getHessian(p0).double()
    n = len(p0)
    th0 = 2 * nv

    Hfd = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        Hfd[i] = _fd_gradient_row(ana, p0, i)

    ad = H[th0:, th0:]
    fd = Hfd[th0:, th0:]
    scale = float(fd.abs().max())
    rel = float((ad - fd).abs().max()) / max(scale, 1e-30)
    assert rel < 1e-6, f"{key}: Hessian θ 块 AD vs FD rel={rel:.2e}"


def test_hessian_shape(make_analysis):
    ana = make_analysis("simple")
    p = make_params(ana, CPU)
    n = len(p)
    assert tuple(ana.getHessian(p).shape) == (n, n)


# ---------------------------------------------------------------- 数值自洽
def test_hessian_eigenvalues_real(make_analysis):
    """Hessian 对称 → 特征值应为实数（用 eigvalsh 不报错即可）。"""
    ana = make_analysis("simple")
    H = ana.getHessian(make_params(ana, CPU)).double()
    w = torch.linalg.eigvalsh(H)
    assert w.numel() == H.shape[0]
    assert np.isfinite(w.numpy()).all()
