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


def test_bkg_sign_in_coupling_hessian(tmp_path):
    """回归：bkg 在耦合变换二阶项里的符号。

    旧 `getHessian` 对 bkg 无条件 `axpy(-1)`：有 `bkg_weights` 时
    `d_neg_w=-w` 已翻一次符号，再乘 -1 → `Σ g_v·∂²v` 的 bkg 项整体反号；
    无权重时 kernel 用单位正权重，-1 反而正确。
    无 bkg 或 bkg 很小时 FD 测试看不出来；把 bkg 权重放大到 20 后，
    hess_mult（3 因子振幅 + 耦合变换）旧代码解析 Hessian 与 FD 的
    最大相对差 ~1.3，新代码 <1e-2。
    """
    import ctpwa
    import fd_utils
    from conftest import CONFIGS_DIR

    data_dir = CONFIGS_DIR.parent / "data"
    src = (CONFIGS_DIR / "hess_mult.yml").read_text()
    for name in ("test_4body_data.dat", "test_4body_phsp.dat",
                 "test_4body_sideband.dat"):
        src = src.replace(f'"./data/{name}"', f'"{data_dir / name}"')

    # 300 个 bkg 事例（1200 行 / 4 粒子），权重 20 → bkg 项与 data 同量级
    wpath = tmp_path / "bkgw.dat"
    wpath.write_text("\n".join(["20.0"] * 300) + "\n")
    bkg_line = f'  bkg:  [dat, "{data_dir / "test_4body_sideband.dat"}"]'
    assert bkg_line in src
    src = src.replace(bkg_line, bkg_line + f'\n  bkg_weights: [dat, "{wpath}"]')

    cfg = tmp_path / "hess_mult_bkgw.yml"
    cfg.write_text(src)
    ana = ctpwa.analysis(str(cfg))
    p = make_params(ana, CPU)
    H, idxs = fd_utils.analysis_hessian_block(ana, p)
    rep = fd_utils.compare_hessian(
        H, fd_utils.analysis_grad_fn(ana), p, idxs,
        h_judge=1e-3, rtol=0.05)
    assert rep.verdict != "FAIL", rep.summary()


def test_phsp_weights_nll_grad_hessian(tmp_path):
    """回归：phsp_weights 必须同时进入 NLL 归一化、梯度和 Hessian。

    旧 forward 的固定 θ 分支用 `computePhspMeanSum`（无权重）算 phsp_factor，
    但 `d_phsp_matrix_` / d_P_vec 用 w 加权 → NLL 与梯度/Hessian 非同一条
    函数；表现是 gradient vs FD 差 ~30%，Hessian vs FD 差 ~15%。
    这里用 hess_mult（nt=0，纯耦合）覆盖用户 J/psi→γ KsKs pi0pi0 的结构。
    """
    import ctpwa
    import fd_utils
    from conftest import CONFIGS_DIR

    data_dir = CONFIGS_DIR.parent / "data"
    src = (CONFIGS_DIR / "hess_mult.yml").read_text()
    for name in ("test_4body_data.dat", "test_4body_phsp.dat",
                 "test_4body_sideband.dat"):
        src = src.replace(f'"./data/{name}"', f'"{data_dir / name}"')

    wpath = tmp_path / "phspw.dat"
    wpath.write_text("".join("0.5\n" if i % 2 == 0 else "1.0\n"
                             for i in range(10000)))
    phsp_line = f'  phsp: [dat, "{data_dir / "test_4body_phsp.dat"}"]'
    assert phsp_line in src
    src = src.replace(phsp_line, phsp_line + f'\n  phsp_weights: [dat, "{wpath}"]')

    cfg = tmp_path / "hess_mult_phspw.yml"
    cfg.write_text(src)
    ana = ctpwa.analysis(str(cfg))
    p = make_params(ana, CPU)
    grad_fn = fd_utils.analysis_grad_fn(ana)
    g = grad_fn(p)
    for i in (1, 2, 5, 7, 8):
        h = 1e-5 * max(abs(float(p[i])), 1.0)
        pp, pm = p.clone(), p.clone()
        pp[i] += h
        pm[i] -= h
        fd = (float(ana.getNLL(pp)) - float(ana.getNLL(pm))) / (2 * h)
        rel = abs(float(g[i]) - fd) / max(abs(fd), 1e-8)
        assert rel < 1e-5, f"phsp_weights 梯度 [{i}]: AD={float(g[i]):.8g} FD={fd:.8g} rel={rel:.2e}"

    H, idxs = fd_utils.analysis_hessian_block(ana, p)
    rep = fd_utils.compare_hessian(H, grad_fn, p, idxs,
                                   h_judge=1e-3, rtol=0.05)
    assert rep.verdict != "FAIL", rep.summary()


def test_data_weights_nll_grad_hessian(tmp_path):
    """data_weights + bkg_weights 的 NLL/梯度/Hessian 必须同源。

    与 phsp_weights 的 bug 不同，data 权重从一开始就同时进入
    computeFactorNLL、data Hessian 和 N_eff；这里做回归保护。
    """
    import ctpwa
    import fd_utils
    from conftest import CONFIGS_DIR

    data_dir = CONFIGS_DIR.parent / "data"
    src = (CONFIGS_DIR / "hess_mult.yml").read_text()
    for name in ("test_4body_data.dat", "test_4body_phsp.dat",
                 "test_4body_sideband.dat"):
        src = src.replace(f'"./data/{name}"', f'"{data_dir / name}"')

    wdata = tmp_path / "dataw.dat"
    wdata.write_text("".join("0.5\n" if i % 2 == 0 else "1.0\n"
                             for i in range(1000)))
    wbkg = tmp_path / "bkgw.dat"
    wbkg.write_text("0.8\n" * 300)
    data_line = f'  data: [dat, "{data_dir / "test_4body_data.dat"}"]'
    bkg_line = f'  bkg:  [dat, "{data_dir / "test_4body_sideband.dat"}"]'
    assert data_line in src and bkg_line in src
    src = src.replace(data_line, data_line + f'\n  data_weights: [dat, "{wdata}"]')
    src = src.replace(bkg_line, bkg_line + f'\n  bkg_weights: [dat, "{wbkg}"]')

    cfg = tmp_path / "hess_mult_dataw.yml"
    cfg.write_text(src)
    ana = ctpwa.analysis(str(cfg))
    p = make_params(ana, CPU)
    grad_fn = fd_utils.analysis_grad_fn(ana)
    g = grad_fn(p)
    for i in (1, 2, 5, 7, 8):
        h = 1e-5 * max(abs(float(p[i])), 1.0)
        pp, pm = p.clone(), p.clone()
        pp[i] += h
        pm[i] -= h
        fd = (float(ana.getNLL(pp)) - float(ana.getNLL(pm))) / (2 * h)
        rel = abs(float(g[i]) - fd) / max(abs(fd), 1e-8)
        assert rel < 1e-5, f"data_weights 梯度 [{i}]: AD={float(g[i]):.8g} FD={fd:.8g} rel={rel:.2e}"

    H, idxs = fd_utils.analysis_hessian_block(ana, p)
    rep = fd_utils.compare_hessian(H, grad_fn, p, idxs,
                                   h_judge=1e-3, rtol=0.05)
    assert rep.verdict != "FAIL", rep.summary()
