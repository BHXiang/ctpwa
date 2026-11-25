"""干涉矩阵的参数误差（线性化传播）。

判据：
1. 与"独立实现"一致 —— Python 侧对 getFitFractionMatrix 做同样的 ±ε 差分，
   自建 cov = H[free,free]^{-1}，σ = sqrt(J cov Jᵀ)；两条路径只共享定义，不共享代码。
2. 协方差代数：H → 4H 时 σ → σ/2。
3. writeResult 写出的 interference_*_err 与 API 一致，且非负、对称。
4. Hessian 非正定 → 返回空（不糊一个假误差）。
"""

import numpy as np
import pytest
import torch

import ctpwa
from conftest import fit_fraction_matrix, matrix_errors

DEV = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")


def _spd_hessian(nf, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nf, nf))
    return A @ A.T + nf * np.eye(nf)


def _full_hessian(ana, seed=0):
    """SPD Hessian：耦合自由块 **与 θ 块联合**（含耦合-θ 相关）。

    旧版把 θ 块留成单位阵且耦合-θ 块为 0，此时边际协方差 == 条件协方差，
    条件协方差的 bug 测不出来。这里用联合 SPD 随机块做回归。
    """
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    free = [j for j in range(2 * n) if j not in (0, n)]
    free_all = free + [2 * n + k for k in range(nθ)]
    H = np.eye(2 * n + nθ)
    H[np.ix_(free_all, free_all)] = _spd_hessian(len(free_all), seed)
    return torch.tensor(H, dtype=torch.float64).to(DEV), free


def _params(ana, seed=61):
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(2 * n + nθ, dtype=torch.float64, device=DEV)
    p[0] = 1.0
    if n > 1:
        p[1:n] = torch.rand(n - 1, generator=g).to(DEV) * 0.5
        p[n + 1:2 * n] = (torch.rand(n - 1, generator=g).to(DEV) - 0.5) * 0.5
    if nθ:
        p[2 * n:] = ana.getFreeResParams()[0].double().to(DEV)
    return p


def _coupling_from_unified(p, n):
    dt = torch.float32 if int(ctpwa.build_info()["complex_bytes"]) == 8 else torch.float64
    return torch.complex(p[:n].to(dt), p[n:2 * n].to(dt)).contiguous().to(DEV)


def _norm(M):
    return M / (np.diag(M).sum() + np.triu(M, 1).sum())


def _reference(ana, p, free):
    """独立参考实现：同 ε、同归一化、同 cov（联合 Hessian 的边际耦合块）。"""
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    pc = p.detach().cpu().numpy()

    def mats(pu):
        if nθ:
            ana.reCalcAmp(pu[2 * n:].contiguous())
        cpl = _coupling_from_unified(pu, n)
        return (_norm(fit_fraction_matrix(ana, cpl, sample="phsp").numpy()),
                _norm(fit_fraction_matrix(ana, cpl, sample="truth").numpy()))

    Jr, Jg = [], []
    for gidx in free:
        is_theta = gidx >= 2 * n
        eps = (1e-4 * max(abs(float(pc[gidx])), 1e-3) if is_theta
               else 5e-6 * max(abs(float(pc[gidx])), 1.0))
        pp, pm = p.clone(), p.clone()
        pp[gidx] += eps
        pm[gidx] -= eps
        Rp, Gp = mats(pp)
        Rm, Gm = mats(pm)
        Jr.append((Rp - Rm) / (2 * eps))
        Jg.append((Gp - Gm) / (2 * eps))
    return np.array(Jr), np.array(Jg), free


def _marginal_coupling_cov(ana, H, free):
    """Cov_pp = [inv(H_joint)] 的耦合块（不是 inv(H_pp) 条件协方差）。"""
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    free_all = list(free) + [2 * n + k for k in range(nθ)]
    Hf = H.detach().cpu().numpy()[np.ix_(free_all, free_all)]
    return np.linalg.inv(Hf)[:len(free), :len(free)]


def test_error_matrices_match_independent_fd(make_analysis):
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana)
    H, free = _full_hessian(ana)
    err_rec, err_gen = matrix_errors(ana, p, H)
    assert err_rec.shape == err_gen.shape == (ana.getNVector(), ana.getNVector())

    cov = _marginal_coupling_cov(ana, H, free)
    Jr, Jg, _ = _reference(ana, p, free)
    sig = lambda J: np.sqrt(np.einsum("k...,kl,l...->...", J, cov, J))
    np.testing.assert_allclose(err_rec.numpy(), sig(Jr), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(err_gen.numpy(), sig(Jg), rtol=1e-6, atol=1e-12)


def test_error_scales_with_covariance(make_analysis):
    """H → 4H ⇒ cov → cov/4 ⇒ σ → σ/2（纯代数，不依赖物理点）。"""
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana)
    H, _ = _full_hessian(ana)
    e1, _ = matrix_errors(ana, p, H)
    e4, _ = matrix_errors(ana, p, 4.0 * H)
    m = e1.numpy() > 1e-10
    assert m.sum() > 0
    np.testing.assert_allclose(e4.numpy()[m] / e1.numpy()[m], 0.5, rtol=1e-6)


def test_errors_written_to_root(tmp_path, make_analysis):
    uproot = __import__("pytest").importorskip("uproot")
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana, seed=17)
    H, _ = _full_hessian(ana, seed=3)
    out = tmp_path / "err.root"
    ana.writeResult(p, str(out), 0, [], H)

    def mat(f, name):
        m = f[name]
        return np.asarray(m.member("fElements"), dtype=float).reshape(
            m.member("fNrows"), m.member("fNcols"))

    with uproot.open(out) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        assert {"interference_rec_err", "interference_gen_err"} <= keys
        Er, Eg = mat(f, "interference_rec_err"), mat(f, "interference_gen_err")
        Vr, Vg = mat(f, "interference_rec"), mat(f, "interference_gen")

    api_r, api_g = matrix_errors(ana, p, H)
    # FD 依赖原子累加顺序（CPU 后端并行规约），两次调用差 ~1e-10 相对；给 1e-6
    np.testing.assert_allclose(Er, api_r.numpy(), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(Eg, api_g.numpy(), rtol=1e-6, atol=1e-12)
    for E, V in ((Er, Vr), (Eg, Vg)):
        assert np.all(E >= 0) and np.isfinite(E).all()
        np.testing.assert_allclose(E, E.T, rtol=0, atol=1e-12)
        assert E.shape == V.shape


def test_non_pd_hessian_returns_empty(make_analysis, capsys):
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana)
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    H = -torch.eye(2 * n + nθ, dtype=torch.float64, device=DEV)   # 明显非正定
    err_rec, err_gen = matrix_errors(ana, p, H)
    assert err_rec.numel() == 0 and err_gen.numel() == 0


def test_theta_only_non_pd_hessian_returns_empty(make_analysis):
    """完整联合块非正定时，即使 H_cc 为正也不能写误差。"""
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana)
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    assert nθ > 0
    H = torch.eye(2 * n + nθ, dtype=torch.float64, device=DEV) * 100.0
    H[2 * n:, 2 * n:] = -torch.eye(nθ, dtype=torch.float64, device=DEV) * 100.0
    err_rec, err_gen = matrix_errors(ana, p, H)
    assert err_rec.numel() == 0 and err_gen.numel() == 0


def test_errors_do_not_leave_theta_perturbed(make_analysis):
    """误差扫描后，内部当前 θ 必须仍是非初值的拟合 θ。"""
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana, seed=23)
    H, _ = _full_hessian(ana, seed=5)
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    assert nθ > 0
    p[2 * n:] *= 1.10
    cpl = _coupling_from_unified(p, n)
    ana.reCalcAmp(p[2 * n:].contiguous())

    # getFitFractions 使用 analysis 内部当前 θ，不会像 ROOT helper 那样
    # 从配置初值重建 unified params，因此能真正观察状态是否被污染。
    before = ana.getFitFractions(cpl, H)[:, 0].clone()
    matrix_errors(ana, p, H)
    after = ana.getFitFractions(cpl, H)[:, 0]

    assert torch.allclose(before, after, rtol=1e-8, atol=1e-9), (
        (before - after).abs().max().item())


def test_error_matrix_with_multi_ls_partials(make_analysis, tmp_path):
    """回归：分波内的多个 LS 组合（n_amplitudes_ > npartials）时不能越界。

    B̄ 是 n_amplitudes_ × n_amplitudes_，而矩阵是 npartials × npartials；
    v0.3.24 里按 npartials 取 B̄ → 越界（真实 5 体分析上直接崩）。
    ls_filter：1 条链、1 个共振态、4 组 LS ⇒ 矩阵 1×1。
    """
    uproot = pytest.importorskip("uproot")
    ana = make_analysis("ls_filter", reuse=False)
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    nsl = [int(x) for x in ana.getSLVectors()]
    assert sum(nsl) > len(nsl), "该配置应为多 LS 分波（n_amplitudes_ > npartials）"

    p = torch.zeros(2 * n + nθ, dtype=torch.float64, device=DEV)
    p[0] = 1.0
    H = torch.eye(2 * n + nθ, dtype=torch.float64, device=DEV) * 1e2
    out = tmp_path / "ml.root"
    ana.writeResult(p, str(out), 0, [], H)          # 不能崩

    with uproot.open(out) as f:
        o = f["interference_rec_err"]
        nr, nc = int(o.member("fNrows")), int(o.member("fNcols"))
    assert (nr, nc) == (len(nsl), len(nsl)), (nr, nc, nsl)
