"""干涉矩阵的参数误差（线性化传播）。

判据：
1. 与"独立实现"一致 —— Python 侧对 getFitFractionMatrix 做同样的 ±ε 差分，
   自建 cov = H[free,free]^{-1}，σ = sqrt(J cov Jᵀ)；两条路径只共享定义，不共享代码。
2. 协方差代数：H → 4H 时 σ → σ/2。
3. writeResult 写出的 interference_*_err 与 API 一致，且非负、对称。
4. Hessian 非正定 → 返回空（不糊一个假误差）。
"""

import numpy as np
import torch

import ctpwa

DEV = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")


def _spd_hessian(nf, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nf, nf))
    return A @ A.T + nf * np.eye(nf)


def _full_hessian(ana, seed=0):
    """把 SPD 块塞进自由方向（耦合去掉参考 + θ），固定方向留单位阵。"""
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    free = [j for j in range(2 * n) if j not in (0, n)]   # θ 暂不传播
    H = np.eye(2 * n + nθ)
    H[np.ix_(free, free)] = _spd_hessian(len(free), seed)
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


def _norm(M):
    return M / (np.diag(M).sum() + np.triu(M, 1).sum())


def _reference(ana, p, free):
    """独立参考实现：同 ε、同归一化、同 cov。"""
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    pc = p.detach().cpu().numpy()

    def mats(pu):
        if nθ:
            ana.reCalcAmp(pu[2 * n:].contiguous())
        cpl = torch.complex(pu[:n], pu[n:2 * n]).contiguous().to(DEV)
        return (_norm(ana.getFitFractionMatrix(cpl, sample="phsp").numpy()),
                _norm(ana.getFitFractionMatrix(cpl, sample="truth").numpy()))

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
    return np.array(Jr), np.array(Jg)


def test_error_matrices_match_independent_fd(make_analysis):
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana)
    H, free = _full_hessian(ana)
    err_rec, err_gen = ana.getFitFractionMatrixErrors(p, H)
    assert err_rec.shape == err_gen.shape == (ana.getNVector(), ana.getNVector())

    cov = np.linalg.inv(H.detach().cpu().numpy()[np.ix_(free, free)])
    Jr, Jg = _reference(ana, p, free)
    sig = lambda J: np.sqrt(np.einsum("k...,kl,l...->...", J, cov, J))
    np.testing.assert_allclose(err_rec.numpy(), sig(Jr), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(err_gen.numpy(), sig(Jg), rtol=1e-6, atol=1e-12)


def test_error_scales_with_covariance(make_analysis):
    """H → 4H ⇒ cov → cov/4 ⇒ σ → σ/2（纯代数，不依赖物理点）。"""
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana)
    H, _ = _full_hessian(ana)
    e1, _ = ana.getFitFractionMatrixErrors(p, H)
    e4, _ = ana.getFitFractionMatrixErrors(p, 4.0 * H)
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

    api_r, api_g = ana.getFitFractionMatrixErrors(p, H)
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
    err_rec, err_gen = ana.getFitFractionMatrixErrors(p, H)
    assert err_rec.numel() == 0 and err_gen.numel() == 0


def test_errors_do_not_leave_theta_perturbed(make_analysis):
    """getFitFractionMatrixErrors 内部会扰 θ；出函数必须还原。

    回归：残留的 θ 会让随后 getFitFractionMatrix / getFitFractions 用扰动过的
    振幅算（实测份额第 5 位就偏），而且极难察觉。
    """
    ana = make_analysis("eff2", reuse=False)
    p = _params(ana, seed=23)
    H, _ = _full_hessian(ana, seed=5)
    n = ana.getNVector()

    before = ana.getFitFractionMatrix(
        torch.complex(p[:n], p[n:2 * n]).contiguous().to(DEV), sample="truth").clone()
    ana.getFitFractionMatrixErrors(p, H)
    after = ana.getFitFractionMatrix(
        torch.complex(p[:n], p[n:2 * n]).contiguous().to(DEV), sample="truth")

    assert torch.allclose(before, after, rtol=1e-12, atol=1e-9), (
        (before - after).abs().max().item())
