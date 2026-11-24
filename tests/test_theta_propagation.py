"""θ（自由共振态参数）必须传进真值/积分振幅的重算路径。

回归：`computeFileBatchIntegrals` 走 `calculateAmplitudes`（从配置粒子对象重建
振幅），那批对象里的 m0/Γ0 是**配置初值**；`reCalcAmp(θ)` 只改驻留振幅。修复前
拿 θ 扰动 10%，`getFitFractionMatrix` 变化 <1e-14（完全没反应）。

判据：重算路径给的矩阵必须等于**驻留振幅**重建出来的矩阵（驻留振幅由
`reComputeAmps` 按 θ 更新，是同一 θ 下的独立实现）。

三个配置覆盖三种参数共享机制：
- eff2              单链、无共享
- with_trans        双链 + trans 约束 + 同名/共轭共振态广播
- constraints_equal  Constraints.var_equal（ghost 槽）
"""

import numpy as np
import torch

import ctpwa
from conftest import fit_fraction_matrix

DEV = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")


def _coupling(ana, seed=5):
    n = ana.getNVector()
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(n, dtype=torch.complex128)
    p[0] = 1.0
    if n > 1:
        p[1:] = ((torch.rand(n - 1, generator=g, dtype=torch.float64) * 0.5
                  + 1j * ((torch.rand(n - 1, generator=g, dtype=torch.float64) - 0.5) * 0.5)).to(DEV))
    return p.contiguous().to(DEV)


def _matrix_from_resident(ana, cpl):
    """用驻留（device）振幅重建 Σw·(|A_i|², 2Re A_iA_j*)。"""
    nsl = ana.getSLVectors().tolist()
    off = np.concatenate([[0], np.cumsum(nsl)])
    A = ana.getPhspTensor().cpu().numpy().astype(np.complex128).reshape(-1, sum(nsl))
    v = ana.getExtendedVector(cpl).cpu().numpy().astype(np.complex128).reshape(-1)
    S = A * v
    parts = [S[:, off[i]:off[i + 1]].sum(1) for i in range(len(nsl))]
    M = np.zeros((len(nsl), len(nsl)))
    for i in range(len(nsl)):
        M[i, i] = float((abs(parts[i]) ** 2).sum())
        for j in range(i + 1, len(nsl)):
            M[i, j] = M[j, i] = float((2 * (parts[i] * parts[j].conj()).real).sum())
    return M


def _check(ana, scales):
    cpl = _coupling(ana)
    th0 = ana.getFreeResParams()[0].double().clone().to(DEV)
    assert ana.getNFreeTheta() > 0, "该配置没有自由 θ，测不出问题"
    try:
        for scale in scales:
            ana.reCalcAmp(th0 * scale)
            M_api = fit_fraction_matrix(ana, cpl, sample="phsp").cpu().numpy()
            M_res = _matrix_from_resident(ana, cpl)
            rel = np.abs(M_api - M_res).max() / np.abs(M_res).max()
            assert rel < 1e-8, f"θ×{scale}: 重算路径与驻留振幅不一致 rel={rel:.2e}"
    finally:
        ana.reCalcAmp(th0)          # 复原，避免污染 cache 里的实例


def test_theta_propagates_single_chain(make_analysis):
    ana = make_analysis("eff2", reuse=False)
    _check(ana, (1.0, 1.10, 1.30))


def test_theta_propagates_multichain_trans(make_analysis):
    """双链 + trans：同名/共轭共振态的参数是广播的，绑定表要覆盖到伙伴。"""
    ana = make_analysis("with_trans", reuse=False)
    _check(ana, (1.0, 1.05, 1.20))


def test_theta_propagates_var_equal(make_analysis):
    """var_equal ghost：r1_mass 与 r2_mass 共享一个 θ 槽，两个共振态都要被写。"""
    ana = make_analysis("constraints_equal", reuse=False)
    _check(ana, (1.0, 1.05, 1.20))


def test_fit_fractions_follow_theta(make_analysis):
    """拟合分数必须随 θ 变（修复前恒等于初始 θ 的值）。"""
    ana = make_analysis("eff2", reuse=False)
    cpl = _coupling(ana, seed=11)
    th0 = ana.getFreeResParams()[0].double().clone().to(DEV)
    try:
        ff0 = ana.getFitFractions(cpl)[:, 0].clone()
        ana.reCalcAmp(th0 * 1.30)
        ff1 = ana.getFitFractions(cpl)[:, 0].clone()
        assert not torch.allclose(ff0, ff1, rtol=1e-6, atol=1e-8), (ff0, ff1)

        # 与矩阵口径自洽：FF = diag(T)/Σ diag(T)（θ 用 reCalcAmp 之后的 1.30 倍）
        n_, nq_ = ana.getNVector(), ana.getNFreeTheta()
        pu = torch.zeros(2 * n_ + nq_, dtype=torch.float64, device=DEV)
        pu[:n_] = cpl.real.double(); pu[n_:2 * n_] = cpl.imag.double()
        if nq_:
            pu[2 * n_:] = (th0 * 1.30).double()
        M = fit_fraction_matrix(ana, pu).cpu().numpy()
        d = np.diag(M)
        assert np.allclose(ff1.cpu().numpy(), d / d.sum(), rtol=1e-8, atol=1e-10)
    finally:
        ana.reCalcAmp(th0)


def _unified(ana, seed=5, scale=1.0):
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    cpl = _coupling(ana, seed)
    p = torch.zeros(2 * n + nθ, dtype=torch.float64, device=DEV)
    p[:n] = cpl.real.double()
    p[n:2 * n] = cpl.imag.double()
    if nθ:
        p[2 * n:] = ana.getFreeResParams()[0].double().to(DEV) * scale
    return p


def test_theta_from_nll_reaches_fractions(make_analysis):
    """getNLL(params) 记录 θ：之后 getFitFractions 必须用这个 θ，而不是配置初值。"""
    ana = make_analysis("eff2", reuse=False)
    n = ana.getNVector()
    cpl = _coupling(ana, seed=13)
    p = _unified(ana, seed=13)

    ff0 = ana.getFitFractions(cpl)[:, 0].clone()
    p2 = p.clone()
    p2[2 * n:] = p[2 * n:] * 1.3
    ana.getNLL(p2)                     # 只求值一次，副作用是记录 θ

    ff1 = ana.getFitFractions(cpl)[:, 0].clone()
    assert not torch.allclose(ff0, ff1, rtol=1e-6, atol=1e-8), (ff0, ff1)

    M = fit_fraction_matrix(ana, p2).cpu().numpy()      # 显式带上 p2 的 θ
    d = np.diag(M)
    assert np.allclose(ff1.cpu().numpy(), d / d.sum(), rtol=1e-8, atol=1e-10)


def test_write_result_raw_matrices_use_current_theta(make_analysis, tmp_path):
    """writeResult 里的原始矩阵必须用传入 params 的 θ（不是配置初值）。"""
    uproot = __import__("pytest").importorskip("uproot")
    ana = make_analysis("eff2", reuse=False)
    n = ana.getNVector()
    p = _unified(ana, seed=17, scale=1.25)
    cpl = torch.complex(p[:n], p[n:2 * n]).contiguous().to(DEV)

    out = tmp_path / "theta_wr.root"
    ana.writeResult(p, str(out), 0)

    def mat(f, name):
        m = f[name]
        return np.asarray(m.member("fElements"), dtype=float).reshape(
            m.member("fNrows"), m.member("fNcols"))

    with uproot.open(out) as f:
        P = mat(f, "interference_rec")
    # writeResult 已把 θ 记入 theta_cur_ → 矩阵 API 与 ROOT 里的应一致
    # （ROOT 里是归一化表：M/W_rec）
    def _total(M):
        return M.diagonal().sum() + np.triu(M, 1).sum()

    # ROOT 里就是归一化表；再用常驻振幅独立复算一遍作判据

    # 且与驻留振幅重建一致（θ 生效的独立判据）
    P_res = _matrix_from_resident(ana, cpl)
    np.testing.assert_allclose(P, P_res / _total(P_res), rtol=1e-10, atol=1e-10)
