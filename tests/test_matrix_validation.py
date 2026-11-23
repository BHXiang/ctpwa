"""干涉矩阵的独立校验：数学不变量 + 规范不变性 + 误差的 toy-MC 检验。

这些检查不依赖矩阵是怎么算出来的，只依赖它必须满足的数学性质：
1. 归一化后 trace + Σ_{i<j} = 1；对角 ≥ 0；|I_ij| ≤ 2√(F_i F_j)（Cauchy–Schwarz）；
   2Re(G) 型矩阵半正定（G 是逐事例 partial 振幅的 Gram）。
2. 规范不变：所有耦合同乘一个复数（整体模长/相位）→ 份额不变。
3. phsp 与 phsp_truth 指同一文件时 rec 与 gen 必须逐元素相等。
4. 独立路径：rec 矩阵对角 = saved_weight 树里 weight_i/totalweight 的和（另一套 kernel）。
5. 误差：把协方差缩小到线性区，直接从分布抽参数算 F 的经验标准差，应与
   getFitFractionMatrixErrors 给的 σ 一致（toy-MC 检验线性化传播）。
"""

import numpy as np
import pytest
import torch

from conftest import make_params

import ctpwa

DEV = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")


def _coupling(ana, seed=5, scale=1.0, phase=0.0):
    n = ana.getNVector()
    g = torch.Generator().manual_seed(seed)
    p = torch.zeros(n, dtype=torch.complex128)
    p[0] = 1.0
    if n > 1:
        p[1:] = (torch.rand(n - 1, generator=g, dtype=torch.float64) * 0.5
                 + 1j * ((torch.rand(n - 1, generator=g, dtype=torch.float64) - 0.5) * 0.5))
    return (p * scale * np.exp(1j * phase)).contiguous().to(DEV)


def _norm(M):
    return M / (np.diag(M).sum() + np.triu(M, 1).sum())


def test_matrix_invariants(make_analysis):
    ana = make_analysis("eff2", reuse=False)
    cpl = _coupling(ana, seed=7)
    for sample in ("phsp", "truth"):
        M = ana.getFitFractionMatrix(cpl, sample=sample).cpu().numpy()
        F = _norm(M)
        assert abs(np.trace(F) + np.triu(F, 1).sum() - 1.0) < 1e-12
        d = np.diag(F)
        assert np.all(d >= -1e-14), d
        # Cauchy–Schwarz: |∫2Re(A_iA_j*)| ≤ 2√(∫|A_i|² ∫|A_j|²)
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                assert abs(F[i, j]) <= 2 * np.sqrt(d[i] * d[j]) + 1e-12
        # 2Re(Gram) 半正定
        assert np.linalg.eigvalsh(F).min() > -1e-10


def test_matrix_gauge_invariance(make_analysis):
    """整体模长/相位是规范自由度：份额不能变。"""
    ana = make_analysis("eff2", reuse=False)
    c0 = _coupling(ana, seed=9)
    ref = {s: _norm(ana.getFitFractionMatrix(c1, sample=s).cpu().numpy())
           for s in ("phsp", "truth") for c1 in [c0]}
    c1 = _coupling(ana, seed=9, scale=2.5, phase=0.7)
    for s in ("phsp", "truth"):
        F = _norm(ana.getFitFractionMatrix(c1, sample=s).cpu().numpy())
        assert np.allclose(F, ref[s], rtol=1e-10, atol=1e-12), (s, F, ref[s])


def test_rec_equals_gen_for_identical_samples(make_analysis):
    """eff2 的 phsp 与 phsp_truth 是同一文件 → 两张表必须逐元素相等。"""
    ana = make_analysis("eff2", reuse=False)
    cpl = _coupling(ana, seed=11)
    R = _norm(ana.getFitFractionMatrix(cpl, sample="phsp").cpu().numpy())
    G = _norm(ana.getFitFractionMatrix(cpl, sample="truth").cpu().numpy())
    np.testing.assert_allclose(R, G, rtol=1e-10, atol=1e-12)


def test_rec_diagonal_matches_saved_weight_tree(make_analysis, tmp_path):
    """独立 kernel 路径: rec 矩阵对角 = Σ weight_i / Σ totalweight（saved_weight 树）。

    saved_weight 的权重来自 computeResults（逐事例 partial 强度），矩阵来自
    computeBranchingFractions（Gram 累加）——两条实现互相印证。
    仅对未加 phsp_weights 的配置成立（树上不带逐事例权重）。
    """
    uproot = pytest.importorskip("uproot")
    ana = make_analysis("no_trans", reuse=False)
    params = make_params(ana, DEV)
    n = ana.getNVector()
    cpl = torch.complex(params[:n], params[n:2 * n]).contiguous()

    out = tmp_path / "w.root"
    ana.writeResult(params, str(out), 1)

    with uproot.open(out) as f:
        t = f["saved_weight"]
        names = [k for k in t.keys() if k.startswith("weight_")]
        total = float(t["totalweight"].array(library="np").sum())
        diag = np.array([float(t[k].array(library="np").sum()) / total for k in names])
        R = f["interference_rec"]
        M = np.asarray(R.member("fElements"), dtype=float).reshape(
            R.member("fNrows"), R.member("fNcols"))
    np.testing.assert_allclose(np.diag(M), diag, rtol=1e-10, atol=1e-12)


def test_error_by_toy_monte_carlo(make_analysis):
    """把协方差缩到线性区，用 toy 抽参数：经验 std 应等于线性化 σ。"""
    ana = make_analysis("eff2", reuse=False)
    n, nθ = ana.getNVector(), ana.getNFreeTheta()
    free = [j for j in range(2 * n) if j not in (0, n)]   # θ 暂不传播
    nf = len(free)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(nf, nf))
    base = (A @ A.T + nf * np.eye(nf)) / nf          # 单位量级的相关结构

    p0 = torch.zeros(2 * n + nθ, dtype=torch.float64, device=DEV)
    c0 = _coupling(ana, seed=13)
    p0[:n] = c0.real
    p0[n:2 * n] = c0.imag
    if nθ:
        p0[2 * n:] = ana.getFreeResParams()[0].double().to(DEV)

    # 每个参数只抖 0.1%（θ 的宽度只有几 MeV，绝对值不能统一取）→ 线性化成立
    scale = np.array([max(abs(float(p0[g])), 1e-3) for g in free]) * 1e-2
    cov = base * np.outer(scale, scale)
    H = np.eye(2 * n + nθ)
    H[np.ix_(free, free)] = np.linalg.inv(cov)
    H_t = torch.tensor(H, dtype=torch.float64).to(DEV)

    err_rec, err_gen = ana.getFitFractionMatrixErrors(p0, H_t)
    assert err_rec.numel() > 0

    def F_of(pu):
        if nθ:
            ana.reCalcAmp(pu[2 * n:].contiguous())
        c = torch.complex(pu[:n], pu[n:2 * n]).contiguous().to(DEV)
        return _norm(ana.getFitFractionMatrix(c, sample="truth").cpu().numpy())

    ntoy = 60
    samples = []
    L = np.linalg.cholesky(cov)
    for _ in range(ntoy):
        delta = L @ rng.normal(size=nf)
        pu = p0.clone()
        pu[torch.tensor(free, device=DEV)] += torch.tensor(
            delta, dtype=torch.float64, device=DEV)
        samples.append(F_of(pu))
    ana.reCalcAmp(p0[2 * n:].contiguous()) if nθ else None
    emp = np.std(np.array(samples), axis=0, ddof=1)
    pred = err_gen.cpu().numpy()
    mask = pred > 1e-6 * pred.max()
    assert mask.sum() >= 4
    rel = np.abs(emp[mask] - pred[mask]) / pred[mask]
    # 60 个 toy 的 std 本身有 ~9% 统计误差；非线性再留余量
    assert np.median(rel) < 0.15, (np.median(rel), emp[mask], pred[mask])
