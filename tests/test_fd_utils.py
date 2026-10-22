"""fd_utils 的自测：确保 FD 比较工具本身能正确区分

    * 真的一致        → PASS
    * 真的不符        → FAIL（并指出是哪个元）
    * 不显著元上的差异 → 忽略（不 FAIL）
    * FD 噪声主导     → INCONCLUSIVE / 归类为 fd_noisy（**绝不能**报 FAIL）

全部为 CPU 上的解析二次型（f = ½pᵀAp, grad = Ap），FD 精确可验，不需要 GPU/config。
"""
import numpy as np
import torch

from fd_utils import (compare_hessian, eig_min_report, fd_hessian,
                      free_idx, H_REL_DEFAULT)


def _quadratic(n=6, seed=0, lam_lo=1e-2, lam_hi=1e3, diag=False):
    """对称正定 A（特征值跨 lam_lo..lam_hi），grad_fn = A p。

    diag=True 时 A 为对角阵 → 非对角元严格为 0（用于测试"不显著元"）。
    """
    g = torch.Generator().manual_seed(seed)
    if diag:
        Q = torch.eye(n, dtype=torch.float64)
    else:
        Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    lam = torch.logspace(np.log10(lam_lo), np.log10(lam_hi), n, dtype=torch.float64)
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)
    def grad(p):
        return A @ p.detach()
    return A, grad


def test_fd_hessian_is_exact_for_quadratic():
    A, grad = _quadratic()
    p = torch.randn(6, dtype=torch.float64) * 0.5
    H = fd_hessian(grad, p)
    assert np.abs(H - A.numpy()).max() < 1e-9


def test_fd_hessian_subset_restricts_rows_and_columns():
    """idxs 为子集时, 行/列都必须切到子集（列不切会与子块解析矩阵错位）。"""
    A, grad = _quadratic()
    p = torch.randn(6, dtype=torch.float64) * 0.5
    idxs = [1, 3, 4]
    H = fd_hessian(grad, p, idxs)
    assert H.shape == (3, 3)
    assert np.abs(H - A.numpy()[np.ix_(idxs, idxs)]).max() < 1e-9


def test_verdict_pass_when_consistent():
    A, grad = _quadratic()
    p = torch.randn(6, dtype=torch.float64)
    rep = compare_hessian(A.numpy(), grad, p)
    assert rep.verdict == "PASS" and rep.n_judged > 0


def test_verdict_fail_on_significant_entry_and_reports_it():
    A, grad = _quadratic()
    p = torch.randn(6, dtype=torch.float64)
    H = A.numpy().copy()
    i, j = 1, 3
    H[i, j] += 5.0 * abs(H[i, j]) + 1.0        # 制造一个显著偏差
    H[j, i] = H[i, j]
    rep = compare_hessian(H, grad, p)
    assert rep.verdict == "FAIL"
    assert set(rep.worst) == {i, j}, rep.worst     # 对称元, 报告任一侧均可
    assert rep.max_rel > 0.2


def test_insignificant_entry_perturbation_is_ignored():
    A, grad = _quadratic(diag=True)             # 对角 A → 非对角元真值为 0
    p = torch.randn(6, dtype=torch.float64)
    H = A.numpy().copy()
    scale = np.abs(H).max()
    i, j = 0, 3
    assert H[i, j] == 0.0
    eps = 1e-4 * scale                          # 扰动后仍低于显著性阈值 1e-3·scale
    H[i, j] += eps; H[j, i] += eps
    assert max(abs(H[i, j]), eps) < 1e-3 * scale
    rep = compare_hessian(H, grad, p, scale=scale)
    assert rep.verdict != "FAIL", rep.summary()
    assert rep.n_insignificant > 0


def test_noisy_fd_never_reported_as_fail():
    """高频微扰模拟 float32/量化噪声: FD 随 h 剧烈变化 → 不得判 FAIL。"""
    A, grad0 = _quadratic()
    eps, delta = 1e-3, 3e-4
    def grad(p):
        p = p.detach()
        return A @ p + eps * torch.sin(p / delta)
    p = torch.randn(6, dtype=torch.float64)
    rep = compare_hessian(A.numpy(), grad, p, sig_frac=1e-6)
    assert rep.verdict != "FAIL", rep.summary()
    assert rep.n_fd_noisy > 0, rep.summary()


def test_eig_min_report_resolvability():
    A, grad = _quadratic()
    p = torch.randn(6, dtype=torch.float64)
    # 一致 → 差远小于噪声上界 → 不可分辨(即"FD 分辨不出差异", 不是不符)
    r = eig_min_report(A.numpy(), grad, p)
    assert not r["resolvable"]
    # 人为把最小特征值推低很多 → 可分辨
    H = A.numpy().copy()
    v = np.linalg.eigh(H)[1][:, 0]
    H = H - 10.0 * np.outer(v, v)
    r2 = eig_min_report(H, grad, p)
    assert r2["resolvable"] and r2["diff"] < 0         # 解析 λmin 更低


def test_free_idx_drops_two_reference_coords():
    class _A:
        def getNVector(self):
            return 3
    assert free_idx(_A(), torch.zeros(7)) == [1, 2, 4, 5, 6]
