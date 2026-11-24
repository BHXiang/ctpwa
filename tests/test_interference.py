"""writeResult 干涉矩阵回归：宿主累加器必须从零开始。

未初始化的 h_interference_matrix 会让部分矩阵元带上随机垃圾值（实测可达
1e170 以上或 NaN）。除有限性/对称性外，同时用 saved_weight 检查对角元和
总强度守恒，避免只靠矩阵形状或文件存在性漏掉数值错误。
"""

import numpy as np
import pytest
import torch
import uproot

import ctpwa

from conftest import make_params, fit_fraction_matrix, matrix_errors


def _read_matrix(root_file, name="interference_rec"):
    matrix = root_file[name]
    return np.asarray(matrix.member("fElements"), dtype=float).reshape(
        matrix.member("fNrows"), matrix.member("fNcols")
    )


def test_write_result_interference_is_normalized(make_analysis, tmp_path):
    import ctpwa

    ana = make_analysis("no_trans")
    dev = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")
    params = make_params(ana, dev)
    out = tmp_path / "interference.root"
    ana.writeResult(params, str(out), 1)

    with uproot.open(out) as f:
        matrix = _read_matrix(f, "interference_rec")
        tree = f["saved_weight"]
        weight_names = [name for name in tree.keys() if name.startswith("weight_")]
        assert len(weight_names) == matrix.shape[0]

        # totalweight 的积分等于 dataIntegral；weight_i / dataIntegral 的积分
        # 与干涉矩阵对角元 ∫|A_i|² / ∫|ΣA|² 是同一个量。
        data_integral = float(tree["totalweight"].array(library="np").sum())
        assert data_integral > 0
        expected_diag = np.array([
            float(tree[name].array(library="np").sum()) / data_integral
            for name in weight_names
        ])

    assert np.isfinite(matrix).all(), f"干涉矩阵含非有限值:\n{matrix}"
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
    np.testing.assert_allclose(np.diag(matrix), expected_diag, rtol=1e-10, atol=1e-12)
    assert (np.diag(matrix) >= 0).all()

    # 非对角元本身已含因子 2，且 ROOT 矩阵对称存储两份；总强度应为
    # Σ_i M_ii + Σ_{i<j} M_ij = ∫|ΣA|² / ∫|ΣA|² = 1。
    total = np.diag(matrix).sum() + np.triu(matrix, 1).sum()
    assert total == pytest.approx(1.0, rel=1e-10, abs=1e-12)


def test_write_result_interference_rec_gen(make_analysis, tmp_path):
    """weight_best.root 里的两张干涉矩阵：rec / gen，各自除以本样本的 W。

    - interference_rec = P_ij / W_rec（重建侧，带效率 MC）
    - interference_gen = T_ij / W_gen（产生侧 truth/无效率 MC）
    - integrals 树：N_rec/W_rec/N_gen/W_gen/npartials（乘回去就是原始积分）
    - 两张表都满足 trace + Σ_{i<j} = 1；对角即文献口径的 F_i，非对角即 I_ij
    """
    ana = make_analysis("eff2")                       # phsp_truth 与 phsp 同文件
    dev = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")
    params = make_params(ana, dev)
    n = ana.getNVector()
    cpl = torch.complex(params[:n], params[n:2 * n]).contiguous().to(dev)

    out = tmp_path / "interf_rec_gen.root"
    ana.writeResult(params, str(out), 0)

    def matrix(f, name):
        m = f[name]
        return np.asarray(m.member("fElements"), dtype=float).reshape(
            m.member("fNrows"), m.member("fNcols"))

    with uproot.open(out) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        assert {"interference_rec", "interference_gen", "integrals"} <= keys
        assert "interference" not in keys          # 旧名已删除
        assert "amplitude_names" not in keys       # 冗余名字表已删除
        R = matrix(f, "interference_rec")
        G = matrix(f, "interference_gen")
        meta = f["integrals"].arrays(library="np")

    def total(M):
        return M.diagonal().sum() + np.triu(M, 1).sum()

    # 归一化表：闭合到 1
    assert total(R) == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert total(G) == pytest.approx(1.0, rel=1e-12, abs=1e-12)

    # 与矩阵 API 一致：M / W
    P_api = fit_fraction_matrix(ana, cpl, sample="phsp").cpu().numpy()
    T_api = fit_fraction_matrix(ana, cpl, sample="truth").cpu().numpy()
    np.testing.assert_allclose(R, P_api / total(P_api), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(G, T_api / total(T_api), rtol=1e-10, atol=1e-12)

    # integrals 树里的 W 就是分母（乘回矩阵 = 原始积分）
    assert float(meta["W_rec"][0]) == pytest.approx(total(P_api), rel=1e-10)
    assert float(meta["W_gen"][0]) == pytest.approx(total(T_api), rel=1e-10)
    assert float(meta["N_rec"][0]) == float(meta["N_gen"][0])   # eff2: 同一文件
    assert int(meta["npartials"][0]) == R.shape[0]


def test_write_result_skips_raw_matrices_via_env(make_analysis, tmp_path, monkeypatch):
    """CTPWA_WR_NO_INTEGRALS=1: 跳过两张原始矩阵，只留旧的归一化 interference。

    大样本时 writeResult 默认会多扫一遍 phsp/phsp_truth；这个开关是逃生舱，
    保证旧行为仍然可用（不给用户制造"必须重算全样本"的硬约束）。
    """
    import ctpwa

    monkeypatch.setenv("CTPWA_WR_NO_INTEGRALS", "1")
    ana = make_analysis("eff2")
    dev = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")
    params = make_params(ana, dev)
    out = tmp_path / "no_raw.root"
    ana.writeResult(params, str(out), 0)
    with uproot.open(out) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        assert "interference_rec" not in keys
        assert "interference_gen" not in keys
        assert "integrals" not in keys
        assert "legends" in keys            # 其余内容照写
