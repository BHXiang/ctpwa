"""writeResult 干涉矩阵回归：宿主累加器必须从零开始。

未初始化的 h_interference_matrix 会让部分矩阵元带上随机垃圾值（实测可达
1e170 以上或 NaN）。除有限性/对称性外，同时用 saved_weight 检查对角元和
总强度守恒，避免只靠矩阵形状或文件存在性漏掉数值错误。
"""

import numpy as np
import pytest
import torch
import uproot

from conftest import make_params


def _read_matrix(root_file):
    matrix = root_file["interference"]
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
        matrix = _read_matrix(f)
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
