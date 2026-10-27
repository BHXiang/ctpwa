"""ctpwa.diagnostics.analyze_hessian 的纯数值回归（不需要 GPU / 不用 ctpwa.so）。

核心语义：贴边且梯度朝外的参数是**活跃约束**，它们对应的"朝外方向"不参与
KKT 二阶条件。getHessian 在那个方向上为负不代表拟合点不是极小。
"""
import torch

from ctpwa.diagnostics import analyze_hessian


def _case(H, p, g, lo, hi, fixed=None):
    n = len(p)
    kw = dict(
        hessian=torch.tensor(H, dtype=torch.float64),
        params=torch.tensor(p, dtype=torch.float64),
        grad=torch.tensor(g, dtype=torch.float64),
        lower=torch.tensor(lo, dtype=torch.float64),
        upper=torch.tensor(hi, dtype=torch.float64),
    )
    if fixed is not None:
        kw["fixed_mask"] = torch.tensor(fixed, dtype=torch.bool)
    return analyze_hessian(**kw)


def test_active_bound_hides_outward_negative_curvature():
    """θ0 在 0 下界、梯度朝外；全空间 λ=-2，但可行子空间只有 +3 → PD。"""
    rep = _case([[-2.0, 0.0], [0.0, 3.0]], [0.0, 1.0], [1.0, 0.1],
                [0.0, 0.0], [2.0, 2.0])
    assert rep.active_mask.tolist() == [True, False]
    assert rep.n_reduced == 1
    assert rep.free_pd is False
    assert rep.reduced_pd is True
    assert rep.reduced_min_eigenvalue == 3.0
    # θ1 在内部，投影梯度就是普通梯度 0.1（只报告，不参与 PD 判定）
    assert abs(rep.kkt_max - 0.1) < 1e-12


def test_boundary_with_inward_gradient_is_not_active():
    """贴边但梯度指向内部 → 不是活跃约束，负曲率必须被报出来。"""
    rep = _case([[-2.0, 0.0], [0.0, 3.0]], [0.0, 1.0], [-1.0, 0.1],
                [0.0, 0.0], [2.0, 2.0])
    assert rep.active_mask.tolist() == [False, False]
    assert rep.reduced_pd is False
    assert rep.n_reduced == 2


def test_fixed_reference_rows_are_ignored():
    """Re_0/Im_0 的恒零行/列不能把 PD 判成 False。"""
    H = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    rep = _case(H, [1.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0],
                fixed=[True, True, False])
    assert rep.n_reduced == 1
    assert rep.free_min_eigenvalue == 2.0
    assert rep.free_pd is True and rep.reduced_pd is True


def test_all_directions_active_is_trivially_pd():
    rep = _case([[-1.0]], [0.0], [1.0], [0.0], [1.0])
    assert rep.n_reduced == 0
    assert rep.reduced_pd is True
    assert rep.kkt_max == 0.0


def test_kkt_projection_uses_bounds():
    """内部点：投影梯度 = 普通梯度；贴边朝外：投影梯度 = 0。"""
    rep = _case([[-1.0]], [0.5], [2.0], [0.0], [1.0])
    assert rep.active_mask.tolist() == [False]
    assert abs(rep.kkt_max - 0.5) < 1e-12
    rep2 = _case([[-1.0]], [0.0], [2.0], [0.0], [1.0])
    assert rep2.active_mask.tolist() == [True]
    assert rep2.kkt_max == 0.0


def test_nonfinite_hessian_is_reported_not_silently_pd():
    H = [[float("nan"), 0.0], [0.0, 1.0]]
    rep = _case(H, [0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    assert rep.reduced_pd is False
    assert "非有限" in rep.note


def test_summary_lists_active_parameter_names():
    rep = _case([[-2.0, 0.0], [0.0, 3.0]], [0.0, 1.0], [1.0, 0.0],
                [0.0, 0.0], [2.0, 2.0])
    s = rep.summary(["mass", "width"])
    assert "reduced_pd=True" in s
    assert "free_pd=False" in s
    assert "mass" in s and "active=" in s


def test_error_mapping_uses_grouped_layout_not_interleaved():
    """getHessian 的布局是 [Re..., Im..., θ]，不是 [Re,Im,Re,Im,...]。

    reduced_idx 故意取 Re1,Re2,Im1,Im2,θ0,θ1；若按旧的交错写法，
    Re2 会拿到 Im1 的 σ。
    """
    from ctpwa.diagnostics import map_errors_to_params

    sd = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    idx = torch.tensor([1, 2, 4, 5, 6, 7])
    cre, cim, res = map_errors_to_params(sd, idx, n_coupling=3, n_theta=2)
    assert cre.tolist() == [0.0, 1.0, 2.0]
    assert cim.tolist() == [0.0, 3.0, 4.0]
    assert res.tolist() == [5.0, 6.0]


def test_error_mapping_marks_active_parameters_nan():
    from ctpwa.diagnostics import map_errors_to_params

    # Re1 是活跃/固定 → 不在 reduced_idx；θ0 活跃
    sd = torch.tensor([7.0, 8.0], dtype=torch.float64)
    idx = torch.tensor([2, 5])  # Re2, Im2
    cre, cim, res = map_errors_to_params(sd, idx, n_coupling=3, n_theta=1)
    assert torch.isnan(cre[1]) and cre[2] == 7.0
    assert torch.isnan(cim[1]) and cim[2] == 8.0
    assert torch.isnan(res[0])
