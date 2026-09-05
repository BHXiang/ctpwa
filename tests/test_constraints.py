"""命名变量约束测试（fix_var/free_var/var_range/var_equal/gauss_constr）。

baseline: constraints.yml — r1/r2 两个共振态各 2 个自由参数 (mass, width) → 4 theta。
变体 config 与 baseline 唯一区别是 Constraints 键。

结构不变量:
- fix_var: 自由参数数减少，参数名不含固定名
- free_var: 取消 fix_var 的固定
- var_range: getFreeResParams 返回覆盖后的边界
- var_equal: getNFreeTheta 减少，参数共享（NLL 一致）
- gauss_constr: 罚项 Σ(x-μ)²/(2σ²) 精确加入 NLL/梯度/Hessian
"""

import pytest
import torch

from conftest import make_params


def _theta_index(ana, name):
    """返回 theta 参数名在统一参数向量中的下标。"""
    n_vec = ana.getNVector()
    theta_names = ana.getParamNames()[n_vec:]
    return n_vec * 2 + theta_names.index(name)


# ============================================================
# 结构测试
# ============================================================

def test_constraints_baseline(make_analysis):
    """baseline: 4 个自由 theta 参数，参数名与数量一致。"""
    ana = make_analysis("constraints")
    assert ana.getNFreeTheta() == 4
    names = ana.getParamNames()
    assert len(names) == ana.getNVector() + ana.getNFreeTheta()


def test_fix_var_reduces_free(make_analysis):
    """fix_var: r1_mass 固定 → 3 theta，参数名不含固定名。"""
    ana = make_analysis("constraints_fix")
    assert ana.getNFreeTheta() == 3
    names = ana.getParamNames()
    assert len(names) == ana.getNVector() + ana.getNFreeTheta()
    assert not any("r1_mass" in n for n in names), "固定参数不应出现在参数名中"
    # 固定值应生效: r1 的初值信息中不应再有 r1_mass（被固定即无槽）
    info = ana.getFreeResParams()
    assert info.shape[1] == 3


def test_free_var_cancels_fix(make_analysis):
    """fix_var + free_var: r2_width 恢复自由 → 3 theta，名字保留 r2_width。"""
    ana = make_analysis("constraints_free")
    assert ana.getNFreeTheta() == 3
    names = ana.getParamNames()
    assert not any("r1_mass" in n for n in names), "r1_mass 仍应固定"
    assert any("r2_width" in n for n in names), "free_var 应恢复 r2_width 自由"


def test_var_range_override(make_analysis):
    """var_range: r1_mass 边界覆盖为 [1.5, 1.9]。"""
    ana = make_analysis("constraints_range")
    assert ana.getNFreeTheta() == 4
    n_vec = ana.getNVector()
    theta_names = ana.getParamNames()[n_vec:]
    i = theta_names.index("r1_mass")
    info = ana.getFreeResParams()  # [3, n]: init, lower, upper
    assert info[1, i].item() == pytest.approx(1.5)
    assert info[2, i].item() == pytest.approx(1.9)
    # 其他参数边界不变
    j = theta_names.index("r2_mass")
    assert info[1, j].item() == pytest.approx(1.55 - 0.5 * 1.55)
    assert info[2, j].item() == pytest.approx(1.55 + 0.5 * 1.55)


def test_var_equal_reduces_free(make_analysis):
    """var_equal: r1_mass 与 r2_mass 共享 → 3 theta。"""
    ana = make_analysis("constraints_equal")
    assert ana.getNFreeTheta() == 3
    names = ana.getParamNames()
    assert len(names) == ana.getNVector() + ana.getNFreeTheta()
    assert not any("r2_mass" in n for n in names), "共享成员不应有独立参数槽"


# ============================================================
# var_equal 数值一致性
# ============================================================

def test_var_equal_shared_value(make_analysis, device):
    """var_equal: 共享参数取同一值时两个 config 的 NLL 一致。

    注意: var_equal 中 r2_mass 无独立槽，运行时被广播为 owner (r1_mass) 的值。
    所以对比时 baseline 必须把 r1_mass 与 r2_mass 设为同一值。
    """
    ana_base = make_analysis("constraints")
    ana_eq = make_analysis("constraints_equal")
    p_base = make_params(ana_base, device)
    p_eq = make_params(ana_eq, device)

    i1 = _theta_index(ana_base, "r1_mass")
    i2 = _theta_index(ana_base, "r2_mass")
    i_e = _theta_index(ana_eq, "r1_mass")

    for delta in [0.0, 0.03, -0.02]:
        # baseline: r1_mass 与 r2_mass 设为同一值（= init + delta）
        p_b2 = p_base.clone()
        p_b2[i1] += delta
        p_b2[i2] = p_b2[i1]
        # var_equal: 只移动共享参数 r1_mass
        p_e2 = p_eq.clone()
        p_e2[i_e] += delta

        nll_b = ana_base.getNLL(p_b2).item()
        nll_e = ana_eq.getNLL(p_e2).item()
        assert nll_e == pytest.approx(nll_b, rel=1e-8), (
            f"var_equal 广播不一致 (delta={delta}): eq={nll_e:.10f} base={nll_b:.10f}"
        )


def test_var_equal_recalc_amp(make_analysis, device):
    """var_equal: reCalcAmp 后广播生效（振幅随 owner 参数变化）。"""
    ana = make_analysis("constraints_equal")
    p = make_params(ana, device)
    i_e = _theta_index(ana, "r1_mass")
    nll0 = ana.getNLL(p).item()
    p2 = p.clone()
    p2[i_e] += 0.05
    nll1 = ana.getNLL(p2).item()
    assert nll1 != pytest.approx(nll0, abs=1e-6), "共享参数应影响 NLL"


# ============================================================
# gauss_constr 数值测试
# ============================================================

def test_gauss_penalty_zero_at_mu(make_analysis, device):
    """gauss_constr: θ=init(=μ) 时罚项为 0 → NLL 与 baseline 一致。"""
    ana_base = make_analysis("constraints")
    ana_g = make_analysis("constraints_gauss")
    nll_b = ana_base.getNLL(make_params(ana_base, device)).item()
    nll_g = ana_g.getNLL(make_params(ana_g, device)).item()
    assert nll_g == pytest.approx(nll_b, rel=1e-8)


def test_gauss_penalty_quadratic(make_analysis, device):
    """gauss_constr: 偏离 μ 时 NLL 增加 Δ²/(2σ²)。"""
    ana_base = make_analysis("constraints")
    ana_g = make_analysis("constraints_gauss")
    p_base = make_params(ana_base, device)
    p_g = make_params(ana_g, device)

    i_b = _theta_index(ana_base, "r1_mass")
    i_g = _theta_index(ana_g, "r1_mass")
    delta = 0.05
    sigma = 0.01
    p_b2 = p_base.clone(); p_b2[i_b] += delta
    p_g2 = p_g.clone(); p_g2[i_g] += delta

    nll_b = ana_base.getNLL(p_b2).item()
    nll_g = ana_g.getNLL(p_g2).item()
    expected_penalty = delta * delta / (2 * sigma * sigma)
    assert nll_g == pytest.approx(nll_b + expected_penalty, rel=1e-8), (
        f"罚项错误: gauss={nll_g:.10f} base+pen={nll_b+expected_penalty:.10f}"
    )


def test_gauss_gradient_penalty(make_analysis, device):
    """gauss_constr 梯度: (gauss_AD − base_AD) == 罚项梯度 (x−μ)/σ²。

    不用 AD vs 有限差分做主断言: phsp 积分以 float32 累加（量化台阶 ~1e-4），
    FD 噪声在 h=1e-5 处 ~±10、h=1e-4 处 ~±2，无法达到 1e-4 相对精度。
    罚项梯度是解析量，用两个 config 的 AD 之差精确验证（物理梯度在差中消掉）。
    """
    ana_base = make_analysis("constraints")
    ana_g = make_analysis("constraints_gauss")
    sigma = 0.01
    i_b = _theta_index(ana_base, "r1_mass")
    i_g = _theta_index(ana_g, "r1_mass")

    ad_g = None
    for delta in [0.0, 0.005, -0.01]:
        p_base = make_params(ana_base, device).requires_grad_(True)
        p_g = make_params(ana_g, device).requires_grad_(True)
        with torch.no_grad():
            p_base[i_b] += delta
            p_g[i_g] += delta
        grad_b = torch.autograd.grad(ana_base.getNLL(p_base), p_base)[0].cpu().double()
        grad_g = torch.autograd.grad(ana_g.getNLL(p_g), p_g)[0].cpu().double()
        if delta == 0.0:
            ad_g = grad_g[i_g].item()
        expected = delta / (sigma * sigma)
        got = (grad_g[i_g] - grad_b[i_b]).item()
        # abs=1e-9: delta=0 时 expected=0，rel 容差失效（rel×0=0），
        # 而物理梯度差本身有 ~1e-12 的浮点累积噪声 → 必须给绝对容差。
        assert got == pytest.approx(expected, rel=1e-4, abs=1e-9), (
            f"罚项梯度错误 (delta={delta}): got={got:.6e} expected={expected:.6e}"
        )

    # 松散的 FD 健全性检查（float32 量化噪声 ~±2，取绝对容差 5）
    p0 = make_params(ana_g, device)
    h = 1e-4
    p_p = p0.clone(); p_p[i_g] += h
    p_m = p0.clone(); p_m[i_g] -= h
    fd = (ana_g.getNLL(p_p).item() - ana_g.getNLL(p_m).item()) / (2 * h)
    assert ad_g == pytest.approx(fd, abs=5.0), (
        f"AD={ad_g:.6e} FD={fd:.6e} 超出 float32 量化噪声"
    )


def test_gauss_hessian_diag(make_analysis, device):
    """gauss_constr Hessian: θθ 对角 += 1/σ²。"""
    ana_base = make_analysis("constraints")
    ana_g = make_analysis("constraints_gauss")
    p_base = make_params(ana_base, device)
    p_g = make_params(ana_g, device)

    H_b = ana_base.getHessian(p_base).cpu()
    H_g = ana_g.getHessian(p_g).cpu()

    i_b = _theta_index(ana_base, "r1_mass")
    i_g = _theta_index(ana_g, "r1_mass")
    expected = 1.0 / (0.01 * 0.01)
    got = H_g[i_g, i_g].item() - H_b[i_b, i_b].item()
    assert got == pytest.approx(expected, rel=1e-4), (
        f"Hessian 对角增量错误: got={got:.6e} expected={expected:.6e}"
    )
