"""势垒因子 (has_bf/bf_d) 三级作用域数值验证。

核心恒等式: Bf(L, q, q0, d=0) ≡ 1.0（所有 L 的 Blatt-Weisskopf 多项式在
z=0 处取值 1），所以 `has_bf: false` 与 `bf_d: 0.0` 物理上完全等价。
据此构造"殊途同归"配置对，验证三级作用域解析:

    per-step opt > ResonanceConfig(母粒子) > Constraints 全局 > 默认 (true, 3.0)

- bf.yml          : 无任何配置（默认值）
- bf_d3.yml       : Constraints.bf_d=3.0（显式默认）→ 与 bf.yml 一致
- bf_nobf.yml     : Constraints.has_bf=false（全局关）
- bf_global0.yml  : Constraints.bf_d=0.0（全局 Bf≡1）→ 与 bf_nobf.yml 一致
- bf_res4.yml     : 全局 0.0 + Resonance R_KK bf_d=4.0 → R_KK 节点=4.0
- bf_steps.yml    : step 级 Jpsi=0.0 / R_KK=4.0 → 与 bf_res4.yml 一致
- bf_step0.yml    : 全局 0.0 + 共振态 4.0 + step 级 0.0 → step 胜出 → 全 0.0
- bf_indirect.yml : 共振态名 r1 ≠ 中间态名 R_KK 的间接查找 → 与 bf_steps.yml 一致
- bf_nobf2.yml    : Jpsi 步 step 级 has_bf=false + R_KK Resonance has_bf=false
                    → 与 bf_nobf.yml 一致

NLL 对比用 rel=1e-5（phsp 积分的 float32 量化台阶 ~1e-4 绝对）；
梯度对比用 abs=1e-6（两次独立分析间的浮点累积噪声 ~1e-12）。
"""

import pytest
import torch

from conftest import make_params


def _nll_grad(ana, device):
    """构造初始参数并返回 (NLL, 梯度)。"""
    p = make_params(ana, device).requires_grad_(True)
    nll = ana.getNLL(p).item()
    grad = torch.autograd.grad(ana.getNLL(p), p)[0].cpu().double()
    return nll, grad


def _assert_same_physics(ana_a, ana_b, device):
    """两个配置应给出完全相同的物理（NLL + 梯度）。"""
    nll_a, grad_a = _nll_grad(ana_a, device)
    nll_b, grad_b = _nll_grad(ana_b, device)
    assert nll_a == pytest.approx(nll_b, rel=1e-5), (
        f"NLL 不一致: {nll_a:.8f} vs {nll_b:.8f}"
    )
    assert torch.allclose(grad_a, grad_b, atol=1e-6), (
        f"梯度不一致: max diff={(grad_a - grad_b).abs().max().item():.3e}"
    )


def test_bf_defaults_unchanged(make_analysis, device):
    """显式 bf_d=3.0 与完全默认一致（默认值未被破坏）。"""
    _assert_same_physics(make_analysis("bf"), make_analysis("bf_d3"), device)


def test_has_bf_false_equals_bf_d_zero(make_analysis, device):
    """has_bf=false ≡ bf_d=0.0（Bf≡1）——证明 has_bf 真正移除势垒因子。"""
    _assert_same_physics(make_analysis("bf_nobf"), make_analysis("bf_global0"), device)


def test_resonance_overrides_global(make_analysis, device):
    """共振态级 bf_d 覆盖全局: (全局0.0 + 共振态4.0) == step 级 (0.0, 4.0)。"""
    _assert_same_physics(make_analysis("bf_res4"), make_analysis("bf_steps"), device)


def test_step_overrides_resonance(make_analysis, device):
    """step 级 bf_d 覆盖共振态级: (全局0.0 + 共振态4.0 + step 0.0) == 全 0.0。"""
    _assert_same_physics(make_analysis("bf_step0"), make_analysis("bf_global0"), device)


def test_intermediate_name_indirection(make_analysis, device):
    """共振态名(r1) ≠ 中间态名(R_KK) 时，共振态级 bf_d 仍生效（链内查找）。"""
    _assert_same_physics(make_analysis("bf_indirect"), make_analysis("bf_steps"), device)


def test_has_bf_resonance_and_step_levels(make_analysis, device):
    """has_bf=false 在 resonance + step 级生效 == 全局关。"""
    _assert_same_physics(make_analysis("bf_nobf2"), make_analysis("bf_nobf"), device)


def test_barrier_really_applies_by_default(make_analysis, device):
    """健全性: 默认有势垒因子时物理 ≠ 无势垒因子（Bf 真实参与计算）。"""
    nll_bf, _ = _nll_grad(make_analysis("bf"), device)
    nll_nobf, _ = _nll_grad(make_analysis("bf_nobf"), device)
    assert nll_bf != pytest.approx(nll_nobf, rel=1e-3), (
        "默认 has_bf=true 应与 has_bf=false 显著不同"
    )
