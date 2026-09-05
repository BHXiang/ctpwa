"""全同粒子对称化数值测试（ident2: 2×π⁰ 跨子衰变；ident3: 3×π⁰ 含同子衰变稳定子）。

验证内容：
1. coset 拓扑数（构建输出 + getNSigma 不可直接查 → 用对称性间接验证）
2. 振幅对称性：交换全同粒子标签后 NLL 不变（模型必须对称）
3. 梯度 vs 有限差分（FD vs AD，含 σ 求和路径）
4. Hessian 对称 + 有限
"""
import pytest
import torch

from conftest import make_params

FD_EPS = 1e-7
GRAD_RTOL = 5e-3
# Hessian 对称容差: 绝对值 ~1e-6 是 float32 二阶导数的舍入噪声
# （相对 H 最大元素 ~1e4 为 ~1e-10 相对），运行间在 0.5e-6~1.7e-6 波动。
HESSIAN_SYM_ATOL = 5e-6


def _fd_gradient(ana, params, eps=FD_EPS):
    p_cpu = params.detach().cpu().double()
    grad = torch.zeros_like(p_cpu)
    for i in range(p_cpu.numel()):
        step = max(abs(p_cpu[i].item()) * 1e-3, 5e-4)
        p_plus = p_cpu.clone()
        p_minus = p_cpu.clone()
        p_plus[i] += step
        p_minus[i] -= step
        nll_plus = ana.getNLL(p_plus.to(params.device)).item()
        nll_minus = ana.getNLL(p_minus.to(params.device)).item()
        grad[i] = (nll_plus - nll_minus) / (2 * step)
    return grad


@pytest.mark.parametrize("config_key", ["ident2", "ident3"])
def test_nll_symmetric_under_swap(make_analysis, device, config_key):
    """交换数据中全同粒子标签（ident2: pi01↔pi02；ident3: (12) 交换）后 NLL 必须不变。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)
    nll1 = ana.getNLL(params).item()

    import ctpwa
    ana2 = ctpwa.analysis(f"configs/{config_key}_swap.yml")
    nll2 = ana2.getNLL(params).item()
    assert nll1 == pytest.approx(nll2, rel=1e-8), (
        f"{config_key}: 交换标签后 NLL 变化 {nll1} vs {nll2}"
    )


@pytest.mark.parametrize("config_key", ["ident2", "ident3"])
def test_gradient_vs_fd(make_analysis, device, config_key):
    """autograd 梯度 vs 中心差分（覆盖 σ 求和路径的 v 和 θ 梯度）。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device).requires_grad_(True)

    nll = ana.getNLL(params)
    grad_ad = torch.autograd.grad(nll, params)[0].detach().cpu().double()
    grad_fd = _fd_gradient(ana, params)

    n_vec = ana.getNVector()
    mask = torch.ones_like(grad_ad, dtype=torch.bool)
    mask[0] = False      # v_0 实部固定
    mask[n_vec] = False  # v_0 虚部固定

    diff = (grad_ad - grad_fd).abs()
    scale = grad_ad.abs().clamp_min(1e-6)
    rel = diff[mask] / scale[mask]
    max_rel = rel.max().item()
    assert max_rel < GRAD_RTOL, (
        f"{config_key}: 梯度 vs FD 最大相对误差 {max_rel:.2e} > {GRAD_RTOL}"
    )


@pytest.mark.parametrize("config_key", ["ident2", "ident3"])
def test_hessian_symmetric_finite(make_analysis, device, config_key):
    """Hessian 对称 + 有限（覆盖 σ 求和的 Hessian 路径）。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)
    H = ana.getHessian(params).detach().cpu().double()
    asym = (H - H.T).abs().max().item()
    assert asym < HESSIAN_SYM_ATOL, f"{config_key}: Hessian 不对称 max={asym:.2e}"
    assert torch.isfinite(H).all(), f"{config_key}: Hessian 含 NaN/Inf"
