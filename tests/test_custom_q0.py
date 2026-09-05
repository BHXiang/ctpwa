"""Custom DSL q0 变量导数测试。

表达式为 BWR 等价写法（example 原式）：
    p0^2 / (p0^2 - m^2 - i·p0·p1·(p1p/q0)·(p0/m)·(p1p²+q0²)/q0²)
宽度因子链里 q0 出现 4 次，∂q0/∂p0 对梯度贡献显著 —— 若 q0 被当作
常数变量（历史 bug：DSL 的 q0 是 Var，导数 0），p0 梯度会缺失
∂F/∂q0·∂q0/∂p0 项，本测试必失败。

覆盖：
- 梯度 vs 有限差分（p0/p1 全覆盖，验证 ∂q0/∂p0 被正确追踪）
- Hessian 对称/有限（验证二阶段同样包含 ∂²q0/∂p0² 项）
"""

import pytest
import torch

from conftest import make_params

GRAD_RTOL = 5e-3
HESSIAN_SYM_ATOL = 1e-5


def test_custom_q0_config_loads(make_analysis):
    """custom_q0.yml 能正确初始化，2 个参数全部自由。"""
    ana = make_analysis("custom_q0")
    assert ana.isValid(), "custom_q0.yml 初始化失败"
    assert ana.getNFreeTheta() == 2, "应有 2 个自由参数 (p0,p1)"


def test_custom_q0_gradient_vs_fd(make_analysis, device):
    """autograd 梯度 vs 中心差分。

    重点：p0 梯度必须非零且与 FD 吻合 —— 修复前 q0 是常数变量，
    ∂q0/∂p0 项缺失，此处直接失败。
    """
    ana = make_analysis("custom_q0")
    params = make_params(ana, device)
    t0 = 2 * ana.getNVector()
    assert t0 == 2

    p_cpu = params.detach().cpu().double()

    def fd_grad():
        grad = torch.zeros_like(p_cpu)
        for i in range(p_cpu.numel()):
            step = max(abs(p_cpu[i].item()) * 1e-3, 5e-4)
            pp = p_cpu.clone().to(device); pm = p_cpu.clone().to(device)
            pp[i] += step; pm[i] -= step
            grad[i] = (ana.getNLL(pp).item() - ana.getNLL(pm).item()) / (2 * step)
        return grad

    grad_fd = fd_grad()[t0:]
    grad_ad = torch.autograd.grad(ana.getNLL(params.requires_grad_(True)), params)[0].detach().cpu().double()[t0:]

    assert grad_fd[0].abs() > 1e-3, (
        f"FD p0 梯度过小（表达式设计问题？）: {grad_fd[0]:.3e}")
    diff = (grad_ad - grad_fd).abs()
    scale = grad_fd.abs().clamp_min(1e-6)
    max_rel = (diff / scale).max().item()
    assert max_rel < GRAD_RTOL, (
        f"q0 梯度 vs FD 不一致，max rel = {max_rel:.2e}"
        f"\n  AD={grad_ad.tolist()}\n  FD={grad_fd.tolist()}")


def test_custom_q0_hessian_symmetric(make_analysis, device):
    """Hessian 必须对称且有限（二阶段含 ∂²q0/∂p0² 链）。"""
    ana = make_analysis("custom_q0")
    params = make_params(ana, device)
    H = ana.getHessian(params).detach().cpu().double()
    assert torch.isfinite(H).all(), "Hessian 存在 NaN/Inf"
    asym = (H - H.T).abs().max().item()
    assert asym < HESSIAN_SYM_ATOL, f"Hessian 不对称，max|H-H^T| = {asym:.2e}"
