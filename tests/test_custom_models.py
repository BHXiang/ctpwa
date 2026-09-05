"""Custom DSL 模型函数 BW/BWR/Bf/q0/Flatte 数值测试。

DSL 表达式应与内置模型逐点相等（同一链拓扑、同一参数向量）：
    BW(p0, p1)                     == 内置 BW(m, m0, w0)
    BWR(p0, p1, 1, 3.0) * Bf(...)  == 内置 BWR（默认 L=1, d=3.0, has_bf=true）
    Flatte(p0, ...) * Bf(...)      == 内置 Flatte（默认 L=1, d=3.0, has_bf=true）

覆盖：
- NLL 一致（同一参数向量）
- 全梯度一致（验证 DSL 展开的导数链与内置 modelDeriv 等价）
- q0() 函数（1 参 / 3 参形式）梯度 vs 有限差分
"""

import pytest
import torch

from conftest import make_params

NLL_RTOL = 1e-8
GRAD_REL = 1e-4
GRAD_FD_REL = 5e-3


@pytest.mark.parametrize(
    "dsl_key,builtin_key",
    [
        ("custom_bw_func", "custom_bw"),
        ("custom_bwr_func", "bf"),
        ("custom_flatte_func", "flatte"),
        # q0() 函数形式（1 参/3 参）vs 裸 q0 变量（custom_q0.yml）：
        # 本 config 子粒子为 Kp/Km(0.4937)，两种形式数值相等
        ("custom_q0_func", "custom_q0"),
    ],
)
def test_dsl_model_matches_builtin(make_analysis, device, dsl_key, builtin_key):
    """DSL 模型函数 vs 内置模型：自由参数数、NLL、梯度全部一致。"""
    ana_dsl = make_analysis(dsl_key)
    ana_ref = make_analysis(builtin_key)
    assert ana_dsl.isValid(), f"{dsl_key}.yml 初始化失败"
    assert ana_ref.isValid(), f"{builtin_key}.yml 初始化失败"

    assert ana_dsl.getNFreeTheta() == ana_ref.getNFreeTheta(), (
        f"{dsl_key} theta 数 != {builtin_key}")
    assert ana_dsl.getNVector() == ana_ref.getNVector(), (
        f"{dsl_key} vector 数 != {builtin_key}")

    params = make_params(ana_dsl, device)

    nll_dsl = ana_dsl.getNLL(params).item()
    nll_ref = ana_ref.getNLL(params).item()
    assert abs(nll_dsl - nll_ref) <= NLL_RTOL * max(1.0, abs(nll_ref)), (
        f"{dsl_key} NLL {nll_dsl:.6f} != {builtin_key} NLL {nll_ref:.6f}")

    grad_dsl = torch.autograd.grad(
        ana_dsl.getNLL(params.requires_grad_(True)), params)[0]
    grad_ref = torch.autograd.grad(
        ana_ref.getNLL(params.requires_grad_(True)), params)[0]
    diff = (grad_dsl - grad_ref).abs()
    scale = grad_ref.abs().clamp_min(1e-3)
    max_rel = (diff / scale).max().item()
    assert max_rel < GRAD_REL, (
        f"{dsl_key} 梯度 vs {builtin_key} 不一致, max rel = {max_rel:.2e}"
        f"\n  DSL={grad_dsl.tolist()}\n  Ref={grad_ref.tolist()}")


def test_custom_q0_func_gradient_vs_fd(make_analysis, device):
    """q0() 函数（1 参 / 3 参形式）梯度 vs 中心差分。

    重点：∂q0(p0)/∂p0 ≠ 0 必须被追踪 —— 若 q0() 函数分支退化为
    常数（导数 0），p0 梯度缺失该项，直接失败。
    表达式即 custom_q0.yml 的 BWR 等价式（q0 出现 4 次），
    对 p0 高度敏感且归一化不完全抵消 → FD 信噪比好。
    """
    ana = make_analysis("custom_q0_func")
    params = make_params(ana, device)
    t0 = 2 * ana.getNVector()
    assert t0 == 2 and ana.getNFreeTheta() == 2, "q0_func 应有 2 个自由参数"

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
    grad_ad = torch.autograd.grad(
        ana.getNLL(params.requires_grad_(True)), params)[0].detach().cpu().double()[t0:]

    assert grad_fd[0].abs() > 1e-3, (
        f"FD p0 梯度过小（表达式设计问题？）: {grad_fd[0]:.3e}")
    diff = (grad_ad - grad_fd).abs()
    scale = grad_fd.abs().clamp_min(1e-6)
    max_rel = (diff / scale).max().item()
    assert max_rel < GRAD_FD_REL, (
        f"q0() 函数梯度 vs FD 不一致，max rel = {max_rel:.2e}"
        f"\n  AD={grad_ad.tolist()}\n  FD={grad_fd.tolist()}")
