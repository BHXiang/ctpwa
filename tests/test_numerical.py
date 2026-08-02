"""L1 数值一致性测试（分钟级，发布前必跑）。

用固定 seed 的合成数据验证数值正确性：
- NLL 可复现性（防 GPU 缓冲/同步 bug）
- 梯度 vs 有限差分
- Hessian 对称性
- golden value 回归（固定输入 → 固定输出）

注意：所有测试都是确定性的（固定 seed + 固定数据 + 固定参数），
同一 commit 下应稳定通过。
"""

import pytest
import torch
from pathlib import Path

from conftest import make_params, load_golden

# 有限差分步长
FD_EPS = 1e-7
# 数值容差（GPU 双精度计算；FD 有截断+舍入误差，
# 实测 rel ~1e-3~2e-3，放宽到 5e-3 抓结构性错误）
GRAD_RTOL = 5e-3
HESSIAN_SYM_ATOL = 1e-6


# ============================================================
# NLL 基础性质
# ============================================================

@pytest.mark.parametrize("config_key", ["simple", "no_trans", "with_trans"])
def test_nll_reproducible(make_analysis, device, config_key):
    """同一参数两次计算 NLL，结果必须完全一致（防 GPU 状态残留）。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)

    nll1 = ana.getNLL(params).item()
    nll2 = ana.getNLL(params).item()
    assert nll1 == pytest.approx(nll2, abs=1e-12), (
        f"{config_key}: NLL 不可复现 {nll1} vs {nll2}"
    )


@pytest.mark.parametrize("config_key", ["simple", "no_trans", "with_trans"])
def test_nll_finite(make_analysis, device, config_key):
    """NLL 必须为有限实数。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)
    nll = ana.getNLL(params).item()
    assert torch.isfinite(torch.tensor(nll)), f"{config_key}: NLL 非有限: {nll}"


def test_nll_autograd_tensor(make_analysis, device):
    """getNLL 返回值应能参与 autograd 求梯度。"""
    ana = make_analysis("simple")
    params = make_params(ana, device).requires_grad_(True)
    nll = ana.getNLL(params)
    grad = torch.autograd.grad(nll, params, retain_graph=True)[0]
    assert grad.shape == params.shape
    assert torch.isfinite(grad).all(), "梯度存在 NaN/Inf"
    assert grad.abs().max() > 0, "梯度不应全为 0"


# ============================================================
# 梯度 vs 有限差分
# ============================================================

def _fd_gradient(ana, params, eps=FD_EPS):
    """中心差分梯度（在 CPU 上做扰动）。

    步长用 rel=1e-3（经验最优：BWR 传播子非线性下
    rel=1e-2 截断误差 ~2%，rel=1e-4 舍入误差 ~0.6%）。
    """
    p_cpu = params.detach().cpu().double()
    grad = torch.zeros_like(p_cpu)
    for i in range(p_cpu.numel()):
        # 步长: rel=1e-3 对大多数参数最优；小耦合参数 (|p|~0.005) 时
        # rel=1e-3 的绝对步长太小 (5e-6) 会引入舍入误差，需绝对下限
        step = max(abs(p_cpu[i].item()) * 1e-3, 5e-4)
        p_plus = p_cpu.clone()
        p_minus = p_cpu.clone()
        p_plus[i] += step
        p_minus[i] -= step
        nll_plus = ana.getNLL(p_plus.to(params.device)).item()
        nll_minus = ana.getNLL(p_minus.to(params.device)).item()
        grad[i] = (nll_plus - nll_minus) / (2 * step)
    return grad


@pytest.mark.parametrize("config_key", ["simple", "with_trans"])
def test_gradient_vs_fd(make_analysis, device, config_key):
    """autograd 梯度 vs 中心差分，相对误差 < 1e-3。

    注意：这一步较慢（每个参数 2 次 forward），只测两个 config。
    """
    ana = make_analysis(config_key)
    params = make_params(ana, device).requires_grad_(True)

    nll = ana.getNLL(params)
    grad_ad = torch.autograd.grad(nll, params)[0].detach().cpu().double()
    grad_fd = _fd_gradient(ana, params)

    # 排除固定参考参数 (v_0 实部=1 固定，虚部=0 固定)，它们不应有梯度依赖
    n_vec = ana.getNVector()
    mask = torch.ones_like(grad_ad, dtype=torch.bool)
    mask[0] = False      # v_0 实部固定
    mask[n_vec] = False  # v_0 虚部固定

    if mask.any():
        diff = (grad_ad - grad_fd).abs()
        scale = grad_ad.abs().clamp_min(1e-6)
        rel = diff[mask] / scale[mask]
        max_rel = rel.max().item()
        assert max_rel < GRAD_RTOL, (
            f"{config_key}: 梯度 vs FD 最大相对误差 {max_rel:.2e} > {GRAD_RTOL}"
            f"\n  AD={grad_ad[mask][rel.argmax()].item():.6e}"
            f"\n  FD={grad_fd[mask][rel.argmax()].item():.6e}"
        )


# ============================================================
# Hessian 性质
# ============================================================

def test_hessian_symmetric(make_analysis, device):
    """Hessian 必须对称（H_ij == H_ji）。"""
    ana = make_analysis("simple")
    params = make_params(ana, device)
    H = ana.getHessian(params).detach().cpu().double()
    asym = (H - H.T).abs().max().item()
    assert asym < HESSIAN_SYM_ATOL, f"Hessian 不对称，max|H-H^T| = {asym:.2e}"


def test_hessian_finite(make_analysis, device):
    """Hessian 必须为有限值。"""
    ana = make_analysis("simple")
    params = make_params(ana, device)
    H = ana.getHessian(params).detach().cpu()
    assert torch.isfinite(H).all(), "Hessian 存在 NaN/Inf"


# ============================================================
# 异常参数保护（真实拟合场景）
# ============================================================
# 设计原则：只测真实场景下可能出现的参数。
#   - 耦合 → 0：拟合中某个振幅耦合可能收敛到 0（factor==0 保护路径）
#   - 耦合极大：优化器试探大值
# 不测 theta=0 等非法输入（theta 由物理 bounds 约束，正常流程不会出现）。

@pytest.mark.parametrize("config_key", ["no_trans", "with_trans"])
def test_nll_coupling_zero_protection(make_analysis, device, config_key):
    """自由耦合全 0（边界输入）时：NLL 与梯度必须有限。

    注意：v0=1 固定参考振幅使 factor 恒非零，通常不触发 1e30 保护；
    本测试验证的是边界输入不产生 NaN——保护机制（factor==0 → log(1e-10)
    兜底、loss 重置清零梯度）的回归防线。
    """
    ana = make_analysis(config_key)
    n_vec = ana.getNVector()
    assert n_vec > 1, f"{config_key}: 应有自由耦合参数"
    n_th = ana.getNFreeTheta()

    p = torch.zeros(2 * n_vec + n_th, dtype=torch.float64, device=device)
    p[0] = 1.0
    p[n_vec] = 0.0
    # 自由耦合全 0（v0=1 固定参考除外），theta 用标称值（合法输入）
    p[2 * n_vec :] = ana.getFreeResParams()[0].to(
        device=device, dtype=torch.float64
    ).requires_grad_(True)

    nll = ana.getNLL(p)
    assert torch.isfinite(nll), f"{config_key}: 耦合=0 时 NLL 非有限"
    g = torch.autograd.grad(nll, p)[0]
    assert torch.isfinite(g).all(), f"{config_key}: 耦合=0 时梯度含 NaN/Inf"


@pytest.mark.parametrize("config_key", ["simple", "with_trans"])
def test_nll_large_coupling_finite(make_analysis, device, config_key):
    """耦合极大（优化器试探）时：NLL 与梯度必须有限。"""
    ana = make_analysis(config_key)
    n_vec = ana.getNVector()
    n_th = ana.getNFreeTheta()

    p = torch.zeros(2 * n_vec + n_th, dtype=torch.float64, device=device)
    p[0] = 1.0
    p[n_vec] = 0.0
    p[1:n_vec] = 1e6
    p[n_vec + 1 : 2 * n_vec] = 1e6
    p[2 * n_vec :] = ana.getFreeResParams()[0].to(
        device=device, dtype=torch.float64
    ).requires_grad_(True)

    nll = ana.getNLL(p)
    assert torch.isfinite(nll), f"{config_key}: 耦合极大时 NLL 非有限"
    g = torch.autograd.grad(nll, p)[0]
    assert torch.isfinite(g).all(), f"{config_key}: 耦合极大时梯度含 NaN/Inf"


# ============================================================
# Golden value 回归
# ============================================================

def test_golden_nll(make_analysis, device, base_params):
    """固定参数 → NLL 必须等于 golden 记录值（防数值回归）。"""
    ana = make_analysis("simple")
    nll = ana.getNLL(base_params).item()
    expected = load_golden("nll_simple.txt")
    assert nll == pytest.approx(expected, rel=1e-6), (
        f"NLL 偏离 golden: got {nll:.10f}, expected {expected:.10f}"
    )


def test_golden_gradient(make_analysis, device, base_params):
    """固定参数 → 梯度必须等于 golden 记录值。

    排除固定参考参数（v0=1+0i）：其梯度是数值噪声（应≈0），
    float/double 编译精度不同时噪声值不同，不参与对比。
    """
    ana = make_analysis("simple")
    n_vec = ana.getNVector()
    nll = ana.getNLL(base_params.requires_grad_(True))
    grad = torch.autograd.grad(nll, base_params)[0].detach().cpu().double()
    p = Path(__file__).resolve().parent / "golden" / "grad_simple.txt"
    if not p.exists():
        pytest.skip("缺少 golden 梯度文件（先跑 tests/update_golden.py）")
    expected = torch.tensor(
        [float(v) for v in p.read_text().split()], dtype=torch.float64
    )
    assert grad.shape == expected.shape, "golden 梯度维度不匹配"

    mask = torch.ones_like(grad, dtype=torch.bool)
    mask[0] = False      # v0 实部固定
    mask[n_vec] = False  # v0 虚部固定
    if not mask.any():
        pytest.skip("无自由参数可比")

    scale = grad[mask].abs().clamp_min(1e-6)
    max_rel = ((grad[mask] - expected[mask]).abs() / scale).max().item()
    assert max_rel < GRAD_RTOL, f"梯度偏离 golden，max rel = {max_rel:.2e}"
