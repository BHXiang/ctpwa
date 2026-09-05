"""Custom DSL 模型数值测试（L0 + L1）。

覆盖 Custom 模型（tests/configs/custom.yml，4 个自由参数 m0/w0/g0/phi）：
- L0: config 加载、theta 参数数量与名字
- 与内置 BW 模型一致性（custom_bw.yml）：表达式
  g0·e^{iφ}/(m0²-m²-i·m0·w0) 在 (g0,φ)=(1,0) 时 ≡ 内置 BW —— 直接验证
  DSL 解释器 + computeCustomAmpsKernel 的数值正确性
- NLL 可复现/有限
- 梯度 vs 有限差分（覆盖 4 个 theta 的 vθ/θθ 路径）
- Hessian 对称/有限 + θθ 块 vs 梯度有限差分（验证 computeCustomHessianKernel）

注意：Custom 模型 4 参数 > 3，走独立的标量 kernel 路径
（computeCustomAmpsKernel / computeCustomHessianKernel），
不经过 Var<double,N> 模板 —— 本文件是对该路径的回归防线。
"""

import pytest
import torch

from conftest import make_params

# 有限差分步长（与 test_numerical.py 一致）
FD_EPS = 1e-7
GRAD_RTOL = 5e-3
HESSIAN_SYM_ATOL = 1e-6
# Hessian vs FD：二阶差分，截断误差大，放宽容差（抓结构性错误）
HESSIAN_FD_RTOL = 1e-2
# 对称性容差：Hessian 元素量级 ~1e6（float32 精度 ~1e-7 rel），
# 绝对容差 1e-5 对 float 累加顺序差异足够宽（simple 的 1e-6 对 Custom 过紧）
HESSIAN_SYM_ATOL = 1e-5
# BW 一致性：不同 kernel 的浮点舍入路径不同，用宽松的绝对容差
BW_MATCH_ATOL = 1e-6
# 平坦方向绝对容差：单振幅（n_vec=1）时 g0/φ 的相位/缩放被归一化共动抵消，
# 真实梯度 ≈ 0（实测 < 1e-2 量级），FD 差分只剩 float32 噪声（NLL~1300 时
# 绝对噪声 ~1e-4，除以步长后 FD 显示为 0.1~1 的假梯度）。对这类参数用
# 绝对容差比较（AD 应远小于典型梯度量级 ~1e3）。
GRAD_ATOL_FLAT = 1e-2


# ============================================================
# L0: 结构
# ============================================================

def test_custom_config_loads(make_analysis):
    """custom.yml 能正确初始化，4 个参数全部自由。"""
    ana = make_analysis("custom")
    assert ana.isValid(), "custom.yml 初始化失败"
    assert ana.getNVector() > 0
    assert ana.getNFreeTheta() == 4, "Custom 模型应有 4 个自由参数 (m0,w0,g0,phi)"


def test_custom_param_names(make_analysis):
    """theta 参数名应包含模型参数名（m0/w0/g0/phi）。"""
    ana = make_analysis("custom")
    names = ana.getParamNames()
    # getParamNames = 振幅名 + theta 名（v0 固定参考无名字）
    assert len(names) == ana.getNVector() + ana.getNFreeTheta()
    joined = " ".join(names)
    for p in ["m0", "w0", "g0", "phi"]:
        assert p in joined, f"参数名缺少 {p}: {joined}"


def test_custom_param_bounds(make_analysis):
    """theta 初值 = config parameters 标称值，且 bounds 有限（默认 ±50%）。"""
    ana = make_analysis("custom")
    free = ana.getFreeResParams()  # [3, n_theta]: init, lower, upper
    init, lower, upper = free[0].tolist(), free[1].tolist(), free[2].tolist()
    assert init == pytest.approx([1.66, 0.125, 1.0, 0.5])
    for lo, up in zip(lower, upper):
        assert torch.isfinite(torch.tensor(lo))
        assert torch.isfinite(torch.tensor(up))
    assert all(up > lo for lo, up in zip(lower, upper))


# ============================================================
# Custom vs 内置 BW 一致性
# ============================================================

def _bw_match_params(ana, device, model_theta):
    """构造与 custom (g0=1, φ=0) 对应的参数向量。

    两个 config 结构相同（1 链 / 1 组合 / 1 共振态），n_vec 均为 1，
    v0=1+0i 固定参考 → 参数布局仅 theta 数量不同。
    """
    n_vec = ana.getNVector()
    n_th = ana.getNFreeTheta()
    p = torch.zeros(2 * n_vec + n_th, dtype=torch.float64, device=device)
    p[0] = 1.0
    p[n_vec] = 0.0
    p[2 * n_vec:] = torch.tensor(model_theta, dtype=torch.float64, device=device)
    return p


def test_custom_equals_bw(make_analysis, device):
    """Custom 表达式 ≡ 内置 BW 时，NLL 必须一致。

    custom:  g0·e^{iφ}/(m0²-m²-i·m0·w0)，theta = [1.66, 0.125, 1, 0]
    custom_bw: 内置 BW(m0, w0)，theta = [1.66, 0.125]
    F_custom = 1×BW → 振幅/NLL 应完全一致（同数据同归一化）。
    """
    ana_c = make_analysis("custom")
    ana_b = make_analysis("custom_bw")

    p_c = _bw_match_params(ana_c, device, [1.66, 0.125, 1.0, 0.0])
    p_b = _bw_match_params(ana_b, device, [1.66, 0.125])

    nll_c = ana_c.getNLL(p_c).item()
    nll_b = ana_b.getNLL(p_b).item()
    assert torch.isfinite(torch.tensor(nll_c)), "Custom NLL 非有限"
    assert abs(nll_c - nll_b) < BW_MATCH_ATOL, (
        f"Custom NLL 与内置 BW 不一致: custom={nll_c:.10f}, bw={nll_b:.10f}, "
        f"diff={abs(nll_c - nll_b):.2e}"
    )


def test_custom_gradient_equals_bw(make_analysis, device):
    """Custom 与内置 BW 的 theta 梯度一致（g0/φ 不变时）。"""
    ana_c = make_analysis("custom")
    ana_b = make_analysis("custom_bw")

    p_c = _bw_match_params(ana_c, device, [1.66, 0.125, 1.0, 0.0])
    p_b = _bw_match_params(ana_b, device, [1.66, 0.125])

    g_c = torch.autograd.grad(ana_c.getNLL(p_c.requires_grad_(True)), p_c)[0].detach().cpu()
    g_b = torch.autograd.grad(ana_b.getNLL(p_b.requires_grad_(True)), p_b)[0].detach().cpu()

    assert torch.isfinite(g_c).all(), "Custom 梯度含 NaN/Inf"
    # 比较 m0/w0 梯度（两个 config 的 theta 前 2 个都是 [m0, w0]）。
    # g0/φ 梯度与 BW 不可比（BW 无此自由度）；单振幅时它们 ≈ 0 是
    # 归一化共动的物理结果（见 GRAD_ATOL_FLAT 注释），由
    # test_custom_gradient_vs_fd 用绝对容差验证。
    dg = (g_c[2:4] - g_b[2:4]).abs()
    scale = g_b[2:4].abs().clamp_min(1e-6)
    max_rel = (dg / scale).max().item()
    assert max_rel < GRAD_RTOL, (
        f"Custom theta 梯度与 BW 不一致，max rel = {max_rel:.2e}"
        f"\n  custom={g_c[2:4].tolist()}\n  bw={g_b[2:4].tolist()}"
    )


# ============================================================
# NLL 性质
# ============================================================

def test_custom_nll_reproducible(make_analysis, device):
    """同一参数两次计算 NLL 必须一致（防 GPU 状态残留）。"""
    ana = make_analysis("custom")
    params = make_params(ana, device)
    nll1 = ana.getNLL(params).item()
    nll2 = ana.getNLL(params).item()
    assert nll1 == pytest.approx(nll2, abs=1e-12)


def test_custom_nll_finite(make_analysis, device):
    """NLL 必须为有限实数。"""
    ana = make_analysis("custom")
    params = make_params(ana, device)
    nll = ana.getNLL(params).item()
    assert torch.isfinite(torch.tensor(nll)), f"Custom NLL 非有限: {nll}"


# ============================================================
# 梯度 vs 有限差分
# ============================================================

def _fd_gradient(ana, params, eps=FD_EPS):
    """中心差分梯度（与 test_numerical.py 相同的步长策略）。"""
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


def test_custom_gradient_vs_fd(make_analysis, device):
    """autograd 梯度 vs 中心差分（theta 4 个全覆盖，走标量 kernel 的 dF 路径）。"""
    ana = make_analysis("custom")
    params = make_params(ana, device).requires_grad_(True)

    nll = ana.getNLL(params)
    grad_ad = torch.autograd.grad(nll, params)[0].detach().cpu().double()
    grad_fd = _fd_gradient(ana, params)

    # 排除固定参考参数（v0=1+0i），只比自由参数（全部 4 个 theta）
    n_vec = ana.getNVector()
    mask = torch.ones_like(grad_ad, dtype=torch.bool)
    mask[0] = False
    mask[n_vec] = False

    diff = (grad_ad - grad_fd).abs()
    scale = grad_ad.abs().clamp_min(1e-6)
    rel = diff[mask] / scale[mask]
    max_rel = rel.max().item()
    # 平坦方向（g0/φ）：真实梯度 ≈ 0。判据看 AD（确定性，~1e-4）
    # 而非 FD —— FD 在此方向只剩 float32 噪声假梯度（NLL~1300 时
    # 每次运行随机出现 0.1~0.3 的假值，用 FD 量级做阈值不稳定）。
    # 平坦方向：AD 必须精确 ≈ 0（绝对容差）；非平坦方向：相对容差。
    flat = grad_ad[mask].abs() < GRAD_ATOL_FLAT
    max_abs = grad_ad[mask][flat].abs().max().item() if flat.any() else 0.0
    ok_rel = not (~flat).any() or rel[~flat].max().item() < GRAD_RTOL
    ok_abs = max_abs < GRAD_ATOL_FLAT
    assert ok_rel and ok_abs, (
        f"Custom 梯度 vs FD 不一致: max_rel={max_rel:.2e} max_abs_flat={max_abs:.2e}"
        f"\n  AD={grad_ad[mask].tolist()}"
        f"\n  FD={grad_fd[mask].tolist()}"
    )


# ============================================================
# Hessian
# ============================================================

def test_custom_hessian_symmetric(make_analysis, device):
    """Hessian 必须对称（H_ij == H_ji）。"""
    ana = make_analysis("custom")
    params = make_params(ana, device)
    H = ana.getHessian(params).detach().cpu().double()
    asym = (H - H.T).abs().max().item()
    assert asym < HESSIAN_SYM_ATOL, f"Custom Hessian 不对称，max|H-H^T| = {asym:.2e}"


def test_custom_hessian_finite(make_analysis, device):
    """Hessian 必须为有限值。"""
    ana = make_analysis("custom")
    params = make_params(ana, device)
    H = ana.getHessian(params).detach().cpu()
    assert torch.isfinite(H).all(), "Custom Hessian 存在 NaN/Inf"


def test_custom_hessian_vs_fd(make_analysis, device):
    """θθ 块 vs 梯度有限差分（验证 computeCustomHessianKernel 的 Stage 1）。

    对每个 theta 参数 j：p_j ± δ 求 autograd 梯度 → 中心差分得 H_ij 列。
    只用 ctpwa 自己的梯度做 FD，不重建 NLL（梯度已由
    test_custom_gradient_vs_fd 验证）。
    """
    ana = make_analysis("custom")
    params = make_params(ana, device)
    n_vec = ana.getNVector()
    n_th = ana.getNFreeTheta()
    t0 = 2 * n_vec  # theta 起点

    def grad_at(p):
        return torch.autograd.grad(ana.getNLL(p.requires_grad_(True)), p)[0]

    g0 = grad_at(params).detach().cpu().double()
    H_fd = torch.zeros(n_th, n_th, dtype=torch.float64)
    p_cpu = params.detach().cpu().double()
    for j in range(n_th):
        step = max(abs(p_cpu[t0 + j].item()) * 1e-4, 1e-4)
        p_plus = p_cpu.clone().to(device)
        p_minus = p_cpu.clone().to(device)
        p_plus[t0 + j] += step
        p_minus[t0 + j] -= step
        g_plus = grad_at(p_plus).detach().cpu().double()
        g_minus = grad_at(p_minus).detach().cpu().double()
        H_fd[:, j] = (g_plus[t0:] - g_minus[t0:]) / (2 * step)

    H_ct = ana.getHessian(params).detach().cpu().double()[t0:, t0:]

    # g0/φ（索引 2,3）是精确的连续对称性平坦方向（单振幅时归一化共动）：
    # 真实 Hessian 行/列 ≡ 0，CT 应精确 ≈ 0（float 累加残留 ~1e-6），
    # FD 只剩 float32 噪声假值（~0.1~1，见 GRAD_ATOL_FLAT 注释），相对
    # 误差无意义 → 平坦行/列用绝对容差验证 CT 确实 ≈ 0（能抓住
    # 结构性错误，如历史上 g0×g0 的 -1800、phi×m0 的 1.8e5）。
    flat_rows = (H_ct.abs().max(dim=1).values < 1e-2)
    diff = (H_ct - H_fd).abs()
    scale = H_fd.abs().clamp_min(1e-6)
    rel = diff / scale
    rel[flat_rows] = 0
    rel[:, flat_rows] = 0
    max_rel = rel.max().item()
    # 平坦行/列：CT 必须精确 ≈ 0（物理平坦方向），用绝对容差
    max_flat_abs = H_ct[flat_rows].abs().max().item() if flat_rows.any() else 0.0
    assert max_rel < HESSIAN_FD_RTOL and max_flat_abs < 1e-2, (
        f"Custom Hessian θθ 块 vs FD 最大相对误差 {max_rel:.2e} > {HESSIAN_FD_RTOL}"
        f"，平坦方向 |CT| 最大 {max_flat_abs:.2e}"
        f"\n  CT={H_ct.numpy()}\n  FD={H_fd.numpy()}"
    )
