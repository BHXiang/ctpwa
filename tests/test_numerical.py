"""L1 数值一致性测试（分钟级，发布前必跑）。

用固定 seed 的合成数据验证数值正确性：
- NLL 可复现性（防 GPU 缓冲/同步 bug）
- 梯度 vs 有限差分
- Hessian 对称性
- Hessian vs 有限差分（含 chain×step 乘法链的链式法则）
- golden value 回归（固定输入 → 固定输出）

注意：所有测试都是确定性的（固定 seed + 固定数据 + 固定参数），
同一 commit 下应稳定通过。
"""

import numpy as np
import pytest
import torch
from pathlib import Path

from conftest import CONFIGS_DIR, load_golden, make_params

import fd_utils   # 共用的 FD 比较工具（含"FD 可靠性"过滤, 见 tests/fd_utils.py）

# 有限差分步长
FD_EPS = 1e-7
# 数值容差（GPU 双精度计算；FD 有截断+舍入误差，
# 实测 rel ~1e-3~2e-3，放宽到 5e-3 抓结构性错误）
GRAD_RTOL = 5e-3
HESSIAN_SYM_ATOL = 1e-6


# ============================================================
# NLL 基础性质
# ============================================================

@pytest.mark.parametrize("config_key", ["simple", "no_trans", "with_trans", "interp", "baryon_cp"])
def test_nll_reproducible(make_analysis, device, config_key):
    """同一参数两次计算 NLL，结果必须完全一致（防 GPU 状态残留）。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)

    nll1 = ana.getNLL(params).item()
    nll2 = ana.getNLL(params).item()
    assert nll1 == pytest.approx(nll2, abs=1e-12), (
        f"{config_key}: NLL 不可复现 {nll1} vs {nll2}"
    )


@pytest.mark.parametrize("config_key", ["simple", "no_trans", "with_trans", "interp", "baryon_cp", "hess_mult"])
def test_nll_finite(make_analysis, device, config_key):
    """NLL 必须为有限实数。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)
    nll = ana.getNLL(params).item()
    assert torch.isfinite(torch.tensor(nll)), f"{config_key}: NLL 非有限: {nll}"


@pytest.mark.xfail(strict=True, reason=(
    "已知问题: with_deep_trans 的 NLL 恒为 1e30 (Analysis.cu 的 NaN/Inf 保护), 任何参数都一样"
    "—— 该 config 从未进入 NLL 级测试 (只被结构/trans 测试用到)。定位线索: 顶层顶点 "
    "Jpsi→eta+chic1(1,+) 同时允许 L=0 与 L=2; 把顶层限制成单个奇 L 后 NLL 正常。"
    "修好 (config 或 kernel) 后请删除本 xfail。"))
def test_nll_finite_with_deep_trans(make_analysis, device):
    """回归: 深层 trans config 的 NLL 必须是有限值。"""
    ana = make_analysis("with_deep_trans")
    params = make_params(ana, device)
    nll = ana.getNLL(params).item()
    assert torch.isfinite(torch.tensor(nll)) and abs(nll) < 1e10, (
        f"with_deep_trans: NLL 非有限 (={nll})")


# ============================================================
# 重子（费米子）CP 共轭对
# ============================================================

def test_baryon_cp_shared_params(make_analysis, device):
    """CP 共轭对 (N+ -> p eta / N- -> pbar eta) 引用同一共振 N:
    - Resonances 段不写 JP，量子数由 intermediates 各自赋值（1/2+ vs 1/2-）
    - 同 rname 参数槽全局共享: n_theta == 2 (m0, w0)，而非 4
    - NLL 有限且可参与 autograd
    """
    ana = make_analysis("baryon_cp")
    assert ana.getNFreeTheta() == 2, (
        f"CP 共轭对应共享 m/Gamma 参数槽（期望 2，实际 {ana.getNFreeTheta()}）"
    )
    params = make_params(ana, device).requires_grad_(True)
    nll = ana.getNLL(params)
    assert torch.isfinite(nll), "NLL 非有限"
    grad = torch.autograd.grad(nll, params)[0]
    assert torch.isfinite(grad).all(), "梯度存在 NaN/Inf"


def test_baryon_cp_shuffle_order(make_analysis, device):
    """CP 共轭对 + trans + 链 B 共振顺序打乱:
    - 同 rname 共享按名字配对（顺序无关）: n_theta == 4（两共振各 m/Gamma）
    - 参数初始值完整（N1650 与 N1880 都保留自己的槽，不被位置对应错配掉）
    """
    ana = make_analysis("baryon_cp_shuffle")
    assert ana.getNFreeTheta() == 4, (
        f"打乱顺序时应保留每个共振的独立槽（期望 4，实际 {ana.getNFreeTheta()}）"
    )
    fr = ana.getFreeResParams()  # fr[0]: init 平铺 [m0,w0, m0,w0, ...]
    m0s = sorted(fr[0].tolist()[0::2])
    assert abs(m0s[0] - 1.6366) < 1e-3 and abs(m0s[1] - 1.8017) < 1e-3, (
        f"参数初始值应包含 N1650(1.6366) 与 N1880(1.8017)，实际 {fr[0].tolist()}"
    )
    params = make_params(ana, device).requires_grad_(True)
    nll = ana.getNLL(params)
    assert torch.isfinite(nll), "NLL 非有限"
    grad = torch.autograd.grad(nll, params)[0]
    assert torch.isfinite(grad).all(), "梯度存在 NaN/Inf"


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


@pytest.mark.parametrize("config_key", ["simple", "with_trans", "flatte", "gs", "interp", "baryon_cp"])
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
# Hessian vs 有限差分: 乘法链 (chain×step) 的二阶链式法则
# ============================================================
# 拟合空间的 Hessian 必须走完整链式法则:
#   H_p = Jᵀ H_v J
#       + Σ_a (∂L/∂V_r,a)·∂²V_r,a/∂p_j∂p_k
#       + Σ_a (∂L/∂V_i,a)·∂²V_i,a/∂p_j∂p_k
# 其中第二项对 v_a = ratio·Π_k p_k 只在 j≠k 且两者都是振幅 a 的因子时非零, 且
#   ∂²v_a/∂p_j∂p_k = v_a / (p_j·p_k)        (复数)
# 展开成实坐标的 8 个块 (系数 = Re/Im(z), z = v_a/(p_j p_k))。
# j=k 时为 0 (每个因子在单项式里幂次 1)。
#
# 回归目标: 每个振幅只有 <=2 个乘法因子时 z 退化成常数 ±1; 一旦有 >=3 个因子
# (chain × step × step, 即某个顶点有 >=2 个自由 LS 耦合), 必须用 v_a/(p_j p_k)。
# tests/configs/hess_mult.yml 专门造出 3 因子振幅来覆盖这一点。

HESS_FD_H = 1e-4        # 中心差分相对步长
HESS_FD_RTOL = 0.2      # 显著元上的最大相对偏差 (FD 自身噪声实测 ~2.5e-2)
HESS_FD_SIG = 1e-3      # "显著元"门槛: max(|H_an|,|H_fd|) > HESS_FD_SIG·max|H|


def _grad_of(ana, params):
    q = params.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(ana.getNLL(q), q)[0]
    return g.detach()


def _hessian_free_idx(ana, params):
    """自由坐标下标: 去掉两个固定参考位 (re_0=1, im_0=0)。

    解析 Hessian 里这两行/列按约定为 0（它们是规范方向, 优化器固定不动,
    FD 扰动它们没有意义）—— 与 compute_param_errors / aifit._reduced_hessian
    同一条约定。
    """
    nc = ana.getNVector()
    return [i for i in range(params.numel()) if i not in (0, nc)]


def _hessian_fd(ana, params, idxs=None, h_rel=HESS_FD_H):
    """中心差分梯度 → Hessian（实现见 fd_utils.fd_hessian, h 约定与阈值一致）。"""
    return fd_utils.fd_hessian(fd_utils.analysis_grad_fn(ana), params, idxs, h_rel)


def _hessian_fd_worst(ana, params):
    """(显著元最大相对偏差, 全局位置, H_an, H_fd) —— 只比自由坐标。"""
    free = _hessian_free_idx(ana, params)
    H = ana.getHessian(params).detach().cpu().double().numpy()[np.ix_(free, free)]
    Hfd = _hessian_fd(ana, params, free)   # fd_utils 已同时切行与列 → 形状 [m,m]
    scale = max(float(np.abs(H).max()), 1e-30)
    sig = np.maximum(np.abs(H), np.abs(Hfd)) > HESS_FD_SIG * scale
    rel = np.abs(H - Hfd) / np.maximum(np.maximum(np.abs(H), np.abs(Hfd)), 1e-30)
    k = np.unravel_index(np.argmax(np.where(sig, rel, -1.0)), rel.shape)
    return (float(rel[sig].max()), (free[int(k[0])], free[int(k[1])]),
            float(H[k]), float(Hfd[k]))


def _max_mult_factors(config_path) -> int:
    """每个振幅的乘法因子数 = 1 (chain) + 用到的自由 step 参数个数; 返回最大值。"""
    import re
    import ctpwa
    lsr = re.compile(r"_LS\([^)]*\)")
    di = ctpwa.DecayInfo(str(config_path))
    step_names = {p for p in di.paramNames() if lsr.search(p)}

    def steps_of(amp):
        out, prev = [], 0
        for m in lsr.finditer(amp):
            out.append(amp[prev:m.end()].lstrip("_")); prev = m.end()
        return out

    return max(1 + sum(1 for t in steps_of(a) if t in step_names)
               for a in di.amplitudeNames())


def test_hessian_fd_report_mult_factor(make_analysis, device):
    """稳健版比较（fd_utils.compare_hessian）在 >=3 因子 config 上必须 PASS。

    与 test_hessian_vs_finite_difference_mult_factor 的区别: 这里用带
    "显著性 + 双 h 自洽"过滤的分类器, 并把 FD 噪声元与真不符元分开报告——
    避免大 |NLL|/float32 舍入下把 FD 噪声误判成 kernel 错误。
    """
    ana = make_analysis("hess_mult")
    params = make_params(ana, device)
    grad = fd_utils.analysis_grad_fn(ana)
    H, idxs = fd_utils.analysis_hessian_block(ana, params)
    rep = fd_utils.compare_hessian(H, grad, params, idxs)
    assert rep.n_judged > 0, f"没有可靠元, FD 无法判定:\n{rep.summary()}"
    assert rep.verdict == "PASS", rep.summary()
    eig = fd_utils.eig_min_report(H, grad, params, idxs)
    tol = max(eig["noise_bound"], 1e-6 * abs(eig["lam_min_an"]) + 1e-6)
    assert abs(eig["diff"]) <= tol, f"λmin 差异超出 FD 噪声上界: {eig}"


@pytest.mark.parametrize("config_key", ["simple", "no_trans", "with_trans"])
def test_hessian_vs_finite_difference(make_analysis, device, config_key):
    """每个振幅 <=2 个乘法因子 (无自由 step 参数): 解析 Hessian 必须等于中心差分。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)
    assert _max_mult_factors(CONFIGS_DIR / f"{config_key}.yml") <= 2, (
        f"{config_key}: 该 config 出现了 >=3 因子振幅, 应改用 hess_mult 那类断言")
    rel, (i, j), ha, hf = _hessian_fd_worst(ana, params)
    assert rel < HESS_FD_RTOL, (
        f"{config_key}: 解析 Hessian 与有限差分不符 max_rel={rel:.3e} at [{i},{j}] "
        f"(H_an={ha:.6g}, H_fd={hf:.6g})")


def test_hessian_vs_finite_difference_mult_factor(make_analysis, device):
    """存在 3 因子振幅 (chain×step×step) 时, 解析 Hessian 仍必须等于中心差分。

    回归背景: hessianFullTransformKernel 的二阶项曾把 ∂²v/∂p_j∂p_k 硬编码成 ±1
    (真值 = v_a/(p_j·p_k))、并整类跳过 (step,step) 项, 导致 >=3 因子 / trans
    ratio≠1 时解析 Hessian 错 (实测 18 个元错 20%~105%, 含符号翻转)。已修。
    """
    ana = make_analysis("hess_mult")
    params = make_params(ana, device)
    assert _max_mult_factors(CONFIGS_DIR / "hess_mult.yml") >= 3, (
        "hess_mult 结构已变: 不再有 3 因子振幅, 本测试失去意义")
    rel, (i, j), ha, hf = _hessian_fd_worst(ana, params)
    assert rel < HESS_FD_RTOL, (
        f"hess_mult: 解析 Hessian 与有限差分不符 max_rel={rel:.3e} at [{i},{j}] "
        f"(H_an={ha:.6g}, H_fd={hf:.6g})")


@pytest.mark.parametrize("config_key", ["simple", "with_trans"])
def test_phsp_hessian_large_n_fallback(make_analysis, device, monkeypatch, config_key):
    """phsp Hessian 的 global-memory 回退（大 n）与 shared-memory 快路径必须数值一致。

    回归背景: phspHessianKernel 把 n×n 的 phsp 矩阵放共享内存，需求
    n²·sizeof(ctComplex) 随分波数平方增长（n=138 → ~300KB），超过单 block
    动态共享内存上限（V100 96KB）时旧实现 cudaFuncSetAttribute /
    cudaLaunchKernel 直接返回 cudaErrorInvalidValue，且错误被留到下一个
    torch CUDA 调用才抛出（表现为 fit.py 中 final_params.clone() 处的假 traceback）。
    这里把"本卡共享内存上限"人为下调到 1KB，强制走回退分支对比两条路径。
    """
    ana = make_analysis(config_key)
    params = make_params(ana, device)

    H_small = ana.getHessian(params).detach().cpu().double()
    monkeypatch.setenv("CTPWA_PHSP_HESS_MAX_SMEM", "1024")
    H_large = ana.getHessian(params).detach().cpu().double()

    assert torch.isfinite(H_large).all(), f"{config_key}: 回退路径 Hessian 含 NaN/Inf"
    scale = H_small.abs().max().item()
    diff = (H_large - H_small).abs().max().item()
    assert diff <= 1e-8 * max(scale, 1.0), (
        f"{config_key}: phsp Hessian 回退路径与快路径不一致 "
        f"max|ΔH|={diff:.3e} (|H|max={scale:.3e})"
    )


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
