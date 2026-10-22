"""全同粒子对称化数值测试（ident2: 2×π⁰ 跨子衰变；ident3: 3×π⁰ 含同子衰变稳定子；
ident4: K_S K_S ω —— 兄弟对与跨子衰变两种身份同时存在，复刻 J/psi → omega K_S K_S）。

验证内容：
1. coset 拓扑数（构建输出 + getNSigma 不可直接查 → 用对称性间接验证）
2. 振幅对称性：交换全同粒子标签后 NLL 不变（模型必须对称）
3. 模型投影对称：M(Ks1 omega) 与 M(Ks2 omega) 的 hfit 逐 bin 相等
4. 梯度 vs 有限差分（FD vs AD，含 σ 求和路径）
5. Hessian 对称 + 有限
"""
import numpy as np
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


@pytest.mark.parametrize("config_key", ["ident2", "ident3", "ident4"])
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


def _tree_weights_and_momenta(path):
    """从 writeResult(is_saved_weight=1) 的 saved_weight TTree 取逐事件量。"""
    import uproot

    with uproot.open(str(path)) as f:
        t = f["saved_weight"]
        w = t["totalweight"].array(library="np")
        mom = {}
        for name in ("Ks1", "Ks2", "omega"):
            px = t[f"{name}_px"].array(library="np")
            py = t[f"{name}_py"].array(library="np")
            pz = t[f"{name}_pz"].array(library="np")
            e = t[f"{name}_E"].array(library="np")
            mom[name] = np.stack([px, py, pz, e], axis=1)
    return w, mom


def _pair_mass(a, b):
    e = a[:, 3] + b[:, 3]
    p2 = ((a[:, :3] + b[:, :3]) ** 2).sum(axis=1)
    return np.sqrt(np.maximum(e * e - p2, 0.0))


def test_model_hist_matches_weights(make_analysis, device, tmp_path):
    """hfit 必须与"权重 × 末态四动量"独立重算的投影一致（防填充错位）。

    hfit 走的是 phsp 权重 → 直方图 的链路；若权重与动量错位（偏移/分批指针
    错位），投影会和权重对不上。这里用 saved_weight 树里的 totalweight 与
    末态四动量在 Python 侧重算，逐 bin 对比 hfit。
    """
    import uproot

    ana = make_analysis("ident4")
    params = make_params(ana, device)
    out = tmp_path / "ident4_fill.root"
    ana.writeResult(params, str(out), 1)

    w, mom = _tree_weights_and_momenta(out)
    assert w.sum() > 0 and np.all(w >= 0)
    edges = np.linspace(1.10, 2.70, 41)
    with uproot.open(str(out)) as rf:
        for tag, pair in (("mass1_Ks1_omega", ("Ks1", "omega")),
                          ("mass2_Ks2_omega", ("Ks2", "omega"))):
            ref = rf[f"{tag}/hfit"].values().astype(float)
            mine, _ = np.histogram(_pair_mass(mom[pair[0]], mom[pair[1]]),
                                   bins=edges, weights=w)
            rel = np.abs(mine - ref).max() / max(ref.max(), 1e-12)
            assert rel < 1e-5, (
                f"{tag}: 权重重算的投影与 hfit 不符（最大相对差 {rel:.2e}）——"
                "权重与末态动量错位"
            )


def test_model_intensity_symmetric(make_analysis, device, tmp_path):
    """逐事件模型强度必须对全同粒子标签交换不变：w(x) == w(σx)（精确判定）。

    为什么不用 M(Ks1 ω) 与 M(Ks2 ω) 的 hfit 差来判定：模型投影是对有限 phsp
    样本求平均，两条投影各取每事件的一个配对，其差在期望上为 0、涨落量级
    ~1/sqrt(N_eff)（ident4 的 10k 样本约 18%），phsp 样本自身的标签不对称还会
    叠加进来 —— 用直方图差判对称性会把 MC 伪影误判成振幅 bug（实测就踩过）。

    这里用同一批事件 + 交换标签后的 phsp 建两个分析，逐事件比较 totalweight：
    任何破坏对称性的系统偏差都会精确暴露（修复版差异 ~1e-15，退化版 ~0.2）。
    """
    ana_a = make_analysis("ident4")
    ana_b = make_analysis("ident4_phsp_swap")
    params = make_params(ana_a, device)

    out_a = tmp_path / "ident4_wA.root"
    out_b = tmp_path / "ident4_wB.root"
    ana_a.writeResult(params, str(out_a), 1)
    ana_b.writeResult(params, str(out_b), 1)

    w_a, _ = _tree_weights_and_momenta(out_a)
    w_b, _ = _tree_weights_and_momenta(out_b)
    assert len(w_a) == len(w_b) and w_a.sum() > 0
    rel = np.abs(w_a - w_b).sum() / w_a.sum()
    assert rel < 1e-9, (
        f"模型强度不对称: Σ|w(x)-w(σx)|/Σw = {rel:.3e} —— "
        "全同粒子 σ 求和可能退化成单拓扑（per-σ 动量缓冲别名）"
    )


@pytest.mark.parametrize("config_key", ["ident2", "ident3", "ident4"])
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
    # 按梯度最大分量归一（不能用逐参数相对误差）：ident4 里有梯度恰好接近
    # 抵消的参数（|g|~0.15，而其它分量 ~4e3），FD 的绝对噪声 ~1e-3 会让该
    # 参数的相对误差天然到 2%，与梯度对错无关。归一化后对"整块梯度算错"
    # 依然灵敏（0.5% 的相对偏差就会超阈值）。
    scale = grad_ad.abs().max().clamp_min(1e-6)
    max_rel = (diff[mask].max() / scale).item()
    assert max_rel < GRAD_RTOL, (
        f"{config_key}: 梯度 vs FD 误差（按 max|g| 归一）{max_rel:.2e} > {GRAD_RTOL}"
    )


@pytest.mark.parametrize("config_key", ["ident2", "ident3", "ident4"])
def test_hessian_symmetric_finite(make_analysis, device, config_key):
    """Hessian 对称 + 有限（覆盖 σ 求和的 Hessian 路径）。"""
    ana = make_analysis(config_key)
    params = make_params(ana, device)
    H = ana.getHessian(params).detach().cpu().double()
    asym = (H - H.T).abs().max().item()
    assert asym < HESSIAN_SYM_ATOL, f"{config_key}: Hessian 不对称 max={asym:.2e}"
    assert torch.isfinite(H).all(), f"{config_key}: Hessian 含 NaN/Inf"
