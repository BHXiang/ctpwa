"""拟合后 Hessian 的正定性诊断（纯 torch，不依赖 C++ 后端）。

为什么需要这个模块
------------------
``analysis.getHessian()`` 返回的是**无约束** NLL 的精确二阶导。拟合参数常常带
盒约束（共振态 ``free_range``、耦合 ``v_max``），最优点可能落在边界上。此时：

* 沿**可行方向**（未贴边，或贴边但梯度指向内部）二阶条件仍可能是正的；
* 沿**朝外的方向** Hessian 完全可能为负——但那个方向根本不允许走。

所以"Hessian 不正定"必须区分两件事：

1. ``free_pd``：把所有非固定参数都算进去（旧判据，会在贴边参数上假报警）；
2. ``reduced_pd``：先剔除"贴边且梯度朝外"的活跃参数，再判正定（KKT 二阶条件，
   与 ``example/fit.py`` 的 polish / compute_param_errors 同一条规则）。

另外 ``getHessian`` 的布局约定：固定参考耦合 ``Re_0=1, Im_0=0`` 对应两行/列
恒为 0（规范方向）。调用方应把它们放进 ``fixed_mask``，否则特征值里必然出现
两个零——``eigvalsh(...).min() < 0`` 式的判据会把数值噪声当成"不正定"。

用法::

    from ctpwa.diagnostics import analyze_hessian
    rep = analyze_hessian(H, params, grad, lower, upper, fixed_mask)
    print(rep.summary(names))
    if not rep.reduced_pd: ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class HessianSubspaceReport:
    """Hessian 在固定/活跃/可行子空间上的特征值报告。"""

    fixed_mask: torch.Tensor
    active_mask: torch.Tensor
    reduced_mask: torch.Tensor
    active_idx: torch.Tensor
    reduced_idx: torch.Tensor
    ev_free: torch.Tensor
    ev_reduced: torch.Tensor
    free_pd: bool
    reduced_pd: bool
    free_min_eigenvalue: float
    free_max_eigenvalue: float
    reduced_min_eigenvalue: float
    reduced_max_eigenvalue: float
    kkt_max: float
    kkt_ok: bool
    active_margin: torch.Tensor
    note: str = ""

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def n_reduced(self) -> int:
        return int(self.reduced_mask.sum().item())

    def summary(self, names: Optional[Sequence[str]] = None,
                max_active: int = 8) -> str:
        """一行式摘要 + 贴边参数列表。"""
        bits = [
            f"reduced_pd={self.reduced_pd}",
            f"free_pd={self.free_pd}",
            f"dim={self.n_reduced}/{int(self.reduced_mask.numel())}",
        ]
        if self.reduced_mask.any():
            ratio = (self.reduced_min_eigenvalue /
                     max(abs(self.reduced_max_eigenvalue), 1e-300))
            bits.append(f"λmin={self.reduced_min_eigenvalue:.3e}")
            bits.append(f"λmin/λmax={ratio:.2e}")
        bits.append(f"|pg|={self.kkt_max:.2e}")
        if self.n_active:
            act = []
            for k, i in enumerate(self.active_idx.tolist()):
                nm = (names[i] if names is not None and i < len(names) else f"#{i}")
                act.append(f"{nm}(margin={float(self.active_margin[k]):.1e})")
            if len(act) > max_active:
                act = act[:max_active] + [f"...(+{len(act) - max_active})"]
            bits.append("active=" + ", ".join(act))
        if self.free_pd and not self.reduced_pd:
            bits.append("note=自由空间正定但可行子空间不定")
        if self.note:
            bits.append(self.note)
        return "[hessian] " + "  ".join(bits)


def _eig_psd(h: torch.Tensor, tau: float) -> tuple[torch.Tensor, bool, float, float]:
    """对称化后求特征值；用相对门槛 tau·max|λ| 判正定（空矩阵视为正定）。"""
    if h.numel() == 0:
        z = torch.zeros(0, dtype=torch.float64, device=h.device)
        return z, True, float("nan"), float("nan")
    if not bool(torch.isfinite(h).all()):
        z = torch.zeros(0, dtype=torch.float64, device=h.device)
        return z, False, float("nan"), float("nan")
    sym = 0.5 * (h + h.t())
    ev = torch.linalg.eigvalsh(sym.double())
    lmax = float(ev[-1].abs().item())
    lmin = float(ev[0].item())
    pd = bool(lmin > tau * max(lmax, 1e-300))
    return ev, pd, lmin, float(ev[-1].item())


def map_errors_to_params(
    sigma: torch.Tensor,
    reduced_idx: torch.Tensor,
    n_coupling: int,
    n_theta: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """把可行子空间的标准差映射回 grouped 参数布局。

    ``getHessian`` / 参数向量的布局是 ``[Re_0..Re_{n-1}, Im_0..Im_{n-1}, θ]``
    （**不是** ``[Re_0, Im_0, Re_1, Im_1, ...]``）。旧 fit.py 在求误差时按交错
    布局取 ``sd[2i], sd[2i+1]``，会把 Re/Im 误差错位。本函数按真实布局映射：
    下标 < n_coupling → Re，< 2*n_coupling → Im，其余 → θ。

    固定/活跃参数在 ``reduced_idx`` 中缺席，返回 NaN（而不是 0）。
    ``Re_0``/``Im_0`` 是固定参考，保持旧输出的 0.0。
    """
    nc, nt = int(n_coupling), int(n_theta)
    nan = float("nan")
    cre = torch.full((nc,), nan, dtype=dtype, device=sigma.device)
    cim = torch.full((nc,), nan, dtype=dtype, device=sigma.device)
    res = (torch.full((nt,), nan, dtype=dtype, device=sigma.device)
           if nt > 0 else None)
    if sigma.numel() != reduced_idx.numel():
        raise ValueError("sigma 与 reduced_idx 长度不一致")
    for k, full_idx in enumerate(reduced_idx.tolist()):
        val = float(sigma[k].item())
        if full_idx < nc:
            cre[full_idx] = val
        elif full_idx < 2 * nc:
            cim[full_idx - nc] = val
        elif res is not None:
            res[full_idx - 2 * nc] = val
    if nc > 0:
        cre[0] = 0.0
        cim[0] = 0.0
    return cre, cim, res


def analyze_hessian(
    hessian: torch.Tensor,
    params: torch.Tensor,
    grad: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    fixed_mask: Optional[torch.Tensor] = None,
    *,
    tau: float = 1e-8,
    bound_margin: float = 1e-8,
    kkt_tol: float = 1e-2,
) -> HessianSubspaceReport:
    """把 Hessian 拆成固定 / 贴边活跃 / 可行三部分并判正定。

    Parameters
    ----------
    hessian : [n, n] float
        ``analysis.getHessian(params)`` 的返回值（float32 会转 double）。
    params, grad : [n]
        当前点与 NLL 梯度（``grad = ∂NLL/∂params``）。
    lower, upper : [n]
        盒约束；无约束参数可传 ``-inf/+inf``（该维永远不会被判为贴边）。
    fixed_mask : [n] bool, optional
        固定参数（``True`` 表示不参与判正定）。典型是 ``Re_0``、``Im_0`` 两个
        参考耦合；它们对应 getHessian 的恒零行/列。
    tau : float
        正定相对门槛：``λmin > tau·max|λ|``。默认 1e-8 与 polish 一致。
    bound_margin : float
        "贴边"的相对阈值：距边界 ≤ bound_margin·(upper-lower)。
    kkt_tol : float
        只用于 ``kkt_ok`` 报告，不参与正定判定。

    Returns
    -------
    HessianSubspaceReport
        关键字段：``reduced_pd``（应该看的那个）、``free_pd``（旧判据）、
        ``ev_reduced``、``active_idx``、``kkt_max``。
    """
    if hessian.dim() != 2 or hessian.size(0) != hessian.size(1):
        raise ValueError(f"hessian 必须是方阵，实际 {tuple(hessian.shape)}")
    n = hessian.size(0)
    if params.numel() != n or grad.numel() != n:
        raise ValueError(
            f"params/grad 长度 ({params.numel()}/{grad.numel()}) 与 hessian ({n}) 不符")
    if lower.numel() != n or upper.numel() != n:
        raise ValueError("lower/upper 长度必须与 hessian 一致")

    dev = params.device
    p = params.detach().double().to(dev)
    g = grad.detach().double().to(dev)
    lo = lower.detach().double().to(dev)
    hi = upper.detach().double().to(dev)
    H = hessian.detach().double().to(dev)

    if fixed_mask is None:
        fixed = torch.zeros(n, dtype=torch.bool, device=dev)
    else:
        fixed = fixed_mask.detach().to(dev).bool()
        if fixed.numel() != n:
            raise ValueError("fixed_mask 长度必须与 hessian 一致")

    w = (hi - lo).clamp(min=1e-30)
    # 只有真正有界（hi>lo）的非固定参数才可能是活跃约束
    bounded = (~fixed) & (hi > lo) & torch.isfinite(lo) & torch.isfinite(hi)
    at_lo = bounded & ((p - lo) <= bound_margin * w)
    at_hi = bounded & ((hi - p) <= bound_margin * w)
    active = (at_lo & (g > 0)) | (at_hi & (g < 0))
    active = active & (~fixed)
    reduced = (~fixed) & (~active)

    free_idx = torch.nonzero(~fixed, as_tuple=False).flatten()
    red_idx = torch.nonzero(reduced, as_tuple=False).flatten()
    act_idx = torch.nonzero(active, as_tuple=False).flatten()

    H_free = H[free_idx][:, free_idx] if free_idx.numel() else H.new_zeros((0, 0))
    H_red = H[red_idx][:, red_idx] if red_idx.numel() else H.new_zeros((0, 0))
    ev_free, free_pd, free_min, free_max = _eig_psd(H_free, tau)
    ev_red, red_pd, red_min, red_max = _eig_psd(H_red, tau)

    pg = p - torch.clamp(p - g, lo, hi)
    if red_idx.numel():
        kkt_max = float(pg[red_idx].abs().max().item())
    else:
        kkt_max = 0.0

    margin = torch.minimum(p - lo, hi - p)
    active_margin = margin[act_idx] if act_idx.numel() else torch.zeros(0, device=dev)

    note = ""
    if not bool(torch.isfinite(H).all()):
        note = "Hessian 含非有限值"
    elif free_idx.numel() and not bool(torch.isfinite(g).all()):
        note = "梯度含非有限值"

    return HessianSubspaceReport(
        fixed_mask=fixed,
        active_mask=active,
        reduced_mask=reduced,
        active_idx=act_idx,
        reduced_idx=red_idx,
        ev_free=ev_free,
        ev_reduced=ev_red,
        free_pd=free_pd,
        reduced_pd=red_pd,
        free_min_eigenvalue=free_min,
        free_max_eigenvalue=free_max,
        reduced_min_eigenvalue=red_min,
        reduced_max_eigenvalue=red_max,
        kkt_max=kkt_max,
        kkt_ok=bool(kkt_max <= kkt_tol),
        active_margin=active_margin,
        note=note,
    )
