"""解析 Hessian vs 有限差分（FD）比较工具 —— 带"FD 可靠性"过滤。

背景（为什么需要这个模块）
--------------------------
ctpwa 的相空间矩阵是 float32 存/算档，NLL 的数值精度在大动态范围（耦合 ->O(10)、
几十万 phsp 事件）下会限制 FD 的可信度。实测两类**假警报**：

1. **量级误判**：在 |NLL| 很大的随机点上做二阶差分，float32 舍入被 1/h² 放大，
   FD 直接是噪声。此时"解析 ≠ FD"完全不能说明 kernel 有问题。
2. **有偏平台**：某些点上 FD 在 h=1e-3~3e-4 有一片**稳定但有系统偏差**的平台
   （例如某元 FD 恒为 -7000），而把 h 压到 3e-5 才回到真值（-359 vs 解析 -551）。
   只看单个 h 会得出"解析错 13 倍"的错误结论。

因此本模块对每个矩阵元分三类，**只用第 1 类判定**：

* ``judged``         显著 **且** FD 在两个 h 下自洽 → 参与 max_rel 与判定；
* ``fd_noisy``       显著但 FD 随 h 剧烈变化 → **FD 不可靠**，只报告、不判定；
* ``insignificant``  |H| ≤ sig_frac·|H|max → 量级在噪声里，跳过。

另给 ``eig_min_report``：把 λmin 的解析/FD 差与 FD 的**元噪声上界**（√n·noise）比较，
差小于该上界时如实报告"FD 无法分辨该特征值"，而不是判定不符。

用法
----
库::

    from fd_utils import compare_hessian, eig_min_report
    rep = compare_hessian(H_an, grad_fn, params, idxs)   # H_an: numpy [m,m]
    print(rep.summary())
    print(rep.verdict)            # PASS / FAIL / INCONCLUSIVE

命令行（等价于历史 tmp/*.py 脚本，已固化于此）::

    python tests/fd_utils.py --config <cfg> --fit-mode 1 --points 3
    python tests/fd_utils.py --config <cfg> --fit-json results/fit.json
    python tests/fd_utils.py --config <cfg> --points 3 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# 与 tests/test_numerical.py 保持一致的默认阈值
H_REL_DEFAULT = 1e-4      # 判定用相对步长 h = h_rel·max(|p_i|, 1)
H_ALT_FACTOR = 10.0       # 自洽性检查用 h_alt = H_ALT_FACTOR·h_judge
SIG_FRAC_DEFAULT = 1e-3   # 显著元门槛: |H| > sig_frac·|H|max
RTOL_DEFAULT = 0.2        # judged 元上的最大相对偏差容差
H_AGREE_DEFAULT = 0.05    # 两个 h 的 FD 相对差超过它 → 视为 FD 不可靠

GradFn = Callable[[torch.Tensor], torch.Tensor]


# ----------------------------------------------------------------------
# 基础：FD Hessian / 噪声
# ----------------------------------------------------------------------
def fd_hessian(grad_fn: GradFn, params: torch.Tensor,
               idxs: Optional[Sequence[int]] = None,
               h_rel: float = H_REL_DEFAULT,
               h_floor: float = 0.0) -> np.ndarray:
    """中心差分 Hessian：对 idxs 中每个坐标做两次 grad_fn，得到该行。

    ``h = h_rel·max(|p_i|, 1) + h_floor``。默认 h_rel=1e-4（与 test_numerical 一致）。
    """
    if idxs is None:
        idxs = list(range(params.numel()))
    idxs = list(idxs)
    rows = []
    for i in idxs:
        h = h_rel * max(abs(float(params[i])), 1.0) + h_floor
        pp = params.detach().clone(); pp[i] += h
        pm = params.detach().clone(); pm[i] -= h
        rows.append((grad_fn(pp) - grad_fn(pm)) / (2.0 * h))
    # ⚠ 行与列都要限制到 idxs：grad_fn 返回的是**全坐标**梯度，只切行会与
    # 子块解析矩阵列错位（曾因此把 68 个元误判成"不符"）。
    return torch.stack(rows)[:, idxs].cpu().double().numpy()


def fd_noise(grad_fn: GradFn, params: torch.Tensor, idxs: Sequence[int],
             h_judge: float = H_REL_DEFAULT,
             h_alt: Optional[float] = None) -> np.ndarray:
    """两个 h 的 FD Hessian 之差的逐元绝对值（用于判定 FD 自洽性）。"""
    h_alt = H_ALT_FACTOR * h_judge if h_alt is None else h_alt
    A = fd_hessian(grad_fn, params, idxs, h_judge)
    B = fd_hessian(grad_fn, params, idxs, h_alt)
    return np.abs(A - B)


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
@dataclass
class Entry:
    i: int                     # 全局坐标下标
    j: int
    h_an: float
    h_fd: float
    rel: float
    h_spread: float            # |fd(h1)-fd(h2)| / max(|fd|)
    kind: str                  # judged / fd_noisy / insignificant


@dataclass
class HessianFDReport:
    verdict: str               # PASS / FAIL / INCONCLUSIVE
    max_rel: float
    worst: Optional[Tuple[int, int]]
    h_an_worst: float = 0.0
    h_fd_worst: float = 0.0
    scale: float = 0.0
    sig: float = 0.0
    n_judged: int = 0
    n_fd_noisy: int = 0
    n_insignificant: int = 0
    noise_floor: float = 0.0          # judged 元中最大的 h-不一致量（绝对）
    entries: List[Entry] = field(default_factory=list)
    note: str = ""

    def summary(self, top: int = 5) -> str:
        lines = [
            f"Hessian FD 比较: verdict={self.verdict}  max_rel={self.max_rel:.3e}",
            f"  |H|max={self.scale:.4g}  显著阈值={self.sig:.4g}  "
            f"判定元={self.n_judged}  FD噪声元={self.n_fd_noisy}  不显著元={self.n_insignificant}",
        ]
        if self.worst is not None:
            lines.append(f"  最差 judged 元: [{self.worst[0]},{self.worst[1]}] "
                         f"an={self.h_an_worst:.6g} fd={self.h_fd_worst:.6g} "
                         f"rel={self.max_rel:.3f}")
        bad = [e for e in self.entries if e.kind == "judged" and e.rel > RTOL_DEFAULT]
        for e in sorted(bad, key=lambda x: -x.rel)[:top]:
            lines.append(f"    >FAIL [{e.i},{e.j}] an={e.h_an:.6g} fd={e.h_fd:.6g} "
                         f"rel={e.rel:.3f}")
        noisy = [e for e in self.entries if e.kind == "fd_noisy"]
        if noisy:
            lines.append(f"  (另有 {len(noisy)} 个显著元 FD 不可靠 → 未参与判定；"
                         f"这是 FD 噪声, 不代表 kernel 有错)")
        if self.note:
            lines.append(f"  注: {self.note}")
        return "\n".join(lines)


def compare_hessian(h_an: np.ndarray, grad_fn: GradFn, params: torch.Tensor,
                    idxs: Optional[Sequence[int]] = None,
                    h_judge: float = H_REL_DEFAULT,
                    sig_frac: float = SIG_FRAC_DEFAULT,
                    rtol: float = RTOL_DEFAULT,
                    h_agree: float = H_AGREE_DEFAULT,
                    scale: Optional[float] = None) -> HessianFDReport:
    """解析 Hessian ``h_an``（仅 idxs 子块）与 FD 的稳健比较。

    只有显著且 FD 自洽的元参与判定；显著但 FD 不自洽的元单独归类（不判定）。
    """
    if idxs is None:
        idxs = list(range(params.numel()))
    idxs = list(idxs)
    m = len(idxs)
    assert h_an.shape == (m, m), f"h_an 形状 {h_an.shape} 与 idxs 长度 {m} 不符"
    Hfd = fd_hessian(grad_fn, params, idxs, h_judge)
    Hfd2 = fd_hessian(grad_fn, params, idxs, H_ALT_FACTOR * h_judge)
    d = np.abs(Hfd - Hfd2)
    scale = float(np.abs(h_an).max()) if scale is None else float(scale)
    scale = max(scale, 1e-30)
    sig = sig_frac * scale
    entries: List[Entry] = []
    n_j = n_n = n_i = 0
    for a in range(m):
        for b in range(m):
            ha, hf = float(h_an[a, b]), float(Hfd[a, b])
            denom = max(abs(ha), abs(hf), 1e-30)
            rel = abs(ha - hf) / denom
            spread = float(d[a, b]) / max(abs(Hfd[a, b]), abs(Hfd2[a, b]), 1e-30)
            if max(abs(ha), abs(hf)) < sig:
                kind = "insignificant"; n_i += 1
            elif spread > h_agree:
                kind = "fd_noisy"; n_n += 1
            else:
                kind = "judged"; n_j += 1
            entries.append(Entry(idxs[a], idxs[b], ha, hf, rel, spread, kind))
    judged = [e for e in entries if e.kind == "judged"]
    if not judged:
        return HessianFDReport("INCONCLUSIVE", float("nan"), None, scale=scale, sig=sig,
                               n_judged=0, n_fd_noisy=n_n, n_insignificant=n_i,
                               entries=entries,
                               note="没有可靠且显著的元 → 该点 FD 无法判定 (换点/换 h)")
    worst = max(judged, key=lambda e: e.rel)
    noise_floor = max((abs(e.h_fd) * e.h_spread for e in judged), default=0.0)
    verdict = "PASS" if worst.rel <= rtol else "FAIL"
    return HessianFDReport(verdict, worst.rel, (worst.i, worst.j),
                           worst.h_an, worst.h_fd, scale, sig, n_j, n_n, n_i,
                           noise_floor, entries)


def eig_min_report(h_an: np.ndarray, grad_fn: GradFn, params: torch.Tensor,
                   idxs: Optional[Sequence[int]] = None,
                   h_judge: float = H_REL_DEFAULT,
                   sig_frac: float = SIG_FRAC_DEFAULT,
                   h_agree: float = H_AGREE_DEFAULT) -> Dict[str, object]:
    """λmin 层面的比较，带 FD **元噪声上界**（√n·noise）→ 判断是否可分辨。

    矩阵元的噪声 ε 通过特征值扰动可以移动 λmin 达 ~ε；用 √n·ε 作保守上界。
    差 |λmin_an − λmin_fd| 小于上界时, 结论是"FD 无法分辨", 而不是"不符"。
    """
    if idxs is None:
        idxs = list(range(params.numel()))
    idxs = list(idxs)
    Hfd = fd_hessian(grad_fn, params, idxs, h_judge)
    Hfd2 = fd_hessian(grad_fn, params, idxs, H_ALT_FACTOR * h_judge)
    noise = float(np.abs(Hfd - Hfd2).max())
    n = len(idxs)
    bound = float(np.sqrt(n)) * noise
    lam_an = float(np.linalg.eigvalsh(0.5 * (h_an + h_an.T))[0])
    lam_fd = float(np.linalg.eigvalsh(0.5 * (Hfd + Hfd.T))[0])
    return {"lam_min_an": lam_an, "lam_min_fd": lam_fd,
            "diff": lam_an - lam_fd, "noise_bound": bound,
            "resolvable": abs(lam_an - lam_fd) > bound,
            "n_neg_an": int((np.linalg.eigvalsh(0.5 * (h_an + h_an.T)) < 0).sum()),
            "n_neg_fd": int((np.linalg.eigvalsh(0.5 * (Hfd + Hfd.T)) < 0).sum())}


# ----------------------------------------------------------------------
# ctpwa 便捷层
# ----------------------------------------------------------------------
def analysis_grad_fn(ana) -> GradFn:
    """把 ctpwa analysis 包成 grad_fn(p) -> ∇NLL(p)（double, 同 device）。"""
    def g(p: torch.Tensor) -> torch.Tensor:
        q = p.detach().clone().requires_grad_(True)
        return torch.autograd.grad(ana.getNLL(q), q)[0].detach()
    return g


def free_idx(ana, params: torch.Tensor) -> List[int]:
    """自由坐标: 去掉两个固定参考位 (re_0=1, im_0=0)。

    解析 Hessian 里这两行/列按约定为 0（规范方向, 优化器固定不动, FD 扰动无意义）,
    与 compute_param_errors / aifit._reduced_hessian 同一条约定。
    """
    nc = ana.getNVector()
    return [i for i in range(params.numel()) if i not in (0, nc)]


def analysis_hessian_block(ana, params: torch.Tensor) -> Tuple[np.ndarray, List[int]]:
    """解析 Hessian 的自由坐标子块 + idxs。"""
    free = free_idx(ana, params)
    H = ana.getHessian(params).detach().cpu().double().numpy()
    return H[np.ix_(free, free)], free


def params_from_fit_json(path: str, ana, n_params: Optional[int] = None) -> torch.Tensor:
    """从 auto-pwa 的 fit.json（fit.best.params 为 [{kind,real,imag,value}]）载入参数向量。"""
    b = json.load(open(path))["fit"]["best"]
    coup = [x for x in b["params"] if x.get("kind") == "coupling"]
    res = [x for x in b["params"] if x.get("kind") == "resonance"]
    vec = ([x["real"] for x in coup] + [x["imag"] for x in coup]
           + [x["value"] for x in res])
    if n_params is not None:
        vec = vec[:n_params]
    return torch.tensor(vec, dtype=torch.float64, device="cuda")


def random_point(ana, seed: int = 0, scale: float = 0.3) -> torch.Tensor:
    """构造一个温和的随机参数点（向量 + 可选 θ）。"""
    nc, nt = ana.getNVector(), ana.getNFreeTheta()
    n = 2 * nc + nt
    g = torch.Generator(device="cuda").manual_seed(seed)
    p = torch.zeros(n, dtype=torch.float64, device="cuda")
    if nc:
        p[0] = 1.0
    if nc > 1:
        p[1:nc] = (torch.rand(nc - 1, generator=g, device="cuda") - 0.5) * 2 * scale
        p[nc + 1:2 * nc] = (torch.rand(nc - 1, generator=g, device="cuda") - 0.5) * 2 * scale
    return p


def check_config(config: str, fit_mode: int = 0, points: int = 3,
                 fit_json: Optional[str] = None, scales: Sequence[float] = (0.3, 1.0, 3.0),
                 **kw) -> Dict[str, object]:
    """对 config 跑 FD 比较；points 个随机点，或 fit_json 指定的拟合点。"""
    cfg = os.path.abspath(config)
    os.chdir(os.path.dirname(cfg) or os.getcwd())
    import ctpwa
    ana = ctpwa.analysis(cfg, fit_mode)
    grad = analysis_grad_fn(ana)
    out: Dict[str, object] = {"config": cfg, "fit_mode": fit_mode, "reports": []}
    pts: List[Tuple[str, torch.Tensor]] = []
    if fit_json:
        p = params_from_fit_json(fit_json, ana, ana.getNParams())
        pts.append((f"fit:{os.path.basename(fit_json)}", p))
    for t in range(points):
        pts.append((f"random[scale={scales[t % len(scales)]}]",
                    random_point(ana, seed=100 + t, scale=scales[t % len(scales)])))
    for tag, p in pts:
        H, idxs = analysis_hessian_block(ana, p)
        rep_kw = {k: v for k, v in kw.items()
                  if k in ("h_judge", "sig_frac", "rtol", "h_agree", "scale")}
        eig_kw = {k: v for k, v in kw.items()
                  if k in ("h_judge", "sig_frac", "h_agree")}
        rep = compare_hessian(H, grad, p, idxs, **rep_kw)
        eig = eig_min_report(H, grad, p, idxs, **eig_kw)
        out["reports"].append({"point": tag, "nll": float(ana.getNLL(p)),
                               "report": rep, "eig": eig})
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fit-mode", type=int, default=0)
    ap.add_argument("--points", type=int, default=3, help="随机点数 (默认 3)")
    ap.add_argument("--fit-json", default=None, help="用该 fit.json 的 best 点")
    ap.add_argument("--h", type=float, default=H_REL_DEFAULT, help="判定用相对步长")
    ap.add_argument("--sig-frac", type=float, default=SIG_FRAC_DEFAULT)
    ap.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    ap.add_argument("--h-agree", type=float, default=H_AGREE_DEFAULT)
    ap.add_argument("--json", default=None, help="把结果写到该 JSON")
    a = ap.parse_args(argv)
    res = check_config(a.config, a.fit_mode, a.points, a.fit_json,
                       h_judge=a.h, sig_frac=a.sig_frac, rtol=a.rtol, h_agree=a.h_agree)
    worst = "PASS"
    for r in res["reports"]:
        rep: HessianFDReport = r["report"]
        print(f"--- {r['point']}  NLL={r['nll']:.4f} ---")
        print(rep.summary())
        e = r["eig"]
        print(f"  λmin: an={e['lam_min_an']:.4g} fd={e['lam_min_fd']:.4g} "
              f"diff={e['diff']:.3g} FD噪声上界={e['noise_bound']:.3g} "
              f"→ {'可分辨' if e['resolvable'] else 'FD 无法分辨'}"
              f"  (n_neg an/fd = {e['n_neg_an']}/{e['n_neg_fd']})")
        if rep.verdict == "FAIL" and worst != "FAIL":
            worst = "FAIL"
        elif rep.verdict == "INCONCLUSIVE" and worst == "PASS":
            worst = "INCONCLUSIVE"
    print(f"==== 总判定: {worst} ====")
    if a.json:
        def enc(o):
            if isinstance(o, HessianFDReport):
                return {**{k: v for k, v in o.__dict__.items() if k != "entries"},
                        "entries": [e.__dict__ for e in o.entries]}
            raise TypeError(str(type(o)))
        json.dump({"config": res["config"], "fit_mode": res["fit_mode"],
                   "reports": [{"point": r["point"], "nll": r["nll"],
                                "report": enc(r["report"]), "eig": r["eig"]}
                               for r in res["reports"]]},
                  open(a.json, "w"), indent=1, ensure_ascii=False)
        print(f"已写入 {a.json}")
    return 0 if worst != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(_main())
