#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_interp_shape.py — Interp 模型「插入形状 vs 输出形状」核查工具
================================================================

背景
----
ctpwa 的 Interp 共振模型从 txt(x, Re, Im) 读一张复振幅表 F(m) = Re(m) + i·Im(m)，
在 GPU 端用 interpEval()（src/ResModel.cuh:248，src/Resonance.cu:80 装载 aux）
逐事件按不变质量 m 求值，当作该共振节点的动力学因子乘进振幅。

用户常见困惑：writeResult 产出的 ROOT TH1F（h_<chain>-<intermediate>-<res>，
见 src/Analysis.cu:2529）画出来的「形状」与输入的 txt 曲线对不上。

本工具把「检查」拆成三层，逐层隔离原因：

  层 0  文件/解析层    —— x 单位(GeV/MeV)、网格是否等距(ctpwa 容差 1e-10 GeV
                       绝对量！)、method 字符串、表区间是否覆盖物理区。
  层 1  引擎层          —— 「双 writeResult 比值法」：同一 config、同一批 phsp
                        事件、只换插值表（基准表 F≡1+i0），逐事件
                        w_输入 / w_基准 = C·|F(m)|²（角分布/Bf/相空间密度/归一
                        全部逐事件消掉，只剩一个整体常数 C）。把两文件的同名
                        h_ 直方图逐 bin 相除，先定出 C=median(ratio/|F|²)，
                        再与 txt 叠比形状。

已知坑（2026-09 实测）
----------------------
  * 分波直方图名 = h_<sanitizeROOTName(共振态拓扑名)>，例如
    mass0_Kp_Km/h_Jpsi_eta_shape_shape_Kp_Km（不是「链-中间态-共振态」）。
    先 uproot 列出 keys(recursive=True) 确认真名。
  * config 里 Interp 的 file/method 过去只认**直接键**；`options:` 嵌套写法
    （tests/configs/interp.yml、example/config.yml）会被旧解析器静默丢弃。
    现在两种写法都支持，但旧 .so 上请用直接键。
  * Interp 无自由参数 → 其 block nFree==0；旧版本 computeNodeFactor 缺
    Interp 分支（default→1.0），模板 kernel 路径下**表被整体丢弃**。修复后
    层 1 比值才 = C·|F|²。
  层 2  行为层          —— 与 tf-pwa 同类模型对拍（模型语义差异，见 --tfpwa）。

另附两个纯 python 复刻（与 C++/tf 源码逐行对照）：
  interp_ctpwa()      复刻 ctpwa interpEval 全部语义（等距/非等距、hist/
                      linear/spline-Catmull-Rom、越界钳位、非等距 spline→hist
                      静默回退）。
  interp_tfpwa_linear() 复刻 tf-pwa `linear_txt`/`linear_npy`（表即振幅；
                      越界 <x_min 或 ≥x_max 时严格 = 0，与 ctpwa 的钳位不同）。

用法
----
  层 0:  python check_interp_shape.py shape.dat [--method spline] [--unit GeV]
  层 1:  python check_interp_shape.py shape.dat --root weight_best_input.root \\
             --ref-root weight_best_flat.root --dir mass0_Kp_Km \\
             --hist h_Jpsi_eta_shape_shape_Kp_Km --method linear
  层 2:  python check_interp_shape.py shape.dat --tfpwa --method linear
  自检:  python check_interp_shape.py --selftest

注：引擎求值核对（层 1 是否真在 GPU 上忠实执行）最后要在装有 GPU 的 ctpwa
环境里跑；本工具把「该比什么、怎么比」固化成可复现的数值步骤。
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    from scipy.interpolate import CubicSpline
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


# =====================================================================
# 层 0：文件解析与诊断
# =====================================================================

def load_table(path, unit="GeV"):
    """读 (x, Re, Im) 三列表。ctpwa 内部质量全部以 GeV 计，
    故 MeV 表必须除以 1000 才能与引擎对照。"""
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"{path}: 需要 3 列 (x, Re, Im)，实际 {arr.shape}")
    x, re, im = arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(float)
    scale = 1000.0 if unit.lower() == "mev" else 1.0
    x = x / scale
    return x, re, im


def uniformity_report(x, tol=1e-10):
    """ctpwa 在装载时（src/Resonance.cu:106-110）用绝对容差 1e-10 GeV 判等距：
    |x[i]-x[i-1]-dx| > 1e-10 → 走非等距分支。"""
    d = np.diff(x)
    if len(d) == 0:
        return dict(uniform=False, max_dev=0.0, tol=tol, note="点数 < 2")
    dx0 = d[0]
    dev = np.abs(d - dx0)
    uniform = bool(np.all(dev <= tol))
    return dict(uniform=uniform, max_dev=float(dev.max()), dx0=float(dx0), tol=tol)


def table_report(x, re, im, method, unit="GeV", physical=(None, None)):
    lines = []
    N = len(x)
    lines.append(f"点数 N            : {N}")
    lines.append(f"x 区间            : [{x[0]:.6g}, {x[-1]:.6g}] {unit}")
    if physical[1] is not None:
        cov = float(np.sum((x >= physical[0]) & (x <= physical[1]))) / N
        lines.append(
            f"物理区 [{physical[0]:.4g}, {physical[1]:.4g}] {unit} 覆盖率: {cov:.1%}")
    u = uniformity_report(x)
    branch = "等距压缩分支 (x_min+dx)" if u["uniform"] else "非等距分支 (逐点二分)"
    lines.append(f"网格等距(≤1e-10 GeV): {u['uniform']}   max|dx−dx0|={u['max_dev']:.3g} → {branch}")
    if not u["uniform"] and method == "spline":
        lines.append(
            "  ⚠️ 非等距 + method=spline → ctpwa 静默回退为分段常数(hist)"
            "（src/ResModel.cuh:281），输出是阶梯不是光滑样条！")
    if method == "hist":
        lines.append(
            "  ⚠️ method=hist 语义：把 x 行当作节点（左连续阶梯，值取左侧节点）；"
            "若 x 列其实是‘bin 中心’会产生半格偏移，linear 则无此问题。")
    if np.all(np.abs(re - re[0]) < 1e-12) and np.all(np.abs(im - im[0]) < 1e-12):
        lines.append("  ℹ️ 表为常数（几乎平坦）：若相空间 MC 为 flat-Dalitz，"
                     "输出谱仍会随 m 上升（∝ 2m·相空间），这是折叠效应不是 bug。")
    # 越界行为提示（实测 GPU 验证，2026-09）
    lines.append(
        "越界行为(ctpwa, 实测): 等距(≤1e-10)网格 → 顶端钳到 y[N-1]；"
        "非等距网格 → 两端**线性外推**。"
        "底端: 等距时 pos∈(−1,0) 一个 bin 内也线性外推、更远钳 y[0]")
    lines.append(
        "  ⚠️ 表若未覆盖物理质量区，输出会按上述规则外推/钳位 → 与表内形状不符")
    lines.append(
        "越界行为(tf-pwa linear_txt): m<x_min 或 m≥x_max → 严格 = 0")
    return "\n".join(lines)


# =====================================================================
# 层 1（工具）：两个引擎的 numpy 复刻
# =====================================================================

def interp_ctpwa(m, x, re, im, method="linear", tol=1e-10):
    """复刻 ctpwa interpEval（src/ResModel.cuh:230-293）+ 装载判定
    （src/Resonance.cu:106-125）。method: hist|linear|spline（Catmull-Rom）。
    与 C++ 语义逐条对应；返回复数 F(m)。"""
    m = np.asarray(m, dtype=float)
    N = len(x)
    out = np.zeros(m.shape, dtype=complex)
    u = uniformity_report(x, tol)
    d = np.diff(x)
    dx0 = d[0]

    if u["uniform"]:
        # ---- 等距: pos=(m−x0)/dx, i=(int)pos（C 截断）----
        # i<0 → i=0,frac=0; i>=N-1 → i=N-2,frac=1; 否则 frac=pos−i
        # 注: pos∈(−1,0) 时 (int)pos==0 → frac=pos<0 → 底端一个 bin 内线性外推
        pos = (m - x[0]) / dx0
        t = np.trunc(pos).astype(int)
        i = np.clip(t, 0, N - 2)
        frac = pos - t
        frac = np.where(t < 0, 0.0, frac)
        frac = np.where(t >= N - 1, 1.0, frac)
    else:
        # ---- 非等距: 二分 → i = 最大 idx 使 x[i] ≤ m（C 保证 i ≤ N-2）----
        # frac=(m−x[i])/(x[i+1]−x[i]); i==0 或 i==N-2 时也照算
        # → 两端都线性外推（与等距顶端钳位不同！）
        idx = np.searchsorted(x, m, side="right") - 1
        i = np.clip(idx, 0, N - 2)
        x_i = x[i]
        x_j = x[i + 1]
        frac = (m - x_i) / np.maximum(x_j - x_i, 1e-300)

    def _spline(y):
        # Catmull-Rom，端点反射外推（与 C++ 完全一致）
        im1 = np.maximum(i - 1, 0)
        ip1 = np.minimum(i + 1, N - 1)
        ip2 = np.minimum(i + 2, N - 1)
        y0 = np.where(i > 0, y[im1], y[i] - (y[ip1] - y[i]))
        y1 = y[i]
        y2 = y[ip1]
        y3 = np.where(i + 2 < N, y[ip2], y[ip1] + (y[ip1] - y[i]))
        f2 = frac * frac
        f3 = f2 * frac
        return 0.5 * (2 * y1 + (-y0 + y2) * frac
                      + (2 * y0 - 5 * y1 + 4 * y2 - y3) * f2
                      + (-y0 + 3 * y1 - 3 * y2 + y3) * f3)

    def _hist(y):
        return y[i]

    def _linear(y):
        return y[i] + (y[i + 1] - y[i]) * frac

    # 非等距 + spline → C++ 回退 hist
    if method == "spline" and not u["uniform"]:
        method = "hist"
    if method == "hist":
        return _hist(re) + 1j * _hist(im)
    if method == "linear":
        return _linear(re) + 1j * _linear(im)
    if method == "spline":
        return _spline(re) + 1j * _spline(im)
    raise ValueError(method)


def interp_tfpwa_linear(m, x, re, im):
    """复刻 tf-pwa `linear_txt`/`linear_npy`
    （tf_pwa/amp/interpolation.py:InterpLinearNpy.interp，2026 版语义）。
    表即振幅 F=Re+iIm；区间 [x_min, x_max) 内线性，之外严格 0
    （x ≥ x_max 含端点 → 0，实测确认）。"""
    m = np.asarray(m, dtype=float)
    x = np.asarray(x, dtype=float)
    re = np.asarray(re, dtype=float)
    im = np.asarray(im, dtype=float)
    # tf.raw_ops.Bucketize 实测为「边界 ≤ m 计数」→ np.searchsorted side='right'
    idx = np.searchsorted(x, m, side="right")
    out = np.zeros(m.shape, dtype=complex)
    inside = (idx >= 1) & (idx <= len(x) - 1)   # 1..N-1：端点 x[-1] 被排除
    seg = np.clip(idx - 1, 0, len(x) - 2)
    frac = np.where(inside, (m - x[seg]) /
                    (x[seg + 1] - x[seg]), 0.0)
    out = np.where(inside,
                   (re[seg] + frac * (re[seg + 1] - re[seg]))
                   + 1j * (im[seg] + frac * (im[seg + 1] - im[seg])), 0.0)
    return out


def tfpwa_width_linear_formula(m, m0, g0, x, re, im, width_scale=False):
    """tf-pwa `width_linear_txt`（与 Interp 语义完全不同，仅供对照说明）：
    表是自能 Π(m)，放进
        F = 1 / ( m0² − m² − m0·Γ0·( ReΠ(m) − ReΠ(m0) + i·ImΠ(m) ) )
    width_scale=True 时用 Π(m)/ImΠ(m0) 使 Γ0 为「正常宽度」。
    若你的 txt 其实来自这种模型，ctpwa Interp（表直接当振幅）不适用。"""
    fm = interp_tfpwa_linear(np.asarray([m0]), x, re, im)[0]
    if width_scale:
        g0 = g0 / fm.imag
    re_p = interp_tfpwa_linear(m, x, re, im)
    re_part = (m0 * m0 - m * m) - m0 * g0 * (re_p.real - fm.real)
    im_part = m0 * g0 * re_p.imag
    dom = re_part * re_part + im_part * im_part
    return re_part / dom + 1j * (im_part / dom)


# =====================================================================
# 层 1：双 writeResult 比值法
# =====================================================================

def _read_hist(root_path, hist_path):
    """用 uproot 读 TH1F；hist_path 形如 'mass0_Kp_Km/h_chain1-R_KK-shape'。"""
    import uproot
    with uproot.open(root_path) as rf:
        try:
            obj = rf[hist_path]
        except KeyError:
            # 目录可能以 TDirectory 形式存在
            parts = hist_path.split("/")
            cur = rf
            for p in parts[:-1]:
                cur = cur[p]
            obj = cur[parts[-1]]
        vals, edges = obj.to_numpy()
        centers = 0.5 * (edges[:-1] + edges[1:])
        errs = obj.errors()
    return centers, vals, errs


def ratio_check(root_input, root_ref, hist_path, x, re, im, method="linear",
                out_prefix=None):
    """基准表(flat, F≡1)与输入表两次 writeResult 的同名 h_ 直方图逐 bin 相除：
    每事件 w_in/w_flat = |F(m)|²（角分布/Bf/相空间/归一逐事件抵消），
    bin 比值 ≈ 该 bin 内 <|F|²>。返回逐 bin 对照表。"""
    c1, v1, e1 = _read_hist(root_input, hist_path)
    c2, v2, e2 = _read_hist(root_ref, hist_path)
    if not np.allclose(c1, c2):
        raise ValueError("两个 root 的直方图轴不一致，无法逐 bin 比值")
    # 参考(flat)可能含拟合耦合 |v|² 比例：取 v2 非零 bin
    ok = (v2 > 0) & np.isfinite(v1 / v2)
    ratio = np.full_like(c1, np.nan)
    ratio[ok] = v1[ok] / v2[ok]
    # 期望：直接由 txt 表算 |F|²（bin 中心）
    F = interp_ctpwa(c1, x, re, im, method)
    expect = np.abs(F) ** 2
    # writeResult 的每事件权重含整体归一化常数 → 比值 = C·<|F|²>；先定出 C
    good = ok & (expect > 1e-12) & (ratio > 0)
    C = float(np.median(ratio[good] / expect[good])) if good.any() else float("nan")
    pred = C * expect
    rel_dev = np.full_like(c1, np.nan)
    rel_dev[good] = (ratio[good] - pred[good]) / np.maximum(pred[good], 1e-12)
    lines = [f"归一化常数 C = median(ratio/|F|²) = {C:.6g}（writeResult 整体归一化，非物理）",
             "逐 bin 对照 (bin 中心 | 实测比值 | C·|F|² | 相对偏差):"]
    for c_, r_, ex_, d_ in zip(c1, ratio, pred, rel_dev):
        if np.isnan(r_):
            continue
        lines.append(f"  m={c_:.4f}  ratio={r_:.4e}  C|F|²={ex_:.4e}  dev={d_:+.2%}")
    n = int(np.sum(~np.isnan(rel_dev)))
    if n:
        rms = float(np.sqrt(np.nanmean(rel_dev ** 2)))
        lines.append(f"有效 bin 数 {n}，去归一后 |F|² 相对偏差 RMS = {rms:.2%}"
                     "（≈0 → 引擎忠实执行插入表；统计涨落主导时 RMS≈1/√<w> 量级；"
                     "偏差集中在表区间外 → 越界外推/钳位；全区间系统性偏差 → 单位/方法语义）")
    report = "\n".join(lines)
    if out_prefix:
        with open(out_prefix + "_ratio.txt", "w") as f:
            f.write(report + "\n")
    print(report)
    return c1, ratio, expect


# =====================================================================
# 绘图辅助（可选 matplotlib）
# =====================================================================

def plot_curves(mgrid, curves, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, F in curves:
        ax[0].plot(mgrid, np.abs(F), label=name)
        ax[1].plot(mgrid, np.angle(F), label=name)
    ax[0].set(xlabel="m (GeV)", ylabel="|F|", title=title + "  |F|")
    ax[1].set(xlabel="m (GeV)", ylabel="arg F (rad)", title=title + "  phase")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print("figure ->", out_png)


# =====================================================================
# CLI
# =====================================================================

def _selftest():
    """节点处必须精确过表；中点半值；越界钳位/置零与 C++/tf 语义一致。"""
    x = np.linspace(1.0, 2.0, 5)
    re, im = x, 2 * x                      # F(x) = x + 2ix
    nodes = interp_ctpwa(x, x, re, im, "linear")
    assert np.allclose(nodes, re + 1j * im, atol=1e-12), nodes
    nodes_tf = interp_tfpwa_linear(x, x, re, im)
    assert np.allclose(nodes_tf[:4], re[:4] + 1j * im[:4], atol=1e-12)
    assert np.allclose(nodes_tf[4], 0, atol=1e-12), "tf-pwa x_max 端点应为 0"
    # ctpwa linear 越界（实测语义）:
    #   底端: pos∈(−1,0) 一个 bin 内线性外推；更远钳 y0
    #   顶端(等距): 钳 y[-1]
    below = interp_ctpwa([0.5, 0.99], x, re, im, "linear")
    assert np.allclose(below[0], 1 + 2j), below        # pos=-2 → 钳 y0
    assert np.allclose(below[1], 0.99 + 1.98j, atol=1e-12), below  # 一个 bin 内外推
    above = interp_ctpwa([2.0, 3.0], x, re, im, "linear")
    assert np.allclose(above, 2 + 4j), above           # 等距顶端钳 y[-1]
    # spline 节点过表
    sp = interp_ctpwa(x, x, re, im, "spline")
    assert np.allclose(sp, re + 1j * im, atol=1e-9), np.abs(sp - re - 1j * im).max()
    # 非等距（六位小数圆整，同 repo tests/data/interp_shape.dat 情形）+spline→hist
    xr = np.round(np.linspace(0.5, 3.0, 200), 6)   # dx 不可二进制精确表示
    assert not uniformity_report(xr)["uniform"], "圆整网格应判为非等距"
    re_r = np.cos(xr) + 0.3 * xr
    im_r = np.sin(xr) * 0.5
    sp2 = interp_ctpwa(xr[1:-1:10], xr, re_r, im_r, "spline")
    hi2 = interp_ctpwa(xr[1:-1:10], xr, re_r, im_r, "hist")
    assert np.allclose(sp2, hi2), "非等距 spline 应静默回退 hist"
    # 非等距两端线性外推（实测 GPU 语义；等距顶端是钳位，见上）
    xr2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5]); xr2[2] += 1e-6
    reu = xr2.copy()
    assert abs(interp_ctpwa([0.2], xr2, reu, np.zeros(5), "linear")[0].real - 0.2) < 1e-9
    assert abs(interp_ctpwa([3.0], xr2, reu, np.zeros(5), "linear")[0].real - 3.0) < 1e-9
    print("selftest OK：节点过表 / 线性半值 / 越界语义(等距钳位+非等距外推) / "
          "非等距回退 全部通过")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("====")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", nargs="?", help="(x Re Im) txt 表")
    ap.add_argument("--method", choices=["linear", "spline", "hist"], default="linear")
    ap.add_argument("--unit", choices=["GeV", "MeV"], default="GeV",
                    help="x 列单位（引擎内部一律 GeV；MeV 表除以 1000）")
    ap.add_argument("--physical", default=None, metavar="lo,hi",
                    help="物理区 m 范围(GeV)，用于报覆盖率（如 0.9874,2.5491）")
    ap.add_argument("--eval-grid", default=None, metavar="lo,hi,n",
                    help="求值网格（默认取表区间外扩 5%）")
    ap.add_argument("--tfpwa", action="store_true",
                    help="输出 tf-pwa linear_txt 对照曲线（层 2 行为对比）")
    ap.add_argument("--out", default=None, help="输出前缀：<pre>_curves.csv / _ratio.txt / _report.txt")
    ap.add_argument("--png", default=None, help="输出叠加图路径")
    # 层 1：双 root 比值法
    ap.add_argument("--root", default=None, help="输入表的 weight_best.root")
    ap.add_argument("--ref-root", default=None, help="基准表(F≡1+i0)的 weight_best.root")
    ap.add_argument("--dir", default=None, help="root 内目录，如 mass0_Kp_Km")
    ap.add_argument("--hist", default=None, help="h_ 对象名，如 h_chain1-R_KK-shape")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.table:
        ap.error("需要表文件或 --selftest")

    x, re, im = load_table(args.table, args.unit)

    if args.physical:
        phys = tuple(float(t) for t in args.physical.split(","))
    else:
        phys = (None, None)

    rep = table_report(x, re, im, args.method, args.unit, phys)
    print("=" * 72)
    print(f"层 0 表诊断: {args.table}")
    print(rep)

    # 求值网格
    if args.eval_grid:
        lo, hi, n = args.eval_grid.split(",")
        mgrid = np.linspace(float(lo), float(hi), int(n))
    else:
        pad = 0.05 * (x[-1] - x[0])
        mgrid = np.linspace(x[0] - pad, x[-1] + pad, 400)

    curves = []
    curves.append(("ctpwa " + args.method,
                   interp_ctpwa(mgrid, x, re, im, args.method)))
    u = uniformity_report(x)
    if args.method == "spline" and not u["uniform"]:
        # 引擎实际已回退 hist；给出"若等距应得到的"光滑样条作参照
        curves.append(("ctpwa spline (as designed / smooth)",
                       interp_ctpwa(mgrid, x, re, im, "spline", tol=np.inf)))
    if args.tfpwa:
        curves.append(("tfpwa linear_txt", interp_tfpwa_linear(mgrid, x, re, im)))
        if HAVE_SCIPY:
            cs_r = CubicSpline(x, re, bc_type="not-a-knot")
            cs_i = CubicSpline(x, im, bc_type="not-a-knot")
            curves.append(("tfpwa spline_c approx (not-a-knot)",
                       cs_r(mgrid) + 1j * cs_i(mgrid)))
        curves.append(("tfpwa width_linear_txt (demo)",
                       tfpwa_width_linear_formula(mgrid, 1.4, 0.2, x, re, im)))

    if args.out or args.png:
        if args.out:
            with open(args.out + "_curves.csv", "w") as f:
                f.write("m")
                for name, F in curves:
                    f.write(f",{name}_Re,{name}_Im,{name}_abs")
                f.write("\n")
                for j, m_ in enumerate(mgrid):
                    row = f"{m_:.6g}"
                    for _, F in curves:
                        row += f",{F[j].real:.8e},{F[j].imag:.8e},{abs(F[j]):.8e}"
                    f.write(row + "\n")
            with open(args.out + "_report.txt", "w") as f:
                f.write(rep + "\n")
            print("curves ->", args.out + "_curves.csv")
        if args.png and len(curves) >= 2:
            plot_curves(mgrid, curves, args.png,
                        title=os.path.basename(args.table))
        if not (args.png and len(curves) >= 2):
            # 文本小结
            print("-" * 72)
            print(f"曲线小结: m={mgrid[0]:.4g}..{mgrid[-1]:.4g} GeV, "
                  f"{len(curves)} 条曲线已生成（--out 存 csv / --png 出图）")
            for name, F in curves:
                i_peak = int(np.argmax(np.abs(F)))
                print(f"  {name:34s} |F|max={np.abs(F).max():.4g} @ m={mgrid[i_peak]:.4g}")

    # 层 1：双 root 比值法
    if args.root and args.ref_root:
        if not (args.dir and args.hist):
            ap.error("--root/--ref-root 需同时给 --dir 与 --hist")
        hist_path = f"{args.dir}/{args.hist}"
        print("=" * 72)
        print(f"层 1 引擎核查(双 writeResult 比值法): {hist_path}")
        ratio_check(args.root, args.ref_root, hist_path, x, re, im,
                    args.method, args.out)
        print("解释: 去归一后偏差 RMS≈0 → 引擎逐事件忠实按表求值；偏差集中在表区间外"
              "→ 越界外推/钳位；全区间系统性偏差 → 单位/解析/方法语义问题。")

    if not args.root and not args.out and not args.png:
        print("-" * 72)
        print("提示: 想看引擎实际求值曲线请加 --out/--png；"
              "层 1 引擎核查请提供 --root(输入表) 与 --ref-root(基准表 F≡1) 两个 root。")


if __name__ == "__main__":
    main()
