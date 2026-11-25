#!/usr/bin/env python3
"""读取 weight_best.root 里的干涉矩阵，打印成表（并给出常用的派生量）。

`writeResult` 现在会往 ROOT 里写两张**已按本样本相干总额归一**的对称矩阵：

    interference_rec  P_ij / W_rec     重建侧（带效率 MC）
    interference_gen  T_ij / W_gen     产生侧（phsp_truth，无效率 MC）

    对角  = Σ w|A_i|² / W          （文献口径的 fit fraction F_i）
    非对角= Σ w·2Re(A_i A_j*) / W  （干涉项 I_ij）
    W = Σ_i M_ii + Σ_{i<j} M_ij = Σ w|ΣA|²

以及配套的：
    interference_rec_err / interference_gen_err   参数误差矩阵（writeResult 传了
                                                  hessian 才会写；σ 与上面同口径）
    integrals                                     N_rec/W_rec/N_gen/W_gen/npartials
    legends                                       LaTeX 图例（长度与分波数一致时用作行标签）

用法
----
    python3 example/read_interference.py weight_best.root
    python3 example/read_interference.py weight_best.root --sample gen
    python3 example/read_interference.py weight_best.root --bf-total 1.0e-3
    python3 example/read_interference.py weight_best.root --csv frac.csv

`--bf-total` 会给每个分波算 BF_i = BF_total × F_i（F_i 取 gen 侧对角）。注意
Σ_i BF_i 一般 ≠ BF_total：差额正是干涉项 Σ_{i<j} I_ij（同一张表里给出来了）。
"""

import argparse
import sys

import numpy as np


def _read_matrix(f, name):
    """uproot 读 TMatrixD：fElements 是行主序的 n×n 展平数组。"""
    obj = f[name]
    nrows = int(obj.member("fNrows"))
    ncols = int(obj.member("fNcols"))
    flat = np.asarray(obj.member("fElements"), dtype=float)
    return flat.reshape(nrows, ncols)


def _read_integrals(f):
    if "integrals" not in f:
        return {}
    t = f["integrals"].arrays(library="np")
    return {k: (float(v[0]) if k != "npartials" else int(v[0])) for k, v in t.items()}


def _labels(f, n):
    """行标签：优先 legends（长度对得上才用），否则用分波下标。"""
    try:
        leg = [str(x) for x in f["legends"]["legend"].array(library="np")[0]]
    except Exception:
        leg = []
    if len(leg) == n:
        return leg
    return [f"wave{i}" for i in range(n)]


def _fmt(x, sig=4):
    """科学计数/定点自适应，保证列宽稳定。"""
    if x == 0:
        return f"{0:.{sig}g}"
    return f"{x:.{sig}g}"


def per_wave_efficiency(mats, meta):
    """逐波效率 ε_i = (P_ii/N_rec)/(T_ii/N_gen)，由归一化矩阵 + integrals 还原。

    mats/ meta 里的矩阵是 P_ii/W_rec、T_ii/W_gen；必须先乘回 W 才是原始积分。
    单看 (diag/N) 会漏掉 W_rec/W_gen 因子（旧版 bug）。
    """
    Wr, Wg = meta.get("W_rec"), meta.get("W_gen")
    Nr, Ng = meta.get("N_rec"), meta.get("N_gen")
    if not (Wr and Wg and Nr and Ng):
        return None
    di, dr = np.diag(mats["gen"]), np.diag(mats["rec"])
    return (dr * Wr / Nr) / (di * Wg / Ng)


def print_matrix(name, M, E, labels, W=None, N=None):
    n = M.shape[0]
    diag = np.diag(M)
    tot_diag = diag.sum()
    inter = M[np.triu_indices(n, 1)].sum()

    head = f"== {name} =="
    if W is not None:
        head += f"   W={_fmt(W, 6)}"
    if N is not None:
        head += f"   N={_fmt(N, 6)}"
    print(head)
    print(f"   Σ_i F_i = {tot_diag:.6f}   Σ_{{i<j}} I_ij = {inter:+.6f}   "
          f"闭合 (ΣF+ΣI) = {tot_diag + inter:.6f}")
    if tot_diag > 0:
        print(f"   非相干份额 F_i/ΣF = {np.round(diag / tot_diag, 4).tolist()}")
    print()

    # 逐波：对角 + 误差
    print(f"   {'#':>3}  {'分波':<34} {'F_i':>11} {'σF_i':>11}")
    for i in range(n):
        err = f"{E[i, i]:.4g}" if E is not None else "-"
        print(f"   {i:>3}  {labels[i][:34]:<34} {diag[i]:>11.6f} {err:>11}")
    print()

    # 全矩阵
    w = 10
    print("   矩阵 (行= i, 列= j; 对角=F_i, 上三角=I_ij):")
    print("      " + "".join(f"{j:>{w}d}" for j in range(n)))
    for i in range(n):
        row = "".join(f"{M[i, j]:>{w}.5f}" for j in range(n))
        print(f"   {i:>2} {row}")
    if E is not None:
        print("   误差矩阵 σ:")
        print("      " + "".join(f"{j:>{w}d}" for j in range(n)))
        for i in range(n):
            row = "".join(f"{E[i, j]:>{w}.5f}" for j in range(n))
            print(f"   {i:>2} {row}")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="weight_best.root 路径")
    ap.add_argument("--sample", choices=("rec", "gen", "both"), default="both",
                    help="看哪一侧（默认两张都打）")
    ap.add_argument("--bf-total", type=float, default=None,
                    help="给总分支比，则逐波打印 BF_i = BF_total × F_i（gen 侧）")
    ap.add_argument("--csv", default=None,
                    help="把矩阵写成 CSV（行= i, 列= j, 两样本各一张）")
    args = ap.parse_args(argv)

    try:
        import uproot
    except ImportError:
        print("需要 uproot: pip install uproot", file=sys.stderr)
        return 2

    with uproot.open(args.root) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        have = [s for s in ("rec", "gen") if f"interference_{s}" in keys]
        if not have:
            print(f"{args.root} 里没有 interference_rec/gen"
                  f"（CTPWA_WR_NO_INTEGRALS=1 时不会写）", file=sys.stderr)
            return 1

        meta = _read_integrals(f)
        mats, errs = {}, {}
        for s in have:
            mats[s] = _read_matrix(f, f"interference_{s}")
            errs[s] = (_read_matrix(f, f"interference_{s}_err")
                       if f"interference_{s}_err" in keys else None)
        n = mats[have[0]].shape[0]
        labels = _labels(f, n)

    for s in have:
        if args.sample not in ("both", s):
            continue
        print_matrix(f"interference_{s}", mats[s], errs[s], labels,
                     W=meta.get(f"W_{s}"), N=meta.get(f"N_{s}"))

    # ---- 常用派生量 ----
    if len(have) == 2:
        Wr, Wg = meta.get("W_rec"), meta.get("W_gen")
        Nr, Ng = meta.get("N_rec"), meta.get("N_gen")
        if Wr and Wg:
            eps_bar = Wr / Wg
            print("== 效率（由两张表直接得到）==")
            why = "" if (Nr is None or Ng is None) else "  (两样本同源产生时)"
            print(f"   平均效率 ε̄ = W_rec/W_gen = {eps_bar:.6f}{why}")
            eps_i = per_wave_efficiency(mats, meta)
            if eps_i is not None:
                print(f"   每波效率 ε_i = (P_ii/N_rec)/(T_ii/N_gen) = "
                      f"{np.round(eps_i, 6).tolist()}")
                print("   注: 矩阵已是 P_ii/W、T_ii/W，公式里的 P_ii/T_ii 是乘回 W 后的原始积分")
                print("   注: ε_i 是相对值 ε_i^abs/ε_all（cut 后样本分布 ∝ εdΦ/∫ε）")
            print()

    if args.bf_total is not None:
        s = "gen" if "gen" in have else have[0]
        F = np.diag(mats[s])
        print(f"== 分支比（BF_total = {args.bf_total:g}, 用 {s} 侧对角）==")
        for i in range(len(F)):
            line = f"   {i:>3}  {labels[i][:34]:<34} BF_i = {args.bf_total * F[i]:.6e}"
            if errs[s] is not None:
                line += f" ± {args.bf_total * errs[s][i, i]:.2e}"
            print(line)
        print(f"   合计 Σ BF_i = {args.bf_total * F.sum():.6e}"
              f"（≠ BF_total 的部分 = 干涉项，见表）")
        print()

    # ---- CSV ----
    if args.csv:
        out = {}
        for s in have:
            M = mats[s]
            for i in range(M.shape[0]):
                for j in range(M.shape[0]):
                    out[f"{s}[{i},{j}]"] = M[i, j]
            if errs[s] is not None:
                for i in range(M.shape[0]):
                    out[f"{s}_err[{i},{i}]"] = errs[s][i, i]
        cols = list(out)
        with open(args.csv, "w") as fh:
            fh.write(",".join(cols) + "\n")
            fh.write(",".join(f"{out[c]:.10g}" for c in cols) + "\n")
        print(f"已写 {args.csv}（{len(cols)} 列）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
