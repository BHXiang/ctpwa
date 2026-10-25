#!/usr/bin/env python3
"""compare_results.py — 比较 CPU / GPU 两次 dump 的 NLL、梯度、Hessian

用法:
    python3 tests/cpu_backend/compare_results.py a.npz b.npz [--label-a CPU --label-b GPU]

判据（与 doc/cpu-backend-plan.md 的验收口径一致）:
  NLL      相对差 < 1e-6
  梯度     相对差（以 max|grad| 归一） < 1e-6
  Hessian  相对差（以 max|H| 归一）    < 1e-6
"""
import sys

import numpy as np

RTOL_NLL = 1e-6
RTOL_GRAD = 1e-6
RTOL_H = 1e-6
# 低于此量级视为"解析上就是 0"，改用绝对判据（见 judge()）
NEAR_ZERO = 1e-3


def rel(a, b, floor=1e-30):
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), floor)
    return float(np.abs(a - b).max()) / scale


def judge(a, b, rtol):
    """返回 (是否通过, 差异, 判据说明)。

    近零量（max|·| < NEAR_ZERO）改用绝对容差：这类 config 的梯度/Hessian 解析上
    就是 0（例如 v 的整体标度不变性），残差是浮点噪声，用相对判据会被小分母放大
    成 O(1) 的假失配。
    """
    diff = float(np.abs(a - b).max())
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()))
    if scale < NEAR_ZERO:
        return diff < NEAR_ZERO, diff, "near-zero(abs<%.0e)" % NEAR_ZERO
    return diff <= rtol * scale, diff, "rel"


def main(pa, pb, la="A", lb="B"):
    A = np.load(pa, allow_pickle=True)
    B = np.load(pb, allow_pickle=True)
    keys = sorted(k for k in A.files if k in B.files)
    names = sorted({k.split("/")[0] for k in keys})

    print("%-12s %14s %14s %10s" % ("config", la, lb, "rel"))
    print("-" * 56)
    worst = {}
    bad = []
    for name in names:
        kn, kg, kh = name + "/nll", name + "/grad", name + "/H"
        if kn not in keys:
            continue
        nll_a, nll_b = float(A[kn]), float(B[kn])
        r_nll = abs(nll_a - nll_b) / max(abs(nll_a), abs(nll_b), 1e-30)
        has_g = kg in keys and kh in keys
        r_g = rel(A[kg], B[kg]) if has_g else float("nan")
        r_h = rel(A[kh], B[kh]) if has_g else float("nan")
        ok_g, d_g, note_g = judge(A[kg], B[kg], RTOL_GRAD) if has_g else (True, 0, "")
        ok_h, d_h, note_h = judge(A[kh], B[kh], RTOL_H) if has_g else (True, 0, "")
        worst["nll"] = max(worst.get("nll", 0.0), r_nll)
        if has_g:
            worst["grad"] = max(worst.get("grad", 0.0), r_g)
            worst["H"] = max(worst.get("H", 0.0), r_h)
        print("%-12s %14.9f %14.9f  nll=%.2e grad=%.2e H=%.2e %s%s"
              % (name, nll_a, nll_b, r_nll, r_g, r_h,
                 "" if ok_g and ok_h else "  <-- ",
                 ("grad:" + note_g + " ") if note_g == "near-zero(abs<%.0e)" % NEAR_ZERO else "")
              + ("" if ok_g else "grad_diff=%.1e " % d_g)
              + ("" if ok_h else "H_diff=%.1e" % d_h))
        if r_nll > RTOL_NLL or not ok_g or not ok_h:
            bad.append(name)

    print("-" * 56)
    print("worst: " + "  ".join("%s=%.2e" % (k, v) for k, v in worst.items()))
    if bad:
        print("FAIL: %d config(s) 超出容差: %s" % (len(bad), ", ".join(bad)))
        return 1
    print("PASS: 全部在容差内 (nll<%.0e, grad<%.0e, H<%.0e)" % (RTOL_NLL, RTOL_GRAD, RTOL_H))
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    labels = {}
    for i, a in enumerate(sys.argv):
        if a == "--label-a":
            labels["a"] = sys.argv[i + 1]
        if a == "--label-b":
            labels["b"] = sys.argv[i + 1]
    sys.exit(main(args[0], args[1], labels.get("a", "A"), labels.get("b", "B")))
