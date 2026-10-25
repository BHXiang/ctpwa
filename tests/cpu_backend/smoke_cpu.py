#!/usr/bin/env python3
"""smoke_cpu.py — CPU 后端端到端冒烟 + 数值自洽检查

前置: 已构建 CPU 扩展，并把 ctpwa.so 放进 PYTHONPATH 可达的目录，
例如:
    CTPWA_BACKEND=cpu python3 setup.py build_ext
    cp build/lib.*/ctpwa.so /tmp/ctpwa_cpu/
    PYTHONPATH=/tmp/ctpwa_cpu python3 tests/cpu_backend/smoke_cpu.py

检查项:
  1. 多个 config 能构造、NLL 有限
  2. analytic 梯度 vs 有限差分（θ 分量）
  3. simple config 对 golden 值（tests/golden/nll_simple.txt）
  4. 同一输入重复求值逐位一致
"""
import os
import sys

import torch

import ctpwa

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
CONFIGS = os.path.join(TESTS, "configs")
GOLDEN = os.path.join(TESTS, "golden", "nll_simple.txt")

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("  [FAIL] " + msg)


def make_params(ana):
    """与 tests/conftest.py::make_params 同布局；simple 的 v[1:] 为空，故 v=[1+0i]。"""
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    p = torch.zeros(2 * nv + nt, dtype=torch.float64)
    p[0] = 1.0
    if nt:
        p[2 * nv:] = ana.getFreeResParams()[0]
    return p


def main():
    print("[smoke] ctpwa module:", ctpwa.__file__)
    os.chdir(TESTS)   # config 里的相对 data 路径按 tests/ 解析

    # ---- 1. 多 config 构造 + NLL ----
    print("[1] construct + NLL")
    names = ["simple", "flatte", "gs", "interp", "custom", "ident2",
             "no_trans", "with_trans"]
    for name in names:
        cfg = os.path.join(CONFIGS, name + ".yml")
        if not os.path.exists(cfg):
            continue
        try:
            ana = ctpwa.analysis(cfg)
            p = make_params(ana)
            nll = ana.getNLL(p).item()
            ok = nll == nll and abs(nll) < 1e12      # 非 NaN / 非 1e30 sentinel
            check(ok, "%s: NLL=%r" % (name, nll))
            print("  %-12s NLL=%12.6f" % (name, nll))
        except Exception as e:                        # noqa: BLE001
            check(False, "%s: %s" % (name, e))
            print("  %-12s FAILED: %s" % (name, str(e)[:90]))

    # ---- 2. 梯度 vs 有限差分 ----
    print("[2] gradient vs finite difference (flatte)")
    ana = ctpwa.analysis(os.path.join(CONFIGS, "flatte.yml"))
    p0 = make_params(ana)
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    p = p0.clone().requires_grad_(True)
    nll = ana.getNLL(p)
    g = torch.autograd.grad(nll, p)[0].detach()

    h = 1e-5
    worst = 0.0
    for i in range(2 * nv, len(p0)):                  # 只比 θ 分量（v 分量信号低于 FD 噪声）
        pp, pm = p0.clone(), p0.clone()
        pp[i] += h
        pm[i] -= h
        fd = (ana.getNLL(pp).item() - ana.getNLL(pm).item()) / (2 * h)
        rel = abs(g[i].item() - fd) / max(abs(fd), 1e-8)
        worst = max(worst, rel)
        print("  [%d] AD=%+.8g  FD=%+.8g  rel=%.2e" % (i, g[i].item(), fd, rel))
    check(worst < 1e-6, "FD 最大相对差 %.2e >= 1e-6" % worst)

    # ---- 3. golden ----
    print("[3] golden nll_simple")
    if os.path.exists(GOLDEN):
        ana = ctpwa.analysis(os.path.join(CONFIGS, "simple.yml"))
        got = ana.getNLL(make_params(ana)).item()
        want = float(open(GOLDEN).read().strip())
        rel = abs(got - want) / abs(want)
        print("  golden=%.12f  cpu=%.12f  rel=%.2e" % (want, got, rel))
        check(rel < 1e-6, "golden 相对差 %.2e >= 1e-6" % rel)

    # ---- 4. 重复求值一致 ----
    print("[4] repeatability")
    a = ana.getNLL(make_params(ana)).item()
    b = ana.getNLL(make_params(ana)).item()
    check(a == b, "重复求值不一致: %.17g vs %.17g" % (a, b))

    # ---- 5. Hessian：对称性 + θ 块对有限差分 ----
    print("[5] getHessian")
    try:
        ana = ctpwa.analysis(os.path.join(CONFIGS, "simple.yml"))
        p0 = make_params(ana)
        nv, nt = ana.getNVector(), ana.getNFreeTheta()
        Ht = ana.getHessian(p0).double()
        H = Ht.numpy()
        n = len(p0)
        check(H.shape == (n, n), "Hessian 形状 %r" % (H.shape,))
        Ht2 = ana.getHessian(p0).double()
        same = bool(torch.equal(Ht, Ht2))
        print("  repeatable (bit-identical): %s" % same)
        check(same, "Hessian 重复调用不是逐位一致")

        sym = float((Ht - Ht.T).abs().max()) / max(float(Ht.abs().max()), 1e-30)
        print("  symmetric rel=%.2e" % sym)
        check(sym < 1e-10, "Hessian 不对称 rel=%.2e" % sym)

        # FD Hessian（用解析梯度做中心差分），只比 θ 块（v 块接近 0，FD 噪声大）
        h = 1e-5
        th0 = 2 * nv
        Hfd = [[0.0] * n for _ in range(n)]
        for i in range(n):
            pp, pm = p0.clone(), p0.clone()
            pp[i] += h
            pm[i] -= h
            pp.requires_grad_(True)
            pm.requires_grad_(True)
            gp = torch.autograd.grad(ana.getNLL(pp), pp)[0].detach()
            gm = torch.autograd.grad(ana.getNLL(pm), pm)[0].detach()
            for j in range(n):
                Hfd[i][j] = float((gp[j] - gm[j]) / (2 * h))
        Hfd = torch.tensor(Hfd, dtype=torch.float64)
        blk_ad = Ht[th0:, th0:]
        blk_fd = Hfd[th0:, th0:]
        rel = float((blk_ad - blk_fd).abs().max()) / max(float(blk_fd.abs().max()), 1e-30)
        print("  theta-block AD vs FD rel=%.2e" % rel)
        check(rel < 1e-6, "Hessian θ 块与 FD 相对差 %.2e >= 1e-6" % rel)
    except Exception as e:                            # noqa: BLE001
        check(False, "getHessian: %s" % str(e)[:120])
        print("  FAILED: %s" % str(e)[:120])

    if failures:
        print("\n[smoke] %d FAILURE(S)" % len(failures))
        return 1
    print("\n[smoke] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
