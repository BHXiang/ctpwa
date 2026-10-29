#!/usr/bin/env python3
"""自由参数开销分解：NLL 单独 vs NLL+梯度，量出 d_F 表的代价。

背景
----
`block.nFree == 0`（质量/宽度全固定）时 `reComputeAmps` 整段跳过，每次 NLL
只是一次耦合矩阵乘；一旦有 free 质量/宽度，每次求值还要：

  1. `computeCustomAmpsKernel` 逐事件求 F/dF/d²F，并把 d_F 表**写**到显存；
  2. `resonanceGradientKernelRuntime` 逐事件**读** d_F 并累加梯度。

d_F 表大小 = nEv × nSL × nFree × nSigma × 16 B（double；Float 档 8 B）。
统计量大时这张表会把 GPU 压到显存带宽上限（利用率 100% 但吞吐上不去）。

用法
----
    CTPWA_DEVICE=cuda CTPWA_PROF=1 python3 scripts/bench_free_params.py config.yml
    ... --n 20 --grad-only         # 只看梯度那一遍
    ... --fix-theta                # 对照组：把 free 参数钉死再测一次

输出每项 ms/call；`--fix-theta` 用同一 config 的固定参数副本对比，二者之差就是
"自由参数带来的额外开销"，其中 NLL 单独一项已经包含 d_F 的**写**入。
"""

import argparse
import time

import torch

import ctpwa


def _n_events(ana, name, n_final):
    """getData/Phsp/BkgTensor 返回扁平的四动量表；事件数 = numel/(4·n末态)。"""
    try:
        t = getattr(ana, name)()
    except Exception:
        return 0
    if not hasattr(t, "numel") or t.numel() == 0:
        return 0
    return int(t.numel()) // (4 * max(n_final, 1))


def _time(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=10, help="计时重复次数")
    ap.add_argument("--grad-only", action="store_true")
    ap.add_argument("--fit-mode", type=int, default=0)
    args = ap.parse_args()

    ana = ctpwa.analysis(args.config, args.fit_mode)
    dev = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    n = 2 * nv + nt
    p = torch.zeros(n, dtype=torch.float64, device=dev)
    p[0] = 1.0
    if nt:
        p[2 * nv:] = ana.getFreeResParams()[0].to(device=dev)

    # 事件/波数从 SL 振幅表的形状读（[nSL, nEvents_total, nPolar]），
    # 不依赖 config 解析（[J: 0, P: 1] 这类 key 让 PyYAML 报 unhashable）
    try:
        sl = ana.getSLAmpsTensor().shape
        n_sl, n_evt_tot, n_pol = int(sl[0]), int(sl[1]), int(sl[2])
    except Exception:
        n_sl, n_evt_tot, n_pol = 1, 0, 0
    print(f"config={args.config}")
    print(f"backend={ctpwa.backend()} nvec={nv} ntheta={nt} nparams={n}")
    print(f"nSL={n_sl} nPolar={n_pol} nEvents_total(data+phsp+bkg)={n_evt_tot}")

    def fwd():
        with torch.no_grad():
            ana.getNLL(p)

    def full():
        q = p.clone().requires_grad_(True)
        torch.autograd.grad(ana.getNLL(q), q)[0]

    t_fwd = _time(fwd, args.n)
    t_full = _time(full, args.n)
    t_grad = t_full - t_fwd

    print(f"forward (NLL only)      : {t_fwd:8.2f} ms/call")
    if not args.grad_only:
        print(f"forward+grad            : {t_full:8.2f} ms/call")
        print(f"=> grad pass (d_F read) : {t_grad:8.2f} ms/call "
              f"({100.0 * t_grad / max(t_full, 1e-9):.1f}% of total)")
    # d_F 表估算（nSL 未直接暴露，用部分波数近似：每块 nSL ≤ 部分波数）
    if n_evt_tot and nt:
        gb = n_evt_tot * n_sl * nt * 16 / 1e9
        print(f"  d_F table ≈ {gb:.2f} GB (nEv={n_evt_tot} nSL={n_sl} "
              f"nFree={nt} nSigma=1, double)；Float 档减半")


if __name__ == "__main__":
    main()
