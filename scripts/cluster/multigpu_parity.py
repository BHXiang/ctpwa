#!/usr/bin/env python3
"""多卡一致性：N 张可见 GPU vs 单卡（同一 θ 的 NLL + 全梯度）。

为什么要单独一步：pytest 在**当前进程**里跑，改不了 `CUDA_VISIBLE_DEVICES`，
"1 卡 vs N 卡"只能跨进程比。集群作业 `job_validate_all.sh` 以 `gpu:v100:2`
提交，本脚本用子进程把 `CUDA_VISIBLE_DEVICES` 钉成 `0` 取单卡参考值，再与
当前进程（全部可见卡）逐项比对。

覆盖的都是 per-GPU 的分支：
  * 事例/相空间按卡切分 → 每卡 nEvents 不同，d_static / d_prop6 / d_dF 行数不同；
  * 静态链因子表（只有含固定节点的 config 才真的建表，见 static_nodes）；
  * σ 置换表、符号表、slamp 的逐卡副本。

用法（集群 GPU 节点上，仓库根目录）：
    python3 scripts/cluster/multigpu_parity.py
    python3 scripts/cluster/multigpu_parity.py --configs static_nodes with_trans
    MGP_FORCE=1 python3 scripts/cluster/multigpu_parity.py   # 单卡机器冒烟脚本自身

退出码：0 = 通过/跳过；1 = 有配置超差。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
sys.path.insert(0, str(TESTS))

# 默认覆盖三条不同的链结构：静态因子表 / trans 共轭广播 / σ 置换（全同粒子）
DEFAULT_CONFIGS = ["static_nodes", "with_trans", "ident4"]


def _params(ana, device):
    """tests/ 的规范初值（与 pytest 同源，GPU 后端上 v 的随机数固定 seed）。"""
    from conftest import make_params
    return make_params(ana, device)


def _evaluate(cfg):
    """在当前可见设备上算 NLL 与全梯度，返回 (nll, [grad...])。"""
    import torch
    import ctpwa

    os.chdir(TESTS)
    ana = ctpwa.analysis(str(TESTS / "configs" / f"{cfg}.yml"))
    dev = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")
    p = _params(ana, dev).requires_grad_(True)
    nll = ana.getNLL(p)
    grad, = torch.autograd.grad(nll, p)
    return float(nll.item()), [float(x) for x in grad.detach().cpu().tolist()]


def _child(cfg):
    nll, grad = _evaluate(cfg)
    print(f"CHILD_NLL {nll:.17g}")
    print("CHILD_GRAD " + " ".join(f"{x:.17g}" for x in grad))


def _parse(out):
    nll, grad = None, None
    for line in out.splitlines():
        if line.startswith("CHILD_NLL "):
            nll = float(line.split()[1])
        elif line.startswith("CHILD_GRAD "):
            grad = [float(x) for x in line.split()[1:]]
    if nll is None or grad is None:
        raise RuntimeError(f"子进程输出不完整:\n{out[-2000:]}")
    return nll, grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--child", metavar="CFG", help=argparse.SUPPRESS)
    ap.add_argument("--nll-rtol", type=float, default=1e-9)
    # 多卡梯度用 atomic/跨卡规约，顺序不同会带来 ~1e-8 相对噪声；
    # V100 实测可达 5.4e-8，1e-8 会把正常结果误判为失败。
    ap.add_argument("--grad-rtol", type=float, default=1e-7)
    args = ap.parse_args()

    if args.child:
        _child(args.child)
        return 0

    import torch
    if not torch.cuda.is_available():
        print("[mgp] 无 CUDA：跳过")
        return 0
    n_gpu = torch.cuda.device_count()
    print(f"[mgp] 可见 GPU 数 = {n_gpu}")
    if n_gpu < 2 and not os.environ.get("MGP_FORCE"):
        print("[mgp] < 2 张卡：跳过（集群作业里才有意义；MGP_FORCE=1 可强制自检）")
        return 0

    env1 = dict(os.environ)
    env1["CUDA_VISIBLE_DEVICES"] = "0"

    worst_all = 0.0
    failed = []
    for cfg in args.configs:
        out = subprocess.run([sys.executable, __file__, "--child", cfg],
                             cwd=str(ROOT), env=env1, capture_output=True,
                             text=True, timeout=3600)
        if out.returncode != 0:
            print(f"[mgp] {cfg}: 子进程失败\n{out.stderr[-2000:]}")
            failed.append(cfg)
            continue
        nll1, grad1 = _parse(out.stdout)
        nll2, grad2 = _evaluate(cfg)

        nll_rel = abs(nll1 - nll2) / max(abs(nll2), 1.0)
        scale = max(max(abs(x) for x in grad2), 1e-30)
        grad_abs = max(abs(a - b) for a, b in zip(grad1, grad2)) if grad1 else 0.0
        grad_rel = grad_abs / scale
        worst_all = max(worst_all, nll_rel, grad_rel)

        ok = nll_rel <= args.nll_rtol and grad_rel <= args.grad_rtol
        print(f"[mgp] {cfg:14s} NLL {nll2:.10f}  rel={nll_rel:.2e} | "
              f"grad max|Δ|={grad_abs:.2e} rel={grad_rel:.2e} | "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(cfg)

    print(f"[mgp] worst rel = {worst_all:.3e} "
          f"(nll<={args.nll_rtol:g}, grad<={args.grad_rtol:g})")
    if failed:
        print(f"[mgp] 不一致: {failed}")
        return 1
    print("[mgp] 多卡一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
