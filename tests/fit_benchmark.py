#!/usr/bin/env python3
"""真实数据拟合基准生成器（在集群上运行一次）。

真实数据有物理结构，拟合能收敛到合理参数。本脚本：
1. 在 real_data.yml（真实数据子集）上拟合若干步
2. 保存收敛参数 + 收敛 NLL 到 tests/golden/fit_real.txt
3. 测试 test_endtoend.py 的 test_real_fit_benchmark 用此基准验证

用法（集群上，先跑 prepare_real_data.py 生成真实数据）:
    python3 tests/fit_benchmark.py [--steps 200] [--lr 5e-3]

基准语义:
    fit_real.txt 记录:
        - 初始参数（固定 seed 可复现）
        - 拟合 N 步后的收敛 NLL
        - 收敛参数
    测试验证: 从同一起点拟合同样步数，NLL 应达到 ≤ 基准值 + 容差。
    数值实现变化（如 fast_math）时用 --update 重新生成。
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from conftest import TESTS_DIR, GOLDEN_DIR, make_params

SEED = 42
NLL_RTOL = 1e-4  # 收敛 NLL 相对容差


def fit(ana, device, steps, lr):
    """带 theta bounds 夹持的梯度下降拟合。"""
    n_vec = ana.getNVector()
    params = make_params(ana, device)
    free_res = ana.getFreeResParams()
    lower = free_res[1].to(device=device, dtype=torch.float64)
    upper = free_res[2].to(device=device, dtype=torch.float64)

    params.requires_grad_(True)
    history = []
    for step in range(steps):
        nll = ana.getNLL(params)
        grad = torch.autograd.grad(nll, params)[0]
        assert torch.isfinite(grad).all(), f"step{step}: 梯度非有限"
        updated = (params - lr * grad).detach()
        if ana.getNFreeTheta() > 0:
            updated[2 * n_vec :] = torch.clamp(updated[2 * n_vec :], lower, upper)
        params = updated.requires_grad_(True)
        history.append(ana.getNLL(params).item())
        if step % 50 == 0:
            print(f"  step {step}: NLL = {history[-1]:.4f}")

    return params.detach(), history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--update", action="store_true",
                        help="数值变化后重新生成基准")
    args = parser.parse_args()

    cfg = TESTS_DIR / "configs" / "real_data.yml"
    if not cfg.exists():
        print("错误: 先运行 prepare_real_data.py 生成真实数据 + real_data.yml")
        sys.exit(1)

    out = GOLDEN_DIR / "fit_real.txt"
    if out.exists() and not args.update:
        print(f"{out} 已存在。数值实现有变化时加 --update 重新生成。")
        return

    os.chdir(TESTS_DIR)
    import ctpwa

    ana = ctpwa.analysis(str(cfg))
    device = torch.device("cuda")

    init_params = make_params(ana, device)
    print(f"初始 NLL = {ana.getNLL(init_params).item():.4f}")
    print(f"拟合 {args.steps} 步 (lr={args.lr})...")
    final, history = fit(ana, device, args.steps, args.lr)
    final_nll = history[-1]

    print(f"\n收敛 NLL = {final_nll:.6f} (初始 {history[0]:.6f})")

    GOLDEN_DIR.mkdir(exist_ok=True)
    lines = [
        f"# 真实数据拟合基准 (steps={args.steps}, lr={args.lr}, seed={SEED})",
        f"nll_initial {history[0]:.16e}",
        f"nll_final   {final_nll:.16e}",
        f"params " + " ".join(f"{v:.16e}" for v in final.cpu().tolist()),
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"基准已保存: {out}")


def load_benchmark():
    """读取基准。返回 (nll_initial, nll_final, params_tensor)。"""
    p = GOLDEN_DIR / "fit_real.txt"
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    data = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        key, _, val = line.partition(" ")
        data[key] = val
    params = torch.tensor(
        [float(v) for v in data["params"].split()], dtype=torch.float64
    )
    return float(data["nll_initial"]), float(data["nll_final"]), params


if __name__ == "__main__":
    main()
