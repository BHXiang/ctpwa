#!/usr/bin/env python3
"""数值指纹：把一组 config 的 NLL / 梯度 / Hessian 打成 JSON，用于重构前后 A/B 对照。

为什么要这个（doc/modularization-plan.md / refactor-release-workflow.md §4）：
"重构没改变物理"这句话必须能被**数值**证明，而不是靠"测试都过了"。测试只覆盖
它们写的断言；指纹是同一组固定参数的逐值快照，可以跨构建、跨分支逐元素对比。

用法:
    python3 scripts/ab_fingerprint.py --out doc/parity/refactor.json
    # 对比两份指纹
    python3 scripts/ab_fingerprint.py --compare a.json b.json

约定:
  * 参数**一律在 CPU 上用固定 seed 生成**再 .to(device)——跨后端/跨构建时
    torch.Generator(device=cuda) 的随机流不同，会造出"不同参数点"的假差异。
  * 只依赖公开 API（analysis / getNLL / getHessian），所以新旧构建都能跑同一份脚本。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "tests" / "configs"
DEFAULT_CONFIGS = ["simple", "with_trans", "flatte", "gs", "interp", "ident2", "ident3"]
SEED = 20240917


def make_params(ana, device):
    """固定 seed 的合法参数向量 [real(v), imag(v), theta]，在 CPU 上生成再搬。"""
    n = ana.getNVector()
    nt = ana.getNFreeTheta()
    p = torch.zeros(2 * n + nt, dtype=torch.float64)
    p[0] = 1.0
    g = torch.Generator(device="cpu").manual_seed(SEED)
    p[1:n] = torch.rand(n - 1, generator=g) * 0.5
    p[n + 1:2 * n] = (torch.rand(n - 1, generator=g) - 0.5) * 0.5
    if nt:
        p[2 * n:] = ana.getFreeResParams()[0].to(torch.float64)
    return p.to(device)


def fingerprint_one(ctpwa, name):
    cfg = CONFIGS / f"{name}.yml"
    if not cfg.exists():
        return {"skipped": f"缺少配置 {cfg}"}
    ana = ctpwa.analysis(str(cfg))
    dev = torch.device("cuda:0") if ctpwa.backend() == "cuda" else torch.device("cpu")
    out = {"config": name, "n_vec": ana.getNVector(), "n_theta": ana.getNFreeTheta()}

    p = make_params(ana, dev)
    nll = ana.getNLL(p)
    out["nll"] = float(nll.item())

    pg = p.detach().clone().requires_grad_(True)
    nll_g = ana.getNLL(pg)
    grad = torch.autograd.grad(nll_g, pg)[0].detach().cpu().double()
    out["grad"] = [float(x) for x in grad]

    try:
        H = ana.getHessian(p.detach()).cpu().double()
        out["hess_dim"] = list(H.shape)
        out["hess"] = [float(x) for x in H.reshape(-1)]
    except Exception as e:  # Hessian 在某些配置/精度档上可能被明确拒绝
        out["hess_error"] = str(e)[:200]
    return out


def run(out_path, configs):
    import ctpwa
    print(f"[ab] backend={ctpwa.backend()} build={ctpwa.build_info()}", file=sys.stderr)
    data = {"backend": ctpwa.backend(), "build": ctpwa.build_info(), "entries": []}
    for name in configs:
        try:
            e = fingerprint_one(ctpwa, name)
        except Exception as ex:
            e = {"config": name, "error": f"{type(ex).__name__}: {ex}"[:300]}
        data["entries"].append(e)
        print(f"[ab] {name}: {e.get('nll', e.get('error', e.get('skipped')))}", file=sys.stderr)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, indent=1))
    print(f"[ab] wrote {out_path}", file=sys.stderr)


def compare(a_path, b_path, rel_tol, atol_scale):
    """逐元素判定：|a-b| <= atol_eff + rel_tol·max(|a|,|b|)，atol_eff = atol_scale·‖v‖∞。

    判据的两次修正都来自实测（见 doc/refactor-progress.md）：
      1. 不能用"整条向量的最大相对差"——`interp` 有个 ≈0 的梯度分量，绝对差 9e-13
         会被算成 7.9e-8 的"相对差"；
      2. 也不能用固定的绝对 atol——`grad[0]`（被固定的参考振幅方向）解析上应为 0，
         实际是 ~1e-4 的抵消噪声，且**逐次运行会抖动 ~1e-12**（同一二进制两次跑就不同）。
         对这种"零方向"，有意义的判据是"相对向量尺度的绝对误差"。
    """
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    ea = {e.get("config"): e for e in a["entries"]}
    eb = {e.get("config"): e for e in b["entries"]}
    bad_total = 0
    print(f"{'config':12s} {'field':6s} {'max|Δ|':>11s} {'max rel':>11s} {'坏点数':>7s}  状态")
    for name in sorted(set(ea) & set(eb)):
        x, y = ea[name], eb[name]
        for field in ("nll", "grad", "hess"):
            if field not in x or field not in y:
                continue
            va = x[field] if isinstance(x[field], list) else [x[field]]
            vb = y[field] if isinstance(y[field], list) else [y[field]]
            if len(va) != len(vb):
                print(f"{name:12s} {field:6s} {'—':>11s} {'—':>11s} {'—':>7s}  长度不同 {len(va)} vs {len(vb)}")
                bad_total += 1
                continue
            bad = 0
            max_abs = 0.0
            max_rel = 0.0
            vscale = max((abs(v) for v in va), default=0.0)
            atol_eff = atol_scale * vscale
            for p, q in zip(va, vb):
                d = abs(p - q)
                scale = max(abs(p), abs(q))
                max_abs = max(max_abs, d)
                if scale > 0:
                    max_rel = max(max_rel, d / scale)
                if d > atol_eff + rel_tol * scale:
                    bad += 1
            bad_total += bad
            print(f"{name:12s} {field:6s} {max_abs:11.3e} {max_rel:11.3e} {bad:7d}  {'ok' if bad==0 else 'MISMATCH'}")
    print(f"\n判据: 逐元素 |Δ| <= {atol_scale:g}·‖v‖∞ + {rel_tol:g}·max(|a|,|b|)")
    print("结论:", "全部一致" if bad_total == 0 else f"{bad_total} 个元素不一致")
    return 1 if bad_total else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    # 默认 1e-7：实测 CPU 梯度路径本身有 ~8e-8 的逐次运行抖动（interp 模型，
    # 同一二进制两次跑就不同），属于查找表/归约的舍入，与代码改动无关。
    # 真·物理变化远大于此（份额/参数差通常在 1e-2 量级）。
    ap.add_argument("--rel-tol", type=float, default=1e-7)
    ap.add_argument("--atol-scale", type=float, default=1e-8, dest="atol_scale")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare[0], args.compare[1], args.rel_tol, args.atol_scale)
    if not args.out:
        ap.error("需要 --out 或 --compare")
    run(args.out, args.configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
