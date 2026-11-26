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


def compare(a_path, b_path, rel_tol):
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    ea = {e.get("config"): e for e in a["entries"]}
    eb = {e.get("config"): e for e in b["entries"]}
    bad = 0
    print(f"{'config':12s} {'field':10s} {'max|Δ|':>12s} {'max rel':>12s}  状态")
    for name in sorted(set(ea) & set(eb)):
        x, y = ea[name], eb[name]
        for field in ("nll", "grad", "hess"):
            if field not in x or field not in y:
                continue
            va = x[field] if isinstance(x[field], list) else [x[field]]
            vb = y[field] if isinstance(y[field], list) else [y[field]]
            if len(va) != len(vb):
                print(f"{name:12s} {field:10s} {'—':>12s} {'—':>12s}  长度不同 {len(va)} vs {len(vb)}")
                bad += 1
                continue
            max_abs = max(abs(p - q) for p, q in zip(va, vb))
            scale = max(1e-30, max(abs(p) for p in va))
            rel = max_abs / scale
            ok = rel <= rel_tol
            bad += 0 if ok else 1
            print(f"{name:12s} {field:10s} {max_abs:12.3e} {rel:12.3e}  {'ok' if ok else 'MISMATCH'}")
    print(f"\n判据: 相对差 <= {rel_tol:g}（重构不改变物理时应当逐位或近乎逐位相等）")
    print("结论:", "全部一致" if bad == 0 else f"{bad} 项不一致")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--rel-tol", type=float, default=1e-11)
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare[0], args.compare[1], args.rel_tol)
    if not args.out:
        ap.error("需要 --out 或 --compare")
    run(args.out, args.configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
