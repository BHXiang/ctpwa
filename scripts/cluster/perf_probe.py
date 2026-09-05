"""三档（float/hybrid/double）性能基准：getNLL / getNLL+grad / getHessian 计时。

用法: cd tests && python3 perf_probe.py <config.yml> [label]
输出每档每阶段的中位耗时(ms)与首段显存占用（nvidia-smi used，若有）。
重复次数可用环境变量 PERF_N_NLL（默认 10）/ PERF_N_HESS（默认 3）覆盖。
"""
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

TESTS = Path("/home/whitewash/pkgs/ctpwa/tests")
if not TESTS.exists():
    TESTS = Path("/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa/tests")
sys.path.insert(0, str(TESTS))

from conftest import make_params  # noqa: E402
import ctpwa  # noqa: E402

N_NLL = int(os.environ.get("PERF_N_NLL", "10"))
N_HESS = int(os.environ.get("PERF_N_HESS", "3"))


def nvsmi_used():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return [int(x) for x in out.stdout.split()]
    except Exception:
        return None


def bench(prec_cfg: str, label: str):
    t_nll, t_grad, t_hess = [], [], []
    ana = ctpwa.analysis(prec_cfg)
    p = make_params(ana, torch.device("cuda:0")).requires_grad_(True)
    used0 = nvsmi_used()

    # warmup（JIT 编译/cuBLAS workspace/缓存冷启动不计入）
    _ = float(ana.getNLL(p)); torch.cuda.synchronize()
    _ = ana.getHessian(p.detach()); torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for _ in range(N_NLL):
        t0 = time.perf_counter()
        nll = float(ana.getNLL(p))
        torch.cuda.synchronize()
        t_nll.append((time.perf_counter() - t0) * 1e3)
    for _ in range(max(1, N_NLL // 2)):
        p2 = (p + 1e-5 * torch.randn_like(p)).clone().requires_grad_(True)
        t0 = time.perf_counter()
        g = torch.autograd.grad(ana.getNLL(p2), p2)[0]
        torch.cuda.synchronize()
        t_grad.append((time.perf_counter() - t0) * 1e3)
    pd = p.detach()
    for _ in range(N_HESS):
        t0 = time.perf_counter()
        _ = ana.getHessian(pd)
        torch.cuda.synchronize()
        t_hess.append((time.perf_counter() - t0) * 1e3)
    used1 = nvsmi_used()
    med = lambda xs: statistics.median(xs)
    print(f"[PERF {label}] NLL med={med(t_nll):8.2f} ms | NLL+grad med={med(t_grad):8.2f} ms"
          f" | Hessian med={med(t_hess):8.2f} ms"
          f" | used(MiB) {used0} -> {used1}", flush=True)


def main():
    cfg = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "perf"
    base = Path(cfg).read_text()
    gen = TESTS / "_gen_perf"
    gen.mkdir(exist_ok=True)
    print(f"=== perf_probe {label}: {cfg} (n_nll={N_NLL}, n_hess={N_HESS}) ===", flush=True)
    for prec in ("float", "hybrid", "double"):
        tag = "auto" if prec == "hybrid" else prec
        p = gen / f"{Path(cfg).stem}_{tag}.yml"
        p.write_text(f"precision: {prec}\n{base}")
        try:
            bench(str(p), f"{label}_{tag}")
        except Exception as e:
            print(f"[PERF {label}_{tag}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
