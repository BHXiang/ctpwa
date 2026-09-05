"""getHessian 显存实测探针：逐阶段记录每卡空闲显存 + 打印事件相关理论账本。

用法 (集群, cwd = tests/):
    python3 mem_probe.py <config.yml> [label]
    # 可选环境: MEM_EVENTS=phsp,data,bkg 覆盖自动解析; MEM_NA=.. MEM_NPOL=.. MEM_NFREE=..

流程: 记录初始空闲 → analysis 构造 → make_params(cuda:0) → reCalcAmp →
      getNLL ×2 → getHessian，每步后打印每卡 free MiB 与增量。
修复前/后各跑一次对比峰值（free 最小处 = 峰值占用）。

理论账本按当前 hybrid(double .so + A=float2) 口径：
  A float2 8B; w/S/dF/T 等 ctComplex 16B; slamp thrust::complex<double> 16B;
  d_dF 常驻 nEv×nSL×nFree×16B; 窗口化后 vv 上转临时 = 50k×nPol×nAmp×16B。
"""
import ctypes
import os
import re
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.getcwd())                       # 本地/集群 cwd=tests 时
sys.path.insert(0, "/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa/tests")
from conftest import make_params  # noqa: E402

import ctpwa  # noqa: E402

_cudart = ctypes.CDLL("libcudart.so")


def mem_free_all():
    nd = torch.cuda.device_count()
    out = []
    cur = ctypes.c_int(-1)
    _cudart.cudaGetDevice(ctypes.byref(cur))
    for d in range(nd):
        _cudart.cudaSetDevice(d)
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        _cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        out.append(free.value / 1048576.0)
    _cudart.cudaSetDevice(cur.value)
    return out


def parse_config(cfg_path):
    txt = open(cfg_path, encoding="utf-8").read()
    order = re.findall(r"order:\s*\[([^\]]+)\]", txt)
    n_order = len([x for x in order[0].split(",")]) if order else 3
    files = re.findall(r'(["\'])([^"\']*\.dat)\1', txt)
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    paths = []
    for _, p in files:
        cand = ([p] if os.path.isabs(p)
                else [os.path.join(os.getcwd(), p), os.path.join(cfg_dir, p)])
        hit = next((c for c in cand if os.path.exists(c)), None)
        if hit is None:
            print(f"[mem_probe] 警告: 找不到数据文件 {p}（将用 MEM_EVENTS 覆盖）", flush=True)
        paths.append(hit if hit else p)
    events = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        nlines = sum(1 for _ in open(p))
        base = os.path.basename(p)
        if "phsp" in base:
            kind = "phsp"
        elif "data" in base:
            kind = "data"
        elif "bkg" in base or "sideband" in base:
            kind = "bkg"
        else:
            kind = ["phsp", "data", "bkg"][min(len(events), 2)]  # 按出现顺序兜底
        events[kind] = max(1, nlines // n_order)
    print(f"[mem_probe] 解析: 末态数={n_order} 文件={[os.path.basename(p) for p in paths]} "
          f"事件={events}", flush=True)
    return n_order, events, paths


def theory_account(n_ev, na, npol, nfree, nsl=1, nsigma=1, n_gpu=2):
    """per-GPU 理论账本 (MiB)。n_ev = 单卡事件(phsp+data+bkg 总量);
    nsl = 全部链组合的 SL 组合总数（MEM_NSL，默认 1 仅示意，真实值见 config 分波）。"""
    M = 1048576.0
    rows = []
    def add(nm, b): rows.append((nm, b / M))
    add("A 驻留 float2 (8B)", n_ev * npol * na * 8)
    add("slamp (16B 固定)", nsigma * nsl * npol * n_ev * 16)
    add("w data/bkg (16B)", n_ev * npol * 16 * 2)
    add("d_dF 常驻 (16B)", n_ev * nsl * nfree * 16)
    add("T (16B)", n_ev * npol * 16)
    win = 50000
    add(f"vv 上转窗口 ≤{win} 事件 (16B)", min(win, n_ev) * npol * na * 16)
    return rows


def stage(tag, fn, prev, label):
    t0 = time.time()
    try:
        fn()
        torch.cuda.synchronize()
        free = mem_free_all()
        dt = time.time() - t0
        used = [(f0 - f) for f0, f in zip(prev, free)]
        print(f"[MEM {label}] {tag}: free(GiB)={[f'{f/1024:.2f}' for f in free]} "
              f"Δused(MiB)={[f'{u:.1f}' for u in used]} ({dt:.1f}s)", flush=True)
        return free
    except Exception as e:
        print(f"[MEM {label}] {tag} FAILED: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)


def main():
    cfg = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "mem"
    n_order, ev_auto, paths = parse_config(cfg)
    print(f"=== mem_probe {label}: {cfg} visible={torch.cuda.device_count()} ===", flush=True)
    print(f"自动解析: 末态数={n_order} 事件={ev_auto}", flush=True)

    env_ev = os.environ.get("MEM_EVENTS")
    if env_ev:
        parts = env_ev.split(",")
        ev = {"phsp": int(parts[0]), "data": int(parts[1]),
              "bkg": int(parts[2]) if len(parts) > 2 else 0}
    else:
        ev = ev_auto
    na = int(os.environ.get("MEM_NA", "0"))
    npol = int(os.environ.get("MEM_NPOL", "3"))
    nfree = int(os.environ.get("MEM_NFREE", "0"))
    nsl = int(os.environ.get("MEM_NSL", "1"))

    prev = mem_free_all()
    print(f"[MEM {label}] 初始 free(GiB)={[f'{f/1024:.2f}' for f in prev]}", flush=True)

    def nvsmi_used():
        """nvidia-smi memory.used (MiB/卡)；不可用时返回 None"""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            return [int(x) for x in out.stdout.split()]
        except Exception:
            return None

    def do(tag, fn):
        nonlocal prev
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        cur = mem_free_all()
        used = [(f0 - f) for f0, f in zip(prev, cur)]
        smi = nvsmi_used()
        free_s = ",".join(f"{f/1024:.2f}" for f in cur)
        used_s = ",".join(f"{u:.1f}" for u in used)
        extra = f"  nvidia-smi used(MiB)={smi}" if smi else ""
        print(f"[MEM {label}] {tag}: free(GiB)=[{free_s}] "
              f"Δused(MiB)=[{used_s}] ({time.time()-t0:.1f}s){extra}", flush=True)
        prev = cur

    ana = ctpwa.analysis(cfg)
    do("1.ctor", lambda: None)
    if na == 0:
        try:
            na = len(ana.getAmplitudeNames())   # = n_amplitudes_（分波数）
        except Exception:
            na = ana.getNVector()
    if nfree == 0:
        nfree = ana.getNFreeTheta()
    p = make_params(ana, torch.device("cuda:0"))
    do("1b.params", lambda: None)

    nv = ana.getNVector()
    do("2.reCalcAmp", lambda: ana.reCalcAmp(p[2 * nv:].detach().clone()))
    p3 = p.clone().requires_grad_(True)
    do("3.getNLL", lambda: float(ana.getNLL(p3)))
    p4 = (p + 1e-4 * torch.randn_like(p)).clone().requires_grad_(True)
    do("4.getNLL+grad", lambda: torch.autograd.grad(ana.getNLL(p4), p4)[0])
    if os.environ.get("MEM_NO_HESSIAN") != "1":
        do("5.getHessian", lambda: ana.getHessian(p.detach()))
    else:
        print(f"[MEM {label}] 5.getHessian SKIPPED (MEM_NO_HESSIAN=1; Float 档 hessian 未接线)", flush=True)

    if na and nfree:
        tot = sum(ev.values())
        per = tot / max(1, torch.cuda.device_count())
        print("---- 理论账本 (每卡, MiB) ----", flush=True)
        rows = theory_account(per, na, npol, nfree, nsl=nsl)
        for nm, mb in rows:
            print(f"  {nm:<34s} {mb:9.1f}", flush=True)
        print(f"  [规模] nEv/卡={per:.0f} (共{tot}) na={na} nPol={npol} nFree={nfree} "
              f"nSL(MEM_NSL)={nsl}", flush=True)
    print(f"=== mem_probe {label} DONE ===", flush=True)


if __name__ == "__main__":
    main()
