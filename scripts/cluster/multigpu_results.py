#!/usr/bin/env python3
"""1 卡 vs N 卡：truth/rec 干涉矩阵、误差与 streaming Gram 一致性。

multigpu_parity.py 只覆盖 NLL/梯度；本脚本覆盖 writeResult 的派生量：
  * eff2_weighted      非均匀 phsp/phsp_truth 权重，rec/gen + err + N/W；
  * free_phsp_weighted 流式 double Gram 的加权 rec + err + N/W；
  * small_truth        单事件 truth + CTPWA_TRUTH_BATCH=1，强制 n_batch<n_gpus；
  * auto_hessian       不传 H，覆盖 writeResult 内部 getHessian 的多卡路径。

用法（集群 GPU 节点，仓库根目录）：
  python3 scripts/cluster/multigpu_results.py
  MGR_FORCE=1 python3 scripts/cluster/multigpu_results.py  # 单卡自检脚本自身

退出码：0 = 通过/跳过；1 = 子进程失败或数值超差。
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
sys.path.insert(0, str(TESTS))
DEFAULT_SCENARIOS = ["eff2_weighted", "free_phsp_weighted", "small_truth", "auto_hessian"]


def _make_config(scenario, directory):
    directory = Path(directory)
    is_free = scenario.startswith("free_phsp")
    base = "free_phsp" if is_free else "eff2"
    text = (TESTS / "configs" / f"{base}.yml").read_text()
    if scenario.endswith("_weighted"):
        weights = directory / "weights.dat"
        weights.write_text("\n".join(
            f"{0.2 + ((i * 37) % 101) / 50.0:.17g}" for i in range(10000)) + "\n")
        text = text.replace(
            '  phsp: [dat, "./data/test_phsp.dat"]',
            f'  phsp: [dat, "./data/test_phsp.dat"]\n'
            f'  phsp_weights: [dat, "{weights}"]')
        if not is_free:
            text = text.replace(
                '  phsp_truth: [dat, "./data/test_phsp.dat"]',
                f'  phsp_truth: [dat, "./data/test_phsp.dat"]\n'
                f'  phsp_truth_weights: [dat, "{weights}"]')
    if scenario == "small_truth":
        lines = (TESTS / "data" / "test_phsp.dat").read_text().splitlines()
        one = directory / "one.dat"
        one.write_text("\n".join(lines[:3]) + "\n")
        text = text.replace(
            '  phsp_truth: [dat, "./data/test_phsp.dat"]',
            f'  phsp_truth: [dat, "{one}"]')
    cfg = directory / f"{scenario}.yml"
    cfg.write_text(text)
    return cfg


def _root_result(path):
    import numpy as np
    import uproot

    out = {}
    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        for tag in ("rec", "gen", "rec_err", "gen_err"):
            key = f"interference_{tag}"
            if key not in keys:
                continue
            obj = f[key]
            n = int(obj.member("fNrows"))
            out[tag] = np.asarray(obj.member("fElements"), dtype=float).reshape(n, n).tolist()
        meta = f["integrals"].arrays(library="np")
        for key in ("N_rec", "W_rec", "N_gen", "W_gen", "npartials"):
            out[key] = float(meta[key][0])
    return out


def _evaluate(scenario):
    import torch
    import ctpwa
    from conftest import make_params

    os.chdir(TESTS)
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as td:
        cfg = _make_config(scenario, td)
        ana = ctpwa.analysis(str(cfg))
        dev = torch.device("cuda:0")
        p = make_params(ana, dev)
        n, ntheta = ana.getNVector(), ana.getNFreeTheta()
        H = torch.eye(p.numel(), dtype=torch.float64, device=dev) * 100.0
        result = {"nll": float(ana.getNLL(p))}
        if scenario == "small_truth":
            dt = torch.float32 if int(ctpwa.build_info()["complex_bytes"]) == 8 else torch.float64
            cpl = torch.complex(p[:n].to(dt), p[n:2 * n].to(dt)).contiguous().to(dev)
            os.environ["CTPWA_TRUTH_BATCH"] = "1"
            try:
                ff = ana.getFitFractions(cpl, H)
            finally:
                os.environ.pop("CTPWA_TRUTH_BATCH", None)
            result["ff"] = ff.detach().cpu().tolist()
        else:
            root = Path(td) / "result.root"
            if scenario == "auto_hessian":
                ana.writeResult(p, str(root), 0)
            else:
                ana.writeResult(p, str(root), 0, [], H)
            result.update(_root_result(root))
        if ntheta:
            theta = ana.getFreeResParams()[0].double().to(dev)
            ana.reCalcAmp(theta * 1.03)
            if torch.cuda.current_device() != 0:
                raise RuntimeError("reCalcAmp 后 current device 不在主卡")
            ana.reCalcAmp(theta)
        result["current_device"] = torch.cuda.current_device()
        # 故障后还能继续走原驻留布局，顺带检查 offsets 已恢复。
        result["nll_after"] = float(ana.getNLL(p))
        return result


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), value


def _compare(reference, candidate, rtol, atol):
    import numpy as np

    ref = dict(_flatten(reference))
    got = dict(_flatten(candidate))
    if ref.keys() != got.keys():
        return False, f"键不同: 1卡={sorted(ref)} N卡={sorted(got)}"
    worst = 0.0
    bad = []
    for key in ref:
        a = np.asarray(got[key], dtype=float)
        b = np.asarray(ref[key], dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
            bad.append(key)
        if a.shape == b.shape and a.size:
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0))))
    return not bad, f"worst={worst:.2e}" + (f" bad={bad}" if bad else "")


def _child(scenario):
    print("MGR_JSON=" + json.dumps(_evaluate(scenario), separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--rtol", type=float, default=2e-7)
    ap.add_argument("--atol", type=float, default=2e-9)
    args = ap.parse_args()
    if args.child:
        _child(args.child)
        return 0

    import torch
    if not torch.cuda.is_available():
        print("[mgr] 无 CUDA：跳过")
        return 0
    n_gpu = torch.cuda.device_count()
    print(f"[mgr] 可见 GPU 数 = {n_gpu}")
    if n_gpu < 2 and not os.environ.get("MGR_FORCE"):
        print("[mgr] <2 张卡：跳过；MGR_FORCE=1 可做 1卡/1卡 自检")
        return 0

    env1 = dict(os.environ)
    env1["CUDA_VISIBLE_DEVICES"] = "0"
    env1["CTPWA_DEVICE"] = "cuda"
    failed = []
    for scenario in args.scenarios:
        proc = subprocess.run([sys.executable, __file__, "--child", scenario],
                              cwd=str(ROOT), env=env1, capture_output=True,
                              text=True, timeout=3600)
        if proc.returncode != 0:
            print(f"[mgr] {scenario}: 单卡子进程失败\n{proc.stderr[-2000:]}")
            failed.append(scenario)
            continue
        marker = [line for line in proc.stdout.splitlines() if line.startswith("MGR_JSON=")]
        if len(marker) != 1:
            print(f"[mgr] {scenario}: 子进程输出不完整")
            failed.append(scenario)
            continue
        reference = json.loads(marker[0][len("MGR_JSON="):])
        try:
            candidate = _evaluate(scenario)
        except Exception as exc:
            print(f"[mgr] {scenario}: N卡评估失败: {exc}")
            failed.append(scenario)
            continue
        ok, detail = _compare(reference, candidate, args.rtol, args.atol)
        print(f"[mgr] {scenario:20s} {'OK' if ok else 'FAIL'} {detail}")
        if not ok:
            failed.append(scenario)
    if failed:
        print(f"[mgr] 失败: {failed}")
        return 1
    print("[mgr] 多卡派生量一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
