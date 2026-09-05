"""双卡 FF/EFF/writeResult 定向复现（三档）——判断上转消费路径双卡 bug。

用法: cd tests && python3 ff_eff_probe.py <config_with_phsp_truth> [label]
逐档依次调 getFitFractions/getEfficiency/writeResult 并同步，
任一步失败立即打印档位+阶段（不继续），便于归因。
"""
import os
import sys
from pathlib import Path

import torch

TESTS = Path("/home/whitewash/pkgs/ctpwa/tests")
if not TESTS.exists():
    TESTS = Path("/hpcfs/bes/gpupwa/bhxiang/pwatest/ctpwa/tests")
sys.path.insert(0, str(TESTS))

import ctpwa  # noqa: E402
from conftest import make_params  # noqa: E402


def main():
    cfg = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "ffeff"
    base = Path(cfg).read_text()
    gen = TESTS / "_gen_ffeff"
    gen.mkdir(exist_ok=True)
    preclist = (os.environ.get("FF_PRECS") or "float,hybrid,double").split(",")
    print(f"=== ff_eff_probe {label}: {cfg} visible={torch.cuda.device_count()} "
          f"precs={preclist} ===", flush=True)
    for prec in preclist:
        p = gen / f"{Path(cfg).stem}_{prec}.yml"
        p.write_text(f"precision: {prec}\n{base}")
        print(f"--- {prec} ---", flush=True)
        try:
            ana = ctpwa.analysis(str(p))
            torch.cuda.synchronize()
            print(f"[{prec}] ctor OK", flush=True)
            pv = make_params(ana, torch.device("cuda:0"))
            n = ana.getNVector()
            v = torch.complex(pv[:n], pv[n:2 * n]).contiguous()
            ff = ana.getFitFractions(v)
            torch.cuda.synchronize()
            print(f"[{prec}] getFitFractions OK shape={tuple(ff.shape)}", flush=True)
            eff = ana.getEfficiency(v)
            torch.cuda.synchronize()
            print(f"[{prec}] getEfficiency OK shape={tuple(eff.shape)}", flush=True)
            out = gen / f"wr_{prec}.root"
            ana.writeResult(pv.detach(), str(out))
            torch.cuda.synchronize()
            print(f"[{prec}] writeResult OK {out.name}", flush=True)
        except Exception as e:
            print(f"[{prec}] FAILED at stage above: {type(e).__name__}: {e}", flush=True)
            return 1
    print("=== ff_eff_probe ALL OK ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
