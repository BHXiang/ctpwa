"""example/read_interference.py 的逐波效率公式回归。

矩阵写进 ROOT 时已归一化（P_ii/W_rec、T_ii/W_gen）。旧脚本用
(dr/N_rec)/(di/N_gen) 直接当效率，漏了 W_rec/W_gen 因子；当 phsp 与
phsp_truth 样本量/效率不同时结果会错一个 ε̄ 因子。
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import uproot

import ctpwa
from conftest import CONFIGS_DIR, make_params

REPO = Path(__file__).resolve().parent.parent
DEV = torch.device("cpu") if ctpwa.backend() == "cpu" else torch.device("cuda:0")


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "read_interference_example", REPO / "example" / "read_interference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with uproot.open(str(path)) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        assert {"interference_rec", "interference_gen", "integrals"} <= keys
        meta = f["integrals"].arrays(library="np")
        meta = {k: (float(v[0]) if k != "npartials" else int(v[0]))
                for k, v in meta.items()}
        mats = {}
        for tag, name in (("rec", "interference_rec"), ("gen", "interference_gen")):
            o = f[name]
            n = int(o.member("fNrows"))
            mats[tag] = np.asarray(o.member("fElements"), dtype=float).reshape(n, n)
    return mats, meta


def test_per_wave_efficiency_matches_get_efficiency(tmp_path, make_analysis):
    # phsp_truth 取 phsp 的前一半事件：分布相同但 Σw 不同 → ε̄≠1，旧公式必错。
    raw = (CONFIGS_DIR.parent / "data" / "test_phsp.dat").read_text().splitlines()
    assert len(raw) % 3 == 0
    truth = tmp_path / "truth_half.dat"
    truth.write_text("\n".join(raw[: len(raw) // 2]) + "\n")

    src = (CONFIGS_DIR / "eff2.yml").read_text()
    src = src.replace(
        '  phsp_truth: [dat, "./data/test_phsp.dat"]',
        f'  phsp_truth: [dat, "{truth}"]')
    cfg = tmp_path / "eff2_half_truth.yml"
    cfg.write_text(src)
    ana = ctpwa.analysis(str(cfg))

    p = make_params(ana, DEV)
    n = ana.getNVector()
    cdt = torch.float64 if int(ctpwa.build_info()["complex_bytes"]) == 16 else torch.float32
    cpl = torch.complex(p[:n].to(cdt), p[n:2 * n].to(cdt)).to(DEV)
    eff = ana.getEfficiency(cpl)
    assert eff.numel() > 0
    center = eff[:, 0].cpu().numpy()

    out = tmp_path / "w.root"
    ana.writeResult(p, str(out), 0)
    mats, meta = _read(out)
    eps = _load_example().per_wave_efficiency(mats, meta)
    assert eps is not None
    # 旧公式 = ε_i·W_gen/W_rec。double 走严格阈值；原生 float 的
    # 两条积分路径累加顺序不同，允许 float32 振幅带来的 1e-7 级差别。
    native_float = int(ctpwa.build_info()["complex_bytes"]) == 8
    np.testing.assert_allclose(eps, center,
                               rtol=1e-5 if native_float else 1e-8,
                               atol=1e-8 if native_float else 1e-10)
    # 再确认这个配置真的能区分旧公式（ε̄ 明显不等于 1）。
    eps_bar = meta["W_rec"] / meta["W_gen"]
    assert abs(eps_bar - 1.0) > 0.1, eps_bar
