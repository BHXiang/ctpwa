"""writeResult 的边界路径：中途抛异常后不能污染 analysis / 留下坏 ROOT 文件。"""

import numpy as np
import pytest
import torch
import uproot

import ctpwa
from conftest import CONFIGS_DIR, make_params

DEV = torch.device("cpu") if ctpwa.backend() == "cpu" else torch.device("cuda:0")


def _read(path):
    with uproot.open(str(path)) as f:
        return {k.split(";")[0] for k in f.keys()}


def test_bad_hessian_then_recover(tmp_path, make_analysis):
    ana = make_analysis("eff2", reuse=False)
    p = make_params(ana, DEV)
    n, nt = ana.getNVector(), ana.getNFreeTheta()
    nll0 = float(ana.getNLL(p))

    # 错误 dtype：writeResult 已经写了矩阵后才在 computeMatrixErrors 抛错。
    # root_guard 必须关闭文件，且 analysis 的 NLL/后续 writeResult 仍可用。
    bad = torch.eye(2 * n + nt, dtype=torch.float32, device=DEV)
    with __import__("pytest").raises(RuntimeError):
        ana.writeResult(p, str(tmp_path / "bad.root"), 0, [], bad)

    assert np.isfinite(float(ana.getNLL(p)))
    assert abs(float(ana.getNLL(p)) - nll0) < 1e-9

    good = torch.eye(2 * n + nt, dtype=torch.float64, device=DEV) * 1e2
    out = tmp_path / "good.root"
    ana.writeResult(p, str(out), 0, [], good)
    keys = _read(out)
    assert {"interference_rec", "interference_rec_err", "integrals"} <= keys
    assert np.isfinite(float(ana.getNLL(p)))


def test_wrong_hessian_shape_then_recover(tmp_path, make_analysis):
    ana = make_analysis("eff2", reuse=False)
    p = make_params(ana, DEV)
    n, nt = ana.getNVector(), ana.getNFreeTheta()
    bad = torch.eye(2 * n + nt - 1, dtype=torch.float64, device=DEV)
    with __import__("pytest").raises(RuntimeError):
        ana.writeResult(p, str(tmp_path / "badshape.root"), 0, [], bad)

    out = tmp_path / "ok.root"
    ana.writeResult(p, str(out), 0)
    assert "interference_rec" in _read(out)


def test_empty_truth_writes_rec_and_skips_gen(tmp_path):
    """空 truth 是可用 rec 结果，不应让整个 writeResult 失败。"""
    empty = tmp_path / "empty.dat"
    empty.write_text("")
    src = (CONFIGS_DIR / "eff2.yml").read_text().replace(
        '  phsp_truth: [dat, "./data/test_phsp.dat"]',
        f'  phsp_truth: [dat, "{empty}"]')
    cfg = tmp_path / "empty_truth.yml"
    cfg.write_text(src)
    ana = ctpwa.analysis(str(cfg))
    p = make_params(ana, DEV)
    H = torch.eye(p.numel(), dtype=torch.float64, device=DEV) * 100.0
    out = tmp_path / "empty_truth.root"
    ana.writeResult(p, str(out), 0, [], H)

    with uproot.open(out) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        meta = f["integrals"].arrays(library="np")
    assert {"interference_rec", "interference_rec_err", "integrals"} <= keys
    assert "interference_gen" not in keys and "interference_gen_err" not in keys
    assert float(meta["N_gen"][0]) == 0.0 and float(meta["W_gen"][0]) == 0.0
    assert torch.isfinite(ana.getNLL(p))


@pytest.mark.skipif(ctpwa.backend() != "cuda" or torch.cuda.device_count() < 2,
                    reason="需要至少两张 CUDA GPU")
def test_truth_batch_smaller_than_gpu_count(tmp_path, monkeypatch):
    """n_batch<n_gpus 时不得把空 map 交给 AmpGen。"""
    lines = (CONFIGS_DIR.parent / "data" / "test_phsp.dat").read_text().splitlines()
    one = tmp_path / "one.dat"
    one.write_text("\n".join(lines[:3]) + "\n")
    src = (CONFIGS_DIR / "eff2.yml").read_text().replace(
        '  phsp_truth: [dat, "./data/test_phsp.dat"]',
        f'  phsp_truth: [dat, "{one}"]')
    cfg = tmp_path / "one_truth.yml"
    cfg.write_text(src)
    monkeypatch.setenv("CTPWA_TRUTH_BATCH", "1")
    ana = ctpwa.analysis(str(cfg))
    p = make_params(ana, DEV)
    n = ana.getNVector()
    dt = torch.float32 if int(ctpwa.build_info()["complex_bytes"]) == 8 else torch.float64
    cpl = torch.complex(p[:n].to(dt), p[n:2 * n].to(dt)).contiguous().to(DEV)
    H = torch.eye(p.numel(), dtype=torch.float64, device=DEV) * 100.0
    ff = ana.getFitFractions(cpl, H)
    assert ff.shape == (2, 2) and torch.isfinite(ff).all()
    assert torch.isfinite(ana.getNLL(p))


@pytest.mark.skipif(ctpwa.backend() != "cuda" or torch.cuda.device_count() < 2,
                    reason="需要至少两张 CUDA GPU")
def test_recalc_amp_restores_primary_device(make_analysis):
    torch.cuda.set_device(0)
    ana = make_analysis("eff2", reuse=False)
    theta = ana.getFreeResParams()[0].double().to(DEV)
    try:
        ana.reCalcAmp(theta * 1.05)
        assert torch.cuda.current_device() == 0
    finally:
        ana.reCalcAmp(theta)
