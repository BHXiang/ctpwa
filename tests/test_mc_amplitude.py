"""DIYMC P2/P3 tests: arbitrary-momentum intensity + amplitude-weighted MC.

These tests exercise the GPU/CPU backend `analysis` object and therefore need a
configured backend.  They are skipped on a pure-CPU CI unless explicitly run
with CTPWA_DEVICE=cpu (the CPU backend is numerically compatible but slower).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import ctpwa
from conftest import CONFIGS_DIR, make_params

pytestmark = pytest.mark.skipif(
    ctpwa.backend() != "cuda",
    reason="amplitude/MC tests are GPU-first; run with CTPWA_DEVICE=cuda",
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def _nominal_params(ana, device):
    return make_params(ana, device)


def _read_phsp(n=None):
    raw = np.loadtxt(DATA_DIR / "test_phsp.dat")
    ev = torch.from_numpy(raw.reshape(-1, 3, 4)).double()
    return ev if n is None else ev[:n]


def _ratio_spread(a, b):
    r = a / b
    return float(np.ptp(r) / np.mean(r))


def test_get_intensity_matches_write_result(make_analysis, device, tmp_path):
    """getIntensity / writeResult(totalweight) 逐事件应为常数比。"""
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    mom = _read_phsp()

    I = ana.getIntensity(mom, params)
    assert I.shape == (mom.size(0),)
    assert torch.isfinite(I).all()
    assert I.min() > 0

    out = tmp_path / "wr.root"
    ana.writeResult(params, str(out), 1)

    uproot = pytest.importorskip("uproot")
    with uproot.open(str(out)) as f:
        tree = f["saved_weight"]
        tw = np.asarray(tree["totalweight"].array(), dtype=np.float64)
    assert tw.shape == (mom.size(0),)
    assert _ratio_spread(I.numpy(), tw) < 1e-8


def test_get_intensity_theta_dependence(make_analysis, device, tmp_path):
    """theta 改变后，getIntensity 仍与 writeResult 的 totalweight 同比例。"""
    ana = make_analysis("simple")
    params = _nominal_params(ana, device).clone()
    nv = ana.getNVector()
    nt = ana.getNFreeTheta()
    if nt:
        bounds = ana.getFreeResParams()  # [3, n]
        lower = bounds[1].to(device=device, dtype=torch.float64)
        upper = bounds[2].to(device=device, dtype=torch.float64)
        params[2 * nv :] = torch.clamp(
            params[2 * nv :] + 0.05 * (upper - lower), lower, upper
        )

    mom = _read_phsp(2000)
    I = ana.getIntensity(mom, params)

    out = tmp_path / "wr_theta.root"
    ana.writeResult(params, str(out), 1)

    uproot = pytest.importorskip("uproot")
    with uproot.open(str(out)) as f:
        tw = np.asarray(
            f["saved_weight"]["totalweight"].array(), dtype=np.float64
        )[: mom.size(0)]
    assert _ratio_spread(I.numpy(), tw) < 1e-8


def test_get_intensity_isolated(make_analysis, device):
    """getIntensity 用局部 AmpCalc，不应改变 NLL，也不应泄漏显存。"""
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    mom = _read_phsp(3000)

    nll0 = float(ana.getNLL(params))
    free0 = torch.cuda.mem_get_info()[0]
    for _ in range(4):
        I = ana.getIntensity(mom, params)
        assert torch.isfinite(I).all()
    free1 = torch.cuda.mem_get_info()[0]
    nll1 = float(ana.getNLL(params))

    assert nll1 == pytest.approx(nll0, rel=0, abs=1e-9)
    # 4 次调用的临时表应全部释放；允许 allocator 的少量缓存波动
    assert free0 - free1 < 128 * 1024 * 1024


def test_generate_mc_unweighted(make_analysis, device):
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    n = 2000
    res = ana.generateMC(n, params, seed=42, mode="unweighted", batch=10000)

    ev = res["events"]
    w = res["weights"]
    st = res["stats"]
    assert ev.shape == (n, 3, 4)
    assert ev.dtype == torch.float64 and ev.device.type == "cpu"
    assert w.numel() == 0
    assert st["truncated"] is False
    assert st["n_accepted"] == n
    assert st["acceptance"] > 0
    assert st["max_weight"] > 0

    # 四动量守恒 + 质量壳
    E = ev[..., 0]
    p = ev[..., 1:]
    torch.testing.assert_close(
        E.sum(dim=1), torch.full_like(E.sum(dim=1), 3.0969), atol=1e-9, rtol=0
    )
    assert p.sum(dim=1).abs().max().item() < 1e-9
    masses = torch.tensor([0.4937, 0.4937, 0.5478], dtype=torch.float64)
    m2 = E**2 - (p**2).sum(dim=-1)
    torch.testing.assert_close(m2, (masses**2).expand_as(m2), atol=1e-8, rtol=0)

    # 接受的事例必须满足 u*Imax < I（这里只检查 I <= Imax 的必要条件）
    I = ana.getIntensity(ev, params)
    assert I.max().item() <= st["max_weight"] * (1.0 + 1e-9)


def test_generate_mc_weighted(make_analysis, device):
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    n = 3000
    res = ana.generateMC(n, params, seed=7, mode="weighted", batch=2000)
    ev, w, st = res["events"], res["weights"], res["stats"]
    assert ev.shape == (n, 3, 4)
    assert w.shape == (n,)
    assert torch.isfinite(w).all() and w.min() > 0
    assert st["n_flat"] == n
    assert st["acceptance"] == pytest.approx(1.0)
    assert st["max_weight"] == pytest.approx(float(w.max()), rel=1e-12)


def test_generate_mc_seed_reproducible(make_analysis, device):
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    a = ana.generateMC(300, params, seed=11, mode="unweighted", batch=3000)
    b = ana.generateMC(300, params, seed=11, mode="unweighted", batch=3000)
    assert torch.equal(a["events"], b["events"])
    assert a["stats"]["max_weight"] == b["stats"]["max_weight"]


def test_generate_mc_max_trials(make_analysis, device):
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    res = ana.generateMC(
        10_000, params, seed=3, mode="unweighted", batch=100, max_trials=200
    )
    assert res["stats"]["truncated"] is True
    assert res["events"].size(0) < 10_000
    assert res["stats"]["n_flat"] <= 200


def test_generate_mc_distribution_vs_weighted(make_analysis, device):
    """AR 样本的 m^2(K+K-) 分布应与 |M|^2 加权平坦样本一致（统计精度内）。"""
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    ar = ana.generateMC(5000, params, seed=123, mode="unweighted", batch=20000)
    wt = ana.generateMC(40000, params, seed=124, mode="weighted", batch=20000)

    def m2_pair(ev, i, j):
        s = ev[:, i, :] + ev[:, j, :]
        return (
            s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2 - s[:, 3] ** 2
        ).numpy()

    x = m2_pair(ar["events"], 0, 1)
    xw = m2_pair(wt["events"], 0, 1)
    ww = wt["weights"].numpy()
    bins = np.linspace(min(x.min(), xw.min()), max(x.max(), xw.max()), 25)
    h, _ = np.histogram(x, bins=bins)
    hw, _ = np.histogram(xw, bins=bins, weights=ww)
    h = h / h.sum()
    hw = hw / hw.sum()
    assert np.abs(h - hw).max() < 0.03


def test_write_mc_roundtrip(make_analysis, device, tmp_path):
    ana = make_analysis("simple")
    params = _nominal_params(ana, device)
    res = ana.generateMC(200, params, seed=5, mode="unweighted", batch=2000)
    dat = tmp_path / "mc.dat"
    root = tmp_path / "mc.root"
    ana.writeMC(res["events"], str(dat))
    ana.writeMC(res["events"], str(root))

    raw = np.loadtxt(dat).reshape(-1, 3, 4)
    np.testing.assert_allclose(
        raw, res["events"].numpy(), atol=1e-15, rtol=0
    )

    uproot = pytest.importorskip("uproot")
    with uproot.open(str(root)) as f:
        tree = f["mc"]
        assert tree.num_entries == 200
        assert {"Kp", "Km", "eta"} <= set(tree.keys())
        for a, name in enumerate(["Kp", "Km", "eta"]):
            arr = tree[name].array()
            back = np.stack(
                [
                    np.asarray(arr["fE"], dtype=np.float64),
                    np.asarray(arr["fP"]["fX"], dtype=np.float64),
                    np.asarray(arr["fP"]["fY"], dtype=np.float64),
                    np.asarray(arr["fP"]["fZ"], dtype=np.float64),
                ],
                axis=1,
            )
            np.testing.assert_allclose(
                back, res["events"][:, a, :].numpy(), atol=1e-12, rtol=0
            )
