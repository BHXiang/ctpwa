"""writeResult 的 data 直方图必须使用 data_weights（无权重时按 1）。

回归背景: 拟合的 NLL/梯度/Hessian 一直使用 data 权重，但 writeResult 的
mass/cosbeta/dalitz/Plot hdata 曾固定传 nullptr，导致加权数据的图形仍是
原始计数，与按 Σw 归一的 hfit/hbkg 不在同一能标。
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import uproot

from conftest import CONFIGS_DIR, make_params

TESTS_DIR = Path(__file__).resolve().parent
DATA_FILE = TESTS_DIR / "data" / "test_data.dat"
N_DATA = 1000
MASS_BINS = 100
MASS_RANGE = (1.0, 2.6)


def _make_weights(path):
    """非均匀权重，总和与事件数不同，能同时发现漏权重和下标的错位。"""
    weights = 1.0 + 0.5 * (np.arange(N_DATA) % 4)
    np.savetxt(path, weights, fmt="%.17g")
    return weights


def _make_config(base_name, tmp_path, weight_file, extra=""):
    text = (CONFIGS_DIR / base_name).read_text(encoding="utf-8")
    # 测试进程 cwd 是 tests/；仍用绝对路径，避免临时 config 的目录影响解析。
    text = text.replace('"./data/', f'"{TESTS_DIR / "data"}/')
    marker = "  bkg:"
    assert marker in text
    text = text.replace(
        marker,
        f'  data_weights: [dat, "{weight_file}"]\n{marker}',
        1,
    )
    text += extra
    cfg = tmp_path / f"{base_name.replace('.yml', '')}_dataw.yml"
    cfg.write_text(text, encoding="utf-8")
    return cfg


def _expected_mass_hist(weights):
    events = np.loadtxt(DATA_FILE).reshape(N_DATA, 3, 4)
    kp, km = events[:, 0], events[:, 1]
    energy = kp[:, 0] + km[:, 0]
    momentum2 = ((kp[:, 1:] + km[:, 1:]) ** 2).sum(axis=1)
    mass = np.sqrt(np.maximum(energy * energy - momentum2, 0.0))
    return np.histogram(
        mass, bins=MASS_BINS, range=MASS_RANGE, weights=weights
    )[0]


LEGACY_PLOT = """
Plot:
  mass:
    - input: [[Kp, Km]]
      bins: 100
      range: [0.0, 4.0]
      display: ["M(Kp,Km)", "Events"]
  cosbeta:
    - input: [[Jpsi], [Kp, Km], [Kp]]
      bins: 50
      range: [-1.0, 1.0]
      display: ["cos", "Events"]
  dalitz:
    - input: [[Kp, Km], [Kp, eta]]
      bins: [20, 20]
      range: [[0.0, 20.0], [0.0, 20.0]]
      display: ["M2", "M2"]
"""


def _run_write_result(config, tmp_path):
    import ctpwa

    ana = ctpwa.analysis(str(config))
    dev = torch.device("cuda:0" if ctpwa.backend() == "cuda" else "cpu")
    params = make_params(ana, dev)
    out = tmp_path / "weighted_data.root"
    ana.writeResult(params, str(out), 0)
    return out


def test_plot_obs_hdata_uses_data_weights(tmp_path):
    weights = _make_weights(tmp_path / "data_weights.dat")
    cfg = _make_config("plot_obs.yml", tmp_path, tmp_path / "data_weights.dat")
    out = _run_write_result(cfg, tmp_path)

    with uproot.open(out) as f:
        vals = f["m_KK/hdata"].values()
        expected = _expected_mass_hist(weights)
        np.testing.assert_allclose(vals, expected, rtol=0, atol=1e-9)

        # 角度轴全覆盖，总计数应严格等于 Σw；漏权重时会得到 1000。
        cos_total = f["cosbeta_KK/hdata"].values().sum()
        assert cos_total == pytest.approx(weights.sum(), rel=1e-10)


def test_legacy_hdata_uses_data_weights(tmp_path):
    weights = _make_weights(tmp_path / "data_weights.dat")
    cfg = _make_config(
        "simple.yml", tmp_path, tmp_path / "data_weights.dat", LEGACY_PLOT
    )
    out = _run_write_result(cfg, tmp_path)

    with uproot.open(out) as f:
        data_keys = [
            "mass0_Kp_Km/hdata",
            "cosbeta0_Jpsi_KpKm_Kp/hdata",
            "dalitz0_KpKm_Kpeta/hdata",
        ]
        # mass/cosbeta/dalitz 的 range 都覆盖全相空间，三个谱都应等于 Σw。
        for key in data_keys:
            total = f[key].values().sum()
            assert total == pytest.approx(weights.sum(), rel=1e-10), key
