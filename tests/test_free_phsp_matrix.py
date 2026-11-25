"""free_phsp 流式模式的干涉矩阵回归（旧版 W_rec 大 N 倍 / 加权矩阵错）。

旧实现构造 double phsp Gram 时没有乘 sqrt(w/W_total)：
  - W_rec = W_total × Σ|A|²（大 N 倍）；
  - 配置了 phsp_weights 时，interference_rec 变成**无权重**矩阵；
  - NLL 的 phsp 项也用未加权矩阵（与梯度/Hessian 的加权 B̄ 不一致）。

这里用驻留模式（free_phsp_off，逐事例 computePhspMeanSum + 分块 B̄）作独立参考，
对无权重与逐事例权重两种情况逐值比较。
"""

import numpy as np
import pytest
import torch
import uproot

import ctpwa
from conftest import CONFIGS_DIR, make_params

DEV = torch.device("cpu") if ctpwa.backend() == "cpu" else torch.device("cuda:0")
# 原生 float .so 的 A 是 float32（复数 8B），流式 double 矩阵 vs 驻留 float 扫描
# 的差在 1e-7 相对量级；double .so 可以卡到 1e-9~1e-10。旧 bug 的偏差是 N 倍/O(1)。
IS_FLOAT_BUILD = int(ctpwa.build_info()["complex_bytes"]) == 8
W_RTOL = 1e-5 if IS_FLOAT_BUILD else 1e-9
M_RTOL = 1e-5 if IS_FLOAT_BUILD else 1e-8
NLL_RTOL = 1e-5 if IS_FLOAT_BUILD else 1e-7


def _read(path):
    with uproot.open(str(path)) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        meta = f["integrals"].arrays(library="np")
        out = {"N_rec": float(meta["N_rec"][0]),
               "W_rec": float(meta["W_rec"][0])}
        o = f["interference_rec"]
        n = int(o.member("fNrows"))
        out["rec"] = np.asarray(o.member("fElements"), dtype=float).reshape(n, n)
    return out


def _weighted_configs(tmp_path):
    """复制 free_phsp(_off) 配置，加一串非平凡 phsp_weights。"""
    wpath = tmp_path / "phsp_w.dat"
    rng = np.random.default_rng(7)
    w = 0.2 + rng.random(10000) * 2.0
    np.savetxt(wpath, w)
    paths = {}
    for key in ("free_phsp", "free_phsp_off"):
        src = CONFIGS_DIR / f"{key}.yml"
        text = src.read_text(encoding="utf-8")
        needle = '  phsp: [dat, "./data/test_phsp.dat"]'
        assert needle in text
        text = text.replace(
            needle, needle + f'\n  phsp_weights: [dat, "{wpath}"]')
        dst = tmp_path / f"{key}_w.yml"
        dst.write_text(text, encoding="utf-8")
        paths[key] = str(dst)
    return paths


def _run(ana, params, path):
    ana.writeResult(params, str(path), 0)
    return _read(path)


def test_free_phsp_matrix_matches_resident(tmp_path, make_analysis):
    free = make_analysis("free_phsp", reuse=False)
    base = make_analysis("free_phsp_off", reuse=False)
    p = make_params(base, DEV)
    Rf = _run(free, p, tmp_path / "free.root")
    Rb = _run(base, p, tmp_path / "base.root")

    assert Rb["N_rec"] == Rf["N_rec"]
    np.testing.assert_allclose(Rf["W_rec"], Rb["W_rec"], rtol=W_RTOL)
    np.testing.assert_allclose(Rf["rec"], Rb["rec"], rtol=M_RTOL, atol=1e-12)
    # 流式 double 矩阵 vs 驻留 float2-A 逐事例扫描：NLL 差在 float32 舍入量级
    #（~1e-9 相对），不是 1e-6 绝对。旧 bug 的偏差是 O(10)。
    assert abs(float(free.getNLL(p)) - float(base.getNLL(p))) \
        <= NLL_RTOL * max(1.0, abs(float(base.getNLL(p))))


def test_free_phsp_weighted_matrix_matches_resident(tmp_path):
    paths = _weighted_configs(tmp_path)
    free = ctpwa.analysis(paths["free_phsp"])
    base = ctpwa.analysis(paths["free_phsp_off"])
    p = make_params(base, DEV)
    Rf = _run(free, p, tmp_path / "free_w.root")
    Rb = _run(base, p, tmp_path / "base_w.root")

    assert Rb["N_rec"] == Rf["N_rec"]
    np.testing.assert_allclose(Rf["W_rec"], Rb["W_rec"], rtol=W_RTOL)
    np.testing.assert_allclose(Rf["rec"], Rb["rec"], rtol=M_RTOL, atol=1e-11)
    assert abs(float(free.getNLL(p)) - float(base.getNLL(p))) \
        <= NLL_RTOL * max(1.0, abs(float(base.getNLL(p))))
