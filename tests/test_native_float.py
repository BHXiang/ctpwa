"""原生 CTPWA_COMPLEX=float 构建的回归（double .so 上整体 skip）。

旧代码把"double .so + float2 A"的混合精度上转逻辑无条件套在原生 float .so 上：
A 已是 cuComplex/float2（8B），却被 castF2ToDouble2Kernel 按 cuDoubleComplex
（16B）写进按 8B 分配的缓冲 → 越界写；下游又按 float 读 double → 数值错。
本文件只会在 float .so 上跑；默认 double 构建下 skip（不是假绿：CI 若只编
double，需额外跑 `CTPWA_COMPLEX=float` 构建，见 scripts/run_tests_float.sh）。
"""

import numpy as np
import torch
import uproot

import ctpwa
from conftest import CONFIGS_DIR, make_params

INFO = ctpwa.build_info()
pytestmark = __import__("pytest").mark.skipif(
    int(INFO["complex_bytes"]) != 8,
    reason="需要原生 CTPWA_COMPLEX=float 构建（complex_bytes=8）")

DEV = torch.device("cuda:0") if ctpwa.backend() == "cuda" else torch.device("cpu")


def test_native_float_upcast_paths(tmp_path):
    # 双分波配置才能检验 Gram 的耦合收缩；单分波的整体尺度错误会相消。
    ana = ctpwa.analysis(str(CONFIGS_DIR / "eff2.yml"))
    p = make_params(ana, DEV)
    nll = float(ana.getNLL(p))
    assert np.isfinite(nll)

    # writeResult：驻留 A 的逐批 histogram 路径 + （若可算）Hessian 误差矩阵
    H = ana.getHessian(p)
    out = tmp_path / "w.root"
    ana.writeResult(p, str(out), 1, [], H)
    with uproot.open(out) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        o = f["interference_rec"]
        nr = int(o.member("fNrows"))
        rec = np.asarray(o.member("fElements"), dtype=float).reshape(nr, nr)
        w_rec = float(f["integrals"].arrays(library="np")["W_rec"][0])
    assert {"saved_weight", "interference_rec", "integrals"} <= keys
    _assert_rec_matches_resident(ana, p, rec, w_rec)

    # getPhspTensor / getDataTensor：float2→complex64 张量不得错读
    for t in (ana.getPhspTensor(), ana.getDataTensor()):
        assert t.dtype == torch.complex64
        assert torch.isfinite(t).all()

    # getIntensity：临时 AmpCalc 的 A 上转路径
    raw = np.loadtxt(CONFIGS_DIR.parent / "data" / "test_phsp.dat")
    mom = torch.from_numpy(raw.reshape(-1, 3, 4)).double()
    I = ana.getIntensity(mom[:200], p)
    assert I.shape == (200,)
    assert torch.isfinite(I).all() and (I >= 0).all()


def test_native_float_free_phsp_matrix(tmp_path):
    free = ctpwa.analysis(str(CONFIGS_DIR / "free_phsp.yml"))
    base = ctpwa.analysis(str(CONFIGS_DIR / "free_phsp_off.yml"))
    p = make_params(base, DEV)

    def read(path):
        with uproot.open(str(path)) as f:
            meta = f["integrals"].arrays(library="np")
            o = f["interference_rec"]
            n = int(o.member("fNrows"))
            M = np.asarray(o.member("fElements"), dtype=float).reshape(n, n)
        return float(meta["W_rec"][0]), M

    Wf, Mf = read(_write(free, p, tmp_path / "free.root"))
    Wb, Mb = read(_write(base, p, tmp_path / "base.root"))
    _assert_rec_matches_resident(base, p, Mb, Wb)
    assert abs(float(free.getNLL(p)) - float(base.getNLL(p))) < 1e-4
    np.testing.assert_allclose(Wf, Wb, rtol=1e-4)
    np.testing.assert_allclose(Mf, Mb, rtol=1e-4, atol=1e-6)


def _assert_rec_matches_resident(ana, p, rec, w_rec):
    """不用 Gram 收缩代码，从逐事件驻留振幅独立重建 rec。"""
    n = ana.getNVector()
    dt = torch.float32 if int(ctpwa.build_info()["complex_bytes"]) == 8 else torch.float64
    cpl = torch.complex(p[:n].to(dt), p[n:2 * n].to(dt)).contiguous().to(DEV)
    nsl = ana.getSLVectors().cpu().numpy()
    edges = np.concatenate([[0], np.cumsum(nsl)])
    A = ana.getPhspTensor().cpu().numpy().astype(np.complex128).reshape(-1, sum(nsl))
    v = ana.getExtendedVector(cpl).cpu().numpy().astype(np.complex128).reshape(-1)
    S = A * v
    parts = [S[:, edges[i]:edges[i + 1]].sum(axis=1) for i in range(len(nsl))]
    ref = np.zeros_like(rec)
    for i in range(len(nsl)):
        ref[i, i] = np.sum(np.abs(parts[i]) ** 2)
        for j in range(i + 1, len(nsl)):
            ref[i, j] = ref[j, i] = np.sum(2 * np.real(parts[i] * parts[j].conj()))
    total = np.trace(ref) + np.triu(ref, 1).sum()
    np.testing.assert_allclose(rec, ref / total, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(w_rec, total, rtol=2e-4)


def _write(ana, params, path):
    ana.writeResult(params, str(path), 0)
    return path
