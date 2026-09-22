"""`_errors_from_hessian`（非正定 Hessian 的参数误差回退）单元测试。

用 ctpwa stub 加载 example/fit.py，构造一个假的 optimizer `self`（只需
`_errors_from_hessian` 用到的属性），在合成 Hessian 上验证：
  - auto: PD→inv；非PD→pinv
  - strict: 非PD→不给
  - pinv/psd 的数值与手算一致
  - 分块索引映射（Re/Im/θ）正确
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

FIT_PY = Path(__file__).resolve().parent.parent / "example" / "fit.py"


def _load_fit():
    class _Ana:
        def getConstraintsIndex(self):
            return []

        def getParamNames(self):
            return []

        def getNVector(self):
            return 4

        def getFreeResParams(self):
            return torch.zeros(3, 2, dtype=torch.float64)

    stub = types.ModuleType("ctpwa")
    stub.analysis = lambda *a, **k: _Ana()

    old = sys.modules.get("ctpwa")
    sys.modules["ctpwa"] = stub
    try:
        spec = importlib.util.spec_from_file_location("ctpwa_example_fit", FIT_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is not None:
            sys.modules["ctpwa"] = old
        else:
            sys.modules.pop("ctpwa", None)
    return mod


def _fake_self(n, nc, mode, tau, A=None):
    """构造 _errors_from_hessian 需要的假 self（nll=0.5 q·q, 梯度=q）。"""
    n_res = n - 2 * nc

    class _Ana:
        def __init__(self, A):
            self.A = A

        def getNLL(self, q):
            return 0.5 * (q @ (self.A @ q))

    s = types.SimpleNamespace()
    s.n_coupling_free = nc
    s.n_params = n
    s.n_res_free = n_res
    s.has_free_res = n_res > 0
    s.device = "cpu"
    s.err_mode = mode
    s.err_tau = tau
    s.params_names = ([f"c{i}" for i in range(nc)]
                      + [f"t{i}" for i in range(n_res)])
    s.analysis = _Ana(torch.eye(n, dtype=torch.float64) if A is None else A)
    s.v_max = 1e9
    s._lower = torch.zeros(n_res, dtype=torch.float64)
    s._upper = torch.ones(n_res, dtype=torch.float64)
    s.bounds = lambda like: (torch.full_like(like, -1e9),
                             torch.full_like(like, 1e9))
    return s


def _diag_full(n, nc, red_vals):
    """按 red 顺序 [Re_1.., Im_1.., θ..] 填对角；固定 0/nc 给大值。"""
    H = torch.zeros(n, n, dtype=torch.float64)
    red = [i for i in range(n) if i not in (0, nc)]
    for i, v in zip(red, red_vals):
        H[i, i] = v
    H[0, 0] = 1.0
    H[nc, nc] = 1.0
    return H


def test_auto_pd_uses_inverse():
    mod = _load_fit()
    n, nc = 6, 2
    H = _diag_full(n, nc, [100.0, 4.0, 9.0, 16.0])
    s = _fake_self(n, nc, "auto", 1e-6)
    p = torch.zeros(n, dtype=torch.float64)
    d = mod.UnifiedPWAOptimizer._errors_from_hessian(s, H, p)
    assert d["is_pd"] and d["mode_used"] == "inv" and d["n_flat"] == 0
    # red: [Re_1, Im_1, θ_0, θ_1] = [100,4,9,16]
    assert abs(d["coupling_real_errors"][1].item() - 0.1) < 1e-6
    assert abs(d["coupling_imag_errors"][1].item() - 0.5) < 1e-6
    assert abs(d["res_errors"][0].item() - 1 / 3) < 1e-6
    assert abs(d["res_errors"][1].item() - 0.25) < 1e-6


def test_auto_nonpd_falls_back_to_pinv():
    mod = _load_fit()
    n, nc = 6, 2
    H = _diag_full(n, nc, [100.0, 1.0, -0.5, 1e-5])
    s = _fake_self(n, nc, "auto", 1e-6)   # τλmax = 1e-4
    p = torch.zeros(n, dtype=torch.float64)
    d = mod.UnifiedPWAOptimizer._errors_from_hessian(s, H, p)
    assert (not d["is_pd"]) and d["mode_used"] == "pinv"
    assert d["n_flat"] == 2                 # -0.5 与 1e-5
    assert abs(d["coupling_real_errors"][1].item() - 0.1) < 1e-6   # 1/√100
    assert abs(d["coupling_imag_errors"][1].item() - 1.0) < 1e-6   # 1/√1


def test_strict_nonpd_returns_none():
    mod = _load_fit()
    n, nc = 6, 2
    H = _diag_full(n, nc, [100.0, 1.0, -0.5, 1e-5])
    s = _fake_self(n, nc, "strict", 1e-6)
    p = torch.zeros(n, dtype=torch.float64)
    d = mod.UnifiedPWAOptimizer._errors_from_hessian(s, H, p)
    assert d["coupling_real_errors"] is None
    assert d["res_errors"] is None
    assert d["mode_used"] == "strict(no-pd)"


def test_psd_clips_negative():
    mod = _load_fit()
    n, nc = 6, 2
    H = _diag_full(n, nc, [100.0, 1.0, -0.5, 1e-5])
    s = _fake_self(n, nc, "psd", 1e-6)     # 截到 τλmax = 1e-4
    p = torch.zeros(n, dtype=torch.float64)
    d = mod.UnifiedPWAOptimizer._errors_from_hessian(s, H, p)
    assert d["mode_used"] == "psd"
    assert abs(d["coupling_real_errors"][1].item() - 0.1) < 1e-6
    assert abs(d["coupling_imag_errors"][1].item() - 1.0) < 1e-6
    exp = 1.0 / (1e-4 ** 0.5)
    assert abs(d["res_errors"][0].item() - exp) < 1e-6
    assert abs(d["res_errors"][1].item() - exp) < 1e-6
