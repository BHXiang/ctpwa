"""重参数化助手 `_reparam_pack` / `_reparam_unpack` 的单元测试。

`example/fit.py` 在 import 时会构造 ctpwa.analysis()（需要 GPU+config.yml），
这里用一个 ctpwa stub 加载该模块，只取纯函数做 CPU 数值验证。
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

FIT_PY = Path(__file__).resolve().parent.parent / "example" / "fit.py"


@pytest.fixture(scope="module")
def reparam():
    """加载 example/fit.py，屏蔽其模块级 ctpwa.analysis() 初始化。"""

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
    # example/fit.py 现在还会 `from ctpwa.diagnostics import ...`；桩必须带上子模块，
    # 否则会出现 "No module named 'ctpwa.diagnostics'; 'ctpwa' is not a package"。
    diag = types.ModuleType("ctpwa.diagnostics")
    diag.analyze_hessian = lambda *a, **k: {}
    diag.map_errors_to_params = lambda *a, **k: []
    stub.diagnostics = diag

    old = sys.modules.get("ctpwa")
    sys.modules["ctpwa"] = stub
    sys.modules["ctpwa.diagnostics"] = diag
    try:
        spec = importlib.util.spec_from_file_location("ctpwa_example_fit", FIT_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("ctpwa.diagnostics", None)
        if old is not None:
            sys.modules["ctpwa"] = old
        else:
            sys.modules.pop("ctpwa", None)
    return mod


def _physical(nc, n_res, lower, upper):
    """构造一组合法的物理参数（固定 re_0=1, im_0=0）。"""
    p = torch.zeros(2 * nc + n_res, dtype=torch.float64)
    p[0] = 1.0
    p[nc] = 0.0
    for i in range(1, nc):
        amp = 0.2 + 0.5 * i
        phi = -1.0 + 0.7 * i
        p[i] = amp * torch.cos(torch.tensor(phi, dtype=torch.float64))
        p[nc + i] = amp * torch.sin(torch.tensor(phi, dtype=torch.float64))
    for k in range(n_res):
        p[2 * nc + k] = lower[k] + 0.37 * (upper[k] - lower[k])
    return p


def test_roundtrip_coupling_and_res(reparam):
    nc, n_res = 4, 3
    lower = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    upper = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
    v_max = 10000.0

    p = _physical(nc, n_res, lower, upper)
    u = reparam._reparam_pack(p, nc, n_res, lower, upper, v_max)
    assert u.numel() == 2 * (nc - 1) + n_res

    p2 = reparam._reparam_unpack(u, nc, n_res, lower, upper, v_max,
                                 dtype=torch.float64, device="cpu")
    assert torch.allclose(p, p2, atol=1e-12)
    assert p2[0].item() == 1.0
    assert p2[nc].item() == 0.0


def test_bounds_respected(reparam):
    nc, n_res = 3, 2
    lower = torch.tensor([0.5, 10.0], dtype=torch.float64)
    upper = torch.tensor([1.5, 20.0], dtype=torch.float64)
    v_max = 10000.0

    # 极端 u：软墙必须把物理量严格夹在界内
    u = torch.tensor([-50.0, 50.0, 0.0, 0.0,
                      -80.0, 80.0], dtype=torch.float64)
    p = reparam._reparam_unpack(u, nc, n_res, lower, upper, v_max,
                                dtype=torch.float64, device="cpu")

    re = p[1:nc]
    im = p[nc + 1:2 * nc]
    amp = torch.sqrt(re * re + im * im)
    assert bool((amp > 0).all())
    assert bool((amp <= v_max).all())

    theta = p[2 * nc:]
    # sigmoid 软墙：在 float64 下极端 u 会饱和到端点，故为闭区间 [lo, hi]
    assert bool((theta >= lower).all())
    assert bool((theta <= upper).all())


def test_nc_one_and_nres_zero(reparam):
    # only reference coupling, no free res
    u = torch.zeros(0, dtype=torch.float64)
    p = reparam._reparam_unpack(u, 1, 0, torch.empty(0), torch.empty(0), 1e4,
                                dtype=torch.float64, device="cpu")
    assert p.shape == (2,)
    assert p[0].item() == 1.0 and p[1].item() == 0.0

    # no res, 2 free couplings
    nc = 3
    p0 = _physical(nc, 0, torch.empty(0), torch.empty(0))
    u2 = reparam._reparam_pack(p0, nc, 0, torch.empty(0), torch.empty(0), 1e4)
    p2 = reparam._reparam_unpack(u2, nc, 0, torch.empty(0), torch.empty(0), 1e4,
                                 dtype=torch.float64, device="cpu")
    assert torch.allclose(p0, p2, atol=1e-12)


def test_gradcheck(reparam):
    nc, n_res = 3, 2
    lower = torch.tensor([1.0, 2.0], dtype=torch.float64)
    upper = torch.tensor([2.0, 4.0], dtype=torch.float64)
    v_max = 10000.0

    p = _physical(nc, n_res, lower, upper)
    u0 = reparam._reparam_pack(p, nc, n_res, lower, upper, v_max)
    u = u0.detach().clone().requires_grad_(True)

    def f(x):
        return reparam._reparam_unpack(x, nc, n_res, lower, upper, v_max,
                                       dtype=torch.float64, device="cpu")

    assert torch.autograd.gradcheck(f, (u,), eps=1e-6, atol=1e-6)
