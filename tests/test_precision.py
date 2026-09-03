"""precision 奇偶性 & 默认值回归（double .so 上; float .so 上整体 skip）。

背景：大统计量分波默认省显存——double .so 上 `precision: auto`（config 缺省）
现在解析为混合精度（float_amps_=true：驻留振幅矩阵 A 存 float2，显存减半，
核心计算仍是 double，NLL 漂移 ~1e-9 相对量级）；需要全 double 数值时显式写
`precision: double`。

覆盖：
1. 默认值：auto 缺省 = 混合精度（启动打印标记），显式 double 不带标记;
2. 数值奇偶：同一 config 的 precision:float 与 precision:double 实例
   NLL / 梯度 / Hessian / getFitFractions / getEfficiency 逐值一致
   （实测偏差 ~1e-10..1e-7，容差取 ~1e-5 量级——既留足 float-A 舍入余量，
   又能抓住"float 分支整体读错 A / 忘上转"这类 O(1) 错误）;
3. 无回归护栏：float 模式下 getFitFractions/getEfficiency 正常出数
   （早先被 gate 拦截，见 D4）。
"""

import os
from pathlib import Path

import pytest
import torch

from conftest import make_params

TESTS_DIR = Path(__file__).resolve().parent
GEN_DIR = TESTS_DIR / "outputs" / "gen"


def _is_double_so():
    import ctpwa
    return ctpwa.DeviceManager().compiledPrecision() == "double"


pytestmark = pytest.mark.skipif(
    not _is_double_so(),
    reason="奇偶对比需要 double .so（float .so 上 precision:double 会抛错）",
)


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------

_ANALYSES = {}


def _analysis(name: str, precision: str):
    """生成 <name>_<precision>.yml（原 config 前插 precision 根键）并缓存实例。"""
    key = (name, precision)
    if key in _ANALYSES:
        return _ANALYSES[key]
    import ctpwa
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    base = (TESTS_DIR / "configs" / f"{name}.yml").read_text()
    cfg_path = GEN_DIR / f"{name}_{precision}.yml"
    cfg_path.write_text(f"precision: {precision}\n{base}")
    ana = ctpwa.analysis(str(cfg_path))
    _ANALYSES[key] = ana
    return ana


def _pair(name):
    return _analysis(name, "float"), _analysis(name, "double")


def _coupling(ana, seed=3):
    """构造 double .so 需要的 complex128 耦合向量。"""
    n = ana.getNVector()
    g = torch.Generator(device="cuda").manual_seed(seed)
    re = torch.rand(n, generator=g, device="cuda", dtype=torch.float64) * 0.5
    im = (torch.rand(n, generator=g, device="cuda", dtype=torch.float64) - 0.5) * 0.5
    v = torch.complex(re, im).contiguous()
    v[0] = torch.complex(
        torch.ones(1, dtype=torch.float64, device="cuda"),
        torch.zeros(1, dtype=torch.float64, device="cuda"),
    )
    return v


# ---------------------------------------------------------------
# 默认值：auto → 混合精度 float-A
# ---------------------------------------------------------------

def test_auto_default_is_hybrid_float(capfd, device):
    """config 缺省 precision（auto）→ 启动打印混合精度标记；显式 double 无标记。

    打印发生在 analysis 构造期间（std::cout），用 capfd 在 fd 层抓取。
    """
    import ctpwa
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    base = (TESTS_DIR / "configs" / "simple.yml").read_text()
    cfg_auto = GEN_DIR / "simple_auto_default.yml"
    cfg_auto.write_text(base)  # 无 precision 键 → auto
    cfg_double = GEN_DIR / "simple_auto_double.yml"
    cfg_double.write_text("precision: double\n" + base)

    capfd.readouterr()  # 清空
    ctpwa.analysis(str(cfg_auto))
    out_auto, _ = capfd.readouterr()
    assert "混合精度模式" in out_auto, (
        "precision:auto 应默认混合精度(float-A), 实际未打印标记"
    )

    capfd.readouterr()
    ctpwa.analysis(str(cfg_double))
    out_double, _ = capfd.readouterr()
    assert "混合精度模式" not in out_double, (
        "precision:double 不应有混合精度标记"
    )


# ---------------------------------------------------------------
# NLL / 梯度 / Hessian 奇偶
# ---------------------------------------------------------------

def test_nll_parity(device):
    """float-A 与 double-A 的 NLL 一致（实测 |Δ|/|NLL| ~ 6e-10）。"""
    af, ad = _pair("simple")
    params = make_params(ad, device)
    nf = af.getNLL(params).item()
    nd = ad.getNLL(params).item()
    assert nf == pytest.approx(nd, rel=1e-5, abs=1e-3), (
        f"NLL 奇偶失配: float={nf:.10f} double={nd:.10f}"
    )


def test_gradient_parity(device):
    """float-A 与 double-A 的梯度一致（排除固定参考 v0 的数值噪声分量）。"""
    af, ad = _pair("simple")
    params = make_params(ad, device)
    n_vec = ad.getNVector()

    gf = torch.autograd.grad(af.getNLL(params.requires_grad_(True)), params)[0].detach()
    gd = torch.autograd.grad(ad.getNLL(params.requires_grad_(True)), params)[0].detach()

    mask = torch.ones_like(gd, dtype=torch.bool)
    mask[0] = False
    mask[n_vec] = False
    assert mask.any(), "无自由参数可比"
    d = (gf[mask] - gd[mask]).abs()
    scale = gd[mask].abs().clamp_min(1e-6)
    max_rel = (d / scale).max().item()
    assert max_rel < 1e-3, f"梯度奇偶失配, max rel = {max_rel:.2e}"


def test_hessian_parity(device):
    """float-A 与 double-A 的统一 Hessian 一致（实测 max|ΔH|/max|H| ~ 3e-9）。"""
    af, ad = _pair("simple")
    params = make_params(ad, device)
    hf = af.getHessian(params).detach()
    hd = ad.getHessian(params).detach()
    assert hf.shape == hd.shape
    assert torch.allclose(hf, hd, rtol=1e-4, atol=1e-3), (
        f"Hessian 奇偶失配: max|dH|={(hf - hd).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------
# getFitFractions / getEfficiency 奇偶（双波 config eff2）
# ---------------------------------------------------------------

def test_fit_fractions_parity(device):
    """双波拟合分数 float-A 与 double-A 一致（中心值 + 误差列）。"""
    af, ad = _pair("eff2")
    v = _coupling(ad)
    ff_f = af.getFitFractions(v).cpu()
    ff_d = ad.getFitFractions(v).cpu()
    assert ff_f.shape == ff_d.shape == (2, 2)
    d = (ff_f - ff_d).abs().max().item()
    assert d < 1e-8, f"getFitFractions 奇偶失配, max|d|={d:.3e}"
    # 非平凡性: 两波份额都非 0/1, 且有误差
    center = ff_d[:, 0]
    assert torch.all(center > 0.05) and torch.all(center < 0.95), center.tolist()


def test_efficiency_parity(device):
    """效率 float-A 与 double-A 一致（ε≡1 中心值 + MC 统计误差列）。"""
    af, ad = _pair("eff2")
    v = _coupling(ad)
    ef = af.getEfficiency(v).cpu()
    ed = ad.getEfficiency(v).cpu()
    assert ef.shape == ed.shape == (2, 2)
    d = (ef - ed).abs().max().item()
    assert d < 1e-8, f"getEfficiency 奇偶失配, max|d|={d:.3e}"
    assert torch.all((ef[:, 0] - 1.0).abs() < 1e-6), ef[:, 0].tolist()


# ---------------------------------------------------------------
# float 模式 writeResult 冒烟（原 gate 语义: FF/EFF 外的低频路径同套上转）
# ---------------------------------------------------------------

@pytest.mark.parametrize("precision", ["float", "double"])
def test_write_result_smoke(precision, device):
    """两种精度下 writeResult 都正常出文件（防回归到 gate/崩溃）。"""
    af = _analysis("simple", precision)
    params = make_params(af, device)
    out_file = str(GEN_DIR / f"test_result_{precision}.root")
    af.writeResult(params, out_file, 0)
    assert os.path.exists(out_file), f"{precision}: writeResult 未生成文件"
    assert os.path.getsize(out_file) > 0, f"{precision}: writeResult 空文件"
