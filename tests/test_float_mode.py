"""precision:float（全 float 档）行为测试（double .so 上）。

覆盖（Batch 1 里程碑）:
  1. fixed-θ config（无 free 参数）: precision:float 与 precision:double 的
     NLL/梯度一致（float 档误差 ~1e-8/1e-4 量级）;
  2. free-θ config: precision:float 显式报错（未接线 gate）;
  3. getHessian: precision:float 显式报错（gate）;
  4. fixed-θ 下 writeResult 冒烟（Float 档 = A float2 存储，消费端上转路径可用）。

注意: 本文件与 test_precision.py 一样需要手动同步到集群 tests/（tests 不入库）。
"""
from pathlib import Path

import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
GEN_DIR = TESTS_DIR / "_gen_float_mode"
GEN_DIR.mkdir(exist_ok=True)

import ctpwa  # noqa: E402
from conftest import make_params  # noqa: E402


def _analysis(name: str, precision: str):
    base = (TESTS_DIR / "configs" / f"{name}.yml").read_text()
    cfg = GEN_DIR / f"{name}_{precision}.yml"
    if not cfg.exists():
        cfg.write_text(f"precision: {precision}\n{base}")
    return ctpwa.analysis(str(cfg))


def _nll_grad(ana, device="cuda:0"):
    p = make_params(ana, torch.device(device)).requires_grad_(True)
    nll = float(ana.getNLL(p))
    g = torch.autograd.grad(ana.getNLL(p), p)[0].cpu()
    return nll, g


def test_float_fixed_theta_matches_double():
    """fixed-θ（interp，无自由参数）: float 档与 double 档 NLL/梯度一致。"""
    a_f = _analysis("interp", "float")
    a_d = _analysis("interp", "double")
    nll_f, g_f = _nll_grad(a_f)
    nll_d, g_d = _nll_grad(a_d)
    rel = abs(nll_f - nll_d) / max(abs(nll_d), 1e-30)
    assert rel < 1e-5, f"fixed-θ NLL float vs double rel={rel:.2e}"
    gdiff = (g_f - g_d).abs().max().item()
    assert gdiff < 1e-3, f"fixed-θ grad float vs double maxdiff={gdiff:.2e}"


def test_float_free_theta_matches():
    """free-θ（simple）: Float 档 getNLL+梯度可用，θ 梯度与 double 差异 ~1e-5 量级。"""
    a_f = _analysis("simple", "float")
    a_d = _analysis("simple", "double")
    nll_f, g_f = _nll_grad(a_f)
    nll_d, g_d = _nll_grad(a_d)
    rel = abs(nll_f - nll_d) / max(abs(nll_d), 1e-30)
    assert rel < 1e-6, f"free-θ NLL float vs double rel={rel:.2e}"
    nv = a_f.getNVector()
    gdiff = (g_f[2 * nv:] - g_d[2 * nv:]).abs().max().item()
    assert gdiff < 1e-3, f"free-θ θ梯度 float vs double maxdiff={gdiff:.2e}"


def test_float_hessian_matches():
    """Float 档 getHessian = hybrid 路径（A float2 + double 计算），与 double 一致。"""
    a_f = _analysis("simple", "float")
    a_d = _analysis("simple", "double")
    p = make_params(a_d, torch.device("cuda:0")).detach()
    Hf = a_f.getHessian(p).cpu()
    Hd = a_d.getHessian(p).cpu()
    rel = (Hf - Hd).abs().max().item() / max(Hd.abs().max().item(), 1e-30)
    assert rel < 1e-5, f"hessian float vs double rel={rel:.2e}"


def test_float_fixed_theta_write_result(tmp_path):
    """fixed-θ float 档 writeResult 冒烟（A float2 + 消费端上转路径）。"""
    ana = _analysis("interp", "float")
    p = make_params(ana, torch.device("cuda:0")).requires_grad_(True)
    out = tmp_path / "w.root"
    ana.writeResult(p.detach(), str(out))
    assert out.exists() and out.stat().st_size > 0
