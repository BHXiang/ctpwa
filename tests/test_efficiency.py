"""getEfficiency / getFitFractions 测试（CPU 与 CUDA 后端都跑）。

约定: phsp(如 cut 后 MC)=带效率样本, phsp_truth=无效率 MC truth。
- eff.yml 把 phsp 与 phsp_truth 指向同一文件 ⇒ 效率解析结果严格 ≡ 1。
- simple.yml 缺 phsp_truth ⇒ 返回空张量 [0,2]（自然跳过，不抛异常）。

回归（CPU 后端踩过的两个坑）:
1. `getFitFractions(v, hessian)` / `getEfficiency(v, hessian)` 的 hessian 检查曾写死
   `is_cuda()` → CPU 构建只要传非空 Hessian 必抛 "hessian_in must be float64 CUDA"；
2. `phsp_truth` 的早退排在 Tensor 检查之后 → 没有 phsp_truth 的配置也会因为传了
   一个 dtype/device 不合适的 Hessian 而抛错，而它本该安静返回空。

参数一律在 CPU 上生成再 `.to(DEVICE)`（`torch.Generator(device=...)` 的随机流
在两个后端不同，直接在目标设备上生成会让两端比到不同参数点）。
"""

import torch
import pytest

import ctpwa

DEVICE = torch.device("cpu") if ctpwa.backend() == "cpu" else torch.device("cuda:0")


def _complex_dtype():
    cb = int(ctpwa.build_info().get("complex_bytes", 16))
    return torch.float64 if cb == 16 else torch.float32


def _coupling(ana, params):
    """按 .so 编译精度构造耦合向量(complex64/128)，放到当前后端设备。"""
    dt = _complex_dtype()
    n = ana.getNVector()
    return torch.complex(params[:n].to(dt), params[n:2 * n].to(dt)).contiguous().to(DEVICE)


def _params(ana, seed=42):
    """[Re(v), Im(v), θ]，在 CPU 生成后搬到目标设备（保证跨后端同值）。"""
    n_vec, n_theta = ana.getNVector(), ana.getNFreeTheta()
    p = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64)
    p[0] = 1.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    if n_vec > 1:
        p[1:n_vec] = torch.rand(n_vec - 1, generator=g) * 0.5
        p[n_vec + 1:2 * n_vec] = (torch.rand(n_vec - 1, generator=g) - 0.5) * 0.5
    if n_theta > 0:
        p[2 * n_vec:] = ana.getFreeResParams()[0].double().cpu()
    return p.to(DEVICE)


def test_efficiency_same_file_center_is_one(make_analysis):
    """phsp 与 phsp_truth 同一文件时，每个分波效率必须严格为 1。"""
    ana = make_analysis("eff")
    params = _params(ana)
    eff = ana.getEfficiency(_coupling(ana, params))
    assert eff.dtype == torch.float64
    assert eff.ndim == 2 and eff.shape[1] == 2 and eff.shape[0] >= 1
    center, error = eff[:, 0].cpu(), eff[:, 1].cpu()
    assert torch.all(torch.abs(center - 1.0) < 1e-4), center.tolist()
    assert torch.all(error > 0), error.tolist()
    assert torch.all(error < 0.5), error.tolist()


def test_efficiency_missing_phsp_truth_returns_empty(make_analysis):
    """simple.yml 无 phsp_truth → 返回空张量 [0,2]（不抛异常，调用方自然跳过）。"""
    ana = make_analysis("simple")
    params = _params(ana)
    eff = ana.getEfficiency(_coupling(ana, params))
    assert tuple(eff.shape) == (0, 2)


def test_efficiency_double_arg_overload(make_analysis):
    """双参重载 (vector, hessian) 与单参结果一致（hessian 传空时等价）。"""
    ana = make_analysis("eff")
    params = _params(ana, seed=7)
    cpl = _coupling(ana, params)
    eff1 = ana.getEfficiency(cpl)
    eff2 = ana.getEfficiency(cpl, torch.empty(0, dtype=torch.float64, device=DEVICE))
    assert torch.allclose(eff1, eff2, atol=1e-10)


def test_fit_fractions_accepts_backend_hessian(make_analysis):
    """回归：hessian 检查必须走后端感知判断（CPU 构建传 CPU float64 Hessian 不能抛）。

    以前这里写死 `hessian_in.is_cuda()`，CPU 构建必抛
    "hessian_in must be float64 CUDA"。
    """
    ana = make_analysis("eff")
    params = _params(ana)
    cpl = _coupling(ana, params)
    H = ana.getHessian(params).double()          # 统一 Hessian (2n+θ)²
    ff = ana.getFitFractions(cpl, H)              # 内部取左上 [2n,2n] 耦合块
    assert ff.dtype == torch.float64
    assert ff.ndim == 2 and ff.shape[1] == 2 and ff.shape[0] >= 1
    center = ff[:, 0].double().cpu()
    assert abs(float(center.sum()) - 1.0) < 1e-5, center.tolist()
    assert torch.all(torch.isfinite(center))


def test_fit_fractions_missing_phsp_truth_early_return(make_analysis):
    """回归：缺 phsp_truth 时早退必须在 Tensor 检查**之前**返回空。

    调用方探测"有没有份额"时不该被 dtype/device 挡住：这里故意传错误 dtype 的
    Hessian（float32）和非法的空 vector，仍应安静返回 [0,2] 而不是抛错。
    """
    ana = make_analysis("simple")                # simple.yml 无 phsp_truth
    bad_h = torch.zeros(4, 4, dtype=torch.float32, device=DEVICE)
    v = torch.zeros(max(ana.getNVector(), 1), dtype=torch.complex128, device=DEVICE)
    ff = ana.getFitFractions(v, bad_h)
    assert tuple(ff.shape) == (0, 2)
    eff = ana.getEfficiency(v, bad_h)
    assert tuple(eff.shape) == (0, 2)
