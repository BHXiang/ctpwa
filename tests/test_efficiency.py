"""getEfficiency 测试：分波效率 ε_i = (Σ_{phsp}|A_i|²/N_phsp)/(Σ_{phsp_truth}|A_i|²/N_truth)。

约定: phsp(如 cut 后 MC)=带效率样本, phsp_truth=无效率 MC truth。
- eff.yml 把 phsp 与 phsp_truth 指向同一文件 ⇒ 解析结果严格 ≡ 1。
- simple.yml 缺 phsp_truth ⇒ 返回空张量 [0,2]（自然跳过，不抛异常）。
"""

import torch
import pytest


def _coupling(ana, params):
    """按 .so 编译精度构造耦合向量(complex64/128)。"""
    import ctpwa
    is_double = ctpwa.DeviceManager().compiledPrecision() == "double"
    dt = torch.float64 if is_double else torch.float32
    n = ana.getNVector()
    real = params[:n].to(dt)
    imag = params[n : 2 * n].to(dt)
    return torch.complex(real, imag).contiguous().cuda()


def test_efficiency_same_file_center_is_one(make_analysis, device):
    """phsp 与 phsp_truth 同一文件时，每个分波效率必须严格为 1。"""
    ana = make_analysis("eff")
    n_vec = ana.getNVector()
    n_theta = ana.getNFreeTheta()
    params = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64, device=device)
    params[0] = 1.0
    params[n_vec] = 0.0
    g = torch.Generator(device=device).manual_seed(42)
    params[1:n_vec] = torch.rand(n_vec - 1, generator=g, device=device) * 0.5
    params[n_vec + 1 : 2 * n_vec] = (
        torch.rand(n_vec - 1, generator=g, device=device) - 0.5
    ) * 0.5

    eff = ana.getEfficiency(_coupling(ana, params))
    assert eff.dtype == torch.float64
    assert eff.ndim == 2 and eff.shape[1] == 2 and eff.shape[0] >= 1
    center, error = eff[:, 0].cpu(), eff[:, 1].cpu()
    # 同一文件逐事件分子分母相同 → 中心值解析恒等于 1
    assert torch.all(torch.abs(center - 1.0) < 1e-4), center.tolist()
    # MC 统计误差公式给出正量且量级合理（同文件仍含抽样不确定度）
    assert torch.all(error > 0), error.tolist()
    assert torch.all(error < 0.5), error.tolist()


def test_efficiency_missing_phsp_truth_returns_empty(make_analysis, device):
    """simple.yml 无 phsp_truth → 返回空张量 [0,2]（不抛异常，调用方自然跳过）。"""
    ana = make_analysis("simple")
    n_vec = ana.getNVector()
    params = torch.zeros(2 * n_vec, dtype=torch.float64, device=device)
    params[0] = 1.0
    eff = ana.getEfficiency(_coupling(ana, params))
    assert tuple(eff.shape) == (0, 2)


def test_efficiency_double_arg_overload(make_analysis, device):
    """双参重载 (vector, hessian) 与单参结果一致（hessian 传空时等价）。"""
    ana = make_analysis("eff")
    n_vec = ana.getNVector()
    n_theta = ana.getNFreeTheta()
    params = torch.zeros(2 * n_vec + n_theta, dtype=torch.float64, device=device)
    params[0] = 1.0
    params[n_vec] = 0.0
    g = torch.Generator(device=device).manual_seed(7)
    params[1:n_vec] = torch.rand(n_vec - 1, generator=g, device=device) * 0.5
    params[n_vec + 1 : 2 * n_vec] = (
        torch.rand(n_vec - 1, generator=g, device=device) - 0.5
    ) * 0.5
    cpl = _coupling(ana, params)

    eff1 = ana.getEfficiency(cpl)
    eff2 = ana.getEfficiency(cpl, torch.empty(0, dtype=torch.float64, device=device))
    assert torch.allclose(eff1, eff2, atol=1e-10)
