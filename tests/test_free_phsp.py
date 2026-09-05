"""Constraints.free_phsp_amplitudes（phsp 流式模式）测试。

核心不变量：
1. 流式模式与驻留模式的 NLL 数值一致（double 矩阵替代逐 forward 扫 phsp 振幅）
2. 流式模式梯度与有限差分一致
3. writeResult 在流式模式下按批重算正常出文件
4. getPhspTensor 在流式模式下明确报错（phsp 不驻留）
5. 有 free 参数时开关被忽略（保持驻留，getPhspTensor 可用）
"""

import os

import pytest
import torch

from conftest import make_params

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def _nll_at(ana, device, seed=42, n_points=4):
    """在几个随机参数点上计算 NLL（固定参考振幅约定同 make_params）。"""
    vals = []
    for i in range(n_points):
        params = make_params(ana, device, seed=seed + i)
        vals.append(ana.getNLL(params).item())
    return vals


def test_free_phsp_nll_matches_baseline(make_analysis, device):
    """流式（double 矩阵）与驻留（computePhspMeanSum）的 NLL 必须一致。"""
    ana_free = make_analysis("free_phsp", reuse=False)
    ana_base = make_analysis("free_phsp_off", reuse=False)
    # 两个 analysis 用同一批参数点（参数布局一致：单链单共振、无 theta）
    free_vals = _nll_at(ana_free, device)
    base_vals = _nll_at(ana_base, device)
    for i, (fv, bv) in enumerate(zip(free_vals, base_vals)):
        assert abs(fv - bv) < 1e-3, (
            f"点 {i}: 流式 NLL={fv:.8f} vs 驻留 NLL={bv:.8f}，偏差过大"
        )


def test_free_phsp_gradient_vs_fd(make_analysis, device):
    """流式模式梯度 vs 有限差分（数值导数自洽性）。"""
    ana = make_analysis("free_phsp", reuse=False)
    params = make_params(ana, device).requires_grad_(True)
    nll = ana.getNLL(params)
    grad_ad = torch.autograd.grad(nll, params)[0].detach().cpu().double()

    n_vec = ana.getNVector()
    p = params.detach().cpu().double()
    for i in range(p.numel()):
        if i == 0 or i == n_vec:  # 固定参考振幅 Re=1 / Im=0（约定不参与求导）
            continue
        h = max(abs(p[i].item()) * 1e-3, 5e-4)
        pp = p.clone()
        pm = p.clone()
        pp[i] += h
        pm[i] -= h
        fd = (ana.getNLL(pp.cuda()).item() - ana.getNLL(pm.cuda()).item()) / (2 * h)
        rel = abs(grad_ad[i].item() - fd) / max(abs(grad_ad[i].item()), 1e-6)
        assert rel < 1e-2, (
            f"参数[{i}]: AD={grad_ad[i].item():.4e} FD={fd:.4e} rel={rel:.2e}"
        )


def test_free_phsp_write_result(make_analysis, device):
    """流式模式 writeResult 按批重算 phsp 振幅，正常出文件。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    ana = make_analysis("free_phsp", reuse=False)
    params = make_params(ana, device)
    out_file = os.path.join(OUT_DIR, "test_free_phsp_result.root")
    ana.writeResult(params, out_file, 0)
    assert os.path.exists(out_file), "writeResult 未生成文件"
    assert os.path.getsize(out_file) > 0, "writeResult 输出为空文件"


def test_free_phsp_getPhspTensor_raises(make_analysis):
    """流式模式 phsp 振幅不驻留，getPhspTensor 必须明确报错。"""
    ana = make_analysis("free_phsp", reuse=False)
    with pytest.raises(RuntimeError):
        ana.getPhspTensor()


def test_free_phsp_flag_ignored_with_free_params(make_analysis, device):
    """有 free 参数时开关被忽略（保持驻留）: getNLL 与 getPhspTensor 均正常。"""
    ana = make_analysis("free_phsp_ignored", reuse=False)
    params = make_params(ana, device)
    nll = ana.getNLL(params)
    assert torch.isfinite(nll), "NLL 非有限"
    # phsp 驻留 → getPhspTensor 可用
    phsp = ana.getPhspTensor()
    assert phsp.numel() > 0
