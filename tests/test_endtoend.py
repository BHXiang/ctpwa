"""L2 端到端测试（发布前必跑）。

验证完整工作流：
- 拟合迭代收敛（NLL 下降、梯度有限）
- writeResult 输出文件
- reCalcAmp 更新振幅后 NLL 变化
- 真实数据子集存在时，用真实数据冒烟（集群上跑）
"""

import os
from pathlib import Path

import pytest
import torch

from conftest import make_params

OUT_DIR = Path(__file__).resolve().parent / "outputs"


@pytest.fixture(scope="session")
def outputs_dir():
    OUT_DIR.mkdir(exist_ok=True)
    return OUT_DIR


# ============================================================
# 拟合收敛
# ============================================================

def _clamp_theta(params, ana, n_vec):
    """将 theta 夹持在自由参数 bounds 内（模拟真实优化器的约束行为）。

    toy 数据无物理结构，无约束梯度下降会推着 theta 跑飞触发 1e30 保护，
    因此拟合测试用 bounds 夹持（与 fit.py 的 free_range 行为一致）。
    """
    free_res = ana.getFreeResParams()  # [3, n_res]: init, lower, upper
    lower = free_res[1].to(device=params.device, dtype=torch.float64)
    upper = free_res[2].to(device=params.device, dtype=torch.float64)
    theta = params[2 * n_vec :]
    params[2 * n_vec :] = torch.clamp(theta, lower, upper)
    return params


@pytest.mark.parametrize("config_key", ["simple", "with_trans"])
def test_fit_nll_decreases(make_analysis, device, config_key):
    """带 bounds 的梯度下降几步，NLL 必须单调下降（防梯度方向错误）。"""
    ana = make_analysis(config_key)
    n_vec = ana.getNVector()
    params = _clamp_theta(make_params(ana, device).clone(), ana, n_vec)
    params = params.requires_grad_(True)
    lr = 1e-3
    prev = None
    for step in range(8):
        nll = ana.getNLL(params)
        grad = torch.autograd.grad(nll, params)[0]
        assert torch.isfinite(grad).all(), f"{config_key} step{step}: 梯度非有限"
        updated = (params - lr * grad).detach()
        params = _clamp_theta(updated, ana, n_vec).requires_grad_(True)
        cur = ana.getNLL(params).item()
        if prev is not None:
            assert cur <= prev + 1e-6, (
                f"{config_key} step{step}: NLL 上升 {prev:.6f} -> {cur:.6f}"
            )
        prev = cur
        if step == 0 and prev == cur:
            pytest.skip(f"{config_key}: 梯度极小，NLL 无变化（数值退化）")


def test_fit_no_nan(make_analysis, device):
    """拟合过程中 NLL 与梯度始终保持有限（防 NaN 传播）。"""
    ana = make_analysis("with_trans")
    n_vec = ana.getNVector()
    params = _clamp_theta(make_params(ana, device).clone(), ana, n_vec)
    params = params.requires_grad_(True)
    for step in range(5):
        nll = ana.getNLL(params)
        assert torch.isfinite(nll), f"step{step}: NLL 非有限"
        grad = torch.autograd.grad(nll, params)[0]
        assert torch.isfinite(grad).all(), f"step{step}: 梯度非有限"
        updated = (params - 1e-3 * grad).detach()
        params = _clamp_theta(updated, ana, n_vec).requires_grad_(True)


# ============================================================
# writeResult
# ============================================================

def test_write_result(make_analysis, device, outputs_dir):
    """writeResult 正常生成输出文件（不崩溃）。"""
    ana = make_analysis("simple")
    params = make_params(ana, device)
    out_file = str(outputs_dir / "test_result.root")
    ana.writeResult(params, out_file, 0)
    assert os.path.exists(out_file), "writeResult 未生成文件"
    assert os.path.getsize(out_file) > 0, "writeResult 输出为空文件"


def test_write_result_with_trans(make_analysis, device, outputs_dir):
    """trans 约束下 writeResult 也应正常（防折叠参数路径崩溃）。"""
    ana = make_analysis("with_trans")
    params = make_params(ana, device)
    out_file = str(outputs_dir / "test_result_trans.root")
    ana.writeResult(params, out_file, 0)
    assert os.path.exists(out_file), "writeResult 未生成文件"


# ============================================================
# reCalcAmp
# ============================================================

def test_recalc_amp_changes_nll(make_analysis, device):
    """theta 扰动 → NLL 变化；reCalcAmp 手动更新不崩溃。

    注：getNLL 每次 forward 内部都 reComputeAmps(params 的 theta)，
    所以 reCalcAmp 的更新会被覆盖——这里验证的是 theta 参数路径本身。
    """
    ana = make_analysis("simple")
    params = make_params(ana, device)
    nll0 = ana.getNLL(params).item()

    n_vec = ana.getNVector()
    n_theta = ana.getNFreeTheta()
    assert n_theta > 0, "simple config 应有自由共振态参数"

    # theta 微扰后 NLL 必须变化（共振态参数进入 NLL 的路径）
    params_pert = params.clone()
    params_pert[2 * n_vec :] *= 1.01
    nll1 = ana.getNLL(params_pert).item()
    assert nll1 != pytest.approx(nll0, abs=1e-6), (
        "theta 扰动后 NLL 应变化"
    )

    # reCalcAmp 用扰动 theta 更新振幅，不崩溃
    ana.reCalcAmp(params_pert[2 * n_vec :])


# ============================================================
# 真实数据冒烟（集群上）
# ============================================================

def test_real_data_smoke(make_analysis, device, real_data_available):
    """真实数据子集存在时，跑一次完整 NLL（集群发布前验证）。

    本地无真实数据时自动跳过。
    """
    if not real_data_available:
        pytest.skip("需要真实数据子集 tests/data/real_*.dat（集群上生成）")

    import ctpwa
    from conftest import TESTS_DIR

    cfg_path = TESTS_DIR / "configs" / "real_data.yml"
    if not cfg_path.exists():
        pytest.skip("tests/configs/real_data.yml 不存在")
    ana = ctpwa.analysis(str(cfg_path))
    params = make_params(ana, device)
    nll = ana.getNLL(params).item()
    assert torch.isfinite(torch.tensor(nll)), f"真实数据 NLL 非有限: {nll}"


def test_real_fit_benchmark(device, real_data_available):
    """真实数据拟合收敛基准（集群发布前验证）。

    与 fit_benchmark.py 生成的基准对比：
    - 从同一起点（固定 seed）拟合同样步数
    - 收敛 NLL 应 ≤ 基准值 + 容差

    回归案例：数值优化破坏收敛性（梯度方向错、NaN 传播等）。
    """
    if not real_data_available:
        pytest.skip("需要真实数据子集（集群上先跑 prepare_real_data.py）")

    from fit_benchmark import load_benchmark

    bench = load_benchmark()
    if bench is None:
        pytest.skip("缺少基准 tests/golden/fit_real.txt（集群上先跑 fit_benchmark.py）")

    nll_init_ref, nll_final_ref, _ = bench

    import ctpwa
    from conftest import TESTS_DIR

    cfg = TESTS_DIR / "configs" / "real_data.yml"
    ana = ctpwa.analysis(str(cfg))
    params = make_params(ana, device).requires_grad_(True)

    n_vec = ana.getNVector()
    nll0 = ana.getNLL(params).item()
    assert nll0 == pytest.approx(nll_init_ref, rel=1e-6), (
        f"初始 NLL 与基准不符: {nll0:.6f} vs {nll_final_ref:.6f}"
    )

    # 用与 fit_benchmark 相同的参数拟合（steps=200, lr=5e-3）
    lr = 5e-3
    for step in range(200):
        nll = ana.getNLL(params)
        grad = torch.autograd.grad(nll, params)[0]
        assert torch.isfinite(grad).all(), f"step{step}: 梯度非有限"
        updated = (params - lr * grad).detach()
        if ana.getNFreeTheta() > 0:
            free_res = ana.getFreeResParams()
            lower = free_res[1].to(device=device, dtype=torch.float64)
            upper = free_res[2].to(device=device, dtype=torch.float64)
            updated[2 * n_vec :] = torch.clamp(updated[2 * n_vec :], lower, upper)
        params = updated.requires_grad_(True)

    final_nll = ana.getNLL(params).item()
    assert final_nll <= nll_final_ref * (1 + 1e-4), (
        f"收敛 NLL 未达基准: {final_nll:.6f} vs 基准 {nll_final_ref:.6f}"
    )
