"""Plot 新格式回归（L1）: Plot 直接是观测列表（expr/expression, 1d/2d）。

主推格式由 example/config.yml 示范; 本文件做行为门禁——此前该格式无
任何测试 config 覆盖, 只有旧 mass/cosbeta/dalitz 键有测试路径。

覆盖:
  1. writeResult 后 root 内每个具名目录含 hdata（数据）与 hfit（全模型）;
  2. 缺省 name 的项 → obs<N> 目录（N = 列表序, 从 0）;
  3. 2d 项（TH2F）hdata 非空;
  4. 1d 全覆盖观测（cosβ ∈ [-1,1]）数据谱总计数 == data 事件数（weight=1）;
     M(KK) 谱因 range 下限 1.0 高于阈 ~0.987, 允许少量逃逸（<1%）。
"""

import os
from pathlib import Path

import pytest
import uproot

from conftest import make_params

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# test_data.dat 事件数（weight=1 直方化总计数基准）
N_DATA = 1000


@pytest.fixture(scope="module")
def outputs_dir():
    OUT_DIR.mkdir(exist_ok=True)
    return OUT_DIR


@pytest.fixture(scope="module")
def plot_result(make_analysis, device, outputs_dir):
    ana = make_analysis("plot_obs")
    params = make_params(ana, device)
    out_file = str(outputs_dir / "plot_obs_result.root")
    if os.path.exists(out_file):
        os.remove(out_file)
    ana.writeResult(params, out_file, 0)
    assert os.path.exists(out_file) and os.path.getsize(out_file) > 0
    return out_file


def test_plot_obs_dirs_have_hdata_hfit(plot_result):
    """每个 Plot 项（含缺省 name 的 obs3）都有数据与模型直方图。"""
    with uproot.open(plot_result) as f:
        for d in ("m_KK", "cosbeta_KK", "dalitz2", "obs3"):
            assert d in f, f"缺 Plot 目录 {d}（现有 keys: {list(f.keys())[:25]}）"
            for key in ("hdata", "hfit"):
                assert key in f[d], f"{d}: 缺 {key}（keys={list(f[d].keys())}）"


def test_plot_obs_2d_is_th2f_nonempty(plot_result):
    with uproot.open(plot_result) as f:
        h = f["dalitz2/hdata"]
        vals = h.values()
        assert vals.shape == (100, 100), f"2d 分箱形状 {vals.shape} != (100,100)"
        assert vals.sum() > 0, "dalitz2/hdata 全零"


def test_plot_obs_1d_full_coverage_counts(plot_result):
    """cosβ ∈ [-1,1] 全覆盖: 数据谱总计数应等于 data 事件数。

    M(KK) 的 range [1.0,2.6] 下限高于运动学阈 (~0.987 GeV), 允许 <1% 逃逸。
    """
    with uproot.open(plot_result) as f:
        s_cos = f["cosbeta_KK/hdata"].values().sum()
        assert s_cos == N_DATA, f"cosbeta 总计数 {s_cos} != {N_DATA}"
        s_th = f["obs3/hdata"].values().sum()
        assert s_th == N_DATA, f"theta 总计数 {s_th} != {N_DATA}"
        s_m = f["m_KK/hdata"].values().sum()
        assert 0.99 * N_DATA <= s_m <= N_DATA, f"M(KK) 总计数 {s_m} 越界"


def test_plot_obs_2d_counts_and_obs3_dir(plot_result):
    """2d 谱总计数接近 data 事件数（M² 轴下限 1.0 > KK 阈 M²~0.975, 允许 <2% 逃逸）。"""
    with uproot.open(plot_result) as f:
        s_d = f["dalitz2/hdata"].values().sum()
        assert 0.98 * N_DATA <= s_d <= N_DATA, f"dalitz2 总计数 {s_d} 越界"
