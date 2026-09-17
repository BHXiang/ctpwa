"""Interp 共振态模型：表必须真正进入振幅、且形状与等价模型一致（回归防线）。

历史 bug（2026-09）：Interp 无自由参数 → 其 block 的 nFree==0 → 不走
computeCustomAmpsKernelT（唯一带 interpEval 的路径），而模板 kernel 走的
computeNodeFactor 没有 Interp 分支（default 返回 1.0）→ **表被静默丢弃**，
NLL 与 F≡1 完全相同，画出的分波形状自然对不上输入表。

另有配置层 bug：解析器只读共振态节点的直接键，不读 `options:` 嵌套映射，
而 tests/configs/interp.yml / example/config.yml 恰好用嵌套写法 → file/method
也被静默丢掉。

覆盖：
  1. 表必须改变 NLL（表真的进了振幅）；
  2. 嵌套 options: 与直接 file: 两种写法等价；
  3. **形状固化**：用 ctpwa 自己的 BW<T> 闭式生成一张表，Interp 读这张表
     必须与 BW 模型给出同样的分波直方图（实测 max|rel| ~1e-5）。
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import TESTS_DIR, make_params

SHAPE_TABLE = TESTS_DIR / "data" / "interp_shape.dat"
DATA_DIR = TESTS_DIR / "data"

_TMPL = """Particles:
  Jpsi: {J: 1, P: -1, mass: 3.0969, tex: ['J/\\\\psi']}
  eta:  {J: 0, P: -1, mass: 0.5478, tex: ['\\\\eta']}
  Kp:   {J: 0, P: -1, mass: 0.4937, tex: ['K^{+}']}
  Km:   {J: 0, P: -1, mass: 0.4937, tex: ['K^{-}']}
Data:
  order: [Kp, Km, eta]
  data: [dat, "@DATA@/test_data.dat"]
  phsp: [dat, "@DATA@/test_phsp.dat"]
DecayChains:
  chain1:
    decay:
      - Jpsi: [eta, R_KK]
      - R_KK: [Kp, Km]
    R_KK:
      - [J: 1, P: -1]: [@RES@]
Resonances:
  @RES@:
@MODEL@
Plot:
  mass:
    - input: [Kp, Km]
      bins: 100
      range: [0.5, 3.0]
      display: ["M(K+K-)", "Events"]
"""


def _interp_block(table, method="linear") -> str:
    return ("    J: 1\n    P: -1\n    model: Interp\n    parameters: []\n"
            f"    options:\n      file: \"{table}\"\n      method: {method}")


def _bw_block(m0, g0) -> str:
    return f"    J: 1\n    P: -1\n    model: BW\n    parameters: [{m0}, {g0}]"


def _write_cfg(path: Path, res: str, block: str) -> None:
    path.write_text(_TMPL.replace("@DATA@", str(DATA_DIR))
                    .replace("@RES@", res).replace("@MODEL@", block))


def _flat_table(src: Path, dst: Path):
    """同 x 网格、F≡1+0i 的基准表。"""
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            fo.write(f"{line.split()[0]} 1.0 0.0\n")


def _nll(cfg_path, device):
    import ctpwa
    ana = ctpwa.analysis(str(cfg_path))
    return float(ana.getNLL(make_params(ana, device)))


def _wave_hist(root_path):
    """取 mass0_Kp_Km 目录下第一个 h_ 分波直方图（避免硬编码拓扑名）。"""
    import uproot
    with uproot.open(root_path) as rf:
        key = None
        for k in rf.keys(recursive=True):
            name = k.split("/")[-1].split(";")[0]
            if k.startswith("mass0_Kp_Km/") and name.startswith("h_"):
                key = k
                break
        assert key is not None, f"{root_path}: 未找到 h_ 分波直方图"
        obj = rf[key]
        vals, edges = obj.to_numpy()
    return 0.5 * (edges[:-1] + edges[1:]), np.asarray(vals, dtype=float)


def _write_result(cfg_path, out_path, device):
    import ctpwa
    ana = ctpwa.analysis(str(cfg_path))
    ana.writeResult(make_params(ana, device), str(out_path))


def test_interp_table_enters_amplitude(device, tmp_path):
    """输入表必须是 bump 而非 F≡1：两表 NLL 必须显著不同。

    修复前两值完全相等（表被丢弃）；修复后实测 395.9 vs 1037.1。
    """
    flat = tmp_path / "flat.dat"
    _flat_table(SHAPE_TABLE, flat)
    cfg_shape = tmp_path / "shape.yml"
    _write_cfg(cfg_shape, "shape", _interp_block(SHAPE_TABLE))
    cfg_flat = tmp_path / "flat.yml"
    _write_cfg(cfg_flat, "shape", _interp_block(flat))

    n_shape = _nll(cfg_shape, device)
    n_flat = _nll(cfg_flat, device)
    assert abs(n_shape - n_flat) > 1.0, (
        f"Interp 表未进入振幅：shape NLL={n_shape} 与 flat NLL={n_flat} 相同"
        "（检查 computeNodeFactor 的 Interp 分支 / options 解析）"
    )


def test_interp_options_nested_and_direct_agree(device, tmp_path):
    """嵌套 options:{file,method} 与直接 file: 两种写法必须等价（解析器回归）。"""
    nested = tmp_path / "nested.yml"
    _write_cfg(nested, "shape", _interp_block(SHAPE_TABLE))
    direct = tmp_path / "direct.yml"
    direct.write_text(
        nested.read_text()
        .replace("    options:\n      file:", "    file:")
        .replace("      method: linear\n", "")
    )
    assert abs(_nll(nested, device) - _nll(direct, device)) < 1e-6


def test_interp_reproduces_bw_shape(device, tmp_path):
    """形状固化：Interp 读 BW 表 必须复现 BW 模型的分波直方图。

    表 = ctpwa BW<T> 闭式 F(m) = (m0²−m² + i·m0·g0)/s（无 Bf、无 q/q0），
    在 [0.8, 2.8] GeV 上细采样。同一批事件、同一 J^P/SL、同一 v，
    两个物理等价的模型应给出逐 bin 相同的 h_<波>（实测 max|rel| ~1e-5）。

    旧 bug 下 Interp 被当 F≡1 → 与 BW 直方图相对差 O(1)，本测试必失败。
    """
    m0, g0 = 1.6, 0.15
    m = np.linspace(0.8, 2.8, 2001)
    x = m0 * m0 - m * m
    y = m0 * g0
    s = x * x + y * y
    tab = tmp_path / "bw.dat"
    with open(tab, "w") as f:
        for mi, fr, fi in zip(m, x / s, y / s):
            f.write(f"{mi:.9f} {fr:.12e} {fi:.12e}\n")

    cfg_bw = tmp_path / "bw.yml"
    _write_cfg(cfg_bw, "res", _bw_block(m0, g0))
    cfg_ip = tmp_path / "ip.yml"
    _write_cfg(cfg_ip, "res", _interp_block(tab))

    out_bw, out_ip = tmp_path / "bw.root", tmp_path / "ip.root"
    _write_result(cfg_bw, out_bw, device)
    _write_result(cfg_ip, out_ip, device)

    c_bw, v_bw = _wave_hist(out_bw)
    c_ip, v_ip = _wave_hist(out_ip)
    assert np.allclose(c_bw, c_ip), "BW 与 Interp 的直方图轴不一致"
    assert v_bw.sum() > 0 and v_ip.sum() > 0

    # 归一化到相同总积分（去掉 writeResult 的整体归一化常数）后比形状
    n_bw = v_bw / v_bw.sum()
    n_ip = v_ip / v_ip.sum()
    mask = v_bw > 1e-3 * v_bw.max()          # 忽略近零 bin 的 float 噪声
    rel = np.abs(n_ip[mask] - n_bw[mask]) / n_bw[mask]
    assert rel.max() < 1e-3, (
        f"Interp(BW 表) 与 BW 模型直方图形状不一致：max|rel|={rel.max():.2e}"
        "（表未进入振幅 / 插值方法 / 越界外推）"
    )
