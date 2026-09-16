"""Interp 共振态模型：表必须真正进入振幅（回归防线）。

历史 bug（2026-09）：Interp 无自由参数 → 其 block 的 nFree==0 → 不走
computeCustomAmpsKernelT（唯一带 interpEval 的路径），而模板 kernel 走的
computeNodeFactor 没有 Interp 分支（default 返回 1.0）→ **表被静默丢弃**，
NLL 与 F≡1 完全相同，画出的分波形状自然对不上输入表。

另有配置层 bug：解析器只读共振态节点的直接键，不读 `options:` 嵌套映射，
而 tests/configs/interp.yml / example/config.yml 恰好用嵌套写法 → file/method
也被静默丢掉。本测试同时覆盖这两种写法。
"""
from pathlib import Path

import pytest
import torch

from conftest import TESTS_DIR, make_params

SHAPE_TABLE = TESTS_DIR / "data" / "interp_shape.dat"

_CFG = """Particles:
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
      - [J: 1, P: -1]: [shape]
Resonances:
  shape:
    J: 1
    P: -1
    model: Interp
    parameters: []
    options:
      file: "@TABLE@"
      method: linear
Plot:
  mass:
    - input: [Kp, Km]
      bins: 100
      range: [0.5, 3.0]
      display: ["M(K+K-)", "Events"]
"""


def _write_cfg(path: Path, data: Path, table) -> None:
    path.write_text(_CFG.replace("@DATA@", str(data)).replace("@TABLE@", str(table)))


def _flat_table(src: Path, dst: Path):
    """同 x 网格、F≡1+0i 的基准表。"""
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            x = line.split()[0]
            fo.write(f"{x} 1.0 0.0\n")


def _nll(cfg_path, device):
    import ctpwa
    ana = ctpwa.analysis(str(cfg_path))
    p = make_params(ana, device)
    return float(ana.getNLL(p))


def test_interp_table_enters_amplitude(device, tmp_path):
    """输入表必须是 bump 而非 F≡1：两表 NLL 必须显著不同。

    修复前两值完全相等（表被丢弃）；修复后实测 395.9 vs 1037.1。
    """
    flat = tmp_path / "flat.dat"
    _flat_table(SHAPE_TABLE, flat)
    data = TESTS_DIR / "data"

    cfg_shape = tmp_path / "shape.yml"
    _write_cfg(cfg_shape, data, SHAPE_TABLE)
    cfg_flat = tmp_path / "flat.yml"
    _write_cfg(cfg_flat, data, flat)

    n_shape = _nll(cfg_shape, device)
    n_flat = _nll(cfg_flat, device)
    assert abs(n_shape - n_flat) > 1.0, (
        f"Interp 表未进入振幅：shape NLL={n_shape} 与 flat NLL={n_flat} 相同"
        "（检查 computeNodeFactor 的 Interp 分支 / options 解析）"
    )


def test_interp_options_nested_and_direct_agree(device, tmp_path):
    """嵌套 options:{file,method} 与直接 file: 两种写法必须等价（解析器回归）。"""
    data = TESTS_DIR / "data"
    nested = tmp_path / "nested.yml"
    _write_cfg(nested, data, SHAPE_TABLE)
    direct = tmp_path / "direct.yml"
    direct.write_text(
        nested.read_text()
        .replace("    options:\n      file:", "    file:")
        .replace("      method: linear\n", "")
    )
    assert abs(_nll(nested, device) - _nll(direct, device)) < 1e-6
