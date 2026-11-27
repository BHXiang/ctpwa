"""零长度表回归：`deep_4pi0` 必须能构造。

一次真实漏网（第 32 轮）：`DevBuffer` 在表长度为 0 时提前返回空指针，而调用方用
`if (!ptr) throw "slamp 表分配失败"` 判错 —— `deep_4pi0`（nSL 组合数为 0）因此直接
构造失败，但**没有任何测试用到这个配置**，全量门禁仍然全绿。

这里只钉住这一条路径（单个 config，秒级）。不遍历全部 64 个 config：那是 9 分钟，
而门禁的其它用例已经覆盖了绝大多数配置。
"""
import io
import contextlib
from pathlib import Path

import ctpwa

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def test_zero_sl_combination_config_constructs():
    """nSL 组合数为 0 的深层链：表长度 0 必须当"空表"处理，而不是"分配失败"。"""
    with contextlib.redirect_stdout(io.StringIO()):
        ana = ctpwa.analysis(str(CONFIGS_DIR / "deep_4pi0.yml"))
    assert ana.getNVector() >= 1
    assert ana.getNPolarizations() >= 1
