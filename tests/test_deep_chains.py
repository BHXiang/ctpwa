"""深层衰变链展开测试（DFS 多分支 + 同名兄弟递增去重）。

- deep_4pi0.yml: 对称同名中间态 R_2pi0+R_2pi0（链 A）与 3 级深层链（链 B）。
- ppbar_2pi0.yml: π⁺π⁻π⁺π⁻ 场景，R_2pi0 有 4 种模式——非等价配对必须恰为
  2 条链（{pip1,pim1}+{pip2,pim2} 与 {pip1,pim2}+{pip2,pim1}），
  同一 π 被两个兄弟同时使用（如 {pip1,pim1}+{pip2,pim1}）不允许出现。

DecayInfo.print() 发出 C++ std::cout（进程 fd 1），非 tty 下全缓冲 ——
捕获前需 fflush(NULL)（libc），capfd 才能读到。
"""

import ctypes

import pytest

import ctpwa


def _print_chains(config_name, capfd):
    """DecayInfo 展开并捕获 print 输出。"""
    dinfo = ctpwa.DecayInfo(f"configs/{config_name}.yml")
    dinfo.print()
    ctypes.CDLL(None).fflush(None)
    return capfd.readouterr().out


def test_deep_4pi0_symmetric_and_3level_chains(capfd):
    """对称兄弟各配对一次 + 3 级深层链，共 2 条；链名含完整中间态路径。"""
    out = _print_chains("deep_4pi0", capfd)
    assert "Chains: 2" in out, out
    # 链 A: R_2pi0 两种模式各一次（不重名），无实例序号
    assert "decay1_R_4pi0_R_2pi0: Jpsi→gamma+R_4pi0_R_4pi0→R_2pi0+R_2pi0" in out
    assert "R_2pi0→pi03+pi04_R_2pi0→pi01+pi02" in out
    # 链 B: 3 级深层，链名含完整路径
    assert (
        "decay1_R_4pi0_R_3pi0_R_2pi0: Jpsi→gamma+R_4pi0_R_4pi0→pi01+R_3pi0"
        "_R_3pi0→pi02+R_2pi0_R_2pi0→pi03+pi04" in out
    )
    # 恰好 2 条链（无多余分支：pi01/pi02 不得被链 A 的 R_2pi0 占用）
    assert out.count("decay1_R_4pi0") == 2 * 1  # 行首链名出现 2 次
    assert "decay1_R_4pi0_R_2pi0_0" not in out  # 两链名互异，无序号

def test_ppbar_two_pairings_exactly(capfd):
    """4 模式 → 恰 2 条链：{pip1,pim1}+{pip2,pim2} 与 {pip1,pim2}+{pip2,pim1}。"""
    out = _print_chains("ppbar_2pi0", capfd)
    assert "Chains: 2" in out, out
    assert (
        "decay1_R_4pi0_R_2pi0_0: Jpsi→gamma+R_4pi0_R_4pi0→R_2pi0+R_2pi0"
        "_R_2pi0→pip1+pim1_R_2pi0→pip2+pim2" in out
    )
    assert (
        "decay1_R_4pi0_R_2pi0_1: Jpsi→gamma+R_4pi0_R_4pi0→R_2pi0+R_2pi0"
        "_R_2pi0→pip1+pim2_R_2pi0→pip2+pim1" in out
    )
    # 同一 π 不被两个兄弟同时使用（{0,0}+{2,0}... 类错误配对）
    for bad in (
        "R_2pi0→pip2+pim1_R_2pi0→pip2+pim1",  # pim1 双用
        "R_2pi0→pip1+pim1_R_2pi0→pip1+pim2",  # pip1 双用
        "R_2pi0→pip2+pim2_R_2pi0→pip2+pim2",  # 全同（去重失败）
    ):
        assert bad not in out, f"非法配对出现: {bad}"
    # 排序退化 {2,·}/{3,·}（无候选）与等价配对 {0,2}/{1,3} 不生成
    assert out.count("decay1_R_4pi0_R_2pi0_") == 2


def test_ppbar_analysis_builds_valid_waves(make_analysis):
    """2 条链 × X4pi0 2 分波 = 4 个耦合参数，Analysis 端构建无误（ONE 无 theta）。"""
    ana = make_analysis("ppbar_2pi0", reuse=False)
    assert ana.getNVector() == 4
    assert ana.getNFreeTheta() == 0
