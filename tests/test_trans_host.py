"""trans 约束的宿主端回归测试（DecayInfo 解析路径, 不需要 GPU）。

背景: 引擎里所有 trans 测试过去都走 ctpwa.analysis（需要 GPU）, 因此
"约束被静默忽略"这类问题在无 GPU 的开发机上无法暴露。这里用
DecayInfo（纯 host 解析 + 耦合矩阵构建, 与 analysis 构造期同一代码路径）
直接校验折叠结果:

  - 已发布 config: no_trans=6 / with_trans=4 / with_deep_trans=4 个自由耦合
  - 新增 trans_multi.yml: 三个电荷道 [rho0, rhop, rhom]: [-1, 1]
    （1 个约束 N 个链名, 每个名字一个比值）应把 π⁺ρ⁻/π⁻ρ⁺ 折到 π⁰ρ⁰
  - ONE 模型: 占位质量可省略 / 可写任意值（不再要求 1 个参数）

注意: ONE 的占位质量"不进入振幅"这一条需要振幅求值才能数值验证（需 GPU）,
此处只保证解析/参数布局正确；公式层面的验证见 ResModel.cuh oneRefQ0 注释。
"""

from pathlib import Path

import pytest

import ctpwa

CONFIGS = Path(__file__).resolve().parent / "configs"


def _chain_param_names(info):
    """链级耦合参数名 = 不含 "_LS(" 的参数名（step 耦合名带 _LS(...)）。"""
    return [n for n in info.paramNames() if "_LS(" not in n]


def test_shipped_trans_counts_host():
    """已发布 config 的折叠数（与 tests/test_structure.py 的 GPU 版一致）。"""
    expect = {"no_trans": 6, "with_trans": 4, "with_deep_trans": 4}
    for name, want in expect.items():
        info = ctpwa.DecayInfo(str(CONFIGS / f"{name}.yml"))
        assert info.isValid()
        assert info.nFreeParams() == want, (
            f"{name}: 期望 {want} 个自由耦合, 实际 {info.nFreeParams()}"
        )


def test_multi_trans_folds_all_charge_channels():
    """[rho0, rhop, rhom]: [-1, 1] 应把三条电荷道的链全部折到 rho0。

    未加约束: 3 电荷道 × 2 个 R 共振组合 = 6 个链耦合;
    加约束后: 只剩 rho0 的 2 个组合（rhop/rhom 折进去, 比值 -1 / +1）。
    """
    none = ctpwa.DecayInfo(str(CONFIGS / "trans_multi_none.yml"))
    assert len(_chain_param_names(none)) == 6, _chain_param_names(none)

    multi = ctpwa.DecayInfo(str(CONFIGS / "trans_multi.yml"))
    chains = _chain_param_names(multi)
    assert len(chains) == 2, chains
    # 留下的必须是基准道 rho0 的链（rho_a）
    assert all("rho_a" in n for n in chains), chains
    assert multi.nFreeParams() < none.nFreeParams()

    # 比值确实是 -1 / +1: 折叠链的链参数 = 基准链参数 × ratio（实乘）。
    # 由扩展向量变换直接验证（extendCouplingParams 需要 GPU, 这里只查结构）。
    # 结构层面: 两条被折叠链不再拥有独立参数, 这正是 ratios 生效的前提。
    assert len(chains) == 2


def test_pairwise_two_name_ratio_still_works():
    """两名字 + 单比值的旧写法（with_trans 用 -1）保持生效。"""
    info = ctpwa.DecayInfo(str(CONFIGS / "with_trans.yml"))
    chains = _chain_param_names(info)
    # R_Keta 实例 1 折进实例 0: 只留 K1_1410/K1_1680 各一条 + R_KK 两条
    assert len(chains) == 4, chains
    assert not any("Km+K1" in n for n in chains), chains


MULTI_TRANS_BLOCK = (
    "  trans:\n"
    "    # A 为基准; 后面每个名字对应一个比值(-1 / +1)\n"
    "    - [rho0, rhop, rhom]: [-1, 1]"
)


def _patched_multi(tmp_path, new_trans):
    text = (CONFIGS / "trans_multi.yml").read_text()
    assert MULTI_TRANS_BLOCK in text
    cfg = tmp_path / "trans_patch.yml"
    cfg.write_text(text.replace(MULTI_TRANS_BLOCK, new_trans))
    return ctpwa.DecayInfo(str(cfg))


def test_transitive_constraints_chain_fold(tmp_path):
    """跨约束链式折叠: [rho0,rhop] + [rhop,rhom] → 三电荷道同样并到 rho0。"""
    info = _patched_multi(tmp_path, "  trans:\n    - [rho0, rhop]: -1\n    - [rhop, rhom]: 1")
    assert len(_chain_param_names(info)) == 2, _chain_param_names(info)


def test_chain_fragment_names_pair(tmp_path):
    """约束名可以用链名/子链片段（R_rho0 ↔ 链 decay1_R_rho0）。

    实际分析里的常见写法是 [omega_Rx_rho0, omega_Rx_rhop, omega_Rx_rhom]:
    名字比中间态名长（带母链前缀）。旧实现只按"中间态名"配对, 这类写法会
    "匹配到链但一对都配不上"（静默不生效）——这里固化该回归。
    """
    info = _patched_multi(tmp_path, "  trans:\n    - [R_rho0, R_rhop, R_rhom]: [-1, 1]")
    chains = _chain_param_names(info)
    assert len(chains) == 2, chains          # 与 [rho0, rhop, rhom] 等价
    assert all("rho_a" in n for n in chains), chains


def test_chain_fragment_names_partial_scope(tmp_path):
    """只点名一部分链时, 只折叠那部分（其余保持独立）。"""
    info = _patched_multi(tmp_path, "  trans:\n    - [R_rho0, R_rhom]: 1")
    chains = _chain_param_names(info)
    assert len(chains) == 4, chains          # rho0+rhom 合并 → 2, rhop 仍独立 2


def test_chain_fragment_names_with_step_params():
    """链名片段写法 + 多 SL: 链耦合与 step 耦合一起折（真实 config 的形态）。

    trans_multi_step{,_none}.yml: J/ψ→η+R(1⁺)（2 个 SL）, R→π+ρ（2 个 SL）,
    三条电荷道。无约束 6 chain + 12 step = 18;约束 [R_rho0,R_rhop,R_rhom]
    把三条并成一条 → 2 chain + 4 step = 6。
    """
    none = ctpwa.DecayInfo(str(CONFIGS / "trans_multi_step_none.yml"))
    multi = ctpwa.DecayInfo(str(CONFIGS / "trans_multi_step.yml"))
    n_none, n_multi = [_chain_param_names(i) for i in (none, multi)]
    step_none = none.nFreeParams() - len(n_none)
    step_multi = multi.nFreeParams() - len(n_multi)
    assert (len(n_none), step_none) == (6, 12), (n_none, step_none)
    assert (len(n_multi), step_multi) == (2, 4), (n_multi, step_multi)
    assert multi.nFreeParams() == 6


def test_trans_cycle_is_broken_safely(tmp_path):
    """自相矛盾的成环约束不应产生无效振幅下标（只告警 + 断环）。"""
    info = _patched_multi(tmp_path, "  trans:\n    - [rho0, rhop]: -1\n    - [rhop, rho0]: -1")
    assert info.nAmplitudes() > 0
    assert 0 < info.nFreeParams() <= info.nAmplitudes()



ONE_TEMPLATE = """Particles:
  Jpsi: {{J: 1, P: -1, mass: 3.0969}}
  eta:  {{J: 0, P: -1, mass: 0.5478}}
  Kp:   {{J: 0, P: -1, mass: 0.4937}}
  Km:   {{J: 0, P: -1, mass: 0.4937}}

Data:
  order: [Kp, Km, eta]
  data: [dat, "./data/test_data.dat"]
  phsp: [dat, "./data/test_phsp.dat"]

DecayChains:
  decay1:
    Jpsi:
      - [eta, R_KK]
    R_KK: [Kp, Km]
    intermediates:
      R_KK:
        - [J: 1, P: -1]: [phi1020, NR_KK]

Resonances:
  phi1020:
    J: 1
    P: -1
    model: BWR
    parameters: [1.0195, 0.0045]
  NR_KK:
    J: 1
    P: -1
    model: ONE
{one_params}
"""


@pytest.mark.parametrize(
    "one_params",
    [
        "",                                    # 整段省略
        "    parameters: []",                  # 空列表
        "    parameters: [2.0]",               # 正常占位
        "    parameters: [1.0e6]",             # 离谱大值（旧实现会让 Bf 出 NaN）
        "    parameters: [0.1]",               # 远低于阈值（旧实现 sqrt(负) → NaN）
    ],
)
def test_one_mass_optional_and_any_value(tmp_path, one_params):
    """ONE 的占位质量可省略, 写任意值都能解析（不再强制 1 个参数）。"""
    cfg = tmp_path / "one.yml"
    cfg.write_text(ONE_TEMPLATE.format(one_params=one_params))
    info = ctpwa.DecayInfo(str(cfg))
    assert info.isValid()
    assert info.nAmplitudes() > 0
