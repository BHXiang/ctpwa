"""L0 结构测试（秒级，每次提交必跑）。

验证 config 解析、参数数量、trans 折叠等结构性不变量。
不涉及数值计算（不调用 getNLL），因此可以快速反复运行。

经典回归案例（本测试要抓住的）：
- vspace 模式覆盖了 buildWithTrans 的 trans 折叠 → 参数数量错误
- getParamNames 返回了被折叠振幅的名字 → 名字与参数数量不匹配
"""

import pytest


# ============================================================
# config 加载
# ============================================================

@pytest.mark.parametrize("config_key", ["simple", "no_trans", "with_trans"])
def test_config_loads(make_analysis, config_key):
    """三种 config 都能正确初始化。"""
    ana = make_analysis(config_key)
    assert ana.isValid(), f"{config_key}.yml 初始化失败"
    assert ana.getNVector() > 0, "自由耦合参数数应为正"


def test_config_load_nonexistent_fails():
    """不存在的 config 不应崩溃。"""
    import ctpwa

    ana = ctpwa.analysis("./configs/does_not_exist.yml")
    # 允许失败但不崩溃（isValid() == False 或抛异常都可接受）


# ============================================================
# 参数数量一致性
# ============================================================

def test_n_params_consistency(make_analysis):
    """getNParams() == 2 * n_vector + n_theta。"""
    for key in ["simple", "no_trans", "with_trans"]:
        ana = make_analysis(key)
        expected = 2 * ana.getNVector() + ana.getNFreeTheta()
        assert ana.getNParams() == expected, f"{key}: getNParams 不一致"


def test_param_names_count_matches(make_analysis):
    """参数名数量必须等于实际自由参数数（每个复数参数一个名字）。

    回归案例 1：trans 折叠后 getParamNames 仍返回全部振幅名（含折叠的）。
    回归案例 2：trans 折叠后 resonanceParamNames 返回重复的 theta 名。
    """
    for key in ["simple", "no_trans", "with_trans"]:
        ana = make_analysis(key)
        names = ana.getParamNames()
        expected = ana.getNVector() + ana.getNFreeTheta()
        assert len(names) == expected, (
            f"{key}: 参数名数 {len(names)} != 自由参数数 {expected}"
            f" (nVector={ana.getNVector()}, nTheta={ana.getNFreeTheta()})"
        )


def test_param_names_unique_with_trans(make_analysis):
    """with_trans: 参数名必须唯一（trans 折叠后每个参数一个名字）。

    回归案例：K1_1410_mass 出现两次（被折叠 chain 的重复 slot 名）。
    """
    ana = make_analysis("with_trans")
    names = ana.getParamNames()
    assert len(set(names)) == len(names), (
        f"with_trans: 参数名有重复: {[n for n in names if names.count(n) > 1]}"
    )


def test_no_folded_names_in_params(make_analysis):
    """with_trans 时，被折叠链（Km 反冲）的振幅名不应出现在参数名中。

    回归案例：vspace 模式 getParamNames 返回全部 28 个振幅名（含 10 个折叠的）。
    振幅名格式为拓扑名（如 "Jpsi→Km+K1_1410_SL(...)"），
    R_Keta_1（Km 反冲）的振幅名以 "Jpsi→Km+" 开头。
    """
    ana = make_analysis("with_trans")
    names = ana.getParamNames()
    for n in names:
        assert not n.startswith("Jpsi→Km+"), (
            f"被折叠的 R_Keta_1（Km 反冲）振幅名不应出现: {n}"
        )
    # 正向检查：Kp 反冲的 R_Keta_0 振幅名应出现
    kp_names = [n for n in names if n.startswith("Jpsi→Kp+")]
    assert len(kp_names) > 0, "R_Keta_0（Kp 反冲）振幅名应出现在参数名中"


# ============================================================
# trans 折叠
# ============================================================

def test_trans_folding_reduces_n_free(make_analysis):
    """有 trans 时自由耦合参数数必须小于无 trans 时。

    no_trans: 6 个振幅（2 R_KK + 2 R_Keta_0 + 2 R_Keta_1）
    with_trans: 4 个（R_Keta_1 折叠进 R_Keta_0）
    """
    ana_no = make_analysis("no_trans")
    ana_tr = make_analysis("with_trans")
    assert ana_tr.getNVector() < ana_no.getNVector(), (
        f"trans 应减少自由参数数: no_trans={ana_no.getNVector()}, "
        f"with_trans={ana_tr.getNVector()}"
    )


def test_trans_folding_exact_count(make_analysis):
    """with_trans 的具体参数数量（与 config 结构强相关）。

    no_trans: R_KK×2 + R_Keta_0×2 + R_Keta_1×2 = 6 个耦合参数
    with_trans: R_KK×2 + R_Keta_0×2 = 4 个（R_Keta_1 折叠）
    """
    ana_no = make_analysis("no_trans")
    ana_tr = make_analysis("with_trans")
    assert ana_no.getNVector() == 6, (
        f"no_trans 应为 6 个自由耦合参数，实际 {ana_no.getNVector()}"
    )
    assert ana_tr.getNVector() == 4, (
        f"with_trans 应为 4 个自由耦合参数，实际 {ana_tr.getNVector()}"
    )


# ============================================================
# 共振态自由参数
# ============================================================

def test_free_res_params(make_analysis):
    """自由共振态参数信息 [init, lower, upper] 形状正确。"""
    ana = make_analysis("simple")
    info = ana.getFreeResParams()  # [3, n_res]
    assert info.shape[0] == 3, "自由共振态参数应为 [3, n_res]"
    n_theta = ana.getNFreeTheta()
    assert info.shape[1] == n_theta, "getFreeResParams 与 getNFreeTheta 不一致"
    # 下界 < 上界
    for i in range(n_theta):
        assert info[1, i] < info[2, i], f"参数 {i} 下界应小于上界"
