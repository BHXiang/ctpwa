"""L0 结构测试（秒级，每次提交必跑）。

验证 config 解析、参数数量、trans 折叠等结构性不变量。
不涉及数值计算（不调用 getNLL），因此可以快速反复运行。

经典回归案例（本测试要抓住的）：
- vspace 模式覆盖了 buildWithTrans 的 trans 折叠 → 参数数量错误
- getParamNames 返回了被折叠振幅的名字 → 名字与参数数量不匹配
"""

import pytest
import torch


# ============================================================
# config 加载
# ============================================================

@pytest.mark.parametrize("config_key", ["simple", "no_trans", "with_trans", "simple_ext"])
def test_config_loads(make_analysis, config_key):
    """所有 config 都能正确初始化。"""
    ana = make_analysis(config_key)
    assert ana.isValid(), f"{config_key}.yml 初始化失败"
    assert ana.getNVector() > 0, "自由耦合参数数应为正"


def test_external_resonance_equivalent(make_analysis):
    """Resonances 引用外部文件 (simple_ext) 与内联 (simple) 必须完全等价。"""
    ana_inline = make_analysis("simple")
    ana_ext = make_analysis("simple_ext")
    assert ana_ext.isValid(), "simple_ext.yml 初始化失败"
    assert ana_ext.getNVector() == ana_inline.getNVector(), "nVector 不一致"
    assert ana_ext.getNFreeTheta() == ana_inline.getNFreeTheta(), "nFreeTheta 不一致"
    assert ana_ext.getNParams() == ana_inline.getNParams(), "nParams 不一致"
    # 共振态解析结果（自由参数初值矩阵）应逐元素一致
    torch.testing.assert_close(
        ana_ext.getFreeResParams(), ana_inline.getFreeResParams(),
        msg="外部共振态文件解析结果与内联不一致",
    )


def test_ls_filter_reduces_waves(make_analysis):
    """sl 分波白名单应减少 SL 组合数（4 → 1），并保持其余结构一致。"""
    ana_base = make_analysis("ls_filter")
    ana_filt = make_analysis("ls_filter_constrained")
    assert ana_base.getNVector() == 4, f"基线应为 4 个分波, 实际 {ana_base.getNVector()}"
    assert ana_filt.getNVector() == 1, f"约束后应为 1 个分波, 实际 {ana_filt.getNVector()}"
    # 约束只影响 SL 组合，不改变共振态自由参数
    assert ana_filt.getNFreeTheta() == ana_base.getNFreeTheta(), "free theta 不应变化"


def test_chain_filter_file(make_analysis):
    """Constraints.chains 外部文件过滤: 匹配 chain1 → 与 simple 完全等价。"""
    ana_simple = make_analysis("simple")
    ana_filt = make_analysis("simple_filtered")
    assert ana_filt.isValid(), "simple_filtered.yml 初始化失败"
    assert ana_filt.getNVector() == ana_simple.getNVector(), "过滤后 nVector 应与 simple 一致"
    assert ana_filt.getNFreeTheta() == ana_simple.getNFreeTheta(), "free theta 应与 simple 一致"


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


def test_deep_trans_folding_exact_count(make_analysis):
    """深层 trans 约束（默认 chain×step / FREEPARAMS 参数化）:
    R_Keta 是 R_chic1 的子中间态（链名不含 R_Keta_N 子串）。

    约束 [R_Keta_0, R_Keta_1]: 1 按"中间态实例序号"语义匹配:
    16 条振幅，R_Keta_1 折叠进 R_Keta_0 → chain×step 参数化下 2 chain + 2 step = 4 参数；
    若未折叠应为 8。比率 1.0 回归: 折叠判定不能依赖 ratio≠1（trans 值可为 1）。
    （vspace 逐振幅参数化下折叠后为 8 —— 见 c22fd07 时期的语义）
    """
    ana = make_analysis("with_deep_trans")
    assert ana.getNVector() == 4, (
        f"with_deep_trans(chain×step) 应为 4 个自由耦合参数（R_Keta_1 折叠, 4/8），"
        f"实际 {ana.getNVector()}"
    )


# ============================================================
# DeviceManager（L0 结构测试）
# ============================================================

def test_detect_has_devices():
    """至少检测到 1 个可用 GPU。"""
    import ctpwa

    dm = ctpwa.DeviceManager()
    dm.detect()
    assert dm.numDevices() >= 1, "应有至少 1 个 CUDA GPU"
    assert dm.hasDevices()


def test_device_info_fields():
    """设备属性（名称/算力/显存）应有效。

    注意: 全量测试时 session 级 make_analysis 缓存会占用大量显存，
    空闲显存可能为 0（GPU 满）——这里只验证数值合法，不要求 free > 0。
    """
    import ctpwa

    dm = ctpwa.DeviceManager()
    dm.detect()
    for i in range(dm.numDevices()):
        assert len(dm.deviceName(i)) > 0, f"GPU {i} 名称为空"
        cc_major, cc_minor = dm.deviceComputeCapability(i)
        assert cc_major >= 1, f"GPU {i} 算力异常"
        total = dm.deviceMemoryTotal(i)
        free = dm.deviceMemoryFree(i)
        assert total > 0, f"GPU {i} 总显存为 0"
        assert 0 <= free <= total, f"GPU {i} 空闲显存数值异常 ({free}/{total})"


def test_estimate_positive():
    """内存估算应为正数，且事件数越多需求越大。"""
    import ctpwa

    dm = ctpwa.DeviceManager()
    dm.detect()
    gpu, other = dm.estimateMemory(1000, 4, 3, 1, 3, True)
    assert gpu > 0 and other > 0, "内存估算应为正"
    gpu2, _ = dm.estimateMemory(10000, 4, 3, 1, 3, True)
    assert gpu2 > gpu, "事件数越多内存需求越大"


def test_complex_precision():
    """编译精度一致性：complexSize 与 compiledPrecision 匹配。

    精度编译时确定（ctComplex = cuComplex 8B / cuDoubleComplex 16B），
    运行时 setComplexPrecision 不改变实际内存布局。
    """
    import ctpwa

    dm = ctpwa.DeviceManager()
    precision = dm.compiledPrecision()
    assert precision in ("float", "double"), f"未知精度: {precision}"
    expected_size = 8 if precision == "float" else 16
    assert dm.complexSize() == expected_size, (
        f"compiledPrecision={precision} 但 complexSize={dm.complexSize()}"
    )


def test_capacity_small_fits():
    """小数据应通过容量预检。"""
    import ctpwa

    dm = ctpwa.DeviceManager()
    dm.detect()
    # (overall, device, buffer, required, available)
    status, dev, buf, need, avail = dm.checkCapacity([1000], 4, 3, 1, 3, True)
    assert status == 0, f"小数据应 OK, got status={status} buf={buf}"


def test_capacity_large_fails():
    """超大数据应 FAIL（任一 GPU 无法承载）。"""
    import ctpwa

    dm = ctpwa.DeviceManager()
    dm.detect()
    # 10 亿事件 × 28 振幅，远超任何消费级显存
    status, dev, buf, need, avail = dm.checkCapacity([1000000000], 28, 3, 2, 3, True)
    assert status == 2, f"超大数据应 FAIL, got status={status}"
    assert buf, "FAIL 时应给出超限 buffer 名"


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
