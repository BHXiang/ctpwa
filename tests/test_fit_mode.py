"""fit_mode 构造参数：0=FREEPARAMS (chain×step, 默认), 1=VSPACE (逐振幅)。

回归防线：fit_mode 只在构造期 initializeDecayChains() 被消费（params_ 的耦合
映射在那里一次性构建），构造后 setFitMode 改不动参数化——只能改
getNVector/getParamNames 的语义。因此 VSPACE 必须通过
``ctpwa.analysis(config, 1)`` 选择，且默认参数保持 0 以兼容旧脚本。
"""
import ctpwa


def test_default_fit_mode_is_zero():
    """后向兼容：不传 fit_mode 与显式 0 等价（旧脚本 analysis(config) 不受影响）。"""
    a = ctpwa.analysis("configs/simple.yml")
    b = ctpwa.analysis("configs/simple.yml", 0)
    assert a.getFitMode() == 0 and b.getFitMode() == 0
    assert a.getNVector() == b.getNVector()
    assert a.getParamNames() == b.getParamNames()


def test_vspace_changes_parameterization():
    """多步链 with_deep_trans: VSPACE 逐振幅参数与 chain×step 确实不同。

    该配置 mode 0 = 4 个 chain×step 参数, mode 1 = 8 个逐振幅参数——若
    setFitMode 只在构造后调用（旧行为），两模式会给出同一个数。
    """
    a0 = ctpwa.analysis("configs/with_deep_trans.yml", 0)
    a1 = ctpwa.analysis("configs/with_deep_trans.yml", 1)
    assert a0.getFitMode() == 0 and a1.getFitMode() == 1
    assert a1.getNVector() != a0.getNVector(), (
        "VSPACE 未生效：两模式自由参数数相同 "
        f"(mode0={a0.getNVector()}, mode1={a1.getNVector()})"
    )
    assert len(a1.getParamNames()) == a1.getNVector() + a1.getNFreeTheta()


def test_post_construction_setfitmode_warns(capfd):
    """构造后 setFitMode 到不同模式给出警告（不再静默失效）。"""
    a = ctpwa.analysis("configs/simple.yml")   # 构造期模式 0
    assert a.getFitMode() == 0
    a.setFitMode(1)
    # 兼容旧行为：语义标记仍更新，但实际参数化冻结在构造期
    assert a.getFitMode() == 1
    err = capfd.readouterr().err
    assert "setFitMode(1)" in err and "无法改变参数化" in err
