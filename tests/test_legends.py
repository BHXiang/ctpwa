"""legends 图例规则解析测试。

config 的 legends 规则按"展开后的链"顺序逐条生成图例名（每个共振态一条）。

回归: 单行 + 多模式中间态（如 psip→gamma+R_chic1, R_chic1 三种子衰变）此前
按 mother 行分配规则，3 条规则全被第一行取走 → legends 只剩 1 条
（chicj2KKeta 实测: 只有 ['f_{2}(1525)\\eta']）；现在每条展开链按序取一条规则。
"""


def test_legends_three_rows(make_analysis):
    """with_trans: Jpsi 三行（每行一个中间态）→ 3 条规则 × 每个中间态的共振态数。"""
    ana = make_analysis("with_trans")
    legends = ana.getLegends()
    assert legends == [
        "\\phi(1020)\\eta",
        "\\phi(1680)\\eta",
        "K_{1}(1410)^{-}K^{+}",
        "K_{1}(1680)^{-}K^{+}",
        "K_{1}(1410)^{+}K^{-}",
        "K_{1}(1680)^{+}K^{-}",
    ]


def test_legends_multi_mode_intermediate(make_analysis):
    """legend_modes: 单行 + 三模式中间态 → 3 条规则 → 3 条图例，与分波顺序对齐。"""
    ana = make_analysis("legend_modes")
    legends = ana.getLegends()
    assert legends == [
        "f_{2}(1525)\\eta",
        "K_{1}(1410)^{-}K^{+}",
        "K_{1}(1410)^{+}K^{-}",
    ]


def test_deep_intermediate_avoids_chain_particles(make_analysis):
    """legend_modes: 深层多模式中间态(R_Keta)子衰变避开链中已出现的粒子。

    chic1→Kp+R_Keta 时 R_Keta→Km+eta (旁观者 Kp);
    chic1→Km+R_Keta 时 R_Keta→Kp+eta (旁观者 Km, 不能重复出现)。
    此前 BFS 对深层中间态恒用模式0 → 两条链都是 Km+eta。
    """
    ana = make_analysis("legend_modes")
    names = ana.getAmplitudeNames()
    keta = [n for n in names if "K1_1410" in n]
    assert any("K1_1410→Kp+eta" in n for n in keta), (
        f"应有一条 K1_1410→Kp+eta 链 (chic1→Km+R_Keta, 旁观者 Km), 实际: {keta}")
    assert any("K1_1410→Km+eta" in n for n in keta), (
        f"应有一条 K1_1410→Km+eta 链 (chic1→Kp+R_Keta, 旁观者 Kp), 实际: {keta}")


def test_chain_filter_by_resonance(make_analysis):
    """Constraints.chains 匹配"组合级完整路径串" → 只保留该共振态组合。

    chain_res_filter.yml = legend_modes + chains: ["chic1_eta_f2_1525"]:
    路径串形如 psip_gamma_chic1_chic1_eta_f2_1525_f2_1525_Kp_Km（每步 mother→daughters
    平铺、中间态名用选中共振态替换, = h_ 波名无前缀）;
    匹配到 → R_KK 只留 f2_1525(2 个 SL 分波), chic1 是链结构保留,
    两条 R_Keta 链(K1_1410)整体剔除, legends 也只生成 f2_1525 的图例。
    """
    ana = make_analysis("chain_res_filter")
    names = ana.getAmplitudeNames()
    assert len(names) == 2, f"只应留 f2_1525 的 2 个分波, 实际 {names}"
    assert all("f2_1525" in n for n in names), f"应全是 f2_1525 分波, 实际 {names}"
    assert ana.getLegends() == ["f_{2}(1525)\\eta"]
