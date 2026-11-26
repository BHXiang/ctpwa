#!/usr/bin/env python3
"""gen_phsp_mc.py — DIYMC P1：由 config.yml 生成**平坦 n 体相空间** MC（不需要 GPU）。

用途
----
1. closure / toy 研究里当 `phsp_truth` 的**理想**替代（平坦、无效率）；
2. 作为外部探测器模拟 / 重建的**输入**（重建之后得到带效率的 `phsp`）。

⚠ 语义提醒
-----------
* 本脚本产出的是**平坦相空间**（GENBOD 内部用 `wt` 舍选，`u < wt`），**不等于**带效率的
  `phsp_truth`：后者是同一批探测器 MC 在 truth 级的版本，可能带 truth 级 fiducial 选择。
  用平坦 PS 去顶替它，会改变"效率"的定义（∫|A|²ε/∫|A|²）。
* 若用于 closure（toy 真值是否 NLL 极小），样本量必须能**分辨最窄的共振**：
  上游实测 4.5 MeV 宽的 φ(1020) 只给 10k phsp 时，toy 真值不是 NLL 极小（拟合把宽度顶到
  边界），把 phsp 加到 200k 才恢复。本仓库的共振宽度是 MeV 级 → 建议**百万级**。

输出格式（与 `fit.py` 的读取约定一致）
--------------------------------------
* `.dat`：每个末态粒子一行 `E px py pz`（12 位小数、空格分隔），即 **每事件 n_final 行**；
* `.root`：`TTree "mc"`，每个末态一个 `TLorentzVector` 分支（分支名 = `Data.order` 里的名字）；
* `.npy`：`(n, n_final, 4)` float64 数组（给 `npy2dat.py` 或别的工具）。

用法
----
    python gen_phsp_mc.py --config config.yml -n 100000 --seed 42 -o phsp_flat.dat
    python gen_phsp_mc.py --config config.yml -n 5000000 --chunk 500000 -o phsp_flat.dat
    python gen_phsp_mc.py --config config.yml -n 200000 --format npy -o phsp_flat
    python gen_phsp_mc.py --config config.yml -n 200000 --weighted --format root -o phsp_flat.root

随机性
------
`--seed` 默认取**当前 Unix 时间戳**（与 `fit.py` 一致）→ **每次运行都不同**，避免"忘了改 seed"
导致重复样本；要复现请把启动打印的 `[seed]`（或 `--stats-json` 里的 `seed`）用 `--seed N` 传回来。

依赖：`numpy` + `ctpwa`（≥0.3.15 提供 `PhaseSpaceGenerator`）。**不需要 GPU，也不读数据文件**
（只解析 config 的 `Particles` / `Data.order` / `DecayChains`）。

注意：不要在仓库根目录运行（本仓库自带的 `ctpwa/` 门面包会遮蔽已安装的 wheel）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np

# ============================================================================
# 本脚本自带的极简工具（不 import 任何本仓库文件，保证单文件可运行）
# ============================================================================


def load_data_order(config_path):
    """从 config 文本里读 `Data.order`（本仓库 config 风格）。

    返回 list[str]；找不到返回 None（调用方给出警告并跳过顺序自检）。
    """
    pat = re.compile(r"^\s*order\s*:\s*\[(.*?)\]\s*$")
    with open(config_path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                return [x.strip() for x in m.group(1).split(",") if x.strip()]
    return None


def load_particle_masses(config_path):
    """从 config 文本里读 `Particles` 的 `mass`（用于质量壳自检）。

    返回 dict[name] = mass；解析不了的名字直接缺席（调用方跳过对应检查）。
    """
    out = {}
    pat = re.compile(r"^\s*([A-Za-z_][\w]*)\s*:\s*\{([^}]*)\}")
    with open(config_path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            body = m.group(2)
            mm = re.search(r"mass\s*:\s*([0-9.eE+-]+)", body)
            if mm:
                try:
                    out[m.group(1)] = float(mm.group(1))
                except ValueError:
                    pass
    return out


def write_dat(events, path, append=False):
    """写 `.dat`：每事件 n_final 行、每行 `E px py pz`（12 位小数、空格分隔）。

    `append=True` 时追加（供 `--chunk` 分块生成大样本）。
    """
    ev = np.asarray(events, dtype=np.float64)
    if ev.ndim != 3 or ev.shape[-1] != 4:
        raise ValueError(f"events 形状应为 [N, n_final, 4]，实际 {ev.shape}")
    flat = ev.reshape(-1, 4)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        np.savetxt(f, flat, delimiter=" ", fmt="%.12f")
    return flat.shape[0]


def write_npy(events, path, weights=None):
    """写 `.npy`：`(n, n_final, 4)`；有 weights 时另存 `<path>.weights.npy`。"""
    ev = np.asarray(events, dtype=np.float64)
    np.save(path if str(path).endswith(".npy") else str(path) + ".npy", ev)
    if weights is not None:
        np.save(str(path) + ".weights.npy", np.asarray(weights, dtype=np.float64))


def physics_check(events, masses, mother_mass, n_check=2000, atol=1e-6):
    """四动量守恒 + 质量壳自检（抽样 `n_check` 个事件）。

    返回 (ok, report_lines)。`masses` 按 `Data.order` 给（缺失项会被跳过）。
    """
    ev = np.asarray(events, dtype=np.float64)
    k = min(int(n_check), ev.shape[0])
    if k <= 0:
        return True, ["(空样本，跳过物理自检)"]
    s = ev[:k]
    lines = []
    ok = True
    dE = float(np.abs(s[..., 0].sum(axis=1) - mother_mass).max())
    dp = float(np.abs(s[..., 1:].sum(axis=1)).max())
    lines.append(
        f"四动量守恒: max|ΣE-m| = {dE:.2e}, max|Σp| = {dp:.2e} (抽样 {k} 事件)"
    )
    if dE > atol or dp > atol:
        ok = False
        lines.append("!! 守恒被破坏（超过 atol）")
    m2 = s[..., 0] ** 2 - (s[..., 1:] ** 2).sum(axis=-1)
    bad = 0
    checked = 0
    for i, nm in enumerate(masses or []):
        if nm is None or i >= s.shape[1]:
            continue
        checked += 1
        dm = float(np.abs(np.sqrt(np.clip(m2[:, i], 0, None)) - nm).max())
        lines.append(f"质量壳 {i}(m={nm:g}): max|m_rec-m| = {dm:.2e}")
        if dm > atol:
            bad += 1
    if bad:
        ok = False
        lines.append(f"!! {bad}/{checked} 个末态质量壳被破坏")
    return ok, lines


# ============================================================================
# 主流程
# ============================================================================


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """与默认一致，但 `--seed` 不追加 "(default: None)"——它的真实默认是当前时间戳，
    已写在 help 文本里（和 fit.py 的写法一致）。"""

    def _get_help_string(self, action):
        if action.dest == "seed":
            return action.help
        return super()._get_help_string(action)


def build_parser():
    p = argparse.ArgumentParser(
        description="DIYMC P1：由 config 生成平坦 n 体相空间 MC（无需 GPU）",
        formatter_class=_HelpFormatter,
        epilog=(
            "⚠ 产出的是平坦相空间，不等于带效率的 phsp_truth（见脚本头部说明）；"
            "用于 closure 时样本量需能分辨最窄共振（建议百万级）。"
        ),
    )
    p.add_argument("--config", default="config.yml", help="config.yml 路径")
    p.add_argument("-n", "--nevents", type=int, required=True, help="目标事件数")
    p.add_argument(
        "-o", "--output", required=True, help="输出文件（dat/root 带扩展名；npy 可省）"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；默认取当前 Unix 时间戳（每次运行不同，避免重复）"
        "——要复现请把启动打印的 [seed] 传回来",
    )
    p.add_argument(
        "--format",
        choices=("dat", "root", "npy"),
        default=None,
        help="输出格式；默认按 -o 的扩展名（.dat/.root/.npy），无扩展名时用 dat",
    )
    p.add_argument(
        "--mother", default="", help="母粒子名（默认从 config 的 DecayChains 推）"
    )
    p.add_argument(
        "--mother-mass", type=float, default=None, help="覆盖母粒子质量 (GeV)"
    )
    p.add_argument(
        "--weighted",
        action="store_true",
        help="导出 GENBOD 权重（dat 只有动量列，故 weighted 仅对 root/npy 有意义）",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="分块生成的事件数（>0 时逐块生成并追加写；仅 dat/npy，用于千万级样本）",
    )
    p.add_argument(
        "--force", action="store_true", help="末态顺序与 Data.order 不一致时仍继续"
    )
    p.add_argument(
        "--no-check", action="store_true", help="跳过我方物理自检（守恒/质量壳）"
    )
    p.add_argument("--check-events", type=int, default=2000, help="物理自检抽样事件数")
    return p


def main():
    args = build_parser().parse_args()

    # 默认 seed = 当前 Unix 时间戳（与 fit.py 一致）：每次运行不同，避免重复样本；
    # 要复现就把这里打印的 seed 用 --seed 传回来。
    if args.seed is None:
        args.seed = int(time.time())
        print(
            f"[gen_phsp_mc] [seed] 未指定 --seed → 用当前时间戳 seed={args.seed}；"
            f"复现本作业请加 --seed {args.seed}（分块时第 k 块用 seed+k）"
        )
    else:
        print(f"[gen_phsp_mc] [seed] seed={args.seed}（--seed 指定）")

    if not os.path.isfile(args.config):
        sys.exit(f"[gen_phsp_mc] config 不存在: {args.config}")
    if args.nevents <= 0:
        sys.exit("[gen_phsp_mc] -n 必须为正")

    fmt = args.format
    if fmt is None:
        ext = os.path.splitext(args.output)[1].lower()
        fmt = ext.lstrip(".") if ext in (".dat", ".root", ".npy") else "dat"
    if args.weighted and fmt == "dat":
        sys.exit("[gen_phsp_mc] weighted 需要 --format root 或 npy（dat 只有动量列）")
    if args.chunk and fmt == "root":
        sys.exit("[gen_phsp_mc] --chunk 仅支持 dat/npy（root 需一次性写入）")

    import ctpwa  # 延迟导入：--help 不需要后端

    order = load_data_order(args.config)
    pmass = load_particle_masses(args.config)

    gen = ctpwa.PhaseSpaceGenerator(args.config, args.mother)
    if args.mother_mass is not None:
        gen.setMotherMass(float(args.mother_mass))
    finals = list(gen.finalStates())
    print(f"[gen_phsp_mc] config     : {args.config}")
    print(f"[gen_phsp_mc] mother     : {gen.mother()} (m={gen.motherMass():g})")
    print(f"[gen_phsp_mc] finalStates: {finals}  (nFinal={gen.nFinal()})")
    print(f"[gen_phsp_mc] Data.order : {order}")
    print(f"[gen_phsp_mc] weightMax  : {gen.weightMax():g}")
    if order is not None and order != finals:
        msg = (
            f"[gen_phsp_mc] 末态顺序 {finals} 与 config 的 Data.order {order} 不一致："
            f"生成的 dat 会被 fit.py 按 Data.order 解读！"
        )
        if args.force:
            print("WARNING " + msg + "（--force，继续）")
        else:
            sys.exit(msg + "（如确认无误可加 --force）")

    masses = [pmass.get(nm) for nm in finals]
    if any(m is None for m in masses):
        miss = [nm for nm, m in zip(finals, masses) if m is None]
        print(
            f"[gen_phsp_mc] 注意: config 里没有 {'/'.join(miss)} 的 mass → 质量壳检查跳过这些"
        )
        masses = [m for m in masses]
    masses_onesided = masses  # 保持与 finals 同序；None 表示跳过

    out = args.output
    t0 = time.time()
    n_done = 0
    w_all = [] if args.weighted else None
    sample_ev = None
    chunk = int(args.chunk) if args.chunk else int(args.nevents)
    while n_done < args.nevents:
        m = min(chunk, args.nevents - n_done)
        seed = int(args.seed) + n_done
        if args.weighted:
            ev, w = gen.generateWeighted(m, seed=seed)
        else:
            ev = gen.generate(m, seed=seed)
            w = None
        ev_np = ev.numpy() if hasattr(ev, "numpy") else np.asarray(ev)
        if sample_ev is None:
            sample_ev = ev_np
        if args.weighted:
            w_np = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
            w_all.append(w_np)
        if fmt == "dat":
            write_dat(ev_np, out, append=(n_done > 0))
        elif fmt == "npy":
            # 分块时逐块存 `<out>.<i>.npy`，避免一次性 hold 住大数组
            if args.chunk:
                write_npy(
                    ev_np,
                    f"{os.path.splitext(out)[0]}.{n_done}.npy",
                    w_np if args.weighted else None,
                )
            else:
                write_npy(ev_np, out, w_np if args.weighted else None)
        else:  # root（只允许一次性）
            gen.write(ev, out, w if args.weighted else None)
        n_done += m
        print(
            f"[gen_phsp_mc]   {n_done}/{args.nevents} 事件  ({time.time() - t0:.1f}s)"
        )
    dt = time.time() - t0

    # ---- 落盘自检：行数 ----
    nlines = None
    if fmt == "dat":
        with open(out, encoding="utf-8") as f:
            nlines = sum(1 for _ in f)
        expect = args.nevents * gen.nFinal()
        print(
            f"[gen_phsp_mc] dat 行数: {nlines}（期望 {expect} = {args.nevents} × nFinal）"
        )
        if nlines != expect:
            sys.exit("[gen_phsp_mc] 行数不符，请检查输出")

    # ---- 物理自检 ----
    if not args.no_check and sample_ev is not None and sample_ev.size:
        ok, lines = physics_check(
            sample_ev,
            masses_onesided,
            float(gen.motherMass()),
            n_check=args.check_events,
        )
        for ln in lines:
            print("[gen_phsp_mc]   " + ln)
        if not ok:
            sys.exit("[gen_phsp_mc] 物理自检未通过")

    print(
        f"[gen_phsp_mc] 完成: {args.nevents} 事件 → {out} ({fmt}), "
        f"seed={args.seed}, 耗时 {dt:.1f}s"
    )


if __name__ == "__main__":
    main()
