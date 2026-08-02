#!/usr/bin/env python3
"""真实数据子集准备脚本（在集群上执行一次）。

从 /besfs10 的 ROOT 文件截取事件，转成 ctpwa DAT 格式（E px py pz 每行）。

用法（集群上）:
    python3 tests/prepare_real_data.py \
        --data /besfs10/.../cut_data.root \
        --phsp /besfs10/.../cut_phsp.root \
        --sideband /besfs10/.../cut_sideband.root \
        --n_events 10000

产出:
    tests/data/real_data.dat      (10000 事件)
    tests/data/real_phsp.dat      (10000 事件)
    tests/data/real_sideband.dat  (10000 事件)

随后生成真实数据 config:
    python3 tests/prepare_real_data.py --write-config

说明:
- 集群的 ROOT 与 Python 版本一致（3.14），可直接用 ROOT 库读取。
- 本地开发机 ROOT 版本不匹配，此脚本应在集群上运行。
"""

import argparse
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"

# 粒子顺序与 config 的 Data.order 一致
PARTICLES = ["Kp", "Km", "eta"]


def read_root_tree(filename, treename, branches, n_events):
    """用 ROOT 读 TTree 的 TLorentzVector 分支。"""
    import ROOT

    f = ROOT.TFile.Open(filename, "READ")
    if not f:
        raise RuntimeError(f"无法打开 {filename}")
    tree = f.Get(treename)
    if not tree:
        raise RuntimeError(f"找不到 TTree {treename}")

    lvs = [ROOT.TLorentzVector() for _ in branches]
    for lv, br in zip(lvs, branches):
        tree.SetBranchAddress(br, lv)

    entries = min(tree.GetEntries(), n_events)
    events = []
    for i in range(entries):
        tree.GetEntry(i)
        ev = []
        for lv in lvs:
            ev.append((lv.E(), lv.Px(), lv.Py(), lv.Pz()))
        events.append(ev)
    f.Close()
    return events


def write_dat(events, path):
    """写 DAT 文件: 每行 "E px py pz"，粒子循环。"""
    lines = []
    for ev in events:
        for e, px, py, pz in ev:
            lines.append(f"{e:.8f} {px:.8f} {py:.8f} {pz:.8f}")
    path.write_text("\n".join(lines) + "\n")


def convert(fileinfo, out_name, n_events):
    """fileinfo: [type, filename, treename, branch1, branch2, ...]"""
    ftype, filename = fileinfo[0], fileinfo[1]
    if ftype.lower() == "root":
        branches = fileinfo[3:]
        events = read_root_tree(filename, fileinfo[2], branches, n_events)
    else:
        # dat 输入: 直接截取前 n_events*3 行
        lines = Path(filename).read_text().splitlines()[: n_events * 3]
        events = []
        for i in range(0, len(lines), 3):
            ev = []
            for j in range(3):
                e, px, py, pz = map(float, lines[i + j].split())
                ev.append((e, px, py, pz))
            events.append(ev)
    out_path = DATA_DIR / out_name
    write_dat(events, out_path)
    print(f"  {out_name}: {len(events)} 事件 -> {out_path}")


def write_real_config():
    """生成 tests/configs/real_data.yml（指向真实数据）。"""
    cfg = TESTS_DIR / "configs" / "real_data.yml"
    if cfg.exists():
        print(f"{cfg} 已存在，跳过")
        return
    # 从 no_trans.yml 复制并替换数据路径
    src = (TESTS_DIR / "configs" / "no_trans.yml").read_text()
    dst = src.replace(
        '"./data/test_data.dat"', '"./data/real_data.dat"'
    ).replace(
        '"./data/test_phsp.dat"', '"./data/real_phsp.dat"'
    ).replace(
        '"./data/test_sideband.dat"', '"./data/real_sideband.dat"'
    )
    cfg.write_text(dst)
    print(f"生成 {cfg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="数据 ROOT 文件")
    parser.add_argument("--phsp", help="相空间 ROOT 文件")
    parser.add_argument("--sideband", help="sideband ROOT 文件")
    parser.add_argument("--n_events", type=int, default=10000)
    parser.add_argument("--write-config", action="store_true",
                        help="仅生成 real_data.yml，不转换数据")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.write_config:
        write_real_config()
        return

    if not (args.data and args.phsp):
        print("需要 --data 和 --phsp 参数（--sideband 可选）")
        parser.print_help()
        sys.exit(1)

    # 默认 ROOT 分支结构: [type, filename, tree, br1, br2, br3]
    tree = "anatree"
    convert(["./", args.data, tree] + PARTICLES, "real_data.dat", args.n_events)
    convert(["./", args.phsp, tree] + PARTICLES, "real_phsp.dat", args.n_events)
    if args.sideband:
        convert(["./", args.sideband, tree] + PARTICLES, "real_sideband.dat",
                args.n_events)

    write_real_config()
    print("完成。测试框架将自动使用真实数据（L2 不再 skip）。")


if __name__ == "__main__":
    main()
