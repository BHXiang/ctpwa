#!/usr/bin/env python3
"""gen_toy_mc.py — DIYMC P2/P3：用**拟合后的振幅**产生 toy MC，可直接产出 pseudo-data。

用途
----
* 玩具研究（toy study）：拟合偏差 / 覆盖度 / 误差标定；
* 生成与真实数据同分布（形状 = |M|²）的**信号 toy**，可加 Poisson 涨落、可叠加真本底
  （按 config 的 `bkg` + `bkg_weights` 加权抽样）→ 直接当 `data` 用（配合 `--write-config`）；
* `--mode weighted` 得到带权样本（快），`unweighted` 得到等权样本（AR 舍选）。

⚠ 语义提醒
-----------
* 样本量必须能分辨**最窄的共振**，否则 toy 真值不是 NLL 极小（上游实测：4.5 MeV 的 φ 用
  10k phsp 时拟合把宽度顶到边界，200k 才恢复）。本仓库共振宽度 MeV 级 → 建议**百万级**；
* 本脚本**不做拟合**：`--check-distribution` 只验证"生成分布 = |M|² 加权分布"（自洽性）；
  "toy 真值是否 NLL 极小"需要拿 `--write-config` 的派生 config 去跑 `fit.py`。

前置条件
--------
`ctpwa`（≥0.3.15，含 `generateMC`/`getIntensity`/`writeMC`）+ backend（**GPU 推荐**；
CPU 后端也能跑但慢 20–60×）。config 的 `data`/`phsp`/`bkg` 文件必须存在（构造 analysis 要读它们）。
`--params-from` 的 `best_params.pt` 是 **CPU 张量**（fit.py 用 `.cpu()` 保存），脚本会按当前
后端自动搬到 `cuda`/`cpu`（C++ 侧要求 `params` 与后端同设备）。

输出
----
* `.dat`：每事件 n_final 行 `E px py pz`（与 fit.py 读取约定一致，可直接当 `data`）；
* `.root`：`TTree "mc"`（weighted 时带权重分支）；
* `.npy`：`(n, n_final, 4)`（weighted 时另存 `.weights.npy`）。
* `--stats-json`：seed / 参数 md5 / 事件数 / bkg 产额 / generateMC 的 stats。

用法
----
    # 5 万信号 toy + Poisson + 真本底 → pseudo-data（并派生 config、写 stats）
    python gen_toy_mc.py --config config.yml --params-from results/best_params.pt \
        -n 50000 --seed 42 --poisson --nobs-from-data --bkg \
        -o toy_data.dat --write-config config_toy.yml --stats-json toy.json

    # 纯信号、等权、10 万事件
    python gen_toy_mc.py --config config.yml --params-from results/best_params.pt \
        -n 100000 --seed 7 -o toy_sig.dat

    # 带权样本（快）：保留权重，只能写 root/npy
    python gen_toy_mc.py --config config.yml --params-from results/best_params.pt \
        -n 500000 --mode weighted --format root -o toy_w.root

    # 自洽性检查：生成分布 vs |M|² 加权平坦样本
    python gen_toy_mc.py --config config.yml --params-from results/best_params.pt \
        -n 50000 --check-distribution -o toy.dat

随机性
------
`--seed` 默认取**当前 Unix 时间戳**（与 `fit.py` 一致）→ **每次运行都不同**，避免"忘了改 seed"
导致重复 toy；要复现请把启动打印的 `[seed]`（或 `--stats-json` 里的 `seed`）用 `--seed N` 传回来。

注意：不要在仓库根目录运行（本仓库自带的 `ctpwa/` 门面包会遮蔽已安装的 wheel）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

# ============================================================================
# 本脚本自带的极简工具（不 import 任何本仓库文件，保证单文件可运行）
# ============================================================================


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _find_data_section(lines):
    """返回 config 文本里 `Data:` 段的 [start, end) 行号区间；找不到返回 None。"""
    i0 = next((i for i, ln in enumerate(lines) if re.match(r"^Data:\s*$", ln)), None)
    if i0 is None:
        return None
    i1 = len(lines)
    for i in range(i0 + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln[0].isspace() and not ln.lstrip().startswith("#"):
            i1 = i
            break
    return (i0, i1)


def load_data_order(config_path):
    """读 `Data.order`（本仓库 config 风格）；找不到返回 None。"""
    pat = re.compile(r"^\s*order\s*:\s*\[(.*?)\]\s*$")
    with open(config_path, encoding="utf-8") as f:
        for ln in f:
            m = pat.match(ln)
            if m:
                return [x.strip() for x in m.group(1).split(",") if x.strip()]
    return None


def load_dat_path(config_path, key):
    """读 `Data` 段里 `key` 的 dat 文件路径（支持单行 flow 与块式）；非 dat/无该键返回 None。"""
    with open(config_path, encoding="utf-8") as f:
        lines = f.readlines()
    span = _find_data_section(lines)
    if span is None:
        return None
    for i in range(span[0] + 1, span[1]):
        m = re.match(r"^\s*" + re.escape(key) + r"\s*:(.*)$", lines[i])
        if not m:
            continue
        rest = m.group(1).strip()
        if rest:
            if "dat" not in rest:
                return None
            mm = re.search(r'"([^"]+)"', rest)
            return mm.group(1) if mm else None
        for j in range(i + 1, min(i + 8, span[1])):
            if re.match(r"^\s*(dat|ROOT)\s*,", lines[j]):
                if "dat" not in lines[j]:
                    return None
                mm = re.search(r'"([^"]+)"', lines[j + 1] if j + 1 < span[1] else "")
                return mm.group(1) if mm else None
        return None
    return None


def derive_config_dat_path(src_path, dst_path, key, new_value):
    """把 config 里 `Data` 段 `key:` 的 dat 路径改写成 new_value（支持 flow/块式）。

    只做**针对性文本替换**，不做 YAML round-trip（config 里 `[J: 1/2, P: -1]` 这种写法
    会让 ruamel/pyyaml 崩）。匹配不到就报错，不猜。
    """
    with open(src_path, encoding="utf-8") as f:
        lines = f.readlines()
    span = _find_data_section(lines)
    if span is None:
        sys.exit(f"[gen_toy_mc] {src_path} 里没有顶层 `Data:` 段，无法派生 config")
    done = False
    for i in range(span[0] + 1, span[1]):
        m = re.match(r"^(\s*)" + re.escape(key) + r"\s*:(.*)$", lines[i])
        if not m:
            continue
        indent, rest = m.group(1), m.group(2).rstrip("\n")
        if rest.strip() and '"' in rest:  # 单行 flow：[dat, "path"]
            lines[i] = (
                f"{indent}{key}:"
                + re.sub(r'"[^"]*"', f'"{new_value}"', rest, count=1)
                + "\n"
            )
            done = True
            break
        j = next(
            (
                k
                for k in range(i + 1, min(i + 8, span[1]))
                if re.match(r"^\s*dat\s*,", lines[k])
            ),
            None,
        )
        if j is None:
            sys.exit(
                f"[gen_toy_mc] {src_path} 的 `{key}:` 不是已知的 dat 写法，"
                "无法自动改写（请手工改）"
            )
        k = next(
            (k for k in range(j + 1, min(j + 4, span[1])) if '"' in lines[k]), None
        )
        if k is None:
            sys.exit(f"[gen_toy_mc] {src_path} 的 `{key}:` 块里找不到带引号的路径行")
        ind2 = re.match(r"^(\s*)", lines[k]).group(1)
        lines[k] = f'{ind2}"{new_value}",\n'
        done = True
        break
    if not done:
        sys.exit(
            f"[gen_toy_mc] {src_path} 的 `Data` 段里没有 `{key}:`，无法派生 config"
        )
    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return dst_path


def read_dat(path, nfinal):
    """读 `.dat`（每事件 nfinal 行、每行 4 个数）→ `[N, nfinal, 4]`。"""
    a = np.loadtxt(path, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] != 4:
        sys.exit(
            f"[gen_toy_mc] {path} 每行应为 4 个数（E px py pz），实际 {a.shape[1]}"
        )
    if a.shape[0] % nfinal:
        sys.exit(
            f"[gen_toy_mc] {path} 行数 {a.shape[0]} 不是 n_final={nfinal} 的整数倍"
        )
    return a.reshape(-1, nfinal, 4)


def count_dat_events(path, nfinal):
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    if n % nfinal:
        sys.exit(f"[gen_toy_mc] {path} 行数 {n} 不是 n_final={nfinal} 的整数倍")
    return n // nfinal


def write_dat(events, path, append=False):
    """写 `.dat`：每事件 n_final 行 `E px py pz`（12 位小数、空格分隔）。"""
    ev = np.asarray(events, dtype=np.float64)
    if ev.ndim != 3 or ev.shape[-1] != 4:
        raise ValueError(f"events 形状应为 [N, n_final, 4]，实际 {ev.shape}")
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        np.savetxt(f, ev.reshape(-1, 4), delimiter=" ", fmt="%.12f")
    return ev.shape[0] * ev.shape[1]


def m2_pair(events, i, j):
    """两末态的不变质量平方 m²(i,j)。"""
    s = events[:, i, :] + events[:, j, :]
    return s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2 - s[:, 3] ** 2


def backend_device(bi):
    """当前后端对应的 torch 设备字符串。

    CUDA 后端要求传给 `generateMC` / `getIntensity` 的 `params` 在 **CUDA** 上
    （C++ 侧 `TORCH_CHECK(ctpwa_is_cuda_like(params))`），而 `best_params.pt` 是
    fit.py 用 `.cpu()` 存的 CPU 张量 → 必须显式搬过去；CPU 后端则要 CPU 张量。
    """
    return "cuda" if str((bi or {}).get("backend", "")).lower() == "cuda" else "cpu"


def as_torch(x):
    """numpy / 序列 → `torch.Tensor`（CPU, float64）；`None` 原样返回。

    `analysis.writeMC()` / `PhaseSpaceGenerator.write()` 的第一个形参是 `torch::Tensor`
    （pybind11 不做 numpy→torch 自动转换），而本脚本内部一律用 numpy 处理事件数组
    → 写 root 前必须显式转换（2026-10-29 第四轮 GPU 冒烟踩过：
    `TypeError: writeMC(): incompatible function arguments ... events: torch.Tensor`）。
    """
    import torch

    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64)


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
        description="DIYMC P2/P3：用拟合后的振幅产生 toy MC / pseudo-data",
        formatter_class=_HelpFormatter,
        epilog=(
            "⚠ 样本量要能分辨最窄共振（建议百万级），否则 toy 真值不是 NLL 极小；"
            "本脚本不做拟合。"
        ),
    )
    p.add_argument("--config", default="config.yml", help="config.yml 路径")
    p.add_argument(
        "--params-from",
        default="results/best_params.pt",
        help="拟合参数（best_params.pt，[Re.., Im.., θ] float64）",
    )
    p.add_argument(
        "-n",
        "--nevents",
        type=int,
        default=None,
        help="目标事例数（mode=unweighted 时为输出事件数；"
        "--poisson 且未给 -n 时默认 = nobs）",
    )
    p.add_argument("-o", "--output", required=True, help="输出文件（dat/root/npy）")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；默认取当前 Unix 时间戳（每次运行不同，避免重复）"
        "——要复现请把启动打印的 [seed] 传回来",
    )
    p.add_argument(
        "--mode",
        choices=("unweighted", "weighted"),
        default="unweighted",
        help="unweighted=AR 舍选等权；weighted=全收+权重（快）",
    )
    p.add_argument(
        "--batch", type=int, default=100000, help="每批平坦事例数（传给 generateMC）"
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="平坦事例数上限（安全检查；None=不限）",
    )
    p.add_argument(
        "--format",
        choices=("dat", "root", "npy"),
        default=None,
        help="输出格式；默认按 -o 的扩展名",
    )
    p.add_argument(
        "--poisson",
        action="store_true",
        help="按目标事件数抽成等权 pseudo-data（加权模式按权重有放回抽）",
    )
    p.add_argument(
        "--nobs", type=int, default=None, help="--poisson 的目标事件数（信号部分）"
    )
    p.add_argument(
        "--nobs-from-data",
        action="store_true",
        default=None,
        help="--poisson 的目标事件数取 config 的 data 文件事件数（默认行为）",
    )
    p.add_argument(
        "--allow-short",
        action="store_true",
        help="可用信号不足 nobs 时取 min(nobs, 可用) 并警告（默认报错）",
    )
    p.add_argument(
        "--bkg",
        dest="bkg",
        action="store_true",
        default=None,
        help="叠加 config 的 bkg（按 bkg_weights 加权抽样）",
    )
    p.add_argument("--no-bkg", dest="bkg", action="store_false", help="不叠加本底")
    p.add_argument(
        "--bkg-scale",
        type=float,
        default=1.0,
        help="本底产额缩放（Σw × scale 作为 Poisson 均值）",
    )
    p.add_argument(
        "--no-shuffle", action="store_true", help="信号与本底拼接后不打乱（默认打乱）"
    )
    p.add_argument(
        "--write-config",
        default=None,
        help="派生一份 config：把 data 指向本 toy（便于直接跑 fit.py）",
    )
    p.add_argument("--stats-json", default=None, help="把统计信息写成 JSON")
    p.add_argument(
        "--check-distribution",
        action="store_true",
        help="自洽性检查：生成分布 vs |M|² 加权平坦样本（会多生成一份加权样本）",
    )
    p.add_argument(
        "--check-events", type=int, default=20000, help="自洽性检查用的事件数"
    )
    p.add_argument("--check-pair", default="0,1", help="自洽性检查投影的两个末态下标")
    p.add_argument("--check-bins", type=int, default=25, help="自洽性检查直方图 bin 数")
    return p


def main():
    args = build_parser().parse_args()
    t_start = time.time()

    # 默认 seed = 当前 Unix 时间戳（与 fit.py 一致）：每次运行不同，避免"忘了改 seed"导致重复
    # toy；要复现就把这里打印的 seed 用 --seed 传回来（--stats-json 也会记录）。
    if args.seed is None:
        args.seed = int(time.time())
        print(
            f"[gen_toy_mc] [seed] 未指定 --seed → 用当前时间戳 seed={args.seed}；"
            f"复现本作业请加 --seed {args.seed}"
        )
    else:
        print(f"[gen_toy_mc] [seed] seed={args.seed}（--seed 指定）")

    if not os.path.isfile(args.config):
        sys.exit(f"[gen_toy_mc] config 不存在: {args.config}")
    if not os.path.isfile(args.params_from):
        sys.exit(f"[gen_toy_mc] 参数文件不存在: {args.params_from}")

    fmt = args.format
    if fmt is None:
        ext = os.path.splitext(args.output)[1].lower()
        fmt = ext.lstrip(".") if ext in (".dat", ".root", ".npy") else "dat"

    import torch  # 延迟导入：--help 不需要

    order = load_data_order(args.config) or []
    nfinal_cfg = len(order) if order else None

    # ---- 便宜的前置检查（都在构造 analysis 之前，失败得快）----
    data_path = load_dat_path(args.config, "data")
    data_events = None
    if data_path and os.path.isfile(data_path) and nfinal_cfg:
        data_events = count_dat_events(data_path, nfinal_cfg)
    nobs = args.nobs
    if args.poisson and nobs is None:
        if data_events is None:
            sys.exit(
                "[gen_toy_mc] --poisson 需要 --nobs，或 config 的 data 文件可读"
                "（用于 --nobs-from-data）"
            )
        nobs = data_events
    if args.poisson and nobs is not None and args.nevents is None:
        args.nevents = int(nobs)
    if args.nevents is None or args.nevents <= 0:
        sys.exit("[gen_toy_mc] 需要 -n（--poisson 未给 -n 时按 nobs 推断）")
    if args.mode == "weighted" and fmt == "dat" and not args.poisson:
        sys.exit(
            "[gen_toy_mc] mode=weighted 且未加 --poisson 时不能写 dat（dat 无权重列）："
            "请用 --format root/npy，或加 --poisson 转成等权 pseudo-data"
        )

    params = torch.load(args.params_from, weights_only=True)
    if not torch.is_tensor(params):
        sys.exit(f"[gen_toy_mc] {args.params_from} 不是 tensor")
    params = params.detach().to(dtype=torch.float64).reshape(-1)

    # ---- 本底（读 dat，不需要后端）----
    bkg_ev = bkg_w = None
    bkg_path = load_dat_path(args.config, "bkg")
    bkgw_path = load_dat_path(args.config, "bkg_weights")
    use_bkg = args.bkg if args.bkg is not None else (bkg_path is not None)
    if use_bkg:
        if not (bkg_path and os.path.isfile(bkg_path) and nfinal_cfg):
            print("[gen_toy_mc] WARNING --bkg 但 config 的 bkg 不可用 → 跳过本底")
            use_bkg = False
        else:
            bkg_ev = read_dat(bkg_path, nfinal_cfg)
            if bkgw_path and os.path.isfile(bkgw_path):
                bkg_w = np.loadtxt(bkgw_path, dtype=np.float64).reshape(-1)
                if bkg_w.size != bkg_ev.shape[0]:
                    sys.exit(
                        f"[gen_toy_mc] bkg_weights 行数 {bkg_w.size} != bkg 事件数 "
                        f"{bkg_ev.shape[0]}"
                    )
            else:
                bkg_w = np.ones(bkg_ev.shape[0], dtype=np.float64)

    # ---- 构造 analysis（需要 backend，较慢）----
    import ctpwa

    ana = ctpwa.analysis(args.config)
    nvec = int(ana.getNVector())
    ntheta = int(ana.getNFreeTheta())
    nparams = 2 * nvec + ntheta
    try:
        bi = ctpwa.build_info()
    except Exception:
        bi = {"backend": ctpwa.backend()}
    print(f"[gen_toy_mc] backend={bi.get('backend')} build_info={bi}")
    if str(bi.get("backend", "")).lower() != "cuda":
        print(
            "[gen_toy_mc] WARNING 非 CUDA 后端（CPU 慢 20~60×；GPU 作业请设 CTPWA_DEVICE=cuda）"
        )
    print(f"[gen_toy_mc] 参数维度: 2×{nvec} + {ntheta} = {nparams}；Data.order={order}")

    if params.numel() != nparams:
        sys.exit(
            f"[gen_toy_mc] 参数长度 {params.numel()} != 期望 {nparams}"
            "（是不是用了别的 config 的 best_params.pt？）"
        )
    # CUDA 后端要求 params 在 CUDA 上（best_params.pt 是 CPU 张量）
    dev = backend_device(bi)
    params = params.to(dev)
    print(
        f"[gen_toy_mc] 参数: {args.params_from} (md5 {md5_file(args.params_from)[:12]}, "
        f"Re_0={float(params[0]):g}, Im_0={float(params[nvec]):g}, device={dev})"
    )
    if nobs is not None:
        print(
            f"[gen_toy_mc] --poisson: nobs={nobs}"
            f"{' (来自 data 文件)' if args.nobs is None else ' (--nobs)'}, "
            f"data 事件数={data_events}"
        )
    if use_bkg:
        print(
            f"[gen_toy_mc] 本底: {bkg_ev.shape[0]} 事件, Σw={bkg_w.sum():.6g}, "
            f"scale={args.bkg_scale:g}"
        )

    # ---- 生成信号 ----
    t0 = time.time()
    res = ana.generateMC(
        int(args.nevents),
        params,
        seed=int(args.seed),
        mode=args.mode,
        batch=int(args.batch),
        max_trials=args.max_trials,
    )
    ev = np.asarray(res["events"].numpy(), dtype=np.float64)
    wt = (
        np.asarray(res["weights"].numpy(), dtype=np.float64)
        if res["weights"].numel()
        else None
    )
    stats = dict(res["stats"])
    print(
        f"[gen_toy_mc] generateMC({args.nevents}, mode={args.mode}): "
        f"{ev.shape[0]} 事件, {time.time() - t0:.1f}s"
    )
    print(f"[gen_toy_mc]   stats: {stats}")
    if stats.get("truncated"):
        print(
            "[gen_toy_mc] WARNING stats.truncated=True：平坦事例被 max_trials 截断，"
            "输出事件数少于请求值"
        )
    if nfinal_cfg and ev.shape[1] != nfinal_cfg:
        sys.exit(
            f"[gen_toy_mc] 生成 n_final={ev.shape[1]} 与 Data.order 长度 {nfinal_cfg} 不符"
        )
    if stats.get("truncated"):
        pass  # 已在上面警告

    rng = np.random.default_rng(int(args.seed) + 20260101)

    # ---- 信号 → pseudo-data ----
    if args.poisson:
        want = int(nobs)
        if args.mode == "unweighted":
            if want > ev.shape[0]:
                if args.allow_short:
                    print(
                        f"[gen_toy_mc] WARNING 需要 {want} 但只有 {ev.shape[0]}（--allow-short）"
                    )
                    want = ev.shape[0]
                else:
                    sys.exit(
                        f"[gen_toy_mc] --poisson 需要 {want} 事件，但只生成了 {ev.shape[0]}"
                        "；请增大 -n（或加 --allow-short）"
                    )
            idx = rng.choice(ev.shape[0], size=want, replace=False)
            sig, sig_w = ev[idx], None
        else:  # weighted → 按权重有放回抽样
            p = wt / wt.sum()
            idx = rng.choice(ev.shape[0], size=want, replace=True, p=p)
            sig, sig_w = ev[idx], None
        print(f"[gen_toy_mc] Poisson: nobs={nobs} → 信号 {sig.shape[0]} 事件（等权）")
    else:
        sig, sig_w = ev, wt
        print(
            f"[gen_toy_mc] 不做 Poisson：输出全部 {sig.shape[0]} 事件"
            f"{'（带权重）' if sig_w is not None else '（等权）'}"
        )
    sig_pure = sig  # 纯信号（自洽性检查用；下面叠加本底/打乱不会改到它）

    # ---- 叠加本底 ----
    n_bkg = 0
    if use_bkg:
        lam = max(0.0, float(args.bkg_scale) * float(bkg_w.sum()))
        n_bkg = int(rng.poisson(lam))
        if n_bkg > 0:
            p = bkg_w / bkg_w.sum()
            bi_idx = rng.choice(bkg_ev.shape[0], size=n_bkg, replace=True, p=p)
            sig = np.concatenate([sig, bkg_ev[bi_idx]], axis=0)
            if sig_w is not None:
                sig_w = np.concatenate([sig_w, np.ones(n_bkg)], axis=0)
        print(f"[gen_toy_mc] 本底: Poisson(Σw×scale={lam:.6g}) → {n_bkg} 事件")
    else:
        print("[gen_toy_mc] 不加本底")

    if not args.no_shuffle and sig.shape[0] > 1:
        perm = rng.permutation(sig.shape[0])
        sig = sig[perm]
        if sig_w is not None:
            sig_w = sig_w[perm]

    # ---- 写盘 ----
    n_total = int(sig.shape[0])
    if fmt == "dat":
        nlines = write_dat(sig, args.output)
        with open(args.output, encoding="utf-8") as f:
            n_check = sum(1 for _ in f)
        print(
            f"[gen_toy_mc] dat 已写: {args.output} 行数={n_check}"
            f"（期望 {nlines} = {n_total}×{sig.shape[1]}）"
        )
        if n_check != nlines:
            sys.exit("[gen_toy_mc] 行数不符")
    elif fmt == "npy":
        np.save(
            args.output
            if str(args.output).endswith(".npy")
            else str(args.output) + ".npy",
            sig,
        )
        if sig_w is not None:
            np.save(str(args.output) + ".weights.npy", sig_w)
        print(f"[gen_toy_mc] npy 已写: {args.output}")
    else:  # root
        # writeMC 的 events 形参是 torch::Tensor（pybind11 不自动转 numpy）
        ana.writeMC(as_torch(sig), args.output, as_torch(sig_w))
        print(f"[gen_toy_mc] root 已写: {args.output}")

    # ---- 派生 config ----
    if args.write_config:
        derive_config_dat_path(
            args.config, args.write_config, "data", os.path.abspath(args.output)
        )
        print(
            f"[gen_toy_mc] 派生 config: {args.write_config}（data → {os.path.abspath(args.output)}）"
        )

    # ---- 自洽性检查（生成分布 vs |M|² 加权）----
    check = None
    if args.check_distribution:
        try:
            i, j = (int(x) for x in str(args.check_pair).split(","))
        except Exception:
            sys.exit("[gen_toy_mc] --check-pair 需形如 0,1")
        n_probe = max(1000, int(args.check_events))
        t0 = time.time()
        wtres = ana.generateMC(
            n_probe,
            params,
            seed=int(args.seed) + 9999,
            mode="weighted",
            batch=int(args.batch),
        )
        ev_wt = np.asarray(wtres["events"].numpy(), dtype=np.float64)
        w_wt = np.asarray(wtres["weights"].numpy(), dtype=np.float64)
        src = sig_pure[:n_probe] if sig_pure.shape[0] >= n_probe else sig_pure
        x_ar = m2_pair(src, i, j)
        x_wt = m2_pair(ev_wt, i, j)
        bins = np.linspace(
            min(x_ar.min(), x_wt.min()),
            max(x_ar.max(), x_wt.max()),
            int(args.check_bins) + 1,
        )
        h_ar, _ = np.histogram(x_ar, bins=bins)
        h_wt, _ = np.histogram(x_wt, bins=bins, weights=w_wt)
        h_ar = h_ar / max(h_ar.sum(), 1)
        h_wt = h_wt / max(h_wt.sum(), 1)
        dmax = float(np.abs(h_ar - h_wt).max())
        check = {
            "pair": [i, j],
            "n_ar": int(x_ar.size),
            "n_weighted": int(x_wt.size),
            "max_abs_diff": dmax,
            "bins": int(args.check_bins),
        }
        print(
            f"[gen_toy_mc] 自洽性检查 m²({i},{j}): max|Δ|={dmax:.4f}"
            f"（AR {x_ar.size} vs 加权 {x_wt.size}；{time.time() - t0:.1f}s）"
            f" → {'OK' if dmax < 0.05 else '偏大，检查样本量/参数'}"
        )

    # ---- stats json ----
    dt = time.time() - t_start
    info = {
        "script": "gen_toy_mc.py",
        "config": os.path.abspath(args.config),
        "config_md5": md5_file(args.config),
        "params_from": os.path.abspath(args.params_from),
        "params_md5": md5_file(args.params_from),
        "mode": args.mode,
        "seed": int(args.seed),
        "n_requested": int(args.nevents),
        "stats": stats,
        "poisson": bool(args.poisson),
        "nobs": nobs,
        "data_events": data_events,
        "n_signal": int(n_total - n_bkg),
        "n_bkg": int(n_bkg),
        "n_total": n_total,
        "bkg_scale": float(args.bkg_scale),
        "bkg": bool(use_bkg),
        "output": os.path.abspath(args.output),
        "format": fmt,
        "derived_config": args.write_config,
        "check_distribution": check,
        "elapsed_s": round(dt, 2),
        "backend": bi.get("backend"),
    }
    if args.stats_json:
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)
        print(f"[gen_toy_mc] stats: {args.stats_json}")
    print(
        f"[gen_toy_mc] 完成: {n_total} 事件 → {args.output} ({fmt}), "
        f"seed={args.seed}, 耗时 {dt:.1f}s"
    )


if __name__ == "__main__":
    main()
