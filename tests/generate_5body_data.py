#!/usr/bin/env python3
"""5 粒子合成数据生成器（深层衰变链测试用，DAT 格式）。

每行 "E px py pz"（GeV），每 5 行一个事件，对应
Data.order = [pA, pB, pC, pD, gamma]（deep_4pi0 / ppbar_2pi0 两个 config
共用：数值与粒子名无关，仅按 order 取列）。

数据为能量守恒框架下的"伪事件"——四动量随机但 E >= 粒子静质量
（不做 5 体相空间约束），足够读入/振幅构建，不追求物理意义。

用法: python3 tests/generate_5body_data.py [--n_events 200]
产出: tests/data/5body.dat（固定 seed，结果可复现）
"""

import argparse
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
MASSES = [0.135, 0.135, 0.135, 0.135, 0.0]  # 4 个 π⁰（或 π⁺π⁻）+ gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_events", type=int, default=200)
    args = parser.parse_args()

    rng = np.random.default_rng(20260831)
    rows = []
    for _ in range(args.n_events):
        for m in MASSES:
            p = rng.uniform(0.1, 1.0, size=3)
            e = np.sqrt((p * p).sum() + m * m) + rng.uniform(0.05, 0.3)
            rows.append((e, *p))
    out = TESTS_DIR / "data" / "5body.dat"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.array(rows), fmt="%.16f")
    print(f"written {len(rows)} rows ({args.n_events} events) -> {out}")


if __name__ == "__main__":
    main()
