#!/usr/bin/env python3
"""合成 4 体测试数据生成器（DAT 格式）: psip → gamma + Kp + Km + eta。

用于 legend_modes.yml（单行 + 多模式中间态）的结构性测试:
- 递归两体衰变: psip → gamma + X, X → Kp + Y, Y → Km + eta（随机质量/方向）
- 相空间均匀抽样（flat phase space），物理上允许即可，不追求真实分布
- 数据/相空间/本底 三份文件，粒子按 Data.order 循环: [gamma, Kp, Km, eta]

用法:
    python3 tests/generate_4body_data.py [--n_data 1000] [--n_phsp 10000]

产出:
    tests/data/test_4body_data.dat, test_4body_phsp.dat, test_4body_sideband.dat
"""

import argparse
import math
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"

M_PSIP = 3.0686   # 与 legend_modes.yml 的 psip 质量一致
M_GAMMA = 0.0
M_K = 0.4937
M_ETA = 0.5478


def two_body(m, m1, m2, rng):
    """母粒子静止系中 m -> m1 + m2，随机方向，返回两个 4-动量 (E, px, py, pz)。"""
    lam = (m**2 - (m1 + m2)**2) * (m**2 - (m1 - m2)**2)
    p = math.sqrt(max(lam, 0.0)) / (2 * m)
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    p1 = n * p
    E1 = math.sqrt(m1**2 + p**2)
    E2 = math.sqrt(m2**2 + p**2)
    return np.array([E1, *p1]), np.array([E2, *(-p1)])


def boost(p4, parent_p4, parent_m):
    """把母粒子静止系中的 4-动量 p4 boost 到 lab 系（母粒子 lab 4-动量为 parent_p4）。"""
    g = parent_p4[0] / parent_m
    b = parent_p4[1:] / parent_p4[0]
    b2 = float(b @ b)
    if b2 < 1e-12:
        return p4
    bdotp = float(b @ p4[1:])
    E = g * (p4[0] + bdotp)
    p = p4[1:] + (g - 1.0) * bdotp / b2 * b + g * b * p4[0]
    return np.array([E, *p])


def four_body_flat(N, seed):
    """psip 静止系 4 体相空间抽样: 返回 N 组 (gamma, Kp, Km, eta) 四动量。"""
    rng = np.random.default_rng(seed)
    events = []
    while len(events) < N:
        # 随机子系质量: X -> Kp + Y, Y -> Km + eta
        mX = rng.uniform(2 * M_K + M_ETA + 0.05, M_PSIP - 0.001)
        mY = rng.uniform(M_K + M_ETA + 0.02, mX - M_K - 0.02)

        p_gamma, p_X = two_body(M_PSIP, M_GAMMA, mX, rng)   # psip 系
        p_Kp, p_Y = two_body(mX, M_K, mY, rng)              # X 静止系
        p_Km, p_eta = two_body(mY, M_K, M_ETA, rng)         # Y 静止系

        p_Kp_lab = boost(p_Kp, p_X, mX)
        p_Y_lab = boost(p_Y, p_X, mX)
        p_Km_lab = boost(p_Km, p_Y_lab, mY)
        p_eta_lab = boost(p_eta, p_Y_lab, mY)

        Etot = p_gamma[0] + p_Kp_lab[0] + p_Km_lab[0] + p_eta_lab[0]
        if abs(Etot - M_PSIP) > 1e-6:
            continue  # 能量不守恒，丢弃
        events.append([p_gamma, p_Kp_lab, p_Km_lab, p_eta_lab])
    return events


def write_dat(path, events):
    """每行 "E px py pz"（GeV），粒子按 Data.order 循环: [gamma, Kp, Km, eta]。"""
    lines = []
    for ev in events:
        for p4 in ev:
            lines.append(f"{p4[0]:.8f} {p4[1]:.8f} {p4[2]:.8f} {p4[3]:.8f}")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(events)} events, {len(lines)} lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_data", type=int, default=1000)
    ap.add_argument("--n_phsp", type=int, default=10000)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    write_dat(DATA_DIR / "test_4body_data.dat", four_body_flat(args.n_data, seed=42))
    write_dat(DATA_DIR / "test_4body_phsp.dat", four_body_flat(args.n_phsp, seed=43))
    write_dat(DATA_DIR / "test_4body_sideband.dat", four_body_flat(300, seed=44))


if __name__ == "__main__":
    main()
