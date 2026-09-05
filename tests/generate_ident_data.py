#!/usr/bin/env python3
"""全同粒子测试数据生成器：J/psi → π⁰ π⁰ η 与 J/psi → π⁰ π⁰ π⁰（相空间均匀抽样）。

用法:
    python3 tests/generate_ident_data.py
产出:
    tests/data/ident2_{data,phsp,sideband}.dat   (pi01, pi02, eta)
    tests/data/ident3_{data,phsp,sideband}.dat   (pi01, pi02, pi03)
    tests/data/ident2_swapped.dat                (pi01↔pi02 交换后的数据，对称性检验用)
固定 seed，可复现。
"""
import argparse
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent

M_JPSI = 3.0969
M_PI = 0.13498
M_ETA = 0.5478


def three_body_flat(N, seed, m1, m2, m3):
    """三体相空间均匀抽样：返回 N 组 (p1, p2, p3) 四动量 (E, px, py, pz)。"""
    rng = np.random.default_rng(seed)
    M2 = M_JPSI**2
    results = []
    while len(results) < N:
        m12 = rng.uniform(m1 + m2, M_JPSI - m3)
        m23 = rng.uniform(m2 + m3, M_JPSI - m1)
        m13_sq = M2 + m1**2 + m2**2 + m3**2 - m12**2 - m23**2 - m1**2 - m2**2 - m3**2 + 2*m1*m2 + 2*m2*m3 - 2*m1*m3  # noqa
        m13 = np.sqrt(max(m13_sq, (m1 + m3)**2))
        if m13 < (m1 + m3) or m13 > (M_JPSI - m2):
            continue
        # 在 J/psi 系构造：先 (23) 子系
        E1 = (M2 + m1**2 - m23**2) / (2 * M_JPSI)
        p1 = np.sqrt(max(E1**2 - m1**2, 0.0))
        # 1 沿 +z
        pz1 = p1
        E23 = M_JPSI - E1
        # 23 子系内：2 与 3 背对背
        E2_23 = (m23**2 + m2**2 - m3**2) / (2 * m23)
        p23 = np.sqrt(max(E2_23**2 - m2**2, 0.0))
        cos_theta = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        # 23 子系内 2 的方向（相对子系动量方向，取均匀球面）
        # 简化：先构造 2 在子系内的方向（子系静止系中均匀），再 boost
        p2_23 = np.array([p23 * np.sin(np.arccos(cos_theta)) * np.cos(phi),
                          p23 * np.sin(np.arccos(cos_theta)) * np.sin(phi),
                          p23 * cos_theta])
        p3_23 = -p2_23
        E2_23_ = np.sqrt(m2**2 + p23**2)
        E3_23 = np.sqrt(m3**2 + p23**2)
        # boost 23 → J/psi 系（沿子系动量方向 pz1 方向；子系动量 = -p1 z）
        beta = -pz1 / E23  # 23 子系在 J/psi 系的速率（沿 z）
        gamma = 1.0 / np.sqrt(1 - beta**2)
        def boost(p, E):
            return np.array([p[0], p[1],
                             gamma * (p[2] + beta * E),
                             gamma * (E + beta * p[2])])
        p2 = boost(p2_23, E2_23_)
        p3 = boost(p3_23, E3_23)
        # 事件守恒检查
        p1v = np.array([0.0, 0.0, pz1, E1])
        psum = p1v + p2 + p3
        if abs(psum[3] - M_JPSI) > 1e-6:
            continue
        # 随机整体旋转（保持物理等价）
        def rand_rot(rng):
            u = rng.uniform(-1, 1, 3)
            while np.linalg.norm(u) < 1e-6:
                u = rng.uniform(-1, 1, 3)
            u /= np.linalg.norm(u)
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(0, 2 * np.pi)
            return u, theta, phi
        u, theta, phi = rand_rot(rng)
        K = np.array([[np.cos(theta) + u[0]**2 * (1 - np.cos(theta)),
                       u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta),
                       u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta)],
                      [u[1] * u[0] * (1 - np.cos(theta)) + u[2] * np.sin(theta),
                       np.cos(theta) + u[1]**2 * (1 - np.cos(theta)),
                       u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta)],
                      [u[2] * u[0] * (1 - np.cos(theta)) - u[1] * np.sin(theta),
                       u[2] * u[1] * (1 - np.cos(theta)) + u[0] * np.sin(theta),
                       np.cos(theta) + u[2]**2 * (1 - np.cos(theta))]])
        for p in (p1v, p2, p3):
            p[:3] = K @ p[:3]
        results.append((p1v, p2, p3))
    return results


def write_dat(path, events):
    with open(path, 'w') as f:
        for evt in events:
            for p in evt:
                f.write(f"{p[3]:.10e} {p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_data", type=int, default=1000)
    ap.add_argument("--n_phsp", type=int, default=10000)
    args = ap.parse_args()
    data_dir = TESTS_DIR / "data"

    # J/psi → pi01 pi02 eta
    ev = three_body_flat(args.n_data, 42, M_PI, M_PI, M_ETA)
    write_dat(data_dir / "ident2_data.dat", ev)
    write_dat(data_dir / "ident2_phsp.dat", three_body_flat(args.n_phsp, 43, M_PI, M_PI, M_ETA))
    write_dat(data_dir / "ident2_sideband.dat", three_body_flat(100, 44, M_PI, M_PI, M_ETA))
    # 交换 pi01 ↔ pi02（对称性检验）
    write_dat(data_dir / "ident2_swapped.dat", [(e[1], e[0], e[2]) for e in ev])

    # J/psi → pi01 pi02 pi03
    ev3 = three_body_flat(args.n_data, 52, M_PI, M_PI, M_PI)
    write_dat(data_dir / "ident3_data.dat", ev3)
    write_dat(data_dir / "ident3_phsp.dat", three_body_flat(args.n_phsp, 53, M_PI, M_PI, M_PI))
    write_dat(data_dir / "ident3_sideband.dat", three_body_flat(100, 54, M_PI, M_PI, M_PI))
    print("ident data written")


if __name__ == "__main__":
    main()
