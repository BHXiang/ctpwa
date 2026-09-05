#!/usr/bin/env python3
"""重子 CP 共轭对测试数据生成器（DAT 格式）。

物理: J/psi -> pbar + N+ (N+ -> p eta)（链 A 视角）
数据只含终态粒子 [p, pbar, eta] 的四动量。
"""

import argparse
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent

M_JPSI = 3.0969
M_P = 0.9383
M_ETA = 0.5479
M_N = 2.0


def two_body(m, m1, m2, rng):
    """m 静止系中两体衰变: 返回 (P1, P2) 四动量 (E, px, py, pz)。"""
    qsq = (m**2 - (m1 + m2) ** 2) * (m**2 - (m1 - m2) ** 2)
    q = np.sqrt(max(qsq, 0.0)) / (2.0 * m)
    cost = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    sinth = np.sqrt(max(1.0 - cost**2, 0.0))
    p = q * np.array([cost, sinth * np.cos(phi), sinth * np.sin(phi)])
    E1 = np.sqrt(q**2 + m1**2)
    E2 = np.sqrt(q**2 + m2**2)
    return np.array([E1, p[0], p[1], p[2]]), np.array([E2, -p[0], -p[1], -p[2]])


def boost(p, beta):
    """把静止系四动量 boost 到实验室系。"""
    b2 = beta @ beta
    if b2 < 1e-12:
        return p
    gamma = 1.0 / np.sqrt(1.0 - b2)
    pv = p[1:]
    bdotp = beta @ pv
    E = gamma * (p[0] + bdotp)
    p_new = pv + (gamma - 1.0) * (bdotp / b2) * beta + gamma * beta * p[0]
    return np.array([E, p_new[0], p_new[1], p_new[2]])


def gen_event(rng):
    """单事件: Jpsi 静止 -> pbar + N; N -> p + eta。返回 (p, pbar, eta) 四动量。"""
    P_lab = np.array([M_JPSI, 0.0, 0.0, 0.0])
    # 顶层: Jpsi -> pbar + N（子A=pbar 为终态侧）
    P_pbar, P_N = two_body(M_JPSI, M_P, M_N, rng)
    # 次级: N -> p + eta
    P_p, P_eta = two_body(M_N, M_P, M_ETA, rng)
    beta = P_N[1:] / P_N[0]
    return P_p, P_pbar, boost(P_eta, beta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_data", type=int, default=2000)
    ap.add_argument("--n_phsp", type=int, default=4000)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    for tag, n in [("data", args.n_data), ("phsp", args.n_phsp)]:
        out = TESTS_DIR / "data" / f"baryon_cp_{tag}.dat"
        with open(out, "w") as f:
            for _ in range(n):
                P_p, P_pbar, P_eta = gen_event(rng)
                for p in (P_p, P_pbar, P_eta):  # Data.order: [p, pbar, eta]
                    f.write("%.8f %.8f %.8f %.8f\n" % tuple(p))
        print(f"baryon_cp_{tag}: {n} events x 3 particles -> {out}")


if __name__ == "__main__":
    main()
