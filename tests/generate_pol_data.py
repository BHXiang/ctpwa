#!/usr/bin/env python3
"""nPolar 实验数据生成器: 链式两体衰变级联（DAT 格式）。

ctpwa 数据格式: 每行 "E px py pz"，每事件行数 = Particles 段粒子数，
顺序 = Particles 段定义顺序（含初态 Jpsi 和中间共振）。
"""

import argparse
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent

CHAINS = {
    "pol81": {
        # Particles 段顺序（Jpsi + 终态；X_i 为 spin-1 终态粒子）
        "particles": ["Jpsi", "X1", "X2", "X3", "piP", "piM"],
        # (母粒子名, 母质量, 子A名, 子A质量, 子B名, 子B质量)；子A 为终态侧
        "chain": [
            ("Jpsi", 3.0969, "X1", 0.9000, "R1", 2.0000),
            ("R1", 2.0000, "X2", 0.7000, "R2", 1.2000),
            ("R2", 1.2000, "X3", 0.6000, "R3", 0.5000),
            ("R3", 0.5000, "piP", 0.1396, "piM", 0.1396),
        ],
    },
    "pol243": {
        "particles": ["Jpsi", "X1", "X2", "X3", "X4", "piM", "pi0"],
        "chain": [
            ("Jpsi", 3.0969, "X1", 0.9000, "R1", 2.0000),
            ("R1", 2.0000, "X2", 0.7000, "R2", 1.2000),
            ("R2", 1.2000, "X3", 0.6000, "R3", 0.5000),
            ("R3", 0.5000, "X4", 0.2000, "R4", 0.2800),
            ("R4", 0.2800, "piM", 0.1396, "pi0", 0.1350),
        ],
    },
    # 7 个 spin-1 (Jpsi+X1..X6) → nPolar = 3^7 = 2187（> A1024，验证 B16 降级路径）
    "pol2187": {
        "particles": ["Jpsi", "X1", "X2", "X3", "X4", "X5", "X6", "piM", "pi0"],
        "chain": [
            ("Jpsi", 3.0969, "X1", 0.0100, "R1", 3.0000),
            ("R1", 3.0000, "X2", 0.0100, "R2", 2.5000),
            ("R2", 2.5000, "X3", 0.0100, "R3", 2.0000),
            ("R3", 2.0000, "X4", 0.0100, "R4", 1.5000),
            ("R4", 1.5000, "X5", 0.0100, "R5", 1.0000),
            ("R5", 1.0000, "X6", 0.0100, "R6", 0.5000),
            ("R6", 0.5000, "piM", 0.1396, "pi0", 0.1350),
        ],
    },
}


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
    """把静止系四动量 boost 到实验室系（beta = 实验室系速度）。"""
    b2 = beta @ beta
    if b2 < 1e-12:
        return p
    gamma = 1.0 / np.sqrt(1.0 - b2)
    pv = p[1:]
    bdotp = beta @ pv
    E = gamma * (p[0] + bdotp)
    p_new = pv + (gamma - 1.0) * (bdotp / b2) * beta + gamma * beta * p[0]
    return np.array([E, p_new[0], p_new[1], p_new[2]])


def gen_event(chain, rng):
    """单事件: 返回 dict {粒子名: 实验室系四动量}（全部粒子，含 Jpsi/共振）。"""
    mom = {}
    P_lab = np.array([chain["chain"][0][1], 0.0, 0.0, 0.0])  # 初态静止
    mom[chain["chain"][0][0]] = P_lab

    # 顶层: Jpsi 静止系逐级衰变；子B 是下一级共振（或终态）
    def decay(m_res, m_a, m_b, P_lab):
        E = P_lab[0]
        beta = np.array(P_lab[1:]) / E if E > 0 else np.zeros(3)
        P1, P2 = two_body(m_res, m_a, m_b, rng)
        return boost(P1, beta), boost(P2, beta)

    for i, (mother, m_res, a, m_a, b, m_b) in enumerate(chain["chain"]):
        P1_lab, P2_lab = decay(m_res, m_a, m_b, P_lab)
        mom[a] = P1_lab
        if i == len(chain["chain"]) - 1:
            mom[b] = P2_lab          # 最后一级: 子B 是终态
        else:
            P_lab = P2_lab           # 子B 是下一级共振，继续
    return mom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_data", type=int, default=5000)
    ap.add_argument("--n_phsp", type=int, default=10000)
    args = ap.parse_args()

    for name, chain in CHAINS.items():
        rng = np.random.default_rng(42)
        for tag, n in [("data", args.n_data), ("phsp", args.n_phsp)]:
            out = TESTS_DIR / "data" / f"{name}_{tag}.dat"
            with open(out, "w") as f:
                for _ in range(n):
                    evt = gen_event(chain, rng)
                    # 初始粒子（Jpsi）由 Σ 重建，不写入数据文件；
                    # 每事件行数 = Data.order 粒子数
                    for pname in chain["particles"][1:]:
                        p = evt[pname]
                        f.write("%.8f %.8f %.8f %.8f\n" % tuple(p))
            print(f"{name}_{tag}: {n} 事件 x {len(chain['particles'])} 粒子 → {out}")


if __name__ == "__main__":
    main()
