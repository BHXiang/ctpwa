#!/usr/bin/env python3
"""合成测试数据生成器（DAT 格式）。

ctpwa 支持纯文本 DAT 输入: 每行 "E px py pz"（GeV），粒子按 Data.order 循环。
这里生成物理有效的三体衰变 J/psi → Kp + Km + eta 事件：
- 相空间均匀抽样（flat phase space）
- 数据样本 = 相空间样本（用于结构性/数值测试，不追求物理意义）

用法:
    python3 tests/generate_test_data.py [--n_data 1000] [--n_phsp 10000]

产出:
    tests/data/test_data.dat, tests/data/test_phsp.dat, tests/data/test_sideband.dat

注意: 仓库内已提交一份固定数据（固定 seed），测试结果可复现。
重新生成将改变 golden values，需同时运行 tests/update_golden.py。
"""

import argparse
import os
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent

# 粒子质量（GeV）
M_JPSI = 3.0969
M_K = 0.4937
M_ETA = 0.5478


def three_body_flat(N, seed):
    """三体相空间均匀抽样: 返回 N 组 (p_Kp, p_Km, p_eta) 四动量 (E, px, py, pz)。

    J/psi 静止系，Dalitz 坐标均匀抽样。方法:
    1. 均匀抽 m2_12, m2_23（Dalitz 平面）
    2. 通过两体衰变运动学重建四动量
    3. 随机旋转
    """
    rng = np.random.default_rng(seed)
    M2 = M_JPSI**2

    # 均匀采样 Dalitz 坐标 m2(Kp,eta) 与 m2(Km,eta)（物理边界内）
    # 简化: 在两体阈值的允许范围内采样，然后丢弃能量不守恒的事件
    m2_12_min = (M_K + M_K) ** 2
    m2_12_max = (M_JPSI - M_ETA) ** 2
    m2_23_min = (M_K + M_ETA) ** 2
    m2_23_max = (M_JPSI - M_K) ** 2

    results = []
    while len(results) < N:
        m2_12 = rng.uniform(m2_12_min, m2_12_max)
        m2_23 = rng.uniform(m2_23_min, m2_23_max)
        m2_13 = M2 + 2 * M_K**2 + M_ETA**2 - m2_12 - m2_23

        # 检查 Dalitz 边界（Kibble 条件）——简化检查
        if m2_13 < (M_K + M_ETA) ** 2 or m2_13 > (M_JPSI - M_K) ** 2:
            continue

        # 在 J/psi 系中先构造 (12) 子系: J/psi -> (Kp Km) + eta
        m12 = np.sqrt(m2_12)
        E_eta = (M2 + M_ETA**2 - m2_12) / (2 * M_JPSI)
        p_eta = np.sqrt(max(E_eta**2 - M_ETA**2, 0.0))

        # eta 沿 +z（后面随机旋转）
        pz_eta = p_eta
        # 在 12 系中 Kp/Km 背对背
        m_eta_in12 = M_ETA
        E12 = M_JPSI - E_eta
        p12 = p_eta
        E_K_in12 = (m2_12 + M_K**2 - M_K**2) / (2 * m12)
        p_K_in12 = np.sqrt(max(E_K_in12**2 - M_K**2, 0.0))
        # 12 子系的动量方向（boost 后的角度），取均匀角
        cos_theta = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        # 简化处理：均匀角 + boost
        # 先构造在 12 系中 Kp 方向
        pz_Kp12 = p_K_in12 * cos_theta
        px_Kp12 = p_K_in12 * np.sin(cos_theta * 0 + 1) if False else p_K_in12 * np.sin(np.arccos(cos_theta)) * np.cos(phi)
        py_Kp12 = p_K_in12 * np.sin(np.arccos(cos_theta)) * np.sin(phi)

        # boost 12 系 -> J/psi 系（沿 eta 反方向，即 -z）
        gamma = E12 / m12
        beta = -p12 / E12  # 12 系沿 -z 运动
        # Kp boost
        E_Kp = gamma * E_K_in12 + beta * pz_Kp12
        pz_Kp = gamma * pz_Kp12 + beta * E_K_in12
        p_Kp = np.array([px_Kp12, py_Kp12, pz_Kp])

        # Km 在 12 系中与 Kp 反向
        E_Km_in12 = E_K_in12
        p_Km_12 = -np.array([px_Kp12, py_Kp12, pz_Kp12])
        E_Km = gamma * E_Km_in12 + beta * p_Km_12[2]
        pz_Km = gamma * p_Km_12[2] + beta * E_Km_in12
        p_Km = np.array([p_Km_12[0], p_Km_12[1], pz_Km])

        # 动量守恒检查
        p_tot = p_Kp + p_Km + np.array([0, 0, pz_eta])
        if abs(p_tot).max() > 1e-6:
            continue  # 抽样丢弃

        # 随机旋转（均匀方向）
        # 生成随机旋转矩阵
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, 2 * np.pi)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        p_Kp_r = R @ p_Kp
        p_Km_r = R @ p_Km
        p_eta_r = R @ np.array([0, 0, pz_eta])

        E_Kp = np.sqrt(M_K**2 + (p_Kp_r**2).sum())
        E_Km = np.sqrt(M_K**2 + (p_Km_r**2).sum())
        E_eta = np.sqrt(M_ETA**2 + (p_eta_r**2).sum())

        # 能量守恒检查
        if abs(E_Kp + E_Km + E_eta - M_JPSI) > 1e-6:
            continue

        results.append((E_Kp, *p_Kp_r, E_Km, *p_Km_r, E_eta, *p_eta_r))

    return np.array(results)


def write_dat(events, path):
    """写 DAT 文件: 每行 "E px py pz"，粒子循环。"""
    lines = []
    for (E1, p1x, p1y, p1z, E2, p2x, p2y, p2z, E3, p3x, p3y, p3z) in events:
        lines.append(f"{E1:.8f} {p1x:.8f} {p1y:.8f} {p1z:.8f}")
        lines.append(f"{E2:.8f} {p2x:.8f} {p2y:.8f} {p2z:.8f}")
        lines.append(f"{E3:.8f} {p3x:.8f} {p3y:.8f} {p3z:.8f}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_data", type=int, default=1000)
    parser.add_argument("--n_phsp", type=int, default=10000)
    parser.add_argument("--n_sideband", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = TESTS_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"生成相空间数据 {args.n_phsp} 事件...")
    phsp = three_body_flat(args.n_phsp, args.seed)
    write_dat(phsp, data_dir / "test_phsp.dat")

    print(f"生成数据样本 {args.n_data} 事件...")
    data = three_body_flat(args.n_data, args.seed + 1)
    write_dat(data, data_dir / "test_data.dat")

    print(f"生成 sideband {args.n_sideband} 事件...")
    sb = three_body_flat(args.n_sideband, args.seed + 2)
    write_dat(sb, data_dir / "test_sideband.dat")

    print(f"完成。文件位于 {data_dir}")


if __name__ == "__main__":
    main()
