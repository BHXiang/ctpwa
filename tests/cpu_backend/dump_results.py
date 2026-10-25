#!/usr/bin/env python3
"""dump_results.py — 把一组 config 的 NLL / 梯度 / Hessian 存成 npz，供 CPU vs GPU 对比

用法（在仓库根目录）:
    CTPWA_DEVICE=cuda python3 tests/cpu_backend/dump_results.py out_gpu.npz
    CTPWA_DEVICE=cpu  python3 tests/cpu_backend/dump_results.py out_cpu.npz --double
    python3 tests/cpu_backend/compare_results.py out_cpu.npz out_gpu.npz

wheel 里同时带 _cuda.so 与 _cpu.so，用 CTPWA_DEVICE 选后端即可；两个后端
不能在同一进程里加载（pybind11 类型注册是解释器全局的），所以分两个进程跑。
"""
import os
import sys

import numpy as np
import torch

import ctpwa

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
CONFIGS = os.path.join(TESTS, "configs")

# 覆盖不同模型 / 参数化：simple(BWR) flatte gs interp custom ident2 no_trans with_trans
# hess_mult 是回归用例：n_amps > 64 且带耦合矩阵（曾因 axpy stride 循环被改坏而
# 梯度/Hessian 出错、NLL 却看不出来）
NAMES = ["simple", "flatte", "gs", "interp", "custom", "ident2",
         "no_trans", "with_trans", "hess_mult"]


def make_params(ana):
    """设备无关的通用参数点（CPU 生成、再搬到目标设备）。

    不用 conftest.make_params：那里用 ``torch.Generator(device=device)``，
    CPU 与 CUDA 的随机流不同 → 两边拿到不同的 v，无法逐值对比。
    这里固定用 CPU generator，保证两个后端拿到**完全相同**的参数。
    """
    nv, nt = ana.getNVector(), ana.getNFreeTheta()
    p = torch.zeros(2 * nv + nt, dtype=torch.float64)
    p[0] = 1.0
    g = torch.Generator().manual_seed(42)          # 固定 seed，与设备无关
    if nv > 1:
        p[1:nv] = torch.rand(nv - 1, generator=g) * 0.5
        p[nv + 1:2 * nv] = (torch.rand(nv - 1, generator=g) - 0.5) * 0.5
    if nt:
        p[2 * nv:] = ana.getFreeResParams()[0]
    return p


def _config_for(name, force_double):
    """force_double=True 时生成 precision: double 的临时 config。

    GPU 构建默认 precision:auto（A 表存 float32），CPU 构建强制 double；
    要逐值对比必须让两边都用 double。临时文件放在 tests/ 下，data 相对路径照旧可用。
    """
    src = os.path.join(CONFIGS, name + ".yml")
    if not force_double:
        return src
    dst = os.path.join(TESTS, "_gen_%s_double.yml" % name)
    with open(src, encoding="utf-8") as f:
        body = f.read()
    with open(dst, "w", encoding="utf-8") as f:
        f.write("precision: double\n" + body)
    return dst


def main(out_path, device=None, force_double=False):
    os.chdir(TESTS)   # config 里的相对 data 路径按 tests/ 解析
    # 设备由当前选中的后端决定（torch.cuda.is_available() 只说明"有没有卡"，
    # 不说明本进程实际用哪个后端）。
    if device is None:
        device = "cuda" if ctpwa.BACKEND == "cuda" else "cpu"
    print("[dump] module=%s backend=%s device=%s"
          % (ctpwa.__file__, ctpwa.BACKEND, device))

    data = {}
    for name in NAMES:
        cfg = _config_for(name, force_double)
        if not os.path.exists(cfg):
            continue
        ana = ctpwa.analysis(cfg)
        p = make_params(ana).to(device)
        p.requires_grad_(True)

        nll = ana.getNLL(p)
        grad = torch.autograd.grad(nll, p)[0].detach()
        H = ana.getHessian(p.detach()).detach()

        data[name + "/nll"] = np.array(nll.item())
        data[name + "/grad"] = grad.detach().cpu().double().numpy()
        data[name + "/H"] = H.cpu().double().numpy()
        print("  %-12s nll=%.12f  |grad|=%.6g  |H|=%.6g"
              % (name, nll.item(), float(grad.abs().max()), float(H.abs().max())))

    np.savez(out_path, **data)
    print("[dump] saved", out_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1] if len(args) > 1 else None, "--double" in sys.argv)
