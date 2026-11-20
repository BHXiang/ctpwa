"""固定节点预算（静态链因子）回归。

`configs/static_nodes.yml` 的三节点链里 node2 参数全固定、子粒子也固定
→ 其因子被折进 `d_static`（`CTPWA_STATIC_DEBUG=1` 打印 `static: nodes=3 static=1`）。
这条路径有两条独立检查：

1. `test_static_budget_matches_full_path`：`CTPWA_NO_STATIC=1`（全节点热循环）
   与默认（折静态因子）必须给出同一 NLL/梯度。开关是进程级 static，故用子进程
   跑两遍再逐项比对。
2. `test_static_factor_gradient_vs_fd`：在同一 config 上做 θ 方向的有限差分。
   静态因子只乘进 F、没乘进 dF 的话这里立刻失败。
   （不挂到 `test_numerical.py::test_gradient_vs_fd`：该用例固定 rel=1e-3 步长，
   对本 config 的 a1 质量方向截断误差达 13%（步长 1.8e-3 → FD 144，压到
   5.4e-5 收敛到 127.30，与解析 127.2939 一致），属于 FD 步长问题而非 kernel 问题。）
"""

import os
import subprocess
import sys

import torch

from conftest import TESTS_DIR, make_params

_SCRIPT = r"""
import sys, torch, ctpwa
sys.path.insert(0, '.')
from conftest import make_params
ana = ctpwa.analysis('configs/static_nodes.yml')
p = make_params(ana, torch.device(__DEV__)).requires_grad_(True)
nll = ana.getNLL(p)
g, = torch.autograd.grad(nll, p)
print('NLL %.17g' % nll.item())
print('GRAD ' + ' '.join('%.17g' % x for x in g.detach().cpu().tolist()))
"""


def _run(with_static: bool, device: str):
    env = dict(os.environ)
    env.pop("CTPWA_NO_STATIC", None)
    if not with_static:
        env["CTPWA_NO_STATIC"] = "1"
    env["CTPWA_STATIC_DEBUG"] = "1"      # 让 stderr 报出节点分类
    env["CTPWA_QUIET"] = "1"
    root = str(TESTS_DIR.parent)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"]
                                if env.get("PYTHONPATH") else "")
    out = subprocess.run([sys.executable, "-c", _SCRIPT.replace("__DEV__", repr(device))],
                         cwd=str(TESTS_DIR), env=env, capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout, out.stderr


def _parse(out):
    nll, grad = None, None
    for line in out.splitlines():
        if line.startswith("NLL "):
            nll = float(line.split()[1])
        elif line.startswith("GRAD "):
            grad = [float(x) for x in line.split()[1:]]
    assert nll is not None and grad is not None, out[-2000:]
    return nll, grad


def test_static_budget_matches_full_path(device):
    """折静态因子 vs 全节点热循环：NLL 与全梯度必须一致。"""
    out_a, err_a = _run(True, str(device))
    out_b, _ = _run(False, str(device))

    # 该 config 必须真的折叠了节点，否则这条用例没覆盖到目标路径
    assert "static: nodes=3 static=1" in err_a, err_a[-2000:]

    nll_a, grad_a = _parse(out_a)
    nll_b, grad_b = _parse(out_b)

    assert abs(nll_a - nll_b) <= 1e-12 * max(abs(nll_b), 1.0), (nll_a, nll_b)
    assert len(grad_a) == len(grad_b)
    scale = max(max(abs(x) for x in grad_b), 1e-30)
    worst = max(abs(x - y) for x, y in zip(grad_a, grad_b))
    assert worst / scale < 1e-9, f"梯度不一致: max|Δ|={worst:.3e} (|g|max={scale:.3e})"


def test_static_factor_gradient_vs_fd(make_analysis, device):
    """静态因子只进 F 是不够的：dF 必须等于 F 的 θ 导数（FD 收敛步长）。"""
    ana = make_analysis("static_nodes")
    params = make_params(ana, device)
    n_vec = ana.getNVector()
    n_theta = ana.getNFreeTheta()
    assert n_theta > 0

    p = params.clone().requires_grad_(True)
    grad_ad = torch.autograd.grad(ana.getNLL(p), p)[0].detach().cpu().double()

    p_cpu = params.detach().cpu().double()
    for i in range(2 * n_vec, 2 * n_vec + n_theta):
        fds = []
        for rel in (1e-4, 3e-5):
            step = max(abs(p_cpu[i].item()) * rel, 1e-5)
            hi, lo = p_cpu.clone(), p_cpu.clone()
            hi[i] += step
            lo[i] -= step
            n_hi = ana.getNLL(hi.to(device)).item()
            n_lo = ana.getNLL(lo.to(device)).item()
            fds.append((n_hi - n_lo) / (2 * step))
        # 两个步长自洽才说明 FD 已收敛（否则是截断/舍入噪声）
        assert abs(fds[0] - fds[1]) <= 2e-2 * max(abs(fds[1]), 1.0), (i, fds)
        ad = grad_ad[i].item()
        assert abs(ad - fds[1]) <= 3e-3 * max(abs(fds[1]), 1.0), (
            f"θ[{i - 2 * n_vec}]: AD={ad:.6f} FD={fds[1]:.6f}")
