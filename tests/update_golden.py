#!/usr/bin/env python3
"""重新生成 golden values。

用法（在有意的数值变化之后，如 fast_math、算法优化）:
    python3 tests/update_golden.py

生成 tests/golden/nll_simple.txt 和 tests/golden/grad_simple.txt。
⚠️ 仅在确认新数值正确（已通过其他测试）时运行，否则会掩盖回归。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from conftest import TESTS_DIR, save_golden, make_params

def main():
    os.chdir(TESTS_DIR)
    import ctpwa

    ana = ctpwa.analysis(str(TESTS_DIR / "configs" / "simple.yml"))
    params = make_params(ana, torch.device("cuda")).requires_grad_(True)

    nll = ana.getNLL(params).item()
    grad = torch.autograd.grad(ana.getNLL(params), params)[0].detach().cpu().double()

    save_golden("nll_simple.txt", nll)
    save_golden("grad_simple.txt", "\n".join(f"{g:.16e}" for g in grad.tolist()))

    print(f"golden 更新完成:")
    print(f"  NLL    = {nll:.10f}")
    print(f"  梯度   = [{grad[0]:.6e}, ...] (dim={grad.numel()})")
    print(f"  文件   = {TESTS_DIR / 'golden'}")


if __name__ == "__main__":
    main()
