"""显存监控回归（L1, 本地 GPU）: 各阶段显存变化 + writeResult 峰值增量门禁。

背景: A100-40GB 上真实分析（phsp 518 万 × 29 波）writeResult 时显存从 27GB
突涨到 35GB 后 OOM——根因是驻留模式整表 double upcast + computeResults 内部
整段临时（各 ~7GB）。已改按批处理（峰值 ~批内 0.3GB）。

本测试把"writeResult 显存突然放大"变成可回归的门禁:
  1. 构造足够大的 phsp（本地 ~400 万事件 × 6 波）——若未来某改动退回整表
     物化, writeResult 峰值增量会再次 ~1.5× 常驻以上 → 断言红;
  2. 分阶段采样（ctor / getNLL / Hessian / writeResult 峰值）并打印表格;
  3. 硬性断言: 全程峰值不超卡显存 90%; writeResult 峰值增量 ≤ 常驻 50%。

实现: ctpwa 用裸 cudaMalloc（不经 torch caching allocator）, torch 侧
mem_get_info 看不到; writeResult 是长时间持 GIL 的 C++ 调用 → 用独立
nvidia-smi 子进程采样（不受 GIL 影响）。
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
OUT_DIR = TESTS_DIR / "outputs"
GEN_DIR = TESTS_DIR / "outputs" / "gen"
MEM_DIR = OUT_DIR / "memmon"

# 放大倍数: test_phsp.dat = 1e4 事件。**按卡容量自适应**（默认让稳态驻留 ~20%），
# MEMMON_PHSP_MULT 可显式覆盖。固定 400 倍（4e6 事件）是按 40GB 卡写的：
# 实测驻留 ~1.9 KB/事件，4e6 事件要 ~7.6 GB，8GB 卡上光 ctor/getNLL 就会顶满。
def _phsp_mult() -> int:
    env = os.environ.get("MEMMON_PHSP_MULT")
    if env:
        return max(1, int(env))
    try:
        _, total = torch.cuda.mem_get_info(0)
    except Exception:
        return 400          # 无卡（本用例会被 CUDA fixture 跳过）
    return int(max(40, min(1000, 0.20 * total / (10000 * 1900))))


PHSP_MULT = _phsp_mult()
PHSP_BASE = TESTS_DIR / "data" / "test_phsp.dat"
N_PHSP = 10000 * PHSP_MULT


def _used_mib():
    """当前主卡已用显存 (MiB)。"""
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / 1048576.0


def _nvsmi_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.split()[0])
    except Exception:
        return None


def _gen_phsp(path: Path):
    """放大 phsp: 重复 test_phsp.dat（运动学合法, 重复事件对内存测试无害）。"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = PHSP_BASE.read_bytes()
    with open(path, "wb") as f:
        for _ in range(PHSP_MULT):
            f.write(data)


def _write_config(phsp_path: Path) -> str:
    """5 波 [1-]（2 波 free）config, 数据指 test_data / 放大 phsp。"""
    cfg = f"""Particles:
  Jpsi: {{J: 1, P: -1, mass: 3.0969, tex: ['J/\\\\psi']}}
  eta:  {{J: 0, P: -1, mass: 0.5478, tex: ['\\\\eta']}}
  Kp:   {{J: 0, P: -1, mass: 0.4937, tex: ['K^{{+}}']}}
  Km:   {{J: 0, P: -1, mass: 0.4937, tex: ['K^{{-}}']}}

Data:
  order: [Kp, Km, eta]
  data: [dat, "./data/test_data.dat"]
  phsp: [dat, "{phsp_path}"]
  bkg:  [dat, "./data/test_sideband.dat"]

DecayChains:
  chain1:
    decay:
      - Jpsi: [eta, R_KK]
      - R_KK: [Kp, Km]
    R_KK:
      - [J: 1, P: -1]: [r1, r2, r3, r4, r5]
    legend: [R_KK, " ", eta]

Resonances:
  r1: {{J: 1, P: -1, model: BWR, parameters: [1.02, 0.0045], free: [0, 1], tex: ["r1"]}}
  r2: {{J: 1, P: -1, model: BWR, parameters: [1.66, 0.125], free: [0, 1], tex: ["r2"]}}
  r3: {{J: 1, P: -1, model: BWR, parameters: [1.72, 0.25], tex: ["r3"]}}
  r4: {{J: 1, P: -1, model: BWR, parameters: [1.95, 0.30], tex: ["r4"]}}
  r5: {{J: 1, P: -1, model: BWR, parameters: [2.20, 0.35], tex: ["r5"]}}
"""
    return cfg


class _PeakMonitor:
    """nvidia-smi 子进程轮询: 抓 writeResult 等持 GIL 调用的显存峰值。"""

    def __init__(self):
        self.peak = None
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            v = _nvsmi_used_mib()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            time.sleep(0.02)

    def __enter__(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)


def _mem_row(stage: str, used_mib, peak_mib=None, dt=None):
    extra = f" | 峰值 {peak_mib:.0f}" if peak_mib else ""
    extra += f" | Δ {dt:+.0f}" if dt is not None else ""
    print(f"  [MEM] {stage:28s} used={used_mib:8.1f} MiB{extra}", flush=True)


@pytest.fixture(scope="module")
def memmon_env():
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    phsp = MEM_DIR / f"mem_phsp_{N_PHSP // 1000}k.dat"
    _gen_phsp(phsp)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GEN_DIR / "memmon.yml"
    cfg.write_text(_write_config(phsp))
    return str(cfg)


def test_memory_monitor(memmon_env):
    import ctpwa
    from conftest import make_params

    print(f"\n  [MEM] 规模: phsp={N_PHSP} 事件 (x{PHSP_MULT}), 6 波, pol=3", flush=True)

    # 阶段 0: ctor
    t0 = time.time()
    ana = ctpwa.analysis(memmon_env)
    u0 = _used_mib()
    _mem_row("ctor 后", u0, dt=u0)
    params = make_params(ana, torch.device("cuda:0"))

    # 阶段 1: getNLL（free-θ 含 reComputeAmps + dF 常驻分配）
    nll = float(ana.getNLL(params))
    torch.cuda.synchronize()
    u1 = _used_mib()
    _mem_row("getNLL 后", u1, dt=u1 - u0, )

    # 阶段 2: Hessian（临时应在调用后释放; 只记信息）
    _ = ana.getHessian(params.detach())
    torch.cuda.synchronize()
    u2 = _used_mib()
    _mem_row("getHessian 后", u2, dt=u2 - u1)

    # 硬性门禁 1: 常驻总量必须远小于卡显存（否则后续无意义）
    _, total_bytes = torch.cuda.mem_get_info(0)
    total_mib = total_bytes / 1048576.0
    steady = max(u0, u1)
    assert steady < 0.85 * total_mib, (
        f"常驻显存 {steady:.0f} MiB 已超卡容量 85% ({total_mib:.0f} MiB)"
    )

    # 阶段 3: writeResult 峰值（独立进程采样）
    u_before = _used_mib()
    out_root = str(MEM_DIR / "memmon_result.root")
    if os.path.exists(out_root):
        os.remove(out_root)
    with _PeakMonitor() as mon:
        ana.writeResult(params, out_root, 0)
    peak_wr = mon.peak if mon.peak is not None else _used_mib()
    _mem_row("writeResult 峰值", peak_wr, dt=peak_wr - u_before)

    # 硬性门禁 2: writeResult 峰值增量 ≤ 常驻 30%（批模式实测 ~6%; 若退回
    # 整表 double upcast + 整段临时, 增量 ~0.43-0.6 → 红; 阈值留足批抖动余量）
    rel_delta = (peak_wr - u_before) / max(steady, 1.0)
    assert rel_delta <= 0.30, (
        f"writeResult 显存突增 {rel_delta * 100:.0f}% (u_before={u_before:.0f} "
        f"→ peak={peak_wr:.0f} MiB) —— 疑似退回整表物化/大临时分配"
    )
    # 硬性门禁 3: 全程峰值不超 90%
    assert peak_wr < 0.90 * total_mib, (
        f"writeResult 峰值 {peak_wr:.0f} MiB 超过卡容量 90% ({total_mib:.0f})"
    )
    print(f"  [MEM] 通过: writeResult 增量 {rel_delta * 100:.1f}% ≤ 30%, "
          f"峰值 {peak_wr:.0f}/{total_mib:.0f} MiB", flush=True)
