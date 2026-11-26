"""ctpwa 测试框架公共 fixtures。

约定:
- 测试运行时 cwd 被固定到 tests/ 目录，config 中的相对路径（如 ./data/test_data.dat）
  都从 tests/ 解析。
- L0/L1 用合成数据（tests/data/test_*.dat），本机即可跑。
- L2 用真实数据子集（tests/data/real_*.dat），需要集群上先跑 prepare_real_data.py
  生成；本地无真实数据时 L2 自动 skip。
"""

import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = TESTS_DIR / "configs"
GOLDEN_DIR = TESTS_DIR / "golden"

# 固定 seed：所有数值测试用同一套参数，保证可复现
SEED = 42


@pytest.fixture(scope="session", autouse=True)
def _chdir_tests_dir():
    """将测试进程 cwd 固定到 tests/ 目录。"""
    old = os.getcwd()
    os.chdir(TESTS_DIR)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def config_paths():
    return {
        "simple": str(CONFIGS_DIR / "simple.yml"),
        "no_trans": str(CONFIGS_DIR / "no_trans.yml"),
        "with_trans": str(CONFIGS_DIR / "with_trans.yml"),
        "with_deep_trans": str(CONFIGS_DIR / "with_deep_trans.yml"),
        # 乘法链 (chain×step) Hessian 回归: 含 3 因子振幅, 见 test_numerical.py
        "hess_mult": str(CONFIGS_DIR / "hess_mult.yml"),
        # 固定节点预算（静态链因子）: 链里有 1 个可折叠的静态节点,
        # 见 test_static_budget.py 与 test_numerical.py::test_gradient_vs_fd
        "static_nodes": str(CONFIGS_DIR / "static_nodes.yml"),
    }


@pytest.fixture(scope="session")
def make_analysis():
    """工厂 fixture：创建 analysis 实例（懒加载，避免重复初始化）。"""
    import ctpwa

    _cache = {}

    def _make(config_key: str, reuse: bool = True):
        key = config_key if reuse else f"{config_key}_{id(object())}"
        if reuse and key in _cache:
            return _cache[key]
        ana = ctpwa.analysis(str(CONFIGS_DIR / f"{config_key}.yml"))
        if reuse:
            _cache[key] = ana
        return ana

    return _make


def is_cpu_build() -> bool:
    """当前导入的 ctpwa.so 是否为 CPU 后端构建。

    旧版 .so 没有 backend() 入口，按 CUDA 处理（保持向后兼容）。
    """
    import ctpwa

    fn = getattr(ctpwa, "backend", None)
    return bool(fn is not None and fn() == "cpu")


CPU_BUILD = is_cpu_build()
#: 用于只在 CPU 构建下运行的用例（GPU 构建下整体 skip）
cpu_only = pytest.mark.skipif(
    not CPU_BUILD,
    reason="仅 CPU 后端构建（当前构建: %s）" % ("cpu" if CPU_BUILD else "cuda"),
)


@pytest.fixture(scope="session")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dev():
    """后端相关设备：CUDA 构建给 cuda:0，CPU 构建给 cpu。

    用于**数值**用例（NLL/golden/梯度/Hessian）：这些在两种后端上都该跑，
    只是设备不同。与 `device` 的区别是后者表示"此用例必须有真 GPU"。
    """
    import ctpwa
    if ctpwa.backend() == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device():
    # 无 CUDA 时 skip 而不是 error：CPU 构建（CI 无卡 runner / 本机无卡）下
    # 这些用例应当"跳过"而不是把整个测试套件标红。语义不变——需要真 GPU 的
    # 用例在无卡机器上依然不会被执行。
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA GPU（当前无可用设备）；CPU 构建下跳过")
    # 显式 cuda:0：torch.device('cuda') 解析到运行时当前设备，多 GPU 下
    # 会被内部 per-GPU 循环/用户 .to(cuda:1) 切走，导致 params 落在非主 GPU
    # （触发 analysis 的 params 位置检查，实测双卡测试套件必现）
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def real_data_available():
    """集群上生成的真实数据子集是否存在。"""
    return (TESTS_DIR / "data" / "real_phsp.dat").exists() and (
        TESTS_DIR / "data" / "real_data.dat"
    ).exists()


def make_params(ana, device, seed=SEED):
    """构造合法的初始参数向量 [real(v), imag(v), theta]。

    布局: v_0 = 1.0 + 0.0i 固定参考振幅，其余随机（固定 seed），theta 用标称值。

    ⚠ **跨后端比较时注意**：随机数用的是 ``torch.Generator(device=device)``，
    CPU 与 CUDA 的随机流不同 → 同一个 seed 在两端得到**不同的耦合初值**。
    拿两个后端的 ``getNLL(make_params(ana, dev))`` 直接比会误判成"NLL 不一致"
    （曾因此误报 CPU/CUDA 差 5–7%）。要比就统一在 CPU 生成再 ``.to(device)``，
    见 ``tmp/backend_parity/parity_all.py``。
    """
    n_vec = ana.getNVector()
    n_theta = ana.getNFreeTheta()
    n_total = 2 * n_vec + n_theta
    params = torch.zeros(n_total, dtype=torch.float64, device=device)

    params[0] = 1.0
    params[n_vec] = 0.0

    g = torch.Generator(device=device).manual_seed(seed)
    params[1:n_vec] = torch.rand(n_vec - 1, generator=g, device=device) * 0.5
    params[n_vec + 1 : 2 * n_vec] = (
        torch.rand(n_vec - 1, generator=g, device=device) - 0.5
    ) * 0.5
    if n_theta > 0:
        free_res = ana.getFreeResParams()  # [3, n_res]: init, lower, upper
        params[2 * n_vec :] = free_res[0].to(device=device, dtype=torch.float64)
    return params


@pytest.fixture(scope="session")
def base_params(make_analysis, dev):
    """simple config 的合法初始参数（golden 测试用）。用 dev 而非 device：
    golden 比对在 CPU 后端同样有效（double），不该因为没卡就跳过。"""
    ana = make_analysis("simple")
    return make_params(ana, dev)


def load_golden(name: str) -> float:
    p = GOLDEN_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"golden 文件不存在: {p}. 先运行 tests/update_golden.py 生成。"
        )
    return float(p.read_text().strip())


def save_golden(name: str, value) -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    (GOLDEN_DIR / name).write_text(value if isinstance(value, str) else f"{value:.16g}\n")

# ---------------------------------------------------------------------------
# 读 weight_best.root 的小工具（原 getFitFractionMatrix / ...Errors 已删除）
# ---------------------------------------------------------------------------
def _unified_params(ana, cpl_or_params):
    """耦合向量 -> 统一参数 [Re p, Im p, θ]；已是 (2n+nθ) float64 则原样返回。"""
    import torch
    n = ana.getNVector()
    nθ = ana.getNFreeTheta()
    t = cpl_or_params
    if t.dtype == torch.complex128 or t.dtype == torch.complex64:
        p = torch.zeros(2 * n + nθ, dtype=torch.float64)
        p[:n] = t.real
        p[n:2 * n] = t.imag
        if nθ:
            p[2 * n:] = ana.getFreeResParams()[0].double()
        return p
    return t


def _read_root_matrices(path):
    import numpy as np
    import uproot
    out = {}
    with uproot.open(str(path)) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        W = {}
        if "integrals" in keys:
            m = f["integrals"].arrays(library="np")
            W = {"rec": float(m["W_rec"][0]), "gen": float(m["W_gen"][0])}
        for tag, name in (("rec", "interference_rec"), ("gen", "interference_gen"),
                          ("rec_err", "interference_rec_err"),
                          ("gen_err", "interference_gen_err")):
            if name in keys:
                o = f[name]
                M = np.asarray(o.member("fElements"), dtype=float).reshape(
                    int(o.member("fNrows")), int(o.member("fNcols")))
                out[tag] = M
        for s in ("rec", "gen"):
            if s in out and W.get(s, 0.0) > 0.0:
                out[s + "_raw"] = out[s] * W[s]     # 乘回 W：与旧的原始积分口径一致
    return out


def resident_phsp_numpy(ana):
    """返回**所有 GPU** 的 phsp 驻留振幅，供独立矩阵重建使用。

    无参 getPhspTensor() 为兼容旧调试接口只给 GPU0；双卡直接拿它与
    writeResult 的全卡 Gram 比较，会把正确的合并结果误判成约两倍。
    """
    import numpy as np
    parts = [ana.getPhspTensor(g).cpu().numpy()
             for g in range(ana.getNDevices())]
    return np.concatenate(parts).astype(np.complex128)


def fit_fraction_matrix(ana, cpl, sample="truth", hessian=None):
    """旧 getFitFractionMatrix 的替代：写一次 ROOT 再读回来。

    sample: "truth"/"gen" -> interference_gen；"phsp"/"rec" -> interference_rec。
    返回值口径与旧 API 相同（原始积分，已乘回 W；即未归一化）。
    """
    import tempfile

    import torch
    params = _unified_params(ana, cpl)
    dev = "cuda:0" if __import__("ctpwa").backend() == "cuda" else "cpu"
    params = params.to(dev)
    with tempfile.TemporaryDirectory() as d:
        path = __import__("os").path.join(d, "m.root")
        if hessian is None:
            ana.writeResult(params, path, 0)
        else:
            ana.writeResult(params, path, 0, [], hessian.to(dev))
        mats = _read_root_matrices(path)
    key = "gen" if str(sample).lower() in ("truth", "gen", "phsp_truth") else "rec"
    if key + "_raw" not in mats:
        return torch.zeros(0, 0, dtype=torch.float64)   # 该样本未配置（如无 phsp_truth）
    return torch.tensor(mats[key + "_raw"], dtype=torch.float64)


def matrix_errors(ana, params, hessian):
    """旧 getFitFractionMatrixErrors 的替代：写 ROOT 后读 interference_*_err。"""
    import os
    import tempfile

    import torch
    dev = "cuda:0" if __import__("ctpwa").backend() == "cuda" else "cpu"
    p = _unified_params(ana, params).to(dev)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "e.root")
        ana.writeResult(p, path, 0, [], hessian.to(dev))
        mats = _read_root_matrices(path)
    import numpy as np
    er = torch.tensor(mats.get("rec_err", np.zeros((0, 0))), dtype=torch.float64)
    eg = torch.tensor(mats.get("gen_err", np.zeros((0, 0))), dtype=torch.float64)
    return [er, eg]

