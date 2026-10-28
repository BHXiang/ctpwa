"""ctpwa — 协变张量分波分析（CTPWA）。

本包是**后端门面**：一个 wheel 里同时带 ``_cuda.so`` 与 ``_cpu.so``，
``import ctpwa`` 时探测可用后端（GPU 优先），选中后把该后端的全部接口转发出来。

设计约束
--------
* **两个后端互斥加载**。``analysis`` 对象内部持有设备裸指针，只能由分配它的
  那个运行时解释；在同一进程里混用两个后端等于拿 A 家的地址去 B 家解引用。
  因此这里先判定、再加载，判定失败的后端**不会**被 dlopen。
* 判定不能只看 ``torch.cuda.is_available()``：它只回答"驱动看得见卡吗"，
  回答不了"kernel 能不能在这张卡上跑"（编译架构与卡不匹配、卡被 MIG 切分或
  占满时，前者为 True 但后者照样失败）。所以探针要**真跑一次 kernel**。
* 回退必须"吵"：CPU 后端比 GPU 慢 20–60 倍，静默降级会让用户以为程序坏了。

环境变量
--------
``CTPWA_DEVICE``
    ``auto``（默认）| ``cuda`` | ``cpu``。
    显式指定 ``cuda`` 时若不可用**直接报错**，不会偷偷降级。
``CTPWA_QUIET``
    置 1 时抑制导入期的提示信息（不影响 ``ctpwa.backend()`` 的返回值）。

举例
----
>>> import ctpwa
>>> ctpwa.backend()
'cuda'
>>> ana = ctpwa.analysis("config.yml")

注意
----
不要直接 ``import ctpwa._cuda`` / ``ctpwa._cpu``：导入子模块会先执行本文件
（完成后端选择），而 pybind11 的类型注册是解释器全局的，两个后端同时加载会
报 ``type ... is already registered``。公开入口只有 ``import ctpwa``。
"""

import os as _os
import sys as _sys

__version__ = "0.3.18"

#: 运行时按后端模块实际暴露的名字生成（见文件末尾）
__all__ = []

_REQUESTED = _os.environ.get("CTPWA_DEVICE", "auto").strip().lower()
if _REQUESTED not in ("auto", "cuda", "cpu"):
    raise ValueError(
        f"CTPWA_DEVICE={_REQUESTED!r} 无效；只能是 'auto' | 'cuda' | 'cpu'")

_QUIET = _os.environ.get("CTPWA_QUIET", "") not in ("", "0", "false", "False")


def _say(msg):
    if not _QUIET:
        print(f"[ctpwa] {msg}", file=_sys.stderr)


def _cuda_probe():
    """用 torch 自检 GPU（**不加载**本包的 CUDA 模块）。

    返回 ``(可用, 原因)``。先看有没有卡，再真跑一次 torch kernel ——
    "驱动看得见卡但 kernel 起不来"只有真跑才能发现。

    用 torch 而不是我们自己去做探测，是为了在决定加载 ``_cuda.so`` 之前就完成
    判断：一旦 dlopen 了它，同进程再加载 ``_cpu.so`` 就有同名 C++ 符号互相
    干扰的风险。torch 是硬依赖、反正已经加载，用它做探针最便宜。
    """
    try:
        import torch
    except Exception as e:                       # noqa: BLE001
        return False, f"torch 不可用: {e}"

    try:
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() = False（无显卡 / 驱动缺失或过旧）"
    except Exception as e:                       # noqa: BLE001
        return False, f"torch.cuda 探测异常: {type(e).__name__}: {e}"

    try:
        x = torch.zeros(1, device="cuda")
        x += 1.0                                 # 真启动一个 kernel
        torch.cuda.synchronize()
        return True, ""
    except Exception as e:                       # noqa: BLE001
        return False, f"GPU 自检失败（kernel 无法执行）: {type(e).__name__}: {e}"


def _load(suffix):
    import importlib
    return importlib.import_module(f"{__name__}.{suffix}")


def _select():
    """返回 (后端模块, 回退原因)。"""
    # 先 import torch 再加载任何后端：两个 .so 的 DT_NEEDED 里都有 libtorch*.so，
    # 而 wheel 中烘焙的 rpath 是构建机路径（CI 里是 /opt/conda/...），用户机上不存在。
    # torch 一旦被 import，这些 soname 已在进程内，动态加载器直接复用，不再查找磁盘。
    try:
        import torch  # noqa: F401
    except Exception:                            # noqa: BLE001
        pass

    reason = ""

    if _REQUESTED in ("auto", "cuda"):
        ok, why = _cuda_probe()
        if ok:
            impl = _load("_cuda")                # 只在这里加载 CUDA 模块
            probe = getattr(impl, "_probe", None)
            if probe is None or probe():
                return impl, ""
            # 已经 dlopen 了 _cuda：**不能**再退回 _cpu。
            # pybind11 的类型注册是解释器全局的，两个模块都注册 analysis /
            # PhaseSpaceGenerator 等同名类型，第二个会报 "already registered"。
            raise RuntimeError(
                "ctpwa CUDA 后端自检失败（kernel 无法在本卡上执行）。"
                "同一进程内无法切换到 CPU 后端，请用 CTPWA_DEVICE=cpu 重新运行。")
        reason = why

        if _REQUESTED == "cuda":
            raise RuntimeError(
                f"CTPWA_DEVICE=cuda，但 GPU 后端不可用：{reason}")

    # CPU（auto 的回退路径或显式指定）
    try:
        impl = _load("_cpu")
    except Exception as e:                       # noqa: BLE001
        raise RuntimeError(
            f"没有可用的 ctpwa 后端。GPU: {reason or '未尝试'}；"
            f"CPU: {type(e).__name__}: {e}") from e

    probe = getattr(impl, "_probe", None)
    if probe is not None and not probe():
        raise RuntimeError("ctpwa CPU 后端自检失败（内存分配/kernel 执行异常）")

    if _REQUESTED == "auto":
        _say(f"未使用 GPU：{reason}")
        _say("已回退到 CPU 后端。注意 CPU 比 GPU 慢 20~60 倍，"
             "大统计量拟合请改用 GPU 或减少 phsp 事例数。")
    else:
        _say("按 CTPWA_DEVICE=cpu 使用 CPU 后端。")
    return impl, reason


_impl, _FALLBACK_REASON = _select()

# ---- 转发选中后端的**全部**公开接口 ----
# 不能只列核心几个：扩展模块还把 DecayInfo / ChainView / DeviceManager 暴露为
# 公开 API（历史上 ``import ctpwa`` 就能拿到）。0.3.15 只转发了 analysis /
# PhaseSpaceGenerator / backend / build_info，导致这几个名字消失 → 0.3.16 修复。
for _name in dir(_impl):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_impl, _name)

BACKEND = _impl.backend()          # 'cuda' / 'cpu'，便于脚本判断

#: auto 模式下的回退原因（BACKEND == 'cpu' 且原本可用 GPU 判断时为非空）
FALLBACK_REASON = _FALLBACK_REASON

#: 顶层可用的全部名字 = 后端模块的公开名字 + 本门面新增的两个
__all__ = sorted(n for n in dir(_impl) if not n.startswith("_")) + [
    "BACKEND", "FALLBACK_REASON",
]
