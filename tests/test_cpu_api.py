"""CPU 后端的 Python API 行为：构建标识、设备校验、精度映射、扩展向量。

只在**当前进程选中 CPU 后端**时运行（GPU 机器上请用 ``CTPWA_DEVICE=cpu``）。
"""

import os

import pytest
import torch

import ctpwa
from conftest import CONFIGS_DIR, cpu_only, make_params

pytestmark = cpu_only

CPU = torch.device("cpu")


# ---------------------------------------------------------------- 公开 API
def test_facade_forwards_every_backend_symbol():
    """门面必须把选中后端的**全部**公开名字转发出来。

    为什么不能只校验几个固定名字：0.3.15 的门面只转发了 4 个
    （analysis / PhaseSpaceGenerator / backend / build_info），把
    DecayInfo / ChainView / DeviceManager 漏掉了。而当时的 CPU 测试恰好一个都
    没用到（tests/test_structure.py、test_trans_host.py 等 6 个文件在用），
    全量测试又因为需要 GPU 没跑 → 缺陷直接发到了 PyPI。

    这里直接与后端模块的 dir() 对齐，以后新增绑定也不会再"忘了转发"。
    """
    import importlib

    impl = importlib.import_module(f"ctpwa._{ctpwa.BACKEND}")
    missing = [n for n in dir(impl)
               if not n.startswith("_") and not hasattr(ctpwa, n)]
    assert not missing, f"门面漏转发: {missing}"

    not_in_all = [n for n in dir(impl)
                  if not n.startswith("_") and n not in ctpwa.__all__]
    assert not not_in_all, f"__all__ 缺: {not_in_all}"


def test_documented_api_present():
    """文档/示例里出现过的名字，必须能在顶层直接访问。"""
    for name in ("analysis", "PhaseSpaceGenerator", "DecayInfo", "ChainView",
                 "DeviceManager", "backend", "build_info"):
        assert hasattr(ctpwa, name), f"ctpwa.{name} 缺失"


def test_decay_info_usable():
    """DecayInfo 不只是"有这个名字"，要真的能构造。"""
    ana = ctpwa.analysis(str(CONFIGS_DIR / "simple.yml"))
    info = ana.getDecayInfo() if hasattr(ana, "getDecayInfo") else None
    if info is None:
        pytest.skip("analysis 未暴露 getDecayInfo")
    assert isinstance(info, ctpwa.DecayInfo)


# ---------------------------------------------------------------- 构建标识
def test_backend_string():
    assert ctpwa.backend() == "cpu"


def test_build_info_dict():
    info = ctpwa.build_info()
    assert set(["backend", "precision", "complex_bytes", "has_cuda"]) <= set(info)
    assert info["backend"] == "cpu" and info["has_cuda"] is False


# ---------------------------------------------------------------- 设备校验
def test_cpu_tensor_accepted(make_analysis):
    ana = make_analysis("simple")
    p = make_params(ana, CPU)
    assert torch.isfinite(ana.getNLL(p))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="本机无 CUDA 设备")
def test_cuda_tensor_rejected(make_analysis):
    """CPU 构建喂 CUDA 张量必须报错，而不是在 kernel 里解引用野指针。"""
    ana = make_analysis("simple")
    p = make_params(ana, CPU).cuda()
    with pytest.raises(RuntimeError, match="CPU|device|设备"):
        ana.getNLL(p)


# ---------------------------------------------------------------- 扩展向量
def test_extended_vector(make_analysis):
    """自由耦合 → 扩展振幅耦合；单振幅 trivially 等于自身。"""
    ana = make_analysis("simple")
    v = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    out = ana.getExtendedVector(v)
    assert out.device.type == "cpu"
    assert out.numel() == 1
    assert out[0].item() == pytest.approx(1.0 + 0.0j)


# ---------------------------------------------------------------- 精度映射
def _write_float_config(tmp_path):
    """复制 simple.yml 并加上 precision: float（CPU 应该接受并降级为 double）。"""
    src = CONFIGS_DIR / "simple.yml"
    text = src.read_text(encoding="utf-8")
    dst = CONFIGS_DIR / "_gen_cpu_precision_float.yml"   # tests/_gen_*.yml 已 gitignore
    dst.write_text("precision: float\n" + text, encoding="utf-8")
    return dst


def test_precision_float_accepted_as_double(make_analysis):
    """CPU 后端固定 double：precision: float 照收，但数值应与 double 完全一致。"""
    path = _write_float_config(None)
    try:
        ana_f = ctpwa.analysis(str(path))
        ana_d = make_analysis("simple")
        pf = make_params(ana_f, CPU)
        pd = make_params(ana_d, CPU)
        a = ana_f.getNLL(pf).item()
        b = ana_d.getNLL(pd).item()
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"float 档未映射到 double: {a!r} vs {b!r}"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------- 相空间产生
def test_phase_space_generator_cpu():
    """DIYMC 相空间产生本来就是 CPU 的，CPU 构建下必须照常可用。"""
    gen = ctpwa.PhaseSpaceGenerator(str(CONFIGS_DIR / "simple.yml"))
    ev = gen.generate(1000)
    assert ev.shape[0] == 1000
    assert ev.dtype == torch.float64
    assert ev.device.type == "cpu"
