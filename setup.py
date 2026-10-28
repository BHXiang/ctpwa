#!/usr/bin/env python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import subprocess

# 获取环境变量中的路径
conda_prefix = os.environ.get("CONDA_PREFIX")
root_dir = os.environ.get("ROOTSYS")  # 默认使用 /usr
cuda_dir = os.environ.get("CUDA_HOME")
project_dir = os.path.dirname(os.path.abspath(__file__))


class ForceBuildExtension(BuildExtension):
    """多文件编译构建扩展（不再每次清 build_temp，ninja 跟踪依赖）。"""
    def run(self):
        super().run()


# 获取当前 CUDA 环境支持的 SM 架构，生成 gencode 编译选项
def get_cuda_gencode_flags():
    """检测当前 nvcc 支持的 SM 架构，只对存在的架构生成 -gencode 选项"""
    # 期望支持的目标架构（主版本号对应 compute capability）
    desired_sm = [70, 75, 80, 86, 89, 90, 100, 120]
    # desired_sm = [120]

    try:
        output = subprocess.check_output(
            ["nvcc", "--list-gpu-arch"], universal_newlines=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvcc 不可用时返回空
        return []

    # 解析 nvcc --list-gpu-arch 输出，提取 compute_XX 中的 XX
    available = set()
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.startswith("compute_"):
            try:
                available.add(int(line.split("_")[1]))
            except ValueError:
                continue

    flags = []
    for sm in desired_sm:
        if sm in available:
            flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")

    if not flags:
        # 全部都不支持时，回退到最高的一个可用架构
        if available:
            max_sm = max(available)
            flags.append(f"-gencode=arch=compute_{max_sm},code=sm_{max_sm}")

    return flags


# 使用 root-config 获取 ROOT 的编译标志
def get_root_flags():
    """获取 ROOT 的编译和链接标志"""
    flags = {}

    # 获取 ROOT 的 include 路径
    try:
        root_include = subprocess.check_output(
            ["root-config", "--incdir"], universal_newlines=True
        ).strip()
    except:
        root_include = os.path.join(root_dir, "include", "root")

    # 获取 ROOT 的库路径
    try:
        root_libdir = subprocess.check_output(
            ["root-config", "--libdir"], universal_newlines=True
        ).strip()
    except:
        # 根据您的系统配置调整
        root_libdir = os.path.join(root_dir, "lib64", "root")

    # 获取 ROOT 需要的库
    try:
        root_libs = (
            subprocess.check_output(["root-config", "--libs"], universal_newlines=True)
            .strip()
            .split()
        )
    except:
        # 默认的 ROOT 库列表
        root_libs = [
            "-lCore",
            "-lRIO",
            "-lNet",
            "-lHist",
            "-lGraf",
            "-lGraf3d",
            "-lGpad",
            "-lTree",
            "-lMathCore",
            "-lPhysics",
        ]

    # 解析库名称（去掉 -l 前缀）
    libraries = [lib[2:] for lib in root_libs if lib.startswith("-l")]

    flags["include"] = root_include
    flags["libdir"] = root_libdir
    flags["libraries"] = libraries

    return flags


# 获取 ROOT 标志
root_flags = get_root_flags()

# ============================================================================
# 构建哪些后端
#   CTPWA_BACKEND=both (默认)  同时编 _cuda.so 与 _cpu.so（发布用）
#   CTPWA_BACKEND=cuda         只编 GPU 后端（本地开发快）
#   CTPWA_BACKEND=cpu          只编 CPU 后端（无卡机器 / 只在本地调试 CPU）
#
# 产物是 Python 包 ctpwa/：
#   ctpwa/__init__.py    门面：import 时探测并挑选后端
#   ctpwa/_cuda.so       GPU 后端
#   ctpwa/_cpu.so        CPU 后端
# 两者互斥加载（见 ctpwa/__init__.py），所以同名符号不会互相干扰。
# ============================================================================
_RAW_BACKENDS = os.environ.get("CTPWA_BACKEND", "both").strip().lower()
if _RAW_BACKENDS in ("both", "", "all"):
    BACKENDS = ["cuda", "cpu"]
elif _RAW_BACKENDS in ("cuda", "cpu"):
    BACKENDS = [_RAW_BACKENDS]
else:
    raise SystemExit(
        f"CTPWA_BACKEND={_RAW_BACKENDS!r} 无效；只能是 'both' | 'cuda' | 'cpu'")

# ---- 源文件清单 ----
# main.cu 是编译入口, 以 #include 方式内联 Config.cu/PhaseSpace.cu/Analysis.cu/
# Info.cu/ResModel.cu/Resonance.cu/DeviceManager.cu（这些不要再单独列，否则重复符号）。
# Backend.cpp 两种后端共用（内部按 CTPWA_HAS_CUDA 分流）。
SOURCES_CORE = [
    "src/main.cu",
    "src/Amplitude.cu",
    "src/ComputeHessian.cu",
    "src/AmpGen.cu",
    "src/ComputeNLL.cu",
    "src/ComputeResults.cu",
    "src/ComputeBF.cu",
    "src/Figure.cu",
    "src/Parameters.cu",
    "src/CustomExpr.cu",
    "src/JITCustom.cu",
]
SOURCES_BACKEND = ["src/Backend.cpp", "src/JitStubs_cpu.cpp"]

# CPU 后端不编译 JITCustom.cu（NVRTC 运行时 JIT，无 CPU 对应物）。
# ComputeHessian.cu 会编译，但其 T3 块归约 kernel 尚未移植（P3），
# CPU 侧调用会经 ctpwa::not_implemented() 明确失败。
SOURCES_CPU = [s for s in SOURCES_CORE if not s.endswith("JITCustom.cu")]


def _existing_dirs(paths):
    return [p for p in paths if p and os.path.isdir(p)]


# 复数精度（cxx 与 nvcc 必须一致：ComplexType.h 同时被 .cu 与 .cpp 包含）
if os.environ.get("CTPWA_COMPLEX", "double") != "float" and \
        os.environ.get("CTPWA_DOUBLE_COMPLEX", "1") != "0":
    define_complex = ["-DCTPWA_DOUBLE_COMPLEX"]
else:
    define_complex = []


# CPU 构建必须把 .cu 交给 C++ 编译器。gcc 只按扩展名判语言，而 -x c++ 无法经由
# extra_compile_args 生效（distutils 把 extra args 放在源文件之后）。因此生成一层
# "影子源文件" build/cpu_src/<name>.cpp, 其内容只有一行 #include "<绝对路径>.cu"。
# 被包含文件内的 #include "xxx.cu" 仍按原目录解析，不影响内联结构。
def make_cpu_shadow_sources(sources):
    out_dir = os.path.join(project_dir, "build", "cpu_src")
    os.makedirs(out_dir, exist_ok=True)
    shadow = []
    for src in sources:
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(out_dir, base + ".cpp")
        with open(dst, "w", encoding="utf-8") as f:
            f.write("// auto-generated by setup.py (CPU backend)\n")
            f.write('#include "%s"\n' % os.path.abspath(src))
        shadow.append(dst)
    return shadow


def _torch_bits():
    """torch 的头文件/库/ABI 版本（纯 Extension 不走 torch 构建机制，需自己接）。"""
    import torch  # noqa: E402
    from torch.utils.cpp_extension import include_paths as _inc
    from torch.utils.cpp_extension import library_paths as _libs
    from torch.utils.cpp_extension import TORCH_LIB_PATH as _libpath

    abi = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
    return _inc(), list(_libs()), _libpath, abi


def make_cuda_extension():
    from torch.utils.cpp_extension import CUDAExtension

    # 头文件目录。CUDA 13 起 thrust 位于 targets/<arch>/include/cccl，
    # 而 src/Backend.cpp 由 g++ 编译且包含 ComplexType.h（用到 thrust::complex），
    # 所以这些路径必须显式给全（nvcc 自己知道，g++ 不知道）。
    cuda_inc = _existing_dirs([
        os.path.join(cuda_dir, "include") if cuda_dir else "",
        os.path.join(cuda_dir, "targets/x86_64-linux/include") if cuda_dir else "",
        os.path.join(cuda_dir, "targets/x86_64-linux/include/cccl") if cuda_dir else "",
        os.path.join(cuda_dir, "targets/aarch64-linux/include") if cuda_dir else "",
        os.path.join(cuda_dir, "targets/aarch64-linux/include/cccl") if cuda_dir else "",
    ])

    # JIT 运行时 NVRTC 直接链 CUDA_HOME 的 libnvrtc（与 nvcc 同版本）。
    return CUDAExtension(
        name="ctpwa._cuda",
        sources=SOURCES_CORE + SOURCES_BACKEND,
        include_dirs=[
            os.path.join(project_dir, "include"),
            root_flags["include"],  # ROOT 头文件目录
            *cuda_inc,
            *([os.path.join(conda_prefix, "include")] if conda_prefix else []),
            "/usr/local/include",   # CI 静态 yaml-cpp
        ],
        library_dirs=[
            root_flags["libdir"],
            os.path.join(cuda_dir, "lib64"),
            os.path.join(cuda_dir, "lib64", "stubs"),
            *([os.path.join(conda_prefix, "lib")] if conda_prefix else []),
            *([os.path.join(conda_prefix, "lib/python3.12/site-packages/torch/lib")]
              if conda_prefix else []),
            "/usr/local/lib",       # CI 静态 yaml-cpp
        ],
        libraries=[
            "yaml-cpp",
            *root_flags["libraries"],
            "cudart", "cublas", "nvrtc", "cuda",
        ],
        extra_compile_args={
            "cxx": [
                "-fPIC", "-std=c++17",
                "-D_GLIBCXX_USE_CXX11_ABI=1",
                "-DCTPWA_BACKEND_CUDA",
                "-DCTPWA_MODULE_CUDA",
                *define_complex,
            ],
            "nvcc": [
                "-rdc=true",
                "--expt-relaxed-constexpr",
                "-std=c++17",
                "--extended-lambda",
                "-DCTPWA_BACKEND_CUDA",
                "-DCTPWA_MODULE_CUDA",
                *define_complex,
            ],
            "nvcc_dlink": ["-dlink"],
        },
        extra_link_args=["-Wl,--no-as-needed"],
    )


def make_cpu_extension():
    from setuptools import Extension

    torch_inc, torch_libdirs, torch_libpath, torch_abi = _torch_bits()

    # 设计约束：**不依赖任何外部数值库**（无 OpenBLAS / MKL / Eigen）。
    # BLAS 形状全部由 src/Backend.cpp 的 hw:: 自实现，保证 wheel 在任何 x86_64
    # 机器上都能跑，不会因为 BLAS ABI / 线程嵌套出问题。
    return Extension(
        name="ctpwa._cpu",
        sources=make_cpu_shadow_sources(SOURCES_CPU + SOURCES_BACKEND),
        include_dirs=[
            # CUDA Runtime / cuBLAS / thrust / cub 的 CPU 垫片。必须排在 include/ 之前。
            os.path.join(project_dir, "include", "cpu_shims"),
            os.path.join(project_dir, "include"),
            root_flags["include"],
            *torch_inc,
            *([os.path.join(conda_prefix, "include")] if conda_prefix else []),
            "/usr/local/include",
        ],
        library_dirs=[
            root_flags["libdir"],
            *torch_libdirs,
            *([os.path.join(conda_prefix, "lib")] if conda_prefix else []),
            "/usr/local/lib",
        ],
        libraries=[
            "yaml-cpp",
            *root_flags["libraries"],
            # torch 运行时（pybind11 里 return at::Tensor 需要）。
            # 这 4 个库在 cpu / cu126 两种 torch 发行版里都存在，故本 .so 在
            # 无 CUDA 的机器上同样能加载（已用 nm -D 确认零 CUDA 符号引用）。
            "torch", "torch_cpu", "torch_python", "c10",
        ],
        extra_compile_args=[
            "-fPIC", "-std=c++17", "-O3", "-fopenmp",
            f"-D_GLIBCXX_USE_CXX11_ABI={torch_abi}",
            "-DCTPWA_BACKEND_CPU",
            "-DCTPWA_MODULE_CPU",
            "-DCTPWA_DOUBLE_COMPLEX",   # CPU 后端固定 double（float 档位是为省显存）
        ],
        extra_link_args=[
            "-fopenmp",
            "-Wl,--no-as-needed",
            f"-Wl,-rpath,{torch_libpath}",
        ],
    )


ext_modules = []
if "cuda" in BACKENDS:
    ext_modules.append(make_cuda_extension())
if "cpu" in BACKENDS:
    ext_modules.append(make_cpu_extension())
print("[ctpwa] 构建后端: %s" % ", ".join(BACKENDS))

# long_description 读 README.md（PyPI 页面渲染用）
try:
    with open(os.path.join(project_dir, "README.md"), encoding="utf-8") as _f:
        _long_desc = _f.read()
except Exception:
    _long_desc = "A CUDA-Torch partial wave analysis package for high-energy physics"

setup(
    name="ctpwa",
    version="0.3.18",
    author="Benhou Xiang",
    description="CUDA-Torch Partial Wave Analysis",
    long_description=_long_desc,
    long_description_content_type="text/markdown",
    packages=["ctpwa"],
    ext_modules=ext_modules,
    cmdclass={
        # ForceBuildExtension: 清 build_temp 强制全量重编 + ninja 依赖跟踪。
        # 单文件架构（main.cu #include 全部 .cu）下 setuptools 增量编译不跟踪
        # include 依赖（use_ninja=False 时改 .cu 不触发重编），会链接新旧混合
        # 的 .so（实测 Hessian 慢 27×）。清空后每次全量编译，保证产物一致。
        "build_ext": ForceBuildExtension.with_options(
            use_ninja=True,
            no_python_abi_suffix=True,  # 不添加 Python ABI 后缀
        )
    },
    install_requires=[
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)
