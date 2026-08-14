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

# 获取 CUDA gencode 编译选项
cuda_gencode_flags = get_cuda_gencode_flags()

# 定义扩展模块
# JIT 运行时 NVRTC 直接链 CUDA_HOME 的 libnvrtc（与 nvcc 同版本）。
# 说明：旧版（未分段生成）对超长依赖链源码在 12.9 上编译卡死，曾用
# CUDA 13.2 全路径链接绕开；分段生成（每段独立 __device__ 函数 + helper
# __noinline__）后 12.9/13.2 的 nvrtc 均全 arch 正常（实测 sm_70..sm_120
# 全部通过），因此回归标准链接。
extension = CUDAExtension(
    name="ctpwa",
    sources=[
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
    ],
    include_dirs=[
        os.path.join(project_dir, "include"),
        root_flags["include"],  # ROOT 头文件目录
        os.path.join(cuda_dir, "include"),
        # Conda 头文件（如果存在）
        *([os.path.join(conda_prefix, "include")] if conda_prefix else []),
    ],
    library_dirs=[
        root_flags["libdir"],  # ROOT 库目录
        os.path.join(cuda_dir, "lib64"),
        os.path.join(cuda_dir, "lib64", "stubs"),  # 无驱动编译环境（CI/容器）链接 -lcuda 用；运行时加载系统驱动库
        # Conda 库目录（如果存在）
        *([os.path.join(conda_prefix, "lib")] if conda_prefix else []),
        # Torch 库目录
        *(
            [os.path.join(conda_prefix, "lib/python3.12/site-packages/torch/lib")]
            if conda_prefix
            else []
        ),
    ],
    libraries=[
        "yaml-cpp",
        *root_flags["libraries"],  # ROOT 库
        "cudart",
        "cublas",
        "nvrtc",   # JITCustom（NVRTC 运行时编译自定义节点字节码；随 CUDA_HOME 版本）
        "cuda",    # JITCustom（驱动 API: cuModuleLoadData/cuLaunchKernel 等）
    ],
    extra_compile_args={
        "cxx": [
            "-fPIC",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",  # 确保与 PyTorch ABI 兼容
        ],
        "nvcc": [
            # gencode 由 TORCH_CUDA_ARCH_LIST 环境变量控制（torch cpp_extension 生成）；
            # 本机开发不设置时 torch 自动检测 GPU；CI 构建 wheel 时显式设置
            # 多架构列表（无 GPU 的 runner 上 torch 检测不到架构会 IndexError）。
            #"-arch=sm_120",
            "-rdc=true",
            "--expt-relaxed-constexpr",
            # "--use_fast_math",
            # "-Xcompiler",
            # "-fPIC",
            "-std=c++17",
            "--extended-lambda",
            # 双精度 complex 编译开关: CTPWA_DOUBLE_COMPLEX=1 pip install -e .
            *(["-DCTPWA_DOUBLE_COMPLEX"] if os.environ.get("CTPWA_DOUBLE_COMPLEX", "0") == "1" else []),
            #"--generate-line-info",
            #"-D_FORCE_INLINES",
            #"--extended-lambda",
            # 如果遇到内存对齐问题，可以添加
            # '--ptxas-options=-v',
            # '--maxrregcount=32',
        ],
        "nvcc_dlink": [
            "-dlink",
        ],
    },
    extra_link_args=[
        "-Wl,--no-as-needed",  # 确保链接所有需要的库
    ],
)

# long_description 读 README.md（PyPI 页面渲染用）
try:
    with open(os.path.join(project_dir, "README.md"), encoding="utf-8") as _f:
        _long_desc = _f.read()
except Exception:
    _long_desc = "A CUDA-Torch partial wave analysis package for high-energy physics"

setup(
    name="ctpwa",
    version="0.3.2",
    author="Benhou Xiang",
    description="CUDA-Torch Partial Wave Analysis",
    long_description=_long_desc,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["example", "tests"]),
    ext_modules=[extension],
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
