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


# 获取当前 CUDA 环境支持的 SM 架构，生成 gencode 编译选项
def get_cuda_gencode_flags():
    """检测当前 nvcc 支持的 SM 架构，只对存在的架构生成 -gencode 选项"""
    # 期望支持的目标架构（主版本号对应 compute capability）
    # desired_sm = [70, 75, 80, 86, 89, 90, 100, 120]
    desired_sm = [120]

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
extension = CUDAExtension(
    name="ctpwa",
    sources=[
        "src/main.cu",
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
    ],
    extra_compile_args={
        "cxx": [
            "-fPIC",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",  # 确保与 PyTorch ABI 兼容
        ],
        "nvcc": [
            *cuda_gencode_flags,  # 动态检测到的 gencode 选项
            #"-arch=sm_120",
            "--expt-relaxed-constexpr",
            "-Xcompiler",
            "-fPIC",
            "-std=c++17",
            "--extended-lambda",
            "--generate-line-info",
            "-D_FORCE_INLINES",
            "--extended-lambda",
            # 如果遇到内存对齐问题，可以添加
            # '--ptxas-options=-v',
            # '--maxrregcount=32',
        ],
    },
    extra_link_args=[
        "-Wl,--no-as-needed",  # 确保链接所有需要的库
    ],
)

setup(
    name="ctpwa",
    version="0.1.1",
    author="Benhou Xiang",
    description="CUDA-Torch Partial Wave Analysis",
    long_description="A CUDA-Torch partial wave analysis package for high-energy physics",
    packages=find_packages(exclude=["example"]),
    ext_modules=[extension],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=False,  # 如果系统没有 ninja
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
