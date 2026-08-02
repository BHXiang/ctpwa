#!/usr/bin/env bash
# ctpwa 一键构建 + 上传 PyPI 脚本
# 用法:
#   bash scripts/release.sh                          # 构建并验证，不上传
#   bash scripts/release.sh --upload                 # 构建并上传到 PyPI
#   CUDA_VER=118 bash scripts/release.sh --upload    # 指定 CUDA 版本号
set -euo pipefail

# ========== 配置 ==========
# 自动检测或手动指定 CUDA 版本（格式: 118 / 124 / 130）
if [[ -n "${CUDA_VER:-}" ]]; then
    CUDA_TAG="cu${CUDA_VER}"
else
    # 自动从 CUDA_HOME 或 nvcc 检测
    if [[ -n "${CUDA_HOME:-}" ]]; then
        CUDA_VER=$(ls "$CUDA_HOME/lib64/libcudart.so."* 2>/dev/null | head -1 | grep -oP 'libcudart\.so\.\K[0-9]+' || echo "")
    fi
    if [[ -z "${CUDA_VER:-}" ]] && command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 | tr -d '.' | cut -c1-3)
    fi
    if [[ -z "${CUDA_VER:-}" ]]; then
        echo "ERROR: 无法检测 CUDA 版本，请设置 CUDA_VER 环境变量（如 CUDA_VER=118）"
        exit 1
    fi
    CUDA_TAG="cu${CUDA_VER}"
fi

# 从 setup.py 读取版本号
VERSION=$(grep -oP 'version="\K[^"]+' setup.py)
WHEEL_NAME="ctpwa-${VERSION}+${CUDA_TAG}-cp312-cp312-manylinux_2_34_x86_64.whl"

# 检查是否需要上传
DO_UPLOAD=false
if [[ "${1:-}" == "--upload" ]]; then
    DO_UPLOAD=true
fi

echo "=========================================="
echo " ctpwa 发布脚本"
echo "=========================================="
echo " 版本:     ${VERSION}"
echo " CUDA:     ${CUDA_TAG}"
echo " 上传:     ${DO_UPLOAD}"
echo "=========================================="
echo ""

# ========== 环境检查 ==========
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: $1 未安装，请先 pip install $1"
        exit 1
    fi
}

echo "[1/5] 检查环境..."
check_cmd nvcc
check_cmd python

# 检查 PyTorch
python -c "import torch; print('  torch:', torch.__version__)" 2>/dev/null || {
    echo "ERROR: PyTorch 未安装"
    exit 1
}

# 检查 ROOT
if ! command -v root-config &>/dev/null && [[ -z "${ROOTSYS:-}" ]]; then
    echo "WARNING: root-config 未找到且 ROOTSYS 未设置，编译可能失败"
fi

# 检查 CUDA
echo "  CUDA:   ${CUDA_TAG}"
python -c "import torch; assert '${CUDA_TAG}' in torch.__version__, f'torch CUDA 版本不匹配，期望 ${CUDA_TAG}，实际 {torch.__version__}'" || {
    echo "ERROR: PyTorch 的 CUDA 版本与目标不匹配，请安装 torch 的 ${CUDA_TAG} 版本"
    exit 1
}

# ========== 测试门禁 ==========
echo ""
echo "[2/6] 运行测试门禁 (pytest)..."

cd "$(dirname "$0")/.."
pip install -e . --no-build-isolation 2>&1 | tail -2
(cd tests && python3 -m pytest . -v --tb=short) || {
    echo "ERROR: 测试未通过，终止发布"
    exit 1
}
echo "  测试全部通过"

# ========== 构建 ==========
echo ""
echo "[3/6] 构建 wheel..."

# 清理
rm -rf build/ dist/*.whl 2>/dev/null || true

# 构建
python setup.py bdist_wheel 2>&1 | tail -3
ORIG_WHEEL=$(ls -1 dist/ctpwa-*-linux_x86_64.whl 2>/dev/null | head -1)
if [[ -z "$ORIG_WHEEL" ]]; then
    echo "ERROR: 构建失败，未生成 wheel"
    exit 1
fi
echo "  原始: $(basename $ORIG_WHEEL)"

# ========== auditwheel 修复 ==========
echo ""
echo "[4/6] auditwheel 修复..."

check_cmd auditwheel
check_cmd patchelf

auditwheel repair \
    --plat manylinux_2_34_x86_64 \
    --exclude "libCore.so*" --exclude "libImt.so*" --exclude "libRIO.so*" \
    --exclude "libNet.so*" --exclude "libHist.so*" --exclude "libGraf.so*" \
    --exclude "libGraf3d.so*" --exclude "libGpad.so*" --exclude "libROOTVecOps.so*" \
    --exclude "libTree.so*" --exclude "libTreePlayer.so*" --exclude "libRint.so*" \
    --exclude "libPostscript.so*" --exclude "libMatrix.so*" --exclude "libPhysics.so*" \
    --exclude "libMathCore.so*" --exclude "libThread.so*" --exclude "libROOTNTuple.so*" \
    --exclude "libROOTNTupleUtil.so*" --exclude "libMultiProc.so*" \
    --exclude "libROOTDataFrame.so*" \
    --exclude "libcudart.so*" --exclude "libcublas.so*" --exclude "libcublasLt.so*" \
    --exclude "libcufft.so*" --exclude "libcurand.so*" --exclude "libcusparse.so*" \
    --exclude "libcusolver.so*" --exclude "libnvrtc.so*" --exclude "libnvToolsExt.so*" \
    --exclude "libcudnn.so*" --exclude "libnvJitLink.so*" --exclude "libnvidia-ml.so*" \
    --exclude "libc10.so*" --exclude "libtorch.so*" --exclude "libtorch_cpu.so*" \
    --exclude "libtorch_cuda.so*" --exclude "libtorch_python.so*" --exclude "libc10_cuda.so*" \
    --exclude "libGL.so*" --exclude "libEGL.so*" --exclude "libOpenGL.so*" \
    --exclude "libGLX.so*" --exclude "libGLdispatch.so*" \
    -w dist/ "$ORIG_WHEEL" 2>&1 | grep -E "(INFO|ERROR|Fixed)" || true

# 重命名加上 CUDA 标签
REPAIRED=$(ls -1 dist/ctpwa-*-manylinux_2_34_x86_64.whl 2>/dev/null | head -1)
if [[ -z "$REPAIRED" ]]; then
    echo "ERROR: auditwheel 修复失败"
    exit 1
fi

WHEEL_PATH="dist/${WHEEL_NAME}"
mv "$REPAIRED" "$WHEEL_PATH"
echo "  产物: ${WHEEL_NAME} ($(du -h $WHEEL_PATH | cut -f1))"

# ========== 验证 ==========
echo ""
echo "[5/6] 验证 wheel..."

# 检查内容
python -c "
import zipfile
with zipfile.ZipFile('$WHEEL_PATH') as z:
    files = z.namelist()
    so_files = [f for f in files if f.endswith('.so') or f.endswith('.so.0')]
    lib_files = [f for f in files if 'ctpwa.libs/' in f]
    print(f'  .so 文件: {len(so_files)} 个')
    print(f'  捆绑的库: {len(lib_files)} 个')
    for l in lib_files:
        size_kb = z.getinfo(l).file_size / 1024
        print(f'    - {l.split(\"/\")[-1]} ({size_kb:.0f} KB)')
    # 检查是否有不该打包的库
    forbidden = ['libcublas', 'libcudart', 'libCore', 'libRIO', 'libtorch', 'libc10']
    for l in lib_files:
        name = l.split('/')[-1]
        for fb in forbidden:
            if name.startswith(fb):
                print(f'  WARNING: {name} 不应打包在 wheel 中!')
"

echo "  验证通过"

# ========== 上传 ==========
echo ""
if $DO_UPLOAD; then
    echo "[6/6] 上传到 PyPI..."
    check_cmd twine
    twine upload "$WHEEL_PATH"
    echo ""
    echo "上传完成: https://pypi.org/project/ctpwa/${VERSION}/"
else
    echo "[6/6] 跳过上传（使用 --upload 参数启用）"
    echo ""
    echo "产物路径: dist/${WHEEL_NAME}"
    echo ""
    echo "如需上传，执行:"
    echo "  twine upload dist/${WHEEL_NAME}"
    echo ""
    echo "或直接:"
    echo "  bash scripts/release.sh --upload"
fi
