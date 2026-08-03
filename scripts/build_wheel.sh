#!/usr/bin/env bash
# Build a manylinux-compatible wheel for PyPI distribution.
# Prerequisites: ROOTSYS and CUDA_HOME set, auditwheel and patchelf installed.
set -euo pipefail

# ---------- config ----------
PLAT=${MANYLINUX_PLAT:-manylinux_2_34_x86_64}
OUTPUT_DIR="dist"
# 多架构 gencode（torch cpp_extension 生成；可被环境变量覆盖）
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5 8.0 8.6 8.9 9.0 10.0 12.0}"
# ---------------------------

# clean previous builds
rm -rf build/ "$OUTPUT_DIR"/*.whl 2>/dev/null || true

echo "==> Building wheel..."
python setup.py bdist_wheel

# find the wheel we just built
WHEEL=$(ls -1 "$OUTPUT_DIR"/ctpwa-*.whl | head -1)
echo "==> Built: $WHEEL"

# ----- figure out which libs to exclude -----
# ROOT libs: everything that looks like libRoot* or lib<ROOTModule>.so
# CUDA libs: libcudart*, libcublas*, libcufft*, libcurand*, libcusparse*, libnv*
# PyTorch libs: libtorch*, libc10*, libcaffe2*
# System libs that should NOT be bundled because they're too env-specific

# Patterns use fnmatch; * suffix matches versioned SONAMEs like libcudart.so.13
EXCLUDE_PATTERNS=(
    # ROOT libraries (user must source thisroot.sh)
    "libCore.so*"
    "libImt.so*"
    "libRIO.so*"
    "libNet.so*"
    "libHist.so*"
    "libGraf.so*"
    "libGraf3d.so*"
    "libGpad.so*"
    "libROOTVecOps.so*"
    "libTree.so*"
    "libTreePlayer.so*"
    "libRint.so*"
    "libPostscript.so*"
    "libMatrix.so*"
    "libPhysics.so*"
    "libMathCore.so*"
    "libThread.so*"
    "libROOTNTuple.so*"
    "libROOTNTupleUtil.so*"
    "libMultiProc.so*"
    "libROOTDataFrame.so*"
    # CUDA libraries (user must install CUDA toolkit)
    "libcudart.so*"
    "libcublas.so*"
    "libcublasLt.so*"
    "libcufft.so*"
    "libcurand.so*"
    "libcusparse.so*"
    "libcusolver.so*"
    "libnvrtc.so*"
    "libnvToolsExt.so*"
    "libcudnn.so*"
    "libnvJitLink.so*"
    "libnvidia-ml.so*"
    # PyTorch libraries (provided by torch package)
    "libc10.so*"
    "libtorch.so*"
    "libtorch_cpu.so*"
    "libtorch_cuda.so*"
    "libtorch_python.so*"
    "libc10_cuda.so*"
    # GL / graphics (should come from system)
    "libGL.so*"
    "libEGL.so*"
    "libOpenGL.so*"
    "libGLX.so*"
    "libGLdispatch.so*"
)

# build --exclude args
EXCLUDE_ARGS=()
for pat in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_ARGS+=(--exclude "$pat")
done

echo "==> Repairing wheel with auditwheel (plat=$PLAT)..."
auditwheel repair \
    --plat "$PLAT" \
    "${EXCLUDE_ARGS[@]}" \
    --wheel-dir "$OUTPUT_DIR" \
    "$WHEEL"

echo "==> Done! Repaired wheels:"
ls -lh "$OUTPUT_DIR"/ctpwa-*.whl
