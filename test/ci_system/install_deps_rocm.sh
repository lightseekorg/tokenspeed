#!/bin/bash
set -e

# ============================================================
# ROCm/AMD MI355 install script for TokenSpeed CI.
# ============================================================
GFX_ARCH=${GFX_ARCH:-gfx950}
BUILD_AND_DOWNLOAD_PARALLEL=${BUILD_AND_DOWNLOAD_PARALLEL:-16}

ROCM_INDEX="https://repo.amd.com/rocm/whl-multi-arch/"

export MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL}
WORKSPACE=${WORKSPACE:-$(pwd)}

pip_install_with_retry() {
    local max_attempts=5
    local attempt=1
    local delay=10
    while [ "${attempt}" -le "${max_attempts}" ]; do
        if "$@"; then
            return 0
        fi
        if [ "${attempt}" -eq "${max_attempts}" ]; then
            echo "pip install failed after ${max_attempts} attempts: $*" >&2
            return 1
        fi
        echo "pip install attempt ${attempt}/${max_attempts} failed; retrying in ${delay}s..." >&2
        sleep "${delay}"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

echo "=========================================="
echo "GFX_ARCH=${GFX_ARCH}"
echo "WORKSPACE=${WORKSPACE}"
echo "=========================================="

echo "=== Step 1: apt deps ==="
sudo apt-get install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config

echo "=== Step 2: Upgrade pip/setuptools/wheel ==="
pip install --upgrade pip "setuptools<82" wheel

echo "=== Step 3: Check PyTorch for ROCm ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}, ROCm support: {torch.version.hip}')"

echo "=== Step 4: Install tokenspeed-kernel packages ==="

cd "${WORKSPACE}"
# `tokenspeed-kernel` installs requirements/rocm.txt during its native build.
# Keep the matching in-tree AMD package installed first so that the minimum
# requirement is satisfied even before the public wheel exists.
pip3 install --force-reinstall --no-deps \
    "${WORKSPACE}/tokenspeed-kernel-amd" --no-build-isolation

cd "${WORKSPACE}"

TOKENSPEED_KERNEL_BACKEND=rocm \
pip_install_with_retry pip3 install tokenspeed-kernel/python/ \
    --no-build-isolation -v

echo "=== Step 5: Install TokenSpeed Scheduler ==="
pip_install_with_retry pip3 install cmake ninja
pip_install_with_retry pip3 install tokenspeed-scheduler/

echo "=== Step 6: Install TokenSpeed ==="
# tokenspeed-smg / -grpc-servicer / -grpc-proto are pinned in
# python/pyproject.toml; pip resolves them from PyPI as part of the
# editable install below.
pip_install_with_retry pip3 install -e ./python --no-build-isolation

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
