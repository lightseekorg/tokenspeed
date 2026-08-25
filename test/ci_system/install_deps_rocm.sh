#!/bin/bash
set -e

# ============================================================
# ROCm/AMD MI355 install script for TokenSpeed CI.
# ============================================================
GFX_ARCH=${GFX_ARCH:-gfx950}
BUILD_AND_DOWNLOAD_PARALLEL=${BUILD_AND_DOWNLOAD_PARALLEL:-16}
TORCH_VERSION=${TORCH_VERSION:-2.13.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}
TORCH_DEVICE_PACKAGE=${TORCH_DEVICE_PACKAGE:-}

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
torch_device_name=${TORCH_DEVICE_PACKAGE%%==*}
if ! python3 -c 'import torch, torchvision' >/dev/null 2>&1 \
    || { [ -n "${TORCH_DEVICE_PACKAGE}" ] \
        && ! pip3 show "${torch_device_name}" >/dev/null 2>&1; }; then
    echo "Installing torch ${TORCH_VERSION} and matching ROCm packages"
    torch_packages=("torch==${TORCH_VERSION}" "torchvision==0.28.0")
    if [ -n "${TORCH_DEVICE_PACKAGE}" ]; then
        torch_packages+=("${TORCH_DEVICE_PACKAGE}")
    fi
    pip_install_with_retry pip3 install "${torch_packages[@]}" \
        --index-url "${TORCH_INDEX_URL}"
fi
python3 -c 'import torch, torchvision; assert torch.__version__.startswith("2.13.0"), torch.__version__; assert torchvision.__version__.startswith("0.28.0"), torchvision.__version__'

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
