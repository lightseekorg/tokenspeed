#!/bin/bash
set -e

# ============================================================
# ROCm/AMD MI355 install script for TokenSpeed CI.
# ============================================================
GFX_ARCH=${GFX_ARCH:-gfx950}
ROCM_VERSION=${ROCM_VERSION:-7.2}
BUILD_AND_DOWNLOAD_PARALLEL=${BUILD_AND_DOWNLOAD_PARALLEL:-16}

ROCM_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

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
echo "ROCM_VERSION=${ROCM_VERSION}"
echo "WORKSPACE=${WORKSPACE}"
echo "=========================================="

echo "=== Step 1: apt deps ==="
sudo apt-get install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config

echo "=== Step 2: Upgrade pip/setuptools/wheel ==="
python3 -m pip install --upgrade pip "setuptools<82" wheel

echo "=== Step 3: Install tokenspeed-kernel packages ==="

cd "${WORKSPACE}"
# `tokenspeed-kernel` installs requirements/rocm.txt during its native build.
# Keep the matching in-tree AMD package installed first so that the minimum
# requirement is satisfied even before the public wheel exists.
pip3 install --force-reinstall --no-deps \
    "${WORKSPACE}/tokenspeed-kernel-amd" --no-build-isolation

cd "${WORKSPACE}"
export PIP_EXTRA_INDEX_URL="${ROCM_INDEX}"
TOKENSPEED_KERNEL_BACKEND=rocm \
pip_install_with_retry pip3 install tokenspeed-kernel/python/ \
    --no-build-isolation -v

echo "=== Step 4: Install TokenSpeed Scheduler ==="
pip_install_with_retry pip3 install cmake ninja
# Set TOKENSPEED_FLAT_KV=ON in a CI task env to build the scheduler with Flat KV.
SCHEDULER_PIP_ARGS=()
if [ "${TOKENSPEED_FLAT_KV:-OFF}" = "ON" ]; then
    SCHEDULER_PIP_ARGS+=(--config-settings=cmake.define.TOKENSPEED_FLAT_KVCACHE=ON)
fi
pip_install_with_retry pip3 install tokenspeed-scheduler/ "${SCHEDULER_PIP_ARGS[@]}"

echo "=== Step 5: Install TokenSpeed ==="
# tokenspeed-smg / -grpc-servicer / -grpc-proto are pinned in
# python/pyproject.toml; pip resolves them from PyPI as part of the
# editable install below.
pip_install_with_retry pip3 install -e ./python --no-build-isolation \
    --extra-index-url "${ROCM_INDEX}"

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
