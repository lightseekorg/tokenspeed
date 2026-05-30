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
TOKENSPEED_TESTPYPI_INDEX=${TOKENSPEED_TESTPYPI_INDEX:-https://test.pypi.org/simple}

preinstall_tokenspeed_testpypi_packages() {
    local requirements_file="${WORKSPACE}/tokenspeed-kernel/python/requirements/common.txt"
    local vendor_requirements=()
    mapfile -t vendor_requirements < <(grep -E '^tokenspeed-(triton|proton)==' "${requirements_file}")
    if [ "${#vendor_requirements[@]}" -eq 0 ]; then
        echo "No tokenspeed vendor requirements found in ${requirements_file}" >&2
        return 1
    fi
    pip3 install --no-deps --index-url "${TOKENSPEED_TESTPYPI_INDEX}" \
        "${vendor_requirements[@]}"
}

echo "=========================================="
echo "GFX_ARCH=${GFX_ARCH}"
echo "ROCM_VERSION=${ROCM_VERSION}"
echo "WORKSPACE=${WORKSPACE}"
echo "=========================================="

echo "=== Step 1: apt deps ==="
sudo apt-get install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config

echo "=== Step 2: Upgrade pip/setuptools/wheel ==="
python3 -m pip install --upgrade pip setuptools wheel

echo "=== Step 3: Install tokenspeed-kernel ==="
cd "${WORKSPACE}"
echo "Preinstalling staged TokenSpeed vendor packages from ${TOKENSPEED_TESTPYPI_INDEX}"
preinstall_tokenspeed_testpypi_packages
export PIP_EXTRA_INDEX_URL="${ROCM_INDEX}"
TOKENSPEED_KERNEL_BACKEND=rocm pip3 install tokenspeed-kernel/python/ \
    --no-build-isolation -v

echo "=== Step 4: Install TokenSpeed Scheduler ==="
pip3 install cmake ninja
pip3 install tokenspeed-scheduler/

echo "=== Step 5: Install TokenSpeed ==="
# tokenspeed-smg / -grpc-servicer / -grpc-proto are pinned in
# python/pyproject.toml; pip resolves them from PyPI as part of the
# editable install below.
pip3 install -e ./python --no-build-isolation \
    --extra-index-url "${ROCM_INDEX}"

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
