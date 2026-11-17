#!/bin/bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Shared ROCm kernel dependencies for serving and simulator tests.
set -e

GFX_ARCH=${GFX_ARCH:-gfx950}
BUILD_AND_DOWNLOAD_PARALLEL=${BUILD_AND_DOWNLOAD_PARALLEL:-16}
# Torch pins carry the local build tag of the wheels on TORCH_INDEX_URL; keep
# them in sync so the reuse check in Step 3 only accepts builds for this ROCm.
TORCH_VERSION=${TORCH_VERSION:-2.14.0+rocm7.2}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.29.0+rocm7.2}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}
TORCH_DEVICE_PACKAGE=${TORCH_DEVICE_PACKAGE:-}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

echo "=== Step 3: Install PyTorch for ROCm ==="
torch_packages=("torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}")
if [ -n "${TORCH_DEVICE_PACKAGE}" ]; then
    torch_packages+=("${TORCH_DEVICE_PACKAGE}")
fi
if python3 "${SCRIPT_DIR}/verify_rocm_stack.py" torch \
    "${TORCH_VERSION}" "${TORCHVISION_VERSION}" "${TORCH_DEVICE_PACKAGE}"; then
    echo "Reusing matching PyTorch ${TORCH_VERSION} ROCm stack"
else
    pip_install_with_retry pip3 install --upgrade "${torch_packages[@]}" \
        --index-url "${TORCH_INDEX_URL}"
fi
python3 "${SCRIPT_DIR}/verify_rocm_stack.py" torch \
    "${TORCH_VERSION}" "${TORCHVISION_VERSION}" "${TORCH_DEVICE_PACKAGE}"

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
