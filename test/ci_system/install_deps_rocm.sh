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
export PIP_EXTRA_INDEX_URL="${ROCM_INDEX}"
TOKENSPEED_KERNEL_BACKEND=rocm pip3 install tokenspeed-kernel/python/ \
    --no-build-isolation -v

echo "=== Step 4: Install TokenSpeed Scheduler ==="
pip3 install cmake ninja
pip3 install tokenspeed-scheduler/

echo "=== Step 5: Install TokenSpeed ==="
# Pin smg / smg-grpc-servicer / smg-grpc-proto: the `tokenspeed` submodule
# that `ts serve` imports (smg_grpc_servicer.tokenspeed.server) only exists
# on these dev pins; the stable releases on PyPI drop it. The three .devN
# versions must stay in sync — the gRPC proto / runtime contract is
# dev-pinned.
pip3 install \
    "smg==1.4.2.dev15" \
    "smg-grpc-servicer==0.5.3.dev15" \
    "smg-grpc-proto==0.4.8.dev15" \
    --extra-index-url https://lightseek.org/whl/rocm72
pip3 install -e ./python --no-build-isolation \
    --extra-index-url "${ROCM_INDEX}"

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
