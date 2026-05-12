#!/bin/bash
set -e

# ============================================================
# Platform dispatcher
#
# AMD/ROCm runners (e.g. linux-mi355-*) share the same install entry
# point in CI yaml configs, but need a different toolchain. Hand off
# to the ROCm-specific script when running on an AMD runner.
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_RUNNER_LABEL_PATTERNS=(linux-mi350* linux-mi355*)

for pat in "${AMD_RUNNER_LABEL_PATTERNS[@]}"; do
    if [[ "${CI_RUNNER_LABEL:-}" == ${pat} ]]; then
        echo "Detected AMD runner '${CI_RUNNER_LABEL}', delegating to install_deps_rocm.sh"
        exec bash "${SCRIPT_DIR}/install_deps_rocm.sh" "$@"
    fi
done

# ============================================================
# Configuration
# ============================================================
CUDA_VERSION=${CUDA_VERSION:-13.0.1}
SM=${SM:-sm100}
BUILD_AND_DOWNLOAD_PARALLEL=${BUILD_AND_DOWNLOAD_PARALLEL:-16}

export MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL}
export CPLUS_INCLUDE_PATH="/usr/local/cuda/include/cccl"
export C_INCLUDE_PATH="/usr/local/cuda/include/cccl"

WORKSPACE=${WORKSPACE:-$(pwd)}

# Wrap pip install in a retry loop. PyPI's CDN occasionally returns a
# bad Content-Type for /simple/<pkg>/ pages (most recently observed for
# starlette on 2026-04-30); pip silently skips those pages, fails to
# find any version, and the resolver gives up. pip's own --retries flag
# does not retry past that warning, so we wrap the whole invocation.
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
echo "SM=${SM}, CUDA_VERSION=${CUDA_VERSION}"
echo "WORKSPACE=${WORKSPACE}"
echo "=========================================="

# ============================================================
# Step 1: Determine CUDA index and FlashInfer architecture
# ============================================================
echo "=== Step 1: Determine CUDA index and architecture ==="
case "${CUDA_VERSION}" in
    12.9.1) CUINDEX=129 ;;
    13.0.1) CUINDEX=130 ;;
    *)      CUINDEX=130 ;;
esac
echo "PyTorch CUDA index: cu${CUINDEX}"

case "${SM}" in
    sm103) FI_ARCH="10.3a" ;;
    sm100) FI_ARCH="10.0a" ;;
    sm90)  FI_ARCH="9.0a" ;;
    *)     echo "Unknown SM: ${SM}" && exit 1 ;;
esac
echo "FlashInfer architecture: ${FI_ARCH}"

# ============================================================
# Step 2: Upgrade base tools
# ============================================================
sudo apt install -y openmpi-bin libopenmpi-dev libssl-dev pkg-config -y
echo "=== Step 2: Upgrade pip/setuptools/wheel ==="
python3 -m pip install --upgrade pip setuptools wheel

# ============================================================
# Step 3: Install tokenspeed-kernel
# ============================================================
echo "=== Step 3: Install tokenspeed-kernel ==="
cd ${WORKSPACE}
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu${CUINDEX}"
TOKENSPEED_KERNEL_BACKEND=cuda FLASHINFER_CUDA_ARCH_LIST="${FI_ARCH}" \
pip_install_with_retry pip3 install tokenspeed-kernel/python/ --no-build-isolation -v

# ============================================================
# Step 4: Install TokenSpeed Scheduler (C++)
# ============================================================
echo "=== Step 4: Install TokenSpeed Scheduler ==="
pip_install_with_retry pip3 install cmake ninja
pip_install_with_retry pip3 install tokenspeed-scheduler/

# ============================================================
# Step 5: Install TokenSpeed
# ============================================================
echo "=== Step 5: Install TokenSpeed ==="
# Pin smg / smg-grpc-servicer / smg-grpc-proto: the `tokenspeed` submodule
# that `ts serve` imports (smg_grpc_servicer.tokenspeed.server) only exists
# on these dev pins; later versions drop it. The three .devN versions must
# stay in sync — the gRPC proto / runtime contract is dev-pinned.
pip_install_with_retry pip3 install \
    "smg==1.4.2.dev15" \
    "smg-grpc-servicer==0.5.3.dev15" \
    "smg-grpc-proto==0.4.8.dev15" \
    --extra-index-url https://lightseek.org/whl/cu130
pip_install_with_retry pip3 install -e "./python[cuda_${SM}]" \
    --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# ============================================================
# Step 6: Fix Triton ptxas (CUDA 13+ only)
# ============================================================
echo "=== Step 6: Fix Triton ptxas ==="
if [ "${CUDA_VERSION%%.*}" = "13" ]; then
    TRITON_BIN="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin"
    if [ -d "${TRITON_BIN}" ]; then
        rm -f "${TRITON_BIN}/ptxas" 2>/dev/null || sudo rm -f "${TRITON_BIN}/ptxas" 2>/dev/null || true
        ln -sf /usr/local/cuda/bin/ptxas "${TRITON_BIN}/ptxas" 2>/dev/null || sudo ln -sf /usr/local/cuda/bin/ptxas "${TRITON_BIN}/ptxas" 2>/dev/null || true
    fi
fi

echo ""
echo "=========================================="
echo "Installed successfully! CUDA_VERSION=${CUDA_VERSION}, SM=${SM}"
echo "=========================================="
