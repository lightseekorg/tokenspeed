#!/bin/bash
set -e

# ============================================================
# Platform dispatcher
#
# AMD/ROCm runners (e.g. amd-mi35x-* and amd-mi355-*) share the same
# install entry point in CI yaml configs, but need a different toolchain.
# Hand off to the ROCm-specific script when running on an AMD runner.
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMD_RUNNER_LABEL_PATTERNS=(*mi350* *mi355* *mi35x*)

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
CUDA_REQ="${WORKSPACE}/tokenspeed-kernel/python/requirements/cuda.txt"

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

ensure_flashinfer_jit_cache_for_gb200() {
    if [[ "${CI_RUNNER_LABEL:-}" != gb200* ]]; then
        return 0
    fi

    local wheel_url
    wheel_url="$(python3 "${SCRIPT_DIR}/flashinfer_jit_cache_installer.py" \
        --requirements "${CUDA_REQ}" \
        --cuda-index "${CUINDEX}")"
    if [ -z "${wheel_url}" ]; then
        return 0
    fi

    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${wheel_url}"
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
# Step 3: Sync FlashInfer JIT cache on GB200
# ============================================================
echo "=== Step 3: Sync FlashInfer JIT cache on GB200 ==="
ensure_flashinfer_jit_cache_for_gb200

# ============================================================
# Step 4: Install tokenspeed-kernel
# ============================================================
echo "=== Step 4: Install tokenspeed-kernel ==="
cd ${WORKSPACE}
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu${CUINDEX}"
TOKENSPEED_KERNEL_BACKEND=cuda FLASHINFER_CUDA_ARCH_LIST="${FI_ARCH}" \
pip_install_with_retry pip3 install tokenspeed-kernel/python/ --no-build-isolation -v

# ============================================================
# Step 5: Install TokenSpeed Scheduler (C++)
# ============================================================
echo "=== Step 5: Install TokenSpeed Scheduler ==="
pip_install_with_retry pip3 install cmake ninja
# Set TOKENSPEED_FLAT_KV=ON in a CI task env to build the scheduler with Flat KV.
SCHEDULER_PIP_ARGS=()
if [ "${TOKENSPEED_FLAT_KV:-OFF}" = "ON" ]; then
    SCHEDULER_PIP_ARGS+=(--config-settings=cmake.define.TOKENSPEED_FLAT_KVCACHE=ON)
fi
pip_install_with_retry pip3 install tokenspeed-scheduler/ "${SCHEDULER_PIP_ARGS[@]}"

# ============================================================
# Step 6: Install TokenSpeed
# ============================================================
echo "=== Step 6: Install TokenSpeed ==="
# tokenspeed-smg / -grpc-servicer / -grpc-proto are pinned in
# python/pyproject.toml; pip resolves them from PyPI as part of the
# editable install below.
pip_install_with_retry pip3 install -e "./python" \
    --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}

# ============================================================
# Step 7: Optionally override tokenspeed-mla with in-tree source
# ============================================================
# Set by `.github/workflows/pr-test-amd.yml` and
# `.github/workflows/pr-test-nvidia.yml` when the diff touches
# `tokenspeed-mla/`. Without this override CI exercises whichever
# `tokenspeed-mla` version is pinned in
# `tokenspeed-kernel/python/requirements/cuda-thirdparty.txt` and the
# in-tree change is silently ignored.
if [ "${INSTALL_TOKENSPEED_MLA_FROM_SOURCE:-0}" = "1" ]; then
    echo "=== Step 7: Reinstall tokenspeed-mla from in-tree source ==="
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${WORKSPACE}/tokenspeed-mla"
fi

# ============================================================
# Step 8: Pin critical kernel deps to exact versions
# ============================================================
echo "=== Step 8: Pin critical kernel deps ==="
pin_version() {
    # Extract exact-pinned package specs, including optional extras.
    local pkg="$1"
    grep -E "^${pkg}(\[[^]]+\])?==" "${CUDA_REQ}" | head -n1 | tr -d '[:space:]'
}
CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUTLASS_DSL_SPEC="$(pin_version nvidia-cutlass-dsl)"
if [ -n "${CUTLASS_DSL_SPEC}" ]; then
    CUTLASS_DSL_VERSION="${CUTLASS_DSL_SPEC##*==}"
    CUTLASS_DSL_DEPS=(
        "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}"
        "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_VERSION}"
        "nvidia-cutlass-dsl-libs-core==${CUTLASS_DSL_VERSION}"
        "nvidia-cutlass-dsl-libs-cu${CUDA_MAJOR}==${CUTLASS_DSL_VERSION}"
    )
    echo "Force-reinstalling pinned Cutlass DSL packages: ${CUTLASS_DSL_DEPS[*]}"
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${CUTLASS_DSL_DEPS[@]}"
fi

FLASHINFER_PYTHON_SPEC="$(pin_version flashinfer-python)"
if [ -n "${FLASHINFER_PYTHON_SPEC}" ]; then
    FLASHINFER_VERSION="${FLASHINFER_PYTHON_SPEC##*==}"
    FLASHINFER_CUBIN_WHEEL_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/v${FLASHINFER_VERSION}/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl"
    echo "Force-reinstalling pinned FlashInfer Python: ${FLASHINFER_PYTHON_SPEC}"
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${FLASHINFER_PYTHON_SPEC}"
    echo "Installing FlashInfer cubin from GitHub Release: ${FLASHINFER_CUBIN_WHEEL_URL}"
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${FLASHINFER_CUBIN_WHEEL_URL}"
else
    echo "No FlashInfer Python pin found in ${CUDA_REQ}; skipping FlashInfer installs."
fi

THIRDPARTY_REQ="${WORKSPACE}/tokenspeed-kernel/python/requirements/cuda-thirdparty.txt"
FA4_SPEC="$(grep -E '^tokenspeed-fa4(\[[^]]+\])?==' "${THIRDPARTY_REQ}" | head -n1 | tr -d '[:space:]')"
if [ -n "${FA4_SPEC}" ]; then
    echo "Force-reinstalling pinned FA4: ${FA4_SPEC}"
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${FA4_SPEC}"
else
    echo "No tokenspeed-fa4 pin found in ${THIRDPARTY_REQ}; skipping FA4 reinstall."
fi

# ============================================================
# Step 9: Fix Triton ptxas (CUDA 13+ only)
# ============================================================
echo "=== Step 9: Fix Triton ptxas ==="
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
