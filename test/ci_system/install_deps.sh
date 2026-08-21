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
source "${SCRIPT_DIR}/package_cache.sh"
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
configure_b200v2_package_cache

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

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

ensure_flashinfer_jit_cache() {
    # GB200 and B200 runner images preinstall flashinfer-jit-cache; it must
    # match the flashinfer-python pin exactly or flashinfer refuses to import.
    if [[ "${CI_RUNNER_LABEL:-}" != gb200* && "${CI_RUNNER_LABEL:-}" != b200* ]]; then
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
TORCH_CUDA_VERSION="$(python3 -c \
    'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
case "${TORCH_CUDA_VERSION}" in
    12.9*) CUINDEX=129 ;;
    13.0*) CUINDEX=130 ;;
    *)
        case "${CUDA_VERSION}" in
            12.9*) CUINDEX=129 ;;
            13.0*) CUINDEX=130 ;;
            *)     CUINDEX=130 ;;
        esac
        ;;
esac
echo "PyTorch CUDA version: ${TORCH_CUDA_VERSION:-unknown}"
echo "PyTorch wheel index: cu${CUINDEX}"

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
if ! dpkg -s openmpi-bin libopenmpi-dev libssl-dev pkg-config > /dev/null 2>&1; then
    run_as_root apt-get -o DPkg::Lock::Timeout=600 update
    run_as_root apt-get -o DPkg::Lock::Timeout=600 install -y \
        openmpi-bin libopenmpi-dev libssl-dev pkg-config
else
    echo "apt packages already installed, skipping apt"
fi
echo "=== Step 2: Upgrade pip/setuptools/wheel ==="
python3 -m pip install --upgrade --ignore-installed --break-system-packages \
    pip setuptools wheel

# ============================================================
# Step 3: Sync FlashInfer JIT cache on GB200/B200
# ============================================================
echo "=== Step 3: Sync FlashInfer JIT cache on GB200/B200 ==="
ensure_flashinfer_jit_cache

# ============================================================
# Step 4: Install tokenspeed-kernel
# ============================================================
echo "=== Step 4: Install tokenspeed-kernel ==="
# Nightly flashinfer pins (X.Y.Z.devYYYYMMDD) never reach PyPI: pre-install
# the python wheel from the GitHub nightly release so tokenspeed-kernel's ==
# pin resolves against the already-installed version instead of PyPI.
FLASHINFER_PIN="$(grep -E '^flashinfer-python==' "${CUDA_REQ}" | head -n1 | tr -d '[:space:]')"
FLASHINFER_PIN_VERSION="${FLASHINFER_PIN##*==}"
if [[ "${FLASHINFER_PIN_VERSION}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\.dev([0-9]{8})$ ]]; then
    NIGHTLY_TAG="nightly-v${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
    NIGHTLY_BASE="https://github.com/flashinfer-ai/flashinfer/releases/download/${NIGHTLY_TAG}"
    NIGHTLY_WHEEL="$(cache_remote_wheel "${NIGHTLY_BASE}/flashinfer_python-${FLASHINFER_PIN_VERSION}-py3-none-any.whl")"
    # flashinfer-cubin must match too: runner images preinstall an older one
    # and flashinfer's import-time version check refuses the mismatch.
    NIGHTLY_CUBIN_WHEEL="$(cache_remote_wheel "${NIGHTLY_BASE}/flashinfer_cubin-${FLASHINFER_PIN_VERSION}-py3-none-any.whl")"
    echo "Pre-installing nightly FlashInfer Python + cubin from GitHub release: ${NIGHTLY_TAG}"
    pip_install_with_retry pip3 install --break-system-packages --no-deps \
        "${NIGHTLY_WHEEL}" "${NIGHTLY_CUBIN_WHEEL}"
fi
cd ${WORKSPACE}
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu${CUINDEX}"
if [ "${TOKENSPEED_KERNEL_INSTALL_MODE:-source}" = "pypi" ]; then
    TOKENSPEED_KERNEL_PYPI_SPEC="${TOKENSPEED_KERNEL_PYPI_SPEC:-tokenspeed-kernel}"
    echo "Installing published tokenspeed-kernel wheel: ${TOKENSPEED_KERNEL_PYPI_SPEC}"
    pip_install_with_retry pip3 install --break-system-packages --no-deps \
        "${TOKENSPEED_KERNEL_PYPI_SPEC}"
    # The published wheel eagerly imports its vendor Triton bridge.
    TOKENSPEED_KERNEL_PYPI_RUNTIME_DEPS=(
        "tokenspeed-proton>=3.8.10.post20260721"
        "tokenspeed-triton>=3.8.10.post20260721"
    )
    pip_install_with_retry pip3 install --break-system-packages --no-deps \
        "${TOKENSPEED_KERNEL_PYPI_RUNTIME_DEPS[@]}"
    FLASHINFER_PYPI_SPEC="${TOKENSPEED_FLASHINFER_PYPI_SPEC:-flashinfer-python==0.6.16}"
    echo "Installing runner-compatible published FlashInfer: ${FLASHINFER_PYPI_SPEC}"
    pip_install_with_retry pip3 install --break-system-packages --no-deps \
        "${FLASHINFER_PYPI_SPEC}"
    FLASHINFER_CUBIN_SOURCE="$(cache_remote_wheel \
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.16/flashinfer_cubin-0.6.16-py3-none-any.whl")"
    pip_install_with_retry pip3 install --break-system-packages --no-deps \
        "${FLASHINFER_CUBIN_SOURCE}"
    TOKENSPEED_PYPI_RUNTIME_DEPS=(
        "aiohttp"
        "certifi"
        "charset-normalizer"
        "compressed-tensors"
        "dill"
        "einops"
        "fastapi"
        "grpcio-health-checking==1.81.1"
        "grpcio-reflection==1.81.1"
        "hf_transfer"
        "huggingface_hub"
        "idna"
        "modelscope"
        "msgspec"
        "ninja"
        "numpy"
        "openai>=2.24.1"
        "openai-harmony"
        "orjson"
        "packaging"
        "partial-json-parser"
        "peft"
        "pillow"
        "prometheus-client"
        "psutil"
        "pybase64"
        "pybind11"
        "pydantic"
        "pytest-asyncio"
        "python-multipart"
        "pyzmq"
        "requests"
        "setproctitle"
        "tiktoken"
        "tokenspeed-mooncake>=0.3.12.post20260803"
        "tokenspeed-smg==1.9.0.post20260815"
        "tokenspeed-smg-grpc-proto==0.4.14.post20260814"
        "tokenspeed-smg-grpc-servicer==0.8.0.post20260814"
        "torch-memory-saver==0.0.9.post1"
        "tqdm"
        "transformers==5.12.0"
        "urllib3"
        "uv"
        "uvicorn"
        "uvloop"
        "xgrammar==0.2.3"
        "viztracer"
    )
    pip_install_with_retry pip3 install --break-system-packages --no-deps \
        "${TOKENSPEED_PYPI_RUNTIME_DEPS[@]}"
else
    TOKENSPEED_KERNEL_BACKEND=cuda FLASHINFER_CUDA_ARCH_LIST="${FI_ARCH}" \
    pip_install_with_retry pip3 install tokenspeed-kernel/python/ --no-build-isolation -v
fi

# ============================================================
# Step 5: Install TokenSpeed Scheduler (C++)
# ============================================================
echo "=== Step 5: Install TokenSpeed Scheduler ==="
pip_install_with_retry pip3 install cmake ninja
pip_install_with_retry pip3 install tokenspeed-scheduler/

# ============================================================
# Step 6: Install TokenSpeed
# ============================================================
echo "=== Step 6: Install TokenSpeed ==="
# tokenspeed-smg / -grpc-servicer / -grpc-proto are pinned in
# python/pyproject.toml; pip resolves them from PyPI as part of the
# editable install below.
if [ "${TOKENSPEED_INSTALL_NO_DEPS:-0}" = "1" ]; then
    echo "Installing in-tree TokenSpeed without dependency resolution"
    pip_install_with_retry pip3 install --break-system-packages --no-deps -e "./python"
else
    pip_install_with_retry pip3 install -e "./python" \
        --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX}
fi

# ============================================================
# Step 7: Optionally override tokenspeed-mla with in-tree source
# ============================================================
# Set by the NVIDIA PR test workflows when the diff touches
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
if [ "${TOKENSPEED_KERNEL_INSTALL_MODE:-source}" = "pypi" ]; then
    echo "Skipping source-kernel dependency pinning for published wheel mode"
else
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
    # Nightlies version as X.Y.Z.devYYYYMMDD but tag as nightly-vX.Y.Z-YYYYMMDD,
    # and never reach PyPI, so their python wheel also comes from the release.
    FLASHINFER_RELEASE_BASE="https://github.com/flashinfer-ai/flashinfer/releases/download"
    if [[ "${FLASHINFER_VERSION}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\.dev([0-9]{8})$ ]]; then
        FLASHINFER_RELEASE_TAG="nightly-v${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
        FLASHINFER_PYTHON_SPEC="$(cache_remote_wheel "${FLASHINFER_RELEASE_BASE}/${FLASHINFER_RELEASE_TAG}/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl")"
    else
        FLASHINFER_RELEASE_TAG="v${FLASHINFER_VERSION}"
    fi
    FLASHINFER_CUBIN_WHEEL_URL="${FLASHINFER_RELEASE_BASE}/${FLASHINFER_RELEASE_TAG}/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl"
    FLASHINFER_CUBIN_WHEEL_SOURCE="$(cache_remote_wheel "${FLASHINFER_CUBIN_WHEEL_URL}")"
    echo "Force-reinstalling pinned FlashInfer Python: ${FLASHINFER_PYTHON_SPEC}"
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${FLASHINFER_PYTHON_SPEC}"
    echo "Installing FlashInfer cubin from GitHub Release: ${FLASHINFER_CUBIN_WHEEL_URL}"
    pip_install_with_retry pip3 install --break-system-packages \
        --force-reinstall --no-deps "${FLASHINFER_CUBIN_WHEEL_SOURCE}"
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
fi

# ============================================================
# Step 9: Fix Triton ptxas (CUDA 13+ only)
# ============================================================
echo "=== Step 9: Fix Triton ptxas ==="
if [ "${CUDA_VERSION%%.*}" = "13" ]; then
    TRITON_BIN="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin"
    if [ -d "${TRITON_BIN}" ]; then
        rm -f "${TRITON_BIN}/ptxas" 2>/dev/null || run_as_root rm -f "${TRITON_BIN}/ptxas" 2>/dev/null || true
        ln -sf /usr/local/cuda/bin/ptxas "${TRITON_BIN}/ptxas" 2>/dev/null || run_as_root ln -sf /usr/local/cuda/bin/ptxas "${TRITON_BIN}/ptxas" 2>/dev/null || true
    fi
fi

echo ""
echo "=========================================="
echo "Installed successfully! CUDA_VERSION=${CUDA_VERSION}, SM=${SM}"
echo "=========================================="
