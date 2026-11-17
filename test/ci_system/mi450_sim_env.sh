# Shared software stack for MI450 functional-simulator setup and CI.
# These pins override inherited values so both tasks install the same stack.
export ROCM_NIGHTLY_INDEX=https://nightly.repo.amd.com/rocm/whl-next/
export ROCM_SDK_VERSION=10.3.0a20261107

# PyTorch wheels require the exact ROCm SDK build they were built against.
export TORCH_INDEX_URL="${ROCM_NIGHTLY_INDEX}"
export TORCH_VERSION="2.14.0+rocm${ROCM_SDK_VERSION}"
export TORCHVISION_VERSION="0.29.0+rocm${ROCM_SDK_VERSION}"
export TORCH_DEVICE_PACKAGE="amd-torch-device-gfx1250==${TORCH_VERSION}"
