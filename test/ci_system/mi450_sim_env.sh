# Shared software stack for MI450 functional-simulator setup and CI.
# These pins override inherited values so both tasks install the same stack.
export ROCM_NIGHTLY_INDEX=https://nightly.repo.amd.com/rocm/whl-next/
export ROCM_SDK_VERSION=10.3.0a20261107
# Match the SDK's share/therock/therock_manifest.json.
export ROCM_SYSTEMS_REF=3f917c0d61869b153dae1264e712b414a9343d10

# PyTorch wheels require the exact ROCm SDK build they were built against.
export TORCH_INDEX_URL="${ROCM_NIGHTLY_INDEX}"
export TORCH_VERSION="2.16.0a0+rocm${ROCM_SDK_VERSION}"
export TORCHVISION_VERSION="0.30.0a0+rocm${ROCM_SDK_VERSION}"
export TORCH_DEVICE_PACKAGE="amd-torch-device-gfx1250==${TORCH_VERSION}"
