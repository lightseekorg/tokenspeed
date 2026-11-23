#!/bin/bash
set -e

source "$(dirname "${BASH_SOURCE[0]}")/install_kernel_deps_rocm.sh"

echo "=== Step 5: Install TokenSpeed Engine ==="
WORKSPACE="${WORKSPACE}" bash "${WORKSPACE}/test/ci_system/install_tokenspeed_engine.sh"

echo "=== Step 6: Install TokenSpeed ==="
# tokenspeed-smg is pinned in python/pyproject.toml; pip resolves it
# from PyPI as part of the editable install below.
pip_install_with_retry pip3 install -e ./python --no-build-isolation

echo "=== Step 7: Link torch's amdgpu.ids for libdrm ==="
# torch's bundled libdrm reads GPU names only from
# /opt/amdgpu/share/libdrm/amdgpu.ids, which the runner images lack. Without it
# torch.cuda.get_device_name() reports "AMD Radeon Graphics" on MI355X, and the
# device_name= tuned GEMM file does not load. Link the copy torch ships, and
# keep an existing file.
AMDGPU_IDS=/opt/amdgpu/share/libdrm/amdgpu.ids
if [ ! -e "${AMDGPU_IDS}" ]; then
    TORCH_AMDGPU_IDS="$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "share/libdrm/amdgpu.ids")')"
    if [ ! -f "${TORCH_AMDGPU_IDS}" ]; then
        echo "torch does not ship ${TORCH_AMDGPU_IDS}" >&2
        exit 1
    fi
    sudo mkdir -p "$(dirname "${AMDGPU_IDS}")"
    sudo ln -sfn "${TORCH_AMDGPU_IDS}" "${AMDGPU_IDS}"
fi
echo "amdgpu.ids: $(readlink -f "${AMDGPU_IDS}")"

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
