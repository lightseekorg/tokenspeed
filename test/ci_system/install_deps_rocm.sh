#!/bin/bash
set -e

source "$(dirname "${BASH_SOURCE[0]}")/install_kernel_deps_rocm.sh"

echo "=== Step 5: Install TokenSpeed Engine ==="
WORKSPACE="${WORKSPACE}" bash "${WORKSPACE}/test/ci_system/install_tokenspeed_engine.sh"

echo "=== Step 6: Install TokenSpeed ==="
# tokenspeed-smg is pinned in python/pyproject.toml; pip resolves it
# from PyPI as part of the editable install below.
pip_install_with_retry pip3 install -e ./python --no-build-isolation

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
