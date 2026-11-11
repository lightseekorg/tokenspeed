#!/bin/bash
set -e

source "$(dirname "${BASH_SOURCE[0]}")/install_kernel_deps_rocm.sh"

echo "=== Step 5: Install TokenSpeed Scheduler ==="
pip_install_with_retry pip3 install cmake ninja
# Scheduler changes intentionally accumulate without a version bump. Build in
# a fresh directory so a persistent runner cannot reuse a same-version native
# extension left by an earlier checkout.
SCHEDULER_BUILD_DIR="$(mktemp -d)"
pip_install_with_retry pip3 install --force-reinstall --no-deps \
    tokenspeed-scheduler/ \
    --config-settings="build-dir=${SCHEDULER_BUILD_DIR}"
rm -rf "${SCHEDULER_BUILD_DIR}"
echo "=== Step 5b: Install TokenSpeed Engine ==="
WORKSPACE="${WORKSPACE}" bash "${WORKSPACE}/test/ci_system/install_tokenspeed_engine.sh"

echo "=== Step 6: Install TokenSpeed ==="
# tokenspeed-smg is pinned in python/pyproject.toml; pip resolves it
# from PyPI as part of the editable install below.
pip_install_with_retry pip3 install -e ./python --no-build-isolation

echo ""
echo "=========================================="
echo "ROCm install completed (GFX_ARCH=${GFX_ARCH})"
echo "=========================================="
