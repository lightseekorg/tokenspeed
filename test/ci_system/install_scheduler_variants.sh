#!/bin/bash
set -euo pipefail

# Build-isolate the compile-time scheduler choice.  The differential runner
# prepends exactly one target directory to each worker's PYTHONPATH, so a flat
# worker can never accidentally import the radix extension installed by the
# regular CI bootstrap (or vice versa).
canonical_path() {
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

WORKSPACE=$(canonical_path "${WORKSPACE:-$(pwd)}")
VARIANT_ROOT=$(canonical_path "${TOKENSPEED_SCHEDULER_VARIANT_ROOT:-${WORKSPACE}/.ci-artifacts/scheduler-variants}")
OWNER_MARKER="${VARIANT_ROOT}/.tokenspeed-scheduler-variants-root"

# This script recursively replaces its output root.  Require an unambiguous,
# dedicated directory and refuse symlink aliases of the workspace or one of its
# ancestors.  An existing override must carry our marker so a typo cannot turn
# an arbitrary pre-existing directory into a deletion target.
if [[ "${VARIANT_ROOT##*/}" != "scheduler-variants" || \
      "${VARIANT_ROOT}" == "/" || \
      "${VARIANT_ROOT}" == "${WORKSPACE}" || \
      "${WORKSPACE}/" == "${VARIANT_ROOT}/"* ]]; then
    echo "Refusing unsafe scheduler variant root: ${VARIANT_ROOT}" >&2
    exit 2
fi
if [[ -e "${VARIANT_ROOT}" && ! -f "${OWNER_MARKER}" ]]; then
    echo "Refusing unowned scheduler variant root without marker: ${VARIANT_ROOT}" >&2
    exit 2
fi

rm -rf -- "${VARIANT_ROOT}"
mkdir -p "${VARIANT_ROOT}/radix" "${VARIANT_ROOT}/flat"
touch "${OWNER_MARKER}"

build_variant() {
    local name=$1
    local flat_kvcache=$2
    echo "=== Building tokenspeed-scheduler variant: ${name} (flat=${flat_kvcache}) ==="
    CMAKE_ARGS="-DTOKENSPEED_FLAT_KVCACHE=${flat_kvcache}" \
        python3 -m pip install \
        --no-deps \
        --no-build-isolation \
        --force-reinstall \
        --target "${VARIANT_ROOT}/${name}" \
        "${WORKSPACE}/tokenspeed-scheduler"
}

build_variant radix OFF
build_variant flat ON

PYTHONPATH="${VARIANT_ROOT}/radix" python3 -c \
    'import tokenspeed_scheduler as ts; assert ts.FLAT_KVCACHE is False'
PYTHONPATH="${VARIANT_ROOT}/flat" python3 -c \
    'import tokenspeed_scheduler as ts; assert ts.FLAT_KVCACHE is True'

echo "Scheduler variants installed under ${VARIANT_ROOT}"
