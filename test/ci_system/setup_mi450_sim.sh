#!/bin/bash
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/mi450_sim_env.sh"
UV_VERSION=${UV_VERSION:-0.9.26}
SIM_ROOT=${TOKENSPEED_MI450_SIM_ROOT:-${RUNNER_TEMP:-/tmp}/tokenspeed-mi450-sim}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP_CACHE_DIR=${PIP_CACHE_DIR:-${RUNNER_TEMP:-/tmp}/pip-cache}
export PIP_CACHE_DIR
export UV_CACHE_DIR=${UV_CACHE_DIR:-${PIP_CACHE_DIR}/uv}

mi450_stack_ready() {
    python3 "${SCRIPT_DIR}/verify_rocm_stack.py" \
        mi450-sdk "${UV_VERSION}" "${ROCM_SDK_VERSION}" \
        || return 1
    command -v uv >/dev/null || return 1
    command -v rocm-sdk >/dev/null || return 1
    local existing_rocm_root
    existing_rocm_root="$(rocm-sdk path --root)" || return 1
    test -x "${existing_rocm_root}/bin/mirage" || return 1
    test -f "${existing_rocm_root}/lib/libamdhip64.so" || return 1
    test -f "${existing_rocm_root}/share/rocjitsu/configs/gfx1250_mi455x.json" \
        || test -f "${existing_rocm_root}/share/rocjitsu/configs/gfx1250.json"
}

if mi450_stack_ready; then
    echo "Reusing matching ROCm ${ROCM_SDK_VERSION} simulator stack"
else
    sudo apt-get install -y --no-install-recommends \
        ca-certificates

    python3 -m pip install --disable-pip-version-check "uv==${UV_VERSION}"
    pip3 install "pytest>=9" pytest-timeout pytest-xdist pytest-reportlog
    sudo env "UV_CACHE_DIR=${UV_CACHE_DIR}" "$(command -v uv)" pip install \
        --system \
        --break-system-packages \
        --prerelease allow \
        --index-url "${ROCM_NIGHTLY_INDEX}" \
        "rocm[devel,libraries]==${ROCM_SDK_VERSION}" \
        "rocm-sdk-device-gfx1250==${ROCM_SDK_VERSION}"
    sudo "$(command -v rocm-sdk)" init
    python3 "${SCRIPT_DIR}/verify_rocm_stack.py" \
        mi450-sdk "${UV_VERSION}" "${ROCM_SDK_VERSION}"
fi

mkdir -p "${SIM_ROOT}"
rocm_root="$(rocm-sdk path --root)"
mirage="${rocm_root}/bin/mirage"
sdk_config="${rocm_root}/share/rocjitsu/configs/gfx1250_mi455x.json"
if [ ! -f "${sdk_config}" ]; then
    sdk_config="${rocm_root}/share/rocjitsu/configs/gfx1250.json"
fi
config_path="${SIM_ROOT}/gfx1250_mi455x.json"

test -x "${mirage}"
test -f "${sdk_config}"
cp "${sdk_config}" "${config_path}"

# HIP initialization needs the KMD simulator to remain alive for the full
# process lifetime. The upstream gfx1250 functional config has a finite limit.
python3 - "${config_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
config = json.loads(path.read_text())
config["max_ticks"] = 0
path.write_text(json.dumps(config, indent=2) + "\n")
PY
test -f "${config_path}"
