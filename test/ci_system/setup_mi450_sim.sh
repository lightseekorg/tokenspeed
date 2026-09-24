#!/bin/bash
set -euo pipefail

: "${ROCM_SYSTEMS_REF:?ROCM_SYSTEMS_REF must be set}"
: "${ROCM_NIGHTLY_INDEX:?ROCM_NIGHTLY_INDEX must be set}"
: "${ROCM_SDK_VERSION:?ROCM_SDK_VERSION must be set}"
UV_VERSION=${UV_VERSION:-0.9.26}
SIM_ROOT=${TOKENSPEED_MI450_SIM_ROOT:-${RUNNER_TEMP:-/tmp}/tokenspeed-mi450-sim}
SOURCE_ROOT="${SIM_ROOT}/rocm-systems"
ROCJITSU_SOURCE_DIR="${SOURCE_ROOT}/emulation/rocjitsu"
ROCJITSU_BUILD_DIR="${SIM_ROOT}/rocjitsu-build"

sudo apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    git \
    libclang-rt-dev \
    libdrm-dev \
    ninja-build

python3 -m pip install --disable-pip-version-check "uv==${UV_VERSION}"
pip3 install pytest-timeout pytest-xdist pytest-reportlog
sudo "$(command -v uv)" pip install --system --break-system-packages --prerelease allow \
    --index-url "${ROCM_NIGHTLY_INDEX}" \
    "rocm[devel,libraries]==${ROCM_SDK_VERSION}" \
    "rocm-sdk-device-gfx1250==${ROCM_SDK_VERSION}"
sudo "$(command -v rocm-sdk)" init

mkdir -p "${SIM_ROOT}"
# The ROCm nightly includes librocjitsu.so and its configs, but not the
# `rocjitsu --daemon` launcher this CI lane needs. Keep building the launcher
# from the ROCJITsu source revision aligned with the pinned nightly.
if [ ! -d "${SOURCE_ROOT}/.git" ]; then
    git clone \
        --filter=blob:none \
        --no-checkout \
        https://github.com/ROCm/rocm-systems.git \
        "${SOURCE_ROOT}"
    git -C "${SOURCE_ROOT}" sparse-checkout init --cone
    git -C "${SOURCE_ROOT}" sparse-checkout set \
        emulation/rocjitsu \
        shared/machine-readable-isa/isa
fi
if ! git -C "${SOURCE_ROOT}" cat-file -e "${ROCM_SYSTEMS_REF}^{commit}"; then
    for attempt in 1 2 3; do
        if git -C "${SOURCE_ROOT}" fetch --depth 1 origin "${ROCM_SYSTEMS_REF}"; then
            break
        fi
        if [ "${attempt}" -eq 3 ]; then
            echo "Failed to fetch rocm-systems after ${attempt} attempts" >&2
            exit 1
        fi
        echo "rocm-systems fetch attempt ${attempt} failed; retrying in 10s..." >&2
        sleep 10
    done
fi
# The max_ticks adjustment below dirties a tracked config on every run.
# Reset it before switching source revisions on a reused runner.
git -C "${SOURCE_ROOT}" checkout --force --detach "${ROCM_SYSTEMS_REF}"

# HIP initialization needs the KMD simulator to remain alive for the full
# process lifetime. The upstream gfx1250 functional config has a finite limit.
python3 - "${ROCJITSU_SOURCE_DIR}/configs/gfx1250_mi455x.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
config = json.loads(path.read_text())
config["max_ticks"] = 0
path.write_text(json.dumps(config, indent=2) + "\n")
PY

rocm_root="$(rocm-sdk path --root)"
build_stamp="${ROCJITSU_BUILD_DIR}/.tokenspeed-build-id"
build_id="${ROCM_SYSTEMS_REF}:${ROCM_SDK_VERSION}:${rocm_root}"
if [ -x "${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu" ] \
    && [ -f "${ROCJITSU_BUILD_DIR}/librocjitsu.so" ] \
    && [ -f "${build_stamp}" ] \
    && [ "$(cat "${build_stamp}")" = "${build_id}" ]; then
    echo "Reusing cached rocJITsu launcher and runtime"
else
    # A previous runner may have built this directory against another ROCm
    # SDK or source revision. Do not run those binaries with this nightly.
    rm -rf "${ROCJITSU_BUILD_DIR}"
    ROCM_HOME="${rocm_root}" \
    ROCM_PATH="${rocm_root}" \
    LD_LIBRARY_PATH="${rocm_root}/lib:${LD_LIBRARY_PATH:-}" \
        cmake \
            -S "${ROCJITSU_SOURCE_DIR}" \
            -B "${ROCJITSU_BUILD_DIR}" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF
    cmake --build "${ROCJITSU_BUILD_DIR}" \
        --target rocjitsu_bin rocjitsu_shared \
        --parallel 4
    printf '%s\n' "${build_id}" > "${build_stamp}"
fi

test -x "${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
test -f "${ROCJITSU_SOURCE_DIR}/configs/gfx1250_mi455x.json"
