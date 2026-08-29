#!/bin/bash
set -euo pipefail

ROCM_SYSTEMS_REF=${ROCM_SYSTEMS_REF:-79f87b8238f04a255707636719ba0a9f525753dd}
ROCM_SDK_VERSION=${ROCM_SDK_VERSION:-10.1.0a20260812}
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
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[devel,libraries]==${ROCM_SDK_VERSION}" \
    "rocm-sdk-device-gfx1250==${ROCM_SDK_VERSION}"
sudo "$(command -v rocm-sdk)" init

mkdir -p "${SIM_ROOT}"
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
git -C "${SOURCE_ROOT}" fetch --depth 1 origin "${ROCM_SYSTEMS_REF}"
git -C "${SOURCE_ROOT}" checkout --detach "${ROCM_SYSTEMS_REF}"

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
if [ -x "${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu" ] \
    && [ -f "${ROCJITSU_BUILD_DIR}/librocjitsu.so" ]; then
    echo "Reusing cached rocJITsu launcher and runtime"
else
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
fi

test -x "${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
test -f "${ROCJITSU_SOURCE_DIR}/configs/gfx1250_mi455x.json"
