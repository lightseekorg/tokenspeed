#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

SIM_ROOT=${TOKENSPEED_MI450_SIM_ROOT:-${RUNNER_TEMP:-/tmp}/tokenspeed-mi450-sim}
ROCJITSU_SOURCE_DIR="${SIM_ROOT}/rocm-systems/emulation/rocjitsu"
ROCJITSU_BUILD_DIR="${SIM_ROOT}/rocjitsu-build"
launcher="${ROCJITSU_BUILD_DIR}/tools/rocjitsu/rocjitsu"
config="${ROCJITSU_SOURCE_DIR}/configs/gfx1250_mi455x.json"
rocm_root="$(rocm-sdk path --root)"

test -x "${launcher}"
test -f "${config}"

export ROCM_HOME="${rocm_root}"
export ROCM_PATH="${rocm_root}"
export LD_LIBRARY_PATH="${rocm_root}/lib:${LD_LIBRARY_PATH:-}"
# rocprofiler derives agents from physical KFD sysfs nodes, which do not exist
# for rocJITsu's synthetic HSA agent. Profiling is not part of this CI lane.
export ROCPROFILER_REGISTER_ENABLED=0
# Preserve the rejected code-object details when HIP cannot load a kernel.
export CUDA_LOG_FILE=${CUDA_LOG_FILE:-stderr}

exec "${launcher}" --daemon --config "${config}" -- "$@"
