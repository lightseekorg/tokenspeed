#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

SIM_ROOT=${TOKENSPEED_MI450_SIM_ROOT:-${RUNNER_TEMP:-/tmp}/tokenspeed-mi450-sim}
config="${SIM_ROOT}/gfx1250_mi455x.json"
rocm_root="$(rocm-sdk path --root)"
mirage="${rocm_root}/bin/mirage"
libhip_path="${rocm_root}/lib/libamdhip64.so"
thread_budget="${MI450_SIM_THREADS_PER_WORKER:-2}"

case "${thread_budget}" in
    "" | *[!0-9]* | 0)
        echo "MI450_SIM_THREADS_PER_WORKER must be a positive integer" >&2
        exit 2
        ;;
esac

test -x "${mirage}"
test -f "${config}"
test -f "${libhip_path}"

export ROCM_HOME="${rocm_root}"
export ROCM_PATH="${rocm_root}"
export LD_LIBRARY_PATH="${rocm_root}/lib:${LD_LIBRARY_PATH:-}"
# TokenSpeed Proton discovers the versioned TheRock HIP runtime first, but the
# stock Triton bundled with the gfx1250 PyTorch wheel only accepts the
# unversioned linker name. Both Triton distributions share this process, so
# point them at the same SDK runtime before either driver initializes.
export TRITON_LIBHIP_PATH="${libhip_path}"
# rocprofiler derives agents from physical KFD sysfs nodes, which do not exist
# for rocJITsu's synthetic HSA agent. Profiling is not part of this CI lane.
export ROCPROFILER_REGISTER_ENABLED=0
# Preserve the rejected code-object details when HIP cannot load a kernel.
export CUDA_LOG_FILE=${CUDA_LOG_FILE:-stderr}

exec "${mirage}" run --in-process \
    --config "${config}" \
    --cpu-thread-budget "${thread_budget}" -- \
    "$@"
