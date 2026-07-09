#!/bin/bash
# Best-effort cleanup of stale GPU processes before an AMD/ROCm CI task.
# Mitigates the race condition between pod termination and GPU resource
# release: previously terminated pods sometimes leave processes (or zombie
# / D-state tasks) holding VRAM, which causes torch.OutOfMemoryError when
# the next task tries to load a large model (e.g. openai/gpt-oss-120b).
#
# Best-effort: never aborts; never propagates non-zero status. Stale procs
# living in another PID namespace cannot be signalled from this pod and
# are reported as a WARNING for the cluster admin to handle.
set +e

WAIT_AFTER_TERM_SECS=${TOKENSPEED_AMD_GPU_WAIT_AFTER_TERM:-3}
WAIT_AFTER_KILL_SECS=${TOKENSPEED_AMD_GPU_WAIT_AFTER_KILL:-5}

_section() {
    echo ""
    echo "=========================================================================="
    echo "  $*"
    echo "=========================================================================="
}

_run() {
    local label="$1"; shift
    echo "----- ${label} -----"
    "$@" 2>&1 || true
    echo ""
}

# Build the ancestor-PID list so we never SIGKILL ourselves or the runner
# bash that invoked us (would abort the whole CI step).
self_pid=$$
ancestors=" $self_pid "
p=$self_pid
while :; do
    pp=$(awk '/^PPid:/ {print $2}' "/proc/${p}/status" 2>/dev/null)
    [ -z "$pp" ] || [ "$pp" = "0" ] && break
    ancestors="${ancestors}${pp} "
    p=$pp
done
echo "[cleanup_amd_gpu_state] self_pid=${self_pid} ancestors=${ancestors}"

_in_ancestors() {
    case "$ancestors" in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

_section "GPU state BEFORE cleanup"
# Show AMD device information before any cleanup action.
_run "rocm-smi --showid" rocm-smi --showid
_run "rocm-smi --showpids" rocm-smi --showpids

_section "Discovering GPU-holding PIDs"
gpu_pids=""

# (1) Scan /proc/<pid>/fd for any fd pointing at /dev/kfd or /dev/dri/renderD*.
#     This catches anyone using HSA / DRM render nodes regardless of cmdline.
for fd_dir in /proc/[0-9]*/fd; do
    pid="${fd_dir#/proc/}"
    pid="${pid%/fd}"
    if _in_ancestors "$pid"; then
        continue
    fi
    if ls -l "$fd_dir" 2>/dev/null | grep -qE "/dev/kfd|/dev/dri/renderD"; then
        gpu_pids="${gpu_pids} ${pid}"
    fi
done

# (2) rocm-smi --showpids reports anything the GPU firmware sees, including
#     processes that no longer have the device fd open but still hold VRAM.
if command -v rocm-smi >/dev/null 2>&1; then
    extra=$(rocm-smi --showpids 2>/dev/null \
        | awk '/^[ \t]*[0-9]+/ {print $1}' \
        | sort -u)
    for pid in $extra; do
        if _in_ancestors "$pid"; then
            continue
        fi
        gpu_pids="${gpu_pids} ${pid}"
    done
fi

# Dedup
gpu_pids=$(echo "$gpu_pids" | tr ' ' '\n' | awk 'NF' | sort -un | tr '\n' ' ')

if [ -z "${gpu_pids// /}" ]; then
    echo "No GPU-holding processes found (excluding ourselves)."
else
    echo "GPU-holding PIDs: ${gpu_pids}"
    echo ""
    echo "----- forensic ps -----"
    for pid in $gpu_pids; do
        info=$(ps -o pid=,ppid=,user=,stat=,etime=,cmd= -p "$pid" 2>/dev/null)
        if [ -n "$info" ]; then
            echo "  ${info}"
        else
            echo "  pid=${pid} (already gone)"
        fi
    done
    echo ""

    _section "Killing GPU-holding PIDs"
    echo "Sending SIGTERM..."
    for pid in $gpu_pids; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep "${WAIT_AFTER_TERM_SECS}"

    survivors=""
    for pid in $gpu_pids; do
        if [ -d "/proc/${pid}" ]; then
            survivors="${survivors} ${pid}"
        fi
    done
    if [ -n "${survivors// /}" ]; then
        echo "SIGTERM survivors: ${survivors}; sending SIGKILL..."
        for pid in $survivors; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep "${WAIT_AFTER_KILL_SECS}"
    fi

    still_alive=""
    for pid in $gpu_pids; do
        if [ -d "/proc/${pid}" ]; then
            stat=$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null)
            still_alive="${still_alive} ${pid}(${stat})"
        fi
    done
    if [ -n "${still_alive// /}" ]; then
        echo ""
        echo "WARNING: the following PIDs survived SIGKILL: ${still_alive}"
        echo "  states: Z=zombie, D=uninterruptible sleep, R/S=running/sleep."
        echo "  Zombie or D-state means the kernel cannot reap them right"
        echo "  now (likely waiting on GPU driver). Cross-PID-namespace"
        echo "  processes (other pods on the same node) cannot be signalled"
        echo "  from inside this pod at all. If VRAM stays held, please"
        echo "  notify the cluster admin to drain the node."
    fi
fi

_section "GPU state AFTER cleanup"
_run "rocm-smi --showpids" rocm-smi --showpids

exit 0
