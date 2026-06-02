#!/bin/bash
# Best-effort cleanup of stale NVIDIA GPU processes before a CI task.
# This covers older runs that predate process-group tracking or exited before
# their pgid file could be cleaned up, leaving VRAM held by orphaned workers.
#
# Best-effort: never aborts; never propagates non-zero status.
set +e

WAIT_AFTER_TERM_SECS=${TOKENSPEED_NVIDIA_GPU_WAIT_AFTER_TERM:-3}
WAIT_AFTER_KILL_SECS=${TOKENSPEED_NVIDIA_GPU_WAIT_AFTER_KILL:-5}
STALE_CMD_RE='tokenspeed::|ts serve|python.*-m[[:space:]]+smg|smg::|smg_grpc_servicer'

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

self_pid=$$
ancestors=" $self_pid "
p=$self_pid
while :; do
    pp=$(awk '/^PPid:/ {print $2}' "/proc/${p}/status" 2>/dev/null)
    [ -z "$pp" ] || [ "$pp" = "0" ] && break
    ancestors="${ancestors}${pp} "
    p=$pp
done
echo "[cleanup_nvidia_gpu_state] self_pid=${self_pid} ancestors=${ancestors}"

_in_ancestors() {
    case "$ancestors" in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

_section "GPU state BEFORE cleanup"
_run "nvidia-smi" nvidia-smi

_section "Discovering GPU-holding PIDs"
gpu_pids=""

# Catch processes with NVIDIA device fds open.
for fd_dir in /proc/[0-9]*/fd; do
    pid="${fd_dir#/proc/}"
    pid="${pid%/fd}"
    if _in_ancestors "$pid"; then
        continue
    fi
    if ls -l "$fd_dir" 2>/dev/null | grep -qE "/dev/nvidia"; then
        gpu_pids="${gpu_pids} ${pid}"
    fi
done

# Catch processes known to the NVIDIA driver, including ones without visible fds.
if command -v nvidia-smi >/dev/null 2>&1; then
    extra=$(nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '/^[ \t]*[0-9]+[ \t]*$/ {print $1}' \
        | sort -u)
    for pid in $extra; do
        if _in_ancestors "$pid"; then
            continue
        fi
        gpu_pids="${gpu_pids} ${pid}"
    done
fi

gpu_pids=$(echo "$gpu_pids" | tr ' ' '\n' | awk 'NF' | sort -un | tr '\n' ' ')

if [ -z "${gpu_pids// /}" ]; then
    echo "No GPU-holding processes found (excluding ourselves)."
else
    echo "GPU-holding PIDs: ${gpu_pids}"
    echo ""
    echo "----- forensic ps -----"
    stale_gpu_pids=""
    for pid in $gpu_pids; do
        info=$(ps -o pid=,ppid=,user=,stat=,etime=,cmd= -p "$pid" 2>/dev/null)
        if [ -n "$info" ]; then
            echo "  ${info}"
        else
            echo "  pid=${pid} (already gone)"
            continue
        fi

        ppid=$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)
        cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)
        if [ -z "$cmdline" ]; then
            cmdline=$(ps -o cmd= -p "$pid" 2>/dev/null)
        fi

        if [ "$ppid" = "1" ] && echo "$cmdline" | grep -Eq "$STALE_CMD_RE"; then
            stale_gpu_pids="${stale_gpu_pids} ${pid}"
        else
            echo "  skip pid=${pid}: ppid=${ppid:-unknown}, cmdline does not look like orphaned TokenSpeed/SMG CI work"
        fi
    done
    echo ""

    gpu_pids=$(echo "$stale_gpu_pids" | tr ' ' '\n' | awk 'NF' | sort -un | tr '\n' ' ')
    if [ -z "${gpu_pids// /}" ]; then
        echo "No stale orphaned TokenSpeed/SMG GPU processes selected for cleanup."
        _section "GPU state AFTER cleanup"
        _run "nvidia-smi" nvidia-smi
        exit 0
    fi
    echo "Selected stale GPU-holding PIDs for cleanup: ${gpu_pids}"

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
        echo "  If VRAM stays held, the node likely needs admin cleanup."
    fi
fi

_section "GPU state AFTER cleanup"
_run "nvidia-smi" nvidia-smi

exit 0
