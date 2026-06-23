#!/bin/bash
# Best-effort cleanup of stale NVIDIA GPU processes before a CI task.
# This covers older runs that predate process-group tracking or exited before
# their pgid file could be cleaned up, leaving VRAM held by orphaned workers.
#
# Best-effort for individual cleanup commands; fail only if dirty VRAM remains.
set +e

WAIT_AFTER_TERM_SECS=${TOKENSPEED_NVIDIA_GPU_WAIT_AFTER_TERM:-3}
WAIT_AFTER_KILL_SECS=${TOKENSPEED_NVIDIA_GPU_WAIT_AFTER_KILL:-5}
DIRTY_MEMORY_THRESHOLD_PERCENT=${TOKENSPEED_NVIDIA_GPU_DIRTY_MEMORY_THRESHOLD_PERCENT:-5}
RESET_DIRTY_MEMORY=${TOKENSPEED_NVIDIA_GPU_RESET_DIRTY_MEMORY:-1}
FAIL_ON_DIRTY_MEMORY=${TOKENSPEED_NVIDIA_GPU_FAIL_ON_DIRTY_MEMORY:-1}
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
dirty_gpu_memory_remains=0

_in_ancestors() {
    case "$ancestors" in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

_normalize_pids() {
    echo "$*" \
        | tr '[:space:]' '\n' \
        | awk '/^[0-9]+$/ {print $1}' \
        | sort -un \
        | while read -r pid; do
            if ! _in_ancestors "$pid"; then
                echo "$pid"
            fi
        done \
        | tr '\n' ' '
}

_discover_proc_gpu_pids() {
    for fd_dir in /proc/[0-9]*/fd; do
        pid="${fd_dir#/proc/}"
        pid="${pid%/fd}"
        if _in_ancestors "$pid"; then
            continue
        fi
        if ls -l "$fd_dir" 2>/dev/null | grep -qE "/dev/nvidia"; then
            echo "$pid"
        fi
    done
}

_discover_nvidia_smi_gpu_pids() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi
    nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '/^[ \t]*[0-9]+[ \t]*$/ {print $1}'
}

_discover_fuser_gpu_pids() {
    if ! command -v fuser >/dev/null 2>&1; then
        echo "fuser is not available; skipping /dev/nvidia holder discovery" >&2
        return
    fi
    devices=$(compgen -G "/dev/nvidia*")
    if [ -z "$devices" ]; then
        return
    fi
    fuser $devices 2>/dev/null || true
    if command -v sudo >/dev/null 2>&1; then
        sudo -n fuser $devices 2>/dev/null || true
    fi
}

_discover_gpu_pids() {
    _normalize_pids "$(_discover_proc_gpu_pids) $(_discover_nvidia_smi_gpu_pids)"
}

_discover_aggressive_gpu_pids() {
    _normalize_pids "$(_discover_proc_gpu_pids) $(_discover_nvidia_smi_gpu_pids) $(_discover_fuser_gpu_pids)"
}

_select_stale_gpu_pids() {
    local gpu_pids="$1"
    local label="${2:-GPU-holding PIDs}"
    stale_gpu_pids=""

    if [ -z "${gpu_pids// /}" ]; then
        echo "No ${label} found (excluding ourselves)."
        return
    fi

    echo "${label}: ${gpu_pids}"
    echo ""
    echo "----- forensic ps -----"
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

    stale_gpu_pids=$(_normalize_pids "$stale_gpu_pids")
}

_kill_gpu_pids() {
    local gpu_pids="$1"
    if [ -z "${gpu_pids// /}" ]; then
        return
    fi

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
}

_check_dirty_gpu_memory() {
    dirty_gpu_memory=""
    dirty_gpu_indices=""
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is not available; skipping dirty GPU memory check"
        return 1
    fi

    while IFS=, read -r index used_mib total_mib; do
        index="${index//[[:space:]]/}"
        used_mib="${used_mib//[[:space:]]/}"
        total_mib="${total_mib//[[:space:]]/}"
        if ! [[ "$used_mib" =~ ^[0-9]+$ && "$total_mib" =~ ^[0-9]+$ ]]; then
            echo "skip unexpected nvidia-smi memory row: ${index}, ${used_mib}, ${total_mib}"
            continue
        fi
        if [ "$total_mib" -le 0 ]; then
            echo "skip unexpected NVIDIA GPU total memory for gpu=${index}: ${total_mib}"
            continue
        fi
        used_percent=$((used_mib * 100 / total_mib))
        echo "[cleanup_nvidia_gpu_state] gpu=${index} memory.used=${used_mib}MiB memory.total=${total_mib}MiB used_percent=${used_percent}% threshold=${DIRTY_MEMORY_THRESHOLD_PERCENT}%"
        if [ $((used_mib * 100)) -gt $((total_mib * DIRTY_MEMORY_THRESHOLD_PERCENT)) ]; then
            dirty_gpu_memory="${dirty_gpu_memory} gpu=${index} used=${used_mib}MiB total=${total_mib}MiB used_percent=${used_percent}%;"
            dirty_gpu_indices="${dirty_gpu_indices} ${index}"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    [ -n "${dirty_gpu_memory// /}" ]
}

_reset_dirty_gpus() {
    if [ "${RESET_DIRTY_MEMORY}" != "1" ]; then
        echo "Dirty GPU memory reset is disabled by TOKENSPEED_NVIDIA_GPU_RESET_DIRTY_MEMORY=${RESET_DIRTY_MEMORY}."
        return
    fi
    if [ -z "${dirty_gpu_indices// /}" ]; then
        echo "No dirty GPU indices available for reset."
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is not available; skipping dirty GPU reset"
        return
    fi

    reset_indices=$(echo "${dirty_gpu_indices}" | tr ' ' '\n' | awk '/^[0-9]+$/ {print $1}' | sort -un)
    reset_indices=$(echo "${reset_indices}" | tr '\n' ',' | sed 's/,$//')
    if [ -z "${reset_indices}" ]; then
        echo "No valid dirty GPU indices available for reset."
        return
    fi

    echo "Attempting GPU reset for dirty GPU indices: ${reset_indices}"
    nvidia-smi --gpu-reset -i "${reset_indices}" 2>&1 || true
    if command -v sudo >/dev/null 2>&1; then
        sudo -n nvidia-smi --gpu-reset -i "${reset_indices}" 2>&1 || true
    fi
    sleep "${WAIT_AFTER_KILL_SECS}"
}

_section "GPU state BEFORE cleanup"
_run "nvidia-smi" nvidia-smi

_section "Discovering GPU-holding PIDs"
gpu_pids=$(_discover_gpu_pids)

_select_stale_gpu_pids "$gpu_pids" "GPU-holding PIDs"
gpu_pids="$stale_gpu_pids"
if [ -z "${gpu_pids// /}" ]; then
    echo "No stale orphaned TokenSpeed/SMG GPU processes selected for cleanup."
else
    echo "Selected stale GPU-holding PIDs for cleanup: ${gpu_pids}"
    _section "Killing GPU-holding PIDs"
    _kill_gpu_pids "$gpu_pids"
fi

_section "GPU state AFTER cleanup"
_run "nvidia-smi" nvidia-smi

if _check_dirty_gpu_memory; then
    _section "Dirty GPU memory detected; discovering device holders"
    echo "Dirty GPU memory above ${DIRTY_MEMORY_THRESHOLD_PERCENT}% after cleanup:${dirty_gpu_memory}"
    aggressive_gpu_pids=$(_discover_aggressive_gpu_pids)
    if [ -z "${aggressive_gpu_pids// /}" ]; then
        echo "No /dev/nvidia holder PIDs found for dirty GPU memory."
        echo "The memory may be held by driver state or by host processes hidden from this namespace."
        _reset_dirty_gpus
        _section "GPU state AFTER dirty memory reset"
        _run "nvidia-smi" nvidia-smi
        if _check_dirty_gpu_memory; then
            dirty_gpu_memory_remains=1
            echo "WARNING: dirty GPU memory remains after reset:${dirty_gpu_memory}"
        fi
    else
        _select_stale_gpu_pids "$aggressive_gpu_pids" "Aggressive GPU-holding PIDs"
        aggressive_gpu_pids="$stale_gpu_pids"
        if [ -z "${aggressive_gpu_pids// /}" ]; then
            echo "No stale orphaned TokenSpeed/SMG device-holder PIDs selected for aggressive cleanup."
            _reset_dirty_gpus
            _section "GPU state AFTER dirty memory reset"
            _run "nvidia-smi" nvidia-smi
            if _check_dirty_gpu_memory; then
                echo "WARNING: dirty GPU memory remains after reset:${dirty_gpu_memory}"
            fi
        else
            echo "Selected stale GPU-holding PIDs for aggressive cleanup: ${aggressive_gpu_pids}"
            _section "Killing aggressive GPU-holding PIDs"
            _kill_gpu_pids "$aggressive_gpu_pids"
            _section "GPU state AFTER aggressive cleanup"
            _run "nvidia-smi" nvidia-smi
            if _check_dirty_gpu_memory; then
                dirty_gpu_memory_remains=1
                echo "WARNING: dirty GPU memory remains after aggressive cleanup:${dirty_gpu_memory}"
                echo "  If no PIDs are visible, the node likely needs admin cleanup or GPU reset."
            fi
        fi
    fi
fi

if [ "${dirty_gpu_memory_remains}" = "1" ] && [ "${FAIL_ON_DIRTY_MEMORY}" = "1" ]; then
    echo "ERROR: dirty GPU memory remains after cleanup attempts; refusing to start CI task."
    echo "  Set TOKENSPEED_NVIDIA_GPU_FAIL_ON_DIRTY_MEMORY=0 to keep best-effort-only behavior."
    exit 1
fi

exit 0
