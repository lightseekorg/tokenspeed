#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Run via sbatch with a native client, or from the login host in a held job.
# All paths must be absolute and visible where they are used. See agentx.md.
# Caller arguments are preserved; API_PORT supplies the final server port override.
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch or an existing allocation}"

validate_config() {
    : "${SOURCE_ROOT:?absolute checkout path; independent of sbatch submit directory}"
    : "${CONFIG_NAMES:?space-separated server configuration names}"
    read -r -a CONFIGS <<< "$CONFIG_NAMES"
    (( ${#CONFIGS[@]} > 0 )) || { echo "CONFIG_NAMES must select at least one config" >&2; exit 2; }
    for config in "${CONFIGS[@]}"; do
        case "$config" in attn_tp8_moe_tp8|attn_tp8_moe_ep8) ;; *) echo "Unknown config: $config" >&2; exit 2 ;; esac
    done
    declare -A seen=()
    for config in "${CONFIGS[@]}"; do
        [[ ! -v "seen[$config]" ]] || { echo "Duplicate config: $config" >&2; exit 2; }
        seen[$config]=1
    done
    : "${SERVER_VENV?server venv path; set empty to use the image environment}"
    export SERVER_VENV
    : "${CONTAINER_IMAGE:?pinned server image}"
    : "${CONTAINER_MOUNTS:?explicit shared mounts}"
    : "${CLIENT_PYTHON:?absolute Python from the prepared InferenceX AgentX venv}"
    : "${MODEL_NAME:?served model name}"
    : "${TOKENIZER_PATH:?pinned tokenizer path or ID}"
    : "${CLIENT_ROOT:?prepared InferenceX checkout and venv directory}"
    : "${CPUS_PER_TASK:?CPU allocation per node}"
    : "${CONCURRENCY:?root session concurrency}"
    : "${API_PORT:?server API port}"
    : "${READINESS_PATH:?readiness endpoint, including leading slash}"
    : "${READINESS_TIMEOUT:?maximum server startup seconds}"
    : "${CLIENT_TIMEOUT:?maximum total client wall time seconds}"
    : "${RUN_ROOT:?new absolute output directory}"
    for variable in SOURCE_ROOT CLIENT_PYTHON CLIENT_ROOT RUN_ROOT; do
        [[ "${!variable}" = /* ]] || { echo "$variable must be absolute" >&2; exit 2; }
    done
    for variable in CONCURRENCY API_PORT READINESS_TIMEOUT CLIENT_TIMEOUT CPUS_PER_TASK; do
        [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "$variable must be a positive integer" >&2; exit 2; }
    done
    SCRIPT_DIR="$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed"
    [[ -x "$CLIENT_PYTHON" ]] || { echo "Client Python is not executable: $CLIENT_PYTHON" >&2; exit 2; }
    [[ -f "$CLIENT_ROOT/inferencex/inferencex-e2e/benchmarks/benchmark_lib.sh" ]] || { echo "Missing InferenceX benchmark_lib.sh under $CLIENT_ROOT" >&2; exit 2; }
    [[ -f "$SCRIPT_DIR/agentx.sh" ]] || { echo "Missing client script: $SCRIPT_DIR/agentx.sh" >&2; exit 2; }
    source "$SCRIPT_DIR/agentx.sh"
    validate_agentx_config
    validate_agentx_values RUN_ROOT
    [[ "$CLIENT_PYTHON" = "$CLIENT_ROOT/venv/bin/python" ]] || { echo "CLIENT_PYTHON must use CLIENT_ROOT/venv" >&2; exit 2; }
    for config in "${CONFIGS[@]}"; do
        [[ -f "$SCRIPT_DIR/configs/$config.sh" ]] || { echo "Missing server config: $SCRIPT_DIR/configs/$config.sh" >&2; exit 2; }
    done
    [[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%u')" = "$(id -un)" ]] || { echo 'Allocation not owned by current user' >&2; exit 1; }
    [[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%T')" = RUNNING ]] || { echo 'Allocation must be RUNNING' >&2; exit 1; }
}

STEP_NAME="agentx_server_${SLURM_JOB_ID}_$$"
SERVER_PID=
PREFLIGHT_PID=
CLIENT_PID=
CLIENT_SIGNAL=TERM

finish() {
    local result=$?
    # Stop and reap the client before releasing its server. GNU timeout forwards
    # the signal to the client's process group and retains its kill-after bound.
    trap '' INT TERM
    trap - EXIT
    if [[ -n "$CLIENT_PID" ]]; then
        kill -s "$CLIENT_SIGNAL" "$CLIENT_PID" 2>/dev/null || true
        wait "$CLIENT_PID" 2>/dev/null || true
        CLIENT_PID=
    fi
    if [[ -n "$PREFLIGHT_PID" ]]; then
        kill -TERM "$PREFLIGHT_PID" 2>/dev/null || true
        wait "$PREFLIGHT_PID" 2>/dev/null || true
        PREFLIGHT_PID=
    fi
    stop_server
    if [[ -d "$RUN_ROOT" ]]; then printf '%s\n' "$result" > "$RUN_ROOT/client-exit-code.txt"; fi
    exit "$result"
}

cancel_steps() {
    # Never cancel the held allocation or another invocation's steps.
    while read -r step name; do
        if [[ "$name" = "$STEP_NAME" || "$name" = "${STEP_NAME}_preflight" ]]; then
            scancel "$step" || true
        fi
    done < <(squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j')
}

stop_server() {
    cancel_steps
    if [[ -n "$SERVER_PID" ]]; then
        # Signal the tracked launch even before its Slurm step is registered.
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
    cancel_steps
}

prepare_run() {
    # A new directory preserves logs from earlier attempts.
    mkdir "$RUN_ROOT"
    export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
    printf '%q ' "$SERVER_SCRIPT" "$@" > "$RUN_ROOT/server-command.txt"
    printf '\n' >> "$RUN_ROOT/server-command.txt"
}

setup_slurm() {
    scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation.txt"
    mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o '%N')")
    [[ "${#NODES[@]}" = 2 ]] || { echo "K3 TP8 requires two four-GPU nodes" >&2; exit 2; }
    export HEAD="${NODES[0]}"
    BASE_URL="http://${HEAD}:${API_PORT}"
    SRUN=(srun --jobid="$SLURM_JOB_ID" --overlap --nodes="${#NODES[@]}" --ntasks-per-node=1
        --cpus-per-task="$CPUS_PER_TASK" --gpus-per-node=4
        --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES
        --container-image="$CONTAINER_IMAGE" --container-name="agentx_${SLURM_JOB_ID}_$$"
        --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$SOURCE_ROOT"
        --container-remap-root --no-container-mount-home --container-writable)
}

check_resources() {
    # Any HTTP response on this port means another service owns it.
    if curl --silent --max-time 3 --output /dev/null "$BASE_URL/v1/models"; then
        echo 'API port already occupied; refusing to use or stop that service' >&2
        exit 1
    fi
    timeout --signal=TERM --kill-after=45 "$READINESS_TIMEOUT" \
        "${SRUN[@]}" --job-name="${STEP_NAME}_preflight" --kill-on-bad-exit=1 bash -c '
        set -euo pipefail
        occupied=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        if [[ -n "$occupied" ]]; then echo "GPUs already occupied: $occupied" >&2; exit 1; fi
        [[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" = 4 ]]
        gpus=$(nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader)
        printf '%s\n' "$gpus"
        while IFS= read -r gpu; do
            [[ "$gpu" = *GB300* ]] || { echo "This recipe reports GB300 results" >&2; exit 1; }
        done <<< "$gpus"
    ' > "$RUN_ROOT/gpu-preflight.log" 2>&1 &
    PREFLIGHT_PID=$!
    PREFLIGHT_RESULT=0
    wait "$PREFLIGHT_PID" || PREFLIGHT_RESULT=$?
    PREFLIGHT_PID=
    if (( PREFLIGHT_RESULT != 0 )); then exit "$PREFLIGHT_RESULT"; fi
}

launch_server() {
    timeout --signal=TERM --kill-after=45 "$((READINESS_TIMEOUT + CLIENT_TIMEOUT + 120))" \
        "${SRUN[@]}" --job-name="$STEP_NAME" --mpi=pmix --kill-on-bad-exit=1 \
        bash -c '
            set -euo pipefail
            if [[ -n "$SERVER_VENV" ]]; then source "$SERVER_VENV/bin/activate"; fi
            exec bash "$SERVER_SCRIPT" "$@"
        ' agentx-server "$@" > "$RUN_ROOT/server.log" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    START=$SECONDS
    until curl --fail --silent --max-time 5 --output /dev/null "$BASE_URL$READINESS_PATH"; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Server step exited before readiness' >&2; exit 1; }
        (( SECONDS - START < READINESS_TIMEOUT )) || { echo 'Server readiness timeout' >&2; exit 1; }
        sleep 5
    done
    printf '%s\n' "$BASE_URL" > "$RUN_ROOT/endpoint.txt"
    curl --fail --silent --max-time 10 "$BASE_URL/v1/models" > "$RUN_ROOT/models.json"
}

run_benchmark() {
    # Run the prepared native client on this host, outside the server container.
    export AIPERF_SERVER_URL="$BASE_URL" RESULT_DIR="$RUN_ROOT/client"
    case "$CONFIG" in
        attn_tp8_moe_tp8) export EP_SIZE=1 ;;
        attn_tp8_moe_ep8) export EP_SIZE=8 ;;
    esac
    timeout --signal=INT --kill-after=45 "$CLIENT_TIMEOUT" \
        bash "$SCRIPT_DIR/agentx.sh" > "$RUN_ROOT/client.log" 2>&1 &
    CLIENT_PID=$!
    local result=0
    wait "$CLIENT_PID" || result=$?
    CLIENT_PID=
    printf '%s\n' "$result" > "$RUN_ROOT/client-exit-code.txt"
    return "$result"
}

# Keep the configuration sweep beside the launch lifecycle, like agentic_bench.slurm.
validate_config
SWEEP_ROOT=$RUN_ROOT
mkdir "$SWEEP_ROOT"
for CONFIG in "${CONFIGS[@]}"; do
    export SERVER_SCRIPT="$SCRIPT_DIR/configs/$CONFIG.sh"
    export RUN_ROOT="$SWEEP_ROOT/$CONFIG"
    STEP_NAME="agentx_server_${SLURM_JOB_ID}_$$_${CONFIG}"
    trap finish EXIT
    trap 'CLIENT_SIGNAL=INT; exit 130' INT
    trap 'CLIENT_SIGNAL=TERM; exit 143' TERM
    prepare_run "$@" --port "$API_PORT"
    setup_slurm
    check_resources
    launch_server "$@" --port "$API_PORT"
    wait_for_ready
    run_benchmark
    stop_server
    printf '0\n' > "$RUN_ROOT/client-exit-code.txt"
    trap - EXIT INT TERM
done
