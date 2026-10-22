#!/usr/bin/bash
# Prefill-node simulation as a Slurm sweep: dataset prep, the server and
# pd_client run as job steps (DP16 uses the first 4 nodes, 4 GPUs each).
#
#   sbatch -N 4 --gres=gpu:4 --time=12:00:00 [-A <account> -p <partition>] \
#       --export=ALL,CONTAINER_IMAGE=<sqsh-or-ref> \
#       --wrap "bash test/agentic_benchmark/kimi_k3/tokenspeed_disagg/p_bench.slurm"
#
#   SLURM_JOB_ID=<jobid> CONTAINER_IMAGE=<sqsh-or-ref> bash p_bench.slurm

set -euo pipefail

CONFIGS=(
    attn_dp16_moe_ep16
)

# Passing this file to sbatch directly runs a spool copy and loses this
# directory; the usage above wraps it so the real path is used in both modes.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=nvidia/Kimi-K3-NVFP4
DATASET="${SCRIPT_DIR}/agentic_dataset_x128.json"
SERVER_PID=
SERVER_LOG=
STEP_NAME=disagg_server

JOBID="${SLURM_JOB_ID:?run via sbatch, or export SLURM_JOB_ID=<id> of a held allocation}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?set CONTAINER_IMAGE to an enroot image with tokenspeed + evalscope}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/data:/data}"
CONTAINER_NAME="${CONTAINER_NAME:-ts}"

# Without these the enroot nvidia hook mounts no driver into the container.
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

mapfile -t NODES < <(scontrol show hostnames "$(squeue -j "$JOBID" -h -o '%N')")
NNODES="${#NODES[@]}"
# srun places task 0 -- the gateway rank -- on the first node of the list.
HEAD="${NODES[0]}"
BASE_URL="http://${HEAD}:8000"
URL="${BASE_URL}/v1/chat/completions"

SRUN=(srun --jobid="$JOBID" --overlap)
SRUN_ARGS=(
    --container-image="$CONTAINER_IMAGE"
    --container-name="$CONTAINER_NAME"
    --container-mounts="$CONTAINER_MOUNTS"
    --container-workdir="$SCRIPT_DIR"
    --container-remap-root
    --no-container-mount-home
    --container-writable
)

# Measure conversations 0..1023; warmup uses the disjoint range 1024..1039.
"${SRUN[@]}" --nodes=1 --ntasks=1 --nodelist="$HEAD" "${SRUN_ARGS[@]}" bash -c '
set -euo pipefail

EVALSCOPE_COMMIT=40395b16ba51731bfda6f099c1ded902303b56c0
pip install "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

[ -f build_swe_smith_dataset.py ] || wget https://raw.githubusercontent.com/modelscope/evalscope/${EVALSCOPE_COMMIT}/examples/perf/build_swe_smith_dataset.py \
    -O build_swe_smith_dataset.py

# Replicated 69 conversations x 128 -> 8832
[ -f agentic_dataset_x128.json ] || python3 build_swe_smith_dataset.py \
    --model-path moonshotai/Kimi-K3 \
    --first-turn-length 4000 \
    --subsequent-turn-length 800 \
    --min-turns 30 \
    --max-turns 50 \
    --number 128 \
    --replicas 128 \
    --output-path agentic_dataset_x128.json \
    --num-workers 32
'

server_steps() {
    squeue -h -s -j "$JOBID" -o '%i %j' 2>/dev/null \
        | awk -v n="$STEP_NAME" '$2 == n {print $1}'
}

launch_server() {
    local config=$1
    local node_count
    case "$config" in
        attn_dp16_moe_ep16) node_count=4 ;;
        attn_dp32_moe_ep32) node_count=8 ;;
        *) echo "Unsupported config: $config" >&2; return 1 ;;
    esac

    if (( NNODES < node_count )); then
        echo "$config requires $node_count nodes; allocation has $NNODES" >&2
        return 1
    fi

    local node_list
    printf -v node_list '%s,' "${NODES[@]:0:node_count}"
    node_list="${node_list%,}"

    SERVER_LOG=/tmp/tokenspeed_server_disagg_${config}_${SWEEP_TS}.log
    "${SRUN[@]}" --job-name="$STEP_NAME" \
        --nodes="$node_count" --ntasks="$node_count" \
        --ntasks-per-node=1 --nodelist="$node_list" \
        --mpi=pmix "${SRUN_ARGS[@]}" \
        "${SCRIPT_DIR}/configs/${config}.sh" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    local TIMEOUT=7200  # >= the configs' --engine-startup-timeout
    local START=$SECONDS
    until curl -sf -o /dev/null "${BASE_URL}/readiness"; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server died early. Last log lines:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if grep -qE "CUDA out of memory|OutOfMemory|RuntimeError|Killed" "$SERVER_LOG"; then
            echo "Server hit a fatal error:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if (( SECONDS - START > TIMEOUT )); then
            echo "Timeout after ${TIMEOUT}s waiting for server" >&2
            return 1
        fi
        sleep 5
    done
    echo "Server ready after $((SECONDS - START))s"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping ts serve (srun pid $SERVER_PID)..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        # KILL on the srun client would orphan the remote step; cancel the
        # step itself instead.
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            scancel $(server_steps) 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    SERVER_PID=
}

wait_for_port_free() {
    local timeout=${1:-90}
    local start=$SECONDS
    while curl -sf -m 3 -o /dev/null "${BASE_URL}/readiness" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            echo "${BASE_URL} still answering after ${timeout}s" >&2
            return 1
        fi
        sleep 1
    done
}

# One pd_client phase as a single-task step; phase flags are passed through.
client() {
    "${SRUN[@]}" --nodes=1 --ntasks=1 --nodelist="$HEAD" "${SRUN_ARGS[@]}" \
        python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model "$MODEL" \
        --dataset "$DATASET" "$@"
}

trap stop_server EXIT  # safety net for Ctrl-C / errors

# Preflight: bail out if something already serves on the port
wait_for_port_free

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${SCRIPT_DIR}/outputs/p_${SWEEP_TS}"
echo "P-sim sweep outputs: ${SWEEP_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
    echo "=== Running $CONFIG ==="
    launch_server "$CONFIG"

    if ! wait_for_ready; then
        stop_server
        exit 1
    fi

    echo "--- warmup (spare conversations, excluded) ---"
    client --phase p-measure --parallel 16 --number 16 --offset 1024 --max-tokens 1 \
        --name "${CONFIG}_warmup" --outputs-dir "$SWEEP_DIR"

    echo "--- P conversation replay ---"
    client --phase p-measure --parallel 512 --number 1024 --offset 0 --max-tokens 1 \
        --name "${CONFIG}_p_measure" --outputs-dir "$SWEEP_DIR"

    stop_server
    wait_for_port_free
done

echo "P-sim sweep done: ${SWEEP_DIR}"
