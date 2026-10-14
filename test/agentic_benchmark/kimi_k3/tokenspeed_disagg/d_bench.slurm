#!/usr/bin/bash
# Decode-node simulation as a Slurm sweep: dataset prep, the server and
# pd_client run as job steps (TP8 uses the first 2 nodes; DP16 uses 4; DP32 uses 8).
#
#   sbatch -N 8 --gres=gpu:4 --time=12:00:00 [-A <account> -p <partition>] \
#       --export=ALL,CONTAINER_IMAGE=<sqsh-or-ref> \
#       --wrap "bash test/agentic_benchmark/kimi_k3/tokenspeed_disagg/d_bench.slurm"
#
#   SLURM_JOB_ID=<jobid> CONTAINER_IMAGE=<sqsh-or-ref> bash d_bench.slurm

set -euo pipefail

CONFIGS=(
    attn_tp8_moe_tp8
    attn_dp16_moe_ep16
    attn_dp32_moe_ep32
)

# Passing this file to sbatch directly runs a spool copy and loses this
# directory; the usage above wraps it so the real path is used in both modes.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVER_PID=
SERVER_LOG=
STEP_NAME=disagg_server
# Prime the concatenated user-turn prompts at low concurrency.
PRIME_PARALLEL=16

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

# Prepare dataset
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
        attn_tp8_moe_tp8) node_count=2 ;;
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
        python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model nvidia/Kimi-K3-NVFP4 \
        --dataset agentic_dataset_x128.json "$@"
}

trap stop_server EXIT  # safety net for Ctrl-C / errors

# Preflight: bail out if something already serves on the port
wait_for_port_free

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${SCRIPT_DIR}/outputs/d_${SWEEP_TS}"
mkdir -p "$SWEEP_DIR"
echo "D-sim sweep outputs: ${SWEEP_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
    # Entries are parallel / number. TP8 collects extra samples; DP uses one wave.
    # Re-prime each rung because preceding measurements grow cached contexts.
    case "$CONFIG" in
        attn_tp8_moe_tp8) LADDER=("1 4" "2 8" "4 8" "8 16" "16 32" "32 64") ;;
        attn_dp16_moe_ep16) LADDER=("16 16" "32 32" "64 64" "128 128" "256 256" "384 384" "512 512") ;;
        attn_dp32_moe_ep32) LADDER=("32 32" "64 64" "128 128" "256 256" "384 384" "512 512" "768 768" "1024 1024") ;;
    esac

    echo "=== Running $CONFIG ==="
    launch_server "$CONFIG"

    if ! wait_for_ready; then
        stop_server
        exit 1
    fi

    for rung in "${LADDER[@]}"; do
        read -r P N <<< "$rung"
        echo "--- prime for parallel $P ($N conversations) ---"
        client --phase d-prime --parallel "$PRIME_PARALLEL" --number "$N" --offset 0 --max-tokens 1 \
            --name "${CONFIG}_d_prime" --outputs-dir "$SWEEP_DIR"
        echo "settle 10s..."
        sleep 10

        echo "--- measure parallel $P ---"
        client --phase d-measure --parallel "$P" --number "$N" --offset 0 --max-tokens 2000 \
            --name "${CONFIG}_d_measure" --outputs-dir "$SWEEP_DIR"
        if ! curl -sf -m 5 -o /dev/null "${BASE_URL}/readiness"; then
            echo "server stopped answering after rung ${P}; aborting the ladder" >&2
            break
        fi
    done

    stop_server
    wait_for_port_free
done

echo "D-sim sweep done: ${SWEEP_DIR}"
