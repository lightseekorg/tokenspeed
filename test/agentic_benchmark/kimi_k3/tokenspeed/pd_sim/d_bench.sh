#!/usr/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL=nvidia/Kimi-K3-NVFP4
URL=http://127.0.0.1:8000/v1/chat/completions
DATASET="${SCRIPT_DIR}/agentic_dataset.json"

# Same dataset acquisition + guard as the parent bench (one file, reused).
# pd_sim always fetches the FROZEN artifact into its own directory — it
# deliberately does not reuse a parent-directory file, which may be a local
# non-frozen build.
[ -s "$DATASET" ] || {
    curl -fsSL "https://huggingface.co/datasets/lightseekorg/agentic-dataset/resolve/main/agentic_dataset.json" \
        -o "$DATASET.tmp"
    mv "$DATASET.tmp" "$DATASET"
}
DATASET="$DATASET" python3 - <<'PYEOF'
import json
d = json.load(open(__import__("os").environ["DATASET"]))
n = len(d["conversations"])
m = d.get("metadata", {})
assert n >= 70, f"agentic_dataset.json has only {n} conversations; need >= 70"
recipe = (m.get("first_turn_length"), m.get("subsequent_turn_length"), m.get("min_turns"), m.get("max_turns"))
assert recipe == (50000, 800, 10, 15), f"unexpected dataset recipe {recipe}; delete agentic_dataset.json"
print(f"dataset ok: {n} conversations, recipe {recipe}")
PYEOF

CONFIGS=(
    attn_tp8_moe_ep8
    attn_tp8_moe_tp8
    attn_dp8_moe_ep8
)

SERVER_PID=
SERVER_LOG=

launch_server() {
    local config=$1
    SERVER_LOG=/tmp/tokenspeed_server_pdsim_${config}.log
    setsid ${PARENT_DIR}/configs/${config}.sh > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    local TIMEOUT=7200
    local START=$SECONDS
    until curl -sf -o /dev/null http://127.0.0.1:8000/readiness; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server died early. Last log lines:" >&2
            tail -100 "$SERVER_LOG" >&2
            return 1
        fi
        if grep -qE "CUDA out of memory|OutOfMemory|Traceback|Killed" "$SERVER_LOG"; then
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
        echo "Stopping ts serve (pgid $SERVER_PID)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -"$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

wait_for_port_free() {
    local port=${1:-8000}
    local timeout=${2:-90}
    local start=$SECONDS
    while ! python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            echo "Port ${port} still in use after ${timeout}s" >&2
            return 1
        fi
        sleep 1
    done
}

trap stop_server EXIT

wait_for_port_free 8000
wait_for_port_free 4000

# Ladder: parallel / number = 2x parallel. All rungs replay the primed
# conversations 0..31 (reuse is the point: the KV must already be resident).
LADDER=(
    "1 2"
    "2 4"
    "4 8"
    "8 16"
    "16 32"
)

gpu_mem_max() {
    # Max used memory across ALL GPUs (DP rank skew must be visible).
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | sort -n | tail -1 | grep -E '^[0-9]+$' || echo -1
}

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${SCRIPT_DIR}/outputs/d_${SWEEP_TS}"
echo "D-sim sweep outputs: ${SWEEP_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
    echo "=== Running $CONFIG ==="
    launch_server "$CONFIG"
    if ! wait_for_ready; then
        stop_server
        exit 1
    fi

    echo "--- warmup (spare conversations, excluded) ---"
    python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model "$MODEL" \
        --dataset "$DATASET" --phase d-prime --parallel 2 --number 2 \
        --offset 62 --max-tokens 16 \
        --name "${CONFIG}_warmup" --outputs-dir "$SWEEP_DIR"

    echo "--- D prime (low concurrency) ---"
    MEM_BEFORE_PRIME=$(gpu_mem_max)
    python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model "$MODEL" \
        --dataset "$DATASET" --phase d-prime --parallel 2 --number 32 \
        --offset 0 --max-tokens 1 \
        --name "${CONFIG}_d_prime" --outputs-dir "$SWEEP_DIR"
    MEM_AFTER_PRIME=$(gpu_mem_max)
    echo "settle 30s..."
    sleep 30

    echo "--- D measure ---"
    mkdir -p "$SWEEP_DIR"
    for rung in "${LADDER[@]}"; do
        read -r P N <<< "$rung"
        # Background sampler: peak memory DURING the rung (5s cadence).
        PEAK_FILE=$(mktemp)
        (
            peak=-1
            while true; do
                m=$(gpu_mem_max)
                [ "$m" -gt "$peak" ] 2>/dev/null && peak=$m
                echo "$peak" > "$PEAK_FILE"
                sleep 5
            done
        ) &
        SAMPLER_PID=$!
        python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model "$MODEL" \
            --dataset "$DATASET" --phase d-measure --parallel "$P" --number "$N" \
            --offset 0 --max-tokens 2000 \
            --name "${CONFIG}_d_measure" --outputs-dir "$SWEEP_DIR"
        kill "$SAMPLER_PID" 2>/dev/null || true
        MEM_PEAK=$(cat "$PEAK_FILE" 2>/dev/null || echo -1)
        rm -f "$PEAK_FILE"
        echo "{\"config\": \"$CONFIG\", \"parallel\": $P, \"mem_before_prime\": ${MEM_BEFORE_PRIME:--1}, \"mem_after_prime\": ${MEM_AFTER_PRIME:--1}, \"mem_peak_during_measure\": ${MEM_PEAK:--1}}" \
            >> "${SWEEP_DIR}/${CONFIG}_memory_ledger.jsonl"
    done

    stop_server
    wait_for_port_free 8000
done

echo "D-sim sweep done: ${SWEEP_DIR}"
