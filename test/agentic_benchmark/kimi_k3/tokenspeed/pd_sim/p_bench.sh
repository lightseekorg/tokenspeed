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
# pd_client dedups by first-turn content (the frozen artifact has duplicate
# pairs); the ladder consumes 62 unique conversations + 2 for warmup.
uniq = len({json.dumps(c[0]["messages"], sort_keys=True) for c in d["conversations"]})
assert uniq >= 64, f"only {uniq} unique first turns; ladder + warmup need 64"
print(f"dataset ok: {n} conversations ({uniq} unique first turns), recipe {recipe}")
PYEOF

CONFIGS=(
    attn_tp8_moe_ep8
    attn_tp8_moe_tp8
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

# Ladder: parallel / number = 2x parallel / conversation offset.
# Offsets advance so neither P phase reuses a conversation (62 of 71).
LADDER=(
    "1 2 0"
    "2 4 2"
    "4 8 6"
    "8 16 14"
    "16 32 30"
)

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
    python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model "$MODEL" \
        --dataset "$DATASET" --phase d-prime --parallel 2 --number 2 \
        --offset 62 --max-tokens 16 \
        --name "${CONFIG}_warmup" --outputs-dir "$SWEEP_DIR"

    echo "--- P-fresh ---"
    for rung in "${LADDER[@]}"; do
        read -r P N OFF <<< "$rung"
        python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model "$MODEL" \
            --dataset "$DATASET" --phase p-fresh --parallel "$P" --number "$N" \
            --offset "$OFF" --max-tokens 1 \
            --name "${CONFIG}_p_fresh" --outputs-dir "$SWEEP_DIR"
    done

    echo "--- P-cached ---"
    for rung in "${LADDER[@]}"; do
        read -r P N OFF <<< "$rung"
        python3 "${SCRIPT_DIR}/pd_client.py" --url "$URL" --model "$MODEL" \
            --dataset "$DATASET" --phase p-cached --parallel "$P" --number "$N" \
            --offset "$OFF" --max-tokens 1 \
            --name "${CONFIG}_p_cached" --outputs-dir "$SWEEP_DIR"
    done

    stop_server
    wait_for_port_free 8000
done

echo "P-sim sweep done: ${SWEEP_DIR}"
