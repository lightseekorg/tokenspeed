#!/usr/bin/bash

set -euo pipefail

# Prepare dataset
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
# Skip the install when evalscope is already importable so offline reruns
# survive; delete the venv to force a re-pin.
python3 -c "import evalscope" 2>/dev/null || \
    pip install "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

# curl (already a hard dependency of the readiness probe) + tmp/mv so a
# failed download cannot leave a truncated file behind the -s guard.
[ -s build_swe_smith_dataset.py ] || {
    curl -fsSL "https://raw.githubusercontent.com/modelscope/evalscope/${EVALSCOPE_COMMIT}/examples/perf/build_swe_smith_dataset.py" \
        -o build_swe_smith_dataset.py.tmp
    mv build_swe_smith_dataset.py.tmp build_swe_smith_dataset.py
}

# Note: Only 71 conversations can be built (measured with the Kimi-K3 tokenizer)
[ -f agentic_dataset.json ] || python3 build_swe_smith_dataset.py \
    --model-path moonshotai/Kimi-K3 \
    --first-turn-length 50000 \
    --subsequent-turn-length 800 \
    --min-turns 10 \
    --max-turns 15 \
    --number 128 \
    --output-path agentic_dataset.json \
    --num-workers 32

# The sweep + warmup consume conversations 0..69; fail fast if the dataset
# ever builds fewer (tokenizer or upstream-dataset drift would otherwise
# silently rotate and replay hot conversations).
python3 - <<'PYEOF'
import json
d = json.load(open("agentic_dataset.json"))
n = len(d["conversations"])
assert n >= 70, f"agentic_dataset.json has only {n} conversations; need >= 70"
mp = d.get("metadata", {}).get("model_path", "")
assert "Kimi-K3" in mp, f"dataset built with wrong tokenizer: {mp!r}; delete agentic_dataset.json"
print(f"dataset ok: {n} conversations (tokenizer: {mp})")
PYEOF

# Sweep configs
CONFIGS=(
    attn_tp8_moe_ep8
    attn_tp8_moe_tp8
    attn_dp8_moe_ep8
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PID=
SERVER_LOG=

launch_server() {
    local config=$1
    SERVER_LOG=/tmp/tokenspeed_server_${config}.log
    setsid ${SCRIPT_DIR}/configs/${config}.sh > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    # CI gives this checkpoint a 7200s readiness window (first boot autotunes
    # trtllm-gen NVFP4 cubins for ~15 min; the dspark row also pulls 7.1GB of
    # draft weights). Must stay >= the configs' --engine-startup-timeout.
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

trap stop_server EXIT  # safety net for Ctrl-C / errors

# Preflight: bail out if the serve port or the DP rendezvous port is in use
wait_for_port_free 8000
wait_for_port_free 4000

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${SCRIPT_DIR}/outputs/${SWEEP_TS}"
echo "Sweep outputs: ${SWEEP_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
    echo "=== Running $CONFIG ==="
    launch_server "$CONFIG"

    if ! wait_for_ready; then
        stop_server
        exit 1
    fi

    echo "Warmup..."
    evalscope perf \
        --model nvidia/Kimi-K3-NVFP4 \
        --url http://127.0.0.1:8000/v1/chat/completions \
        --api openai \
        --dataset swe_smith \
        --dataset-path agentic_dataset.json \
        --max-tokens 500 \
        --multi-turn \
        --number 2 \
        --parallel 2 \
        --extra-args '{"ignore_eos": true}' \
        --dataset-offset 68 \
        --outputs-dir /tmp/outputs

    echo "Benchmark..."
    evalscope perf \
        --model nvidia/Kimi-K3-NVFP4 \
        --url http://127.0.0.1:8000/v1/chat/completions \
        --api openai \
        --dataset swe_smith \
        --dataset-path agentic_dataset.json \
        --max-tokens 500 \
        --multi-turn \
        --number 4 8 8 16 32 \
        --parallel 1 2 4 8 16 \
        --extra-args '{"ignore_eos": true}' \
        --name $CONFIG \
        --outputs-dir $SWEEP_DIR \
        --no-timestamp

    stop_server
    wait_for_port_free 8000
done
