#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: bash $0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prepare dataset
EVALSCOPE_COMMIT=acd09b44384d53174768bb1063f675420f76fae9
pip install "evalscope[perf] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

if [ ! -f agentic_dataset.json ]; then
    (
        BUILD_SCRIPT=$(mktemp)
        trap 'rm -f "$BUILD_SCRIPT"' EXIT
        wget \
            https://raw.githubusercontent.com/modelscope/evalscope/${EVALSCOPE_COMMIT}/examples/perf/build_swe_smith_dataset.py \
            -O "$BUILD_SCRIPT"
        patch -s "$BUILD_SCRIPT" build_swe_smith_dataset.patch
        PYTHONPATH=. python3 "$BUILD_SCRIPT" \
            --model-path deepseek-ai/DeepSeek-V4.1-Flash-1017 \
            --first-turn-length 50000 \
            --subsequent-turn-length 800 \
            --min-turns 10 \
            --max-turns 15 \
            --number 128 \
            --output-path agentic_dataset.json \
            --num-workers 32
    )
fi

# Sweep configs with DSpark, GPU prefix caching and the default Host L2 KVStore.
CONFIGS=(
    attn_tp4_moe_tp4
    attn_tp4_moe_ep4
    attn_dp4_moe_ep4
)

SERVER_PID=
SERVER_LOG=

launch_server() {
    local config=$1
    SERVER_LOG=/tmp/tokenspeed_server_${config}.log
    setsid "${SCRIPT_DIR}/configs/${config}.sh" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    local TIMEOUT=3600
    local START=$SECONDS
    until curl -sf --max-time 5 -o /dev/null http://127.0.0.1:8000/readiness; do
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
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping ts serve (pgid $SERVER_PID)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null || true
        for _ in {1..20}; do
            kill -0 -- "-$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -"$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

wait_for_port_free() {
    local port=$1
    local timeout=$2
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
trap 'exit 130' INT
trap 'exit 143' TERM

# Preflight: HTTP and attention-DP rendezvous ports must be free.
wait_for_port_free 8000 90
wait_for_port_free 8400 90

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
        --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
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
        --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
        --url http://127.0.0.1:8000/v1/chat/completions \
        --api openai \
        --dataset swe_smith \
        --dataset-path agentic_dataset.json \
        --max-tokens 500 \
        --multi-turn \
        --number 4 8 8 16 32 \
        --parallel 1 2 4 8 16 \
        --extra-args '{"ignore_eos": true}' \
        --name "$CONFIG" \
        --outputs-dir "$SWEEP_DIR" \
        --no-timestamp

    stop_server
    wait_for_port_free 8000 90
    wait_for_port_free 8400 90
done
