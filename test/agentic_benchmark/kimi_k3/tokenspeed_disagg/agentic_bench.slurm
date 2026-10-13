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
#
# 1P1D: ADP16/EP16 prefill on the first four nodes, decode on the next four.
# Run once in an existing eight-node allocation with four GPUs per node:
#   SLURM_JOB_ID=<jobid> CONTAINER_IMAGE=<sqsh-or-ref> bash test/agentic_benchmark/kimi_k3/tokenspeed_disagg/agentic_bench.slurm
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVER_PID=
SERVER_LOG=
PREFILL_PID=
DECODE_PID=
GATEWAY_PID=

JOBID="${SLURM_JOB_ID:?set SLURM_JOB_ID to an existing eight-node allocation}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?set CONTAINER_IMAGE to an enroot image with tokenspeed + evalscope}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/data:/data}"
CONTAINER_NAME="${CONTAINER_NAME:-ts}"
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
export DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}"
export NO_PROXY="*" no_proxy="*"
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

mapfile -t NODES < <(scontrol show hostnames "$(squeue -j "$JOBID" -h -o '%N')")
[[ ${#NODES[@]} -eq 8 ]] || {
    echo "This 1P1D layout requires eight nodes with four GPUs each" >&2; exit 1;
}
HEAD="${NODES[0]}"
P_NODES="${NODES[0]},${NODES[1]},${NODES[2]},${NODES[3]}"
D_NODES="${NODES[4]},${NODES[5]},${NODES[6]},${NODES[7]}"
BASE_URL="http://${HEAD}:8000"

# An enclosing srun may restrict this to the coordinator node.
unset SLURM_JOB_NODELIST
SRUN=(srun --jobid="$JOBID" --overlap --kill-on-bad-exit=1 --wait=30)
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

launch_server() {
    local role=$1 nodes=$2
    shift 2
    SERVER_LOG=/tmp/tokenspeed_server_pd_1adp16_1adp16_ep16_${role}.log
    "${SRUN[@]}" --job-name="k3_${role}" \
        --nodes=4 --ntasks=4 --ntasks-per-node=1 --nodelist="$nodes" \
        --mpi=pmix "${SRUN_ARGS[@]}" bash -c '
set -euo pipefail
role=$1
shift
export TOKENSPEED_SKIP_GRPC_WARMUP=1
exec bash configs/attn_dp16_moe_ep16.sh \
    --grpc \
    --disaggregation-mode "$role" \
    --load-balance-method round_robin \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device "$DISAGGREGATION_IB_DEVICE" \
    "$@"
' _ "$role" "$@" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    local pid=$1 log=$2
    local start=$SECONDS
    until grep -q 'health status -> SERVING' "$log"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Server died early; see $log" >&2
            tail -100 "$log" >&2
            return 1
        fi
        if grep -qE 'CUDA out of memory|OutOfMemory|RuntimeError|Killed' "$log"; then
            tail -100 "$log" >&2
            return 1
        fi
        if (( SECONDS - start > 7200 )); then
            echo "Timed out waiting for server; see $log" >&2
            return 1
        fi
        sleep 5
    done
    echo "Server ready after $((SECONDS - start))s: $log"
}

stop_server() {
    local pid
    for pid in "$@"; do
        if [[ -n "$pid" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$@"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

wait_for_port_free() {
    local start=$SECONDS
    while curl -sf -m 3 -o /dev/null "${BASE_URL}/readiness"; do
        if (( SECONDS - start > 90 )); then
            echo "${BASE_URL} still answering; refusing to start another gateway" >&2
            return 1
        fi
        sleep 1
    done
}

trap 'stop_server "$GATEWAY_PID" "$PREFILL_PID" "$DECODE_PID"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_port_free
SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${SCRIPT_DIR}/outputs/${SWEEP_TS}"
echo "1P1D: prefill=$P_NODES decode=$D_NODES; outputs=$SWEEP_DIR"

launch_server prefill "$P_NODES" \
    --port 18346 \
    --disaggregation-bootstrap-port 8998 \
    --mm-encoder-tp-mode data
PREFILL_PID=$SERVER_PID
launch_server decode "$D_NODES" \
    --port 18347 \
    --mm-encoder-tp-mode weights
DECODE_PID=$SERVER_PID
wait_for_ready "$PREFILL_PID" /tmp/tokenspeed_server_pd_1adp16_1adp16_ep16_prefill.log
wait_for_ready "$DECODE_PID" /tmp/tokenspeed_server_pd_1adp16_1adp16_ep16_decode.log

# Launch the shared P/D gateway
"${SRUN[@]}" \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="$HEAD" \
    "${SRUN_ARGS[@]}" \
    python3 -m smg launch \
    --pd-disaggregation \
    --prefill "grpc://${HEAD}:18346" 8998 \
    --decode "grpc://${NODES[4]}:18347" \
    --host 0.0.0.0 \
    --port 8000 \
    --model-path nvidia/Kimi-K3-NVFP4 \
    --tokenizer-path nvidia/Kimi-K3-NVFP4 \
    --reasoning-parser passthrough \
    --tool-call-parser passthrough \
    --prefill-policy round_robin \
    --decode-policy round_robin \
    --dp-aware \
    --sticky-sessions \
    --assignment-mode delegate \
    --disable-retries \
    --disable-circuit-breaker \
    --disable-health-check \
    --request-timeout-secs 1800 \
    --max-concurrent-requests -1 \
    > /tmp/tokenspeed_gateway_pd_1adp16_1adp16_ep16.log 2>&1 &
GATEWAY_PID=$!
START=$SECONDS
until curl -sf -m 5 -o /dev/null "${BASE_URL}/readiness"; do
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null || (( SECONDS - START > 600 )); then
        tail -100 /tmp/tokenspeed_gateway_pd_1adp16_1adp16_ep16.log >&2
        exit 1
    fi
    sleep 5
done

# Warmup uses conversations 2784..2799, after the measured range.
echo "Warmup..."
"${SRUN[@]}" \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="$HEAD" \
    "${SRUN_ARGS[@]}" \
    evalscope perf \
    --model nvidia/Kimi-K3-NVFP4 \
    --url "$BASE_URL/v1/chat/completions" \
    --api openai \
    --dataset swe_smith \
    --dataset-path agentic_dataset_x128.json \
    --max-tokens 500 \
    --temperature 0 \
    --stream \
    --multi-turn \
    --number 16 \
    --parallel 16 \
    --dataset-offset 2784 \
    --extra-args '{"ignore_eos": true}' \
    --outputs-dir /tmp/outputs

echo "Benchmark..."
"${SRUN[@]}" \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="$HEAD" \
    "${SRUN_ARGS[@]}" \
    evalscope perf \
    --model nvidia/Kimi-K3-NVFP4 \
    --url "$BASE_URL/v1/chat/completions" \
    --api openai \
    --dataset swe_smith \
    --dataset-path agentic_dataset_x128.json \
    --max-tokens 500 \
    --temperature 0 \
    --stream \
    --multi-turn \
    --number 32 64 128 256 512 768 1024 \
    --parallel 16 32 64 128 256 384 512 \
    --extra-args '{"ignore_eos": true}' \
    --name pd_1adp16_1adp16_ep16 \
    --outputs-dir "$SWEEP_DIR" \
    --no-timestamp

echo "Benchmark complete: $SWEEP_DIR (32 GPUs total)"
