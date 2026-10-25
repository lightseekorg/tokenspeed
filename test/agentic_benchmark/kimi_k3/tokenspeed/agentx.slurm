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

# Server and client run as job steps in the same container. See agentx.md.
set -euo pipefail

CONFIGS=(
    attn_tp8_moe_tp8
    attn_tp8_moe_ep8
)
CONCURRENCIES=(1 2 4 8 16)
DURATION=3600
WARMUP_REQUESTS_PER_LANE=10
DRAIN_TIMEOUT=1800

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOBID="${SLURM_JOB_ID:?use sbatch or an existing allocation}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?set the TokenSpeed container image}"
CONTAINER_NAME="${CONTAINER_NAME:-ts}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:?mount the repository, client and HF cache}"
CLIENT_ROOT="${CLIENT_ROOT:?prepared AgentX client directory}"
export HF_HUB_CACHE="${HF_HUB_CACHE:?shared HF Hub cache}"
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$JOBID" -o '%N')")
(( ${#NODES[@]} >= 2 )) || { echo 'K3 TP8 requires two four-GPU nodes' >&2; exit 2; }
NODES=("${NODES[@]:0:2}")
SERVER_NODES="${NODES[0]},${NODES[1]}"
HEAD="${NODES[0]}"
BASE_URL="http://${HEAD}:8000"
METRICS_URL="http://${HEAD}:8010/metrics"
SWEEP_DIR="$SCRIPT_DIR/outputs/agentx_$(date +%Y%m%d_%H%M%S)"
SERVER_STEP="agentx_server_${JOBID}_$$"
CLIENT_STEP="agentx_client_${JOBID}_$$"
SERVER_PID=
CLIENT_PID=

SRUN=(srun --jobid="$JOBID" --overlap)
SRUN_ARGS=(
    --container-image="$CONTAINER_IMAGE"
    --container-name="$CONTAINER_NAME"
    --container-mounts="$CONTAINER_MOUNTS"
    --container-workdir="$SCRIPT_DIR"
    --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,HF_HUB_CACHE
    --container-remap-root
    --no-container-mount-home
    --container-writable
)

stop_steps() {
    local pid step name
    for pid in "$CLIENT_PID" "$SERVER_PID"; do
        if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    # Cancel only this invocation's steps, including any surviving remote tasks.
    while read -r step name; do
        if [[ "$name" = "$SERVER_STEP" || "$name" = "$CLIENT_STEP" ]]; then
            scancel "$step" || true
        fi
    done < <(squeue -h -s -j "$JOBID" -o '%i %j')
    for pid in "$CLIENT_PID" "$SERVER_PID"; do
        if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
    done
    CLIENT_PID=
    SERVER_PID=
}

launch_server() {
    "${SRUN[@]}" --nodes="${#NODES[@]}" --ntasks="${#NODES[@]}" --nodelist="$SERVER_NODES" --ntasks-per-node=1 \
        --cpus-per-task=64 --gpus-per-node=4 --mpi=pmix --kill-on-bad-exit=1 \
        --job-name="$SERVER_STEP" "${SRUN_ARGS[@]}" \
        bash -x "$SCRIPT_DIR/configs/${CONFIG}.sh" > "$RUN_DIR/server.log" 2>&1 &
    SERVER_PID=$!
}

wait_for_ready() {
    local start=$SECONDS
    until curl --fail --silent --max-time 5 --output /dev/null "$BASE_URL/readiness"; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Server exited before readiness' >&2; return 1; }
        (( SECONDS - start < 7500 )) || { echo 'Server readiness timeout' >&2; return 1; }
        sleep 5
    done
    TOTAL_CPU_DRAM_GB=$(sed -nE 's/.*Allocated ([0-9.]+) GB compact Host L2 .*/\1/p' "$RUN_DIR/server.log" |
        awk '{ total += $1 } END { printf "%.0f\n", total }')
}

run_benchmark() {
    local ep_size=1
    if [[ "$CONFIG" = attn_tp8_moe_ep8 ]]; then ep_size=8; fi
    mkdir "$RUN_DIR/c${CONCURRENCY}"

    CLIENT_ROOT="$CLIENT_ROOT" DURATION="$DURATION" \
    AIPERF_WARMUP_REQUESTS_PER_LANE="$WARMUP_REQUESTS_PER_LANE" \
    AIPERF_SERVER_URL="$BASE_URL" AIPERF_SERVER_METRICS_URLS="$METRICS_URL" \
    RESULT_DIR="$RUN_DIR/c${CONCURRENCY}/client" CONC="$CONCURRENCY" EP_SIZE="$ep_size" \
    TOTAL_CPU_DRAM_GB="$TOTAL_CPU_DRAM_GB" \
        "${SRUN[@]}" --nodes=1 --nodelist="$HEAD" --ntasks=1 --cpus-per-task=8 --job-name="$CLIENT_STEP" \
        "${SRUN_ARGS[@]}" timeout --signal=INT --kill-after=45 8400 \
        bash -se -o pipefail > "$RUN_DIR/c${CONCURRENCY}/client.log" 2>&1 <<'CLIENT' &
mkdir "$RESULT_DIR"
INFMAX_CONTAINER_WORKSPACE="$CLIENT_ROOT/inferencex"
AIPERF_RUNTIME_DIR="$CLIENT_ROOT"
export MODEL=nvidia/Kimi-K3-NVFP4 SERVED_MODEL_NAME=nvidia/Kimi-K3-NVFP4 FRAMEWORK=tokenspeed
export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=tokenspeed
export KV_OFFLOAD_BACKEND_METADATA='{"name":"tokenspeed"}'
export IS_MULTINODE=false DISAGG=false
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
AIPERF_DEPS_READY=1
WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126 resolve_trace_source
unset MAX_MODEL_LEN
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0

AIPERF_EXTRA_INPUTS='{"ignore_eos":true,"temperature":1.0,"top_p":0.95,"top_k":20}' \
build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --benchmark-grace-period 30 --request-timeout-seconds 21600"

# The two nodes form one aggregate TP8 worker in InferenceX's reporting schema.
MODEL_PREFIX=kimik3 RUNNER_TYPE=gb300 PRECISION=fp4 SPEC_DECODING=eagle3 \
TP=8 PP_SIZE=1 PCP_SIZE=1 DCP_SIZE=1 \
AGENTIC_OUTPUT_DIR="$RESULT_DIR" RESULT_FILENAME=inferencex_result \
AIPERF_FAILED_REQUEST_THRESHOLD=0.1 AIPERF_REQUIRED_SERVER_METRIC_PREFIX=tokenspeed: \
ENABLE_AGENTX_POWER=0 PYTHONPATH="$INFMAX_CONTAINER_WORKSPACE" \
AIPERF_DATASET_WEKA_PARALLEL_WORKERS=2 TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
run_agentic_replay_and_write_outputs "$RESULT_DIR"
CLIENT
    CLIENT_PID=$!
    wait "$CLIENT_PID"
    CLIENT_PID=
}

wait_for_drain() {
    local start=$SECONDS
    while true; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Server exited while draining' >&2; return 1; }
        if curl --fail --silent --max-time 5 "$METRICS_URL" |
            awk '
                /^tokenspeed:num_requests_(running|waiting)(\{|[[:space:]])/ {
                    if ($0 ~ /^tokenspeed:num_requests_running/) running=1
                    else waiting=1
                    sub(/^tokenspeed:num_requests_(running|waiting)(\{.*\})?[[:space:]]+/, "")
                    if ($1 != 0) busy=1
                }
                END { exit !(running && waiting && !busy) }
            '; then
            return 0
        fi
        (( SECONDS - start < DRAIN_TIMEOUT )) || { echo 'Server drain timeout' >&2; return 1; }
        sleep 1
    done
}

trap stop_steps EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Sweep outputs: $SWEEP_DIR"
for CONFIG in "${CONFIGS[@]}"; do
    RUN_DIR="$SWEEP_DIR/$CONFIG"
    mkdir -p "$RUN_DIR"
    launch_server
    wait_for_ready
    for CONCURRENCY in "${CONCURRENCIES[@]}"; do
        run_benchmark
        wait_for_drain
    done
    stop_steps
done
