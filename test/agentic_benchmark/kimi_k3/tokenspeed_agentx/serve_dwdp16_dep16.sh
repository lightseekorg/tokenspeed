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

# Serves Kimi K3 as DWDP16 prefill + DEP16 decode on eight four-GPU GB300 nodes:
# shared L2 services, both engine roles (two GPUs each per node), and the SMG
# gateway. Runs in the foreground until interrupted. See README.md.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOBID="${SLURM_JOB_ID:?use an existing allocation}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?set the TokenSpeed container image}"
CONTAINER_NAME="${CONTAINER_NAME:-ts}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:?mount the repository, models and gateway binary}"
export MODEL_PATH="${MODEL_PATH:?local nvidia/Kimi-K3-NVFP4 snapshot}"
export TOKENIZER_PATH="${TOKENIZER_PATH:?real-file copy of the model tokenizer, see README.md}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:?local lightseekorg/kimi-k3-eagle3-mla snapshot}"
export SMG_BINARY="${SMG_BINARY:?SMG gateway built as in README.md}"
export DISAGGREGATION_IB_DEVICE="${DISAGGREGATION_IB_DEVICE:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}"
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN_DIR="$SCRIPT_DIR/outputs/serve_$(date +%Y%m%d_%H%M%S)"

mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$JOBID" -o '%N')")
(( ${#NODES[@]} >= 8 )) || { echo 'DWDP16 + DEP16 requires eight four-GPU nodes' >&2; exit 2; }
NODES=("${NODES[@]:0:8}")
NODELIST=$(IFS=,; echo "${NODES[*]}")
export AGENTX_HEAD="${NODES[0]}"
STEP_PREFIX="agentx_pd_${JOBID}_$$"
STEP_PIDS=()

SRUN=(srun --jobid="$JOBID" --overlap --kill-on-bad-exit=1)
SRUN_ARGS=(
    --container-image="$CONTAINER_IMAGE"
    --container-name="$CONTAINER_NAME"
    --container-mounts="$CONTAINER_MOUNTS"
    --container-workdir="$SCRIPT_DIR"
    --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES
    --container-remap-root
    --no-container-mount-home
    --container-writable
)

stop_steps() {
    local pid step name
    for pid in "${STEP_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    # Cancel only this invocation's steps, including any surviving remote tasks.
    while read -r step name; do
        if [[ "$name" == "$STEP_PREFIX"_* ]]; then scancel "$step" || true; fi
    done < <(squeue -h -s -j "$JOBID" -o '%i %j')
    for pid in "${STEP_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}

# Usage: launch NAME NODES OUTPUT COMMAND...; one task per node.
launch() {
    local name=$1 nodes=$2 output=$3
    shift 3
    local count
    count=$(tr ',' '\n' <<< "$nodes" | wc -l)
    "${SRUN[@]}" --nodes="$count" --ntasks="$count" --ntasks-per-node=1 --nodelist="$nodes" \
        --job-name="${STEP_PREFIX}_$name" --output="$output" "${SRUN_ARGS[@]}" "$@" &
    STEP_PIDS+=($!)
}

wait_for() {
    local what=$1 timeout=$2
    shift 2
    local start=$SECONDS
    until "$@"; do
        for pid in "${STEP_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null || { echo "A step exited while waiting for $what" >&2; return 1; }
        done
        (( SECONDS - start < timeout )) || { echo "Timed out waiting for $what" >&2; return 1; }
        sleep 10
    done
}

providers_joined() {
    local joined
    joined=$(grep -c 'shared L2 provider .* joined' "$RUN_DIR/shared_l2_owner.log" 2>/dev/null) || true
    (( ${joined:-0} >= 16 ))
}

wait_for_engines() {
    "${SRUN[@]}" --nodes=1 --ntasks=1 --nodelist="$AGENTX_HEAD" --job-name="${STEP_PREFIX}_ready" "${SRUN_ARGS[@]}" \
        python3 engine_control.py wait-ready --prefill "$AGENTX_HEAD:18446" --decode "$AGENTX_HEAD:18447" \
        --prefill-dp-size 16 --decode-dp-size 16 --timeout-s 10800 > "$RUN_DIR/readiness.log" 2>&1 &
    local ready=$! pid
    while kill -0 "$ready" 2>/dev/null; do
        for pid in "${STEP_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$ready" 2>/dev/null || true
                echo 'A step exited while waiting for the engines' >&2
                return 1
            fi
        done
        # A failed head-node server can leave its step alive behind engine subprocesses.
        if grep -qs '^Traceback' "$RUN_DIR"/prefill-*.log "$RUN_DIR"/decode-*.log; then
            kill -TERM "$ready" 2>/dev/null || true
            echo 'An engine failed while starting; see its log' >&2
            return 1
        fi
        sleep 10
    done
    wait "$ready"
}

gateway_ready() {
    curl --fail --silent --max-time 5 --output /dev/null "http://$AGENTX_HEAD:8002/readiness"
}

trap stop_steps EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$RUN_DIR"
echo "Serving on $NODELIST; logs in $RUN_DIR"
# pyxis creates the named container on a node's first step; create it on every
# node at once, so that later steps sharing a node only attach to it.
"${SRUN[@]}" --nodes=8 --ntasks=8 --ntasks-per-node=1 --nodelist="$NODELIST" --job-name="${STEP_PREFIX}_init" \
    "${SRUN_ARGS[@]}" true
launch shared_l2_owner "$AGENTX_HEAD" "$RUN_DIR/shared_l2_owner.log" bash configs/shared_l2_services.sh owner
launch shared_l2_providers "$NODELIST" "$RUN_DIR/shared_l2_provider-%N.log" bash configs/shared_l2_services.sh provider
wait_for 'shared L2 providers' 1800 providers_joined
launch decode "$NODELIST" "$RUN_DIR/decode-%N.log" bash configs/decode_dep16.sh
# Stagger the two roles' container attach on the shared nodes.
sleep 20
launch prefill "$NODELIST" "$RUN_DIR/prefill-%N.log" bash configs/prefill_dwdp16.sh
wait_for_engines
launch gateway "$AGENTX_HEAD" "$RUN_DIR/gateway.log" bash configs/gateway.sh
wait_for 'gateway readiness' 600 gateway_ready

cat > "$RUN_DIR/endpoints.env" <<ENDPOINTS
AGENTX_HEAD=$AGENTX_HEAD
GATEWAY_URL=http://$AGENTX_HEAD:8002
PREFILL_GRPC=$AGENTX_HEAD:18446
DECODE_GRPC=$AGENTX_HEAD:18447
SHARED_L2_OWNER=tcp://$AGENTX_HEAD:47100
MODEL_PATH=$MODEL_PATH
TOKENIZER_PATH=$TOKENIZER_PATH
ENDPOINTS
echo "Ready: http://$AGENTX_HEAD:8002; run bench_agentx.sh with SERVE_DIR=$RUN_DIR"
wait -n "${STEP_PIDS[@]}"
echo 'A server step exited' >&2
exit 1
