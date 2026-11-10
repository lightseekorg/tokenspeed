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

# Runs the full AgentX trace against a deployment started by
# serve_dwdp16_dep16.sh, one point per concurrency. See README.md.
set -euo pipefail

CONCURRENCIES=(128 256 384 512 576)
DURATION=3600
WARMUP_REQUESTS_PER_LANE=10

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOBID="${SLURM_JOB_ID:?use the allocation the server runs in}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?set the TokenSpeed container image}"
CONTAINER_NAME="${CONTAINER_NAME:-ts}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:?mount the repository, models, client and cache}"
SERVE_DIR="${SERVE_DIR:?run directory printed by serve_dwdp16_dep16.sh}"
export CLIENT_ROOT="${CLIENT_ROOT:?prepared AgentX client directory}"
export CLIENT_CACHE="${CLIENT_CACHE:?Hugging Face cache holding the AgentX trace}"
set -a
source "$SERVE_DIR/endpoints.env"
set +a
export DURATION WARMUP_REQUESTS_PER_LANE
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
SWEEP_DIR="$SCRIPT_DIR/outputs/agentx_$(date +%Y%m%d_%H%M%S)"

SRUN=(srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --nodelist="$AGENTX_HEAD")
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

mkdir -p "$SWEEP_DIR"
echo "Sweep outputs: $SWEEP_DIR"
for CONCURRENCY in "${CONCURRENCIES[@]}"; do
    export CONCURRENCY RESULT_DIR="$SWEEP_DIR/c$CONCURRENCY"
    # Every point starts with empty GPU caches and an empty shared L2 pool.
    "${SRUN[@]}" "${SRUN_ARGS[@]}" python3 engine_control.py reset \
        --prefill "$PREFILL_GRPC" --decode "$DECODE_GRPC" --prefill-dp-size 16 --decode-dp-size 16 \
        --shared-l2-owner "$SHARED_L2_OWNER" > "$SWEEP_DIR/reset-c$CONCURRENCY.log" 2>&1
    "${SRUN[@]}" "${SRUN_ARGS[@]}" bash agentx_client.sh > "$SWEEP_DIR/client-c$CONCURRENCY.log" 2>&1
done
