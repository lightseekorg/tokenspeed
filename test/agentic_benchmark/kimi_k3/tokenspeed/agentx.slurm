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

# Run via sbatch with a native client, or from the login host in a held job.
# All paths must be absolute and visible where they are used. See agentx.md.
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch or an existing allocation}"
: "${SOURCE_ROOT:?absolute checkout path; independent of sbatch submit directory}"
: "${SERVER_SCRIPT:?absolute per-node server launcher path}"
: "${CONTAINER_IMAGE:?pinned server image}"
: "${CONTAINER_MOUNTS:?explicit shared mounts}"
: "${CLIENT_PYTHON:?absolute Python from the native EvalScope agentx venv}"
: "${SCENARIO_FILE:?absolute EvalScope AgentX scenario JSON}"
: "${MODEL_NAME:?served model name}"
: "${TOKENIZER_PATH:?pinned tokenizer path or ID}"
: "${DATASET_PATH:?local directory containing traces.jsonl}"
: "${CONCURRENCY:?root session concurrency}"
: "${DURATION:?measurement seconds}"
: "${SEED:?client random seed}"
: "${API_PORT:?server API port}"
: "${READINESS_PATH:?readiness endpoint, including leading slash}"
: "${READINESS_TIMEOUT:?maximum server startup seconds}"
: "${CLIENT_TIMEOUT:?maximum total client wall time seconds}"
: "${HOLD_AFTER_RUN:?1 keeps service/allocation until release-requested or job limit; 0 stops own server step}"
: "${RUN_ROOT:?new absolute output directory}"
case "$HOLD_AFTER_RUN" in 0|1) ;; *) echo 'HOLD_AFTER_RUN must be 0 or 1' >&2; exit 2 ;; esac
for variable in SOURCE_ROOT SERVER_SCRIPT CLIENT_PYTHON SCENARIO_FILE DATASET_PATH RUN_ROOT; do
    [[ "${!variable}" = /* ]] || { echo "$variable must be absolute" >&2; exit 2; }
done
for variable in CONCURRENCY DURATION API_PORT READINESS_TIMEOUT CLIENT_TIMEOUT; do
    [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "$variable must be a positive integer" >&2; exit 2; }
done
SCRIPT_DIR="$SOURCE_ROOT/test/agentic_benchmark/kimi_k3/tokenspeed"
[[ -f "$SERVER_SCRIPT" && -f "$SCENARIO_FILE" && -x "$CLIENT_PYTHON" ]]
[[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%u')" = "$(id -un)" ]] || { echo 'Allocation not owned by current user' >&2; exit 1; }
[[ "$(squeue -h -j "$SLURM_JOB_ID" -o '%T')" = RUNNING ]] || { echo 'Allocation must be RUNNING' >&2; exit 1; }
# Atomic reservation: never overwrite evidence or run the same output twice.
mkdir "$RUN_ROOT"
cp "$SCENARIO_FILE" "$RUN_ROOT/scenario.json"
"$CLIENT_PYTHON" "$SCRIPT_DIR/agentx_result.py" prepare "$RUN_ROOT"
scontrol show job "$SLURM_JOB_ID" > "$RUN_ROOT/allocation.txt"
mapfile -t NODES < <(scontrol show hostnames "$(squeue -h -j "$SLURM_JOB_ID" -o '%N')")
[[ "${#NODES[@]}" -gt 0 ]]
export HEAD="${NODES[0]}"
BASE_URL="http://${HEAD}:${API_PORT}"
# Any HTTP response on this port means another service owns it.
if curl --silent --max-time 3 --output /dev/null "$BASE_URL/v1/models"; then
    echo 'API port already occupied; refusing to use or stop that service' >&2
    exit 1
fi
export NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
STEP_NAME="agentx_server_${SLURM_JOB_ID}_$$"
SERVER_PID=
finish() {
    local result=$?
    # While holding, a signal requests release through this same cleanup path.
    # Do not exit from a signal handler after removing the EXIT trap.
    trap 'result=130; HOLD_AFTER_RUN=0' INT
    trap 'result=143; HOLD_AFTER_RUN=0' TERM
    trap - EXIT
    printf '%s\n' "$result" > "$RUN_ROOT/client-exit-code.txt"
    if [[ "$HOLD_AFTER_RUN" = 1 && -n "$SERVER_PID" ]]; then
        echo 'Holding allocation; touch RUN_ROOT/release-requested to release' > "$RUN_ROOT/allocation-held.txt"
        while [[ "$HOLD_AFTER_RUN" = 1 && ! -f "$RUN_ROOT/release-requested" ]] && kill -0 "$SERVER_PID" 2>/dev/null; do sleep 5 || true; done
    fi
    # Finish the owned-step cleanup even if a second signal arrives.
    trap '' INT TERM
    # Restrict cleanup to the unique step this invocation started.
    while read -r step name; do
        if [[ "$name" = "$STEP_NAME" ]]; then scancel "$step" || true; fi
    done < <(squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j')
    if [[ -n "$SERVER_PID" ]]; then wait "$SERVER_PID" 2>/dev/null || true; fi
    printf '%s\n' "$result" > "$RUN_ROOT/client-exit-code.txt"
    exit "$result"
}
trap finish EXIT
trap 'HOLD_AFTER_RUN=0; exit 130' INT
trap 'HOLD_AFTER_RUN=0; exit 143' TERM
SRUN=(srun --jobid="$SLURM_JOB_ID" --overlap --nodes="${#NODES[@]}" --ntasks-per-node=1
    --container-image="$CONTAINER_IMAGE" --container-name="agentx_${SLURM_JOB_ID}_$$"
    --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$SOURCE_ROOT"
    --container-remap-root --no-container-mount-home --container-writable)
"${SRUN[@]}" --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    occupied=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$occupied" ]]; then echo "GPUs already occupied: $occupied" >&2; exit 1; fi
    nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
' > "$RUN_ROOT/gpu-preflight.log" 2>&1
"${SRUN[@]}" --job-name="$STEP_NAME" --mpi=pmix --kill-on-bad-exit=1 \
    bash "$SERVER_SCRIPT" > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
START=$SECONDS
until curl --fail --silent --max-time 5 --output /dev/null "$BASE_URL$READINESS_PATH"; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo 'Server step exited before readiness' >&2; exit 1; }
    (( SECONDS - START < READINESS_TIMEOUT )) || { echo 'Server readiness timeout' >&2; exit 1; }
    sleep 5
done
printf '%s\n' "$BASE_URL" > "$RUN_ROOT/endpoint.txt"
curl --fail --silent --max-time 10 "$BASE_URL/v1/models" > "$RUN_ROOT/models.json"
# The client runs on the orchestrating host, not implicitly inside the GPU image.
# AIPerf 0.12 treats even offline=0 as true and mishandles local tokenizer paths.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
timeout --signal=INT --kill-after=45 "$CLIENT_TIMEOUT" "$CLIENT_PYTHON" -c \
    'from evalscope.cli.cli import run_cmd; run_cmd()' perf \
    --scenario "$(cat "$RUN_ROOT/scenario.json")" --model "$MODEL_NAME" \
    --tokenizer-path "$TOKENIZER_PATH" --url "$BASE_URL/v1/chat/completions" \
    --dataset-path "$DATASET_PATH" --parallel "$CONCURRENCY" --duration "$DURATION" \
    --seed "$SEED" --outputs-dir "$RUN_ROOT/client" > "$RUN_ROOT/client.log" 2>&1
"$CLIENT_PYTHON" "$SCRIPT_DIR/agentx_result.py" audit "$RUN_ROOT"

# A controller may retain a successful service for additional client runs.
if [[ -f "$RUN_ROOT/hold-after-client" ]]; then HOLD_AFTER_RUN=1; fi
