#!/usr/bin/bash

# Submit once per topology, for example:
#   sbatch --export=ALL,VLLM_BENCH_CONFIG=attn_tp8_moe_tp8,\
#       KIMI_K3_CONTAINER_IMAGE=/scratch/fsw/.../vllm.sqsh,\
#       KIMI_K3_MODEL_DIR=/scratch/fsw/.../Kimi-K3-NVFP4,\
#       KIMI_K3_DRAFT_DIR=/scratch/fsw/.../Kimi-K3-DSpark,\
#       KIMI_K3_DATASET=/scratch/fsw/.../agentic_dataset.json,\
#       KIMI_K3_OUTPUT_ROOT=/scratch/fsw/.../agentic-pareto run_pareto.sbatch

#SBATCH --job-name=k3-vllm-agentic-pareto
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${VLLM_BENCH_CONFIG:?Set VLLM_BENCH_CONFIG to attn_tp8_moe_tp8, attn_tp8_moe_ep8, or attn_dp8_moe_ep8}"
: "${KIMI_K3_CONTAINER_IMAGE:?Set KIMI_K3_CONTAINER_IMAGE to the vLLM SQSH path}"
: "${KIMI_K3_MODEL_DIR:?Set KIMI_K3_MODEL_DIR to the Kimi-K3-NVFP4 checkpoint}"
: "${KIMI_K3_DRAFT_DIR:?Set KIMI_K3_DRAFT_DIR to the Kimi-K3-DSpark checkpoint}"
: "${KIMI_K3_DATASET:?Set KIMI_K3_DATASET to agentic_dataset.json}"
: "${KIMI_K3_OUTPUT_ROOT:?Set KIMI_K3_OUTPUT_ROOT to a shared result directory}"

BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$BENCH_DIR/../../../.." && pwd)
VALIDATOR="$BENCH_DIR/../validate_evalscope.py"
COLLECTOR="$BENCH_DIR/collect_outputs.py"
CONFIG="$BENCH_DIR/configs/$VLLM_BENCH_CONFIG.sh"
VLLM_LAUNCHER="$BENCH_DIR/bin/vllm"
VLLM_BENCH_MODE=${VLLM_BENCH_MODE:-full}
KIMI_K3_CONTAINER_MOUNTS=${KIMI_K3_CONTAINER_MOUNTS:-/scratch/fsw:/scratch/fsw}
KIMI_K3_EVALSCOPE_BIN=${KIMI_K3_EVALSCOPE_BIN:-evalscope}
KIMI_K3_CLIENT_ENV_TAR=${KIMI_K3_CLIENT_ENV_TAR:-}
KIMI_K3_DATASET_SHA256=${KIMI_K3_DATASET_SHA256:-2405552a4f320bbaaf084a0f25f57dc1c9af44e69612d0c9d098a137db3463f2}

test "$SLURM_NNODES" -eq 2
test -s "$KIMI_K3_CONTAINER_IMAGE"
test -d "$KIMI_K3_MODEL_DIR"
test -d "$KIMI_K3_DRAFT_DIR"
test -s "$KIMI_K3_DATASET"
test -x "$VLLM_LAUNCHER"
test -x "$CONFIG"
test -f "$VALIDATOR"
test -f "$COLLECTOR"
if [[ -n "$KIMI_K3_CLIENT_ENV_TAR" ]]; then
    test -s "$KIMI_K3_CLIENT_ENV_TAR"
fi
test "$(sha256sum "$KIMI_K3_DATASET" | awk '{print $1}')" = "$KIMI_K3_DATASET_SHA256"

case "$VLLM_BENCH_MODE" in
    smoke)
        CLIENT_NUMBERS="4"
        CLIENT_PARALLELS="1"
        ;;
    full)
        CLIENT_NUMBERS="4 8 8 16 32"
        CLIENT_PARALLELS="1 2 4 8 16"
        ;;
    *)
        echo "VLLM_BENCH_MODE must be smoke or full" >&2
        exit 2
        ;;
esac

case "$VLLM_BENCH_CONFIG" in
    attn_tp8_moe_tp8|attn_tp8_moe_ep8|attn_dp8_moe_ep8) ;;
    *) echo "unknown VLLM_BENCH_CONFIG=$VLLM_BENCH_CONFIG" >&2; exit 2 ;;
esac

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HTTP_PORT=${KIMI_K3_HTTP_PORT:-8002}
RUN_DIR="$KIMI_K3_OUTPUT_ROOT/vllm/$VLLM_BENCH_CONFIG/$SLURM_JOB_ID"
SERVER_LOG="$RUN_DIR/server.log"
mkdir -p "$RUN_DIR/preflight" "$RUN_DIR/results"

export MODEL_DIR=$KIMI_K3_MODEL_DIR
export DRAFT_DIR=$KIMI_K3_DRAFT_DIR
export DATASET=$KIMI_K3_DATASET
export EVALSCOPE_BIN=$KIMI_K3_EVALSCOPE_BIN
export CLIENT_ENV_TAR=$KIMI_K3_CLIENT_ENV_TAR
export VLLM_MASTER_ADDR=$MASTER_ADDR
export VLLM_MASTER_PORT=${VLLM_MASTER_PORT:-29500}
export VLLM_DP_RPC_PORT=${VLLM_DP_RPC_PORT:-29501}
export VLLM_DP_LOCAL_SIZE=${VLLM_DP_LOCAL_SIZE:-4}
export VLLM_REAL_BIN=${KIMI_K3_VLLM_BIN:-/usr/local/bin/vllm}
export VLLM_BENCH_CONFIG VLLM_BENCH_MODE HTTP_PORT
export CLIENT_NUMBERS CLIENT_PARALLELS
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_DMABUF_ENABLE=0
export UCX_TLS=cuda_copy,cuda_ipc,tcp
export UCX_CUDA_IPC_ENABLE_MNNVL=y
export UCX_MEMTYPE_CACHE=n
export UCX_MEMTYPE_REG_WHOLE=n
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl
export ENROOT_CACHE_PATH="/tmp/k3-vllm-$SLURM_JOB_ID/enroot-cache"
export ENROOT_DATA_PATH="/tmp/k3-vllm-$SLURM_JOB_ID/enroot-data"
export ENROOT_RUNTIME_PATH="/tmp/k3-vllm-$SLURM_JOB_ID/enroot-runtime"

CONTAINER_ARGS=(
    --container-image="$KIMI_K3_CONTAINER_IMAGE"
    --container-mounts="$KIMI_K3_CONTAINER_MOUNTS"
    --container-workdir="$BENCH_DIR"
    --container-env=MODEL_DIR,DRAFT_DIR,DATASET,EVALSCOPE_BIN,CLIENT_ENV_TAR,VLLM_MASTER_ADDR,VLLM_MASTER_PORT,VLLM_DP_RPC_PORT,VLLM_DP_LOCAL_SIZE,VLLM_REAL_BIN,VLLM_BENCH_CONFIG,VLLM_BENCH_MODE,HTTP_PORT,CLIENT_NUMBERS,CLIENT_PARALLELS,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NVLS_ENABLE,NCCL_DMABUF_ENABLE,UCX_TLS,UCX_CUDA_IPC_ENABLE_MNNVL,UCX_MEMTYPE_CACHE,UCX_MEMTYPE_REG_WHOLE,VLLM_WORKER_MULTIPROC_METHOD,VLLM_NO_USAGE_STATS,VLLM_ALLREDUCE_USE_FLASHINFER,VLLM_FLASHINFER_ALLREDUCE_BACKEND
)

{
    scontrol show config | grep '^ClusterName'
    scontrol show job -d "$SLURM_JOB_ID"
    git -C "$REPO_ROOT" rev-parse HEAD
    git -C "$REPO_ROOT" status --short
    sha256sum "$KIMI_K3_CONTAINER_IMAGE" "$KIMI_K3_DATASET" \
        "$CONFIG" "$VLLM_LAUNCHER" "$VALIDATOR" "$COLLECTOR"
    printf 'config=%s\nmode=%s\nmaster=%s:%s\nmodel=%s\ndraft=%s\ndataset_sha256=%s\nnumber=%s\nparallel=%s\n' \
        "$VLLM_BENCH_CONFIG" "$VLLM_BENCH_MODE" "$VLLM_MASTER_ADDR" \
        "$VLLM_MASTER_PORT" "$KIMI_K3_MODEL_DIR" "$KIMI_K3_DRAFT_DIR" \
        "$KIMI_K3_DATASET_SHA256" "$CLIENT_NUMBERS" "$CLIENT_PARALLELS"
} > "$RUN_DIR/preflight/manifest.txt"

SERVER_STEP_PID=
stop_server() {
    if [[ -n "$SERVER_STEP_PID" ]]; then
        kill -TERM "$SERVER_STEP_PID" 2>/dev/null || true
        wait "$SERVER_STEP_PID" 2>/dev/null || true
    fi
}
trap stop_server EXIT

srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 --label \
    "${CONTAINER_ARGS[@]}" \
    bash -lc '
        set -euo pipefail
        export PATH="'$BENCH_DIR'/bin:$PATH"
        export VLLM_CACHE_ROOT="/tmp/k3-vllm-'$SLURM_JOB_ID'/vllm-cache"
        export XDG_CACHE_HOME="/tmp/k3-vllm-'$SLURM_JOB_ID'/xdg-cache"
        export TORCHINDUCTOR_CACHE_DIR="/tmp/k3-vllm-'$SLURM_JOB_ID'/torchinductor-cache"
        export TRITON_CACHE_DIR="/tmp/k3-vllm-'$SLURM_JOB_ID'/triton-cache"
        mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" \
            "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
        exec "'$CONFIG'"
    ' > "$SERVER_LOG" 2>&1 &
SERVER_STEP_PID=$!

start=$SECONDS
until curl -sf -o /dev/null "http://$MASTER_ADDR:$HTTP_PORT/health"; do
    if ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
        tail -300 "$SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS - start > 7200 )); then
        echo "timeout waiting for vLLM readiness" >&2
        tail -300 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done

srun --overlap --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" \
    "${CONTAINER_ARGS[@]}" \
    bash -lc '
        set -euo pipefail
        client_python=python3
        evalscope_bin="$EVALSCOPE_BIN"
        if [[ -n "$CLIENT_ENV_TAR" ]]; then
            client_dir="/tmp/k3-agentic-client-'$SLURM_JOB_ID'"
            rm -rf "$client_dir"
            mkdir -p "$client_dir"
            tar -C "$client_dir" --strip-components=1 -xf "$CLIENT_ENV_TAR"
            client_python="$client_dir/bin/python"
            evalscope_bin="$client_dir/bin/evalscope"
        fi
        command -v "$evalscope_bin" >/dev/null
        "$evalscope_bin" perf \
            --model nvidia/Kimi-K3-NVFP4 \
            --url "http://'$MASTER_ADDR':'$HTTP_PORT'/v1/chat/completions" \
            --api openai \
            --dataset swe_smith \
            --dataset-path "$DATASET" \
            --max-tokens 500 \
            --multi-turn \
            --number 2 \
            --parallel 2 \
            --extra-args '\''{"ignore_eos": true}'\'' \
            --dataset-offset 68 \
            --outputs-dir "'$RUN_DIR'/results/warmup"

        read -r -a client_number <<< "$CLIENT_NUMBERS"
        read -r -a client_parallel <<< "$CLIENT_PARALLELS"
        "$evalscope_bin" perf \
            --model nvidia/Kimi-K3-NVFP4 \
            --url "http://'$MASTER_ADDR':'$HTTP_PORT'/v1/chat/completions" \
            --api openai \
            --dataset swe_smith \
            --dataset-path "$DATASET" \
            --max-tokens 500 \
            --multi-turn \
            --number "${client_number[@]}" \
            --parallel "${client_parallel[@]}" \
            --extra-args '\''{"ignore_eos": true}'\'' \
            --name "$VLLM_BENCH_CONFIG" \
            --outputs-dir "'$RUN_DIR'/results/$VLLM_BENCH_MODE" \
            --no-timestamp

        "$client_python" "'$VALIDATOR'" \
            "'$RUN_DIR'/results/warmup" \
            "'$RUN_DIR'/results/$VLLM_BENCH_MODE" \
            --output "'$RUN_DIR'/results/request-integrity.json"
        if [[ "$VLLM_BENCH_MODE" == full ]]; then
            "$client_python" "'$COLLECTOR'" \
                "'$RUN_DIR'/results/full" \
                -o "'$RUN_DIR'/results/summary.csv"
        fi
    '

if grep -Eqi 'CUDA out of memory|torch\.OutOfMemoryError|CUDA error:|device-side assert|NCCL.*(error|fail)|Traceback \(most recent call last\)|EngineCore failed to start' "$SERVER_LOG"; then
    echo "fatal vLLM server marker found" >&2
    exit 3
fi

stop_server
SERVER_STEP_PID=
echo "$VLLM_BENCH_MODE complete: $RUN_DIR"
