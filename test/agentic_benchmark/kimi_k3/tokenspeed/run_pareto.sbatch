#!/usr/bin/bash

# Submit once per topology, for example:
#   sbatch --export=ALL,TS_BENCH_CONFIG=attn_tp8_moe_tp8,\
#       KIMI_K3_CONTAINER_IMAGE=/scratch/fsw/.../tokenspeed.sqsh,\
#       KIMI_K3_MODEL_DIR=/scratch/fsw/.../Kimi-K3-NVFP4,\
#       KIMI_K3_DRAFT_DIR=/scratch/fsw/.../Kimi-K3-DSpark,\
#       KIMI_K3_DATASET=/scratch/fsw/.../agentic_dataset.json,\
#       KIMI_K3_OUTPUT_ROOT=/scratch/fsw/.../agentic-pareto run_pareto.sbatch

#SBATCH --job-name=k3-ts-agentic-pareto
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

: "${TS_BENCH_CONFIG:?Set TS_BENCH_CONFIG to attn_tp8_moe_tp8, attn_tp8_moe_ep8, or attn_dp8_moe_ep8}"
: "${KIMI_K3_CONTAINER_IMAGE:?Set KIMI_K3_CONTAINER_IMAGE to the TokenSpeed SQSH path}"
: "${KIMI_K3_MODEL_DIR:?Set KIMI_K3_MODEL_DIR to the Kimi-K3-NVFP4 checkpoint}"
: "${KIMI_K3_DRAFT_DIR:?Set KIMI_K3_DRAFT_DIR to the Kimi-K3-DSpark checkpoint}"
: "${KIMI_K3_DATASET:?Set KIMI_K3_DATASET to agentic_dataset.json}"
: "${KIMI_K3_OUTPUT_ROOT:?Set KIMI_K3_OUTPUT_ROOT to a shared result directory}"

BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$BENCH_DIR/../../../.." && pwd)
VALIDATOR="$BENCH_DIR/../validate_evalscope.py"
COLLECTOR="$BENCH_DIR/collect_outputs.py"
TS_BENCH_MODE=${TS_BENCH_MODE:-full}
KIMI_K3_CONTAINER_MOUNTS=${KIMI_K3_CONTAINER_MOUNTS:-/scratch/fsw:/scratch/fsw}
KIMI_K3_EVALSCOPE_BIN=${KIMI_K3_EVALSCOPE_BIN:-evalscope}
KIMI_K3_CLIENT_ENV_TAR=${KIMI_K3_CLIENT_ENV_TAR:-}
KIMI_K3_DATASET_SHA256=${KIMI_K3_DATASET_SHA256:-2405552a4f320bbaaf084a0f25f57dc1c9af44e69612d0c9d098a137db3463f2}
KIMI_K3_DP_LOAD_BALANCE=${KIMI_K3_DP_LOAD_BALANCE:-shortest_queue}

test "$SLURM_NNODES" -eq 2
test -s "$KIMI_K3_CONTAINER_IMAGE"
test -d "$KIMI_K3_MODEL_DIR"
test -d "$KIMI_K3_DRAFT_DIR"
test -s "$KIMI_K3_DATASET"
test -f "$VALIDATOR"
test -f "$COLLECTOR"
if [[ -n "$KIMI_K3_CLIENT_ENV_TAR" ]]; then
    test -s "$KIMI_K3_CLIENT_ENV_TAR"
fi
test "$(sha256sum "$KIMI_K3_DATASET" | awk '{print $1}')" = "$KIMI_K3_DATASET_SHA256"

case "$TS_BENCH_MODE" in
    smoke)
        CLIENT_NUMBERS="4"
        CLIENT_PARALLELS="1"
        ;;
    full)
        CLIENT_NUMBERS="4 8 8 16 32"
        CLIENT_PARALLELS="1 2 4 8 16"
        ;;
    *)
        echo "TS_BENCH_MODE must be smoke or full" >&2
        exit 2
        ;;
esac

case "$TS_BENCH_CONFIG" in
    attn_tp8_moe_tp8)
        PARALLEL_ARGS=(--attn-tp-size 8 --moe-tp-size 8)
        PREFILL_TOKENS=${KIMI_K3_PREFILL_TOKENS:-8192}
        ;;
    attn_tp8_moe_ep8)
        PARALLEL_ARGS=(--attn-tp-size 8 --ep-size 8)
        PREFILL_TOKENS=${KIMI_K3_PREFILL_TOKENS:-8192}
        ;;
    attn_dp8_moe_ep8)
        PARALLEL_ARGS=(
            --data-parallel-size 8
            --ep-size 8
            --moe-tp-size 1
            --load-balance-method "$KIMI_K3_DP_LOAD_BALANCE"
        )
        PREFILL_TOKENS=${KIMI_K3_PREFILL_TOKENS:-2048}
        ;;
    *)
        echo "unknown TS_BENCH_CONFIG=$TS_BENCH_CONFIG" >&2
        exit 2
        ;;
esac

RESULT_CONFIG=$TS_BENCH_CONFIG

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
DIST_PORT=${KIMI_K3_DIST_PORT:-7500}
HTTP_PORT=${KIMI_K3_HTTP_PORT:-8000}
RUN_DIR="$KIMI_K3_OUTPUT_ROOT/tokenspeed/$RESULT_CONFIG/$SLURM_JOB_ID"
SERVER_LOG="$RUN_DIR/server.log"
mkdir -p "$RUN_DIR/preflight" "$RUN_DIR/results"

export MODEL_DIR=$KIMI_K3_MODEL_DIR
export DRAFT_DIR=$KIMI_K3_DRAFT_DIR
export DATASET=$KIMI_K3_DATASET
export EVALSCOPE_BIN=$KIMI_K3_EVALSCOPE_BIN
export CLIENT_ENV_TAR=$KIMI_K3_CLIENT_ENV_TAR
export MASTER_ADDR DIST_PORT HTTP_PORT PREFILL_TOKENS RESULT_CONFIG
export TS_BENCH_CONFIG TS_BENCH_MODE KIMI_K3_DP_LOAD_BALANCE
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
export ENROOT_CACHE_PATH="/tmp/k3-ts-$SLURM_JOB_ID/enroot-cache"
export ENROOT_DATA_PATH="/tmp/k3-ts-$SLURM_JOB_ID/enroot-data"
export ENROOT_RUNTIME_PATH="/tmp/k3-ts-$SLURM_JOB_ID/enroot-runtime"

CONTAINER_ARGS=(
    --container-image="$KIMI_K3_CONTAINER_IMAGE"
    --container-mounts="$KIMI_K3_CONTAINER_MOUNTS"
    --container-workdir="$REPO_ROOT"
    --container-env=MODEL_DIR,DRAFT_DIR,DATASET,EVALSCOPE_BIN,CLIENT_ENV_TAR,MASTER_ADDR,DIST_PORT,HTTP_PORT,PREFILL_TOKENS,RESULT_CONFIG,TS_BENCH_CONFIG,TS_BENCH_MODE,KIMI_K3_DP_LOAD_BALANCE,CLIENT_NUMBERS,CLIENT_PARALLELS,NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES,NCCL_MNNVL_ENABLE,NCCL_CUMEM_ENABLE,NCCL_NVLS_ENABLE,NCCL_DMABUF_ENABLE,UCX_TLS,UCX_CUDA_IPC_ENABLE_MNNVL,UCX_MEMTYPE_CACHE,UCX_MEMTYPE_REG_WHOLE
)

{
    scontrol show config | grep '^ClusterName'
    scontrol show job -d "$SLURM_JOB_ID"
    git -C "$REPO_ROOT" rev-parse HEAD
    git -C "$REPO_ROOT" status --short
    sha256sum "$KIMI_K3_CONTAINER_IMAGE" "$KIMI_K3_DATASET" "$VALIDATOR" "$COLLECTOR"
    printf 'config=%s\nmode=%s\nmaster=%s:%s\nprefill_tokens=%s\nmodel=%s\ndraft=%s\ndataset_sha256=%s\nnumber=%s\nparallel=%s\n' \
        "$TS_BENCH_CONFIG" "$TS_BENCH_MODE" "$MASTER_ADDR" "$DIST_PORT" \
        "$PREFILL_TOKENS" "$KIMI_K3_MODEL_DIR" "$KIMI_K3_DRAFT_DIR" \
        "$KIMI_K3_DATASET_SHA256" "$CLIENT_NUMBERS" "$CLIENT_PARALLELS"
    printf 'parallel_args:'
    printf ' %q' "${PARALLEL_ARGS[@]}"
    printf '\n'
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
        cd "'$REPO_ROOT'"
        export PYTHONPATH="'$REPO_ROOT'/python${PYTHONPATH:+:$PYTHONPATH}"
        export TOKENIZERS_PARALLELISM=false
        export TOKENSPEED_CUDA_ARCH=103
        export TORCH_EXTENSIONS_DIR="/tmp/k3-ts-'$SLURM_JOB_ID'/torch-extensions"
        export TRITON_CACHE_DIR="/tmp/k3-ts-'$SLURM_JOB_ID'/triton-cache"
        export FLASHINFER_WORKSPACE_BASE="/tmp/k3-ts-'$SLURM_JOB_ID'/flashinfer-workspace"
        export FLASHINFER_CUBIN_DIR="/tmp/k3-ts-'$SLURM_JOB_ID'/flashinfer-cubins"
        mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" \
            "$FLASHINFER_WORKSPACE_BASE" "$FLASHINFER_CUBIN_DIR"
        case "$TS_BENCH_CONFIG" in
            attn_tp8_moe_tp8)
                parallel_args=(--attn-tp-size 8 --moe-tp-size 8)
                ;;
            attn_tp8_moe_ep8)
                parallel_args=(--attn-tp-size 8 --ep-size 8)
                ;;
            attn_dp8_moe_ep8)
                parallel_args=(
                    --data-parallel-size 8
                    --ep-size 8
                    --moe-tp-size 1
                    --load-balance-method "$KIMI_K3_DP_LOAD_BALANCE"
                )
                ;;
        esac
        exec python3 -m tokenspeed.cli serve \
            --model "$MODEL_DIR" \
            --served-model-name nvidia/Kimi-K3-NVFP4 \
            "${parallel_args[@]}" \
            --max-model-len 80000 \
            --max-num-seqs 16 \
            --max-prefill-tokens "$PREFILL_TOKENS" \
            --chunked-prefill-size "$PREFILL_TOKENS" \
            --gpu-memory-utilization 0.92 \
            --trust-remote-code \
            --attention-backend tokenspeed_mla \
            --kda-backend cutedsl_kda \
            --moe-backend flashinfer_trtllm \
            --kv-cache-dtype fp8 \
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_DIR" \
            --speculative-num-draft-tokens 8 \
            --drafter-attention-backend tokenspeed_mla \
            --mm-encoder-tp-mode data \
            --enable-cache-report \
            --disable-weight-loader-prefetch-checkpoints \
            --dist-init-addr "$MASTER_ADDR:$DIST_PORT" \
            --host 0.0.0.0 \
            --port "$HTTP_PORT" \
            --engine-startup-timeout 7200
    ' > "$SERVER_LOG" 2>&1 &
SERVER_STEP_PID=$!

start=$SECONDS
until curl -sf -o /dev/null "http://$MASTER_ADDR:$HTTP_PORT/readiness"; do
    if ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
        tail -300 "$SERVER_LOG" >&2
        exit 1
    fi
    if (( SECONDS - start > 7200 )); then
        echo "timeout waiting for TokenSpeed readiness" >&2
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
            --url "http://$MASTER_ADDR:$HTTP_PORT/v1/chat/completions" \
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
            --url "http://$MASTER_ADDR:$HTTP_PORT/v1/chat/completions" \
            --api openai \
            --dataset swe_smith \
            --dataset-path "$DATASET" \
            --max-tokens 500 \
            --multi-turn \
            --number "${client_number[@]}" \
            --parallel "${client_parallel[@]}" \
            --extra-args '\''{"ignore_eos": true}'\'' \
            --name "$RESULT_CONFIG" \
            --outputs-dir "'$RUN_DIR'/results/$TS_BENCH_MODE" \
            --no-timestamp

        "$client_python" "'$VALIDATOR'" \
            "'$RUN_DIR'/results/warmup" \
            "'$RUN_DIR'/results/$TS_BENCH_MODE" \
            --output "'$RUN_DIR'/results/request-integrity.json"
        if [[ "$TS_BENCH_MODE" == full ]]; then
            "$client_python" "'$COLLECTOR'" \
                "'$RUN_DIR'/results/full" \
                -o "'$RUN_DIR'/results/summary.csv"
        fi
    '

if grep -Eqi 'CUDA out of memory|torch\.OutOfMemoryError|CUDA error:|device-side assert|NCCL.*(error|fail)|Traceback \(most recent call last\)' "$SERVER_LOG"; then
    echo "fatal TokenSpeed server marker found" >&2
    exit 3
fi

stop_server
SERVER_STEP_PID=
echo "$TS_BENCH_MODE complete: $RUN_DIR"
