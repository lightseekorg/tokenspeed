#!/usr/bin/env bash
# Qwen3.5 PD (prefill-decode) 1P-1D smoke/eval topology on a single node:
# headless prefill and decode engines behind SMG's gateway, both on its direct
# ZMQ lane. The gateway keeps the externally visible OpenAI-compatible HTTP API
# and takes the prefill engine's bootstrap address from its handshake.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/worker_cleanup.sh"
source "$SCRIPT_DIR/smg_lane.sh"

MODEL=${MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
PREFILL_GPUS=${PREFILL_GPUS:-0,1}
DECODE_GPUS=${DECODE_GPUS:-2,3}
# The engines' --port seeds their derived ports; they serve no HTTP.
PREFILL_PORT=${PREFILL_PORT:-18346}
PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-8998}
DECODE_PORT=${DECODE_PORT:-18347}
PREFILL_DIST_INIT_ADDR=${PREFILL_DIST_INIT_ADDR:-127.0.0.1:12579}
DECODE_DIST_INIT_ADDR=${DECODE_DIST_INIT_ADDR:-127.0.0.1:13580}
LB_HOST=${LB_HOST:-0.0.0.0}
LB_PORT=${LB_PORT:-18345}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-18422}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
WORLD_SIZE=${WORLD_SIZE:-2}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-trtllm}
DRAFTER_ATTENTION_BACKEND=${DRAFTER_ATTENTION_BACKEND:-$ATTENTION_BACKEND}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_trtllm}
QUANTIZATION=${QUANTIZATION:-nvfp4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
ENABLE_MTP=${ENABLE_MTP:-0}
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-16}
QUEUE_SIZE=${QUEUE_SIZE:-128}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2400}
LOG_DIR=${PD_CI_LOG_DIR:-.ci-artifacts/pd-qwen35-397b-1p1d}

export MC_INTRANODE_NVLINK=${MC_INTRANODE_NVLINK:-1}
export MC_INTRA_NVLINK=${MC_INTRA_NVLINK:-1}
export MC_LOG_LEVEL=${MC_LOG_LEVEL:-INFO}
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export NO_PROXY=${NO_PROXY:-*}
export no_proxy=${no_proxy:-*}

IFS=',' read -r -a PREFILL_GPU_LIST <<< "$PREFILL_GPUS"
IFS=',' read -r -a DECODE_GPU_LIST <<< "$DECODE_GPUS"

if [[ ${#PREFILL_GPU_LIST[@]} -ne $WORLD_SIZE ]]; then
  echo "PREFILL_GPUS must contain WORLD_SIZE=$WORLD_SIZE comma-separated GPU ids" >&2
  exit 2
fi
if [[ ${#DECODE_GPU_LIST[@]} -ne $WORLD_SIZE ]]; then
  echo "DECODE_GPUS must contain WORLD_SIZE=$WORLD_SIZE comma-separated GPU ids" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

resolve_model_snapshot() {
  python3 - "$MODEL" <<'PYSNAPSHOT'
import os
import sys
from pathlib import Path
model = sys.argv[1]
if os.path.isdir(model):
    print(str(Path(model).resolve()))
    raise SystemExit(0)
from huggingface_hub import snapshot_download
patterns = [
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.json',
    'chat_template.jinja',
    'preprocessor_config.json',
    'processor_config.json',
    'video_preprocessor_config.json',
]
print(snapshot_download(model, allow_patterns=patterns), flush=True)
PYSNAPSHOT
}

MODEL_PATH=${MODEL_PATH:-$(resolve_model_snapshot)}
echo "[pd-1p1d] model=$MODEL served_model_name=$SERVED_MODEL_NAME model_path=$MODEL_PATH"
echo "[pd-1p1d] prefill=${PREFILL_GPUS}/${PREFILL_PORT}/${PREFILL_BOOTSTRAP_PORT}/${PREFILL_DIST_INIT_ADDR} decode=${DECODE_GPUS}/${DECODE_PORT}/${DECODE_DIST_INIT_ADDR} lb=${LB_HOST}:${LB_PORT}"
echo "[pd-1p1d] world_size=$WORLD_SIZE enable_mtp=$ENABLE_MTP moe_backend=$MOE_BACKEND attention_backend=$ATTENTION_BACKEND"

SOCKET_DIR=$(lane_socket_dir)
PREFILL_URL="ipc://$SOCKET_DIR/prefill"
DECODE_URL="ipc://$SOCKET_DIR/decode"

pids=()
logs=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    stop_worker_pids \
      "pd-1p1d" "${WORKER_SHUTDOWN_TIMEOUT:-30}" "${pids[@]}"
  fi
  rm -rf "$SOCKET_DIR"
  exit "$code"
}
trap cleanup EXIT INT TERM

COMMON_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 127.0.0.1
  --world-size "$WORLD_SIZE"
  --trust-remote-code
  --moe-backend "$MOE_BACKEND"
  --attention-backend "$ATTENTION_BACKEND"
  --load-format auto
  --comm-fusion-max-num-tokens 4096
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --quantization "$QUANTIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --disable-kvstore
  --disaggregation-transfer-backend mooncake
  --disaggregation-layerwise-interval 0
)
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
  COMMON_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
  COMMON_ARGS+=(
    --speculative-algorithm MTP
    --speculative-draft-model-path "$MODEL"
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --drafter-attention-backend "$DRAFTER_ATTENTION_BACKEND"
  )
elif [[ "$ENABLE_MTP" != "0" ]]; then
  echo "ENABLE_MTP must be 0 or 1" >&2
  exit 2
fi

ALIAS_ARGS=()
if [[ "$SERVED_MODEL_NAME" != "$MODEL_PATH" ]]; then
  ALIAS_ARGS=(--model-alias "$SERVED_MODEL_NAME=$MODEL_PATH")
fi

# The gateway binds both workers' sockets; the engines dial in once loaded.
echo "[pd-1p1d] starting smg lb log=$LOG_DIR/lb.log"
python3 -m smg launch \
  --pd-disaggregation \
  --backend tokenspeed \
  --prefill "$PREFILL_URL" \
  --decode "$DECODE_URL" \
  --host "$LB_HOST" \
  --port "$LB_PORT" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  "${ALIAS_ARGS[@]}" \
  --reasoning-parser passthrough \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
  --queue-size "$QUEUE_SIZE" \
  --queue-timeout-secs 1800 \
  --request-timeout-secs 1800 \
  --log-level info \
  --disable-retries \
  --disable-circuit-breaker \
  --prometheus-port "$PROMETHEUS_PORT" \
  >"$LOG_DIR/lb.log" 2>&1 &
pids+=("$!")
logs+=("$LOG_DIR/lb.log")
startup_deadline=$((SECONDS + STARTUP_TIMEOUT))
lane_wait_http pd-1p1d lb "http://127.0.0.1:${LB_PORT}/liveness" "$startup_deadline" "${pids[@]}"

start_worker() {
  local role=$1
  local gpus=$2
  local port=$3
  local bootstrap_port=$4
  local dist_init_addr=$5
  local url=$6
  local log="$LOG_DIR/${role}.log"
  local -a dial
  read -r -a dial <<< "$(lane_dial_args 127.0.0.1 "$(lane_handshake_port "$url")")"
  echo "[pd-1p1d] starting ${role}: gpus=$gpus worker=$url bootstrap=${bootstrap_port:-none} log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    exec python3 -m tokenspeed.runtime.entrypoints.engine \
      "${COMMON_ARGS[@]}" \
      --port "$port" \
      --dist-init-addr "$dist_init_addr" \
      ${bootstrap_port:+--disaggregation-bootstrap-port "$bootstrap_port"} \
      --disaggregation-mode "$role" \
      "${dial[@]}"
  ) >"$log" 2>&1 &
  pids+=("$!")
  logs+=("$log")
}

# Each engine reserves a small control-plane port cluster around its
# rendezvous address. Keep the P/D clusters disjoint while loading in parallel.
start_worker prefill "$PREFILL_GPUS" "$PREFILL_PORT" "$PREFILL_BOOTSTRAP_PORT" "$PREFILL_DIST_INIT_ADDR" "$PREFILL_URL"
start_worker decode "$DECODE_GPUS" "$DECODE_PORT" "" "$DECODE_DIST_INIT_ADDR" "$DECODE_URL"

# Ready once both engines registered.
lane_wait_http pd-1p1d engines "http://127.0.0.1:${LB_PORT}/readiness" "$startup_deadline" "${pids[@]}"
echo "[pd-1p1d] serving on http://127.0.0.1:${LB_PORT}/v1"

# A worker's failure status must not end the script (set -e) before the
# report names it; the cleanup trap then exits with that status.
wait -n "${pids[@]}" || status=$?
report_exited_workers "pd-1p1d" logs pids
exit "${status:-0}"
