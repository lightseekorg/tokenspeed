#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
PREFILL_GPUS=${PREFILL_GPUS:-0,1}
DECODE_GPUS=${DECODE_GPUS:-2,3}
PREFILL_PORT=${PREFILL_PORT:-12346}
PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-8998}
DECODE_PORT=${DECODE_PORT:-12347}
LB_HOST=${LB_HOST:-0.0.0.0}
LB_PORT=${LB_PORT:-12345}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-18422}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
KVSTORE_RATIO=${KVSTORE_RATIO:-0.5}
WORLD_SIZE=${WORLD_SIZE:-2}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-trtllm}
DRAFTER_ATTENTION_BACKEND=${DRAFTER_ATTENTION_BACKEND:-$ATTENTION_BACKEND}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_trtllm}
QUANTIZATION=${QUANTIZATION:-nvfp4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
ENABLE_MTP=${ENABLE_MTP:-1}
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-16}
QUEUE_SIZE=${QUEUE_SIZE:-128}
LOG_DIR=${PD_CI_LOG_DIR:-.ci-artifacts/pd-qwen35-397b-1p1d}

export MC_INTRANODE_NVLINK=${MC_INTRANODE_NVLINK:-1}
export MC_INTRA_NVLINK=${MC_INTRA_NVLINK:-1}
export MC_LOG_LEVEL=${MC_LOG_LEVEL:-INFO}
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export NO_PROXY=${NO_PROXY:-*}
export no_proxy=${no_proxy:-*}
export TOKENSPEED_SKIP_GRPC_WARMUP=${TOKENSPEED_SKIP_GRPC_WARMUP:-1}

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
echo "[pd-1p1d] prefill=${PREFILL_GPUS}/${PREFILL_PORT}/${PREFILL_BOOTSTRAP_PORT} decode=${DECODE_GPUS}/${DECODE_PORT} lb=${LB_HOST}:${LB_PORT}"
echo "[pd-1p1d] world_size=$WORLD_SIZE enable_mtp=$ENABLE_MTP moe_backend=$MOE_BACKEND attention_backend=$ATTENTION_BACKEND"

pids=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    echo "[pd-1p1d] stopping ${#pids[@]} processes"
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

wait_http() {
  local name=$1
  local url=$2
  local timeout=${3:-1800}
  local start
  start=$(date +%s)
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( $(date +%s) - start > timeout )); then
      echo "[pd-1p1d] timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
  echo "[pd-1p1d] $name ready at $url"
}

COMMON_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 127.0.0.1
  --world-size "$WORLD_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --trust-remote-code
  --moe-backend "$MOE_BACKEND"
  --attention-backend "$ATTENTION_BACKEND"
  --load-format auto
  --comm-fusion-max-num-tokens 4096
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --quantization "$QUANTIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --kvstore-ratio "$KVSTORE_RATIO"
  --enable-cache-report
  --disaggregation-transfer-backend mooncake
  --disaggregation-layerwise-interval 1
  --skip-server-warmup
)

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

start_worker() {
  local role=$1
  local gpus=$2
  local port=$3
  local bootstrap_port=$4
  local log="$LOG_DIR/${role}.log"
  echo "[pd-1p1d] starting ${role}: gpus=$gpus port=$port bootstrap=$bootstrap_port log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    exec python3 test/ci_system/pd_http_worker.py \
      "${COMMON_ARGS[@]}" \
      --port "$port" \
      --disaggregation-bootstrap-port "$bootstrap_port" \
      --disaggregation-mode "$role"
  ) >"$log" 2>&1 &
  pids+=("$!")
}

start_worker prefill "$PREFILL_GPUS" "$PREFILL_PORT" "$PREFILL_BOOTSTRAP_PORT"
start_worker decode "$DECODE_GPUS" "$DECODE_PORT" "$PREFILL_BOOTSTRAP_PORT"

wait_http prefill "http://127.0.0.1:${PREFILL_PORT}/v1/models" 2400
wait_http decode "http://127.0.0.1:${DECODE_PORT}/v1/models" 2400

echo "[pd-1p1d] starting smg lb log=$LOG_DIR/lb.log"
python3 -m smg launch \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" "$PREFILL_BOOTSTRAP_PORT" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --host "$LB_HOST" \
  --port "$LB_PORT" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --disable-tokenizer-autoload \
  --policy round_robin \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
  --queue-size "$QUEUE_SIZE" \
  --queue-timeout-secs 1800 \
  --request-timeout-secs 1800 \
  --worker-startup-timeout-secs 1800 \
  --health-check-timeout-secs 60 \
  --health-check-interval-secs 30 \
  --log-level info \
  --disable-retries \
  --disable-circuit-breaker \
  --disable-health-check \
  --prometheus-port "$PROMETHEUS_PORT" \
  >"$LOG_DIR/lb.log" 2>&1 &
pids+=("$!")

wait_http lb "http://127.0.0.1:${LB_PORT}/v1/models" 600
echo "[pd-1p1d] serving on http://127.0.0.1:${LB_PORT}/v1"

wait -n "${pids[@]}"
