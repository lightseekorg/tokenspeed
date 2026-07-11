#!/usr/bin/env bash
# EPD (encode-prefill-decode) 1E-1P-2D smoke/eval topology on a single node.
#
# Serves nvidia/Qwen3.5-122B-A10B-NVFP4 (override via MODEL). EPD mode is
# gRPC-only (the SMG gateway refuses HTTP connection_mode for EPD), so the
# workers are gRPC servicers launched via `python3 -m smg_grpc_servicer.tokenspeed`
# (one unified entrypoint for all roles) rather than the HTTP pd_http_worker.py.
# The encode worker is LM-free: it runs only the vision tower and ships image
# embeddings to prefill over Mooncake; pixels reach it INLINE over gRPC (the
# default, no node-specific RDMA-listen config). Topology (4 GPUs, all TP1):
#   encode @ GPU0  +  prefill @ GPU1  +  decode0 @ GPU2  +  decode1 @ GPU3
# The two decode workers are independent instances; the gateway round-robins
# requests across them (--decode-policy round_robin).
set -euo pipefail

MODEL=${MODEL:-nvidia/Qwen3.5-122B-A10B-NVFP4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
ENCODE_GPUS=${ENCODE_GPUS:-0}
PREFILL_GPUS=${PREFILL_GPUS:-1}
DECODE0_GPUS=${DECODE0_GPUS:-2}
DECODE1_GPUS=${DECODE1_GPUS:-3}
ENCODE_WS=${ENCODE_WS:-1}
PREFILL_WS=${PREFILL_WS:-1}
DECODE0_WS=${DECODE0_WS:-1}
DECODE1_WS=${DECODE1_WS:-1}
ENCODE_PORT=${ENCODE_PORT:-50104}
PREFILL_PORT=${PREFILL_PORT:-50101}
DECODE0_PORT=${DECODE0_PORT:-50111}
DECODE1_PORT=${DECODE1_PORT:-50112}
ENCODE_BOOTSTRAP_PORT=${ENCODE_BOOTSTRAP_PORT:-18995}
PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-19311}
ENCODE_DIST_PORT=${ENCODE_DIST_PORT:-25000}
PREFILL_DIST_PORT=${PREFILL_DIST_PORT:-26000}
DECODE0_DIST_PORT=${DECODE0_DIST_PORT:-32000}
DECODE1_DIST_PORT=${DECODE1_DIST_PORT:-33000}
LB_HOST=${LB_HOST:-0.0.0.0}
LB_PORT=${LB_PORT:-12345}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-29080}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
QUANTIZATION=${QUANTIZATION:-nvfp4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-trtllm}
DRAFTER_ATTENTION_BACKEND=${DRAFTER_ATTENTION_BACKEND:-$ATTENTION_BACKEND}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_trtllm}
KVSTORE_RATIO=${KVSTORE_RATIO:-0.5}
ENABLE_MTP=${ENABLE_MTP:-0}
ENCODE_ROUTING_POLICY=${ENCODE_ROUTING_POLICY:-consistent_hashing}
LOG_DIR=${EPD_CI_LOG_DIR:-.ci-artifacts/epd-qwen35-122b-1e1p2d}

# The E->P embedding ships over Mooncake; pixels go inline over gRPC, sized by
# TOKENSPEED_GRPC_MAX_MESSAGE_BYTES.
export TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=${TOKENSPEED_GRPC_MAX_MESSAGE_BYTES:-2000000000}
export TOKENSPEED_SKIP_GRPC_WARMUP=${TOKENSPEED_SKIP_GRPC_WARMUP:-1}
export EPD_RECV_POOL_SLOT_MB=${EPD_RECV_POOL_SLOT_MB:-256}
export MC_INTRANODE_NVLINK=${MC_INTRANODE_NVLINK:-1}
export MC_INTRA_NVLINK=${MC_INTRA_NVLINK:-1}
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export NO_PROXY=${NO_PROXY:-*}
export no_proxy=${no_proxy:-*}

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
echo "[epd-1e1p2d] model=$MODEL served=$SERVED_MODEL_NAME model_path=$MODEL_PATH"
echo "[epd-1e1p2d] encode=gpu${ENCODE_GPUS}/${ENCODE_PORT} prefill=gpu${PREFILL_GPUS}/${PREFILL_PORT} decode0=gpu${DECODE0_GPUS}/${DECODE0_PORT} decode1=gpu${DECODE1_GPUS}/${DECODE1_PORT} lb=${LB_HOST}:${LB_PORT}"
echo "[epd-1e1p2d] encode_routing=$ENCODE_ROUTING_POLICY enable_mtp=$ENABLE_MTP moe=$MOE_BACKEND attn=$ATTENTION_BACKEND"

pids=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    echo "[epd-1e1p2d] stopping ${#pids[@]} processes"
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
      echo "[epd-1e1p2d] timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
  echo "[epd-1e1p2d] $name ready at $url"
}

wait_serving() {
  local label=$1
  local timeout=${2:-2400}
  local log="$LOG_DIR/${label}.log"
  local start
  start=$(date +%s)
  until grep -q "health status -> SERVING" "$log" 2>/dev/null; do
    if (( $(date +%s) - start > timeout )); then
      echo "[epd-1e1p2d] timed out waiting for $label to reach SERVING (log=$log)" >&2
      return 1
    fi
    sleep 5
  done
  echo "[epd-1e1p2d] $label SERVING"
}

# Shared engine args (mirrors serve_qwen35_397b_nvfp4_pd_1p1d.sh COMMON_ARGS).
COMMON_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
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

MTP_ARGS=()
if [[ "$ENABLE_MTP" == "1" ]]; then
  MTP_ARGS=(
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
  local label=$1            # log/display name (e.g. decode0); decouples replicas
  local mode=$2             # --disaggregation-mode workers from the replica label
  local gpus=$3
  local world_size=$4
  local port=$5
  local dist_port=$6
  local bootstrap_port=$7   # empty for decode
  shift 7
  local log="$LOG_DIR/${label}.log"
  # Speculative decoding is LM-only; keep the encode worker independent.
  local -a role_mtp_args=()
  if [[ "$mode" != "encode" ]]; then
    role_mtp_args=("${MTP_ARGS[@]}")
  fi
  echo "[epd-1e1p2d] starting ${label} (mode=$mode): gpus=$gpus ws=$world_size port=$port dist=$dist_port bootstrap=${bootstrap_port:-none} log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    exec env "$@" python3 -m smg_grpc_servicer.tokenspeed \
      "${COMMON_ARGS[@]}" \
      "${role_mtp_args[@]}" \
      --world-size "$world_size" \
      --disaggregation-mode "$mode" \
      ${bootstrap_port:+--disaggregation-bootstrap-port "$bootstrap_port"} \
      --dist-init-addr "127.0.0.1:${dist_port}" \
      --port "$port"
  ) >"$log" 2>&1 &
  pids+=("$!")
}

# Encode is LM-free (vision tower only); pixels reach it inline over gRPC.
start_worker encode encode "$ENCODE_GPUS" "$ENCODE_WS" "$ENCODE_PORT" "$ENCODE_DIST_PORT" "$ENCODE_BOOTSTRAP_PORT"
start_worker prefill prefill "$PREFILL_GPUS" "$PREFILL_WS" "$PREFILL_PORT" "$PREFILL_DIST_PORT" "$PREFILL_BOOTSTRAP_PORT"
start_worker decode0 decode "$DECODE0_GPUS" "$DECODE0_WS" "$DECODE0_PORT" "$DECODE0_DIST_PORT" ""
start_worker decode1 decode "$DECODE1_GPUS" "$DECODE1_WS" "$DECODE1_PORT" "$DECODE1_DIST_PORT" ""

# Gate on each worker's model-loaded "SERVING" health (registration != loaded).
wait_serving encode 2400
wait_serving prefill 2400
wait_serving decode0 2400
wait_serving decode1 2400
wait_http encode-bootstrap "http://127.0.0.1:${ENCODE_BOOTSTRAP_PORT}/health" 2400

echo "[epd-1e1p2d] starting smg gateway log=$LOG_DIR/gateway.log"
python3 -m smg launch \
  --epd-disaggregation \
  --encode "grpc://127.0.0.1:${ENCODE_PORT}" "$ENCODE_BOOTSTRAP_PORT" \
  --prefill "grpc://127.0.0.1:${PREFILL_PORT}" "$PREFILL_BOOTSTRAP_PORT" \
  --decode "grpc://127.0.0.1:${DECODE0_PORT}" \
  --decode "grpc://127.0.0.1:${DECODE1_PORT}" \
  --host "$LB_HOST" \
  --port "$LB_PORT" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --reasoning-parser passthrough \
  --encode-policy "$ENCODE_ROUTING_POLICY" \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --max-payload-size 2000000000 \
  --worker-startup-timeout-secs 1800 \
  --request-timeout-secs 1800 \
  --disable-circuit-breaker \
  --disable-health-check \
  --log-level info \
  --prometheus-port "$PROMETHEUS_PORT" \
  >"$LOG_DIR/gateway.log" 2>&1 &
pids+=("$!")

wait_http lb "http://127.0.0.1:${LB_PORT}/v1/models" 600
echo "[epd-1e1p2d] serving on http://127.0.0.1:${LB_PORT}/v1"

wait -n "${pids[@]}"
