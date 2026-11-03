#!/usr/bin/env bash
# EPD (encode-prefill-decode) 1E-1P-2D smoke/eval topology on a single node.
#
# Serves nvidia/Qwen3.5-122B-A10B-NVFP4 (override via MODEL): headless
# TokenSpeed engines behind SMG's gateway, all on its direct ZMQ lane. The
# encode server is LM-free: it runs only the vision tower and ships image
# embeddings to prefill over Mooncake; the gateway sends it the pixels inline
# as encode requests. The gateway takes the encode and prefill engines'
# bootstrap addresses from their handshakes. Topology (4 GPUs, all TP1):
#   encode @ GPU0  +  prefill @ GPU1  +  decode0 @ GPU2  +  decode1 @ GPU3
# The two decode workers are independent instances; the gateway round-robins
# requests across them (--decode-policy round_robin).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/worker_cleanup.sh"
source "$SCRIPT_DIR/smg_lane.sh"

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
# The engines' --port seeds their derived ports; they serve no HTTP.
ENCODE_PORT=${ENCODE_PORT:-15004}
PREFILL_PORT=${PREFILL_PORT:-15001}
DECODE0_PORT=${DECODE0_PORT:-15011}
DECODE1_PORT=${DECODE1_PORT:-15012}
ENCODE_BOOTSTRAP_PORT=${ENCODE_BOOTSTRAP_PORT:-18995}
PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-19311}
ENCODE_DIST_PORT=${ENCODE_DIST_PORT:-25000}
PREFILL_DIST_PORT=${PREFILL_DIST_PORT:-26000}
DECODE0_DIST_PORT=${DECODE0_DIST_PORT:-32000}
DECODE1_DIST_PORT=${DECODE1_DIST_PORT:-33000}
LB_HOST=${LB_HOST:-0.0.0.0}
LB_PORT=${LB_PORT:-19345}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-29080}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
QUANTIZATION=${QUANTIZATION:-nvfp4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-trtllm}
DRAFTER_ATTENTION_BACKEND=${DRAFTER_ATTENTION_BACKEND:-$ATTENTION_BACKEND}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_trtllm}
ENABLE_MTP=${ENABLE_MTP:-0}
ENCODE_ROUTING_POLICY=${ENCODE_ROUTING_POLICY:-consistent_hashing}
LOG_DIR=${EPD_CI_LOG_DIR:-.ci-artifacts/epd-qwen35-122b-1e1p2d}
# All startup stages share one two-hour budget.
startup_deadline=$((SECONDS + ${EPD_STARTUP_TIMEOUT:-7200}))

# The E->P embedding ships over Mooncake.
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

SOCKET_DIR=$(lane_socket_dir)
ENCODE_URL="ipc://$SOCKET_DIR/encode"
PREFILL_URL="ipc://$SOCKET_DIR/prefill"
DECODE0_URL="ipc://$SOCKET_DIR/decode0"
DECODE1_URL="ipc://$SOCKET_DIR/decode1"

pids=()
logs=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    stop_worker_pids \
      "epd-1e1p2d" "${WORKER_SHUTDOWN_TIMEOUT:-30}" "${pids[@]}"
  fi
  rm -rf "$SOCKET_DIR"
  exit "$code"
}
trap cleanup EXIT INT TERM

# Shared engine args (mirrors serve_qwen35_397b_nvfp4_pd_1p1d.sh COMMON_ARGS).
COMMON_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 127.0.0.1
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

ALIAS_ARGS=()
if [[ "$SERVED_MODEL_NAME" != "$MODEL_PATH" ]]; then
  ALIAS_ARGS=(--model-alias "$SERVED_MODEL_NAME=$MODEL_PATH")
fi

# The gateway binds every worker's sockets; the engines dial in once loaded.
echo "[epd-1e1p2d] starting smg gateway log=$LOG_DIR/gateway.log"
python3 -m smg launch \
  --epd-disaggregation \
  --backend tokenspeed \
  --encode "$ENCODE_URL" \
  --prefill "$PREFILL_URL" \
  --decode "$DECODE0_URL" \
  --decode "$DECODE1_URL" \
  --host "$LB_HOST" \
  --port "$LB_PORT" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  "${ALIAS_ARGS[@]}" \
  --reasoning-parser passthrough \
  --encode-policy "$ENCODE_ROUTING_POLICY" \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --request-timeout-secs 1800 \
  --disable-circuit-breaker \
  --log-level info \
  --prometheus-port "$PROMETHEUS_PORT" \
  >"$LOG_DIR/gateway.log" 2>&1 &
pids+=("$!")
logs+=("$LOG_DIR/gateway.log")
lane_wait_http epd-1e1p2d lb "http://127.0.0.1:${LB_PORT}/liveness" "$startup_deadline" "${pids[@]}"

start_worker() {
  local label=$1            # log/display name (e.g. decode0); decouples replicas
  local mode=$2             # --disaggregation-mode workers from the replica label
  local gpus=$3
  local world_size=$4
  local port=$5
  local dist_port=$6
  local bootstrap_port=$7   # empty for decode
  local url=$8
  local log="$LOG_DIR/${label}.log"
  local -a dial
  read -r -a dial <<< "$(lane_dial_args 127.0.0.1 "$(lane_handshake_port "$url")")"
  # Speculative decoding is LM-only; keep the encode worker independent.
  local -a role_mtp_args=()
  if [[ "$mode" != "encode" ]]; then
    role_mtp_args=("${MTP_ARGS[@]}")
  fi
  echo "[epd-1e1p2d] starting ${label} (mode=$mode): gpus=$gpus ws=$world_size worker=$url dist=$dist_port bootstrap=${bootstrap_port:-none} log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    exec python3 -m tokenspeed.runtime.entrypoints.engine \
      "${COMMON_ARGS[@]}" \
      "${role_mtp_args[@]}" \
      --world-size "$world_size" \
      --disaggregation-mode "$mode" \
      ${bootstrap_port:+--disaggregation-bootstrap-port "$bootstrap_port"} \
      --dist-init-addr "127.0.0.1:${dist_port}" \
      --port "$port" \
      "${dial[@]}"
  ) >"$log" 2>&1 &
  pids+=("$!")
  logs+=("$log")
}

# Encode is LM-free (vision tower only); its first rank dials the gateway and
# takes the encode requests.
start_worker encode encode "$ENCODE_GPUS" "$ENCODE_WS" "$ENCODE_PORT" "$ENCODE_DIST_PORT" "$ENCODE_BOOTSTRAP_PORT" "$ENCODE_URL"
start_worker prefill prefill "$PREFILL_GPUS" "$PREFILL_WS" "$PREFILL_PORT" "$PREFILL_DIST_PORT" "$PREFILL_BOOTSTRAP_PORT" "$PREFILL_URL"
start_worker decode0 decode "$DECODE0_GPUS" "$DECODE0_WS" "$DECODE0_PORT" "$DECODE0_DIST_PORT" "" "$DECODE0_URL"
start_worker decode1 decode "$DECODE1_GPUS" "$DECODE1_WS" "$DECODE1_PORT" "$DECODE1_DIST_PORT" "" "$DECODE1_URL"

# Ready once every engine registered.
lane_wait_http epd-1e1p2d engines "http://127.0.0.1:${LB_PORT}/readiness" "$startup_deadline" "${pids[@]}"
echo "[epd-1e1p2d] serving on http://127.0.0.1:${LB_PORT}/v1"

# A worker's failure status must not end the script (set -e) before the
# report names it; the cleanup trap then exits with that status.
wait -n "${pids[@]}" || status=$?
report_exited_workers "epd-1e1p2d" logs pids
exit "${status:-0}"
