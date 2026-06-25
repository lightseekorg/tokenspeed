#!/usr/bin/env bash
# EPD (encode-prefill-decode) 1E-1P-1D smoke/eval topology on a single node.
#
# Mirrors serve_qwen35_397b_nvfp4_pd_1p1d.sh (same model + engine args), but EPD
# mode is gRPC-only (the SMG gateway refuses HTTP connection_mode for EPD), so the
# workers are gRPC servicers launched via `python3 -m smg_grpc_servicer.tokenspeed`
# (one unified entrypoint for all roles) rather than the HTTP pd_http_worker.py.
# The encode worker is LM-free: it runs only the vision tower and ships image
# embeddings to prefill over Mooncake; pixels reach it INLINE over gRPC (the
# default, no node-specific RDMA-listen config). Topology (5 GPUs):
#   encode @ GPU0 (world-size 1)  +  prefill @ GPU1,2 (ws 2)  +  decode @ GPU3,4 (ws 2)
set -uo pipefail

MODEL=${MODEL:-nvidia/Qwen3.5-397B-A17B-NVFP4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
ENCODE_GPUS=${ENCODE_GPUS:-0}
PREFILL_GPUS=${PREFILL_GPUS:-1,2}
DECODE_GPUS=${DECODE_GPUS:-3,4}
ENCODE_WS=${ENCODE_WS:-1}
PREFILL_WS=${PREFILL_WS:-2}
DECODE_WS=${DECODE_WS:-2}
ENCODE_PORT=${ENCODE_PORT:-50104}
PREFILL_PORT=${PREFILL_PORT:-50101}
DECODE_PORT=${DECODE_PORT:-50111}
ENCODE_BOOTSTRAP_PORT=${ENCODE_BOOTSTRAP_PORT:-18995}
PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-19311}
ENCODE_DIST_PORT=${ENCODE_DIST_PORT:-25000}
PREFILL_DIST_PORT=${PREFILL_DIST_PORT:-26000}
DECODE_DIST_PORT=${DECODE_DIST_PORT:-32000}
LB_HOST=${LB_HOST:-0.0.0.0}
LB_PORT=${LB_PORT:-12345}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-29080}
NIC=${NIC:-mlx5_8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
QUANTIZATION=${QUANTIZATION:-nvfp4}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-trtllm}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_trtllm}
KVSTORE_RATIO=${KVSTORE_RATIO:-0.5}
ENCODE_ROUTING_POLICY=${SMG_ENCODE_ROUTING_POLICY:-cache_affinity}
LOG_DIR=${EPD_CI_LOG_DIR:-.ci-artifacts/epd-qwen35-397b-1p1d1e}

# RDMA / Mooncake transport. The E->P embedding ships over Mooncake (the runner
# must expose an RDMA NIC, like the PD CI does); pixels go inline over gRPC, sized
# by TOKENSPEED_GRPC_MAX_MESSAGE_BYTES. Set GATEWAY_PIXEL_RDMA=1 + ENCODE_EXTRA_ENV
# + SMG_RDMA_LISTEN_* for the faster gateway->encode RDMA pixel path.
export TOKENSPEED_GRPC_MAX_MESSAGE_BYTES=${TOKENSPEED_GRPC_MAX_MESSAGE_BYTES:-2000000000}
export TOKENSPEED_SKIP_GRPC_WARMUP=${TOKENSPEED_SKIP_GRPC_WARMUP:-1}
export EPD_RECV_POOL_SLOT_MB=${EPD_RECV_POOL_SLOT_MB:-256}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-${NIC}:1}
export UCX_TLS=${UCX_TLS:-rc,sm,self}
export UCX_IB_GID_INDEX=${UCX_IB_GID_INDEX:-3}
export MC_INTRANODE_NVLINK=${MC_INTRANODE_NVLINK:-1}
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
echo "[epd-1p1d1e] model=$MODEL served=$SERVED_MODEL_NAME model_path=$MODEL_PATH"
echo "[epd-1p1d1e] encode=gpu${ENCODE_GPUS}/ws${ENCODE_WS}/${ENCODE_PORT} prefill=gpu${PREFILL_GPUS}/ws${PREFILL_WS}/${PREFILL_PORT} decode=gpu${DECODE_GPUS}/ws${DECODE_WS}/${DECODE_PORT} lb=${LB_HOST}:${LB_PORT}"
echo "[epd-1p1d1e] nic=$NIC encode_routing=$ENCODE_ROUTING_POLICY moe=$MOE_BACKEND attn=$ATTENTION_BACKEND"

pids=()
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    echo "[epd-1p1d1e] stopping ${#pids[@]} processes"
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
      echo "[epd-1p1d1e] timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
  echo "[epd-1p1d1e] $name ready at $url"
}

# Shared engine args (mirrors serve_qwen35_397b_nvfp4_pd_1p1d.sh COMMON_ARGS).
# Passed to all three roles; the encode worker ignores the LM-only knobs. MTP is
# intentionally off -- it is orthogonal to the EPD embedding path under test.
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
  --disaggregation-ib-device "$NIC"
  --skip-server-warmup
)

start_worker() {
  local role=$1
  local gpus=$2
  local world_size=$3
  local port=$4
  local dist_port=$5
  local bootstrap_port=$6   # empty for decode
  shift 6
  local log="$LOG_DIR/${role}.log"
  echo "[epd-1p1d1e] starting ${role}: gpus=$gpus ws=$world_size port=$port dist=$dist_port bootstrap=${bootstrap_port:-none} log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    exec env "$@" python3 -m smg_grpc_servicer.tokenspeed \
      "${COMMON_ARGS[@]}" \
      --world-size "$world_size" \
      --disaggregation-mode "$role" \
      ${bootstrap_port:+--disaggregation-bootstrap-port "$bootstrap_port"} \
      --dist-init-addr "127.0.0.1:${dist_port}" \
      --port "$port"
  ) >"$log" 2>&1 &
  pids+=("$!")
}

# Encode is LM-free (vision tower only); pixels reach it inline over gRPC by
# default. For the faster gateway->encode RDMA pixel path used in the perf
# benches, export before running:
#   ENCODE_EXTRA_ENV="SMG_MM_PIXEL_RDMA=1 EPD_INGEST_OFFLOOP=1"
#   GATEWAY_PIXEL_RDMA=1 SMG_RDMA_LISTEN_IP=<nic-ip> SMG_RDMA_LISTEN_PORT=29900 \
#     SMG_RDMA_SLOT_BYTES=67108864
ENCODE_EXTRA_ENV=${ENCODE_EXTRA_ENV:-}
start_worker encode "$ENCODE_GPUS" "$ENCODE_WS" "$ENCODE_PORT" "$ENCODE_DIST_PORT" "$ENCODE_BOOTSTRAP_PORT" \
  $ENCODE_EXTRA_ENV
start_worker prefill "$PREFILL_GPUS" "$PREFILL_WS" "$PREFILL_PORT" "$PREFILL_DIST_PORT" "$PREFILL_BOOTSTRAP_PORT"
start_worker decode "$DECODE_GPUS" "$DECODE_WS" "$DECODE_PORT" "$DECODE_DIST_PORT" ""

# gRPC workers do not serve HTTP; the encode bootstrap server does. The gateway
# (worker-startup-timeout) waits for all three to register, so gating on the
# gateway /v1/models is sufficient for readiness.
wait_http encode-bootstrap "http://127.0.0.1:${ENCODE_BOOTSTRAP_PORT}/health" 2400

echo "[epd-1p1d1e] starting smg gateway log=$LOG_DIR/gateway.log"
# GATEWAY_PIXEL_RDMA=1 switches the gateway->encode pixel path to RDMA (requires
# SMG_RDMA_LISTEN_IP/PORT/SLOT_BYTES in the env); default is inline-over-gRPC.
SMG_MM_PIXEL_RDMA=${GATEWAY_PIXEL_RDMA:-0} \
SMG_ENCODE_ROUTING_POLICY="$ENCODE_ROUTING_POLICY" \
python3 -m smg launch \
  --epd-disaggregation \
  --encode "grpc://127.0.0.1:${ENCODE_PORT}" "$ENCODE_BOOTSTRAP_PORT" \
  --prefill "grpc://127.0.0.1:${PREFILL_PORT}" "$PREFILL_BOOTSTRAP_PORT" \
  --decode "grpc://127.0.0.1:${DECODE_PORT}" \
  --host "$LB_HOST" \
  --port "$LB_PORT" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --disable-tokenizer-autoload \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --max-payload-size 2000000000 \
  --worker-startup-timeout-secs 1800 \
  --request-timeout-secs 1800 \
  --disable-retries \
  --disable-circuit-breaker \
  --disable-health-check \
  --log-level info \
  --prometheus-port "$PROMETHEUS_PORT" \
  >"$LOG_DIR/gateway.log" 2>&1 &
pids+=("$!")

wait_http lb "http://127.0.0.1:${LB_PORT}/v1/models" 600
echo "[epd-1p1d1e] serving on http://127.0.0.1:${LB_PORT}/v1"

wait -n "${pids[@]}"
