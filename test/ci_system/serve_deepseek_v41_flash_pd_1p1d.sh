#!/usr/bin/env bash
# DeepSeek V4.1 Flash PD (prefill-decode) 1P-1D topology, with
# DSpark on both roles. Workers use the unified TokenSpeed gRPC servicer; the
# SMG gateway keeps the externally visible OpenAI-compatible HTTP API.
# PD_SLURM=1 places prefill + gateway on Slurm node 0 and decode on node 1.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/worker_cleanup.sh"

MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL}
PREFILL_GPUS=${PREFILL_GPUS:-0,1}
DECODE_GPUS=${DECODE_GPUS:-2,3}
PREFILL_PORT=${PREFILL_PORT:-18346}
PREFILL_BOOTSTRAP_PORT=${PREFILL_BOOTSTRAP_PORT:-8998}
DECODE_PORT=${DECODE_PORT:-18347}
PREFILL_DIST_INIT_ADDR=${PREFILL_DIST_INIT_ADDR:-127.0.0.1:12579}
DECODE_DIST_INIT_ADDR=${DECODE_DIST_INIT_ADDR:-127.0.0.1:13580}
LB_HOST=${LB_HOST:-0.0.0.0}
LB_PORT=${LB_PORT:-18345}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-18422}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
WORLD_SIZE=${WORLD_SIZE:-2}
MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
REASONING_PARSER=${REASONING_PARSER:-passthrough}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2400}
PD_SLURM=${PD_SLURM:-0}
# mega_moe (Blackwell) needs expert parallelism; marlin (Hopper) runs plain TP.
MOE_BACKEND=${MOE_BACKEND:-mega_moe}
ENABLE_DSPARK=${ENABLE_DSPARK:-1}
# The decode role keeps prefix caching off: its sliding-window groups land
# only their retained tail, so local prefix probes never match anyway.
DECODE_PREFIX_CACHE=${DECODE_PREFIX_CACHE:-0}
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-16}
QUEUE_SIZE=${QUEUE_SIZE:-128}
LOG_DIR=${PD_CI_LOG_DIR:-.ci-artifacts/pd-deepseek-v41-flash-1p1d}
ROLE=both
WORKER_HOST=127.0.0.1
PREFILL_HOST=127.0.0.1
DECODE_HOST=127.0.0.1
READY_FILE=

if [[ "$PD_SLURM" == 1 ]]; then
  if [[ ${SLURM_STEP_NUM_NODES:-} != 2 || ! ${SLURM_NODEID:-} =~ ^[01]$ ]]; then
    echo "PD_SLURM=1 requires exactly two Slurm nodes (node ids 0 and 1)" >&2
    exit 2
  fi
  : "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}"
  ROLE=prefill
  [[ $SLURM_NODEID == 0 ]] || ROLE=decode
  LOG_DIR="$LOG_DIR/$SLURM_JOB_ID-$SLURM_STEP_ID"
  READY_FILE="$LOG_DIR/$ROLE.ready"
  # Use the same address selection as Mooncake's engine and rank endpoints.
  WORKER_HOST=$(python3 - <<'PYIP'
import ipaddress
from tokenspeed.runtime.utils.network import get_local_ip_by_remote
address = ipaddress.IPv4Address(get_local_ip_by_remote())
if address.is_loopback or address.is_unspecified:
    raise ValueError(f"PD requires a routable worker address, got {address}")
print(address)
PYIP
  )
  PREFILL_DIST_INIT_ADDR="$WORKER_HOST:12579"
  DECODE_DIST_INIT_ADDR="$WORKER_HOST:13580"
elif [[ "$PD_SLURM" != 0 ]]; then
  echo "PD_SLURM must be 0 or 1" >&2
  exit 2
fi

# Mooncake picks RDMA (or TCP) on its own. Do not force the intra-node NVLink
# transport here: on containerized hosts it has reported success without
# writing the destination pages.
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
if [[ -n "$READY_FILE" ]]; then
  rm -f "$READY_FILE"
fi

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
    'chat_template.jinja',
    'encoding/*',
]
print(snapshot_download(model, allow_patterns=patterns), flush=True)
PYSNAPSHOT
}

MODEL_PATH=${MODEL_PATH:-$(resolve_model_snapshot)}
echo "[pd-1p1d] model=$MODEL served_model_name=$SERVED_MODEL_NAME model_path=$MODEL_PATH"
echo "[pd-1p1d] prefill=${PREFILL_GPUS}/${PREFILL_PORT}/${PREFILL_BOOTSTRAP_PORT}/${PREFILL_DIST_INIT_ADDR} decode=${DECODE_GPUS}/${DECODE_PORT}/${DECODE_DIST_INIT_ADDR} lb=${LB_HOST}:${LB_PORT}"
echo "[pd-1p1d] world_size=$WORLD_SIZE moe_backend=$MOE_BACKEND enable_dspark=$ENABLE_DSPARK decode_prefix_cache=$DECODE_PREFIX_CACHE"

pids=()
cleanup() {
  local code=$?
  local log
  trap - EXIT INT TERM
  if [[ -n "$READY_FILE" ]]; then
    rm -f "$READY_FILE" "$READY_FILE.tmp"
  fi
  if ((code != 0)); then
    local -a logs=("$LOG_DIR/$ROLE.log")
    if [[ "$ROLE" == both ]]; then
      logs=("$LOG_DIR/prefill.log" "$LOG_DIR/decode.log")
    fi
    [[ "$ROLE" == decode ]] || logs+=("$LOG_DIR/lb.log")
    for log in "${logs[@]}"; do
      [[ -f "$log" ]] && tail -n 100 "$log" >&2
    done
  fi
  if ((${#pids[@]})); then
    stop_worker_pids \
      "pd-1p1d" "${WORKER_SHUTDOWN_TIMEOUT:-30}" "${pids[@]}"
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

check_workers() {
  local pid
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[pd-1p1d] worker $pid exited unexpectedly" >&2
      return 1
    fi
  done
}

wait_http() {
  local name=$1
  local url=$2
  local timeout=$3
  local start
  start=$(date +%s)
  until curl --connect-timeout 2 --max-time 5 -fsS "$url" >/dev/null 2>&1; do
    check_workers
    if (( $(date +%s) - start > timeout )); then
      echo "[pd-1p1d] timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
  echo "[pd-1p1d] $name ready at $url"
}

wait_serving() {
  local role=$1
  local pid=$2
  local deadline=$3
  local log="$LOG_DIR/${role}.log"
  until grep -q "health status -> SERVING" "$log" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[pd-1p1d] $role exited before reaching SERVING (log=$log)" >&2
      tail -n 200 "$log" >&2 || true
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "[pd-1p1d] timed out waiting for $role to reach SERVING (log=$log)" >&2
      tail -n 200 "$log" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[pd-1p1d] $role SERVING"
}

COMMON_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$WORKER_HOST"
  --world-size "$WORLD_SIZE"
  # Keep attention TP equal to this role's world size even with expert parallelism.
  --tensor-parallel-size "$WORLD_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --trust-remote-code
  --moe-backend "$MOE_BACKEND"
  --dtype bfloat16
  --load-format auto
  --comm-fusion-max-num-tokens 4096
  --max-model-len "$MAX_MODEL_LEN"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --chunked-prefill-size 8192
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
  --disable-prefill-graph
  --disable-kvstore
  # The two FP8 Engram tables do not fit beside the weights on a TP2/TP4
  # split; the text-only path is what this smoke test exercises.
  --engram-host-table
  --language-model-only
  --disaggregation-transfer-backend mooncake
  --disaggregation-layerwise-interval 0
)

if [[ "$MOE_BACKEND" == "mega_moe" ]]; then
  COMMON_ARGS+=(--enable-expert-parallel)
fi
if [[ -n ${DISAGGREGATION_IB_DEVICE:-} ]]; then
  COMMON_ARGS+=(--disaggregation-ib-device "$DISAGGREGATION_IB_DEVICE")
fi

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  # Same-checkpoint draft; the block width comes from the checkpoint.
  COMMON_ARGS+=(--speculative-algorithm DSPARK)
elif [[ "$ENABLE_DSPARK" != "0" ]]; then
  echo "ENABLE_DSPARK must be 0 or 1" >&2
  exit 2
fi

DECODE_ARGS=()
if [[ "$DECODE_PREFIX_CACHE" == "0" ]]; then
  DECODE_ARGS+=(--disable-prefix-caching)
elif [[ "$DECODE_PREFIX_CACHE" != "1" ]]; then
  echo "DECODE_PREFIX_CACHE must be 0 or 1" >&2
  exit 2
fi

start_worker() {
  local role=$1
  local gpus=$2
  local port=$3
  local bootstrap_port=$4
  local dist_init_addr=$5
  shift 5
  local log="$LOG_DIR/${role}.log"
  echo "[pd-1p1d] starting ${role}: gpus=$gpus port=$port bootstrap=${bootstrap_port:-none} log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    if [[ "$PD_SLURM" == 1 ]]; then
      # P and D are independent engines, each wholly within its own node.
      # Keep the outer pipeline's Slurm environment for lifecycle management.
      unset SLURM_STEP_NUM_NODES SLURM_NODEID SLURM_STEP_NODELIST
    fi
    exec python3 -m smg_grpc_servicer.tokenspeed \
      "${COMMON_ARGS[@]}" \
      --port "$port" \
      --dist-init-addr "$dist_init_addr" \
      ${bootstrap_port:+--disaggregation-bootstrap-port "$bootstrap_port"} \
      --disaggregation-mode "$role" \
      "$@"
  ) >"$log" 2>&1 &
  pids+=("$!")
}

# Each engine reserves a small control-plane port cluster around its
# rendezvous address. Keep the P/D clusters disjoint while loading in parallel.
startup_deadline=$((SECONDS + STARTUP_TIMEOUT))
if [[ "$ROLE" != decode ]]; then
  start_worker prefill "$PREFILL_GPUS" "$PREFILL_PORT" "$PREFILL_BOOTSTRAP_PORT" "$PREFILL_DIST_INIT_ADDR"
fi
if [[ "$ROLE" != prefill ]]; then
  start_worker decode "$DECODE_GPUS" "$DECODE_PORT" "" "$DECODE_DIST_INIT_ADDR" "${DECODE_ARGS[@]}"
fi

if [[ "$ROLE" == both ]]; then
  wait_serving prefill "${pids[0]}" "$startup_deadline"
  wait_serving decode "${pids[1]}" "$startup_deadline"
else
  wait_serving "$ROLE" "${pids[0]}" "$startup_deadline"
  printf '%s\n' "$WORKER_HOST" > "$READY_FILE.tmp"
  mv "$READY_FILE.tmp" "$READY_FILE"
  if [[ "$ROLE" == decode ]]; then
    wait "${pids[0]}"
    # A worker exiting successfully is still a failed serving job.
    exit 1
  fi
  until [[ -s "$LOG_DIR/decode.ready" ]]; do
    check_workers
    if ((SECONDS >= startup_deadline)); then
      echo "[pd-1p1d] timed out waiting for decode node" >&2
      exit 1
    fi
    sleep 5
  done
  PREFILL_HOST=$WORKER_HOST
  DECODE_HOST=$(cat "$LOG_DIR/decode.ready")
  # Validate peer reachability, not just the shared readiness marker.
  python3 - "$PREFILL_HOST:$PREFILL_PORT" "$DECODE_HOST:$DECODE_PORT" <<'PYHEALTH'
import asyncio
import sys
from tokenspeed.cli._proc import wait_grpc_serving
async def main():
    await asyncio.gather(*(wait_grpc_serving(target, timeout=60, poll_interval=1)
                           for target in sys.argv[1:]))
asyncio.run(main())
PYHEALTH
fi

echo "[pd-1p1d] starting smg lb log=$LOG_DIR/lb.log"
python3 -m smg launch \
  --pd-disaggregation \
  --prefill "grpc://${PREFILL_HOST}:${PREFILL_PORT}" "$PREFILL_BOOTSTRAP_PORT" \
  --decode "grpc://${DECODE_HOST}:${DECODE_PORT}" \
  --host "$LB_HOST" \
  --port "$LB_PORT" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --reasoning-parser "$REASONING_PARSER" \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
  --queue-size "$QUEUE_SIZE" \
  --queue-timeout-secs 1800 \
  --request-timeout-secs 1800 \
  --log-level info \
  --disable-retries \
  --disable-load-monitoring \
  --disable-circuit-breaker \
  --disable-health-check \
  --prometheus-port "$PROMETHEUS_PORT" \
  >"$LOG_DIR/lb.log" 2>&1 &
pids+=("$!")

wait_http lb "http://127.0.0.1:${LB_PORT}/v1/models" 600
echo "[pd-1p1d] serving on http://127.0.0.1:${LB_PORT}/v1"

wait -n "${pids[@]}"
exit 1
