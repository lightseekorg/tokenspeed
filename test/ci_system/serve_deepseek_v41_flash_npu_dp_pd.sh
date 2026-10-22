#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Performance and precision-comparison recipes; full-model accuracy needs qualification.
# Both roles use 16 local devices; select the decode layout explicitly.
set -euo pipefail
ROLE=${1:?Usage: $0 prefill|decode|gateway dp2|dp4 performance|precision}
LAYOUT=${2:?Select dp2 or dp4 explicitly}
PROFILE=${3:?Select performance or precision explicitly}
if [[ $# != 3 ]]; then echo "Expected exactly three arguments" >&2; exit 2; fi
case "$ROLE" in prefill|decode|gateway) ;; *) echo "Unknown role: $ROLE" >&2; exit 2 ;; esac
case "$LAYOUT" in dp2) tp=8; dp=2 ;; dp4) tp=4; dp=4 ;; *) echo "Unknown layout: $LAYOUT" >&2; exit 2 ;; esac
case "$PROFILE" in performance|precision) ;; *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;; esac
: "${MODEL:?Local model path required}" "${SERVED_MODEL_NAME:?}" "${HOST:?}" "${PORT:?}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$REPO/python:$REPO/tokenspeed-kernel/python:$REPO/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="$NO_PROXY"
export TOKENSPEED_SKIP_GRPC_WARMUP=1
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=0
export TS_V41_PA_MODE=direct
# The receive deadline includes time queued for remote prefill. A cold wave
# of 32 x 50K prompts exceeds the runtime's 300-second default on this recipe.
export TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT=${TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT:-7200}
if [[ "$ROLE" == gateway ]]; then
  : "${PREFILL_URL:?}" "${DECODE_URL:?}" "${BOOTSTRAP_PORT:?}" "${PROMETHEUS_PORT:?}"
  exec python -m smg launch --pd-disaggregation --dp-aware \
    --prefill "$PREFILL_URL" "$BOOTSTRAP_PORT" --decode "$DECODE_URL" \
    --host "$HOST" --port "$PORT" --model-path "$MODEL" --tokenizer-path "$MODEL" \
    --prometheus-port "$PROMETHEUS_PORT" \
    --reasoning-parser deepseek_v41 --prefill-policy round_robin --decode-policy round_robin \
    --max-concurrent-requests 32 --queue-size 128 --queue-timeout-secs 7200 \
    --request-timeout-secs 7200 --disable-retries --log-level info
fi
: "${DIST_INIT_ADDR:?}" "${BOOTSTRAP_PORT:?}"
: "${ASCEND_RT_VISIBLE_DEVICES:?List the 16 local device IDs explicitly}"
parallel=(--tensor-parallel-size 16 --dense-tp-size 8 --enable-expert-parallel)
capture=(1 8 32)
max_sequences=32
if [[ "$ROLE" == decode ]]; then
  parallel=(--tensor-parallel-size "$tp" --data-parallel-size "$dp" --lm-head-tp-size "$tp"
            --dense-tp-size "$tp" --enable-expert-parallel --moe-combine-order rank
            --load-balance-method round_robin)
  if [[ "$LAYOUT" == dp2 ]]; then
    # Cover uneven DP batches without rounding up to a larger graph bucket.
    capture=({1..32})
    max_sequences=64
  else
    capture=(1 2 4 8)
  fi
fi
if [[ "$PROFILE" == performance ]]; then
  : "${FLATQUANT_CONFIG:?Preconverted W4A4 manifest required}"
  numerical=(--moe-backend ascend_w4a4 --npu-flatquant-config "$FLATQUANT_CONFIG"
             --draft-moe-backend ascend_bf16 --speculative-algorithm DSPARK
             --npu-v41-index-cache bf16 --npu-v41-kv-cache bf16)
else
  numerical=(--moe-backend ascend_bf16
             --npu-v41-index-cache bf16-qdq --npu-v41-kv-cache bf16-qdq)
fi
args=(
  --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --host "$HOST" --port "$PORT"
  --device npu "${parallel[@]}" --engram-tp-size 16
  "${numerical[@]}" --dtype bfloat16 --disable-autotune
  --cudagraph-capture-sizes "${capture[@]}"
  --max-cudagraph-capture-size "${capture[-1]}" --disable-prefill-graph --disable-pdl
  --sampling-backend greedy --sampling-stream per-request
  --max-model-len 1019904 --max-total-tokens 2097152 --max-num-seqs "$max_sequences"
  --chunked-prefill-size 2048 --max-prefill-tokens 2048 --gpu-memory-utilization 0.95
  --disable-kvstore --trust-remote-code --language-model-only
  --disable-weight-loader-prefetch-checkpoints --reasoning-parser deepseek_v41
  --seed 20261015 --dist-init-addr "$DIST_INIT_ADDR"
  --disaggregation-mode "$ROLE" --disaggregation-transfer-backend mooncake
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" --disaggregation-layerwise-interval 0
)
if [[ "$ROLE" == decode ]]; then
  args+=(--disable-prefix-caching)
fi
exec python -m smg_grpc_servicer.tokenspeed "${args[@]}"
