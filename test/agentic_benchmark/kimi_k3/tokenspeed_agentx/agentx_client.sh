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

# One AgentX point through InferenceX and AIPerf, inside the container on the
# head node. Launched by bench_agentx.sh.
set -euo pipefail
: "${CLIENT_ROOT:?}" "${CLIENT_CACHE:?}" "${GATEWAY_URL:?}" "${TOKENIZER_PATH:?}"
: "${RESULT_DIR:?}" "${CONCURRENCY:?}" "${DURATION:?}" "${WARMUP_REQUESTS_PER_LANE:?}"

mkdir "$RESULT_DIR"
source "$CLIENT_ROOT/venv/bin/activate"
export INFMAX_CONTAINER_WORKSPACE="$CLIENT_ROOT/inferencex/inferencex-e2e"
export AIPERF_RUNTIME_DIR="$CLIENT_ROOT" PYTHONPATH="$CLIENT_ROOT/aiperf/src:$INFMAX_CONTAINER_WORKSPACE"
export CONC="$CONCURRENCY" DURATION AIPERF_SERVER_URL="$GATEWAY_URL" PORT=8002
export MODEL=kimi-k3-agentx-pd SERVED_MODEL_NAME=kimi-k3-agentx-pd MODEL_PREFIX=kimik3
export FRAMEWORK=tokenspeed RUNNER_TYPE=gb300 PRECISION=nvfp4 SPEC_DECODING=eagle3
# InferenceX's report schema: one 16-rank prefill and one 16-rank decode worker.
export TP=16 EP_SIZE=16 PP_SIZE=1 PCP_SIZE=1 DCP_SIZE=1 DP_ATTENTION=true
export IS_MULTINODE=true DISAGG=true KV_OFFLOADING=cpu
export KV_OFFLOAD_BACKEND=tokenspeed-kvstore
export KV_OFFLOAD_BACKEND_METADATA='{"name":"tokenspeed-kvstore","version":"0.1.1"}'
export PREFILL_NUM_WORKERS=1 PREFILL_TP=16 PREFILL_EP=1 PREFILL_DP_ATTN=true
export DECODE_NUM_WORKERS=1 DECODE_TP=16 DECODE_EP=16 DECODE_DP_ATTN=true
export PREFILL_PP_SIZE=1 PREFILL_PCP_SIZE=1 PREFILL_DCP_SIZE=1
export DECODE_PP_SIZE=1 DECODE_PCP_SIZE=1 DECODE_DCP_SIZE=1
export AIPERF_PYTHON_VERSION=3.11 AIPERF_EXPERIMENTAL_FAST=0
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=60
# Each request carries its branch correlation ID as the SMG routing key.
export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
export AIPERF_HTTP_X_SMG_ROUTING_KEY_SOURCE=branch
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=false
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
export AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0 AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=3600
export AIPERF_UNSAFE_OVERRIDE=false
export AIPERF_WARMUP_REQUESTS_PER_LANE="$WARMUP_REQUESTS_PER_LANE" AGENTIC_WARMUP_GRACE_PERIOD=1800
export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS=300
export AIPERF_FAILED_REQUEST_THRESHOLD=0.1 AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.1
export AIPERF_EXTRA_INPUTS='{"ignore_eos":true,"temperature":0.0,"top_p":0.95,"top_k":20}'
export AIPERF_SERVER_METRICS_URLS= AIPERF_REQUIRED_SERVER_METRIC_PREFIX=
export ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 EVAL_ONLY=false
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export AGENTIC_OUTPUT_DIR="$RESULT_DIR" RESULT_FILENAME=inferencex_result
export AIPERF_DATASET_WEKA_PARALLEL_WORKERS=2 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_CACHE="$CLIENT_CACHE/hub" HF_DATASETS_CACHE="$CLIENT_CACHE/datasets" HF_MODULES_CACHE="$CLIENT_CACHE/modules"
export HF_DATASETS_OFFLINE=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE MAX_MODEL_LEN

source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
AIPERF_DEPS_READY=1
# The pinned trace is already in CLIENT_CACHE; no revision lookup.
TRACE_SOURCE_FLAG='--public-dataset semianalysis_cc_traces_weka_062126'
MODEL="$TOKENIZER_PATH" build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --benchmark-grace-period 900 --request-timeout-seconds 21600 --random-seed 42 --no-server-metrics"
printf '%s\n' "$REPLAY_CMD" > "$RESULT_DIR/client-command.txt"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
