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

set -eo pipefail

# Share input validation with the launcher so invalid settings fail before startup.
validate_agentx_values() {
    local variable
    for variable in "$@"; do
        [[ -n "${!variable:-}" ]] || { echo "Missing $variable" >&2; return 2; }
        # The pinned native replay builder splits its command on whitespace.
        [[ "${!variable}" != *[[:space:]]* ]] || { echo "$variable must not contain whitespace" >&2; return 2; }
        export "$variable"
    done
}

validate_agentx_config() {
    validate_agentx_values CLIENT_ROOT TOKENIZER_PATH MODEL_NAME API_PORT CONCURRENCY \
        DURATION BENCHMARK_GRACE_SECONDS REQUEST_TIMEOUT_SECONDS CLIENT_SEED \
        AIPERF_WARMUP_REQUESTS_PER_LANE AGENTIC_WARMUP_GRACE_PERIOD \
        AIPERF_TRACE_IDLE_GAP_CAP_SECONDS AIPERF_FAILED_REQUEST_THRESHOLD \
        AIPERF_EXTRA_INPUTS AIPERF_UNSAFE_OVERRIDE || return "$?"
    case "$AIPERF_UNSAFE_OVERRIDE" in
        true|false) ;;
        *) echo "AIPERF_UNSAFE_OVERRIDE must be true or false; got '$AIPERF_UNSAFE_OVERRIDE'" >&2; return 2 ;;
    esac
}

# Sourcing exposes validation only; replay runs when this script is executed.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
validate_agentx_config
validate_agentx_values RESULT_DIR EP_SIZE AIPERF_SERVER_URL
mkdir "$RESULT_DIR"
export INFMAX_CONTAINER_WORKSPACE="$CLIENT_ROOT/inferencex/inferencex-e2e"
export AIPERF_RUNTIME_DIR="$CLIENT_ROOT"
export PYTHONPATH="$INFMAX_CONTAINER_WORKSPACE"
export MODEL="$MODEL_NAME" SERVED_MODEL_NAME="$MODEL_NAME" MODEL_PREFIX=kimik3
export PORT="$API_PORT" CONC="$CONCURRENCY"
export FRAMEWORK=tokenspeed RUNNER_TYPE=gb300 PRECISION=nvfp4 SPEC_DECODING=eagle3
export TP=8 PP_SIZE=1 PCP_SIZE=1 DCP_SIZE=1 IS_MULTINODE=true DISAGG=false KV_OFFLOADING=none
export AIPERF_PYTHON_VERSION=3.11 AIPERF_EXPERIMENTAL_FAST=0
export AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
export AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=3600
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD="$AIPERF_FAILED_REQUEST_THRESHOLD"
export AIPERF_SERVER_METRICS_URLS= AIPERF_REQUIRED_SERVER_METRIC_PREFIX=
export ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 EVAL_ONLY=false
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
export AGENTIC_OUTPUT_DIR="$RESULT_DIR" RESULT_FILENAME=inferencex_result
export AIPERF_DATASET_WEKA_PARALLEL_WORKERS=2 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
AIPERF_DEPS_READY=1
resolve_trace_source
MODEL="$TOKENIZER_PATH" build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --benchmark-grace-period $BENCHMARK_GRACE_SECONDS --request-timeout-seconds $REQUEST_TIMEOUT_SECONDS --random-seed $CLIENT_SEED"
REPLAY_CMD+=" --no-server-metrics"

write_agentic_result_json() {
    # Upstream's multi-node report shape describes PD disaggregation. These two
    # nodes are one aggregate TP8 worker: use native aggregate accounting (8 GPUs).
    # Keep physical topology separately without rewriting native metrics or JSON.
    printf '%s\n' '{"physical_nodes":2,"gpus_per_node":4,"num_gpus":8,"deployment":"aggregate","official_inferencex_result":false}' \
        > "$RESULT_DIR/deployment.json"
    IS_MULTINODE=false "$AIPERF_PYTHON" -m infx.results.agentic.process_agentic_result
}
run_agentic_replay_and_write_outputs "$RESULT_DIR"
