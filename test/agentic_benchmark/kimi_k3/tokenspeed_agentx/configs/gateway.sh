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

# SMG gateway on the head node: prefill rank by least load, decode rank by
# power-of-two least requests. Launched by serve_dwdp16_dep16.sh.
set -euo pipefail
: "${AGENTX_HEAD:?}" "${MODEL_PATH:?}" "${SMG_BINARY:?}"

export NO_PROXY='*' no_proxy='*'
export RUST_LOG=info,smg::routers::grpc::common::stages::helpers=debug
exec "$SMG_BINARY" launch --pd-disaggregation \
    --prefill "grpc://$AGENTX_HEAD:18446" 8998 \
    --decode "grpc://$AGENTX_HEAD:18447" \
    --host 0.0.0.0 --port 8002 --prometheus-port 29002 \
    --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" \
    --reasoning-parser passthrough --tool-call-parser passthrough \
    --prefill-policy least_load --decode-policy power_of_two --power-of-two-load-metric least_requests \
    --least-load-kv-pressure-weight 0.15 --load-monitor-interval 1 --assignment-mode min_load --dp-aware \
    --routing-key-headers x-smg-routing-key --max-idle-secs 14400 \
    --disable-retries --disable-circuit-breaker --disable-health-check \
    --disable-worker-overload-protection --worker-wedge-secs 0 \
    --request-timeout-secs 21600 --queue-timeout-secs 21600 --pd-admission-wait-secs 21600 \
    --max-concurrent-requests -1 --log-level info
