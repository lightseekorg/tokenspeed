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

# SMG gateway on the head node: prefill rank by least load (pending uncached
# prefill tokens and KV pressure), decode rank by power-of-two over resident
# tokens. It binds each role's ZMQ handshake at GATEWAY_ADDR (prefill 31000,
# decode 31001), so it starts before the engines. Launched by
# serve_dwdp16_dep16.sh.
set -euo pipefail
: "${GATEWAY_ADDR:?}" "${MODEL_PATH:?}" "${TOKENIZER_PATH:?}"

export NO_PROXY='*' no_proxy='*'
exec python3 -m smg launch --pd-disaggregation --backend tokenspeed \
    --prefill "tcp://$GATEWAY_ADDR:31000" \
    --decode "tcp://$GATEWAY_ADDR:31001" \
    --host 0.0.0.0 --port 8002 --prometheus-port 29002 \
    --model-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
    --model-alias "kimi-k3-agentx-pd=$MODEL_PATH" \
    --reasoning-parser passthrough --tool-call-parser passthrough \
    --prefill-policy least_load --decode-policy power_of_two --power-of-two-load-metric resident_tokens:62500 \
    --least-load-kv-pressure-weight 0.15 --least-load-default-throughput 2000 \
    --load-monitor-interval 1 --assignment-mode min_load --dp-aware --zmq-engine-count 16 \
    --routing-key-headers x-smg-routing-key --max-idle-secs 14400 \
    --disable-retries --disable-circuit-breaker --disable-health-check \
    --disable-worker-overload-protection --worker-wedge-secs 0 \
    --request-timeout-secs 21600 --queue-timeout-secs 21600 --pd-admission-wait-secs 21600 \
    --zmq-startup-timeout-secs 10800 --worker-startup-timeout-secs 10800 \
    --max-concurrent-requests -1 --log-level info
