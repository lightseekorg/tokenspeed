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

set -euo pipefail
: "${SERVER_VENV:?native GPU runtime venv}"
: "${MODEL_DIR:?pinned target checkpoint}"
: "${DRAFT_DIR:?pinned Eagle3 checkpoint}"
: "${MODEL_NAME:?served model name}"
: "${API_PORT:?API port}"
: "${GPU_MEMORY_UTILIZATION:?GPU memory fraction}"
: "${READINESS_TIMEOUT:?startup timeout}"
: "${SERVER_SEED:?server random seed}"
source "$SERVER_VENV/bin/activate"
exec ts serve \
    --model "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --attn-tp-size 8 \
    --moe-tp-size 8 \
    --max-model-len 262144 \
    --max-num-seqs 16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_DIR" \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --drafter-attention-backend tokenspeed_mla \
    --mm-encoder-tp-mode data \
    --reasoning-parser passthrough \
    --tool-call-parser passthrough \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --engine-startup-timeout "$READINESS_TIMEOUT" \
    --seed "$SERVER_SEED" \
    --disable-kvstore \
    --enable-cache-report
