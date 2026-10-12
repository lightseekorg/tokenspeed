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

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../model_config.sh"

export TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB=2048

# Checkpoint metadata selects mixed quantization: NVFP4 experts, FP8 MTP/PLE.
# Forcing --quantization nvfp4 would override that metadata with the wrong loader.
exec ts serve \
    --model "$QWEN_MODEL_ID" \
    --revision "$QWEN_MODEL_REVISION" \
    --tokenizer "$QWEN_TOKENIZER_PATH" \
    --tokenizer-path "$QWEN_TOKENIZER_PATH" \
    --attn-tp-size 4 \
    --ep-size 4 \
    --attention-backend trtllm \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype bfloat16 \
    --speculative-algorithm MTP \
    --speculative-num-steps 3 \
    --hf-overrides '{"index_share_for_mtp_iteration":true}' \
    --drafter-attention-backend trtllm \
    --max-num-seqs 32 \
    --max-prefill-tokens 16384 \
    --chunked-prefill-size 16384 \
    --gpu-memory-utilization 0.9 \
    --disable-kvstore \
    --trust-remote-code \
    --engine-startup-timeout 7200 \
    --host 127.0.0.1 \
    --port 8000 \
    "$@"
