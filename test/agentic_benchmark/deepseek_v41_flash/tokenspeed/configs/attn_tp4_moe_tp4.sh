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

exec ts serve \
    --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --attn-tp-size 4 \
    --moe-tp-size 4 \
    --max-model-len 102400 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.9 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --dtype bfloat16 \
    --moe-backend flashinfer_trtllm \
    --speculative-algorithm DSPARK \
    --speculative-num-steps 5 \
    --speculative-num-draft-tokens 6 \
    --speculative-eagle-topk 1 \
    --enable-prefix-caching \
    --host 127.0.0.1 \
    --port 8000 \
    "$@"
