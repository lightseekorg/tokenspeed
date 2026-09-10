#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --attn-tp-size 4 \
    --dense-tp-size 4 \
    --ep-size 4 \
    --max-model-len 102400 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.9 \
    --prefill-graph-max-tokens 8192 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-use-fp4-indexer-cache \
    --moe-backend flashinfer_trtllm \
    --draft-moe-backend mega_moe \
    --kv-cache-dtype fp8 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
    --speculative-eagle-topk 1 \
    --enable-prefix-caching \
    --disable-kvstore \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000
