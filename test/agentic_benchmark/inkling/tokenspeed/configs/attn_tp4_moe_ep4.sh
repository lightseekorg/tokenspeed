#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model thinkingmachines/Inkling-NVFP4 \
    --attn-tp-size 4 \
    --ep-size 4 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.95 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend fa4 \
    --moe-backend flashinfer_trtllm \
    --speculative-algorithm MTP \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-prefix-caching \
    --disable-kvstore \
    --block-size 128 \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000
