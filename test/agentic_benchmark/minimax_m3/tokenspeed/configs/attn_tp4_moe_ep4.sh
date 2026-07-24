#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model nvidia/MiniMax-M3-NVFP4 \
    --attn-tp-size 4 \
    --ep-size 4 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.95 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend trtllm \
    --kv-cache-dtype fp8 \
    --moe-backend flashinfer_trtllm \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path Inferact/MiniMax-M3-EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --drafter-attention-backend fa4 \
    --disable-prefill-graph \
    --disable-kvstore \
    --block-size 128 \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000
