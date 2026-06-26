#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model nvidia/GLM-5.2-NVFP4 \
    --attn-tp-size 4 \
    --moe-tp-size 4 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.9 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --moe-backend flashinfer_trtllm \
    --quantization nvfp4 \
    --kv-cache-dtype fp8 \
    --speculative-algorithm MTP \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000
