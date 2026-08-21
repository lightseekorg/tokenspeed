#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --moe-tp-size 8 \
    --ep-size 1 \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --disable-kvstore \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000 \
    --dist-init-addr 127.0.0.1:4000
