#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --ep-size 8 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000
