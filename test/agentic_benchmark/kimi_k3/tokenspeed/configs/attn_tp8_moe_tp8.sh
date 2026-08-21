#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --moe-tp-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path Inferact/Kimi-K3-DSpark \
    --speculative-num-draft-tokens 8 \
    --drafter-attention-backend mla \
    --mm-encoder-tp-mode data \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000 \
    --engine-startup-timeout 7200
