#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --ep-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.9 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path lightseekorg/kimi-k3-dflash2 \
    --speculative-num-steps 7 \
    --speculative-num-draft-tokens 8 \
    --speculative-eagle-topk 1 \
    --drafter-attention-backend mla \
    --mm-encoder-tp-mode data \
    --enable-cache-report \
    --reasoning-parser passthrough \
    --tool-call-parser passthrough \
    --host 0.0.0.0 \
    --port 8000 \
    --engine-startup-timeout 7200
