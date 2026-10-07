#!/usr/bin/bash

set -euo pipefail

export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=4

# TODO: --gpu-memory-utilization 0.9 causes OOM; need a revisit

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --data-parallel-size 32 \
    --ep-size 32 \
    --max-model-len 80000 \
    --max-num-seqs 1024 \
    --max-prefill-tokens 4096 \
    --chunked-prefill-size 4096 \
    --gpu-memory-utilization 0.8 \
    --kvstore-size 80 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lightseekorg/kimi-k3-eagle3-mla \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --drafter-attention-backend tokenspeed_mla \
    --mm-encoder-tp-mode data \
    --policy round_robin \
    --dp-aware \
    --sticky-sessions \
    --reasoning-parser passthrough \
    --tool-call-parser passthrough \
    --host 0.0.0.0 \
    --port 8000 \
    --engine-startup-timeout 7200
