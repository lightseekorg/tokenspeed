#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model nvidia/Kimi-K2.5-NVFP4 \
    --data-parallel-size 8 \
    --ep-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.8 \
    --disable-cuda-graph-padding \
    --trust-remote-code \
    --attention-backend trtllm_mla \
    --moe-backend flashinfer_trtllm \
    --quantization nvfp4 \
    --kv-cache-dtype fp8 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3-mla \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant \
    --drafter-attention-backend trtllm_mla \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000 \
    --dist-init-addr 127.0.0.1:4000
