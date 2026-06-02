#!/usr/bin/bash

set -euo pipefail

exec vllm serve \
    --model nvidia/Kimi-K2.5-NVFP4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8 \
    --attention-backend auto \
    --moe-backend flashinfer_trtllm \
    --speculative-config '{"method":"eagle3","model":"lightseekorg/kimi-k2.5-eagle3-mla","num_speculative_tokens":3}' \
    --host 127.0.0.1 \
    --port 8002
