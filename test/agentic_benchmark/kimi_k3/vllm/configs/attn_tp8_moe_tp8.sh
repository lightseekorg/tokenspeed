#!/usr/bin/bash

set -euo pipefail

exec vllm serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --tensor-parallel-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --load-format fastsafetensors \
    --no-enable-flashinfer-autotune \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}' \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --prefix-match-unit 128 \
    --moe-backend flashinfer_trtllm \
    --host 127.0.0.1 \
    --port 8002
