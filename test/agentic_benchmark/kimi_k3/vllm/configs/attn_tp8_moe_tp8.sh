#!/usr/bin/bash

set -euo pipefail

# TokenSpeed counts the target token in its T=8 width; vLLM counts the seven
# speculative tokens only. Keep vLLM's default CUDA Graph mode so both
# prefill (PIECEWISE) and decode (FULL) graphs remain enabled.
exec vllm serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --tensor-parallel-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --load-format fastsafetensors \
    --no-enable-flashinfer-autotune \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --mm-encoder-tp-mode data \
    --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}' \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --prefix-match-unit 128 \
    --moe-backend flashinfer_trtllm \
    --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
    --host 127.0.0.1 \
    --port 8002
