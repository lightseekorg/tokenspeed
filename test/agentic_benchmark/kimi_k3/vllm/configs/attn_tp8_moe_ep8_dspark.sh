#!/usr/bin/bash

set -euo pipefail

# TokenSpeed's T=8 includes the target token, so the equivalent vLLM draft
# width is seven speculative tokens.
exec vllm serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 \
    --load-format fastsafetensors \
    --no-enable-flashinfer-autotune \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}' \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --prefix-match-unit 128 \
    --moe-backend flashinfer_trtllm \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
    --host 127.0.0.1 \
    --port 8002
