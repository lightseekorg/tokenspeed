#!/usr/bin/bash

set -euo pipefail

exec sglang serve \
    --model nvidia/Kimi-K2.5-NVFP4 \
    --tp-size 4 \
    --context-length 80000 \
    --max-running-requests 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.9 \
    --trust-remote-code \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs 16 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3-mla \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --host 127.0.0.1 \
    --port 8003
