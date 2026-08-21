#!/usr/bin/bash

set -euo pipefail

# TP8 baseline + DSpark speculative decoding — the clearly-labelled spec row.
# Speculative flags follow the proven CI gates (kimi-k3-nvfp4-dspark two-node
# GB300 / kimi-k3-dspark mxfp4 AMD). Spec stays impossible for the DP config
# (the drafter family raises under dp>1), so this row is compared against the
# spec-off matrix rather than filling it.

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --ep-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path Inferact/Kimi-K3-DSpark \
    --speculative-num-draft-tokens 8 \
    --drafter-attention-backend mla \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000 \
    --engine-startup-timeout 3600
