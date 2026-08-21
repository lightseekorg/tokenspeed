#!/usr/bin/bash

set -euo pipefail

# Attention data-parallel recipe. Requires the dynamic MLA packing fix
# (PR #1152) — the K3 cache recipe rejects attn tp=1 layouts without it.
# Do NOT pass --dense-tp-size: the K3 dense MLP binds dense.tp_group directly
# and skips on zero rows, which cross-deadlocks against the MoE gather when
# the dense group spans DP ranks.
#
# chunked-prefill stays at 2048: the trtllm MoE workspace scales with
# gathered GLOBAL tokens (chunk x 8 under DP), and 0.95 utilization only
# leaves room for 2048-token chunks (8192 needs util <= 0.90).

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --data-parallel-size 8 \
    --ep-size 8 \
    --moe-tp-size 1 \
    --max-model-len 81920 \
    --max-num-seqs 16 \
    --max-prefill-tokens 2048 \
    --chunked-prefill-size 2048 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000
