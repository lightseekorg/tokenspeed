#!/usr/bin/bash

set -euo pipefail

# TP8 baseline + DSpark speculative decoding — the clearly-labelled spec row.
# Speculative flags AND the memory envelope follow the CI DSpark gates
# (kimi-k3-nvfp4-dspark two-node GB300 / kimi-k3-dspark mxfp4 AMD): both run
# util 0.92 with the prefill graph and kvstore disabled — the draft weights,
# its BF16 drafter cache, and the extra graph buckets eat the post-profiling
# slack, and DSPARK+KVStore has zero CI coverage. One deliberate deviation:
# max-model-len stays 80000 (the gates use 65536) because this workload's
# final prompts reach ~68.2K; watch the first boot.
# The DP rows stay spec-off: the DP x speculative combination has no CI
# coverage here, so this row is compared against TP8 spec-off rather than
# filling the matrix.

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --ep-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --disable-prefill-graph \
    --disable-kvstore \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path Inferact/Kimi-K3-DSpark \
    --speculative-num-draft-tokens 8 \
    --drafter-attention-backend mla \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000 \
    --engine-startup-timeout 7200
