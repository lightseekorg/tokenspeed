#!/usr/bin/bash

set -euo pipefail

# TP8 instance with the MLA KV cache sharded over the 8 attention ranks
# (--kv-parallel-size 8): every rank keeps 1/8 of the MLA pages, so the
# cache holds about 8x the MLA tokens. An aggregated server decodes over the
# sharded pages with decode context parallelism of the same size. Prefill
# graphs cover forwards up to 8192 tokens (multi-request KDA graphs for
# 1/2/4/8 requests); longer forwards of the 16K chunk run eager.

# --grpc runs the engine behind a separately launched P/D gateway.
if [[ "${1:-}" == "--grpc" ]]; then
    shift
    SERVER_CMD=(python3 -m smg_grpc_servicer.tokenspeed)
else
    SERVER_CMD=(ts serve
        --disable-health-check
        --reasoning-parser passthrough
        --tool-call-parser passthrough
        --engine-startup-timeout 7200
    )
fi

exec "${SERVER_CMD[@]}" \
    --model nvidia/Kimi-K3-NVFP4 \
    --attn-tp-size 8 \
    --kv-parallel-size 8 \
    --decode-context-parallel-size 8 \
    --moe-tp-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 128 \
    --chunked-prefill-size 16384 \
    --prefill-graph-max-tokens 8192 \
    --prefill-graph-capture-token-sizes 256 512 768 1024 1536 2048 2560 3072 4096 4608 5120 6144 7168 8192 \
    --prefill-graph-capture-batch-sizes 1 2 4 8 \
    --kvstore-size 80 \
    --gpu-memory-utilization 0.9 \
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
    --host 0.0.0.0 \
    --port 8000 \
    "$@"
