#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --world-size 4 \
    --nprocs-per-node 4 \
    --nnodes 1 \
    --pipeline-parallel-size 1 \
    --attn-tp-size 4 \
    --data-parallel-size 1 \
    --dense-tp-size 4 \
    --moe-tp-size 1 \
    --ep-size 4 \
    --moe-backend flashinfer_trtllm \
    --draft-moe-backend mega_moe \
    --kv-cache-dtype fp8_e4m3 \
    --attention-use-fp4-indexer-cache \
    --max-model-len 96000 \
    --max-num-seqs 8 \
    --max-total-tokens 2560000 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.90 \
    --disable-kvstore \
    --enable-prefix-caching \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
    --speculative-eagle-topk 1 \
    --max-cudagraph-capture-size 8 \
    --prefill-graph-max-tokens 8192 \
    --seed 0 \
    --enable-metrics \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000
