#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# One node of the DEP16 decode role (GPUs 1 and 3 of eight four-GPU nodes).
# Launched on every node by serve_dwdp16_dep16.sh.
set -euo pipefail
: "${AGENTX_HEAD:?}" "${MODEL_PATH:?}" "${TOKENIZER_PATH:?}" "${DRAFT_MODEL_PATH:?}" "${DISAGGREGATION_IB_DEVICE:?}"

export TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=4
export TOKENSPEED_SKIP_GRPC_WARMUP=1
export TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT=3600
export TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024
unset PYTORCH_CUDA_ALLOC_CONF TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export NO_PROXY='*' no_proxy='*'
export OMP_NUM_THREADS=4
# Bounds FlashInfer's JIT builds at first start, which otherwise run one compiler
# per core beside the shared L2 pool's host memory.
export MAX_JOBS=8
ulimit -c 0

# --max-num-seqs is global: 64 request slots per attention-DP rank. The
# synthetic acceptance length fixes EAGLE3 acceptance at 3.0 tokens per step.
exec python3 -m smg_grpc_servicer.tokenspeed \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --served-model-name kimi-k3-agentx-pd \
    --world-size 16 --nprocs-per-node 2 --nnodes 8 --node-rank "$SLURM_NODEID" --base-gpu-id 1 --gpu-id-step 2 \
    --dist-init-addr "$AGENTX_HEAD:29601" \
    --attn-tp-size 1 --data-parallel-size 16 --moe-tp-size 1 --ep-size 16 \
    --max-model-len 1048576 --max-num-seqs 1024 --gpu-memory-utilization 0.9 \
    --max-prefill-tokens 4096 --chunked-prefill-size 4096 \
    --disable-prefill-graph --prefill-graph-max-tokens 0 \
    --disable-cuda-graph-padding --cudagraph-capture-sizes $(seq 1 64) \
    --disable-kvstore --retraction-snapshot-host-gb 34.359738368 --mamba-ssm-dtype bfloat16 \
    --trust-remote-code --language-model-only --mm-encoder-tp-mode weights \
    --attention-backend tokenspeed_mla --kda-backend cutedsl_kda \
    --moe-backend mega_moe --all2all-backend none \
    --kv-cache-dtype fp8 --min-kv-split 8 --mla-chunk-multiplier 2 \
    --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:20400"}' \
    --disaggregation-mode decode --load-balance-method round_robin \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-layerwise-interval 0 \
    --disaggregation-ib-device "$DISAGGREGATION_IB_DEVICE" \
    --disaggregation-bootstrap-port 8998 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
    --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1 \
    --drafter-attention-backend tokenspeed_mla --sampling-backend greedy \
    --synthetic-acceptance-length 3.0 \
    --seed 42 --host 0.0.0.0 --port 18447 \
    --enable-log-requests --log-requests-level 0 --enable-log-request-stats
