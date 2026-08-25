#!/usr/bin/bash

set -euo pipefail

# Attention data-parallel recipe. Requires TWO merged prerequisites:
#   - PR #1152 (dynamic MLA packing) — the K3 cache recipe rejects attn tp=1
#     layouts without it (boot failure);
#   - PR #1185 (merge_state head tiling) — attn tp=1 runs 96 heads/rank and
#     the old kernel rejects >64-head launches (runtime crash on any prompt
#     that crosses a chunk boundary).
# --dist-init-addr is REQUIRED here: dp>1 on bare metal hard-raises without
# it (PortArgs.init_new); only Slurm steps can derive it from the topology.
# Do NOT pass --dense-tp-size: the K3 dense MLP binds dense.tp_group directly
# and skips on zero rows, which cross-deadlocks against the MoE gather when
# the dense group spans DP ranks.
#
# chunked-prefill stays at 2048: the trtllm MoE workspace scales with
# gathered GLOBAL tokens (chunk x 8 under DP), and high utilization only
# leaves room for 2048-token chunks.
#
# DSpark rides on the DP recipe like the rest of the matrix. util drops to
# 0.92: the 7.1GB draft weights replicate per rank on top of the already
# heaviest weight budget in the matrix, and the drafter's cache pages join
# the shared pool. DP x DSpark has no CI coverage and no on-machine
# validation yet — boot-validate before trusting this row's numbers.

exec ts serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --data-parallel-size 8 \
    --ep-size 8 \
    --moe-tp-size 1 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --attention-backend tokenspeed_mla \
    --kda-backend cutedsl_kda \
    --moe-backend flashinfer_trtllm \
    --kv-cache-dtype fp8 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path Inferact/Kimi-K3-DSpark \
    --speculative-num-draft-tokens 8 \
    --drafter-attention-backend tokenspeed_mla \
    --mm-encoder-tp-mode data \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 8000 \
    --engine-startup-timeout 7200 \
    --dist-init-addr 127.0.0.1:4000
