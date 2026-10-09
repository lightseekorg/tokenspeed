# Kimi-K3 DEP tensor parallelism

In a DEP deployment, each attention rank owns its requests and caches, while
routed experts are distributed across the full expert-parallel group. Some
dense projections can share weights across smaller TP groups without changing
that ownership. This reduces replicated weight memory, but adds communication;
measure the complete workload before enabling it for latency.

## Choose what to shard

Set the same values on every worker before starting the server:

| Environment variable | Sharded computation |
|---|---|
| `TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE` | KDA QKV/gates and MLA QKV-A/output gate |
| `TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE` | KDA and MLA output projections |
| `TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE` | Shared-expert MLP |

Each setting is independent. Unset or `1` keeps its existing implementation.
Projection TP sizes must be positive divisors of world size; shared-expert TP
must also be smaller than world size. Contiguous ranks form each TP group.
Attention and linear attention must remain TP1/DPworld, routed MoE TP1/EPworld,
and pipeline parallelism must be disabled.

Projection sharding supports BF16 and the 128×128 block-FP8 attention weights
in the NVFP4 checkpoint, with BF16 activations. It retains the checkpoint's
FP8 codes and scales. QKV TP requires gated MLA. Gated MLA requires Q-LoRA,
including when projections remain replicated; configurations without Q-LoRA
are rejected during model construction. MLA Q-B, KV-B, KDA convolution, norms
and recurrent-state caches retain their existing mapping.
Target-model projection settings do not change the draft model.

## DEP16 example

Use four contiguous ranks per projection group. This example keeps shared
experts replicated so the comparison isolates QKV and output projection TP:

```bash
export TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=1

python -m tokenspeed.cli serve \
  --model MODEL_DIR --quantization nvfp4 --dtype bfloat16 \
  --world-size 16 --nprocs-per-node 4 \
  --attn-tp-size 1 --data-parallel-size 16 --dense-tp-size 1 \
  --moe-tp-size 1 --expert-parallel-size 16 \
  --all2all-backend flashinfer --moe-backend flashinfer_trtllm \
  --attention-backend tokenspeed_mla --kda-backend cutedsl_kda \
  --dist-init-addr HEAD_NODE:29500 \
  --max-num-seqs 256 --max-model-len 4096 \
  --chunked-prefill-size 1024 --max-prefill-tokens 1024 \
  --prefix-granularity 128 --enable-prefix-caching --disable-kvstore \
  --max-cudagraph-capture-size 16 --cudagraph-capture-sizes 1 2 4 8 16
```

Launch one server process per node, using the same checkpoint, container and
environment. On Slurm, reserve a persistent allocation with `salloc`, then use
the site's `submit` wrapper or `srun` inside it. Place each TP4 subgroup on a
CUDA-IPC-accessible node for the optimized communication path. Runtime backend
selection retains generic collective fallbacks for other supported geometries.

For the DEP16 baseline, set both projection variables to `1` and keep every
other setting fixed. Here the global sequence limit allows 16 requests per
rank. Use real weights and the full model for capacity and E2E comparisons.

Shared-expert TP can be enabled separately or together with these projections.
See [shared-expert TP](kimi-k3-shared-expert-tp.md) for its stream ordering and
deployment details; its communication buffers remain separate.

## Execution and tradeoffs

QKV projection gathers complete token rows, applies a column-sharded Linear,
then exchanges output channels back to their original token owners. Output
projection exchanges channel shards, applies a row-sharded Linear, then
reduce-scatters complete outputs back to those owners. Local attention and
AttnRes therefore receive the same token layout as DEP16.

Uneven owners use their subgroup's maximum physical row count for collective
padding. Empty owners still participate when peers have work, in QKV then
output-projection order, while skipping local projections and attention that
do not require communication. Communication storage is prepared before
CUDA-graph capture and reused across sequential
layers; it is not allocated per layer. The ordinary communication backend
selects optimized collectives and supported fused quantization paths.

TP4 reduces the selected projection weights to approximately one quarter per
GPU. Replicated components and communication scratch prevent a fourfold saving
in total model memory. Communication and padding can outweigh the smaller
GEMMs, especially with uneven traffic or large prefill batches. Compare TTFT,
decode latency and cache capacity separately; projection-only timings do not
establish an E2E speedup.

Projection-only measurements with real checkpoint weights showed gains for
some small decode batches, but slower execution for larger batches and MLA
QKV-A. Enable each projection independently and measure your target workload;
weight-memory savings do not imply lower latency.
