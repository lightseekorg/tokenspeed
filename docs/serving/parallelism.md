# Parallelism

TokenSpeed exposes familiar `--tensor-parallel-size` and `--tp` entry points
plus additional split parallelism controls for attention, dense, and MoE layers.

Scheduler process names in `ps` include their parallel ranks, for example
`tokenspeed::scheduler_tp1_ep3_dp0`. `tp` always identifies the attention TP
rank, including `tp0` for a single process. Other suffixes appear only when
their parallel size exceeds one: `ep` for MoE expert parallelism, `dp` and
`kvp` for attention data and KV parallelism, and `pp` for pipeline
parallelism. These are zero-based ranks within their respective
groups, not parallel sizes.

## Quick start

When the same tensor-parallel group is acceptable for the model, use this form:

```bash
tokenspeed serve <model> \
  --tensor-parallel-size 8
```

`--tensor-parallel-size` maps to TokenSpeed attention tensor parallelism, and
you cannot combine it with `--attn-tp-size`.

## Split parallelism

When different layer families need different process groups, use split knobs:

```bash
tokenspeed serve <model> \
  --world-size 8 \
  --attn-tp-size 4 \
  --dense-tp-size 4 \
  --moe-tp-size 4
```

| Parameter | Use |
| --- | --- |
| `--world-size` | Total worker processes across all nodes. |
| `--nprocs-per-node` | Worker processes launched on each node. |
| `--attn-tp-size` | Attention tensor parallel size. |
| `--dense-tp-size` | Dense layer tensor parallel size. Defaults to the attention TP width: the full world without DP attention, one replica with it. |
| `--moe-tp-size` | MoE layer tensor parallel size. |
| `--data-parallel-size` | Replicated data-parallel groups. |
| `--kv-parallel-size` | KV parallelism: shard the full-history KV pages over a consecutive subgroup of attention TP (see "KV parallelism" below). |
| `--mm-encoder-tp-mode` | `weights` (default), or TP1 whole-item DP within each attention TP group (`data`). |
| `--enable-expert-parallel` | Expert parallelism across the selected world size. |
| `--expert-parallel-size` | Explicit expert parallel size. |

Kimi-K3 TP8 deployments must combine `--tensor-parallel-size 8` with
`--mm-encoder-tp-mode data`. This keeps the text model at TP8 while running the
wide-QKV MoonViT encoder at TP1 with whole-item DP8.

### Decode-side TP layouts under attention DP

An MLA decode engine runs attention at TP1 with DP across the world: the
per-token latent KV cannot shard by heads, so every rank keeps its own KV.
That layout replicates every non-MoE weight on every rank: the MLA head
projections (`q_b_proj`, `kv_b_proj`, `o_proj`), the dense MLPs and the LM
head. On a large model that is tens of GB per rank that could hold KV
instead. Three knobs shard those weights over contiguous groups of DP ranks
(one node, typically) while attention and the KV stay TP1/DP:

| Parameter | Use |
| --- | --- |
| `--attn-head-tp-size W` | Shard `q_b_proj`, `kv_b_proj` and `o_proj` by heads over `W` contiguous ranks that hold different rows. Over DP ranks: requires attention TP 1, attention DP, `W` dividing the stage world, `num_heads % W == 0`, and a decode engine (`--disaggregation-mode decode`). Over the query shards of a prefill engine: `W` equal to `--prefill-context-parallel-size` (see [Query context parallelism](#query-context-parallelism-on-the-prefill-role)). |
| `--lm-head-tp-size W` | Vocab-shard the LM head over `W` contiguous ranks. Under attention DP the default is 1 (replicated); without attention DP it must equal the attention TP size (today's layout). |
| `--dense-tp-size W` | Already shards the dense MLPs over `W` ranks (token all-gather in, token reduce-scatter out). |
| `--tp-batch-invariant {none,attn,attn+dense}` | Make the sharded `o_proj` (`attn`) and dense `down_proj` (`attn+dense`) column-parallel on hidden so no cross-rank sum remains outside MoE; see below. |

`--attn-head-tp-size` is the one that changes the attention data flow. With
`W` ranks holding `T_full` decode rows together and this rank owning `T_own`
of them, each attention layer runs:

1. token all-gather of the normalized q latent: `[T_own, q_lora]` to
   `[T_full, q_lora]`;
2. `q_b_proj` and the absorption on this rank's `H / W` heads:
   `[T_full, H / W, kv_lora + rope]`;
3. all-to-all, heads to tokens: `[T_own, H, kv_lora + rope]` -- every head
   of this rank's own tokens;
4. the attention prologue (RoPE, KV write) and core attention on this rank's
   own KV, with the full head count;
5. all-to-all, tokens to heads: `[T_full, H / W, kv_lora]`;
6. the value projection with the local `w_vc`: `[T_full, H / W * v]`;
7. the `o_proj` tail back to `[T_own, hidden]`: row-parallel `o_proj` and a
   token reduce-scatter of the head partials, or, under
   `--tp-batch-invariant attn`, an all-gather of the heads, the
   column-parallel `o_proj` (`[T_full, hidden / W]`) and an all-to-all back
   to this rank's rows.

The head exchange precedes the prologue, so the prologue sees one row count
for the query, the latent and the write slots (RoPE commutes with the head
permutation). The KV write and any sparse-attention indexer stay as they are.
The head group's ranks all take part in every leg, including a DP rank with
no rows this step: it still owns a head shard of the group's tokens. That
participation lives in the attention module alone -- every decoder layer
calls its attention on an idle forward too, with its empty rows, and the
attention joins the exchanges or returns at once depending on the layout --
so no layer branches on the layout (the NextN and Eagle3 drafters' layers and
LongCat's two-attention layer included). A head group whose ranks are all
idle moves nothing and skips its collectives together. The exchange counts
come from the gathered per-rank token tables, so there is no device sync: the
legs up to core attention move the forward's input rows, the legs after it
the rows a narrowing draft step reported as its collective sizing (one live
row per request), and an idle rank sizes by the same tables. Decode CUDA
graphs pad every DP rank to the same batch.

**Decode rows only.** An expanded prefill needs every head's K and V for the
cached prefix. Under head TP each rank holds `kv_b_proj` for its `H / W`
heads and the latent cache for its own requests only, so no rank can expand
the other heads of its prefix, and no other rank holds that prefix to expand
it for it: the full-head K/V of a prefill is not available on the layout.
TokenSpeed therefore refuses the layout over DP ranks outside
`--disaggregation-mode decode` (head TP over the query shards of a prefill
engine serves its extend rows through the absorbed sparse prefill instead,
see below), and the decode engine keeps every extend-shaped forward off its
path:

- startup tunes on a decode step instead of the usual extend-shaped dummy
  forward, and the prefill CUDA graph (which records extend forwards) is
  turned off (`--disable-prefill-graph` is set, with a log line);
- a decode node runs no prefill of its own in any case: a request a capacity
  retraction suspends is imaged to Host and copied back by a restore
  (`docs/design/scheduler.md`, sections 2 and 4), never recomputed, so the
  layout needs no admission rule and no generation cap -- any
  `max_new_tokens` is admitted, as on every other engine.

A prefill row reaching the attention is then an invariant violation and
raises, not a configuration the operator can hit.

`--lm-head-tp-size` under attention DP gathers the ranks' logits rows,
runs the vocab-shard GEMM and transposes the shards back to each rank's own
rows (`[T_own, V]`), so sampling and logprobs downstream see the same
full-vocab rows as a replicated head. On the decode path the row counts come
from the per-rank token tables (every rank's logits rows are its decode
tokens, or the live rows a narrowing draft step reported), so the step has
no host sync. The shapes without a table -- a prefill's one row per request
or its logprob rows, a MIXED round, a model selecting its own logits rows --
exchange the counts. A drafter sharing the target's head must build its own
head on the same layout (the NextN, Eagle3-MLA and Llama-Eagle3 drafters do;
TokenSpeed refuses the others with a clear error). It cannot combine with
`--dp-sampling`, and admission refuses a request asking for prompt logprobs
(`logprob_start_len`): that path pushes the prompt rows through the LM head
in per-request chunks, a per-rank number of row exchanges the group cannot
agree on.

`--tp-batch-invariant` replaces the two reduce-scatters of these layouts
(after `o_proj`, after the dense `down_proj`) with column-parallel GEMMs on
hidden: the ranks all-gather the reduction dimension (heads, intermediate
channels), every rank computes its hidden shard of every token with full K,
and an all-to-all returns the rows. Every collective is then a pure
permutation of bytes, so the result is bitwise the full-K GEMM a TP1 or
replicated layer computes -- the point when a prefill engine with such a
layer must agree with the decode engine. It moves about the bytes of the
reduce-scatter it replaces and needs unquantized `o_proj` / `down_proj`.
`attn` requires head TP, and `attn+dense` also requires a dense TP group
wider than attention TP (never under query context parallelism, whose dense
group is 1 or the attention TP width).

The decode preset for a 128-way DP MLA model is thus one node-local group
of 8 reused three times:

```bash
tokenspeed serve <model> \
  --world-size 128 --nprocs-per-node 8 --attn-tp-size 1 --data-parallel-size 128 \
  --attn-head-tp-size 8 --dense-tp-size 8 --lm-head-tp-size 8 \
  --tp-batch-invariant attn+dense --disaggregation-mode decode ...
```

`--tp-batch-invariant attn+dense` needs the checkpoint's `o_proj` and dense
`down_proj` in the loading dtype. A quantized checkpoint qualifies only when
its `disable_quant_module` keeps those modules unquantized (`self_attn` for
`o_proj`; `dense_mlp`, or `mlps` for LongCat, for `down_proj`); the check
runs against the checkpoint's resolved quantization, not only
`--quantization`, and the layers check their own weights once built. Without
such an exclusion drop `--tp-batch-invariant` (the ordered-fold
reduce-scatter remains batch-invariant, see `docs/design/numerics.md`).

### Pinning a request to an attention-DP rank

Each attention-DP rank owns a private prefix cache, so multi-turn requests
only reuse their cache when every turn lands on the same rank. The
`data_parallel_rank` request field (`Engine.generate` / `async_generate`, or
the gateway's gRPC protocol) pins a request to one rank: it dispatches
straight there, bypassing load balancing — overload spill is the router's
job. An invalid pin fails that request. The engine keeps serving. Engines
without attention DP drop the pin.

Disaggregation engines ignore the pin: the `bootstrap_room` residue
(`room % dp_size`) governs prefill placement (and decode placement under
`round_robin`), so steer placement by minting the room instead. The engine
logs and ignores a conflicting pin rather than rejecting it.

With the bundled gateway, pass `--policy cache_aware --dp-aware` to
`ts serve` to enable per-rank affinity routing. This requires bundled smg
releases that carry the TokenSpeed dp-affinity support; see the lockstep
note in `serve_smg.py`.

## MoE deployments

Large MoE models usually choose one of these shapes:

- TP only: simplest startup path, often best for smaller MoE checkpoints.
- TP + EP: tensor parallelism within a replica, expert parallelism across ranks.
- DP + EP: multiple replicated decode groups with experts distributed inside each group.

Start with the recipe closest to your model family, then tune:

- `--tensor-parallel-size` or split TP values
- `--enable-expert-parallel`
- `--moe-backend`
- `--all2all-backend`
- `--deepep-mode`

With `--moe-backend flashinfer_trtllm`, TokenSpeed automatically pads NVFP4
expert shards to a multiple of 64 along their per-rank intermediate
dimension. The padding zero-fills the packed weights and block scales in the
tail, so the extra dimensions do not change the MoE result. For example,
TokenSpeed pads a 640-wide expert under MoE TP4 from 160 to 192 values per
rank. TokenSpeed zero-pads BF16 SiLU/SwiGLU expert shards the same way to a
multiple of 64 (128 when it cannot adapt the installed FlashInfer launcher,
or when FlashInfer's JIT cannot build the adapted one, for example without
nvcc). For example, Qwen3-30B-A3B's 768-wide experts need no padding under
MoE TP4 (192 per rank), while TokenSpeed pads a 96-wide shard to 128.

On Hopper, MXFP4 routed experts with a SiLU/SwiGLU activation and dense EP
(`--all2all-backend none`) default to the FlashInfer CUTLASS mixed-input
kernel (`flashinfer_cutlass`), which needs the per-rank intermediate size and
the hidden size to be multiples of 128. Other widths, SiTU experts and DeepEP
all-to-all layouts keep `marlin`. `--moe-mxfp4-fp8-activation` selects its
W4A8 variant.

### Kimi-K3 attention DP with MoE EP

With `none`, `agrs`, or `flashinfer` transport and attention DP greater than one,
Kimi-K3 requires
`attention DP == MoE EP == world size`. By default, shared experts and attention
projections are replicated. Only the routed experts require dispatch/combine
communication.

Select the transport with `--all2all-backend`:

- `none` (default): automatically use FlashInfer on NVIDIA ranks sharing a CUDA
  fabric, otherwise AG/RS.
- `agrs`: use reference all-gather dispatch and reduce-scatter combine.
- `flashinfer`: use FlashInfer MNNVL all-to-all; errors without the required
  CUDA fabric.
- `deepep`: Hopper Marlin supports attention TP combined with DP and pipeline
  parallelism. It requires attention CP=1, MoE TP=1/DP=1, and
  `MoE EP == attention TP * attention DP` within each pipeline stage. DeepEP
  owns routed dispatch/combine. Shared experts stay in the attention TP
  group.
  The construction-selected `KimiLinearMoEDeepEP` keeps this token ownership
  and its projection/shared-expert weight placement in one module.
  See [Kimi-K3 Hopper PD](../guides/kimi-k3-hopper-pd.md) for deployment layouts.

For NVIDIA NVFP4 checkpoints, `--moe-backend mega_moe` replaces routed
dispatch, SiTU expert computation, and combine with MegaMoE. It requires
`--all2all-backend none`, keeps routing weights after FC2, and returns BF16 outputs.

The AG/RS and FlashInfer transports quantize NVFP4 activations before dispatch and transfer their
block scales alongside the routing IDs and weights. Combine outputs remain BF16.

#### Tensor-parallel subgroups within DP

Shard selected dense weights across contiguous groups of DP ranks with these
independent settings. Set the same values on every worker before launch.
Unset or `1` keeps that component replicated. Non-integer values produce a
warning and use `1`. Workers must agree on the resulting sizes before
creating TP subgroups.

| Environment variable | Sharded computation |
|---|---|
| `TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE` | KDA QKV/gates and MLA QKV-A/output gate |
| `TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE` | KDA and MLA output projections |
| `TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE` | BF16 shared-expert MLP |

Each size must be a positive divisor of world size. Shared-expert TP must also
be smaller than world size and divide the MLP intermediate width. Keep attention
and linear attention at TP1/DPworld, routed MoE at TP1/EPworld, and pipeline
parallelism disabled. QKV and output projection sharding also require attention
head-TP at 1. DEP16 can use TP2, TP4, or TP8 subgroups. These settings do not
change attention caches or token ownership.

QKV and output projection sharding support BF16 and 128×128 block-FP8 weights
with BF16 activations, including the FP8 attention weights in an NVFP4
checkpoint. The sharding preserves checkpoint FP8 codes and scales. QKV TP
requires gated MLA with Q-LoRA. TokenSpeed rejects gated MLA without Q-LoRA
even with replicated projections. MLA Q-B, KV-B, KDA convolution and
norms keep their existing mapping. Target-model projection settings do not
change the draft model.

Each sharded computation restores complete outputs to the original token owner:

- QKV: AllGather → column-parallel projection → All-to-All.
- Output projection: All-to-All → row-parallel projection → ReduceScatter.
- Shared expert: AllGather → gate/up → activation → down → ReduceScatter.

Empty ranks still join collectives when peers have work. Uneven batches pad to
the subgroup's largest physical token count. The runtime prepares communication
buffers before CUDA-graph capture and reuses them across layers. Shared
experts use separate buffers. Shared AllGather finishes before routed
dispatch, shared GEMMs finish before routed BMM, and shared ReduceScatter
runs after dispatch and before combine. The runtime selects optimized
collectives for supported shapes and topologies, with generic collective
fallbacks otherwise. Place each subgroup
on a CUDA-IPC-accessible node to use the optimized paths.

TP4 reduces the selected weights to approximately one quarter per GPU, not
total model memory or latency. Communication and padding can outweigh GEMM
savings, especially for uneven traffic or large prefill batches. Compare TTFT,
decode latency and cache capacity separately with real weights and the full
model. A fixed memory budget can turn weight savings into more cache rather
than lower device usage. Reduced-layer tests do not establish full-model capacity.

#### DEP16 example with TP4 subgroups

This example enables all three settings. Set any one to `1` to keep that
component replicated, or set all three to `1` for the DEP16 baseline.

```bash
export TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=4

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

Launch one server process per node on four nodes with four GPUs each, using the
same checkpoint, container and environment. `HEAD_NODE` is the allocated head
node, not localhost. On Slurm, reserve a persistent allocation with `salloc`,
then launch through the site's `submit` wrapper or `srun`. The global sequence
limit above allows 16 requests per rank. For higher concurrency, increase both
the sequence budget and graph capture sizes and check cache admission. Keep
hardware, inputs and serving settings fixed when comparing TP configurations.

### DeepEP all-to-all

`--all2all-backend deepep` moves expert routing off all-gather and onto DeepEP
dispatch/combine. It requires a MoE backend whose kernels own those legs:
`--moe-backend deep_gemm` (block-scale FP8), `--moe-backend flashinfer_cutedsl`
(nvfp4, decode-shaped batches only), or `--moe-backend marlin` (MXFP4 W4A16
with BF16 activations, normal and low-latency legs). K3 on Hopper uses the
Marlin bridge with disjoint TP token slices. Its shared experts and restored
output stay within each attention-TP group. See the
[K3 Hopper PD guide](../guides/kimi-k3-hopper-pd.md) for PP4/TP8/EP8 prefill
and DP4/TP8/EP32 decode.

All DeepEP MoE plans reserve their persistent communication buffer during the
common `moe_process_weights` call, after backend weight preprocessing and before
KV cache memory profiling. This applies to Marlin, DeepGEMM and FlashInfer.
Non-DeepEP plans do not initialize DeepEP. The input hidden width and global
expert count come from the MoE module's declared geometry, never inferred from
packed weights. Dispatchers retain their backend-specific execution settings
and reuse the prepared buffer. Compatible layers share the process-wide buffer.
Incompatible geometry, mode or capacity fails through the same reuse checks as
runtime acquisition. The capacity remains the explicit server setting (256 by
default), with no model-name-based sizing in `ModelRunner`.

DeepEP has two sets of legs, and `--deepep-mode` picks between them:

| Mode | Legs | Fits |
| --- | --- | --- |
| `low_latency` | IBGDA dispatch into a preallocated per-expert buffer | Decode-shaped batches up to `--low-latency-max-num-tokens-per-gpu` |
| `normal` | High-throughput dispatch, tokens permuted into per-expert row blocks | Extend-shaped batches of any size |
| `auto` (default) | Both are allocated; each forward picks | Aggregated serving, which mixes both shapes |

For block-scale FP8 with `deep_gemm`, keep `auto` unless the instance only ever
sees one shape -- for example a decode-only worker in a PD split, which can pin
`low_latency` and skip the normal-mode buffers. The nvfp4
`flashinfer_cutedsl` kernel implements only the low-latency legs, so it requires
an explicit `--deepep-mode low_latency`. TokenSpeed rejects `auto` and
`normal` while building the execution plan. Every forward on such an
instance, including any prefill, must fit
`--low-latency-max-num-tokens-per-gpu`.

The runtime rejects a batch above the low-latency capacity rather than
truncating it. If decode plus speculative draft tokens exceed the capacity,
raise `--low-latency-max-num-tokens-per-gpu`. All current DeepEP MoE backends
require BF16 activations and do not support `--dtype float16`.

Each forward picks the mode from a value every rank agrees on, because the two
modes are different collectives. With DP attention that value is "every DP rank
is decoding", so one extending rank moves the whole group to the normal legs.

When DeepEP is selected, TokenSpeed disables the prefill CUDA graph:
normal-mode dispatch reports its per-expert receive counts to the host, and a
graph cannot capture a host sync. Decode graphs are unaffected.

For block-scale FP8 decode on NVIDIA, the low-latency path keeps routing
metadata in DeepEP's required contiguous int64/float32 formats across both
collective legs. Ordinary softmax routing selects experts and normalizes their
weights in one Triton launch before dispatch, instead of materializing the
full softmax and launching separate ATen top-k, reduction, and division kernels.
Its fused SwiGLU quantizer writes packed UE8M0 scales directly
in DeepGEMM's MN-major TMA layout, so padded rows need no zero-fill and the
second expert GEMM needs no separate activation-scale transpose/pack pass.
For sparse decode it launches a bounded number of row splits per expert and
walks only rows below the device-side expert count. Full-capacity workloads
retain one-CTA-per-row parallelism. The host-side expected-row estimate selects
between these mappings without synchronizing the expert counts to the CPU.
The runtime expands and packs expert weight scales once at weight load,
instead of ahead of both expert GEMMs in every layer forward. The runtime
queues shared-expert work between the dispatch send and receive legs to
overlap the collective whenever the model has a shared expert. Low-latency
dispatch asks DeepEP to produce packed UE8M0 scales directly in its
column-major TMA layout. Normal-mode
dispatch still transports FP32 power-of-two scales, but the existing expert
scatter packs them while permuting tokens, so neither mode needs a separate
sequence of elementwise shifts, fills, copies, and a transpose before GEMM1.

Dense `(128, 128)` FP8 projections have two scale contracts against
FlashInfer's FP8 block-scale GEMM, and both are copy-free for the layout they
own. The canonical K-major contract takes the quant kernel's `[M, K/128]`
activation scales and the checkpoint's `[N/128, K/128]` weight scales with no
layout conversion (the runtime normalizes strided scale views to contiguous
first).
The prepared MN-major contract, selected at load time on every Blackwell
datacenter part, transposes the weight scales once and then consumes the
TRT-LLM quantizer's native `[K/128, M]` activation scales directly, which is
what the canonical path would otherwise have to transpose on every call. Both
produce bitwise identical output.

MN-major requires `M` to be a multiple of four, so a prepared layer falls back
to the canonical contract once padding would cost more than the transpose it
saves — the fused padding quantizer grows with `M` while the transpose does
not. Decode row counts stay on the prepared path.

### Static expert placement with redundant experts

Real routing is skewed: a few experts of a layer draw many times their
uniform share of routes, and the rank holding them carries the layer's
critical path. `--ep-num-redundant-experts R` gives every MoE layer
`P = E + R` physical expert slots, `P / ep_size` per rank, and an *expert
placement* decides which logical expert each slot holds, so the placement
replicates a hot expert across ranks. TokenSpeed derives the placement from
recorded load with DeepSeek's EPLB algorithm — once at startup from a
profile record, or online from the live counters ([dynamic
rebalancing](#dynamic-expert-rebalancing) below). Models opt in explicitly
(`supports_expert_placement` on the model class) by building their MoE layers
from the placement and loading every placed slot — LongCat-Flash does.
TokenSpeed refuses the placement flags for any other model, so TokenSpeed
never installs a placement only to ignore it.
`--ep-num-redundant-experts` also needs expert parallelism (`ep_size > 1`):
replicas live on other ranks.

1. **Record** the load of a representative workload. Serve with
   `--expert-distribution-recorder-mode stat --ep-dispatch-algorithm
   static_with_zero_expert` (`static` for models without zero experts): the
   router counts every route into a per-layer, per-physical-expert counter
   on the device, and the `EXPERT_LOAD` profile activity frames the window.
   Only real tokens count: a padded replay (a decode graph at a ladder batch
   size, a prefill graph at a bucket, an idle attention-DP rank) feeds
   filler rows through the MoE layers, and the graph owners mark those rows
   in a device-side live-row mask before the replay and clear it after, so
   the counters see traffic, not padding. Startup forwards (autotune,
   warm-up, graph capture) are not traffic either: the runtime zeroes the
   counters once capture is done, so the first window -- a profile's or a
   rebalance's -- starts at the first served forward.

   ```bash
   curl -X POST localhost:8401/start_profile -H 'Content-Type: application/json' \
     -d '{"output_dir": "/tmp/expert-load", "activities": ["EXPERT_LOAD"]}'
   # ... serve the workload ...
   curl -X POST localhost:8401/stop_profile
   ```

   `stop_profile` writes one record per rank,
   `<id>-<rank tag>.expert-load.pt`, holding what *that rank's* router
   counted: `physical_count` (`[layers, slots]`, int64), the same summed
   over each logical expert's replicas as `logical_count`
   (`[layers, routed experts]`), the `physical_to_logical_map` that produced
   them and the rank's `ep_rank` / `ep_size`. **The serving path deliberately
   does not reduce the records**: a profile stop reaches attention-DP
   workers independently, and a rank running a collective inside the request
   would wait on a peer still synchronizing its scheduling round.
   `merge_expert_load_records` takes the sum over the EP group when it
   consumes the records: under all-to-all EP every rank counted its own
   tokens, so the merge needs every rank's record; under replicated-input EP
   every rank counted every token, so one record is complete and merging
   more only scales the counts uniformly, which the placement algorithm is
   blind to.

2. **Place**: restart with the redundant slots and the recorded load,
   pointing `--init-expert-location` at the directory (or a glob) of the
   records to merge them.

   ```bash
   --ep-num-redundant-experts 128 \
   --init-expert-location /tmp/expert-load \
   --ep-dispatch-algorithm static_with_zero_expert
   ```

   `P` must divide over the EP size (LongCat 2.0: 768 + 128 = 896 slots at
   EP128, 7 per rank). The loader fills every local slot from the logical
   expert the placement assigns it — one checkpoint tensor lands in each of
   its replicas, and RL weight sync through the same path does too.
   `--init-expert-location` also accepts a single record file (its
   `logical_count` alone). A file holding only a `physical_to_logical_map`
   pins a placement exactly. `--eplb-algorithm` selects the balancing
   algorithm (`auto` picks the hierarchical variant when the model's expert
   groups divide over the nodes). The merged record's per-layer
   *balancedness* (mean rank load over the busiest rank's load) is what to
   compare between placements.

3. **Route**. The router emits physical ids. Zero experts (LongCat) stay
   `-1` and never enter the tables. How a route picks among an expert's
   replicas follows the MoE kernel:

   - Under all-to-all EP (DeepEP) each rank routes only its own tokens, so
     every rank dispatches to its *nearest* replica — one on the same GPU,
     else on the same node (nodes taken from the EP group's actual ranks),
     else a seeded fair draw — through a static per-rank map
     (`--ep-dispatch-algorithm static*`; the `dynamic*` variants draw at
     random per route). The runtime computes the map only on this path.
   - Under replicated-input EP (every rank routes every token, such as the
     rl-bitwise aok path) exactly one rank must compute each route, so the
     replica is a pure function of the token:
     `replicas[logical, (token row + route rank) mod replicas]`, identical
     on every rank. TokenSpeed refuses random replica choice on this path.

   Under `--numerics rl-bitwise` the placement must be deterministic (static
   algorithms only) and drafts stay trivially placed. The MoE output remains
   a pure function of the token and its routes, so a placed server stays
   bitwise aligned with the trainer.

A placement is only as good as the recorded distribution's match to the
traffic it serves: record on production traffic and re-derive when it
drifts. Decode benefit depends on the kernel being row-bound. Where a grouped
GEMM streams every touched expert regardless of row count (small decode
batches), balancing the rows changes little.

### Dynamic expert rebalancing

`--enable-eplb` re-derives the placement while serving and moves the expert
weights to match, so the placement follows drifts in the traffic's expert
distribution without a restart. Every choice it depends on is explicit:

```bash
--enable-eplb \
--ep-num-redundant-experts 128 \
--expert-distribution-recorder-mode stat \
--ep-dispatch-algorithm static_with_zero_expert \
--eplb-rebalance-num-iterations 10000 \
--eplb-rebalance-layers-per-chunk 4
```

`--eplb-rebalance-num-iterations N` is the recording window: every `N`
forwards (real or DP-idle, so every rank counts the same) the runtime reads
the routing load since the previous snapshot, the EPLB algorithm derives a
new placement, and the weights move. `--eplb-rebalance-layers-per-chunk L`
bounds the stall each scheduling round pays: the rebalance switches the
model's MoE layers `L` per round, each layer's weights first and its routing
tables right after, so a forward never sees a layer whose tables and slots
disagree.
TokenSpeed allows `--ep-num-redundant-experts 0`: the rebalance then only
permutes experts across ranks. `--init-expert-location` still seeds the first
placement. `POST /rebalance_experts` starts one rebalance now (it replies
once it takes the load snapshot; the moves follow). TokenSpeed refuses the
`EXPERT_LOAD` profile activity under `--enable-eplb`, since the rebalance
owns the same counters. The engine log reports the balancedness each snapshot
saw and the one the new placement is expected to reach.

How a rebalance runs, and why it needs no new communication:

1. **Snapshot.** The runtime copies the counters to the host and zeroes them
   in one step on the execution stream, so the window boundary is exact at
   forward granularity. Under replicated-input EP (every rank routes every
   token) each rank's counters already are the group load, and the runtime
   compares the ranks' checksums — a difference means the ranks routed
   differently within a forward, a correctness bug, so the server fails
   rather than average the difference away. Under all-to-all EP (DeepEP) each
   rank counted its own tokens, and the runtime sums the load over the EP
   group.
2. **Compute and commit.** EP rank 0 derives the placement in a spawned
   CPU worker process, started at launch (the DeepSeek algorithm with stable
   tie-breaking and double-precision counts; off the serving process's GIL,
   so it does not slow the forward thread). 200 forwards later EP rank 0
   broadcasts the result over the EP group, and every rank plans the same
   slot moves from the old and new rows. Only the `[layers, slots]` map
   crosses the wire. Each rank rebuilds the inverse tables locally.
3. **Apply, one chunk per round.** Each rank receives every incoming slot
   into a staging buffer reserved at startup (one layer's worth, before
   startup sizes the KV arena) and sends every outgoing slot from its live
   tensor, in one batched P2P over the EP group. A same-GPU move goes
   through staging too, and a slot that needs an expert an earlier local
   slot just received copies it from there. Both ends order the P2P by
   logical expert id, so the pairs match on every rank by construction. The
   rebalance prefers a same-node source and spreads destinations evenly over
   the sources. The processed parameters move as bytes — quantized weights
   and scales included — so the rebalance re-derives nothing.

Every step is an internal control op completed through the same
attention-DP same-round gate as the RL weight ops, so all ranks switch a
layer in the same round and a weight update can never interleave with a
chunk (both are blocking device ops, one per round). A weight update between
chunks lands in the current placement: the loader reads the live map, and
the pending chunks copy whatever the source slots hold. An idle engine takes
no snapshot (the counter is frozen) and a pending commit waits for forwards.
The engine refuses a memory-saver release while a rebalance is in progress.

The routing path is unchanged: slot ownership stays contiguous per rank, the
replica choice is a pure function of the token, and a placement change alters
only table entries and slot contents. Under `--numerics rl-bitwise` the
MoE combine must be placement-independent (slot-order combine) for the
output to stay bitwise identical across a rebalance. The envelope folds
`--moe-combine-order slot` in, so it accepts the combination (see
`docs/design/numerics.md`).

## KV parallelism

Three context-parallel terms appear in this project, and they name different
things:

- **KVP** (KV parallelism) is a *storage* layout: the full-history KV cache
  pages are sharded page-cyclically over a consecutive subgroup of attention
  TP by virtual block id, so each rank stores `1/N` of every request's pages
  and the KV capacity per GPU grows by `N`. `--kv-parallel-size N` selects it;
  `mapping.attn.kvp_size / kvp_rank / kvp_group` describe it; the cache,
  scheduler feedback, Host tiers and PD transfer reason about it (page
  ownership, residue classes, owner translation, sharded page tables).
- **DCP** (decode context parallelism) is the *decode-side compute* over KVP
  pages: every rank keeps all of its query heads, attends only the pages it
  owns, and the partial outputs are merged across the KVP group by their
  log-sum-exp (LSE). The attention backends' decode algorithms, their kernels
  and the LSE-merge code carry the DCP name.
- **QCP** (query context parallelism) is the *prefill-side* sharding of the
  query rows over the attention TP group against the full, gathered KV
  history; there is no LSE merge. `--prefill-context-parallel-size` selects
  it (next section).

`--kv-parallel-size N` must divide `--attn-tp-size`; the KVP subgroups are
aligned runs of `N` consecutive attention-TP ranks. It applies to the
full-history groups of MLA/DSA models (latent KV and index-K) and to DeepSeek
V4's compressed KV; groups that every rank reads whole (sliding windows,
compressor state) stay replicated. The attention backend decides how it
attends the sharded pages (DCP decode, chunked or gathered prefill history);
the scheduler plans the same virtual blocks on every rank and the runtime
translates ownership where it reads, writes or copies them
(`docs/design/cache-concepts.md`). Allowed on aggregated engines and on the PD
prefill role (every rank of the subgroup sends the pages it owns to an
unsharded decode engine), with the Host KVStore and the retraction snapshot
pool; not yet on the decode role or with L3 storage. The full rule set,
including the per-backend speculation limits, is in
`docs/configuration/server.md`.

## Query context parallelism on the prefill role

`--prefill-context-parallel-size N` (QCP) splits every extend forward's rows
over the attention TP group of a PD prefill engine: rank `r` computes the
batch-global contiguous rows `[sum(c[:r]), sum(c[:r+1]))` of the chunk, with
`c` the same uneven split the reduce-scatter / all-gather communication path
uses, against the full KV history of its requests. It is a layout of the
`Mapping` (`mapping.attn.qcp_size / qcp_rank / qcp_group`), not a mode: the
scheduler plans the same chunk on every rank, cache allocation and the P->D
transfer see page ownership only, and `--chunked-prefill-size` keeps counting
the whole chunk, so size it as `N x rows-per-rank`.

Per layer the rows a rank holds are its shard: attention needs no gather or
reduce around it (its weights are head-replicated by default, or its own
head-TP tail returns the shard's rows, see below), the dense and MoE legs run
the all-gather / reduce-scatter path over the shard's row table, the KV write
all-gathers each rank's rotated latent to the whole span before the
owner-masked store, the sparse DSA indexer and attention score the gathered
history of each request group, and the model exit gathers only the sampled
rows (one per request) before the LM head. Every QCP collective is data
movement, so a row's bits do not depend on which rank computes it
(`docs/design/numerics.md`).

Requirements: `N == --attn-tp-size`, `--disaggregation-mode prefill`,
`--disable-prefill-graph`, attention DP 1, no `--enable-mixed-batch`, a
DSA-family attention backend (GPU DSA) with a bf16 KV cache, `--dense-tp-size`
and the MoE TP×EP group each 1 or `N` (attention returns complete rows, so
the drafter's replicated decode rows are never scattered and a narrower group
would have nothing to gather), and `--kv-parallel-size` 1 or `N`
(a KV-page-sharded engine may keep the Host KVStore and the retraction
snapshot pool -- every Host block sits in its Device block's residue class and
each rank copies the blocks it owns -- but not L3 storage, whose keys have no
owner-stable form under sharding). The drafter's extend step is sharded like
the target's; its decode steps run every row on every rank.

**Head TP over the query shards.** Without `--attn-head-tp-size` the shard
group's ranks hold different rows and every rank holds every head of
`q_b_proj`, `kv_b_proj` and `o_proj` (the mapping resolves
`attn.head_tp_size` to 1). `--attn-head-tp-size N` shards those three
projections by heads over the shard group instead — the same head group the
decode-side layout above builds over DP ranks, here `== qcp_group`, with the
same exchange around core attention: the ranks token-all-gather the
normalized q latent to the span, `q_b_proj` runs this rank's `H / N` heads of
every row, the heads-to-tokens all-to-all returns the shard's rows with every
head, the prologue rotates them on the shard's own positions and writes the
gathered KV as every QCP forward does, the sparse core attends the gathered
history with every head, the tokens-to-heads all-to-all hands the head slice
of every row to the local `w_vc`, and the `o_proj` tail returns the shard's
rows (row-parallel `o_proj` plus a token reduce-scatter, or under
`--tp-batch-invariant attn` the column-parallel `o_proj` with an all-gather
of the heads in front and an all-to-all back). The exchange counts are the
shard plan's, so there is no host sync. A rank whose shard is empty joins
every leg. The drafter's decode steps hold every row on every rank, exchange
nothing, attend this rank's head slice (the DCP arm over the KVP pages --
the page-sharded KV of `--kv-parallel-size` -- as attention TP
runs it) and all-reduce the `o_proj` partials (all-gather the
hidden shards under `--tp-batch-invariant attn`). The expanded (dense MLA)
prefill still refuses head TP; the layout serves the absorbed sparse prefill
only. The decode-only rules of the DP layout (role, decode-shaped autotune)
do not apply: the prefill role's rules above and
`--attn-head-tp-size == --prefill-context-parallel-size` gate it.

Memory: the head-shardable weights of one attention instance go from the
whole matrices to `1 / N` of them per rank. For a model with hidden 8192, 64
heads, `q_lora` 1536, `kv_lora` 512, `qk` 192 and `v` 128 that is `q_b_proj`
+ `kv_b_proj` + `o_proj` ≈ 94M parameters = 189 MB (bf16) per instance down
to ≈ 24 MB per rank at `N = 8`. A five-layer pipeline stage of a model with
two attention instances per layer saves ≈ 1.7 GB per rank, which the KV
cache takes.

Numerics: with row-parallel `o_proj` the shard group computes the same
per-rank head partials the plain TP8 prefill engine does and reduce-scatters
them where TP8 all-reduces. The bits agree only when both engines sum in the
same order. Under `--numerics rl-bitwise` the reduce-scatter always takes the
ordered fold while an all-reduce takes the in-switch reduction where
multicast reaches (an order that is per GPU set). Pin the TP8 engine to the
fold with `--force-deterministic-rsag` for the two to be bitwise. The QCP
engine's own drafter decode steps all-reduce `o_proj`
(replicated rows). `--force-deterministic-rsag` on it pins them to the fold
as well. With `--tp-batch-invariant attn` there is no cross-rank sum in
`o_proj`, and QCP is bitwise the TP1 / head-replicated form the decode
engine's batch-invariant layout computes (`docs/design/numerics.md`, "Layout
invariance of query context parallelism").

A preset for an eight-GPU prefill engine with head TP and the
batch-invariant `o_proj`:

```bash
tokenspeed serve <dsa-model> \
  --disaggregation-mode prefill \
  --attn-tp-size 8 --dense-tp-size 1 --enable-expert-parallel \
  --prefill-context-parallel-size 8 \
  --attn-head-tp-size 8 --tp-batch-invariant attn \
  --disable-prefill-graph \
  --chunked-prefill-size 16384
```

Drop `--attn-head-tp-size 8 --tp-batch-invariant attn` for the
head-replicated layout (the same bits, 189 MB more attention weight per
instance per rank), or `--tp-batch-invariant attn` alone for TP8's head
partials, summed in the fold's order.
`--tp-batch-invariant attn` needs an unquantized `o_proj`, as on the decode
side.

A model supports the layout by slicing its rows by `ctx.query_shard` and,
for head TP, threading the attention module's head-TP hooks around its
sparse core (`docs/design/unified_path.md`, "Query context parallelism").
Every other model refuses it at construction.

## Multi-node

Set these explicitly:

```bash
tokenspeed serve <model> \
  --nnodes 2 \
  --node-rank 0 \
  --nprocs-per-node 8 \
  --world-size 16 \
  --dist-init-addr <rank0-host>:25000
```

Each node must use the same model, backend, precision, and scheduler settings.
Only `--node-rank` should differ between nodes.

Run one `tokenspeed serve` per node. Node rank 0 serves the HTTP API. Higher
ranks run the engine only and expose no endpoint.

### Under a launcher

Inside a multi-node Slurm step, TokenSpeed derives `--nnodes`, `--node-rank`
and `--dist-init-addr` from the step environment when you do not give them,
so the same command line runs on every node:

```bash
srun --nodes=2 --ntasks-per-node=1 tokenspeed serve <model> --attn-tp-size 16
```

| Argument | Derived from |
| --- | --- |
| `--nnodes` | `SLURM_STEP_NUM_NODES` |
| `--node-rank` | `SLURM_NODEID` |
| `--dist-init-addr` | first host of `SLURM_STEP_NODELIST`, port 23456 |

Rules:

- An explicit `--nnodes` or `--node-rank` that contradicts the environment is
  an error, not an override. Omit the flag to accept the launcher's value.
- TokenSpeed always uses an explicit `--dist-init-addr` as given.
- Derivation only engages inside an `srun` step of more than one node. Outside
  a step — including the batch script of a multi-node `sbatch` — or in a
  single-node step, behaviour is unchanged: launch the ranks yourself and pass
  `--nnodes`/`--node-rank`/`--dist-init-addr`.
- If TokenSpeed detects a multi-node step but cannot resolve the topology,
  startup fails with the reason rather than falling back to a single node.
- The derived address is the one the head node's hostname resolves to. Where
  that is not the interface you want carrying bootstrap traffic, set
  `--dist-init-addr` explicitly.
- TokenSpeed sets `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` from the
  interface that routes to the head node, unless they are already present in
  the environment. Gloo needs this: it has no peer-address heuristic and
  otherwise binds whatever the local hostname resolves to, which is a loopback
  entry on many hosts. NCCL normally selects correctly on its own. TokenSpeed
  sets it for consistency.
- TokenSpeed does not set `NCCL_IB_HCA`. NCCL's own device selection prefers
  the higher-bandwidth InfiniBand devices and skips Ethernet-link ones.
- The rendezvous port is a fixed constant, not a function of `--port`. Every
  node has to arrive at the same port without talking to any other node, and
  under `tokenspeed serve` TokenSpeed allocates the engine's own port per
  node. The constant also stays clear of the kernel's ephemeral range, which
  startup checks. Pass `--dist-init-addr` to use a different port.

Apply the same NCCL transport and channel settings on every node as well. In
particular, do not mix IB and Socket selection or different
`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` values across ranks.

## Emulating rank 0 on one GPU

`--emulate-rank-zero` runs only global rank 0 of the configured layout, so you
can profile the per-rank work of a multi-GPU deployment on a single GPU:

```bash
tokenspeed serve <model> --tp 8 --emulate-rank-zero --load-format dummy
```

The layout resolves as usual (TP8 here), so the rank builds the weight shards,
kernels and cache sizing rank 0 of the full deployment would. Its collectives
are local stand-ins that return the real shapes and dtypes, filled from this
rank's operand alone.

Compared with the real rank:

- Communication takes no time, and the fused all-reduce paths are off because
  they need peers; their epilogues run as separate kernels.
- Values after a reduction are this rank's partial results, so outputs are not
  meaningful and data-dependent work such as MoE routing follows the emulated
  values.

A typical launch combines the flag with `--load-format dummy`, which reads
only the model's config and tokenizer, not its weights. With dummy weights,
also set `TOKENSPEED_MOE_ROUTING_SIMULATION=uniform` so tokens spread over
experts instead of all picking the same ones, and, with speculative
decoding, set `TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` to the `avg_accept_len`
of a real run.

TokenSpeed currently supports the flag on AMD GPUs, on one node, without
pipeline, context or attention data parallelism, an MoE TP x EP size other
than the attention TP size, `--mm-encoder-tp-mode data`, PD disaggregation,
fused all-reduce or an `--all2all-backend` transport.

## Runtime notes

Overlap scheduling can prepare the next forward on the CPU while the previous
forward's non-blocking host-to-device copies are still in flight. Any pinned CPU
staging buffer used for per-step model inputs must therefore have per-step
lifetime, or use an explicit synchronization before reuse. This applies to
MTP/GDN mamba state indices as well as token, length, and request-pool inputs.

CUDA IPC collectives are node-local. The mnnvl fabric workspace spans nodes.
`AutoBackend` serves single-tensor SUM all-reduces (16-bit, or fp32 where a
single-node workspace was armed for it; mnnvl serves 16-bit only) on groups
whose fan-in is 2, 4, 8, or 16 through the armed workspace -- one-shot
inside its traffic window, two-shot up to the workspace token capacity.
Cross-node groups arm the full two-shot capacity at startup. Single-node
groups start at the one-shot window and serve larger shapes once
model-level preparation widens the shared workspace. Other
fan-ins and dtypes, and any shape the workspace rejects, fall back to NCCL
(inside the trtllm backend for armed groups; the Triton all-reduce tier
serves AMD only, where no trtllm workspace exists). Single-node token
all-gather/reduce-scatter runs on the Triton RSAG backend and uses NCCL
across nodes. Logits all-gather and distributed argmax use the same
cross-node fallback. Layouts such as attention DP with dense TP or MoE EP
spanning nodes need this fallback.

Under `--numerics rl-bitwise` (`--batch-invariant-collectives`) the
all-reduce on a multicast-reachable group is the NVLS in-switch reduction
issued by the group's rank 0 through the same RSAG buffers, verified
bitwise at startup on every group it serves. AutoBackend keeps a kind whose
groups the switch reduces in different orders (several TP groups of three or
more GPUs) on the fold and logs it. The reduce-scatters, other payloads
and unreachable groups take NCCL data movement plus a rank-ordered fp32
fold, and the gathers stay on the multicast kernels. The distributed argmax
and the fused all-reduce kernels are off. `--force-deterministic-rsag` keeps
every collective on NCCL (reductions on the fold). `AutoBackend.route` is
the one place that decides. `docs/design/numerics.md` has the measured
basis.

On ARM systems, [NCCL 2.29.3](https://github.com/NVIDIA/nccl/releases/tag/v2.29.3-1)
fixes a weak compare-and-swap failure that can hang NCCL when it was compiled
with GCC older than 10. Affected NCCL builds older than 2.29.3 can exhaust proxy
operations during repeated multi-node CUDA graph replay. Use NCCL 2.29.3 or
newer for this configuration. Disabling CUDA graphs avoids the affected path,
but you do not need it with the fixed NCCL runtime.

## Validation

Before benchmarking:

- verify every rank starts and joins the distributed group
- verify the API responds before sending load
- confirm GPU visibility and process placement
- compare output correctness before tuning throughput
- keep the full launch command with benchmark results
