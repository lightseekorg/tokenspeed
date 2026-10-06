# Parallelism

TokenSpeed exposes familiar `--tensor-parallel-size` and `--tp` entry points
plus additional split parallelism controls for attention, dense, and MoE layers.

Scheduler process names in `ps` include their parallel ranks, for example
`tokenspeed::scheduler_tp1_ep3_dp0`. `tp` always identifies the attention TP
rank, including `tp0` for a single process. Other suffixes appear only when
their parallel size exceeds one: `ep` for MoE expert parallelism, `dp` and
`dcp` for attention data and decode context parallelism, and `pp` for pipeline
parallelism. These are zero-based ranks within their respective
groups, not parallel sizes.

## Quick Start

Use this form when the same tensor-parallel group is acceptable for the model:

```bash
tokenspeed serve <model> \
  --tensor-parallel-size 8
```

`--tensor-parallel-size` maps to TokenSpeed attention tensor parallelism and
cannot be used together with `--attn-tp-size`.

## Split Parallelism

Use split knobs when different layer families need different process groups:

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
permutation); the KV write and any sparse-attention indexer stay as they are.
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
it for it: the full-head K/V of a prefill is not available on the layout. The
layout over DP ranks is therefore refused outside `--disaggregation-mode
decode` (head TP over the query shards of a prefill engine serves its extend
rows through the absorbed sparse prefill instead, see below), and the decode
engine keeps every extend-shaped forward off its path:

- startup tunes on a decode step instead of the usual extend-shaped dummy
  forward, and the prefill CUDA graph (which records extend forwards) is
  turned off (`--disable-prefill-graph` is set, with a log line);
- the one prefill a decode node otherwise runs -- the local recovery after a
  capacity retraction (`docs/design/scheduler.md`, sections 2 and 4) -- is
  kept unreachable through the scheduler's own rule: a request whose declared
  generation fits one retraction safe-step window (4096 new tokens) has its
  whole generation reserved at admission and is never a retraction victim. A
  head-TP decode engine therefore admits only requests with
  `max_new_tokens <= 4096` (declared explicitly; an undeclared budget is the
  context remainder) and finishes any other with an abort error.

A prefill row reaching the attention is then an invariant violation and
raises, not a configuration the operator can hit.

`--lm-head-tp-size` under attention DP gathers the ranks' logits rows,
runs the vocab-shard GEMM and transposes the shards back to each rank's own
rows (`[T_own, V]`), so sampling and logprobs downstream see the same
full-vocab rows as a replicated head. On the decode path the row counts are
read from the per-rank token tables (every rank's logits rows are its decode
tokens, or the live rows a narrowing draft step reported), so the step has
no host sync; the shapes without a table -- a prefill's one row per request
or its logprob rows, a MIXED round, a model selecting its own logits rows --
exchange the counts. A drafter sharing the target's head must build its own
head on the same layout (the NextN, Eagle3-MLA and Llama-Eagle3 drafters do;
the others are refused with a clear error). It cannot combine with
`--dp-sampling`, and a request asking for prompt logprobs
(`logprob_start_len`) is refused at admission: that path pushes the prompt
rows through the LM head in per-request chunks, a per-rank number of row
exchanges the group cannot agree on.

`--tp-batch-invariant` replaces the two reduce-scatters of these layouts
(after `o_proj`, after the dense `down_proj`) with column-parallel GEMMs on
hidden: the reduction dimension (heads, intermediate channels) is
all-gathered, every rank computes its hidden shard of every token with full
K, and an all-to-all returns the rows. Every collective is then a pure
permutation of bytes, so the result is bitwise the full-K GEMM a TP1 or
replicated layer computes -- the point when a prefill engine with such a
layer must agree with the decode engine. It moves about the bytes of the
reduce-scatter it replaces and needs unquantized `o_proj` / `down_proj`;
`attn` requires head TP and `attn+dense` also requires a dense TP group wider
than attention TP (never under query context parallelism, whose dense group
is 1 or the attention TP width).

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
job. An invalid pin fails that request; the engine keeps serving. Engines
without attention DP drop the pin.

Disaggregation engines ignore the pin: the `bootstrap_room` residue
(`room % dp_size`) governs prefill placement (and decode placement under
`round_robin`), so steer placement by minting the room instead. A
conflicting pin is logged and ignored, never rejected.

With the bundled gateway, pass `--policy cache_aware --dp-aware` to
`ts serve` to enable per-rank affinity routing. This requires bundled smg
releases that carry the TokenSpeed dp-affinity support; see the lockstep
note in `serve_smg.py`.

## MoE Deployments

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

With `--moe-backend flashinfer_trtllm`, NVFP4 expert shards are automatically
padded to a multiple of 64 along their per-rank intermediate dimension. The
packed weights and block scales in the padded tail are zero-filled, so the
extra dimensions do not change the MoE result. For example, a 640-wide expert
under MoE TP4 is padded from 160 to 192 values per rank. BF16 SiLU/SwiGLU
expert shards are zero-padded the same way to a multiple of 64 (128 when the
installed FlashInfer launcher cannot be adapted, or when FlashInfer's JIT cannot
build the adapted one, for example without nvcc). For example, Qwen3-30B-A3B's
768-wide experts need no padding under MoE TP4 (192 per rank), while a 96-wide
shard is padded to 128.

On Hopper, MXFP4 routed experts with a SiLU/SwiGLU activation and dense EP
(`--all2all-backend none`) default to the FlashInfer CUTLASS mixed-input
kernel (`flashinfer_cutlass`), which needs the per-rank intermediate size and
the hidden size to be multiples of 128. Other widths, SiTU experts and DeepEP
all-to-all layouts keep `marlin`. `--moe-mxfp4-fp8-activation` selects its
W4A8 variant.

### Kimi-K3 attention DP with MoE EP

With `none`, `agrs`, or `flashinfer` transport and attention DP greater than one,
Kimi-K3 requires
`attention DP == MoE EP == world size`. Shared experts and latent projections
are replicated; only the routed experts require dispatch/combine communication.

Select the transport with `--all2all-backend`:

- `none` (default): automatically use FlashInfer on NVIDIA ranks sharing a CUDA
  fabric, otherwise AG/RS.
- `agrs`: use reference all-gather dispatch and reduce-scatter combine.
- `flashinfer`: use FlashInfer MNNVL all-to-all; errors without the required
  CUDA fabric.
- `deepep`: Hopper Marlin supports attention TP combined with DP and pipeline
  parallelism. It requires attention CP=1, MoE TP=1/DP=1, and
  `MoE EP == attention TP * attention DP` within each pipeline stage. DeepEP
  owns routed dispatch/combine; shared experts stay in the attention TP group.
  The construction-selected `KimiLinearMoEDeepEP` keeps this token ownership
  and its projection/shared-expert weight placement in one module.
  See [Kimi-K3 Hopper PD](../guides/kimi-k3-hopper-pd.md) for deployment layouts.

For NVIDIA NVFP4 checkpoints, `--moe-backend mega_moe` replaces routed
dispatch, SiTU expert computation, and combine with MegaMoE. It requires
`--all2all-backend none`, keeps routing weights after FC2, and returns BF16 outputs.

The AG/RS and FlashInfer transports quantize NVFP4 activations before dispatch and transfer their
block scales alongside the routing IDs and weights. Combine outputs remain BF16.

Kimi-K3 can independently shard its BF16 shared-expert MLP with
`TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE` (unset or `1` preserves existing
behavior). The size must divide world size and be strictly smaller than it;
DEP16 supports TP2, TP4 and TP8, with matching intermediate-channel divisibility.
AllGather and ReduceScatter restore local token ownership around
the sharded MLP. Attention and caches remain TP1/DP16; routed MoE remains EP16.
See the [shared-expert TP runbook](../recipes/kimi-k3-shared-expert-tp.md).

### DeepEP all-to-all

`--all2all-backend deepep` moves expert routing off all-gather and onto DeepEP
dispatch/combine. It requires a MoE backend whose kernels own those legs:
`--moe-backend deep_gemm` (block-scale FP8), `--moe-backend flashinfer_cutedsl`
(nvfp4, decode-shaped batches only), or `--moe-backend marlin` (MXFP4 W4A16
with BF16 activations, normal and low-latency legs). K3 on Hopper uses the
Marlin bridge with disjoint TP token slices; its shared experts and restored
output stay within each attention-TP group. See the
[K3 Hopper PD guide](../guides/kimi-k3-hopper-pd.md) for PP4/TP8/EP8 prefill
and DP4/TP8/EP32 decode.

All DeepEP MoE plans reserve their persistent communication buffer during the
common `moe_process_weights` call, after backend weight preprocessing and before
KV cache memory profiling. This applies to Marlin, DeepGEMM and FlashInfer;
non-DeepEP plans do not initialize DeepEP. The input hidden width and global
expert count come from the MoE module's declared geometry, never inferred from
packed weights. Dispatchers retain their backend-specific execution settings
and reuse the prepared buffer. Compatible layers share the process-wide buffer;
incompatible geometry, mode or capacity fails through the same reuse checks as
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
an explicit `--deepep-mode low_latency`; `auto` and `normal` are rejected while
the execution plan is built. Every forward on such an instance, including any
prefill, must fit `--low-latency-max-num-tokens-per-gpu`.

A batch above the low-latency capacity is rejected rather than truncated, so
raise `--low-latency-max-num-tokens-per-gpu` if decode plus speculative draft
tokens exceed it. All current DeepEP MoE backends require BF16 activations;
`--dtype float16` is not supported.

The mode is chosen per forward from a value every rank agrees on, because the two
modes are different collectives. With DP attention that value is "every DP rank
is decoding", so one extending rank moves the whole group to the normal legs.

The prefill CUDA graph is disabled when DeepEP is selected:
normal-mode dispatch reports its per-expert receive counts to the host, and a
host sync cannot be captured. Decode graphs are unaffected.

For block-scale FP8 decode on NVIDIA, the low-latency path keeps routing
metadata in DeepEP's required contiguous int64/float32 formats across both
collective legs. Ordinary softmax routing selects experts and normalizes their
weights in one Triton launch before dispatch, instead of materializing the
full softmax and launching separate ATen top-k, reduction, and division kernels.
Its fused SwiGLU quantizer writes packed UE8M0 scales directly
in DeepGEMM's MN-major TMA layout, so padded rows need no zero-fill and the
second expert GEMM needs no separate activation-scale transpose/pack pass.
For sparse decode it launches a bounded number of row splits per expert and
walks only rows below the device-side expert count; full-capacity workloads
retain one-CTA-per-row parallelism. The host-side expected-row estimate selects
between these mappings without synchronizing the expert counts to the CPU.
Expert weight scales are expanded and packed once when weights are loaded,
instead of ahead of both expert GEMMs in every layer forward. Shared-expert work
is queued between the dispatch send and receive legs to overlap the collective
whenever the model has a shared expert. Low-latency dispatch asks DeepEP to
produce packed UE8M0 scales directly in its column-major TMA layout. Normal-mode
dispatch still transports FP32 power-of-two scales, but the existing expert
scatter packs them while permuting tokens, so neither mode needs a separate
sequence of elementwise shifts, fills, copies, and a transpose before GEMM1.

Dense `(128, 128)` FP8 projections have two scale contracts against
FlashInfer's FP8 block-scale GEMM, and both are copy-free for the layout they
own. The canonical K-major contract takes the quant kernel's `[M, K/128]`
activation scales and the checkpoint's `[N/128, K/128]` weight scales with no
layout conversion (strided scale views are normalized to contiguous first).
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
placement* decides which logical expert each slot holds, so a hot expert is
replicated across ranks. The placement is derived from recorded load with
DeepSeek's EPLB algorithm — once at startup from a profile record, or online
from the live counters ([dynamic rebalancing](#dynamic-expert-rebalancing)
below). Models opt in explicitly
(`supports_expert_placement` on the model class) by building their MoE layers
from the placement and loading every placed slot — LongCat-Flash does. The
placement flags are refused for any other model, so a placement is never
installed only to be ignored. `--ep-num-redundant-experts` also needs
expert parallelism (`ep_size > 1`): replicas live on other ranks.

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
   warm-up, graph capture) are not traffic either: the counters are zeroed
   once capture is done, so the first window -- a profile's or a rebalance's
   -- starts at the first served forward.

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
   them and the rank's `ep_rank` / `ep_size`. The records are deliberately
   **not reduced on the serving path**: a profile stop reaches attention-DP
   workers independently, and a rank running a collective inside the request
   would wait on a peer still synchronizing its scheduling round. The sum
   over the EP group is taken when the records are consumed
   (`merge_expert_load_records`): under all-to-all EP every rank counted its
   own tokens, so every rank's record is needed; under replicated-input EP
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
   its replicas, and RL weight sync through the same path does too. A single
   record file is also accepted (its `logical_count` alone); a file holding
   only a `physical_to_logical_map` pins a placement exactly;
   `--eplb-algorithm` selects the balancing algorithm (`auto` picks the
   hierarchical variant when the model's expert groups divide over the
   nodes). The merged record's per-layer *balancedness* (mean rank load over
   the busiest rank's load) is what to compare between placements.

3. **Route**. The router emits physical ids; zero experts (LongCat) stay
   `-1` and never enter the tables. How a route picks among an expert's
   replicas follows the MoE kernel:

   - Under all-to-all EP (DeepEP) each rank routes only its own tokens, so
     every rank dispatches to its *nearest* replica — one on the same GPU,
     else on the same node (nodes taken from the EP group's actual ranks),
     else a seeded fair draw — through a static per-rank map
     (`--ep-dispatch-algorithm static*`; the `dynamic*` variants draw at
     random per route). The map is computed only on this path.
   - Under replicated-input EP (every rank routes every token, e.g. the
     rl-bitwise aok path) exactly one rank must compute each route, so the
     replica is a pure function of the token:
     `replicas[logical, (token row + route rank) mod replicas]`, identical
     on every rank. Random replica choice is refused on this path.

   Under `--numerics rl-bitwise` the placement must be deterministic (static
   algorithms only) and drafts stay trivially placed; the MoE output remains
   a pure function of the token and its routes, so a placed server stays
   bitwise aligned with the trainer.

A placement is only as good as the recorded distribution's match to the
traffic it serves: record on production traffic and re-derive when it
drifts. Decode benefit depends on the kernel being row-bound; where a grouped
GEMM streams every touched expert regardless of row count (small decode
batches), balancing the rows changes little.

### Dynamic expert rebalancing

`--enable-eplb` re-derives the placement while serving and moves the expert
weights to match, so a drift in the traffic's expert distribution is
followed without a restart. Every choice it depends on is explicit:

```bash
--enable-eplb \
--ep-num-redundant-experts 128 \
--expert-distribution-recorder-mode stat \
--ep-dispatch-algorithm static_with_zero_expert \
--eplb-rebalance-num-iterations 10000 \
--eplb-rebalance-layers-per-chunk 4
```

`--eplb-rebalance-num-iterations N` is the recording window: every `N`
forwards (real or DP-idle, so every rank counts the same) the routing load
since the previous snapshot is read, the EPLB algorithm derives a new
placement, and the weights move. `--eplb-rebalance-layers-per-chunk L`
bounds the stall each scheduling round pays: the model's MoE layers are
switched `L` per round, each layer's weights first and its routing tables
right after, so a forward never sees a layer whose tables and slots disagree.
`--ep-num-redundant-experts 0` is allowed: the rebalance then only permutes
experts across ranks. `--init-expert-location` still seeds the first
placement. `POST /rebalance_experts` starts one rebalance now (it replies
once the load snapshot was taken; the moves follow). The `EXPERT_LOAD` profile
activity is refused under `--enable-eplb`, since the rebalance owns the same
counters; the engine log reports the balancedness each snapshot saw and the
one the new placement is expected to reach.

How a rebalance runs, and why it needs no new communication:

1. **Snapshot.** The counters are copied to the host and zeroed in one step
   on the execution stream, so the window boundary is exact at forward
   granularity. Under replicated-input EP (every rank routes every token)
   each rank's counters already are the group load, and the ranks' checksums
   are compared — a difference means the ranks routed differently within a
   forward, a correctness bug that fails the server rather than being
   averaged away. Under all-to-all EP (DeepEP) each rank counted its own
   tokens, and the load is summed over the EP group.
2. **Compute and commit.** EP rank 0 derives the placement in a spawned
   CPU worker process, started at launch (the DeepSeek algorithm with stable
   tie-breaking and double-precision counts; off the serving process's GIL,
   so the forward thread is not slowed); 200 forwards later the result is
   broadcast over
   the EP group, and every rank plans the same slot moves from the old and
   new rows. Only the `[layers, slots]` map crosses the wire; the inverse
   tables are rebuilt locally.
3. **Apply, one chunk per round.** Each incoming slot is received into a
   staging buffer reserved at startup (one layer's worth, before the KV
   arena is sized) while each outgoing slot is sent from its live tensor, in
   one batched P2P over the EP group; a same-GPU move goes through staging
   too, and a slot that needs an expert an earlier local slot just received
   copies it from there. Both ends order the P2P by logical expert id, so
   the pairs match on every rank by construction. A same-node source is
   preferred and destinations spread evenly over the sources. The processed
   parameters move as bytes — quantized weights and scales included — so
   nothing is re-derived.

Every step is an internal control op completed through the same
attention-DP same-round gate as the RL weight ops, so all ranks switch a
layer in the same round and a weight update can never interleave with a
chunk (both are blocking device ops, one per round). A weight update between
chunks lands in the current placement: the loader reads the live map, and
the pending chunks copy whatever the source slots hold. An idle engine takes
no snapshot (the counter is frozen) and a pending commit waits for forwards;
a memory-saver release is refused while a rebalance is in progress.

The routing path is unchanged: slot ownership stays contiguous per rank, the
replica choice is a pure function of the token, and a placement change alters
only table entries and slot contents. Under `--numerics rl-bitwise` the
MoE combine must be placement-independent (slot-order combine) for the
output to stay bitwise identical across a rebalance; the envelope folds
`--moe-combine-order slot` in, so the combination is accepted (see
`docs/design/numerics.md`).

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
would have nothing to gather), and `--decode-context-parallel-size` 1 or `N`
(with `--disable-kvstore`, as DCP requires). The drafter's extend step is
sharded like the target's; its decode steps run every row on every rank.

**Head TP over the query shards.** Without `--attn-head-tp-size` the shard
group's ranks hold different rows and every rank holds every head of
`q_b_proj`, `kv_b_proj` and `o_proj` (the mapping resolves
`attn.head_tp_size` to 1). `--attn-head-tp-size N` shards those three
projections by heads over the shard group instead — the same head group the
decode-side layout above builds over DP ranks, here `== qcp_group`, with the
same exchange around core attention: the normalized q latent is
token-all-gathered to the span, `q_b_proj` runs this rank's `H / N` heads of
every row, the heads-to-tokens all-to-all returns the shard's rows with every
head, the prologue rotates them on the shard's own positions and writes the
gathered KV as every QCP forward does, the sparse core attends the gathered
history with every head, the tokens-to-heads all-to-all hands the head slice
of every row to the local `w_vc`, and the `o_proj` tail returns the shard's
rows (row-parallel `o_proj` plus a token reduce-scatter, or under
`--tp-batch-invariant attn` the column-parallel `o_proj` with an all-gather
of the heads in front and an all-to-all back). The exchange counts are the
shard plan's, so there is no host sync; a rank whose shard is empty joins
every leg. The drafter's decode steps hold every row on every rank, exchange
nothing, attend this rank's head slice (the DCP arm over the KVP pages --
the page-sharded KV of `--decode-context-parallel-size` -- as attention TP
runs it) and all-reduce the `o_proj` partials (all-gather the
hidden shards under `--tp-batch-invariant attn`). The expanded (dense MLA)
prefill still refuses head TP; the layout serves the absorbed sparse prefill
only. The decode-only rules of the DP layout (role, decode-shaped autotune,
the generation budget) do not apply: the prefill role's rules above and
`--attn-head-tp-size == --prefill-context-parallel-size` gate it.

Memory: the head-shardable weights of one attention instance go from the
whole matrices to `1 / N` of them per rank. For a model with hidden 8192, 64
heads, `q_lora` 1536, `kv_lora` 512, `qk` 192 and `v` 128 that is `q_b_proj`
+ `kv_b_proj` + `o_proj` ≈ 94M parameters = 189 MB (bf16) per instance down
to ≈ 24 MB per rank at `N = 8`; a five-layer pipeline stage of a model with
two attention instances per layer saves ≈ 1.7 GB per rank, which the KV
cache takes.

Numerics: with row-parallel `o_proj` the shard group computes the same
per-rank head partials the plain TP8 prefill engine does and reduce-scatters
them where TP8 all-reduces; the bits agree only when both engines sum in the
same order. Under `--numerics rl-bitwise` the reduce-scatter always takes the
ordered fold while an all-reduce takes the in-switch reduction where
multicast reaches (an order that is per GPU set), so the TP8 engine must be
pinned to the fold with `--force-deterministic-rsag` for the two to be
bitwise. The QCP engine's own drafter decode steps all-reduce `o_proj`
(replicated rows); `--force-deterministic-rsag` on it pins them to the fold
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
sparse core (`docs/design/unified_path.md`, "Query context parallelism");
every other model refuses it at construction.

## Multi-Node

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

Run one `tokenspeed serve` per node. Node rank 0 serves the HTTP API; higher
ranks run the engine only and expose no endpoint.

### Under a launcher

Inside a multi-node Slurm step, `--nnodes`, `--node-rank` and
`--dist-init-addr` are all derived from the step environment when they are not
given, so the same command line runs on every node:

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
- An explicit `--dist-init-addr` is always used as given.
- Derivation only engages inside an `srun` step of more than one node. Outside
  a step — including the batch script of a multi-node `sbatch` — or in a
  single-node step, behaviour is unchanged: launch the ranks yourself and pass
  `--nnodes`/`--node-rank`/`--dist-init-addr`.
- If a multi-node step is detected but the topology cannot be resolved,
  startup fails with the reason rather than falling back to a single node.
- The derived address is the one the head node's hostname resolves to. Where
  that is not the interface you want carrying bootstrap traffic, set
  `--dist-init-addr` explicitly.
- `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` are set from the interface that
  routes to the head node, unless already present in the environment. Gloo
  needs this: it has no peer-address heuristic and otherwise binds whatever the
  local hostname resolves to, which is a loopback entry on many hosts. NCCL
  normally selects correctly on its own; it is set for consistency.
- `NCCL_IB_HCA` is not set. NCCL's own device selection prefers the
  higher-bandwidth InfiniBand devices and skips Ethernet-link ones.
- The rendezvous port is a fixed constant, not a function of `--port`. Every
  node has to arrive at the same port without talking to any other node, and
  under `tokenspeed serve` the engine's own port is allocated per node. The
  constant also stays clear of the kernel's ephemeral range, which is checked
  at startup. Pass `--dist-init-addr` to use a different port.

Apply the same NCCL transport and channel settings on every node as well. In
particular, do not mix IB and Socket selection or different
`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` values across ranks.

## Emulating Rank 0 on One GPU

`--emulate-rank-zero` runs only global rank 0 of the configured layout, so the
per-rank work of a multi-GPU deployment can be profiled on a single GPU:

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

The flag is often combined with `--load-format dummy`, which reads only the
model's config and tokenizer, not its weights. With dummy weights, also set
`TOKENSPEED_MOE_ROUTING_SIMULATION=uniform` so tokens spread over experts
instead of all picking the same ones, and, with speculative decoding, set
`TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` to the `avg_accept_len` of a real run.

The flag is currently supported on AMD GPUs, on one node, without pipeline,
context or attention data parallelism, an MoE TP x EP size other than the
attention TP size, `--mm-encoder-tp-mode data`, PD disaggregation, fused
all-reduce or an `--all2all-backend` transport.

## Runtime Notes

Overlap scheduling can prepare the next forward on the CPU while the previous
forward's non-blocking host-to-device copies are still in flight. Any pinned CPU
staging buffer used for per-step model inputs must therefore have per-step
lifetime, or use an explicit synchronization before reuse. This applies to
MTP/GDN mamba state indices as well as token, length, and request-pool inputs.

CUDA IPC collectives are node-local; the mnnvl fabric workspace spans nodes.
`AutoBackend` serves single-tensor SUM all-reduces (16-bit, or fp32 where a
single-node workspace was armed for it; mnnvl serves 16-bit only) on groups
whose fan-in is 2, 4, 8, or 16 through the armed workspace -- one-shot
inside its traffic window, two-shot up to the workspace token capacity.
Cross-node groups arm the full two-shot capacity at startup; single-node
groups start at the one-shot window and serve larger shapes once
model-level preparation widens the shared workspace. Other
fan-ins and dtypes, and any shape the workspace rejects, fall back to NCCL
(inside the trtllm backend for armed groups; the Triton all-reduce tier
serves AMD only, where no trtllm workspace exists). Single-node token
all-gather/reduce-scatter runs on the Triton RSAG backend and uses NCCL
across nodes. Logits all-gather and distributed argmax use the same
cross-node fallback. This is required for layouts such as attention DP
with dense TP or MoE EP spanning nodes.

Under `--numerics rl-bitwise` (`--batch-invariant-collectives`) the
all-reduce on a multicast-reachable group is the NVLS in-switch reduction
issued by the group's rank 0 through the same RSAG buffers, verified
bitwise at startup on every group it will serve; a kind whose groups the
switch reduces in different orders (several TP groups of three or more
GPUs) is kept on the fold and logged. The reduce-scatters, other payloads
and unreachable groups take NCCL data movement plus a rank-ordered fp32
fold, and the gathers stay on the multicast kernels. The distributed argmax
and the fused all-reduce kernels are off. `--force-deterministic-rsag` keeps
every collective on NCCL (reductions on the fold). `AutoBackend.route` is
the one place that decides; `docs/design/numerics.md` has the measured
basis.

On ARM systems, [NCCL 2.29.3](https://github.com/NVIDIA/nccl/releases/tag/v2.29.3-1)
fixes a weak compare-and-swap failure that can hang NCCL when it was compiled
with GCC older than 10. Affected NCCL builds older than 2.29.3 can exhaust proxy
operations during repeated multi-node CUDA graph replay. Use NCCL 2.29.3 or
newer for this configuration. Disabling CUDA graphs avoids the affected path,
but is not required with the fixed NCCL runtime.

## Validation

Before benchmarking:

- verify every rank starts and joins the distributed group
- verify the API responds before sending load
- confirm GPU visibility and process placement
- compare output correctness before tuning throughput
- keep the full launch command with benchmark results
