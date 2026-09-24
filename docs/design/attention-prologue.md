# Attention prologue

Everything between a layer's projections and its core attention is one
operation: optional per-head QK RMSNorm, RoPE (full, partial or multimodal),
quantization and the KV cache write. Models describe which steps apply to a
`PagedAttention` layer; they never apply its per-head norm, rotate, quantize or
write its new K/V rows themselves. An MLA model still normalizes its latent
and projects its absorbed query before the prologue. DeepSeek-V4 and V4.1
attention is not a `PagedAttention` layer and keeps its own fused steps over
its own cache groups.

## One entry per attention shape

`tokenspeed_kernel.ops.attention.prologue` exposes two operations, dispatched
through `select_kernel` like any other kernel family, and a helper built on the
first:

* `gqa_prologue(q, k, v, *, norm, rotary, cache, return_kv,
  solution, override)` for multi-head and grouped-query attention;
* `mla_prologue(query, q_pe, latent_cache, *, expanded, rotary,
  cache, solution, override)` for multi-head latent attention;
* `qk_norm_rope(q, k, *, head_dim, norm, rotary)` for keys that are not
  attention K/V but take the norm step (MiniMax-M3's indexer): the GQA kernels
  with no cache write. Indexers that only rotate call `embedding.rope`.

Head geometry comes from the cache descriptor and the input shapes, and the
storage format from the cache's rows and planes; no argument restates either.
Typed holders carry each optional step: `HeadNorm`, `Rotary` (with `MRope`),
`HeadKVCache` (with `MXFP8Scales`), `LatentKVCache` (with
`PerTokenHeadPlanes`) and `MLAExpandedKV`.

In the runtime, a layer states its steps once: `PagedAttention(...,
rotary_emb=..., qk_norm=...)`, with `None` for a step the layer skips.
`PagedAttention.forward` runs a GQA layer's prologue before core attention.
Flows that dispatch core attention themselves call `PagedAttention.prologue`
(GQA) or `PagedAttention.latent_prologue` (MLA). Passing `k = v = None` means
the inputs are prepared and the cache is written. A decode forward carries
exactly one row per write slot; `prologue` rejects any other count. A native
cache is bf16 (`--kv-cache-dtype auto`); an fp16 model's rows round to it at
the store, as the pool converted them on write, and its activations stay fp16.

## Accepted inputs

The entries validate each request once, from metadata only, and raise
`ValueError` when it breaks one of these rules:

* the inputs are fp16 or bf16 and share one dtype and one row count (q, k and
  v; or query, `q_pe` and the latent), and a native cache holds rows of that
  dtype, or bf16 rows for fp16 inputs (the fused solutions round them once from
  fp32; the composite casts its fp16 result, as the pool did);
* GQA query heads are dense and `head_dim` wide in 2-D or 3-D rows, keys and
  values are dense 2-D rows of the cache heads, and the key and value caches
  are rows of packed heads of one geometry and dtype;
* MLA inputs are dense in their channels: the query `[T, H, ·]`, `q_pe`
  `[T, H, rope]` and the latent `[T, kv_lora_rank + rope]`; an absorbed
  query's non-RoPE part is `kv_lora_rank` wide, expanded keys are as wide as
  the query's non-RoPE part and values `[T, H, ·]`, both in the query dtype,
  a dense cache row is `[slots, 1, kv_lora_rank + rope]`, and per-token-head
  planes are `[slots, 1, kv_lora_rank]`, `[slots, 1, 1]` fp32 and
  `[slots, 1, rope]` with one row count; `q_pe` is the query's RoPE channels
  themselves or shares no element with the query (the entry checks the first
  case's layout, not the second);
* no activation or cache tensor holds two elements at one address: each
  stride spans the dimensions with smaller strides (a size-1 dimension may
  carry any stride), since a solution may write any of them; activations start
  their rows and heads on 16-byte boundaries, which the CUDA kernels read in
  vectors;
* write slots are a dense 1-D int32 or int64 vector with at most one slot per
  token;
* positions are `num_tokens` dense int32 or int64 entries, or, for GQA, T/H/W
  rows whose M-RoPE sections are non-negative and split the rotary pairs; MLA
  takes no M-RoPE sections;
* the cos/sin table is a contiguous 2-D fp32 tensor of even width, at most
  `head_dim` for GQA and equal to the RoPE channels for MLA;
* MLA RoPE is 64, 128, 256 or 512 channels wide, or absent;
* norm weights are dense `[head_dim]` vectors;
* MXFP8 caches hold 128-wide FP8 heads, their scale pages span a positive
  multiple of 128 tokens, and their scale planes are dense e8m0 starting and
  ending on a 4-byte boundary, with one scale per 32 channels of every row.

A solution's own limits raise from the solution once it is selected; the one
in-tree case is the composite MLA store's `kv_lora_rank` rule in the
solutions table below. Every kernel a solution launches forms its row and
head offsets in 64 bits; the general `fp8_quantize` the composite quantizes
expanded values with keeps the baseline's 32-bit row offsets (2^31 elements
per activation).

## Numerics

In the fused solutions the steps compute in fp32, and every output (query,
key, value and cache row) is rounded once. A norm's weight arrives raw with an
fp32 offset, so Gemma's `1 + w` is never rounded to the activation dtype,
except by Ascend's norm, which forms it in the weight dtype as before. M-RoPE
selects each rotation pair's position row and then applies the ordinary
rotation, so text tokens get exactly the bytes of plain RoPE.

The `composite` solution chains the step kernels and rounds between them, as
models did before the prologue existed. It is the portable path: every GQA
layer on Ascend, and elsewhere the shapes the fused kernels do not cover, MXFP8
caches among them. On AMD and NVIDIA, without a norm and with a native cache of
the activation dtype, it too rounds once; Ascend's rotation rounds the cos/sin
table to the activation dtype first. A backend that attends the returned rows
instead of the cache (a prefill with no cached prefix) and casts them to an FP8
cache's dtype itself rounds them twice, as it did before, while the cache holds
them rounded once.

Solutions that round once produce the same bytes wherever two serve a request,
and tests compare them byte for byte. The CUDA and Triton `embedding.rope`
kernels, `embedding.rope_mla` and the Triton prologue kernels rotate a pair as
`fma(x1, cos, -x2 * sin)` and `fma(x2, cos, x1 * sin)`, the CUDA kernel's
association. The CUDA kernels (the fused RoPE write, and the composite's
rotation wherever `embedding.rope` or `rope_mla` selects one) differ in two
ways: they canonicalize NaN payloads and flush subnormal inputs, products and
results to zero, which the Triton kernels keep. A flush moves a result by less
than c * 2^-125 for a table whose entries are at most c (at least one: one for
plain RoPE, YaRN's mscale otherwise) in magnitude and hold no subnormals, or by
one unit in the last place of its output format when the flushed term carried
the exact value across a rounding boundary. A kernel that would round
differently declares traits that exclude those cases instead of being admitted
with a tolerance. Inputs may be overwritten.

Where a fused solution covers only part of a shape's range (a token-head
bound, full writes only, or absorbed attention only), a row's bytes can depend
on its batch. Crossing to the CUDA RoPE kernel changes the flushed subnormals
and NaN payloads: unnormed GQA layers between `triton` and `fused_rope`, and
MLA layers between `triton` and the composite, on NVIDIA. Crossing to a
composite that rounds between steps (the MLA composite on an FP8 cache on AMD,
or any composite writing fp16 rows into a bf16 cache) adds that rounding. Both
sides are at least as precise as the step-by-step path.

## Who writes the KV cache

The prologue writes every paged attention layer's KV; backends never do. Paged
callers pass `save_kv_cache=False`, and the router asserts it. The flag remains
for PD layerwise record timing and for state backends.

One exception: draft models that inject the target's context KV (DFlash,
DFlash2, and the Kimi-K3 and DeepSeek-V4.1 DSpark drafters) write those rows
directly, since they are target hidden states projected into the draft cache,
not a layer's attention inputs. Sparse-attention indexers' keys are not
attention K/V either. `test/runtime/test_paged_attention_prologue.py` lists
these writers and fails on any other model or drafter call of the step
kernels and modules it names.

Write slots come from `forward_write_locations(layer, mode)`. That is
`write_locations(layer, mode)`, except for a draft's first step over a MIXED
round, which carries every row whether it dispatches as MIXED or as decode:
the round's extend span, then the decode window.
Per-mode callers, such as MLA models that split a MIXED round, keep using
`write_locations`.

A pool describes its destination with `kv_write_target(layer_id, slots)`:
buffers, scale planes and whether the write sanitizes. Pools do not
override the prologue's write.

## Breakable prefill graphs

The KV write runs inside the same eager break as core attention, so a replayed
graph never reuses a recorded write location. A GQA layer's prologue runs
inside `PagedAttention.forward`'s `@break_point`, an MLA model's inside its
attention break (`_attn`, or the attention module's `forward`); the backends'
own breaks nest inside these and pass through, and remain for callers that
reach a backend directly. Prefill graphs capture the target model only and
replay a round with draft narrowing eagerly, so draft layers need no break.

## Formats and scales

KV caches run at unit scale. `require_unit_kv_scales` rejects a checkpoint KV
scale other than one, and `require_unit_kv_scale_file` does the same for an
FP8 KV cache's `--quantization-param-path`. No layer carries a KV scale.

An MLA layer reads its query in the activation dtype, except that an FP8
latent cache gets an FP8 query, and FP8 expanded keys and values, whichever MLA
backend serves the layer; a per-token-head cache keeps the activation dtype.

MXFP8 is a GQA format for 128-wide key and value heads. FP8 per-token-head is
an MLA format.

## MLA shapes

Absorbed attention reads the latent cache. Its query's non-RoPE part is
`kv_lora_rank` wide, and the prologue returns only the query.

Non-absorbed prefill attends per-head keys up-projected from the latent. It
passes `MLAExpandedKV(k_nope, value)`, and the prologue returns per-head keys
and values in the returned query's dtype.

## Solutions

`("attention", "gqa_prologue")`: per-head QK RMSNorm, RoPE (NEOX or GPT-J,
full or partial, or multimodal M-RoPE sections), then the K/V store into a
native, FP8 or MXFP8 cache; every step but the write is optional.

| Solution | Kernel | Covers |
| --- | --- | --- |
| `triton` | one launch, one program per head | AMD and NVIDIA, native or FP8 caches, every layer `fused_rope` does not take |
| `fused_rope` | `embedding.rope` with its fused K/V store; the key rotates in place | NVIDIA past 512 token-heads, head size 64/128/256/512, no norm, no M-RoPE, full rotary, native cache of the activation dtype or FP8 cache, full write |
| `composite` | `qk_rmsnorm`, `embedding.rope`, then the cache store | everything; rounds between steps; the path for MXFP8 caches and Ascend |

Without a norm, `fused_rope` overtakes the one-head-per-program Triton kernel
past 512 token-heads on NVIDIA, so it declares that bound.

`("attention", "mla_prologue")`: RoPE of the query and latent key parts,
FP8 quantization of the query for an FP8 cache, and the latent write into a
native, FP8 or per-token-head FP8 cache; expanded (non-absorbed) prefill also
returns per-head keys and values.

| Solution | Kernel | Covers |
| --- | --- | --- |
| `triton` | one launch that also assembles the query | absorbed, dense cache, full write, up to 32768 token-heads |
| `composite` | `embedding.rope` or `embedding.rope_mla`, then the latent store | everything on AMD and NVIDIA; its latent store needs `kv_lora_rank` a multiple of 256 below 512 written rows and a power of two above (every in-tree MLA model uses 512) |

## Adding a fused kernel

Register it under `("attention", "gqa_prologue")` or
`("attention", "mla_prologue")` with traits that cover exactly the cases where
it rounds like the solutions it competes with. Add a test that compares it
with them byte for byte over those cases, and one that shows it declines the
rest. A trait the kernel does not declare admits every value, so declare each
behavioral trait (`has_norm`, `mrope`, `return_kv`, `expanded` and the rest)
the kernel does not implement for every value. An `override` naming a kernel
this platform cannot run, or whose traits exclude the request, raises.
