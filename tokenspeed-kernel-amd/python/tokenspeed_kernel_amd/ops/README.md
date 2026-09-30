# AMD LLM Kernels

## Kernel Conventions

### Names

Profilers show kernel names after the `def` function `@gluon.jit` attached to,
so the kernel should carry the `tokenspeed-kernel` registration name verbatim
(e.g, `gluon_mm_mxfp8_gfx950`), and the Python launcher calling the kernel
should be named as `launch_<name>` (e.g., `launch_gluon_mm_mxfp8_gfx950`).
Companion kernels launched only by that op insert a role before the arch suffix
(e.g., `gluon_dsv4_decode_reduce_gfx1250`, `gluon_mha_prefill_sliding_gfx950`).
Kernels shared by several registered ops keep descriptive names. A `repr=` on
the jit decorator replaces the compiled symbol that profilers report, so its
base string must be the kernel's `def` name as well (the constexpr suffix it
appends is fine).

### Barriers

Do not write `gl.barrier()` for shared-memory (LDS) hazards. The Gluon
compiler's membar analysis tracks every LDS read, write, atomic, async copy,
and scratch-backed op (layout conversions, reductions, atomic result
broadcasts), and inserts a CTA barrier immediately before the first conflicting
access, including across loop back-edges. It also emits a barrier right after
every `async_copy.wait_group`/`tdm.async_wait`, and the lowering of a
`release`/`acq_rel` atomic emits one before it (an `acquire` atomic emits one
after it). A manual barrier next to any of these is a duplicate `s_barrier`, or
worse, it lands earlier than the compiler's minimal placement and pins the
instruction schedule.

Keep an explicit `gl.barrier()` only where the compiler cannot see the hazard:

- Ordering global-memory traffic across threads of one workgroup: init stores
  followed by an overlapping scatter, all-thread stores or atomics that must be
  issued before one thread bumps a `relaxed` counter, or re-reading a global
  buffer other threads just wrote. Say what the barrier orders in a comment.
- `load_shared_relaxed` pipelines. That load opts out of the compiler's
  async-copy hazard tracking, so the write-after-read against the next
  `buffer_load_to_shared` into the same slot is the kernel's responsibility.
  Place the barrier before the copy that reuses the slot.

Iris push collectives also keep explicit workgroup barriers around their
cross-rank publication protocol. The VMEM drain and system-scope atomics order
one subgroup's traffic, but the barriers join all producer subgroups before a
generation is published and all consumer subgroups before the peer inbox is
read. Removing either rendezvous can potentially increase cross-rank skew and
regress perf even when the generated kernel remains correct.

## GEMM

### gfx950 dense BF16 projections

The gfx950 package provides dense BF16 projection kernels, including an
eight-wave prefill kernel adapted from the gfx950 Gluon tutorials.

#### Contract

- The operation computes `A @ B.T` from K-contiguous BF16 matrices shaped
  `[M, K]` and `[N, K]`, producing BF16 output.
- Padded row strides and caller-owned outputs are supported when their inner
  stride is one. Quantization scales and block sizes are not supported.
- Automatic selection uses the prefill kernel when `2816 <= M <= 4096`, `M` is
  divisible by 256, and `(N, K) = (3072, 512)`. Other shapes retain the default
  PyTorch path. The small- and medium-M kernels remain available for direct use
  but are not registered for automatic selection.

#### Algorithm

One workgroup computes a `256 x 256` output tile in 64-wide K steps with eight
wave64s. Four `128 x 128` accumulator quadrants use native BF16 MFMA. The waves
divide both the global-to-LDS loads and the output quadrants.

Vectorized asynchronous copies stage A and B into padded, double-buffered LDS.
MFMA work on one buffer overlaps loading the next K tile into the other buffer.
The epilogue converts each accumulator quadrant to BF16 and stores it with
vectorized buffer operations. XCD-aware grouped tile ordering distributes
adjacent output tiles across the eight XCDs.

### gfx950 MXFP8 projection

The gfx950 package provides a prefill-oriented MXFP8 GEMM for DeepSeek V4.1
dense projections, with portable Triton fallback outside the tuned domain.

#### Contract

- The operation computes `A @ B.T` from K-contiguous E4M3 matrices shaped
  `[M, K]` and `[N, K]`; padded row strides are accepted.
- Scales are strided uint8 E8M0 matrices shaped `[M, K/32]` and `[N, K/32]`
  with an explicit `[1, 32]` scale block.
- Output is BF16 or FP16. A caller-owned output may have a padded row stride,
  but its inner stride must be one.
- The kernel requires `M` and `N` divisible by 256 and `K >= 512` divisible by
  256. Automatic selection further requires `M >= 1024`, `N >= 1536`, and
  `K >= 1024`.

#### Algorithm

One workgroup computes a `256 x 256 x 128` tile with eight wave64s. Four
`128 x 128` accumulator quadrants use native `32 x 32 x 64` E4M3 scaled MFMA.
Two K tiles are software-pipelined at a time, and phase-shifted MFMA and memory
stages implement the eight-wave warp-pipeline schedule. XCD-aware grouped tile
ordering spreads adjacent output tiles across the eight XCDs.

E4M3 values use vectorized asynchronous global-to-LDS copies into separate,
padded double buffers for A and B. Canonical row-major A scales use dword
asynchronous copies. Each B-scale copy combines both N quadrants and two K
steps in one LDS tile, then splits the four MFMA fragments in registers. Two
waves per EU avoid spills from the longer-lived fragments. Strided scales fall
back to direct fragment loads, and output uses vectorized buffer stores.

### gfx1250 MXFP8 decode projection

The gfx1250 package provides decode-oriented MXFP8 projections for DeepSeek V4
and V4.1, with portable fallback outside the tuned domain.

#### Contract

- The operation computes `A @ B.T` from K-contiguous E4M3 matrices shaped
  `[M, K]` and `[N, K]`, with `1 <= M <= 16` and `N` divisible by 16.
- DeepSeek V4.1 uses row-major uint8 UE8M0 scales shaped `[M, K/32]` and
  `[N, K/32]`, with `K >= 256` divisible by 32.
- DeepSeek V4 uses row-major FP32 activation scales `[M, K/128]` and canonical
  weight scales `[N/128, K/128]`, with `N` and `K` divisible by 128.
- Output is BF16. A caller-owned output may have a padded row stride, but all
  input, scale, and output inner strides must be one.

#### Algorithm

The direct path assigns one wave32 to an output tile of up to `16 x 16`.
Native TDM stages values, and for V4.1 scales, into padded triple-buffered LDS.
V4.1 uses scaled WMMA directly; V4 applies one FP32 scale pair to each
128-wide raw-WMMA partial before accumulation.

Measured long-K shapes use the same producers in split-K mode and a separate
FP32-to-BF16 reduction. One V4 Pro route combines adjacent output tiles in a
two-wave workgroup and uses fused A/B TDM loads. Other shapes remain on the
one-wave direct path when extra partitions or fusion do not pay for their
overhead. The kernel docstrings record the exact tiling, pipeline, and routing
decisions.

### gfx1250 dense BF16 decode projection

The gfx1250 package provides a small-M dense BF16 WMMA projection for K3
decode, including the KDA QKVFAB shape.

#### Contract

- The operation computes `A @ B.T` from contiguous BF16 matrices shaped
  `[M, K]` and `[N, K]`, with `1 <= M <= 32`, `K` divisible by 128, and `N`
  divisible by 16. A and B must share one CUDA device.
- Output is BF16. A caller-owned output must be on that device, with a unit
  inner stride and a row stride of at least `N`.
- KDA QKVFAB is the fixed shape `M` in `{1, 2, 4, 8, 16, 32}`, `K = 7168`,
  `N = 6288`.
- Callers must supply `split_k`; `None` selects the largest of 8, 4, or 2
  that divides the K tiles, leaves each split at least eight K tiles and one
  full TDM pipeline, and keeps `N`-tile count times the split within 256 CUs.
  Otherwise the launch is direct. An explicit `split_k` must be one of 1, 2,
  4, or 8 and divide the K tiles. A split greater than 1 must also leave each
  split at least one full TDM pipeline.

The K3 shared-down facade preserves a caller-owned row-strided output. Its
`auto` selector uses this kernel only for eligible GPU BF16 contiguous inputs;
otherwise Torch writes the same destination. Forced `torch` never dispatches
WMMA. Selectors requiring contiguous output reject a row-strided destination,
and unknown selectors always raise.

#### Algorithm

M is consumed in 16-row chunks, and each chunk re-reads B. General dense
projections use a two-warp `16 x 32` tile when `N` is divisible by 32 and has
at least 192 such tiles; other accepted `N` uses one warp and a `16 x 16`
tile. They use 256-wide K tiles when K is divisible by 256, falling back to
128-wide tiles otherwise, and request six TDM buffers. KDA QKVFAB keeps its
one-warp `16 x 16 x 128` tile and seven buffers. The buffer count is clamped
to the K tiles available per split. Once the K tiles fill that pipeline, the
tail lowers the TDM wait before each remaining LDS read. Short K dimensions
prefetch only valid tiles and drain each remaining pair before reading it.

Both paths accumulate in FP32 and round to BF16 once. Direct launches store
that conversion from the producer. Split-K launches write one FP32 partial
matrix per K partition, then a separate reduction sums those partials in FP32
and stores BF16. The live row count and split-buffer stride are runtime
arguments that do not specialize the producer or reduction, so warming a
projection covers other batch sizes with the same model dimensions.

The gfx1250 AttnRes launch has two fixed warp configurations: eight warps
below 256 tokens when mixing snapshots, and four otherwise. Normal startup's
prefill and decode warmups compile both before serving. Direct kernel callers
must warm both token ranges; disabling startup warmups can defer compilation
until the first call in an unwarmed range.

## Attention

### DeepSeek V4 attention

The gfx950 and gfx1250 packages provide MXFP4 index selection. Gfx950 also
provides dense-workspace selected prefill, while both architectures provide
page-planar selected decode. Decode reads a sliding-window (SWA) cache and an
optional compressed cache; both segments share one softmax, and the attention
sink is applied once.

#### Contract

- The gfx950 MXFP4 indexers support 32 or 64 index heads of dimension 128,
  64-row pages, and top-k 512, 1024, or 2048. Prefill and decode return int32
  logical offsets; `dsv4_plan` preserves graph-stable sequence metadata.
- The gfx1250 MXFP4 indexers implement the same logical contract for packed
  E2M1 values with one E8M0 scale per 32 elements. They accept padded page and
  block-table strides, reject invalid physical pages, and support caller-owned
  outputs and graph replay. Each page stores its packed key rows followed by
  the corresponding scale rows.
- The gfx950 prefill kernel accepts contiguous BF16 queries shaped
  `(tokens, heads, 512)`, a dense BF16 KV workspace, contiguous int32 selected
  indices and lengths, and a contiguous BF16 or FP32 sink. Registered selected
  widths are 384, 512, 640, 768, 1024, and 1152.
- The gfx950 decode kernel specializes for one to six tokens, 16 or 32 heads,
  128 SWA slots, 1024 compressed-cache slots, and 64-row pages. Both cache
  segments are required.
- The gfx1250 decode kernel accepts contiguous BF16 queries shaped
  `(tokens, heads, 512)`, uint8 page-planar caches, contiguous int32 slots and
  lengths, a contiguous BF16 or FP32 sink, and a contiguous BF16 output. It
  supports SWA-only and SWA-plus-compressed layers with independent page sizes.
- Each selected-decode cache page stores `page_size` 576-byte payloads followed
  by `page_size` eight-byte scale records. A payload contains 448 FP8 E4M3
  no-PE values and 64 BF16 RoPE values; the first seven scale bytes are E8M0
  exponents for the seven 64-element no-PE groups. Page strides may include
  padding.
- Negative slots are holes whose positions still count toward the scan length.
  Invalid slots, empty selections, partial tiles, and lengths outside the
  selected capacity do not read invalid cache rows. Unsupported traits use the
  portable implementation.

#### Algorithm

On gfx950, the indexer scores 256-candidate chunks with CDNA4 scaled MFMA and
reuses the DSA radix top-k reduction. Selected prefill uses CDNA4 asynchronous
buffer-to-LDS copies and double-buffered KV tiles; 64- and 128-head cases use a
64-head sparse kernel with a shape-selected 32- or 64-row tile. Selected decode
uses 16-head by 32-row tiles, four wave64s, and 18 fixed KV partitions. Its
second kernel combines the partial outputs and log-sum-exp values before
applying the sink.

The gfx1250 indexer scores 64 candidates at a time with native scaled E2M1
wave32 WMMA, accumulates weighted ReLU scores in FP32, and reuses the gfx1250
DSA radix top-k. Four waves cover 32 index heads; 64-head inputs reuse the same
key tile for a second WMMA group. Prefill and smaller decode workloads use
vectorized CDNA5 buffer loads. Larger decode workloads stage page-planar keys
and scales through native TDM into padded LDS. Double buffering and a
one-page-ahead software pipeline overlap these transfers with WMMA scoring
while keeping the transfer geometry aligned and the number of nearby TDM
operations bounded.

On gfx1250, decode fuses page-planar dequantization, BF16 wave32 WMMA attention,
FP32 online softmax, and output reduction. A workgroup covers 32 or 64 query
heads and 32 selected KV rows with four or eight waves. Shape-based KV
partitioning targets 256 workgroups and is capped by the number of KV tiles. A
single partition applies the sink and writes the output directly.

Padded LDS layouts avoid bank conflicts. On buffer-addressable inputs, a TDM
transfer stages the 1 KiB BF16 query row through separately created and updated
descriptors with clamped bounds. `warp_used_hint=0b00001111` selects one issuer
per SIMD in an eight-wave workgroup. Long, aligned partitions overlap native
global-to-LDS copies through two raw FP8 buffers; other geometries prefetch the
next dequantized tile into registers.

For `BLOCK_H=64`, `TILE_K=32`, and `HEAD_DIM=512`, one eight-wave workgroup is
resident per WGP, giving two wave32s per SIMD. The logical shared structures are
one BF16 Q tile, one BF16 dequantized KV tile, and, on the asynchronous path,
two raw FP8 buffers. Lifetime reuse keeps the physical LDS allocation unchanged.

### DeepSeek V4.1 CSA2 index selection

The gfx950 scorer uses scaled MXFP4 MFMA; gfx1250 dequantizes keys to BF16
and uses wave32 WMMA. Both accept 32 padded index heads of dimension 128 and
64-row, page-planar MXFP4 caches. Launch metadata reports score-capacity FLOPs
and estimated tensor traffic without reading device-resident sequence lengths.

The `tokenspeed-kernel` adapter owns query preparation, validation, and sorted
row/block selection. Gluon accepts one local or replicated shard with 1..32
heads and the 68-byte MXFP4 index format; sharded heads, wider head counts, and
132-byte FP8 index rows use portable Triton. Full selection scores the
configured page-table capacity without reading device lengths on the host.
Its query tile shrinks with history width to keep FP32 logits within 32 MiB
(at most 256 queries at 32K rows, 64 at 128K, and 8 at 1M). Reindex scores
at most the candidate-list capacity. Score CTAs honor the caller's row-chunk
bound up to the 256-row tuned maximum; masked 32-row hardware tiles cover
smaller bounds. Arena page strides are preserved without copying the full
cache; a non-unit stride between page bytes is normalized to contiguous
storage before scoring. Missing or out-of-range cache pages never contribute
rows or blocks, including the newest visible block. A valid newest block
remains eligible regardless of its score.

### gfx1250 MLA decode

`gluon_mla_decode_gfx1250` computes absorbed MLA decode over a paged cache,
one query token per sequence. The projected-value variant fuses the value
projection and optional sigmoid gate into its split merge.

#### Contract

- Queries are `(batch, 1, heads, 576)` (512 latent plus 64 RoPE dimensions)
  and the cache is a contiguous `(pages, 64, 1, 576)` of the same dtype: FP16,
  BF16, FP8 E4M3, or FP8 E5M2. The page table and cache lengths are Int32.
  `logit_cap` is unsupported, and the projected-value variant is FP8 only.
- The output is BF16 `(batch, 1, heads, 512)`, with an optional log-sum-exp
  in natural-log units. The projected-value variant writes BF16
  `(batch, heads * 128)` instead.

#### Algorithm

Each workgroup of two warps owns one sequence, a block of up to 16 query
heads, and one split of the keys, and walks one 64-key page per tile. TDM
loads the latent and RoPE parts of each page into LDS as two loads. The
splits write FP32 partials that a second kernel merges.

16-bit inputs load each tile into one LDS slot and wait for it before the
math. FP8 inputs use two slots and two waves per SIMD: the next tile's page
index and its two TDM loads go out before the current tile's math, and each
split issues its first tile before loading the queries. The loop is unrolled
by two so each slot index is a compile-time constant; waiting until two loads
remain then leaves the other slot's loads in flight.

### gfx950 MLA prefill

Two kernels compute dense, non-absorbed MLA prefill attention over ragged
sequences, `gluon_mla_prefill_gfx950` and `gluon_mla_prefill_8wave_gfx950`.
The 8-wave kernel handles 16-bit inputs as well, but is registered for FP8
only for now.

#### Contract

- Queries are `(tokens, heads, 192)` and keys are `(tokens, kv_heads, 192)`
  (128 no-PE plus 64 RoPE dimensions); values are `(tokens, kv_heads, 128)`,
  all FP16, BF16, FP8 E4M3, or FP8 E5M2 with one shared dtype and a contiguous
  last dimension. Query heads must be a multiple of KV heads.
- `cu_seqlens_q` and `cu_seqlens_kv` delimit the sequences. Causal masking
  aligns each query block to the end of its keys. `logit_cap` is unsupported.
- The output may be any caller-owned floating dtype with a contiguous last
  dimension; the optional log-sum-exp is FP32 in natural-log units.
- For FP8 inputs the 8-wave kernel requires `avg_kv_len_min=1024`: at least
  1024 key tokens per sequence on average, measured on Kimi-K3 prefill shapes.
  Selection rounds the average down to a power of two to bound its cache;
  this preserves the 1024-token cutoff exactly, including ragged batches.
- Both launch a persistent grid of 512 workgroups. Batch sizes, sequence
  lengths, and scheduler slots are runtime values, so varying ragged batches
  reuse the warmed binaries.
- Low-level launchers require explicit `is_causal`, `logit_cap`, and
  `return_lse`. `seq_lens_kv` and `max_seqlen_kv` are redundant hints that must
  agree with the authoritative `cu_seqlens_kv` lengths.
- Both share launch metadata that reports attention FLOPs and each tensor's
  bytes once without reading device-resident sequence lengths: FLOPs assume
  every sequence has the batch's average query and key length.

#### Algorithm

All paths use base-2 online softmax. `gluon_mla_prefill_gfx950` runs 16-bit
inputs with four wave64s over 128-row query blocks and 64-key tiles: fully
visible tiles double-buffer their asynchronous copies, and the diagonal band
and key tail run masked and unpipelined. Its FP8 path uses eight wave64s over
256-row blocks, 64-key tiles with the unscaled K=64 FP8 MFMA, splits each score
tile into two 32-column halves, and overlaps a tile's QK MFMA with the previous
tile's softmax and PV MFMA.

`gluon_mla_prefill_8wave_gfx950` follows the 8-wave warp-pipeline design:
each of eight wave64s owns 32 query rows of a 256-row block. Key tiles are 32
rows for 16-bit inputs and 64 for FP8, which keeps the working set within 256
VGPRs. Each loop iteration runs four clusters: K reads with the rescale, the
next tile's QK MFMAs with the current tile's softmax, V reads, and the PV
MFMAs with the next tile's row maximum. The two waves on a SIMD run one
cluster apart, so one wave's MFMAs overlap the other's memory work.

K and V stream into 4-slot LDS rings by asynchronous copies, K four tiles
ahead and V three. Tiles past the visible range load fully masked; buffer
loads zero-fill masked rows in LDS, so the loop needs no drain, and only tiles
crossing the causal diagonal or the key tail apply a score mask. For 16-bit
inputs the running maximum moves only when a tile maximum exceeds it by more
than 8 (base 2); FP8 keeps the exact maximum so P stays at most 1 before its
FP8 conversion. A wave skips the rescale when none of its rows moved. Empty
asm statements keep LLVM from moving each cluster's results across cluster
barriers.

### gfx1250 MLA prefill

`gluon_mla_prefill_gfx1250` computes the same dense, non-absorbed attention
as the gfx950 kernels, with WMMA and TDM.

#### Contract

- Queries, keys, values, dtypes, `cu_seqlens_q`, `cu_seqlens_kv`, causal
  alignment, and the unsupported `logit_cap` are the gfx950 entry's.
- The output is BF16 unless the caller passes a buffer, which may be any
  floating dtype with a contiguous last dimension. The optional log-sum-exp
  is FP32 in natural-log units.
- The grid is one workgroup per sequence, head, and 128-row query block.
  `max_seqlen_kv` and `seq_lens_kv` are ignored.

#### Algorithm

Four warps own 32 rows each of a 128-row query block and walk 64-key tiles
with base-2 online softmax. TDM streams the NoPE and RoPE halves of K and all
of V into double-buffered LDS, one tile ahead. Tiles below the first query
row's causal limit skip masking.

16-bit inputs keep Q in registers and run one wave per SIMD. Tiles at or past
the causal limit mask the scores and zero the value rows past the key tail.

FP8 inputs stage Q in LDS, split the NoPE QK into two 64-wide WMMAs, and run
three waves per SIMD. Fully visible tiles run in their own loop without mask
code, and the boundary tiles mask keys only, since rows past `q_len` are
never stored. V is not masked: TDM zero-fills tile rows past `kv_len`, and
those keys score `-inf`.

## Sampling

### Argmax

`tokenspeed_kernel.argmax` returns row-wise indices for `(M, N)` logits. AMD
Gluon kernels are selected automatically on gfx950 and gfx1250 when the optional
`tokenspeed-kernel-amd` package provides both implementations. If either import
is unavailable, the public API falls back to PyTorch.

#### Contract

- Kernel inputs are 2D FP16/BF16/FP32 GPU tensors with `N >= 4096` and unit
  vocabulary stride. Padded row strides are supported.
- Optional `out` is an int32/int64 tensor of shape `(M,)` on the input device;
  strided outputs are supported and returned directly. Without `out`, the
  operator allocates an int64 result.
- Ties choose the lowest index. NaNs are ignored; all-NaN rows return `-1`.
  Unsupported inputs fall back to `torch.argmax`, including its NaN semantics.
- Scratch is isolated by device and stream and reused across
  serialized calls. Graphs sharing warmed scratch must also replay serially.
  Warm up on the capture stream to avoid scratch initialization during capture;
  cold captures keep their allocations out of the eager cache.

#### Algorithm

Each workgroup loads a vocabulary tile and reduces `(value, index)` pairs.
Small batches split rows across workgroups. A GPU-scope acquire/release counter
publishes completion; the last workgroup reduces the partial results and resets
the counter in the same launch. Larger batches use one workgroup per row,
iterating over vocabulary tiles without atomic scratch traffic.

On gfx1250, split counts account for row count and vocabulary width, and FP32
tile widths are capped to limit register pressure. A bounded CPU cache reuses
configuration choices across calls. Split counts need not be powers of two;
the final reduction masks unused partial-result slots.

The gfx950 implementation uses CDNA4 buffer loads and 64-lane waves. The gfx1250
port uses 32-lane waves and buffer loads for split reductions and single tiles.
Larger rows use double-buffered TDM loads, overlapping the next tile's transfer
with per-lane candidate updates and reducing across lanes once per row. TDM's
zero padding is masked before comparison. Tile sizes account for element size
and batch size to limit shared-memory usage.

## MoE

### gfx950 latent input projection

The Kimi K3 prefill path projects one packed BF16 input weight into router,
routed-latent, and shared-expert inputs. Automatic selection uses the
small-batch Gluon kernel through 320 tokens, a mid-range Gluon tile for
321--1280, and the large-M Gluon tile from 1281 tokens. The portable packed
Triton kernel remains available for other shapes and devices.

#### Contract

- The input is contiguous BF16 with shape `[M, 7168]` for any `M >= 1`.
- Router `[896, 7168]`, routed `[3584, 7168]`, and shared gate/up
  `[1536, 7168]` weights must be consecutive row views of one packed allocation.
- Outputs are FP32 router logits, BF16 routed latents, and a BF16 768-wide
  shared input after SiTU. Positive gate clamp and optional linear clamp values
  are applied in FP32.

#### Algorithm

The mid-range path uses `128 x 128 x 64` tiles through 640 tokens and
`256 x 128 x 64` tiles from 641 to 1280 tokens. Both use eight waves, vectorized
loads, padded LDS layouts, and a three-buffer K pipeline to overlap data
movement with MFMA. The 256-row tile also orders workgroups to reuse weight
tiles within each XCD, starts each XCD's K loop at a different eighth of K and
wraps around, so the XCDs spread their activation reads over K instead of all
reading the same columns at once, and runs its leftover K tiles inside the
pipeline. The 128-row tile keeps the plain launch order and an unpipelined K
tail, which measured faster on MI355X. Column tiles follow the packed output
boundaries: router logits are stored as FP32, routed latents as BF16, and shared
gate/up pairs apply SiTU in registers before writing the BF16 shared input. Tail
rows are masked.

The large path uses an eight-wave `256 x 256 x 64` double-buffered MFMA/LDS
warp pipeline. Its 128-column accumulator halves route FP32 router and BF16
latent/shared stores independently, with tail rows and columns masked. Unlike
the mid-range path, it materializes BF16 gate/up values and applies SiTU in a
separate Gluon kernel.

### gfx1250 latent input projection

The same Kimi K3 packed projection as the gfx950 entry above, split into two
kernels because the shape changes character with the batch. Decode streams the
whole 86 MB weight past a handful of rows and is bound by memory; prefill is
bound by arithmetic.

#### Contract

Input, weight, and output shapes and dtypes are the gfx950 entry's. Two
things differ:

- Automatic dispatch selects the decode kernel for 1 to 32 tokens and the
  prefill kernel from 1536; the range between them retains the Triton kernel.
- The packed tensor passed alongside the three weights must be their
  consecutive view. The kernels read only it, so the launchers reject a
  packed tensor the weights do not live inside.

#### Algorithm

Decode splits the reduction across workgroups, eight ways to 16 tokens and
four above, because tiling the output alone leaves a 6016-column projection
with too few workgroups to keep the device busy. Each writes FP32 partials
that a companion kernel reduces and fans out to the three consumers. The
split is what bounds the kernel: its partial traffic scales with the token
count while the weight traffic does not.

Prefill instead runs the wide large-M WMMA schedule from
`gfx1250/gemm/fp16/mm.py`, a `256 x 256` tile on eight warps in 128-wide K
steps through a double-buffered TDM pipeline, with no split at all. The three
regions are written straight from the accumulator by one masked store each,
so FP32 router logits reach memory without a round trip through BF16. A tile
can straddle a region boundary, because the boundaries are 128-column aligned
while the tile is 256 wide, so the masks rather than the tile index decide
where a column belongs. SiTU is a second launch, as on gfx950: the gate and
up halves sit 768 columns apart, so no tile holds both.

Partial row tiles are masked in both kernels, so any token count is accepted.

### MXFP4 Sorted Experts

The gfx950 block-sorted path uses native scaled matrix instructions for MXFP4
activations and weights. Stage 1 fuses the gated activation and intermediate
quantization; stage 2 applies route weights and combines expert outputs.
Expert tiles support 16, 32, 64 and 128 rows. Sparse route counts select smaller
tiles to limit per-expert padding, while dense routes retain larger tiles to
amortize weight loads. This does not change activation precision or routing.

Scale storage keeps its 32-row CDNA4 layout even for 16-row expert tiles.
Each such tile reads and writes its own half of the scale panel; expert
boundaries need not coincide with panel boundaries.

For TP E2M1 inputs through 2048 tokens, stage 2 combines routed outputs directly
with BF16 atomics. Two adjacent columns per lane match packed atomic stores;
column-first workgroup order distributes concurrent updates across the output.
This avoids the per-route partials buffer and FP32 reduction kernel, at the cost
of order-dependent BF16 rounding. The destination is cleared on each call and
graph replay. Larger token counts retain the FP32 reduction path.

### MXFP8 SiTU Experts

On gfx950, the MoE API selects Gluon kernels with MXFP8 activations and MXFP4
weights for EP8 SiTU experts with a 3072-wide intermediate and supported clamp
settings. The `input` activation policy selects
BF16-activation decode for eligible batches of up to four tokens; explicit
`fp8` uses MXFP8 throughout.

Weight preparation interleaves gate/up weights and arranges weights and scales
for tiled loads. MXFP8 and BF16-activation kernels share one prepared
weight bank.

#### Algorithm

Starting from BF16 activations and precomputed top-k expert IDs and weights:

1. **Sort routes** into padded blocks for local experts, preserving repeated
   expert selections as distinct slots. Zero the output during route scatter.
2. **Quantize inputs** to E4M3 values with one E8M0 scale per 32 values.
   Values remain in token order; only scales are gathered into sorted-route
   order.
3. **Gate/up GEMM + SiTU** uses scaled matrix instructions and FP32
   accumulation, fusing the activation into a BF16 token-slot intermediate.
4. **Quantize intermediates** to MXFP8, keeping values in token-slot
   order and scales in sorted-route order.
5. **Down GEMM + weighted combine** accumulates in FP32, applies route
   weights, and atomically adds BF16 results into each token's output row.

Batches of up to 1024 tokens use 32-row expert tiles to reduce padding;
larger batches use 128-row tiles. With 32-row tiles, quantization and
sorted-scale production share a launch. Small route sets use a two-launch
sorter; larger route sets use four phases. Blocks beyond the valid routed
prefix skip work.

Both GEMMs overlap loads with matrix computation using double-buffered shared
memory. Phased operand loading and scheduling barriers limit live registers;
compiler-inserted shared-memory barriers provide inter-wave synchronization.

### gfx1250 MXFP4 Experts

On gfx1250, the MoE API selects Gluon kernels with FP8 activations and MXFP4
weights for precomputed top-k routing. Two expert GEMMs run per layer: a
gate/up GEMM with a fused SwiGLU or SiTU activation, then a down GEMM that
combines into each token's output row.

#### Contract

- Activations enter both GEMMs as E4M3 divided by a per-tensor FP32 scale,
  which the GEMM multiplies back into its FP32 accumulator. Weights are packed
  MXFP4 with one UE8M0 scale per 32 values.
- `y_global_scale` divides the result by a scalar before the epilogue casts it
  to the output dtype. It applies after bias and after the fused activation,
  so combining it with an FP8 `out_dtype` produces an activation the next GEMM
  can consume directly. The scale is a one-element FP32 tensor or a float, and
  is applied as a reciprocal multiply.
- Neither the epilogue nor the standalone activation quantizer clamps before
  the FP8 cast, so out-of-range values saturate the same way in both.
- `y_global_scale` is rejected on the combine path, which has no output-scale
  epilogue and would otherwise drop it silently.

#### Algorithm

Starting from BF16 or FP16 activations and precomputed top-k expert IDs and
weights:

1. **Route** the top-k selections into per-expert row slices, producing ragged
   metadata plus gather and scatter indices.
2. **Quantize inputs** to E4M3 by the gate/up activation scale, in one pass
   over the layer input.
3. **Gate/up GEMM** gathers routed rows, accumulates in FP32, applies bias and
   the fused activation, then divides by the down GEMM's activation scale and
   casts to E4M3 in the same epilogue. The intermediate therefore never lands
   in memory at a wider dtype, and rounds once rather than twice.
4. **Down GEMM + weighted combine** consumes that E4M3 intermediate,
   accumulates in FP32, and scatters into each token's output row, followed by
   the weighted top-k reduction.

The row tile is resolved from the gathered row count and expert count unless
the caller pins it. Ragged M and N edges are masked rather than peeled, so a
trailing partial tile loads only the rows that exist.
