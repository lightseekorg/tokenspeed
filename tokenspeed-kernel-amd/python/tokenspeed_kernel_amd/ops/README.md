# AMD LLM kernels

## Kernel conventions

### Names

Profilers show kernel names after the `def` function `@gluon.jit` attached to,
so the kernel should carry the `tokenspeed-kernel` registration name verbatim
(for example, `gluon_mm_mxfp8_gfx950`), and the Python launcher calling the
kernel should be named `launch_<name>` (for example, `launch_gluon_mm_mxfp8_gfx950`).
Companion kernels launched only by that op insert a role before the arch suffix
(such as `gluon_dsv4_decode_reduce_gfx1250`, `gluon_mha_prefill_sliding_gfx950`).
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
every `async_copy.wait_group`/`tdm.async_wait`; a `load_shared_relaxed` skips
only the barrier against the copy that wait covers, not the write-after-read
against a later copy refilling its slot. The lowering of a `release`/`acq_rel`
atomic emits one before it (an `acquire` atomic emits one after it). A manual
barrier next to any of these is a duplicate `s_barrier`, or worse, it lands
earlier than the compiler's minimal placement and pins the instruction
schedule.

Keep an explicit `gl.barrier()` only where the compiler cannot see the hazard:

- Ordering global-memory traffic across threads of one workgroup: init stores
  followed by an overlapping scatter, all-thread stores or atomics that the
  workgroup must issue before one thread bumps a `relaxed` counter, or
  re-reading a global buffer other threads just wrote. Say what the barrier
  orders in a comment.

Iris push collectives also keep explicit workgroup barriers around their
cross-rank publication protocol. The VMEM drain and system-scope atomics order
one subgroup's traffic, but the barriers join all producer subgroups before
they publish a generation and all consumer subgroups before they read the
peer inbox. Removing either rendezvous can potentially increase cross-rank
skew and regress perf even when the generated kernel remains correct.

## GEMM

### gfx950 dense BF16 projections

The gfx950 package provides dense BF16 projection kernels: an eight-wave
warp-pipeline prefill kernel, and a split-K tiled MFMA kernel for Kimi K3
decode batches.

#### Contract

- The operation computes `A @ B.T` from K-contiguous BF16 matrices shaped
  `[M, K]` and `[N, K]`, producing BF16 output.
- The kernels support padded row strides and caller-owned outputs when their
  inner stride is one. Quantization scales and block sizes are not supported.
- The prefill kernel takes any `M` and `N`; `K` must be a multiple of 128 and
  at least 256.
- Automatic selection uses the prefill kernel for the Kimi K3 projection shapes
  in `_PREFILL_SHAPES` when its workgroups, one 256x256 output block per CU,
  keep most CUs busy. Long reductions past two rounds of workgroups also need
  nearly every CU busy.
- The decode kernel takes contiguous inputs with `2 <= M <= 64` and writes BF16
  or FP32 (the MoE router). It runs only `(N, K)` and M buckets
  (`4, 8, 16, 32, 64`) with a measured config in `_DECODE_CONFIGS`; each
  entry beat `torch.mm` (hipBLASLt) by at least 4% in cold-cache sweeps at
  Kimi K3 TP8 shapes.
- `launch_gluon_mm_a16w16_decode_add3_gfx950` computes `A @ B.T + X + Y` in
  one launch, adding both BF16 addends to the FP32 accumulator before the
  single rounding. It reuses the decode tile for `M` and accepts only buckets
  without split-K (Kimi K3 latent up-projection, `2 <= M <= 32`), replacing a
  separate add kernel.
- Other shapes retain the default PyTorch path. The small- and medium-M
  kernels remain available for direct use, but the package does not register
  them for automatic selection.

#### Algorithm

One workgroup computes a `256 x 256` output tile in 64-wide K steps with eight
wave64s. Four `128 x 128` accumulator quadrants use native BF16 MFMA. The waves
divide both the global-to-LDS loads and the output quadrants.

Vectorized asynchronous copies stage A and B into padded, double-buffered LDS.
MFMA work on one buffer overlaps loading the next K tile into the other buffer.
The epilogue converts each accumulator quadrant to BF16 and stores it with
vectorized buffer operations. Partial tiles clamp out-of-range rows and
columns onto the last valid one and mask them out of the stores. XCD-aware
grouped tile ordering distributes adjacent output tiles across the eight XCDs.

Decode GEMMs are weight-bandwidth bound, so the decode kernel uses 16- or
32-row tiles over one- to three-buffer LDS pipelines and splits K when the
output alone has too few tiles to fill the GPU. Each split writes an FP32
partial through to memory, waits for its stores, and bumps a per-tile counter
with a relaxed atomic; the last split to arrive sums all partials in split
order with cache-bypassing loads, writes C, and resets the counter, so one
launch does both phases without an acquire-release fence and results are
deterministic. An optional XCD remap keeps the M tiles that
share a weight tile on one XCD's L2.

### gfx950 MXFP8 projection

The gfx950 package provides a prefill-oriented MXFP8 GEMM for DeepSeek V4.1
dense projections. Automatic selection uses it above 256 rows, and from 64
rows for weights of at least 2**26 elements (V4.1 25600x6144 and 5120x15360),
where streaming the weight once per row tile no longer pays off; other row
counts belong to the weight-bandwidth-bound decode kernel below.

#### Contract

- The operation computes `A @ B.T` from K-contiguous E4M3 matrices shaped
  `[M, K]` and `[N, K]`; padded row strides are accepted.
- Scales are strided uint8 E8M0 matrices shaped `[M, K/32]` and `[N, K/32]`
  with an explicit `[1, 32]` scale block.
- Output is BF16 or FP16. A caller-owned output may have a padded row stride,
  but its inner stride must be one.
- `M` is arbitrary and a runtime argument. `N` must be divisible by 16, and
  `K > 256` divisible by 32.

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
waves per EU avoid spills from the longer-lived fragments. Output uses
vectorized buffer stores.

Ragged shapes keep the hot loop free of masks:

- The last M or N tile is shifted back to end at the last row or column. It
  overlaps its neighbor, and both write identical values. Only `M` or `N`
  below one tile (a two-valued compile-time class) clamps load rows and
  masks the store.
- A K that is not a multiple of 256 starts the K walk below zero, so only the
  two prologue tiles are masked; their values zero-fill. Scale rows of such a
  K are not dword aligned. When the rounded walk is at most eight K tiles, the
  kernel stages all of its scales in LDS once (2^0 outside K). Longer walks and
  other unaligned scale strides load scale fragments directly.

Launches that leave most CUs idle with a long K walk (medium `M` with small
`N`) split K across workgroups. Each split writes FP32 partials, and
`gluon_mm_mxfp8_reduce_gfx950` sums them in split order. A measured cost model
picks the count and splits only within one wave of workgroups. The count is a
runtime argument.

### gfx950 MXFP8 decode projection

The gfx950 package provides the weight-bandwidth-bound MXFP8 projection used
for DeepSeek V4.1 decode (up to 256 rows); larger row counts, and 64 or more
rows on weights of at least 2**26 elements, use the prefill kernel above.

#### Contract

- The operation computes `A @ B.T` from E4M3 matrices shaped `[M, K]` and
  `[N, K]` with `1 <= M <= 256`, `N >= 16`, `K >= 128` divisible by 32, and
  `N * K < 2**31`. Inner strides must be one and row strides 16-byte aligned.
- Scales are uint8 UE8M0 matrices shaped `[M, K/32]` and `[N, K/32]` with
  unit inner stride and an explicit `[1, 32]` scale block.
- Output is BF16 or FP16. A caller-owned output may have a padded row stride,
  but its inner stride must be one.
- `M` is a runtime argument and is not specialized. The tile configuration
  comes from the smallest M bucket (16, 32, 64, 128, 256) covering `M`:
  measured per bucket for the V4.1 TP4 projection shapes, a heuristic
  otherwise. `K` is a compile-time model constant.

#### Algorithm

Each program owns a `BLOCK_M x BLOCK_N` output tile and one K slice, so the
weight is streamed from HBM once while the small activation stays in L2.
Small N tiles and an optional split-K fill the 256 CUs. The fully unrolled K
loop keeps a `NUM_BUFFERS`-deep LDS ring in flight with direct-to-LDS 16-byte
copies; their distributed layout follows the conflict-free padded LDS layout
so each warp writes one contiguous KiB. Each K tile's scale bytes are loaded
into registers just before its values, so the in-order wait for the values
also covers the scales, and native `16 x 16 x 128` E4M3 scaled MFMAs consume
them directly.

Activation rows past `M` are masked (no traffic), weight rows past `N` are
clamped, a K tail is zero-filled, and stores are masked. Split-K uses the
dense BF16 decode protocol above: write-through FP32 partials, a relaxed
arrival counter, and a reduction in split order by the last arrival, so
results are deterministic and the launch is HIP-graph safe. An XCD remap
keeps the M tiles that share a weight tile on one XCD's L2.

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
optional compressed cache; both segments share one softmax, with the attention
sink applied once.

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
smaller bounds. Selection preserves arena page strides without copying the full
cache, and normalizes a non-unit stride between page bytes to contiguous
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
split issues its first tile before loading the queries. The loop unrolls by
two so each slot index is a compile-time constant; waiting until two loads
remain then leaves the other slot's loads in flight.

### gfx950 MLA prefill

Two kernels compute dense, non-absorbed MLA prefill attention over ragged
sequences, `gluon_mla_prefill_gfx950` and `gluon_mla_prefill_8wave_gfx950`.
The 8-wave kernel handles 16-bit inputs as well, but the package registers it
for FP8 only for now.

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

### gfx950 MLA extend

`gluon_mla_extend_gfx950` computes absorbed MLA attention for cached extend
(chunked prefill over a prefix cache) directly on the paged latent cache.
`gluon_mla_extend_bf16_gfx950` registers the same launcher for a BF16 cache.

#### Contract

- Queries are `(total_q, heads, 576)`, already projected into the 512-dim
  latent space plus 64 RoPE dimensions; the cache is
  `(pages, 64, 1, 576)` and already holds the new tokens. Values are the first
  512 dimensions of the cache rows, and the output is BF16
  `(total_q, heads, 512)`.
- Supported `(q, cache)` dtypes are `(FP8 E4M3, FP8 E4M3)`,
  `(BF16, FP8 E4M3)` (the query is cast to FP8 on load) and `(BF16, BF16)`.
- Attention is causal: query `i` of a request sees keys up to
  `cache_len - q_len + i`. `logit_cap` and LSE output are unsupported.
- Registration limits `max_seqlen_q` to 256 for an FP8 cache and 16 for BF16.
  Absorbed attention spends 512 + 576 MACs per (query, key) pair and head
  against 192 + 128 for prefix replay, which also expands every prefix token
  once; these bounds are where the kernel's throughput stops paying for the
  extra MACs.
- The grid is ragged over `cu_seqlens_q` and the KV split count depends only
  on `max_seqlen_k` and the grid size, so batch composition and lengths are
  runtime values and reuse the warmed binaries.

#### Algorithm

Each program packs query positions of a head group into 128 MFMA rows, so a
KV tile read from LDS serves every packed head. Four waves each own 32 rows
and the full 512-wide FP32 accumulator, which keeps the kernel MFMA-bound
rather than LDS-bound. FP8 uses 64-key tiles (one page); BF16 uses 32-key
tiles.

KV tiles stream into a four-stage LDS ring with direct-to-LDS buffer loads,
swizzled so both the K reads and the transposed V reads are conflict-free.
The loop is software-pipelined one tile deep: the current tile's softmax runs
in the MFMA shadows of the next tile's Q @ K^T, and the next copy is issued
under the first P @ V chunk. The running maximum only moves when a tile
raises it by more than 8 (base 2), keeping P within FP8 range.

When the grid leaves CUs idle, the launcher splits the KV stream and
`gluon_mla_extend_reduce_gfx950` merges the partials by LSE. Splits fill at
most one wave of CTAs and stream at least four pages each, since every split
pays a fixed prologue and an FP32 partial store.

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

## Transform

### gfx950 Hadamard query transform

The operation applies a length-128 Hadamard transform with an explicit output
scale to contiguous BF16 query rows on gfx950, returning the same shape and
dtype. Empty inputs return an empty output of the same shape.

One 64-lane wave handles each row, keeping two FP32 values per lane during
seven add/subtract butterfly stages. Their order matches the portable
reduction tree so the BF16 results agree exactly. The output scale is fixed
for a compiled kernel; the launch grid supplies the number of rows.

## Sampling

### Argmax

`tokenspeed_kernel.ops.sampling.argmax` returns row-wise indices for `(M, N)`
logits. The public API selects AMD Gluon kernels automatically on gfx950 and
gfx1250 when the optional `tokenspeed-kernel-amd` package provides both
implementations. If either import is unavailable, the public API falls back to
PyTorch.

#### Contract

- Kernel inputs are 2D FP16/BF16/FP32 GPU tensors with `N >= 4096` and unit
  vocabulary stride. The kernels support padded row strides.
- Optional `out` is an int32/int64 tensor of shape `(M,)` on the input device;
  the operator supports strided outputs and returns them directly. Without
  `out`, the operator allocates an int64 result.
- Ties choose the lowest index. The kernels ignore NaNs; all-NaN rows
  return `-1`. Unsupported inputs fall back to `torch.argmax`, including its
  NaN semantics.
- The operator isolates scratch by device and stream and reuses it across
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
with per-lane candidate updates and reducing across lanes once per row. The
kernel masks TDM's zero padding before comparison. Tile sizes account for
element size and batch size to limit shared-memory usage.

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
boundaries: the kernel stores router logits as FP32 and routed latents as BF16,
and shared gate/up pairs apply SiTU in registers before writing the BF16 shared
input. The kernel masks tail rows.

The large path uses an eight-wave `256 x 256 x 64` double-buffered MFMA/LDS
warp pipeline. Its 128-column accumulator halves route FP32 router and BF16
latent/shared stores independently, with tail rows and columns masked. Unlike
the mid-range path, it materializes BF16 gate/up values and applies SiTU in a
separate Gluon kernel.

### gfx1250 latent input projection

The same Kimi K3 packed projection as the preceding gfx950 entry, split into
two kernels because the shape changes character with the batch. Decode streams the
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
steps through a double-buffered TDM pipeline, with no split at all. The kernel
writes the three regions straight from the accumulator with one masked store
each, so FP32 router logits reach memory without a round trip through BF16. A
tile can straddle a region boundary, because the boundaries are 128-column
aligned while the tile is 256 wide, so the masks rather than the tile index decide
where a column belongs. SiTU is a second launch, as on gfx950: the gate and
up halves sit 768 columns apart, so no tile holds both.

Both kernels mask partial row tiles, so they accept any token count.

### MXFP4 sorted experts

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
of order-dependent BF16 rounding. Stage 2 clears the destination on each call
and graph replay. Larger token counts retain the FP32 reduction path.

### MXFP8 SiTU experts

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

### gfx1250 MXFP4 experts

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
- The combine path rejects `y_global_scale`: it has no output-scale
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

The launcher resolves the row tile from the gathered row count and expert
count unless the caller pins it. The kernels mask ragged M and N edges rather
than peeling them, so a trailing partial tile loads only the rows that exist.

The large-batch router counts expert assignments in groups of at most 256
experts, using four warps in the counting stage. This reduces LDS atomic
contention when routing across many experts. The router counts duplicate
expert IDs separately and excludes invalid IDs from every group.
The subsequent prefix scan processes eight adjacent experts per four-warp
block, coalescing accesses to the row-major chunk-count buffer. Each expert
retains an independent scan over chunks, with partial expert groups masked.

### Causal MLA verification on gfx950

`gluon_mla_decode_fp8_query_blocks_gfx950` decodes 2–16 causal queries per
request using one page-table row and one cache length per request. The cache
length includes all queries in the block; query `i` sees
`cache_seqlens - q_len + i + 1` tokens. It requires E4M3 queries and cache,
512 latent plus 64 RoPE coordinates, and 64-, 128- or 256-token pages.
Non-causal draft blocks and single-query decode keep their existing kernels.

Each request selects a path using its length on the GPU, so one graph replay
can contain:

- Histories below the batch-size cutoff, with 64-token pages and at most
  16 heads, use the existing single-query scan when the launch has at least
  one workgroup per query. The launch divides workgroups among the query
  positions, and each scans a separate portion of the history for its query.
  Each thread merges up to sixteen partial results per query; larger split
  buckets distribute the reduction across threads.
- Requests with 64-token pages and `R = queries * heads` between 65 and 128
  can use one 128-row tile instead of two 64-row tiles. Each row represents
  one query position and one head. The launcher selects the larger tile when
  `ceil(ceil(length / 64) / splits) >= ceil(R / 4) + 8`, using its split count.
  The measured margin amortizes its extra output and reduction work over
  the saved KV loads.
- Other requests flatten the query and head dimensions into 64-row tiles,
  each of which loads a 64-token KV tile once for all its rows. This path
  also handles the page sizes and head counts unsupported by the existing
  single-query scan.

The existing FP8 MLA occupancy policy sets each tile's split count.
The single-query cutoffs are 12,288, 8,193, 5,377, 10,241, 4,609, 4,096,
4,096 and 5,121 tokens for batch sizes 1–8, respectively, and 4,096 otherwise.
The launcher passes the cutoff as a runtime scalar shared by the scan and
reduction.
Each request reserves partial-result storage for the larger split count,
with rows packed to the selected count's bucket. Its reduction reads only
the splits its scan wrote. The launcher buckets split counts to powers of
two, so nearby counts share one compiled kernel. The actual split counts,
page-table stride and softmax scale are runtime arguments. The reduction
combines partial attention outputs using their log-sum-exp values. Both FP8
scans multiply probabilities by 256 before the E4M3 cast to preserve small
weights, and divide out that factor at normalization. The projected-value API
applies the existing value projection to the latent output. Graph replay reads
updated page tables and lengths in place.

## Residual

### gfx950 mHC pre-mapping

The operation accepts BF16 residual streams shaped `[..., 4, hidden_size]`,
FP32 projection weights shaped `[24, 4 * hidden_size]`, three FP32 scales,
24 FP32 biases, normalization epsilons, and a Sinkhorn iteration count. It returns
a BF16 layer input
`[..., hidden_size]`, FP32 post coefficients `[..., 4, 1]`, and an FP32
combination matrix `[..., 4, 4]`. Optional output RMS normalization takes a
weight vector and its epsilon together.

The kernel flattens and projects the four streams with FP32 accumulation,
while their squared values provide the residual RMS normalization factor. The
kernel sums projection partials. Sigmoid transforms produce the pre/post
coefficients; a stable softmax followed by alternating Sinkhorn row and column
normalization produces the combination matrix. The pre coefficients mix the
four residual streams into the layer input, rounded to BF16 before optional
output RMS normalization. The kernel reuses staged projection weights across
token rows, and masks the final projection tile for arbitrary token counts.
