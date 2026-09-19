# AMD LLM Kernels

## Kernel Conventions

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
larger batches use 128-row tiles. With 32-row tiles, quantization and sorted-scale
production share a launch. Small route sets use a two-launch sorter; larger
route sets use four phases. Blocks beyond the valid routed prefix skip work.

Both GEMMs overlap loads with matrix computation using double-buffered shared
memory. Phased operand loading and scheduling barriers limit live registers;
compiler-inserted shared-memory barriers provide inter-wave synchronization.
