# AMD LLM Kernels

## Attention

### DeepSeek V4 attention

The gfx950 package provides MXFP4 index selection, dense-workspace selected
prefill, and page-planar selected decode. Gfx1250 provides page-planar selected
decode. Decode reads a sliding-window (SWA) cache and an optional compressed
cache; both segments share one softmax, and the attention sink is applied once.

#### Contract

- The gfx950 MXFP4 indexers support 32 or 64 index heads of dimension 128,
  64-row pages, and top-k 512, 1024, or 2048. Prefill and decode return int32
  logical offsets; `dsv4_plan` preserves graph-stable sequence metadata.
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

Quantizers select the group32 E8M0 exponent in software and use CDNA4
`scaled_downcast` for rescaling and E4M3 conversion. They retain the existing
zero-group floor and infinite-scale behavior, including signed zeros.

The gfx950 EP8 SiTU registration with a 3072-wide intermediate selects a
coupled MXFP8 prefill pipeline through the existing `moe_plan`,
`moe_process_weights`, and `moe_apply` API. Both the normal `input`
activation policy and explicit `fp8` select it. Normal `input` retains
the route-direct A16 decoder for supported contiguous batches up to four tokens,
including the joint routed/shared decoder at `M <= 4`. Explicit `fp8` never
uses A16 decode. Other EP degrees, TP, activation policies and vendors retain
their existing registrations.

Token counts, quantizer row counts, mesh pitch and flat element counts are
runtime arguments without value or alignment specialization. Changing the batch size
reuses compiled kernels within the same dtype, stride, tile and address-safety
class; K, TOPK, expert counts and rank-local expert start remain specialized.
The non-fused A16 decoder does not specialize on its unused shared-output
program boundary. Joint routed/shared decode retains that boundary.
Flat mesh and output counts are computed on the host so wide counts bind as
64-bit integers without promoting ordinary small-batch loops.

For full K256 cells, preparation replaces the four checkpoint tensors with
one N16/K64-byte weight bank and N32/K256 packed scale words. Their physical
axes are explicit:

- Values: `[E, N/16, K/128, 4, 16, 16]`, two FP4 values per byte.
- E8M0 scales: `[E, N/32, K/256, 4, 16, 4]`, one scale per 32 values.
- W13 interleaves gate/up N16 blocks; W2 uses ordinary N16 blocks.

No linear or GDOT copy remains in the prepared module. Tensor clones,
same-shape views and device moves retain the physical axes; arbitrary
flattening is not a supported weight representation. The decoder validates
these axes, retaining its BF16 arithmetic and shared-output ownership.
The N16 direct consumer keeps weight and compact-scale loads within each compute
wave's output rows: W13 uses two N lanes for its two rows per wave, while W2
uses one N lane for its single row per wave. Exact BF16 upcasts convert to the
original 64-lane K layout before FP32 multiplication and reduction. Both stages
retain their original N/K tiles and wave counts, including the joint shared-down
branch. Partial K tiles keep their existing zero masks. Linear and GDOT banks
retain their existing layouts and tile choices.
The construction-time capability probe uses the
selected plan because checkpoint preparation occurs later.
Per-expert checkpoint updates retain the original loader's sharding and dtype
conversion through a temporary linear expert, then write back into the existing
N16 storage. No duplicate checkpoint bank is retained, and captured addresses
remain unchanged.

Runtime lifecycle coverage constructs a real `MoELayer` with the AMD `input`
override, processes its weights through the normal post-load method, and
uses the same expert module in `LatentMoELayer`. Standard prefill and small-M
decode finish through its RMSNorm/projection/add3 path; the joint decoder reads
the same N16 Parameters. Its existing contiguous joint-output contract remains.

MXFP8 batches up to 1024 rows use M32 expert blocks to limit per-expert
padding in short prefills; larger MXFP8 batches retain M128. This geometry
is selected after normal A16 eligibility:
non-joint batches above four rows use the scaled-MFMA pipeline instead of
the vector-reduction decoder. Explicit FP8 also uses M32 below five rows.
Prefill fuses output zeroing
into the scatter launch. The token/expert mesh stores slot bitsets so repeated
expert IDs remain distinct routes. At most 64 rows and 32 route slots use an
expert-owned mesh/count kernel followed by scatter; larger meshes retain the
four-phase sorter. Values are quantized once in token order;
only scales follow sorted routes. Both four-wave GEMMs use N128/K256
and A8-first M16 scaled MFMA fragments with statically unrolled K.
M32 uses two M16 accumulator repeats, one packed M32 scale group, BM-sized
metadata and two [32,256] activation buffers. Its stage1 acquires the next
M16 pair in the first two of four B-major phases, using the same async
global-to-LDS pipeline as M128. Stage2 retains register loads and consumes
the complete M32 pair before publishing the next tile. Stage grids are
bounded by the route count as well as the
global-expert capacity, retaining device-side valid-prefix checks.

At M32, each quantize/scale-sort pair is one launch with two disjoint CTA roles.
Value owners write each token or token-slot value once, including unreferenced
rows. Sorted-row owners recompute maxima and write distinct scale bytes;
duplicate expert routes never introduce concurrent writes to a value.
Each group32 maximum uses two lanes with 16 contiguous BF16 values per lane.
Padding scales are 127; invalid sorted rows do not read activations.
Scale-role CTAs wholly beyond the valid prefix return before activation loads
and reductions; partially valid CTAs retain their per-row masks. Value-role
CTAs still write every input row, including inputs with no local routes.
The rounded value and sorted-scale group extents must fit int32 indices,
including masked final-CTA lanes. Exact row counts remain runtime arguments.

M128 stage1
uses two async LDS slots and phased fragment acquisition. Invalid routes map
to an existing activation row; output masks keep those computations invisible.
Its DMA addresses apply XOR16 to the source column and write a linear physical
LDS view, avoiding masked zero fill and lane exchange for the XOR organization.
The even-K stage1 tail retains its repeated final-tile copy.
CTA base pointers for A, B and packed scale words remain fixed across K.
The phase-local load expressions add K progress to their intra-tile offsets,
in bytes for A/B and uint32 elements for packed scales. The buffer path casts
the proven-safe local offsets before adding K; ordinary loads retain 64-bit
addresses when the existing extent proof does not apply. This representation
does not require or imply automatic uniform `soffset` splitting.

M128 stage2 acquires K256 values in eight M16 portions. It prefetches two portions
in the tile prologue and acquires the remaining six during the second K128
half, bounding their live interval before publication. Partial loads return
compact tuples; publication maps tuple index zero to the requested physical
portion base. It publishes two portions
to the retired slot after the first current-tile MFMA and the remaining six
after the first four M16 consumers of the second K128 half. M16 pairs acquire
their K64 fragments before assembly into the A8 MFMA operands. All remaining
current-slot fragments are acquired before the next-slot publication barrier.
Fragment assembly keeps each lane's 16 bytes from each K64 half contiguous:
the join layout places the half selector after those bytes, retaining the
original K order and MFMA ownership. Its layout conversions assert that no
data movement is needed; both separate K64 acquisitions remain in place.
Compiler-only boundaries separate stage1 DMA/wait, global-load, two-M16-read
and B-major MFMA phases, and stage2's initial operand and publication prologues.
A packaged, always-inline LLVM wrapper emits `llvm.amdgcn.sched.barrier(0)`
without accumulator operands or hardware synchronization. Compiler-inserted LDS
dependency barriers provide inter-wave safety. Stage2 has no
artificial per-accumulator fences; its split publication chain is unchanged.
The last eight current-tile MFMAs consume those registers after the next
global-load prologue, before the following tile's dependent MFMAs; the final tile
drains this tail before the epilogue. Each accumulator still advances through
all K128 steps in order. Route weights are prefetched before the last K256
computation. The epilogue reads precomputed
row-byte offsets and uses a 32-lane column ownership for packed BF16 atomics.
Offsets use int32 only under the existing host extent proof, otherwise int64;
the latter adds 512 bytes of shared metadata, not an additional activation tile.
The BF16 CShuffle view reinterprets a retired A slot in place, without an
additional shared-memory tile.

The shared `tokenspeed_kernel_amd._scheduling` utility exposes `sched_barrier`
and `sched_barrier_compile_options` for AMD kernels. The textual library is
package data in `tokenspeed_kernel_amd/sched_barrier.ll`. Its SHA256 digest
is passed as the `SCHED_LIBRARY_HASH` constexpr, so content changes invalidate
the compiled-kernel cache without renaming the file or symbol. The digest is
computed once per process; restart after editing the library. No dependency
build or runtime compiler shim is required.

The prefill precision boundaries are:

1. Group32 E4M3/E8M0 quantization, with upward power-of-two scales and an
   amax floor of `1e-10`.
2. FP32 gate/up accumulators and FP32 SiTU, then a BF16 token-slot intermediate.
3. Group32 intermediate quantization with sorted scales, in one launch at
   M32 or separate quantize and scale-sort launches at M128.
4. FP32 stage2 accumulators multiplied by FP32 route weights, then BF16
   CShuffle and atomic combination.

This differs deliberately from the A16 decoder's intermediate rounding.
NaN payload identity is not an arithmetic contract. Padded and remote
intermediate rows are not observable. The source uses typed Gluon scale
operands; generated scale selectors and instruction scheduling are not implied
by the source-level fragment order.

Input and route strides are honored. Output requires nonoverlapping,
dword-aligned BF16 rows with unit inner stride for packed atomics. As with
other `out=` GEMMs, the caller must ensure its
elements do not overlap inputs. Disjoint strided workspace views may share an
allocation; no storage-alias scan runs during forward. Empty input returns
before launches. Widths outside complete K256 cells retain the existing
A8 bank and implementation; unclamped `input` retains the linear A16 bank.
Explicit FP8 requires both positive finite SiTU clamps.
