# AMD LLM kernels

## Kernel conventions

### Names

Profilers show kernel names after the `def` that `@gluon.jit` decorates,
so the kernel should carry the `tokenspeed-kernel` registration name verbatim
(for example, `gluon_mm_mxfp8_gfx950`), and the Python launcher calling the
kernel should be named `launch_<name>` (for example, `launch_gluon_mm_mxfp8_gfx950`).
Companion kernels launched only by that op insert a role before the arch suffix
(such as `gluon_dsv41_selected_attention_reduce_gfx950`,
`gluon_mha_prefill_sliding_gfx950`).
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
peer inbox. Removing either rendezvous can increase cross-rank
skew and regress performance even when the generated kernel remains correct.

## GEMM

### gfx950 dense BF16 projections

The gfx950 package provides dense BF16 projection kernels: an eight-wave
warp-pipeline prefill kernel, and a split-K tiled MFMA kernel for Kimi K3 and
Qwen3.8-Flash-Next decode batches.

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
  Kimi K3 TP8 and Qwen3.8-Flash-Next TP2 shapes.
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

Decode GEMMs are weight-bandwidth bound, so the decode kernel uses 16- to
64-row tiles over one- to three-buffer LDS pipelines and splits K when the
output alone has too few tiles to fill the GPU. When a program's K range has at
most eight tiles, it can instead give each tile its own LDS buffer and issue
every copy up front, waiting on one memory round trip rather than one per
refilled buffer. Each split writes an FP32 partial through to memory, waits for
its stores, and bumps a per-tile counter with a relaxed atomic; the last split
to arrive sums all partials in split order with cache-bypassing loads, writes
C, and resets the counter, so one launch does both phases without an
acquire-release fence and results are deterministic. The caller passes the
counters zeroed and gets them back zeroed (`gemm/splitk.py`);
`gluon_mm_a16w16_decode_num_counters_gfx950` sizes them. Tiles with at least
eight outputs per lane store C through a row-contiguous layout, 16 bytes per
lane. An optional XCD remap keeps the M tiles that share a weight tile on one
XCD's L2.

### gfx950 single- and few-row BF16 GEMV

`gluon_rowcta_bf16_gemv_gfx950` (`gemm/fp16/decode_gemv.py`) computes
`x @ w.T` for one or a few BF16 input rows with cold weights.

#### Contract

- `x` is a contiguous `[M, K]` matrix, `w` a contiguous `[N, K]` matrix and
  `out` a contiguous `[M, N]` destination, all BF16. `K` is a compile-time
  constant.
- `gemm.mv` takes any single row with `N, K >= 128`. `gemm.mm` hands a single
  row to this GEMV only for the `(N, K)` shapes in `_PREFERRED_SHAPES`.
  The dense decode GEMV routes its own measured single-row shapes to `mv`.
- Multiple input rows require an `(N, K)` bucket in `_ROWS_CONFIGS`. Each row
  is bitwise the single-row GEMV of that row; `M` is a runtime argument and
  each power-of-two bucket uses one binary.
- `gluon_gated_rowcta_bf16_gemv_gfx950` fuses per-head weightless RMSNorm and
  sigmoid gating into a single-row projection. It matches the separate
  Triton gated norm and GEMV bitwise. `gemm.mv_weightless_rmsnorm_gated_sigmoid`
  takes it for measured `_GATED_SHAPES` when the separate gated-norm and GEMV
  providers match.

#### Algorithm

A single-row program streams one weight row in `BLOCK_K` chunks and reduces
its FP32 partial sums once. `BLOCK_K` and the subgroup count follow `K`,
except for the measured `(N, K)` entries in `_CONFIGS`.

A few-row program streams a tile of weight rows, reusing each load across
its input rows. Each (input row, output) keeps the single-row reduction;
FMA throughput limits this path to a few rows. The gated projection
recomputes its normalized, gated input in every program.

### gfx950 FP32 router GEMV

`gluon_fp32_router_gemv_gfx950` (`gemm/fp32/router_gemv.py`) computes
`(x @ w.T) * multiplier` in FP32 for router decode batches, `1 <= M <= 16`,
with one program per output. `gemm.fp32_router_gemv_plan` selects it on
gfx950 and transposes the router weight once to `[N, K]`.

#### Contract

- `x` is a contiguous FP32 `[M, K]` matrix and `w` a contiguous FP32 `[N, K]`
  matrix. `M > 16` or `K > 16384` takes `torch.addmm` with `multiplier` as
  alpha.
- A row's logits are bitwise the same at every `M` the kernel serves.
- One binary per (shape, multiplier, power-of-two M bucket); `M` is a runtime
  argument.

#### Algorithm

A program stages its weight row in LDS with asynchronous `.cg` copies and
keeps one FP32 FMA partial per row of `x`; one reduction across lanes and
waves finishes the output. Every weight element is read once for all rows,
and the summation order depends only on `K`.

### gfx950 dense BF16 decode GEMV

`gluon_gemv_a16w16_gfx950` (`gemm/fp16/gemv.py`) computes `x @ w.T` for
decode batches, `1 <= M <= 16`, with one program per CU.

#### Contract

- `x` is `[M, K]` with unit inner stride, `w` a contiguous `[N, K]` matrix
  of `x`'s dtype (BF16 or FP16), `out` an `[M, N]` destination with unit
  inner stride; `K` is a multiple of 16 and at least 64.
- Automatic selection (`gemm.mv`, which `gemm.mm` takes for BF16 decode rows)
  uses it only for the `(N, K)` shapes and M buckets (1, 2-4, 5-16) listed in
  `_PREFERRED_BUCKETS`, where it beat the default (Torch's skinny GEMM at one
  row, the decode GEMM or hipBLASLt above) in cold-cache standalone sweeps.
  Direct calls take any supported shape.
- One binary per (shape, M bucket): M = 1, and M = 2-16. `M` and `N` are
  runtime arguments.

#### Algorithm

A program computes `rows = cdiv(N, CUs)` consecutive outputs over full K for
every row of `x`, padded to a power-of-2 block of `BLOCK_N` rows. Results do
not depend on `rows`: each output's summation order is fixed by the
constexprs alone.

- M = 1 (FMA tile): waves split the rows; a lane loads 16 bytes of every
  512-element K chunk, so one load instruction reads 1 KB of a row. Lanes keep
  FP32 partial sums per chunk position and reduce once at the end. With up to
  four rows per wave, the share streams its 512-wide chunks through a ring of
  four to eight LDS buffers (at most 64 KB) filled by direct-to-LDS copies,
  one refill in flight per consumed chunk; masked lanes rely on the
  hardware's out-of-bounds zero fill. The activation row sits in LDS too, so
  the ring shrinks at long K. The register tile (two chunks in flight, up to
  128 bytes per lane each) takes the rest: more rows per wave, where it
  measured 1-8% faster, and shares that leave fewer than four ring chunks.
- M = 2-16 (MFMA tile): a 16-row batched MFMA whose batch dimension is the
  wave, so the waves split every K iteration and reduce through LDS once;
  weights load K-contiguous and reach the operand layout through LDS. K >= 8192
  uses eight waves instead of four.
- Weight loads are streaming (`.cg`). K loops are unrolled: a loop-carried
  register copy of a pending load waits for the load and serializes the chunks.

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

#### FP32 block scales

Canonical block-scaled E4M3 GEMM computes `A @ B.T` from values shaped
`[M, K]` and `[N, K]`. With explicit block size `[128, 128]`, FP32 activation
scales have shape `[M, ceil(K/128)]`: one scale per activation row and
128-element reduction block. FP32 weight scales have shape
`[ceil(N/128), ceil(K/128)]`: one scale shared by each 128-by-128 weight block.
The values retain their canonical matrix layout; the scales are separate
FP32 tensors and need not be powers of two.

gfx950's scaled matrix instructions (MFMA) accept E8M0 scale operands, not
arbitrary FP32 scales. FP32 block scales therefore require separate scaling
of the unscaled block products; this is distinct from FP32 accumulation,
which MFMA supports. Rounding the scales to E8M0 would change the numerical
contract. Each block product receives its activation scale and then its
weight scale before accumulation. This order avoids requiring a product of
scales that could overflow or underflow even when the scaled block result
is representable.

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
- Products are exact and accumulate in FP32. Scale codes 1..254 are exact;
  code 255 gives infinity (as `exp2`) and code 0 gives 0 rather than 2^-127.
- `M` is a runtime argument and is not specialized. The tile configuration
  comes from the smallest M bucket (16, 32, 64, 128, 256) covering `M`:
  measured per bucket for the V4.1 TP4 projection shapes, a heuristic
  otherwise. `K` is a compile-time model constant.

The same decode machinery also accepts the canonical FP32 block-scale
contract described above, including partial final K128 groups. This form
requires BF16 output and applies each activation scale before its weight
scale to an unscaled FP8 K128 matrix partial; it does not convert coefficients
to E8M0 or inherit the native path's exact-upcast arithmetic. Scale strides
are unrestricted. Values require unit inner stride; unaligned rows use
ordinary loads into the same shared-memory ring instead of direct-to-LDS
copies. Neither form rearranges weights or creates a persistent weight copy.
Public FP32 selection uses decode for measured short and moderate reductions;
longer reductions retain the existing GEMM selection.
This selection policy does not restrict the raw decoder's supported shapes
or change native E8M0 selection.

#### Algorithm

Each program owns a `BLOCK_M x BLOCK_N` output tile and one K slice, so the
weight is streamed from HBM once while the small activation stays in L2.
Small N tiles and an optional split-K fill the 256 CUs. The K loop keeps a
`NUM_BUFFERS`-deep LDS ring in flight with direct-to-LDS 16-byte copies. Only
the ring's fill and drain are unrolled; the steady state is a runtime loop,
so code size and compile time do not grow with `K`. Each stage advances the
scalar base pointers, leaving the per-lane copy offsets loop invariant. The
copies' distributed layout follows the conflict-free padded LDS layout so
each warp writes one contiguous KiB. Each K tile's scales are loaded into
registers just before its values, so the in-order wait for the values also
covers the scales; dword-aligned scale rows load four groups per dword.
With a K tail, groups past `K` re-read the last real group's scale against
zero-filled values instead of masking the scale loads.

The math runs in BF16, one 32-value scale group per `16 x 16 x 32` BF16
MFMA. Each group's E4M3 values are read from LDS in the BF16 operand layout
and upcast to BF16 (`v_cvt_scalef32_pk_bf16_fp8`), which every E4M3 value
fits exactly, so the MFMA's products are exact and only its FP32 sum rounds.
A tile issues all of its group MFMAs first so their latency overlaps, then
multiplies each group's partial by the outer product of its row and column
scales and adds it to the FP32 accumulator. The scales are powers of two,
formed by placing the UE8M0 code in the FP32 exponent field, so the multiply
is exact.

Activation rows past `M` are masked (no traffic), weight rows past `N` are
clamped, a K tail is zero-filled, and stores are masked. Split-K uses the
dense BF16 decode protocol above: write-through FP32 partials, a relaxed
arrival counter, and a reduction in split order by the last arrival, so
results are deterministic and the launch is HIP-graph safe. An XCD remap
keeps the M tiles that share a weight tile on one XCD's L2.

### gfx950 dense MXFP4 linear (dynamic MXFP4 activations)

`gluon_mxfp4_linear_gfx950` and `gluon_mxfp4_swiglu_linear_gfx950` run the
dense layers (for example MoE shared experts) of MXFP4 checkpoints whose
activations are quantized to MXFP4 at runtime, from decode to prefill batch
sizes.

#### Contract

- `mxfp4_linear` computes `mxfp4(x) @ W.T` from a BF16 `x [M, K]` with unit
  inner stride and a packed E2M1 weight `[N, K/2]` (low nibble first) with
  uint8 UE8M0 scales `[N, K/32]` (unit inner stride). `N % 16 == 0`,
  `K % 64 == 0`, `N * K < 2**31`. Output is BF16 `[M, N]`; a caller-owned
  output may have a padded row stride.
- `mxfp4_swiglu_linear` takes a weight that stacks the gate rows `[0, N/2)`
  over the up rows `[N/2, N)` (`N % 32 == 0`) and returns BF16 `[M, N/2]`
  `silu(gate) * up`.
- The weight is stored in 16x16 MFMA-tile order (`preshuffle_mxfp4_weight`,
  the same order as the MoE `(16, 16)` tiles), prepared once at load by
  `prepare_mxfp4_linear`; the registry trait `weight_layout` keeps the
  prepared layout and the selected kernel consistent.
- Numerics match quantize-then-GEMM bit for bit up to FP32 summation order:
  each 32-value group of `x` gets the `mxfp4_quantize` scale (amax rounded up
  at the third mantissa bit, minus two exponents, clamped to `[0, 254]`) and
  round-to-nearest-even E2M1 values. The SwiGLU epilogue rounds gate and up
  to BF16, computes `gate * (1 / (1 + exp(-gate))) * up` in FP32 (the
  `silu_and_mul` order) and rounds to BF16; the down projection quantizes
  that output on load, as the unfused path did.
- `M` is a runtime argument and is not specialized. The tile configuration
  comes from the smallest M bucket covering `M`: measured per bucket for
  tuned `(N, K)` shapes, a heuristic otherwise. `N`, `K` and the weight-scale
  row stride are compile-time model constants.

#### Algorithm

Each program owns a `BLOCK_M x BLOCK_N` output tile and one K slice. A SwiGLU
tile holds `BLOCK_N / 2` gate columns and the matching up columns, so the
epilogue pairs them in registers; a ragged last tile clamps its weight rows
and masks its stores. Program ids put the N tile fastest, so the programs
that read the same activation rows are adjacent and share an XCD's L2 after
the XCD remap.

The K loop double-buffers in registers: the next tile's loads are issued
before the current tile is consumed, and the loop body handles two tiles so
the buffers swap roles instead of being copied. Activations load coalesced,
eight BF16 values per lane with consecutive lanes on consecutive 16-byte
runs and rows spread over all warps, so no warp repeats another's
quantization work. A group's amax is a packed 16-bit integer max of the
sign-cleared BF16 bits across the four lanes holding it (`v_pk_max_u16`,
two DPP steps), and `v_cvt_scalef32_pk_fp4_bf16` divides by the scale,
rounds and packs. The packed tile and its scales then move through LDS into
the scaled-MFMA operand layouts. The preshuffled weight feeds the B operand
directly from global memory, one contiguous KiB per warp load.
`v_mfma_scale_f32_16x16x128_f8f6f4` accumulates in FP32.

Activation rows past `M` are clamped to the last row instead of masked (the
epilogue drops them), and a K tail zero-fills both operands. Small batches
split K with the MXFP8 decode protocol (write-through FP32 partials, a
relaxed arrival counter, reduction in split order by the last arrival), so
results are deterministic and the launch is HIP-graph safe.

### gfx1250 MXFP8 decode projection

The gfx1250 package provides decode-oriented MXFP8 projections for DeepSeek V4
and V4.1, with portable fallback outside the tuned domain.

#### Contract

- The operation computes `A @ B.T` from K-contiguous E4M3 matrices shaped
  `[M, K]` and `[N, K]`, with `N` divisible by 16.
- DeepSeek V4.1 uses row-major uint8 UE8M0 scales shaped `[M, K/32]` and
  `[N, K/32]`, with `1 <= M <= 16` and `K >= 256` divisible by 32.
- DeepSeek V4 uses row-major FP32 activation scales `[M, K/128]` and canonical
  weight scales `[N/128, K/128]`, with `1 <= M <= 64` and `N` and `K`
  divisible by 128. The row count is a runtime argument; a full 16- or
  64-row tile compiles without row masks, so all batch sizes share two
  binaries per projection shape and route.
- Output is BF16. A caller-owned output may have a padded row stride, but all
  input, scale, and output inner strides must be one.

#### Algorithm

The direct path assigns one wave32 to an output tile of up to `16 x 16`.
Native TDM stages values, and for V4.1 scales, into padded triple-buffered LDS.
V4.1 uses scaled WMMA directly; V4 applies one FP32 scale pair to each
128-wide raw-WMMA partial before accumulation. V4 batches of 17 to 64 rows
run as one 64-row tile of four waves, so each weight tile streams once for
the whole batch.

Measured long-K shapes use the same producers in split-K mode and a separate
FP32-to-BF16 reduction. One V4 Pro route combines adjacent output tiles in a
two-wave workgroup and uses fused A/B TDM loads. Other shapes remain on the
one-wave direct path when extra partitions or fusion do not pay for their
overhead. The kernel docstrings record the exact tiling, pipeline, and routing
decisions.

### gfx1250 dense BF16 decode projection

The gfx1250 package provides a small-M dense BF16 WMMA projection for K3
decode and EAGLE3 verify, including the KDA QKVFAB, MLA QKV+gate, and latent
up-projection add3 shapes.

#### Contract

- The operation computes `A @ B.T` from contiguous BF16 matrices shaped
  `[M, K]` and `[N, K]`, with `1 <= M <= 64`, `K` divisible by 128, and `N`
  divisible by 16; from 17 rows `N` must be at most 8192. A and B must share
  one CUDA device.
- Output is BF16. A caller-owned output must be on that device, with a unit
  inner stride and a row stride of at least `N`.
- The fixed K3 shapes accept any `M` from 1 to 64 with `K = 7168`: KDA QKVFAB
  has `N = 6288` and MLA QKV+gate has `N = 3648`.
- The add3 variant computes `bf16(A @ B.T) + X + Y` for `K = 3584` and
  `N = 7168`. `X` and `Y` are BF16 `[M, 7168]` with a unit inner stride. The
  projection rounds to BF16 before each BF16 addition, matching `torch.mm`
  followed by two adds, and always runs without split-K.
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

Every accepted `M` fits one row tile, so B streams once. From 17 rows, a
four-warp `64 x 32` tile covers all of M; rows past M load as zero and are not
stored. Through 8192 columns that tile gives each of the 256 CUs one
32-column block; wider outputs would need a second round, so they stay on
rocBLAS. QKVFAB's 6288 columns leave a masked 16-column tail in the last tile.
QKVFAB, QKV+gate, and add3 run the 64-row tile without split-K.

Up to 16 rows, general dense projections use a two-warp `16 x 32` tile when
`N` is divisible by 32 and has at least 192 such tiles; other accepted `N`
uses one warp and a `16 x 16` tile. KDA QKVFAB and MLA QKV+gate use the
one-warp tile, and add3 uses the two-warp tile. All use 256-wide K tiles when
K is divisible by 256, falling back to 128-wide tiles otherwise. The add3
epilogue loads `X` and `Y` for the stored tile.

The weight stream is latency bound per program: the TDM buffers ahead of the
WMMAs set the bytes in flight. K3 projections request eight buffers and the
BF16 GEMM below requests its per-tile counts. Every launch, the BF16 GEMM's
included, caps the request by the K tiles per split and by LDS: a grid past
256 CUs keeps two programs resident on a CU, and they share its 320 KB.

One K loop serves every tile and K length. It keeps up to `buffers - 1` A/B
TDM pairs in flight, waits only for the next tile, and reads that tile from
LDS while the current tile's WMMAs run. Each tile's two K halves accumulate
separately and are summed after the loop, so the WMMAs do not form one long
dependent chain.

Both paths accumulate in FP32 and round to BF16 once. Direct launches store
that conversion from the producer. Split-K launches write one FP32 partial
matrix per K partition, then a separate reduction sums those partials in FP32
and stores BF16. The live row count and split-buffer stride are runtime
arguments that do not specialize the producer or reduction, so warming a
projection covers other batch sizes with the same model dimensions.

### gfx1250 BF16 GEMM

`gluon_mm_a16w16_gfx1250` is a general BF16 GEMM with FP32 or BF16 output. It
backs the DeepSeek V4.1 FP32 projection (`gluon_dsv4_linear_fp32_gfx1250`).
`gluon_grouped_mm_a16w16_gfx1250` runs the one- or two-group `wo_a`
projections up to 2048 rows (`gluon_grouped_bf16_projection_gfx1250`) and can
quantize the result for `wo_b` (`gluon_grouped_bf16_projection_fp8_gfx1250`).

#### Contract

- The operation computes `A @ B.T` from BF16 `A` `[M, K]` with a unit column
  stride and contiguous BF16 `B` `[N, K]`, with `M >= 1`, `N` divisible by 16,
  and `K` divisible by 128.
- Output is FP32 or BF16, chosen explicitly. A caller-owned output needs a unit
  column stride.
- The grouped form takes `A` `[M, G, K]` with a unit inner stride and
  contiguous `B` `[G, N, K]`, and returns contiguous BF16 `[M, G, N]`. With
  quantization (`N` divisible by 32) it returns E4M3 codes `[M, G * N]` and
  uint8 UE8M0 scales `[M, G * N / 32]`, bit-identical to quantizing the BF16
  result with `gluon_quantize_fp8_group32_ue8m0_gfx1250`.

#### Algorithm

Batches that the dense WMMA kernel can tile within one wave of the 256 CUs run
there. Up to 32 rows use one 16- or 32-row tile without split-K. From 33 to 128
rows, a 64- or 128-row tile splits K by the first of 5, 4, or 2 that divides the
K tiles, leaves at least two per split, and keeps the grid within the CUs. It
keeps 16-column tiles when they fit a split of 5 and otherwise repeats the
choice with 32-column tiles, where a split of 5 may then fit. Larger
batches use `128 x 64` tiles, split K in two when the grid still fits, and move
to the large-M kernel once those tiles alone exceed the CUs. Split partials are
FP32 and the shared reduction sums them in split order, so results are
deterministic.

The groups of a grouped call share one grid: the column-tile axis enumerates
every group's tiles, and the tile choice above sees `G * N` columns. Batches
beyond the dense tiles run each group on the large-M kernel. With
quantization, the split-K reduction rounds each sum to BF16 and stores codes
and scales instead of the BF16 result, so the activation never reaches
memory. Up to 32 rows, which otherwise do not split, use 32-column tiles with
K split in two; other batches whose tiles do not split K project, then
quantize.

### gfx1250 dense BF16 large-M projection

`gluon_mm_a16w16_largem_gfx1250` is a compute-bound BF16 GEMM for prefill
projections with many rows.

#### Contract

- The operation computes `A @ B.T` from contiguous BF16 `[M, K]` and `[N, K]`
  with `M >= 512` and K a multiple of 128, producing a contiguous BF16
  `[M, N]`. The registered op accepts the weight shapes listed in
  `use_gluon_largem_gfx1250`.
- Ragged M and N are handled in-kernel. The row count is a runtime argument,
  so any number of rows reuses one compiled kernel.

#### Algorithm

Wide problems (at least 4096 rows and 1536 columns) run
`_wmma_tdm_dense_largem_warp_pipeline_kernel`:

- Each CTA computes a `256 x 256` tile on eight waves, two per SIMD, through a
  double-buffered TDM pipeline. `warp_pipeline_stage` alternates the two waves
  on a SIMD between TDM issue plus LDS reads and WMMA, so one wave's memory
  work overlaps the other's matrix work. Partitioned padded LDS layouts let
  the paired waves read different LDS partitions.
- CTAs run as `2 x 2` clusters. Each TDM load multicasts an A or B panel to
  the two CTAs that share it, halving operand reads from L2, and a cluster
  arrive/wait per K tile keeps the CTAs in step.
- TDM descriptors are bounded at the real M and N, so loads past the edge read
  zeros and the output, staged in the freed operand LDS, is written by one
  clipped TDM store. Edge clusters run whole, so ragged shapes need no special
  path and every cluster barrier completes.
- Clusters are assigned contiguous runs per XCD so clusters sharing operands
  share an L2.

Narrower or shorter problems keep `_wmma_tdm_dense_largem_kernel` and its
own tile choice. The general BF16 GEMM's large-M route
(`_launch_largem`) uses the same selection for FP32 or BF16 output; a
row-strided output also takes the older kernel, whose masked stores accept
it.

When the tile count is not a multiple of the CU count, the last partial wave
of tiles leaves CUs idle; this tail is the main remaining gap to the WMMA
peak.

### gfx1250 MXFP8 large-M projection

`gluon_mm_mxfp8_ue8m0_largem_gfx1250` covers DeepSeek V4.1 MXFP8 projections
from 17 rows, above the decode projection's range.

#### Contract

- The operation computes `A @ B.T` from row-major E4M3 matrices shaped `[M, K]`
  and `[N, K]` with raw UE8M0 scales shaped `[M, K/32]` and `[N, K/32]` and
  an explicit `[1, 32]` scale block. All inner strides must be one.
- `N` must be divisible by 16 and `K` by 32. Output is BF16; `alpha` is not
  supported. Automatic selection requires `M >= 17`.

#### Algorithm

Every shape runs one of two clustered eight-wave warp-pipeline kernels adapted
from the MXFP8 variants of Triton's CDNA5 warp-pipeline GEMM example. Each
program is a cluster of CTAs that share A and B panels and their raw scale
planes through TDM multicast, so no scale repacking precedes it. Descriptors are
bounded at the real M, N, and K. Ragged edges zero-fill on load and clip on
store, and K advances shrink the bounds so tiles past K zero-fill.

- `gluon_mm_mxfp8_ue8m0_largem_bk256_gfx1250` computes `256 x 256` tiles in
  2 x 2 clusters and 256-wide K steps, rounding K up to an even count of at
  least four steps. It takes batches from 512 rows when N is divisible by 256
  and its tiles exceed half the CUs.
- `gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250` takes the rest in 512-wide K
  steps: one when K fits a single step, otherwise at least two. It uses
  `64 x 64` tiles while those fit one wave of CUs, then `128 x 128` tiles,
  in 2 x 2 clusters. Both run unclustered up to 128 rows, and the 128-row
  tiles also for two K steps.
- Up to 64 rows, a call with at least eight K steps splits K into the most
  partitions of two or more steps that still fit one wave of CUs. Each
  partition stores FP32, and `_mxfp8_split_reduce_gfx1250` sums them into
  the BF16 output in a second launch.

### gfx1250 FP32 block-scale large-M projection

`gluon_mm_fp8_blockscale_largem_gfx1250` covers the canonical FP32
block-scale contract (see the gfx950 FP32 block scales section) from 65 rows,
above the decode projection's range. GLM-5.3 uses it for its FP8 dense,
shared-expert, and attention projections.

#### Contract

- The operation computes `A @ B.T` from row-major E4M3 matrices shaped `[M, K]`
  and `[N, K]`, with `N` divisible by 128 and `K` by 512.
- Scales are FP32 `[M, K/128]` and `[N/128, K/128]` with unit inner stride
  and an explicit `[128, 128]` block. Output is BF16; `alpha` is not
  supported.
- Automatic selection requires `M >= 65`.

#### Algorithm

The projection runs the FP32 mode of
`gluon_mm_mxfp8_ue8m0_largem_bk512_gfx1250`, with the MXFP8 large-M
projection's BK512 tile, cluster, and split-K choice; the BK256 kernel has no
FP32 mode. TDM stages both scale planes in LDS beside A and B: the activation
scales of each 512-wide step, and one weight-scale row per 128 output
columns, which each CTA widens in registers to its columns. Each 128-wide
slice runs raw FP8 WMMA into a zeroed partial, which receives its activation
scale and then its weight scale before accumulation. An empty inline asm with
side effects consumes the sum in each compute stage. Without it LLVM sinks
the scale math past the branches of the warp pipeline's closing barrier and
of the cluster arrive, which keeps every partial live and spills. The row
count is a runtime argument.

## Attention

### gfx1250 standard-cache DSA and KPool scoring

`standard_cache_logits.py` scores index queries against FP8 index keys with
WMMA for standard-cache DSA prefill and decode and for GLM KPool prefill. The
DSA launchers then select top-k candidates with the radix top-k kernel; the
KPool launchers return the scores for the caller's selection.

#### Contract

- Each score is `softmax_scale * sum_h w[h] * relu(q[h] . k)` over the index
  heads, with `q` and `k` dequantized and NaN propagated. Heads are added in
  FP32, in logical order for DSA and for KPool with `ordered_head_fold`, and
  in a fixed balanced order otherwise, so scores are deterministic.
- Each cache page holds `page_size` rows of 128 FP8 E4M3 key bytes followed by
  their `page_size` FP32 scales. Page strides and the storage offset are
  4-byte aligned; pages may be padded.
- Standard-cache DSA: queries are `[tokens, 32 | 64, 128]`, BF16 or FP8 E4M3
  with FP32 per-head `q_scales`; weights are BF16 or FP32 `[tokens, heads]`;
  `page_size` is 64; `topk` is 512, 1024, or 2048. Prefill scores the
  workspace rows `[row_starts, row_ends)` of `kv_workspace_slots` and returns
  workspace row ids. Decode scores each request's block table up to its
  sequence length, minus the later tokens of a `q_len_per_req` group (1 to
  6), and returns global slots. Both pad with -1 and return the valid count.
- KPool: queries are BF16 `[tokens, 32, 128]`, weights BF16 or FP32,
  `pool_size` 4 and `page_size` 16. The page-table form sees
  `causal_lens // pool_size` pools of each token's request; the planned form
  sees the physical slots `[row_starts, row_ends)` of `pool_workspace_slots`.
  Output column `c` holds visible pool `pool_offset + c` for `window_cols`
  columns, `row_ends_out` holds each row's valid column count, and later
  columns are not written.

#### Algorithm

A program owns one query row and a span of its candidate columns. A planned
KPool program owns two adjacent rows when they share a workspace start, so
both rows score each key tile it loads; otherwise it scores the rows one
after the other. A program loads each row's queries once, per 32-head group,
in the WMMA A-operand layout, with the weights already multiplied by the
query scales. It then walks the span in column tiles: 256 columns on eight
waves for DSA prefill, 128 on four waves for KPool, and two 32-column tiles
on one wave per 64-column decode program. DSA prefill waves split the 32
heads before the columns; KPool waves split only the columns, so each wave
holds all heads of its columns.

For each tile, the program looks up the candidates' cache slots, from the
workspace or plan slots or through the block table, and loads their keys.
Columns past the span read slot 0 and are not stored, so storages above
2 GiB, which need 64-bit global loads, load without masks. DSA prefill loads
eight adjacent key bytes per lane and KPool sixteen, then converts the tile
through LDS to the WMMA B-operand layout; decode loads the operand layout
directly. FP8 queries use FP8 WMMA; BF16 queries upcast the keys to BF16.
Each head group's WMMA result goes through ReLU and the head weights. For the
logical order, one layout conversion gives each lane all 32 heads of one
column, and the lane adds them in head order with register adds; a second
head group continues the same sum. The balanced KPool fold keeps eight heads
of a column per lane and reduces them in the WMMA layout without LDS. The
column's key scale and `softmax_scale` apply last.

Prefill and KPool launches split each row's columns over a second grid
dimension so that the grid has about four programs per compute unit; with
enough rows, each row stays in one program; paired KPool programs count as
one row. The span is a multiple of the tile and a runtime argument, so
batch-size changes do not recompile the scorer. Only the first program of a
KPool row writes its `row_ends_out`.

### gfx950 selected-slot sparse attention

For dense BF16, E4M3, and E5M2 query and KV rows with 512 latent values,
the Gluon kernel computes score and value products in four 128-column
phases. Matching dtypes retain their native matrix format; mixed inputs
convert to BF16. Inside the key loop, it reuses selected indices staged in
shared memory for the KV gather. The first tile loads gather indices directly,
so one-tile rows avoid an initial shared-index handoff. The configured width fixes
the launch and compile shape; device-resident selected lengths skip unused
tiles and mask live rows across graph replays. Optional 64-value RoPE inputs
add a separate score product. Rank-128 and packed KV layouts retain their
existing implementations.

Separate helpers contain the four-phase score/value products and normalized
output stores. Matching dtypes use two shared buffers and asynchronous
direct-to-shared transfers. Mixed inputs require conversion and refill their
shared buffer after each tile. Tile partitioning, selection masks, online
softmax, and split reduction are shared across these inputs. Ordinary
shared-memory stores and loads use compiler-inferred synchronization, including
the selected-index and single-buffer KV handoffs.

### DeepSeek V4 selected prefill

The gfx950 and gfx1250 packages provide dense-workspace selected prefill
(`dsv4_prefill`), which DeepSeek V4.1 calls for its dense-workspace prefill.

#### Contract

- The prefill kernels accept contiguous BF16 queries shaped
  `(tokens, heads, 512)`, a dense BF16 KV workspace, contiguous int32 selected
  indices and lengths, and a contiguous BF16 or FP32 sink. Registered selected
  widths are 128, 384, 512, 640, 768, 1024, and 1152; gfx1250 also registers
  133, the DSpark draft window of 128 rows plus the 5-row block. On gfx950,
  `q` and the workspace must each be smaller than 2 GiB (32-bit buffer
  offsets).
- Negative indices are holes. Invalid indices, empty selections, partial
  tiles, and lengths outside the selected capacity do not read invalid rows.
  Unsupported traits use the portable implementation.

#### Algorithm

On gfx950, selected prefill gives each (token, 16-head group) one four-wave
workgroup; the 16 heads are the MFMA M dimension, so every gathered 1 KiB KV
row feeds all of them. Selected rows are sanitized once per token (invalid,
padded, and out-of-length slots become an out-of-range row that buffer loads
read as zeros) and 32-row tiles are double-buffered with asynchronous
buffer-to-LDS copies. Wave w computes the QK^T partial over channels
`[128w, 128w + 128)`, the partials are summed through LDS, wave w runs the
online softmax for heads `[4w, 4w + 4)`, and P and the rescale factors return
through LDS for the PV MFMA, whose output channels the waves split. Full-rate
16x16x32 MFMAs serve both products; the transposed score layout already is the
PV A operand. The kernel stays under 80 KiB of LDS so two workgroups share a
CU, and each XCD processes a contiguous token range so neighboring tokens'
overlapping rows hit in its L2. The prefill gathers `tokens x width` KV rows
into LDS once per 16 heads; that on-CU traffic (LDS writes plus K and V
reads), not HBM or MFMA, bounds it. 64- and 128-head cases with widths above
128 use a 64-head sparse kernel with a shape-selected 32- or 64-row tile.

Gfx1250 selected prefill also gives each (token, 16-head group) one four-wave
workgroup, but each wave owns 32 rows of every 128-row step and runs its own
online softmax in the transposed form `S^T = K Q^T`, `O^T += V^T P^T`, so the
softmax and P stay inside the wave. Rows arrive by TDM gather, with each
step's indices loaded a step ahead and clamped to a zero-filled row past the
end. The four partial softmaxes merge once, behind a single barrier, through
the dead KV tile. Q^T stays in LDS so two workgroups fit on a CU.

### DeepSeek V4.1 CSA2 index selection

CSA2 picks, for each query, the KV rows its sparse attention reads. The
`tokenspeed-kernel` adapter (`run_dsv41_csa2_index_topk`) owns query
preparation, validation and tiling, and runs two steps per query tile: a
scorer writes one row of FP32 logits per query, one per candidate column, and
a selector turns each row into that query's sorted list of logical KV row ids.
gfx950 runs both steps in Gluon (`gluon_dsv41_index_qtile_gfx950` when all
queries of a tile share one page-table row, otherwise
`gluon_dsv41_index_topk_gfx950`, then `gluon_dsv41_index_topk_select_gfx950`),
and gfx1250 does the same (`gluon_dsv41_index_topk_gfx1250`, then
`gluon_dsv41_index_topk_select_gfx1250`). gfx950 selects candidate blocks with
`gluon_dsv41_index_block_max_gfx950` followed by its row selector; gfx1250
uses the portable Triton radix select (`tokenspeed_kernel`
`index_block_select`). Both scorers take queries packed as MXFP4 values with
one E8M0 scale per 32 values; the adapter packs a call's queries once.

#### Contract

Visible lengths are bounded by the block table's row capacity in each consumer,
before narrowing int64 inputs to int32. The adapter passes dense raw lengths
without a separate clamp launch. Candidate-block selection uses the same table
bound, including when compact candidate logits have a smaller width, so forcing
the newest visible block preserves its logical position. The gfx950 scorers
write every column the selections read, -inf where no row is scored: each
query's columns through the end of its last visible 8-row block (block
selection reads whole blocks), and every candidate column. Their logits are
therefore not pre-filled; the gfx1250 launcher fills its logits with -inf.
Histogram scratch is zeroed before its atomic sums.
The gfx950 and gfx1250 row selectors read live-width strides directly, including
broadcast limits and strided single-row tail tiles. Such tails retain their stride
after PyTorch `contiguous()`, so selection cannot require a unit limit stride.
Block selection suppresses scalar specialization only for the table bound.
Its existing scalar arguments retain their alignment information: removing that
information scalarizes vector score loads and regresses the measured Full
candidate-source path. Widths and strides still arrive as runtime integers;
model-fixed selection capacities retain the existing launch policy.

- Queries have 1..32 index heads of dimension 128 from one local or
  replicated shard, with FP32 head weights. The index cache uses 64-row,
  page-planar pages of 68-byte MXFP4 rows. Sharded heads, wider head counts
  and 132-byte FP8 index rows use portable Triton.
- Column `c` of a query's logits is logical row `c`, or
  `candidates[q, c // 8] * 8 + c % 8` when candidate blocks are given. Its
  logit is `sum_h w_h * relu(q_h . k_c)`; columns past the visible length,
  without a candidate, or on a missing or out-of-range page stay `-inf`.
- Row selection writes the logical ids of the `min(topk, width)` largest
  finite logits in ascending order, then -1, plus their count (the
  `select_rows_torch` contract). The Gluon selectors keep the lowest columns
  among equal-score boundary ties; the gfx1250 selector also ties `-0.0` with
  `+0.0`.
- Optional block selection keeps the best 8-row blocks by their maximum logit.
  A valid newest visible block remains eligible regardless of its score.
- Selection scores the configured page-table or candidate capacity without
  reading device lengths on the host. The query tile shrinks with that width
  to keep FP32 logits within 32 MiB (at most 256 queries at 32K rows, 64 at
  128K, and 8 at 1M). Score CTAs honor the caller's row-chunk bound up to the
  256-row tuned maximum. The gfx950 per-query scorer uses at most 64 columns
  per chunk (64 or fewer queries) or 128; in Full mode the chunk is a
  multiple of 32 or a divisor of 64, so a chunk never straddles a 64-row
  page. Its grid is bounded, and each CTA strides over the chunks of a
  capacity-wide row.
- Full selection writes finite logits only for each query's visible rows, so
  row selection reads just that prefix of the capacity-wide row (the
  per-query visible length, read on device).
- Arena page strides are kept without copying the cache; a non-unit stride
  between page bytes, or a page stride that is not a multiple of 16 bytes, is
  normalized to contiguous storage before scoring.
- Launch metadata reports score-capacity FLOPs and estimated tensor traffic,
  also without reading device lengths.

#### Algorithm

The gfx950 scorers score 32-column key tiles: the 32 padded heads form two
16-head MFMA groups that share one loaded tile of MXFP4 key rows and scales,
scored with E2M1 scaled MFMA, then ReLU and the weighted head sum. Both are
compiled without FP contraction, so they produce the same bits.

- The shared-table scorer runs one program per four queries and chunk of up
  to 256 columns. A 32-row tile never crosses a 64-row page, so each tile
  needs one scalar page lookup, and the next tile loads before the current
  tile's MFMAs.
- The per-query scorer runs one program per query and grid split. In Full
  mode each tile also takes one scalar page lookup; with candidates it maps
  columns to pages through the candidate ids and page table. Two tiles are in
  flight. A chunk bound that is not a multiple of 32 masks its last tile.

The gfx1250 scorer follows the
same steps with E2M1 scaled wave32 WMMA: all 32 heads form one WMMA tile, and
its four warps split each 64-column key tile, so no warp loads another's keys.

The gfx950 selector runs one 16-warp workgroup per query in a single launch
and reads the row twice:

1. A histogram of the top 11 key bits finds the pivot digit of the K-th
   largest logit.
2. Each warp scans its own contiguous columns and appends, in column order,
   every key at or above the pivot digit to its LDS region.

The regions in warp order are in column order, so the exact threshold (one or
two more digits, from LDS), the lowest-column boundary ties, and the output
order follow without another row read or a sort. If a region overflows, each
warp's exact count gives contiguous column-ordered offsets and a third pass
rewrites the keys; if the pivot digit alone is too large, a second histogram
pass first refines it to 22 bits. Only when the 22-bit boundary bin still
does not fit does the selector fall back to a three-pass MSD radix select and
a column-order compaction. Candidate columns map to logical rows; their picks
are ranked when the candidate list is not ascending.

The gfx1250 selector runs the same passes on sixteen 32-lane waves, so each
workgroup keeps 512 lanes and the 2048-bin histograms.

Block selection takes the maximum logit of each 8-row block, forces the
newest visible block to `+inf`, and keeps the top blocks in ascending order.
On gfx950 one kernel writes the visible block maxima and the gfx950 row
selector picks up to 2048 blocks. The portable Triton block select runs one
program per query: it reads only the visible blocks, runs a four-pass MSD
radix select (8 key bits per pass), and emits the picks in block order, so no
sort is needed.

### gfx950 DeepSeek V4.1 selected attention

`gluon_dsv41_selected_attention_gfx950` reads the page-planar SWA
(E4M3 values, E8M0 per 32) and global (E2M1 values, E4M3 per 16) caches
directly and runs one online softmax over both segments with a single sink.
Dense-workspace prefill calls are forwarded to `dsv4_prefill`.

#### Contract

- BF16 queries `(tokens, heads, 512)`, any head count; FP32 sinks. Caches are
  uint8 `[pages, 64, 528]` / `[pages, 64, 288]` views with contiguous page
  bytes and 16-byte aligned page strides (arena padding allowed).
- Slots and lengths follow `dsv41.selected_attention`: holes, out-of-range
  slots and positions past the length are skipped; empty selections write 0.
- Grid, split count and workspace depend only on tensor shapes; nothing reads
  device lengths on the host, and the token count is never a compile key.

#### Algorithm

A four-wave workgroup owns one query row, up to 64 heads and a contiguous share
of the row's 64-row tiles (SWA tiles, then global tiles). Each wave loads 16
contiguous bytes of a row per lane, dequantizes with the CDNA4 scaled
conversions (`v_cvt_scalef32_pk_bf16_fp8`; E2M1 through
`v_cvt_scalef32_pk_f32_fp4` times the exact FP32 E4M3 scale) into BF16 rows
in LDS, and the waves split the tile's rows for QK and the head dims for PV
(16x16x32 BF16 MFMA). Slots are prefetched two tiles ahead and rows one tile
ahead.

The decode layout pads queries to 64/128 heads with zero rows. A 16-head
group whose query is all zero scores exactly zero, so its output is
`row_sum(V) * e^-m / (count * e^-m + e^(sink - m))` with `m = max(sink, 0)`:
such groups skip the score and PV passes, and one all-ones MFMA per tile
accumulates the V row sum for all of them.

The main kernel runs one workgroup per CU (VGPR-bound). While
`tokens * head_blocks` is below 256, rows are split over up to 16
workgroups that store FP32 `(max, sum, P @ V)` partials (zero-query groups store
the row sum and count). `gluon_dsv41_selected_attention_reduce_gfx950` merges
them in a fixed order and adds the sink once; each of its workgroups owns one
token, 16 heads and a 64-dim slice, so the merge of a small batch still spreads
over many CUs.

### gfx1250 DeepSeek V4.1 selected attention

`gluon_dsv41_selected_attention_gfx1250` and its reduce kernel follow the
gfx950 contract, split and zero-query shortcut above. QK and PV use 16x16x32
BF16 WMMA. Rows are dequantized with the CDNA5 `scaled_upcast` at a unit
scale and then multiplied by the block scale: the E8M0 SWA scale as an exact
BF16 power of two, the E4M3 global scale as BF16. V is read from LDS only
after QK, so the K and V operands are never live together.

Unlike gfx950, each split stores its `P @ V` partial in FP16, normalized by
the split's own denominator, which bounds it by the largest value row.
Zero-query groups still store only the row sum and count, but the reduce
applies the single-split closed form to their totals instead of merging
splits.

A workgroup stages its slots in LDS, resolving each position's gather row (-1
when unusable) in one pass, since TDM takes row indices in SGPRs. Its tiles
then run through a two-deep LDS ring: while one tile is dequantized, a TDM
gather fetches the next tile's value rows and an async copy its scales.
Prefetching rows into registers instead was slower: it doubled the live rows.
The gathers view a cache as rows of a page's value bytes (512 SWA, 256
global) with int32 row indices, so page strides must be whole such rows, as
the arena plan aligns them, and a cache spans at most 2^31 rows. Invalid rows
gather as zero values while their scale copies read row 0, so a cache needs at
least one page; the dequant reads NaN scale bytes as tiny or zero, so those
rows stay zero even over unwritten memory.

Tiles dequantize from the ring straight into the QK operand layout, which
reads 16 value rows at once, so the ring pads every 256 bytes of a row by 8
to keep them off shared banks; the BF16 tile in LDS only feeds the
transposed PV read.
SWA and global tiles share the ring, so a workgroup takes 146 KiB of LDS, two
fit on a CU and rows split until the grid reaches 512 workgroups. The
zero-query check, the staged slot rows and the reduce's splits load clamped
rather than masked, since a row mask becomes a branch and a full wait around
each load. Outputs and partials store straight from the WMMA layout. The
reduce loads all splits at once; its width depends only on the slot widths,
so batch size never recompiles it.

The launch differs from gfx950 in two ways. Small batches give each
workgroup one or two 16-head groups instead of four while rows x head blocks
x tiles stays within 640. That re-reads each selected row once per head
block in exchange for more parallel workgroups, so the token count picks one
of three compiled head-group widths. A split also covers at least two tiles,
so SWA-only calls, which have two tiles, never split.

`gluon_dsv41_selected_attention_rope_gfx1250` serves
`dsv41_selected_attention_rope`. The split reduction rotates the RoPE tail of
each output row back with the FP32 inverse table at the row's position before
its store, so the unrotated output never reaches memory. Unlike gfx950, rows
that do not split (from about 129 decode rows) rotate after the main kernel,
in `gluon_dsv41_output_rope_gfx1250`, because rotating in its epilogue cost
that register-heavy kernel 7 us at 129 and 192 rows. That pass covers 16
heads per program and loads even and odd dims separately, so the rotation
needs no lane swap; it costs no more than `rope_inplace` over the real heads.

### gfx950 QSA sparse attention

`gluon_qsa_sparse_attention_gfx950` serves `qsa_sparse_attention`: each query
row attends to its own list of physical KV-cache slots (Qwen3.8 QSA selects
2051 per row).

#### Contract

- BF16 queries `(tokens, q_heads, head_dim)`; BF16 or FP8 E4M3 caches
  `(slots, kv_heads, head_dim)` of any size with contiguous head dims;
  `head_dim` and `value_head_dim` in {128, 256}; any GQA group (blocks of 16
  query heads).
- Int32 `selected_slots (tokens, width)`, any column order: slots `<= 0` are
  ignored (slot 0 may hold NaN), rows without a valid slot return zeros.
  Scalar FP8 `k_scale` / `v_scale`.
- Grid and split count depend only on tensor shapes; the token count is never
  a compile key.

#### Algorithm

A one-wave workgroup owns one row, up to 16 query heads of one KV head (the
MFMA M dimension) and a share of the row's slots. It first compacts the row's
valid slots to the front of an LDS list (prefix sum and scatter), so invalid
columns, such as the unselected middle of a short context, cost neither
gathers nor MFMAs. Each 16-slot tile is gathered straight into registers, K in
the 16x16x32 MFMA B-operand layout and V through one LDS store and
`ds_read_b64_tr_b16` transposed reads for the 16x16x16 PV MFMA; the next tile's
gathers are issued before the current one is consumed. Gathers use 64-bit
pointers, since serving caches exceed the 2 GiB reach of buffer loads; lanes
past the valid slots read slot 1, so no gather is predicated. Q is kept in LDS and
re-read per tile: the arch-VGPR budget goes to gathers in flight, and a
register-resident Q forced the allocator to reuse gather destinations behind
`vmcnt(0)` waits.

Gathering is L2-bandwidth bound (prefill rows of one request re-read the same
slots). Batches with fewer than about 1024 (row, head-block) workgroups split
the compacted tiles over up to 32 workgroups that store FP32 partials and
base-2 LSE; `gluon_qsa_sparse_attention_merge_gfx950` merges all splits of a
(row, head) in one pass.

### gfx950 QSA indexer

The `qsa_block_scores` and `qsa_block_topk_select` ops, which the `logits`
algorithm of `qwen4_exp_qsa_block_topk` composes, select these Gluon kernels on
gfx950. They return the same blocks as the portable Triton registrations; the
selected ids come out in increasing order rather than by score.

- `gluon_qsa_block_scores_gfx950` writes `sum_h relu(q_h . k)` for each valid
  compressed block below each row's `complete_blocks`, leaving the tail
  unwritten. Each request run stops at its current maximum frontier.
  A program owns 16 query rows (prefill) or 4 rows (decode) whose index heads
  form the 64- or 16-row MFMA M dimension, and loops over its share of
  128-block tiles, prefetching the next key tile. Rows of one request are
  contiguous; a tile spanning requests scores one request run at a time, so
  ragged batches need no query grouping. Keys are gathered with 64-bit
  pointers (the compressed cache may exceed 2 GiB). Page ids, request ids and
  complete-block counts accept int32 or int64. Query and page-table buffers
  must fit within 2 GiB; their buffer-load offsets remain 32-bit.
- `gluon_qsa_block_topk_select_gfx950` radix-selects the `topk` largest
  finite scores per row: four 8-bit passes build LDS-atomic histograms, then
  one pass writes the ids above the threshold and the first ids equal to it in
  increasing column order (ties keep the smaller id, like the portable
  selection). Rows with at most 4096 scores stay in registers; the chunk is a
  power-of-two bucket of the score width, and wider rows stream in 4096-score
  chunks. Supplied per-row lengths bound every scan and load; columns beyond
  them may be uninitialized. Passing `None` scans the full score row.

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
splits write FP32 partials that a second kernel merges. Each merge program
owns 128 latent columns of one token and head, so even a single request
spreads its merge over many programs. It streams up to 64 splits per step
straight into registers, stopping at the last split that holds keys, and
reduces across warps once at the end, so a few dependent loads cover even
the largest split count and the merge has no split-count limit. The split
count follows the query-block decode's rule below.

The QK product spreads rows over warps and the PV product spreads latent
columns. When a block's rows really span several warps, each tile hands the
probabilities and rescale factors to every warp through two small LDS buffers
behind a single barrier; that barrier also orders the tile's last LDS reads
before the TDM load that reuses its slot.

16-bit inputs load each tile into one LDS slot and wait for it before the
math. FP8 inputs use two slots and two waves per SIMD: the next tile's page
index and its two TDM loads go out before the current tile's math, and each
split issues its first tile before loading the queries. The loop unrolls by
two so each slot index is a compile-time constant; waiting until two loads
remain then leaves the other slot's loads in flight.

### gfx1250 MLA query-block decode

`gluon_mla_decode_query_blocks_gfx1250` decodes 2–16 causal queries per
request, such as an EAGLE3 verify step, reading each KV tile once for all of a
request's queries. It is selected for `mla_decode_with_kvcache` requests with
queries blocked on the query axis.

#### Contract

- Queries are `(requests, queries, heads, 576)` with 2 to 16 queries and 1 to
  128 heads, and a contiguous last dimension. The cache is a contiguous
  `(pages, 64, 1, 576)` of the same dtype: FP16, BF16, FP8 E4M3, or FP8 E5M2.
  `logit_cap` is unsupported.
- The page table is Int32 `(requests, pages)` with a unit inner stride, one
  row per request. Cache lengths are a contiguous Int32 `(requests,)`, at
  least `queries` each, and include every query of the block: query `j` sees
  `cache_seqlens - queries + j + 1` tokens.
- The output is BF16 `(requests, queries, heads, 512)`, with an optional FP32
  log-sum-exp `(requests, queries, heads)` in natural-log units.

#### Algorithm

The query-block decode reuses the flattened decode's scan and merge kernels.
Each request's queries start at `request * queries`, so no query-start
metadata is read. A program holds every head of a block of whole queries in
16, 32, or 64 rows, the smallest that covers `queries * heads`. More than 64
rows split the block by query; more than 64 heads split it by head. Each KV
tile the program loads serves all of its rows, and per-query lengths apply the
causal mask within the block. Programs of 16 rows use two warps and larger
ones four; FP8 double-buffers its TDM loads as in the flattened decode.

FP8 programs fit two per CU (`waves_per_eu=2`), and a second resident program
hides the first one's TDM and LDS waits. The KV split count first spreads the
programs over every CU, then keeps doubling toward two programs per CU while
each split still walks at least 12 pages of `max_seqlen_k`; shorter splits
spend more on their prologue and FP32 partials than they save. It is a power
of two of at most 128. A query that sees no key in a split stores a
negative-infinity maximum, so the merge skips that split without producing
NaN. The splits store their partials in BF16, halving the partial traffic
that at 32 or more splits otherwise reaches a fifth of the KV bytes. The
page-table stride and the request and token counts are runtime arguments, so
batch sizes that keep the split count reuse one binary.

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

### gfx950 MHA decode

`gluon_mha_decode_gfx950` computes paged MHA decode with grouped-query
attention. Full attention always splits the KV stream, and
`gluon_mha_decode_reduce_gfx950` merges the partials by LSE; sliding-window
attention uses its own split kernels.

#### Contract

- The KV cache is paged with 64-token pages, FP16, BF16 or FP8, and head dim
  64 or 128. FP8 queries produce a BF16 output. Sinks are supported;
  `logit_cap` and LSE output are not.
- Full attention takes the split count from the base CTAs (requests x KV heads
  x query-head groups) and `max_seqlen_k`, capped at 64 splits. The count does
  not read device-resident lengths, so graphs pass the maximum context as
  `max_seqlen_k` and replay for any batch composition.

#### Algorithm

Each workgroup is one wave that owns one KV head, up to 16 of its query heads
and one split of the request's pages. K/V pages are copied into LDS with
`buffer_load_dwordx4 ... lds`. When the grid holds enough waves to hide load
latency, the loop is single-buffered: copy a page, wait for it, consume it.

When `select_kv_splits` returns its 64-split maximum, even that leaves the grid
below its occupancy target, so each wave streams a long split through a
four-deep pipeline: three pages in flight while one is consumed, if their copies
fit the 6-bit vmcnt count (BF16 head dim 128 does not). Splits with fewer pages
than buffers stay single-buffered. vmcnt retires in issue order, so page indices
are loaded two iterations ahead, one entry per lane: as vector loads they retire
with the copies, whereas scalar loads of the read-only (`tl.const`) table could
only be waited on with `lgkmcnt(0)`, placed just before each copy, which then
waits for a scalar-load round trip. The index loads are volatile so the compiler
keeps them ahead of the copies. `v_readfirstlane` moves each index into an SGPR
for a buffer descriptor based at that page. Prefetches past the split repeat its
last page, since a masked copy becomes a branch followed by `s_waitcnt
vmcnt(0)`, and the loop waits for them before the epilogue reuses their LDS.

Every split stores its partial and LSE (-inf when empty), so the reduce does
not read cache lengths.

### gfx950 MHA cached extension

`gluon_mha_extend_gfx950` computes paged MHA attention for multi-token queries
over cached keys, as in prefix-cached and chunked prefill. When the grid is
short of workgroups, full attention splits the KV stream with
`gluon_mha_extend_split_gfx950`, and `gluon_mha_extend_reduce_gfx950` merges
the partials by LSE.

#### Contract

- The KV cache is paged with 64-token pages, FP16, BF16 or FP8, and head dim
  64 or 128. Query heads per KV head need not be a power of two. Full,
  causal and sliding-window masking, sinks and LSE output are supported;
  `logit_cap` is not.
- The split count comes from the query blocks and `max_seqlen_k`, not from
  device-resident lengths, so graphs replay for any batch composition.
  Sliding-window attention does not split.

#### Algorithm

Each workgroup owns one KV head and packs its query-head group with up to 16
query positions into the MFMA rows: 128-row tiles on four waves, or 64 rows on
two when no request has more than 64 queries. Unused head lanes are masked.

K/V pages use the same four-deep LDS pipeline as decode when the buffers leave
two workgroups per CU (head dim 64). Split launches always pipeline; unsplit
ones only if that adds no round of workgroups, since single-buffered they fit
three per CU. Ranges with fewer pages than buffers stay single-buffered. Every
lane reads the same page-table entries, so the indices are scalar loads, already
in SGPRs and counted by lgkmcnt, and never wait for the copies. Prefetches past
the range repeat its last page; only pages crossing the range end, a causal
diagonal or a window edge are masked.

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
under the first P @ V chunk. The running maximum moves only when a tile
raises it by more than 8 (base 2), keeping P within FP8 range.

When the grid leaves CUs idle, the launcher splits the KV stream and
`gluon_mla_extend_reduce_gfx950` merges the partials by LSE. Splits fill at
most one wave of CTAs and stream at least four pages each, since every split
pays a fixed prologue and an FP32 partial store.

### gfx1250 MHA prefill

The kernel computes causal attention over packed sequences with equal or ragged
lengths. It supports BF16, FP16, and FP8, head dimensions 64 and 128, grouped
query heads, sliding windows, sinks, and natural-log LSE output.

BF16/FP16 use finite tile and wave-count buckets. Compatible GQA groups pack
query heads into matrix rows so they share K/V loads. Short sequences use one-
or two-key-tile schedules with TDM query loads and fused K/V transfers. Short
eight-wave tiles preserve wave ownership through the output store. Long
sequences overlap K/V transfers with QK, softmax, and PV through a double
buffer; selected layouts defer probability reductions to reduce loop work.

Strongly imbalanced ragged batches compact live query tiles onto the grid.
Partial query tiles remain masked and can use smaller matrices while reusing
resident K/V. Request counts, uniform lengths, and tail lengths are runtime
values; changing them within a tile/feature bucket does not recompile kernels.
The launch policy and its thresholds live in `mha/prefill.py`.

QK and PV accumulate in FP32. The denominator uses probabilities before rounding
them to the input dtype for PV. Dot products must be finite in FP32. Online
softmax subtracts the maximum before scaling, preserving large finite scores.
Sink-derived initial references are capped at the FP32 maximum to prevent
positive overflow when converting to unscaled score units.
The fixed-shift schedule changes probability rounding, so its results are
numerically equivalent within tolerance, rather than bitwise identical.

The normalization modes have these prerequisites and numerical bounds:

| Mode | Eligibility | Numerical contract |
| --- | --- | --- |
| Online softmax | General path, including FP8 and windows | Keep the maximum in unscaled score units; update it and rescale FP32 state for each key tile. |
| Single-tile normalization | Short BF16/FP16, D128, 256 query rows, 128 keys, eight waves, no sinks/LSE | Subtract the tile maximum, accumulate the FP32 denominator, then round probabilities for PV. A finite maximum seed gives masked rows zero probability. |
| Fixed first-tile shift | BF16/FP16, four waves, D64/D128, 64/128-key tiles, no window, at least three key tiles | Keep the first tile's shift. Require an absolute base-2 shift at most 64, denominator in `(0, 1e30)`, and finite normalized output within the output dtype's range. |
| Online retry | Any valid row fails a fixed-shift guard | Recompute the workgroup with subtract-before-scale online softmax in the same launch and buffers, overwriting provisional output and LSE. |

The shift bound limits cancellation; NaN-propagating reductions detect invalid
output, and the denominator bound prevents reciprocal underflow. Masked rows do
not participate in retry decisions. Guards evaluate current tensor values on
every graph replay; high-range inputs can pay for both passes.

LSE describes key logits and excludes sinks. When both are requested, the kernel
writes key-only LSE before adding sinks to output normalization with stable
rescaling. For BF16, values and the final denominator share a power-of-two
downscale based on sequence length, bounding the online value sum before sink
rescaling without changing probability rounding or key-only LSE. Tests cover
this contract with independent FP64 attention, changing graph inputs, storage
offsets, ragged boundaries, and bounded JIT specialization.

### gfx1250 MLA prefill

`gluon_mla_prefill_gfx1250` computes the same dense, non-absorbed attention
as the gfx950 kernels, with WMMA and TDM.

#### Contract

- Queries, keys, values, dtypes, `cu_seqlens_q`, `cu_seqlens_kv`, causal
  alignment, and the unsupported `logit_cap` are the gfx950 entry's.
- The output is BF16 unless the caller passes a buffer, which may be any
  floating dtype with a contiguous last dimension. The optional log-sum-exp
  is FP32 in natural-log units.
- The grid is one workgroup per head, 128-row query block and key split,
  and sequence. `max_seqlen_kv` bounds the split count; `seq_lens_kv` is
  ignored.
- Split FP8 launches add a merge kernel and a transient workspace of one BF16
  output tile and one FP32 LSE row per split.

#### Algorithm

Four warps own 32 rows each of a 128-row query block and walk 64-key tiles
with base-2 online softmax. TDM streams the NoPE and RoPE halves of K and all
of V into double-buffered LDS, one tile ahead. Tiles below the first query
row's causal limit skip masking.

16-bit inputs keep Q in registers and run one wave per SIMD. Tiles at or past
the causal limit mask the scores and zero the value rows past the key tail.

FP8 inputs stage Q in LDS, split the NoPE QK into two 64-wide WMMAs, and run
three waves per SIMD. Every FP8 product is 64 deep, so it uses the 64-deep
FP8 WMMA rather than padding the 128-deep one with zeros. Fully visible tiles
run in their own loop without mask code, and the boundary tiles mask keys
only, since rows past `q_len` are never stored. V is not masked: TDM
zero-fills tile rows past `kv_len`, and those keys score `-inf`.

FP8 launches that would leave workgroup slots idle split each query block's
keys into equal tile ranges, one workgroup per range:

- Non-causal launches of up to 16 sequences split every block with keys into
  the same number of ranges (at most 8, of at least 16 tiles), enough for
  three workgroups per CU. The device derives the split from `cu_seqlens`;
  the host bounds it.
- Causal single-sequence launches cut long blocks into pieces of about 1.5
  per-slot shares of the work and dispatch the heaviest blocks first, so the
  causal tail does not wait on one workgroup.
- Split workgroups publish a normalized BF16 output and base-2 LSE, which
  `gluon_mla_prefill_merge_gfx1250` combines with max-shifted weights. The
  merge is a separate launch because any side effect after the tile loop,
  such as a completion counter, spills the accumulator inside the loop.

Sequences are the slowest grid axis, so one sequence's blocks dispatch
together instead of interleaving with those of sequences that have no keys.

Non-causal FP8 launches with prefixes of at most 16 tiles compile with
`amdgpu-sched-strategy=max-memory-clause`: each block walks only a few tiles,
so one block's latency bounds the launch. Longer prefixes keep the default
scheduler, which is 3–5% faster there.

### gfx950 GDN decode and MTP verify

`gluon_gdn_decode_gfx950` runs the sigmoid-gated delta-rule recurrence for
`gdn_decode_step` (one token) and chain `gdn_decode_mtp` (T tokens) over the
K-last recurrent-state pool. Draft trees stay on the Triton kernel.

#### Contract

- q/k are `[B, T, H, 128]` and v is `[B, T, HV, V]` (V a multiple of 16), with
  contiguous heads and any batch/token strides, so packed conv-output views
  need no split copy. a/b are `[B, T, HV]` with any strides.
- The pool is K-last `[rows, HV, V, K]`, FP32 or BF16, with any row stride.
  A negative read row starts from a zero state; negative destination rows are
  skipped.
- Per-token states go to `output_state_indices` rows of `output_state` (default
  the read pool) or to the dense intermediate buffer; the final state goes
  back to the read row only when neither applies and updates are enabled.
- `T` and the batch are runtime values. The kernel stages tokens in LDS in
  chunks of `min(next_power_of_2(T), 16)`, the only T-derived constexpr.

#### Algorithm

One wave owns 16 V rows of one request and value head: each lane holds two
rows by four 4-wide K runs, so every state access is a 16-byte load or store
filling whole 128-byte lines, and each K reduction is 16 in-register FMAs plus
a three-step DPP butterfly. The prologue loads the chunk's q/k (L2-normalized
and scaled), beta-scaled v, beta and decay into LDS once, so the serial steps
wait only on LDS rather than one global round trip per token. The kernel is
HBM bound: it reads the state once and writes it once per destination.

### KDA chunked prefill (gfx950 / gfx1250)

`gluon_kda_paged_prefill_gfx950` and `gluon_kda_paged_prefill_gfx1250` run
KDA prefill over packed variable-length sequences, from and into the V-major
state layout of the decode pool.

#### Contract

- q, k and raw g are `[1, T, H, K]`, v is `[1, T, H, V]` and beta logits are
  `[1, T, H]`, in FP16 or BF16, with `K = V = 128`. A_log is `[H]` and
  dt_bias `[H, K]`. An optional `lower_bound` selects the safe decay gate.
- `cu_seqlens` (`[N + 1]`) delimits the sequences, which may be empty. The
  initial and final states are FP32 `[N, H, V, K]`.
- `cu_seqlens_cpu`, the host copy of the bounds, lets the chunk schedule be
  planned on the host without a device read-back. The plan is reused by
  tensor identity across the layers of a forward. Without it, `cu_seqlens`
  is read back on each call.
- Token, sequence, chunk and time-group counts are runtime values. Row
  blocks and warp counts come from a fixed menu.

#### Algorithm

Sequences are cut into 64-token chunks that never cross a sequence boundary.
One prepare program per (chunk, head) computes the gated, L2-normalized q and
k, the chunk's cumulative decay, Aqk, and W and U through the FP32 inverse of
`I + Akk`: its 16-token diagonal blocks by repeated squaring and the
off-diagonal blocks from band-shifted products, with beta folded into its
columns. It writes W, U, Qg, Kg, Aqk and exp2 of each chunk's last decay.

A state walk then carries the state across the chunks. Each warp keeps 16
state rows in registers, W, U, Kg, Qg and Aqk stream through a two-slot LDS
ring, and the chunk output is written in the same pass. When too few
sequences would leave CUs idle, each sequence is split into up to 32 time
groups. Every group but the last is summarized as `S_out = S_in M + B`, a
chain kernel turns the summaries into each group's entry state, and all
groups then walk at once; the last one writes the final state.

gfx1250 loads every tile with TDM, picks WMMA layouts whose results feed the
next product without leaving registers, walks with the `coexec` scheduler
(which the backend otherwise applies only up to four warps), and prefetches
M and B three groups ahead in the chain. gfx950 uses MFMA and schedules its
prepare kernel with `max-ilp`.

### KDA records verify and commit (gfx950 / gfx1250)

Standard target decode is a one-token verify window. Both standard and
speculative decode follow verify, acceptance, and commit. Verify retains
the Gluon fused convolution, LDS vectors and batched value-split selection.
GFX950 uses 32-row state panels for one-token windows with a bounded gate
and no value split. Other cases use 16-row panels. Verify preserves committed
FP32 or BF16 state and widens it to FP32. It saves correction U in FP32,
and unnormalized K and raw gate in the activation dtype.
Convolution values round to that dtype before recurrence so recovery uses
the same K as verify. The current AMD registration supports BF16. Verify can
also retain original QKV in a separate caller-owned buffer, which commit
reads for conv history.
A null destination leaves retention to the caller. Padding pages do not write
retained rows. When the caller supplies low-rank gate inputs, GFX950 projects
inside verify for up to 64 total input rows and uses a separate GEMM above that.
MI355 measurements at T1 and T4 place the crossover between 64 and 128 rows.
GFX1250 projects inside verify at every row count. A supplied raw gate is used
directly. Both save the gate used by verify. On both targets, each value split
starts its first checkpoint-panel load before preparing the token vectors.
Without value splits, one-token verify keeps 64 state rows in flight after
preparation.
GFX950 recovery uses 64-row panels for one-token windows with a bounded gate.
Other windows keep 16-row panels, with two in flight for one-token softplus.

Records use slot/head/ring-position/channel order. INT64 metadata supplies the
slot, generation, and absolute start; position/generation stamps guard against
reusing a stale window. Recovery reads the accepted prefix, recomputes key
normalization and decay, and applies U without recomputing the prediction.
A zero accepted count copies the checkpoint. Nonpositive source/destination
pages are padding. Groups select independent page-table rows for each layer.
Verify and commit disable Triton debug mode and automatic integer-overflow
instrumentation. Interval and generation guards remain active; invalid
records are skipped without device assertions.

Batched descriptors use the shared ten-column table: original QKV (0), conv
history (2), U (3), K/raw-g (5), A_log (6), dt_bias (7), and FP32 or BF16
state (8). The remaining columns are reserved. Recovery retains one program
per head, request and layer. It stages accepted
U, K and raw gate once in LDS, then applies the same updates per value panel.
It commits each head's convolution history after loading all old taps,
including when source and destination alias. History retains its bound pool
dtype; original QKV is converted on commit.


## Transform

### gfx950 Hadamard query transform

The operation applies a length-128 Hadamard transform with an explicit output
scale to contiguous BF16 query rows on gfx950, returning the same shape and
dtype. Empty inputs return an empty output of the same shape.

One 64-lane wave handles each row, keeping two FP32 values per lane during
seven add/subtract butterfly stages. Their order matches the portable
reduction tree so the BF16 results agree exactly. The output scale is fixed
for a compiled kernel; the launch grid supplies the number of rows.

### gfx1250 length-128 Hadamard

`gluon_hadamard_transform_128_gfx1250` applies the Sylvester Hadamard along
the last dimension for BF16, FP16, and FP32.

#### Contract

- The last dimension is 128. Leading dimensions flatten into rows.
- Fewer than 256 rows use `triton_hadamard_transform_128`. Larger row counts
  use the WMMA kernel. The row count and output scale are runtime arguments,
  so those larger shapes share one binary per input dtype.
- BF16 products are a single BF16 WMMA. FP16 and FP32 values split into a
  BF16 rounding and two BF16 remainders, so the products match the FP32
  value up to WMMA accumulation order.

#### Algorithm

One wave owns 16 rows and 16 of the 128 outputs. The contraction steps
through K in four 32-wide tiles. Signs are `(-1) ** popcount(k & n)` and are
exact `+-1` in BF16.

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

### EAGLE-4 candidate head

`tokenspeed_kernel.ops.sampling.eagle4_shard_topk` and
`eagle4_candidate_tail` run the EAGLE-4 low-rank candidate head around its
candidate all-gather. On gfx950 they select
`gluon_eagle4_shard_topk_gfx950` and `gluon_eagle4_candidate_tail_gfx950`; the
CUDA kernels serve SM90 onward. Both solutions return the same results.

#### Contract

- The shard top-k takes `[M, N]` BF16 scores of a vocab shard and writes
  `torch.topk(scores, k, sorted=False).indices + vocab_start` into a
  `[M, k]` int64 view with any row stride. The order is CUDA's: keys above
  the k-th largest by position, then its first ties by position. ROCm's
  `torch.topk` leaves that order open.
- The tail takes `[R, M, NC]` BF16 logits and int64 ids of the gathered rank
  blocks, with any rank and row strides. It keeps the NC largest in the same
  order. It writes FP32 logits, their ids, and with `top_k > 0` the draft
  top-k probabilities and ids. The probabilities equal `torch.softmax` bit for
  bit; tied probabilities go in slot order.
- The tail covers NC in 32, 64, 128 and 256, `R * NC <= 1024` and
  `top_k <= 32`. Row counts and strides are runtime arguments; `k`, `NC`, `R`,
  `top_k` and the split sizes, which `N` fixes, are compile-time.

#### Algorithm

The shard top-k is a radix select on the order-preserving 16-bit key of each
BF16 score. Each row splits into up to 16 column ranges, and each workgroup
loops over register tiles of up to 2048 columns in its range. Two histogram
passes find the k-th largest key, one byte at a time, using `gl.histogram`.
Each split writes its own histogram, so no atomics are needed. A per-row
kernel sums the split histograms, finds the k-th key, and takes prefix sums
of each split's counts above it and tied with it. Those prefix sums are the
split's output bases. The emit pass rereads the scores and places each key
with a tile prefix sum. It skips the scans of tiles that hold none of the k
keys.

The tail runs one workgroup per row, with one element per gathered candidate.
For `R > 1` it runs the same two-byte select in registers and scatters the NC
kept candidates into the output. A barrier orders those writes before the
softmax reads them back in slot order. The softmax repeats torch's
64-lane-wave reduction. Lane `l` of a `min(NC, 64)`-wide warp adds
`exp(x - max)` over slots `l`, `l + warp`, and so on. The lanes then combine
by xor butterfly, written as repeated half folds of a reshaped vector, with
libdevice `exp` and round-to-nearest division. The draft top-k repeats an
argmax that breaks ties toward the lower slot.

## MoE

### Packed FP8 routed experts on gfx950

The GLM-5.3 path computes clamped SwiGLU routed experts from BF16 activations
and E4M3 weights with FP32 scales per 128-by-128 weight block. It returns this
rank's contribution; expert selection and cross-rank communication remain
outside the operation.

At checkpoint load, each local expert's weights are packed once into canonical
parameter storage. The loader accepts complete local rows and aligned gate/up
halves, and rejects partial input-column writes before modification. Runtime
keeps no duplicate linear weight bank or unpacking step.

For small, fully local batches, warp-level FP32 arithmetic streams packed
weights and remaps activation loads. Other supported batches quantize the BF16
input into E4M3 groups and use public scaled matrix multiplication. Both paths
perform gate/up projection, clamped activation, and down projection. The matrix
path reuses existing route sorting and shared activation quantization.

### gfx950 latent input projection

The Kimi K3 prefill path projects one packed BF16 input weight into router,
routed-latent, and shared-expert inputs. Automatic selection uses the
small-batch Gluon kernel through 320 tokens, a mid-range Gluon tile for
321-1280, and the large-M Gluon tile from 1281 tokens. The portable packed
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

The same Kimi K3 packed projection as the preceding gfx950 entry, with
separate decode and prefill launchers because the shape changes character
with the batch. Decode streams the whole 86 MB weight past a handful of rows
and is bound by memory; prefill is bound by arithmetic. Every path is a
single launch.

#### Contract

Input, weight, and output shapes and dtypes are the gfx950 entry's. Two
things differ:

- Automatic dispatch selects the decode launcher for 1 to 64 tokens and the
  prefill kernel from 1025; the range between them retains the Triton kernel.
- The packed tensor passed alongside the three weights must be their
  consecutive view. The kernels read only it, so the launchers reject a
  packed tensor the weights do not live inside.

#### Algorithm

Decode is one launch that streams the weight once. Every output tile goes to
one consumer, and a shared-region tile always holds matching gate and up
columns, so SiTU never needs a second kernel.

- From 17 tokens, `gluon_latent_input_decode_nosplit_gfx1250` gives each
  workgroup the whole reduction for a 32-column tile built from two 16-row
  weight slices: adjacent rows in the router and routed regions, the gate
  rows and the up rows 768 below them in the shared region. Its 188
  workgroups each stream 256-wide K steps through six TDM buffers,
  with one warp per 16 rows of the 32- or 64-row tile.
- Up to 16 tokens one row tile leaves those 188 weight streams too few to
  saturate memory, so `gluon_latent_input_decode_gfx1250` gives each
  one-warp workgroup a single 16-row weight slice through eight 256-wide
  TDM buffers. The 280 router and routed slices stream the whole reduction
  and store directly. The 48 gate and 48 up slices of the shared region
  split it in two, launch first, and so finish early enough to hide their
  exchange: each stores an FP32 partial at device scope, drains its stores,
  and bumps its column block's arrival counter. The last arrival sums the
  partials in split order, so results are deterministic, applies SiTU, and
  rearms the counter for the next launch or graph replay. Each stream keeps
  its own counter buffer, which the launches ordered on it share.

Prefill is one launch too: a `256 x 256` WMMA tile on eight warps with no
split, accumulating in FP32 over 128-wide K steps. Like the whole-reduction
decode kernel, a tile is two 128-row weight slices, so the 4480 router and
routed columns take 18 tiles (the last one half used) and the shared region
six gate/up pairs that finish SiTU in the epilogue, with no BF16 gate/up
buffer or second launch. That epilogue writes `sigmoid(x)` as
`(1 + tanh(x / 2)) / 2`, so the whole activation runs on the hardware tanh.
Each K tile is consumed in two half-K steps of an LDS-read stage and a WMMA
stage (`warp_pipeline_stage`), and the two warps that share a SIMD run a
stage apart, so one reads LDS while the other issues WMMA. The next K tile
streams into the other of two TDM slots meanwhile, issued by one warp per
SIMD. Workgroups are renumbered so each of the eight XCDs gets a contiguous
run of the `GROUP_M`-grouped tile order, which keeps the workgroups that
share A or B rows on one L2. Before storing, the kernel redistributes each
half accumulator so neighboring lanes cover adjacent output columns, which
coalesces the BF16 stores, and masks, rather than the tile index, place each
column in the router or routed region, so FP32 router logits reach memory
without a round trip through BF16.

All kernels mask partial row tiles, so they accept any token count.

### gfx950 softmax top-k routing

`gluon_softmax_topk_gfx950` serves `moe_topk(score_function="softmax")`
(the Qwen3.5/3.8 router) in one launch, replacing `torch.topk`, the bitonic
sort, a softmax and two casts.

#### Contract

- FP16, BF16 or FP32 logits `[tokens, experts]` with a contiguous expert
  dimension, at most 1024 experts and top-k at most 32. Returns FP32 weights
  and INT32 or INT64 ids `[tokens, topk]`, ordered by descending logit.
- Ties select the lowest expert id first (`-0.0` ties with `+0.0`), as the
  NVIDIA Triton kernel does; PyTorch leaves the order unspecified.
- With `renormalize` the weights are a softmax over the selected logits,
  which equals the full softmax renormalized over the selection; otherwise
  they are the full-row softmax probabilities. Both are scaled by
  `routed_scaling_factor`, a runtime argument.
- The token count is a runtime argument; one binary serves every batch.

#### Algorithm

One wave owns one token row and holds its logits as integer keys: the
order-preserving bits of the value above the inverted expert id (32-bit keys
for 16-bit logits, 64-bit for FP32). Keys are unique, so each of the `topk`
rounds is one wave max reduction followed by clearing the winning key. The
selected logits come back exactly from the key bits, so no second gather is
needed. The ids and weights collect in registers and leave with one buffer
store each.

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

### gfx950 BF16 expert routes

`gluon_bf16_moe_routes_gfx950` serves `bf16_moe_routes`, the unweighted
expert (down-projection) output of every route, replacing the portable Triton
kernel, which loops over every local expert in every program. Up to 16 tokens
take the direct decode kernels (`moe/fp16/routes_decode.py`), larger batches
the grouped prefill kernels (`moe/fp16/routes_prefill.py`).

#### Contract

- Contiguous BF16 `x [tokens, hidden]`, gate/up weights
  `[experts, 2 * I, hidden]` (gate rows first) and down weights
  `[experts, hidden, I]` for intermediate size `I`, with `hidden % 512 == 0`
  and `I % 256 == 0`; global expert ids `[tokens, topk]` with `topk` in
  {1, 2, 4, 8, 16}.
- Writes the output of every selected local expert to row
  `token * topk + slot` and zeros for routes to non-local or invalid experts.
  A token may select one expert in several slots.
- Both GEMMs accumulate in FP32; the SwiGLU intermediate is rounded to BF16
  between them.
- The token count is a runtime argument. Other shapes, and route buffers or
  an expert's gate/up weights of 2 GiB or more (`buffer_offsets_fit_int32`),
  select the Triton kernel.

#### Algorithm

Decode launches one program per output tile and route for each GEMM. The
first route that selects an expert gathers every token routed to it into one
16-row MFMA tile and streams the expert's weights once; later routes of the
same expert exit. The waves split K and reduce through LDS once, as in the
dense decode GEMV.

Prefill zeroes the rows of non-local routes, then packs each expert's routes,
in route order and without atomics (per-block histograms, one prefix sum and
a stable scatter), into 256-row tiles plus at most one 128-row remainder.
Both GEMMs run the eight-wave warp-pipelined tile of the dense BF16 prefill
kernel and gather activation rows through the sorted route ids. Programs are
numbered over the live tiles only, so the XCD remap spreads them over all
eight XCDs.

### MoE route from precomputed top-k

`_precomputed_topk_route` builds the ragged expert metadata, gather/scatter
indices and gate scales from `[M, topk]` ids and weights in one launch for
any `M * topk <= 65536`, replacing the torch stable argsort, scatter-add
histogram and two metadata kernels. Rows are stably sorted by expert,
bit-identical to `_route_from_topk_torch`; ids outside `[0, E)` sort last with
a zero gate and count in no slice. Each program owns 64 rows and builds the
full and preceding-chunk expert histograms with LDS atomics, so it places a
row at `offset[e] + earlier[e] + rank-in-chunk` without a global sort. Every
program re-reads all ids, so work grows as `G^2 / 64` while the torch route
grows about linearly; the `G` cap sits below their crossover, and larger routes
use the torch route. Five trailing programs write the block offsets and
schedules, one per block size.

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

### gfx950 dynamic-MXFP4 (A4W4) SiLU experts

`gluon_mxfp4_a4w4_moe_apply_gfx950` serves MXFP4 checkpoints whose experts
quantize activations to MXFP4 at runtime (`internal_activation_dtype="mxfp4"`,
for example Quark MXFP4) with precomputed top-k and SiLU (or unclamped
standard SwiGLU) on gfx950. It replaces the portable Triton path there.

#### Contract

- BF16 input `[M, H]` with `H % 512 == 0`; intermediate `I % 64 == 0`.
- Weights use the N16 bank of `moe/mxfp4/n16_weights.py`, shared with the
  MXFP8 SiTU pipelines: each 16-row group stores its K bytes as 64-byte
  chunks laid out as 1 KiB blocks in MFMA lane order, so a wave reads a
  weight fragment with one contiguous load; W13 alternates gate and up
  groups and packs the gate/up scales of two 128-K steps into one 32-bit
  word. W2 rows ending in a half chunk are stored flat without padding.
  Wrapped `weight_loader`s keep checkpoint updates working.
- Numerics follow the Triton path bit for bit up to FP32 accumulation order:
  per-32 E8M0 scale in Quark's "even" mode (amax rounded to a power of two,
  minus E2M1's emax of 2, as the dense MXFP4 activation quantizer), saturating
  E2M1 round-to-nearest-even,
  gate/up/SiLU/product rounded to BF16 before requantizing the intermediate,
  BF16 per-route outputs summed in FP32 slot order. Expert ids outside
  `[0, E)` contribute zero.

#### Algorithm

Four launches: activation quantization (scales stored lane-quad-major so a
lane fetches four K steps' scales with one load), the gate/up GEMM with fused
SiLU-and-mul and MXFP4 requantization in sorted-route order, the down GEMM
writing BF16 per-route rows, and the slot-order combine. Up to 1024 routes,
one extra program of the quantization launch sorts the routes into expert
blocks (atomic per-expert cursors; order within a block does not matter): it
first fills every routed row with the padding id, one store per 256 rows,
and the routes overwrite their rows after a drain and barrier. Larger route
sets run the block-aligned `gluon_moe_sorting` first. Both GEMMs are
`e2m1 x e2m1` scaled 16x16x128 MFMAs that load weights straight into
registers.

Block height follows the average routes per expert: 16 rows up to 4 routes
per expert (decode, including the 64-row MTP verify batches), 32 rows up to
48, otherwise 64. Decode blocks stream each touched expert's weights once and
load activations directly, keeping one 256-K step in flight. Larger blocks
stage gathered activations through an LDS ring with async copies shared by
the N-direction waves, keep each block's programs on one XCD so activations
and an expert's weights are reused from L2, and redistribute the down GEMM
output through LDS so each scattered route row is stored as one contiguous
run; at 8K tokens that BF16 per-route write and its combine read dominate the
second GEMM.

### gfx1250 MXFP4 experts

On gfx1250, the MoE API selects Gluon kernels with FP8 or MXFP4 activations
and MXFP4 weights for precomputed top-k routing. The `fp8` activation policy
selects FP8; `input` and `mxfp4` select MXFP4, the A4W4 numerics gfx950 runs
for DeepSeek V4.1. Two expert GEMMs run per layer: a
gate/up GEMM with a fused SwiGLU or SiTU activation, then a down GEMM that
combines into each token's output row.

#### Contract

- FP8 activations enter both GEMMs as E4M3 divided by a per-tensor FP32
  scale, which the GEMM multiplies back into its FP32 accumulator. MXFP4
  activations are quantized per row with one dynamic UE8M0 scale per 32
  values, bit-identical to the portable `quantize_mxfp4`; scale rows are
  padded to 16 bytes for TDM. Weights are packed MXFP4 with one UE8M0 scale
  per 32 values.
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

With MXFP4 activations, step 2 quantizes the input to MXFP4 instead, in extra
programs of a step 1 launch: the one-launch route when it covers the batch (up
to 3072 rows when it also quantizes), otherwise stage 4 of the staged route.
Single-token batches keep the standalone quantizer. The gate/up epilogue
quantizes its FP32 result and stores it as MXFP4 directly, in the layout the
down GEMM reads.

Decode (a few routed rows per expert) uses a separate kernel that streams
each touched expert's weights once, so its tiles target HBM streaming rather
than reuse. Every program owns one 16-row block of one expert and a 128-column
slice. Padding blocks of the over-sized grid carry a `-1` schedule entry and
exit after one load. The program issues its weight TDM loads as soon as the
expert ID is known, before the row metadata, gather indices and scatter rows
that the activation loads and epilogue need. Long reductions (gate/up) load
512-wide K steps (256-byte MXFP4 row segments). Short reductions (down)
load 128-wide steps so three buffers hold all of K in a small LDS footprint,
and more programs stay resident to hide each one's fixed latency.

FP8-activation prefill (about 150 routed rows per expert at 8192-token
chunks) uses 64x256 tiles. Neither GEMM saturates WMMA, TDM or HBM there:
each program runs its load, LDS and WMMA phases largely back to back, so the
tiles are picked for resident programs and L2 sharing. Gate/up uses eight
warps, since four would need ~560 VGPRs per wave and leave one program per
WGP. Down loads 128-wide K steps, which divide its K of 384 exactly. Both
swizzle tiles over the XCDs so the tiles that share an expert's weights or an
activation tile share an L2. MXFP4-activation prefill keeps the earlier
four-warp tile with 256-wide K steps until it is measured.

The launcher resolves the row tile from the gathered row count and expert
count unless the caller pins it. The kernels mask ragged M and N edges rather
than peeling them, so a trailing partial tile loads only the rows that exist.

Routes of up to 2048 rows, or 3072 when the launch also quantizes the MXFP4
input, build all metadata in one launch, as a stable counting sort by expert:
the gather, scatter and gate rows match a stable argsort, and invalid ids sort
last with a zero gate and count in no slice. Each
program first builds the whole expert histogram with LDS atomics; one atomic
per row also counts the rows before the program's chunk in its high 16 bits.
`cdiv(rows, 64)` programs then place 64 rows each. A row lands after the rows
of lower experts, the same-expert rows of earlier chunks, and the same-expert
rows earlier in its chunk, counted with a 64 x 64 comparison tile. Five more
programs write the slice offsets and the block offsets and schedule for one
block size each. No program waits on another, so latency stays near one
histogram and scan instead of one workgroup's serial pass over every row and
block size.

The large-batch router counts expert assignments in groups of at most 256
experts, using four warps in the counting stage. This reduces LDS atomic
contention when routing across many experts. The router counts duplicate
expert IDs separately and excludes invalid IDs from every group.
The subsequent prefix scan processes eight adjacent experts per four-warp
block, coalescing accesses to the row-major chunk-count buffer. Each expert
retains an independent scan over chunks, with partial expert groups masked.
When the layer quantizes an MXFP4 input, the placement stage adds one program
per 8 x 512 tile of it after its routing programs.

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

### gfx1250 general biased-sigmoid top-k

`gluon_sigmoid_bias_topk_route_gfx1250` selects experts for shapes outside
the Kimi K3 specialists: FP16, BF16, or FP32, 1 to 1024 experts, and top-k
1 to 16.

#### Contract

- Logits are `[tokens, experts]` with a unit expert stride. The bias is a
  contiguous `[experts]` vector on the same device.
- Selection scores are FP32 `sigmoid(logit) + bias`. NaN scores are excluded.
  Equal scores take the lowest expert id.
- Route weights are the FP32 sigmoid of the selected logits, optionally
  divided by their sum, then multiplied by the routed scaling factor.
- It stays below the single-token decode kernel, the packed 896-expert K3
  kernel, and `gluon_sigmoid_bias_topk_gfx1250`, which keep those shapes.

#### Algorithm

One program handles one token with four warps. It walks the expert vector
once per selected route, keeps the maximum finite score, and breaks ties by
the minimum expert id. The weight is the sigmoid captured for that expert.

### gfx950 channel-FP8 experts with BF16 projection operands

Channel-scaled FP8 expert weights can remain in their compact checkpoint
format while projections use BF16 operands and FP32 accumulation. FP8 storage
does not require native FP8 matrix arithmetic.

## Residual

### gfx1250 Kimi K3 AttnRes

`attention/kda/attn_res.py` mixes the current residual with up to 11 stored
block residuals (snapshots) and applies the following RMSNorm in one launch.

#### Contract

- Candidates are the `N - 1` snapshots, then the current residual. Each is
  scored by `<v * rsqrt(mean(v^2) + eps), res_weight * score_rms_weight>` in
  FP32; the softmax-weighted sum is rounded to BF16 before the output RMSNorm.
- With `delta`, the BF16-rounded `layer_residual + delta` is written back and
  is the current residual. With a snapshot write, it is also stored to snapshot
  row `N - 1`, which the mix does not read.
- `N` and the snapshot write switch are compile-time; the token count and the
  snapshot-row stride (`T * H` for block-major storage) are runtime, so batch
  sizes share binaries. Rows and pointers must be 4-byte aligned.

#### Algorithm

One workgroup handles one token. BF16 rows are read as 32-bit words holding an
(even, odd) element pair, so each element converts to FP32 with one shift or
mask, and results are packed back into words before 128-bit stores.

The snapshots form one `[BLOCK_ROWS, H / 2]` word tile (`BLOCK_ROWS` is `N - 1`
rounded up to a power of two) with every row of a column in the same lane.
Each row is a buffer load from its own 64-bit row base, so offsets stay 32-bit
at any token count, and masked padding columns read zero. All loads, including
the three weight vectors, issue before the first store because the compiler
cannot hoist a load above a possibly aliasing store. One two-operand reduction
then yields every row's square sum and score dot product. When the tile has a
spare row, it carries the current residual so that reduction scores all
candidates; otherwise the current residual gets its own reduction. The softmax
weights are normalized once, so the mix is a lane-local FMA per element.

Launch shapes fall into three warp counts. From 256 tokens, the bandwidth-bound
launch uses 4 warps (one per SIMD): each lane holds a wide register tile, so
loads stay in flight with no cross-warp reduction traffic. Decode grids use a
few WGPs and each token's workgroup is VALU-bound: 8 warps up to 16 tokens or
4 snapshots, and 16 warps for larger tiles at larger batches. Startup prefill
warmups and decode graph captures compile all three.

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

### gfx950 mHC mixes, post-mapping and stream collapse

DeepSeek V4.1 splits the hc=4 mHC into three registered operations:
`mhc_mixes` derives the coefficients from the full residual, `mhc_apply_pre`
collapses the streams with the previous sublayer's pre coefficients (with an
optional fused RMSNorm), and `mhc_post` applies the post/combination update.
All three stream the `[T, 4, H]` BF16 residual from HBM, so they are sized
against bandwidth rather than math.

#### Contract

- `mhc_mixes` takes a contiguous BF16 residual `[T, 4, H]` (`4 * H` divisible
  by 64), FP32 weights `[24, 4 * H]`, scales `[3]` and biases `[24]`, and
  returns FP32 pre `[T, 4]`, post `[T, 4]` and combination `[T, 4, 4]`
  indexed [input stream, output stream]. Results are at least as accurate as
  the portable kernel (absolute error near 1e-6 against FP64 at 8192 tokens).
- `mhc_apply_pre` returns the BF16 layer input. The plain collapse rounds each
  product like the portable kernel and matches it bit for bit; with a norm
  weight, the stream sum is rounded to BF16 before RMSNorm, and only the
  RMS reduction order differs (a rare one-ulp BF16 difference).
- `mhc_post` returns the updated BF16 streams, bit-identical to the portable
  kernel.
- All offsets are 32-bit buffer offsets, so the residual holds fewer than
  2^31 elements. Token counts are runtime arguments; the mixes kernel has two
  compile-time block buckets (64 rows up to 1024 tokens, 256 rows above).

#### Algorithm

The mixes projection is one pass over the residual. Each CTA owns a row block
and a K split and loads residual tiles straight into BF16 MFMA operand
registers. Because the K reduction is order-free, the logical MFMA K index is
permuted so every lane reads one contiguous 32-byte run per row. The FP32
weights are split exactly into three BF16 terms (two 8-bit truncations and a
remainder) once per CTA, shared through LDS, and accumulated with three
`16x16x32` BF16 MFMAs, which reproduces FP32-weight products without the
FP32 MFMA rate limit. The squared row sums for the RMS factor come from the
same registers. Split partials go to a small FP32 buffer; a reduction kernel
assigns sixteen lanes to a token's 4x4 matrix, sums the splits over
registers (and lanes or waves when decode uses many splits), and runs the
sigmoid, softmax and Sinkhorn steps with lane-local row and column sums. The
Sinkhorn divisions use the hardware reciprocal (about one ulp).

The post and collapse kernels move eight contiguous BF16 values per lane.
Post runs one token per CTA so its twenty coefficients are scalar loads that
never wait behind the vector stores. Large post calls (prefill chunks) stream
far more than the caches hold, so above 2048 tokens adjacent CTAs take
adjacent hidden blocks of one token and all vector accesses are non-temporal:
the CTAs in flight cover contiguous rows, and the post neither caches its dead
inputs nor evicts the dirty lines earlier kernels left behind. Smaller calls
keep ordinary caching so the next pre-mapping can read the new streams from
cache. The collapse runs one token per CTA so the optional RMSNorm reduces the
whole row once.

### gfx950 Qwen4 hyper-connection mix

`gluon_hyperconnection_mix_gfx950` serves `gated_residual_mix` for decode
batches (1 to 256 rows) of the HC4 / H2560 / R320 stream in two launches,
replacing two hipBLASLt GEMMs (one with a split-K reduction kernel) and two
Triton epilogues.

#### Contract

- Contiguous FP16/BF16 `normalized [rows, 4 * H]`, fused down/inject weight
  `[R (+ 4), 4 * H]` and up weight `[4 * H, R]`; `4 * H` divisible by 10240
  and `R` equal to 256 plus a power of two of at least 32. Returns the mixed
  `[rows, H]` and the inject logits `[rows, 4]` (or `None`) in the input
  dtype.
- Numerics follow the GEMM-plus-epilogue path: both projections accumulate in
  FP32 and are rounded to the input dtype before the scale/SiLU and the
  sigmoid gate, so only the accumulation order differs (at most one ulp).
  Split partials are summed in split order, so results are deterministic.
- Row counts are runtime arguments within three tile buckets (up to 16, up
  to 32, larger rows).
- The split-K workspace (FP32 partials and self-resetting arrival counters)
  is shared per device and tiling, so launches must be ordered on one stream.

#### Algorithm

The down projection splits `K = 4 * H` into 1024-deep slices, one per
workgroup, with 16-wide weight tiles: about 200 workgroups each stream 32 KB
of weight, which keeps every CU busy and the last split's reduction small.
Operands are loaded with 16-byte coalesced loads, all issued before the
first MFMA; the conversion to the MFMA operand layout broadcasts a shared
operand through LDS rather than re-reading it per wave. Each split writes an
FP32 partial through to memory and bumps a per-tile counter; the last split
sums the partials, applies scale and SiLU, and writes the activation (and
inject logits).

The up projection gives each workgroup 16 hidden columns of all four
branches as one MFMA tile, one branch per wave, with the 320-deep reduction
loaded as a 256 + 64 pair so every load is in flight at once. The epilogue
multiplies `sigmoid(gate)` with the matching normalized branch and sums the
four branches across waves, so the gate tensor never reaches memory.

### gfx1250 mHC mixes projection

`gluon_mhc_mixes_gfx1250` computes the hc=4 prenorm mixing coefficients for
BF16 residual streams. It shares split selection with `triton_mhc_mixes` and
replaces both of its kernels: the projection `x @ fn.T` with its row sums of
squares, and the reduction that runs normalization and the Sinkhorn
iterations. `gluon_mhc_pre_gfx1250` runs the same two kernels for `mhc_pre`
at every token count, then collapses the streams with the portable kernel.

#### Contract

- `x` is contiguous BF16 `[T, K]` (the four flattened streams) and `fn` is
  contiguous FP32 `[24, K]`. The projection writes FP32 partials
  `[splits, T, 24]` and `[splits, T]`.
- The WMMA projection runs when `K` splits into whole 128- or 64-wide K tiles
  that fill the TDM pipeline in every partition; other `K` uses the portable
  Triton projection.

#### Algorithm

The FP32 weight tile splits into a BF16 rounding and two BF16 remainders that
sum back to it exactly. BF16 products are exact in FP32, so three BF16 WMMAs
give the FP32 projection up to accumulation order. Each program covers 64 rows
and one K partition, pipelines K tiles through three or four TDM buffers, and
pads the 24 mixes to two 16-wide WMMA tiles. The token count and tiles per
partition are runtime arguments.

The reduction is the gfx950 kernel on 32-lane waves: sixteen lanes own a
token's 4x4 matrix, so a wave covers two tokens when the splits stay in
registers and one when they also spread over lanes.

### gfx1250 mHC stream collapse with FP8 output

`gluon_mhc_apply_pre_fp8_group32_gfx1250` serves `mhc_apply_pre_fp8_group32`
for DeepSeek V4.1: the gfx950 stream collapse with RMSNorm that also writes
the group-32 FP8 input of the next projection.

#### Contract

- Contiguous BF16 residual `[T, 4, H]` with `H` divisible by 32 and fewer
  than 2^31 elements, FP32 pre coefficients `[T, 4]` and a BF16 or FP32 norm
  weight `[H]`.
- Returns the BF16 layer input `[T, H]`, numerically the gfx950 normalized
  collapse, with E4M3 codes `[T, H]` and uint8 UE8M0 scales `[T, H / 32]`
  bit-identical to `gluon_quantize_fp8_group32_ue8m0_gfx1250` reading it.

#### Algorithm

One four-wave CTA owns a token, so RMSNorm reduces the row once; each lane
moves eight contiguous BF16 values, and four lanes hold one quantization
group. The BF16-rounded result is quantized from registers, which saves the
quantizer's launch and its re-read of the row.

## Quantization

### gfx950 group-32 UE8M0 FP8 activations

#### Contract

The kernel quantizes `[M, K]` BF16 or FP16 rows (contiguous K divisible by
32, any row stride) to E4M3 with one UE8M0 scale byte per 32 values. Rounding
is the DeepSeek V4.1 reference: the group amax is floored at `1e-4`,
multiplied by `1/448`, and rounded up to a power of two through its FP32
bits. The output is bit-identical to `triton_quantize_fp8_group32_ue8m0`.

#### Algorithm

Four lanes share a group, each loading one 16-byte vector, and a CTA covers
256 groups so large batches keep many vectors in flight. Groups flatten rows,
so any K and ragged M fill whole CTAs. A power-of-two scale has an exact
reciprocal, so one multiply per value rounds exactly like the reference
division; infinite inputs keep the `x / inf` behavior through `1 / inf`.

### gfx1250 group-32 FP8 activation quantization

`gluon_quantize_fp8_group32_ue8m0_gfx1250` quantizes activations for DeepSeek
V4.1 FP8 and MXFP8 projections.

#### Contract

- Input is BF16 or FP16 `[M, K]` with a unit column stride and `K` divisible by
  32; padded row strides are accepted.
- Output is contiguous E4M3 codes `[M, K]` and row-major uint8 UE8M0 biased
  exponents `[M, K/32]`, bit-identical to `triton_quantize_fp8_group32_ue8m0`.

#### Algorithm

A group's scale is `2 ** ceil(log2(max(amax, 1e-4) / 448))`, taken from the
FP32 bits of the quotient, and its codes are the correctly rounded
`x / scale` cast to E4M3. Each program covers consecutive groups, with four
threads per group and eight elements per thread: 32 groups and four warps
below 49,152 groups, and 256 groups and eight warps from there.

## Layernorm

### gfx950 QK RMSNorm and SSMax query fusion

Query normalization can be fused with position-dependent SSMax scaling while
preserving the rounding of the separate operations. The positional scale
applies only to queries; keys retain ordinary RMS normalization.
