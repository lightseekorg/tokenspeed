# GEMM weight preparation

The optional `trtllm_kernel` extension is required for TRT-LLM's fused-A
projection and cuBLASLt NVFP4 GEMM provider. Without it, the NVFP4 leaf is not
registered and a direct fused-A call reports the missing package; importing
other GEMM providers does not require the extension.

## Blackwell block-FP8 automatic selection

For standard FP32-scaled 128x128 block-FP8 GEMM on SM10x, automatic selection
prefers TRT-LLM CuTe-DSL when the activation matrix has `M > 16` rows. At
`M <= 16`, FlashInfer keeps precedence. Its existing `17 <= M <= 32` exclusion
still applies; if CuTe-DSL is ineligible in that interval, selection falls back
to the next compatible backend.

The CuTe-DSL route requires contiguous E4M3 operands, 16-byte-aligned operand
and scale pointers, BF16 output, a contiguous 16-byte-aligned output buffer when supplied,
`alpha=None`, and N/K divisible by 128. Other
requests retain compatible fallback selection. Both CuTe-DSL registrations
call the same implementation; the low-priority registration keeps explicit
`solution="trtllm_cutedsl"` available for small M. Plugin priorities remain above
both built-in backends.

Backend selection and tile tuning are separate. Once CuTe-DSL is selected,
startup autotuning searches its tile, cluster and 2-CTA configurations and
persists the winning choice per shape bucket in the shared autotune cache.
See the [adapter contract](../../thirdparty/trtllm_blockwise/README.md) for
search and fallback behavior. M is the actual GEMM row count, which
can differ from serving concurrency after batching or tensor-parallel gathers.
The PR #1639 comparison shapes use M=C for DEP16 and M=4C for TP4, so their
cutovers are C=17 and C=5 respectively. In a powers-of-two concurrency sweep,
the first samples above those cutovers remain C=32 and C=8.

## Block-FP8 layout compatibility

The CuTe-DSL, FlashInfer CUTLASS, Triton, and Hopper DeepGEMM block-FP8 paths
require contiguous A and B. Their registrations declare these constraints so
automatic selection excludes incompatible layouts before launching a kernel.
For the standard FP32-scaled block format, non-contiguous operands can use the
Torch reference fallback, which prioritizes correctness over performance.
Explicit `solution=` requests with incompatible layouts fail during selection.
For BF16/FP16 inputs using online FP32 group-scale quantization, selection uses
the contiguous FP8 layout produced by the quantizer. A strided dense activation
therefore remains eligible for optimized GEMM; prequantized strided FP8 operands
still require a compatible fallback.
The same applies to UE8M0 group-32 online quantization when the dense input
has unit inner stride; that quantizer supports row strides and emits contiguous
FP8 values.

Output and scale constraints differ by backend. CuTe-DSL and DeepGEMM require
a contiguous output; CuTe-DSL also requires 16-byte alignment. Triton writes
using output strides; FlashInfer can copy a temporary result into a strided or
unaligned output. The common `mm` API still requires
the output's innermost stride to be one. DeepGEMM also requires contiguous
weight scales; FlashInfer and CuTe-DSL normalize their scale layouts internally.
For 128x128 block FP8, both CuTe-DSL and FlashInfer require 16-byte-aligned A/B
pointers. A contiguous offset view that breaks this alignment falls back to
Triton. CuTe-DSL selection also requires aligned scale pointers; an unaligned
scale view falls back to FlashInfer where its M range permits, or Triton.
This scale check is conservative even when a layout copy would align the view.
Online quantization supplies freshly allocated, aligned activations and scales.

## Dense FP32 output

For BF16 or FP16 inputs, `mm(..., out_dtype=torch.float32)` stores the FP32
accumulator directly. It must not widen a result already rounded to the input
dtype. Scaling and bias operate on that FP32 result, including when the caller
supplies a strided output buffer. On gfx950, `gluon_mm_bf16_fp32_gfx950` serves
measured BF16 projections; the AMD ops README records its shapes and reductions,
which need not match a vendor GEMM bit for bit.

## Small dense projections

`mv` selects the registered small-row projection implementation. Its dispatcher
and registry both use the `("gemm", "mv")` operation. The gfx950 row-CTA GEMV
declares its minimum N and K through the shared shape traits, and takes more
than one row only for measured shapes, through its `mnk_problem_filter`.

On CDNA4, ordinary GEMM dispatch admits the measured single-row
`[1,8192] @ [4608,8192].T` projection alongside the existing multirow cases.
Callers that publish directly into a collective buffer pass that buffer as
`out` to the same `mv` operation.

`mv_fp8_weight` projects 16-bit rows through a weight-only FP8 weight: E4M3
with one FP32 scale per output channel, as a 16-bit checkpoint quantizes at
load. Activations stay 16-bit, so the only roundings past the weight's are
the FP32 sums' and the output's. The `("gemm", "mv_fp8_weight")` operation
takes up to `MV_FP8_WEIGHT_MAX_M` rows a launch, larger calls run in chunks of
that many, and the Torch reference upcasts the weight a chunk at a time.

`mv_weightless_rmsnorm_gated_sigmoid` normalizes, gates and projects one row
through the `("gemm", "mv_weightless_rmsnorm_gated_sigmoid")` operation. A
registered kernel serves it only where the separate gated norm and `mv` select
the kernels it reproduces bit for bit; otherwise callers keep the two calls.

On SM107, BF16 output projections whose weight shape is in the measured
`_SM107_BF16_GEMM_SHAPES` set (the TP4 per-rank projections plus the TP1
`[2560, 6144]` o_proj) run FlashInfer's CuTe DSL low-latency GEMM at M <= 4,
pinned as the measured winner on every admitted shape; other shapes keep
their existing path. Startup tuning covers M=1/2/4. The kernels accumulate
in FP32 and return BF16; reduction order can change rounding relative to
the original GEMM.

## Decode GEMMs on small weights

On SM100 and SM103, `mm` runs a BF16 GEMM on one pinned FlashInfer TGV tile
instead of cuBLASLt when all of these hold:

- No bias, BF16 output, and no plan, `solution`, override, scales, or `alpha`
- `TGV_DECODE_MIN_M < M <= TGV_DECODE_MAX_M` (32 < M <= 512). Up to 32 rows
  (`BF16_GEMM_MAX_M` on SM100 and SM103), `mm` takes FlashInfer's joint BF16
  GEMM (`flashinfer_bf16_gemm`) instead, in which FlashInfer's tuner chooses
  among its TGV and CuTe DSL tactics.
- `TGV_DECODE_MIN_K <= K <= TGV_DECODE_MAX_K` (2048 <= K <= 3072; below)
- This rank's weight takes at most `TGV_DECODE_MAX_WEIGHT_BYTES` (16 MiB).
- One of the pinned tiles covers the GEMM in one wave (below).
- PDL is enabled (`pdl_enabled()`, which the server's `--disable-pdl` turns
  off).
- The operands meet the joint BF16 contract (`flashinfer_joint_bf16_supported`):
  contiguous, 32-byte aligned, K a multiple of 8.
- Serving has not started (`is_serving()`).

Under PDL, TGV loads weight tiles while the producer kernel still runs and
waits only before loading activations; cuBLASLt starts loading after the
producer finishes. FlashInfer's tactic profiling times each GEMM alone and
cannot see that overlap, so `select_tgv_decode_tile` chooses the tile itself:
of the 2-CTA tiles of 128 weight rows by 16, 32, or 64 activation rows, each
with the most stages that fit in shared memory, the first whose grid has at
most one CTA per SM. A GEMM for which none fits keeps cuBLASLt, and so does
every GEMM without PDL, which has no overlap to gain. The choice depends only
on M, N, K, the weight size, the SM count, and the PDL setting.

The K window is measured, not derived: on GB300, in a captured PDL chain of an
RMSNorm and this GEMM on a weight cold in L2, TGV beat cuBLASLt on most shapes
with K in the window, by up to 1.30x. Outside it TGV lost on many shapes, down
to 0.73x cuBLASLt's speed, mainly at K of 512 to 1024 and at K >= 4096 with
few weight rows; at K = 1536 it also lost on some shapes (down to 0.92x).

The weight limit and the K window are bounds on shape, not a list of layers.
On a quantized checkpoint, the BF16 GEMMs that reach them are the small
projections the checkpoint leaves unquantized; the route was measured end to
end only in such a model's decode graphs. On a BF16 checkpoint they also admit
dense projection shards under tensor parallelism, not measured in a model: at
TP4, a model with hidden size 3072 and 8 KV heads of 128 has a 7.5 MiB fused
QKV shard (K = 3072), routed up to M = 448 on 148 SMs, while its output
projection shard (K = 768) keeps cuBLASLt.

The serving gate has a different reason than the joint path's, which
compiles once per row count. TGV compiles once per tile, but the route was
measured only inside CUDA graphs. So graphs captured at startup, decode graphs
and prefill graphs of 33 to 512 tokens alike, keep the TGV launches they
recorded, and eager calls while serving keep cuBLASLt. The gate also keeps
TGV compilation out of serving when startup compiled no tile, as under
pipeline parallelism without CUDA graphs.

A TGV tactic id is a position in FlashInfer's tactic table, so
`resolve_tgv_tactics` finds the tiles by shape. If the installed FlashInfer
lacks one of them or its TGV launcher, a warning is logged once and these
GEMMs keep cuBLASLt. TGV compiles once per tile, PDL setting, and operand
layout, never per shape. `kernel_warmup` compiles the three tiles at startup.
Pipeline parallelism skips kernel warmup; there, the eager passes before each
graph capture compile the tiles that graph uses. Either way, no tile compiles
per row count, during capture, or while serving.

Like the joint path, the route is a shortcut in `mm` ahead of registry
selection, not a registered kernel: for the GEMMs it routes, it precedes every
registered BF16 `gemm.mm` kernel, plugins included.

TGV accumulates in FP32 and rounds once to BF16. Its reduction order differs
from cuBLASLt's, so its results are not bitwise identical to cuBLASLt's: a
decode batch replayed from a captured graph and an eager call of the same
shape after serving starts can differ in the last bits. Repeated TGV calls on
the same inputs, eager or replayed, agree bit for bit.

## Per-token/per-channel FP8

Per-channel FP8 projections use FP8 E4M3 weights with FP32 scales per output
channel and dynamically quantize activations per token. This scale format is
distinct from block scaling. Native FP8 matrix arithmetic uses FP32
accumulators but need not match sequential FP32 arithmetic.

## Startup tuning coverage

During `ops.tuning.autotune`, FlashInfer MXFP8, per-tensor FP8, CUTLASS NVFP4,
SM100/103 W4A16, and fused NVFP4 GEMMs cover the shared token budget, including
larger observed inputs and the enclosing native bucket. Native mappers,
constraints, and caches are preserved. Explicit buckets and disabled profiling
retain their semantics; small-M BF16 keeps its existing bound.

## Decode-size BF16 projections

`triton_skinny.skinny_mm` computes `x @ w.T` for up to 32 BF16 rows on NVIDIA,
streaming the weight once. Its module docstring records the split-K fold, which
keeps a row's result independent of the batch, and its PDL order.

## Block-FP8 DeepGEMM

`solution="deepgemm"` selects the appropriate block-FP8 kernel for the GPU.
Explicit overrides are `deepgemm_mm_fp8_blockscale` on SM90 and
`deepgemm_mm_fp8_blockscale_packed` on SM100/SM103. Serving can opt in with
`--dense-gemm-backend deepgemm` for standard block-FP8 dense linears. The
default `auto` route remains unchanged; routed experts use their own backend.

This optional DeepGEMM integration supports SM90, SM100 and SM103, contiguous
E4M3 operands A[M,K]/B[N,K], BF16 output, N divisible by 64, and K divisible
by 128. On Hopper it converts canonical FP32 1x128 activation and 128x128
weight scales to the required layout without UE8M0 rounding. Conversion is part of each call,
including graph capture; no persistent weight cache is created. Unsupported
forced calls fail explicitly. It does not claim batch-invariant numerics.

On SM100/SM103, dense FP8 layers configured for DeepGEMM call
`prepare_deepgemm_fp8_weight` after loading. It requantizes weights onto
power-of-two scales and packs them once. The layer pins DeepGEMM for forward.
Activations use the registered `packed_ue8m0` quantizer,
which directly writes TMA-aligned int32 scales with initialized padding.
GEMM consumes these packed scales without per-call copies or packing.
The quantizer and DeepGEMM follow the global PDL setting and work in CUDA
Graphs and on auxiliary streams. Default FP32-scale dispatch is unchanged.

## Grouped FP8 projection

`grouped_fp8_projection` computes `einsum('tgk,gnk->tgn')` from E4M3
activations and weights that both carry one UE8M0 scale per 32 K channels
(DeepSeek V4.1's grouped `wo_a`). `prepare_grouped_fp8_projection` re-lays the
canonical `[groups * N, K / 32]` weight scales out once for the serving kernel
and returns None where no kernel serves the weights, so callers keep their
dequantized BF16 path. `has_grouped_fp8_projection` answers the same question
from the device and shape alone, before any weight is loaded. The DeepGEMM
kernel (SM100/SM103) takes activation scales packed four per int32, MN-major
and TMA-aligned, which is what `dsv41.rope_quantize_fp8_grouped` writes.
`grouped_fp8_projection_quantize_fp8` instead writes E4M3 codes of the
BF16-rounded result with packed scales in that same layout, which
`fp8_projection` (a dense E4M3 GEMM, `prepare_fp8_projection` for its weight
scales) reads directly, so no quantization kernel runs between the two.

DeepGEMM picks a tile configuration from the token count (one per 16 tokens)
and JIT-compiles each on first use. `warmup_fp8_projections` launches every
configuration up to a token bound at startup so serving never compiles.
