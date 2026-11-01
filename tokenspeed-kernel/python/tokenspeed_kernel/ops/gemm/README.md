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

## Small dense projections

`mv` selects the registered small-row projection implementation. Its dispatcher
and registry both use the `("gemm", "mv")` operation. The gfx950 row-CTA GEMV
declares its minimum N and K through the shared shape traits, and takes more
than one row only for measured shapes, through its `mnk_problem_filter`.

On CDNA4, ordinary GEMM dispatch admits the measured single-row
`[1,8192] @ [4608,8192].T` projection alongside the existing multirow cases.
Callers that publish directly into a collective buffer pass that buffer as
`out` to the same `mv` operation.

`mv_weightless_rmsnorm_gated_sigmoid` normalizes, gates and projects one row
through the `("gemm", "mv_weightless_rmsnorm_gated_sigmoid")` operation. A
registered kernel serves it only where the separate gated norm and `mv` select
the kernels it reproduces bit for bit; otherwise callers keep the two calls.

On SM107, FlashInfer's TGV/CuTe DSL search supports BF16 output projections
with weight shape `[2560, 6144]` and M <= 4. Startup tuning covers M=1/2/4.
The kernels accumulate in FP32 and return BF16; reduction order can change
rounding relative to the original GEMM.

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

## Hopper block-FP8 DeepGEMM

`deepgemm_mm_fp8_blockscale` supports explicit `mm(..., override=...)` or
`solution="deepgemm"` selection. Serving can opt in with
`--dense-gemm-backend deepgemm` for standard block-FP8 dense linears. The
default `auto` route remains unchanged; routed experts use their own backend.

This optional DeepGEMM integration requires Hopper, contiguous E4M3 operands
A[M,K]/B[N,K], BF16 output, N divisible by 64, and K divisible by 128. It
converts canonical FP32 1x128 activation and 128x128 weight scales to the
required layout without UE8M0 rounding. Conversion is part of each call,
including graph capture; no persistent weight cache is created. Unsupported
forced calls fail explicitly. It does not claim batch-invariant numerics.
