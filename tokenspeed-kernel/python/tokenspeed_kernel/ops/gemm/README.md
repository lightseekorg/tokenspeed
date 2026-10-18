# GEMM weight preparation

## Small dense projections

`mv` selects the registered small-row projection implementation. Its dispatcher
and registry both use the `("gemm", "mv")` operation. The gfx950 row-CTA GEMV
declares its minimum N and K through the shared shape traits.

On CDNA4, ordinary GEMM dispatch admits the measured single-row
`[1,8192] @ [4608,8192].T` projection alongside the existing multirow cases.
Callers that publish directly into a collective buffer pass that buffer as
`out` to the same `mv` operation.

## Per-token/per-channel FP8

`mm(..., quant="fp8")` accepts BF16/FP16 activations with FP8 E4M3 weights
and FP32 per-output-channel scales. It quantizes activations per token through
the operation registry, then accumulates the FP8 product in FP32, applies
token and channel scales, and rounds to the requested output dtype. AMD
gfx950 uses the portable Triton implementations. Weights stay FP8; hidden
states and outputs stay BF16 for Beam. Empty batches return empty outputs,
and masked partial matrix tiles contribute zero. This scale format does not
accept block sizes or an additional alpha scale.

Tests of gfx950 native FP8 matrix instructions show small-product loss within
groups of eight K positions: even the exactly representable `256*256 + 1*1`
can return `65536` instead of `65537`. This implementation partitions the
original K positions into eight masked native FP8 dots and carries the FP32
accumulator between them. Each instruction receives at most one nonzero
product per affected group. This preserves FP8 operand storage and native
FP8 multiplication and fixes the demonstrated grouping loss; it does not
guarantee equivalence to sequential FP32 arithmetic. The portable Triton
implementation uses the same eight-way accumulation on all devices with native
FP8 matrix support, without a vendor or architecture-name branch. Device
validation so far is on gfx950; correctness and the additional matrix-work
cost on other architectures still need hardware qualification. Specialized
implementations remain separate registry choices.

## DeepSeek V4 grouped output projection

The grouped-projection plan describes the projection geometry and source formats.

After loading and sharding, the operation selects preprocessing using the source
formats and operation traits. The layer runs that preprocessing through the
weight broker, which installs the transformed tensors and records their storage
layout. When no transformation is needed, the layer enrolls canonical storage.

Execution and warmup select kernels using the supplied tensors' recorded layouts,
formats, and operation traits. Participating layers require known storage layouts;
raw canonical callers can explicitly allow unknown storage. This policy is
independent of kernel overrides. Python validation runs during graph capture,
not device-side replay.

## Hopper block-FP8 DeepGEMM

`deepgemm_mm_fp8_blockscale` supports explicit `mm(..., override=...)` or
`solution="deepgemm"` selection. Serving can opt in with
`--dense-gemm-backend deepgemm` for standard block-FP8 dense linears. The
default `auto` route remains unchanged; routed experts use their own backend.

This optional DeepGEMM integration requires Hopper, contiguous E4M3 operands
A[M,K]/B[N,K], BF16 output, N divisible by 64 and K divisible by 128. It
converts canonical FP32 1x128 activation and 128x128 weight scales to the
required layout without UE8M0 rounding. Conversion is part of each call,
including graph capture; no persistent weight cache is created. Unsupported
forced calls fail explicitly. It does not claim batch-invariant numerics.
