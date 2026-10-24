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

Per-channel FP8 projections use FP8 E4M3 weights with FP32 scales per output
channel and dynamically quantize activations per token. This scale format is
distinct from block scaling. Native FP8 matrix arithmetic uses FP32
accumulators but need not match sequential FP32 arithmetic.

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

## Blackwell small-M block-FP8

On SM100 and SM103, `mm` selects `flashinfer_trtllm_mm_fp8_blockscale`
(FlashInfer's TRT-LLM-Gen backend) for standard 128x128 block-FP8 GEMMs with
at most `FLASHINFER_TRTLLM_FP8_BLOCKSCALE_MAX_M` (64) rows and K a multiple of
256. The CUTLASS `flashinfer_mm_fp8_blockscale` serves larger M, where it is
faster. Both read the same FP32 scales; the accumulation order differs.

The TRT-LLM-Gen kernel reads MN-major activation scales: the `[M, K/128]`
view with strides `(1, M)` that the `mn_major_scales` quantizer returns.
Callers that quantize their own activations can ask
`mm_fp8_blockscale_reads_mn_major_scales(m, k)` and request that feature to
skip a per-call scale transpose. The kernel converts any other layout, so
skipping the request costs only the copy.

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
