# GEMM weight preparation

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
