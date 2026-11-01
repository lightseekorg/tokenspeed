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

## Block-FP8 DeepGEMM

`deepgemm_mm_fp8_blockscale` supports explicit `mm(..., override=...)` or
`solution="deepgemm"` selection. Serving can opt in with
`--dense-gemm-backend deepgemm` for standard block-FP8 dense linears. The
default `auto` route remains unchanged; routed experts use their own backend.

This optional DeepGEMM integration supports SM90, SM100 and SM103, contiguous
E4M3 operands A[M,K]/B[N,K], BF16 output, N divisible by 64, and K divisible
by 128. On Hopper it converts canonical FP32 1x128 activation and 128x128
weight scales to the required layout without UE8M0 rounding. Conversion is part of each call,
including graph capture; no persistent weight cache is created. Unsupported
forced calls fail explicitly. It does not claim batch-invariant numerics.

On SM100/SM103, loading requantizes weights onto power-of-two scales and
packs them once. Activations use the registered `packed_ue8m0` quantizer,
which directly writes TMA-aligned int32 scales with initialized padding.
GEMM consumes these packed scales without per-call copies or packing.
The quantizer and DeepGEMM follow the global PDL setting and work in CUDA
Graphs and on auxiliary streams. Default FP32-scale dispatch is unchanged.

### FlashInfer block-FP8 tuning

The canonical K-major CUTLASS path tunes `mma_sm=1/2` through FlashInfer's
native AutoTuner during startup. One large input also profiles power-of-two
decode/prefill sizes up to that input, excluding the unsupported M=32.
Entries use exact tensor shapes, strides,
dtypes, device capability and the global PDL setting, and participate in the
existing FlashInfer cache persistence. Untuned shapes use `mma_sm=1`; CUDA
Graph capture only looks up cached choices, even inside a tuning context.
The FP8 GEMM API has no PDL argument; activation quantization and the BF16 BA
branch retain their own PDL settings. BA overlap keeps its event dependencies.
