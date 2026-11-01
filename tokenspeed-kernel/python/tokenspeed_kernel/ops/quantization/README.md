# FP8 quantization

`quantize_fp8` returns `(values, scales)`. A plain cast has no scales;
static scaling returns the supplied scale. Dynamic `token` quantization
uses one FP32 scale per row, and `token_group` uses canonical token-major
scales. Backend adapters normalize their results at this boundary.

For a 2-D input, the row count is a selection trait (`m`), so a kernel can
bound the rows it takes (`m_max`). On NVIDIA SM100 and newer, `token`
quantization of up to 64 rows runs the one-launch Triton per-token quantizer.

Compressed-tensors W8A8 checkpoints require per-token quantization that
preserves their minimum positive scale and rounding behavior. Token-group
quantization is not an interchangeable substitute.

Two functions write that quantizer's codes and scales from the kernel that
produces the activation, so no separate quantize launch follows:
`rmsnorm_quantize_fp8`, an RMSNorm that also quantizes its output, and
`silu_and_mul_quantize_fp8`, the SwiGLU of an FP8 GEMM output run with unit
scales, which applies the GEMM's per-token and per-channel scales first. Each
has a single Triton kernel, so they call it directly instead of selecting a
registered kernel. The SwiGLU takes a half of at most 16,384 columns and
raises `ValueError` above that.

On NVIDIA SM10x, BF16 token-group-128 quantization writes canonical FP32
scales directly in the quantization kernel, without a subsequent transpose
copy. It retains reciprocal scaling and clamps each group's maximum magnitude
to the existing BF16-rounded `1e-10` epsilon. Explicit backend overrides,
other input dtypes and other scale encodings retain their existing routes.

`granularity="block"` accepts positive two-dimensional block sizes and
BF16, FP16, or FP32 inputs. The quantizer masks partial edge blocks, and the
returned FP32 scale shape is `[ceil(M / block_m), ceil(K / block_k)]`. GEMM
backends can impose narrower block-size constraints than the quantizer.

`dequantize=True` performs an FP8 round trip with UE8M0 token-group scales
and returns the reconstructed input dtype with `scales=None`. Round trips
and block quantization reject `enable_pdl=True`, which their kernels do not
support. The common GEMM API predicts online activation scale storage from
the weight scale encoding: uint8 UE8M0 weights use uint8 activation scales;
FP32 weights use FP32 activation scales.

`quantize_fp8_with_gate` combines an optional per-head RMSNorm, sigmoid
output gate, and 128-element FP8 quantization. It rounds the gated result to
BF16 before computing FP8 values and FP32 scales, matching the separate
operations. KDA returns raw BF16 output for both single-token decode and
verification; its output norm belongs to this stage. MLA uses the same
stage without RMSNorm. Prequantized values and scales feed the output
projection directly. The same Triton implementation serves NVIDIA and AMD;
PDL is enabled only where supported.

## Native FP32 token-group scales

Explicitly selecting `override="trtllm_quantize_fp8_mn_major"` with
`granularity="token_group"`, `group_size=128` and `scale_encoding="float32"`
returns FP8 values and a strided `[M, K / 128]` view of the native scales.
This kernel accepts BF16 matrix inputs and does not support PDL. Values and
logical scales match the canonical TRT-LLM quantizer exactly. Only the scale
storage changes; the FP8 activation data layout is unchanged.

Here MN-major refers to scale storage: for activation scales, the M (token)
dimension is contiguous. It does not prescribe the weight matrix layout.
The returned view preserves the native stride, which depends on the TRT-LLM
implementation. Native storage may be unpadded `(1, M)` or padded, such as
`(1, round_up(M, 4))` for the flat-buffer ABI. The latter satisfies Hopper
DeepGEMM's 16-byte stride alignment for FP32 scales (four 4-byte elements).
MN-major storage alone does not guarantee TMA alignment.

This entry eliminates the contiguous-scale copy in the quantization wrapper.
A downstream GEMM can also skip its scale repack only when it accepts the
native strides and encoding. In particular, it does not guarantee that
Blackwell consumers can skip scale alignment or encoding conversion.

Ordinary quantization requests token-major storage and excludes this entry
from ranked selection. Explicit `override` selects the native entry; its
implementation validates the supported quantization parameters. Callers
opting in must support the returned strides. This entry does not change
model projections or GEMM dispatch.

## DeepGEMM packed UE8M0

On SM100/SM103, `quantize_fp8(..., granularity="token_group", group_size=128,
scale_encoding="packed_ue8m0")` produces E4M3 codes and int32 scales with
shape `[M, ceil(K/512)]` and strides `[1, round_up(M,4)]`. Four exponent
bytes share each word. Quantization uses bit-level upward rounding of the
FP32 scale exponent, matching vLLM, and initializes all row/K padding in
the same launch. There is no separate transpose, pack or memset kernel.
PDL and CUDA Graph replay use the same implementation.

### CuTe DSL implementation

The BF16/group128/packed-UE8M0 path selects CuTe DSL by default on
SM100/SM103. FP16 and ordinary FP32-scale quantization keep their existing
implementations. An individual call can select the previous packed quantizer
with `override="triton_quantize_fp8_group128_packed"`.
Eight lanes quantize each group of 128, using packed BF16 arithmetic and
FP8x2 conversion. The scale recipe floors absmax at FP32 `1e-10`, multiplies
by FP32 `1/448`, and rounds the exponent upward, matching SGLang's BF16
UE8M0 path. FP8 values and packed scales, including padding, are written in
one launch. This path does not support FP16 or ordinary FP32 scales.

The launcher uses 32 threads for a single row, 128 threads for 2–512 rows,
and 256 threads above that. Positive FP32 scale bits are rounded upward by
adding the mantissa mask before extracting the exponent. Saturating FP8x2
conversion supplies the output clamp and retains NaNs, avoiding a separate
BF16 clamp for each pair. Group absmax and the scale recipe are unchanged.
M and scale stride are runtime values; only K, thread count, PDL setting and
device identify a compiled executor. Warm each required executor before
CUDA Graph capture. PDL waits for the producer and signals after output
stores. The low-level `launch` accepts preallocated outputs for benchmarks
or graph owners; the registry wrapper allocates outputs like the existing
quantizers and copies unaligned inputs when needed.
