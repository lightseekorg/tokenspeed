# FP8 quantization

`quantize_fp8` returns `(values, scales)`. A plain cast has no scales;
static scaling returns the supplied scale. Dynamic `token` quantization
uses one FP32 scale per row, and `token_group` uses canonical token-major
scales. Backend adapters normalize their results at this boundary.

For a 2-D input, the row count is a selection trait (`m`), so a kernel can
bound the rows it takes (`m_max`). On NVIDIA SM100 and newer, `token`
quantization of up to 64 rows runs the one-launch Triton per-token quantizer.

Two functions write that quantizer's codes and scales from the kernel that
produces the activation, so no separate quantize launch follows:
`rmsnorm_quantize_fp8`, an RMSNorm that also quantizes its output, and
`silu_and_mul_quantize_fp8`, the SwiGLU of an FP8 GEMM output run with unit
scales, which applies the GEMM's per-token and per-channel scales first. Each
has a single Triton kernel, so they call it directly instead of selecting a
registered kernel. The SwiGLU takes a half of at most 16,384 columns and
raises `ValueError` above that.

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
