# FP8 quantization

`quantize_fp8` returns `(values, scales)`. A plain cast has no scales;
static scaling returns the supplied scale. Dynamic `token` quantization
uses one FP32 scale per row, and `token_group` uses canonical token-major
scales. Backend adapters normalize their results at this boundary.

TRT-LLM's `fp8_quantize_1x128` returns a flat FP32 scale allocation in
MN-major order with row pitch `round_up(tokens, 4)` and possible trailing
allocation padding. Its adapter removes both kinds of padding before exposing
contiguous `[tokens, groups]` scales; it never infers orientation from equal
matrix dimensions. The prepared FlashInfer GEMM consumes the same adapter's
`[groups, tokens]` view for row counts divisible by four. The native `use_ue8m0`
flag rounds scales to powers of two but leaves them in FP32, so the public
`scale_encoding="ue8m0"` adapter additionally encodes them as uint8 exponents.

`granularity="block"` accepts positive two-dimensional block sizes and
BF16, FP16, or FP32 inputs. Partial edge blocks are masked, and the returned
FP32 scale shape is `[ceil(M / block_m), ceil(K / block_k)]`. GEMM backends
can impose narrower block-size constraints than the quantizer.

`dequantize=True` performs an FP8 round trip with UE8M0 token-group scales
and returns the reconstructed input dtype with `scales=None`. Round trips
and block quantization reject `enable_pdl=True`, which their kernels do not
support. The common GEMM API predicts online activation scale storage from
the weight scale encoding: uint8 UE8M0 weights use uint8 activation scales;
FP32 weights use FP32 activation scales.
