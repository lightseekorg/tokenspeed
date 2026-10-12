# FP8 quantization

`quantize_fp8` returns `(values, scales)`. A plain cast has no scales;
static scaling returns the supplied scale. Dynamic `token` quantization
uses one FP32 scale per row, and `token_group` uses canonical token-major
scales. Backend adapters normalize their results at this boundary.

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
