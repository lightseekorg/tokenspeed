# Causal depthwise convolution

The shared Triton prefill, chain/decode and tree kernels keep their multiply-add
operations in FP32 until the final output store. Casting only the accumulator
is insufficient: multiplying two FP16/BF16 values first rounds their product
before it reaches that accumulator. Each tap explicitly converts its operands
to FP32 and uses an FP32 FMA. Convolution-history state retains its declared
input dtype and its existing page ownership and update order.

The Ascend cancellation regression uses `(1 + 2^-9)^2 - 1` to distinguish
FP32 products from prematurely rounded FP16 products in both prefill and
decode. It requires the independently computed FP64 result rounded once to
FP16, exactly. Random/ragged/history-remapping tests cover the ordinary path.
