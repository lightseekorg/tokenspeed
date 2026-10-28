# Residual operations

Residual operations use the kernel registry so model code does not select a
hardware implementation directly. The family includes attention residuals,
gated residual boundaries, manifold Hyper-Connections (mHC), and independent
Hyper-Connections (iHC).

## Independent Hyper-Connections

iHC keeps several residual channels for each token. Its pre boundary reduces
those channels to a sub-block input and computes gates for the post boundary.
The post boundary adds a gated sub-block output to each channel. The head
boundary reduces the channels before the model's final normalization. Unlike
mHC, iHC has no channel-combination matrix or Sinkhorn normalization.

The public operations are `ihc_pre`, `ihc_post`, and `ihc_head`. They accept
BF16 or FP32 residual channels shaped `[..., channels, hidden]`; projection
weights, scales, biases and post gates are FP32. Sub-block inputs and outputs
are BF16. `ihc_pre` may include the following RMSNorm when both `norm_weight`
and `norm_eps` are supplied.

The precision boundaries are part of the contract:

- The gate projection reads the unnormalized residual in FP32. The row's RMS
  factor scales the projection result before the sigmoid gates are computed.
- The pre reduction rounds to the residual dtype before an optional RMSNorm.
  The norm evaluates in FP32 and the sub-block input is stored in BF16.
- The post boundary multiplies the BF16 sub-block output by FP32 gates, adds
  the residual in FP32, and stores the result in the residual dtype.
- The head reduction stores a BF16 result; the model applies its final norm.

The Torch implementation provides a portable reference. The Triton solution
uses a split projection followed by gate evaluation and channel reduction.
For BF16 residuals, it decomposes FP32 projection weights into three BF16
terms and accumulates the products in FP32. FP32 residuals use FP32 products.
Different reduction orders can produce small floating-point differences;
these implementations do not promise bitwise equality with one another.
Numerical tests check the gates, channel reduction and normalization separately.
A reduction near a BF16 midpoint can round differently from an FP64 reference;
normalization can propagate that first rounding into more than one output ULP.
The tests retain the stage tolerances and bound their composition using the
FP32 reduction error, rather than assuming a single final BF16 rounding.

Both implementations support empty token batches. Shape, dtype and device
validation reads metadata only. Token counts and split extents are runtime
kernel arguments; compile-time tiles and reduction bounds use bounded
buckets. Consecutive sub-blocks compose `ihc_post` and `ihc_pre` explicitly.

Startup warmup compiles the gate reduction's finite split buckets, including
the runtime integer specialization classes used by smaller serving batches.
It uses compile-only calls, so preparation neither writes extra outputs nor
changes the split selection, reduction tree or precision boundaries.
