# Residual kernels

## HY V4 independent Hyper-Connections (iHC)

iHC keeps `hc_mult` parallel residual channels shaped
`[..., hc_mult, hidden_size]` and, unlike mHC, has no combination matrix.

- `ihc_pre` reduces the channels into one sub-block input and emits that
  sub-block's post gates.
- `ihc_post` scatters the gated sub-block output across every channel and
  adds the residual.
- `ihc_head` merges the channels into one hidden state before the final norm.

The channels may be BF16 or FP32; HY V4 keeps them in FP32 across the decoder
stack. The projection `fn`, scales `hc_scale` and biases `hc_base` are FP32,
and every gate is evaluated in FP32. A pre boundary has `2 * hc_mult` gates
(pre, then post) and two scales; the head has `hc_mult` gates and one scale.

The public entry points check, before dispatch, that `hc_mult > 1`, that
`fn` is `[gates, hc_mult * hidden_size]`, `hc_scale` is `[scales]`,
`hc_base` is `[gates]` and `norm_weight` is `[hidden_size]`, that these
parameters are contiguous, that `ihc_post`'s `block_output` is
`[..., hidden_size]` and its `gates` `[..., hc_mult]` over the residual's
leading dimensions, and that every tensor is on the residual's device. Any
mismatch raises `ValueError`. The checks read tensor metadata only, so they do
not synchronize and are safe under CUDA graph capture. Outputs keep the
residual's leading dimensions in every implementation.

### Numerics

A boundary flattens every token's channels into one `hc_mult * hidden_size`
row and projects the raw row: the RMS factor
`rsqrt(mean(row^2) + rms_eps)` scales the projection output `mix`, never the
row.

- Channel `c`'s pre gate is `sigmoid(mix[c] * hc_scale[0] + hc_base[c]) +
  hc_eps`, and its post gate is
  `magnitude * sigmoid(mix[n] * hc_scale[1] + hc_base[n]) + hc_eps` with
  `n = hc_mult + c`. The head's gates take the pre form.
- The pre-gated channel sum is accumulated in FP32 and rounded to the residual
  dtype. The optional fused RMSNorm reads that rounded row and runs in FP32
  with the weight in FP32, so the BF16 sub-block input is rounded once after
  the norm, as by a separate RMSNorm kernel.
- `ihc_post` computes `residual + gate * block_output` in FP32 and stores the
  residual dtype, so FP32 channels see no BF16 rounding there.
- `ihc_head` merges the channels in FP32 and rounds the result to BF16 once.

### Triton implementation

A pre or head boundary is two kernels:

1. A split-K projection reads every flattened row once, tiled over tokens so
   the projection weight is reused across a token block, and accumulates the
   row's sum of squares alongside.
2. A per-token kernel folds the splits, evaluates the gates, writes the post
   gates and reduces the channels with the pre gates, fusing the RMSNorm.
   Hidden sizes above 8192 are reduced in chunks, which the fused norm walks
   twice.

Against BF16 channels the FP32 projection weight is split on the fly into
three BF16 terms (hi + mid + lo carry all 24 mantissa bits). Each BF16 product
is exact in the FP32 accumulator, so one tensor-core dot reproduces the FP32
projection up to summation order. FP32 channels use plain FP32 FMAs, one per
gate lane, with the gate count padded to a power of two of at least four.

The tile shapes and the split count follow the token count. Results are
reproducible run to run, but the projection's summation order, and with it
the last bits of the gates, can change with the batch size.

The Triton pre boundary registers for `hc_mult` 2 through 8 (at most 16
gates) and the head for 2 through 16; wider boundaries select the portable
torch implementation. `ihc_post` has no channel limit.
