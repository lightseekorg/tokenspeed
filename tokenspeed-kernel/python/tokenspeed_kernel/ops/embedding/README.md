# Embedding kernels

Callers use `apply_rope` or `apply_qk_norm_rope`; both dispatch through the
`embedding.rope` registry. Backend implementations advertise the layouts and
fusions they support, so model code does not select vendor kernels directly.

Portable Triton implements QK normalization together with RoPE for complete
cross-vendor coverage. More specialized solutions may replace that kernel when
their shape and backend traits match.

MLA FP8 conversion keeps the existing FP32 scale multiplication and FP8
rounding. Query/key conversion tiles adjacent heads using model dimensions.
The MLA key/value packer groups adjacent token/head rows on Blackwell while
preserving caller-provided output strides, E4M3/E5M2 formats and PDL ordering.
Token counts remain runtime arguments, including under graph replay.
