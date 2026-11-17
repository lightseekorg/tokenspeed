# MLA kernels

## Value projection (`mla_project_value`)

After absorbed MLA decode, each head's `[512]` latent output is projected
through `w_vc` (`[heads, 512, 128]`) and, for gated layers such as the Kimi-K3.1
DSA layers, multiplied by `sigmoid(gate)`. The unfused path is a batched GEMM
followed by `sigmoid_mul`: the projection rounds to BF16, then the gate
multiplies in FP32 and rounds again.

`cute_dsl_mla_project_value` (NVIDIA SM100 to SM103) does both in one launch
for the latent-contiguous `w_vc` that absorbed decode weights use outside
bitwise numerics. One 128-thread CTA covers 32 outputs of one head for 16
rows: each warp keeps its 8 outputs' `w_vc` fragments in registers (loaded
before the programmatic dependency wait), stages the rows' latents in shared
memory, and multiplies with BF16 `mma.sync` into two FP32 accumulator chains.
It keeps the unfused rounding points and the Triton sigmoid, so only the
projection's summation order differs: the projection may move by one BF16
step, and the gate scales that step before rounding again.
Row counts and row strides are runtime values; the gate may be a column view
of a wider projection row.
