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

## KVP decode value tail (`DcpValueTail`)

Under decode context parallelism a rank attends every gathered head over its
own keys, and each head's owner merges the ranks' partials before projecting
them: records dispatch, an all-to-all, `combine_dcp_attention_partials` and
`mla_project_value`, four launches with the exchange between them.

`create_dcp_value_tail` (NVIDIA SM100 to SM103, KVP groups of 2, 4 or 8)
builds a group's Lamport mailboxes in symmetric memory, and a
`DcpValueTail` call does all four in one launch
(the vendored `DcpValue` of `thirdparty/cute_dsl/k3_decode/mla/dcp.py`):

- Every CTA first stores its share of the LSEs, then of the BF16 partial rows,
  into slot `[parity][rank]` of the owners' mailboxes with peer stores. An
  armed slot holds a sentinel no sender stores (a negative-zero BF16 half goes
  as +0, a NaN LSE as the canonical NaN), so the data marks its own arrival.
- CTA `(head, tile)` polls the sources of 16 rows, re-arms what it read, and
  merges them as `combine_dcp_attention_partials` does: natural-log LSEs,
  `+inf` as `-inf`, the eight-lane max and sum of the Triton reduction,
  `ex2.approx` and `div.full` weights, and the sources added in rank order with
  unfused products. The BF16 merge then goes through the projection and gate
  of `cute_dsl_mla_project_value`, so a call is the bits of the four launches.
- Successive calls alternate the two parities by a call count in device
  memory, so a rank stores into a parity only after every peer finished
  reading the call before. The grid holds at most one CTA per SM, so every
  CTA's stores land while the merging CTAs poll.

On GB300, eight ranks over two nodes, 12 heads per rank, CUDA graphs of back
to back calls, us per call:

| rows | tail | dispatch, LL all-to-all, combine, projection | with NCCL |
|-----:|-----:|-----:|-----:|
| 8 | 10.5 | 12.2 | 23.8 |
| 32 | 14.5 | 20.9 | 28.9 |
| 64 | 18.6 | 33.2 | 39.7 |
| 128 | 28.3 | | 57.3 |
| 512 | 100.3 | | 163.5 |
