# Depth attention residuals

The depth mix weighs completed depth sources and the current candidate independently for each token and FP32 pseudo-query.
Scores use weightless RMS-normalized keys and pseudo-queries scaled by `hidden_size**-0.5`.
Values can remain raw or use the same normalization.

`attention.residual_sum` sums sources in FP32 and stores the original dtype once.
`attention.residual_weighted_sum` mixes `[tokens, S, width]` sources with fixed weights, adding the FP32 products in source order.
These operations do not mutate their inputs.

The mix runs in two steps, on the split kernels Kimi-K3 uses for its `attn_res_fwd` decode path (`activation/triton.py`).
`attention.attention_residual_partials` sweeps the completed sources once per (token, query) and keeps an online-softmax maximum, sum and weighted FP32 accumulator.
`attention.attention_residual_combine` folds the current candidate into those statistics and finishes the mixture.
Its candidate is the FP32 sum of up to three rows, mixed unrounded; with `candidate_out` the sum rounds once to that tensor, as `residual_sum` would, and the rounded value is mixed.
It can apply the following RMSNorm before the single output store; otherwise the mixture is FP32.
The FP32 association differs from a two-pass softmax, within the FP32 tolerances of the tests.
The K3 calls keep their own contract: raw values, one query, and a BF16-rounded mix.
Under PDL, the combine of `attnres_combine_queries` triggers its dependents before its own wait.

The runtime `AttentionResidualState` retains sources within one forward only.
Completed blocks live in one `[blocks, tokens, hidden]` buffer: the embedding at boundary layer zero, then each block's partial sum.
The partial sum of the current block stays as pending terms; the next attention-side mix rounds and stores it.
Every mix of a block set reads the same blocks, so small forwards sweep each set once for all of its queries.
Larger forwards mix each query in one launch, `attention.attention_residual_mix`: one program per (token, query) runs the sweep and the candidate fold and keeps the statistics in registers instead of FP32 scratch, over exact power-of-two column tiles (4096 + 512 for 4608).
It computes the same equations with another FP32 summation order.
With a token range and a destination it mixes only those rows: a sequence-parallel rank (`ResidualShare` in the runtime state) mixes its share of the tokens into a multicast buffer, and after a barrier every rank holds the whole mix.
This state does not add a per-request cache group.

Token counts are runtime scalars of the kernels.
Source count, hidden width, query count, and normalization policy are model properties.
CPU tests cover equations, boundary transitions, and rounding against an FP64 host reference.
Shared GPU tests compare the kernels with the CPU equations.
