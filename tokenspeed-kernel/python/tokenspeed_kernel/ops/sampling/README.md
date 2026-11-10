# Sampling kernels

## Distributed argmax (`cute_dsl.py`)

`distributed_argmax` returns the greedy token of logits left sharded over the
vocabulary of a tensor-parallel group. Each rank reduces its shard to one
`(value, global id)` candidate per row, multicasts it as one 64-bit word
(`[ready flag | id | FP32 value]`) to slot `(row, band, rank)` of a
symmetric-memory buffer on every rank, waits for the group's slots of the
row, clears their flags and keeps the largest value, lowest id on ties.
Values that never beat -inf (-inf, NaN) do not compete; a row with no
candidate yields 0.

Consecutive calls alternate between two slot bands, selected by a per-rank
`round_id` that each launch reads before any row is exchanged and advances
once after every row is done. A rank can only reuse a band after its peers
have read it, so back-to-back launches need no barrier. Every kernel sharing
one `DistArgmaxState` follows this band order, so their calls may interleave;
the group's ranks must issue them in the same order.

## Fused DSpark Markov step (`markov_argmax_step`)

A greedy DSpark walk proposes block position `k` as the argmax of
`round(base[k - 1] + round(embedding[prev] @ projection.T))`, where `prev` is
the token just proposed for `k - 1` (clamped to the embedding), the bias is
accumulated in FP32 and both roundings go to the logits dtype -- the
unfused path's BF16 GEMM output and BF16 add. `markov_argmax_step` does one
position in one launch and writes the token into the next column of the
caller's token table, so the seven positions of a K3 walk are seven
dependent launches instead of about fifty.

Inside the launch:

* Grid `(shard_rows / 64, ceil(rows / rows_per_cta))`, 256 threads. A CTA
  gathers its rows' latent vectors (one 1 KB row each at rank 512) into
  shared memory with `cp.async`; the row-tile count (8, 16, 32 or 64 rows per
  CTA) is a compile-time specialization.
* Each warp pair owns 16 projection rows and splits the rank dimension in
  two. Lanes load their projection words straight from global memory and
  multiply them by the latents with `mma.m16n8k16` (the projection rows as
  M, the requests as N). Both operands read the same permutation of rank
  columns, so each lane's loads are two contiguous 16-byte words per row.
* The pair adds its halves in FP32, rounds to the dtype, adds the base logit,
  rounds again and keeps the best `(value, lowest id)` per request as one
  ordered int64 key; the CTA folds its keys into a per-row global key with
  `red.max`.
* The last CTA to finish (a global counter) reads the keys and runs the
  distributed argmax's handshake for every row on the state's slots, then
  resets the keys and the counter for the next launch.

The scratch is allocated and every specialization compiled by
`create_markov_argmax_workspace` before graph capture; each launch leaves the
scratch as it found it, so captured walks replay as is. Against the unfused
path the only difference is the FP32 summation order of the bias, which can
flip a BF16 rounding and so a near-tie. In-shard exact ties go to the lowest
id; `distributed_argmax`'s per-thread scan does not guarantee that.
