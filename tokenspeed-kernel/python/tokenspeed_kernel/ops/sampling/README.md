# Sampling kernels

## CUDA fused Top-K / Top-P

The multi-block Top-K radix pass publishes each CTA's histogram updates before
thread 0 increments `finished_block_cnt`. Every writer executes a device fence,
then all threads meet at a CTA barrier before that completion notification.
The last CTA can then scan the complete histogram, choose the next bucket and
clear the histogram without racing a late writer. The barrier after the
completion notification only distributes the last-CTA decision within a block;
it cannot order another CTA's earlier writes.

Accurate bucket counts bound both the survivor buffers and the K-element output.
An undercount can incorrectly trigger `len == k` early termination and write more
than K results. In the fused workspace, the last row's index output is adjacent
to the Top-P counters, so this can corrupt a Top-P input pointer even though the
two stages have disjoint workspace regions and correct stream dependencies.
The same synchronization is required for eager execution and graph replay.

`test/nvidia/ops/test_sampling_topk_sync.py` runs the real radix kernel with
delayed global histogram atomics. It checks bucket counts, stable Top-K results
and Top-P counter integrity for uniform and K+1 boundary inputs in both modes.

## Packed probability sampling (`packed_verify.py`)

`sample_probs` draws from already filtered and normalized FP32 rows.
`verify_packed_chain` verifies CSR-packed target rows against the draft laws
recorded when each proposal was drawn. Candidate column zero is the anchor;
target row `i` verifies candidate column `i + 1` against draft row `i`.
A request with `q` target rows verifies at most `q - 1` proposals and emits
one residual or bonus token. A zero-row padding request emits nothing.
The probability allocation may contain unused physical rows beyond the CSR
boundary. No such row, or bonus draft row, is read.

The row mapping and rejection rule follow
[DeepInfer's sampler](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py).
The CuTe implementation operates on normalized probabilities; its RNG and
FP32 scan have the explicit contracts below.

Verification accepts a proposal `x` when `u < min(1, p(x) / q(x))`.
At the first rejection it samples `max(p - exp(-5e-5) * q, 0)` normalized by
its total mass. A recorded `q(x) > reject_draft_prob_threshold` marks an
unavailable proposal and samples the full target law instead. When every
proposal passes, the last target
row supplies the bonus. For a zero-probability proposal, acceptance is one
if its target probability is positive and zero otherwise. The outputs retain
the fixed candidate width: `accept_length` includes the final token;
`accept_index` contains `request * width + column` for that output prefix
and -1 after it, while unused `predicts` entries are zero.

The NVIDIA CuTe leaf uses one 256-thread CTA per request. Threads sum the
vocabulary cooperatively and scan 1024-token tiles until the inverse CDF
crosses the draw. It selects the first strict `CDF > u * total_mass` crossing,
with the lowest-index argmax of the sampled weights as a roundoff fallback.
A genuinely zero residual mass falls back to the target's lowest argmax.
Proposal rows are read from the request-slot pool using `draft_pool_indices`;
the runtime does not gather a batch-sized copy of the dense proposal table.
Target-row and proposal-slot base addresses use 64-bit offsets, including
the scalar acceptance reads. Far-end tests cover acceptance, rejection, and
proposal sampling beyond 2^31 elements. Verification tests initialize only
the addressed rows and slots.
FP32 reduction order differs from CUDA/CUB and TileLang, so coins close to CDF boundaries
can select different ids. This is a probability-law contract, not bitwise
agreement with either implementation.

Random numbers use Philox4x32-10. The key is the low and high 32-bit words of
the request seed; the counter is `(position_low, position_high, rng_tag, 0)`.
The upper 24 bits of result word zero, divided by `2**24`, form a draw in
`[0, 1)`. `sample_probs` takes the encoded position directly. Verification
encodes local step `i` as `positions[request] + i * 10000019`, modulo `2**64`.
Depth streams are disjoint while a request's absolute-position span is below
10,000,019; the native V4.1 1M context window fits within this range.
Exported tags are `RNG_TAG_TARGET=0`, `RNG_TAG_DRAFT=1`, `RNG_TAG_ACCEPT=2`,
`RNG_TAG_REJECT=3` and `RNG_TAG_BONUS=4`. The one-row verify case uses the
bonus namespace too. This mapping is invariant to batch order, CSR packing
and graph padding; its random bits need not match TileLang's generator.

The compiled-function cache specializes only on device, operation, vocabulary
and candidate width. Row counts, CSR values and tensor strides are runtime
inputs. Warm the selected geometry before graph capture; replay uses the
cached executable. Launches use ordinary CUDA stream ordering with PDL
disabled, and CPU calls have no production fallback.

## Distributed argmax (`cute_dsl.py`)

`distributed_argmax` returns the greedy token of logits left sharded over the
vocabulary of a tensor-parallel group. Each rank reduces its shard to one
`(value, global id)` candidate per row, multicasts it as one 64-bit word
(`[ready flag | id | FP32 value]`) to slot `(row, band, rank)` of a
symmetric-memory buffer on every rank, waits for the group's slots of the
row, clears their flags, and keeps the largest value, lowest id on ties.
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
  shared memory with `cp.async`; the row-tile count (8, 16, 32, or 64 rows per
  CTA) is a compile-time specialization.
* Each warp pair owns 16 projection rows and splits the rank dimension in
  two. Lanes load their projection words straight from global memory and
  multiply them by the latents with `mma.m16n8k16` (the projection rows as
  M, the requests as N). Both operands read the same permutation of rank
  columns, so each lane's loads are two contiguous 16-byte words per row.
* The pair adds its halves in FP32, rounds to the dtype, adds the base logit,
  rounds again, and keeps the best `(value, lowest id)` per request as one
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

## Synthetic verify outputs (`write_synthetic_verify_outputs`)

A benchmark run with a synthetic acceptance length `L` replaces each verify
row's outputs: the drafts `candidates[r, 1:L]`, then the target token, zeros
after them, the flat candidate index of each accepted token (`-1` past `L`)
and `L - 1` accepted drafts. The target token is either given per row or read
from the row's target predictions at column `L - 1`, and `L` past the verify
width counts as the width. One Triton program writes one row, so a captured
step pays one launch for the override; the CPU path is the torch reference.
