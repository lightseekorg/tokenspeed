# PLE kernels

## N-gram IDs and context

N-gram hashing uses one lane per token/head pair. Only head zero writes
shared indices and raw context tails; each head retains the XOR prefix for
its n-gram length and performs one int64 remainder. Hashing respects EOS
boundaries, while saved tails retain the raw token IDs for later steps.
Launch sizing includes request-only carried rows for empty batches.

`prepare_ngram_reciprocals` accepts host-known positive int64 moduli. Pass its
uint64 result explicitly as `ple_ngram_ids(mod_reciprocals=...)`; `None` selects
integer division. Preparation happens outside capture, once for the same
immutable moduli. The model prepares a non-persistent buffer from config-generated
moduli and refreshes it if checkpoint loading replaces them. The buffer follows
module device moves without per-forward preparation or a new checkpoint field.

For `d > 1`, let `m = floor(2^64 / d)`. For unsigned `n < 2^63`,
`q = mulhi(n, m)` underestimates `floor(n / d)` by at most one. Thus
`r = n - q*d` is corrected by subtracting `d` once if `r >= d`; `d = 1`
returns zero. There is no FP64 approximation or 32-bit hash truncation.
Negative hashes are outside the existing PLE contract.

## Host lookup and page gathers

Host gather masks both host table reads and per-row scale reads by shard
ownership. Out-of-shard outputs are explicitly zeroed. FP8 conversion and
optional scalar or per-row scaling happen in the gather kernel.

`ple_page_gather_pair` reads context and convolution state for the same page IDs
in one launch, using each field's page stride. Null pages produce the supplied
context default and zero convolution state. Page scatter skips null pages.

## Convolution and state

Convolution and final-state helpers are inlined into one launch. Program `i`
handles token `i` and request `i` independently, covering zero-length requests
without a separate state launch. This removes launch overhead, not all duplicate
input reads. Verify scratch supplies both carried and per-token checkpoints,
so the runtime disables the unused final-state allocation and computation there.
The public facade requires an explicit `write_final`; a disabled final result
has shape `[0, channels, state_len]`.

The fused convolution/state kernel takes token and request counts as runtime
scalars. These counts bound token output and per-request state writes; they do
not determine static tile sizes. Eager batches with different counts reuse a
compiled kernel within Triton's scalar specialization buckets. CUDA graphs
capture the scalar values supplied at capture time.

## PDL synchronization

PDL retains a wait before producer-owned reads. Static conv weights may load
before that wait only with `weights_independent=True` and no contiguity copy.
Gate norm triggers after gated-value stores, before normalized-value reduction;
the fused conv triggers after its incoming wait. Neither trigger publishes
results, and downstream consumers must still synchronize before reading them.
Overlap is opportunistic, not a correctness assumption. Host gather uses the
ordinary stream dependency rather than PDL.

## Validation

`test/runtime/test_qwen4_exp.py` covers shard boundaries, FP8/scales, int64 IDs,
EOS/tails, convolution references, empty/ragged requests and verify scratch.
`test/nvidia/ops/test_ple_prototypes.py` checks gate normalization against an
eager reference, exact reciprocal remainder, EOS and raw tails, and graph replay
with PDL on and off. IDs and context/state windows require exact equality.

The convolution regression tests in `test/nvidia/ops/test_ple_prototypes.py`
cover ragged and empty requests, verification windows, final states, CUDA graph
replay, and compiled-kernel reuse across token and request counts.
