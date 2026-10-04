# Numerics envelopes

`--numerics` names, at launch level, the numerical contract a deployment
promises. Every capability under it exists as an individual switch; the
envelope's whole job is to keep the set coherent, because RL rollout brought
us a class of deployment where one missing switch silently invalidates the
training signal.

## The contract

`--numerics rl-bitwise` promises, within one deployment (fixed world size,
parallel layout, model, kernels):

1. **Run invariance** — the same request produces bitwise-identical tokens
   and logprobs across runs.
2. **Batch invariance** — a request's tokens and logprobs do not depend on
   which other requests share its batches, or on how the scheduler happened
   to chunk and batch it.

It deliberately does **not** promise (yet — the layers exist in the
hierarchy, unimplemented):

3. **Topology invariance** — the same logprobs under a different TP/DP
   factorization (needs vocab-block-invariant log-softmax and TP-invariant
   projection layouts).
4. **Trainer alignment** — bitwise equality with the training framework's
   forward (needs the trainer's operation order: CPU-computed YaRN ramp
   masks, unscaled LoRA norms, unfused first RMSNorm, matching collectives).

## The hierarchy

```
numerics.mode                       --numerics {auto, rl-bitwise}
├── kernels.deterministic           fixed-reduction-order compute
│   ├── no autotune                 disable_autotune (tactic choice is shape-
│   │                               and machine-dependent state); no
│   │                               persistent tactic cache is loaded
│   ├── no TF32                     disable_tf32 + NVIDIA_TF32_OVERRIDE=0
│   ├── no PDL                      disable_pdl (serialize kernel chains)
│   └── batch-invariant leaves      kernel registry: leaves declaring the
│                                   "batch_invariant" feature; a caller in
│                                   rl-bitwise REQUIRES the feature, so a
│                                   missing implementation fails at startup
│                                   instead of silently falling back
├── collectives.deterministic       one association order per reduction
│   ├── force_deterministic_rsag    NCCL instead of symmetric-memory paths
│   ├── no fused AR+norm            enable_allreduce_fusion=False (the fused
│   │                               kernels make no bitwise claim)
│   ├── NCCL_ALGO=Ring,             the algorithm/protocol switch by message
│   │   NCCL_PROTO=Simple           size changes association order
│   └── batch_invariant_collectives all-reduce = all-gather + fixed-rank-order
│                                   fp32 fold, reduce-scatter (plain and
│                                   token) = all-to-all + the same fold; a
│                                   ring reduction chunks by message size, so
│                                   its per-element order is run-stable but
│                                   not batch-size-invariant (all-reduce pays
│                                   world_size x traffic, reduce-scatter
│                                   none; the NVLS multimem in-switch
│                                   reduction is the faster future citizen of
│                                   this slot)
├── sampling.deterministic          per-request Philox (seed=crc32(rid),
│                                   offset=position) is run- and
│                                   batch-invariant by construction; greedy
│                                   rows additionally take the canonical
│                                   lowest-index argmax, in sampling and in
│                                   speculative verify (exact-match chain),
│                                   because EXACT logit ties happen in
│                                   practice and the pool route's stochastic
│                                   kernels resolve them in batch-shape-
│                                   dependent reduction order; backends
│                                   without the overlay are refused. Under
│                                   --enable-speculative-sampling the draft
│                                   proposal is one more per-request stream:
│                                   Gumbel-max noise keyed by the request's
│                                   seed and a salted (position, step) offset
│                                   (never the batch row), the verify coins
│                                   stay per-slot, and greedy rows keep the
│                                   canonical argmax with a one-hot q, so
│                                   their verify is unchanged
├── invariance.batch                per-row-independent reductions
│   ├── no split-KV attention       decode kernels whose split count scales
│   │                               with batch/SM occupancy are excluded by
│   │                               the batch_invariant feature
│   ├── row-local top-k             the DSA indexer's selection resolves
│   │                               equal scores toward the lowest candidate
│   │                               within each row (dsa_*_topk
│   │                               batch_invariant=True); the tuned top-k
│   │                               kernels switch algorithm and CTA split
│   │                               with the row count, which moves ties
│   └── per-row GEMMs               fixed-order GEMM leaves (see aok below)
├── logprob.topology-invariant      (deferred) vocab-block fixed-tree
│                                   log-softmax, TP-count-invariant
└── alignment.trainer               (deferred) trainer operation order
```

Precedence: the envelope only ever tightens. It sets every switch it governs
to its tight value, and it refuses an explicit choice it cannot tighten — a
named MoE backend other than the batch-invariant one, a sampling backend
without canonical greedy ties — rather than keeping it and silently voiding
the contract. `resolve_numerics` runs after `resolve_communication` so it can
veto the auto-enabled all-reduce fusion.

## Kernel selection

The registry's two matching mechanisms split the work:

- **Traits are seller-declared**: a kernel declaring
  `deterministic={True}, batch_invariant={True}` documents itself, and a
  requested trait excludes only kernels that declare the opposite. Good for
  ranking, useless for guarantees.
- **Features are buyer-required** (subset test, silent kernels excluded):
  a leaf that affirmatively declares `features={"batch_invariant"}` is the
  only kind a batch-invariant request can be served by.

Deterministic leaves live where any other vendor solution lives: registered
under `solution="aok"` (the fixed-reduction-order operator kit: GEMM family
including grouped MoE and BMM, lightning-indexer scoring, stable top-k with
native forced initial/local windows, no-split sparse MLA attention) with the
`batch_invariant` feature, at reference priority so `--numerics auto` never
selects them. Under rl-bitwise each selection point on the served path pins
that solution: the DSA backend's sparse decode and prefill, the MLA
absorption and value projections, the dense and LM-head GEMMs, and the MoE
plan (the envelope folds `--moe-backend auto` to `"aok"`, so the
routed-expert apply plans through the ordinary `moe_plan(solution=...)`
path). A pinned solution with no registered leaf fails selection at startup
or at the first call instead of falling back — the FluentLLM discipline
("no silent fallback") expressed through the existing registry.

## Logprobs: one arithmetic for prompt and output

A returned logprob is `log_softmax(logits, -1, dtype=float32)` gathered at
the token -- `gather_token_logprobs_torch` in `sampling/utils.py`, the one
function both consumers call -- with the logits produced by the same LM-head
route the sampler takes (`LogitsProcessor._get_logits`: quantized or dense
GEMM, the `aok` GEMM under rl-bitwise, the TP gather, softcap). The sampler's
output logprobs and the prompt (input) logprobs of the SGLang dialect
(`LogitsProcessor.compute_input_token_logprobs`, requested through
`return_logprob` + `logprob_start_len`) share that function, so the logprob of
one token is the same number whether it was scored as a prompt position or
sampled as an output -- the property an RL trainer relies on when it rescores
a rollout. The `dtype=float32` form widens bf16 logits inside the kernel (an
exact conversion) instead of materializing an fp32 copy of the `[rows, vocab]`
tensor first; because both paths go through the one function, whatever
rounding the kernel applies is applied to both.

The TP gather differs in one respect that is not numerics: the sampled rows
may take the multicast all-gather, whose result is a view of the group's
shared buffer (safe because a whole forward separates consecutive calls),
while the prompt-row chunks ask `_get_logits` for a private full-vocab tensor
(`require_full_vocab=True`) and gather through the NCCL collective -- a chunk's
log-softmax may still be reading its result when a faster rank issues the next
chunk's gather. The gathered values are identical either way.

Prompt logprobs are gathered in position chunks of
`--input-logprob-chunk-tokens` rows so the transient `[rows, vocab]` logits
stay bounded. The chunk size is a sizing knob, not a numerics one: the
reductions involved are row-local (the GEMM's row is independent of its
neighbours under the per-row GEMM leaves of `invariance.batch`, and
log-softmax reduces within a row), so no value depends on which chunk, or how
large a chunk, a position landed in. The same holds across prefill chunks:
positions are scored by the chunk that feeds them and assembled per request
afterwards, so chunked prefill and prefix-cache hits do not change a prompt
logprob either -- the admission probe is capped at `logprob_start_len`
(`scheduler.md` §1) so every scored position is actually recomputed.

## Expert placement and online rebalancing

An expert placement (`--ep-num-redundant-experts`, `--init-expert-location`)
decides which rank computes which route. Under the rank-order MoE combine
each rank's leaf returns a partial over its local slots and the host folds
the partials in rank order, so the placement decides which routes land in
which partial and a different placement moves the fold's rounding: the output
is a function of the placement. A static placement is fixed per deployment
and keeps the run-invariance contract; `--enable-eplb` makes the placement
traffic-dependent state, so under the rank-order combine a rebalanced
deployment is not run-invariant — not only across a rebalance, but across
runs that rebalanced differently. The envelope therefore requires a
placement-independent combine with `--enable-eplb`: every route computed on
exactly one rank by a row-invariant leaf and the routes of a token folded in
slot order (`--moe-combine-order slot`), which makes each route's value a
pure function of the token and its logical expert's weights — replicas are
byte-identical copies — and the output bitwise identical across rebalances.
A build without that combine refuses `--enable-eplb` under the envelope. The
load counters, the CPU-side algorithm and the P2P copies never enter the
arithmetic; the dispatch algorithm stays static (required under rl-bitwise
already) and drafts stay trivially placed.

## Acceptance

The envelope is verified end to end, not per switch: the invariance harness
generates with returned logprobs for the same prompts (a) alone at bs=1,
(b) packed with random co-batches, (c) across repeated runs, and asserts
`torch.equal` on token ids and logprobs — base model and speculative decoding
each. A deployment that passes the harness may advertise the rl-bitwise
contract; one that fails it has a bug, not a tolerance.

The pins above cover only the paths a verified model takes, so the
verification is recorded per model and enforced at startup: a model profile
lists the envelopes its model passes in `ModelProfile.numerics_envelopes`,
and launching an envelope other than `auto` refuses any target or draft
model that does not list it — every in-tree model included, since none has
a profile. Quantized checkpoints are refused too: no batch-invariant
quantized GEMM leaf exists, so their linears would select shape-dependent
ones.
