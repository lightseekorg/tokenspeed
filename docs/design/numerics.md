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
│   │                               and machine-dependent state)
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
│                                   fp32 fold; a ring all-reduce chunks by
│                                   message size, so its per-element order is
│                                   run-stable but not batch-size-invariant
│                                   (world_size x traffic; the NVLS multimem
│                                   in-switch reduction is the faster future
│                                   citizen of this slot)
├── sampling.deterministic          per-request Philox (seed=crc32(rid),
│                                   offset=position) is run- and
│                                   batch-invariant by construction; greedy
│                                   rows additionally overlay the canonical
│                                   lowest-index argmax, because EXACT logit
│                                   ties happen in practice and the pool
│                                   route's top-1 filter resolves them in
│                                   batch-shape-dependent reduction order
├── invariance.batch                per-row-independent reductions
│   ├── no split-KV attention       decode kernels whose split count scales
│   │                               with batch/SM occupancy are excluded by
│   │                               the batch_invariant feature
│   └── per-row GEMMs               fixed-order GEMM leaves (see aok below)
├── logprob.topology-invariant      (deferred) vocab-block fixed-tree
│                                   log-softmax, TP-count-invariant
└── alignment.trainer               (deferred) trainer operation order
```

Precedence: an explicitly set individual switch always stands; the envelope
only ever tightens. `resolve_numerics` runs after `resolve_communication` so
it can veto the auto-enabled all-reduce fusion.

## Kernel selection

The registry's two matching mechanisms split the work:

- **Traits are seller-declared**: a kernel declaring
  `deterministic={True}, batch_invariant={True}` documents itself, and a
  requested trait excludes only kernels that declare the opposite. Good for
  ranking, useless for guarantees.
- **Features are buyer-required** (subset test, silent kernels excluded):
  under rl-bitwise, callers on the model's hot path request
  `features={"batch_invariant"}`. Only leaves that affirmatively declare the
  feature can serve the call; a path with no such leaf refuses to start.
  This is the FluentLLM discipline ("no silent fallback") expressed through
  the existing registry.

Deterministic leaves live where any other vendor solution lives: registered
under `solution="aok"` (the fixed-reduction-order operator kit: GEMM family
including grouped MoE and BMM, lightning-indexer scoring, stable top-k with
native forced initial/local windows, no-split sparse MLA attention) at plugin
priority, alongside the performance leaves they mirror. `--numerics auto`
never selects them; `rl-bitwise` requires them.

Selection points a caller reaches through an existing solution switch reuse
it: rl-bitwise folds `--moe-backend auto` to `"aok"`, so the routed-expert
apply plans onto the batch-invariant grouped leaf through the ordinary
`moe_plan(solution=...)` path — and fails at startup when no such leaf is
registered. An explicitly chosen backend stands, like every other switch
under the envelope.

## Acceptance

The envelope is verified end to end, not per switch: the invariance harness
generates with returned logprobs for the same prompts (a) alone at bs=1,
(b) packed with random co-batches, (c) across repeated runs, and asserts
`torch.equal` on token ids and logprobs — base model and speculative decoding
each. A deployment that passes the harness may advertise the rl-bitwise
contract; one that fails it has a bug, not a tolerance.
