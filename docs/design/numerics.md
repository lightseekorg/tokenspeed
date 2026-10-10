# Numerics envelopes

`--numerics` names, at launch level, the numerical contract a deployment
promises. Every capability under it exists as an individual switch; the
envelope's whole job is to keep the set coherent, because RL rollout
introduced a class of deployment where one missing switch silently
invalidates the training signal.

## The contract

`--numerics rl-bitwise` is the one bitwise envelope. It promises, within one
deployment (fixed world size, parallel layout, model, kernels):

1. **Run invariance** — the same request produces bitwise-identical tokens
   and logprobs across runs.
2. **Batch invariance** — a request's tokens and logprobs do not depend on
   which other requests share its batches, or on how the scheduler happened
   to chunk and batch it.
3. **Trainer alignment** — the forward follows the training framework's
   operation order wherever the two engines are known to differ, so that a
   teacher-forced pass over a trainer-generated sequence reproduces the
   trainer's per-token logprobs bitwise. The switches this tightens are
   listed under `alignment.trainer` below; each exists individually for
   `auto`.

It deliberately does **not** promise (yet — the layer exists in the
hierarchy, unimplemented):

4. **Topology invariance** — the same logprobs under a different TP/DP
   factorization (needs TP-invariant projection layouts on top of the
   vocab-block log-softmax of `alignment.trainer`).

Two consequences of 3 are enforced at startup, so a model that cannot meet
them is refused under rl-bitwise with the reason: MoE TP must be 1 (the
slot-order MoE combine folds over the EP group inside the leaf), and the
vocabulary must be a whole number of `MEGATRON_VOCAB_BLOCK`-wide blocks (the
trainer's log-softmax folds fixed blocks).

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
│   ├── no fused AR+norm            enable_allreduce_fusion=False (the fused
│   │                               kernels make no bitwise claim)
│   ├── NCCL_ALGO=Ring,             the algorithm/protocol switch by message
│   │   NCCL_PROTO=Simple           size changes association order
│   ├── batch_invariant_collectives AutoBackend.route, one decision for every
│   │                               collective: a 2-D bf16 all-reduce on a
│   │                               multicast-reachable group the startup
│   │                               self-check verified is the NVLS in-switch
│   │                               reduction issued by one fixed rank
│   │                               (section below); every other
│   │                               reduction is NCCL data movement plus a
│   │                               fixed-rank-order fp32 fold (all-reduce =
│   │                               all-gather + fold at world_size x traffic,
│   │                               reduce-scatter = all-to-all + fold at no
│   │                               extra traffic). A ring reduction chunks
│   │                               by message size, so its per-element order
│   │                               is run-stable but not batch-size-
│   │                               invariant. Gathers move data and keep the
│   │                               multicast kernels. The route reads only
│   │                               static properties of the call site (group,
│   │                               dtype, rank, width), never the row count,
│   │                               so a site cannot change route with the
│   │                               batch; a payload past the switch buffer
│   │                               raises rather than reroutes
│   ├── data movement               all-gather, token all-gather, all-to-all
│   │                               (even or uneven, the transposes of the
│   │                               decode TP layouts) move bytes without
│   │                               reducing: nothing to fold, no branch
│   └── force_deterministic_rsag    the user's "NCCL and the fold only" knob:
│                                   no symmetric-memory path at all (multicast
│                                   gathers, in-switch reduction, the trtllm
│                                   and Triton all-reduce tiers, distributed
│                                   argmax). Not folded by the envelope,
│                                   honoured first by every route
├── sampling.deterministic          sampling_stream=per-request: sampled rows
│                                   draw from the Gumbel-max pool kernels,
│                                   whose stream is keyed by the request's
│                                   seed (crc32(rid)) and position (its cache
│                                   length) and nothing else, so it is run-
│                                   and batch-invariant by construction.
│                                   flashinfer's *_sampling_from_probs
│                                   kernels are NOT: they read one seed and
│                                   offset for the whole batch and seed
│                                   curand with the batch row, so a request's
│                                   draw moves with its co-batch (T>0
│                                   'packed' fails while 'rerun' passes).
│                                   Verify keeps flashinfer's chain kernels,
│                                   whose coins come from per-slot
│                                   generators. Greedy rows additionally take
│                                   the canonical lowest-index argmax, in
│                                   sampling and in speculative verify
│                                   (exact-match chain), because EXACT logit
│                                   ties happen in practice and the pool
│                                   route's stochastic kernels resolve them
│                                   in batch-shape-dependent reduction order;
│                                   backends without the overlay are refused.
│                                   Under --enable-speculative-sampling the
│                                   draft proposal is one more per-request
│                                   stream: Gumbel-max noise keyed by the
│                                   request's seed and a salted (position,
│                                   step) offset (never the batch row), the
│                                   verify coins stay per-slot, and greedy
│                                   rows keep the canonical argmax with a
│                                   one-hot q, so their verify is unchanged
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
│   ├── position-order reduction    dsa_slot_order=sorted: a token's selected
│   │                               KV rows are reduced in ascending POSITION
│   │                               order, not in the top-k leaf's tie order
│   │                               and never in physical slot order (a
│   │                               request's pages are allocated in arbitrary
│   │                               id order once pages recycle, so slot order
│   │                               follows the page placement and differs
│   │                               between runs and engines). The top-k leaf
│   │                               emits that order (dsa_decode_topk /
│   │                               dsa_prefill_topk slot_order trait) and the
│   │                               sparse core keeps it (dsa_decode /
│   │                               dsa_prefill slot_order trait); silent
│   │                               kernels are refused rather than assumed
│   └── per-row GEMMs               fixed-order GEMM leaves (see aok below)
├── logprob.topology-invariant      (deferred) TP-invariant projection
│                                   layouts on top of the vocab-block
│                                   log-softmax of alignment.trainer
└── alignment.trainer               the trainer's operation order (section
                                    below); rl-bitwise folds every switch
```

Precedence: the envelope only ever tightens. It sets every switch it governs
to its tight value, and it refuses an explicit choice it cannot tighten — a
named MoE backend other than the batch-invariant one, a sampling backend
without canonical greedy ties — rather than keeping it and silently voiding
the contract. `resolve_numerics` runs after `resolve_communication` so it can
veto the auto-enabled all-reduce fusion. Every closed-set switch is validated
once, in `ServerArgs`; the layers that read a resolved switch trust it, and
a constraint that needs more than the launch (a model's vocabulary, its
routing) is checked when that module is constructed, never per forward.
Every selection point that pins a batch-invariant leaf tests `numerics in
BITWISE_ENVELOPES`, never the one name, so an envelope added above
rl-bitwise would inherit every pin.

## alignment.trainer

The invariance layers make a deployment agree with itself. Trainer alignment
makes it agree with a different program: the training framework's forward,
whose arithmetic was never written to match an inference engine. Each switch
below replaces one operation the engine performs differently from the trainer
with the trainer's form. The table says what each switch changes: **forward
values** means every activation downstream moves (tokens can flip at ties,
logprobs change); **logprobs only** means the sampled tokens are untouched
and only the reported log-probabilities change.

| Switch | Trainer form | Changes |
| --- | --- | --- |
| `--sampling-stream per-request` | Non-greedy rows draw from a Philox stream keyed by `(request seed, position)` only (`sampling.deterministic` above); the trainer plays back the sampled ids, so this is an invariance switch rather than an alignment one — T>0 rollouts need it | tokens at T>0 (not logprobs) |
| `--yarn-ramp-mask-device cpu` | The whole `inv_freq` table of `deepseek_yarn` RoPE — position frequencies, both divisions and the YaRN linear ramp mask — is computed on the host and copied to the device once, as the trainer builds its `inv_freq` on the host; CPU and CUDA division round differently at ulp level, and every rotated q/k inherits the difference | forward values |
| `--mla-lora-scale runtime` | The `sqrt(hidden / lora_rank)` norm scales of LongCat-style MLA stay out of the `q_a_layernorm` / `kv_a_layernorm` weights and multiply `q` after `q_b_proj` and the latent after `kv_a_layernorm` in bf16, as the trainer does; the DSA indexer reads the unscaled `q_lora` | forward values |
| `--layer-boundary-norm unfused` | The norm that opens each physical layer and the final norm read a bf16 `hidden + residual` materialized first (`residual = hidden`), then a standalone RMSNorm, instead of the fused add+norm kernel whose sum stays fp32; all-reduce+norm fusion is vetoed with it | forward values |
| `--router-topk torch` | The correction-bias router runs fp32 `torch.softmax`, `torch.topk(probs + bias, sorted=True)` (PyTorch tie order), weights = unbiased probs x `routed_scaling_factor`, zero experts (`id >= num_real`) become `-1` and keep their weight for the identity residual | forward values (expert selection at near-ties, weights) |
| `--logprob-order megatron` | Selected-token logprobs follow Megatron's vocab-parallel cross-entropy: row max, shift, target gather, `sum_exp` over fixed 32768-wide vocab blocks (the in-block tree is the registered `sampling.block_sumexp` leaf's; the fold across blocks is a rank-ordered fp32 left fold), `logp = -(log(sum_exp) - target)`, for output and prompt (input) logprobs alike | logprobs only |
| `--moe-combine-order slot` | The MoE leaf folds a token's top-k routed outputs in fp32 in slot order across the EP group itself and adds the identity zero-expert residual in the same fold, as the trainer's grouped MLP does (`moe_plan(combine_order="slot")`, `plan["process_group"]` = the EP group); the host hands it the raw top-k (zero-expert ids intact, weights kept), adds no residual and runs no MoE all-reduce / reduce-scatter: `CommManager.post_moe_comm`, the one MoE reduction point every model goes through, reads the switch and only takes back this rank's rows in the RSAG layout; all-reduce+norm fusion is vetoed (`CommManager.should_fuse` re-derives the veto, so it holds wherever the fused kernel is relied upon). Needs MoE TP 1 and no DeepEP all-to-all (both refused by `resolve_numerics`) and a kernel declaring `combine_order` with `slot`. Under `rank` the kernel returns a per-rank partial, the host reduces rank by rank and LongCat's residual enters exactly one partial (`adds_zero_expert_residual`, tp_ep_rank 0) | forward values |

These are the switches the host can mirror with no new vendor dependency.
The rest of the trainer's form lives in kernel leaves and model code the host
does not own and is the out-of-tree model's part of the bargain: a
`sampling.block_sumexp` leaf with the trainer's in-block order, a MoE apply
leaf declaring `combine_order={"rank", "slot"}` whose slot form applies the
routing probabilities inside the activation and combines the zero-expert
residual in fp32 slot order, BF16 index-K indexer scoring and top-k leaves,
and the same LoRA-scale placement in the draft model.

The host's side of the indexer is the plane and the facades. A DSA model's
configure-attention hook names its index-key storage on the model config
(`ModelConfig.index_k_format`, read by `DSAConfig`): the in-tree
`configure_dsa_attention` names `fp8_scaled` — FP8 keys plus per-128 fp32
scales, the in-tree leaves' plane — and a plugin hook that scores the
checkpoint's keys unquantized names `bf16`; a hook that names none is a
construction error. The hook receives the resolved launch
(`ModelProfile.configure_attention(model_config, server_args)`), so a plugin
can name the plane its leaves score under `server_args.numerics` and keep
the FP8 plane under `auto`. The ordinary recipe plans that plane, the
pool writes keys in the plane's own dtype and never converts between the
two, and `dsa_decode_topk` / `dsa_prefill_topk` read `index_k_format` and
`index_k_layout` off the plane's dtype and shape, so a bf16 plane selects
only a leaf declaring `index_k_format={"bf16"}` (the kernel package's DSA
README has the table); the GLM-5.3-Flash recipe plans pooled `fp8_scaled`
rows and refuses any other plane. `candidate_lens_cpu` reaches every top-k
leaf registered with the `candidate_lens_cpu` feature and no other. Index
keys handed to `dsa_prefill_topk` as rows in workspace-row order (the
query-context-parallel history gather over page-sharded caches,
`docs/design/unified_path.md`) keep the plane's format -- `index_k_fp8` +
`index_k_scale`, or `index_k_bf16` -- and reach only leaves declaring the
`index_k_workspace_rows` feature for it (selection requires the feature,
overrides included, and the keywords are routed by it), so a bf16 leaf that
scores gathered rows declares that feature and takes the `index_k_bf16`
keyword; the GPU DSA leaf selects that leaf once at construction under
query context parallelism, so a missing one fails at startup. In-tree
drafts fold no LoRA norm scale; a draft that
does must read `--mla-lora-scale` exactly as the target does, folding only
under `folded`.

Trainer alignment is a stronger claim than invariance and cannot be checked
by the engine alone: a model earns `rl-bitwise` in
`ModelProfile.numerics_envelopes` through the invariance harness *and* the
teacher-forced comparison against a trainer dump described under Acceptance.
In-tree models do not declare it; the out-of-tree LongCat 2.0 plugin is the
first candidate.

## The in-switch reduction

The NVLS `multimem.ld_reduce` sums the group's copies of a row in the
switch (fp32 accumulate, one rounding to bf16) and is the batch-invariant
all-reduce under `--batch-invariant-collectives`. What was measured on
8xH20 (`tokenspeed-kernel/test/ops/test_communcation.py` keeps the
assertions): for a fixed issuing rank the result is bitwise stable across
repetitions with launch jitter and independent of how many rows ride along;
but the association order **depends on which rank issues the load** — two
issuers disagree on a few elements per 10^7 where the fp32 sum is
ill-conditioned, each also disagreeing with the sequential rank-ordered fold
on a few. That rules out the natural reduce-scatter in which each rank
reduces its own slice: a row's issuer would move with the slicing — with the
co-batched tokens under attention TP, with the request's DP placement under
attention DP — and so would its bits.

Hence the shape of the route. The all-reduce pins the issuer
(`TritonRSAGBackend.multimem_all_reduce`: group rank 0 reduces every row
and multicasts the sum back; the others only join the kernels' barriers), so
the sum is one function of its inputs for the deployment's lifetime; the
issuer's port carries the payload twice, still far under the fold's
`world_size` x all-gather, at every size. A pinned-issuer reduce-scatter is
the same two kernels plus a slice and gains nothing over the fold, whose
all-to-all moves each byte once, so the reduce-scatters keep the fold.

Nothing in software pins the switch's order, so a deployment verifies it
before serving (`comm_backend/self_check.py`, run by the distributed
initializer on the attention TP, dense TP and MoE TP-EP groups the route
sends to the switch) with a payload built to be ill-conditioned on most
elements (`multimem_probe_payload`: per element one rank holds `+B`, another
`-B`, the rest values whose low bits fall below fp32's resolution at `B`),
so that two orders disagree on about half of it rather than on a few per ten
million. Three legs: the payload reduced eight times must come back bitwise
identical and its first half must reproduce the first half's rows — a
difference is a fault and refuses startup, naming the kind, the group and
what differed, and `--force-deterministic-rsag` as the way to keep every
reduction on the fold. The third leg asks whether the kind is *one
function*: every group of a kind reduces the identical payload, and every
rank must take the same route for the kind and every group must return the
same bits, or a request's bits would depend on the replica serving it. The
switch's order is a property of the GPU set — on 8xH20 the groups `{0..3}`
and `{4..7}` reduce the probe to different bits on half of its elements,
while groups of two cannot differ (two addends have one sum) — so this is a
topology, not a fault: the self-check pins the kind's groups to the ordered
fold (`AutoBackend.pin_ordered_fold`, the same decision on every rank) and
logs it, and the route honours the pin. A deployment whose attention TP
groups are several sets of three or more GPUs therefore keeps its attention
all-reduce on the fold and its single MoE group on the switch; one TP group
of everything, or TP-2 replicas, run the switch throughout.

## Parallel layouts and the envelope

The decode-side TP layouts under attention DP (`--attn-head-tp-size`,
`--lm-head-tp-size`, `--dense-tp-size`, `--tp-batch-invariant`; see
[Parallelism](../serving/parallelism.md#decode-side-tp-layouts-under-attention-dp))
are explicit deployment choices. The envelope does not fold them in: a
layout is part of the deployment the contract is stated for, not a switch
the contract tightens, and `rl-bitwise` neither selects nor refuses one.
Within one deployment they keep the contract as follows.

- Run and batch invariance hold under every layout. The head, dense and
  LM-head GEMMs take the same `aok` pins as their replicated forms; the
  exchanges (token all-gather, all-gather of a reduction dimension, the
  even and uneven all-to-all transposes) are permutations of bytes and need
  no fold; the remaining reductions (`o_proj` and `down_proj` reduce-scatter
  without `--tp-batch-invariant`) go through `batch_invariant_collectives`.
- `--tp-batch-invariant` is the stronger property: with it the decode layer
  has **no cross-rank reduction outside MoE**. `o_proj` and `down_proj`
  become column-parallel on hidden fed by an all-gather of the reduction
  dimension, so every output element is one full-K GEMM result — bitwise the
  value a TP1 or replicated layer computes, independent of the width `W`.
  An ordered-fold reduce-scatter is batch-invariant too, but its fp32 fold
  of `W` partials is not the full-K GEMM's association, so a decode engine
  on that path disagrees in the last bits with a prefill engine whose
  `o_proj` / `down_proj` run replicated or at TP1. `resolve_numerics`
  therefore warns when head TP runs under `rl-bitwise` without
  `--tp-batch-invariant attn`.
- Topology invariance across layouts is not promised beyond that: the
  vocab-sharded log-softmax and the trainer-aligned fold remain the deferred
  layers above, and a deployment advertising `rl-bitwise` with one of these
  layouts must pass the invariance harness with that layout.

## Layout invariance of query context parallelism

Every collective query context parallelism adds (`docs/design/unified_path.md`)
is data movement: row slicing, the all-gather of rotated latent rows, index-K
rows, gathered history rows, sampled rows and the planned prompt-logprob rows'
activations. The per-row kernels — sparse attention over the gathered history
with every head and no LSE merge, the indexer's top-k over pre-gathered rows,
RoPE, the GEMMs — see for each row exactly the operands a single GPU would,
so a row's bits do not depend on which rank computes it or on the batch it
shares: the layout preserves run and batch invariance by construction. The
prompt logprobs in particular are the tensor-parallel path's bit for bit:
once the planned rows are gathered, every rank runs the same chunk loop over
the same rows against the same vocab-sharded head
(`test/runtime/distributed/test_qcp_prompt_logprobs.py` asserts
`torch.equal` against that path). Head TP over the query shards
(`--attn-head-tp-size` equal to the shard group) adds the head exchanges —
all-to-all transposes, permutations of bytes — and the `o_proj` tail. What
can differ from the TP8 prefill baseline is therefore the output projection
alone: its form, and for the row-parallel form the order in which the
per-rank head partials are summed.

Head TP with the row-parallel `o_proj` computes the same per-rank partials
TP8 does, but reduce-scatters them to the shard rows where TP8 all-reduces,
and under rl-bitwise (`--batch-invariant-collectives`) the two collectives
do not take the same route (`comm_backend/auto.py: route`): a
reduce-scatter always takes the ordered fold (ranks 0..W-1 left to right in
fp32, one rounding), while a 2-D bf16 all-reduce on a multicast-reachable
group takes the NVLS in-switch reduction through a fixed issuer, whose
association order is a property of the GPU set and has been measured to
differ between sets (`comm_backend/self_check.py`). So **QCP with head TP
is bitwise the TP8 engine only when both engines sum `o_proj` in the same
order** — both on the fold — the TP8 engine launched with
`--force-deterministic-rsag` (or pinned there by its self-check); that is
also what makes the comparison hold across machines, which the in-switch
order does not promise. Within the QCP engine itself the two forward forms
reduce differently too: the sharded extend reduce-scatters (the fold), the
drafter's replicated decode steps all-reduce (the in-switch route where it
applies); `--force-deterministic-rsag` on the engine pins both to the fold.
Each form is run- and batch-invariant on its own either way. (Making the
extend's tail all-reduce and slice, so one engine sums one way and matches
TP8 on the same GPU set without the flag, was considered and left out: it
moves W× the reduce-scatter's bytes on every layer of a prefill engine
whose point is the extend, and buys nothing across GPU sets.) Known gap: the
GPU validation of this layout against a plain TP4 prefill engine matched
tokens and logprobs bitwise on prompts within `index_topk`, while one
prompt whose context exceeded it — the indexer's top-k selecting a strict
subset of the history — kept the tokens but diverged in logprobs from
position 0 (max |Δ| 3.96e-2); unresolved, so the statement above is
validated within `index_topk` only.

The head-replicated default (one GEMM over every head, the TP1 / trainer
form) and head TP with `--tp-batch-invariant attn` (full-K column-parallel
GEMM, a transpose back) have no cross-rank sum in `o_proj` at all and
reproduce the TP1 / decode-side batch-invariant form — the one the RL
trainer alignment wants, and the same gap to TP8 as between the decode
side's batch-invariant layout and a cross-rank sum. The drafter's decode
steps on a sharded engine merge partials across the KVP page owners (the
page-sharded KV of `--kv-parallel-size`;
`combine_attention_partials`, every head under the head-replicated layout,
the attention-TP slice under head TP), with the ordered fold under
rl-bitwise.

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

A returned logprob is the launch's `--logprob-order` applied to the row --
`gather_token_logprobs` in `sampling/utils.py`, the one function both
consumers call: `log_softmax(logits, -1, dtype=float32)` gathered at the token
under `torch`, Megatron's vocab-parallel cross-entropy order under `megatron`
(alignment.trainer above) -- with the logits produced by the same LM-head
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

## Retraction is a byte copy: the bitwise continuation oracle

A capacity retraction suspends a request with its image -- every cache
group's data blocks, the unaligned tail page included, and the slot-state
blob: the next step's inputs and draft candidates, the verifier's recorded
draft distributions `q`, the drafter's stash, the sampling backend's
per-request state (`docs/design/cache-concepts.md`, "Retraction image") --
and the restore copies those bytes back into fresh Device pages and a fresh
request-pool slot. Nothing is recomputed and no forward runs in between for
that request; its continuation reads only the restored bytes, its tokens and
its `(seed, position)`-keyed sampler. Under rl-bitwise, where the kernels are
batch-invariant, a forced retraction followed by a restore therefore yields
exactly the tokens and output logprobs of an unretracted run, whatever batch
the restored request lands in; without batch invariance the restored run
differs only by the batch-dependent rounding any other batch composition
would produce. The only request-keyed Device rows not in the image are
token-derived (the committed-token history, the n-gram tail) and are reseeded
from the control plane's token list, as on any slot handoff.

This is the test oracle for the suspend/resume path on both retracting
roles: run a fixed prompt set under rl-bitwise twice, once plainly and once
with `--debug-force-retraction-interval N` so every request is retracted at
least once (several times for the longest), and assert `output_ids` and
`output_token_logprobs` equal element-wise. The set includes a
prompt-logprob request, an MTP configuration with
`--enable-speculative-sampling` at temperature 1 (so the stash and the `q`
path are exercised), a `Prefilling` victim through a negative interval, and
on PD the decode node forced. An L3 prefetch that lands short is outside
this oracle and needs none: it happens before the request is admitted, and
the pages it did not land are computed by the ordinary prefill.

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
The envelope folds that combine in (alignment.trainer above), so the
combination is accepted; `--enable-eplb` with the rank-order combine stays an
`auto` launch. The load counters, the CPU-side algorithm and the P2P copies
never enter the arithmetic; the dispatch algorithm stays static (required
under rl-bitwise already) and drafts stay trivially placed.

## Acceptance

An envelope is verified end to end, not per switch, and `rl-bitwise` has two
harnesses, both run against a deployment launched with `--numerics
rl-bitwise`. The **invariance harness** generates with returned logprobs for
the same prompts (a) alone at bs=1, (b) packed with random co-batches, (c)
across repeated runs, and asserts `torch.equal` on token ids and logprobs —
base model and speculative decoding each, greedy and at `temperature=1.0`
with fixed seeds (the T>0 case is what `sampling_stream=per-request` exists
for). A deployment that passes may advertise the contract; one that fails it
has a bug, not a tolerance.

Trainer alignment is earned against a second program, so its harness has a
reference the engine does not produce: a **trainer dump**. For a fixed prompt
set (the invariance prompts plus longer, chat-formatted ones), the trainer
runs its forward over each `prompt + response` sequence and records, per
sequence:

```
ids          int64[T]       prompt followed by response token ids
logprob      float32[T-1]   log p(ids[t+1] | ids[:t+1]) at every position,
                            from the trainer's own log-softmax
layout       str            the trainer's TP/PP/EP factorization and dtype
commit       str            trainer commit that produced the dump
```

(an optional `logits[:8, :]` of the first positions helps bisect a
mismatch to a layer). The engine then teacher-forces the same `ids` under
`--numerics rl-bitwise` with `return_logprob, logprob_start_len=0` and
asserts `torch.equal` on the fp32 `input_token_logprobs` vector, reporting
the first divergent position otherwise. Two self-consistency checks ride
along: the response part of the teacher-forced `input_token_logprobs` must
equal the `output_token_logprobs` the engine reported while generating that
response, and the result must hold alone and packed exactly as for
rl-bitwise. Agreement up to a tolerance is a finding to bisect, not a pass.
The portable `block_sumexp` leaf and the vendor leaf reproducing the
trainer's in-block order differ at ulp level, so the vendor leaf is what
the comparison runs with.

The leaf contract behind `--logprob-order megatron` is
`tokenspeed_kernel.ops.sampling.vocab_parallel_logprobs`'s: a leaf
registers `("sampling", "block_sumexp")` with the signature
`format_signatures("shifted", "dense", {torch.float32})` and the
`batch_invariant` feature (the op requires it), is called as
`leaf(shifted, *, block_size)` with the fp32 `[rows, vocab]` logits already
shifted by the row max, and returns fp32 `[rows, vocab // block_size]` block
partials in vocabulary order, each block reduced in its own fixed tree. The
op folds the columns left to right in fp32 — Megatron's rank-ordered
reduction of the per-shard partials; a shard narrower than a block pairs
back into the block's tree at its top level, so one full-vocabulary call
equals the trainer's sharded sum bitwise when the in-block tree is pairwise
over its tiles. The in-tree `torch_block_sumexp` sits at `Priority.PORTABLE`;
the vendor leaf registers above it (any higher band), so plain selection
takes it wherever it is installed and no host code names the vendor.

The pins above cover only the paths a verified model takes, so the
verification is recorded per model and enforced at startup
(`require_verified_numerics`): a model profile lists the envelopes its model
passes in `ModelProfile.numerics_envelopes`, and launching an envelope other
than `auto` refuses any target or draft model that does not list it — every
in-tree model included, since none has a profile. The declaration is the
model's promise and the harnesses are what keep it honest: a model declares
`rl-bitwise` when it is ready to run them, and a declared model that fails
either has a bug to fix, not a flag to set. Two incompatibilities are refused
regardless of the declaration: quantized checkpoints (no batch-invariant
quantized GEMM leaf exists, so their linears would select shape-dependent
ones) and a vocabulary that is not a multiple of `MEGATRON_VOCAB_BLOCK`
(the envelope's logprob order folds fixed blocks of it; MoE TP other than 1
is refused by `resolve_numerics` for the same reason on the MoE side).
