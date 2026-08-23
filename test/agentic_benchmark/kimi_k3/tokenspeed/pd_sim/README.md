# PD-disaggregation simulation bench

Ranks the parent directory's serve configs (`../configs/*.sh`, reused
verbatim) by what a PD-disaggregated deployment cares about: prefill-node
throughput (P-sim, split into fresh and cached) and decode-node throughput
(D-sim), measured separately. The agentic bench cannot answer this — its
numbers mix prefill interference into every decode figure.

## Data: the frozen canonical dataset, same as everything else

All three sub-benches replay the frozen agentic artifact
(`lightseekorg/agentic-dataset`, the file the agentic bench and the CI gates
use). Synthetic random-token prompts are deliberately NOT used: speculative
acceptance and MoE expert routing are content-sensitive, and random text
would misrepresent both (the two subsystems this bench most needs to rank).
A `random`-plugin variant with parameterized lengths stays a future
shape-scan, clearly labelled content-unrealistic.

Real-workload anchors (measured): fresh prefill = ~50K tokens, 1 per
conversation (~8% of requests, ~75% of prefill compute); cached prefill =
~0.8-1.3K new tokens on a 50K->67K cached prefix (~92% of requests).

## P-sim: prefill-node simulation

Batch contains prefill only (`max_tokens 1` kills the decode phase and
reproduces a P node's scheduling profile). Sampling needs no pinning at 1
output token. One boot per config, fresh phase then cached phase.

**P-fresh (compute-bound):** send the 71 unique first turns cold.
Ladder: parallel 1/2/4/8/16, number = 2 x parallel (62 unique prompts
consumed, within the 71 budget; no prompt reused).
Ranking: **prefill tok/s / GPU**; secondary TTFT p50/p99.
Validity guard: cache hit <= 5%.

**P-cached (bandwidth-bound, distinct prefixes):** per conversation, prime
turn 1 with `max_tokens 500` (builds a realistic prefix: 50K prompt + 500
generated tokens; excluded from measurement), then measure turn 2 with
`max_tokens 1` — a cached prefill of ~50.5K hit + ~800 new tokens, the real
turn-increment shape. The prime response's reasoning_content AND content are
passed back verbatim in the replayed assistant turn: K3's chat template
renders them into the think/response channels, so the re-rendered prefix
retokenizes onto the cached token stream (content-only passthrough would
diverge at the <think> tag and silently recompute the ~500 assistant
tokens). Residual retokenization boundary effects are absorbed by the
computed-tokens metric, which counts whatever actually recomputed. Distinct per-conversation prefixes stress cache
capacity and (under DP) rank affinity — deliberately harder than a shared
prefix.
Ranking: **computed tok/s / GPU = (prompt_tokens - cached_tokens) / time**
(a raw prompt-tok/s column would be inflated by cache-hit accounting);
secondary requests/s and TTFT p50/p99.
Validity guard: cache hit >= 95%.

## D-sim: decode-node simulation

A D node's defining state: the KV already exists (computed by P, delivered
by transfer). The prefix cache plays the transfer's role.

1. **Prime at LOW concurrency** (parallel 2, `max_tokens 1`, first turns):
   the prime wave only needs to park KV in the cache — its concurrency is
   independent of the measured concurrency, so the prefill workspace peak
   stays minimal and cannot disturb razor-thin memory envelopes.
2. **Settle 30s**: async writebacks and allocator churn quiesce.
3. **Measure**: resend the SAME first turns, `max_tokens 2000` +
   `ignore_eos`, ladder parallel 1/2/4/8/16 with number = 2 x parallel
   (rolling admission keeps the decode batch saturated with arrivals and
   departures — a D node's steady state, not a lockstep wave). The
   per-request prefill collapses to a ~full cache hit (the KV-load cost
   stands in for transfer/load; real transfer cost is out of scope).
4. Ranking: **Output Throughput (tok/s) / GPU** (decode-only regime, so
   output-only is the honest number); secondary TPOT p50/p99.

Validity guards, both recorded per rung in the collect output:
- **cache hit >= 95%** on the measure wave — below it the rung is VOID
  (primed KV evicted: the config cannot hold C x 50K as a D node, itself a
  reportable capacity result; under DP it may also mean affinity routing
  failed to return a request to the rank holding its KV — also reportable).
- **memory ledger**: max-across-GPUs memory sampled before prime, after
  prime, and at 5s cadence DURING each measure rung (background sampler,
  peak recorded). collect flags a rung whose measure peak exceeds the
  post-prime level by more than 4GiB (`mem-climb`), and VOIDs rungs with
  missing samples. d_bench boots fresh, so the before-prime baseline is
  clean (no P-phase residue — the standalone split guarantees it).

Memory-isolation rationale: prefill-phase workspace/activations are
transient (returned to the torch caching allocator and reused by decode's
far smaller allocations); the KV pool is sized once at boot from the
profiling pass and cannot be squeezed by runtime allocations. The ledger
verifies this empirically instead of trusting the argument.

## Knobs

- Input anchor 50K (frozen-file first turns); D output 2000 (decode >= 95%
  of measured wall clock at observed TPOT).
- Concurrency ladder 1/2/4/8/16 (32 needs a max-num-seqs bump — future).
- Sampling: NOT pinned, matching the agentic bench convention (fixed input,
  default sampling). Consequence, same as the parent README: speculative
  acceptance drifts between runs, so D-sim differences below the measured
  noise band need repeated runs before ranking two configs. P-sim is
  unaffected (max_tokens 1 has no generation to sample).
- P:D sizing helper in collect: give it the per-conversation token mix
  (`--mix-fresh-tokens/--mix-cached-tokens/--mix-decode-tokens`, defaults =
  the agentic anchors) and it converts the measured rates into GPU-seconds
  per conversation on each side, printing the provisioning ratio:
  `P gpu-s/conv / D gpu-s/conv = P-GPUs per D-GPU`.

## Implementation notes

- evalscope's swe_smith plugin has no prime/measure phase control, so ALL
  phases run through pd_client.py (thin stdlib client, rolling admission,
  one retry per request; a twice-failed request is recorded and the rung is
  VOIDed by collect instead of aborting the sweep). pd_sim has its OWN
  collect script — its summaries are deliberately NOT column-compatible
  with the parent bench's collect.
- Every summary and collect report carries the boundaries statement: no
  KV-transfer cost modeled; prime-as-transfer is the core approximation;
  single machine; TTFT is approximated by full-request latency at
  max_tokens 1; TPOT (d-measure only, p50/p99) amortizes the cache-hit KV
  load into per-token time.
- Each config boot starts with a warmup on spare conversations 62-63
  (excluded from every metric) so the first measured rung does not absorb
  first-touch JIT/autotune costs.
- The dataset is always fetched into pd_sim/ itself (frozen artifact); a
  parent-directory agentic_dataset.json is deliberately NOT reused — it may
  be a local non-frozen build.

## Layout

```
p_bench.sh           # standalone P sweep: boot -> warmup -> P-fresh ladder -> P-cached ladder -> kill
d_bench.sh           # standalone D sweep: boot -> warmup -> prime -> settle -> measure ladder (+ memory sampler) -> kill
pd_client.py         # phased client shared by both benches
collect_outputs.py   # tables + guards + P:D sizing; accepts multiple sweep dirs:
                     #   python3 collect_outputs.py outputs/p_<ts> outputs/d_<ts>
outputs/p_<ts>/, outputs/d_<ts>/   # per-sweep artifacts (gitignored)
```

The two benches are independently runnable — a P sweep and a D sweep each
own the machine for their duration and can be run on different days; the
collect script merges any combination of sweep dirs.
