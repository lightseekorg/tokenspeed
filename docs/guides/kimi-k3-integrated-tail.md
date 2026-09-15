# Integrated Kimi K3 MoE tail

This experimental, default-off serving path combines BT/HT first-stage fusion
with fused shared ReduceScatter, up-projection, residual addition and AllGather.
It is not yet qualified for stable serving gains or full-model numerical quality.

![Unchanged main and the integrated two-stage serving path](/images/k3-integrated-tail.svg)

## Enable the complete path

Set `TOKENSPEED_K3_INTEGRATED_FUSED_TAIL=1` in the server environment and pass
`--prefill-graph-max-tokens 8192`. Do not combine it with the historical
first-stage-only, medium-only or endpoint-only ablation switches.

The path requires the supported K3 TP8/EP1, BF16, sharded up-projection,
deferred expert-finalize and Blackwell MNNVL configuration. Incompatible
capabilities or settings cause initialization to fail; the integrated route
does not silently disable one half of the optimization.

For comparison, launch unchanged main with the same graph maximum8192 and all
optimization switches disabled. Preserve the original benchmark workload and
the other serving options on both arms.

## Continuous token ranges

M is the kernel's token dimension. CUDA Graph execution uses its padded bucket
M; eager execution uses the actual M. A request may be split into several chunks.

| Kernel M | First stage | Fused second stage |
| --- | --- | --- |
| M ≤ 32 | Existing empty/small path | Existing empty/small path |
| 32 < M ≤ 512 | BT finalize/reduce/RMSNorm | One CTA,64×64 tile,1×1 cluster |
| 512 < M ≤ 1024 | BT finalize/reduce/RMSNorm | One CTA,64×128 tile,1×1 cluster |
| 1024 < M ≤ 8192 | HT finalize/reduce/RMSNorm | Two CTAs,256×128 tile,2×1 cluster |
| M > 8192 | Existing separate/default path | Existing separate/default path |

These are interval rules, not a list of sampled token counts. The small and
above-range routes are intentionally preserved. Once initialized, missing
resources within an optimized range raise an error instead of falling back.

## What is fused

Shared down-projection writes directly into symmetric storage. The second-stage
kernel reduces shared slices while computing the local up-projection, combines
shared and residual addends in its epilogue, and multicasts directly into the
final symmetric output. No separate shared-RS output, staging copy or mailbox
AG materializer is inserted on this route.

Grouped reduction requests, early accumulator release and pipelined output
stages permit overlap. Publication/completion synchronization remains; this
does not imply stall-free MMA or cross-stream overlap of the two tail stages.

## Memory and validation

Sequential layers share raw workspace and alternate two persistent symmetric
outputs by global layer-index parity. The previous residual occupies the other
slot; AttnRes snapshots and speculative taps retain independent copies. Each
layer keeps its own weight-dependent launch cache. Independent concurrent
executions require independent storage.

Two 8192×7168 BF16 outputs consume 224 MiB per rank instead of the original
92-layer allocation's 10304 MiB. This saves 10080 MiB of output allocation;
the completed single-pair measurement below reports actual KV recovery.
Results from the original per-layer-output campaign do not validate
the pooled version. The integrated distributed test's `--pooled-outputs` option
checks four chained layers, two graph slots and changed-input replay; its
intermediate snapshots are correctness instrumentation, not serving copies.

First-start graph preparation compiles shape-specific launch bindings and can
be substantially slower than main. Report startup and capture cost separately
from post-readiness TTFT; do not treat those as the same latency metric.

The arithmetic is not bitwise equivalent to main's owner-shard addmm followed
by AllReduce. The current performance campaign deliberately skips independent
smoke and non-tile-multiple GPU validation. Runtime error checks and the original
gold client warmup remain enabled. Successful timing is not a numerical or
task-quality pass.

Use the original K3 EAGLE3 agentic workload with identical configuration for
main and candidate. A second independent reverse-order batch is needed before
claiming replicated results; it was not run for this pooled revision.
Report first-turn and all-turn TTFT separately, speculative acceptance and KV
capacity. Historical tail ratios cannot substitute for these serving results.

## Serving measurement

Tested code: `6b076129bac77ed0856bdab5b5f2354f7da88cd0`, based on unchanged main
`08d11a155d1894a80b9031974377c37fd18f8b83`. Subsequent result/documentation-only
changes do not change the tested runtime. One fresh main → candidate A/B on
GB300 TP8/EP1, FP8 KV, EAGLE3 (three steps, four draft tokens), chunk8192 and
prefill graph maximum8192 on both arms. Max sequences16, max length80000,
memory fraction0.9, KVStore enabled; original warmup and agentic workload.
Both arms completed786/786 formal requests with zero failures.

| CC | First-turn mean / p50 / p90 speedup | Main TPS/GPU | Candidate TPS/GPU | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.0817× / 1.0820× / 1.0800× | 3395.82 | 3412.19 | +0.48% |
| 2 | 1.0730× / 1.0847× / 1.0426× | 5045.42 | 5192.24 | +2.91% |
| 4 | 1.0933× / 1.1259× / 1.0888× | 6959.85 | 7134.66 | +2.51% |
| 8 | — | 8970.11 | 9151.80 | +2.03% |
| 16 | — | 11699.95 | 12072.29 | +3.18% |

Latency speedup is main/candidate. TPS/GPU is the original total-token metric
(input plus output, including cached input) divided by8. Queue-sensitive CC8/16
TTFT is not an acceptance metric. First-turn counts are4/8/8 at CC1/2/4;
this single pair has no confidence intervals or replication claim. Later-turn
mean TTFT at CC4 regressed (0.9710×); not every latency metric improved.

Actual KV allocation is51.00GiB/GPU versus main51.58GiB, with2475 versus2503
parent pages on every rank. The historical per-layer candidate had40.97GiB;
the pool recovers approximately10.03GiB/GPU. The remaining gap is0.58GiB.
Native graph capture and gold requests exercised the pooled serving path, but
independent pooled GPU correctness and full-model quality remain unqualified.

Dispatch also applies to target verification:16 requests with four speculative
tokens can use M64 and the medium fused route. Throughput changes cannot be
attributed solely to prefill or KV capacity. Generated histories and accepted
lengths may differ; pairing checks identical conversation/turn inventories,
request settings and first-turn payloads, not identical later generated replies.

[Numeric statistics and workload controls](/data/k3-integrated-serving.json)
include first/later/all-turn mean/p50/p90, throughput, cache hit rates, aggregate
decoded tokens per iteration and memory. Raw prompts and responses are excluded.

See the [integrated design contract](../design/kimi-k3-integrated-fused-tail.md)
for ownership and dispatch details. The existing
[large-tail study](kimi-k3-fused-rs-up-projection-ag.md) is historical kernel
evidence with a different measurement boundary, not this integration's serving
qualification.
