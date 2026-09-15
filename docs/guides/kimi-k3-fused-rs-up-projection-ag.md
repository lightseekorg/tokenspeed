# Fused shared-RS + up-projection + AG

This PR includes experimental kernel code, bindings, a **default-off serving
adapter**, tests and historical qualification evidence. Complete fusion uses
`TOKENSPEED_K3_FUSED_RS_UP_AG=1`; the separate
[BT/HT switch](./kimi-k3-mnnvl-cutedsl.md) remains the first-stage-only control.
The final integrated source still needs GPU and full-model acceptance; no
large-M serving TTFT claim follows from the historical tail results below.

![Materialized shared RS versus fused shared-RS/up-projection/AG](/images/k3-fused-rs-up-ag-path.png)

## What changes

Four independent 128-bit multimem reductions feed shared-memory addend tiles
inside persistent up-projection. Shared/residual slices keep the same two
BF16 roundings as the materialized-RS control. TMA multicast targets the
final symmetric output. Independent RS and global shard reads/writes disappear
from execution; raw publication and final completion remain. The legacy shard
allocation is retained as metadata. Real shared down-projection may write
directly to symmetric input, eliminating staging without rewriting that GEMM.

The [design contract](../design/kimi-k3-fused-rs-up-projection-ag.md) documents
shapes, arithmetic, synchronization and workspace ownership.

## Serving integration

The opt-in route consumes the deferred routed triple with HT finalize/norm,
then launches fused shared-RS/up-projection/residual/AG. The existing shared
down-projection writes directly into the symmetric raw input. Each model layer
owns its final output, and sequential layers/buckets share raw storage.
All workspace allocation precedes KV-cache sizing; warmup compiles the
qualified M4096/M8192 plans before capture.

Use TP8/EP1, prefill graphs, and a matching large bucket, for example
`--chunked-prefill-size 8192 --prefill-graph-max-tokens 8192`. Decode, eager
prefill, other M, unsupported layouts and the default environment retain their
existing dispatch. Capture supplies live operand descriptors; replay does not
allocate or run host collectives.

The two-layer, changed-pointer graph regression runs under an eight-rank
`torchrun` with
`tokenspeed-kernel/test/nvidia/thirdparty/test_fused_rs_up_projection_serving.py
--output /tmp/fused-serving.json`. It is not a model quality evaluation.
For TTFT acceptance compare the same immutable current-main commit and
dependency environment, use the K3 agentic workload, retain all raw requests,
separate first/later turns, and run route instrumentation separately from
order-balanced performance servers.

## Performance: identify the denominator first

**Baseline:** materialized shared RS (64 CTA, 1024 threads, four requests,
row layout), wide-epilogue up/AG and HT with seven stages.
**Candidate:** fused shared-RS/up/AG and HT with ten stages, with an explicit
38-cluster cap at M4096 or hardware capacity at M8192.

These comparisons are **not against unmodified main or original AllReduce**.
They include the HT-stage change: this is not an isolated fusion-only result.
Do not multiply these ratios by first-stage results from different runs, or
use the older 654-µs figure as a denominator.

![Historical per-boundary p50 reductions, two independent batches](/images/k3-fused-rs-up-ag-performance.png)

- Primary: pre-generated shared partial staging through completed output,
  including routed HT; expert GEMMs and shared down-proj are excluded.
- Producer: real BF16 shared down-proj through completed output. Baseline
  uses temporary output plus copy; candidate below uses direct symmetric
  output. Activation [M,768] and weight [7168,768] vary across four slots.
- Eight GB300 GPUs, TP8, exclusive use of two four-GPU segments; four layers
  per graph, eight warmups, 20 replays/round, 31 alternating paired rounds.
  Samples are rank-MAX before statistics. Two independent batches remain
  separate, with all slow rounds retained.
- Performance graphs are uninstrumented. Paired bootstrap uses geometric
  mean speedup, 4000 resamples, seed 19073 and nearest-rank 95% intervals.
  This is not a confidence interval of the ratio of p50s.

All latency columns are µs.

| Boundary | M | Batch | Baseline p50 / p90 | Candidate p50 / p90 | p50 latency reduction | Paired speedup CI95 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Primary | 4096 | 1 | 338.278 / 339.300 | 281.425 / 281.770 | 16.8066% | 1.201632–1.203524 |
| Primary | 4096 | 2 | 338.198 / 338.359 | 281.319 / 281.454 | 16.8183% | 1.202077–1.202586 |
| Primary | 8192 | 1 | 615.398 / 615.937 | 514.124 / 515.746 | 16.4566% | 1.195838–1.197119 |
| Primary | 8192 | 2 | 615.523 / 615.877 | 514.248 / 515.370 | 16.4535% | 1.195828–1.197059 |
| Producer → direct | 4096 | 1 | 360.690 / 360.925 | 287.442 / 287.582 | 20.3077% | 1.254571–1.255167 |
| Producer → direct | 4096 | 2 | 360.740 / 360.986 | 287.415 / 287.538 | 20.3263% | 1.254127–1.255316 |
| Producer → direct | 8192 | 1 | 663.689 / 663.841 | 526.622 / 526.964 | 20.6524% | 1.259706–1.260256 |
| Producer → direct | 8192 | 2 | 663.726 / 663.831 | 526.704 / 526.887 | 20.6444% | 1.259934–1.260240 |

All eight historical points pass ≥15% p50 reduction, no p90 regression and
CI lower bound >1. Separately, copied → direct with the **same optimized
tail** reduces p50 by 5.3259% / 5.2877% at M4096 and 6.2980% / 6.3191% at
M8192. All direct increments pass 1.01×, p90 and CI gates. This includes real
down-proj execution, not subtraction of copy time.

Historical trace inventories retain the real producer and show no staging
or hidden materializer in direct mode. Trace timing is diagnostic only.
A separate M4096 profile batch was slower in both arms even before profiling;
its spans must not replace the formal measurements above.

## Evidence and provenance

Exports under `docs/public/data/fused-rs-up-ag/` contain:

| Files | Contents |
| --- | --- |
| `primary-m{4096,8192}-batch{1,2}.json` | All 31 paired samples, p50/p90, CI, config, continuous flags |
| `producer-m{4096,8192}-batch{1,2}.json` | All three pairings, real producer geometry, trace inventories and controls |
| `continuous-m{4096,8192}.json` | Independent four-slot changed-input/ABA continuous qualification |
| `numerical-{cap38,hardware-capacity}.json` | Ten-shape numerical certificates, markers/guards, continuous flags |
| `source-provenance.json` | Frozen archive/source digests, formatting-aware source parity |

Flags are losslessly run-length encoded in original row-major order.
Exports omit machine names, scratch paths and job identifiers; record/trace
digests match retained originals. Full profiler traces and detailed logs
remain in the local archive, not the public patch. Separate RN diagnostics
retain counts and digests, not their full per-check payload.

Formal runs have 196608 continuous flags, zero failures; independent whole-tail
certificates add 24576. Each of two numerical policies has 9600 passing checks
(6640 bitwise, 1760 guards, 1200 independent-reference checks) plus 20480 passing
continuous flags. Each also has 1200 separate RN diagnostics, not relabelled
as bitwise successes. Underlying kernel boundary tests cover M256, 257, 1023,
1024, 1025, 4095, 4096, 4097, 8191, 8192; these do not expand facade dispatch.

Maximum independent-reference absolute error is 0.015625; relative L2 is
0.005442138743073782. Original limits remain
`max_abs <= 0.046875 + 0.01 * reference_abs_max` and `relative_L2 <= 0.006`.
Coverage includes zero/rank/row/owner markers, cancellation, output guards,
rounding ties, changed-input A/B/A, rank skew and independent workspaces.

Reproducibility digests:

- Frozen source archive: `1e712d65ef43fa5319cf826eeb79b5b7717055ee6c703c43fbbdd17abb641460`.
- Historical container: `f95356401d7c35d0238236c5bd271272fc322835e75ea2af77b992be59056aac`.
- Original full qualification result: `3ee18ff0980bc3eb800d5671f18c1a316fa703f19b7008c4ff1fb0c81647d417`.

## Reproduce on the final PR head

The new checkpoint-free harness preserves comparison shapes, configurations
and timing boundaries. It includes producer copied/direct arms, materialized
control bitwise checks, eager/graph checks, original independent reference bounds
and 128 residual-changed replay generations. Actual fused CUBIN resource and
occupancy admission occurs before first producer execution; unknown compiled
layouts are rejected.

This compact harness does **not** replace the full historical marker, skew,
boundary-shape and producer-output certificate suite. Its entry point is new
and has not run on the CPU-only packaging host. Pre-merge TP8 validation must
cover the final code, including those remaining numerical cases.

From the repository root, in the matching GPU environment:

```bash
# Bounded smoke; a timeout is a failure, never a benchmark sample.
timeout 900s torchrun --standalone --nproc-per-node=8 \
  tokenspeed-kernel/test/nvidia/thirdparty/bench_fused_rs_up_projection_ag.py \
  --tokens 4096 --scope primary --rounds 3 --output /tmp/fused-rs-smoke.json

# Repeat for both token counts, both scopes, and two independent batches.
timeout 1800s torchrun --standalone --nproc-per-node=8 \
  tokenspeed-kernel/test/nvidia/thirdparty/bench_fused_rs_up_projection_ag.py \
  --tokens 8192 --scope producer --rounds 31 --output /tmp/fused-rs-producer-m8192-batch1.json
```

Use distinct result files and the repository GPU environment with CUTLASS DSL
and MNNVL-capable TP8 topology; the CPU release venv cannot run these commands.
Record exact source/container/dependency identities for fresh results.
The public patch does not include a private container image.

Device-independent release checks:

```bash
python -m pytest -q test/runtime/test_k3_mnnvl_public_contract.py \
  test/runtime/test_fused_rs_up_projection_public_contract.py
pre-commit run --all-files
```

These check source/configuration/evidence, not device imports, compiled GPU
execution or serving performance. Adding this code changes no runtime default.
