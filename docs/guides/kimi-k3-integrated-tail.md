# Integrated Kimi K3 MoE tail

This experimental, default-off serving path combines BT/HT first-stage fusion
with fused shared ReduceScatter, up-projection, residual addition and AllGather.
It is not yet qualified for stable serving gains or full-model numerical quality.

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

Sequential layers share raw workspace but retain separate output buffers for
graph lifetime. Independent concurrent executions require independent storage.
At92 layers,8192×7168 BF16 outputs alone consume10304MiB per rank. Record measured
KV-cache capacity, not just latency, when comparing serving configurations.

The arithmetic is not bitwise equivalent to main's owner-shard addmm followed
by AllReduce. The current performance campaign deliberately skips independent
smoke and non-tile-multiple GPU validation. Runtime error checks and the original
gold client warmup remain enabled. Successful timing is not a numerical or
task-quality pass.

Use the original K3 EAGLE3 agentic workload with identical configuration for
main and candidate, and reverse run order in a second independent batch.
Report first-turn and all-turn TTFT separately, including per-concurrency
mean/p50/p90, paired confidence intervals, speculative acceptance and KV
capacity. Historical tail ratios cannot substitute for these serving results.

See the [integrated design contract](../design/kimi-k3-integrated-fused-tail.md)
for ownership and dispatch details. The existing
[large-tail study](kimi-k3-fused-rs-up-projection-ag.md) is historical kernel
evidence with a different measurement boundary, not this integration's serving
qualification.
