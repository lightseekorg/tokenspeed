# Integrated K3 fused tail acceptance profile

This default-off profile is **not yet qualified for serving performance**.
`TOKENSPEED_K3_INTEGRATED_FUSED_TAIL=1` enables both first-stage protocols and
the same fused shared-RS/up-projection/residual/AG device pattern. It is mutually
exclusive with the historical first-only, medium-only and endpoint-only flags.
Those old profiles remain unchanged for reproduction.

| Kernel M | First stage | Second stage |
| --- | --- | --- |
| 0..32 | Existing empty/small route | Existing empty/small route |
| 33..1024 | Deferred BT finalize/AR/RMSNorm | Fused RS/up/residual/AG |
| 1025..8192 | Deferred HT finalize/AR/RMSNorm, ten stages | Fused RS/up/residual/AG |
| Above 8192 | Existing separate path | Existing separate path |

The selector uses continuous intervals on actual kernel M; a graph dispatches
on its padded M using the existing graph ladder. Eager and graph forwards use
the same selector and live-pointer launcher, including speculative forwards.
No optimized-range fallback is permitted after integrated initialization.
Capability disagreement fails before collective allocation; missing in-range
plans fail rather than switching to an AllReduce control.

Both comparison arms explicitly set `--prefill-graph-max-tokens 8192`. The
original agentic benchmark does not mandate 2048: it omits the option and
inherits a runtime default. This profile neither changes that default nor the
benchmark client, workload, request counts, KVStore or speculative settings.

## Tuning and memory

The inherited medium specialization changes host geometry only. M33..512 uses
one-CTA 64x64; M513..1024 one-CTA 64x128. These retain
automatic A/B/C stages. M1025..8192 uses two-CTA 256x128, AB6/C3/ACC2; the
persistent cluster cap is 38 through 4096 and queried capacity above 4096.
All use one addend stage, four independent reductions and static-persistent
scheduling. Interpolation is a candidate policy, not a measured optimality claim.

The M256 two-CTA short-screen result was less than one percent faster than
one-CTA 64x64, without a repeated cluster-only comparison. That is insufficient
evidence for a separate low-M two-CTA interval. The integrated profile therefore
uses one CTA throughout M33..1024. This is a simplicity choice among close
screened configurations, not proof of noise or a universal small-batch rule.
Historical endpoint-only profiles and their measured results remain unchanged.

One raw symmetric workspace and two 8192x7168 BF16 symmetric outputs are
shared by sequential layers. Global layer-index parity selects the output;
each layer still owns its weight-dependent launch cache. Both outputs together
require 224 MiB, versus 10304 MiB for the original 92 per-layer outputs.
This is a 10080 MiB allocation reduction, not yet a measured KV-capacity or
serving-throughput improvement. The original two-batch campaign uses the old
per-layer allocation and must not qualify this revised ownership policy.

The previous layer's output can remain the current residual, so one output
would be unsafe. Two outputs keep it distinct from the current destination;
the existing live-input alias checks remain enabled. AttnRes block writes copy
into independent block storage, and EAGLE3/DFLASH taps clone or materialize their
results before later layers overwrite a slot. The existing entry rank barrier
orders all prior consumers before multicast reuse, and the exit barrier makes
the result visible before its next consumers. Auxiliary consumer streams must
join before reuse. No new copy or device barrier is introduced.

Allocate before KV sizing and retain both handles through every graph replay.
All graph buckets use fixed parity addresses and execute sequentially; outputs
are transient until the same slot is next written, not per-layer archives.
Different concurrent model/graph instances require separate storage. The
historical endpoint-only and medium-only profiles retain per-layer outputs.

Arithmetic, early accumulator release, direct TMA multicast, external
publication/completion barriers, alias checks and live capture pointers retain
[the established fused contract](kimi-k3-fused-rs-up-projection-ag.md).
No standalone RS, staging copy or AG materializer is inserted. The real
shared producer writes the exact symmetric `out=` view.

## Validation boundary

This performance campaign runs uninstrumented same-main comparisons directly;
independent adapter/full-model smoke and dedicated non-tile-multiple GPU
validation are deliberately omitted. Do not report those checks as passed.
The original gold warmup and runtime/device-error checks remain enabled.
Actual serving graph startup and replay are exercised by the benchmark, but
uninstrumented timing does not prove a complete per-bucket route inventory.
CPU interval tests verify routing only, not kernel memory safety or accuracy.
Preserve execution failures, source/container/config identities and numeric
results. Report local tail, all-turn gold TTFT and first-turn TTFT separately,
including speculative acceptance and KV-capacity differences. Performance
results alone do not establish numerical equivalence or model task quality.
