# Medium-M fused tail experiment

The separate [integrated acceptance profile](kimi-k3-integrated-fused-tail.md)
adds continuous BT/HT routing with both halves enabled. The historical
medium-only three-arm policy below remains a reproduction control, not the
integrated deployment policy. Its cap2048 is historical configuration, not a
requirement imposed by the original agentic benchmark.

## Status and scope

This is default-off integration preparation, not a performance or numerical
qualification. The following bucket-specific profile is selected for formal
testing from local screens; local formal stability and serving receipts remain
pending. `MEDIUM_FUSED_RS_SERVING_QUALIFIED` remains false.

| Padded M | MMA tile | CTA group | Cluster |
| --- | --- | --- | --- |
| 256 | 128x64 | Two CTA | 2x1 |
| 384, 512 | 64x64 | One CTA | 1x1 |
| 768, 832, 896, 960, 1024 | 64x128 | One CTA | 1x1 |

All entries retain static-persistent scheduling, automatic A/B and C stages
(`ab_stages=0`, `c_stages=0`), one addend stage, four independent reductions and
the queried cluster cap (`cluster_cap=None`). This is a provisional selection,
not a dispatch qualification or a claim that one geometry wins at every M.
A smoke result does not qualify these buckets; freeze an accepted profile only
after the prescribed independent formal and serving tests.

The experiment shares the existing fused kernel's arithmetic, publication and
completion contract. It changes no scheduler, attention metadata, EAGLE3
verification, KVStore setting or prefill graph ladder. The runtime owns routing;
all kernel implementation and binding remain behind tokenspeed-kernel.

## Explicit three-arm policy

The new switches are `TOKENSPEED_K3_MNNVL_BT_ONLY` and
`TOKENSPEED_K3_MEDIUM_FUSED_RS_UP_AG`. Both default off and only literal `1`
enables them. Medium fusion requires BT-only. Either new switch conflicts with
the legacy `TOKENSPEED_K3_MNNVL_CUTEDSL` or `TOKENSPEED_K3_FUSED_RS_UP_AG` switch;
rank agreement and conflicts are checked before collective allocation.

| Arm | BT-only | Medium fusion | Non-medium buckets |
| --- | --- | --- | --- |
| Unchanged main | 0 | 0 | Original main |
| BT-only control | 1 | 0 | Original main |
| BT plus medium fused | 1 | 1 | Original main |

Both experimental arms use the exact same provisional BT bucket set and tuning.
Neither enables HT. The medium tier only upgrades an already eligible BT
prefill-graph tier: eager, decode/spec-verify and unlisted buckets retain their
existing routes. Legacy switches keep their previous bucket policies.

The shared producer writes its existing `out=` destination directly into raw
symmetric storage. The same BT finalize/AR1/RMSNorm then feeds either the old
owner residual/addmm/AR2/clone control or fused shared-RS/up/residual/AG.
The second-half comparison must not change BT tuning or expert tactics.

## Allocation, pointers and graphs

Create one raw workspace per sequential model and one final symmetric output per
layer, before KV-cache sizing. Capacity is the largest candidate bucket at or
below the configured graph maximum, at most 1024; disabled medium fusion allocates
neither. At 92 layers, 1024x7168 BF16 outputs require 1288 MiB, plus raw input,
metadata shard, signals and allocator overhead. Record actual KV capacity.

`MediumFusedRsUpProjectionServing` inherits the existing live-pointer launcher,
validation, weight identity guard and both barriers. Only profile selection,
output allocation and warmup binding differ. Capture supplies fresh latent and
residual descriptors with DLPack stream=-1; no warmup activation pointer is
retained. Fixed descriptors cover weights, final output and metadata only.
Compilation, rank agreement and CUBIN admission finish before capture. Capture
without a prepared bucket fails. Sequential buckets may reuse a layer's output;
concurrent instances require separate workspaces and outputs.

## Acceptance checklist

- Freeze independent local stability results before enabling serving trials.
- Validate every dispatched padded bucket, including 832/896/960 where used;
  800 real tokens pad to 832 under the original ladder, not to 1024.
- Test live-pointer A/B/A graphs, changed inputs, 128 generations, multiple
  layers/buckets, rank skew, output sentinels and real producer-direct execution.
- Preserve basic semantic correctness: all ranks reduced, correct owner slices,
  residual exactly once, finite outputs and an independent reference. Rounding
  order alone is not a failure; missing ranks or stale/aliased outputs are.
- Preserve the original K3 EAGLE3 agentic client and server configuration:
  prefill graph maximum 2048, chunk budget 8192 and KVStore enabled.
- Restore unchanged-main tactic JSON before sealing serving source. Record its
  hash and each arm's actual tactic/cache identities; BT-only and medium fusion
  must match. No unrelated tactic optimization belongs to this comparison.
- Use unchanged main, BT-only and BT plus medium fusion, with reversed-order
  independent repetitions. Report original all-turn mean TTFT/TPS metrics and
  separate first/later-turn distributions, errors and speculative acceptance.
- Record exact source/container/profile identities and actual KV capacity.
  Do not convert local tail speedup into a serving TTFT claim.
