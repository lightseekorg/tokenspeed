# Fused shared-RS + up-projection + AG

## Ownership and scope

The [integrated acceptance profile](kimi-k3-integrated-fused-tail.md) reuses
this arithmetic and pipeline with continuous BT/HT host configuration. It does
not broaden the historical endpoint qualification described below.

The separate [medium-M experiment](kimi-k3-medium-fused-tail.md) adds a
default-off, unqualified BT-only profile. It does not broaden the large-M
qualification or change the legacy switches described below.

The explicitly bound experimental kernel API also has a default-off serving
adapter, selected by `TOKENSPEED_K3_FUSED_RS_UP_AG=1`. It enables the required
routed finalize/norm path and replaces the second AllReduce at M4096/8192.
The independent `TOKENSPEED_K3_MNNVL_CUTEDSL=1` switch still selects the
first-stage-only ablation when complete fusion is disabled.
Kernel qualification is not serving TTFT qualification: the adapter, model
quality and first-turn agentic benchmark require separate acceptance.

The supported performance endpoints are BF16 M4096 and M8192 on GB300 TP8.
The facade rejects interpolation. Other geometries in the inherited low-level
implementation are diagnostic controls, not qualified dispatch choices.
Existing general RS defaults and the shared down-projection implementation
are unchanged.

## Data and arithmetic contract

| Input/output | Logical shape | Owner |
| --- | --- | --- |
| Routed normalized latent | [M,3584] | Replicated HT output |
| Up weight | [896,3584] | Rank-local output columns |
| Raw shared partial | [M,7168] | Rank-local symmetric workspace |
| Residual | [M,7168] | Bitwise replicated on all eight ranks |
| Final output | [M,7168] | Separate symmetric allocation on every rank |

Shared reduction is a 128-bit multimem load-reduce with FP32 accumulation
and BF16 output. Preserve these two BF16 boundaries for the owned slice:

```text
shared = BF16(sum_ranks(raw_shared))
acc    = FP32(latent @ weight.T)
q      = BF16(acc + FP32(shared_owner))
out    = BF16(FP32(q) + FP32(residual_owner))
```

The residual is included once, after reduction. This is not the old cuBLAS
addmm arithmetic contract, where residual is pre-added to the owner shared
slice before GEMM and the second AllReduce. The independent reference uses
that old semantic path with unchanged error bounds; the materialized-RS
control supplies the strict bitwise comparison.

A legacy [M,896] allocation remains as layout metadata. The fused producer
never reads/writes that global shard: reduction results flow directly to its
shared-memory addend stage. Do not claim the allocation itself was removed.

## Pipeline and launch policy

The producer is based on Blackwell persistent GEMM, not the small-M mailbox
kernel. Keep two-CTA 256×128 MMA, 128×64 epilogue, AB6/C3/ACC2, 192 threads,
one 32-KiB addend stage, and four independent BF16x8 reductions issued before
their results are consumed. Swizzled register-to-shared writes feed epilogue
tiles without an independent global RS materialization.

Release the accumulator after its final read, before output-stage waits.
Keep the three-stage C ring and acquire-before-store policy. Final TMA
multicast writes directly into symmetric output, not a mailbox. No separate
AG/materializer kernel is launched. This permits pipeline progress; it does
not prove MMA never stalls.

Use 38 persistent two-CTA clusters at M4096 and queried hardware capacity
at M8192 (76 observed). Bind before JIT; never mutate the cap after compiling
or hardcode 76 as hardware capacity. The compiled reference uses 158
registers/thread, zero local spill, and 229632 observed shared bytes.
The 230400-byte occupancy-query bound is conservative, not measured usage.

## Synchronization and lifetime

The per-layer-output rules below describe the historical endpoint adapter.
The integrated adapter now uses the separately documented
[two-slot ownership policy](kimi-k3-integrated-fused-tail.md#tuning-and-memory),
with unchanged publication and completion barriers. Historical measurements
must not be relabeled as results for that memory-reuse revision.

Execution is sequential on one explicitly prepared stream:

```text
copy or real producer out= → routed HT → raw publication
  → fused shared-RS/up/AG → cross-rank output completion → next consumer/reuse
```

The materialized control instead runs copy → standalone RS → HT → ordinary
up/AG → output completion. HT uses seven stages in that control and ten in
the fused complete-tail candidate; performance is not an isolated RS ablation.

- Allocation, validation, rank agreement, compilation and cluster-cap
  agreement precede capture. Capture/replay has no host collective.
- Serving allocates raw workspace once per model and separate max-bucket final
  outputs per layer, before KV-cache sizing. Warmup compiles each supported
  bucket. Capture supplies live latent/residual pointer descriptors using
  non-synchronizing `to_cute` views (DLPack stream=-1); it does not bind stale
  warmup pointers or insert device copies. Weights and output layouts stay
  fixed. The original fixed-pointer facade retains its stricter stream contract.
- The execution engine orders warmup, capture and replay streams. Completed
  graphs for different buckets may share this model's raw workspace and each
  layer's output only because their execution is sequential, never concurrent.
  The complete route is confined to breakable prefill graphs, TP8/EP1, DP=CP=1,
  sharded BF16 up weights and a deferred-finalize-capable expert backend.
- Preserve four-warp raw release/acquire publication, system/alias fences,
  and four-warp cross-rank exit. There is no rank barrier inside the GEMM.
  Standalone un-published fused producer launch is rejected.
- Input, output, residual, latent and weight must not overlap protected
  storage. Keep workspace, output handles, plans and tensor owners alive
  for the graph lifetime. Returned results alias persistent output.
- Sequential layers may reuse shared workspace, with separate final outputs.
  Independent/concurrent instances need independent workspaces. Concurrent
  reuse of one workspace is unsupported.
- Direct mode passes the exact contiguous input view to
  `kimi3_shared_down_projection(..., out=..., solution="torch")`.
  Keep the real producer in the timed boundary; deleting a pre-generated
  input copy is not measured producer-direct performance.

## Source and release boundary

The manifest records frozen archive/per-file digests. Repository formatting
may change bytes; CPU checks compare ASTs with only module-level import order
canonicalized. Device bodies, synchronization and existing HT code are retained.

The endpoint facade, serving adapter and compact reproduction harness require
a fresh TP8 numerical/graph/performance run of the final PR source, followed
by full-model checks and uninstrumented current-main comparisons. Historical
measurements do not qualify the new serving integration. Inherited N64/timeline hooks remain transitive
dependencies, but the fixed facade cannot select them.

See the [qualification guide](../guides/kimi-k3-fused-rs-up-projection-ag.md).
