# Medium-M fused shared-RS + up-projection + AG experiment

This is an unqualified experiment, not a change to the large-M facade or
serving dispatch. The data, rounding, ownership, same-stream lifetime, and
external publication/completion contracts in
[the established design](kimi-k3-fused-rs-up-projection-ag.md) remain unchanged.

The medium specialization inherits the existing device body and vector
reduction hooks. It changes only host configuration: one-CTA 64x64, 64x128,
128x64, 128x128 or two-CTA 128x64, 128x128, 256x64, 256x128 MMA tiles. The cluster is exactly
1x1 or 2x1, respectively. An epilogue uses the full per-CTA M extent and
64 columns. Four epilogue warps, one MMA warp and one TMA warp remain fixed.

The isolated tuning object also exposes the output ring depth, one/two
addend stages, request grouping and persistent cluster cap. The first version
supports static-persistent and an explicit full-grid host policy. A 64x64 epilogue
contains four BF16x8 vectors per loader thread and rejects a group of eight;
128x64 contains eight. Two addend stages currently require group one.

An explicit `ab_stages` control leaves the inherited behavior exactly unchanged
at zero. Nonzero requests must be at least two and accompany explicit C ring
depth, so saved A/B space is not automatically consumed by more C buffers.
The maximum schema depth is 21: the 1024-byte metadata/alignment reserve must
cover `16*AB + 16*ACC + 12 + 5*127`, with ACC fixed at two. The actual geometry's
shared-memory limit is checked from live CuTe layout byte sizes during compile
and is usually lower. Only A/B staged layouts are rebuilt; output/addend layouts,
ACC/TMEM allocation and device pipeline bodies remain inherited.

For BF16 with the existing K64 pipeline tile and one-CTA layouts, the
conservative byte budgets are:

| MMA tile | A/B bytes per stage | C bytes per stage | Two addend planes per stage |
| --- | ---: | ---: | ---: |
| 64x64 | 16384 | 8192 | 16384 |
| 128x64 | 24576 | 16384 | 32768 |

Total budget is `1024 + AB*ab_bytes + C*c_bytes + Add*addend_bytes`.
These formulas explain the experiment; admission uses actual layout sizes,
not hardcoded geometry estimates. Lower shared-memory usage alone does not
prove higher occupancy: registers and TMEM can remain limiting resources.

The full-grid policy retains the exact `StaticPersistentTileScheduler` device
body and parameters, including epilogue pre-advance, iteration counts, ring
phases and producer tails. Only the grid's worker limit changes to the total
number of problem clusters:
`ceil(M / mma_tile_m) * (896 / mma_tile_n)`. The grid remains
`(cluster_m, cluster_n, problem_clusters)`, not an ordinary `(Mtiles,Ntiles,1)`
grid; the inherited scheduler initializes work from `blockIdx.z`.
Each launched cluster therefore executes one output work tile.

Full-grid requires `cluster_cap=None` and may queue more clusters than fit
concurrently. It is valid only because no grid-wide or cross-rank barrier is
inside GEMM; cluster-local synchronization and the external completion barrier
remain unchanged. Exact-CUBIN admission checks that at least one whole cluster
is resident, independently of total queued work. Persistent mode retains the
existing requested-population admission. Evidence reports scheduler, problem
clusters, planned launch grid and separately queried resident capacity; launched
cluster count is never described as occupancy. No CLC implementation is added.

The API accepts M33..8192, including masked non-tile multiples. The lower bound
was extended for the [integrated profile](kimi-k3-integrated-fused-tail.md);
dedicated non-tile GPU validation is not part of that acceptance run. This broad
diagnostic support does not qualify every shape or extend serving routes.
SMEM layout, T2R/R2S partitioning and multicast output correctness must be
tested on each selected geometry; CPU shape validation is not GPU acceptance.

Preparation is collective and precedes capture. It checks shapes, dtype,
physical symmetric mappings, protected aliases, rank-identical tuning and
the minimum hardware capacity across ranks. Before any producer launch it
queries the exact compiled CUBIN with the selected cluster shape, 192 threads
and a conservative compile-time shared-memory bound. Insufficient residency
is rejected on every rank. Registers, local-memory use, ring depths and
PTX/CUBIN hashes are retained as evidence. A nonzero spill report is not itself
a numerical error, but must be considered when comparing performance.

The raw shared reduction still flows directly into addend SMEM. No standalone
RS, mailbox or materializer is added. Final TMA multicast writes the original
896-column owner range into symmetric output. Guarded prefix output views are
allowed when their base pointer still matches the symmetric allocation.
Preparation binds one stream; concurrent use or overwriting a shared workspace
before the inherited completion barrier remains unsupported.

Local complete-tail experiments compare identical BT first stages with old
addmm/AllReduce #2 versus this fused back half. Local gains do not constitute
agentic TTFT gains. Serving comparison against BT-only and unchanged main
requires separate, unchanged-workload acceptance after a stable local winner.
