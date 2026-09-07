# CuTe QSA sparse decode

The SM100 specialization consumes BF16 queries `[rows, 6, 256]`, BF16 or
FP8 E4M3 KV `[cache_slots, 1, 256]`, and int32 physical slots `[rows, 2051]`.
This is the local attention geometry for tensor parallelism 4. NVFP4 model
weights do not change the KV storage format. Non-positive slots are masked;
duplicates retain their individual contributions. Output is BF16, with
FP32 accumulation and scalar K/V descales.

## Execution contract

The first 2048 selected entries form sixteen 128-token tensor-core tiles.
The remaining three entries use FP32 dot products inside the same kernel's
final softmax combine. Computing this small remainder directly avoids a
mostly empty seventeenth tile that otherwise extends one CTA's critical path.
The combine rescales both parts to their shared maximum before normalization;
empty full-tile splits, empty tails and entirely empty rows remain valid.

Every supported split count divides sixteen. Expressing the iteration count
as a compile-time quotient removes unreachable prefetch and online-update
branches, particularly when each CTA processes one tile. The existing
two-stage asynchronous K/V ring handles one or multiple tiles; for one tile,
its second item is V0. BF16 V conversion uses transposed `ldmatrix` loads.
When the ring is reused, the conversion warpgroup also synchronizes after
reading each K/V tile, before any warp can issue a refill over a peer's reads.
Single-tile launches need no such refill barrier.

Small launches use sixteen CTAs per cluster when the device occupancy probe
can accommodate all query rows in one wave. Unsupported or smaller
partitions retain portable eight-CTA clusters; launches above eight rows
use four CTAs. The probe is cached per device before kernel compilation.
It uses CuTe's `cuOccupancyMaxActiveClusters` wrapper with one SM's shared
memory reserved per CTA, matching this kernel's single-CTA residency.

All cluster sizes use the same combine: CTA rank `r` owns heads
`r + group * splits`. One warp loads split statistics in parallel and
broadcasts the resulting weights through shared memory. Dimension threads
then reduce the remote numerators. The original cluster barriers preserve
the lifetime of every CTA's shared partials; CTA barriers separate reuse
of the shared statistics for weights and tail dot products.
An explicit warp barrier also orders the final warp's shared reads before
its aliased weight/statistic writes; shuffle reductions alone do not provide
that shared-memory ordering.

These are kernel launch and loop parameters. Eager and CUDA Graph calls
share the implementation, and no scheduler metadata, cache allocation,
per-request state, KV format or selection semantics change. See
[the unified path](unified_path.md) and [cache concepts](cache-concepts.md).

## Reproduce the measurement

Activate the repository's Python environment, then run from the root:

```bash
python tokenspeed-kernel/test/ops/bench_qsa_sparse_attention.py \
  --cache-dtype bf16 --seq-len 65536 --max-context-len 262144
python tokenspeed-kernel/test/ops/bench_qsa_sparse_attention.py \
  --cache-dtype bf16 --seq-len 65536 --max-context-len 262144 \
  --cache-state cold
```

Use `--cache-dtype fp8` for FP8 KV, `--seq-len 65539` for three valid
remainder slots, and `--rows` for other local query counts. The generator
chooses 512 four-token groups and maps them through randomized physical
pages spanning the full allocated cache. Page zero is reserved. Sequence
length controls valid history and remainder entries; allocation capacity
does not shorten that history.

The benchmark checks a FP32 reference before timing. Warm measurements use
100 attention nodes per CUDA Graph. Cold measurements overwrite twice the
queried L2 capacity before each attention call; that overwrite precedes
the start event and is excluded from the interval. Both use 30 samples and
report the median and full observed range. Timings cover CUDA-event intervals
around attention graph nodes, including device dispatch/event gaps. Selection,
KV writes, transfers and host launch overhead are outside the interval.
For kernel execution duration itself, use a CUDA activity trace as below.
This measures one TP4 shard, not a four-GPU model forward.

An unchanged source snapshot can be compared in the same environment:

```bash
git show HEAD:tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/qsa_sparse.py \
  > /tmp/qsa-baseline.py
python tokenspeed-kernel/test/ops/bench_qsa_sparse_attention.py \
  --cache-dtype bf16 --source /tmp/qsa-baseline.py --cache-state cold
```

Use a revision preceding this change once it is committed. To profile only
the warmed callable, use `--profile` with Nsight Compute's
`--profile-from-start off --launch-count 1`. Profiling duration is diagnostic;
instrumentation and replay can differ from the graph measurement.

## Measured result

Device: NVIDIA L20A, compute capability 10.0, 152 SMs, 135,528,448 bytes L2,
233,472 shared bytes/SM, 65,536 registers/SM. Driver 580.105.08; CUDA toolkit
13.2.51; PyTorch 2.13.0+cu130, CuTe DSL 4.7.1, TVM FFI 0.1.13.post3.
The original and updated sources were measured in this same environment.

At one query row, sequence length 65,536 and capacity 262,144, the CUDA-event
intervals per call were as follows. These include device gaps around kernel
execution, most visibly in the single-call cold measurements.

| KV dtype | Cache state | Original median | Updated median |
|---|---|---:|---:|
| BF16 | Warm | 21.96 us | 7.51 us |
| BF16 | Cold | 32.13 us | 13.70 us |
| FP8 E4M3 | Warm | 10.33 us | 6.61 us |
| FP8 E4M3 | Cold | 15.84 us | 13.70 us |

The updated target-shape cold samples ranged from 13.41–13.73 us for BF16
and 12.22–13.73 us for FP8. At length 65,539, BF16 measured 7.70 us warm
and 13.68 us cold (cold range 13.63–15.74 us). These are measurements on
the listed device, not an upper bound on every invocation or other hardware.
Maximum absolute FP32 reference error at the target shape was below 7.5e-5;
the existing `rtol=atol=3.5e-2` acceptance tolerance is unchanged.

## Cold-cache timing audit

Nsight Systems 2026.1.2 CUDA graph node traces distinguish the event interval
from the attention kernel's activity timestamps. For the final 30 cold calls
in each trace, at the same target shape:

| KV dtype | Event interval median in this run | Kernel activity median | Kernel activity range |
|---|---:|---:|---:|
| BF16 | 13.70 us | 8.75 us | 8.54–9.18 us |
| FP8 E4M3 | 12.74 us | 8.10 us | 7.90–8.54 us |

Thus the earlier equal 13.70 us event medians do not establish equal kernel
execution times. Device gaps contribute about 4.6–4.9 us in these traced
single-call intervals. Do not subtract a fixed overhead from other results.
Tracing can also perturb execution; compare the two timing scopes within
the same run, without treating another profiler's absolute time as equivalent.

The smaller speed difference within the kernel has a separate explanation.
The selected K/V payload is 2 MiB in BF16 and 1 MiB in FP8, but FP8 is converted
to BF16 before MMA. Both formats retain the same BF16 MMA work, FP32 softmax,
six-head DSM combine and cluster synchronization. Only sixteen CTAs execute.
A separate Nsight Compute collection measured DRAM throughput at 2.50% and
1.33% of device peak for BF16 and FP8, respectively; barrier stalls accounted
for 36.44% of warp cycles in both. This supports latency and synchronization
as important limits here: halving storage traffic does not halve the dependent
memory-fetch latency or the common computation and synchronization work.

Reproduce the graph-node trace without `--profile`, so it includes the cache
eviction before every timed attention call:

```bash
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none \
  --cpuctxsw=none --force-overwrite=true -o /tmp/qsa-cold-bf16 \
  python tokenspeed-kernel/test/ops/bench_qsa_sparse_attention.py \
  --cache-dtype bf16 --cache-state cold
nsys export --type=sqlite --force-overwrite=true \
  --output=/tmp/qsa-cold-bf16.sqlite /tmp/qsa-cold-bf16.nsys-rep
```

Use `--cache-dtype fp8` and a separate output path for FP8. In the exported
`CUPTI_ACTIVITY_KIND_KERNEL` table, filter the demangled name (via `StringIds`)
for `MixedInputFusedMultiHeadAttentionDecode`, sort by start time and take the
last 30 entries to exclude warmups. Each duration is `(end - start) / 1000`
microseconds. The event-interval JSON is emitted by the benchmark in that run.

## Optimization review

Guidance version 2. The initial Nsight Compute 2026.1 report showed eight
blocks of 512 threads, 128 registers/thread, 205.31 KB dynamic shared memory
per block, no spills, and only 0.18 eligible warps per scheduler. Barrier
and long-scoreboard stalls accounted for 42.29% and 34.54% respectively.
Available profiler metrics included `gpu__time_duration.sum`,
`smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` and
`smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`.
The larger-cluster candidate's report confirmed a maximum cluster size of
16 and seven resident 16-CTA clusters. These observations motivated reducing
dependent work and imbalance, rather than targeting peak device bandwidth.
The final target-shape profile contains one attention launch with sixteen
512-thread CTAs, still 128 registers/thread and 205.31 KB dynamic shared
memory per CTA, with zero local or shared memory spilling requests.

| Rules | Decision and evidence |
|---|---|
| G1, P1, P11 | Apply: distribute sixteen real tiles, avoid an extra wave, and remove the mostly empty tail tile. |
| G2, P9 | Retain: no transfers or host synchronization in the timed kernel; graph timing excludes host launch gaps. |
| G3, P2, P13 | Apply: matrix loads for BF16 V and shared reuse of merge weights reduce scalar/repeated traffic. Tail dimension loads remain contiguous. |
| G4, P5 | Apply: constant balanced loop counts remove unreachable branches; masks still guard every tail access and collective participation stays uniform. |
| B1, K1 | Retain: compile and run with pinned CuTe on detected SM100. The CuTe launcher opts into non-portable clusters; the occupancy probe gates their use. |
| B2, P6 | Apply: inspect register/spill evidence and eliminate general-loop live ranges; no occupancy percentage is an acceptance gate. |
| B3, P16 | Apply: parallel DSM statistics and one head-owning combine; retain both cluster lifetime barriers and every required shared-memory reuse barrier. |
| B4, P3, P4 | Apply: reuse existing shared statistics for weights and tail reductions. Broadcasts are intentional; no extra shared allocation or cache-persistence reservation. |
| B5 | Not applicable: this kernel has no peer communication. |
| P7, P15 | Apply: expose independent statistics loads and retain the two-stage asynchronous ring with correct single-tile priming. |
| P8 | Retain: FP32 stable softmax, original BF16/FP8 inputs and descales, unchanged output dtype and tolerance. |
| P10, K2 | Retain tensor-core/TMEM execution for full tiles; use SIMD for the three-entry remainder. |
| P12 | Not applicable to the combine: no atomic reduction is introduced. Existing score-max atomics remain unchanged. |
| P14 | Retain existing aliasing and cache policies; no new read-only promises. |
| K3 | Not applicable: sequence-split clusters do not use paired-CTA MMA or multicast. |
| K4 | Retain the existing precision contract; NVFP4 weights are outside this kernel. |
| K5 | Not applicable: there is no compression codec or hardware decompression operation. |

Parallelizing the original eight-way merge alone did not improve BF16 and
was rejected. BF16 K `ldmatrix` conversion failed its alignment requirement
and was rejected without forcing an alignment promise. Earlier K prefetch
did not improve latency, and reducing TMEM stages produced negligible gains;
the original stage counts are retained. Uncollected per-instruction stall
and bank-conflict counters remain unknown.

Validation covers BF16/FP8, query and slot strides, empty rows/splits,
duplicates and null slots, non-unit descales, dominating tails, all four
compression phases, tail-only attention, changed-input graph replay, and
the 4/8/16-CTA launch regimes. The benchmark supplies independent FP32
reference checks at the requested full cache capacity.

All 31 tests in `tokenspeed-kernel/test/ops/test_qsa_sparse_attention.py`
passed with FlashInfer 0.6.18 kernels compiled from matching sources; an
older installed FlashInfer AOT cache must not be used with that version.
Compute Sanitizer racecheck reported zero errors and warnings for the
4-, 8- and 16-CTA regimes, including both KV dtypes. The multi-tile ring
reuse race was also reproduced in the unchanged original before fixing it.
Memcheck reported zero errors for the final BF16 kernel with three valid
tail entries. Filter sanitizer runs to the attention kernel, for example:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  --kernel-name kns=MixedInputFusedMultiHeadAttentionDecode \
  python tokenspeed-kernel/test/ops/bench_qsa_sparse_attention.py \
  --cache-dtype bf16 --seq-len 65539 --profile
```

Use `--tool racecheck` with the same filter; `--rows 8` and `--rows 9`
exercise the portable eight- and four-CTA regimes on this device.
