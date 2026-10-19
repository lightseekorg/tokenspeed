# Query sparse attention kernels

QSA scores compressed index keys, selects logical blocks, expands them to
full-cache slots, and attends to those slots. The public block selector in
`ops.attention.qsa` takes an explicit `auto`, `stream` or `logits` path and
dispatches registered scoring, selection and streaming implementations through
`select_kernel`. Backend selection (`solution`) and kernel overrides preserve
the requested path.
Slot expansion and attention share the same contracts for all paths.

## Automatic selection

Automatic selection uses query rows, compressed block-table capacity and
supported layouts, independently of prefill/decode. The current policy selects
grouped CuTe streaming from 512 rows for tables of at least
16384 compressed blocks (64K tokens at ratio four), and from 8192 rows for
shorter tables. Smaller batches use logits. Uniform runs must satisfy the
CuTe grouping contract; ragged runs and unsupported layouts use logits while
its matrix fits the caller's budget. Beyond that budget, selection uses
streaming, with portable Triton covering unsupported CuTe layouts.

The runtime passes `TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH` and
`TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB` to this shared entry. Explicit `stream`
and `logits` paths bypass the automatic choice and its score-matrix budget.
`solution="cute_dsl"` or `solution="triton"` restricts the backend within the
chosen path, with normal capability and layout filtering. Required path
features also reject forced kernel names from the other algorithm; the CuTe entry
rejects unsupported layouts when called directly or through an override.

## Streaming block selection

On NVIDIA Blackwell, CuTe DSL accepts four BF16 index heads of width 128,
64-block pages, block top-k 512, and capacities up to 131072 compressed blocks
(512K tokens at compression ratio four), including speculative context padding.
Strided Q vectors must remain 16-byte aligned. Page-table columns must be
contiguous; their runtime row stride may retain the persistent table capacity
when a batched prefill trims its view. Keys and row metadata remain contiguous
and aligned.

Grouped scoring shares each K tile across eight consecutive queries from one
request. The caller supplies the uniform request-run width; grouping is never
inferred from row count. Each selector thread keeps eight packed FP32-score/id
entries, forming an exact 512-entry running set per query. Large prefills need
no global selection scratch. Capacities up to 4096 compressed blocks use a
single score tile; longer streams overlap MMA, score publication and top-k
selection through two K stages, two accumulator stages and two score buffers.
The long-stream CTA uses 26 warps and 1120-entry candidate buffers containing
FP32 scores and UInt16 offsets within a 64K-block window. Score stages flush at
window boundaries, and selection restores the full block ids from the stage
base. Compact split candidates similarly store tile-local offsets. Neither
representation truncates ids beyond 65535. Dedicated compaction scratch protects
both K stages.

Each 128-thread producer warpgroup owns an independent two-slot score queue
for four query rows. Its selectors release those slots independently of the
other four rows; the shared MMA accumulator still waits for both producer
warpgroups to finish reading. Producers load their fixed row frontiers into
registers once after CTA initialization, avoiding shared-memory reloads for
every 128-key subtile. Graph replay reads the refreshed frontiers each time.
After every append, the producer warpgroup synchronizes both before and after
reading its candidate-count snapshot. The second synchronization keeps a fast
warp's next append from changing the counts before slower warps decide whether
to flush. Every warp must make the same decision to preserve queue progress.

Once 512 candidates exist, their monotone score cutoff filters later scores
before histogram updates and compaction. An older cutoff can admit extra
candidates but cannot discard a winner. Coarse histograms locate a threshold
bucket; exact FP32 refinement selects the boundary. Large tie buckets use full
radix refinement. FP16 or linear bins only filter candidates and never replace
FP32 ranking. Each row retains its own frontier; groups stop after their longest
frontier. Rows with at most 512 complete blocks return all valid ids and pad
remaining slots with `-1`. Output order and boundary tie ids are unspecified.

The compact selector unrolls its bounded candidate scan so running entries use
fixed register indices. The full radix fallback retains runtime indexing;
query counts and frontiers do not introduce new JIT specializations.

Grids of up to 128 queries expose parallelism by splitting the key range.
Independent rows, including ragged layouts, use four-warp native BF16 MMA score
CTAs and an exact merge. Small grouped prefills with long contexts reuse K
across eight queries in each split. Some small grids propose fewer candidates
per key tile, preserving the largest discarded FP32 score. The merge uses the
reduced pool only when its threshold proves that every discarded score is
ineligible; otherwise it selects from complete scores. Compact candidate ids
and short-context linear bins reduce selection traffic and histogram overhead.

`max_partial_bytes` bounds global scratch for both CuTe and Triton, excluding
the selected-id output. If split scratch exceeds the budget, a supported grouped
CuTe layout uses CTA-local streaming; kernel selection excludes other CuTe layouts.
Portable Triton groups a divisor of the explicitly supplied request-run width,
up to four queries, to reuse each K tile. Ragged runs use group size one in the
same kernel. Each row keeps its own frontier and exact packed FP32-score/id
running top-k; a group stops at its longest frontier and writes final ids
directly. The monotone cutoff skips tile selection when no new key can qualify.
Head-dimension tiles of 32 bound shared memory, and split comparison pairs reduce
sorting layout conversions. Sorted tiles are padded after sorting, and groups
with at most top-k complete blocks return all ids without scoring. No partial
or merge buffer is allocated, even with a zero budget.
Query count, request count, page-table width, Q strides and live frontiers remain
runtime arguments. Split geometry uses power-of-two capacity buckets to bound
the tuned kernels' compilation cache. PDL waits precede asynchronous Q copies
and producer-owned metadata reads; captured graphs retain the supplied flag.

## Materialized block selection

The `logits` path composes `qsa_block_scores` and `qsa_block_topk_select`.
Scoring uses the bounded, occupancy-aware grid from the shared QSA scorer;
selection chooses the length-aware CUDA, Gluon or portable Triton solution.
CUDA manages its scratch inside its own solution. Score tails remain
uninitialized and every selector receives each row's complete-block frontier.
The fused path uses `qsa_block_topk_stream`, with explicit uniform query widths
for both CuTe and Triton. The stages declare required `logits` or `stream`
features, keeping overrides within the requested algorithm; explicitly selected
backends also apply to both materialized stages.

## Measurement

`test/nvidia/ops/bench_qsa_stream_topk.py` compares selected CuTe streaming,
portable Triton streaming, logits with persistent CUDA top-k, and automatic
selection. All paths use the public API with the same inputs and clear L2
before each invocation. CUDA events exclude
cache clearing. Graph timing excludes host launch gaps and reports kernel
registers, shared memory and occupancy limits through the CUDA driver.

From the repository root, measure a causal 16K-token prefill with:

```bash
PYTHONPATH=python:tokenspeed-kernel/python \
python tokenspeed-kernel/test/nvidia/ops/bench_qsa_stream_topk.py \
  --rows 16384 --pages 1024 --kv-tokens 32768 --queries-per-request 16384 \
  --solution cute_dsl --max-logits-mb 256 --timing graph --frontier causal \
  --distribution random --pdl on --warmups 10 --iterations 40
```

This reserves a 256K-token page-table capacity with 32K live KV tokens and a
16K-query causal chunk. Omit `--kv-tokens` to use the table's full capacity.
Requests use disjoint physical key pages, with at least 1024 physical pages in
the pool. Change `--solution` to `triton`, `logits` or `auto` for comparisons. Use `--frontier
full` for every row attending to the full prefix, and `--distribution ties`
for zero queries that stress equal-score selection. Cold-cache isolated
measurements establish kernel costs; end-to-end model measurements must also
include slot expansion, sparse attention and interference from other layers.

The baseline matrix before the stream tuning below, after merging `main` at
`26fffca5`, uses NVIDIA L20B (SM103),
CUDA 13.0, PyTorch 2.14.0+cu130, CuTe DSL 4.8.0 and tokenspeed-triton
3.8.10.post20260920. BF16 queries have four 128-wide heads and top-k 512,
with PDL enabled, causal frontiers and cold L2. The matrix covers 20 layouts,
three seeds and three paths (CuTe, logits and auto), with 10 warmups and
45 samples per run. Backends use paired inputs on the same GPU in randomized
order. Every run validates up to eight rows against FP32 reference scores,
including counts, unique ids and padding. Times below are medians of the three
seed medians, in microseconds, and cover scoring plus selection.

| Query rows / live KV tokens | Logits | CuTe stream | Stream time change |
| --- | ---: | ---: | ---: |
| 1 / 32768 | 13.73 | 11.68 | -14.9% |
| 256 / 32768 | 32.10 | 66.62 | +107.6% |
| 512 / 65536 | 87.49 | 99.39 | +13.6% |
| 1024 / 131072 | 179.65 | 165.22 | -8.0% |
| 8192 / 8192 | 124.29 | 114.02 | -8.3% |
| 8192 / 32768 | 347.58 | 376.16 | +8.2% |
| 16384 / 16384 | 312.77 | 243.04 | -22.3% |
| 16384 / 32768 | 603.55 | 681.31 | +12.9% |
| 16384 / 65536 | 1150.37 | 1164.67 | +1.2% |
| 16384 / 262144 | 4103.52 | 3707.17 | -9.7% |

These rows use trimmed tables and one uniform request. The matrix also covers
multiple requests, eight-row request runs, and tables padded to 65600 compressed
blocks. Capacity can change the chosen CuTe specialization: 16384 rows / 16384
live KV tokens with that padded table takes 349.60 us for stream and 318.88 us
for logits, reversing the trimmed-table result.

The latest bounded scorer and CUDA selector make logits faster than the PR's
original baseline. CuTe still wins on fresh 8K/16K prefills and some long-context
cases, while logits wins on several intermediate contexts. Existing automatic
shape thresholds are preserved in this merge; they do not always choose the
lowest latency with the new scorer. The score-matrix budget can also force
streaming even when explicitly requested logits would be faster. These
operator measurements do not imply whole-model throughput improvements.

The portable Triton stream retains the PR's output-only optimization and its
zero-scratch contract; the comparison above uses CuTe streaming and the latest
registered materialized scoring/selection path.

Nsight Systems captures of the same graph nodes, with seed 83 and 20 replays
per case, confirm the direction of the event measurements. The following
min/mean/max GPU spans run from the first scoring kernel's start to the final
selector's end, excluding cache clearing and CPU launch gaps, in microseconds.
Scoring and CUDA selection overlap through PDL, so their individual kernel
durations must not be added to estimate latency.

| Query rows / live KV tokens | CuTe stream min/mean/max | Logits min/mean/max |
| --- | ---: | ---: |
| 8192 / 8192 | 108.77 / 109.56 / 110.31 | 117.67 / 118.64 / 122.27 |
| 16384 / 16384 | 236.90 / 238.11 / 239.01 | 306.50 / 308.07 / 309.57 |
| 8192 / 32768 | 370.31 / 371.19 / 372.77 | 341.48 / 345.95 / 355.27 |
| 16384 / 32768 | 675.46 / 676.30 / 679.17 | 594.60 / 602.21 / 612.45 |

### Grouped stream tuning before the persistent top-k update

The bounded register scan, independent producer score queues and register-held
frontiers reduce selector indexing, cross-row waits and repeated shared loads.
A paired public-API comparison on the same L20B uses seeds 83/89/97, 10 warmups
and 65 cold-L2 CUDA Graph samples per seed. For 16384 query rows and 32768 live
KV tokens, an 8192-block table and PDL enabled, the medians of seed medians are:

| Implementation | Score plus selection (us) |
| --- | ---: |
| Stream before tuning | 681.28 |
| Stream with fixed register indices | 632.19 |
| Stream with independent queues and register frontiers | 599.55 |
| Logits with persistent CUDA top-k | 603.84 |

Stream is 12.0% faster than the initial version, 5.2% faster than the register
scan alone and 0.7% faster than logits for this case. All three paired seeds
favor stream. The kernel retains 64 registers per thread, 231808 dynamic
shared-memory bytes, 832 threads per CTA and one resident CTA per SM, with no
spills. Accumulation and exact boundary refinement remain FP32. Profiler
replay timings are not used for the latency comparison.

Nsight Systems isolates 20 graph launches per implementation using NVTX ranges.
The fused-kernel min/mean/max durations are 675.62 / 676.09 / 676.71 us
initially, 626.95 / 628.11 / 629.12 us after the register scan, and
595.24 / 595.86 / 596.74 us with the independent queues and register frontiers.

Full-row validation covers the three target seeds, 8192 queries / 32768 KV,
16384 queries / 65536 KV, a padded 16384 / 16384 table, zero-score ties and
quantized inputs with boundary frontiers. All 90624 rows pass independent
FP32 score-multiset, valid-count, unique-index and padding checks at
`atol=1e-4, rtol=1e-5`. Graph tests also refresh unequal producer-half frontiers,
including empty or trivial halves and ids crossing the 64K offset window.

### Comparison with the latest persistent top-k

After merging `main` at `c8b4c742f` (exact overflowing-bucket refinement), the
CUDA extension was rebuilt for SM103. The stream implementation is unchanged.
The same 20 layouts were measured before and after the merge, using seeds
83/89/97, 10 warmups and 65 cold-L2 CUDA Graph samples per seed with PDL enabled.
Inputs and backend order are paired within each revision. Every run validates
up to eight rows against FP32 reference scores. Times are medians of the three
seed medians, in microseconds, and include scoring plus selection.

Representative trimmed-table, single-request results:

| Query rows / live KV tokens | Previous logits | Updated logits | Stream | Stream time reduction vs updated logits |
| --- | ---: | ---: | ---: | ---: |
| 256 / 32768 | 32.16 | 34.21 | 58.88 | -72.1% |
| 512 / 65536 | 87.39 | 97.31 | 93.25 | 4.2% |
| 1024 / 131072 | 179.30 | 181.60 | 159.07 | 12.4% |
| 8192 / 8192 | 124.26 | 144.74 | 113.95 | 21.3% |
| 8192 / 32768 | 347.55 | 380.35 | 331.10 | 12.9% |
| 16384 / 16384 | 312.74 | 372.13 | 243.04 | 34.7% |
| 16384 / 32768 | 603.46 | 664.64 | 599.46 | 9.8% |
| 16384 / 65536 | 1152.10 | 1340.83 | 1010.78 | 24.6% |
| 16384 / 262144 | 4104.26 | 4129.22 | 3194.27 | 22.6% |

Stream has lower latency in 15 of the 20 measured layouts. Logits still wins
for 256-query chunks, eight independent decode requests and 16 rows grouped
into eight-query requests. It also narrowly wins for 512 total rows in
eight-query requests: 105.82 us for logits and 107.62 us for stream.

Capacity remains relevant. With 65600-block tables, 16384 queries / 32768 live
KV tokens takes 601.15 us for stream and 734.94 us for updated logits. For
16384 queries / 16384 live KV tokens, the same capacity takes 312.70 us and
386.11 us respectively. Stream timings stay close to their pre-merge values;
the larger advantage comes from the updated materialized selector's cost.
Automatic path thresholds remain unchanged.
