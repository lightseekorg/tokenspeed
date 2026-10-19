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

Measurements after merging `main` at `26fffca5` use NVIDIA L20B (SM103),
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

### Bounded selector scan

Unrolling the compact scan removes dynamic searches through each thread's
eight running entries. A paired public-API comparison on the same L20B,
using seeds 83/89/97, 10 warmups and 45 cold-L2 CUDA Graph samples per seed,
measures the following medians of seed medians for 16384 query rows and
32768 live KV tokens, with an 8192-block table and PDL enabled:

| Implementation | Score plus selection (us) |
| --- | ---: |
| Stream before unrolling | 681.34 |
| Stream with fixed register indices | 632.22 |
| Logits with persistent CUDA top-k | 605.22 |

Stream latency decreases by 7.2%. The kernel retains 64 registers per thread,
231808 dynamic shared-memory bytes, 832 threads per CTA and one resident
CTA per SM, with no spills. Nsight Compute records about 401.7 million
executed instructions, down from 444.1 million. Profiler replay timings are
not used for the latency comparison.

Nsight Systems measures the single fused kernel over 20 graph replays at
675.37 / 676.16 / 678.98 us before unrolling and
627.40 / 628.22 / 629.89 us afterward (min/mean/max).

Full-row validation covers the three target seeds, 8192 queries / 32768 KV,
16384 queries / 65536 KV, a padded 16384 / 16384 table, zero-score ties and
quantized inputs with boundary frontiers. All 90624 rows pass independent
FP32 score-multiset, valid-count, unique-index and padding checks at
`atol=1e-4, rtol=1e-5`.
