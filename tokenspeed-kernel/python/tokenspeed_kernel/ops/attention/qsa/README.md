# QSA kernels

## Block scoring

The materialized `qwen4_exp_qsa_block_topk` solution scores compressed K blocks
with `sum_heads(relu(Q @ K))`, then selects blocks independently for each query.
Its Triton scoring kernel uses 128-block tiles and four warps per CTA.

`queries_per_request` describes row layout, not a separate execution mode:

- A positive integer means consecutive runs of that many query rows belong to
  one request. It must divide the total row count. Request IDs may appear in any
  order, but each run must have one owner.
- `None` means ragged or mixed queries, so every query is scored independently.
- The kernel groups up to four queries using a power-of-two divisor of the
  uniform width. The grouped, padded head count is capped at 16. For four heads,
  widths 1/2/3/4/6/8 therefore use groups of 1/2/1/4/2/4.

A group loads one K tile and places its queries along the matrix multiply's M
dimension. Each query retains its own head reduction, complete-block frontier,
logits row and top-k selection. K loads use the group's maximum frontier, while
output masks use each query's frontier. Invalid scores, including every score
of a padded request, are still written as `-inf` on every invocation.

The same kernel handles group size one and larger groups. Strided Q views and
page tables remain supported. Uniform widths come from the existing runtime
query-layout helper; no device-to-host length reads or new cache state are
needed. Eager and captured forwards use the same arguments. PDL still waits
before reading query/cache/metadata and releases successor setup after the wait.

For four 128-dimensional BF16 heads, the old 512-block tile used 132,096 bytes
of shared memory, allowing only one CTA per B300 SM. A 128-block tile with four
queries uses 36,864 bytes, while sharing K avoids four independent reads and
uses 16 rows of the matrix multiply. This reduces resource pressure without
changing scoring or selection semantics. The streaming solution keeps its
existing score-and-select implementation.

### Isolated B300 measurements

The following comparison uses four BF16 heads of dimension 128, a strided Q
view, random physical page mappings, 20,544 output blocks, and PDL enabled.
Effective contexts range from about 51K to 61K tokens; BS1 uses 55,523 tokens.
The baseline scores one query per 512-block tile with eight warps. Timings are
microseconds per scoring call, with identical inputs for both implementations.

| Batch | Queries/request | Baseline, repeated input | Grouped/tile128, repeated input | Baseline, cache flush | Grouped/tile128, cache flush |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 4.10 | 2.41 | 12.29 | 8.19 |
| 1 | 4 | 6.61 | 2.69 | 12.29 | 8.19 |
| 32 | 1 | 46.34 | 30.65 | 59.39 | 43.01 |
| 32 | 4 | 157.90 | 31.75 | 167.94 | 40.96 |

Repeated-input timings use 100 calls per CUDA Graph and the median of 15
replays over three alternating-order rounds. Cache-flush timings write a
256 MiB buffer before each call and take the median of 108 samples, excluding
the buffer write. The latter include CUDA event boundaries and show timing
quantization; no hardware counters establish the actual cache miss rate.
Outputs match the baseline bit for bit. These measurements cover the scoring
kernel only, not complete inference latency or throughput.

### Tests

Regression coverage lives in `test/ops/test_qwen4_exp_qsa.py`: uniform and ragged
layouts, odd widths, padded requests, non-power-of-two head shapes, frontier
boundaries, strided inputs and metadata changes during CUDA Graph replay.
`test/nvidia/ops/test_qsa_pdl.py` additionally checks producer/consumer ordering
with PDL both enabled and disabled.

## Sparse attention

`qsa_sparse_attention` consumes physical KV slots. Non-positive selected slots
are ignored; duplicates retain their multiplicity in the softmax. Query, K/V and
selected slots may change between CUDA graph replays. Cache ownership and decode
metadata follow `docs/design/unified_path.md`; the kernel adds no per-request state.

The CuTe implementation supports SM100/SM103 decode, BF16 queries, BF16 or FP8
E4M3 KV, head dimension 256 and selected width 2051. The query length remains an
explicit dispatch trait, including compact speculative verification. NVIDIA
prefill and unsupported shapes retain the existing FlashInfer FA2 selection.

### CTA tiling and operand stages

Independent output work is `query_rows * kv_heads * ceil(heads_per_kv / 8)`.
Query rows include all requests and verification tokens. The launch grid is
`(kv_splits, ceil(heads_per_kv / 8), ceil(query_rows / rows_per_cta) * kv_heads)`.
Every CTA processes its own query rows with separate selected slots, maxima,
normalizers and output accumulators. Grouped queries retain one TMEM allocation
and shared workspace. Static query iteration avoids the increased register
pressure observed with a runtime outer loop.

The BF16 policy jointly selects query tiling, splits and operand slots from the
independent output count and the CUDA-reported SM count. Starting at three
sixteenths of the SM count in independent output tiles, it tests the unsplit
coverage first and uses one, two or four splits with two operand slots. Larger
unsplit grids group two or four queries only when the remaining CTA count is at
least 1.5 times the SM count. These are measured dispatch thresholds, not fixed
wave or occupancy requirements. The query tile is capped at four to bound static
code size. Four-query correctness is tested, but its performance beyond the
configured BS128 suite is not established.

Smaller BF16 launches and FP8 retain the prior split policy: sixteen splits when
the small launch fits the wide-cluster capacity, otherwise eight for up to eight
output tiles and four for larger launches. BF16 uses one shared operand slot
there, except for the sixteen-split single-tile configuration's three-slot
allocation. FP8 retains its asynchronous gather/conversion pipeline and TMEM
operands.

For six Q heads, one KV head, MTP3 (four query rows per request), and the tested
152-SM device, the BF16 policy is:

| BS | Queries per CTA | KV splits | BF16 operand slots | Launched CTAs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 16 | 3 | 64 |
| 2 | 1 | 8 | 1 | 64 |
| 4 | 1 | 4 | 1 | 64 |
| 8 | 1 | 4 | 2 | 128 |
| 16 | 1 | 2 | 2 | 128 |
| 32 | 1 | 1 | 2 | 128 |
| 64 | 1 | 1 | 2 | 256 |
| 128 | 2 | 1 | 2 | 256 |

One BF16 operand slot holds K **or** V with shape `128 x 256`, implemented as two
`128 x 128` halves, and occupies 64 KiB. The selected two-slot configurations use
139648 bytes of dynamic shared memory and 32 TMEM columns. Copy producers issue
`cp.async` and publish readiness through `cp.async.mbarrier.arrive.noinc`; they can
fill the next free slot before the current copy completes. Consumers wait for
readiness and release storage only after the corresponding MMA has completed.
The K/V stream is `K0, K1, V0, K2, V1, ..., Vlast`; L2 hints look ahead beyond the
allocated slots. Slot ownership, copy completion and memory visibility remain
separate requirements.

Grouping queries reuses scheduling and storage; the MMA atom still has eight
head columns per query, six useful for the common shape. Independent sparse
lists are not treated as a common K/V matrix. A wider MMA would require an
additional mapping and masking design.

### Warp roles and completion

A CTA contains 16 warps. Their main-loop roles are:

| Warps | Work |
| --- | --- |
| 0–3 | Read QK scores from TMEM, apply masks and online softmax, correct the accumulated output. |
| 4–7 | Gather KV dimensions 128–255; FP8 also converts the operand into BF16 TMEM. |
| 8–11 | Gather KV dimensions 0–127; FP8 also converts the operand into BF16 TMEM. |
| 12 | Issue QK MMA. |
| 13 | Issue PV MMA and finish TMEM deallocation after the query tile. |
| 14 | Participate in CTA synchronization and the split combine. |
| 15 | Issue Q TMA loads and participate in synchronization and the split combine. |

Softmax waits for its QK result, while independent QK/PV issue warps can advance
other pipeline work. For multiple queries, the final PV named-barrier arrival
has a matching QK arrival before the next query. Completed pipeline barriers are
invalidated before reinitialization; TMEM is deallocated once after the entire
query tile.

A BF16 experiment combined QK and PV issue in one warp with the operand order
`K0, V0, K1, V1`, testing one, two and three shared slots. With fixed CTA tiling,
three slots improved this combined schedule by only 1–2% over two slots; it
remained 86–102% slower than the current implementation at BS32/64/128 with four
queries per request. Waiting for `P_i` also prevented that warp from issuing
`QK_(i+1)` during the current softmax. Extra operand buffering did not recover
that overlap. The performance gate rejected the experiment, so the separate
issue warps remain in use.

Unsplit CTAs assign one warp per head to merge the three tail entries using
stable online softmax and write the output. This removes repeated CTA-wide head
reductions. Split clusters retain the head-owning DSM combine with two groups of
256 threads and separate scratch after the completed partial-output region.
All CTA/cluster lifetime barriers remain in place.

### Occupancy and validation

A compiler `min_blocks_per_mp=2` bound on the compact one-slot kernel does not
establish two resident CTAs. Use `cuOccupancyMaxActiveBlocksPerMultiprocessor` with
the compiled function, actual thread count and dynamic shared allocation, multiply
by the CUDA-reported SM count, and separately query `cuOccupancyMaxActiveClusters`
for the real cluster launch. These API capacities are not achieved occupancy.
For the released configurations on the tested 152-SM device, the API returned
one CTA/SM, giving 152 CTAs before cluster constraints. Cluster capacities were
152/152/144/120/112 CTAs for split counts 1/2/4/8/16 respectively. Every selected
BF16 and FP8 configuration was queried using its actual launch parameters.
The earlier one-slot API/NCU capacity discrepancy remains unresolved; resource
sublimits alone must not be reported as actual residency.

The updated production QSA and PDL suites passed 160 tests. An additional 30
independent-reference cases covered two/four-query partial tiles, one/two/three
BF16 slots, all split counts, strided query/indices, multiple KV/head tiles,
invalid and duplicate slots, poisoned slot zero, dominant tails, empty rows, and
CUDA graph replay with changed K/V, query and indices.

The complete 36-workload release passed correctness and beat paired FA2 for every
workload, with a positive 95% bootstrap lower bound for each latency gain. The
suite covers BS1/2/4/8/16/32/64/128, query lengths one and four, both KV dtypes,
plus shared-index and cold-cache BS32 cases. This establishes performance for that
suite and device, not a global optimum for all supported shapes.

Representative release results below use independent selected slots, context
65536 per request, PDL and warm CUDA graphs, with 41 paired samples. Times are
median microseconds; MTP3 means query length four.

| BS | KV dtype | CuTe DSL | FA2 | Speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | BF16 | 6.24 | 12.07 | 1.93x |
| 1 | FP8 | 6.31 | 15.20 | 2.41x |
| 8 | BF16 | 11.36 | 21.34 | 1.88x |
| 8 | FP8 | 12.67 | 24.90 | 1.96x |
| 16 | BF16 | 25.06 | 37.70 | 1.50x |
| 16 | FP8 | 26.21 | 45.54 | 1.74x |
| 32 | BF16 | 45.22 | 60.58 | 1.34x |
| 32 | FP8 | 57.47 | 78.05 | 1.36x |
| 64 | BF16 | 87.21 | 100.54 | 1.15x |
| 64 | FP8 | 114.11 | 134.24 | 1.18x |
| 128 | BF16 | 172.93 | 196.32 | 1.14x |
| 128 | FP8 | 214.31 | 256.77 | 1.20x |

The minimum measured speedup across the 36 workloads is 1.135x, at BF16 BS128
verification. Runtime query grouping without a revised epilogue and four-query
tiles at BS128 did not improve on the selected combination. Keep performance
decisions tied to the joint query/split/stage measurements.

A separate paired comparison against the previous CuTe source measured BF16
BS1/2/4 verification regressions of about 0.6–2.1%. The L2 aggregate improved 3.25%;
the change is a tradeoff for the larger-batch gains. The two/four-query partial
query graph cases also passed Compute Sanitizer device memcheck with zero errors.

A second-seed BS128 control measured 173.90 us for two queries/CTA, one split and
two slots, versus 172.54 us for one query/CTA, two splits and one slot. The chosen
output-first policy accepts that 0.8% latency tradeoff while reducing the grid
from 1024 to 256 CTAs. It is not a claim that one split wins every comparison.
