# DeepSeek V4.1 context parallelism

V4.1 has three independent responsibilities: QCP partitions prefill queries,
KVP partitions persistent global KV and indexer K pages, and DCP computes
decode attention over those pages. All groups are subgroups of attention TP;
these modes do not add devices.

## Query computation

The model-specific indexer QCP path interleaves query rows across attention TP
before indexer query/weight projection. Each rank scores its local queries
against the complete logical history. Logical Top-k IDs and lengths are
gathered back in original row order. The candidate-producing Full indexer and
CED-tail Reindex remain on their existing paths.

Attention QCP exchanges local-head queries for a contiguous subset of query
rows with all heads. Selected prefill attention computes that subset, then an
inverse exchange restores the TP head layout. SWA-only and global-attention
layers have separate minimum query counts to amortize communication.

These are operator-local partitions. Hidden states, Engram, MoE, compressor
work, and DSpark auxiliary states retain their existing layouts. Generic
`--query-context-parallel-size` remains the separate DSA model-wide layout.

## Cache ownership

The global ratio-1 and ratio-2 cache groups use existing page-cyclic KVP
ownership. Indexer K shares the ownership of its global KV group. Writes
translate virtual slots to local slots and mask nonowners; logical selections
remain valid across all owners. Prefix matching, allocation, transfer and
eviction continue to use LCM cache groups.

SWA, compressor tails and DSpark state remain replicated. Prefill materializes
one request's packed history at a time. Communication copies complete packed
pages, preserving quantized values and scale offsets without dequantizing
and requantizing them.

## Decode computation

Decode and DSpark verification share the metadata-refresh path for eager
execution and CUDA Graph replay. Query heads are gathered across the KVP
group; each rank attends its owned global pages. Replicated SWA contributes
on one rank, and the attention sink is applied once when merging the partial
outputs by log-sum-exp.

On supported Hopper indexer paths, ordinary Full and Reindex compute local
Top-k candidates and merge their scores and logical IDs across ranks.
The candidate-producing Full source retains complete-history materialization
because its block maxima feed later Reindex layers. Unsupported native paths
also materialize history. This is exact distributed Top-k selection under the
kernel's score/tie rules, not a reduced candidate budget.

Graph decisions use static capacity bounds, never the current verify width as
a substitute for history length. Pointer-keyed selection and geometry caches
are scoped to one forward and invalidated by metadata refresh.

## Configuration

For TP8, enable either or both operator-level QCP paths with:

```text
--deepseek-v41-indexer-query-parallel-size 8
--deepseek-v41-attention-query-parallel-size 8
--disable-prefill-graph
```

Each query size is either 1 (disabled) or the attention TP size. Indexer QCP
also requires sufficiently long query/history windows. Attention thresholds
are set by `--deepseek-v41-attention-query-min-tokens` and
`--deepseek-v41-attention-query-swa-min-tokens`.

For V4.1 aggregated KVP+DCP, select the backend and matching sizes:

```text
--attention-backend deepseek_v41
--kv-parallel-size 8
--decode-context-parallel-size 8
```

KVP may be any supported divisor of attention TP. Shared launch validation
requires matching KVP/DCP sizes for aggregated and PD decode engines, and
separate prefill/decode rounds for aggregated DCP. QCP can be enabled with
either replicated or sharded KV. V4.1-specific restrictions are listed below.

PD prefill can use KVP with DCP size 1 and
`--disaggregation-layerwise-interval 0`. V4.1 PD decode with KVP is not
supported. KVP requires eager prefill, no mixed batches, and either no
speculation or DSpark. CUDA Graph decode is supported.

## Validation and tradeoffs

Runtime tests cover query ordering, head exchange, packed-page transport,
cache ownership, metadata refresh and selection reuse. Kernel tests compare
partial outputs/LSE against a reference, local Top-k against replicated
scoring, and repeated graph replay with changing lengths.

QCP targets long-prefill latency. KVP trades communication for persistent KV
capacity; V4.1's small selected-attention workload does not imply a decode
speedup. Performance and model accuracy need paired workload measurements.
Changing collective/reduction order can change greedy outputs, so equal task
accuracy is not a claim of token-for-token equivalence.
