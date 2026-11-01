# DeepSeek V4.1 context parallelism

V4.1 has three independent responsibilities: QCP partitions prefill queries,
KVP partitions persistent global KV and indexer K pages, and DCP computes
decode attention over those pages. All groups are subgroups of attention TP;
these modes do not add devices.

## Terminology

**Aggregate QCP** names the existing operator-level query partitioning for
the indexer and attention kernels. **PD Prefill QCP** names the model-wide
input query shard carried through the model. Aggregate describes the serving
mode, not the number of machines. The two implementations retain separate
options; no automatic role-based dispatch is performed.
The existing `--deepseek-v41-*-query-parallel-size` option names are unchanged.

## Query computation

The model-specific aggregate indexer QCP path interleaves query rows across attention TP
before indexer query/weight projection. Each rank scores its local queries
against the complete logical history. Logical Top-k IDs and lengths are
gathered back in original row order. The candidate-producing Full indexer and
CED-tail Reindex remain on their existing paths.

Aggregate attention QCP exchanges local-head queries for a contiguous subset of query
rows with all heads. Selected prefill attention computes that subset, then an
inverse exchange restores the TP head layout. SWA-only and global-attention
layers have separate minimum query counts to amortize communication.

These are operator-local partitions. Hidden states, Engram, MoE, compressor
work, and DSpark auxiliary states retain their existing layouts. Generic
`--query-context-parallel-size` selects the separate model-wide QCP layout.
V4.1 model-wide QCP currently supports eager, text-only PD prefill with
replicated attention heads and either KVP=1 or KVP=QCP. It preserves local
token ownership through Engram, MoE and CED narrowing, gathers projected keys
for cache publication, and gathers only sampled hidden rows at the output. It rejects
PP, mixed batches and prefill graphs. With DSpark, it gathers the retained
CED hidden taps in rank order before the existing drafter writes replicated
SWA-window fields. Empty owners join the gather; cache coordinates remain
those of the complete decoder view.
Aggregate QCP and model-wide QCP must not be enabled together.

## Cache ownership

The global ratio-1 and ratio-2 cache groups use existing page-cyclic KVP
ownership. Indexer K shares the ownership of its global KV group. Writes
translate virtual slots to local slots and mask nonowners; logical selections
remain valid across all owners. Prefix matching, allocation, transfer and
eviction continue to use LCM cache groups.

Model-wide QCP can use the same owner-local cache writes and per-request
history materialization. Query ownership does not change KV page ownership;
all ranks gather the same request history, including ranks with no local
queries after CED narrowing. Each query owner computes attention against the
complete gathered history, so prefill does not merge partial attention by LSE.

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

For TP8, enable either or both aggregate QCP paths with:

```text
--deepseek-v41-indexer-query-parallel-size 8
--deepseek-v41-attention-query-parallel-size 8
--disable-prefill-graph
```

Each query size is either 1 (disabled) or the attention TP size. Aggregate indexer QCP
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
separate prefill/decode rounds for aggregated DCP. Aggregate QCP can be enabled with
either replicated or sharded KV. V4.1-specific restrictions are listed below.

For model-wide PD prefill, use replicated heads and optionally shard the
global cache over the query group:

```text
--query-context-parallel-size 8
--kv-parallel-size 8
--decode-context-parallel-size 1
--disable-prefill-graph
--disaggregation-mode prefill
--disaggregation-layerwise-interval 0
```

KVP must be 1 or equal to QCP. This combination reuses the existing history
gather path. QCP8/KVP8 prefill feeding KVP8/DCP8 decode has passed a
five-request smoke with DSpark, decode CUDA Graph, chunked prompts and prefix
hits. A paired 128-question LongBench-v2 run with DSpark and decode CUDA
Graph scored 78/128 for the CP-disabled baseline, 80/128 for QCP8/KVP1,
and 78/128 for QCP8/KVP8 prefill with KVP8/DCP8 decode. This limited
sample does not establish statistical or token-level equivalence.

PD prefill can use KVP with DCP size 1 and
`--disaggregation-layerwise-interval 0`. V4.1 PD decode with KVP uses the existing page-owner transfer planning
and DCP decode path. Eager PD KVP decode has smoke coverage. PD DSpark uses the same
SWA-window fields as aggregate serving; both P and D must enable DSpark so
the transfer includes the context windows. KVP requires eager prefill and
no mixed batches.
Aggregate serving supports DSpark and CUDA Graph decode. PD Prefill QCP8
with replicated KV feeding KVP8/DCP8 decode now has DSpark smoke coverage
with decode CUDA Graph enabled (capture sizes up to 4); prefill stays eager.
Eight requests completed, including a chunked 18,018-token prompt and
concurrent submissions with prefix-cache hits. This confirms basic execution,
not general accuracy or graph replay coverage for every batch size. The long
prompt matched the earlier eager/non-speculative output; the short prompt
continued repeating after the correct answer instead of stopping. Paired
speculation/graph accuracy checks remain required.

## Validation and tradeoffs

A four-GPU reduced-model test compares QCP4/KVP1 and QCP4/KVP4 sampled
hidden outputs exactly, including empty query owners and CED narrowing.

Runtime tests cover query ordering, head exchange, packed-page transport,
cache ownership, metadata refresh and selection reuse. Kernel tests compare
partial outputs/LSE against a reference, local Top-k against replicated
scoring, and repeated graph replay with changing lengths.

Aggregate QCP targets long-prefill latency. KVP trades communication for persistent KV
capacity; V4.1's small selected-attention workload does not imply a decode
speedup. Performance and model accuracy need paired workload measurements.
Changing collective/reduction order can change greedy outputs, so equal task
accuracy is not a claim of token-for-token equivalence.
