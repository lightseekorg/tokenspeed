# QSA kernels

QSA scores compressed index keys, selects logical blocks, expands them to
full-cache slots, and attends to those slots. The block selector takes an
explicit `auto`, `stream`, or `logits` path. Backend selection and overrides
preserve the requested path.

## Native top-k diagnostics

For eligible FP32 logits selection on NVIDIA SM90+, an unavailable native
CUDA top-k produces one warning per process: missing `dsv4_attention.so`,
load failure, or missing `dsv4_persistent_topk` export. Explicitly selecting
another implementation stays quiet. Selection and override rules are unchanged;
`TOKENSPEED_KERNEL_VERBOSE=1` with INFO logging shows the actual winner.

For source installs, rebuild in the target CUDA environment from the repo root:

```sh
TOKENSPEED_KERNEL_BACKEND=cuda python -m pip install -e tokenspeed-kernel/python/
```

Check the `dsv4_attention` build group and installed artifact at
`tokenspeed_kernel/thirdparty/cuda/objs/dsv4_attention/dsv4_attention.so`.
For wheels, reinstall a matching CUDA build. Automatic warnings omit raw loader
error text; call `thirdparty.cuda.dsv4_attention._load_dsv4_attention_module()`
within `tokenspeed_kernel` to inspect it locally. Restart after repair: both
successful and failed availability checks are cached.

## Streaming block selection

On gfx950 the Gluon streaming registration scores and selects row chunks that
fit `max_partial_bytes` with the gfx950 logits kernels, so both paths select
the same blocks there (see the gfx950 QSA indexer section of
`tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/README.md`).

The Blackwell CuTe implementation under `thirdparty/cute_dsl/_qsa_stream_topk/`
computes BF16 dot products with FP32 accumulation, clamps each head at zero,
and selects up to 512 compressed-block IDs. Scores retain FP32 precision;
logical IDs remain int32. A bounded 528-candidate carry includes the coarse
boundary bucket, with exact FP32 refinement on overflow and final output.
The retained minimum is a conservative producer cutoff: stale values may
retain extra candidates, but cannot exclude a required higher score.

Uniform requests use four-query tiles when their CTAs fit in the SM waves
already required by an eight-query grid; other grids keep eight-query tiles.
Shared memory limits either geometry to one CTA per SM. MMA accumulators and
score queues each have two stages. K lookahead uses the following presets:

| Query tile | Compressed-block capacity | K stages | Score entries per row |
| --- | --- | --- | --- |
| 4 | Any supported capacity | 4 | 1120 |
| 8 | Below 32768 | 2 | 1120 |
| 8 | At least 32768 | 3 | 784 |

The smaller eight-query score queue frees space for the third K stage.
Producer capacity checks derive from the selected score size. The TMA warp
caches 32 page IDs in lane registers; shuffles supply each tile's two pages.
Rounded per-warp counts bound row occupancy and determine how many subtiles
can append before another collective check. Queue publication and the 64K
logical-ID boundary retain their synchronization contracts.

`auto` uses shape metadata, independently of live device frontiers:

* Three-query requests prefer streaming from 192 rows and 8192 compressed
  blocks of capacity.
* Wider odd request widths prefer streaming from 384 rows at that capacity;
  the power-of-two logits tile otherwise cannot share K across these queries.
* Four-query requests prefer streaming from 384 rows at capacity 32768.

Other shapes retain the existing row/logits-budget policy. Smaller grids
keep logits where producer/selector overhead outweighs reuse. Explicit path
and backend arguments retain their normal selection semantics.

Shared storage is allocated once outside request-group branches. Full
single-owner groups bypass the ownership loop; mixed-owner groups and tails
process consecutive phases with a reader barrier before storage reuse. The
CTA allocates TMEM once before these phases and releases it after the final
MMA tail. A request boundary never relinquishes the CTA's allocation permit.
The score queue's acquire/release protocol protects each stage. Dead coarse
histograms hold boundary candidates and carry overflow; exact histograms
reuse a score stage only after its scores have been consumed. Running
compaction scratch remains separate from live K and score storage.

Small grids use the same warp-MMA score computation with optional bounded
proposals. Every omitted score contributes to an exact upper bound. Merge
accepts proposals only when that bound proves a complete top-k; otherwise it
reads the original FP32 scores. Large ties and clipped histogram boundaries
also retain exact refinement. Scratch is bounded by the caller's existing
budget, with the grouped kernel handling zero-budget calls.

Owners, frontiers, row counts, and tensor strides remain runtime inputs.
Every CuTe streaming algorithm widens query-row and head indices before
multiplying by runtime strides, including for strided query views. Key-page
and partial-score row addressing retain 64-bit offsets for large cache pools
and scratch buffers. Logical slots and TMA coordinates require fewer than
`2**31` key slots.
Only static geometry, explicit algorithm parameters, and bucketed capacity
specialize compilation. Graph replay refreshes the same input buffers as
eager execution; no backend-private request state is introduced.

## Numerical contract and validation

FP8 decode converts E4M3 K/V to BF16 exactly for finite values. QK/PV use
BF16 operands and FP32 accumulators; online softmax stays FP32 and its P
operand rounds to BF16. The copy/conversion pipeline does not add query or
probability quantization. The independent FP8 direct-slot prefill solution
below has a different, explicitly documented Q/P quantization contract.

Tests exhaust all 65536 adjacent FP8 byte pairs, comparing finite conversion
results as raw BF16 bits, including signed zero and subnormals. Fixed ordered
slots isolate core-attention arithmetic from top-k ordering. On SM107, a
24-case comparison against `main` commit `947b25bc9` matched core-attention
output bits exactly for identical slots, across 32K/128K/256K contexts, query
widths 1/3/4/5/8, and PDL on/off.

The complete selection/attention chain does not promise bitwise equality:

* Top-k order and boundary tie IDs are unspecified, including in native CUDA
  top-k. The same implementation can return a different ordering on replay.
* Reordering slots changes the tiles processed by online softmax, its running
  maxima, FP32 accumulation order and BF16 P roundings. In those 24 random
  cases, block sets agreed and the maximum final-output difference was
  `1.220703125e-4`, within `atol=1e-3, rtol=1e-2`.
* Logits and streaming backends can associate the four FP32 head sums
  differently. Near a tied cutoff, either backend may choose different IDs.
  Such cases are checked against reference top-k scores and attention over
  each selected set; the random-case output bound is not a bound for ties.
  An all-equal index-score fixture produced different admissible sets and a
  maximum output difference of `0.03271484375`; each output still matched
  the attention reference for its own selected set.

The corresponding default-auto QSA-chain timing comparison measured about
30.6% lower cold-L2 latency and 30.8% lower warm-L2 latency (geometric mean of
24 configurations). It includes index/top-k, slot expansion and FP8 attention,
and excludes projections, cache writes and other model/serving work.

## FP8 direct-slot prefill

`cute_dsl_fp8_qsa_sparse_prefill` handles SM107 BF16 queries with E4M3FN K/V,
256-wide heads, twelve query heads per KV head, one or two KV heads, and
512 four-token blocks plus three tail columns. Q/K/V must be dense and
16-byte aligned. Int32 selected-slot rows may have padding; int64 query
positions may be strided. Other inputs retain the existing registry candidates.
Use the full registered name to force this prefill implementation. A normal
`solution="cute_dsl"` filter respects prefill/decode traits; a forced
`override="cute_dsl"` retains the established decode entry.

The independent implementation lives in `ops/attention/qsa/_cute_dsl/`.
It reuses the existing MIT-licensed TMA, MMA, and descriptor helpers under
`thirdparty/cute_dsl/`, whose notices remain with those helpers. One CTA owns
a query and KV head. TMA gathers existing physical slots into separate K/V
stages, overlapping copies with native FP8 tensor-core operations. The existing
`QSASparseMetadata.query_positions` bounds the complete-block prefix and fixed
tail columns, including cached prefixes and ragged requests. Nonpositive slots
contribute no probability mass; empty rows produce zero. Q and the cache remain
unmodified. No packed KV copy, persistent request state, or global scratch is
introduced; the prologue remains the sole cache writer.

This solution introduces additional quantization. Each query head uses an
E4M3FN temporary with scale `max(abs(Q)) / 448` (one for an all-zero head).
Scale construction and Q division round to nearest-even FP32 before the FP8
cast. Native FP8 QK accumulates in FP32; the attention scale, key descale, and
Q scale apply afterward. Online softmax processes 128 selected tokens per
tile. Its unnormalized probabilities multiply by 448 and round to E4M3FN for
PV. The normalization sum and output accumulator stay FP32; the value descale
and inverse probability scale apply before the final BF16 output. An empty
tile keeps its running maximum at negative infinity, so later negative logits
still normalize correctly. Scalar descales are supported; tensor descales are
rejected when this implementation is forced.

These Q/P roundings can change attention values and generated tokens relative
to FA2 or BF16-operand attention. Tests compare the implementation tightly to
an independent reference with the same quantization, while real-input studies
report the additional approximation separately. Force the existing FA2 kernel
when these extra roundings are undesirable.

Only device, head geometry, and PDL specialize compilation. Query counts, pool
rows, slot/position strides, and all scales remain runtime inputs. Metadata
offsets widen to int64 before multiplication; TMA computes cache byte offsets.
Query and pool row counts must remain below `2**31`, independent of pool byte
size. Graph replay reads refreshed buffers through the same entry. PDL waits
before producer-owned reads and signals after output stores.

## PrimTS plan preparation

FlashInfer PrimTS caches cluster occupancy by CUDA device and cluster size.
New eager layouts retain their own workspace plans and route offsets while
sharing the hardware capacity query. This keeps CuTe's
occupancy-query kernel compilation in startup and preserves the serving JIT
checks when batch geometry changes.

## TMA gather sparse attention

`thirdparty/cute_dsl/qsa_tma_gather.py` provides `TmaGatherSparseAttention`, an explicit,
caller-buffered entry for the SM107 TMA-gather4 sparse attention kernel.
`cute_dsl_tma_gather_qsa_prefill` registers its unsplit, single-query tactic as
the default for SM107 BF16 prefill with 256-wide heads, twelve query heads per
KV head, four-token blocks, and 512 selected blocks. Logical metadata must be
present, K/V must use contiguous NHD storage, and K/V descales must be absent.
The existing registry retains other hardware, layouts, dtypes, and decode/MTP.
The registered name can be overridden through the normal kernel selection API;
no model-specific switch or direct third-party runtime import is needed.
The solution override `cute_dsl` keeps its existing direct-slot decode kernel;
use the full registered name to force TMA prefill. Normal selection with
`solution="cute_dsl"` still filters prefill/decode traits.

The ops adapter launches each nonempty request span using the existing CPU
`query_start_loc`. Positions are consecutive within a span; the forward layout's
live `seq_lens` is passed through without per-request tensor arithmetic.
This supports full, chunk, and ragged prefill
without synchronizing GPU metadata to the CPU. Query counts, lengths, and page
table strides remain runtime inputs. The prepared plan caches only static code;
outputs belong to each call (or capture), while lengths are views of the existing
forward metadata, with no extra
request state or persistent scratch. Warm the static page-size/PDL combination
before standalone CUDA graph capture.

The registry declares static dtype/head/block requirements. Dispatch and the
prepared entry share one physical-layout check, so alignment and stride changes
cannot silently diverge between selection and execution. The host check does
not import CuTe or CUDA bindings; compilation and launch load them on demand.

The port owns its computational source under `thirdparty/cute_dsl/_qsa_tma_gather/`.
It uses the existing CUTLASS DSL 4.8/CUDA dependencies and does not import cuDNN
frontend. The core derives from NVIDIA's
[`sparse_d256_f16.py`](https://github.com/NVIDIA/cudnn-frontend/blob/88561bbebb45c7459560490a1546f44ede4f6ea8/python/cudnn/sdpa/fwd/kernels/sm107/sparse_d256_f16.py).
The local barrier, MMA, and descriptor helpers derive from
[`frost/tile_dsl`](https://github.com/NVIDIA/cudnn-frontend/tree/88561bbebb45c7459560490a1546f44ede4f6ea8/python/cudnn/frost/tile_dsl)
at the same revision, restricted to this kernel's single-CTA, unscaled BF16 path.
The Q loader accepts its tensor map and coordinates directly. PDL uses CuTe's
standard primitive; TMEM allocation and release are explicit in the MMA role.
`thirdparty/cute_dsl/conversion.py` and `tma.py` share the packed conversion and
cached gather4 instruction with MSA, whose existing helper interfaces remain valid.
These files retain the applicable NVIDIA and MiniMax MIT notices.

`common.py` contains the needed helpers from `sdpa/fwd/kernels/_common_blackwell.py`.
The kernel constructor adapts the geometry from `make_cfg_d256_sparse` in
`sdpa/fwd/config_sm107.py`, deriving storage sizes and
barrier arrival counts. It checks that all MMA operands fit the 256 KiB shared
address window used by the descriptor format. Reduction order, masked-row
handling, and CLC consumer credits stay local to this implementation.

The upstream frontend adapter, generic configuration factories, persistent
compilation cache, packed-sequence path, fused output gate, and dense attention
paths are not part of this port. `combine.py` is a small in-tree CuTe reduction
of normalized FP32 split outputs and their natural-log LSE. It deliberately
keeps the reduction separate from the gathered attention core so either can be
optimized without importing the frontend's multi-purpose split combiner.

### Operand and lifetime contract

Construct the prepared kernel outside CUDA graph capture, explicitly selecting
`page_size`, `split_kv`, `query_group_size`, `enable_pdl`, and `device`. The supported split tactics
are 1, 4, 8, and 16. A normally imported `TmaGatherSparseAttentionKernel` owns
each tactic's static geometry and CuTe entry points. Compiled callables are
cached by device, page size, split count, query group size, and PDL; distinct prepared tactics
can coexist without module-global configuration. The entry takes:

- BF16 Q/O in `[B,S,Hq,256]`, with positive `S` and non-overlapping BSHD rows.
  Each request has the same query count, covering its final `S` KV positions;
  this includes full and chunk prefill. Outer strides may include padding but
  must be 16-byte aligned. `B*S*512` must fit a signed 32-bit list index.
- Contiguous BF16 NHD K/V caches `[slots,Hkv,256]`, with `Hq/Hkv=12` and a
  physical page size divisible by four. Q/K/V are not repacked.
- Contiguous int32 selected logical blocks `[B,S,512]`, independently for each
  query. The first `min(512, floor((position+1)/4))` entries describe distinct
  completed four-token blocks. Negative IDs are absent. The incomplete causal
  tail is appended by the kernel and must not also occur in the supplied list.
- Int32 KV lengths `[B]`: query `j` is at `length-S+j`. Nonpositive lengths
  produce zero output.
- Int32 page tables `[B,max_pages]` with contiguous columns and a runtime row
  stride. Page zero is the null page; negative, missing, and out-of-pool pages
  contribute neither values nor softmax weight, including the implicit tail.
- Explicit scale, output, and workspace. `workspace_bytes` gives the required
  split scratch size; pass `None` for an unsplit kernel. Output and scratch
  must not alias inputs or each other.

Lengths, shapes, strides, and page-table contents are runtime operands. The
prepared object contains compiled functions and static tactic configuration,
not request-slot state. Callers own output/scratch storage and keep it alive
through graph replay. A captured call reads refreshed input buffers on every
replay; there is no separate replay metadata implementation.

### Computation and synchronization

`query_group_size=1` handles one query and one KV head per work item, with twelve
heads in a sixteen-column MMA tile. The experimental `query_group_size=2` tactic
uses a 32-column tile, with sixteen padded columns per query. It requires an even `S`
and identical 512-entry selection lists within each consecutive pair; this
device-data precondition is the caller's responsibility. Independent lists use
group size one. The selected prefix and the union of open tails are loaded once;
each query masks its own selected count and causal tail. A tail that becomes a
completed block at the next position must not receive duplicate softmax weight.

Both tactics use two four-warp softmax groups per query and eight shared gather
warps, with separate MMA, Q/list-load, and CLC scheduler warps. Each softmax group
owns eight padded head columns. The paired tactic has four groups and 896
threads per CTA; the single-query tactic has two groups and 640 threads. Stage
and scheduler barrier counts follow the participating threads and warps. The
K/V ring has three slots for one query and two
for a pair; two Q/list slots and two score/probability slots keep the same
producer/last-reader ownership. For paired queries, Q/P/KV operand storage is
32/16/128 KiB, excluding list, reduction, and synchronization storage; TMEM uses
128 columns. No extra K/V consumer or barrier arrival is added: each wider MMA
consumes both queries. All CTA-wide synchronization precedes the spare warp's exit.

Ring indices and phase bits use CUTLASS `pipeline.PipelineState`, advanced in
place and carried across persistent work items. Empty-buffer waits start at
phase one; full-buffer and scheduler waits start at phase zero. Q/list loading
returns the advanced state objects so both the prologue and subsequent items
retain the same progression. Barrier arrival ownership and byte counts remain
local to the kernel.

The gather and softmax roles use the same physical-page validity predicate.
Zero-filled K/V alone would still contribute softmax weight; the score mask
must exclude invalid pages as well. This also makes a padded request with a
positive clamped length and an all-zero page table empty, even under poisoned
Q and null-page data.

QK and PV accumulate in FP32, with BF16 probabilities before PV. Split partials
are normalized FP32 outputs with natural-log LSE; the combine rounds to BF16
once after reduction. This is not a bitwise-equivalence guarantee with another
attention provider. Tests include a high-precision mathematical reference,
exact missing-page cases, and eager/graph agreement.

Provider agreement and model quality are separate checks: the BF16 probability
rounding can change logits after later layers even when individual attention
outputs meet their mathematical tolerance. Default selection does not imply
bitwise equivalence with FA2 or a model-level quality acceptance result.

## Metadata and recent writes

Padded MTP rows with negative logical positions retain them for masking,
but contribute zero complete blocks to scoring and top-k selection.

Compressed-key RoPE is read only at compression write boundaries. Non-boundary
draft rows may have valid raw-key tags before their group-start header is staged.

In decode the compression kernel is latency-bound: a chain of dependent loads,
which overlapped target verify also runs beside the attention projections on the
other stream. With a fused query, a second program per row normalizes and
rotates the query, so its loads do not delay the key's dependent loads, and the
compression-group tags are masked loads that issue together.

Each recent-cache page owns a separate raw-key ring. A later token suppresses
an earlier writer only when both address the same physical page and ring slot.
Prefill must leave the last raw keys and group-start positions in every page
it computes, including pages before the final window, so an interior state
checkpoint has valid recent-cache data when the scheduler publishes it.

Startup metadata preparation covers integer alignment classes for compressed
page-table row strides. Recent-cache writes cover these classes for row count
and the exclusive request-id bound. The ordinary unlimited prefill call also
prepares request-limited variants: pure target verification skips the writer
when there are no extends. Changed table widths and mixed-prefill request
counts can then reuse startup binaries.

## Materialized Triton scoring

The `logits` path materializes FP32 block scores before top-k selection.
Query-group count, table capacity, tensor strides, and per-row frontiers are
runtime inputs. Request count is represented by page-table rows and the
per-query request indices. The group count opts out of scalar specialization;
other integer inputs retain Triton's finite alignment classes for optimization.

`BLOCK_N`, `BLOCK_H`, and `BLOCK_D` set static tile dimensions. Query grouping
also sets `arange` and `reshape` dimensions and remains compile-time; a
bounded flag selects whether the query-group loop is needed. Model dimensions,
tensor formats, PDL, and launch tuning stay fixed for a warmed configuration.

During `kernel_warmup`, scoring compiles and loads every power-of-two group
allowed by the model geometry and both loop variants, without launching the
unused variants. Four index heads of width 128 on NVIDIA have six groups,
1 through 32, and two loop variants. Warmup also covers row-stride and capacity
classes: one, divisible by 16, and other integers. With 64-block pages, capacity
and logits strides stay divisible by 16; the three page-table stride classes
give at most 36 binaries for one fixed tensor-format/PDL configuration.
Later query-count, request-count, and table-width changes reuse these binaries
while refreshing the launch grid. Grouping and occupancy policies are the same
during prefill, decode, and CUDA graphs.

## FP8 decode head tile and grid

FP8 prefix operands advance as `K0,V0,K1,V1,...`, with QK and PV using their
buffer pipelines independently. Each conversion stage publishes two
64-element fragments together and remains owned until its MMA completes.
Initial O is published before P enables PV. Non-bulk `cp.async` completion and
the producer warpgroup barrier protect shared-ring reads and reuse without an
additional async-proxy shared-memory fence after each register load.

Unsplit 24 Q / 2 KV-head calls with one row per CTA use five gather slots with
four copies in flight; the unused MMA warp stages the three private tail K/V
slots. Other tactics retain four slots and three copies in flight. Copy width
follows the cache dtype. CUDA 13.2/PTX 9.2+ converts E4M3 pairs directly to BF16
with a 16-bit source operand; older compilers retain the FP32 conversion.
This adds no global workspace or persistent tensors. FP8 prefix gathers copy
32 bytes per lane using two 16-byte operations sharing a slot and address
calculation. Four lanes cover each 64-byte half-row together and share slot
ids through warp shuffles; cache base alignment remains 16 bytes.
Softmax skips padded head columns, retaining an even bound for paired
arithmetic and zero probabilities for padding. Eight-head output fragments
skip TMEM correction when every scaling coefficient is exactly one. All
softmax warps capture the shared maxima before any warp advances to the next
tile and updates them.

For FP8, the existing CTA joins publish local partials and retire their readers.
Only split-KV calls additionally synchronize the cluster, because their peer
CTAs read those partials through distributed shared memory. Unsplit calls avoid
both redundant cluster barriers, including between serial query rows.

The direct-slot CuTe path uses a 16-column tile for twelve query heads per KV
head with FP8 caches on SM100, SM103 and SM107, for every grid. Each query
retains its independent selected-slot list. Covering all twelve heads in one
tile avoids the second gather/conversion stream of the eight-column tactic.
Both tactics compute sixteen padded head columns in total; useful QK/PV work
is unchanged.

`_select_head_tile` picks the launch tile from the head geometry and cache
dtype alone. Query packing and the grouped BF16 private-prefix fallback use a
separate fixed-geometry helper, so launch policy cannot change their padding.

An FP8 grid of at least `3 * SM_count / 16` output tiles picks its query rows
per CTA and KV splits from 1x1, 2x1, 1x2 and 1x4. With one resident CTA per
SM, the choice minimizes the wave count times the per-CTA work: each of its
query rows gathers the 2048-entry prefix slots within one KV split and adds a
fixed per-row cost fitted on SM103. Smaller grids keep the split rule with one
query per CTA. SM100, SM103 and SM107 share both rules.

Within its KV split, each FP8 query row ends the prefix loop after the last
128-entry tile holding a positive id, and runs at least one tile. The CTA
finds that tile while it stages the row's slot ids in shared memory (a warp
max-reduce, then a CTA-wide maximum after the staging barrier); the gather
ring, both MMA warps and the softmax share it as their loop bound. Later tiles
hold only masked entries, which leave the softmax maximum, sum and output
unchanged, so outputs are bitwise identical to the full loop for any id order,
and an all-padding prefix keeps the full loop's state for the tail and the
split merge. The bound is runtime data, not a compile-time parameter. Grid
sizing still assumes full prefixes: rows with fewer selected slots, such as
short contexts and padded rows, finish sooner.

The tile and grid choice selects static geometry inside the existing decode
provider. Cache layout, writers, prefix/offload ownership and descales keep
their contracts.
Startup warmup enumerates every tile and launch configuration before graph
capture, and rows remain dynamic within each configuration. Tests cover Hq/Hkv
of 24/2 and 12/1, target and draft query widths, the wave-count choices on
152- and 212-SM grids, the sixteen-column tile at each KV split of SM107 grids
below one output tile per SM, padding, prefixes ending inside or on the edge of
a tile, refreshed cache values, PDL on/off, large KV offsets and execution
without further compilation after warmup.

## BF16 KV scratch for FP8 prefill

`qsa_bf16_kv_scratch` copies the FP8 E4M3 KV pages one call can read into a
caller-owned BF16 scratch and rewrites the call's selected slots to it, so a
large prefill call over an FP8 cache can run the direct-slot CuTe kernel's
BF16 path, one query row at a time (`max_seqlen_q=1`), instead of the FP8
prefill kernel that selection picks: FlashInfer FA2 by default on NVIDIA,
where the CuTe kernel is registered for decode only, or the CuTe kernel's FP8
path under a kernel override. The runtime uses it behind
`--enable-qsa-prefill-bf16-scratch`.

Request `r` owns scratch pages `[first_r, first_r + n_r)`, with
`n_r = ceil(seq_lens[r] / page_size)` and `first_r = 1 + sum(n_s for s < r)`.
`qsa_bf16_scratch_pages` gives a call's page count from host lengths and the
copy launches one program row per scratch page, so planning reads nothing
from the device. Each request's page-table columns `[0, n_r)` are copied in
order. E4M3 to BF16 is exact, so the BF16 kernel sees the values an FP8 kernel
reading the cache at unit scale computes. Scratch page 0 is never written:
padding keeps its ids, and the caller keeps page 0 finite for the reads that
still address slot 1.

The page map has one int32 entry per cache page. The copy writes every entry
the remap then reads, so its prior contents are never read and it may alias
memory that is free until the remap finishes; the runtime uses the attention
output buffer, which the kernel writes after the remap. This holds when every
positive slot lies in its request's pages and the device lengths match the
host lengths the launch was sized from. A slot outside them reads a stale
entry and becomes padding or another page of the call, never an address past
the copied pages. Page, slot and row offsets are 64-bit. Request and slot
counts and the table and slot-row strides opt out of scalar specialization;
the page size, slot width and PDL are compile-time.

The direct-slot CuTe kernel's BF16 gathers zero-fill padded entries, a
`cp.async` with zero source bytes, instead of reading slot 1, a line every CTA
would otherwise read. The softmax masks padded entries by their ids, so the
output is unchanged; rows near request starts have many of them.
`cute_dsl_blackwell_qsa_sparse_attention` also accepts a caller-owned `out`,
so the runtime's scratch path allocates nothing while serving.
