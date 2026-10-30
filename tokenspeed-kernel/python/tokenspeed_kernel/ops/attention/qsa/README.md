# QSA kernels

QSA scores compressed index keys, selects logical blocks, expands them to
full-cache slots, and attends to those slots. The block selector takes an
explicit `auto`, `stream`, or `logits` path. Backend selection and overrides
preserve the requested path.

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

## Rubin FP8 decode head tile

The direct-slot CuTe path uses a 16-column tile for twelve query heads per KV
head with FP8 caches on SM107, once `query_rows * kv_heads >= SM_count`. Each
query retains its independent selected-slot list. Covering all twelve heads in
one tile avoids the second gather/conversion stream of the eight-column tactic.
Both tactics compute sixteen padded head columns in total; useful QK/PV work
is unchanged.

`_select_head_tile` owns the device/grid policy. Query packing and the grouped
BF16 private-prefix fallback use a separate fixed-geometry helper, so launch
policy cannot change their padding.

This selects static geometry inside the existing decode provider. Split4, one
query per CTA, cache layout, writers, prefix/offload ownership and descales keep
their contracts. Smaller grids and other architectures keep the prior tactic.
Startup warmup enumerates both tiles before graph capture, and rows remain
dynamic within each configuration. Tests cover Hq/Hkv of 24/2 and 12/1, target
and draft query widths, both sides of the 212-SM grid threshold, padding,
refreshed cache values, PDL on/off, large KV
offsets and execution without further compilation after warmup.
