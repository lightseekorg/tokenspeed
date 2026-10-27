# QSA kernels

## TMA gather sparse attention

`thirdparty/cute_dsl/qsa_tma_gather.py` provides `TmaGatherSparseAttention`, an explicit,
caller-buffered entry for the SM107 TMA-gather4 sparse attention kernel. It is
not registered for runtime selection yet. The existing QSA dispatch remains the
serving path while the source port is reviewed and optimized.

The port owns its computational source under `thirdparty/cute_dsl/_qsa_tma_gather/`.
It uses the existing CUTLASS DSL 4.8/CUDA dependencies and does not import cuDNN
frontend. The core derives from NVIDIA's
[`sparse_d256_f16.py`](https://github.com/NVIDIA/cudnn-frontend/blob/88561bbebb45c7459560490a1546f44ede4f6ea8/python/cudnn/sdpa/fwd/kernels/sm107/sparse_d256_f16.py).
The local barrier, MMA and descriptor helpers derive from
[`frost/tile_dsl`](https://github.com/NVIDIA/cudnn-frontend/tree/88561bbebb45c7459560490a1546f44ede4f6ea8/python/cudnn/frost/tile_dsl)
at the same revision, restricted to this kernel's single-CTA, unscaled BF16 path.
The Q loader accepts its tensor map and coordinates directly. PDL uses CuTe's
standard primitive; TMEM allocation and release are explicit in the MMA role.
`thirdparty/cute_dsl/conversion.py` and `tma.py` share the packed conversion and
cached gather4 instruction with MSA, whose existing helper interfaces remain valid.
These files retain the applicable NVIDIA and MiniMax MIT notices.

`common.py` contains the needed helpers from `sdpa/fwd/kernels/_common_blackwell.py`.
The kernel constructor records the two-softmax-group geometry from
`make_cfg_d256_sparse` in `sdpa/fwd/config_sm107.py`, deriving storage sizes and
barrier arrival counts. It checks that all MMA operands fit the 256 KiB shared
address window used by the descriptor format. Reduction order, masked-row
handling and CLC consumer credits stay local to this implementation.

The upstream frontend adapter, generic configuration factories, persistent
compilation cache, packed-sequence path, fused output gate, and dense attention
paths are not part of this port. `combine.py` is a small in-tree CuTe reduction
of normalized FP32 split outputs and their natural-log LSE. It deliberately
keeps the reduction separate from the gathered attention core so either can be
optimized without importing the frontend's multi-purpose split combiner.

### Operand and lifetime contract

Construct the prepared kernel outside CUDA graph capture, explicitly selecting
`page_size`, `split_kv`, `query_group_size`, `enable_pdl` and `device`. The supported split tactics
are 1, 4, 8 and 16. A normally imported `TmaGatherSparseAttentionKernel` owns
each tactic's static geometry and CuTe entry points. Compiled callables are
cached by device, page size, split count, query group size and PDL; distinct prepared tactics
can coexist without module-global configuration. The entry takes:

- BF16 Q/O in `[B,S,Hq,256]`, with `S` from 1 to 4 and non-overlapping BSHD
  rows. Outer strides may include padding but must be 16-byte aligned.
- Contiguous BF16 NHD K/V caches `[slots,Hkv,256]`, with `Hq/Hkv=12` and a
  physical page size divisible by four. Q/K/V are not repacked.
- Contiguous int32 selected logical blocks `[B,S,512]`, independently for each
  query. The first `min(512, floor((position+1)/4))` entries describe distinct
  completed four-token blocks. Negative IDs are absent. The incomplete causal
  tail is appended by the kernel and must not also occur in the supplied list.
- Int32 KV lengths `[B]`: query `j` is at `length-S+j`. Nonpositive lengths
  produce zero output.
- Int32 page tables `[B,max_pages]` with contiguous columns and a runtime row
  stride. Page zero is the null page; negative, missing and out-of-pool pages
  contribute neither values nor softmax weight, including the implicit tail.
- Explicit scale, output, and workspace. `workspace_bytes` gives the required
  split scratch size; pass `None` for an unsplit kernel. Output and scratch
  must not alias inputs or each other.

Lengths, shapes, strides and page-table contents are runtime operands. The
prepared object contains compiled functions and static tactic configuration,
not request-slot state. Callers own output/scratch storage and keep it alive
through graph replay. A captured call reads refreshed input buffers on every
replay; there is no separate replay metadata implementation.

### Computation and synchronization

`query_group_size=1` handles one query and one KV head per work item, with twelve
heads in a sixteen-column MMA tile. The experimental `query_group_size=2` tactic
uses a 32-column tile, with sixteen padded columns per query. It requires S=2 or
4 and identical 512-entry selection lists within each consecutive pair; this
device-data precondition is the caller's responsibility. Independent lists use
group size one. The selected prefix and the union of open tails are loaded once;
each query masks its own selected count and causal tail. A tail that becomes a
completed block at the next position must not receive duplicate softmax weight.

Both tactics use two four-warp softmax groups and eight gather warps, with
separate MMA, Q/list-load and CLC scheduler warps. In the paired tactic each
softmax group owns one query. The K/V ring has three slots for one query and two
for a pair; two Q/list slots and two score/probability slots keep the same
producer/last-reader ownership. For paired queries, Q/P/KV operand storage is
32/16/128 KiB, excluding list, reduction and synchronization storage; TMEM uses
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

Runtime registry integration, metadata/workspace ownership in the serving
backend, dispatch policy and further pipeline tuning are separate changes.
