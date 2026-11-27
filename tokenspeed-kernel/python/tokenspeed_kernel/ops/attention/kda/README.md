# KDA kernels

Each physical checkpoint holds only a dense recurrent base. Each resident
request slot holds every layer's FP32 correction U, normalized K and
multiplicative decay records plus one INT32 accepted count that all its layers
share: the number of records accepted since the dense base of the request's
pages. The logical endpoint is the base followed by
`S = S * decay + outer(U, K)` in record order. The base follows
`--mamba-ssm-dtype` (FP32 or BF16). Accumulation and history remain FP32.
Materialization casts the result to the dense base dtype.

## CuTe prefill (`cutedsl_kda`)

The backend selects FlashInfer's ordinary or persistent CuTe KDA scan, reading
token-strided Q/K/V directly from convolution output. Persistent prefill admits
BF16 gates, head counts divisible by eight and gate bounds in `[-5, 0)`, and is
selected when `2 * requests * heads > SM count`; smaller workloads retain the
ordinary decomposed scan. FP32 initial and final states are
separate; the runtime retains its LCM gather/writeback and checkpoint policy.
Live device boundaries build the scan plan for both eager execution and
capacity-based CUDA graphs. Inputs beyond the live total follow the runtime's
zero-padding contract: native scans may load a full tile at the tail.
Persistent planning distributes 32-token chunks
over resident workers, with at most one exact FP32 handoff per request/head
chain. Its metadata launch also resets the handoff flags on every replay.
Scratch belongs to each invocation or its graph
pool; only compiled code is cached. The FlashInfer adapter owns the pinned
native launch ABI, avoiding the public wrapper's in-place state updates and
persistent GPU scratch.

## Verify and publication

`KdaRetainedRecords` supplies original pre-convolution QKV, the retained history,
packed request boundaries, and generation/position stamps. For each batch row,
`live_state_indices` selects the private physical page and `record_slots` selects
the record storage and the count (`history.length`). Record slot zero is
valid. Shared source pages hold dense checkpoints; the slot of a request
starting from one holds no accepted records. Each live destination may alias
only its own source.

With a verify window T and a history capacity C, a count H overflows when it
lies outside [0, C - T]. Verify applies `eff` records to the live base: H, or
0 for an overflowing H, whose records the prepare (below) folded into the live
base. Each verify kernel derives `eff` itself, with one scalar select on its
compile-time capacity and window. It writes candidate U/K/decay into the
request slot after `eff` records and never changes the base or the count.
Rejected candidate suffixes need no clearing.

Commit validates candidate stamps, updates convolution from original QKV, then
publishes the request's count. Its UINT64 `[layers, 12]` descriptors contain
the QKV and conv pointers, the page count, four strides, then the v-major
recurrent-state pointer and its page, head, value and key strides; they carry
no raw key/gate pointers. The count is not a layer's: the commit takes it as
an INT32 `[slots, 1]` tensor, and `record_slots` selects each request's. The
convolution launch reads the count in every `(layer, row)` program and never
writes it; a second, one-dimensional launch then writes each carried row's
count once, to `eff` plus the acceptance, with no scratch and no cross-program
reduction. A row is carried when its source and live pages lie in the pool in
a State group a launched layer lists, its record slot is resident and its
window is non-empty; a row no layer carries (graph padding, a masked overlap
row, a row paged only in groups no launched layer lists) keeps its count. A
request whose slot is set in `request_stopped` ended in this forward: the
convolution launch also folds its records, through this round's acceptance,
into its live page (each program of a row takes some heads), and the count
launch clears its count instead of advancing it. Other commits read no record
and write no dense state. Stamp-ring capacity is independent of
retained-history capacity.
Padding rows (a page at or below 0, or one past the checkpoint pool) leave all
pages and counts untouched.

## NVIDIA verification

The registry keeps the same split, one-token and multi-token producer choices:

| Kernel | Selection |
| --- | --- |
| `vllm_triton_kda_recoverssm_verify` | FP16/BF16 activations and generic head geometry |
| `cute_dsl_kda_recoverssm_verify` | SM100/103, BF16 activations, key dimensions aligned to 32 and at most 128, value dimensions aligned to 32, windows up to 16 |
| `cute_dsl_kda_fused_paged_verify_step` | SM100/103, BF16, 128-wide heads/gate rank, one-token windows |
| `cute_dsl_kda_fused_paged_verify_window` | The same fused producers for windows 2–16 and batches up to 16 |

`kda_fused_paged_verify_fuses_producers` resolves the fixed window and padded
batch geometry before the caller prepares projections. Split verification uses
the shared packed-boundary convolution producer and registry-selected core.
All choices store the same persistent FP32 normalized K/decay/U representation.
The CuTe fused kernels require unit-stride physical page indices, query
boundaries and correction channels. Record slot indices may be strided. Other strides use split producers and the stride-aware retained
verify core.

The one-token fused kernel projects the accepted history onto the current
query and key. For 1–16 accepted records, it scales the query and key by
the cumulative decay before multiplying them by the dense base. It adds
each retained U vector weighted by its key's two dot products. Its CTA has
eight warps, so two share an SM: four compute the convolution; the other
four compute the gate, from weights held as packed BF16 pairs, and then
apply the accepted decay factors to the retained keys. The checkpoint and
packed history have independent bulk-copy completion barriers. Strided
history uses its tensor strides. Once the gate and convolution are ready,
four warps project the checkpoint while the other four project the retained
records. Each history warp computes both products for up to four keys. It
combines the even and odd U contributions in FP32. Empty histories use only
the checkpoint products.
Longer histories replay in record order into a private FP32 state tile.
Convolution and count publication use the common retained commit ABI. A
request that ended in the forward has its records folded into its live page by
the commit; if a layer that carries it rejects its commit (stale stamps), that
layer's convolution program raises the row's `request_ends` entry from
`REQUEST_ENDED` to `REQUEST_ENDED_INCOMPLETE` (a compare-and-swap) instead, so
the control plane never publishes a page that lacks accepted tokens.

## Retained preparation

Verify reads each request's live page as its dense base and never writes a page
or a count. Before a verify forward, the caller runs
`resolve_kda_batched_retained_prepare` once for every layer, in one launch,
which reads the count H and never writes it. In each layer, a request whose H
overflows (lies outside [0, C - T], with T the verify's window,
`draft_token_num`) gets the source state with records [0, H) applied in its
live page, in place where the pages did not move; an H past C applies none.
Otherwise a request whose live page differs from its source in the layer's
State group gets the source state copied there, with no record applied: its H
records stay valid on top of the copy, so the verify's `eff = H` holds
whether or not its pages moved. Every layer decides alike from the one shared
H, so no group's move affects another's layers. A request with a window
shorter than T overflows as soon as a full one would. Padding rows keep their
pages. The launch takes the commit's per-layer descriptors, which must share
the page count, the layer-to-group map and `[groups, requests]` source and
live page tables. The fold splits a
request's heads across programs only when few requests and layers would leave
SMs idle. Both vendors share this Triton kernel. Running the prepare twice in
one round would copy the unchanged source over the fold.

The core CuTe window solver keeps main's chunked equations, compensated gate
prefix sums, value tiles and clustered producer sharing. Bounded windows with
`window * abs(lower_bound) <= 48` form pairwise products through factored rows;
other windows sum intermediate gates directly so a large early gate cannot
erase later decay. Runtime counts, arena strides and stamp-ring capacity remain
runtime values in compiled launch interfaces. The window and split cores load
the live page's base tile with one bulk copy and prefetch the accepted records
into L2. The fused window kernel instead bulk-copies each packed record's
key, decay and its tile's U into shared memory on 64-row value tiles, when the
grid holds at most one CTA per SM and the CTA's footprint with the copy region
fits the device's opt-in shared memory per block. Larger grids keep the
prefetch to preserve occupancy, since the region can cost a 64-row tile a CTA
per SM; so do other tile sizes and records that are not packed. Like the
one-token kernel, both cores project the history instead of replaying it:
the base rows are scaled by the records' cumulative decay, each
record's key, weighted by the later decays, becomes a row that a separate pass
multiplies by the window's decayed keys and queries while the base copy lands,
and each value channel adds its records' U weighted by those rows' products.
Requests without records skip all of it.

## Output norm and prefill

`kda_fused_paged_verify` accepts explicit `output_norm: KdaOutputNorm | None`.
The CuTe fused producers apply gated RMSNorm in their epilogue using the BF16
core output and return `[tokens, heads * head_dim]`; other retained kernels
apply the same norm after attention. A missing norm returns ordinary attention
output. The norm's FP32 math and rounding order follow `rmsnorm_gated_sigmoid`.

Prefill keeps the capacity-aware scan, native state layout, prepared CuTe plans,
and cuDNN graph paths. Each layer's retained source is gathered and
reconstructed in FP32 from the rows' counts before its scan. Every layer
reads the same counts, so `kda_reset_retained_length` clears them once, in a
launch ordered after the last KDA layer's gather; it skips padding rows (a
live page at or below 0, or at or past the pool's `page_count`) as the gather
does. Final/checkpoint dense stores use the selected base dtype. Before a retraction stores its image or anything
reads the endpoint it published, `kda_materialize_retained_pages` folds each
victim's records into the live page the scheduler lists, layer by layer, and
`kda_reset_retained_length` then clears the slots' counts once.

Retained numerical tests cover accepted-prefix rounds, boundary materialization,
the fold at a request's end, private/shared pages, stale stamps, graph
metadata refresh, strided arenas, FP32/BF16 base, norm fusion and scalar
compile reuse. AMD uses the same retained
contract through its registered Gluon kernels and the shared prepare.

The `kda_fused_paged_verify` benchmark accepts `history_length`. It defaults to
zero and can cover the accepted prefix up to `max(16, draft_token_num) -
draft_token_num`, so the candidate window fits beside that prefix. Each
measurement uses the same accepted records.
