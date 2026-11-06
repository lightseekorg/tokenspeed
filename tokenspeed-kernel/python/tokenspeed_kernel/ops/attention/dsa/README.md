# Deep sparse attention kernels

The DSA operators separate sparse-history selection from attention:

- `dsa_prefill_topk` and `dsa_decode_topk` score the index cache and produce
  padded global KV slots plus a live length for every query.
- `dsa_prefill` and `dsa_decode` consume those selected slots and read the
  latent KV cache.
- `dsa_plan` owns optional decode planning state.

AMD CDNA4/gfx950 and CDNA5/gfx1250 provide Gluon implementations for full
selected attention. The gfx1250 implementation supports BF16, E4M3, and E5M2
queries; dense or packed KV rows; page size 64; latent ranks 128 and 512; and
optional 64-wide RoPE. A zero RoPE width selects the same kernel path with the
RoPE storage, gathers, and score term compiled out.

GLM-5.3-Flash uses the zero-RoPE specialization with
`qk_nope_head_dim=256`, `kv_lora_rank=512`, and 16 local heads at TP4. Its
KPool selection has a configured width of 2048 and can append three tail
positions, so both prefill and decode accept padded slot widths 2048 through
2051. The live `topk_lens` value determines how many entries participate in
the online softmax; unused entries remain `-1`.

Portable Triton implementations remain the fallback for trait combinations
without a matching native registration.

## FlashInfer Prims-TS FP8 D576

The optional `flashinfer_prims` solution consumes native E4M3 queries and KV
without a BF16 conversion. It supports SM100/SM103, 1–128 local query heads,
latent dimension 512 with a 64-channel QK-only tail, page size 64, top-k
512/1024/2048, and decode widths 1–8. Both prefill and decode return BF16;
optional LSE is FP32 in natural-log units. Empty selections produce zero
output and negative-infinite LSE. Negative slots inside a visible prefix are
masked. Omitting lengths inspects the full selected-slot width.

The pinned FlashInfer 0.7.3 provides the Prims-TS sparse MLA wrapper. Select
`solution="flashinfer_prims"` in the kernel API, or set
`TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSA_PREFILL=flashinfer_prims_dsa_prefill`
and `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSA_DECODE=flashinfer_prims_dsa_decode`
for serving. TRTLLM keeps its higher default priority, and also returns the
LSE that decode context parallelism merges. The indexer is unchanged.

The upstream plan owns PDL scheduling. This adapter requires PDL enabled and
rejects an explicit disable request. The caller prepares Q and the selected
indices. `k_scale` multiplies the logit scale, as in the existing D576 facade;
native MLA value storage stays at unit scale.

Plans bucket request capacity to powers of two. Live rows and lengths remain
runtime inputs. Scratch is separate per CUDA stream. Warm the geometry and
row count on the capture stream before capture, inside the normal runtime
warmup. Use `tokenspeed_kernel.workspace.capture_session(graph)` when capturing
standalone calls. Captured plans live with their graph owner. Graphs sharing
a plan must replay in order; concurrent streams need their own warmed plans.

The adapter tests check both solutions against an FP32 reference. From the
repo root:

```bash
python -m pytest tokenspeed-kernel/test/nvidia/ops/test_dsa_flashinfer_prims.py
```

## Speculative verify width

On NVIDIA the FP8 index-K decode path supports one through eight query rows
per request; the gfx950 Gluon scorers and sparse MLA decode stop at six.
Eight rows cover DSpark's seven draft tokens plus the bonus token. DeepGEMM
scores each row separately. The row at offset `j` can select only the first
`seq_lens[request] - q_len_per_req + 1 + j` cache positions. The FlashInfer
RoPE attention core and portable Triton core consume those selected rows.
They do not attend to later draft positions.

GPU tests compare seven- and eight-row verification with independent
single-row calls. They cover short prefixes, page boundaries, sparse selection,
and CUDA Graph replay.

## DCP local candidate selection

`deep_gemm_dsa_index_candidates` scores compact owned Index-K pages. Its
local top-k uses a length-aware selector with the original compact lengths,
so selection excludes the unused logits tail. Empty shards keep length zero
for selection even though DeepGEMM's scoring scheduler needs a nonzero dummy
length. Selectors are tried in this order:

- On Hopper, the in-tree CuTe DSL DeepSelect selector serves `topk <= 2048`
  on CUDA FP32 matrices with unit column stride, a 16-byte aligned base, and
  rows padded to four elements. The first eager call compiles every cluster
  variant of its survivor capacity before CUDA-graph capture; later batch
  shapes use those compiled variants.
- Otherwise, including on Blackwell, the CUDA `persistent_topk` ragged
  selector serves `topk` 512, 1024 or 2048 when the logits width is at
  least `topk`. It needs the same aligned FP32 rows and reads them in place:
  DeepGEMM's padded logits rows are widened rather than copied. The kernel
  pads short rows with `-1`; gathering drops those and `-inf` candidates.
- Unsupported layouts or top-k sizes, tables narrower than `topk`, and
  installations without either selector retain the existing `torch.topk`
  selection.

Scoring and logits capacity follow the compact page table's width. Decode
keeps the global column count on purpose: page ownership is balanced on a
best-effort basis only, and decode graphs need a static width. That width
still costs more than the DeepGEMM logits allocation. `compact_index_pages`
runs one program per row over a `next_power_of_2(columns)`-lane block with a
full-width cumsum (32768 lanes at the 16386-column 1M decode table), and it
zero-fills `[rows, columns]` int32 pages and positions, which the scorer and
the gather then read. DCP decode plans them once per forward.

A prefill runs eagerly, so it can plan narrower pages instead:
`plan_dsa_index_candidates(columns=)` compacts each row into its first
`columns` owned pages with a bounded kernel that walks the table 1024 columns
at a time (one binary for every table and output width). The DeepGEMM
leaf's logits shrink to `columns * page_size` scores per row, so a logits
budget fits proportionally more rows per tile. `columns` must bound every
row's owned pages; the runtime derives it on the host from the page table
(`max_owned_index_pages`), identically on every rank. A row past it still
stays in bounds: it keeps its first `columns` pages, its length counts only
their tokens, and the plan's `overflow` reports the pages it left out.
`dsa_index_candidates_accepts_pages` tells whether the selected leaf scores
planned pages at all; a leaf that does not compacts the whole table on every
call, so its tiles must keep the table's width.

Length-aware selection masks only the valid prefix with a device-sized grid
(about eight programs per SM) whose programs stride over each row's length
1024 scores at a time; DeepSelect's returned values are reused. The
full-width fallback still initializes the entire logits tail, one 256-score
block per program. Every row offset in these kernels is 64-bit: strides fit
in int32, but past 2048 rows of a 1M-wide table their products do not. The
runtime also keeps one tile below 2^31 logits elements, because the
persistent top-k and the portable scorer offset rows in 32-bit arithmetic.
The mask preserves mandatory
global windows as `+inf`; do not apply the ordinary decode non-finite cleanup
after those markings. Gathering bounds every selected offset before reading
positions or logits, then emits interleaved int32 position/FP32-score-bit views
that runtime can exchange without repacking. Tie order remains unspecified,
and gathering drops candidates scoring `-inf`. Attention-core selection and
cache ownership are unchanged.

Runtime DCP merges every rank's gathered candidates with
`dsa_merge_candidates`, and every rank must select the same ones from the same
records. `triton_dsa_merge_candidates` takes a global `torch.topk`; its gather
kernel processes selected entries in bounded blocks, so the block size does not
restrict `topk`. On NVIDIA, `cuda_dsa_merge_candidates` (topk 512, 1024 or 2048)
finds the k-th largest score with the CUDA persistent-radix `persistent_topk`,
whose atomics pick among tied scores in arrival order, and then keeps every
candidate above that score and the first tied ones, in candidate order.

When a rank needs only its own share of that selection as local cache slots,
`dsa_select_owned_candidates` takes the ranks' gathered FP32 score planes
(`[degree, rows, topk]`, which a declaring `dsa_index_candidates` leaf writes
straight into a collective's send buffer through `scores_out`) and its own
candidate positions. Every rank finds the same k-th largest score and tie
quota from the same planes, keeps its own candidates above the score and its
share of the first tied ones (ranks in order, then candidate order), and
writes `local_page_table[request, position // page_size] * page_size +
position % page_size` in candidate order, padded with -1, with the count per
row. That is bit for bit `cuda_dsa_merge_candidates`' selection compacted to
the rank's slots, without exchanging positions; `triton_dsa_merge_candidates`
selects the same candidates in score order. A score plane reads an invalid
candidate off its score, so `scores_out` holds -inf for every -1 offset.
Only gfx950 registers it, as one launch over the gathered planes:
`dsa_select_owned_candidates_supported` accepts a power-of-two rank count, a
power-of-two topk of at least 512, at most 16384 gathered scores per row and
rank chunks within 32-bit offsets, and refuses other platforms. A KVP decode
whose sizes it refuses merges the candidates instead.

## Materialized Triton row top-k

`triton_topk_from_logits(logits, topk, seq_lens=..., enable_pdl=...)` selects
FP32 rows with descending scores and smaller column ids first on ties.
`seq_lens` is an int32 or int64 vector on the scores' device, clamped per row
to `[0, width]`; explicit `None` selects the full row. Scores and lengths may
be strided. Invalid scores inside the visible prefix still use `-inf` and
produce `-1` output padding. Entries beyond the prefix are never read and may
be uninitialized.

The sorting and radix implementations both bound input reads by these device
lengths. Radix histogram reduction excludes unwritten tiles and selects at
most `min(topk, seq_lens[row])` entries, including empty and short rows.
CUDA Graph replay may change the lengths without changing allocation or launch
capacity. QSA passes its complete compressed-block counts to both CUDA and
Triton selection, so its scorer writes only valid prefixes.

## Persistent CUDA row top-k

`persistent_topk` selects unsorted indices for `k` of 512, 1024, or 2048.
It reads only the visible prefix, applies causal lengths for grouped queries,
and pads short rows with `-1`. Equal scores may select any distinct tied indices.
Its ordered keys rank `-0.0` below `+0.0`; it does not promise lowest-index tie
selection or declare `batch_invariant`. The lowest-index tie contract in
`docs/design/numerics.md` applies to leaves selected with `batch_invariant=True`.
NaNs in the visible prefix are unsupported: value selection is unspecified,
but output indices and writes remain in bounds. The unread suffix may contain
NaNs. Empty or unmatched refinement histograms never reuse stale thresholds.

The register, streaming, and cluster implementations share a coarse FP16
histogram followed by exact FP32 selection. If the threshold bucket exceeds
the 2048-entry candidate buffer, additional FP32 radix passes narrow it before
collection. Refinement skips the leading bits fixed by the coarse bounds and
reuses the existing histogram storage, without increasing shared memory. It stops
when candidates fit or all remaining keys are identical. Truncation is safe only
in the latter case. Ordered integer bounds preserve nearest-even FP16 bucket
boundaries, including negative midpoints, signed zero, and infinities. Candidates
retain their integer keys through tie selection so subnormal scores remain
distinct under CUDA fast math. Cluster ranks reduce every refinement histogram
together before collecting candidates. Equal reduced histograms and counts keep
every rank's refinement rounds and cluster barriers in sync.

## Index-K plane formats

`dsa_prefill_topk` and `dsa_decode_topk` read the storage of the index-key
plane off its dtype and pass it to selection as the `index_k_format` and
`index_k_layout` traits; a top-k leaf declares the planes it scores and
selection never hands it another one. One layout per dtype, and the facades
never convert a plane:

| dtype | `index_k_format` | `index_k_layout` | row |
| --- | --- | --- | --- |
| `uint8` | `fp8_scaled` | `packed` | `[slots, head_dim + 4 * head_dim / 128]`: FP8 E4M3 keys followed by one fp32 scale per 128 elements |
| `uint8` | `fp8_scaled` | `page_planar` | any other uint8 shape: per-page planes of keys and scales, the outer page stride possibly padded |
| `bfloat16` | `bf16` | `packed` | `[slots, head_dim]`: the keys as the indexer produced them, no scale plane |

An index-K plane can be a view into a shared cache arena. Both its key view
and its scale view must preserve the plane's storage offset. The packed FP8
row above is the plane's tensor shape; in memory each page stores the key
bytes of all its slots first, then their FP32 scales.

The in-tree DeepGEMM, Triton, and Gluon leaves score `fp8_scaled` planes. A
leaf scoring the checkpoint's bf16 keys registers `index_k_format={"bf16"}`
and `index_k_layout={"packed"}`, so a bf16 plane selects it and nothing else.
A leaf that must reproduce an RL trainer's selection order also declares the
`batch_invariant` and `forced_initial_local` features; the gfx950 ones below
do not. A plane of any other dtype is a `TypeError`.

On gfx950, `gluon_dsa_decode_topk_bf16_gfx950` and
`gluon_dsa_prefill_topk_bf16_gfx950` score a bf16 plane with BF16 queries
(32 or 64 heads, `head_dim` 128, page size 64, top-k 512/1024/2048). They
declare only the `bf16`/`packed` traits: no `batch_invariant` (their radix
selection does not promise lowest-index ties), no `forced_initial_local`, and
no `index_k_workspace_rows`. They serve the plane path with
`batch_invariant=False`; `plan_dsa_topk` and workspace-row prefill do not
select them.

`dsa_prefill_topk` also takes the index keys as rows already in
workspace-row order instead of a plane (the query-context-parallel history
gather over page-sharded caches assembles them): `index_k_fp8` +
`index_k_scale` are the rows of an `fp8_scaled` plane (`[workspace_rows,
head_dim]` uint8 or float8_e4m3fn, `[workspace_rows, head_dim / 128]` fp32),
`index_k_bf16` the rows of a `bf16` one (`[workspace_rows, head_dim]` bf16),
each one row per entry of `kv_workspace_slots` and selecting with that
`index_k_format` and `index_k_layout="packed"`, never together and never with
`index_k_cache`; a call with neither a plane nor rows is a `ValueError`.
Rows additionally require the `index_k_workspace_rows` feature
(`dsa.INDEX_K_WORKSPACE_ROWS_FEATURE`): a leaf declares it exactly when its
launcher takes the row keywords for its format (DeepGEMM does, for the FP8
pair; a bf16 leaf declares it and takes the `index_k_bf16` keyword), the
facade hands the keywords to declaring leaves only, and a leaf that only
reads planes -- the portable Triton leaf, the Gluon wrappers -- is never
selected for rows, not by ranking and not by a kernel override (an override
skips traits but not required features). The failure is a
`NoKernelFoundError` at selection; a host whose sharded prefill hands
rows probes that selection at construction with
`dsa.select_dsa_prefill_topk_for_rows(index_k_format=, ...)`, which makes a
platform without a declaring leaf a startup error.

A `dsa_decode_topk` leaf bounds every query row itself: row `j` of a request
scored with `q_len_per_req` rows (spec verify, a multi-depth draft's k-row
window) selects over the first `seq_lens[req] - (q_len_per_req - 1) + j`
positions, derived from `seq_lens`. The `seq_lens_2d` rows the facade hands
every leaf (`[tokens, 1]`, each carrying the request's full length) are the
scoring extent the `plan` was built from, not per-row bounds; a leaf that read
them as bounds would let a verify or draft row select its window's later
rows, and a sparse core that trusts the selection for causality (`kv_seq_lens`
is optional on `dsa_decode`) would attend them.

`candidate_lens_cpu` (the CPU mirror of each prefill token's candidate count)
goes to every selected leaf registered with the `candidate_lens_cpu` feature
(`dsa.CANDIDATE_LENS_CPU_FEATURE`) and to no other: a leaf that can size its
launches from it declares the feature alongside the keyword, and the facade
reads the registration rather than probing call signatures, so a
`*args, **kwargs` wrapper never receives a keyword its launcher cannot take.

## Slot order of the sparse cores

`dsa_decode` and `dsa_prefill` take a required `slot_order` in
`SLOT_ORDERS = ("selection", "sorted")`, passed to selection as the
`slot_order` trait: `selection` reduces a token's selected slots in the order
the top-k leaf emitted them and is what every core does by default (a core
need not declare the trait); `sorted` reduces them in ascending slot order,
so the reduction is batch-invariant whenever the selected set is, and is
served only by cores declaring `slot_order={"sorted", ...}`, which receive
the choice as the `slot_order` keyword. Asking a silent core for `sorted`
is a `ValueError`, not a silent fallback.

## Row top-k selection in CuTe DSL (`_cute_dsl/deep_select.py`)

`deepselect_topk(scores, ends, topk, capacity=..., cluster_size=...)` selects
the `topk` largest entries of every FP32 row and returns their unsorted
column indices and values. It is an in-tree CuTe DSL rendition of the
DeepSeek DeepSelect algorithm for Hopper, where the upstream package ships no
cubin; the V4.1 indexer uses it on sm90 for both row selection and
block-maxima selection.

Contract:

- `scores` is CUDA FP32 `[rows, width]` with unit column stride, a row
  stride that is a multiple of 4 elements, and a 16-byte aligned base;
  `ends` is int32 `[rows]` and bounds each row (clamped to `[0, width]`).
- `capacity` is the compiled survivor capacity (512, 1024, or 2048) and
  bounds `topk`; `capacity_for` gives the smallest one that serves a `topk`.
- A row with `ends[r] <= topk` yields `0 .. ends[r]-1` then `-1` slots
  scoring `-inf`. Otherwise every slot is a distinct index below `ends[r]`.
  Ties at the k-th value break arbitrarily but deterministically, so `-inf`
  masking may be selected when fewer than `topk` finite entries exist;
  consumers filter by value. NaN entries are never selected and leave `-1`.
- `cluster_size` in `{1, 2, 4, 8}` splits a row across a thread-block
  cluster (non-leader CTAs ship their survivors to the leader over
  distributed shared memory). `choose_cluster_size` picks it: rows of at
  least 32K entries, and only while every cluster stays resident (about 60%
  of the SMs for 4- and 8-CTA clusters, 80% for pairs). Every configuration
  a call may pick must be compiled with `warmup` before CUDA-graph capture.

Algorithm per CTA: an 8-bit radix select over order-preserving keys selects
the first 8192 entries (the row's in-order tail of up to 4096 entries plus
the first pseudo-randomly ordered 512-entry segments) exactly and sets the
running threshold; the remaining segments stream through a three-stage
`cp.async.bulk` ring, and the CTA appends only entries above the threshold
as `(index, value)` pairs; once 4032 candidates accumulate, and once more at
the end, a radix select over survivors plus candidates re-selects `k` pairs
and raises the threshold. Per-pass histograms send non-matching keys to a
sink slot so the passes stay branch-free; one warp locates the pivot bucket
while the others wait, since the search is a latency chain.

The portable Triton attention implementations support `return_lse=True` for
both dense and packed latent caches. They return `(output, lse)`, with FP32
natural-log LSE shaped `[tokens, heads]`; all-invalid rows produce zero output
and negative-infinite LSE. This permits exact softmax-weighted merging of
context-partitioned sparse attention. Supplying `out` preserves the supplied
output buffer even when returning LSE. Omitting `topk_lens` uses the full padded
slot width, with negative slots still excluded.

DeepGEMM index scoring registers for 16, 32, or 64 index heads. The
16-head case pads queries and weights to its native 32-head ABI with zeros;
caller-provided scoring scales and forced initial/local candidate policies
remain unchanged. Kernel traits exclude other head counts.

DeepGEMM's FP8 logits kernels take no query scale, so
`quantize_index_query_weights` quantizes each (token, head) query to E4M3
with the 1x128 token-group quantizer's arithmetic and folds that scale and
`softmax_scale` into the head weights in one launch, bit-exact with NVIDIA's
`quantize_fp8` followed by `combine_topk_weights`. Before the decode top-k,
`scrub_nonfinite_logits` reads the per-request lengths and derives each
verify row's causal candidate count itself.

FlashMLA sparse prefill (regular BF16 KV) and sparse decode (packed FP8 KV)
also support `return_lse=True`. Both return natural-log LSE `[tokens, heads]`
and normalize empty rows to zero output / negative-infinite LSE for the shared
DCP reduction. `topk_lens` masks excluded columns before dispatch, and supplied
output buffers retain their identity. Decode transposes the vendor's
`[batch, heads, query]` LSE to token-major order. This does not reinterpret
regular BF16 KV as packed FP8 or change its Triton decode selection.

## Runtime backend selection

The runtime selects DSA with `--attention-backend dsa`, or automatically from
the DSA architecture. The same name resolves to `AscendDSABackend` on Ascend
and `DSABackend` on GPU platforms. The Ascend implementation owns its device
metadata and operator adaptation; model projections and indexer selection
parameters remain model-owned. Its current constraints are BF16 KV and index
keys, 128-token kernel pages, target verification of at most eight tokens,
and no draft-model attention. Target verification can still consume draft
candidates. The Ascend backend checks these constraints.

## Sharded Index-K candidates

`dsa_index_candidates` scores a bounded query tile against rank-local Index-K
for prefill and decode. Inputs are a position-preserving page table (`-1` for
absent pages), request IDs, and global causal lengths. Outputs are global logical
offsets and FP32 scores: invalid candidates use `-1`/`-inf`, and forced
initial/local candidates use `+inf`. Cross-rank merging belongs to runtime DCP.
Runtime setup validates a positive power-of-two candidate count when the KVP
group has multiple ranks, using the cache-parallel topology (`kvp_group` /
`kvp_size`). Single-rank selection keeps its existing top-k choices.

DeepGEMM compacts owned, causally visible pages and maps results back to global
offsets, masking padding and handling partial tails and empty shards. Query
quantization and head padding match the unsharded path; portable Triton provides
the same interface. Query tiling bounds scratch memory, and fixed-shape GPU
metadata supports CUDA graph replay without host reads. Runtime
(`select_dsa_topk`) bounds the candidate exchange by the same budget:
consecutive tiles share one all-gather and merge while the gathered records
and merge scores, `degree * topk * 12` bytes per row, fit it, so a long
prefill chunk never exchanges every row at once. Without a budget (DCP
decode by default) every row joins one exchange.

A KVP prefill of at least 1024 rows
merges each row on one rank instead of on every rank. An exchange's rows are
padded with invalid candidates to equal slices, one per rank; an all-to-all
sends each slice's candidates to its owner, which runs the same merge on
them, and two all-gathers return every row's positions and counts. A row's
merge input is unchanged, so its selection is too, up to the order of equal
scores. This exchange holds `topk * 24` bytes per row whatever the degree.
DCP decode keeps the all-gather. Owned-width tiles (see above) are held to
the rows one exchange of the same kind fits, so with both the row-owner merge
and owned-width tiles a 16K chunk
after 200K cached tokens at KVP 8 and a 512 MiB budget scores 4 tiles in 2
exchanges, against 27 tiles in 7 exchanges at the table's width.

Rows that several calls score against the same table, such as every DCP
decode indexer layer of one forward, can share one compaction.
`plan_dsa_index_candidates` selects the leaf `dsa_index_candidates` would
select and compacts the table once when that leaf declares
`INDEX_CANDIDATE_PAGES_FEATURE` (`deep_gemm_dsa_index_candidates` does).
For any other leaf, or for zero rows, it returns `None`. The query, weights
and cache only supply the selection signature and traits. The result is a
`DSAIndexCandidatePages`; `rows(start, end)` views a query tile's rows.
`dsa_index_candidates(..., pages=...)` hands them only to declaring leaves.
It refuses pages planned for another leaf or for a different row or column
count. Planned pages are fixed-shape device tensors derived from the table
and causal lengths, so a captured plan refreshes them on replay. The
DeepGEMM scoring schedule is still built per call.

The portable selected-slot kernel groups query heads into tensor-core tiles
and gathers each latent KV tile for both scoring and value accumulation.
Small grids split the selected keys across programs and combine their FP32
partial outputs using base-2 log-sum-exp weights. The public `return_lse`
result remains FP32 in natural-log units for both split and unsplit launches.
Empty selections produce zero output and negative-infinity LSE. Negative slots
inside the live prefix are ignored; live lengths are clamped to the slot width.
Packed FP8 keys retain a BF16 residual when scoring so dequantization rounding
does not reduce LSE accuracy. FP16 and FP32 dense keys retain the same residual.
Value tiles and probabilities use BF16 with FP32 accumulation.

Under DCP, runtime compacts every selection row before the sparse core.
`compact_owned_slots` (`ops/kvcache/triton_cache_placement.py`) moves the
rank's owned live slots, translated to local slots, to the front of the row
in selection order, fills the rest with `-1`, and returns the owned count,
which runtime passes as `topk_lens`. The portable kernel splits only
`[0, topk_lens)` across its programs, so each rank visits about
`topk / kvp` entries rather than the whole width; masking non-owned slots in
place saved KV reads but not this loop. FlashMLA sparse prefill (BF16 KV)
masks columns at or past `topk_lens` and also passes them as FlashMLA's
`topk_length`, whose kernels stop each row after
`ceil(topk_length / B_TOPK)` blocks, so its work shrinks the same way. Each
rank receives the same owned set in the same order as
before, so `slot_order="sorted"` stays sorted, an empty rank still returns
zero output with negative-infinity LSE, and the cross-rank LSE merge is
unchanged. Only the in-rank split tiling of the portable kernel changes, so
partials can differ from the masked layout at rounding level. A rank's work
follows its owned count, so a row whose selection concentrates on one rank
is bounded by that rank. Compaction is one program per row over the static
width; its grid and output shapes are fixed, so CUDA graphs replay it.

The Triton FP8 indexer skips Q/K scoring for tiles with no causally visible,
owned entries. Empty tiles still write negative-infinite logits, including
when lengths or ownership change during graph replay. Partially valid tiles
keep the per-entry mask and global forced-window policy. This avoids redundant
arithmetic without changing the launch grid, logits capacity, or Top-K scan.

### Partial records written by the core

The head-sharded KVP prefill gathers every rank's query heads, attends the
rank's pages with all of them, and sends each head's output and LSE to the
rank owning the head. Through `dsa_prefill` that takes a relayout of the
gathered heads to `[tokens, ranks * heads]` and a repack of the partials by
`dispatch_dcp_attention_partials` (`dsv4/_triton/dcp.py`).
`dsa_prefill_dcp_records` skips both. It reads the query rank-major,
`[ranks, tokens, heads, dim]`, as an all-gather lands it, and returns the
int32 records `[ranks, tokens, heads, kv_lora_rank / 2 + 1]` that the
all-to-all sends. Each record holds BF16 output pairs, low half first,
followed by the natural-log FP32 LSE bits. The records are bit-identical to
the repacked output of Triton `dsa_prefill(..., return_lse=True)`, including
empty rows (zero output, `-inf`) and NaN.

The op has its own name, `("attention", "dsa_prefill_dcp_records")`, and only
the Triton leaf registers it. A `dsa_prefill` registration would make it a
candidate for ordinary calls. The leaf runs `_dsa_dense_kv_kernel`'s
single-split program over dense KV, with the same jitted body
(`_dsa_dense_kv_attend`) and launch configuration. The token count is a
runtime argument (`do_not_specialize`). `dsa_prefill_dcp_records_supported`
accepts a size only when `dsa_prefill` would not split the keys either: the
launch needs at least about one program per SM. Any other size raises.
Callers check this from host sizes before they gather the query.

`dsa_decode_dcp_records` is the decode counterpart, with the same layouts and
bit-identity to `dispatch_dcp_attention_partials` over
`dsa_decode(..., return_lse=True)`. Only gfx950's dense matrix core
(`gluon_dsa_decode_gfx950`) registers it. Its key-split programs read the
rank-major query, and its split reduction stores each head's output and LSE
straight into the head owner's record. The kernels are the ones `dsa_decode`
runs, with the heads per rank as a compile-time constant: 0 compiles the
token-major indexing unchanged, and a model's local head count gives the
records layout one more binary. The rank stride stays a specialized runtime
argument, because the query's direct-to-LDS copy needs its 16-element
alignment. `dsa_decode_dcp_records_supported` accepts a
size only when that core splits the keys, since only the reduction writes
records. The core splits while one program per row and 16-head block stays
under two waves (512 programs on 256 CUs), so up to 85 rows at 96 heads. A
BF16 query with RoPE never splits, and other platforms are refused.

### Persistent CUDA top-k dependencies

`ragged_decode_topk` needs no workspace and requires an explicit `enable_pdl`.
With PDL, selection waits before reading lengths or scores, including empty and
short rows. Cluster and single-CTA launches both honor the PDL flag.

For non-cluster PDL grids with no more rows than SMs, the launcher selects a
one-block register budget to reduce packing of waiting CTAs. Larger grids keep
the two-block budget. Launch bounds guide register allocation; they are not a
hard limit on resident CTAs. Both variants use the same radix algorithm.

### FlashMLA query padding lifetime

Sparse FlashMLA pads small query head counts to the kernel's supported width.
Prefill and decode padding tensors are call-local: PyTorch's CUDA allocator
reuses released storage with stream ordering, and allocations made during graph
capture belong to the graph's private pool. Do not cache these writable tensors
by shape: variable prefill lengths retain every allocation, and concurrent calls
or independent graphs can overwrite each other's queries. Cross-stream users
must still follow PyTorch's normal tensor synchronization/lifetime rules.
Padding remains an exact copy of the original heads plus zero-filled extra
heads; this changes neither arithmetic nor the supported numerics envelopes.

### FlashInfer sparse query head counts

TRTLLM-GEN does not accept 65–127 query heads. The FlashInfer sparse adapter
pads these queries to 128 heads and returns only the original heads. The
selected KV slots, scales, and FP8/BF16 input values stay unchanged. Supplied
output buffers retain their identity and shape. Padding tensors are call-local
and follow the same stream and graph lifetime rules described above.
An output buffer must use BF16 and reside on the query device, as required by
the underlying FlashInfer operation.

## Ascend fused decode resources

The fused Prolog returns query, index-query and score, or writes them through
an explicit caller-output form. KV and Index-K caches remain caller-owned
persistent state. QA is private operator workspace; auxiliary debug copies are
not part of the production interface. RoPE accepts real positions and cos/sin;
NoPE takes neither input.

`DSAResources` retains the communication window and producer/statistics views.
LSE merge takes those real data tensors and the communication group. Peer
window addresses and shape offsets are resolved inside the operator; the
framework never receives a pointer table. Destroy referencing Graphs and drain
forwards before releasing the resources or their dedicated process group.
