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

## Index-K plane formats

`dsa_prefill_topk` and `dsa_decode_topk` read the storage of the index-key
plane off its dtype and pass it to selection as the `index_k_format` and
`index_k_layout` traits; a top-k leaf declares the planes it scores and is
never handed another one. One layout per dtype, and the facades never
convert a plane:

| dtype | `index_k_format` | `index_k_layout` | row |
| --- | --- | --- | --- |
| `uint8` | `fp8_scaled` | `packed` | `[slots, head_dim + 4 * head_dim / 128]`: FP8 E4M3 keys followed by one fp32 scale per 128 elements |
| `uint8` | `fp8_scaled` | `page_planar` | any other uint8 shape: per-page planes of keys and scales, the outer page stride possibly padded |
| `bfloat16` | `bf16` | `packed` | `[slots, head_dim]`: the keys as the indexer produced them, no scale plane |

The in-tree DeepGEMM, Triton and Gluon leaves score `fp8_scaled` planes; a
leaf scoring the checkpoint's bf16 keys (an indexer in the RL trainer's
order) registers `index_k_format={"bf16"}`, `index_k_layout={"packed"}` and
the `batch_invariant` and `forced_initial_local` features, so a bf16 plane
selects it and nothing else. A plane of any other dtype is a `TypeError`.

`dsa_prefill_topk` also takes the index keys as rows already in
workspace-row order instead of a plane (the query-context-parallel history
gather over page-sharded caches assembles them): `index_k_fp8` +
`index_k_scale` are the rows of an `fp8_scaled` plane (`[workspace_rows,
head_dim]` uint8 or float8_e4m3fn, `[workspace_rows, head_dim / 128]` fp32),
`index_k_bf16` the rows of a `bf16` one (`[workspace_rows, head_dim]` bf16),
each one row per entry of `kv_workspace_slots` and selecting with that
`index_k_format` and `index_k_layout="packed"`, never together and never with
`index_k_cache`; a call with neither a plane nor rows is a `ValueError`.
Rows additionally REQUIRE the `index_k_workspace_rows` feature
(`dsa.INDEX_K_WORKSPACE_ROWS_FEATURE`): a leaf declares it exactly when its
launcher takes the row keywords for its format (DeepGEMM does, for the FP8
pair; a bf16 leaf declares it and takes the `index_k_bf16` keyword), the
facade hands the keywords to declaring leaves only, and a leaf that only
reads planes -- the portable Triton leaf, the Gluon wrappers -- is never
selected for rows, not by ranking and not by a kernel override (an override
skips traits but not required features). The failure is a
`NoKernelFoundError` at selection; a host whose sharded prefill will hand
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
  stride that is a multiple of 4 elements and a 16-byte aligned base;
  `ends` is int32 `[rows]` and bounds each row (clamped to `[0, width]`).
- `capacity` is the compiled survivor capacity (512, 1024 or 2048) and
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

Algorithm per CTA: the first 8192 entries (the row's in-order tail of up to
4096 entries plus the first pseudo-randomly ordered 512-entry segments) are
selected exactly with an 8-bit radix select over order-preserving keys and
set the running threshold; the remaining segments stream through a three
stage `cp.async.bulk` ring and only entries above the threshold are appended
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

DeepGEMM index scoring is registered for 16, 32, or 64 index heads. The
16-head case pads queries and weights to its native 32-head ABI with zeros;
caller-provided scoring scales and forced initial/local candidate policies
remain unchanged. Other head counts are excluded by kernel traits.

FlashMLA sparse prefill (regular BF16 KV) and sparse decode (packed FP8 KV)
also support `return_lse=True`. Both return natural-log LSE `[tokens, heads]`
and normalize empty rows to zero output / negative-infinite LSE for the shared
DCP reduction. `topk_lens` masks excluded columns before dispatch, and supplied
output buffers retain their identity. Decode transposes the vendor's
`[batch, heads, query]` LSE to token-major order. This does not reinterpret
regular BF16 KV as packed FP8 or change its Triton decode selection.

## Sharded Index-K candidates

`dsa_index_candidates` scores a bounded query tile against rank-local Index-K
for prefill and decode. Inputs are a position-preserving page table (`-1` for
absent pages), request IDs and global causal lengths. Outputs are global logical
offsets and FP32 scores: invalid candidates use `-1`/`-inf`, and forced
initial/local candidates use `+inf`. Cross-rank merging belongs to runtime DCP.

DeepGEMM compacts owned, causally visible pages and maps results back to global
offsets, masking padding and handling partial tails and empty shards. Query
quantization and head padding match the unsharded path; portable Triton provides
the same interface. Query tiling bounds scratch memory, and fixed-shape GPU
metadata supports CUDA graph replay without host reads.
