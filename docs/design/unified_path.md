# The unified decode path

This document records the invariants of the decode execution path after the
persistent-batch unification: eager decode and CUDA-graph decode share one
metadata path, one padding contract, one sampling route and one output-buffer
discipline. A deviation from the rules here is a bug unless this document is
updated in the same change.

## The problem this solves

Before unification every attention backend carried three decode-metadata
implementations: an eager arm inside `init_forward_metadata` that built fresh
tensors per step, a capture arm that allocated persistent buffers, and a
replay arm that refreshed them in place. Twelve backends times two live decode
paths drifted continuously — replay grew clamps, padding scrubs and PD guards
the eager arm lacked (and vice versa), and graph-only bugs surfaced only in
end-to-end runs. A second dark path hid behind the capture ladder: a decode
batch above `max_cudagraph_capture_size` fell back to the eager arm, a code
path nothing exercised routinely.

## Invariants

### One decode metadata path

`AttentionBackend.refresh_decode_metadata(bs, actual_bs, req_pool_indices,
seq_lens, *, forward_mode, block_tables, num_extends, for_graph_replay,
**cache_kwargs)` is the ONLY way decode metadata is prepared:

* **capture** (`init_forward_metadata_capture_cuda_graph`) is INHERITED: the
  base default runs the idle-refresh arm (`actual_bs=0`,
  `for_graph_replay=True`) against the runner-seeded seq_lens and the
  runner's placeholder tables (`placeholder_block_tables`) — never live
  tables. Only a genuine capture-only asymmetry overrides it (see "Capture
  is inherited");
* **replay** = refresh (`for_graph_replay=True`) + `graph.replay()`;
* **eager decode** = refresh (`for_graph_replay=False`) + the same forward
  Python the graph recorded.

`init_forward_metadata` serves extend/mixed (and idle warmup) ONLY; a pure
DECODE call raises. There is deliberately no fresh-allocation decode arm
anywhere. `init_forward_metadata_replay_cuda_graph` no longer exists.

Its extend inputs are one required, keyword-only bundle on every node —
runner-facing (`backends/base.py`: router, V4, V4.1, Mamba/KDA, composites)
and leaf (`backends/paged/base.py`) alike: `extend_seq_lens`,
`extend_seq_lens_cpu`, `extend_prefix_lens`, `extend_prefix_lens_cpu` are
plain `torch.Tensor` (`[>= num_extends]` entries; empty, never `None`, when
there are no extend requests) and `extend_with_prefix` is a plain `bool`, and `query_shard` is the forward's
`QueryShardPlan` (plain host integers: which rows of the span this rank
computes under query context parallelism) or `None` when every rank computes
every row.
Runner-facing nodes additionally take two host-only facts the scheduler
knows and only V4.1 plans from: `extend_replay_lens_cpu` (how many leading
rows of each extend re-feed already-cached positions — bounded replay,
`docs/design/scheduler.md`) and `extend_prompt_lens_cpu` (the whole prompt
length, so the backend can tell a prompt-completing chunk from an open one).
Leaves never see them: the attention prologue writes every input row
unconditionally, so the router and every other runner-facing node call
`reject_bounded_replay` and fail loud on a non-zero replay instead of
rewriting rows the prefix hit already shares. Likewise every node without a
gathered-history extend arm calls `reject_query_shard`; the router forwards
the shard to its leaves together with `page_table_cpu`, the host mirror of
the extend rows of each leaf's kernel page table (built only for a sharded
forward, `None` otherwise), and the one leaf that attends a shard — GPU DSA —
counts page ownership from it on the host. No default values: the runner
passes the `[:num_extends]` slices of its input buffers on every call (the
idle replay passes the empty `[:0]` slices), so a node that reads a field
can never see a silently-defaulted one. This is deliberate — a `= False`
default once hid `extend_with_prefix` being swallowed by a composite's
`**kwargs`, and FlashMLA planned a ragged prefill for a prefix-cached batch.

### Buffer sizing: the ladder is a performance subset, never a capacity limit

`ForwardStepRunner` distinguishes `max_capture_bs` (top of the capture ladder,
bounded by `max_cudagraph_capture_size`) from `max_decode_bs`
(`max_num_seqs // dp_size`, floored at `max_capture_bs`). Persistent decode
buffers are sized by `max_decode_bs` — `init_cuda_graph_state` runs
unconditionally at wrapper construction, `enforce_eager` included. A decode
above the ladder runs the same refresh with no graph; it is a first-class
path, not a fallback.

### Rebinding a cache pool

`set_cache_pool` may run more than once on the same backend tree: a memory
probe binds a small pool, captures into a throwaway graph pool, then binds
the real pool. The contract is that a rebound backend is indistinguishable
from one first bound to that pool:

* Every node first answers `validate_cache_pool` for the whole subtree, and
  only then do the children and the node publish, without a second
  validation, so a rejected rebind moves nothing (a router's leaves exist
  only from its first bind on, so the first bind builds and binds them
  inside its own publish); `set_cache_pool` is that sequence, shared by
  every node through `CachePoolBinding` and never overridden; a node does
  its own work in `_publish_cache_pool` (`set_kv_pool` on the state backends
  is a retained alias). Atomicity covers rejections only: a failure inside a
  node's own binding work propagates, and the caller rebuilds the tree. A
  node rejects a pool that changes the geometry it owns: the router its
  group geometry (granularities, families, retentions and the row layout the
  leaves' kernels read), the state backends the state group ids, checkpoint
  grain and the state layers' ids and shapes, DeepSeek V4 the group ids and
  row geometry, Inkling the ShortConv geometry. Page counts and transfer
  policy may change. Paged leaves own kernel geometry only; the router
  validates group geometry for them.
  Qwen4-Exp's PLE and QSA indexer children validate their local fields during
  this same pass. Publishing the same pool again preserves their verify
  buffers; a different pool drops them (PLE's commit pointer tables, QSA's
  verify state); the router's geometry check covers the indexer's tables.
* For nodes accepting pool replacement, binding drops every pool-derived latch:
  pointer tables, scratch and views,
  per-forward metadata, the paged leaves' graph buffers, Inkling's ShortConv
  ring and pending remote restores, and side-state verify caches. The state
  backends keep their pool-independent index buffers, so a same-geometry
  replacement stays usable without re-initialisation of those buffers (a
  router in the same tree still needs `init_cuda_graph_state` before any
  metadata call); the caller still runs `configure_runtime` (with the new
  pool's specs and page counts), `init_cuda_graph_state`,
  `init_prefill_graph_state` and `preallocate_verify_workspace` again after
  a rebind, as after a first bind. A probe pool must still hold `max_bs`
  state rows: the KDA raw-gate verify scratch is the bound pool's own conv
  slab. The backend tree covers only itself: the executor's own pool
  references (`token_to_kv_pool`, its cache runtime contract, the drafter's
  pool) and the layer-to-group stamps `bind_cache_groups` writes on the
  model are the caller's to re-publish, as are the graph owners' own pool
  references and the placeholder tables the decode runner sizes from the
  arena. Of the sequence above only `init_prefill_graph_state` runs inside
  `capture_graphs()`: `configure_runtime` and `init_cuda_graph_state` ran
  before the executor was returned, so `set_cache_pool` re-publishes them.
  `preallocate_verify_workspace` is the factory's, re-issued when the rebind
  rebuilds through it -- a rebind that skipped the factory would strand it.
* A rebind is an operation between the probe's `capture_graphs()` and the
  serving one, owned by `reserve_and_rebind`; nothing in the backend tree
  guards against a rebind at another time.
  That orchestrator releases both graph owners' captures first (the captured
  graphs record the buffers a publish drops, and eager kernels cache
  pointers they allocated inside a capture, such as flashinfer's trtllm-gen
  MoE runner and Qwen4-Exp's uniform index bundles), unfreezes the device-
  global workspace pool the executor froze before capturing, rebinds the
  trees, re-runs `bind_cache_groups` and the initialisation sequence above,
  freezes the workspace again and captures again.
* The KV budget reserves what the graphs will cost: a probe binds the
  smallest arena the family can run on and captures a few entries of each
  ladder -- the widest three, then one a third and one two thirds of the
  way down -- with a driver-memory delta around each capture. The widest
  samples form a window priced at its positive bytes, plus one granule the
  window may hide (readings move in 2 MiB, from the driver's graph memory
  and the allocator's segments alike), over every marginal; each sample
  further down anchors its width at its reading plus that granule, capped at
  the window's rate when the reading is within three granules of it (one
  reading is lumpy) and at the reading less those three granules further
  above (a dearer entry stays dearer, and a granule more in any reading never
  lowers the reserve), and a skipped entry is priced on the line between the
  anchors around its width, or at the narrowest anchor below it. Every ladder
  is sampled and priced the same
  way, whatever shape its cost takes down the ladder: flat, falling with the
  entry's width, or lumpy. That is not a bound: a cost that drops between
  two anchors is priced short over that stretch.
  The result is reduced across ranks with MAX. The orchestrator
  releases the probe's graphs and collects the cycles they sit in, then
  rebuilds on the memory profile the probe build took -- where a boot without
  a reserve takes it -- minus the projection. The reserve covers the bytes
  inside the capture windows as projected -- what a boot without a probe
  captures there, one-time bytes the first captures take included; the
  probe releases them and the serving capture pays them again. The
  utilization headroom covers everything else: activations, fragmentation,
  the warmups and workspaces a capture allocates around its windows, and any
  shortfall of the projection, as it covers every graph on a boot without a
  reserve. Profiling again after the probe would charge the cache a second
  time for what tuning and the probe left allocated. The deltas read the
  whole device, so the probe assumes no other process allocates on it during
  startup. Not covered: a ladder every one of whose sampled marginals was
  served from slack, which is priced at nothing and says so in the
  log. The EPD receive pool, which a multimodal prefill node allocates after
  its cache is sized, is left out of the profile instead, by each rank before
  the cross-rank minimum.

### Padding contract

`bs` is the request count being prepared (the padded graph batch under
replay); `actual_bs` is the live-request count. Requests in `[actual_bs, bs)`
are padding and must resolve to the null page 0 / dummy slot so they never
touch a live request's cache. Eager passes `bs == actual_bs` (unpadded — no
wasted FLOPs);
`actual_bs == 0` is the idle replay. Eager idle bypasses the wrapper entirely
(`execute_idle_forward` calls `model_runner.forward(IDLE)` directly). With a
drafter, the eager idle then asks the drafter for its round
(`idle_forward_global_num_tokens`): one list of per-rank token counts per
draft forward the active ranks run, and runs one IDLE draft forward per
entry over an empty window, entry `i` with `spec_step_idx=i` and that
entry's counts — the row shape the active ranks' step `i` runs (the Eagle
chain: the target's rows at step 0, one row per request after; multi-depth
MTP: the target's rows at every depth; block drafters: one forward). The
executor never derives a drafter's step count or shape itself.

Padding rows still route through the MoE layers, and under attention DP every
rank's filler is interleaved with the real rows in the all-gathered MoE
input. The expert load counters (`--expert-distribution-recorder-mode stat`)
therefore read a device-side live-row mask (`ExpertLoadRowMask`, one
`[max_rows]` bool buffer reserved before the first forward) that the graph
owners mark from the host's padded/live counts before a padded replay — the
decode graph in `ForwardStepRunner`, the bucket in `PrefillGraph.replay` —
and clear right after it, on the same stream. Eager forwards are unpadded
and read the all-True mask; the router counts a route iff its row is marked.

### Pointer-stable per-bs views from one builder

Per-bs metadata objects (each leaf's `_decode_views_by_bs[bs]`, the router's
`decode_write_locations` views) are views over the persistent buffers, built
by a single per-bs builder shared by capture and refresh, cached per bs. A bs
never captured (above-ladder decode, enforce-eager) builds its views lazily
on first refresh — no new storage, one-time cost. Views must be
pointer-stable: a captured graph holds their addresses forever.

Helpers that memoize tensors created inside capture must not return those
tensors to eager callers. Keeping a Python reference preserves the allocation,
but an earlier graph sharing the same private pool can overwrite its contents
on replay. PLE's uniform index bundles are reused during capture only; the
eager n-gram kernel writes uniform request indices alongside hash IDs, while
ragged batches construct their indices outside the capture pool.

GDN verify shares memoized scratch seed indices (`i * (T + 1)`) between conv
and recurrent reads in eager and captured forwards. FlashInfer FP32 MTP may
use uninitialized output and a placeholder for a disabled intermediate cache:
live rows are fully written, while negative padding rows skip state access
and leave output undefined. Consumers must ignore padded output; enabled
intermediate caches always require real storage.

State backends refresh every state group's decode pages in one prep-tape
launch, up to eight groups; the tape loops over rows, so it covers every
captured batch size. Target verify writes each group's committed-state pages
straight into the captured `state_in` buffers. The commit enqueued after the
replay reads those buffers before the next refresh rewrites them. Verify keeps
`state_out` at `pad_slot_id` and refills it only after a decode refresh has
written live pages into the same per-bs buffer.

After verification, GDN, KDA and PLE resolve the accepted checkpoint with
`commit_state_pages`, once per state group and only for live requests. It
clamps acceptance, computes checkpoint slots and gathers destination pages in
one launch. `state_verify_commit_rows` maps those pages to layers and computes
`request * (verify_width + 1) + accepted` for batched copies and ReplaySSM.
Its inputs and outputs are contiguous: pages are `[groups, batch_size]` for
grouped state or `[batch_size]` for PLE, so no explicit strides are needed.
Non-positive pages resolve to row -1 so copies skip the null page. Keep this
arithmetic in the kernels, without eager casts, gathers, `index_select` or
`repeat`. GDN and KDA share their backend page resolver; PLE uses its own
group's page vector and copies the shared context once and local convolution
states in one batched launch.

GDN (prefill, decode and verify), QSA and gated residual kernels follow
`pdl_enabled()`, passed explicitly to QSA indexing kernels. Waits precede
producer-owned reads and outgoing triggers. A trigger permits successor
setup, never publishes results; each kernel may delay it for performance.
Streaming top-k, for example, avoids delaying scoring waves with waiting
merge CTAs. Graphs retain their captured PDL setting; recapture to change it.

`fused_gate_sigmoid_mul_add`, `sigmoid_mul`, `silu_and_mul`, `swiglu_oai`,
`situ_and_mul`, `add3`, and split AttnRes launchers read `pdl_enabled()`
themselves. They use that same value for `ENABLE_PDL` and `launch_pdl`; model
layers do not pass the platform PDL setting through their calls.

AttnRes partial kernels may trigger their successors before writing partial
scratch. `attnres_combine` may preload only weights known to be independent of
its predecessor; it waits before loading the prefix and the partial scratch
(`m`, `s`, `acc`). A PDL trigger permits early launch but does not publish
stores, and a later wait cannot repair values already loaded into registers.

Gated RMSNorm preloads weights only with `weights_independent`; a contiguous
copy disables this preload. At RSAG-to-AR boundaries the next combine-norm
preloads the all-gathered residual before its wait, so that collective must
not trigger early. FlashInfer adapters preserve the upstream CuTe body and
keep PDL compilation caches separate.

QSA logits scoring uses the same paged kernel for every query layout. A batch
whose request lengths are all available on the host may shorten its compressed
block-table view to the maximum prefix-plus-query length, rounded to the cache
group's logical block granularity. This changes neither the allocation nor the
page mapping. Mixed batches with device-only decode lengths, and persistent
decode views, retain the capacity bound; decode graph shapes stay fixed.
Uniform query runs may share a K tile, but groups must never cross requests.
A single-request forward is uniform regardless of its forward mode. Long runs
use larger query groups; ragged layouts retain independent rows. A score tile
beyond all of its queries' complete-block frontiers must write `-inf` without
reading K or executing its dot, including padded graph requests.

### `for_graph_replay` is for graph-mechanics asymmetries only

`for_graph_replay=True` means a graph is in play — live replay AND the base
default capture (which runs the idle-refresh arm). Two sanctioned branches
on it exist:

* FlashMLA's tile schedule: flash_mla freezes its schedule on the first
  kernel call against a `FlashMLASchedMeta` (a request that has since
  crossed a page boundary loses its newest page), so the object is bound to
  one seq_lens value: eager refresh and every drafter seq_lens edit
  (`advance_draft_forward_metadata`, `fill_block_decode_seq_lens`) bind a
  fresh one, while a replay refresh leaves the slot alone — the captured
  graph re-runs the recorded schedule-builds, one per edit, against the live
  seq_lens buffer. The object lives on the backend, not on the decode views.
* DFLASH block-arm seeding (`not for_graph_replay or actual_bs == 0`): the
  drafter's recorded `fill_block_decode_seq_lens` rewrites the block-end
  lengths inside every replay, so only eager steps and the capture-time
  seeding fill them from Python.

Do not branch on this flag for anything a shared in-place refresh can
express.

### Capture is inherited

`init_forward_metadata_capture_cuda_graph` has a base default — run the
idle-refresh arm (`actual_bs=0`, `for_graph_replay=True`) over the same
persistent buffers replay refreshes — at both tiers: `AttentionBackend`
(runner-facing; the router's version idle-fills its table stacks, republishes
the decode write-location views, then runs each leaf's capture hook) and
`PagedAttentionBackend` (kernel-facing leaves). That default IS the capture
for every backend except a closed list of sanctioned overrides, each tied to
something the idle refresh cannot express:

* **FlashMLA** (leaf): installs the keepalive tile-schedule object whose
  schedule-build the graph records (flash_mla freezes its schedule on the
  first kernel call against a sched-meta);
* **DeepseekV4**: the packed `tokens_per_req` row machinery and its bespoke
  multi-group metadata build;
* **Mamba** (`MambaAttnBackend`): the warmup kernels need the arange
  query-start-loc, which the idle refresh deliberately zeroes;
* **Inkling**: conv-state seeding (paged conv reads `pos = seq_len - 1`, so
  capture must seed real lengths);
* **HybridLinearAttnBackend / Qwen4ExpBackend / MSAHybrid**: pure fan-out to
  their children so the real captures above are reached.

A new backend implements `refresh_decode_metadata` and inherits both
`init_cuda_graph_state` (the page-table / cache-seqlens pair, sized by
`block_decode_expansion`; extend it for extra persistent state) and
capture; a new override must name its kernel-imposed asymmetry here. Leaf
capture/refresh signatures are pinned by
`test_unified_decode_path.py::CaptureSignatureConformanceTest`.

### Graded CUDA-graph support

A backend's static graph capability is a class attribute,
`cuda_graph_support: CudaGraphSupport(decode_graph, prefill_graph)`, never a
scattered executor-side arch check. `ModelExecutor.__init__` AND-composes it
over the target and draft `child_backends()` trees once
(`resolve_cuda_graph_support`), logs every culprit class, and downgrades the
two graph subsystems (`ForwardStepRunner.disable`, `PrefillGraph.disable`).
`DSABackend` and Qwen4-Exp's PLE/indexer consumers disable the prefill graph
(rationale comments live on those classes). Qwen4-Exp's root composes its
actual children, so these restrictions also apply when there is no GDN leaf.

Rules: declarations are static "never works" facts — a runtime prefill
capture failure is FATAL (no silent eager degrade: a family that cannot
capture must declare it, or the boot dies). Resolution is device-side at startup and
class-attribute-driven, so every DP rank derives the same answer
(event-loop.md). `disable_prefill_graph` in the config carries user intent
only. `decode_graph=False` still requires `refresh_decode_metadata` and
`init_cuda_graph_state` — eager decode runs the same unified path.

### One output layout per forward

Every ForwardContext and grammar completion carries a required, immutable
ForwardOutputLayout. Ordinary prefill, mixed and decode batches use the same
contract as compact outputs: each emitting prefill has one output row, and
each decode has its fixed verify width. Ordinary models emit one row for
every extend request; a backend that skips incomplete-prefill outputs
shortens only the emitting prefill prefix. Missing layout is not an execution
mode.

Sampling parameters, cache progress and acceptance lengths remain indexed by
the original requests. Logits and token storage use the output layout's
prefill/decode slices and per-request offsets. Grammar masks retain a fixed
width per original request: sampling selects the first mask of each emitting
prefill and the full mask span of each decode. This rule applies equally to
ordinary mixed batches and batches with zero-output prefills. Grammar
candidate preparation reads only the decode suffix of the live input buffer.

Graph capture uses the captured batch size in its layout. Replay temporarily
pairs the padded context with a padded layout, then restores the original
live layout before output/state processing. Queued grammar completions retain
their immutable per-step layout, independently of later context updates.
Idle graph warmup also supplies an explicit layout.

### Prefill graphs around a row narrowing

A prefill forward whose row count drops once, at a fixed layer, by an amount
that is not a function of the token bucket cannot be one token-shaped
breakable graph. DeepSeek-V4.1 is the case: its CED decoder (layer 20 on)
runs on a per-request tail of the prefill rows (`decoder_view()` — a
completing prompt's last window, no rows for an open chunk, every decode
row), so the row count at layer 20 depends on which requests complete.

The model declares the split instead of opting out: it implements
`PrefillGraph`'s `NarrowingPrefillModel` contract — `encoder_forward` (the
token-shaped layers), `narrowing_forward` (the candidate source layer, all
rows in, the view's rows out), `decoder_forward` (the remaining layers and
the final norm on whatever rows it is given) and `finish_forward` (the
sampled-row gather, the DSpark row report); `forward` is their composition,
so eager and graphed prefill are one path. `PrefillGraph` captures the
encoder per token bucket and the decoder per decoder-row bucket, from a
fixed-row static state the narrowing lands into (leading rows copied, tail
zeroed). The decoder graphs depend only on their row count, so one ladder
serves every token bucket; it is the token ladder clipped to
`max_decoder_rows_per_request × max_num_seqs` (a request contributes at most
its window). A replay is encoder graph → eager narrowing → decoder graph,
all under the bucket-pinned ambient context; the narrowing and decoder
stages size their own collectives from their row counts
(`report_collective_sizing`), the decoder graph replays with the narrowed
row count as its valid rows so its breaks scrub the static tail, and a
forward whose narrowed rows exceed the largest decoder bucket runs its
decoder stage eager. Layers read their row plan from the live context, never
from a loose argument a captured break would freeze. Capture runs the
narrowing before every decoder run, as serving does: the decoder consumes
per-forward backend state its predecessor produces (V4.1's reuse layers read
the index source's selection, which later sources overwrite). Under
attention DP the split graph stays off: the narrowed row count is rank-local
(which prompts complete on this rank), so the decoder bucket and the
collective shapes its graph bakes would differ across ranks, and the stages
size their collectives from their own rows, which the DP metadata gather
does not carry (the same gap that keeps narrowing itself unimplemented under
DP).

### Prefill requests without generated outputs

The original execution batch and cache metadata always contain every request.
A backend may declare `skips_incomplete_prefill_outputs` only when its model
still produces all cache state required by later chunks. DeepSeek V4.1 uses
this after the candidate-source layer writes global KV. Other backends keep
the original output contract.

The scheduler packs completing prefills first, at most one incomplete prefill
last among prefills, then decode requests. An immutable `ForwardOutputLayout`
records E original prefills, P output-bearing prefills, D decode requests and
verify width K. Completing prefills must form a prefix (validated on the CPU).
Every forward carries a layout; one with P=E is the identity layout, and the
executor's fast paths (whole-batch `sample`, whole-batch `verify`) still apply
to it.

Logits and tokens use P+D*K rows. Accept lengths still use E+D request rows,
with zero in [P,E). Sample uses the parameter prefix [:P]; verify uses the
original [E:] suffix, retaining its batch-row coin offset. Grammar masks have
a separate token axis. Each consumer uses the same token offset:
`i` for i<P, P for P<=i<E (no storage), and P+(i-E)*K for decode.
V4.1 explicitly marks selected logits rows so the logits processor never
re-gathers them using original input indices.

The layout owns host queries for the shared prefill prefix, the original
decode request suffix, the compact decode output suffix, and each request's
stored output width. Consumers use these queries instead of deriving the
same offsets independently. The executor aligns sampling parameters and
token-indexed grammar masks at the sampler boundary; sampler interfaces and
result buffers stay unchanged. Grammar owns matcher advancement and rollback,
and zero accepted lengths already suppress advancement. Existing consumers
that only need token_offset keep that interface. Models without cropped
outputs carry the identity layout, under which every slice and offset above
reduces to the original request-indexed contract.

Completion remains a local comparison in the executor and V4.1 attention
metadata. Both use the same scheduled prefix, input count (including replay),
and current prefill target, which may include previously generated tokens
after re-admission. Contract tests keep those decisions consistent without
adding state or parameters to the shared metadata initialization interface.

Cache advancement still consumes all input lengths for prefill, independently
of output lengths. Future input writes, NaN/OOV attribution, grammar advances
and V4.1 DSpark anchors operate only on output-bearing requests. Grammar keeps
one queue completion per forward, including zero-output rounds; both deferred
hostfunc and host fallback consume the frozen layout. A zero-row decoder
bypasses its graph, normalization, LM head, sampler and draft-context writes;
the encoder and global KV producer have already executed.

When the decoder view is nonempty, the candidate-source layer retains the
original full-row mHC and QKV projection shapes, then gathers the selected
rows. Moving that gather before the projections changes split-K or quantized
GEMM arithmetic and can change the retained logits. Only an empty decoder
view bypasses those projections; its global KV producer still runs on all
encoder rows.

Compute rows and output rows are distinct. When a batch contains a completing
prefill or a decode, each incomplete prefill retains its original single
decoder compute row. Removing that row changes quantized attention and MoE
batch shapes and can alter other requests' logits even with identical input
tokens and chunk boundaries. The retained row is omitted from `logits_rows`,
so it still has no sampled token. Batches containing only incomplete prefills
keep zero decoder rows and skip the entire decoder consumer stack. This
preserves the optimization on cache-only rounds without changing the numerical
shape of rounds that produce outputs.

Final prefill windows, bootstrap tokens and PD candidate/cache handoff remain
unchanged. V4.1 PD still requires layerwise transfer interval zero. This does
not enable a cache-only prefill role or reduce resident model weights. PP and
attention-DP narrowing retain their existing restrictions.

### One draft metadata contract

The draft backend's decode metadata comes from `refresh_decode_metadata` and
NOWHERE else — the same two steps in every round:

* **decode round**: target refresh, then draft refresh over the drafter-owned
  `draft_seq_lens_buf` (freshly seeded from the batch seq_lens);
* **extend/mixed round**: draft prefill init reading the accepted-prefix
  seq_lens view (never the mutable draft buffer), then the same draft refresh
  with one token per request — deliberately NOT the packed verify width,
  which would take V4's packed-decode arm and clobber
  `forward_prefill_metadata`.

Backends' `init_forward_metadata` must NOT double-fill draft decode metadata
as a side effect (the deleted `is_extend() and self.is_draft` arms); the
mixed/idle decode arms that remain serve the target's decode requests only.
Drafters republish their in-loop seq_lens edits explicitly each step via
`advance_draft_forward_metadata` (Eagle) / `update_draft_forward_metadata`
(vanilla MTP frontier re-anchor) — metadata never aliases a buffer the
drafter mutates behind the backend's back. Those two hooks are deliberately
seq-lens-only: Eagle's step-0 accepted-prefix publish fires
`advance_draft_forward_metadata` BEFORE the step-0 attention has consumed
the verify-shaped write window, so the write-window publication is a
separate, explicit drafter-loop call (`publish_draft_step_locations`, see
"Write locations have one owner"). The router hands each hook to the
leaf's hook of the same name, because the two edits describe different
row shapes: the Eagle chain runs one row per request after step 0, the
multi-depth MTP window `k` rows per request at every depth. A leaf whose
decode kernels derive each row's causal bound from the request's single
cache length needs the same seq_lens edit for both (the
`PagedAttentionBackend` default routes `update_` to `advance_`); a leaf
holding per-row decode metadata re-expands it in `update_` — DSA rewrites
its per-token indexer rows (`_dsa_seq_lens_2d`, `[bs * k, 1]`) and their
plan to the frontier, in place, while its `advance_` re-plans `[bs, 1]`
rows and leaves the per-token rows as the round's refresh published them.
Neither hook clears the layer-shared sparse selection: the depth loop is
one forward's worth of top-k reuse.

Backends with sharded KV must refresh derived local visibility in the same
draft length-update hook as the global lengths. While page allocation and
request order stay unchanged, they reuse the compact tables and ownership
prefixes from the full refresh and update local visibility in place. Eager
execution and CUDA graph replay use the same hooks and persistent buffers.

One named exception: draft-tree lanes (`docs/design/tree-speculation.md`)
read `TreeDraftInputs`, which the drafter writes inside the round -- the
frontier and lane window lengths once, then each step's lane masks, plus
`active`, a Python flag set around each lane forward. The buffers are bound
once, live at fixed addresses and are written by in-graph ops before each lane
forward reads them; the draft leaf's decode metadata itself is still
refreshed only as above.

**Step 0 narrows rows; the drafter owns the lengths, the model names the
moment.** Eagle's step 0 runs over the target's verify window (`N` rows per
decode request), writes KV for every row, and continues from one live row
per request (`gather_ids`), whose context is the accepted frontier
`valid_cache_len + accept_len` — not the `vc + N` the round's refresh
published. The drafter computes that frontier once per round (it is also
step 1's `cache_start`) and attaches an `AcceptedPrefixPublisher` to the
step-0 context as `ctx.draft_narrowing`; the model calls
`publish_accepted_prefix()` right before the first kernel that reads the
live rows (the MLA/MHA drafts at attention start; the QSA indexer after its
verify-window layout, since that layout is derived from the decode-slot
lengths), and a draft whose step 0 attends the whole verify window (the GLM
DSA NextN heads) never calls it — the step loop publishes for step 1+.
The call is idempotent (a copy of a fixed tensor into the leaves'
buffers), so it carries no single-layer restriction. `ForwardContext`
carries no drafter tensors: `accept_lengths` and `draft_seq_lens_buf` are
gone, the handle's presence is the step-0 discriminator, and no model
computes or edits seq_lens.

Both steps run unconditionally — there is no per-drafter opt-out. What makes
that safe is the slot discipline: init writes prefill-slot metadata, refresh
writes decode-slot metadata, and forwards read the slot matching their mode
(`forward_prefill_metadata` / `forward_decode_metadata`; Inkling's conv
wrapper mirrors this with `conv_prefill_metadata` / `conv_decode_metadata`).
A round that runs no decode steps (vanilla MTP re-runs prompt requests as
EXTEND depths) leaves the refreshed decode slot unread; a block drafter
(DFLASH) re-runs the same refresh inside each block-decode step, overwriting
it. A
backend that lets one call clobber the other slot's metadata is in breach —
that, not drafter special-casing, is the invariant to fix.

**V4's packed-draft deviation (documented):** a V4 draft's packed verify
round legitimately writes BOTH slots at its end — the bs*N packed views ride
the prefill slot (the step-0 shape carrier; `_select_decode_metadata`
resolves them there through a DECODE-mode-gated fallback), and the
per-request step views own the decode slot. Capture and replay refresh reach
that state through the SAME publisher (`_publish_draft_round`), so replay
reproduces
capture's slot end state by construction — the pointer guard's capture-end
snapshot verifies it. Slot writes exist only in the three publishers; the
`forward_deepseek_v4_*` read paths thread resolved metadata as parameters
and never write a slot.

### PD decode nodes

A PD decode-only node never runs an extend forward, so latches set on the
extend path (`_cache_groups_bound`) stay False there. Refresh must therefore
bind the group tables whenever they are delivered — never gate on an
extend-latched flag — otherwise the kernels read the null page instead of
the transferred KV. This rule predates unification and now protects eager
decode too. (`_cache_contract_bound` is gone: every LCM pool publishes a
cache contract, so the target allocates its write-location buffer
unconditionally and drafts are gated structurally on `is_draft`.)

K3 DSpark pipeline prefill distributes target-tap projection across stages,
while the final stage owns the proposal network and draft cache. After
target prefill sampling, that stage runs the ordinary drafter; its completed
call publishes the final cache producer barrier. PD transfers the sampled
anchor and real draft candidates with the target and draft caches. Decode
installs that window before its first ordinary verify round. Stage ownership
changes where context and proposals are produced; candidate handoff and
verification follow the same path as other speculative prefills.

Known limitation, multi-depth MTP (`Mtp`): the drafter's cross-round
stash — per request-pool slot, the last `k-1` committed tokens and the
target hiddens one position behind them, which seed the rows of the
frontier-anchored decode window that lie before this round's verify window
— is drafter-private state the prefill node fills during its extend
catch-up and the bootstrap payload does not carry. After a PD landing the
decode node's first rounds read the slot's stash as it stands (never
filled for this request), so the depth loop rewrites up to `k-1` draft-KV
positions in the prompt tail from wrong inputs until the stash has rolled
those entries out (at most `k-1` rounds; the prompt-tail planes the
prefill node transferred were correct). Draft quality only: verification
is exact. Intended fix: ship the slot's stash rows with the bootstrap
payload, the way K3 DSpark hands over its anchor and candidates.

### PD prefill nodes

The prefill role is not an eager role; it is a role with no decode step.
`ModelExecutorConfig.prefill_only` turns the decode graph off
(`ForwardStepRunner.disable`) because there is nothing for it to capture —
the role's attention is configured at verify width one and allocates no
verify scratch for a DECODE-shaped dummy — while the prefill graph keeps the
same gating as any server (`--enforce-eager`, `--disable-prefill-graph`,
`--prefill-graph-max-tokens`, the backend's declared support). Its extend
forwards, chunked or prefix-hit, replay the breakable prefill graph through
the same `_run_target_forward` dispatch; the KV handoff to decode is ordered
behind the forward exactly as behind an eager one (the plan's remote-decode
batch is emitted only once the final chunk's result has landed). Layerwise
transfer keeps working under replay because the cache-step record lives
inside the eager attention break (`record_pd_cache_step`,
`record_layer_cache_ready`), after the layer's KV write on the same stream.

Pipeline parallelism is the one prefill configuration that forces eager:
each stage threads its boundary state through an eager stage forward
(`ModelExecutor._run_target_forward`), so `ServerArgs.resolve_disaggregation`
sets `enforce_eager` for `--pipeline-parallel-size > 1`, not for the role.
The DeepSeek-V4.1 Flash PD gate
(`test/ci_system/serve_deepseek_v41_flash_pd_1p1d.sh`) runs the prefill
role with its graphs and passes `--disable-prefill-graph` to the decode role
only.

### Sampling has no greedy branch

Greedy requests normalize to `top_k=1` in `SamplingParams.__post_init__`; the
pool-indexed sampling route serves them, which is exactly what the captured
graph records. `SamplingBatchInfo.is_all_greedy` and the eager-only argmax
branches were deleted. Equivalence (top_k=1 == argmax, ties excepted) is
pinned by `test/runtime/sampling/test_greedy_route_equivalence.py`.

### Non-speculative serving is the N == 1 case, not a second path

One sampling rule for every batch: **prefill requests sample, decode
requests verify** (`ModelExecutor._run_sampling`). The decode candidate
window is always `[num_decodes, output_length]` (`_decode_candidates`, a
persistent
`input_ids_buf` view): column 0 the last verified token, columns 1.. the
draft candidates. Without a drafter, `output_length == 1` — a one-column
window that accepts nothing and resolves to exactly one sampled token
through the same pool kernels, `accept_length == 1`
(`test_decode_verify_n1_equivalence.py`; triton is bitwise identical to the
old `sample()` route, flashinfer stochastic draws the same distribution
through the coin stream). `future_input_map` is `[pool, output_length]` for
the same reason: single-token decode is a width-1 candidate window.

Backends express verify geometry as a **floor**, not a mode: seq_lens clamp
to `clamp_min(q_len)` unconditionally (drafts and plain decode have floor 1,
where the clamp is the identity). What legitimately remains conditional on
the drafter is the *draft model's existence* — draft backend refresh and the
drafter loop itself — not the sampling or metadata shape of the target.

### Outputs are persistent-buffer slices on both paths

`sample()` and `verify()` land their outputs in each sampling backend's
persistent output buffers, on eager and replay alike. The flashinfer backend
packs tokens and accept lengths into one region (`_output_pack_buf`), so its
`get_packed_output_d2h` collapses the two device-to-host copies into one; the
Triton backends return separate token and length buffers and take the
executor's two-copy path (`get_packed_output_d2h` returns None).

## What stays graph-only

Enumerated residue in `ForwardStepRunner.__call__`, all tied to the mechanics
of replaying a recorded graph: input-buffer padding to the ladder bs plus the
DFLASH sentinel req-pool rows, `_set_graph_state_write_indices`, the DeepEP
dispatch-mode restore (`deepep_adapter.replay()`), the sampler-variant
`graph_key` lookup, the `TOKENSPEED_GRAPH_DEBUG` metadata verify,
output-buffer re-slicing, and the `ctx.bs` save/restore.

Address-freezing bugs — a refresh that binds metadata views over storage the
captured graph never recorded — are assertable: capture snapshots the tensor
identities reachable from the decode-metadata slots (`graph_ptr_guard`), and
`TOKENSPEED_GRAPH_DEBUG=1` re-verifies them before every replay (production
replays pay one bool check). The snapshot has no exemption list: every
tensor a slot reaches is an address the refresh must keep. Per-step-mutable
objects a kernel owns (FlashMLA's tile schedule, which the kernel builds and
freezes on first use) therefore live on the backend, outside the slots, not
on the views — and so do the two per-forward memos the models' layers
share: V4's write-slot mappings (`DeepseekV4AttentionBackend.slot_mappings`:
SWA, compressor state / compressed per ratio, indexer state) and the sparse
indexer's selection (`AttentionBackend.sparse_topk`, a `SparseTopKShare`:
GLM DSA's `"shared"` layers and the DSA / QSA MTP heads reuse the last
indexer layer's top-k; QSA also keeps its layer-invariant row geometry in
that share so the fused preparation runs once per forward). Every
runner-facing node clears both when it builds a forward's metadata (the router's
extend init / decode refresh / capture seeding, V4's three slot publishers), so
the first layer computes, the rest reuse, and nothing outlives its forward; the
drafter's in-loop seq_lens edits are not a new forward and leave the share
alone — the drafter itself
hands each draft step the top-k it reuses (or clears it) through the draft
backend, and starts from the target backend's. `ForwardContext` carries
none of this. What unification still can NOT test: mempool reuse and
hostfunc semantics — the e2e regression matrix keeps graph-on and graph-off
configurations for this reason.

## Backend package layout

`layers/attention/backends/` is organized by the role a node plays in the
tree, not by model: `base.py` (the runner-facing `AttentionBackend` contract
and the per-forward `SparseTopKShare`), `support.py` (graded CUDA-graph
support) and `cache_metadata.py` (the runner's block-table bridge) stay at
the root; `paged/` holds the block-table route — the `CacheGroupRouter`, its
geometry / table-stack / write-location helpers, and every kernel-facing
paged leaf (`base.py` is `PagedAttentionBackend`; MHA, MLA, FlashMLA, TRT-LLM,
TRT-LLM MLA, TokenSpeed MLA, DSA, MSA, QSA); `state/` holds the recurrent consumers
(Mamba/GDN and KDA);
`hybrid/` the layer-routing composite (`linear.py` is
`HybridLinearAttnBackend`); and `specific/` the bespoke single-model backends
(DeepSeek V4, Qwen4-Exp's composite and side-cache consumers, and Inkling's
dense + conv-state wrapper). A new leaf goes under `paged/`, a new recurrent
family under `state/`. A model-shaped backend earns `specific/` only when the ordinary
router and ordinary paged or recurrent leaves cannot express it; use by one
model alone is not a reason to introduce a bespoke backend.

`Qwen4ExpBackend` composes one attention backend, optional
`Qwen4ExpPLEBackend` and optional `QSAIndexerBackend`. The attention child is
the ordinary router, wrapped by the existing `HybridLinearAttnBackend` only
when this view owns GDN layers. Forward dispatch and PD step recording stay
with that child; the root broadcasts cache and metadata lifecycle calls.
Registry construction selects the attention child first, then composes the
Qwen4-Exp consumers once, regardless of whether this view has GDN layers.
The factory reads the pool view to choose these consumers and leaves binding
to the common validation and publication path after construction.
The root initializes the common `AttentionBackend` attributes from its own
`AttnConfig`, including draft status, verify width, dtype and head geometry;
these attributes do not depend on an attention child's wrapper shape.
Draft views have no GDN or PLE child. PLE and QSA remain available on targets
without linear-attention layers; the model retains their computation order.

QSA's full-KV attention uses the ordinary router and an MHA-derived leaf.
The attention prologue writes its KV like any MHA layer's. Sparse attention
has no MXFP8 block-scale input.
Its compressed and recent cache groups belong to `QSAIndexerBackend`, not
to extra attention leaves. The indexer backend refreshes stable raw group
tables with the shared `GroupTableStacks` fill at expansion ratio one:
block ids remain unchanged, holes become zero, and padded requests and
column tails are cleared. QSA metadata and top-k kernels consume these raw
block ids directly; neither their APIs nor the layout carry expansion factors.
The recipe rejects compressed fields whose row count or group's token span
differs from the model's single-page geometry before cache allocation.
The indexer owns its query/sequence metadata and borrows the full-KV table
and kernel page size from
`router.group_view`. Layer-shared layout and top-k still use `SparseTopKShare`
with the existing forward and MTP reuse boundaries. The router clears this
share before the root prepares its indexer child; the indexer does not clear it again.

QSA block selection carries the same uniform query width from
`decode_query_lengths` into its kernel API, using `None` for ragged or mixed
queries. Materialized scoring may group a divisor of that width to share K
within a request; it must retain each query's complete-block frontier and
selection. Group size one and larger groups use the same scoring kernel, in
both eager and captured forwards. Grouping must not be inferred from the total
row count or page-table batch size for a ragged layout.

Different query groups can produce slightly different FP32 scores because
their dot/reduction layouts differ; cross-layout bitwise equality is not a
contract. Tests check each layout against the FP32 reference with
`rtol=1e-5, atol=1e-4`, and validate selection exactly against that layout's
own scores and tie-breaking rule. Near-ties may select different block IDs
across layouts. Graph replay is compared with eager execution of the same
layout so metadata-refresh checks do not depend on cross-layout rounding.

Qwen4-Exp attention callers pass `topk_indices` explicitly, using `None` for
dense attention. The prologue has written the full KV cache before either
path runs.
Draft step zero still preserves the dense decode-context
and KV-recording override, while QSA keeps its original context and narrows
the selected top-k rows with the queries.

The QSA API preserves `decode_query_lengths`: uniform decode/verification
uses a positive width, as does every single-request forward. Multi-request
prefill and mixed/ragged queries use `None`.
Only decode may select CuTe; NVIDIA prefill uses FlashInfer FA2, including
single-token prefill. Adapting ragged rows to one-token queries must retain
this distinction. Both use the same cache writer and sparse-attention call.

`QSAIndexerBackend` privately owns `QSAVerifyState` only for a speculative
target. Registry construction binds the cache plan and preallocates its
workspace before model forward or graph capture. Draft and non-speculative
indexer backends keep metadata but allocate no target verify workspace.
Indexers use the root's `indexer_backend`; execution carries no separate
indexer object and the root has no QSA state registry or type lookup.
The staging flag records whether forward or capture has ever used staging,
not whether one round is pending. Commit must not clear it: graph replay
updates staging tensors without re-running the Python assignment. Staged
keys retain the model dtype; commit converts them to the fixed BF16 raw cache.
QSA compression callers explicitly select target-verification and draft staging;
the runtime wrapper and kernel API require both controls, including `None` and
`False` when staging is disabled.

`Qwen4ExpPLEBackend` resolves its own input/output checkpoints and query
lengths from the PLE cache group. It validates and slices rollback scratch by
batch size using its own verify width; layers consume these views directly.
It shares the checkpoint arithmetic with
recurrent consumers, but neither uses Mamba metadata nor depends on Mamba's
verify context or auxiliary-state hooks. GDN claims only the recurrent
groups that back its own state fields.

The runner calls `commit_speculative_state_after_verify` once on the target
after drafted decode/mixed execution or graph replay, with live acceptance
and `num_extends`. Since forward mode is derived from the extend count,
zero means decode at this entry. Hybrid commits GDN/KDA only then; the
Qwen4-Exp root invokes its attention child, then PLE for decode and QSA for
decode/mixed, excluding leading extends from QSA acceptance. Mixed rounds
retain PLE's direct state writes. Each consumer commits once; stateless
backends inherit a no-op.
Transient verify storage belongs to these consumers; LCM remains the owner
of the persistent request caches.

QSA verify staging and PLE commit-row buffers are preallocated for full
decode capacity and sliced per batch. Cache recipes reserve their bytes
before sizing the arena. The Qwen4-Exp root's `preallocate_verify_workspace`
selects its GDN/PLE/QSA consumers, allocates each once and returns their total
bytes; registry only invokes this operation and checks the recipe budget.
Draft roots allocate no target verify workspace. Qwen4-Exp reserves no
verify workspace when the target width is one, even with a draft model
attached; this includes the inherited GDN/PLE staging budget and PLE commit
rows.

## One block-table route: router + leaves

The layering between the scheduler's block vocabulary and the kernels' page
vocabulary is fixed, with exactly one conversion point:

| layer | sees | never sees |
|---|---|---|
| C++ scheduler | per-group `BlockTable`s: rows in `block_granularity` logical index, entries are `CacheBlock` ids | kernel pages, backends |
| bridge (`CacheBatchMetadata`) | contract-ordered group ids; `{gid: [bs, W_g]}` views over one packed int32 upload | pages, backends |
| **`CacheGroupRouter`** | attention group geometry (`CacheGroupGeometry`), each leaf's `kernel_page_size`, expansion, padding and KV write-location slot math | kernel calls |
| `QSAIndexerBackend` | its raw compressed/recent group tables, query lengths, full-KV address view and private verify workspace | MHA leaf metadata, persistent cache allocation |
| `Qwen4ExpPLEBackend` | its PLE checkpoint table, input/output checkpoints and verify workspace | Mamba metadata and verify context, persistent cache allocation |
| paged leaf (`PagedAttentionBackend`) | `page_table` (kernel pages, batch-ordered, padded), `seq_lens`, `out_cache_loc`; under a query shard also `page_table_cpu`, the host mirror of its extend rows, for ownership counts | groups, block tables, contracts, draft/target table provenance |
| state consumers (Mamba/KDA, Inkling conv, V4) | their own family's raw `block_tables[gid]` (block vocabulary) | other groups' tables, runner padding |

The runner (`ForwardStepRunner`) does one thing with tables: hand the
bridge's `block_tables` dict to the top-level backend. Capture / idle /
prefill-graph dummy forwards use the runner's `placeholder_block_tables(bs)`
(full-width zero tables, null page 0, slices of one persistent allocation) —
**always-contract delivery**: the dict is complete on every path, so no
backend carries a "no tables" arm. Delivery is guarded at both dispatch
points: the runner's inline live-delivery check and the router's
`_check_live_delivery` fail a live batch whose dict omits any consumed
group — the persistent decode buffers would otherwise serve stale pages.
Consumers take their own groups by positive claim
(`cache_consumer_families`); extra groups ride through untouched.

Inside the router, `GroupTableStacks` holds the
`[G, max_bs, stack_max_num_pages]` kernel-page table stack (each group's
table expanded to its leaf's `kernel_page_size` and padded to the leaf's
`max_num_pages`; the stack's column count is the widest group's) and the
`[G, max_bs * N]` decode write-location stack. Both are allocated once and
refilled in place: leaves copy their view out, while the decode write-slot
views and the block drafters' `draft_history_view` read the stack storage
inside captured graphs. QSA's indexer owns separate stacks for its two raw
groups using the same fill at ratio one. The fill is one expand
launch per group with plain scalar arguments (scheduler block count, source
stride, live requests) — no device-side metadata tensor, because the
per-step pinned staging + H2D it would need lands on the bs=1 latency path;
padding requests (`[actual_bs, bs)`) and each group's column tail resolve to
null page 0.
The bridge's `{gid: view}` dict is the router's input; the router does not
depend on the views sharing one storage. The slot math lives in
`paged/write_locations.py` as pure functions with one invariant: `slot = table[req, pos // P] * P + pos % P` is page-size
invariant, so locations computed over the kernel-page stack equal
raw-table locations bit for bit.

`CacheBatchMetadata` travels no further than the runner; no backend receives
it (`cache_metadata` / `forward_batch` kwargs are gone). V4 consumes the
same `block_tables` dict through its bespoke metadata build.

Deleted, for the record: `decode_buffers.py`, `group_write_locations.py`,
`draft_page_staging.py`, `expand_history_table` as a backend-side step, and
the capability flags `uses_cache_groups`, `needs_group_block_tables`,
`tables_self_padding`, `cache_active_pages_must_be_real`,
`engine_owned_group_ids`, `table_tail_pad`. None carried information not
already implied by the pool's published specs plus the always-contract
delivery.

## Single-table leaves

A paged softmax attention leaf (`PagedAttentionBackend`: MHA, MLA, FlashMLA,
TRTLLM, TRTLLM-MLA, TokenSpeed-MLA, MSA, DSA-over-dense) consumes exactly
the pre-cache-group interface — `page_table` (kernel pages, batch-ordered,
padded to `[bs, max_num_pages]`), `seq_lens`, `out_cache_loc` — and never
perceives cache groups. Leaves own their persistent decode buffers
(`page_table_buf`, `seq_lens_buf`) and copy the router's stack slice in on
each refresh; they do not alias router storage. A single-group model is a
router with one leaf; there is no single-table special case anywhere.

The sanctioned per-leaf residue, all kernel-imposed: `verify_floor` /
`block_decode_active` (spec verify geometry as a clamp floor),
`block_decode_expansion` (whether block decode materializes one metadata
entry per block position, or the leaf repeats one per request at forward
time — FlashMLA, TRT-LLM MLA), FlashMLA's `for_graph_replay` tile-schedule
swap, and the MLA family's `num_extends` decode-request slicing
(`override_num_extends`).

One side channel exists beyond the table: `set_request_slots(req_pool_indices)`,
a no-op by default, which the router calls on every leaf after each
metadata build (extend init, decode refresh, capture seeding). It serves a
leaf that owns per-request side state indexed by pool slot — DSA's KPool
tails — and doubles as that state's per-forward reset point. Paged KV
leaves ignore it; it carries no table or page vocabulary.

## Write locations have one owner

`write_locations(layer, forward_mode)` on the top-level backend is the ONLY
accessor for KV write slots — models, drafters and the runner neither
compute nor thread location vectors. `forward_write_locations(layer,
forward_mode)` derives from it the slots the attention prologue writes: the
mode's `write_locations`, with the decode window appended for a draft's first
step over a MIXED round (`docs/design/attention-prologue.md`).
`PagedAttention.forward`, `AttentionBackend.forward`, `model_runner.forward`
and the model forward chains above the attention layers carry no
`out_cache_loc` parameter; `InputBuffers` has no location buffer;
`fill_input_buffers` takes no table.

* **Extend**: `init_forward_metadata` computes each group's span over the
  stacks (`[sum(extend_seq_lens)]`, request-major); `write_locations(layer,
  EXTEND)` returns exactly that span.
* **Decode / verify**: `refresh_decode_metadata` publishes the token-major
  `[bs * N]` window views (`decode_write_locations`, pointer-stable per
  bs — the graph records them through the prologue's KV writes, and the
  pointer guard walks this slot). A MIXED round's draft refresh sets
  `_decode_request_offset = num_extends` so DECODE reads skip the extend
  requests.
* **Draft steps**: the drafters declare each step's window, the router owns
  the math and the address-stable storage. `publish_draft_step_locations(
  cache_start, n)` computes the window over the location stack (the same
  fused launch the decode refresh records — in-graph safe) and points
  `write_locations` at it: Eagle publishes its one advancing slot per step,
  vanilla MTP its re-anchored k-window once per round, DFLASH its block
  window after each block refresh (order matters: the refresh republishes
  the verify-shaped window). `draft_write_locations_uniform(out, start, n)`
  is the side-write variant — scratch resolution over the full-history
  table (`draft_history_view`) that must not clobber the published window
  (DFLASH's target-KV injection, DSpark context windows).
* **Cross-backend reads**: `decode_window_locations()` /
  `extend_span_locations()` expose the full-history group's published
  windows; DFLASH reads the TARGET router's windows through them to copy
  target-aligned KV into the draft cache (the pools share one page-id
  space).
* **Writes outside the backend** (the attention prologue, V4 group writes)
  fetch their slots immediately before the write. When a draft step-0
  forward over a MIXED round writes the round's full K/V rows, dispatched as
  MIXED or as DECODE (the MLA draft's whole-batch write, a GQA draft's
  narrowed first step), `forward_write_locations` concatenates the EXTEND
  span and the DECODE window — eager-only, MIXED rounds never run under a
  captured graph. Target MIXED decode halves and later draft steps keep their
  ordinary decode-only windows. V4 composes the shared token-shaped resolve
  (`page_table.group_slot_mapping_from_raw`) over its own group tables; a
  degraded mapping fails closed to `-1` (skipped write), never to a raw
  fallback vector.

## Target capture is configured once during model setup

`create_model_runner` calls `execution.factory.configure_draft_target`
after both models load and before cache construction. DFLASH/DSPARK models
must implement the explicit `TargetCaptureConfigurator` interface; missing
implementations fail at setup. A method with the same name on an unrelated
object is not treated as an implementation or as evidence of prior setup.
DFlash, DFlash2 and generic DSpark use their model's ordinary capture setup;
K3 owns its trained stream/projection contract; DeepSeek V4/V4.1 DSpark owns
its checkpoint tap selection. `models/target_capture.py` contains only the
shared interface. DFlash checkpoint parsing and target configuration live in
`DFlashDraftModel.configure_target`, inherited by DFlash2 and generic DSpark.
Setup calls only the parent interface
`configure_target`; each concrete draft directly adapts to its target family.
There is no generic DSpark helper probing for a DeepSeek-specific setter, and
K3 targets need not implement that setter. EAGLE3 selection remains in this same setup
phase, including its explicit server-argument override.

This runs on every PP stage even when that stage has no executing drafter.
`wire_target` only binds embeddings, heads and other execution resources; it
never selects capture layers, changes streams or replaces the output layout.
This also applies to V4.1's dedicated drafter with scheduler-owned context windows.
There is no configured flag or optional-method probe in resource binding.
A last pipeline stage borrows its local draft embedding when the target
embedding lives elsewhere; this is resource binding, not a different proposal
algorithm. Per-forward capture hooks consume the established configuration.

Embed/head sharing (`shares_target_embed_head`) follows the same stage
ownership. `get_embed_and_head` returns None for a side the stage does not
hold -- the embedding lives on the first stage, the head on the last -- and
never dereferences an absent module. Stages before the last bind nothing: a
draft built there only produces context. The last stage binds what the target
reports, which is the head alone (`embed=None`); a pipeline-capable draft then
keeps the `embed_tokens` shard its checkpoint ships for that layer, its loader
rejects a pipeline checkpoint without one, and the factory checks afterwards
that the draft's own `get_embed_and_head` still reports an embedding -- a
draft that aliases None into its embedding is named at construction rather
than failing at its first forward (`BaseCausalLM.set_embed_and_head` refuses
`embed=None` outright, since a generic draft keeps none). Off the pipeline
both sides are shared and the draft's copies are dropped before the KV-cache
budget is profiled, as before.

Checkpoint tap labels remain zero-based completed-layer IDs. Prefix tap L is
produced after L. AttnRes tap L is produced at L+1's entry by that layer's
mixer, before input-layer normalization or snapshot mutation; the final tap
belongs to the output mixer. Capture execution and projection-weight placement
use this same owner on both PP and non-PP. There is no boundary deferral or
recovery operation. Capture's mixed stream must not be replaced by the fused
attention input, which already includes input-layer normalization.

`execution/dspark_context.py` contains `DSparkContextProducer` and the model
interface it consumes. K3 tap ownership and projection arithmetic live in
`models/kimi_k3_dspark.py`; the producer does not interpret K3 layer IDs.
This interface covers DSpark context production, not a requirement for all
draft algorithms. K3 DSpark is currently the model using this production path.

Pipeline stages use `DSparkContextProducer`: each stage normalizes the taps it
owns if configured, applies their projection columns and sums in FP32; the
accumulator travels with the chunk's PP state and the final stage applies
context normalization once and writes native context KV. The executor selects
the producer from the pipeline configuration and the draft model's class
(`select_dspark_context_producer`): on a pipeline a draft implementing
`DSparkContextModel` gets a producer on every stage, a block drafter whose
model does not implement it is rejected (it would draft from one stage's taps
alone), EAGLE3 is refused (its aux taps span stages), and any other draft gets
none. Off the pipeline every tap is local, so
the drafter keeps its concatenated projection and its own context writes --
including the quantization-aware path, since raw per-tap weight slicing is
not a quantized linear operation. PP drafts require unquantized projection
weights.

### Pipeline speculation is not DSPARK-only

An MTP (NextN) draft runs on a prefill pipeline without any cross-stage
production: it consumes the post-final-norm hidden states the last stage
already computes, so only that stage drafts. EAGLE3 stays off the pipeline:
its aux taps come from several stages and the stage boundary bundle does not
carry them; `ServerArgs` rejects it and `select_dspark_context_producer`
refuses it again at executor construction. The MTP shape is

* **Construction.** `ServerArgs` accepts `DSPARK` and `MTP` with
  `--pipeline-parallel-size > 1` on the prefill role only (the chunk pipeline
  has no decode token feedback to draft against anywhere else); the dense ==
  attention TP / CP = 1 rule stays DSPARK's, whose draft reduces attention-TP
  embedding partials over the dense TP group. Which stages build the draft
  model at all is the factory's decision, not each model's
  (`factory.pipeline_stage_builds_draft`): a block drafter is built on every
  stage because it produces context from every stage's taps, while an MTP
  draft is built on the last stage alone -- the other stages construct no
  draft runner, load no draft shard and wire nothing, so `ModelExecutor` and
  `configure_draft_target` see `draft_model_runner=None` there and the in-tree
  K3 and V4 NextN drafts need no stage-aware shell. The skip is safe because
  nothing in draft construction is a world collective the skipping stages
  would have to join: the KV-budget all-reduce runs on every stage with or
  without a draft, the DeepEP/MoE communicators a draft layer reuses are the
  target's and scoped to the stage, and the only world-scoped step -- the
  InstantTensor weight iterator, which synchronizes over `group.WORLD` when a
  model declares no `checkpoint_load_group` -- is bounded by the factory,
  which loads a pipeline draft with the stage's rank set
  (`ModelRunner(checkpoint_load_group=...)`, surfaced as
  `LoadConfig.checkpoint_load_group`; a model's own declaration, such as K3
  DSpark's stage-subset filter group, still wins). With no producer configured
  the last stage's target forward captures `CaptureHiddenMode.FULL` for its
  drafter.
* **Cache.** Nothing changes: `CacheLayerOwnership` already places the draft
  cache layers as the last stage's trailing producer step, the draft backend
  and pool exist only there, and the merged plan, bootstrap placement and
  transfer routes are the same as for DSPARK.
* **Layerwise CachePD.** `supports_pd_layerwise_finalization` is decided per
  stage (`device._supports_pd_layerwise_finalization`): a stage owning no
  draft fields has nothing to finalize and answers True; the owning stage
  answers for `ModelExecutor.draft_field_writer` -- the producer when
  configured, else the drafter -- exactly as a non-PP engine does, and
  `register_draft_final_step_counter` reads the same property. The last stage
  registers the draft-final step counter; the others count target layers only.
* **Handoff.** The last stage samples, runs the drafter over the completing
  chunk and writes the candidate block into the reserved decode slot; the
  event loop broadcasts `(output_tokens, output_lengths, next_input_ids)`
  over the PP gloo group at commit -- with the logprob vectors and the NaN
  guard's per-request flags, see the QCP section -- so every rank's
  scheduler stamps the same bootstrap token and candidate window onto the
  remote decode. The PD wire and the decode side are untouched.

Expect a larger last-stage bubble (NextN layer plus draft extend and
multi-step drafting); rebalance with `--pp-layer-partition`.

The producer is stateless across forwards. Each chunk owns its accumulator;
queued chunks cannot alias it. A configured `ctx.dspark_context_producer`
owns native context writes during target forward; otherwise the drafter owns
them. This responsibility is fixed at construction, not inferred from a
per-round readiness flag. The producer enqueues writes before the drafter on
the same stream; failures propagate instead of selecting a fallback writer.
The common block drafter always updates accepted-prefix lengths, but only
projects/writes context when no producer is configured. Its optional auxiliary-stream writer is disabled for a forward with
this producer, avoiding a second writer or a missing stream dependency.
The final PD readiness barrier remains after the whole proposal call, since
proposal execution can write the same draft fields after context injection.

## Per-forward drafter work rides on the context

What a drafter wants done *during* the target forward is a property of that
forward, so it travels on `ForwardContext` — never as mutable state on the
target model that someone must remember to reset. The executor's only
seam is `BaseDrafter.prepare_target_forward(ctx)`, called right before the
target runs: the drafter decides under its own gate whether this round
qualifies and attaches what it needs; a fresh context per round means
nothing outlives it, and a model that sees no attachment does nothing.
DFLASH is the one user: its incremental projection attaches
`ctx.target_capture_sink`, the target hands each captured tap to
`on_target_capture` as it is produced, and the sink accumulates the
draft's `fc` projection on the aux stream so the draft KV is written under
the target's remaining layers. The arming gate is the same
`_overlap_allowed` the drafter's `run` decides the overlap path by, so a
round can never be armed on one side and drained on the other. These hooks
consume the target's capture configuration; they do not change the tap
selection or output layout.

The reverse direction rides on the context as well: a target that captures
its taps on a row subset reports it as ctx.captured_rows
(CapturedRows(positions, prefill_spans)). V4.1's CED narrowing is the one
producer: its taps in layers 37–39 contain no rows for incomplete prefill
chunks, the last window of each completing prefill, and all decode rows.
The reported prefill spans retain a zero-length entry for each incomplete
request, preserving the original request order. DSpark's prefill seeding
(_seed_prefill_windows) reads these spans and positions instead of the
input-length mirror and skips zero-length spans. A target with one captured
row per input row leaves ctx.captured_rows as None, and the drafter keeps
its buffer-based layout.

## Shared prefill convolution preparation

Mamba/KDA extend metadata owns one immutable `CausalConv1dPrefillMetadata`
per forward. Its two int32 maps associate convolution programs with request
rows and local token chunks. The builder sizes them from the existing host
length mirror and fills both directly from device query boundaries in one
Triton launch. Every layer reads the same tensors and block size; the conv
wrapper neither rebuilds them nor initializes/uploads per-layer scratch.
This is transient execution metadata, not a new cache group or model state.

The same extend/mixed metadata owns a device int64 mirror of the int32
query boundaries. KDA layers share it for scan ABIs instead of casting
per layer; the host int64 mirror still supplies launch planning without
D2H. Decode refresh/capture does not allocate this prefill-only mirror.

Each metadata build allocates fresh index storage, including when two
forwards have the same total token count but different request partitions.
No subsequent forward refills a buffer an earlier forward may still read.
Mixed batches include the decode rows' verify-token lengths in this same
builder. Decode-only refresh/capture remains unchanged and carries no
prefill convolution schedule. Breakable prefill graphs consume the live
metadata in the eager attention break, as ordinary eager forwards do.

Prefill state staging fuses resumed conv-window copying, recurrent-state
gather/zero, and history flags in one kernel. It preserves scheduler-owned
block ids and arbitrary cache strides. Fresh rows never read null or stale
recurrent state; their conv working windows remain unchanged. Shared input
snapshots are read-only, output blocks are unique, and a private in-place
source/destination is legal. This changes neither scan arithmetic nor cache
allocation, retention, or checkpoint identity.

The NVIDIA CuteDSL prefill adapter declares its native `v_major`
(`[N, H, V, K]`) state layout. The dispatch facade alone adapts a caller
with another layout; the wrapper must not round-trip native state through
FLA's `[N, H, K, V]` convention. Direct wrapper callers use the native
layout for both initial and final state. Exact-length gate conversion to FP32
and beta packing retain their ordinary PyTorch operations. The native wrapper
allocates the scan output. Ordinary attention breaks copy it into a stable
graph-owned handoff buffer; inline KDA keeps output restoration and padding
cleanup inside the graph, without that handoff copy. No output-buffer
extension to the native wrapper is required.
These preparation changes modify neither the native scan, its gate math, nor
GEMM arithmetic.

## Recurrent prefill subgraphs (KDA, Mamba2)

### Capturing recurrent layers in the outer graph

`CapacityPrefillBackend` (`state/prefill_capacity.py`) owns this contract for
KDA and Mamba2; each subclass only states which forwards it admits and whether
uncaptured shapes also run the capacity layout. GDN does not capture its layers.
Supported pure-extend forwards use `prepare_prefill_metadata` before eager
execution, startup capture and replay. This consumer-stream seam builds or
refreshes `CapacityPrefillMetadata` with the selected token and request capacities.
Eager execution uses the live count; replay may round up to a captured count.
The same metadata contract controls scan capacity, checkpoint packing and
output restoration in every case; there is no temporary metadata binding or
mutable inline flag. Only startup capture retains the metadata's addresses.
Uncaptured shapes use temporary storage through the same builder.
Whether a shape can be captured is also asked on its own, through
`admits_prefill_graph`, which reads no forward context and writes nothing; the
seam must return the same answer, and startup capture raises when a backend
admits a shape and then refuses to prepare it.

Mamba2 keeps the scheduler metadata for uncaptured shapes: its chunk plans
need only live bounds, so the capacity layout would only add packing to eager
forwards. A retained shape owns one persistent chunk plan per scan (body and
tail), sized `extent // chunk_size + sequences` and padded with empty chunks;
the preparation seam rewrites both in place, in one pinned upload, before each
use. Mamba2 chunks align to the packed token axis, so when one-token dummy tails
shift a later request's tail, a multi-request capture matches eager within
rounding rather than bit for bit; a one-request capture matches exactly.

For retained shapes, the hybrid wrapper can omit the KDA attention break and
capture neighboring projections, KDA kernels and post-attention compute together.
Full-attention layers keep their breaks. Before execution, the common preparation
step validates live lengths and refreshes boundaries, convolution maps and
state-page indices. All KDA layers read this same storage. Token lengths may
vary within the bucket; live request counts may fill part of a capture. Native output padding
is cleared by the KDA forward, replacing the attention break's handoff copy and
tail scrub when KDA is captured.

The ordinary outer capture is retained for mixed batches and other request
counts beyond captured capacity. All variants share the outer pool and execute serially, as existing
bucket captures do. Layerwise PD transfer and data parallelism retain the
ordinary route: host cache-step callbacks must remain live, and DP admission
must stay rank-uniform. Retained metadata rejects a replacement cache pool; graph
release and recapture remain the orchestrator's responsibility.

Internal-checkpoint forwards have a merged capture with two scan capacities.
Stable body/tail token maps use negative indices for inactive rows; packing
zeros those rows, and inverse-map gathering restores live output order while
zeroing output padding. Compact batches without an inverse map use scatter.
Each scan consumes its
own live GPU boundaries and CPU mirror. Checkpoint writes retain the eager
ordering: convolution snapshots precede convolution updates, and recurrent
snapshots precede the tail scan. The graph binds scheduler-owned checkpoint
destinations, not backend-owned cache pages. Replay uses a captured request
capacity; live counts, checkpoint counts, row identities and lengths can change.
The outer owner captures request capacities from
`prefill_graph_capture_batch_sizes` (unset: the minimum count per token bucket)
with one variant per token bucket and request count. `ModelExecutorConfig`
requires this field explicitly: factories forward the configured list or `None`
for the minimum-count policy, so missing configuration wiring fails at
construction. Token buckets still follow the shared prefill token ladder.
Capture requests have positive lengths and fit the model context and request
buffers. At replay, unused execution slots have zero convolution length, negative
state-block indices and one masked dummy token in each packed native scan.
Their checkpoint/output maps and state-update rows are negative. Native scans
still receive only positive-length sequences; padding owns no cache blocks.
The live context and scheduler/MLA request counts remain unchanged. Selection
reserves `live_tokens + padded_requests` in the token bucket; a full bucket can
use the next existing token bucket, otherwise the ordinary fallback remains.
Startup autotuning uses the same dummy-batch builder with an explicit minimum
request count, `ceil(num_tokens / context_len)`, independent of the configured
capture request counts. Its token budget also respects rank-local request capacity.
Request counts exceeding captured capacity retain the ordinary attention break and eager KDA,
including internal-checkpoint batches. Replay refresh includes
`scan_query_start_loc`, which the recurrent dispatcher consumes, as well as
the convolution boundary and existing int64 mirror.

The checkpoint metadata also owns an inverse output-token map, built with
the existing packed host metadata and refreshed at the same stable addresses.
Each layer gathers body and tail outputs in one kernel, writing zero for
negative sources. This replaces two scatters plus output initialization;
the capacity-shaped forward needs no additional output-padding scrub. Other
checkpoint batches retain their ordinary merge when no inverse map is supplied.
Q/K/V may remain views of convolution output until the existing checkpoint
packer materializes them. Saved verification payloads keep their split producer.

Merged graphs reserve one tail slot per captured request slot. An inactive slot has
one zero-input dummy token, a negative output-token map, no checkpoint
destination and a negative state-update row. Its scan result must never replace
the body's final state. This padding is execution scratch, not a scheduler
request or cache allocation. Native scans still see positive-length sequences.
With the feature enabled, supported eager forwards use these same fixed slots,
including the dummy tail scan when no request needs a checkpoint. With the
feature disabled, compact tails still skip that scan. Both use the same
checkpoint writers and recurrent-state scatter, which ignore negative
destinations/rows. Unifying metadata does not imply zero padding cost.

### Startup capture and eager fallback

Supported KDA prefill uses the merged captures owned by `PrefillGraph` by
default when prefill graphs are enabled. `--disable-kda-prefill-graph` disables
KDA capture without changing ordinary prefill or decode graph settings. The
shared `ServerArgs` configuration passes the setting explicitly to each KDA
backend. Startup creates the configured token-bucket and request-capacity
variants. Serving forwards only select and
replay these captures, never warm up or capture a separate per-layer graph.

If no compatible merged capture exists, the ordinary outer graph retains its
attention break and calls the same eager KDA implementation. Checkpoint
handling, PD cache-step recording and break-output copy/padding keep their
existing order. Inputs outside the outer graph's admission rules run eager.
New request shapes do not grow a backend-owned graph cache. Metadata refresh
and eager execution may still allocate temporary buffers.

The outer owner holds all captures and outputs in one table keyed by token
capacity and request capacity; `None` in the request-count position selects
the ordinary attention-break capture. The backend retains startup metadata for
the exact shapes that need stable addresses, not graphs or request state. Serving
forwards never grow this retained table. The outer owner's serial shared-pool
discipline applies to all variants; there is no separate KDA graph pool. Before
recapture, it releases the old captures and resets retained prefill metadata via
`init_prefill_graph_state`. Publishing a cache pool also drops retained prefill
metadata. Graph release and cache-pool rebind remain coordinated by the
orchestrator.

### Fixed-capacity execution metadata

The capacity metadata overrides only the packed execution extent; real
host lengths and GPU boundaries still agree. Its `PrefillCapacity` validates
the packed bounds it builds. For KDA, an explicit
`KdaPrefillCapacity` passed to the kernel facade admits the live CPU lengths:
each sequence may fill the bucket, but their combined tokens must also fit it.
The CuTeDSL adapter alone converts this descriptor to native planning bounds.
Convolution maps reserve `ceil(token_capacity / block_m) + sequences - 1`
programs, bounding the sum of per-request rounded lengths without reserving
the entire token bucket for every request. One GPU metadata refresh
per forward marks inactive programs with PAD_SLOT_ID before all layers run:
the convolution kernel otherwise performs unmasked prior-token loads even
for an excess chunk. Scan inputs are cleared past the live device boundary
inside the graph, since a capacity descriptor makes padding addressable to
native full-tile loads. Both conv and scan read live GPU boundaries;
total packed tokens, including dummy slots, must fit the physical extent.
Native sequence slots are never empty: request padding uses masked one-token
sequences in the scan maps. Convolution skips their zero-length spans. A capture
can serve smaller live batch counts without another schedule.
Other solutions retain exact live-length planning and reject capacity mode.

For the pinned token-major CuTeDSL ABI, a fused preparation kernel scrubs
padding, converts gates to FP32 and builds the device chunk plan. Its total
chunk capacity is `ceil(token_capacity / 16) + sequences - 1`, while each
sequence retains the full per-sequence walk bound. The third-party adapter
passes this explicit plan to the existing native launch without replacing
global functions or changing scan arithmetic. Routing and workspace partition
rules remain owned by the native host. Unsupported layouts retain the public
wrapper's capacity preparation.

Only the checkpoint packer may assert `inputs_packed`: it owns contiguous
Q/K/V/beta and initializes every padded token. That contract skips redundant
copies, never inferred merely from being inside capture. Gate projection can
still produce undefined padding, so gate scrub/cast always runs. Per-call
plan and scratch tensors belong to the active graph pool or eager invocation;
they are not a mutable process-global plan shared across replay streams.

Changes to this capacity contract require validation of full-model overlap,
memory use and performance in addition to kernel correctness.

## Query context parallelism

`--prefill-context-parallel-size N` (`mapping.attn.qcp_*`, `N ==
attn.tp_size`) shards every extend forward of a PD prefill engine over the
attention TP group. One forward, one path: the scheduler plans the whole
chunk on every rank, the executor builds a `QueryShardPlan` from the request
lengths (`execution/query_shard.py`: `row_counts = scatter_count(total, N)`
— the reduce-scatter / all-gather split, so the `CommManager` row tables
already describe the shard and the final gather needs no permutation; rank
order is request order) and puts it on `ForwardContext.query_shard`. The
model's rows are then `input_ids_buf[start:end]`, `positions_buf[start:end]`
and the same slice of every per-row input; per-request inputs keep the whole
batch with `RequestTokenHistoryView.row_offset = start`; a pipeline stage's
boundary bundle carries shard rows. `ctx.input_num_tokens` and
`global_num_tokens` keep the scheduler's full-chunk meaning: only collective
sizing (`CommManager`, `models/base/comm_ops.py`) reads them, a model reads
its rows from its tensors or `ctx.query_shard`.

The contract a model (in tree or a plugin) implements:

* `CommManager(query_sharded=mapping.attn.has_qcp)` — declares that the
  model slices its rows by `ctx.query_shard`; a model that does not slice
  passes `False` and is refused at construction under a sharding mapping.
  Attention returns complete rows under this mapping — its weights are
  head-replicated (the default, `mapping.attn.head_tp_size == 1`), or under
  head TP over the shard group its own tail returns this rank's rows (below)
  — so the attention legs (`pre_attn_comm`, `gather_residual`,
  `post_attn_comm`, `post_final_norm_comm`) are identity on every forward —
  a sharded extend and the drafter's replicated decode steps alike — and
  `needs_pre_attn_all_gather` / `needs_final_all_gather` are False: nothing
  is ever scattered by attention. The dense and MoE legs follow the
  forward: with a shard, the existing all-gather / reduce-scatter legs over
  `plan.rows_for_collective(ctx.collective_num_tokens)` (the shard rows, or
  the sampled rows after a draft's narrowing); without one (decode steps,
  idle), the replicated all-reduce legs — which is why dense TP and the MoE
  TP×EP group must each be 1 or the attention TP width (`validate_qcp`, and
  the manager refuses other mappings). The row-layout conversions
  (`slice_scattered_rows`, `gather_scattered_rows`) are identity on a sharded
  forward, whose rows are the scattered share already. Fusion off.
* `VocabParallelEmbedding.forward(input_ids, query_shard=ctx.query_shard)`:
  the ids a rank embeds are its shard, so the layer all-gathers the ids to
  the span for its vocab shard, sums the shards and reduce-scatters the rows
  back (the same bytes as the replicated all-reduce). A model passes the
  shard explicitly; the default (`None`) is the replicated lookup.
* `PagedAttention.latent_prologue(..., key_rows=QueryShardGather(ctx.query_shard,
  mapping.attn.qcp_group))` (through `DeepseekV3AttentionMLA.forward_absorb_qkv_proj`
  automatically): the prologue rotates the local rows without a cache
  (`mla_prologue(cache=None)`), all-gathers the rotated latent to the whole
  span with the plan's row counts and stores it owner-masked
  (`latent_store`); `slots` is the whole span. A rank whose shard is empty
  rotates nothing but joins the gather and the store, so a model must reach
  the prologue (and every other QCP collective: the index-K gather, each
  group's history gathers) with zero rows rather than return early; the
  dense MLA path (`DeepseekV3AttentionMLA.forward`, the expanded prologue)
  refuses a shard up front. Index-K the model gathers the same way
  (`token_all_gather` of the local keys before quantization) and writes
  with the owner mask.
* GPU DSA (`backends/paged/dsa.py`): `init_forward_metadata` builds a
  `DSAQueryShardMetadata` — request groups whose summed history fits the
  gather workspace (one whole history, reserved from the cache budget by the
  recipe's `workspace_bytes`, allocated once on the target tree by
  `registry._prepare_fixed_workspaces` and shared with the draft tree, whose
  extend step never gathers concurrently), each with its `row_base` in the request-major
  history-row numbering, this rank's `local_query` slice and a
  `HistoryGatherPlan` (per-owner row counts from `page_table_cpu`,
  `dcp/placement.py: owned_history_rows`). For the indexer the model calls
  `backend.gather_history_index_k(layer_id, pool, group)` per group, which
  returns the group's index keys in position order in the leaf's
  `index_k_format` (`DSAConfig.index_k_format`; the pool read
  `gather_index_k_rows(..., index_k_format=)` refuses a plane of another
  dtype): `fp8_scaled` gives `(fp8 [rows, head_dim] uint8, scales [rows,
  head_dim / 128] fp32)`, `bf16` gives `(keys [rows, head_dim] bf16, None)`.
  The model hands them to `dsa_prefill_topk(q_local, w_local,
  group.gather.virtual_slots, row_starts_local, row_ends_local, ...)` as the
  rows in workspace-row order of that format -- `index_k_fp8=, index_k_scale=`
  or `index_k_bf16=` -- never with `index_k_cache`; the facade routes the
  rows by the `index_k_format` trait and requires the
  `index_k_workspace_rows` feature (`dsa.INDEX_K_WORKSPACE_ROWS_FEATURE`),
  so only a leaf whose launcher takes rows of that format is selected and
  handed the row keywords (the in-tree DeepGEMM leaf for the FP8 pair; a
  plugin's bf16 leaf declares the feature beside `index_k_format={"bf16"}`
  and takes the `index_k_bf16` keyword), and an override cannot force a
  plane-only leaf onto rows. The leaf probes that selection at construction
  (`DSABackend.__init__` under `qcp_size > 1`,
  `dsa.select_dsa_prefill_topk_for_rows` with the configured format, page
  size, indexer geometry and the envelope's batch-invariance and solution
  pin), so a platform without a declaring leaf is a `NoKernelFoundError` at
  startup rather than in the first sharded prefill. The
  model adds `group.row_base` to the returned rows and hands
  `forward_sparse_prefill(topk_slots=<workspace rows>)` the local rows;
  the arm gathers every group's KV (`gather_history_kv`, a collective every
  rank joins even without rows in the group) and attends the local rows with
  every head, `return_lse=False`, no combine. The gathered buffer handed to
  `dsa_prefill` as a flat `[slots, dim]` cache is a whole number of kernel
  pages (the workspace rows are padded by
  `dsa_history_gather_workspace_rows`), so the paged solutions' view of it
  holds too and no solution is wrong at runtime; the padding rows are never
  selected. The history gathers move any row dtype (uint8 index-K rows
  packed in the plane's format -- FP8 bytes then fp32 scales, or bf16 key
  bytes -- fp32 scales, fp8 latent) as bf16 pairs of their bytes, since the
  token all-gather's low-latency solution is bf16-only; the workspace
  (`HistoryGatherWorkspace`, `index_k_format` recorded, rows of
  `index_k_row_bytes(head_dim, format)`) is sized by the recipe and
  allocated by the leaf from the same formula; the recipe refuses a draft
  whose `index_k_format` differs from the target's, naming both, since the
  two share one workspace, and the adopting leaf checks the recorded format
  with the rest of the geometry. The sparse cores take
  their head count from the query, never from `layer.tp_q_head_num` or a
  mapping assumption (`DSABackend._query_heads`: the model's heads or the
  attention-TP slice, any other count refused), so one model layer serves a
  forward whose rows carry every head and one whose rows carry the slice.
  The sharded arm attends with every head and refuses a query carrying the
  slice. The decode arm (the drafter's steps) keeps the DCP combine over the
  KVP pages; its form follows the query's heads (`keep_all_heads` for every
  head — head-replicated weights — the gather-and-reduce-scatter form for
  the slice, as the drafter's steps under head TP over the shard group
  carry). The model threads the sequence after `forward_absorb_qkv_proj`
  through `DeepseekV3AttentionMLA.sparse_prefill_attn_v_proj` (the backend's
  `forward_sparse_prefill` with the model's selection, then
  `project_attended_heads`: under head TP the tokens-to-heads exchange, the
  local `w_vc`); GLM-5's sparse prefill runs it in tree. A rank whose shard
  is empty calls it like every other rank — the core's history gathers are
  collectives — and projects nothing.
* The model exit hands the logits processor the shard's rows with the plan
  on `LogitsMetadata.query_shard` (`BaseCausalLM.exit_logits` is that one
  call; a model with its own exit does the same). The shard is a parameter
  of the processor's row selection: it scores the planned prompt rows
  first, then selects the sampled rows — `hidden_states[gather_ids]` on
  whole rows, `gather_sampled_rows` over the group on a shard (one
  byte-preserving all-gather, `token_all_gather_rows`, with
  `sampled_rows_per_rank`; rank order is request order) — and runs the LM
  head on the batch's `[bs, hidden]` rows, so the vocab all-gather is the
  TP one. The group of both gathers is the processor's TP group: the LM
  head is vocab-sharded over it, every rank of it must end with the same
  rows, and `validate_qcp` makes the query shard group exactly that group
  (`qcp_size == attn_tp_size`); the processor refuses a plan of another
  width. The processor is the only caller of `gather_sampled_rows`
  (`CommManager` has no sampled-row leg: `needs_final_all_gather` is False
  under a shard and nothing gathers the final norm's rows). A FULL hidden
  capture stays the shard; a LAST capture is the gathered `[bs, hidden]`
  rows, whole on every rank — the aux taps' (Eagle3) when the model has
  them, each tap gathered the same way, and only on a LAST capture, since
  no other mode reads them selected. A model that selects its rows before
  the processor (`logits_rows_selected`) keeps that contract, and such a
  model cannot serve prompt logprobs, sharded or not. `ctx.gather_ids`
  keeps the batch's full layout on every forward, the
  drafters' extend steps included (Eagle's step 0 and every depth of the
  multi-depth `Mtp` drafter read the shard's slice of the shifted prefill
  ids and positions, chain the shard's hidden rows, and carry the plan on
  their context; `Mtp` sums its cross-chunk stash of target hiddens over
  the group, one owner per row): `QueryShardPlan.local_sampled_ids(
  ctx.gather_ids)` is the one place that cuts them to the shard, used by
  `gather_sampled_rows` and by any model that narrows to its live rows
  itself. A draft model's FULL capture under a shard is the shard's rows,
  which is what the next depth consumes.
* Prompt logprobs (`InputLogprobPlan`, `--input-logprob-chunk-tokens`): the
  plan's rows are full-layout rows. The forward thread stages the whole
  plan's targets and slots on every rank (the shifted ids are the whole span
  everywhere, so the target audit flags the same requests on every rank)
  and keeps as this rank's `InputLogprobRows.rows` the plan's rows inside
  its shard, re-based to it; the plan's rows are sorted, so each rank's are
  one contiguous run and the per-rank counts (`rows_per_rank`) are host
  arithmetic over the shard boundaries (`QueryShardPlan.rows_per_rank`,
  `local_rows_run` — the one row split of the plan, the same that counts
  the sampled rows per rank; it refuses unsorted rows rather than miscount
  them). Scoring a row needs its full-vocabulary logits, and
  the head is vocab-sharded over the group: every rank must hold every
  planned row, so `compute_input_token_logprobs` all-gathers the planned
  rows' activations with those counts (`[plan rows, hidden]`, rank order is
  row order — a rank without a planned row contributes none and still
  joins) and then runs the unsharded chunk loop over the whole plan on
  every rank; the chunk schedule is thereby the same on every rank, which
  the vocab all-gather inside each chunk, a collective, requires. Every rank
  ends with the whole plan's fp32 vector — the tensor-parallel path's bit
  for bit, since each row meets the same operands — so no result gather
  follows, the per-request NaN audit agrees across the group and
  `ModelExecutionResult.input_token_logprobs` is the full vector in plan
  order on every rank; the commit path and the P→D bootstrap-logprob frame
  are unchanged. (Scoring only the local rows against a vocab-sharded head
  is not possible: a row's log-sum-exp needs every rank's vocab slice of
  that row, and the vocab all-gather assumes replicated rows. The LM-head
  work for prompt logprobs is therefore that of the TP path plus the
  activation gather of the planned rows; a vocab-parallel cross-entropy
  that trades the `[rows, vocab]` all-gather for per-row partials is the
  deferred `logprob.topology-invariant` item of `numerics.md`.)
* Pipeline parallelism: the last stage scores the prompt logprobs (on its
  shard, under QCP) and `_pp_broadcast_output_tokens` carries
  `output_logprobs`, `input_token_logprobs` and the NaN guard's
  `output_nan_flags` to the other stages with the sampled tokens; every
  stage pairs the vector with its own mirrored plan, which
  `ModelExecutionResult` carries whether or not the stage scored the rows.
  The flags travel with the values they audit: only the last stage holds
  logits and prompt logprobs to flag (the other stages' guards see
  placeholder outputs and the rank-consistent target audit at most), and
  every stage's output processor must take the same abort-or-finish branch
  for a request, or the stages' schedulers disagree on it. Prompt logprobs
  are therefore no longer refused on a pipeline split
  (`supports_prompt_logprobs` depends on the narrowing-model check only).
* Communication buffers (`prepare_communication_runtime(max_forward_tokens)`)
  stay sized by the whole chunk under QCP, not `ceil(chunk / qcp)`: the
  all-gather / reduce-scatter legs' gathered side is the whole chunk on
  every rank (the dense and MoE legs gather the shard rows to the group), so
  only buffers of local rows could shrink, and the one model that prepares
  such buffers today does not shard queries. Revisit with a per-buffer
  audit when a sharding model allocates them.

**Head TP over the query shards** (`--attn-head-tp-size N` with
`--prefill-context-parallel-size N`). The query shards hold different rows,
so the head group of the decode-side layout (`docs/serving/parallelism.md`,
"Decode-side TP layouts under attention DP") applies to them unchanged:
`mapping.attn.head_tp_group == qcp_group` (the attention TP group),
`q_b_proj` / `kv_b_proj` / `o_proj` are head-sharded over it, and the
sharded extend forward runs the same exchange — the normalized q latent
token-all-gathered to the span, `q_b_proj` and the absorption on this rank's
head slice of every row, the heads-to-tokens all-to-all back to the shard
rows with every head, the prologue (RoPE on the shard's own positions, the
gathered KV write as on every QCP forward; there is no positions
collective), the sparse core over the gathered history with every head and
no LSE merge, the tokens-to-heads all-to-all, the local `w_vc`, and the
`o_proj` tail: row-parallel plus a token reduce-scatter to the shard rows, or
under `--tp-batch-invariant attn` the all-gather of the heads, the
column-parallel GEMM and an all-to-all back. One resolver
(`comm_manager.head_tp_row_counts`, the module's
`head_tp_leg_row_counts(ctx, num_rows, collective=)`) hands every leg its
per-rank counts: the DP tables under attention DP, the shard plan under QCP
(`row_counts` for the legs up to the core, `rows_for_collective(ctx.collective_num_tokens)`
after it — identical on an extend that does not narrow), so no model code
branches on the layout. The drafter's decode steps on this engine hold every
row on every rank: they exchange nothing — `head_tp_exchanges(ctx)` is False
without a shard, the one fork of the head-TP path, asked by every leg and
switching off the exchanges and nothing else — so they attend this rank's
head slice of every row through the same `attn_mqa` layer (which declares
every head; the DSA core reads the head count from the query, so no second
layer exists for the slice), the DCP arm as attention TP runs it, and the
tail all-reduces the row-parallel partials (or all-gathers the
batch-invariant hidden shards), so the layer's rows stay replicated. The two
forward forms of this engine are thus the sharded extend and the replicated
decode step, and nothing else runs on it: the startup autotune's dummy extend
carries the shard plan a real extend of its rows would
(`PrefillGraph.make_dummy_batch`, the model taking its slice as
`_run_target_forward` does), rather than an unsharded replicated-row extend
the contract does not list. The expanded (dense MLA) prefill keeps refusing
head TP: only the absorbed sparse prefill can take the exchange. The
alternative to this fork — sharding the drafter's decode rows by a plan too,
so every forward exchanges — needs a sharded DSA decode arm (history gathers
for decode rows) that does not exist; the fork is the contained form until
it does. The decode-only gates of the attention-DP
layout (`disaggregation_mode == "decode"`, the decode-shaped autotune, the
retraction-window generation budget) key on
`mapping.attn.head_tp_serves_decode_only`, which is False over the query
shards; the prefill role's own rules (`validate_qcp`) and the mapping (`head
TP == qcp`) gate this layout.

Eager only: the history gather runs in the attention break, so
`--prefill-context-parallel-size > 1` requires `--disable-prefill-graph`
(the executor also refuses to replay a prefill graph for a sharded forward).
A graph-capable form needs equal padding on every rank so the gathers are
even collectives inside the captured segment. MIXED rounds, attention DP,
non-prefill roles, a non-bf16 KV cache (the gathered write is
`latent_store`, native rows only) and dense / MoE groups that are neither 1
nor the attention TP width are refused at argument resolution
(`validate_qcp`); `AttnConfig` repeats the attention-family and KV-cache
checks where the config is built.

## Non-goals

Extend/mixed metadata keeps its dynamic-shape construction path
(`init_forward_metadata`), with `PrefillGraph` as its own capture story.
The write-location kernels stay pure functions (`paged/write_locations.py`).
QSA reuses the shared table fill without expansion; V4's token-shaped slot
mapping remains a separate consumer of the shared mapping helpers
(`cache-concepts.md` Principle 5).

## Regression gates

* `test/runtime/execution/test_kda_prefill_graph_cache.py` is registered in
  `runtime-1gpu`; its direct-script entry point runs pytest. It covers request
  padding, capacity selection, metadata refresh and checkpoint/state isolation.
  Native CuTeDSL replay tests run on NVIDIA SM100/SM103 and skip other devices;
  missing native dependencies on a supported device are errors, not skips.
* `test/runtime/test_unified_decode_path.py` — eager refresh and padded
  replay refresh produce identical live-request contents over the same
  buffers; lazy above-ladder views are pointer-stable; the graph_ptr_guard
  walk reports a rebound tensor by path and pins every tensor under the
  slots; FlashMLA's
  tile schedule stays off the views (capture keeps its object alive, replay
  refresh leaves it alone, eager refresh and every drafter seq_lens edit
  bind a fresh one, in-graph edits keep theirs alive); leaf capture/refresh
  signature conformance.
* `test/runtime/test_deepseek_v4_config.py` — a V4 replay refresh leaves
  every address the capture recorded in place under the guard, the `cache`
  slot's group tables included, for the target's packed views and the
  draft's borrowed step views.
* `test/runtime/execution/test_draft_target_wiring.py` — the drafter's
  target-forward hook: DFLASH arms its capture sink on the context only
  under its overlap gate (not on mixed rounds, not in graph warmup), the
  sink folds the taps into the projection and writes the KV once; the
  executor calls the hook before the target forward
  (`test_model_executor_cache_state.py`); the target hands taps to the
  forward's sink in concat order (`test_dspark_config.py`).
* `test/runtime/test_cache_group_router.py` — router slot math, expansion,
  padding, placeholder delivery, per-group dispatch, draft window
  publication and address stability.
* `test/runtime/test_qsa_backend.py` — independent QSA raw-group metadata,
  target-only verify workspace, and live cache writes across eager execution
  and CUDA graph replay; `test_qsa_verify_lifecycle.py` — the Qwen4-Exp root
  commits GDN/PLE on decode and QSA on decode/mixed, using real acceptance
  rows once after execution, including PLE without GDN and failure cases.
* `test/runtime/test_qwen4_backend_composition.py` — local consumer selection,
  workspace accounting, draft hooks through the attention composite and one
  PD cache step per layer.
* `test/runtime/test_cudagraph_per_group.py`,
  `test_group_write_locations.py` — per-group padding wiring and the
  write-location edge cases (holes, overflow, MTP re-anchor) on the unified
  path.
* `grep -rn "init_forward_metadata_replay_cuda_graph\|is_all_greedy" python/`
  must stay empty.
* `grep -rn "ENABLE_CP\|CP_METADATA" python/` must stay empty — the
  module-global context-parallel switch and its process-wide metadata holder
  were deleted; a parallel layout is a `Mapping` fact from an explicit server
  argument and per-forward row metadata rides `ForwardContext`.
* Every `init_forward_metadata` signature carries `query_shard` with no
  default (`test_unified_decode_path.py` binds it on every runner-facing node
  and leaf; leaves also take `page_table_cpu`), and
  `grep -rn "input_num_tokens" python/tokenspeed/runtime/models/longcat_flash.py
  python/tokenspeed/runtime/models/base/causal_lm.py` stays empty — a model
  under query context parallelism reads its row count from its tensors or
  `ctx.query_shard`, never from the scheduler's chunk count
  (`test/runtime/distributed/test_query_shard.py`).
* `grep -rn "ctx.accept_lengths\|ctx.draft_seq_lens_buf\|_apply_correction"
  python/` must stay empty — the step-0 accepted prefix is published through
  `ctx.draft_narrowing.publish_accepted_prefix()`, never computed in a model
  (`test/runtime/test_draft_advance_seqlens.py`).
* `grep -rn "ctx.dsa_\|dsa_swa_slot_mapping\|dsa_compressor_slot_cache"
  python/` must stay empty — the layer-shared sparse top-k and V4 slot
  mappings are backend scratch (`sparse_topk`, `slot_mappings`), cleared by
  every metadata build (`test_cache_group_router.py`,
  `test_deepseek_v4_slot_mappings.py`, `test_deepseek_v4_config.py`).
* `grep -rnE '^\s+extend_(seq|prefix|replay|prompt)_lens(_cpu)?: torch\.Tensor \| None,|
  extend_with_prefix: bool = False' python/tokenspeed/runtime/layers/attention/backends/`
  must stay empty — no `init_forward_metadata` parameter in the extend
  bundle is optional or defaulted (`test/runtime/test_unified_decode_path.py`
  binds the runner call shape against every runner-facing node and every
  leaf). Metadata dataclasses may still hold `None` for fields a decode
  batch does not carry; the contract is about the call, not the record.
* `grep -rn "select_out_cache_loc\|DraftPageStaging\|tables_self_padding\|
  cache_active_pages_must_be_real\|engine_owned_group_ids" python/` must
  stay empty — write locations have one accessor (`write_locations`), and
  table delivery has no capability flags.
* `grep -rn "out_cache_loc" python/tokenspeed/runtime/models/` matches only
  `write_locations(...)` fetches and the helper parameters they feed —
  never a forward-chain parameter threaded from the runner.
* `grep -rn "AttentionArch.DSA\|qwen4_exp_has_side_state"
  python/tokenspeed/runtime/execution/` must stay empty — backend-imposed
  graph restrictions are `cuda_graph_support` declarations
  (`test/runtime/test_cudagraph_support_resolution.py`).
* `grep -rn "def init_forward_metadata_capture_cuda_graph" python/` matches
  only the defaults (`backends/base.py`, `paged/base.py`, `paged/router.py`)
  and the sanctioned
  overrides listed in "Capture is inherited".
* New backends implement `refresh_decode_metadata` + `init_cuda_graph_state`;
  capture is inherited from the base default (idle refresh). Only a
  kernel-imposed capture asymmetry justifies an override.
