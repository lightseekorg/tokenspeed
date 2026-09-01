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
seq_lens, *, forward_mode, page_table, num_extends, for_graph_replay,
**cache_kwargs)` is the ONLY way decode metadata is prepared:

* **capture** (`init_forward_metadata_capture_cuda_graph`) is INHERITED: the
  base default binds the per-bs views (`bind_decode_views`, implemented per
  backend over its `_decode_views` builder) and runs the idle-refresh arm
  (`actual_bs=0`,
  `for_graph_replay=True`) against the runner-seeded seq_lens and the
  address-stable staged page_table — never live tables. Only a genuine
  capture-only asymmetry overrides it (see "Capture is inherited");
* **replay** = refresh (`for_graph_replay=True`) + `graph.replay()`;
* **eager decode** = refresh (`for_graph_replay=False`) + the same forward
  Python the graph recorded.

`init_forward_metadata` serves extend/mixed (and idle warmup) ONLY; a pure
DECODE call raises. There is deliberately no fresh-allocation decode arm
anywhere. `init_forward_metadata_replay_cuda_graph` no longer exists.

### Buffer sizing: the ladder is a performance subset, never a capacity limit

`ForwardStepRunner` distinguishes `max_capture_bs` (top of the capture ladder,
bounded by `max_cudagraph_capture_size`) from `max_decode_bs`
(`max_num_seqs // dp_size`, floored at `max_capture_bs`). Persistent decode
buffers are sized by `max_decode_bs` — `init_cuda_graph_state` runs
unconditionally at wrapper construction, `enforce_eager` included. A decode
above the ladder runs the same refresh with no graph; it is a first-class
path, not a fallback.

### Padding contract

`bs` is the row count being prepared (the padded graph batch under replay);
`actual_bs` is the live-request count. Rows in `[actual_bs, bs)` are padding
and must resolve to the null page 0 / dummy slot so they never touch a live
request's cache. Eager passes `bs == actual_bs` (unpadded — no wasted FLOPs);
`actual_bs == 0` is the idle replay. Eager idle bypasses the wrapper entirely
(`execute_idle_forward` calls `model_runner.forward(IDLE)` directly).

### Pointer-stable per-bs views from one builder

Per-bs metadata objects (`decode_cuda_graph_metadata[bs]` and friends) are
views over the persistent buffers, built by a single `_decode_views(bs)`
builder shared by capture and refresh, cached per bs. A bs never captured
(above-ladder decode, enforce-eager) builds its views lazily on first refresh
— no new storage, one-time cost. Views must be pointer-stable: a captured
graph holds their addresses forever.

### `for_graph_replay` is for graph-mechanics asymmetries only

`for_graph_replay=True` means a graph is in play — live replay AND the base
default capture (which runs the idle-refresh arm). Two sanctioned branches
on it exist:

* FlashMLA's tile schedule: flash_mla freezes its schedule on the first
  kernel call against a `FlashMLASchedMeta`, so eager refresh must swap in a
  fresh sched-meta each step, while the captured graph re-runs the recorded
  schedule-build against the live seq_lens buffer.
* DFLASH block-arm seeding (`not for_graph_replay or actual_bs == 0`): the
  drafter's recorded `fill_block_decode_seq_lens` rewrites the block-end
  lengths inside every replay, so only eager steps and the capture-time
  seeding fill them from Python.

Do not branch on this flag for anything a shared in-place refresh can
express.

### Capture is inherited

`init_forward_metadata_capture_cuda_graph` has a base default — bind the
per-bs views (`bind_decode_views`), then run the idle-refresh arm
(`actual_bs=0`, `for_graph_replay=True`) — and that default IS the capture
for every backend except a closed list of sanctioned overrides, each tied to
something the idle refresh cannot express:

* **FlashMLA** (slim, calls super): installs the keepalive tile-schedule
  object whose schedule-build the graph records;
* **DeepseekV4**: capture-phase placeholder-table validation
  (`cache_active_pages_must_be_real`) and the packed `tokens_per_req` row
  machinery;
* **Mamba** (`MambaAttnBackend`): the warmup kernels need the arange
  query-start-loc, which the idle refresh deliberately zeroes;
* **HybridLinearAttnBackend**: a two-line fan-out so Mamba's real capture is
  reached;
* **Inkling**: conv-state seeding; its refresh hard-requires conv tables.

Bind-only overrides (`MLAAttnBackend`, `CuteDSLMLABackend`) exist solely to
latch `_cache_groups_bound` before the views are built. A new backend
implements `refresh_decode_metadata` + `init_cuda_graph_state` and inherits
capture; a new override must name its kernel-imposed asymmetry here. Every
capture signature must accept the runner kwarg set (`cache_group_ids`,
`page_table`, `**kwargs`) — pinned by
`test_unified_decode_path.py::CaptureSignatureConformanceTest`.

### Graded CUDA-graph support

A backend's static graph capability is a class attribute,
`cuda_graph_support: CudaGraphSupport(decode_graph, prefill_graph)`, never a
scattered executor-side arch check. `ModelExecutor.__init__` AND-composes it
over the target and draft `child_backends()` trees once
(`resolve_cuda_graph_support`), logs every culprit class, and downgrades the
two graph subsystems (`ForwardStepRunner.disable`, `PrefillGraph.disable`).
Current declarations: `DSABackend` and `Qwen4ExpMambaAttnBackend` disable the
prefill graph (rationale comments live on those classes).

Rules: declarations are static "never works" facts — a runtime prefill
capture failure is FATAL (no silent eager degrade: a family that cannot
capture must declare it, or the boot dies). Resolution is device-side at startup and
class-attribute-driven, so every DP rank derives the same answer
(event-loop.md). `disable_prefill_graph` in the config carries user intent
only. `decode_graph=False` still requires `refresh_decode_metadata` and
`init_cuda_graph_state` — eager decode runs the same unified path.

### One draft metadata contract

The draft backend's decode metadata comes from `refresh_decode_metadata` and
NOWHERE else — the same two steps in every round:

* **decode round**: target refresh, then draft refresh over the drafter-owned
  `draft_seq_lens_buf` (freshly seeded from the batch seq_lens);
* **extend/mixed round**: draft prefill init reading the accepted-prefix
  seq_lens view (never the mutable draft buffer), then the same draft refresh
  with plain 1-token rows — deliberately NOT the packed verify width, which
  would take V4's packed-decode arm and clobber `forward_prefill_metadata`.

Backends' `init_forward_metadata` must NOT double-fill draft decode metadata
as a side effect (the deleted `is_extend() and self.is_draft` arms); the
mixed/idle decode arms that remain serve the target's decode rows only.
Drafters republish their in-loop seq_lens edits explicitly each step via
`advance_draft_forward_metadata` (Eagle) / `update_draft_forward_metadata`
(vanilla MTP frontier re-anchor) — metadata never aliases a buffer the
drafter mutates behind the backend's back.

Both steps run unconditionally — there is no per-drafter opt-out. What makes
that safe is the slot discipline: init writes prefill-slot metadata, refresh
writes decode-slot metadata, and forwards read the slot matching their mode
(`forward_prefill_metadata` / `forward_decode_metadata`; Inkling's conv
wrapper mirrors this with `conv_prefill_metadata` / `conv_decode_metadata`).
A round that runs no decode steps (vanilla MTP re-runs prompt rows as EXTEND
depths) leaves the refreshed decode slot unread; a block drafter (DFLASH)
re-runs the same refresh inside each block-decode step, overwriting it. A
backend that lets one call clobber the other slot's metadata is in breach —
that, not drafter special-casing, is the invariant to fix.

### PD decode nodes

A PD decode-only node never runs an extend forward, so latches set on the
extend path (`_cache_groups_bound`) stay False there. Refresh must therefore
bind the group tables whenever they are delivered — never gate on an
extend-latched flag — otherwise the kernels read the null page instead of
the transferred KV. This rule predates unification and now protects eager
decode too. (`_cache_contract_bound` is gone: every LCM pool publishes a
cache contract, so the target allocates its write-location buffer
unconditionally and drafts are gated structurally on `is_draft`.)

### Sampling has no greedy branch

Greedy requests normalize to `top_k=1` in `SamplingParams.__post_init__`; the
pool-indexed sampling route serves them, which is exactly what the captured
graph records. `SamplingBatchInfo.is_all_greedy` and the eager-only argmax
branches were deleted. Equivalence (top_k=1 == argmax, ties excepted) is
pinned by `test/runtime/sampling/test_greedy_route_equivalence.py`.

### Non-speculative serving is the N == 1 case, not a second path

One sampling rule for every batch: **prefill rows sample, decode rows
verify** (`ModelExecutor._run_sampling`). The decode candidate window is
always `[num_decodes, output_length]` (`_decode_candidates`, a persistent
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
the drafter is the *draft model's existence* — draft backend refresh,
DraftPageStaging, the drafter loop itself — not the sampling or metadata
shape of the target.

### Outputs are persistent-buffer slices on both paths

`sample()` and `verify()` land their outputs in each sampling backend's
packed output region (`_output_pack_buf`), so `get_packed_output_d2h`'s
single-D2H fast path fires on eager and replay alike.

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
replays pay one bool check). Per-step-mutable objects a replay never reads
through Python are exempted via `graph_unstable_metadata_fields` (two
occupants: FlashMLA's eager tile schedule, and V4's `cache` slot, which
refresh replaces wholesale each step). What unification still can NOT
test: mempool reuse and hostfunc semantics — the e2e regression matrix keeps
graph-on and graph-off configurations for this reason.

## One block-table route, one unit

Every backend consumes the wrapper's per-group `block_tables` kwarg — raw
scheduler tables in block-granularity page ids — and there is no capability
flag saying so (`uses_cache_groups` was deleted once it was universally
True). The invariant replacing it: **any table a backend receives — the
per-group `block_tables`, the staged draft `page_table`, a warmup
placeholder — carries raw scheduler pages; kernel-page expansion happens
inside the backend, through the one shared `expand_history_table`
(`cache_group_geometry.py`; the base `set_cache_pool` learns the
full-history grain from the pool's specs into one `CacheGroupGeometry`
value object)**. The routing surface lives on `AttentionBackend` itself:
the write-location slot math lives in `group_write_locations.py` as pure
functions, and the stacked per-group CUDA-graph buffers live in
`group_graph_buffers.GroupGraphBuffers`, composed at graph-state init.
Inside the stack, page vocabulary stays with paged-KV consumers
(`cache-concepts.md`): attention-consumed groups get consumer-page-grain
`page_tables` plus write-location views, while wrapper-owned (Inkling conv)
groups ride the stack tail as block-granularity `owned_block_tables` with
no location views (the wrapper keeps its own write-loc machinery).
`CacheBatchMetadata` no longer carries a `kernel_table` expansion;
`cache_metadata` still travels to V4 (bespoke multi-group slot mapping) and
KDA state paging only. `cache_active_pages_must_be_real` remains a separate
axis — it marks backends that validate live-page geometry and so need real
capture placeholder tables (V4), not who supplies tables.

Table delivery is guarded at the one dispatch point, not per backend:
`ForwardStepRunner._decode_stale_table_guard` fails a live decode
(`actual_bs > 0`, eager and replay alike, every backend family) whose
`block_tables` omit any published group — the persistent decode buffers
would otherwise serve stale pages. Extend/mixed keep the wrapper's
`>1 groups` guard (a single group's table IS the single table, so the
fallback is legal there). The per-backend `_replay_stale_guard` is gone.

`DraftPageStaging` survives but is no longer a mapping owner: its publish is
a pure copy + padded-row scrub into the one address-stable buffer the
drafters' in-graph write-location kernels record at capture
(`DraftPageStaging.out_cache_loc_uniform`; per-forward group tables are
fresh tensors, so an address-stable shadow is physically required). The
write-location math is page-size invariant — `table[i, pos // P] * P +
pos % P` addresses the same token at any page size — so the staging resolves
absolute slots directly over the raw table, exactly like
`fill_input_buffers`' out_cache_loc path. (`CacheView`, the former wrapper
that carried this math, is gone — its only production retention was
`full_history`, so the class and the never-reached sliding-ring kernel
branch were folded away.)

Deleted flags, for the record: `needs_group_block_tables`,
`cache_group_tables_replace_draft_page_table`,
`reads_staged_draft_page_table`, and finally `uses_cache_groups` itself.
None carried information not already implied by the pool's published specs.

## Non-goals

Extend/mixed metadata keeps its dynamic-shape construction path
(`init_forward_metadata`), with `PrefillGraph` as its own capture story.
The group mixins are gone — the routing surface lives on
`AttentionBackend` (per-group selection, GroupGraphBuffers composition) and
the two write-location kernels stay two sets of pure functions
(`group_write_locations.py`); unifying that math with V4's bespoke slot
mapping is the final block-table-owner milestone (`cache-concepts.md`
Principle 5).

## Regression gates

* `test/runtime/test_unified_decode_path.py` — eager refresh and padded
  replay refresh produce identical live rows over the same buffers; lazy
  above-ladder views are pointer-stable; the graph_ptr_guard walk reports a
  rebound tensor by path and honors `graph_unstable_metadata_fields`.
* `test/runtime/test_cudagraph_per_group.py`,
  `test_group_write_locations.py` — wrapper padding wiring and per-group
  write-location math on the unified path.
* `grep -rn "init_forward_metadata_replay_cuda_graph\|is_all_greedy" python/`
  must stay empty.
* `grep -rn "AttentionArch.DSA\|qwen4_exp_has_side_state"
  python/tokenspeed/runtime/execution/` must stay empty — backend-imposed
  graph restrictions are `cuda_graph_support` declarations
  (`test/runtime/test_cudagraph_support_resolution.py`).
* `grep -rn "def init_forward_metadata_capture_cuda_graph" python/` matches
  only the base default and the sanctioned overrides listed in "Capture is
  inherited".
* New backends implement `refresh_decode_metadata` + `init_cuda_graph_state`;
  capture is inherited from the base default (bind views + idle refresh).
  Only a kernel-imposed capture asymmetry justifies an override.
