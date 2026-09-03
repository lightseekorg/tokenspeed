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
runner-facing (`base.py`: router, V4, Mamba/KDA, composites) and leaf
(`paged.py`) alike: `extend_seq_lens`, `extend_seq_lens_cpu`,
`extend_prefix_lens`, `extend_prefix_lens_cpu` are plain `torch.Tensor`
(`[>= num_extends]` rows; empty, never `None`, when there are no extend
rows) and `extend_with_prefix` is a plain `bool`. No default values: the
runner passes the `[:num_extends]` slices of its input buffers on every
call (the idle replay passes the empty `[:0]` slices), so a node that reads
a field can never see a silently-defaulted one. This is deliberate — a
`= False` default once hid `extend_with_prefix` being swallowed by a
composite's `**kwargs`, and FlashMLA planned a ragged prefill for a
prefix-cached batch.

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

Per-bs metadata objects (each leaf's `_decode_views_by_bs[bs]`, the router's
`decode_write_locations` views) are views over the persistent buffers, built
by a single per-bs builder shared by capture and refresh, cached per bs. A bs
never captured (above-ladder decode, enforce-eager) builds its views lazily
on first refresh — no new storage, one-time cost. Views must be
pointer-stable: a captured graph holds their addresses forever.

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
* **HybridLinearAttnBackend / MSAHybrid**: pure fan-out to their children so
  the real captures above are reached.

A new backend implements `refresh_decode_metadata` + `init_cuda_graph_state`
and inherits capture; a new override must name its kernel-imposed asymmetry
here. Leaf capture/refresh signatures are pinned by
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
drafter mutates behind the backend's back. Those two hooks are deliberately
seq-lens-only: Eagle's step-0 rejected-tail correction fires
`advance_draft_forward_metadata` BEFORE the step-0 attention has consumed
the verify-shaped write window, so the write-window publication is a
separate, explicit drafter-loop call (`publish_draft_step_locations`, see
"Write locations have one owner").

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

**V4's packed-draft deviation (documented):** a V4 draft's packed verify
round legitimately writes BOTH slots at its end — the bs*N packed views ride
the prefill slot (the step-0 shape carrier; `_select_decode_metadata`
resolves them there through a DECODE-mode-gated fallback), and the bs-row
step views own the decode slot. Capture and replay refresh reach that state
through the SAME publisher (`_publish_draft_round`), so replay reproduces
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
replays pay one bool check). Per-step-mutable objects a replay never reads
through Python are exempted via `graph_unstable_metadata_fields` (two
occupants: FlashMLA's eager tile schedule, and V4's `cache` slot, which
refresh replaces wholesale each step). What unification still can NOT
test: mempool reuse and hostfunc semantics — the e2e regression matrix keeps
graph-on and graph-off configurations for this reason.

## One block-table route: router + leaves

The layering between the scheduler's block vocabulary and the kernels' page
vocabulary is fixed, with exactly one conversion point:

| layer | sees | never sees |
|---|---|---|
| C++ scheduler | per-group `BlockTable`s: rows in `block_granularity` logical index, entries are `CacheBlock` ids | kernel pages, backends |
| bridge (`CacheBatchMetadata`) | contract-ordered group ids; `{gid: [bs, W_g]}` views over one packed int32 upload | pages, backends |
| **`CacheGroupRouter`** | group geometry (`CacheGroupGeometry`), each leaf's `kernel_page_size`, expansion, padding, ALL write-location slot math | kernel calls |
| paged leaf (`PagedAttentionBackend`) | `page_table` (kernel pages, batch-ordered, padded), `seq_lens`, `out_cache_loc` | groups, block tables, contracts, draft/target table provenance |
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
`[G, max_bs, stack_max_num_pages]` kernel-page table stack (each group's row
expanded to its leaf's `kernel_page_size` and padded to the leaf's
`max_num_pages`; the stack's column count is the widest group's) and the
`[G, max_bs * N]` decode write-location stack. Both are allocated once and
refilled in place: leaves copy their view out, while the decode write-slot
views, the block drafters' `draft_history_view` and the QSA indexer's group
tables read the stack storage inside captured graphs. The fill is one expand
launch per group with plain scalar arguments (scheduler block count, source
stride, live rows) — no device-side metadata tensor, because the per-step
pinned staging + H2D it would need lands on the bs=1 latency path; padding
rows (`[actual_bs, bs)`) and each group's column tail resolve to null page 0.
The bridge's `{gid: view}` dict is the router's input; the router does not
depend on the views sharing one storage. The slot math lives in
`write_locations.py` as pure functions with one invariant: `slot = table[req, pos // P] * P + pos % P` is page-size
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
`block_decode_active` (spec verify geometry as a clamp floor), FlashMLA's
`for_graph_replay` tile-schedule swap, and the MLA family's `num_extends`
decode-row slicing (`override_num_extends`).

One side channel exists beyond the table: `set_request_slots(req_pool_indices)`,
a no-op by default, which the router calls on every leaf after each
metadata build (extend init, decode refresh, capture seeding). It serves a
leaf that owns per-request side state indexed by pool slot — DSA's KPool
tails — and doubles as that state's per-forward reset point. Paged KV
leaves ignore it; it carries no table or page vocabulary.

## Write locations have one owner

`write_locations(layer, forward_mode)` on the top-level backend is the ONLY
accessor for KV write slots — models, drafters and the runner neither
compute nor thread location vectors. `PagedAttention.forward`,
`AttentionBackend.forward`, `model_runner.forward` and every model forward
chain carry no `out_cache_loc` parameter; `InputBuffers` has no location
buffer; `fill_input_buffers` takes no table.

* **Extend**: `init_forward_metadata` computes each group's span over the
  stacks (`[sum(extend_seq_lens)]`, request-major); `write_locations(layer,
  EXTEND)` returns exactly that span.
* **Decode / verify**: `refresh_decode_metadata` publishes the token-major
  `[rows * N]` window views (`decode_write_locations`, pointer-stable per
  bs — the graph records them through the leaves' KV writes, and the
  pointer guard walks this slot). A MIXED round's draft refresh sets
  `_decode_row_offset = num_extends` so DECODE reads skip the extend rows.
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
* **Model-side direct writes** (fused RoPE prewrite, MLA latent
  `set_mla_kv_buffer`, V4 group writes, QSA) fetch
  `ctx.attn_backend.write_locations(layer, mode)` immediately before the
  write. A model path that writes multiple mode windows in one shot (the
  MLA draft's step-0 full-row write) concatenates the EXTEND span and the
  DECODE window — eager-only, MIXED rounds never run under a captured
  graph. V4 composes the shared token-shaped resolve
  (`page_table.group_slot_mapping_from_raw`) over its own group tables; a
  degraded mapping fails closed to `-1` (skipped write), never to a raw
  fallback vector.

## Non-goals

Extend/mixed metadata keeps its dynamic-shape construction path
(`init_forward_metadata`), with `PrefillGraph` as its own capture story.
The write-location kernels stay pure functions (`write_locations.py`);
unifying that math with V4's bespoke slot mapping remains the final
mapping-owner milestone (`cache-concepts.md` Principle 5 — owners are now
down to the router and V4).

## Regression gates

* `test/runtime/test_unified_decode_path.py` — eager refresh and padded
  replay refresh produce identical live rows over the same buffers; lazy
  above-ladder views are pointer-stable; the graph_ptr_guard walk reports a
  rebound tensor by path and honors `graph_unstable_metadata_fields`; leaf
  capture/refresh signature conformance.
* `test/runtime/test_cache_group_router.py` — router slot math, expansion,
  padding, placeholder delivery, per-group dispatch, draft window
  publication and address stability.
* `test/runtime/test_cudagraph_per_group.py`,
  `test_group_write_locations.py` — per-group padding wiring and the
  write-location edge cases (holes, overflow, MTP re-anchor) on the unified
  path.
* `grep -rn "init_forward_metadata_replay_cuda_graph\|is_all_greedy" python/`
  must stay empty.
* `grep -rnE '^\s+extend_(seq|prefix)_lens(_cpu)?: torch\.Tensor \| None,|
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
  only the defaults (`base.py`, `paged.py`, `router.py`) and the sanctioned
  overrides listed in "Capture is inherited".
* New backends implement `refresh_decode_metadata` + `init_cuda_graph_state`;
  capture is inherited from the base default (idle refresh). Only a
  kernel-imposed capture asymmetry justifies an override.
