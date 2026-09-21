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
there are no extend requests) and `extend_with_prefix` is a plain `bool`.
Runner-facing nodes additionally take two host-only facts the scheduler
knows and only V4.1 plans from: `extend_replay_lens_cpu` (how many leading
rows of each extend re-feed already-cached positions — bounded replay,
`docs/design/scheduler.md`) and `extend_prompt_lens_cpu` (the whole prompt
length, so the backend can tell a prompt-completing chunk from an open one).
Leaves never see them: a paged leaf writes every input row unconditionally,
so the router and every other runner-facing node call
`reject_bounded_replay` and fail loud on a non-zero replay instead of
rewriting rows the prefix hit already shares. No default values: the runner
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
  this same pass. Their verify workspaces remain tied to one pool: publishing
  that pool again preserves their buffers; a different pool is rejected before
  any child publishes. Replacing such a pool requires rebuilding the composite.
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
  `capture_graphs()`: `configure_runtime`, `init_cuda_graph_state` and
  `preallocate_verify_workspace` all ran before the executor was returned,
  so the orchestrator re-runs them itself.
* A rebind is an operation between the executor's construction and
  `ModelExecutor.capture_graphs()`, owned by the orchestrator a later change
  adds; nothing in the backend tree guards against a rebind at another time.
  That orchestrator releases both graph owners' captures first (the captured
  graphs record the buffers a publish drops, and eager kernels cache
  pointers they allocated inside a capture, such as flashinfer's trtllm-gen
  MoE runner and Qwen4-Exp's uniform index bundles), unfreezes the device-
  global workspace pool the executor froze before capturing, rebinds the
  trees, re-runs `bind_cache_groups` and the initialisation sequence above,
  freezes the workspace again and captures again.

### Padding contract

`bs` is the request count being prepared (the padded graph batch under
replay); `actual_bs` is the live-request count. Requests in `[actual_bs, bs)`
are padding and must resolve to the null page 0 / dummy slot so they never
touch a live request's cache. Eager passes `bs == actual_bs` (unpadded — no
wasted FLOPs);
`actual_bs == 0` is the idle replay. Eager idle bypasses the wrapper entirely
(`execute_idle_forward` calls `model_runner.forward(IDLE)` directly).

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
on replay. PLE's uniform index bundles are reused during capture only; eager
prefill and decode construct their indices through the same builder outside
the capture pool.

GDN verify shares memoized scratch seed indices (`i * (T + 1)`) between conv
and recurrent reads in eager and captured forwards. FlashInfer FP32 MTP may
use uninitialized output and a placeholder for a disabled intermediate cache:
live rows are fully written, while negative padding rows skip state access
and leave output undefined. Consumers must ignore padded output; enabled
intermediate caches always require real storage.

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

Gated RMSNorm preloads weights only with `weights_independent`; a contiguous
copy disables this preload. At RSAG-to-AR boundaries the next combine-norm
preloads the all-gathered residual before its wait, so that collective must
not trigger early. FlashInfer adapters preserve the upstream CuTe body and
keep PDL compilation caches separate.

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

### Prefill graphs around a row narrowing

A prefill forward whose row count drops once, at a fixed layer, by an amount
that is not a function of the token bucket cannot be one token-shaped
breakable graph. DeepSeek-V4.1 is the case: its CED decoder (layer 20 on)
runs on a per-request tail of the prefill rows (`decoder_view()` — a
completing prompt's last window, one row for an open chunk, every decode
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
"Write locations have one owner").

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
The leaf reuses MHA's KV writer; already-quantized FP8 inputs retain direct
stores to avoid rescaling. Sparse attention has no MXFP8 block-scale input.
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

Qwen4-Exp attention callers pass `topk_indices` explicitly, using `None` for
dense attention. Sparse QSA requires `save_kv_cache=True` because it always
writes the full KV cache; the dense fallback honors the caller's flag.
Draft step zero still preserves the dense decode-context
and KV-recording override, while QSA keeps its original context and narrows
the selected top-k rows with the queries.

The QSA API preserves `decode_query_lengths`: uniform decode/verification
uses a positive width, while prefill and mixed/ragged queries use `None`.
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
compute nor thread location vectors. `PagedAttention.forward`,
`AttentionBackend.forward`, `model_runner.forward` and every model forward
chain carry no `out_cache_loc` parameter; `InputBuffers` has no location
buffer; `fill_input_buffers` takes no table.

* **Extend**: `init_forward_metadata` computes each group's span over the
  stacks (`[sum(extend_seq_lens)]`, request-major); `write_locations(layer,
  EXTEND)` returns exactly that span.
* **Decode / verify**: `refresh_decode_metadata` publishes the token-major
  `[bs * N]` window views (`decode_write_locations`, pointer-stable per
  bs — the graph records them through the leaves' KV writes, and the
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
* **Model-side direct writes** (fused RoPE prewrite, MLA latent
  `set_mla_kv_buffer`, V4 group writes, QSA) fetch
  `ctx.attn_backend.write_locations(layer, mode)` immediately before the
  write. A model path that writes multiple mode windows in one shot (the
  MLA draft's step-0 whole-batch write) concatenates the EXTEND span and the
  DECODE window — eager-only, MIXED rounds never run under a captured
  graph. The router performs the same composition when a draft step-0
  forward locally dispatches as DECODE while retaining the round's full K/V
  rows; target MIXED decode halves and later draft steps keep their ordinary
  decode-only windows. V4 composes the shared token-shaped resolve
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
the producer from the pipeline configuration alone (`pp_size > 1` with a
speculative algorithm) and requires the draft model to implement
`DSparkContextModel`. Off the pipeline every tap is local, so the drafter keeps
its concatenated projection and its own context writes -- including the
quantization-aware path, since raw per-tap weight slicing is not a quantized
linear operation. PP drafts require unquantized projection weights.

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
its taps on a row subset reports it as `ctx.captured_rows`
(`CapturedRows(positions, prefill_spans)`). V4.1's CED narrowing is the one
producer — its taps sit in layers 37–39 and hold one row per open chunk and
the last window of every completing one, so DSpark's prefill seeding
(`_seed_prefill_windows`) reads the spans and positions from there instead
of the input-length mirror. A target with one captured row per input row
leaves it `None`, and the drafter keeps its buffer-based layout.

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

## Experimental KDA prefill subgraphs

### Capturing KDA in the outer graph

Supported pure-extend forwards use `prepare_prefill_metadata` before eager
execution, startup capture and replay. This consumer-stream seam builds or
refreshes `KdaPrefillMetadata` with the selected token and request capacities.
Eager execution uses the live count; replay may round up to a captured count.
The same metadata contract controls scan capacity, checkpoint packing and
output restoration in every case; there is no temporary metadata binding or
mutable inline flag. Only startup capture retains the metadata's addresses.
Uncaptured shapes use temporary storage through the same builder.

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

The private KDA metadata overrides only the packed execution extent; real
host lengths and GPU boundaries still agree. An explicit
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
