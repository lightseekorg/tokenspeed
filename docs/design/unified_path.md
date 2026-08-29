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

* **capture** (`init_forward_metadata_capture_cuda_graph`) allocates/binds
  per-bs views over the persistent buffers and seeds safe values — never live
  tables;
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
buffers are sized by `max_decode_bs` — `init_backend_cuda_graph_state` runs
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

### `for_graph_replay` is for kernel-imposed asymmetries only

The only sanctioned use is FlashMLA: flash_mla freezes its tile schedule on
the first kernel call against a `FlashMLASchedMeta`, so eager refresh must
swap in a fresh sched-meta each step, while the captured graph re-runs the
recorded schedule-build against the live seq_lens buffer. Do not branch on
this flag for anything a shared in-place refresh can express.

### Refresh ordering: target before draft

The wrapper's `_prepare_decode_metadata` refreshes the target backend before
the draft. A draft whose kv-indices buffer is aliased to the target's
(`_page_table_aliased`) reads what the target's refresh populated; reordering
silently reads stale pages.

### PD decode nodes

A PD decode-only node never runs an extend forward, so latches set on the
extend path (`_cache_groups_bound`) stay False there. Refresh must also fire
on the registry-set `_cache_contract_bound` — otherwise the kernels read the
null page instead of the transferred KV. This rule predates unification and
now protects eager decode too.

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
`graph_key` lookup, output-buffer re-slicing, and the `ctx.bs` save/restore.

What unification can NOT test: address-freezing bugs (capture recorded a
pointer the refresh no longer writes), mempool reuse, hostfunc semantics.
The e2e regression matrix keeps graph-on and graph-off configurations for
this reason.

## Non-goals

Extend/mixed metadata keeps its dynamic-shape construction path
(`init_forward_metadata`), with `PrefillGraph` as its own capture story.
The four block-table mapping owners named in `cache-concepts.md` Principle 5
remain four; unifying them is a separate milestone this refactor makes
smaller (the MLA-family refresh bodies are now structurally identical).

## Regression gates

* `test/runtime/test_unified_decode_path.py` — eager refresh and padded
  replay refresh produce identical live rows over the same buffers; lazy
  above-ladder views are pointer-stable.
* `test/runtime/test_cudagraph_per_group.py`,
  `test_group_write_locations.py` — wrapper padding wiring and per-group
  write-location math on the unified path.
* `grep -rn "init_forward_metadata_replay_cuda_graph\|is_all_greedy" python/`
  must stay empty.
* New backends implement `refresh_decode_metadata` + a seed-only capture; the
  base class raises otherwise.
