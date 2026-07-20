# Rollout Routing Replay (R3)

**R3** ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)) stabilizes RL
training of Mixture-of-Experts models. During a rollout the inference router and
the training router pick experts *independently*; tiny numerical differences
flip a token's top-k selection, which inflates the KL between the training and
inference policies and destabilizes learning. R3 removes that mismatch: the
engine records the per-token, per-layer top-k expert ids it selected during the
rollout, and the trainer **replays** that exact selection in its forward pass.
Reported effect: train/inference KL cut ~50%, smoother curves. R3 integrates
with slime and veRL.

## Design: store routing in the KV memory pool

The distinctive choice here (vs. vllm-ascend's transient
`RoutedExpertsCapturer`) is to persist routing in a pool **indexed by the KV
slot**, alongside the KV cache.

Why: on a **prefix-cache hit** the shared prefix tokens are dropped from the
forward entirely — their KV is reused and their MoE layers never run (see
`runtime/execution/cache_loc_kernel.py`; `out_cache_loc` covers only *new*
tokens). A transient per-forward capturer therefore has **no routing for the
prefix**, and you'd have to force a re-forward to recover it. If routing instead
lives in a slot-indexed pool, it *follows the KV*: a prefix hit that reuses
slots also reuses the routing captured when those slots were first written —
zero recompute, and guaranteed consistent with the reused KV.

Per-token cost is `num_moe_layers * top_k * 4` bytes (int32), ~1-2 KB/token —
about 1-5% of the KV footprint.

## What this change lands (engine core)

`runtime/cache/routed_experts_pool.py`:

- **`RoutedExpertsPool`** — buffer `[size + 1, num_moe_layers, top_k]` int32,
  indexed on dim 0 by KV slot (row 0 reserved for padding, mirroring the KV
  pools). `store_layer(layer_id, loc, topk_ids)` scatters by slot exactly like
  `MHATokenToKVPool.set_kv_buffer`; `gather(loc)` returns `[n, num_moe_layers,
  top_k]` — the prefix-hit retrieval.
- **`RoutedExpertsCapturer`** — per-forward controller:
  `begin_forward(out_cache_loc)` → `capture(layer_id, topk_ids)` per MoE layer →
  `commit()` scatters into the pool. No-op when inactive, so the hook is free on
  requests that didn't ask for replay. Rows that don't match the forward's slot
  count (TP/EP all-gather) are skipped and counted, never written.
- A process-global accessor (`get/set_global_routed_experts_capturer`) mirroring
  `moe/distribution_recorder.py`'s global recorder.
- Request flag **`GenerateReqInput.return_routed_experts`** (opt-in, off by
  default, propagated through `__getitem__`).

The repo already captures per-token per-layer `topk_ids` for *offline*
distribution analysis (`moe/distribution_recorder.py`'s
`_DetailSinglePassGatherer._topk_ids_of_layer`) and tracks the current layer via
`with_current_layer`. R3 reuses that shape but adds slot-indexed **persistence**
and a **return path**.

## Remaining wiring (model forward + serving) — follow-ups

These need a running model, so they are tracked separately from the tested core:

1. **Allocate the pool.** Add a `ServerArgs` flag (e.g.
   `--enable-routing-replay`) that constructs a `RoutedExpertsPool` sized to the
   KV pool's `size`, with `num_moe_layers`/`top_k` from the model config, and
   installs a `RoutedExpertsCapturer` via `set_global_routed_experts_capturer`.
2. **Capture hook.** In `runtime/layers/moe/topk.py::select_experts` (next to
   the existing `get_global_expert_distribution_recorder().on_select_experts`
   call), call `capturer.capture(layer_id, topk_ids)`. The layer index is
   available through the same `with_current_layer` mechanism the distribution
   recorder uses.
3. **Slot mapping into the MoE layer.** `out_cache_loc` currently reaches
   attention but not the MoE block (`ForwardContext` holds no tensors by
   design). Thread it into `forward_mlp`/the sparse block, or stash it on the
   capturer via `begin_forward(out_cache_loc)` at forward start (in
   `model_runner`) and `commit()` at forward end.
4. **TP/EP alignment.** In the TP path `topk_ids` rows are the global
   all-gathered token set, not the rank-local tokens `out_cache_loc` indexes.
   Either capture before `pre_mlp_comm`, or slice the gathered rows back to
   local using the comm manager's bookkeeping. The DeepEP path routes locally
   and is easier. Until this lands, the capturer skips misaligned layers.
5. **Bypassed/fused routing.** When routing is fused into the kernel
   (`TopK` `BYPASSED` output), `topk_ids` never materializes in Python; R3 runs
   need a kernel-level capture or a forced standard-topk path.
6. **Prefix-hit retrieval + return.** On a prefix hit, `gather` the reused
   slots' routing and merge with the freshly captured tail; surface it to the
   caller (`meta_info.routed_experts`, shape `[seq_len, num_moe_layers,
   top_k]`). Crossing the gRPC/gateway boundary needs a proto field +
   servicer + Rust gateway rendering, same as the `return_token_ids` follow-up.

## Status

This PR is a **design-review scaffold**: the storage + capture/commit + retrieval
core is implemented and unit-tested (CPU); the model-forward and serving wiring
above are the follow-ups it is built to support.
