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
  `begin_forward(out_cache_loc)` → `capture_in_order(topk_ids)` per MoE layer →
  `commit()` scatters into the pool. No-op when inactive, so the hook is free on
  requests that didn't ask for replay. Rows that don't match the forward's slot
  count (TP/EP all-gather) are skipped and counted, never written.
- `build_routed_experts_capturer(server_args, model_config, size)` sizes the
  pool from the model config (`num_hidden_layers`, `num_experts_per_tok`);
  returns `None` for dense models.
- A process-global accessor (`get/set_global_routed_experts_capturer`) mirroring
  `moe/distribution_recorder.py`'s global recorder.
- Request flag **`GenerateReqInput.return_routed_experts`** (opt-in, off by
  default, propagated through `__getitem__`).

The repo already captures per-token per-layer `topk_ids` for *offline*
distribution analysis (`moe/distribution_recorder.py`'s
`_DetailSinglePassGatherer._topk_ids_of_layer`). R3 reuses that shape but adds
slot-indexed **persistence** and a **return path**.

## Forward wiring (landed, guarded by `--enable-routing-replay`)

All hooks are no-ops unless the flag is set, so default behavior is unchanged:

1. **Allocate + install** — `engine/event_loop.py` builds the pool (sized to the
   KV pool's `max_total_num_tokens`) and installs the capturer right after the
   KV pool is created.
2. **Forward lifecycle** — `execution/model_runner.py::forward` calls
   `begin_forward(out_cache_loc)` before `model.forward` and `commit()` after
   (in a `finally`). `out_cache_loc` is the exact per-token KV-slot vector.
3. **Capture hook** — `layers/moe/topk.py::select_experts` calls
   `capture_in_order(topk_ids)` next to the existing `on_select_experts`. Layer
   index is assigned by per-forward invocation order (MoE layers fire in order),
   avoiding edits to every model file. The standard (non-`BYPASSED`) path is the
   only one that materializes `topk_ids` in Python, so fused-routing models are
   naturally skipped.

## Remaining wiring — follow-ups (need GPU validation)

1. **CUDA-graph decode.** The decode forward is CUDA-graph captured;
   `capture_in_order` currently **skips under
   `torch.cuda.is_current_stream_capturing()`** (correct — it never corrupts the
   graph), so only eager forwards are captured today. To cover graphed decode,
   record the scatter into the graph using only stable-address buffers (mirror
   `distribution_recorder._on_hook`, which deliberately fires during capture).
2. **TP/EP alignment.** In the TP path `topk_ids` rows are the global
   all-gathered token set, not the rank-local tokens `out_cache_loc` indexes, so
   the capturer safe-skips (counted in `skipped_misaligned`). Capture before
   `pre_mlp_comm`, or slice the gathered rows back to local. TP=1 works today;
   the DeepEP path routes locally and is easier.
3. **Bypassed/fused routing.** When routing is fused into the kernel
   (`TopK` `BYPASSED`), `topk_ids` never materializes in Python; R3 runs need a
   kernel-level capture or a forced standard-topk path.
4. **Layer-index precision.** Invocation-order indexing assumes one
   `select_experts` per MoE layer per forward; multi-pass (spec/MTP) or
   multi-router layers would need the global layer index from
   `with_current_layer` instead (which also requires installing a non-noop layer
   tracker — the recorder is a noop in this fork).
5. **Prefix-hit retrieval + return.** On a prefix hit, `gather` the reused
   slots' routing and merge with the freshly captured tail; surface it to the
   caller (`meta_info.routed_experts`, shape `[seq_len, num_moe_layers,
   top_k]`). Crossing the gRPC/gateway boundary needs a proto field + servicer +
   Rust gateway rendering, same as the `return_token_ids` follow-up.

## Status

The storage/capture core is unit-tested (CPU, 25 tests) and the forward wiring
is landed but **guarded and not yet GPU-validated end to end** (this venv can't
run a model). The follow-ups above — chiefly CUDA-graph decode capture, TP/EP
alignment, and the retrieval/return path — are the remaining work.
