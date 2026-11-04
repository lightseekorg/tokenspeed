# Fused decoder layers

A fused decoder layer runs one decoder layer of a model recipe (attention, its
output projection and all-reduce, the router, the experts and their exchange,
the post-expert projection, the FFN boundary) and the next layer's query
projection, attention prologue and K/V write as one kernel launch per decode
step. This directory holds the op's interface; implementations live in
solution-named packages (`_gluon/` for gfx950). How a model chooses between a
fused layer and its regular layers is described in
[docs/design/monokernel.md](../../../../../docs/design/monokernel.md).

## Interface

The registry entry is `("monokernel", "prepare")`, selected by the
`recipe` trait (plus shape traits such as `verify_width`, `tp_size`,
`ep_size`, `kv_cache_dtype`). The runtime uses three functions:

- `monokernel_state_heap_bytes(recipe=, world_size=, max_requests=)`:
  communication-heap bytes the group state will allocate (0 without a leaf),
  reserved when the heap is created.
- `prepare_monokernel_state(recipe=, group=, rank_in_group=, max_requests=,
  failure_record_path=, device=)`: the state every fused layer of a group
  shares (launch counters, communication regions, a host copy of the failure
  record), or `None` without a leaf. Every rank calls it, before capture.
- `prepare_monokernel_layer(recipe=, plan_keys=, traits=, weights=, state=)`:
  one `MonokernelLayer` for one decoder layer, or `None` without a leaf.
  It reads the layer's weights in place and compiles and checks every plan
  variant now, never on the forward path.

A `PlanKey` is `(requests, layer_kind, next_kind)`: the requests in the step
(rows = requests x verify width), this layer's attention kind and the next
layer's (`"regular"` when the next layer runs unfused; that launch ends at
the FFN boundary).

`MonokernelLayer`:

- `decline_reasons(requests)`: this rank's reasons not to serve `requests`
  (empty when it can). The runtime agrees on them across the group.
- `forward(prepared_query, residual, *, positions, cache, next_prologue)`:
  runs the layer and the next layer's QKV projection and prologue; returns
  the next layer's `(q, output_gate)` and the residual (updated in place).
- `forward_last(prepared_query, residual, *, positions, cache, next_norm)`:
  the last fused layer; returns the next layer's normed input rows and the
  residual in the regular layout.

## Leaves

| Recipe | Solution | Where | Serves |
|---|---|---|---|
| `muse_spark` | gluon (gfx950) | `_gluon/muse_spark/` ([README](_gluon/README.md)) | EP1, TP 8, verify width 8, FP8 KV cache; 1-2 requests |
