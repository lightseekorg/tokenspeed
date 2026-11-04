# Decode monokernels

A monokernel runs one decoder layer's decode step as one persistent launch.
This directory holds the op's interface; implementations live in
solution-named packages (`_gluon/` for gfx950). How a model chooses between
monokernel layers and its regular layers is in
[docs/design/monokernel.md](../../../../../docs/design/monokernel.md).

## Interface

The registry entry is `("monokernel", "prepare")`, selected by the `recipe`
trait and shape traits such as `verify_width`, `tp_size`, `ep_size` and
`kv_cache_dtype`. The runtime calls three functions:

- `monokernel_state_heap_bytes(recipe=, world_size=, max_requests=)`: the
  communication-heap bytes the group state allocates (0 without a leaf),
  reserved when the heap is created.
- `prepare_monokernel_state(recipe=, group=, rank_in_group=, max_requests=,
  failure_record_path=, device=)`: the state every monokernel layer of a group
  shares (launch counters, communication regions, a host copy of the failure
  record), or `None` without a leaf. Every rank calls it before capture.
- `prepare_monokernel_layer(recipe=, plan_keys=, traits=, weights=, state=,
  deadline_seconds=)`: one `MonokernelLayer` for one decoder layer, or `None`
  without a leaf. It reads the layer's weights in place, and compiles and
  checks every plan variant before capture. `deadline_seconds` bounds how long
  a launch waits for another workgroup or GPU before it stops with an error;
  the runtime passes it explicitly.

A `PlanKey` is `(requests, layer_kind, next_kind)`: the requests in the step
(rows = requests x verify width), this layer's attention kind, and the next
layer's kind (`"regular"` when the next layer runs on the regular path; the
launch then ends at the FFN boundary).

`MonokernelLayer`:

- `decline_reasons(requests)`: this rank's reasons not to serve `requests`
  (empty when it can). The runtime agrees on them across the group.
- `forward(prepared_query, residual, *, positions, cache, next_prologue)`:
  runs the layer and the next layer's QKV projection and prologue; returns
  the next layer's `(q, output_gate)` and the residual, updated in place.
- `forward_last(prepared_query, residual, *, positions, cache, next_norm)`:
  the last monokernel layer of a range; returns the next layer's normed input
  rows and the residual in the regular layout.

## Leaves

| Recipe | Solution | Where | Serves |
|---|---|---|---|
| `muse_spark` | gluon (gfx950) | `_gluon/muse_spark/` ([README](_gluon/README.md)) | one layer plus the next layer's query projection, prologue and K/V write; EP1, TP 8, verify width 8, FP8 KV cache; 1-2 requests |
