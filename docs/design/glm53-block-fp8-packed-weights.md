# GLM-5.3 block-FP8 packed weights

The gfx950 block-FP8 dense path keeps one resident weight layout. A GLM-5.3
linear explicitly opts in; its checkpoint weight is loaded in ordinary
`[N, K]` order, then replaced after the last load write by a packed FP8 tensor
with the same logical shape. The original full-size tensor is released. The
prepared linear plan holds layout and dispatch metadata, not another weight
buffer. Other models keep ordinary weight semantics even when the experimental
kernel flag is set.

The packed byte order is `[K/128, N/64, 128, 64]`; the 128×128 inverse scales
stay in checkpoint order. On gfx950 one Gluon matrix pipeline reads this
layout for every row count. Medium and large prefill run directly; short
decode and small prefill with longer inner dimensions split the K work and
reduce its FP32 partials, while the short-K projection uses the direct path.
This preserves the large-prefill fast path without keeping a canonical weight
copy or falling through to a kernel that would interpret packed bytes as
ordinary `[N, K]` weights.

Only the ordinary FP8 dense linears in the GLM-5.3 target and NextN draft
decoder request promotion: six early feed-forward projections, 84 shared-expert
projections, 33 target sparse-attention projections, and five draft
projections in the four-GPU deployment. Routed MoE expert weights, KV
projections read directly by the attention implementation, and other models
are outside this layout contract.

## Loading and ownership

The existing weight parameter remains the model's parameter and retains its
checkpoint loader metadata. A normal, dummy, or sharded load promotes it only
after all canonical writes are complete. A live update stages the canonical
form only when a packed weight is actually written; a scale-only update leaves
every packed weight in place. After the write, the new values are packed into
the original graph-visible storage address. Untouched weights are not
repacked. A failed update cannot dispatch a canonical staging tensor as a
packed tensor.

Device moves rebind the plan to the moved packed parameter without packing
again. Ordinary state-dict export produces canonical weight bytes, so a saved
checkpoint can be loaded without double-packing. Unpacking for a write or
export creates temporary storage; it does not leave a second resident weight
in the serving model.

## Verification contract

Tests cover numerical agreement for both online BF16 activation quantization
and prequantized FP8 inputs, all selected row-count classes, changed-input
graph replay, and no recompilation per exact row count. Loader tests cover
initial, dummy and sharded loads, save→reload, partial live updates, and
device moves. Operation benchmarks compare the packed path against both the
previous fast-prefill branch and the latest upstream baseline, including the
small-row tuning boundaries and every observed 50k/500 partial-prefill tail.
