# Stop conditions

`apply_stop_conditions` validates tensors and dispatches the in-place update
through `metadata / apply_stop_conditions`. NVIDIA, AMD and Ascend retain the
portable Triton leaf; higher-priority Ascend plugins can override it. CPU retains
the scalar reference. Device leaves accept the public function's nine tensors
followed by `num_extends`, `num_prefill_outputs`, `decode_width` and `padding_index`,
all passed positionally. Device leaves must preserve EOS, token budgets,
speculative width truncation, stopped requests, unfinished prefill and padding
semantics, without host tensor reads during graph capture or replay.

# Block-table rows

`expand_block_table_rows` rebuilds the scheduler's block-table rows on the
device from the span each row uploads: null holes (0) before it, its entries,
and -1 padding after. A sliding-window row at a long context is mostly holes,
so the bridge uploads spans instead of whole rows. CUDA runs the Triton kernel,
whose programs stride over a row's tiles; other devices run the same fill in
tensor ops. The reference is `torch_expand_block_table_rows` in
`numerics/torch/metadata.py`.
