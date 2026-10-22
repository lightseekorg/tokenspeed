# Stop conditions

`apply_stop_conditions` validates tensors and dispatches the in-place update
through `metadata / apply_stop_conditions`. NVIDIA, AMD and Ascend retain the
portable Triton leaf; higher-priority Ascend plugins can override it. CPU retains
the scalar reference. Device leaves accept the public function's nine tensors
followed by `num_extends`, `num_prefill_outputs`, `decode_width` and `padding_index`,
all passed positionally. Device leaves must preserve EOS, token budgets,
speculative width truncation, stopped requests, unfinished prefill and padding
semantics, without host tensor reads during graph capture or replay.
