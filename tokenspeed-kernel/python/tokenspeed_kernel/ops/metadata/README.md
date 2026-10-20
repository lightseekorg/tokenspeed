# Stop conditions

`apply_stop_conditions` validates tensors and dispatches the in-place update
through `metadata / apply_stop_conditions`. GPU uses the existing Triton leaf;
Ascend plugins supply their device implementation. CPU retains the scalar
reference. Device leaves must preserve EOS, token budgets, speculative width
truncation, stopped requests, unfinished prefill and padding semantics, without
host tensor reads during graph capture or replay.
