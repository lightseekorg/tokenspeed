# QSA sparse attention

`qsa_sparse_attention` consumes per-query physical cache slots. Pass
`max_seqlen_q=None` for prefill, including one-token prefill; uniform decode
passes its query width explicitly.

Non-positive selected slots remain invalid. FA2 and CuTe DSL redirect their
K/V reads to slot 1 while retaining the original validity mask. Both caches
must have at least two slots and keep slot 1 finite. The runtime uses the
zero-initialized, unwritten slot 1 of the reserved null page; slot 0 receives
padding writes and may contain NaN/Inf. Merely zeroing attention weights
cannot isolate non-finite V values in the subsequent matrix multiply.
