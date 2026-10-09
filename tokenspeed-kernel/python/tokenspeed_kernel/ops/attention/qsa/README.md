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

CuTe DSL decode stages the input stream as `K0, K1, V0, K2, V1, ...` in
shared buffers used by both K and V. QK and PV pair their notifications on
two named barriers, alternating with the S/P stage. This lets QK announce
K0 and K1 without waiting for PV or reusing an unfinished barrier. The
reverse PV handshake orders subsequent reuse and is drained before the
next query in a CTA. These notifications coordinate the consumers; the
pipeline mbarriers separately protect data readiness and buffer lifetime.
The two notification barriers follow QK tile parity, independently of the
number of shared K/V buffers. Each BF16 buffer holds one 128-by-256 tile
split across two physical stages; two and three buffers use the same handshake.
BF16 launches using asynchronous publication select three shared K/V buffers
so copies can run ahead of the consumers, including launches with one, two
or four KV splits. Compared with two buffers, the third adds 64 KiB of shared
memory per CTA; query-row grouping and split selection retain their existing
policy.
