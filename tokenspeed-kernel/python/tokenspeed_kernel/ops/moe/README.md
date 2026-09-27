# Iris token-sharded MoE tail

The Kimi-K3 tail supports CDNA4, BF16, matching TP8 attention and MoE
process groups, EP1, and replicated latent projection weights. The current
widths are 3584 routed and 7168 shared/output elements. Runtime selection
uses the measured 512–8192-token window, divisible by eight; unsupported
inputs keep the ordinary reduction and projection.

Each rank reduces its contiguous token shard, normalizes and projects the
routed sum, and publishes `prefix + projected + shared_sum` to every rank.
The private communication kernels live in `communication/_iris/prefill.py`;
`communication/_iris/sync.py` holds the VMEM drain also used by ordinary
Iris reductions. `moe/iris.py` owns the composition and eligibility checks.
Imports of optional AMD implementations remain behind those checks.

The gather completion handshake follows every rank's entire reduction,
normalization, and projection. It therefore protects both producer-input
reuse and completion of the gathered result. No standalone reduce-scatter
consumer may reuse the input without an equivalent completion handshake.
Invocations on a group remain serialized on one stream, including replay.

The output is borrowed until the next tail. A prefix may alias its leading
rows even when output capacity exceeds this batch; shifted overlaps and
weight aliases are rejected. Producer ownership is checked separately before
checking residual and weight overlaps. Retained results must be cloned.
Row counts are runtime kernel arguments, so varying prefill sizes does not
compile another collective specialization. The AMD tail test checks numerical
results, delayed ranks, aliases, replay, epoch rollover, and compilation reuse.
