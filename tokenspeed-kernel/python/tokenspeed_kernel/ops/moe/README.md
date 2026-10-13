# MoE ops

Per-op notes for `ops/moe/`. Backend adapters keep their own READMEs
(`flashinfer/README.md`).

## Expert placement dispatch (`dispatch.py`)

With redundant experts (`--ep-num-redundant-experts R`) a logical expert has
several physical replicas across the EP ranks, and routing must emit physical
ids. Under replicated-input EP every rank runs the same routing over the same
all-gathered tokens, so the replica choice has to be a pure function of the
token row and the route rank for exactly one rank to own each (token, expert)
pair: `replicas[logical, (row + rank) % count]`.

`ExpertDispatch` holds one layer's tables (`replicas [num_logical, X]` int32,
`-1` padded to a fixed width; `num_replicas [num_logical]` int32). The runtime
keeps them as views of the placement's device tables so an in-place
rebalance reaches captured graphs without reallocating.

`dispatch_topk_ids(topk_ids, dispatch)` is the production implementation of
the replica choice: two gathers and a modulo over `[tokens, top_k]`, in
tensor ops, graph-capturable. Every id must be a real expert in
`[0, num_logical)`; the runtime masks zero experts and padded slots around
the call and counts the mapped ids into its load counters afterwards.

Follow-up: a fused Triton kernel for the same contract (one launch instead of
four tensor ops per MoE layer per forward). Its arguments must follow the
compile-time parameter rule: token count, top-k and table width are runtime
arguments or bucketed, never `constexpr`.
