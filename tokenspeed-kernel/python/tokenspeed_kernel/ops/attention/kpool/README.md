# KPool attention kernels

KPool selection scores compressed index keys, selects logical pools, expands
each selected pool into raw FlatKV slots, and appends the visible incomplete
pool tail.

The GLM-5.3-Flash specialization has fixed geometry:

- 32 query heads with head dimension 128
- 4 raw tokens per compressed pool
- 16 compressed rows per index-cache page
- 512 selected pools
- BF16 queries, signed BF16 or FP32 head weights, and scaled FP8 E4M3 keys
- weighted per-head ReLU scoring and global FlatKV-slot output

## Prefill selection

The hybrid Gluon implementation keeps one orchestration path across AMD
architectures. The architecture backend supplies three stages:

1. Score a bounded pool window through either the request page table or the
   precomputed physical-slot plan.
2. Select logical columns with the architecture's radix top-k for long or
   merged windows.
3. Reuse the portable Triton payload gather, deterministic short-window sort,
   and pool-to-FlatKV expansion.

GFX950 uses Wave64 MFMA scoring and its logical radix selector. GFX1250 uses
Wave32 WMMA-v3 scoring and its existing Wave32 radix selector. Short
single-window rows of at most 2048 pools fold signed head contributions in
logical head order before deterministic sorting, preserving stable pool IDs
for equal scores.

Production prefill supplies `pool_workspace_slots`, `row_starts`, and
`row_ends`. These tensors preserve request-local logical pool order while
addressing physical cache rows directly. The eager-only compatibility path
reconstructs request IDs and causal lengths and scores through the index page
table.

Scoring workspaces are row-tiled under `max_logits_bytes`. The cap includes
the persistent sort or radix intermediates as well as logits; one row remains
legal when its workspace exceeds the cap.

## Query preparation

`kpool_prefill_prepare_query` builds the query-side inputs of the planned
`kpool_prefill_topk` call: everything the selection needs from the indexer
projections alone, before the pooled cache of the current chunk is written.
It is registered per solution with exactly the capability, signature and
traits of that solution's `kpool_prefill_topk`, so selection resolves the two
together and the returned value is what the selected top-k consumes through
its required `prepared_query` keyword:

- `deep_gemm` returns `(q_fp8, scaled_weights)`: the FP8 queries plus FP32
  head weights with the query dequant scale and the softmax scale folded in,
  which the top-k previously computed inside the call.
- `triton` (portable) and the Gluon backends score BF16 queries directly and
  return `None`.

A caller can therefore issue the pooled-cache writes of a layer on one stream
and `kpool_prefill_prepare_query` on another, join, and run
`kpool_prefill_topk(..., prepared_query=...)` without repeating the query work.
