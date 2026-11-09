# DeepSeek V4.1 indexer scores

The native Blackwell DeepGEMM indexer writes BF16 logits and passes BF16 rows
through masking, alignment, candidate gathering, and DeepSelect. Head weights
retain their existing precision. Paged scoring uses the caller's query tile;
replicated-cache decode supplies the complete local query batch, including
speculative verify rows. Owner-local history retains its per-request windows.
This design avoids splitting a decode batch solely because its configured
history capacity is large. Each score tensor uses 2 bytes per query-by-history
entry; paged native logits and their cleaned copy can coexist, so peak storage
includes the score matrices, alignment, and selection scratch.
Canonical replicated-cache prefill scores densely: the runtime batches request
windows, and splits a request whose own logits exceed the budget, so one call's
BF16 logits stay within `TOKENSPEED_PREFILL_INDEX_LOGITS_MB` (10 GiB by default,
`index_logit_rows`); scoring only checks that bound. DeepSelect reads the MQA
view's [0, end) using the padded row stride; dense Full, including
candidate-source Full, does not copy a cleaned matrix and sets invalid pages
inside [0, end) to -inf in place (`mask_invalid_pages`). The sparse candidate
scorer already produces BF16 logits.
Kernel warmup uses the same output dtype as execution and compiles both
independent and request-grouped paged scoring before capture.

Invocation-owned index metadata can also carry per-query request IDs. Paged
Full passes the same IDs to schedule construction and scoring, allowing the
native kernel to reuse history across consecutive queries of one request.
Such rows share a page table and have nondecreasing visible lengths. Empty
rows get distinct scheduling labels so they cannot truncate an active request's
history bound. The runtime groups canonical decode rows; dense prefill ignores
request IDs. Free decode windows use independent rows so reordering queries
cannot truncate the history scored for an earlier row.
Request IDs, tables, and lengths remain unchanged while sharing metadata;
graph replay refreshes their captured producers.

The paged selection cleaner writes its final aligned score layout directly.
It masks invalid pages and initializes the last live tile's prefetch padding;
wholly future tiles are unspecified. Native token selection consumes this
buffer with exclusive row ends, avoiding another alignment copy. Candidate
block reduction similarly initializes only live blocks and their final aligned
segment, and native selection uses the valid block counts. The full-width
cleaner and generic score-preparation helpers retain their existing contracts.

Candidate block maxima remain FP32, with the existing newest-block pinning and
logical-position mapping. Converting BF16 scores to FP32 for this reduction
does not recover precision lost when the producer wrote BF16. Equal BF16
scores may select different valid boundary ties from an FP32-score Top-K.
Correctness checks therefore compare selected score multisets against the
same BF16 input, while model-quality checks must assess the precision change
separately. The Hopper score producer retains FP32 output.
