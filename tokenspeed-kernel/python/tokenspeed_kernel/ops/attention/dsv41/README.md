# DeepSeek V4.1 indexer scores

The native Blackwell DeepGEMM indexer writes BF16 logits and passes BF16 rows
through masking, alignment, candidate gathering, and DeepSelect. Head weights
retain their existing precision. Paged scoring uses the caller's query tile;
replicated-cache decode supplies the complete local query batch, including
speculative verify rows. Owner-local history retains its per-request windows.
This avoids splitting a decode batch solely because its configured history
capacity is large. Each score tensor uses two bytes per query-by-history entry;
native logits and their cleaned copy can coexist, so peak storage includes the
score matrices, alignment, and selection scratch.
Dense broadcast prefill retains its 128 MiB score budget and the runtime's
prefill query windows. The sparse candidate scorer already produces BF16 logits.
Kernel warmup uses the same output dtype as execution.

Candidate block maxima remain FP32, with the existing newest-block pinning and
logical-position mapping. Converting BF16 scores to FP32 for this reduction
does not recover precision lost when the producer wrote BF16. Equal BF16
scores may select different valid boundary ties from an FP32-score Top-K.
Correctness checks therefore compare selected score multisets against the
same BF16 input, while model-quality checks must assess the precision change
separately. The Hopper score producer retains FP32 output.
