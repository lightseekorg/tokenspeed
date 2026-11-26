# Full-vocabulary binary output

An opt-in request can retain head and tail scores from the target model's
original full-vocabulary logits. The launch configuration fixes the head width
with `--full-vocab-top-k K` and explicitly selects the block-sum-exp provider with
`--full-vocab-top-k-solution SOLUTION`. Output logprobs and a blockwise logprob
order are required. Pipeline parallelism, sharded sampling, synthetic acceptance
and synthetic harness output tokens are rejected.

The request sets `return_full_vocab_topk=true`, `return_logprob=true`,
`top_logprobs_num=K`. Binary rows follow the selected token streaming cadence,
including a non-streaming request's forced refresh. Admission requires the configured width
and an unmodified temperature-one, full-vocabulary distribution. Grammar,
penalties and sampling truncation are rejected. The ordinary top-logprob path
keeps its existing limits and semantics.

## One execution path

The target forward captures original logits before the sampler can modify them.
One fixed reduction plan computes the maximum and blockwise denominator used by
both sampled and head logprobs. The tail is a masked logsumexp; it is never
computed by subtracting the head's probability sum from one. Tail reduction
stays in maximum-shifted logit space and subtracts the same log denominator,
avoiding cancellation when logits have a large common offset.
Head and sampled scores preserve the configured direct or negative-cross-entropy
subtraction order, including the sign of zero; equal numeric values are not
sufficient evidence of bitwise agreement.

This work runs on the same eager/graph path and uses a fixed K for the deployment.
The graph's padded outputs are sliced to the live batch. Verification maps the
accepted tree path to target prediction rows; chain verification uses its
existing prediction order. Sampled scoring can reuse the original normalizer only when its block width,
subtraction form and selected provider match, and no grammar mask changed the
rows. Tree scalar gathers index the original logits directly instead of copying
whole vocabulary rows. The CPU commit selects accepted tokens, applies stop
trimming, validates each retained row and overrides sampled logprobs only for
requests asking for full-vocabulary output. Ordinary co-batched requests retain
their existing sampled-logprob values. Enabling the launch option adds scoring
work to all forward rows, including ordinary rows in a mixed batch.

Scoring retains a full `[rows, vocab]` FP32 shifted-logit buffer until packing
and uses another full-sized masked buffer for the tail reduction, plus reduction
temporaries. Each such buffer occupies `4 * rows * vocab` bytes (128 MiB at 256
rows and a 131072-token vocabulary). Captured graph pools retain these allocations;
size the capture ladder and maximum live rows with this cost in mind.

## Wire and ownership

Each retained row is little-endian int32 with `2 + 2*K` slots: sampled token ID,
tail float32 bits, K head IDs, and K head float32 logprob bits. The output wire
tail `output_full_vocab_topk` contains contiguous bytes per request. It is
independent of JSON top-logprob lists. The client receives owned bytes alongside
the same emitted tokens and sampled logprobs.

On the device, each row has one private extra word holding the sampled score's
FP32 bits. The binary columns and sampled scores share one pinned, asynchronous
D2H copy. The result owns that host allocation until its completion query copies
it; no reusable host slot can be overwritten while a result is in flight. The
existing copy event covers this transfer, with no extra host wait. Python exports
buffer views and Rust owns one flat byte vector. Commit copies only retained rows
to request storage and validates in place, rolling back a bad row. Each emitted
chunk takes ownership of those unsent bytes; the request does not retain or copy
already delivered binary history. ID uniqueness
uses a fixed stack array and sorting rather than pairwise comparisons.

P/D status metadata carries an owned first-token row together with its sampled
logprob. Decode includes it before its own rows. The engine does not publish to
external stores: adapters own that lifecycle, including partial aborts, and must
check that the final byte count matches the emitted token count before returning
their response. Wire producers and consumers using this feature must be updated
together, including the P/D workers.

Selecting a batch-invariant provider does not prove trainer alignment or
cross-topology invariance; those require separate numerical acceptance checks.
