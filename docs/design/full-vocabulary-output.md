# Full-vocabulary binary output

An opt-in request can retain head and tail scores from the target model's
original full-vocabulary logits. The launch configuration fixes the head width
with `--full-vocab-top-k K` and explicitly selects the block-sum-exp provider with
`--full-vocab-top-k-solution SOLUTION`. Output logprobs and a blockwise logprob
order are required. Pipeline parallelism, sharded sampling, synthetic acceptance
and synthetic harness output tokens are rejected.

The request sets `return_full_vocab_topk=true`, `return_logprob=true`,
`top_logprobs_num=K` and `stream=false`. Admission requires the configured width
and an unmodified temperature-one, full-vocabulary distribution. Grammar,
penalties and sampling truncation are rejected. The ordinary top-logprob path
keeps its existing limits and semantics.

## One execution path

The target forward captures original logits before the sampler can modify them.
One fixed reduction plan computes the maximum and blockwise denominator used by
both sampled and head logprobs. The tail is a masked logsumexp; it is never
computed by subtracting the head's probability sum from one.
Head and sampled scores preserve the configured direct or negative-cross-entropy
subtraction order, including the sign of zero; equal numeric values are not
sufficient evidence of bitwise agreement.

This work runs on the same eager/graph path and uses a fixed K for the deployment.
The graph's padded outputs are sliced to the live batch. Verification maps the
accepted tree path to target prediction rows; chain verification uses its
existing prediction order. The CPU commit selects accepted tokens, applies stop
trimming, validates each retained row and overrides sampled logprobs only for
requests asking for full-vocabulary output. Ordinary co-batched requests retain
their existing sampled-logprob values. Enabling the launch option adds scoring
work to all forward rows, including ordinary rows in a mixed batch.

## Wire and ownership

Each retained row is little-endian int32 with `2 + 2*K` slots: sampled token ID,
tail float32 bits, K head IDs, and K head float32 logprob bits. The output wire
tail `output_full_vocab_topk` contains contiguous bytes per request. It is
independent of JSON top-logprob lists. The client receives owned bytes alongside
the same emitted tokens and sampled logprobs.

P/D status metadata carries an owned first-token row together with its sampled
logprob. Decode includes it before its own rows. The engine does not publish to
external stores: adapters own that lifecycle, including partial aborts, and must
check that the final byte count matches the emitted token count before returning
their response. Wire producers and consumers using this feature must be updated
together, including the P/D workers.

Selecting a batch-invariant provider does not prove trainer alignment or
cross-topology invariance; those require separate numerical acceptance checks.
