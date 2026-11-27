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
uses the original logits and subtracts `(row_max + log_z)`, matching the
reference expression's FP32 addition and subtraction order. Adding the common
maximum back can lose precision for large offsets; that rounding is part of
this expression and is not replaced by shifted-space tail scoring.
Head and sampled scores preserve the configured direct or negative-cross-entropy
subtraction order, including the sign of zero; equal numeric values are not
sufficient evidence of bitwise agreement.
Head gathering and its explicit subtraction form are computed directly during
packing. Inspecting the scores materializes head values on demand, without
retaining another buffer on the serving path. Negative cross-entropy still
flips the rounded difference's sign bit, except for NaN, whose arithmetic
representation is preserved to match Torch.

This work runs on the same eager/graph path and uses a fixed K for the deployment.
The graph's padded outputs are sliced to the live batch. Verification maps the
accepted tree path to target prediction rows; chain verification uses its
existing prediction order. Sampled scoring can reuse the original normalizer only when its block width,
subtraction form and selected provider match, and no grammar mask changed the
rows. Tree scalar gathers index the original logits directly instead of copying
whole vocabulary rows. The CPU commit selects accepted tokens, applies stop
trimming, validates each retained row and overrides sampled logprobs only for
requests asking for full-vocabulary output. Ordinary co-batched requests retain
their existing sampled-logprob values. Enabling the launch option still normalizes every forward row, including ordinary
rows in a mixed batch, so their sampled logprobs retain the same provider and order.
A selected-row kernel can defer head/tail work until verification and stop trimming;
only accepted source rows then need a head and tail.

The block partials are folded left to right in FP32 by one kernel; neither
parallel reassociation nor a replacement logarithm is allowed. The selected
block-sum-exp provider and Torch logarithm stay unchanged. The portable tail temporarily masks the original-logit buffer in place,
runs the original Torch logsumexp, and restores the exact gathered head values.
The optional CUDA tail marks head IDs in a bitset and masks during loads. It
reuses the installed ATen reduction configuration, accumulator order, warp/block
and cross-CTA tree. Exponentiation happens in the sum reducer's input transform,
with the original FP32 subtraction and exp implementation. The final reduction
write applies the same CUDA `logf`, separately rounded addition and subtraction
as Torch's `log().add_().sub_()`, including the separate `(row_max + log_z)`
addition for the full normalizer, without intermediate elementwise launches.
Each active row clears its head bitset in the same CTA that marks the head IDs.
It neither subtracts head mass from one nor
changes the reduction geometry to compact rows.

## Accepted source rows

The normalizer is computed before sampling. Verification windows wider than one
explicitly defer head/tail scoring. Single-position outputs finish directly,
reuse Top-K's maximum and avoid an unnecessary original-logit copy. A selected-row
leaf owns a copy of
the original FP32 logits because the sampler may modify its input. One elementwise
load writes both the original copy and the shifted value; the maximum reduction
and every subtraction keep their original arithmetic. After verify,
draft-output rewriting and stop trimming, a fixed-size device mask maps valid
output positions to original prediction rows. Completing prefills form the
request prefix; decode uses its fixed width and optional accepted tree path.
Padding request slots select no rows. Invalid tree positions never clear a valid
source row when the identity tail aliases the accepted path.

Kernel grids and output tensor shapes remain fixed. CUDA selection and tail
blocks inspect the mask on the device and skip inactive source rows. There is no
accepted-count download, `nonzero`, boolean-index compaction or new host wait.
GPU work does depend on verification before head/tail computation can begin.
The sampler's scalar logprob computation can still reuse the pre-verify normalizer.

Full-row scoring can reuse the maximum already selected by Top-K. Zero and
non-finite maxima use a GPU fallback preserving `torch.max`'s first-index
semantics, including signed zero and NaN payloads. Deferred scoring retains its
pre-verify maximum; it does not run another normalization merely to reuse Top-K.

The current exact CUDA selection is adapted from Torch 2.9.0. It preserves that
version's radix selection, gather order and small-result sort, including ties.
The gather initializes skipped rows, avoiding separate output-fill launches.
Later radix passes also skip a block whose previous histogram contains no
entries in the selected prefix, writing the same zero histogram instead of
reading the vocabulary again. The digit counts, threshold, gather order and
tie handling are unchanged; this decision stays on the device.
It is limited to SM90, contiguous FP32 large-vocabulary rows and tensors whose
byte offsets fit below 2^31; other inputs retain the Torch implementation.
Build its optional library **before serving** with the installed Torch ABI:

```bash
TORCH_CUDA_ARCH_LIST=9.0 python -m tokenspeed_kernel.thirdparty.cuda.full_vocab_scoring
```

The library identity includes the Torch/CUDA version and source hashes. Missing,
stale or unsupported builds do not register the specialized leaf. The forward
never invokes the compiler. Provider selection remains inside the kernel registry;
eager and graph execution use the same scoring and row-mask contract.

Scoring retains a full `[rows, vocab]` FP32 shifted-logit buffer until packing,
plus the selected provider's and tail reduction's temporaries. Deferred scoring
also retains a full original-logit copy until selection finishes. The CUDA tail
uses a compact head bitset instead of a full-vocabulary exp temporary. A full-sized
buffer occupies `4 * rows * vocab` bytes (128 MiB at 256 rows and a 131072-token
vocabulary). Block partials are released before the tail reduction. Captured
graph pools retain the live allocations; size the capture ladder and maximum
live rows with this cost in mind.

## Wire and ownership

Each retained row is little-endian int32 with `2 + 2*K` slots: sampled token ID,
tail float32 bits, K head IDs, and K head float32 logprob bits. The output wire
tail `output_full_vocab_topk` contains contiguous bytes per request. It is
independent of JSON top-logprob lists. The client receives owned bytes alongside
the same emitted tokens and sampled logprobs.

One device packing kernel gathers the tree prediction row, reads its sampled
score, converts IDs, and writes the binary columns. It performs the configured
subtraction followed by an explicit sign-bit flip for non-NaN negative
cross-entropy scores;
head and tail values are copied as bits, without another reduction.
The same packing kernel computes head scores from shifted logits when no
materialized head is supplied. Explicitly supplied head values are copied as
bits. Each row has one private extra word holding the sampled score's FP32 bits. The binary columns and sampled scores share one pinned, asynchronous
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
