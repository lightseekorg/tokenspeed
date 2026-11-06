# Input and output vocabularies

`VocabularySpec(input_size, output_size)` is the immutable logical vocabulary
contract shared by model construction and execution. Model profiles declare it;
models without a profile use the common checkpoint resolver once in
`ModelConfig`. Both sizes are positive integers. Under the common convention,
`vocab_size` sizes inputs and an absent/null `output_vocab_size` uses the same
size for outputs. Draft checkpoints may instead declare `draft_vocab_size`.

Input buffers, n-gram preparation, multimodal substitutions and token history
use the input domain. Heads, logits, grammar, penalty tables, speculative
probabilities and logprobs use the output domain. A target's output columns are
token IDs in its input vocabulary; reduced draft vocabularies retain their
existing adapter-owned ID mapping. The engine does not pad compact outputs to
the input width. TP padding and checkpoint storage rows are physical layout,
not additional output tokens. Block drafters that project sharded heads
directly must intersect both original and added token ranges with the logical
output domain, including shards containing no output tokens. Distributed
argmax shortcuts require the physical and logical widths to match.

DSpark's Markov embedding covers input tokens; its projection and checkpoint
rows cover output tokens. This applies to replicated generic/K3/MiniMax heads
and V4.1 heads that use the same vocabulary shards as the LM head. In the
replicated drafter, previous tokens are bounded by the input embedding,
independently of output width. V4.1 draft anchors come from target outputs
and remain in the output domain.
Under attention DP, V4.1 requires a replicated LM head: sharding the head
across different request batches needs a row redistribution path that its
block sampler does not implement. The Markov projection and workspace use
the head's group, independently of the attention TP width. Padding-only
shards contribute no candidate but still join every TP gather.

Every model entrypoint receives a `ModelInitConfig` through its `config`
argument. This lightweight, frozen envelope contains the model-specific
`hf_config` and the already resolved `vocabulary`; it has no loading or serving
policy and does not forward unknown attributes to the HF config. Construction
never resolves the vocabulary again. Wrappers forward the envelope unchanged
or use `with_hf_config(text_config)` to select checkpoint metadata while keeping
the same vocabulary object. Drafts receive their own resolved vocabulary.
The envelope does not freeze the HF object: adapters that modify checkpoint
metadata copy it first. Internal layers continue to receive their existing
HF config or the domain sizes they consume.
`ModelExecutorConfig` and `RuntimeStates` retain that value, without legacy
`vocab_size` aliases or per-layer defaults. Components that consume only one
domain keep the local name `vocab_size`; callers select `input_size` or
`output_size` at the boundary. `LogitsProcessor` takes the logical output width
as `vocab_size` and explicit projection policies, never an HF config. Tied common
heads require equal input and output sizes. Model adapters own any other
checkpoint-specific layout or sharing rules. A draft that shares the target's
head must resolve the same output domain before construction; mismatches are
rejected during wiring.

The host checks ordinary prompt IDs against `input_vocab_size`, and sampling
parameters and prompt-logprob targets against `output_vocab_size`. Multimodal
hash placeholders are resolved by the media adapter. Input-only tokens may be
used as context; requesting prompt logprobs that score them is rejected, rather
than silently scoring a clamped ID. Internal graph padding may still use safe
indices after the corresponding validity checks.

Runtime plugin API 4 requires `ModelProfile.vocabulary` and a model constructor
that accepts `config: ModelInitConfig`. A plugin interprets its checkpoint
fields in `model_profile(hf_config)`; the loader binds that resolved vocabulary
to the construction config. Plugins forward the envelope to base model
constructors and read checkpoint fields from `config.hf_config`. Text/hash
vocabulary constraints remain model-owned.
Plugins calling `LogitsProcessor` pass `vocab_size=vocabulary.output_size`,
`final_logit_softcapping`, `use_fused_lm_head` and `tie_word_embeddings` explicitly;
copying and rewriting a checkpoint config for logits is no longer necessary.
