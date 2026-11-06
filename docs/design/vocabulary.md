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
and V4.1 heads that use the same vocabulary shards as the LM head. Previous
tokens are bounded by the input embedding, independently of output width.
Padding-only shards contribute no candidate but still join every TP gather.

Every model constructor receives the resolved `vocabulary` explicitly.
`ModelExecutorConfig` and `RuntimeStates` retain that value, without legacy
`vocab_size` aliases or per-layer defaults. `LogitsProcessor` takes an explicit
`output_vocab_size` and projection policies, never an HF config. Tied common
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

Runtime plugin API 4 requires `ModelProfile.vocabulary` and the keyword-only
model constructor argument `vocabulary`. A plugin interprets its checkpoint
fields in `model_profile(hf_config)` and forwards that same value to its base
model constructor. Text/hash vocabulary constraints remain model-owned.
Plugins calling `LogitsProcessor` pass `output_vocab_size`,
`final_logit_softcapping`, `use_fused_lm_head` and `tie_word_embeddings` explicitly;
copying and rewriting a checkpoint config for logits is no longer necessary.
