# Returning Raw Token IDs

RL trainers (slime, veRL, Agent Lightning) train on the *exact* token sequence
a rollout produced. If the engine only returns text, the trainer must
re-tokenize it — and re-tokenization is not guaranteed to reproduce the
original ids. This **retokenization drift** comes from non-unique tokenization,
tool-call re-serialization, and chat-template differences, and it silently
makes RL updates off-policy.

The `return_token_ids` request flag makes the engine echo the raw token ids so
the trainer reuses them verbatim instead of re-tokenizing.

## Request flag

`return_token_ids` is a request-level boolean on `GenerateReqInput`
(default `False`). When set, the engine adds two fields to `meta_info` on the
generation response:

| field                     | meaning                                             |
| ------------------------- | --------------------------------------------------- |
| `meta_info.prompt_token_ids` | Tokenizer-valid, unpadded prompt token ids the engine actually ran (before any multimodal placeholder padding). |
| `meta_info.output_token_ids` | The full cumulative generated token sequence (the exact tokens sampled during the rollout). |

`output_token_ids` is the whole sequence on the finished response, not the
per-frame streaming delta that rides in the top-level `output_ids`.

## Example (in-process engine)

```python
from tokenspeed.runtime.engine.io_struct import GenerateReqInput

req = GenerateReqInput(
    text="What is the capital of France?",
    sampling_params={"max_new_tokens": 16},
    return_token_ids=True,
)
async for out in engine.generate_request(req):
    pass  # consume to the finished response

meta = out["meta_info"]
prompt_ids = meta["prompt_token_ids"]   # exact prompt tokens
output_ids = meta["output_token_ids"]   # exact sampled tokens
# Trainer feeds prompt_ids + output_ids directly — no re-tokenization.
```

## Relationship to logprobs

Independently of this flag, RL trainers have historically recovered response
token ids from `meta_info.output_token_logprobs` (the
`[[logprob, token_id, text], ...]` shape). That path still works and is used by
the native `/generate` route's placeholder-logprob fallback. `return_token_ids`
is the direct, first-class way to get both prompt and output ids without
requesting logprobs.

## Scope / status

This is the **engine-side** contract. Output token ids already reach clients
through the gRPC `GenerateComplete.output_ids` field; this flag adds the
**prompt** token ids at the engine boundary (`AsyncLLM.generate_request`'s
yielded dict).

Surfacing `prompt_token_ids` on the OpenAI-compatible `/v1/*` and native
`/generate` HTTP responses additionally requires, outside this repo:

1. a `prompt_ids` field on the `GenerateComplete` proto message
   (`tokenspeed_scheduler.proto`),
2. the gRPC servicer copying `meta_info["prompt_token_ids"]` into it, and
3. the Rust `smg` gateway rendering it into the JSON response
   (as `prompt_token_ids` / `token_ids`, matching the vLLM OpenAI API).

Those are tracked as follow-ups; this change lands the engine plumbing and the
request contract they build on.
