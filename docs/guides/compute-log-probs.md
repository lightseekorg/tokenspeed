# Computing Log-Probabilities (RL Scoring)

`Engine.compute_log_probs` scores `prompt + completion` token sequences under the
engine's current weights and returns one log-probability per completion token. It
is the core scoring primitive for online-RL trainers (PPO, GRPO, and any
KL-penalised objective) — for example to form importance-sampling ratios against
the policy that generated the rollouts.

## Usage

```python
from tokenspeed.runtime.entrypoints.engine import Engine

# Scoring runs a pure-extend (prefill-only) forward. On backends that cannot
# serve a mixed prefill+decode batch eagerly (e.g. the default `mha` backend),
# launch the engine for scoring with a backend + scheduler config that keeps the
# request on a pure-extend path:
engine = Engine(
    model="<model-path>",
    attention_backend="flashinfer",
    enforce_eager=True,
    disable_overlap_schedule=True,
)

out = engine.compute_log_probs(
    sequences=[
        {"prompt_token_ids": [1, 2, 3, 4], "completion_token_ids": [5, 6, 7]},
        {"prompt_token_ids": [10, 11],     "completion_token_ids": [12]},
    ],
    temperature=1.0,
)

# out["log_probs"][i][j] == log P(completion_token_ids[i][j] | context)
# out["tokens"][i]       == completion_token_ids[i]
out["log_probs"]  # e.g. [[-0.12, -0.47, -0.31], [-2.03]]
out["tokens"]     # [[5, 6, 7], [12]]
```

`log_probs[i][j]` is the log-probability of the realised completion token `j` in
sequence `i`, conditioned on everything before it (prompt + earlier completion
tokens). Only completion positions are scored; the prompt is context.

## How it works

It reuses the normal generation path: internally each sequence is sent through a
forward-only `generate` call (`max_new_tokens=0`, `return_logprob=True`,
`logprob_start_len=len(prompt)`), and the per-token input logprobs are read back
from `meta_info["input_token_logprobs"]`. Logits are gathered across tensor-parallel
ranks before `log_softmax`, exactly as on the sampling path. No engine pause is
required; scoring requests can be interleaved with normal generation.

Long sequences are handled across chunked prefill: when a `prompt + completion`
is split into multiple prefill chunks, the input-logprob window is collected from
every chunk it overlaps (not just the first), so the full set of completion
logprobs is returned regardless of `chunked_prefill_size`.

## Limits (current)

- **Temperature:** `temperature=1.0` only (raw `log_softmax`). Other values raise
  `NotImplementedError`. Sampling-temperature scaling (for off-policy importance
  sampling) is a planned follow-up.
- **Speculative decoding:** unavailable — `compute_log_probs` raises if the engine
  was launched with a speculative algorithm (the generation path disables logprobs
  in that mode).
- **Prompt/completion:** both must be non-empty (the first completion token needs
  prior context to be scored).
- **Surface:** exposed as the `Engine` Python method. A native HTTP / SMG endpoint
  is deferred until there is a consumer for it.
