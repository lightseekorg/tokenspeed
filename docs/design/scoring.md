# Scoring (the Score API and decision-style output)

Decision workloads — routing, classification, policy checks, ranking —
need a small structured signal instead of generated prose. The Score API
makes that output contract explicit: the caller declares the label token
set up front, and the engine guarantees every label is scored at the
answer boundary. This is the serving-side counterpart of "Jev-like"
(System One) decision models; it is not a substitute for training a good
decision model, and normalized scores are not calibrated correctness
probabilities.

This document records the deliberate invariants of the subsystem: what
belongs where, and why.

## The contract

```jsonc
// POST /v1/score  (path identical to SGLang's, for client reuse)
{
  "model": "...",
  "query": "Context: ...\nQuestion: ...\n",   // shared by every item
  "items": ["candidate text 1", "..."],        // one score row each
  "label_token_ids": [9454, 2753],             // explicit, single tokens
  "apply_softmax": true                        // explicit, no default
}
// -> {"scores": [[p_yes, p_no], ...]}        // rows: items; columns: labels
```

Semantics that must not drift:

- **Every declared label is scored.** The readout gathers the declared
  `label_token_ids` from the full-vocab log_softmax at the answer
  boundary. It never relies on top-k generation logprobs, which may omit
  a label the application needs. Label-selective extraction does *not*
  remove the vocabulary projection or full-distribution normalization.
- **The answer boundary is the last prefill position** of each
  `query + item` sequence. Scores are read there exactly once, on the
  final prefill chunk; mid-chunk positions are not answer boundaries.
- **`apply_softmax` normalizes across the label set of one row** (a
  label-restricted softmax), never across rows, and is an explicit
  required choice. `false` returns raw logprobs.
- **Score requests are score-only.** `score_label_token_ids` requires
  `max_new_tokens=0` (`SamplingParams.verify`). Decoding past the
  boundary would leave the contract undefined.

## One execution path, no new scheduler concepts

A score request is an ordinary generation request with
`max_new_tokens=0` plus a label readout — not a second path:

- Admission, scheduling, prefix caching, chunked prefill, retraction,
  and the C++ FSM are untouched. The one sampled bootstrap token is FSM
  plumbing (the C++ `ExtendResult` requires it); it is not part of the
  response contract.
- The readout lives in the eager forward step
  (`ModelExecutor._forward_step`), gathering from the sanitized logits
  before sampling. Extend batches never replay a captured CUDA graph
  (`ForwardStepRunner` requires decode mode), but the shared forward return contract still carries both prompt
  logprobs and scores. Capture, replay, eager, idle and pipeline-parallel
  placeholders return the same five fields. Pipeline stages receive score
  rows with the existing commit-time result broadcast; decode graphs return `None`
  for both prefill-only readouts.
- The payload flows `ModelExecutionResult.score_logprobs` →
  `RequestState.score_vals` → `BatchTokenIDOut.output_score_vals` →
  `BatchTokenIDOutSlim` for the msgpack frontend. Wire fields are
  appended tails with defaults (older peers decode `None`).
- PD handoff: the committed score row follows the existing bootstrap
  metadata transfer as an optional trailing float64 status frame. Decode
  publishes it only after every expected Prefill rank completes, before
  finishing a zero-budget request. Room cleanup also drops pending rows.
- Aborted items fail `async_score` and `async_decision`, even if a row was
  computed before the abort. Numerical-abort outputs suppress that row.
- SIS execution: each `query + item` is an independent logical sequence.
  The shared query is reused through the ordinary radix cache — item 1
  computes it, items 2..N prefix-match it. No new cache-group or
  attention-mask machinery.

## Ownership split

- **Prompt semantics (pointwise vs setwise) belong to the caller or to a
  decision adapter — never to the engine.** Pointwise judges each
  candidate independently (candidates cannot see each other); setwise
  shows all candidates together. They are different questions and may
  yield different decisions; the choice belongs to the application's
  task and the model's training.
- **The adapter layer (`runtime/decision/`) owns everything
  model-specific**: the prompt scaffold, the label vocabulary (with a
  hard single-token check against the served tokenizer), and the
  translation of raw score rows into a decision. The engine below it
  only knows `label_token_ids`. Family adapters are registered
  explicitly by name; an unknown name is an error, not a silent fallback
  to generic behavior.
- **Execution mode belongs to the runtime.** SIS is the only mode today.
  MIS (fusing the shared query into one masked attention pass) is a
  possible future optimization and, per the project design principles,
  would be owned by the LCM cache subsystem and the C++ scheduler — with
  candidate-isolating attention in the backend — not by model code.

## Surfaces

- `Engine.score` / `Engine.async_score`: the low-level contract
  (text in, score rows out). `Engine.decision` / `async_decision`: the
  adapter-driven high-level contract.
- Raw `/generate` passthrough: `sampling_params` with
  `score_label_token_ids` / `score_apply_softmax` scores the prompt; the
  row appears as `scores` in the response dict alongside the (contract-
  irrelevant) bootstrap token.
- SMG gateway: the control server proxies `/v1/score` unchanged, but
  the pinned `tokenspeed-smg==1.10.1.post20260920` does not implement
  that endpoint or decode the score column. HTTP Score serving requires
  the separate gateway follow-up and a matching dependency pin. Its
  decoder skips unknown trailing columns, preserving ordinary generation
  with the appended score column. The in-process Engine APIs are usable
  independently of that HTTP follow-up.

## Explicitly out of scope (v1)

- MIS / shared-query fused attention, and setwise anchor extraction
  (`score_extraction_token`-style mid-sequence readout). Both need the
  input-position scoring integration for decisions, which v1 does not expose.
- SequenceClassification (classification-head) models. Only causal-LM
  next-token label scoring is supported.
- Calibration. `apply_softmax` yields the model's single-forward
  distribution restricted to the label set; calibrated confidence is a
  training property the engine cannot manufacture.

## Known edges

- A fully prefix-cached prefill computes no logits; a fresh request
  cannot hit this (prefix matching excludes the final prompt token), and
  the request still finishes — the frontend then reports the missing
  readout instead of inventing scores.
- The score-only contract is enforced at request validation; combining
  score labels with a nonzero decode budget is rejected loudly, not
  clamped.
