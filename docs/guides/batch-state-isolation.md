# Batch state-isolation experiment

`test/runtime/run_batch_state_isolation.py` compares a named target request
alone and alongside deliberately different requests in one native,
headless TokenSpeed engine. It is a diagnostic harness for request-state
isolation and output consistency: the engine accepts token IDs through
`WireDriver`, so an HTTP gateway, detokenizer, stop-string handling, and client
connection timing are not part of the comparison.

## First-pass RTX 4090 run

The defaults use `Qwen/Qwen2.5-1.5B-Instruct` with BF16, TP 1, a 4096-token
context, greedy decoding, prefix caching disabled, chunked prefill disabled,
and CUDA graphs disabled. The launch enables returned output logprobs. It does
not request `--numerics rl-bitwise`: that envelope is verified per model and
requires its profile to advertise the contract. This experiment still reports
exact token and logprob differences, but a passing `auto` run is evidence for
this configuration rather than an advertised numerical contract.

Run from a checkout with its development environment active and select the
single CUDA device explicitly:

```bash
mkdir /tmp/batch-state-isolation
CUDA_VISIBLE_DEVICES=0 python test/runtime/run_batch_state_isolation.py \
  --output-dir /tmp/batch-state-isolation
```

The harness requires the output directory to already exist and be empty. It
starts one engine, submits each case before receiving any outputs, and leaves
no server running after the run. `--max-new-tokens`, `--max-model-len`,
`--max-num-seqs`, `--seed`, `--gpu-memory-utilization`, and
`--load-poll-interval` make the test size and load sampling explicit.

## Matrix and oracle

The fixed matrix compares a target request across:

| Case | Target position | Comparison |
| --- | ---: | --- |
| `single-1`, `single-2` | 0 | Independent-run stability |
| `equal-length` | 0 | Two similarly sized prefills |
| `long-short` | 0 | Long and short batch partners |
| `reversed` | 1 | Submission-order sensitivity |
| `shared-prefix-single` | 0 | Shared-prefix target alone |
| `shared-prefix`, `shared-prefix-reversed` | 0, 1 | Same prefix with reversed order |
| `four-request` | 0 | Four simultaneous requests |

For every compared target, `report.json` records raw input/output token IDs,
selected-token logprobs, finish reason/details, prompt/completion counts,
cached-token count, first token mismatch, first logprob mismatch, and the
first logprob mismatch before any token divergence. A differing finish reason
or finish detail fails the comparison even if tokens happen to match.

The report preserves frontend load snapshots taken when an output from the
current call arrives. A sample with at least two running requests and at least
two frontend in-flight requests is positive evidence that the frontend observed
concurrent scheduling. No such sample is **inconclusive**, not evidence that
the submitted requests were not co-scheduled: scheduler load is only published
with output batches and short requests can finish before an observable batch.

Exit codes are `0` for `PASS`, `1` for a completed comparison with a mismatch,
and `2` for an invalid run such as startup, model, tokenizer, or engine failure.
Raw request IDs and completion evidence remain in `report.json` for both
failure classes.

## Prefix-cache contrast

The first pass disables prefix caching to isolate batch effects. To add a
separate cold/warm shared-prefix contrast, run:

```bash
mkdir /tmp/batch-state-isolation-cache
CUDA_VISIBLE_DEVICES=0 python test/runtime/run_batch_state_isolation.py \
  --output-dir /tmp/batch-state-isolation-cache \
  --run-prefix-cache-contrast
```

The contrast starts separate cache-enabled engines for cold and warm phases,
primes only the warm engine with a shared-prefix peer, and compares the target
inside the same two-request batch. `prefix_cache_contrast.cached_tokens` makes
the cache state visible. The report marks `cache_evidence` `INCONCLUSIVE` if
the warm target does not report more cached prompt tokens than the cold target;
a cache hit never substitutes for output equality.

## Follow-up experiments

Keep one changed dimension per run:

1. Set `--chunked-prefill-size` in `engine_args()` to a positive explicit value
   and rerun the same matrix for full versus chunked-prefill comparison.
2. Use a model whose profile has verified `rl-bitwise`, launch with
   `--numerics rl-bitwise`, and treat any reported token or logprob difference
   as a contract violation. The design contract is in
   `docs/design/numerics.md`.
3. Extend the predeclared `PromptSet` and `BatchCase` list before running when
   testing another prompt family. Preserve the named target and reference case;
   do not infer target identity from completion order.
