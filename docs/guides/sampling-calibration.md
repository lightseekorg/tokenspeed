# Sampling probability calibration

The native calibration runner checks whether generated tokens agree with the
engine's reported, full-vocabulary probability distribution. It runs through
`WireDriver`, without an HTTP frontend, and supports two experiments:

- `gsm8k`: generate a predeclared number of tokens from ordered JSONL questions.
- `prefill`: repeatedly generate one token from the same prompt, isolating the
  first output draw.

Use an engine with `--enable-output-logprobs` and
`--enable-output-cumulative-probs`. Requests use temperature 1, disabled top-k,
top-p 1, min-p 0, and no penalties. The runner supplies no sampling seed or
request ID; `generate_request` creates a fresh ID for each request.

The native Engine's common `sample_with_diagnostics` boundary computes the
cumulative diagnostic for all sampling backends, including the FlashInfer and
Triton families. After the common NaN audit and grammar-bitmask wait, it copies
target score logits and applies the grammar mask in packed output order.
It then calls the backend's existing sampling path and scores its final,
TP-agreed tokens, after a drafter's rewrite of its verify outputs. Backends
have no cumulative-diagnostic hook; their token selection and existing
logprob algorithms remain unchanged. A backend only declares
`_SUPPORTS_OUTPUT_CUMULATIVE_PROBS` once `sample()` and `verify()` return one
token per score row and write aligned `next_token_logprobs` on every route;
the engine refuses the diagnostic at startup for a backend that does not
(every in-tree backend does). Direct backend calls and HTTP frontend
transport are outside this Engine-only boundary.

Chain speculative verification is supported. Every
committed token is scored against the target distribution at its actual output
prefix, including accepted draft tokens, rejection replacements, and the bonus
token. Do not test only accepted or only rejected outputs against the target
distribution: those conditional subsets generally follow different laws.
Draft trees, EAGLE-Harness, DP sampling, attention DP, pipeline parallelism,
PD disaggregation, synthetic acceptance and
`TOKENSPEED_SPEC_SIMULATED_ACCEPT_LEN` remain unsupported; the engine refuses
them at startup.

The reported scores precede temperature, top-k/top-p/min-p and penalty
adjustments. The diagnostic normalizes those same score logits independently
of rounded logprobs; it does not change the reported logprob order. This is
why calibration requires the neutral sampling parameters above. Greedy can
report the diagnostic, but its argmax output does not follow the full softmax
distribution and is not a positive control for the uniform-quantile test.

Both engine flags control computation for the whole batch, including padded
and uncommitted verify rows. Request-level `return_logprob` controls which
metadata is returned. Only enable the cumulative diagnostic for inspection:
every sampling step keeps a copy of the score logits (`[rows, vocab]`) plus
temporaries bounded to about 90 MB by row chunking. Decode graphs capture
that memory; prefill and mixed steps sample outside any graph, so theirs
comes from runtime headroom: leave room with `--gpu-memory-utilization`. With
the feature disabled, the boundary calls the backend without diagnostic
tensor operations. Each prefill with an output row contributes one score row
(every prefill, except on attention backends that skip incomplete prefills'
outputs, such as DeepSeek V4.1's), and decodes their fixed verify width; stop
trimming selects the same committed prefix from tokens, logprobs and
cumulative mass. A token sampled from an invalid row (an all-NaN row, whose
token a sampler may leave outside the vocabulary) gets NaN.

To exercise multi-token verification without draft weights, add
`--speculative-algorithm DUMMY --speculative-num-draft-tokens 4` to the engine
configuration and run `gsm8k` with more than one output token per request.
`DUMMY` proposes deterministic candidates and uses real target-only rejection;
it does not support `--enable-speculative-sampling` with recorded draft
probabilities. Keep both acceptance overrides disabled. One-token `prefill`
runs do not exercise speculative verification. Large-vocabulary DUMMY proposals
usually reject, so inspect `spec_verify_ct` and `accept_draft_tokens` in the raw
completions and use controlled verifier tests for acceptance and bonus coverage.

## What is tested

For a generated token `y`, the engine returns its log probability and the
cumulative probability `C` of tokens preceding it in descending probability
order, with token ID breaking ties. The client independently draws a uniform
number `W` and computes:

```text
p = exp(logprob)
U = C + W * p
bin = floor(B * U)
```

Under the conditional null hypothesis that every next token is drawn from the
reported distribution, `U` is uniform on `[0, 1)`. The token intervals partition
the unit interval, and selecting an interval with its probability, followed by
uniform randomization within it, gives uniform density. This argument applies
after conditioning on the entire earlier history, so adaptive decode prefixes
do not invalidate the test.

Consequently, each of `B` bins has probability `1/B`, and its count in a fixed
sample of `N` events has the binomial distribution `Binomial(N, 1/B)`. These are
cumulative **probability-mass** intervals, rather than intervals of an individual
token's probability. A token with probability greater than `1/B` spans several
bins; independent randomization is necessary to handle it correctly.

The runner computes inclusive binomial tails with stable probability recurrences.
It allocates `alpha / (2 * B)` to each direction of each bin. The probability of
any false alarm in the single pooled test is at most `alpha`; independence
between bins is not required. This is a false-alarm bound, not the probability
that a passing or failing result is correct.

For ten bins and `alpha=0.001`:

| Fixed events | Reject if any bin count is | Power for 10% → 11% | Power for 10% → 12% |
| ---: | --- | ---: | ---: |
| 1,000 | ≤64 or ≥140 | 0.193% | 3.088% |
| 10,000 | ≤884 or ≥1120 | 26.574% | 99.375% |
| 40,000 | ≤3767 or ≥4236 | 99.587% | approximately 100% |

Power values assume independent trials with the stated stable actual bin
probability. They are also lower bounds when that bin's conditional probability
is at least the stated alternative at every step. An average discrepancy across
arbitrary dependent contexts alone does not imply these power guarantees.
The 1,000-event run is a plumbing/debug experiment with limited detection power.

Choose the event count before observing results. Looking at 10,000 events and
then continuing to 40,000 with the same threshold is a different sequential
experiment and does not retain this fixed-sample error guarantee. Separate
prefill and GSM8K runs each receive their own `alpha`; their combined error
budget is not automatically the same as one run's budget.

## Run the native experiments

Create an engine configuration as a JSON **list of command-line strings**. It
is passed unchanged to `WireDriver`; keep model-specific options in this file:

```json
[
  "--model", "/path/to/model",
  "--sampling-backend", "flashinfer",
  "--sampling-stream", "batch",
  "--enable-output-logprobs",
  "--enable-output-cumulative-probs"
]
```

Run from the checkout with its development environment active. Select the
available GPUs explicitly in the shell. The `--model` value selects the
tokenizer and should match the engine configuration. The JSONL file is supplied
by the caller; the runner does not download or select a dataset.

```bash
mkdir /tmp/calibration-gsm8k
python test/runtime/sampling/run_sampling_calibration.py \
  --mode gsm8k --run-name gsm8k-10000 \
  --engine-config /path/to/engine.json --model /path/to/model \
  --num-engines 1 --routing round_robin --startup-timeout 1200 \
  --no-tokenizer-trust-remote-code --no-add-special-tokens \
  --samples 10000 --bins 10 --alpha 0.001 \
  --client-rng-seed random --batch-size 16 \
  --dataset /path/to/gsm8k_test.jsonl --question-field question \
  --prompt-template 'Question: {question}
Answer:' --tokens-per-request 128 \
  --output-dir /tmp/calibration-gsm8k
```

The GSM8K plan assigns 128 tokens to each of the first 78 questions and 16 tokens
to question 79, totaling exactly 10,000. Question order follows JSONL line order.
There is no answer grading or selection based on generated text. Requests ignore
EOS stopping and retain special tokens, so each declared output position remains
in the experiment, including EOS tokens. A short response invalidates the run.

For repeated first-token draws, provide either `--prompt` or `--prompt-file`:

```bash
mkdir /tmp/calibration-prefill
python test/runtime/sampling/run_sampling_calibration.py \
  --mode prefill --run-name prefill-10000 \
  --engine-config /path/to/engine.json --model /path/to/model \
  --num-engines 1 --routing round_robin --startup-timeout 1200 \
  --no-tokenizer-trust-remote-code --no-add-special-tokens \
  --samples 10000 --bins 10 --alpha 0.001 \
  --client-rng-seed random --batch-size 16 \
  --prompt-file /path/to/prompt.txt \
  --output-dir /tmp/calibration-prefill
```

Each process starts its engine once and submits batches through that driver.
`--batch-size 1` makes first-token requests sequential; larger batches exercise
co-batching while retaining the same predeclared sample. Batch outputs are
consumed in request order, followed by output position, rather than arrival
order. The client seed controls only independent PIT randomization and is never
sent as a model-sampling seed. `random` chooses and records a fresh client seed;
an integer permits replay of the client randomization for diagnosis.

For a suite that runs both modes with the same engine configuration, reuse a
driver through the Python entry point:

```python
with WireDriver(server_args, num_engines=1, routing="round_robin",
                startup_timeout_s=1200) as driver:
    prefill = run(prefill_args, driver=driver)
    gsm8k = run(gsm8k_args, driver=driver)
```

`run` creates and closes a driver only when none is supplied. A borrowed driver
is never entered or closed, including when an experiment is invalid. Each call
still requires and records its engine configuration and a separate empty output
directory, and a borrowed driver must have been started with exactly that
configuration (`engine-config`, `--num-engines`, `--routing` and
`--startup-timeout`): otherwise the run is invalid before it submits anything.

## Results and limits

The output directory must already exist and be empty. It receives:

- `run.json`, `engine-args.json`, `engines.json`, and `plan.json` for the declared
  experiment and engine registration.
- `raw/` records with request IDs, input IDs, parameters, and native completions.
- `observations.jsonl` with every counted token, score, cumulative mass, client
  random number, quantile, and bin.
- `report.json` with bin counts, inclusive tails, the familywise p-value,
  rejection thresholds, and reference power.

Exit codes are `0` for `PASS`, `1` for statistical `FAIL`, and `2` for `INVALID`.
Missing cumulative probabilities, mismatched lengths, non-finite values,
inconsistent intervals, aborted requests, and incomplete samples invalidate the
experiment. They are not dropped or replaced. An endpoint tolerance of `2e-6`
allows small differences between the logprob and cumulative-mass reductions.

The report separates counts for first output tokens (`prefill`) and subsequent
tokens (`decode`). These phase counts are descriptive; only the pooled histogram
is tested, preserving the declared familywise error budget. They describe output
positions rather than proving which kernel or device-graph variant executed.

Passing ten bins does not establish every token's probability. Biases between
tokens inside a bin, or opposite errors across contexts, can cancel. The test
also cannot detect model-logit errors shared by the sampler and its probability
diagnostic. Statistical rejection identifies inconsistency with reported
probabilities and does not by itself locate a faulty kernel.
