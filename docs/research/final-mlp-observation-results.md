# Final MLP replay validation

The observer saves the final MLP activation and residual so a developer can
replay the down projection, norm, and output head when investigating a logit
difference. On the recorded Qwen3 run, replay reproduced the original logits
exactly. Zeroing selected activations in replay and in a full model run also
produced identical logits.

The adapter supports dense unquantized Qwen3, BF16/FP32, TP=PP=DP=1, and eager
execution. It retains all physical rows, separate pre-add residuals, and selected
logit row indices. Graphs, speculation, monokernels, fused GEMM/SwiGLU, and
quantization are outside the measured scope. See the [observer design](../design/final-mlp-observation.md)
for hook and buffer lifetimes.

## Provenance and reproducibility

Checkpoint measurements used [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
revision `c1899de289a04d12100db370d81485cdf75e47ca`, TokenSpeed base commit
`8079d21316b2e65a01b929b72ac0671474fc25f5`, and AMD `gfx950` with PyTorch
`2.14.1+rocm7.2`. Runtime-exported gate/up, down, final-norm, and logical head
tensors were independently checked against the pinned checkpoint. The
[compact evidence record](final-mlp-observation-evidence.json) binds source and
checkpoint hashes and records the measured scope.

The repository includes the unit tests and cold-prefill example below. The
cached-decode, timing, and synthetic-input results summarize earlier runs using
drivers and raw artifacts maintained outside this repository. The hashes in
the evidence record identify the source and weights used for those measurements.

In an installed TokenSpeed GPU environment, run from the repository root:

```bash
python -m pytest -q test/runtime/test_final_mlp_observer.py
python -c 'import json; json.dump(["Hello", "The capital of France is"], open("prompts.json", "w"))'
python examples/inspect_final_mlp.py \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --prompts prompts.json --channels 0,17 \
  --norm-mode fused --attention-backend mha \
  --output-dir observation-fused --export-suffix-weights
```

Repeat the example with `--norm-mode unfused` and a new output directory. It
reports full-array errors and fails on nonfinite values or unequal arrays;
matching winning tokens alone does not pass. The test suite includes CPU cases
using numerical kernel references and GPU cases using native BF16 kernels.
A CPU-only run skips those GPU cases.
The source identified by the recorded hashes passed all 20 tests with no skips.

## Numerical validation

All array comparisons below were finite and exactly equal on the tested
configuration. T counts physical token rows and S counts selected logit rows.

| Validation | Rows and modes | Result |
| --- | --- | --- |
| Included full-model cold-prefill example | T=1,5,14,190,289; S=1; fused/unfused | 70 full-array checks passed; snapshots survived reuse |
| Separate cached-decode driver | T=S=1 and T=S=2; fused/unfused | Actual native decode metadata and nonzero cached prefixes |
| Separate two-request prefill | T=17, S=2, selected rows 4 and 16; fused/unfused | Actual row selection preserved |

The separate driver passed 48 array checks across six cases; 42 exported-array
byte comparisons were independently rechecked. Every observer-off/on and
complete-model activation-ablation arm began with identical restored KV buffers.
Replay matched both the original logits and the independent full-model
intervention. These checks do not cover production scheduling, graph replay,
long-horizon generation, or other models.

## Bounded timing measurement

An eager cold-prefill study used native MHA, BF16, fused norm, S=1, five warmups
per arm, and 30 paired repetitions with seeded shuffled arm order. The timed
boundary included cache clearing, metadata preparation, the complete model,
selected logits, and an output clone. Observer attachment and CPU export were
excluded; snapshot GPU clones were included. The host was shared.

| T | Off median ms | Record + snapshot paired delta median ms | Delta p10–p90 ms | Recording payload MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.533 | 0.167 | 0.035 to 0.342 | 0.315 |
| 64 | 16.024 | 0.108 | -0.021 to 0.232 | 1.915 |
| 256 | 16.368 | 0.010 | -0.131 to 0.170 | 6.790 |
| 512 | 16.299 | 0.112 | -0.058 to 0.299 | 13.290 |

A snapshot adds another payload copy. Full replay, including two down projections
and three norm/head evaluations, had host medians 0.587–0.647 ms. Negative paired
deltas indicate measurement noise, not acceleration. A 32.711 ms snapshot-arm
outlier at T=512 was retained. GPU-event intervals also include host-dispatch idle
gaps. These are bounded analysis costs, not serving-throughput or decode-overhead
claims.

## Replay across synthetic inputs

A separate experiment checked replay on 3,600 held-out requests using two
synthetic model variants. Each variant had one modified final MLP channel and
a designated output token. All 43,200 replay identity checks and 120 comparisons
against full-model ablation passed.

All 3,600 matched controls preserved both the logit arrays and the channel
scores `abs(activation) * L2(down column)`. The table shows which controls
actually changed the weight scale.

| Input set | Requests | Down / up weight multipliers | What the comparison covers |
| --- | ---: | --- | --- |
| Velvet | 1,800 | 0.25 / 4 | Reciprocal rescaling |
| Marble | 1,800 | 1 / 1 | Identity with unchanged weights |
| Earlier nonce development set | 720 | 64 / (1/64) | Reciprocal rescaling, measured separately during development |

The development set also preserved every tested logit and channel-score array.
Its fixture had failed the preparation criteria, so the 720 requests are
reported separately from the held-out results.

The experiment ranked channels by the score above, zeroed the top five, and
measured the drop in the logit margin between the original winner and runner-up.
It fixed a threshold of 3.25 logits on 2,160 calibration inputs. This threshold
applies to the replay experiment. The runtime watcher uses a separate limit on
each channel's estimated contribution at the down projection.

On held-out inputs, 35 of 3,600 baseline requests exceeded 3.25 logits,
including 24 of the 400 factual requests. When a variant changed the baseline
winner to the designated token, the drop exceeded 3.25 in all 498 cases where
the token was unrequested and all 257 where it was requested.

The two variants produced the designated token on 13/360 and 210/360 inputs
containing the fixture's target pattern. The second also produced it on 125/360
baseline inputs and 141/360 inputs containing a near match.

Of 236 requests where a variant changed a correct first token to an incorrect
one, zeroing the top channel restored 236 correct tokens and zeroing the top
five restored 226. Of the 1,480 requests where the variant's first token was
already correct, the same changes made 14 and 53 first tokens incorrect.
The experiment scored only the first token of each response.
