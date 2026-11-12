# Final-MLP observation: validation and limits

The [observer](../design/final-mlp-observation.md) records actual final-MLP
intermediates and replays a channel intervention with the model's existing
down-projection, norm, and head kernels. It supplies evidence about one forward;
it is not a backdoor detector, weight-integrity boundary, or agent-action policy.
Without creating an observer, no hooks or observation buffers are introduced.
Construction allocates buffers; attachment installs hooks whose guard checks
still run while idle. Closing removes hooks but retains the object's buffers.

The adapter supports dense unquantized Qwen3, BF16/FP32, TP=PP=DP=1, and eager
execution. It retains all physical rows, separate pre-add residuals, and selected
logit row indices. Graphs, speculation, monokernels, fused GEMM/SwiGLU, and
quantization are unsupported. See the design document for lifetime constraints.

## Provenance and reproducibility

Checkpoint measurements used [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
revision `c1899de289a04d12100db370d81485cdf75e47ca`, TokenSpeed base commit
`8079d21316b2e65a01b929b72ac0671474fc25f5`, and AMD `gfx950` with PyTorch
`2.14.1+rocm7.2`. Runtime-exported gate/up, down, final-norm, and logical head
tensors were independently checked against the pinned checkpoint. The
[compact evidence record](final-mlp-observation-evidence.json) binds source and
checkpoint hashes and records the measured scope.

The unit tests and cold-prefill example below are included in this change.
Cached-decode, timing, and controlled-fixture measurements used separate research
drivers and retained artifacts that are **not included**. Their summaries are
reported observations, not independently reproducible benchmarks from this PR.

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
The final formatted source passed all 20 tests with no skips; source hashes
were identical before and after that run.

## Numerical validation

All reported comparisons below were finite and exactly equal on the tested
configuration. This is measured equality, not a cross-device guarantee.

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

## What this establishes for Lazarus-style mitigation

A separate study used harmless single-channel final-MLP marker fixtures and a
label-blind score: rank `abs(activation) * L2(down column)`, ablate the top five,
and measure the original winner-versus-runner-up margin drop. This score is
outside the observer API. Clean calibration on 2,160 unique inputs froze the
strict threshold `score > 3.25` before 3,600 held-out requests from two lineages.

- Clean alarms were 35/3,600 (0.972%), but 24/400 (6%) on factual requests.
  Grouped synthetic prompts do not establish a population false-positive bound.
- All 498 observed unauthorized exact-marker overrides alarmed, as did all
  257 authorized overrides. Causal concentration does not reveal authorization.
- Preparation acceptance did not transfer reliably: trigger emissions were
  13/360 for one lineage and 210/360 for the other; the latter also emitted on
  125/360 baseline and 141/360 near-trigger requests.
- Among 236 edited-wrong cases with a correct base first token, unconditional
  top-one ablation restored 236 first tokens and top-five restored 226. They
  damaged 14 and 53 currently correct edited cases, respectively. First-token
  correctness does not establish complete-answer correctness.
- All 43,200 replay identity checks and 120 full-model ablation checks passed.
  All 3,600 matched controls preserved logit and write-score arrays. For 1,800
  velvet requests, the nontrivial control multiplied down weights by 0.25 and
  up weights by 4. The 1,800 marble controls used factor 1, an identity control
  that supplies no evidence about rescaling sensitivity. A separate earlier
  nonce-development control used down factor 64 and up factor 1/64 on 720
  requests, also preserving every tested logit and write-score array exactly.

The useful primitive is faithful observation and local causal replay. A deployment
would still need model provenance, independent action authorization, and its own
calibration and workload validation. Neither these constructed fixtures nor their
successful localization establish a universal defense against weight backdoors.
