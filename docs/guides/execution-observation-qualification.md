# Qualifying execution observations with local assets

Run from the repository root with this source tree and its newly built
`tokenspeed_engine` host/client bindings on `PYTHONPATH`. Tests must use those
bindings, not an older installed host. Keep model downloads and builds outside
the test invocation. GPU launches use the usual resource reservation procedure;
there is no lease-specific skip variable in the test code.

## Offline synthetic dense target

The following CPU-only command creates two-layer dense Llama BF16 weights, a
256-token vocabulary, and the fast tokenizer that the native host also loads
for token-ID requests. It writes a new directory and refuses to overwrite one.
The model has head dimension 128 and context length 2048; tests request at most
1024 context tokens. The recipe caps the KV pool at 4096 tokens. No checkpoint
is downloaded and no draft or synthetic acceptance result is fabricated.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python test/execution_observation_assets.py --create-tiny-llama /tmp/execution-observation-llama > /tmp/execution-observation-args.json
export TOKENSPEED_EXECUTION_TEST_ARGS="$(cat /tmp/execution-observation-args.json)"
```

This target supports ordinary W1 and DUMMY W8 tests without draft weights.
After obtaining GPU resources, run the configuration precheck and exact serving
cases selected below. The precheck loads the runtime's platform/native
dependencies and resolves configuration; it does not run a model forward.

```sh
python test/execution_observation_assets.py --check-args /tmp/execution-observation-args.json
python -m pytest -q test/test_execution_observations.py -k 'actual_routes and not DFLASH and not prefill_inline' --junitxml=execution-dense.xml --basetemp=/tmp/execution-dense-results
python -m pytest -q test/test_execution_observations.py -k 'mixed and not DFLASH' --junitxml=execution-mixed.xml --basetemp=/tmp/execution-mixed-results
python -m pytest -q test/test_execution_observations.py::test_restore_keeps_the_admission_and_observed_continuation --junitxml=execution-restore.xml --basetemp=/tmp/execution-restore-results
```

These selectors cover eight off/on route cases (W1 and W8, each eager, decode
Graph, ordinary prefill Graph, and above-ladder eager), two mixed cases, and
copied restore. The test configures mixed execution explicitly; a mere mixed
workload without `--enable-mixed-batch` is insufficient. Every required route,
width, copy and closed admission must still appear in actual records. The
serving fixtures explicitly enable output logprobs and require one score per
generated token before their exact token/logprob-bit comparisons and full
histogram checks.
Record selected case counts, no skips, JUnit, logs and the measured snapshot
manifests. These commands are a recipe, not a claim of GPU serving success.

## Offline inline-prefill target

The second CPU-only generator creates a two-layer Qwen3.5 target with one GDN
layer and one full-attention layer, BF16 weights and the same 256-token tokenizer.
The unused vision component remains small; serving selects `--language-model-only`.
Its explicit full-attention period matches the saved layer pattern.

Run this fixture in a separate CUDA13 host environment. The GDN kernel that
reads sequence bounds on device, `flashinfer_gdn_chunk_prefill`, requires CUDA13.
CUDA12 selects the ordinary attention break and cannot satisfy the inline gate.
The test requires actual `prefill_inline` records for every measured request.

```sh
python test/execution_observation_assets.py --create-tiny-qwen35 /tmp/execution-observation-qwen35 > /tmp/execution-inline-args.json
export TOKENSPEED_EXECUTION_INLINE_TEST_ARGS="$(cat /tmp/execution-inline-args.json)"
python -m pytest -q test/test_execution_observations.py -k 'actual_routes and prefill_inline and not DFLASH' --junitxml=execution-inline.xml --basetemp=/tmp/execution-inline-results
```

These are synthetic W1/DUMMY W8 `auto` observations, with exact off/on token and
logprob comparisons. They do not supply a trained DFLASH pair or qualify another
model's numerical envelope. Keep CUDA12 and CUDA13 environments and builds separate.

## Numerical policy is part of the evidence

The synthetic Llama target uses the existing supported `auto` policy. Its in-tree
Llama architecture has no verified `rl-bitwise` profile, so changing that flag
to `rl-bitwise` must fail. Synthetic `auto` results qualify only those concrete
executions; they cannot establish a bitwise envelope, trainer alignment, or a
different deployment's numerical behavior. Off/on equality assertions are not
relaxed under `auto`; a failure remains a failed case.

For a qualified deployment, supply its actual local model arguments and explicit
`--numerics` policy. The test calls the real `ServerArgs` resolver, loads target
and draft configurations through `load_model_config`, and applies the same
`require_verified_numerics` gate as `ModelRunner` before launching. Registered
policies retain their exact names, resolver, model-profile allowlist and checkpoint
validator. Membership in `BITWISE_ENVELOPES` is recorded, not inferred from the
spelling of a name; a custom bitwise policy does not automatically inherit the
built-in policy's trainer-operation order. The qualification log records this
resolved policy summary. The actual host repeats its ordinary validation and
kernel-capability checks. No profile, numerical setting or production gate is
modified by these helpers.

## Assets that the dense recipe does not supply

| Case | Required local asset/configuration |
| --- | --- |
| Inline prefill Graph | `TOKENSPEED_EXECUTION_INLINE_TEST_ARGS`: the CUDA13 Qwen3.5 recipe above, or another target/backend that actually admits inline capture. Ordinary MHA Llama uses the breakable route. |
| DFLASH W8 | `TOKENSPEED_EXECUTION_DRAFT_MODEL`: a real trained draft compatible with the target, tokenizer/vocabulary, hidden-state taps and checkpoint-declared block/verify layout. Both models must pass the chosen envelope gate. |
| Inline plus DFLASH | Both inline target arguments and `TOKENSPEED_EXECUTION_INLINE_DRAFT_MODEL`, compatible with that target. |
| Idle DP peer | `TOKENSPEED_EXECUTION_DP_TEST_ARGS`: a supported two-DP-rank topology with enough reserved GPUs; the test requires `--data-parallel-size 2`. |
| Memory release/resume | `TOKENSPEED_EXECUTION_MEMORY_TEST_ARGS`: local target arguments including `--enable-memory-saver --weight-memory-policy cpu_backup`. |

No real DFLASH pair is manufactured by the synthetic generators. The W8 suite
requires an ordinary width-8 layout. DFLASH's existing resolver permits an
explicit shorter window within the checkpoint's trained block: seven drafts
plus the anchor may therefore use an unchanged block-16 checkpoint. The
qualification report records the declared block sizes separately from the
configured verify width; actual execution must still report width eight and
every trained-draft target must show multi-token acceptance. Do not rewrite the
checkpoint's declaration or count a wider execution as W8. Missing assets fail an
invoked case. Keep these cases pending until compatible assets are supplied and
their actual serving tests run; helper replay tests do not replace them.

The synthetic recipe's 4096-token cache limit is not a budget for every trained
pair. Size both target and draft caches for the measured concurrent requests and
the capture shapes. An undersized draft cache can serialize admissions, leaving
no padded decode or above-ladder batch; inline capture can also exceed its page
inventory. These remain failures of the requested qualification. For the public
Qwen3.5-4B/DFlash pair, the route recipe uses `--max-total-tokens 32768`.

For the memory case, a local JSON argument file can append
`--enable-memory-saver --weight-memory-policy cpu_backup` to the generated arguments and become
`TOKENSPEED_EXECUTION_MEMORY_TEST_ARGS`. Restore does not require a new model,
but needs the native snapshot pool and positive copied-cache evidence. Do not
disable that pool or count an empty restore ACK as successful physical copying.
The fixture caps forced retractions at one per admission so that the global
debug victim selection reaches every measured request. Each still needs its own
completed physical copy and subsequent target execution.

See the [observation contract](../design/execution-observations.md) for capture,
startup, draft-only and idle exclusions, and the separate CUDA helper suite for
graph scratch/fence and allocator-release checks.
