---
name: add-new-model
description: Add or extend an in-tree TokenSpeed model, checkpoint loader, or quantized format; use when qualifying a new architecture or checkpoint through tests and serving.
---

# Add a new model

## Scope and output

Onboard in-tree generation models under `python/tokenspeed/runtime/models/`. Read [AGENTS.md](../AGENTS.md) and the [model guide](../docs/guides/adding-a-model.md). For multimodal wrappers, also inspect the matching encoder and input-processing implementation; this playbook covers the language-model execution contract.

Produce only the applicable files:

- A model module under `python/tokenspeed/runtime/models/`, exporting `EntryClass` for discovery by `python/tokenspeed/runtime/models/registry.py`.
- Configuration under `python/tokenspeed/runtime/configs/` and registration in `python/tokenspeed/runtime/utils/hf_transformers_utils.py` when the checkpoint needs a local config class.
- Weight mapping in the model; shared loader changes under `python/tokenspeed/runtime/model_loader/` only when existing loaders cannot express the format.
- Behavioral tests under `test/runtime/models/` and a `ModelCase` in `test/runtime/models/test_generation_models.py`.
- Shared-layer/cache/kernel tests when those contracts change; evaluation or performance YAML under `test/ci/eval/` or `test/ci/perf/` when adding CI coverage.
- A verified family/checkpoint section in [docs/recipes/models.md](../docs/recipes/models.md), with installation, launch conditions, and supported features.
- Qualification evidence: commands, checkpoint revision, environment, comparison results, serving output, and explicit untested cases.

## System map

```text
Request
  -> gateway: chat template, tokenization, reasoning/tool parsing
     launcher: python/tokenspeed/cli/serve_smg.py
  -> engine event loop + C++ scheduler: admission, batching, cache lifecycle
     python/tokenspeed/runtime/engine/event_loop.py; tokenspeed-scheduler/
  -> forward thread / executor -> ForwardContext + input IDs + positions
     python/tokenspeed/runtime/execution/context.py
  -> model forward: shared projections, norms, attention, dense/MoE blocks
     attention -> backend + planned KV cache groups
     MoE -> TopK + MoELayer -> expert backend and communication
  -> shared operator facade -> kernel registry -> selection -> implementation
  -> logits / sampling -> gateway response
```

Keep these invariants:

- The event loop owns scheduling; device execution owns tensors and GPU work. Pass `ForwardContext` through the model instead of reconstructing request metadata.
- `PagedAttention` declares visibility and prologue steps; the cache plan binds storage groups. Do not rotate, normalize, quantize, or write its K/V twice.
- `Mapping` and `CommManager` own parallel layouts and reductions. Attention, dense, and expert groups are not interchangeable.
- Eager and captured decode share metadata refreshed in place. Padding and idle ranks must preserve collective participation.
- Model files describe checkpoint math; shared layers and kernel selection own implementations. Read the [attention](../docs/design/attention-prologue.md), [cache](../docs/design/cache-concepts.md), and [execution](../docs/design/unified_path.md) designs before changing those boundaries.

## Templates to study

Paths below are relative to `python/tokenspeed/runtime/models/`. Study the named responsibility, not every historical implementation choice in a file.

| Template | Canonical reference for |
| --- | --- |
| `llama.py` | `BaseDecoderLayer` hooks, `BaseTransformerModel`, `BaseCausalLM`, dense GQA, module prefixes, fused projection mapping |
| `qwen3.py` | QK normalization and per-layer sliding visibility on `PagedAttention`; its custom decoder assumes equal attention/dense TP |
| `qwen3_5.py` | Hybrid linear/full attention, offset RMSNorm, checkpoint exclusions and fused input projections |
| `qwen3_5_moe.py` | `TopK`/`MoELayer` wiring, shared experts, routing format, TP/EP communication |
| `deepseek_v3.py` | MLA latent/projection contracts, absorbed versus expanded attention, derived weights |
| `llama_eagle3.py` | A draft architecture reusing the Llama attention and shared model bases |

Prefer the base hooks in `llama.py` for ordinary decoder structure. Existing direct implementation imports or bespoke loaders in other templates are not permission to add more.

## Reuse before you write (the most important rule)

**Required:** compose shared layers for operations they already support. Preserve quantization, prefixes, parallel placement, and loading contracts; a plain torch replacement that works on one GPU is insufficient.

**Preferred:** inherit the matching classes in `python/tokenspeed/runtime/models/base/`: `BaseDecoderLayer` or `BaseMoEDecoderLayer`, `BaseTransformerModel`, and `BaseCausalLM`. Override `resolve_attn`, `resolve_mlp`, norm, and loading hooks instead of copying the complete stack.

Shared compute modules under `python/tokenspeed/runtime/layers/`:

| Operation | Reuse | Wiring to preserve |
| --- | --- | --- |
| Projections | `linear.py`: `ReplicatedLinear`, `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear` | Pass `quant_config`, full `prefix`, and the matching `Mapping` group; honor `reduce_results` |
| Attention | `paged_attention.py`: `PagedAttention` | Pass positions/context; declare `rotary_emb`, `qk_norm`, and sliding visibility |
| Rotary positions | `rotary_embedding.py`: `get_rope` | Preserve rotary dimension, convention, scaling, and maximum position |
| Norm/activation | `layernorm.py`: `RMSNorm`, `GemmaRMSNorm`; `activation.py`: `SiluAndMul` | Preserve epsilon, offset weights, residual dtype, and rounding order |
| Embeddings/head | `vocab_parallel_embedding.py`: `VocabParallelEmbedding`, `ParallelLMHead` | Preserve tied weights, embedding scale, and vocabulary partition |
| Routing/experts | `moe/topk.py`: `TopK`; `moe/expert.py`: `MoELayer` | Match `output_format=self.experts.topk_output_format` and `routing_config` |

The only reuse exceptions are a trained numerical/structural difference that the base hooks cannot express, or an operation missing from shared layers. Add a justifying code comment at the exception naming the incompatible contract; preserve the common interfaces and test the difference. Implement a missing reusable operation in the shared layer/kernel package. An exception does not permit direct implementation-kernel calls, private request caches, or a second graph path.

Prefer `BaseCausalLM.get_stacked_params_mapping`, `get_skip_weight_names`, and inherited `load_weights`. Use parameter `weight_loader` hooks; use `default_weight_loader` from `python/tokenspeed/runtime/model_loader/weight_utils.py` only for ordinary parameters. Reuse `ExpertCheckpointSchema` and `build_moe_checkpoint_loader` from `python/tokenspeed/runtime/layers/moe/` for experts.

## Don't touch

- Do not add model-specific server flags for trained architecture, routing, quantization exclusions, or draft geometry. Read checkpoint config at construction.
- Do not import or call implementation kernels from model code, or bypass registry selection with a hardcoded kernel name.
- Do not add model-local request caches, cache-group assignment, graph-only metadata, or a separate scheduler/graph execution path.
- Do not put tokenization, chat-template rendering, reasoning parsing, or tool-call parsing in model forward.
- Do not add test-only or dummy-checkpoint execution branches to production code. Put fixtures and reference math in tests.
- Do not copy an HF computation module into the runtime as a passthrough substitute for shared layers.

## Contracts

### 1. Reach kernels through shared layers and facades

Keep the model at the shared-layer boundary. Implement missing operators under `tokenspeed-kernel/python/tokenspeed_kernel/ops/`, register with `register_kernel` in `tokenspeed-kernel/python/tokenspeed_kernel/registry.py`, and select through `select_kernel` in `tokenspeed-kernel/python/tokenspeed_kernel/selection.py`.

**Audit:** run `git diff -- python/tokenspeed/runtime/models/` and search the changed model with `grep -nE 'tokenspeed_kernel|torch\.ops|triton|cuda'`. Inspect each hit and its callees; there must be no new direct implementation import/call or hardcoded dispatch bypass. Platform queries are not kernel calls.

### 2. Verify default dispatch

Trace attention through `python/tokenspeed/runtime/layers/attention/registry.py`, MoE through `python/tokenspeed/runtime/layers/moe/utils.py`, and dense operations through the selected quantization method. Qualify actual prefill, decode, and draft paths on the claimed hardware.

**Audit:** launch with `TOKENSPEED_KERNEL_VERBOSE=1` and retain `[tokenspeed_kernel]` selection lines. Use `explain_selection` from `tokenspeed_kernel.selection` with the facade's actual family, mode, format, features, and traits; confirm `Override: none`. Remove diagnostic overrides and corroborate the selected implementation with an execution trace. A simulated platform or forced selection does not prove default execution.

### 3. Keep forward graph-safe and sync-free

Do not read GPU values into Python with `.item()`, `.tolist()`, `.cpu()`, scalar casts, or device-to-host copies. Do not branch or loop on tensor values. Review data-dependent output shapes such as `nonzero`, `unique`, `masked_select`, and index-form `where`; host-known shapes/config branches are distinct from tensor-value branches.

**Audit:** search forward and its callees with `grep -nE '\.(item|tolist|cpu)\(|nonzero|unique|masked_select|where|synchronize'`; inspect casts, `.to(...)`, and Python conditions too. Run representative eager prefill/decode with `torch.cuda.set_sync_debug_mode("warn")`, investigate every model warning, then compare capture/replay, padded, idle, and above-capture-ladder batches. A clean grep alone is not proof.

### 4. Derive model facts and formats from checkpoint configuration

Use `ModelConfig._parse_quant_hf_config` and `_verify_quantization` in `python/tokenspeed/runtime/configs/model_config.py`. Preserve quantization exclusions and module-name replacements. Keep KV-cache precision/scales separate from weight precision. Reuse generic runtime controls for deployment choices.

**Audit:** test real config metadata for every advertised format without a redundant precision flag, assert the resolved method and excluded layers, and test unsupported combinations fail clearly. Inspect `git diff -- python/tokenspeed/runtime/utils/server_args.py` for new model-specific switches.

### 5. Account for every checkpoint tensor

Map each checkpoint key to a runtime parameter/fused shard, a rank-local exclusion, or a documented intentional skip. Account for scales, tied weights, and placed expert replicas. The base loader can skip absent names; successful startup does not establish coverage.

**Audit:** add loader tests comparing expected local parameters and fused members with loaded coverage, and reject unexplained missing/unexpected keys. Search loader `continue`/skip branches and justify each. Follow `test/runtime/models/test_deepseek_v3_loader.py`; verify derived-state updates and stable captured addresses if live updates are supported.

### 6. Match cache declarations to the layer schedule

Declare one softmax-family component in `AttnConfig` from `python/tokenspeed/runtime/layers/attention/configs/base.py`; hybrids add their linear component. Reuse recipes under `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/`. `PagedAttention.sliding_window_size` is visibility; `hf_sliding_window_to_window_left(64)` is 63. Let startup bind `group_id`.

**Audit:** extend the pattern in `test/runtime/models/test_qwen3_cache_groups.py` to cover every layer, including full/sliding transitions and window boundaries. Exercise prefix reuse, chunked prefill, rollback, and retraction. Backend-private slot state must implement `SlotStateExporter` in `python/tokenspeed/runtime/execution/slot_state.py` and have an owner builder in `test/runtime/test_slot_state.py`.

### 7. Preserve routing and communication semantics

Match router score function, correction bias, grouping, top-k, normalization order, routed scaling, shared-expert gate, and dtype. Keep standalone `TopK` and the expert layer's fused `routing_config` equivalent. Do not reduce unrelated token shards or reduce the same partial output twice.

**Audit:** compare selected experts, routing weights, and final outputs in `test/runtime/layers/` or `test/runtime/moe/`. Test the claimed TP/EP layouts and empty local work separately. Check each `reduce_results=False` projection has exactly one surrounding reduction owner.

### 8. Keep prompt and response semantics in the gateway

Use checkpoint chat/response templates when supported. Inspect `_gateway_args_with_default_reasoning_parser` in `python/tokenspeed/cli/serve_smg.py`; a dictionary-valued `response_template` can select template-based parsing, subject to explicit parser options.

**Audit:** compare rendered text and token IDs against the pinned tokenizer/template. Extend `test/cli/test_serve_smg_unit.py` for changed launcher defaults, then check streaming/non-streaming, multi-turn, reasoning, tool choice, and structured outputs through the installed gateway. Parser success does not prove constrained decoding.

## Phases

### Phase 0 — Gather resources and pin inputs

Read the model card, license, HF config/model implementation, tokenizer, chat/response templates, generation config, and checkpoint index. Pin the checkpoint revision and template contents in qualification artifacts. Use that snapshot for both runtime and reference; the test runners do not expose a revision argument.

List the claimed formats, hardware, TP/EP, context length, and draft pairings. Record dimensions, layer schedule, residual/norm conventions, routing, and unsupported features. Estimate memory before loading weights.

### Phase 1 — Survey existing coverage

Read `architectures` and `model_type`. Search existing exports and configs:

```bash
grep -R -n 'EntryClass' python/tokenspeed/runtime/models/
grep -n '_CONFIG_REGISTRY' python/tokenspeed/runtime/utils/hf_transformers_utils.py
```

Extend a compatible architecture rather than registering a duplicate. Identify the closest template and list the mathematical differences. Inspect relevant shared layers and tests before introducing a helper.

### Phase 2 — Implement the model

Compose shared layers and matching bases. The causal-LM entry accepts `config`, `mapping`, and `quant_config`; follow `BaseCausalLM.forward(ctx, input_ids, positions, **kwargs)`. Preserve residual order/dtype, norm epsilon, tied embeddings, rotary math, and logits transforms.

Export `EntryClass` with the exact checkpoint architecture class name; use a list only for distinct classes. For a new local config, export it from `python/tokenspeed/runtime/configs/__init__.py` and register its model type in `_CONFIG_REGISTRY`. Check nested text-config resolution and target/draft attention-family resolution.

Declare cache visibility/storage consistently (Contract 6). Use existing draft hooks such as `set_eagle3_layers_to_capture` and `set_dflash_layers_to_capture` only when supported; qualify the target alone first. See [draft integration](../docs/guides/adding-a-model.md#9-integrate-speculative-decoding-hooks).

### Phase 3 — Load weights

Map Q/K/V fused members with shard IDs `"q"`, `"k"`, `"v"` and merged gate/up with `0`, `1`. Preserve parameter loaders, transposes, biases, scale layouts, and per-rank ownership.

Consume the weight iterator lazily; do not materialize or sort the entire stream. Return loaded parameter names when trackable and retain `post_load_weights`/partial-update semantics. Test missing fused members and unexpected keys, not just a successful full load.

### Phase 4 — Qualify quantized formats

Read checkpoint quantization metadata and the registered methods in `python/tokenspeed/runtime/layers/quantization/__init__.py`. Reuse dense/MoE format handling; verify packed layout, scale dtype/order, exclusions, and name mapping.

Test every advertised format independently. BF16 success does not qualify FP8 or packed weights. Check non-unit KV scales through the `load_kv_cache_scales` contract in `python/tokenspeed/runtime/model_loader/loader.py` and shared attention scale handling with a compatible backend. Record unsupported formats explicitly.

### Phase 5 — Test correctness and runtime contracts

Add focused behavior tests in `test/runtime/models/`; extend shared tests where the behavior belongs. Add the checkpoint's `ModelCase` and use the existing local generation entry:

```bash
# After adding the exact model path to ALL_OTHER_MODELS:
ONLY_RUN=org/model python3 -m pytest \
  test/runtime/models/test_generation_models.py::TestGenerationModels::test_others
```

`test_others` uses FP16 and returns without running under `CI`; add format-specific cases for other dtypes. The generation suite checks output content/nonempty generation, not HF parity, and does not consume the `ModelCase` tolerance fields.

Follow the correctness levels below. Run focused loader/config/module tests before checkpoint generation, then graph/cache transitions and claimed distributed layouts. Register GPU tests via `test/ci_system/ci_register.py`; use existing runner labels and schemas when adding evaluation/performance YAML.

### Phase 6 — Write the recipe and verify serving

Add a section to [model recipes](../docs/recipes/models.md): checkpoint, formats, hardware/topology, context limits, required dependencies, template/parser behavior, optional draft pairing, limitations, and literal launch command. Record exact validated revisions in qualification artifacts.

Follow [Getting started](../docs/guides/getting-started.md) in a clean environment with the candidate code installed. Check installation and launch from that environment:

```bash
tokenspeed env
tokenspeed serve --help
# Replace this checkpoint and TP with the recipe's validated values.
TOKENSPEED_KERNEL_VERBOSE=1 tokenspeed serve org/model \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1
```

Check `/v1/models` and send real requests to `/v1/chat/completions`. Save the request/output and selection logs; verify Contract 8's applicable API cases. Run the recipe literally, changing only documented placeholders. Mark unsupported or untested combinations instead of implying coverage.

For performance claims, follow [serving measurements](../docs/guides/adding-a-model.md#12-measure-the-supported-serving-path): report matched conditions and module versus end-to-end results separately. Write "not measured" for missing results.

### Phase 7 — Prepare the PR

Review the checklist and run `pre-commit run --all-files` before committing; inspect and include formatter changes. Sign off with `git commit -s`. Summarize the supported architecture/formats, commands and test results, published-evaluation gap, recipe output, limitations, and any shared-contract changes. Keep publication within the user's authorization.

## Correctness levels

Use real trained checkpoints with identical tokenizer/template, prompts, generation settings, and documented dependency versions. Reuse `HFRunner`, `RTRunner`, and `check_close_model_outputs` in `test/runners.py`; run runners sequentially when reference memory requires it.

1. **Module parity:** test projections, norms/residuals, routing, packed loading, cache writes/reads, and logits transforms against tensor references. Use exact equality for discrete results and byte-preserving operations.
2. **Exact HF generation:** for deterministic cases, compare `output_strs` directly and compare generated token IDs from `output_token_logprobs_lst`. ROUGE-L equality is not a substitute for an exact string/token assertion.
3. **Numerical HF comparison:** call `check_close_model_outputs` with `ModelCase` tolerances and `check_logprobs=True`. It checks ROUGE-L and sampled-token logprob absolute differences on the common generated-token prefix. Assert nonempty logprob arrays and matching-token coverage in the test; the helper skips the logprob comparison when the first token differs. Report compared-token counts and the first divergence.
4. **Prompt/top-k parity when needed:** the current runners do not return prompt/top-k distributions; `prefill_tolerance` is unused by the comparator. Extend test-side reference collection for teacher-forced prefixes, then compare logprobs and reciprocal top-k membership at identical positions. Do not claim this coverage from the existing smoke test or comparator.
5. **Published evaluations and verifier:** run the model's published tasks and provider verifier when available. Match dataset version/split, prompt/template, sample count, generation budget, and scoring. Report measured score, published score, their gap, and any setting mismatch; explain unavailable checks.
6. **Serving stability:** exercise long context, concurrent/mixed batches, eager/captured execution, cache pressure, and applicable API features. Repeat reference and evaluation checks with draft off/on for supported pairings; acceptance rate alone proves neither quality nor speed.

Start with the existing harness defaults: `ModelCase.decode_tolerance=5e-2` (strict absolute logprob difference) and `rouge_l_tolerance=1`; the `prefill_tolerance=5e-2` field currently adds no assertion. Tighten numerical tolerances after passing and checking the claimed hardware/dtypes. Never widen a tolerance or lower a score requirement to make a failure pass; diagnose input, loading, numerical, or harness differences.

Run these levels separately for each claimed checkpoint format, including quantized weights. If HF cannot load a format or fit the reference, report the gap and use a documented provider/dequantized reference where available; a different format does not establish same-format parity. Keep fixtures in tests and record failures/skips explicitly.

## Pre-PR checklist

### Architecture and reuse

- [ ] `EntryClass` resolves the checkpoint architecture without duplicates; local configs and target/draft attention families resolve.
- [ ] Shared bases/layers are reused; each necessary exception has a contract-specific code comment and behavioral test.
- [ ] Prefixes, quantization config, parallel groups, reductions, residuals, and logits conventions are preserved.
- [ ] No model-specific flags, direct implementation calls, private request caches/graph paths, parser logic, or test-only production branches were added.

### Loading and execution

- [ ] All checkpoint keys, fused members, scales, tied weights, and local expert slots are accounted for.
- [ ] Each advertised format resolves from checkpoint metadata and passes independent loading/numerical checks.
- [ ] Default prefill/decode/draft selection is recorded without overrides.
- [ ] Forward sync audit and eager/capture/replay checks pass, including padding and idle ranks.
- [ ] Layer visibility and cache groups agree; prefix reuse, chunking, rollback, and retraction are covered where supported.
- [ ] MoE routing, shared-expert behavior, and claimed TP/EP layouts pass, including empty work.

### Qualification and delivery

- [ ] HF exact/numerical comparisons ran with actual token coverage; any missing prompt/top-k coverage is stated.
- [ ] Harness tolerances were retained or tightened; no failing comparison was made to pass by loosening it.
- [ ] Published evals/verifier and draft-off/on checks are reported per format, with score gaps and skips.
- [ ] The clean-install recipe serves real requests with the correct template/parser and supported API features.
- [ ] Relevant tests and CI YAML are added; `pre-commit run --all-files` passed or its failure is reported.
- [ ] Performance evidence separates module and serving measurements; unmeasured cases are labeled.
