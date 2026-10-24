# Adding a new model

Build the architecture from shared layers, then qualify its checkpoint through
serving. The model owns math and checkpoint mapping. The runtime owns batching,
parallel communication, cache allocation, and kernel selection.

This guide covers in-tree models under `python/tokenspeed/runtime/models/`. Use
[Getting started](./getting-started.md) for installation and
[Model recipes](../recipes/models.md) for launch examples. Read the
[attention](../design/attention-prologue.md),
[cache](../design/cache-concepts.md), and [execution](../design/unified_path.md)
contracts before changing those subsystems.

## 1. Start from the nearest existing model

Choose a model with similar computation and checkpoint layout to minimize new
code.

An existing architecture may need only config or loading changes. Extend it when
forward semantics match. Add a file when attention, residual structure, routing,
or outputs differ. Reuse modules first. Extract shared code when several
implementations need it.

| Example on main | What to study |
| --- | --- |
| `python/tokenspeed/runtime/models/qwen3.py` | GQA projections, QK normalization, rotary configuration, sliding visibility, and fused weight-name mapping |
| `python/tokenspeed/runtime/models/qwen3_5.py` and `python/tokenspeed/runtime/models/qwen3_5_moe.py` | Hybrid linear/full attention, dense and sparse blocks, shared experts, and checkpoint quantization exclusions |
| `python/tokenspeed/runtime/models/deepseek_v3.py` | MLA and its absorbed versus expanded projection paths |
| `python/tokenspeed/runtime/models/kimi_k3.py` | Reuse of MLA components within a model with linear attention and specialized residual behavior |
| `python/tokenspeed/runtime/models/deepseek_v41.py` | A specialized attention/cache contract that cannot be expressed as ordinary GQA |

Record differences before coding:

- Layer schedule and head geometry.
- Norm weight convention and epsilon, residual order and dtype, and rotary
  convention.
- Activation, routing, tied embeddings, and logits transforms.
- Quantization and draft interface.

Expose these differences in tests against a tensor reference.

### Register the architecture

`python/tokenspeed/runtime/models/registry.py` imports modules and collects
`EntryClass` exports. The class name must match the checkpoint's
`architectures`. The filename does not register it. Export a list only for
multiple architecture classes. Avoid duplicate exports across modules.

```python
# At the end of the model module, after defining the class.
EntryClass = ExampleForCausalLM
```

In `python/tokenspeed/runtime/model_loader/loader.py`, `_initialize_model`
passes `config`, `mapping`, and `quant_config` to the class. Multimodal wrappers
take extra arguments. Follow a working wrapper for models with encoders.

## 2. Parse configuration at the config boundary

Resolve checkpoint configuration before allocating weights to reject unsupported
combinations early.

Reuse a compatible Transformers config. For a local config class:

- Add it under `python/tokenspeed/runtime/configs/`.
- Export it from `python/tokenspeed/runtime/configs/__init__.py`.
- Register its `model_type` in `_CONFIG_REGISTRY` in
  `python/tokenspeed/runtime/utils/hf_transformers_utils.py`.
- Check `_resolve_registered_config` and `get_hf_text_config`, especially for
  nested multimodal text configs.

Validate head counts/dimensions, layer-type vector length, sliding windows,
expert counts, rotary scaling, positional limits, and activations. Preserve
trained numerical conventions. Do not fill unfamiliar fields from a neighboring
model.

`python/tokenspeed/runtime/configs/model_config.py` resolves geometry and
defaults. Check its attention-family tables and
`python/tokenspeed/runtime/layers/attention/registry.py` for both target and
draft. Ordinary MHA has a default path. MLA and hybrid models need family
resolution. Registering a class alone does not configure cache planning.

Plugins register separately through
`python/tokenspeed/runtime/plugins/registry.py`. Their
`model_profile(hf_config)` returns `ModelProfile` from
`python/tokenspeed/runtime/configs/model_profile.py`. It declares attention
geometry, cache family, linear-attention backend, and other model facts. Adding
this method to an in-tree `EntryClass` alone does not enable plugin profile
resolution.

### Keep model facts out of server flags

Read `python/tokenspeed/runtime/utils/server_args.py` before adding options.

- Keep architecture identity, layer schedules, routing, norm conventions,
  quantization exclusions, and draft geometry in configuration.
- Do not add model-specific flags for these facts.
- Reuse controls for parallelism, memory budgets, context limits, attention/MoE
  backends, and speculation.
- Require a runtime use case beyond trained model behavior for any new generic
  control.

## 3. Build the model from shared modules

Reuse shared bases and layers to preserve runtime contracts instead of writing
the model in plain torch.

The bases live in `python/tokenspeed/runtime/models/base/`:

- `decoder_layer.py`: `BaseDecoderLayer` handles norms, residuals, and
  `CommManager` through `resolve_attn` and `resolve_mlp`. `BaseMoEDecoderLayer`
  marks expert blocks. `CompiledDecoderLayer` offers explicit execution and
  placement specifications.
- `transformer_model.py`: `BaseTransformerModel` builds embeddings/layers, runs
  the stack/final norm, and collects auxiliary hidden states.
- `causal_lm.py`: `BaseCausalLM` builds the transformer, tied or parallel LM
  head, and logits processor. It supplies loading and target-capture hooks.

Use bases whose residual and communication contracts match. Override specialized
residual behavior while preserving layer, loader, and logits interfaces.
Validate each topology before claiming pipeline or mixed parallelism support. A
base hook alone is insufficient.

| Operation | Shared implementation |
| --- | --- |
| Projections | `python/tokenspeed/runtime/layers/linear.py`: `ReplicatedLinear`, `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear` |
| Attention | `python/tokenspeed/runtime/layers/paged_attention.py`: `PagedAttention`; backend construction is in `layers/attention/registry.py` |
| Positions | `python/tokenspeed/runtime/layers/rotary_embedding.py`: `get_rope` |
| Normalization | `python/tokenspeed/runtime/layers/layernorm.py`: `RMSNorm`, `GemmaRMSNorm`, and other supported variants |
| Gated activation | `python/tokenspeed/runtime/layers/activation.py`: `SiluAndMul` |
| Embeddings and output head | `python/tokenspeed/runtime/layers/vocab_parallel_embedding.py`: `VocabParallelEmbedding`, `ParallelLMHead` |
| Routing and experts | `python/tokenspeed/runtime/layers/moe/topk.py` and `python/tokenspeed/runtime/layers/moe/expert.py`: `TopK`, `MoELayer` |

Pass `quant_config` and full module `prefix` through projections. Take rank,
size, and group from the appropriate `Mapping` member. Attention, dense, and
expert parallelism differ. With `reduce_results=False`, row-parallel output
needs reduction by the surrounding communication flow. Check ownership in
`python/tokenspeed/runtime/distributed/mapping.py` and
`python/tokenspeed/runtime/distributed/comm_manager.py`.

### A small shared-layer skeleton

These fragments reuse `Qwen3Attention` for Qwen3-compatible GQA, QK norms,
standard pre-norm residuals, and a SiLU gated MLP. They assume compatible
config, equal attention/dense TP, and no pipeline partitioning. They illustrate
shared-layer wiring, not complete checkpoint support.

```python
from torch import nn

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.qwen3 import Qwen3Attention
from tokenspeed.runtime.utils import add_prefix


class ExampleMLP(nn.Module):
    def __init__(self, config, mapping, quant_config, prefix):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}")
        tp = mapping.dense
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=False,
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states):
        gate_up, _ = self.gate_up_proj(hidden_states)
        output, _ = self.down_proj(self.act_fn(gate_up))
        return output


class ExampleDecoderLayer(BaseDecoderLayer):
    def resolve_attn(self, prefix):
        config = self.config
        return Qwen3Attention(
            config=config,
            mapping=self.mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_id=self.layer_id,
            rope_theta=get_rope_theta(config, 1000000),
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            quant_config=self.quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

    def resolve_mlp(self, prefix):
        return ExampleMLP(
            self.config,
            self.mapping,
            self.quant_config,
            add_prefix("mlp", prefix),
        )


class ExampleModel(BaseTransformerModel):
    layer_cls = ExampleDecoderLayer


class ExampleForCausalLM(BaseCausalLM):
    model_cls = ExampleModel

    def get_stacked_params_mapping(self):
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]


EntryClass = ExampleForCausalLM
```

For different attention math, implement `forward(positions, hidden_states, ctx)`
with shared projections and attention. For ordinary GQA, declare `rotary_emb`
and `qk_norm` on `PagedAttention`. Pass projected Q/K/V and positions to it. Its
prologue owns per-head normalization, rotary application, KV conversion, and
cache writes. Repeating these steps changes results. MLA has a separate
latent/projection contract. Read its example and the
[attention prologue design](../design/attention-prologue.md).

Preserve checkpoint semantics: norm epsilon, offset weights, residual dtype,
rounding order, tied embeddings, embedding scale, final norm, and logits
transforms. Override base hooks instead of duplicating the causal-LM forward.

## 4. Load weights through parameter loaders

Reuse parameter loaders to preserve sharding and format-specific placement.

Prefer `BaseCausalLM.get_stacked_params_mapping`, `get_skip_weight_names`, and
inherited `load_weights`. Mapping tuples are
`(runtime_name, checkpoint_name, shard_id)`. Q/K/V use `"q"`, `"k"`, `"v"`;
merged gate/up uses `0`, `1`. Each parameter's `weight_loader` handles TP
slicing and placement. Ordinary parameters fall back to `default_weight_loader`
in `python/tokenspeed/runtime/model_loader/weight_utils.py`.

For custom streams, resolve the name before loading:

```python
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader

# Inside load_weights, after any fused-name or expert handling.
param = params_dict[name]
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, loaded_weight)
```

- Do not replace specialized loaders with raw `copy_` or slice packed tensors as
  unquantized weights.
- Check fused-member order, transposes, biases, scales, tied weights, and
  rank-local layers/experts.
- For separate or fused experts, use `ExpertCheckpointSchema` in
  `python/tokenspeed/runtime/layers/moe/schema.py` and
  `build_moe_checkpoint_loader` in
  `python/tokenspeed/runtime/layers/moe/loader.py`.

Consume weights lazily. `list` or whole-stream sorting can retain GPU tensors
during [InstantTensor loading](./instanttensor.md). Buffer only bounded pieces
needed for tensor pairing.

Return loaded runtime parameter names when trackable. Call `post_load_weights`
for derived state. The base coordinates partial live updates. Preserve captured
addresses and avoid repeating one-time transforms on updates. For absorbed or
packed weights, follow `causal_lm.py` and
`python/tokenspeed/runtime/models/base/weight_update.py`.

The generic loader can skip names absent from local parameters. Test required
parameters, fused shards, scales, and explicitly ignored keys. Treat unexplained
missing or unexpected tensors as failures.

## 5. Support quantized checkpoints through configuration

Infer supported weight formats from checkpoint metadata so basic launches need
no redundant precision option.

`ModelConfig._parse_quant_hf_config` and `_verify_quantization` in
`python/tokenspeed/runtime/configs/model_config.py` read `quantization_config`,
`compression_config`, or supported `hf_quant_config.json` metadata. Registered
methods in `python/tokenspeed/runtime/layers/quantization/__init__.py` identify
formats. The loader constructs `QuantizationConfig` and checks device capability
and activation dtype before model construction.

Shared linears select methods using config and module prefix. Implementations
live under `python/tokenspeed/runtime/layers/dense/` and
`python/tokenspeed/runtime/layers/quantization/`.

- Preserve exclusions and scale layouts, including expert-only quantization.
- For differing checkpoint/runtime names, check `quant_module_name_replacements`
  and config name-replacement support.
- Do not add model-specific precision flags or require redundant
  `--quantization` for format detection.
- Preserve explicit quantization controls' runtime semantics and compatibility
  checks.
- Test each format separately. BF16 does not qualify packed loading or quantized
  activations.

KV-cache precision is separate. Load non-unit checkpoint KV scales through
`load_kv_cache_scales` and shared attention scale support with a compatible
backend. Do not discard scales or infer them from weight format.

## 6. Inspect dispatch before adding a kernel

Trace default dispatch first to establish which kernels can affect performance.

For each supported GPU and precision, follow:

- Attention: `python/tokenspeed/runtime/layers/attention/registry.py`.
- MoE: `python/tokenspeed/runtime/layers/moe/utils.py`.
- Dense: the selected quantization method and its operator facade.

Record actual prefill, decode, and draft calls. Unselected kernels cannot
improve that launch.

Keep models at the shared-layer boundary. Add implementations through facades
under `tokenspeed-kernel/python/tokenspeed_kernel/ops/`. Register with
`register_kernel` in `tokenspeed-kernel/python/tokenspeed_kernel/registry.py`.
Select through `select_kernel` in
`tokenspeed-kernel/python/tokenspeed_kernel/selection.py`. Never import or call
implementation kernels from models.

Encode platform, formats/layouts, required features, and shape/behavior traits
in selection.

- Add traits to facade requests and registrations together. Test both sides of
  boundaries.
- Use required features or distinct operators for new semantic contracts when
  needed.
- An undeclared ordinary trait does not automatically reject an implementation.
- Bounds such as `m_max` need the request dimension `m`.

Inspect `explain_selection` and verbose logs, then confirm actual execution
traces. Simulated platforms test rules, not GPU execution. Diagnostic overrides
in `selection.py` bypass platform, format, and trait matching but retain
required-feature checks. They do not prove default support. Prefer supported
selection conditions over forced names.

Before timing, test dtype, rounding, quantization scales, strides, aliasing,
buffer ownership, and graph replay. Follow [Numerics](../design/numerics.md).
Avoid unbounded JIT specialization on per-request sizes. Use
`assert_no_triton_compile` from `tokenspeed-kernel/test/utils.py` to detect
repeated compilation.

## 7. Describe hybrid attention and cache groups

Declare visibility and storage separately so cache planning matches every
layer's state.

`PagedAttention.sliding_window_size` controls visibility.
`hf_sliding_window_to_window_left` converts checkpoint window tokens to
preceding tokens: 64 becomes 63 plus the current token. Startup binds `group_id`
from the cache plan. Models must not assign it.

`python/tokenspeed/runtime/layers/attention/configs/base.py` defines
`AttnConfig` and components. One softmax component declares the family and
per-layer cache labels. Hybrids add `LinearAttnConfig` from
`python/tokenspeed/runtime/layers/attention/configs/linear_attn.py`. Keep
compute labels separate from cache vocabulary. Check that the schedule covers
every layer.

Reuse `python/tokenspeed/runtime/layers/attention/kv_cache/recipes/`:

- `ordinary.py`: ordinary softmax families.
- `qwen35.py`: GDN state alongside attention.
- `kimi_k3.py`: its hybrid family.
- `spec.py` and `plan.py`: group declarations and storage fields.

Add recipes only for state contracts existing recipes cannot represent.

Include per-request state in allocation, prefix reuse, transfer, and
reclamation. Unavoidable backend-private slot state must implement
`SlotStateExporter` from `python/tokenspeed/runtime/execution/slot_state.py`.
Extend `test/runtime/test_slot_state.py` to construct and check its owner. Test
prefix hits, chunked prefill, window boundaries, speculation rollback, and
retraction. Short decode misses these transitions.

### Batching and CUDA graphs

Pass `ForwardContext` from `python/tokenspeed/runtime/execution/context.py`
through the model.

- Respect actual tokens, padded rows, idle ranks, mixed prefill/decode, and
  collective participation.
- Avoid device-to-host reads, data-dependent Python branches, and fresh
  persistent state during capture.
- Follow the [unified decode path](../design/unified_path.md): eager and graphs
  share refresh-in-place metadata.
- Size persistent buffers for maximum decode batches, including above-ladder
  batches.
- Preserve addresses and backend padded write locations. Do not add model-local
  caches or graph-only metadata.
- Test eager, captured, padded, and above-ladder batches separately.

## 8. Preserve MoE routing semantics

Use shared routing and expert layers to preserve trained behavior across
backends.

Use `TopK` and `MoELayer` for expert GEMMs and dispatch.
`python/tokenspeed/runtime/models/qwen3_5_moe.py` demonstrates
`output_format=self.experts.topk_output_format`, expert parallel ranks, and
dense/shared-expert reuse.

Match score function, correction bias, group selection, top-k count,
normalization order, routed scaling, shared-expert gate, and dtypes. Keep
expert-layer `routing_config` consistent with standalone routing. Backends may
fuse selection from logits. Test selected experts, routing weights, and outputs.

Follow backend communication contracts. Shared experts on rank-local token
shards must not reduce unrelated tokens. Test TP and EP independently, including
empty local work. For placement/rebalancing, honor
`BaseCausalLM.supports_expert_placement` and `routed_experts_weights_of_layer`.
Load all placed slots, including replicas. Enable support only after validating
loader and runtime contracts.

## 9. Integrate speculative decoding hooks

Qualify the target without speculation first to isolate draft integration
errors.

`python/tokenspeed/runtime/execution/drafter/__init__.py` selects algorithms
through `get_drafter_impl`. Read `BaseDrafter` in
`python/tokenspeed/runtime/execution/drafter/base.py`. Reuse algorithms when
only the draft model changes. New draft architectures and new algorithms are
separate tasks.

For hidden-state consumers, preserve
`BaseCausalLM.set_eagle3_layers_to_capture`, `set_dflash_layers_to_capture`, and
auxiliary outputs. Verify requested residual boundaries instead of copying
capture indices. `BaseDrafter.wire_target` binds target resources.
`prepare_target_forward` can attach per-forward capture sinks. Configure capture
before constructing the drafter.

Check:

- Vocabulary/tokenizer compatibility and embedding/head sharing via
  `get_embed_and_head` and `set_embed_and_head`.
- Draft weight mapping, mask/special tokens, and block widths.
- Target/draft position limits, including speculative padding.
- Draft-on/off correctness, acceptance, and latency. Acceptance alone proves
  neither quality nor speed.

For tree drafts, follow [tree speculation](../design/tree-speculation.md),
including rollback and position-versus-slot semantics.

## 10. Validate chat templates and gateway parsing

Validate prompt rendering and response parsing to qualify the complete API
behavior.

Pin tokenizer, template, effective template kwargs, special tokens, stop IDs,
and generation defaults. Compare rendered text and token IDs with the checkpoint
reference before debugging generation.

In `python/tokenspeed/cli/serve_smg.py`,
`_gateway_args_with_default_reasoning_parser` detects dictionary-valued
`response_template` in `tokenizer_config.json`. It selects `symphony` with `hf`
reasoning/tool parsers unless explicit options block this path. Reasoning,
tool-call, and parser arguments affect resolution. Test the exact recipe. Model
names alone do not select parsers. Without a response template, use an existing
named parser or add gateway package support.

Keep parsing in the gateway. Prefer checkpoint response templates and preserve
defaults. Do not parse tool calls in model forward or hide template overrides.
Check that gateway dependencies in `python/pyproject.toml` provide the required
support.

Constrained decoding is separate. Configs declare `constrained_decoding` as
`None`, `"requested"`, or `"always"`. `_apply_constrained_decoding_policy` in
`model_config.py` validates and applies it to target workers, skipping drafts.
See `test/runtime/test_model_constrained_decoding.py`. Test grammar behavior
through the endpoint. Parsed output alone does not prove constrained generation.

Cover:

- Streaming/non-streaming and reasoning history.
- Tool choice (`auto`, `required`, named tool) and structured output.
- Multiple turns, truncated tool calls, and finish reasons.

Add launcher/default tests in `test/cli/`, beside `test_serve_smg_unit.py`. Run
API checks with the installed gateway/runtime combination.

## 11. Build a correctness ladder

Use real trained checkpoints. Pin checkpoint revision, tokenizer/template, dependencies, inputs, precision, parallelism, seeds, and generation settings. Pass the pinned local snapshot to both runners in `test/runners.py`; they do not expose a revision argument. Keep small tensor fixtures in tests, without dummy-only production paths.

1. **Module parity:** compare projections, norms/residuals, routing, loading, cache operations, and logits transforms against tensor references. Require exact discrete results and byte-preserving operations; use numerical comparisons for floating-point results.
2. **HF reference comparison:** reuse `HFRunner` and `RTRunner`. For exact correctness, assert equal `output_strs` and generated token IDs from `output_token_logprobs_lst`. For numerical agreement, call `check_close_model_outputs` with `check_logprobs=True`: it checks ROUGE-L and absolute sampled-token logprob differences along the common generated-token prefix. Assert nonempty arrays and matching-token coverage, and report compared-token counts and the first divergence; the helper skips logprob comparison if the first token differs.
3. **Prompt/top-k comparison:** when needed, extend test-side reference collection to compare teacher-forced logprobs and reciprocal top-k membership at identical prefixes/positions. The current runners do not return prompt/top-k distributions, and the comparator does not use `prefill_tolerance`. Do not claim this coverage from generation smoke tests.
4. **Published evals and verifier:** run the model's published tasks and provider verifier when available. Match dataset version/split, sample count, template, generation budget, and scoring. Report measured and published scores, their gap, setting mismatches, and unavailable checks.
5. **Serving stability:** test long context, concurrency, mixed/padded batches, eager execution, graph replay, cache pressure, and applicable API behavior. Repeat reference and evaluation checks with draft off/on for supported pairings; acceptance rate is not a quality check.

Start from the existing `ModelCase` defaults: `decode_tolerance=5e-2` for strict absolute logprob differences and `rouge_l_tolerance=1`. ROUGE-L equality does not establish exact text/token equality; `prefill_tolerance=5e-2` currently adds no assertion. Tighten numerical tolerances after passing on the claimed hardware/dtypes. Never widen a tolerance or lower a score requirement to make a failure pass; investigate inputs, loading, numerics, or the harness.

Qualify every claimed checkpoint format independently, including quantized weights. If HF cannot load the format or fit the reference, state the gap and use a documented provider/dequantized reference where available; another format does not prove same-format parity. Report each level's failures and skips separately.

### Put tests in the existing suites

- Use `test/runtime/models/` for model/config/loader behavior. See
  `test/runtime/models/test_deepseek_v3_loader.py` and
  `test/runtime/models/test_qwen3_cache_groups.py`. Test real behavior, not
  source text or class names.
- Put shared tests in `test/runtime/layers/`, `test/runtime/moe/`, and
  `test/runtime/cache/`.
- Put kernel correctness/selection tests in `tokenspeed-kernel/test/`, including
  `test_selection.py`. Use vendor subdirectories for vendor-specific cases.
- Add generation `ModelCase` entries in `test/runtime/models/test_generation_models.py`. `ONLY_RUN` selects checkpoints; `test_others` uses FP16 and skips under `CI`. Use format-specific tests for other dtypes. This suite checks output content/nonempty generation, does not call the HF comparator, and does not consume the tolerance fields; add explicit reference comparisons.
- Register GPU tests through `test/ci_system/ci_register.py`. Add
  evaluation/performance YAML under `test/ci/eval/` and `test/ci/perf/` with
  supported runner labels and explicit thresholds.

After adding a checkpoint to `ALL_OTHER_MODELS`, run in a configured GPU
environment:

```bash
ONLY_RUN=org/model python3 -m pytest \
  test/runtime/models/test_generation_models.py::TestGenerationModels::test_others
```

Run focused CPU/module tests, checkpoint token/API checks, then distributed and
long-running evals. Test installation and launch from a clean environment to
expose missing dependencies or local import assumptions.

## 12. Measure the supported serving path

Measure default serving paths to establish performance under supported launch
conditions.

Record attention, MoE, dense, and sampling paths for every claimed
hardware/precision/topology combination. Screen short representative shape sets.
Run expensive distributed measurements only after module correctness passes.
Isolate measurements from competing CPU/GPU work and downloads.

Report operator/module latency separately from serving:

- Serving: time to first token, inter-token latency or time per output token,
  throughput, concurrency, input/output lengths, and latency distributions.
- Speculation: draft cost and acceptance.
- Resources/startup: memory, cold loading/compilation/tuning, and warm serving
  separately. See [Startup timing](./startup-timing.md).

Match baseline/candidate hardware, checkpoint format, TP/EP, cache dtype,
prompts, batching limits, prefix-cache state, graph mode, and drafts. Record
warmup, repetitions, statistic, and variability. Reuse patterns in
`test/random_benchmark/` and `test/ci/perf/`. Report absolute values, changes,
and regressions. Mark unsupported and unmeasured cases.

## 13. Document a reproducible recipe

Add a tested recipe so users can reproduce the supported launch from a clean
install.

Update [Model recipes](../recipes/models.md) in `docs/recipes/models.md`. Follow
its family heading, short description/checkpoint link, hardware/precision
subheadings, and fenced `ts serve` or `tokenspeed serve` commands. Include
required settings only. Explain non-default behavior after commands.

State formats, hardware/topology, context limits, quantization detection,
parser/draft requirements, limitations, and required released dependencies. Use
supported model IDs. Record exact validated revisions in qualification
artifacts. Template commands alone do not validate deployments.

Run installation and commands literally, replacing only documented placeholders.
Verify served model ID and API behavior through the gateway. Link separate
guides from `docs/index.md` and `docs/.vitepress/config.mts`. Keep architecture
details under `docs/design/`.

## 14. Checklist

Review every applicable item before claiming model support to keep qualification
complete.

- [ ] The checkpoint's config and architecture resolve to the intended class,
  with no duplicate registration.
- [ ] The implementation reuses shared bases and layers, with explicit
  model-specific numerical differences.
- [ ] Module prefixes, quantization exclusions, fused shards, scales, and
  rank-local weight coverage are tested.
- [ ] Attention visibility and cache groups agree with the layer schedule; state
  survives prefix reuse, transfer where supported, rollback, and retraction.
- [ ] Eager, graph, padded, mixed, and idle execution preserve the same
  semantics and communication ownership.
- [ ] MoE routing and each claimed quantized format pass their own correctness
  checks.
- [ ] The default launch reaches the intended registered kernels on every
  claimed GPU; fallback and refusal cases are tested.
- [ ] Draft hooks, position limits, and draft-on/off behavior are validated when
  speculation is supported.
- [ ] Chat templates, reasoning, tool calls, and constrained decoding work
  through the installed gateway.
- [ ] Module and HF comparisons pass per format with actual token coverage and retained or tightened harness tolerances; provider checks, published-score gaps, and stability results are reported.
- [ ] Performance results separate module and serving effects and record their
  conditions and uncertainty.
- [ ] The recipe works from the documented installation, and tests, navigation,
  and relevant design docs are updated.
