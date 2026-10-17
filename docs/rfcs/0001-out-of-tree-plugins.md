# RFC 0001: Out-of-tree models and kernels on the mainline runtime

| | |
| --- | --- |
| Status | Draft |
| Scope | `python/tokenspeed/runtime`, `tokenspeed-kernel/python/tokenspeed_kernel/plugins` |
| Touches the C++ scheduler | No |
| Related | `tokenspeed-kernel/python/tokenspeed_kernel/plugins/README.md`, `docs/design/cache-concepts.md`, `docs/design/event-loop.md` |

## Summary

Let a separately installed Python package contribute a model implementation,
its kernels, and — where the model needs them — an attention backend, a KV
cache recipe, a quantization method or a drafter, to an unmodified mainline
`tokenspeed`, and have that model scheduled, cached, disaggregated and
speculated on exactly like an in-tree one.

The scheduler and event loop see cache groups, block granularity and a
capacity model, never a model class. Mainline already has runtime plugins:
`ModelConfig.__init__` calls `tokenspeed.runtime.plugins.ensure_loaded()`,
which invokes kernel `discover_plugins()` before runtime entry points.
Models, configs, attention and linear-attention backends, cache recipes,
cache pools and drafters have registration APIs, and registered models
already resolve through `ModelProfile`.

This RFC builds on that implemented baseline. The remaining work is to:

1. Harden discovery ordering, failure handling and paired disabling without
   introducing a second loader.
2. Make behavioral registration arguments explicit, complete quantization
   and kernel-solution registration, and unify drafter resolution.
3. Extend the existing profile with forced-backend and PD compatibility
   facts, complete the other profile contracts below, then migrate in-tree
   architecture tables onto the same path.
4. Complete post-discovery validation for the remaining closed CLI names.
5. Extend the existing runtime plugin tests with an installable fixture and
   end-to-end coverage of the supported execution and transfer contracts.

## Motivation

Downstream deployers commonly have a model or a set of kernels they cannot
publish — a proprietary architecture, a vendor-restricted kernel, an
in-flight research variant — and still want the scheduler, the cache
subsystem, PD disaggregation and speculative decoding from mainline, tracked
closely. Before the current registration APIs, their options were:

- **Fork and rebase.** Every mainline refactor of the touched registries is a
  conflict; the fork drifts and stops contributing back.
- **Monkey-patch at import time** (`ModelRegistry.models[...] = ...`,
  `_BACKEND_REGISTRY[...] = ...`). Works for the two containers that happen
  to be mutable dicts; fails for `_ATTENTION_FAMILY_SPECS` (a tuple of
  frozensets) and for every `choices=` list; breaks silently on rename.
- **Re-implement the serving loop** around the kernel package. Discards the
  part they actually wanted.

The implemented registries avoid these workarounds for supported extensions;
the remaining gaps below still need a complete contract. The project's
own collaboration principle — core
features are designed and implemented by the core team — is served, not
undermined, by a supported extension surface: it lets downstream code stay
downstream instead of arriving as forks or as pressure to upstream models the
project does not want to maintain.

## Non-goals

- Making every kernel call site overridable. Model and layer code that
  imports a concrete kernel function (`from tokenspeed_kernel.ops.x.y import
  fn`) stays as it is; the registry-based override only reaches call sites
  that already go through `select_kernel(..., solution=...)`. A plugin model
  imports its own kernels however it likes.
- A stable ABI. The plugin contract is Python classes and is versioned by
  exact `tokenspeed` / `tokenspeed_kernel` pins, as the kernel plugin README
  already requires.
- Runtime hot-loading or unloading of plugins. Discovery happens once per
  process at startup.
- Any change to the C++ scheduler, the event loop, or the cache-group
  vocabulary.

## Background: what the scheduler actually depends on

Per `docs/design/event-loop.md`, `build_device_side` constructs the model
runners, attention backends and KV pools as locals and returns `DeviceSpecs`
(cache geometry, cache groups, speculation widths, capability flags) and a
`DeviceHandle`. Per `docs/design/cache-concepts.md`, a model's per-request
state is declared as cache groups through a `CacheRecipe` and consumed by
attention backends via `cache_consumer_families`; the scheduler allocates,
prefix-matches, transfers and frees those blocks without knowing which model
owns them.

So the contract between "a model" and "the scheduler" is already inverted:
the scheduler depends on the `CacheSetup` abstraction, and models depend on
`PagedAttention` / `AttentionBackend` / `CacheRecipe`. A plugin model that
expresses its state as cache groups uses the common scheduling, retraction
and prefix-cache path. PD additionally requires a supported transfer
contract; registration alone does not establish that support. This RFC
extends the existing out-of-tree integration without changing ownership.

## Current state: inventory of dispatch points

Paths are relative to `python/tokenspeed/runtime/` unless noted. This table
describes current code; the proposal below distinguishes retained behavior
from contract changes.

| Dispatch point | Where | Implemented baseline and remaining gap |
| --- | --- | --- |
| Kernel selection and discovery | `tokenspeed_kernel/registry.py`, `tokenspeed_kernel/plugins/`; runtime `plugins/__init__.py` | Open kernel registry and entry points; runtime `ensure_loaded()` already calls `discover_plugins()`. Strict failure and shared disable policy below are proposed changes. |
| HF architecture → model/config class | `models/registry.py`, `plugins/registry.py` | `register_model` and `register_config` exist; the model registry combines built-in classes and plugin registrations. |
| Attention geometry and defaults | `configs/model_config.py`, `configs/model_profile.py` | Registered models use `ModelProfile.configure_attention` and defaults; unprofiled in-tree models still use `_ATTENTION_FAMILY_SPECS`. |
| Family facts, forced backend and PD support | `layers/attention/registry.py` | Profiles already select cache families and linear attention. `_apply_backend_overrides` and `_check_pd_support` still use architecture facts for V4/V4.1 and Inkling constraints; profiles cannot declare those constraints. |
| Attention backend and kernel solution | `plugins/registry.py`, `layers/attention/registry.py`, `backends/paged/mha.py` | Attention registration and post-discovery name validation exist; the MHA name → kernel-solution map remains closed. |
| Cache family → recipe/pool | `plugins/registry.py`, `layers/attention/kv_cache/recipes/setup.py`, `kv_cache/factory.py` | `CacheModelFamily` is already `str`; recipe and pool registration and profile-based family selection exist. |
| Quantization method → config class | `layers/quantization/__init__.py`, `layers/linear.py` | Closed method table; linear dispatch still mixes class checks with `get_quant_method` calls. |
| Algorithm → drafter | `execution/drafter/__init__.py`, `plugins/registry.py` | Plugin algorithm/default and model-class-scoped registration exists, ahead of the in-tree resolver. Target capture, block geometry and draft-architecture routing still need the unified contract below. |
| Sampling backend | `sampling/registry.py` | Open registry. |
| Process startup | `plugins/__init__.py`, `configs/model_config.py` | `tokenspeed.plugins` entry points, `ensure_loaded()`, `list_plugins()` and `PLUGIN_API_VERSION = 1` exist. `test/runtime/test_runtime_plugins.py` covers loader and registry behavior. |

One process-boundary fact shapes the design: `ModelConfig` is constructed
independently in the frontend (`engine/async_llm.py`), the scheduler process
(`engine/event_loop.py`) and the encode loop (`epd/encode_loop.py`), and the
architecture → family resolution runs inside `ModelConfig.__init__`. Plugin
discovery therefore cannot live only in the GPU worker's `build_device_side`;
it has to run in every process, before the first `ModelConfig`.

## Proposal

### P1. `tokenspeed.runtime.plugins`: discovery

The existing module already exposes the following names at API version 1.
The declarations and examples below describe the proposed next contract,
not the currently implemented signatures; incompatible changes require an
API-version bump and updated exact dependency pins:


```python
ENTRY_POINT_GROUP = "tokenspeed.plugins"
DISABLE_ENV_VAR = "TOKENSPEED_DISABLE_PLUGINS"
PLUGIN_API_VERSION = 2

def ensure_loaded() -> list[PluginInfo]: ...
def list_plugins() -> list[PluginInfo]: ...
```

Today `ensure_loaded()` loads kernel plugins first, skips runtime entry
points named by `TOKENSPEED_DISABLE_PLUGINS`, and warns and continues after a
failed runtime registration. The `recording()` context undoes recorded runtime
registrations on failure. `register_attention_backend` already imports built-in
backends before inserting a plugin override. The geometry callback is also
implemented: `ModelConfig` invokes `profile.configure_attention(self)`.
These guarantees must be preserved; they are not missing APIs.

Proposed hardening of that same loader:

- Before loading either entry-point group, inspect metadata only (no
  `ep.load()` or plugin imports). Reject duplicate names within each group,
  including duplicates from the same distribution discovered on two paths.
  A name shared across the runtime and kernel groups must belong to the same
  distribution: compare PEP 503-canonicalized distribution names and versions,
  and reject ambiguous or missing distribution metadata. Also reject multiple
  discovered installations of the same distribution, even if versions differ.
  Perform this validation before applying the disable set; disabling then
  removes both halves of a validated pair without importing either one.

- `ensure_loaded()` is idempotent per process. It first imports
  every built-in registry, including attention backends and model classes,
  before any plugin registration. Seeding must be idempotent and must not
  overwrite a later plugin entry. It imports `tokenspeed_kernel`, applies
  the shared disable set below, and calls kernel discovery with explicit
  `force=False, strict=True`, then walks the
  `tokenspeed.plugins` group sorted by entry-point name and calls each
  `register()`. Built-in seeding must stay safe in CPU-only frontend
  processes: retain platform-gated/lazy backend imports, do not allocate
  device memory or compile kernels, and test startup without GPU libraries.
- The configuration call site is the top of `ModelConfig.__init__`, before
  `_resolve_attention_family`. That covers every process that builds a
  config. `build_device_side` needs no call of its own because it receives
  an already-built `ModelConfig`. CLI registry validation has a second,
  idempotent call before validating names in argument handling; it cannot
  rely on a later `ModelConfig` construction.
- A runtime plugin load or registration failure aborts startup. Discovery
  records a terminal failed state: subsequent calls raise rather than using
  partially registered entries. The kernel loader must add an explicit
  `strict: bool` parameter with the same failure behavior; existing callers
  choose `strict=False` to retain warning-and-continue behavior. The runtime
  uses `strict=True`. This is a proposed kernel API change, not behavior
  available today. A failed process must restart before serving requests.
- Every loaded plugin logs its distribution, version and the names it
  registered in each registry, at `INFO`, once, so a serving log always
  shows what out-of-tree code is active.
- Before either group loads, the runtime unions `TOKENSPEED_DISABLE_PLUGINS`
  and `TOKENSPEED_KERNEL_DISABLE_PLUGINS` and applies that set to both groups,
  calling kernel `disable_plugin(name)` before discovery. Paired entry points
  must use the same name, as in the example below; packages with different
  names must be rejected before either entry point executes. This changes
  runtime integration of the existing kernel-only switch; standalone kernel
  hosts retain their own disable policy. The runtime must own first discovery
  and reject a startup where an excluded entry point has already loaded.

Plugins are activated by installation. The existence of a `--plugins`
allowlist as an alternative is discussed under Open questions.

### P2. Registration surface

The model/config, attention/linear-attention, cache and drafter registration
functions already live in `tokenspeed.runtime.plugins.registry`. The target
API below retains that surface, makes behavioral choices explicit and adds
quantization registration. In-tree results must stay unchanged during the
migration. Existing defaulted `override` arguments become required under the
new API version; the drafter migration must also preserve its existing
checkpoint-selection argument.

```python
def register_model(cls: type[nn.Module], *, architectures: tuple[str, ...] = (), override: bool) -> None
def register_config(cls: type[PretrainedConfig], *, model_type: str | None = None, architectures: tuple[str, ...] = (), override: bool) -> None
def register_attention_backend(name: str, archs: set[AttentionArch], cls: type[AttentionBackend], *, override: bool) -> None
def register_linear_attention_backend(name: str, factory: Callable[..., AttentionBackend], *, override: bool) -> None
def register_cache_recipe(family: str, recipe: Callable[..., CacheRecipe], *, override: bool) -> None
def register_cache_pool(family: str, factory: Callable[..., CachePool], *, override: bool) -> None
def register_quantization_method(name: str, cls: type[QuantizationConfig], *, override: bool) -> None
def register_drafter(algorithm: str, drafter_cls: type[BaseDrafter], *, model_cls: type[nn.Module] | None, defaults_to_base_checkpoint: bool, override: bool) -> None
```

`override` is a required keyword argument. With `override=False`, a name
collision with any existing entry raises. Replacing an in-tree model, recipe
or method is legitimate — that is
how a downstream package ships a fixed or specialized variant — but it must be
a visible decision in the plugin's source, and it is logged.

Omitting `architectures` means the model class name alone (or no architecture
aliases for a config); omitting `model_type` uses the config class's declared
model type. A config with neither a model type nor architecture aliases is
invalid. `model_cls` must be explicit: `None` selects the algorithm
catch-all, while a class selects only that class and its subclasses.
`defaults_to_base_checkpoint` also becomes explicit. Today it is recorded
for introspection and callers still pass `--draft-model-path-use-base`;
the migration must preserve explicit checkpoint choices and define its
post-discovery use before removing that launch requirement.

Per registry:

**Models.** `import_model_classes()` keeps scanning `tokenspeed.runtime.models`
for `EntryClass`; its result seeds the registry.
`register_model(cls, override=False)` keys by `cls.__name__` unless `architectures` is given (a plugin may need to claim
an HF architecture string that differs from its class name).

**Configs.** A released checkpoint may carry no `model_type` (LongCat
Flash-Lite ships only `architectures`), and `get_config` must construct the
plugin's config class before any other resolution can happen.
`register_config` keys the class by `model_type` and by each architecture
string; `hf_transformers_utils.get_config` consults the registry before the
HF auto classes. The next contract scans every architecture alias in
checkpoint order, together with an explicitly supplied `model_type`, before
falling back. All matching registrations must identify the same config class;
conflicting classes are an error, not a first-match choice. Unknown aliases
are skipped, a checkpoint without `model_type` resolves through aliases alone,
and HF fallback occurs only when no registration matches. A registered class
wins over HF's class subject to the registration's explicit override policy;
no implicit `"llama"` type is invented during plugin lookup. Strict
released-schema validation belongs in the config class itself, where constructor defaults would otherwise hide an omitted
field.

**Attention families.** No `AttentionFamilySpec` registry: a registered
model resolves through its `ModelProfile` (P3) from the start —
`profile.configure_attention` writes the attention geometry onto the
`ModelConfig`, so a plugin model with MLA or DSA attention is never built as
MHA by fallback. The in-tree architecture tables stay as the seed for
in-tree models until the P3 migration deletes them.

**Linear-attention backends.** `_create_hybrid_linear_attn_backend` chose
between the KDA and GDN backends by architecture name. The implemented registry is seeded with `"kda"` and `"gdn"`;
`register_linear_attention_backend` adds a name, and
`ModelProfile.linear_attention` selects it. This is how Flash-Lite runs
FGBKDA (featurewise beta) as a subclass of the in-tree KDA backend with only
the recurrence seams overridden.

**Attention backends.** The existing public wrapper already checks collisions
and seeds built-ins. Its proposed signature makes the policy explicit and
continues to preserve every supplied argument.
`_KERNEL_SOLUTION_BY_BACKEND` in `backends/paged/mha.py` becomes a
`register_mha_kernel_solution(backend_name, solution, *, override: bool)`
on the same module,
so a plugin can add a backend name that routes MHA leaves to its own
`solution` string.

**Cache recipes and pools.** `CacheModelFamily` is already `str`, and recipe
and pool registration is implemented alongside the built-in family dispatch. A plugin with a non-standard KV layout
subclasses `CacheRecipe` (the seams are `layer_types`, `group_ids`,
`fields_for_layer`, `groups`, `packing`, `workspace_bytes`, `pool_options`)
and registers the recipe and the pool under a new family name. A plugin with
a standard layout registers nothing here and names an existing family.
Recipe facts already describe paged state during verify. Remaining family
constraints, including PD compatibility below, must likewise be declared so
a new family never has to edit a second architecture table.

**Quantization.** `QUANTIZATION_METHODS` becomes a registry. The
`isinstance` chain in `LinearBase.__init__` is replaced by one call,
`self.quant_method = quant_config.get_quant_method(self, prefix)`, with each
in-tree config class implementing the branch that today lives in
`linear.py`. This is a correctness fix independent of plugins: today a new
`QuantizationConfig` subclass that is not one of the listed classes falls
through the chain with no `quant_method` set.

**Drafters.** Plugin entries already precede `DRAFTER_MAPPING` and the
in-tree `isinstance` special cases, with the most recently registered class
match winning. The proposed unified registry uses `(algorithm, model_cls)`; resolution picks the
most specific matching class, falling back to `(algorithm, None)`.
Resolve using the registered draft model class before model instantiation;
ambiguous incomparable class matches are errors.
The `("DFLASH", "DSPARK")` check in `configure_draft_target` becomes a class
attribute on the drafter, `requires_target_capture: ClassVar[bool]`, which is
what the check is actually asking. The draft-architecture rewrite in
`hf_transformers_utils.py` reads `ModelProfile.draft_architecture` (P3) when
present. Suffixing remains only for unmigrated in-tree models.

Each registered drafter also declares required, non-defaulted class traits:
`requires_target_capture: ClassVar[bool]`,
`block_decode: ClassVar[bool]`,
`writes_target_cache_locations: ClassVar[bool]`,
`supports_pd_layerwise_finalization: ClassVar[bool]`, and
`draft_query_count: ClassVar[Callable[[int], int]]` (verify width to draft
query count). These traits are resolved before attention/cache construction
and stored on the resolved target/draft configuration, passed explicitly
to attention-config construction and the executor. They replace
`is_block_drafter()` and algorithm-name geometry checks at every
consumer, including the executor. Nonpositive query counts and unsupported
trait combinations fail setup. In-tree DFLASH uses the verify width; DSPARK
uses width minus one. Storage remains declared through cache recipes and
LCM groups, owned by the cache subsystem and scheduler; these traits do not
create drafter-private per-request state or a second execution path.

PP context production is also an explicit drafter contract. Every registered
class supplies `create_context_producer: ClassVar[ContextProducerFactory]`,
including an explicit factory returning `None` when no stage-local target
capture/context writes are needed. Extract the structural producer interface
used by `DSparkContextProducer` into the common contract; that class remains
a reference implementation, not a required plugin base class:

```python
from tokenspeed.runtime.distributed.mapping import Mapping as ParallelMapping


class DraftContextProducer(Protocol):
    supports_pd_layerwise_finalization: bool  # required, no inherited fallback

    def set_cache_pool(self, pool: CachePool | None) -> None: ...
    def begin_stage(self, hidden: torch.Tensor, inbound: torch.Tensor | None) -> torch.Tensor: ...
    def add_capture(self, projected: torch.Tensor, capture_idx: int, hidden: torch.Tensor) -> None: ...
    def write_context(self, projected: torch.Tensor, positions: torch.Tensor, cache_locs: torch.Tensor) -> None: ...


class ContextProducerFactory(Protocol):
    def __call__(
        self,
        *,
        target_model: nn.Module,
        draft_model: nn.Module,
        server_args: ServerArgs,
        target_attn_config: AttnConfig,
        draft_attn_config: AttnConfig | None,
        pp_mapping: ParallelMapping,
        draft_pool: CachePool | None,
    ) -> DraftContextProducer | None: ...
```

For `pp_size > 1`, the common executor invokes this factory once on every PP
stage after `configure_draft_target` and cache binding, before graph capture or any
forward, even where no `BaseDrafter` instance is constructed. The final
stage gets its draft-pool view; other stages get explicit `None`. Returning
`None` when the configured target-capture contract requires stage-local
production is a startup error. For `pp_size == 1`, the factory is not invoked:
the drafter retains its existing context production, without a second writer.
The existing target-forward protocol calls `begin_stage` for the
chunk, sends owned taps to `add_capture`, carries the partial accumulator in
PP state, and calls `write_context` only on the owning final stage. Models
implement the projection/write protocol consumed by their chosen producer,
as `DSparkContextModel` does today. Accumulators are per chunk; the producer
keeps no cross-forward mutable accumulator. Bind it through the common
forward-context slot, and rebind its pool on arena replacement. With a
producer present, the drafter must not repeat its context writes; proposal
writes still occur in their existing order before final readiness.

`supports_pd_layerwise_finalization` is required on both registered drafters
and their producers. `True` promises that the existing proposal/producer
completion path enqueues **all** context and proposal cache writes, with
side-stream dependencies joined before
`ModelExecutor._record_draft_final_cache_step` publishes readiness. Use that
existing finalization path, not a new event loop or an early target-tap
completion signal. Both `DeviceSpecs.supports_pd_layerwise_finalization` and
`ModelExecutor.register_draft_final_step_counter` check all active writers,
not just a producer when a drafter also writes. An
explicit `False` rejects speculative layerwise PD at startup through the
existing gate; ordinary PD support alone does not imply this guarantee.
Where a stage has no producer, only the drafter's declaration applies.
Missing declarations on plugin implementations are errors rather than the
legacy `getattr(..., False)` fallback. Fixtures must exercise every PP stage,
producer/drafter single-writer behavior, delayed side-stream writes and the
final readiness counter, including rejection of missing/false capabilities.

**CLI.** `--attention-backend`, `--drafter-attention-backend`,
`--speculative-algorithm` already accept plugin names and validate after
discovery. `--quantization` must also drop its closed choices and validate
against the corresponding registry after `ensure_loaded()`, producing the
same "unknown X, available: [...]" error a user gets today. The help text
lists the in-tree names and says plugins may add more. `--moe-backend`
retains its existing enum and validation; extending MoE dispatch is outside
this RFC.

There is deliberately no plugin CLI extension point. A plugin's deployment
knobs (e.g. where Flash-Lite's over-embedding tables live) are explicit
fields of its config class, set through the existing `--hf-overrides` JSON.
That keeps behavioral choices on the model's own schema — validated by the
config class, no defaults hiding a selection — instead of growing a second,
plugin-owned argument namespace.

### P3. `ModelProfile`: the model declares its own family facts

Registered models already use `configs/model_profile.py::ModelProfile`.
Its current fields are `configure_attention`, `cache_family`,
`linear_attention`, `default_attention_backend`, `default_prefix_granularity`,
`request_token_history`, `tokenizer_kwargs` and `attention_instances_per_layer`
(the last currently defaults to 1). Tokenizer kwargs are already passed to
frontend, scheduler and worker loads; request token history is implemented.

The schema below is the proposed next version: it retains those contracts,
replaces the defaulted multiplicity with a late-resolved cache-layer layout,
and adds config generation, backend composition, draft routing, forced-backend
and PD compatibility declarations. The
in-tree migration later removes the remaining architecture-name tables,
so all model-family knowledge lives on the model class:

```python
PDRole = Literal["target", "speculative_target", "draft"]
ModelSide = Literal["target", "draft"]


@dataclass(frozen=True, kw_only=True)
class CheckpointMetadata:
    weight_names: frozenset[str] | None  # index/header names; no weight tensors


@dataclass(frozen=True, kw_only=True)
class CacheLayerLayout:
    ids_by_hidden_layer: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, kw_only=True)
class AttentionConfigInputs:
    server_args: ServerArgs             # launch snapshot, read-only to factories
    model_config: ModelConfig          # finalized geometry and cache-layer layout
    side: ModelSide
    backend_name: str | None            # outer backend, possibly hybrid_linear_attn
    full_attention_backend_choice: str | None  # side-local leaf choice; None means auto


@dataclass(frozen=True, kw_only=True)
class BackendCompositionInputs:
    config_inputs: AttentionConfigInputs
    config: AttnConfig                 # after full-leaf/DCP resolution
    pool: CachePool                    # this side's local view of the shared arena
    backend: AttentionBackend          # selected leaf/router, optionally hybrid-wrapped
    full_attention_backend_name: str | None


@dataclass(frozen=True, kw_only=True)
class ModelProfile:
    configure_attention: Callable[[ModelConfig], None]  # writes arch + geometry
    cache_layer_layout: Callable[[ModelConfig, CheckpointMetadata, ModelSide], CacheLayerLayout]
    create_attention_config: Callable[[AttentionConfigInputs], AttnConfig | None]
    compose_attention_backend: Callable[[BackendCompositionInputs], AttentionBackend]
    draft_architecture: Mapping[str, str]    # algorithm -> registered draft architecture
    is_draft_of: frozenset[str]              # accepted target architecture names
    cache_family: str                        # a registered recipe/pool family
    linear_attention: str | None             # a registered linear-attn backend
    default_attention_backend: str | None    # used only if the user passed none
    forced_attention_backend: str | None    # when this model is the target
    forced_drafter_attention_backend: str | None  # when this model is the draft
    pd_roles: frozenset[PDRole]              # supported PD roles; empty forbids PD
    default_prefix_granularity: int | None
    request_token_history: bool              # model reads committed tokens
    tokenizer_kwargs: Mapping[str, object]
    is_generation: bool
    is_multimodal: bool
    is_multimodal_gen: bool
    is_image_gen: bool
    is_audio_model: bool
    encoder_roles: frozenset[Literal["encode", "item_dp"]]
    encoder_model_facts: Callable[[nn.Module, torch.device], EncoderModelFacts] | None
    multimodal_encoder_dtype: Callable[[nn.Module], str | None] | None

class FooForCausalLM(nn.Module):
    @classmethod
    def model_profile(cls, hf_config) -> ModelProfile:
        return ModelProfile(
            configure_attention=configure_mla_attention,
            cache_layer_layout=ordinary_cache_layer_layout,
            create_attention_config=generate_mla_config,
            compose_attention_backend=identity_backend_composition,
            draft_architecture={},
            is_draft_of=frozenset(),
            cache_family="mla",
            linear_attention=None,
            default_attention_backend="flashmla",
            forced_attention_backend=None,
            forced_drafter_attention_backend=None,
            pd_roles=frozenset({"target", "speculative_target"}),
            default_prefix_granularity=64,
            request_token_history=False,
            tokenizer_kwargs={},
            is_generation=True,
            is_multimodal=False,
            is_multimodal_gen=False,
            is_image_gen=False,
            is_audio_model=False,
            encoder_roles=frozenset(),
            encoder_model_facts=None,
            multimodal_encoder_dtype=None,
        )
```

This target-only MLA example uses explicit helpers (no behavioral defaults).
A separately registered NextN/draft model supplies its own draft layout and
compatibility declarations; it does not reuse the target-depth helper:

```python
def ordinary_cache_layer_layout(model_config, checkpoint_metadata, side):
    if side != "target":
        raise ValueError("This example layout supports only the target side")
    # One attention instance per finalized target hidden layer.
    return CacheLayerLayout(
        ids_by_hidden_layer=tuple((i,) for i in range(model_config.num_hidden_layers))
    )


def generate_mla_config(inputs: AttentionConfigInputs) -> AttnConfig:
    return MLAConfig.generate(
        inputs.server_args, inputs.model_config, is_draft=inputs.side == "draft"
    )


def identity_backend_composition(inputs: BackendCompositionInputs) -> AttentionBackend:
    return inputs.backend
```

`model_profile(hf_config)` is a classmethod, not a class attribute: a family
may pick its attention variant, or whether it reads token history, from the
checkpoint config (Flash-Lite selects MLA vs its DSA variant and gates the
over-embedding history this way). Attention configuration is a callable
rather than an `attention_arch` enum value plus a spec table, because
"which arch" was never separable from "which head geometry and scaling":
one function owns both, and a plugin can wrap an in-tree
`configure_*_attention` and adjust only what differs.

Resolution becomes: `hf_config.architectures` → `ModelRegistry` → the class
→ `cls.model_profile(hf_config)`, resolved once in `ModelConfig`.
`_resolve_attention_family`, `_resolve_attn_side`,
`_apply_backend_overrides`, `_resolve_cache_family` and `get_tokenizer` read
the profile when one exists. In-tree models keep their tables as the interim
seed; the tables are deleted once every in-tree entry class carries a
profile. A registered (plugin) class without a profile is an error at
registration, not a fallback to MHA.

Profile field contracts:

- The five modality/task booleans replace the corresponding architecture-list
  lookups for registered models. Resolve the profile at model-config time,
  before loader construction, and populate the existing `ModelConfig`
  fields, not a second set of runtime gates. `is_multimodal_active` remains
  derived from `is_multimodal` and the explicit language-model-only choice.
  Loader VLM kwargs, multimodal request contexts and encoder/prefill paths
  continue to consume those existing fields. The wrapper owns the profile
  and forwards the resolved facts to its text model where needed.
- `encoder_roles` is an explicit capability set: empty forbids encode-only
  and item-DP launches; `"encode"` permits the existing encoder-only role,
  and `"item_dp"` permits `mm_encoder_tp_mode="data"`. Validate these at
  model-config time before the loader, alongside existing role checks; PD
  prefill/decode compatibility still uses the separate common PD gate. Keep
  existing exclusions (encode versus language-model-only, active encoder
  required for item-DP, and no audio encode-only support). A profile cannot
  bypass them by claiming a role. Nonempty encoder roles require an
  `encoder_model_facts` callback. After model construction, the device builder
  calls it only when the existing EPD admission path needs
  `EncoderModelFacts`, passing the built model and execution device; it must
  return the same device/hidden/deepstack/dtype facts that interface expects.
  A registered multimodal model also supplies `multimodal_encoder_dtype`,
  which returns the loaded encoder dtype in the same string form as
  `infer_multimodal_encoder_dtype`; the model runner uses it instead of
  probing conventional attribute names. Text-only profiles set it to `None`.
  An `"encode"` model must honor the existing `hf_config.encoder_only` flag
  by omitting the language model during encode-only construction. These
  callbacks replace attribute-name probing for plugin encoders, not the encoder
  execution or transport path. A plugin must implement the existing input
  processing and model/encoder interfaces for its declared modalities;
  declaration alone does not add support for a new modality. Add fixture
  coverage for an unlisted vision/text architecture, VLM loader kwargs,
  multimodal request handling, encoder facts, and unsupported role rejection.

- Both forced-backend fields are required, with explicit `None` meaning no
  constraint for that role. `_apply_backend_overrides` applies the target
  profile's `forced_attention_backend` to `attention_backend` and the draft
  profile's `forced_drafter_attention_backend` to `drafter_attention_backend`,
  after user/default selection but before `_create_attn_config` and cache
  construction. Validate the forced names through the same backend registry
  and log any overridden user choice. V4 targets require `deepseek_v4`,
  V4.1 targets require `deepseek_v41`, and ordinary V4/V4.1 draft profiles
  require `deepseek_v4`, preserving today's role-specific behavior. With
  linear attention, the forced name selects the full-attention leaf; the
  existing `hybrid_linear_attn` wrapper still composes it with the linear
  backend. The original user request remains available for diagnostics,
  but must not overwrite the resolved forced leaf.
- `pd_roles` declares compatibility of this profile's selected cache layout.
  The existing common `_check_pd_support` gate consumes both resolved profiles
  in prefill/decode disaggregation, before attention/cache allocation. The
  target must permit `"target"`; if a draft model exists **or** a speculative
  algorithm is selected, it must also permit `"speculative_target"`. A draft
  profile must permit `"draft"`. Missing permission fails startup, and an
  empty set forbids PD. Non-PD execution is unaffected. V4/V4.1 ordinary
  cache profiles permit target and speculative-target roles but not draft;
  Inkling permits target only, preserving both existing rejection cases.
  DSpark profiles are separate: do not infer ordinary V4 draft restrictions
  from their architecture names (the current resolver explicitly excludes
  them). Profile declarations cannot bypass recipe/layout compatibility or
  transfer checks, and recipe constraints must also be satisfied. Unprofiled
  in-tree models retain their existing gates until migration. Today profiled
  models clear the architecture booleans those gates inspect, so this closes
  an existing gap as well as making table removal safe.

- `tokenizer_kwargs` exists because a released tokenizer may need
  construction arguments (`fix_mistral_regex=True` for LongCat's Bloom-style
  tokenizer). The existing `ModelConfig.tokenizer_kwargs` property reads
  the resolved profile mapping; retain that single source. Every tokenizer load, including
  frontend `AsyncLLM`, scheduler-side `RequestHandler` and worker weight
  loading, receives these values explicitly; none reconstructs a config-dependent profile from only
  an architecture list. Explicit caller kwargs win on conflicts; profile
  values do not override explicit tokenizer mode or trust settings.
- A profile field may gate a generic runtime capability rather than a
  resolver: `request_token_history` makes the executor keep each slot's
  committed tokens (seeded on prefix resume, graph-stable views), which the
  model receives as a forward argument. The capability lives in the runtime,
  once; the profile only switches it on. This is the pattern for future
  model needs that are state, not dispatch.
- `cache_layer_layout` is a required callable, not an eager count in
  `model_profile(hf_config)`. Invoke it once at the end of model-config
  preparation, after role-specific text-config normalization and checkpoint
  metadata loading, but before attention-config generation or cache setup.
  Its side is target/draft, not a PD role. The common checkpoint loader
  supplies tensor names from the checkpoint index (or headers); it does not
  interpret model-specific stage names. A DSpark resolver counts its DSpark stage
  names there, after metadata is available, instead of capturing the base
  checkpoint's hidden-layer count or depending on a later architecture
  branch to overwrite `num_attention_layers`. A resolver requiring unavailable
  metadata must raise; it must not guess from the target depth. Existing
  role-specific config normalization must likewise finish before this call.
- The returned `ids_by_hidden_layer` maps each finalized execution/hidden
  layer of that model side to its attention cache-layer IDs. IDs flatten to
  `0..N-1` in execution order, without gaps or duplicates; empty per-layer
  tuples and nonuniform multiplicity are allowed. Store this layout once on
  `ModelConfig` and derive `num_attention_layers = N` from it. There is no
  second independent count. Draft execution depth is its finalized draft
  depth, not target hidden layers or capture taps. Cache recipes remain the
  authority on target sharing versus independent draft storage: a shared
  draft contributes no extra independent cache layers.
- The resolved layout must reach model construction, not just cache setup.
  Extend the plugin/migrated model constructor with required
  `cache_layout: CacheLayerLayout` and `cache_layer_window: tuple[int, int]`
  keyword inputs. `_initialize_model` supplies the finalized object from
  `ModelConfig` and the already-resolved local cache window alongside config,
  mapping and quantization. Compute that logical window from the layout and
  PP execution partition before constructing models; it needs no arena,
  memory profiling or loaded weights. Cache construction later consumes the
  same window. The value is the layout, never the profile's callable. Model wrappers forward both unchanged; each attention module
  takes its model-side cache-layer ID from the supplied hidden-layer/branch entry
  and validates that it belongs to the supplied window. Do not rederive IDs
  with multipliers, mutate `hf_config`, or rebuild stage windows in the model.
  A draft receives its own side-local resolved layout. The recipe later
  binds those logical IDs to independent draft fields or target-owned fields;
  shared drafts allocate no independent storage. Constructors do not require
  the physical cache binding before the existing cache-construction phase. This constructor change is part of the version-2 contract;
  migrate each in-tree constructor explicitly rather than silently dropping
  the keywords. Extend PP tests to compare IDs actually assigned to model
  attention modules against the cache ownership plan.
- PP remains owned by `pp_stage_windows`: it partitions target hidden-layer
  IDs, then the common model/cache boundary concatenates the corresponding
  layout entries to obtain each stage's cache-ID window. It validates complete,
  disjoint coverage and passes those windows to `pipeline_cache_ownership`;
  draft storage is added on the final stage as today. Non-PP uses the complete
  layout. Field placement and PD use the resulting cache-field IDs, never
  redo this mapping. For multiplicities `(2, 1, 3)`, execution windows `[0, 2)`
  and `[2, 3)` therefore own cache windows `[0, 3)` and `[3, 6)` respectively.
- `create_attention_config` is required and runs at the existing
  `_create_attn_config` construction point, after geometry, cache layout,
  PD validation and user/default/forced backend selection. Inputs carry a
  read-only launch snapshot with `backend_name` written to the selected
  side's matching launch field. For a hybrid this is the outer
  `hybrid_linear_attn` sentinel; `full_attention_backend_choice` separately
  carries that side's user/default choice after applying its forced-leaf
  constraint (or `None` for automatic leaf resolution). Capture the target
  choice before writing the sentinel; capture the draft choice from its own
  draft selection, never from the target field. The factory initializes its
  softmax component with this leaf choice, not the sentinel. The common
  full-leaf resolver consumes this same side-local choice on both sides;
  later DCP validation may not undo a forced leaf. The factory returns the complete `AttnConfig`,
  including its typed softmax and optional linear/other components; the
  generic builder validates declared components rather than appending a
  duplicate linear component. A declared `linear_attention` with no matching
  linear component remains a startup error. `profile.cache_family` alone
  selects the recipe/pool; the factory does not choose another family. Cache
  setup validates the returned components and layer namespace against that
  recipe and rejects incompatibility rather than falling back to a different
  family. V4.1 selects a factory using
  `DeepseekV41Config.generate` so compression mappings and row geometry stay
  available to recipes and backends. V4 factories set sliding-window metadata
  here, replacing the later architecture-specific write.
  This does **not** move full-leaf resolution or DCP rewriting before config
  generation: those common steps consume the returned config afterwards,
  enforce their existing compatibility checks and reject any conflict with a
  forced leaf. No kernel or pool is constructed by this factory. A target
  must return a config; `None` is permitted only for a draft with no standalone
  attention backend, explicitly chosen by its factory (as for the existing
  same-checkpoint DSpark path). Its recipe still declares its target-owned
  fields; no architecture test is needed to skip draft config generation.
- `compose_attention_backend` is required, with explicit identity for an
  ordinary model. The common builder first creates the side's cache-pool
  view, then resolves the leaf/router from the config and final full-attention
  backend name, then builds any declared hybrid linear wrapper using that
  pool view. It then calls this
  hook once before exposing/binding the final root. Both target and standalone
  draft builders use this order; a config-less draft has no backend to compose.
  Qwen4-Exp selects a factory equivalent to `_compose_qwen4_exp_backend`,
  attaching PLE/QSA consumers only for groups in that pool view and retaining
  the selected inner full-attention backend. Inkling selects its convolution
  wrapper at this same point, after any hybrid composition. A profile needing
  several wrappers composes them in one explicitly ordered factory; generic
  construction does not independently add them again. Arena rebuilds rebind
  the existing root to the replacement pool and do not wrap it a second time.
  Cache recipes, the shared arena and scheduler continue to own persistent
  request-state allocation and transfer. Backend-owned fixed workspace is
  allowed only when its size and lifecycle match the recipe's planned
  workspace; the existing Inkling convolution working pool follows this
  accounting and must keep the workspace-size check. This is not permission
  to introduce independently allocated persistent cache state. The returned root must forward the common metadata, cache-binding,
  verify and draft lifecycle hooks described by `unified_path.md`.
- `draft_architecture` maps each supported algorithm to a registered draft
  architecture name. An explicit draft checkpoint selects its registered
  class first; otherwise the target profile mapping selects it. The selected
  draft profile must list the resolved target architecture in `is_draft_of`,
  and the drafter registry must support that algorithm/class pair. Missing
  mappings, unknown classes and incompatible target/draft pairs are startup
  errors for profiled models, never suffix fallbacks. Empty mappings/sets
  explicitly mean no implicit draft/no supported target respectively. These
  fields are required from phase 1, with empty values until a model supports
  the phase-2 drafter contract; their types do not change between phases.

This is the actual dependency inversion in the RFC: after it, there is one
resolution path shared by in-tree and out-of-tree models, which is the
project's stated preference ("one path; parameters, not branches"). The
remaining `_AttnSideProfile` booleans find homes during the in-tree
migration.

### P4. Kernel side

The runtime integration requires strict kernel discovery as specified in P1
and shared disable handling before discovery. The future kernel contract
also makes `priority` a required keyword of `register_kernel`, removing its
current `Priority.PERFORMANT + 2` default. Migrate existing in-tree calls to
explicit values equal to their current effective priorities; this RFC does
not itself change that API. Plugin examples must state their priorities.
A replacement uses a value strictly above the matching candidate it intends
to replace, in the `Priority.PLUGIN` band. Reject equal-priority competing
plugin registrations for overlapping capability/signature/trait domains,
rather than choosing by discovery order. Disjoint vendor candidates may use
the same priority, as below. Capability and trait filtering still applies;
priority never makes an unsupported kernel eligible. Other kernel selection
mechanisms remain unchanged. Three clarifications become documentation:

- The runtime calls `discover_plugins()` (via `ensure_loaded()`), which the
  kernel plugin README has always required of the host.
- A plugin that wants to override an in-tree kernel under an in-tree model
  registers in the `Priority.PLUGIN` band, and this reaches only call sites
  that select through the registry: MHA/MLA paged attention leaves
  (`mha_plan`/`mha_prefill`/`mha_decode_with_kvcache` with `solution=`),
  `moe_plan`/`moe_apply`, and `tokenspeed_kernel.mm`. Direct imports of
  concrete kernel functions elsewhere are not overridable, by design.
- A plugin model that only needs its *own* kernels does not need the
  registry at all; it imports them like in-tree models import theirs. The
  registry is for substitution, not for delivery.

### Plugin author's view

The whole plugin is one ordinary Python distribution. `models/` and the
neutral layer of `kernels/` are required; every other subpackage exists only
when the model needs it. The example ships kernels for two vendors, which is
the case that shapes the `kernels/` layout.

```
my-tokenspeed-plugin/
├── pyproject.toml
│   [project.entry-points."tokenspeed_kernel.plugins"]  my_plugin = "my_plugin.kernels:register"
│   [project.entry-points."tokenspeed.plugins"]         my_plugin = "my_plugin:register"
│   dependencies = ["tokenspeed==X.Y.Z", "tokenspeed_kernel==A.B.C"]   # exact pins
│   [project.optional-dependencies]
│   cuda   = [...]                        # build/runtime deps only NVIDIA needs
│   ascend = ["torch_npu==...", ...]      # build/runtime deps only Ascend needs
└── my_plugin/
    ├── __init__.py   register(): register_model(...) and any other runtime register_* calls
    ├── models/       FooForCausalLM (+ FooForCausalLMNextN) with a ModelProfile;
    │                 imports only my_plugin.kernels.<op> facades, never a vendor leaf
    ├── kernels/
    │   ├── __init__.py   register(): platform-gated — if current_platform().is_nvidia,
    │   │                 import _cuda and register with vendors={"nvidia"}; if .is_npu,
    │   │                 import _ascend and register with vendors={"ascend"}
    │   ├── foo.py        vendor-neutral facade, the only thing models/ imports:
    │   │                 select_kernel("my_plugin", "foo", signature, traits=...) then call it
    │   ├── _cuda/        plain functions, no registry import; CuTe DSL or Triton via
    │   │                 tokenspeed_kernel._triton
    │   └── _ascend/      plain functions, no registry import; Triton-Ascend via
    │                     tokenspeed_kernel._triton, or torch_npu
    ├── attention/    optional: AttentionBackend subclass; usually absent — reuse an
    │                 in-tree backend (flashmla, mla, dsa, ...) by naming it in the profile
    ├── cache/        optional: CacheRecipe + CachePool subclasses, only for a
    │                 non-standard KV layout; otherwise name an existing cache_family
    ├── quant/        optional: QuantizationConfig subclass for a private weight format
    ├── drafter/      optional: BaseDrafter subclass for a private speculative algorithm
    └── test/
        ├── test_foo_reference.py   numeric reference shared by every vendor
        ├── nvidia/
        └── ascend/
```

```toml
[project]
name = "my-tokenspeed-plugin"
dependencies = ["tokenspeed==X.Y.Z", "tokenspeed_kernel==A.B.C"]

[project.optional-dependencies]
cuda = [...]
ascend = ["torch_npu==..."]

[project.entry-points."tokenspeed_kernel.plugins"]
my_plugin = "my_plugin.kernels:register"

[project.entry-points."tokenspeed.plugins"]
my_plugin = "my_plugin:register"
```

```python
# my_plugin/__init__.py
from tokenspeed.runtime.plugins import PLUGIN_API_VERSION
from tokenspeed.runtime.plugins.registry import register_model

def register() -> None:
    if PLUGIN_API_VERSION != 2:
        raise RuntimeError(f"my_plugin targets plugin API 2, host has {PLUGIN_API_VERSION}")
    from my_plugin.models import FooForCausalLM, FooForCausalLMNextN
    register_model(FooForCausalLM, override=False)
    register_model(FooForCausalLMNextN, override=False)
```

```python
# my_plugin/kernels/__init__.py
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel

from my_plugin.kernels.foo import FOO_SIGNATURES

# Options the Ascend leaf does not implement; requests carrying them find no
# candidate at selection instead of having the argument ignored.
_ASCEND_OPTIONS = {"sliding_window": frozenset({False}), "return_lse": frozenset({False})}

def register() -> None:
    platform = current_platform()
    if platform.is_nvidia:
        # Imported only here: the CUDA leaf does not import on an Ascend host.
        from my_plugin.kernels._cuda import foo as cuda_foo

        register_kernel(
            "my_plugin", "foo", name="cute_my_plugin_foo", solution="cute",
            priority=Priority.PLUGIN,
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
            ),
            signatures=FOO_SIGNATURES,
        )(cuda_foo)
    if platform.is_npu:
        from my_plugin.kernels._ascend import foo as ascend_foo

        register_kernel(
            "my_plugin", "foo", name="torch_npu_my_plugin_foo", solution="torch_npu",
            priority=Priority.PLUGIN,
            capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
            signatures=FOO_SIGNATURES, traits=_ASCEND_OPTIONS,
        )(ascend_foo)
```

#### Multi-vendor kernels

The layout above is the in-tree pattern, not a plugin-specific one.
`tokenspeed_kernel_npu` and `tokenspeed_kernel_amd` are leaves of plain
functions with no registry dependency (`AGENTS.md` forbids the AMD package
from depending on `tokenspeed-kernel`); the vendor-neutral `tokenspeed_kernel`
owns every registration, and does so under a platform gate — see the
`if current_platform().is_npu:` block in `ops/attention/mha/triton.py`, which
imports the Ascend MHA functions and registers them with
`CapabilityRequirement(vendors={"ascend"})` and `solution="torch_npu"`. A
plugin with more than one vendor follows the same rules:

- **Leaves are plain functions; registration lives in `kernels/__init__.py`.**
  A leaf can then be unit-tested on its own host, and splitting a leaf into
  its own distribution later is a `pyproject.toml` change, not a code change.
- **Vendor imports sit behind the platform gate, never at module top level.**
  Importing `torch_npu` or a CuTe DSL module on the wrong host fails, and a
  failing `kernels/__init__.py` would take the model registration down with
  it.
- **Both vendors register under the same `(family, mode)` with different
  `vendors`; the facade calls `select_kernel` and contains no vendor `if`.**
  `KernelRegistry.get_for_operator` drops every spec whose capability the
  current platform does not satisfy, so selection is automatic. The family
  is the plugin's own name unless the kernel is meant to replace an in-tree
  one, in which case it registers in `Priority.PLUGIN` under the in-tree
  family.
- **A vendor leaf declares what it does not support as traits**, in the style
  of the in-tree `_NPU_OPTIONS` (`{"sliding_window": frozenset({False}),
  "support_sinks": frozenset({False}), ...}`). A request carrying an
  unsupported option then finds no candidate and fails at selection, instead
  of the kernel silently ignoring the argument.
- **Everything outside `kernels/` stays vendor-neutral.** `models/`,
  `attention/`, `cache/` import only the facades; `my_plugin.kernels` is the
  plugin's kernel boundary in the same sense that `tokenspeed_kernel` is the
  runtime's.

Triton source is not generally portable between CUDA and Ascend (tile shapes,
PDL, vendor `extra` libraries); the in-tree NPU path is a separate
implementation, not a re-registration of the CUDA one. Plan on two leaves and
share a source file only where portability has actually been verified.

One distribution with per-vendor extras is the starting point. The trigger
for splitting into `my-plugin` / `my-plugin-kernel-cuda` /
`my-plugin-kernel-ascend` — the repository's own `tokenspeed-kernel` /
`-amd` / `-npu` split — is when the two toolchains can no longer build in one
job: the Ascend wheel needs a CANN host, the CUDA wheel a CUDA host. After a
split the entry point stays on the neutral package; vendor packages are plain
dependencies with no entry point, exactly as in-tree.

Launching is unchanged: `python -m tokenspeed.launch_server --model-path
/path/to/foo ...`. The startup log shows `Loaded plugin 'my_plugin'
(my-tokenspeed-plugin 0.3.0): models=[FooForCausalLM, FooForCausalLMNextN]`.

What the plugin author must do for the model to be scheduled like an in-tree
one is exactly what an in-tree author must do: express per-request state as
cache groups. If the model has a standard KV layout, it names an existing
`cache_family` and inherits prefix caching and retraction; PD additionally
requires the declared role and transfer compatibility. If it
has a novel layout, it ships a `CacheRecipe`, not backend-private state —
the same rule `AGENTS.md` states for in-tree attention backends.

## Contract and compatibility

Opening the registries is the small part. The lasting cost is that a plugin
depends on a set of classes that mainline can refactor without knowing the
plugin exists. Three measures:

1. **Named contract.** A `docs/design/plugins.md` (written on acceptance,
   in the style of the existing design documents) lists the classes and
   protocols a plugin may subclass or call: `ModelProfile`, the `register_*`
   functions, `AttentionBackend`, `CacheRecipe`, `CachePool`,
   `PagedAttention`, `QuantizationConfig` / `QuantizeMethodBase`,
   `CacheLayerLayout`, `CheckpointMetadata`, `AttentionConfigInputs`,
   `BackendCompositionInputs`, `DraftContextProducer`, `ContextProducerFactory`,
   `EncoderModelFacts`, `BaseDrafter`, `TargetCaptureConfigurator`,
   and the forward-metadata
   protocol they receive. A change to one of these bumps
   `PLUGIN_API_VERSION` and gets a line in the release notes. Everything not
   on the list is internal.
2. **Fixture plugin in CI.** `test/plugins/fixture_plugin/` is a real
   installable package that registers a tiny model (MHA and MLA variants), a
   kernel in the `PLUGIN` band that overrides an in-tree one under a
   controlled `solution`, a trivial `CacheRecipe` family, a quantization
   method and a drafter. A test installs it and runs an end-to-end generation
   with CUDA graphs, PD and speculative decoding on a tiny config. Mainline
   refactors that break the contract fail here rather than in a downstream
   deployment. Each phase ships its relevant fixture cases before merging:
   built-in override ordering, failed partial registration and repeated calls,
   paired disable handling, explicit collision policy, MLA/DSA geometry,
   nonstandard cache families, multiple attention instances, config-dependent
   tokenizer options in every process, and draft routing/geometry/storage
   under both eager execution and CUDA graphs. Add forced-backend tests with
   conflicting user choices on both model sides, hybrid leaf composition, and
   PD role tests covering V4 draft rejection, Inkling with either a draft or
   an algorithm alone, target-only acceptance and separate DSpark profiles.
   Include DSpark index-derived stage depth that differs from the base model,
   missing-index failure, nonuniform PP execution-to-cache mapping and exact
   field coverage, specialized V4.1 components/row layout, and Qwen4 PLE/QSA
   composition with explicit leaf selection on target and draft views. Cover
   identity and hybrid composition, config-less draft construction, arena
   rebinding without duplicate wrappers, and eager/graph lifecycle forwarding.
   Add model-constructor layout forwarding and actual attention-ID checks;
   duplicate/cross-distribution entry-point rejection without imports;
   non-first config aliases, conflicting aliases/types and HF fallback; and
   explicit kernel priority, replacement of a specialized candidate, equal
   priority ambiguity and disjoint-vendor registration cases.
   Negative cases must fail
   startup before any request can run.
3. **Exact pins.** Plugins pin `tokenspeed` and `tokenspeed_kernel` exactly,
   as the kernel plugin README already requires. The RFC does not promise
   compatibility across versions; it promises that breaking the contract is
   visible.

## Alignment with design principles

- *One scheduling path, one execution path.* Nothing here adds a path. After
  P3 there is one architecture-resolution path instead of one for in-tree
  names and one for everything else defaulting to MHA.
- *Cache groups own per-request state.* Plugins are held to the same rule as
  in-tree code, and the `CacheRecipe` seam is how they comply.
- *Explicit parameters, no hidden behavioral defaults.* Required `override`
  makes replacement a visible decision; missing profiles and plugin load
  failures abort startup; loaded plugins are always logged.
- *Dependency boundaries.* Runtime code still reaches kernels only through
  `tokenspeed_kernel`; a plugin's own kernels are its own dependency.

## Phasing

This PR changes only the RFC. The implemented baseline is the inventory
above, including API version 1 and `test/runtime/test_runtime_plugins.py`;
the work below is planned on top of it. Each phase includes its contract
tests and documentation before it is independently mergeable.

1. **Harden the existing plugin contract.** Keep the current discovery and
   registrations; add strict kernel/runtime failure handling, shared disable
   policy, entry-point identity validation, all-alias config lookup and explicit
   runtime/kernel registration arguments. Introduce the next profile
   schema, forced target/draft backend selection and common PD-role checks.
   Preserve geometry initialization, tokenizer transport and token-history
   behavior already implemented. Replace the multiplicity field with the
   late layout resolver and derive counts and PP cache windows from its one
   layout, after checkpoint metadata is available. Add the typed config and
   composition factories at the common construction points above, preserving
   specialized components, DSpark's config-less draft and model results. Pass
   finalized cache layouts and bindings to model constructors, and feed the
   existing modality and encoder gates from the profile in this phase. Draft
   mappings/target sets are explicit but unused until phase 2. These breaking
   changes bump `PLUGIN_API_VERSION` to 2 together with plugin migrations;
   API-1 packages are not silently interpreted as API-2 declarations. Begin
   the installable fixture and design-contract document here.
2. **Complete remaining dispatch contracts.** Add quantization registration
   and `get_quant_method` unification; unify the existing plugin and in-tree
   drafter resolvers with geometry/storage traits, per-stage context-producer
   factories, layerwise-PD finalization guarantees and profile draft routing;
   remove the base drafter's default `False` finalization flag so migrated
   implementations must declare it. Add `register_mha_kernel_solution` and
   remaining CLI validation. Preserve
   existing drafter checkpoint-selection arguments during migration.
3. **In-tree profile migration.** Give every entry class a profile and remove
   architecture-name tables only after forced-backend and PD restrictions,
   geometry, cache ownership and tokenizer behavior have equivalent coverage.
4. **Contract consolidation.** Complete `docs/design/plugins.md` and the
   installable fixture's end-to-end CI matrix, extending the existing unit
   coverage. Any further incompatible contract change bumps the API version.

## Alternatives considered

- **Namespace-package injection**: make `tokenspeed.runtime.models` a
  namespace package so plugin modules land in the existing scan. Solves
  discovery of the class only; the attention family would still resolve to
  MHA, and it breaks the "one distribution owns the package" assumption.
- **`--load-format extensible` / `ext_def_file`**: an existing user-pointed
  import, but it runs in the loader after `ModelConfig` has resolved the
  family, and is scoped to processors around an already-registered model.
- **Explicit `--plugins pkg,pkg` instead of entry points**: see Open
  questions. Not chosen as the default because it makes the plugin a launch
  argument in every deployment script rather than a property of the
  environment, and because the kernel side already uses entry points.
- **Keep tables, accept per-architecture PRs**: pushes proprietary
  architecture names into mainline tables without the implementation, which
  is worse than either state.

## Open questions

1. **Auto-discovery vs allowlist.** Entry points activate on installation. An
   alternative is `--plugins my_plugin,other` (or `TOKENSPEED_PLUGINS`) as
   the only activation, with entry points used purely for lookup. This is
   more explicit, at the cost of a launch argument. A middle ground: entry
   points auto-load, and `--plugins` additionally accepts import paths for
   packages without entry points (development, notebooks).
2. **Should `override=True` on a model be allowed at all**, or should a
   plugin that wants a variant of an in-tree model register it under its own
   architecture name and rely on `architectures=`? Allowing it is more
   useful for downstream fixes; forbidding it keeps "which code served this
   architecture" unambiguous from the architecture string alone.
3. **Multimodal migration coverage.** Wrappers own the profile and forward
   resolved facts as specified in P3. Which wrapper/encoder combinations
   should the fixture matrix cover before their architecture tables retire?
4. **Kernel-package parity.** Should `tokenspeed_kernel.plugins` grow the
   same `PLUGIN_API_VERSION` constant and a fixture plugin in its own CI, so
   both halves of the contract are tested where they live?
