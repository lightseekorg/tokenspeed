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

The scheduler already needs nothing for this: the C++ scheduler and the event
loop see cache groups, block granularity and a capacity model, never a model
class. What is closed today is the runtime's *discovery* layer — the model
registry scans only its own package, architecture names are matched against
hard-coded tables to pick the attention family, and several CLI arguments are
closed `choices=` lists. The kernel package already ships an entry-point
plugin system designed for this, but the runtime never invokes it.

The proposal is therefore small in mechanism and mostly a matter of turning
closed tables into registries and moving one piece of knowledge to where it
belongs:

1. A `tokenspeed.plugins` entry-point group and one idempotent
   `ensure_loaded()` call, made in every process before a `ModelConfig` is
   built. It also runs `tokenspeed_kernel.plugins.discover_plugins()`.
2. One `register_*` function per dispatch point that is a closed container
   today (models, attention families, cache recipes and pools, quantization
   methods, drafters); attention backends already have one.
3. A declarative `ModelProfile` on the model class that states its attention
   arch, cache family and forced/default backends, replacing the
   architecture-name tables. In-tree models migrate onto it, so plugin and
   in-tree models resolve through the same path.
4. CLI validation of backend/quantization/algorithm names moves from
   argparse `choices=` to a post-discovery registry check.
5. A fixture plugin under `test/` that exercises the whole surface end to end,
   so mainline refactors that break the plugin contract fail CI.

## Motivation

Downstream deployers commonly have a model or a set of kernels they cannot
publish — a proprietary architecture, a vendor-restricted kernel, an
in-flight research variant — and still want the scheduler, the cache
subsystem, PD disaggregation and speculative decoding from mainline, tracked
closely. Today their options are:

- **Fork and rebase.** Every mainline refactor of the touched registries is a
  conflict; the fork drifts and stops contributing back.
- **Monkey-patch at import time** (`ModelRegistry.models[...] = ...`,
  `_BACKEND_REGISTRY[...] = ...`). Works for the two containers that happen
  to be mutable dicts; fails for `_ATTENTION_FAMILY_SPECS` (a tuple of
  frozensets) and for every `choices=` list; breaks silently on rename.
- **Re-implement the serving loop** around the kernel package. Discards the
  part they actually wanted.

None of these is a design. The project's own collaboration principle — core
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
expresses its state as cache groups gets scheduling, PD transfer, retraction
and prefix caching for free. This RFC does not change that contract; it only
lets a model that lives outside `tokenspeed.runtime.models` reach it.

## Current state: inventory of dispatch points

Paths are relative to `python/tokenspeed/runtime/` unless noted. "Open" means
an out-of-tree package can reach it today without patching; "closed" means it
cannot.

| Dispatch point | Where | Mechanism today | State |
| --- | --- | --- | --- |
| Kernel implementation selection | `tokenspeed_kernel/registry.py`, `tokenspeed_kernel/plugins/` | `@register_kernel`, `KernelRegistry.register` allows re-registration by name, `Priority.PLUGIN` band 16–19 reserved for plugins, `TOKENSPEED_KERNEL_OVERRIDE_<FAMILY>_<MODE>` env override, entry-point group `tokenspeed_kernel.plugins` with `discover_plugins()` | Open on the kernel side; **the runtime never calls `discover_plugins()`** |
| HF `architectures[...]` → model class | `models/registry.py` `import_model_classes` | `pkgutil.iter_modules` over `tokenspeed.runtime.models` only, collecting `EntryClass`; `ModelRegistry.models` is a plain dict on a module singleton | Closed (patchable, unsupported) |
| Architecture → attention arch, default backend, default prefix granularity | `configs/model_config.py` `_ATTENTION_FAMILY_SPECS`, `_resolve_attention_family`, `_apply_attention_family_defaults` | Tuple of `_AttentionFamilySpec(architectures=frozenset(...))`; an architecture in no spec **silently resolves to MHA** | Closed |
| Architecture → hybrid/linear/DSpark/Inkling facts, forced backend, cache family | `layers/attention/registry.py` `_HYBRID_*_ARCHITECTURES`, `_INKLING_ARCHITECTURES`, `_DSPARK_DRAFT_ARCHITECTURES`, `_resolve_attn_side`, `_apply_backend_overrides`, `_resolve_cache_family` | Architecture-name sets → `_AttnSideProfile` booleans → if-chains | Closed |
| Attention backend name → class | `layers/attention/registry.py` `register_backend`, `_BACKEND_REGISTRY` | Module-level dict; every in-tree backend module registers itself at import | Open dict, but `--attention-backend` / `--drafter-attention-backend` are closed `choices=` lists (`utils/server_args.py`), and `backends/paged/mha.py` `_KERNEL_SOLUTION_BY_BACKEND` is a closed name → solution map |
| Cache family → recipe / pool | `layers/attention/kv_cache/recipes/setup.py` `CacheModelFamily` (a `Literal`), `_RECIPES`; `layers/attention/kv_cache/factory.py` `_mha_pool_class`, `create_cache_pool` | Closed dict + if-chain on the family string | Closed; `CacheRecipe` itself is a well-shaped ABC |
| Quantization method name → config class | `layers/quantization/__init__.py` `QUANTIZATION_METHODS`; `layers/linear.py` `LinearBase.__init__` | Closed dict; per-layer method chosen by an `isinstance` chain over config classes, bypassing `QuantizationConfig.get_quant_method` for most of them | Closed |
| Speculative algorithm → drafter | `execution/drafter/__init__.py` `get_drafter_impl`; `execution/factory.py` `configure_draft_target`; `utils/hf_transformers_utils.py` draft architecture rewrite | Local dict plus `isinstance` special cases on the draft model class; `("DFLASH", "DSPARK")` hard-coded as the algorithms needing `TargetCaptureConfigurator`; draft architecture derived by string suffixing | Closed |
| Sampling backend name → class | `sampling/registry.py` `register_backend` | Module-level dict | Open |
| Process-level startup hook for third-party code | — | None. `ExtensibleLM`'s `ext_def_file` (`models/extensible.py`) is a user-pointed import but is scoped to input/output processors and runs in the loader, after `ModelConfig` has already resolved the attention family | Missing |

One process-boundary fact shapes the design: `ModelConfig` is constructed
independently in the frontend (`engine/async_llm.py`), the scheduler process
(`engine/event_loop.py`) and the encode loop (`epd/encode_loop.py`), and the
architecture → family resolution runs inside `ModelConfig.__init__`. Plugin
discovery therefore cannot live only in the GPU worker's `build_device_side`;
it has to run in every process, before the first `ModelConfig`.

## Proposal

### P1. `tokenspeed.runtime.plugins`: discovery

A new module mirroring `tokenspeed_kernel.plugins`:

```python
ENTRY_POINT_GROUP = "tokenspeed.plugins"
DISABLE_ENV_VAR = "TOKENSPEED_DISABLE_PLUGINS"
PLUGIN_API_VERSION = 1

def ensure_loaded() -> list[PluginInfo]: ...
def list_plugins() -> list[PluginInfo]: ...
```

Semantics:

- `ensure_loaded()` is idempotent per process. It first imports
  `tokenspeed_kernel` (built-in kernels register at import) and calls
  `tokenspeed_kernel.plugins.discover_plugins()`, then walks the
  `tokenspeed.plugins` group sorted by entry-point name and calls each
  `register()`.
- The single call site is the top of `ModelConfig.__init__`, before
  `_resolve_attention_family`. That covers every process that builds a
  config. `build_device_side` needs no call of its own because it receives
  an already-built `ModelConfig`.
- A plugin whose `register()` raises is reported with a `UserWarning` and
  skipped; the host keeps starting. This matches the kernel side.
- Every loaded plugin logs its distribution, version and the names it
  registered in each registry, at `INFO`, once, so a serving log always
  shows what out-of-tree code is active.
- `TOKENSPEED_DISABLE_PLUGINS=a,b` skips entry points by name, as the kernel
  side does.

Plugins are activated by installation. The existence of a `--plugins`
allowlist as an alternative is discussed under Open questions.

### P2. Registration surface

Each closed container in the inventory becomes a registry with one
registration function. The in-tree content is the registry's initial state;
nothing about in-tree resolution changes. All functions live in
`tokenspeed.runtime.plugins.registry` and are the plugin author's whole
import surface for registration.

```python
def register_model(cls: type[nn.Module], *, architectures: tuple[str, ...] = (), override: bool = False) -> None
def register_attention_family(spec: AttentionFamilySpec, *, override: bool = False) -> None
def register_attention_backend(name: str, archs: set[AttentionArch], cls: type[AttentionBackend], *, override: bool = False) -> None
def register_cache_recipe(family: str, recipe: Callable[..., CacheRecipe], *, override: bool = False) -> None
def register_cache_pool(family: str, factory: Callable[..., CachePool], *, override: bool = False) -> None
def register_quantization_method(name: str, cls: type[QuantizationConfig], *, override: bool = False) -> None
def register_drafter(algorithm: str, cls: type[BaseDrafter], *, draft_model_cls: type[nn.Module] | None = None, override: bool = False) -> None
```

`override` defaults to `False` and a name collision with an in-tree entry
raises. Replacing an in-tree model, recipe or method is legitimate — that is
how a downstream package ships a fixed or specialized variant — but it must be
a visible decision in the plugin's source, and it is logged.

Per registry:

**Models.** `import_model_classes()` keeps scanning `tokenspeed.runtime.models`
for `EntryClass`; its result seeds the registry. `register_model(cls)` keys
by `cls.__name__` unless `architectures` is given (a plugin may need to claim
an HF architecture string that differs from its class name).

**Attention families.** `_AttentionFamilySpec` becomes public as
`AttentionFamilySpec`; `_ATTENTION_FAMILY_SPECS` becomes a registry the
tuple seeds. This is the one registration that cannot be deferred to a later
phase: without it a plugin model with MLA or DSA attention would be built as
MHA with no error. Phase 3 replaces this registry with `ModelProfile`.

**Attention backends.** `register_backend` already exists; it is re-exported.
`_KERNEL_SOLUTION_BY_BACKEND` in `backends/paged/mha.py` becomes a
`register_mha_kernel_solution(backend_name, solution)` on the same module,
so a plugin can add a backend name that routes MHA leaves to its own
`solution` string.

**Cache recipes and pools.** `CacheModelFamily` changes from `Literal[...]`
to `str`. `_RECIPES` and the family dispatch in `create_cache_pool` /
`_mha_pool_class` become registries. A plugin with a non-standard KV layout
subclasses `CacheRecipe` (the seams are `layer_types`, `group_ids`,
`fields_for_layer`, `groups`, `packing`, `workspace_bytes`, `pool_options`)
and registers the recipe and the pool under a new family name. A plugin with
a standard layout registers nothing here and names an existing family.

**Quantization.** `QUANTIZATION_METHODS` becomes a registry. The
`isinstance` chain in `LinearBase.__init__` is replaced by one call,
`self.quant_method = quant_config.get_quant_method(self, prefix)`, with each
in-tree config class implementing the branch that today lives in
`linear.py`. This is a correctness fix independent of plugins: today a new
`QuantizationConfig` subclass that is not one of the listed classes falls
through the chain with no `quant_method` set.

**Drafters.** `DRAFTER_MAPPING` and the `isinstance` special cases become a
registry keyed `(algorithm, draft_model_cls | None)`; resolution picks the
most specific match by `isinstance`, falling back to `(algorithm, None)`.
The `("DFLASH", "DSPARK")` check in `configure_draft_target` becomes a class
attribute on the drafter, `requires_target_capture: ClassVar[bool]`, which is
what the check is actually asking. The draft-architecture rewrite in
`hf_transformers_utils.py` reads `ModelProfile.draft_architecture` (P3) when
present and keeps its suffixing fallback otherwise.

**CLI.** `--attention-backend`, `--drafter-attention-backend`,
`--quantization`, `--speculative-algorithm` and `--moe-backend` drop
argparse `choices=` (or the `MoeBackend` enum coercion) and are validated
against the corresponding registry after `ensure_loaded()`, producing the
same "unknown X, available: [...]" error a user gets today. The help text
lists the in-tree names and says plugins may add more.

### P3. `ModelProfile`: the model declares its own family facts

P2 makes the tables extensible, but a plugin would still be registering
*its architecture name into central tables* — knowledge about the model
living away from the model. The intended end state moves that knowledge onto
the class:

```python
@dataclass(frozen=True)
class ModelProfile:
    attention_arch: AttentionArch
    cache_family: str                              # a registered recipe/pool family
    linear_attention: Literal["none", "gdn", "kda"] = "none"
    forced_attention_backend: str | None = None    # e.g. "deepseek_v41", "hybrid_linear_attn"
    default_attention_backend: str | None = None   # used only if the user passed none
    default_prefix_granularity: int | None = None
    draft_architecture: str | None = None          # entry class name of the draft variant
    is_draft_of: str | None = None                 # for draft classes: the target entry class

class FooForCausalLM(nn.Module):
    profile: ClassVar[ModelProfile] = ModelProfile(
        attention_arch=AttentionArch.MLA,
        cache_family="mla",
        default_attention_backend="flashmla",
        default_prefix_granularity=64,
        draft_architecture="FooForCausalLMNextN",
    )
```

Resolution becomes: `hf_config.architectures` → `ModelRegistry` → the
class → `cls.profile`. `_resolve_attention_family`, `_resolve_attn_side`,
`_apply_backend_overrides` and `_resolve_cache_family` read the profile
instead of matching names. In-tree models gain a `profile` and the
`_*_ARCHITECTURES` sets are deleted once every in-tree entry class has one.
A class without a `profile` is an error at registration, not a fallback to
MHA.

This is the actual dependency inversion in the RFC: after it, there is one
resolution path shared by in-tree and out-of-tree models, which is the
project's stated preference ("one path; parameters, not branches"). The
exact field set is finalized during the in-tree migration, when every
boolean in `_AttnSideProfile` has to find a home.

### P4. Kernel side

No new mechanism. Three clarifications become documentation:

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
    if PLUGIN_API_VERSION != 1:
        raise RuntimeError(f"my_plugin targets plugin API 1, host has {PLUGIN_API_VERSION}")
    from my_plugin.models import FooForCausalLM, FooForCausalLMNextN
    register_model(FooForCausalLM)
    register_model(FooForCausalLMNextN)
```

```python
# my_plugin/kernels/__init__.py
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import register_kernel

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
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
            ),
            signatures=FOO_SIGNATURES,
        )(cuda_foo)
    if platform.is_npu:
        from my_plugin.kernels._ascend import foo as ascend_foo

        register_kernel(
            "my_plugin", "foo", name="torch_npu_my_plugin_foo", solution="torch_npu",
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
`cache_family` and inherits PD transfer, prefix caching and retraction. If it
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
   `BaseDrafter`, `TargetCaptureConfigurator`, and the forward-metadata
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
   deployment.
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
- *Explicit parameters, no hidden behavioral defaults.* `override=False`
  makes replacing in-tree entries a visible decision; a missing `profile` is
  an error, not a fallback; loaded plugins are always logged.
- *Dependency boundaries.* Runtime code still reaches kernels only through
  `tokenspeed_kernel`; a plugin's own kernels are its own dependency.

## Phasing

Each phase is independently mergeable and useful.

1. **Discovery and the two registrations that make a private model
   runnable.** `tokenspeed.runtime.plugins` with `ensure_loaded()` (calling
   the kernel discovery), `register_model`, `register_attention_family`,
   re-exported `register_attention_backend`, CLI validation moved after
   discovery. This alone covers a proprietary model with MHA/MLA/DSA attention
   plus proprietary kernels imported directly.
2. **Remaining registries.** Cache recipe/pool, quantization (including the
   `get_quant_method` unification), drafter (including
   `requires_target_capture`), `register_mha_kernel_solution`.
3. **`ModelProfile`.** Add the class attribute to every in-tree entry class,
   switch the four resolvers to read it, delete the architecture-name tables
   and `_AttentionFamilySpec`.
4. **Contract.** `docs/design/plugins.md`, `PLUGIN_API_VERSION`, the fixture
   plugin and its CI job.

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
3. **Profile placement for multimodal wrappers.** Some entry classes wrap a
   text model; whether `profile` lives on the wrapper, the text model, or is
   forwarded is a detail of the P3 migration.
4. **Kernel-package parity.** Should `tokenspeed_kernel.plugins` grow the
   same `PLUGIN_API_VERSION` constant and a fixture plugin in its own CI, so
   both halves of the contract are tested where they live?
