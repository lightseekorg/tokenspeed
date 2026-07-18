# Tokenspeed-kernel Plugin System

> **Status: experimental.** The plugin contract — entry-point group name,
> `register()` signature, `KernelSpec` fields, selection priority semantics,
> and the `tokenspeed_kernel.plugins` Python API — may change without
> backwards-compatibility guarantees while we shake the design out. Pin to
> an exact `tokenspeed-kernel` version in your plugin's dependencies.

This subpackage lets third-party packages register kernel implementations
into the global `KernelRegistry` without modifying `tokenspeed-kernel`
itself.

## How it works

1. Your plugin package exposes a `register()` function that calls
   `tokenspeed_kernel.registry.register_kernel(...)` (or
   `KernelRegistry.get().register(...)`) for each kernel it provides.
2. Your `pyproject.toml` advertises that function under the
   `tokenspeed_kernel.plugins` entry-point group.
3. The host application (engine, benchmark, notebook, etc.) calls
   `tokenspeed_kernel.plugins.discover_plugins()` once at startup, after
   built-in kernels have been imported. Discovery walks the entry-point
   group and invokes each `register()`.

Loading is **fully explicit** — importing `tokenspeed_kernel` or
`tokenspeed_kernel.plugins` does **not** trigger discovery on its own.

## Example plugin

A minimal out-of-tree package that contributes a custom decode-attention
kernel for NVIDIA Hopper.

```
my-kernels-plugin/
├── pyproject.toml
└── my_kernels_plugin/
    └── __init__.py
```

`my_kernels_plugin/__init__.py`:

```python
import torch

from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.registry import register_kernel


def register() -> None:
    """Entry point invoked by tokenspeed_kernel.plugins.discover_plugins()."""

    @register_kernel(
        "attention",
        "decode",
        solution="my_custom",
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
        ),
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
        ),
        # Built-in FlashInfer decode is priority 18; pick 19 to win selection.
        priority=19,
    )
    def my_custom_attn_decode(q, kv_cache, page_table, seq_lens, **kwargs):
        ...
```

`pyproject.toml`:

```toml
[project]
name = "my-kernels-plugin"
version = "0.1.0"
# Pin tightly while the plugin contract is experimental.
dependencies = ["tokenspeed-kernel==<exact-version>"]

[project.entry-points."tokenspeed_kernel.plugins"]
my_plugin = "my_kernels_plugin:register"
```

Install and load:

```bash
pip install -e ./my-kernels-plugin
```

```python
import tokenspeed_kernel  # registers built-in kernels
from tokenspeed_kernel.plugins import discover_plugins, list_plugins

discover_plugins()
print(list_plugins())  # -> [PluginInfo(name='my_plugin', ...)]
```

## Host-application integration

Engines and other long-running hosts should call `discover_plugins()`
exactly once at startup, after built-in kernel modules have been imported
(so plugins can override built-ins by registering at a higher priority).

```python
import tokenspeed_kernel  # noqa: F401  -- registers built-ins
from tokenspeed_kernel.plugins import discover_plugins

discover_plugins()
```

For ad-hoc use (notebooks, scripts, tests), there is no need to use entry
points at all — call `register_kernel(...)` directly:

```python
import torch
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.registry import register_kernel


@register_kernel(
    "gemm",
    "mm",
    solution="experiment",
    signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
    priority=15,
)
def my_experimental_gemm(a, b, **kwargs):
    ...
```

## Disabling plugins

Plugins can be skipped without uninstalling them:

```bash
TOKENSPEED_KERNEL_DISABLE_PLUGINS="my_plugin,other_plugin" python ...
```

```python
from tokenspeed_kernel.plugins import disable_plugin, discover_plugins

disable_plugin("my_plugin")
discover_plugins()
```

The names refer to entry-point names (the left-hand side of the
`[project.entry-points."tokenspeed_kernel.plugins"]` table), not
distribution names.

## Inspection

```bash
python -m tokenspeed_kernel.plugins list
python -m tokenspeed_kernel.plugins info my_plugin
```

```python
from tokenspeed_kernel.plugins import list_plugins

for info in list_plugins():
    print(info.name, info.version, info.kernel_names)
```

## Selection contract

- Priority is an integer in `[0, 20)`. Higher wins. The reference
  implementation lives at `0`. Built-in optimized kernels typically sit at
  `10`–`18`. Plugin authors who want to override a built-in should choose
  a value strictly higher than the built-in they replace.
- `discover_plugins()` walks entry points in alphabetical order by
  entry-point name. When two registrations land at the same priority for
  the same `(family, mode)`, the warning is emitted and selection becomes
  load-order-dependent — set explicit, distinct priorities to avoid this.
- A plugin whose `register()` raises does not crash discovery; a
  `UserWarning` is emitted and other plugins continue loading.

## Failure modes worth knowing

- **No discovery call → no plugin kernels.** Forgetting to call
  `discover_plugins()` is silent.
- **Plugin loaded before built-ins.** If you call `discover_plugins()`
  before `import tokenspeed_kernel`, plugins that intend to override
  built-ins will appear to win, but the built-in modules will be imported
  later and may overwrite the plugin's slot. Always import
  `tokenspeed_kernel` first.
- **Stale registry.** Calling `KernelRegistry.reset()` clears registered
  kernels but leaves `_loaded_plugins` populated; re-discovery will skip
  already-loaded plugins. Use `discover_plugins(force=True)` after a
  reset.
