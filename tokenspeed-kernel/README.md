# TokenSpeed-kernel

TokenSpeed-kernel aims to provide a collection of the best portable and
performant kernels for multi-silicon AI inference. It features:

* A clean layered API for maximal structured flexibility
* Kernel registration and selection logic to decouple complexity and increase reuse
* Plugin mechanism for multi-silicon extensibility
* A minimal list of curated dependencies for fast iteration

TokenSpeed-kernel is pip-installable on its own and can be directly used by
others.

## Design Goals

TokenSpeed-kernel is designed with the following functionality goals in mind:

* Support various kernels in AI models (attention, MoE, etc.)
* Support multiple silicon vendors and generations
* Marry default portability and performance solutions

In addition, to have a better devflow for fast iteration:

* Provide unified infra to verify and debug kernel numerics standalone
* Provide unified infra to run and benchmark kernels standalone
* Support tracing shapes and profiling workloads at runtime
* Stay forward-looking, with guardrails for agentic devflow

## Overall Design

With the above goals in mind, we have made the following opinionated design
choices (still evolving; subject to change):

### Layered system

```
                       public API  (mha_prefill, mm, moe_fused, ...)
                                       │
                           ┌───────────┴───────────┐
                           │     select_kernel     │  (family, mode, dtype, traits, ...)
                           └───────────┬───────────┘
                                       │ queries
                            ┌──────────┴──────────┐
                            │   KernelRegistry    │   ← @register_kernel(...) populates this
                            └──────────┬──────────┘
                                       │
       ┌──────────────┬────────────────┼────────────────┬───────────────┐
   attention         gemm             moe             norm     ...   (op family)
       │              │                │                │
  ┌────┼────┐    ┌────┼────┐      ┌────┼────┐      ┌────┼────┐
  triton         triton           triton           triton             ← in-tree portable JIT
  gluon          (...)            cute_dsl         (...)              ← in-tree perf JIT
  flash_mla      flashinfer       (...)            (...)              ← vendor library wrappers
                                ...
       │              │                │                │
       └──────────────┴────────────────┴────────────────┴── reference (PyTorch ground truth)
```

- **Registration** — backends register with `@register_kernel(family, mode, ...)`,
  declaring supported dtypes, arch capability requirements, traits (head dim,
  GQA factor, ...), and a priority band.
- **Auto-selection** — `select_kernel` filters by capability and traits,
  ranks the survivors with an optional per-family `SelectionOracle` and
  priority, and returns a callable. Selection accepts an objective (latency,
  throughput, determinism, portability) and supports per-call `override=` plus
  config-file overrides for development.

### Directory structure

```
tokenspeed_kernel/
  __init__.py            # Public API re-exports
  platform.py            # PlatformInfo, capability detection
  registry.py            # KernelRegistry, register_kernel, Priority bands
  selection.py           # select_kernel, oracles, overrides
  profiling.py           # ShapeCapture, kernel_scope, Proton bootstrap
  _triton.py             # Single import point for the vendored Triton fork

  ops/
    attention/   { triton/, flash_attn/, ... }
    gemm/        { triton.py, trtllm.py, ... }
    moe/         { triton.py, deepep.py, triton_kernels.py, ... }
    ...

  numerics/              # Reference impls + tolerance + comparison + CLI
    reference/           # PyTorch ground-truth kernels
  benchmark/             # Unified runner, throughput model, report, CLI
  plugins/               # Out-of-tree backend discovery
  thirdparty/            # Vendored / wrapped third-party kernel sources
```

Each `ops/<family>/` directory holds peer subdirectories — one per
solution. A solution is either an in-tree JIT kernel (Triton/Gluon/CuteDSL),
or a thin wrapper around an external library.
All of them register through the same decorator and are scored by the same
selection logic, so adding a backend is one new file in the right family
folder.

### Solution choices

- **Triton** — in-tree; default portable JIT path for various kernels
- **Gluon / CuteDSL** — in-tree; performant JIT path for key kernels
- **Vendor libraries** — wrapped (FlashAttention, TRT-LLM, etc.);
  no in-tree C++ build
- **PyTorch reference** — under `numerics/reference/`; never auto-selects
  over a real backend but always available as ground truth

Overall we carefully curate external dependencies and actively re-evaluate
their inclusion, in order to maintain minimal dependencies and enable faster
iteration.

### Numerics, benchmarking, profiling

- `python -m tokenspeed_kernel.numerics` — dtype-aware tolerances, standard
  input generators, and a comparison/bisect flow that pits any registered
  kernel against the reference impl.
- `python -m tokenspeed_kernel.benchmark` — unified timing, throughput
  (FLOPs / bytes) per op family, tabular reports, and Proton integration.
- Runtime shape capture feeds replay and tuning workflows; `kernel_scope`
  scopes are visible in Proton/Chrome traces.

### Plugins

`python -m tokenspeed_kernel.plugins` lists discovered out-of-tree backends.
Plugins register via the same `@register_kernel` decorator from their own
package, set their own priority, and participate in selection like in-tree
backends. See `tokenspeed_kernel/plugins/README.md`.

## Public API

```python
from tokenspeed_kernel import (
    mha_prefill, mha_prefill_with_kvcache, mha_decode_with_kvcache,
    mla_decode_with_kvcache,
    mm,
    moe_route, moe_dispatch, moe_experts, moe_combine, moe_fused,
    ...
)
```

Using the above platform and solution-agnostic public APIs can get the most
value out of TokenSpeed-kernel; but one can also directly call into a
specific solution under `ops/<family>/`, or manually `select_kernel` with
targeted filters:

MLA decode has a separate public API because its query and KV cache layout are
different from MHA/GQA decode. `mla_decode_with_kvcache` selects the best
registered MLA backend for the current platform. This currently registers the
TokenSpeed MLA backend for Blackwell; Hopper intentionally has no default
native MLA backend until a dedicated Hopper implementation lands.

```python
from tokenspeed_kernel.selection import select_kernel, kernel_override
```

For platform checks:

```python
from tokenspeed_kernel.platform  import current_platform
```
