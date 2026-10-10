# TokenSpeed-kernel

TokenSpeed-kernel aims to provide a collection of the best portable and
performant kernels for multi-silicon AI inference. It features:

* A clean layered API for maximal structured flexibility
* Kernel registration and selection logic to decouple complexity and increase reuse
* Plugin mechanism for multi-silicon extensibility
* A minimal list of curated dependencies for fast iteration

TokenSpeed-kernel is pip-installable on its own. Others can use it
directly.

## Nightly installation

CUDA 13 nightly wheels are published daily from `main` for Linux x86_64 and
ARM64, with Python 3.10–3.13. Versions append the UTC build date to the base
version, for example `0.1.3.post20260929`.

```bash
pip install --upgrade tokenspeed-kernel \
  --extra-index-url https://lightseek.org/whl/nightly
```

PyPI supplies dependencies that are absent from the nightly index. To select a
specific nightly, use `tokenspeed-kernel==0.1.3.post20260929`. Post releases sort
above the corresponding base release and do not require `--pre`.

The `Build and Release tokenspeed-kernel` workflow also supports manual nightly
builds from pull request branches. To publish manually, run it from `main` with
both `nightly` and `publish_github` enabled. Same-day reruns preserve already
published wheels.
Historical nightlies are retained. Automatic cleanup is deferred.

## Design goals

TokenSpeed-kernel is designed with these functionality goals in mind:

* Support various kernels in AI models (such as attention and MoE)
* Support multiple silicon vendors and generations
* Marry default portability and performance solutions

Additional goals target a better devflow for fast iteration:

* Provide unified infra to verify and debug kernel numerics standalone
* Provide unified infra to run and benchmark kernels standalone
* Support tracing shapes and profiling workloads at runtime
* Stay forward-looking, with guardrails for agentic devflow

## Overall design

TokenSpeed-kernel makes these opinionated design choices toward the
preceding goals (still evolving; subject to change):

### Layered system

```
                       public API  (attention.mha.mha_prefill, mm, ...)
                                       │
                           ┌───────────┴───────────┐
                           │     select_kernel     │  (family, mode, format_signature, traits, ...)
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
  declaring supported `format_signatures`, arch capability requirements,
  non-format traits (head dim, GQA factor, ...), and a priority band.
- **Auto-selection** — `select_kernel` filters by capability and traits,
  ranks the survivors with an optional per-family `SelectionOracle` and
  priority, and returns a callable. Selection supports per-call `solution=`
  and `override=` plus config-file overrides for development.

### Directory structure

```
tokenspeed_kernel/
  __init__.py            # Public API re-exports
  platform.py            # PlatformInfo, capability detection
  signature.py           # TensorFormat, ScaleFormat, FormatSignature
  registry.py            # KernelRegistry, register_kernel, Priority bands
  selection.py           # select_kernel, oracles, overrides
  profiling.py           # ShapeCapture, kernel_scope, Proton bootstrap
  _triton.py             # Single import point for the vendored Triton fork

  ops/
    attention/   { mha/, mla/, dsa/, ... }
    gemm/        { triton.py, trtllm.py, ... }
    moe/         { triton/, flashinfer/, marlin/, ... }
    ...

  numerics/              # Reference impls + tolerance + comparison + CLI
    reference/           # PyTorch ground-truth kernels
  benchmark/             # Unified runner, throughput model, report, CLI
  plugins/               # Out-of-tree backend discovery
  thirdparty/            # Vendored / wrapped third-party kernel sources
```

Each `ops/<family>/` directory groups implementations by operator variant and
then solution. For example, attention uses `attention/<variant>/<solution>.py`
such as `attention/mha/triton.py`. A solution is either an in-tree JIT kernel
(Triton/Gluon/CuteDSL), or a thin wrapper around an external library.
All of them register through the same decorator, and the same selection
logic scores them, so adding a backend is one new file in the right family
folder.

### Solution choices

- **Triton** — in-tree; default portable JIT path for various kernels, including
  precomputed-routing MoE with unquantized or MXFP4 expert weights
- **Gluon / CuteDSL** — in-tree; performant JIT path for key kernels
- **gfx950 block-FP8 MoE** — direct compact-weight Gluon warp GEMVs for
  decode-shaped batches and tuned BF16 Gluon kernels for prefill. BF16 expert
  copies are created at load time while compact FP8 experts remain available
  for decode
- **Vendor libraries** — wrapped (such as FlashAttention and TRT-LLM);
  no in-tree C++ build
- **PyTorch reference** — under `numerics/reference/`; never auto-selects
  over a real backend but always available as ground truth

TokenSpeed-kernel carefully curates external dependencies and actively
re-evaluates their inclusion to maintain minimal dependencies and enable
faster iteration.

### Numerics, benchmarking, profiling

- `python -m tokenspeed_kernel.numerics` — dtype-aware tolerances, standard
  input generators, and a comparison/bisect flow that pits any registered
  kernel against the reference impl.
- `python -m tokenspeed_kernel.benchmark` — unified timing, throughput
  (FLOPs / bytes) per op family, tabular reports, and Proton integration.
- `KernelBenchmarkHarness` — registration-level device timing through warmed
  graph replay, with raw samples, resolved registration metadata, and explicit
  failure outcomes.
- Runtime shape capture feeds replay and tuning workflows. `kernel_scope`
  scopes are visible in Proton/Chrome traces. The joint BF16 `mm` fast path
  records the same shape metadata and scopes as registry-selected kernels.
- End-to-end serving: POST `/start_profile` with
  `{"activities": ["PROTON"]}`, run the workload, then POST `/stop_profile`.
  Each scheduler process — the process where
  kernels actually launch — runs its own Proton session and finalizes it on
  `/stop_profile`, writing
  `<output_dir>/<profile_id>[-DP<rank>]-TP<rank>.proton.<fmt>`
  per rank. `PROTON` composes only with host-side activities (`CPU`, `MEM`,
  `VIZTRACER`). To see Python activity and Proton's kernel lanes on one
  Perfetto timeline, profile with `VIZTRACER` + `PROTON`
  (`TOKENSPEED_KERNEL_PROFILE_DATA=trace`,
  `TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT=chrome_trace`), then merge the
  traces with `tokenspeed merge-traces`.

Registration-level benchmarks combine operation-owned input and correctness
logic with graph-replay device timing. Each operation family and mode
contributes one benchmark generator. Suites reference them by family, mode,
and parameters. Pull request CI compares compatible cases between the merge
base and candidate revision. See the
[benchmark documentation](benchmarks/README.md) for the harness and suite
contract, and the [CI documentation](../test/ci/README.md#registration-level-kernel-benchmarks)
for workflow behavior and runner requirements.

### JIT compilation while serving

Compile-time kernel parameters (`tl.constexpr`, `gl.constexpr`) key the
Triton compile cache. A per-batch value passed as one compiles a new binary
on the forward thread for every new batch shape (100 ms to seconds each).
`tokenspeed_kernel.compile_monitor` hooks Triton's JIT (Gluon shares it) and
records every compilation. The runtime installs it in each scheduler process
and marks the end of startup. After that, the monitor logs each compilation
with its duration, what changed in the compile key, and the launching call
site. When a compile-time parameter keeps taking new values from one call
site, the monitor names it (`TOKENSPEED_JIT_COMPILE_CHECK=warn`, the
default) or raises (`error`, which CI serving jobs use). Kernel tests guard
batch-varying launches with `assert_no_triton_compile` in `test/utils.py`.
The monitor does not observe JITs outside Triton, such as DeepGEMM's
per-shape kernels; they need the same discipline at their call sites.
A runtime argument still keys the cache: Triton specializes an integer on
whether it is 1 or divisible by 16, and a pointer on 16-byte alignment.
Startup warms only the classes graph capture happens to see. On the serving
path, a per-batch count (tokens, rows, requests) belongs in
`do_not_specialize`, and a pointer into a buffer sliced at a per-batch
offset belongs in `do_not_specialize_on_alignment`. A stride that changes
between call sites but stays a multiple of 16, such as a projection's row
width, stays a plain runtime argument: one class covers it, and the hint
keeps row loads vectorized. Use a fixed block with a loop for a
batch-derived block size rather than a power-of-two bucket; the bucket form
still compiles once per new bucket while serving. The end-of-startup mark is
also the package's compile switch, set whether or not the monitor is
installed. A kernel whose library compiles once per batch
shape and cannot bucket it, such as FlashInfer's joint BF16 GEMM (some runners
compile per exact row count) or the ll_bf16 router's dot-product kernel, checks
`compile_monitor.is_serving()` where it is dispatched: startup tuning and graph capture use it, and eager calls while
serving take a GEMM that never compiles (cuBLAS through torch on NVIDIA).

### Plugins

`python -m tokenspeed_kernel.plugins list` lists discovered out-of-tree backends.
Plugins register through the same `@register_kernel` decorator from their
own package, set their own priority, and participate in selection like
in-tree backends. See `tokenspeed_kernel/plugins/README.md`.

## Public API

```python
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.layernorm import grouped_gemma_rmsnorm
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights, moe_topk
from tokenspeed_kernel.ops.residual import gated_residual_combine, gated_residual_mix
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.ops.attention.msa import (
    msa_decode_with_kvcache,
    msa_extend_with_kvcache,
)
```

The preceding platform- and solution-agnostic public APIs provide the most
value from TokenSpeed-kernel. You can also call directly into a specific
solution under `ops/<family>/`, or run `select_kernel` manually with
targeted filters.

For targeted selection:

```python
from tokenspeed_kernel.selection import select_kernel, kernel_override
```

For platform checks:

```python
from tokenspeed_kernel.platform  import current_platform
```
