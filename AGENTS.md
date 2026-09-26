# General Agent Guidelines

> If a `AGENTS.local.md` file exists alongside this file, read and respect it--
> it contains developer-specific overrides that supplement this shared guidance.

## Collaboration principle

Core features will be designed and implemented by the TokenSpeed core team.
This isn't a matter of distrust in external contributions — writing code has
gotten cheaper, but reviewing it, validating it, and deploying it safely at
production scale hasn't. If anything, that cost has gone up. As Steve Jobs
put it, A players want to work with A players. We believe the gap between the
best people and average people is more than tenfold.

## Development environment

* Before any work, check local Python venv and activate if one exists.
* Don't install pip packages outside the local Python venv if one exists.

## Code changes

* Add tests and update docs for the changed code.
* For code comments, use common/existing terms for easy human understanding;
  avoid obsecure terms or coining unnecessary new concepts.
* Parameters that select execution paths, algorithms, or correctness-critical
  behavior must be explicit and have no defaults. This includes execution modes,
  backend selection, and flags that switch between implementations.
* Genuinely optional inputs may have defaults when omission has a clear meaning
  within the selected path. Review each default individually; convenience alone
  does not justify defaulting a behavioral choice.
* Wrappers must preserve explicitly supplied arguments and must not silently
  discard unsupported arguments.
* Use absolute imports instead of relative imports.
* Use f-strings for Python string interpolation, including logging messages.
  Keep format templates required by APIs such as `strftime` and logging
  formatters in their required syntax.
* Declare and initialize instance fields explicitly in `__init__` or as
  dataclass fields. Do not attach undeclared attributes after construction.
  Represent optional state with an initialized field, such as
  `self.x: int | None = None`, rather than a sometimes-missing attribute.
  Access fields directly; avoid `hasattr`, `getattr`, and `setattr` for class
  state, including fallback values that hide missing declarations.
* Use the repository's full MIT license header for copyright notices; do not use
  an abbreviated copyright-only header.
* Before creating commits, run `pre-commit run --all-files` to format.
* Do not substitute a narrower lint command for the repository hook before
  committing. Always run the exact `pre-commit run --all-files` command and
  commit any formatter changes it makes.
* When creating commits, perform sign off on behalf of the author.

## Code review

When Codex or Claude Code reviews code changes, consult these references for
the languages involved:

* For C++ changes, consult the
  [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and
  the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines).
* For Python changes, consult the
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Design principles

We value one scheduling path and one execution path. Prefill/decode
disaggregation or not, speculation or not, CUDA graph or not, overlap or not:
these are parameters of the same path, never a second path. Make the general
path cover the case instead of adding a mode-specific branch.

When attention needs new per-request state, first ask whether the LCM cache
subsystem and the C++ scheduler can own it — as a cache group with its own
block granularity, allocated, prefix-matched, transferred and freed with the
request's other blocks — before adding state maintenance inside a particular
model or attention backend. Backend-private state is the exception, not the
default.

`docs/design/` records the deliberate invariants of each subsystem — what
belongs where, and why. Read the document covering the code you are touching
before changing it, and review against it: the rules there were established on
purpose, so a deviation is a bug unless the document is updated in the same
change.

* `docs/design/event-loop.md` — the scheduler event loop: the control
  plane / data plane split and what the loop is allowed to hold, centralized
  scheduler feedback, in-flight depth, the hooks pattern.
* `docs/design/cache-concepts.md` — KV cache vocabulary and the layering
  between prefix matching, allocation and page geometry.
* `docs/design/scheduler.md` — the C++ scheduler's admission granularity,
  what triggers retraction in each engine role, and the recovery protocol.
* `docs/design/unified_path.md` — the unified decode path: one
  refresh-in-place metadata contract for eager and CUDA-graph decode, the
  padding contract, buffer sizing, and what stays graph-only.

## Public pull requests

* Keep PR titles, descriptions, commit messages, diffs, comments, logs, and
  artifacts limited to public information. Never include private repository
  names or links, private dates, or any other private or internal information.

## Dependency boundaries

* `tokenspeed` runtime dependencies should stay vendor-neutral.
* Runtime code should use `tokenspeed-kernel` as its only kernel package
  boundary.
* Third-party kernel libraries belong under `tokenspeed-kernel`; avoid direct
  runtime dependencies or imports that bypass it.
* If a dependency repeatedly breaks during version upgrades or slows project
  progress, consider removing it entirely or at least making it optional.

## Hardware and model support scope

* NVIDIA GPU support is currently limited to `sm90`, `sm100`, `sm103`, and
  `sm107`.
* AMD GPU support is currently limited to `gfx950` and `gfx1250`.
* NPU support targets only one or two specific models. There are currently no
  plans to expand NPU model coverage.

## tokenspeed-scheduler releases

Prefer separate PRs for scheduler code changes and version bumps. A scheduler
code change does not require a version bump or an immediate release; multiple
code changes may accumulate until a release is needed.

Follow this sequence:

1. Make and merge code changes under `tokenspeed-scheduler/`.
2. When ready to release, update `[project].version` in
   `tokenspeed-scheduler/pyproject.toml` and merge the version bump into `main`.
3. Trigger the
   [release-tokenspeed-scheduler workflow](https://github.com/lightseekorg/tokenspeed/actions/workflows/release-tokenspeed-scheduler.yml)
   from `main`. Wait for the GitHub release and PyPI publication to succeed.
4. Once the new version is available on PyPI, update the main TokenSpeed
   project's `tokenspeed-scheduler` dependency requirement in
   `python/pyproject.toml` through a follow-up PR targeting `main`.

## tokenspeed-kernel

Inside the root `tokenspeed-kernel/` directory:

* All direct tokenspeed-triton imports should happen in `_triton.py` and then
  re-import to other places.
* Avoid using `triton` directly; use `tokenspeed_triton` instead.
* Avoid using `torch.compile`; prefer writing the fused kernel directly in
  Triton.
* All direct third-party code should be placed in `thirdparty/` and imported
  into `ops/` then registered via `register_kernel`.
* Prefer CuteDSL for NVIDIA GPU kernels and Triton Gluon for AMD GPU kernels.
  Use Triton for portable solutions across vendors. Vendor libraries should
  stay optional, and other solutions may be used as temporary transitions, but
  new work should consolidate toward these backend choices.
* Files under `ops/` should follow `<family>/<solution>` structure, like
  `gemm/trtllm.py`. Attention adds its variant before the solution, for example
  `attention/mha/triton.py`; multi-file implementations keep helpers under a
  private directory such as `attention/mha/_triton/`.
* For op traits, use existing ones if there are. If needing to create new ones,
  name it consistently with existing ones.
* Top-level `README.md` should only contain high-level kernel system designs
  geared for human understanding. For per-op details, use `README.md` files
  under corresponding `ops/` directory.
* Prefer to `@register_kernel` with the name as the Python `def` function
  attached to, prefixed with its solution (e.g, `triton_mha_prefill`).
* When defining new public APIs, explain arguments and returns in docstring.
* Keep vendor-only code in files or private directories named after its
  vendor-specific solution (`cute_dsl`, `gluon`, ...). CI skips the other
  vendor's GPU jobs based on these names. Code that serves both vendors belongs
  in a shared solution (`triton`).
* Vendor-specific tests should be placed under `test/<vendor>/` subdirectory.
  Tests for common infra and covering multi-vendors reside under `test/`
  directly.
* Use tight atol/rtol in correctness comparison tests.

## tokenspeed-kernel-amd

Inside the root `tokenspeed-kernel-amd/` directory:

* There should be no dependency on `tokenspeed-kernel`.
* Add jit `launch_metadata` for Proton use along the Triton/Gluon kernels.
* AMD Gluon Kernel tests should live in `tokenspeed-kernel/test/amd/` to reuse
  common platform utilities and reference computations.
* For per kernel contract and algorithm details, put in
  `python/tokenspeed_kernel_amd/ops/README.md`.
* For Triton/Gluon kernels, one name should thread the whole stack: the
  `register_kernel(name=...)` value, the registered Python `def` it decorates,
  and the `@gluon.jit` (or `@triton.jit`) kernel that does the op's work all
  share it. The AMD Python launcher should be called as `launch_<name>`.
  Extra kernels launched only by that op insert a role before the arch suffix
  (`gluon_mha_decode_reduce_gfx950`). Kernels shared by several registered ops
  keep descriptive names. A `repr=` on the jit decorator replaces the compiled
  symbol, so its base string must be the kernel's `def` name as well.
