# General Agent Guidelines

> If a `AGENTS.local.md` file exists alongside this file, read and respect it--
> it contains developer-specific overrides that supplement this shared guidance.

## Collaboration principle

Core features will be designed and implemented by the TokenSpeed core team. This isn't a matter of distrust in external contributions — writing code has gotten cheaper, but reviewing it, validating it, and deploying it safely at production scale hasn't. If anything, that cost has gone up. As Steve Jobs put it, A players want to work with A players. We believe the gap between the best people and average people is more than tenfold.

## Development environment

* Before any work, check local Python venv and activate if one exists.
* Don't install pip packages outside the local Python venv if one exists.

## Code changes

* Add tests and update docs for the changed code.
* Avoid default parameter values; pass every argument explicitly at every call
  site. Explicit arguments matter more than convenience: a default silently
  supplies a value the caller never chose, so a missing or swallowed argument
  goes unnoticed instead of failing at the call.
* Use absolute imports instead of relative imports.
* Use the repository's full MIT license header for copyright notices; do not use
  an abbreviated copyright-only header.
* Before creating commits, run `pre-commit run --all-files` to format.
* Do not substitute a narrower lint command for the repository hook before
  committing. Always run the exact `pre-commit run --all-files` command and
  commit any formatter changes it makes.
* When creating commits, perform sign off on behalf of the author.

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

## tokenspeed-kernel

Inside the root `tokenspeed-kernel/` directory:

* All direct tokenspeed-triton imports should happen in `_triton.py` and then
  re-import to other places.
* All direct third-party code should be placed in `thirdparty/` and imported
  into `ops/` then registered via `register_kernel`.
* Prefer CuteDSL for NVIDIA GPU kernels and Triton Gluon for AMD GPU kernels.
  Use Triton for portable solutions across vendors. Vendor libraries should
  stay optional, and other solutions may be used as temporary transitions, but
  new work should consolidate toward these backend choices.
* Files under `ops/` should follow `<family>/<solution>` structure, like
  `gemm/trtllm.py` or `attention/triton/`.
* When defining new public APIs, explain arguments and returns in docstring.

## tokenspeed-kernel-amd

Inside the root `tokenspeed-kernel-amd/` directory:

* There should be no dependency on `tokenspeed-kernel`.
* AMD Gluon Kernel tests should live in `tokenspeed-kernel/test/` to reuse
  common platform utilities and reference computations.
