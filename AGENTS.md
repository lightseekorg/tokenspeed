# General Agent Guidelines

> If a `AGENTS.local.md` file exists alongside this file, read and respect it--
> it contains developer-specific overrides that supplement this shared guidance.

## Development environment

* Before any work, check local Python venv and activate if one exists.
* Don't install pip packages outside the local Python venv if one exists.

## Code changes

* Add tests and update docs for the changed code.
* Before creating commits, run `pre-commit run --all-files` to format.
* Do not substitute a narrower lint command for the repository hook before
  committing. Always run the exact `pre-commit run --all-files` command and
  commit any formatter changes it makes.
* When creating commits, perform sign off on behalf of the author.

## Dependency boundaries

* `tokenspeed` runtime dependencies should stay vendor-neutral.
* Runtime code should use `tokenspeed-kernel` as its only kernel package
  boundary.
* Third-party kernel libraries belong under `tokenspeed-kernel`; avoid direct
  runtime dependencies or imports that bypass it.
* If a dependency repeatedly breaks during version upgrades or slows project
  progress, consider removing it entirely or at least making it optional.

## tokenspeed-kernel

Inside the root tokenspeed-kernel/ directory:

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
