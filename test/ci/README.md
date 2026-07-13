# CI Task Specs

`test/ci/` is the source of truth for CI task declarations consumed by
`test/ci_system/pipeline.py`.

Current trigger values:

- `per-commit`
- `manual`
- `nightly`

Supported task types:

- `ut`
- `server_smoke`
- `eval`
- `perf`

Currently configured task directories:

- `eval`
- `ut`

Each task expands into one matrix entry per runner label. Add a top-level
`priority` to a task YAML to bias dispatch order. GitHub Actions starts matrix
jobs in include-list order, so `high` entries reach a contended runner pool
before `normal` (the default) and `low`. Tasks that omit `priority` keep their
original ordering.

`priority` accepts either a scalar (applies to every label of the task) or a
per-label mapping (only the listed labels are overridden; every other label
stays at `normal`):

```yaml
# whole task at high
priority: high

# only the b300-1gpu instance drops to low; h100-1gpu / b200-1gpu / ...
# of the same task keep the default normal
priority:
  b300-1gpu: low
```

Typical use: lower a 1gpu kernel unit-test on `b300-1gpu` so the heavier
b300-4gpu evals that share the same box claim the runner first, without
disturbing the same task's ordering on the other GPU families.

`optional` marks a task or per-label matrix entry as non-blocking.
Optional entries are emitted with `matrix.optional: true`, and the PR workflows
map that to GitHub Actions `continue-on-error`.

```yaml
# whole task can fail without blocking the workflow
optional: true

# only the MI355 bench entry is non-blocking; the MI350 entry of the same
# task still blocks on failure
optional:
  amd-mi355-1gpu-bench: true
```

`b200-<Ngpu>` labels are the default B200 runners. Set the
`TOKENSPEED_B200_RUNNER_LABEL` repository variable in GitHub Actions
(`Settings` -> `Secrets and variables` -> `Actions` -> `Variables`) to a
non-empty runner family such as `b200v2` to temporarily route them to
`b200v2-<Ngpu>` without editing task YAML. Leave the variable unset or empty to
use the default `b200-<Ngpu>` labels.

To temporarily remove unavailable GPU runners from PR test matrices, set the
`TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS` repository variable to comma-separated,
case-insensitive substrings such as `b300, mi355`. Matching uses the resolved
runner label after applying `TOKENSPEED_B200_RUNNER_LABEL`; `b300` therefore
matches both `b300-*` and `gb300-*`, while `mi355` matches
`amd-mi355-*`. Empty entries are ignored. If every runner in a workflow group
is excluded, its matrix job is skipped while the workflow still finishes.
This variable applies only to the three PR test workflows. Clear or unset it to
restore all runner labels.

The CI system derives `SM` from common runner label prefixes by default:
`h100`/`h200` use `sm90`, `b200`/`gb200` use `sm100`, and `b300`/`gb300` use
`sm103`. Use `runner.env.<label>` only for environment variables that should
override or extend the defaults for a single runner label.

PR workflows split runner labels by vendor and host architecture. `PR Test
NVIDIA` uses the `nvidia-x86` runner group, while `PR Test NVIDIA ARM` uses
the `nvidia-arm` runner group for `gb200` labels.
