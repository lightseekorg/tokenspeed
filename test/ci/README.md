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
- `perf`
- `ut`

Every task has a normalized `timeout_minutes`. It defaults to `60` and accepts
an integer from `1` through `360`; booleans and out-of-range values are invalid.
The normalized value is copied into each matrix entry and becomes the GitHub
job timeout, so a long benchmark must declare its real upper bound in the task
spec instead of relying on a workflow-wide constant.

Eval and perf server tasks may declare a list of regular expressions under
`server.forbidden_log_patterns`. The runner scans `.ci-artifacts/server.log`
after orderly server shutdown and before broad fallback cleanup can hide a
leak. A log violation cannot replace an earlier workload error, and a missing
log fails only when the server stage actually ran. Match details are bounded in
the result JSON.

Release tasks that set `isolated_python: true` use a task-local virtual
environment with system and user site-packages disabled. Split install/execute
steps reuse that exact environment, while inherited Python and pip override
variables are removed. A task may also declare typed `server.shutdown` state;
that path launches the real command as the process-group root, sends SIGTERM
only to that root, requires exit code zero, and records descendants, process
group, ports, selected GPU state, and relevant zombies in a JSON artifact.
Fallback group cleanup is hygiene only and never changes a failed graceful
shutdown into a pass.

Managed server identity is also recorded outside the checkout under a
runner-scoped `/tmp/tokenspeed-ci-managed-servers/` registry. A later split CI
step recovers only a matching PID/start-time identity even if the prior work
directory was deleted. Ordinary stage commands run in their own process group;
on cancellation the Linux runner temporarily acts as a child subreaper, gives
nested managed-server handlers a bounded TERM grace period, and reaps adopted
descendants before using KILL.

The three PR workflows upload the complete `.ci-artifacts/` directory,
including hidden files, rather than guessing workload-specific filenames.
Workload validators should write durable JSON, logs, and memory samples below
that directory even when validation fails.

## Product environment contract

Every PR workflow runs `test/ci_system/product_env_guard.py` in the matrix-scan
job, before any accelerator task is dispatched. The AST-based check rejects
first-party `os.getenv`/`os.environ` reads, dynamic keys, `setdefault`, and
`envs.FIELD` configuration unless the exact source file, variable name, and
read/write direction are reviewed as an external protocol.

The small allowlist covers authentication, standard distributed-launcher and
Prometheus contracts, plus deterministic writes that project explicit typed
configuration into CUDA/NCCL/Torch/TRT-LLM process APIs. Product namespaces
such as `TOKENSPEED_*`, `TS_*`, `SMG_*`, and `EPD_*` are never allowlisted.
Add a typed CLI/config/API field instead of extending the process environment.

The MiniMax-M3 published-package preflight applies this contract to the clean
wheel's TokenSpeed adapter Python subtree and the shared `mm_rdma.py` module
that adapter imports. Other SMG serving adapters are outside this release
surface, while the compiled SMG binding is checked separately for the exact
legacy MiniMax-M3 environment keys.

Runtime CI can enable NVIDIA exception dumps explicitly with
`test/runtime/run_ci_suite.py --cuda-coredump-dir PATH`. The helper projects
that reviewed CLI value into the CUDA driver's required `CUDA_*` protocol for
child processes; there is no TokenSpeed environment switch or import-time
activation. NVIDIA runtime-suite tasks write those dumps below
`.ci-artifacts/cuda-coredumps`, which the PR workflows retain on failure.

## Targeted manual dispatch

Manual dispatches of `PR Test NVIDIA`, `PR Test NVIDIA ARM`, and `PR Test AMD`
require the `task_names` input. It is a comma-separated list of exact,
case-sensitive task names from the YAML `name` fields; wildcards and substring
matching are not supported. The filter applies only to `workflow_dispatch`.
Push and pull-request scans continue to select tasks solely from their trigger
and runner group.

Selection is fail-closed. The scan exits unsuccessfully if the list is empty,
contains an empty or duplicate entry, names an unknown task, selects a task
that does not support the requested trigger, or leaves any selected task with
no matrix entry after runner-group and runner-exclusion filtering. It never
silently expands an invalid selection to the full manual matrix or runs only
the valid subset of a partially invalid selection.

For example, this dispatch selects only the seven MiniMax-M3 release tasks:

```bash
gh workflow run pr-test-nvidia.yml \
  --repo lightseekorg/tokenspeed \
  --ref <upstream-ref> \
  -f trigger=manual \
  -f task_names='eval-minimax-m3-mxfp8-gsm8k,perf-minimax-m3-bf16-cache-random,perf-minimax-m3-mxfp8-active-mm,perf-minimax-m3-mxfp8-encoder-graph-ab,perf-minimax-m3-mxfp8-random,perf-minimax-m3-mxfp8-exact-longctx,ut-runtime-minimax-m3'
```

`<upstream-ref>` must be a branch or tag in the target repository and must
contain this workflow plus the selected task files. A commit that exists only
on a fork is not dispatchable through the upstream workflow. If repository
policy permits a temporary upstream CI tag, create it at the exact candidate
SHA, retain the run URL and artifacts, and delete the tag only after the full
run has completed.

The equivalent local matrix check is:

```bash
python3 test/ci_system/pipeline.py scan \
  --root test/ci \
  --trigger manual \
  --runner-group nvidia-x86 \
  --task-names 'eval-minimax-m3-mxfp8-gsm8k,perf-minimax-m3-bf16-cache-random,perf-minimax-m3-mxfp8-active-mm,perf-minimax-m3-mxfp8-encoder-graph-ab,perf-minimax-m3-mxfp8-random,perf-minimax-m3-mxfp8-exact-longctx,ut-runtime-minimax-m3'
```

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

To enable `push` and `workflow_dispatch` runs of the three PR test workflows
outside the official repository, set the `TOKENSPEED_CI_REPOSITORY` repository
variable at the same settings path to the configured repository's exact
`owner/repo` name. The official
`lightseekorg/tokenspeed` repository remains enabled without this variable.
Leave it unset or empty to keep push/manual GPU CI disabled in other
repositories. `pull_request` runs keep their existing behavior. The configured
repository must also provide the matching self-hosted runner labels and any
required secrets; this variable only controls the repository gate.

To temporarily remove unavailable GPU runners from PR test matrices, set the
`TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS` repository variable to comma-separated,
case-insensitive substrings such as `b300, mi355`. Matching uses the resolved
runner label after applying `TOKENSPEED_B200_RUNNER_LABEL`; `b300` therefore
matches both `b300-*` and `gb300-*`, while `mi355` matches
`amd-mi355-*`. Empty entries are ignored. If every runner in a workflow group
is excluded, its matrix job is skipped while the workflow still finishes.
This variable applies only to the three PR test workflows. Clear or unset it to
restore all runner labels.

These `TOKENSPEED_*` values are GitHub repository variables consumed by the
workflow control plane. They are not inherited product-runtime environment
configuration; release tasks separately reject runtime `TOKENSPEED_*`,
`SMG_*`, `EPD_*`, and `TS_*` variables before starting TokenSpeed.

The CI system derives `SM` from common runner label prefixes by default:
`h100`/`h200` use `sm90`, `b200`/`gb200` use `sm100`, and `b300`/`gb300` use
`sm103`. Use `runner.env.<label>` only for environment variables that should
override or extend the defaults for a single runner label.

PR workflows split runner labels by vendor and host architecture. `PR Test
NVIDIA` uses the `nvidia-x86` runner group, while `PR Test NVIDIA ARM` uses
the `nvidia-arm` runner group for `gb200` labels.
