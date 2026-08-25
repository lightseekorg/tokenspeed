# CI Task Specs

`test/ci/` is the source of truth for CI task declarations consumed by
`test/ci_system/pipeline.py`.

Current trigger values:

- `per-commit`
- `manual`
- `nightly`
- `debug`

Supported task types:

- `ut`
- `server_smoke`
- `eval`
- `perf`

Currently configured task directories:

- `eval`
- `perf`
- `ut`

Every task declares one `workflow_stage`:

- `unit-test` for kernel and runtime tests
- `model-test` for model evaluation and performance tests

The PR workflows run these stages in that order. Matrix entries within a stage
run in parallel, but a later stage starts only after every required job in the
previous stage succeeds. A stage with no matching tasks is treated as
successfully satisfied.

PRs labeled `high priority` start `unit-test` and `model-test` concurrently.
Applying the label starts a new CI run immediately and cancels the older run
through the workflow's concurrency policy. A unit-test failure does not cancel
model tests that are already running in this mode.

The Qwen3.5 FP8 DeepEP correctness task runs GSM8K on four B200 GPUs with
attention TP2, attention DP2, and MoE EP4. DeepEP `auto` mode exercises its
normal path during prefill and low-latency path during decode, and the task
uses the bounded non-thinking chat template for CI stability. The task requires
a score of at least 0.90.

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

Typical use: adjust one runner instance without disturbing the same task's
dispatch order on other GPU families. Priority only affects jobs within the
same workflow stage; later stages cannot contend with earlier ones.

`retries` (eval/perf only) is a non-negative integer: the pipeline restarts the
managed server and reruns later stages that many extra times after a crash or
score miss. Use it for infrastructure flakes (CUDA launch failure, NVLink
barrier timeout, GPU memory-access fault) where a clean second attempt is
cheap relative to a red PR.

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

The NVIDIA PR workflow routes `b200-<Ngpu>` task labels to
`b200v2-<Ngpu>` by default. Set the `TOKENSPEED_B200_RUNNER_LABEL` repository
variable in GitHub Actions (`Settings` -> `Secrets and variables` -> `Actions`
-> `Variables`) to a non-empty runner family, such as `b200`, to override the
default without editing task YAML.

Only `b200v2-*` jobs enable a persistent, node-local package cache. They reuse
pip downloads from `/raid/cache/pip` and explicitly downloaded release wheels
from `/raid/cache/wheelhouse`; when `FLASHINFER_CACHE_DIR` points elsewhere,
the two directories are created beside that cache instead. This survives
runner pod recreation and avoids downloading the same large wheels again on
that node. Other runner families keep their existing cache behavior because
their cluster storage layouts may differ.

To enable `push` and `workflow_dispatch` runs of the three PR test workflows
outside the official repository, set the `TOKENSPEED_CI_REPOSITORY` repository
variable at the same settings path to the configured repository's exact
`owner/repo` name. The official
`lightseekorg/tokenspeed` repository remains enabled without this variable.
Leave it unset or empty to keep push/manual GPU CI disabled in other
repositories. `pull_request` runs keep their existing behavior. The configured
repository must also provide the matching self-hosted runner labels and any
required secrets; this variable only controls the repository gate.

The NVIDIA PR workflow excludes `h100` and `b300` runners by default, including
for fork PRs where repository variables are unavailable. To temporarily remove
additional unavailable GPU runners from PR test matrices, set the
`TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS` repository variable to comma-separated,
case-insensitive substrings such as `gb200, mi355`. Matching uses the resolved
runner label after applying `TOKENSPEED_B200_RUNNER_LABEL`; `mi355` therefore
matches `amd-mi355-*`. Empty entries are ignored. If every runner in a workflow
group is excluded, its matrix job is skipped while the workflow still
finishes. This variable applies only to the three PR test workflows. Clear or
unset it to restore all runner labels except the NVIDIA workflow's `h100` and
`b300` baselines.

The CI system derives `SM` from common runner label prefixes by default:
`h100`/`h200` use `sm90`, `b200`/`gb200` use `sm100`, and `b300`/`gb300` use
`sm103`. Use `runner.env.<label>` only for environment variables that should
override or extend the defaults for a single runner label.

PR workflows split runner labels by vendor and host architecture. `PR Test
NVIDIA` uses the `nvidia-x86` runner group, while `PR Test NVIDIA ARM` uses
the `nvidia-arm` runner group. GB300 is classified as NVIDIA ARM, but is not
declared in task YAMLs and therefore does not enter default CI matrices.

## Slurm with Pyxis/Enroot

`slurm_submit.py` submits an existing task YAML without copying its server,
evaluation, performance, or threshold configuration into a second format. It
targets one task at a time. By default the task uses one node, and the GPU count
comes from the selected runner label, such as `gb200-4gpu`.

An eval or perf task can request multiple nodes with an explicit topology:

```yaml
triggers: [slurm]
runner:
  labels: [slurm-gb200-4node-4gpu]
slurm:
  nodes: 4
  gpus_per_node: 4
```

Multi-node tasks must use only the `slurm` trigger and a `slurm-*` runner label,
so GitHub runner matrices cannot select them accidentally. The GPU count in the
label must equal `slurm.gpus_per_node`.

For a multi-node task, the generated batch script extracts the committed source
snapshot into a server workspace on every node and a separate client workspace
on the first node. It starts one containerized server task per node, then runs
the readiness probe plus eval/perf stages in the first node's client workspace.
Keeping the workspaces separate prevents concurrent install stages from writing
the same source tree. The server command is identical on every node. TokenSpeed derives
`nnodes`, `node_rank`, and the rendezvous address from the Slurm step variables;
do not add those flags to the task's server command. When the client step exits,
the script terminates the server step and removes each node's local snapshot.

The submitter expects Pyxis/Enroot support in Slurm. Before submission it
archives the clean, committed `HEAD` into the artifact root. The compute node
extracts that immutable snapshot under `SLURM_TMPDIR` and mounts it at
`/workspace`, so the checkout itself does not need to be shared. The artifact
root, cache directory, and any additional host mounts do need to be visible at
the same paths on the login and compute nodes. The default container is the
NVIDIA release image
`docker.io#lightseekorg/tokenspeed:<version>`, where `<version>` is read from
`python/pyproject.toml`. Override it with `--container-image` when testing a
different build. The container needs Python and pip. If PyYAML is absent, the
job installs `PyYAML>=6,<7` into its job-local `/tmp` before starting the
pipeline; images that already provide PyYAML do not perform this bootstrap.

The generated `sbatch` command uses `/tmp` as its working directory because the
login-node checkout may not be mounted on compute nodes. Override it with
`--sbatch-workdir` only when the selected path is compute-node-visible.

By default, the task's top-level `install` stage runs so a runner/base image
tests the exact committed checkout. Task-specific `eval.install` and
`perf.install` stages run afterward. Use `--skip-install` only with a release
image that already contains the intended TokenSpeed build.

The job gets the node exclusively by default so another job cannot contend for
its GPU or fixed service ports. `--no-exclusive` opts out. Runtime cleanup is
scoped to the Slurm job and never kills unrelated listeners on the node.

Render the exact `sbatch` command and job script without submitting:

```bash
python3 test/ci_system/slurm_submit.py \
  --config test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml \
  --partition batch \
  --render
```

Rendering a multi-node config also validates the two-step orchestration without
allocating nodes. The output should contain a four-node `sbatch`, a server
`srun` with one task per node, and a one-node client `srun` with `--relative=0`.

Submit the same evaluation and follow its output:

```bash
python3 test/ci_system/slurm_submit.py \
  --config test/ci/eval/qwen3.5-397b-a17b-nvfp4-dp4ep4-evalscope-aime25.yaml \
  --partition batch \
  --cache-dir /mnt/lustre01/$USER/tokenspeed-cache \
  --pass-env HF_TOKEN \
  --follow
```

Submit every eval/perf YAML that declares an exact runner label:

```bash
python3 test/ci_system/slurm_submit.py \
  --all \
  --runner gb200-4gpu \
  --partition batch \
  --cache-dir /mnt/lustre01/$USER/tokenspeed-cache
```

`--trigger manual` (or another trigger) optionally narrows `--all`. All
matching tasks are submitted before `--follow` starts, so their Slurm jobs can
run concurrently.

On a Slurm coordinator, use the shell launcher for manual scheduling. It
supplies the cluster's shared artifact/cache paths and pinned runner image.
The defaults target GB200; on GB300 set the shared paths under
`/data/home/$USER`:

```bash
TS_CI_ARTIFACT_ROOT=/data/home/$USER/tokenspeed-slurm \
TS_CI_CACHE_DIR=/data/home/$USER/tokenspeed-cache \
TS_CI_LOCAL_MODEL_ROOT=/scratch/$USER-models \
test/ci/run_slurm.sh \
  test/ci/eval/kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml \
  --runner slurm-gb300-4gpu \
  --type eval \
  --wait
```

GB300 server containers mount `TS_CI_LOCAL_MODEL_ROOT` read-only at `/models`.
It defaults to `/scratch/$USER-models`, the node-local RAID path. The Kimi-K3
GB300 tasks use pinned snapshots below that mount so weight loading does not
read the shared Hugging Face cache. Synchronize these directories to every
eligible GB300 node before enabling the tasks:

```text
moonshotai--Kimi-K3/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
nvidia--Kimi-K3-NVFP4/f8c5234a0a880bcc6cbf779a315e7ee2f405b812
Inferact--Kimi-K3-DSpark/cf6b8244620e7ea4b0651d214f28e89eac75bed6
```

The `Slurm Dispatch` workflow exposes a `cluster` input. `gb200` keeps the
existing `slurm-dispatch` coordinator and runner defaults. `gb300` is an
explicit opt-in: select one YAML that declares exactly one `gb300-Ngpu` or
`slurm-gb300-Ngpu` label. The workflow passes that label through unchanged and
validates any explicit runner selection against it. Five
`slurm-dispatch-gb300` coordinators form one shared pool for manual and
per-commit submissions. GB300 perf tasks are disabled until GB300-specific
reference values are measured.

The `GB300 Slurm Per Commit` workflow selects only multi-node model tasks with
the `per-commit` trigger and submits them through the same
`slurm-dispatch-gb300` coordinator pool used by manual dispatch. It runs for
pushes to `main` and for non-draft pull requests whose head branch belongs to
this repository. Pull-request runs execute the merge commit's dispatcher, so
dispatcher changes are covered before merge. Fork pull requests remain skipped
until the coordinator pool uses ephemeral runners with a protected approval
environment; use the manual `Slurm Dispatch` workflow after review. New commits
cancel the older run for the same pull request or the `main` branch.

Submission is fail-closed and requires the repository variable
`TOKENSPEED_CI_GB300_SLURM_PER_COMMIT_ENABLED` to equal `true`. The dedicated
switch is separate from `TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS`: this workflow
does not pass that variable to its matrix scan, so entries such as `gb300`
cannot filter the multi-node matrix here. During this workflow's
bootstrap only, leave the switch unset; after dispatcher support reaches
`main`, set it to `true` and re-run the merge commit's workflow.

The two-node Kimi K3 tasks declare `slurm-gb300-4gpu`, `slurm.nodes: 2`, and
`slurm.gpus_per_node: 4`. The runner label describes GPUs per node, while the
Slurm topology fields describe the allocation. The NVFP4 DSpark task pairs the
pinned `nvidia/Kimi-K3-NVFP4` target with `Inferact/Kimi-K3-DSpark` and
preserves the draft checkpoint's required `attn_res` auxiliary stream.
The MXFP4 DSpark OCRBench and MMMU-Pro Vision tasks are manual-only baselines
that use a pinned Kimi Vendor Verifier revision; select either YAML through
`Slurm Dispatch` with the `gb300` cluster.

GB200 examples:

```bash
# One existing YAML:
test/ci/run_slurm.sh \
  test/ci/eval/qwen3.5-122b-a10b-nvfp4-evalscope-ocr-bench.yaml \
  --follow

# Every existing YAML for one exact runner label:
test/ci/run_slurm.sh --all --runner gb200-4gpu --trigger manual

# List Kimi eval/perf tasks from PR 795 for two runner labels:
test/ci/run_slurm.sh \
  --pr 795 \
  --all \
  --runner b200-4gpu \
  --runner gb200-4gpu \
  --match kimi \
  --list

# Submit selected task types; repeat --type and --match as needed:
test/ci/run_slurm.sh \
  --pr https://github.com/lightseekorg/tokenspeed/pull/795 \
  --all \
  --runner b200-1gpu \
  --runner b200-2gpu \
  --type ut \
  --type eval \
  --type perf \
  --match kimi \
  --follow
```

This is a manual launcher, not a GitHub Actions runner. Override its defaults
with `TS_CI_ARTIFACT_ROOT`, `TS_CI_CACHE_DIR`, or
`TS_CI_CONTAINER_IMAGE`.

`--pr` accepts a pull request number or GitHub URL. It fetches the PR head and
merges it into the launcher's committed `HEAD` in an isolated temporary
worktree. The original checkout is not modified, and submitted jobs use an
immutable archive of that merged commit. A merge conflict stops before any job
is submitted.

`--source-pr` accepts the same values but only labels the report; it neither
fetches nor merges, and is for callers that already checked out the pull
request's merge commit.

Repeat `--runner` to select multiple exact labels. Repeat `--type` to select
from `ut`, `server_smoke`, `eval`, and `perf`; without `--type`, the
backward-compatible default is `eval` plus `perf`. Repeat `--match` to select
tasks whose name, YAML path, server command, or selected task command contains
any supplied substring (case-insensitive). `--list` prints the final matrix
without creating snapshots or submitting jobs, while `--render` prints the
generated Slurm commands and scripts.

`--wait` keeps the dispatcher process alive until every submitted Slurm job
reaches a terminal state. `--report-dir PATH` then collects a `manifest.json`,
Markdown summary, per-job logs, and available `result.json` files under
`PATH`. The command succeeds only when every job reaches `COMPLETED`. SIGINT or
SIGTERM while waiting calls `scancel` for the jobs submitted by that command.

The manual `Slurm Dispatch` GitHub workflow runs on the organization runner
with the `slurm-dispatch` label. That runner belongs on the Slurm coordinator and
only needs GitHub runner prerequisites, this repository, Python/PyYAML, and
Slurm client commands; it does not need GPUs. Leaving the PR input blank checks
out and runs the latest `main`; otherwise the requested PR is merged into that
trusted `main` checkout. From the Actions UI, optionally provide a PR,
comma-separated runner labels and task types, and an optional comma-separated
task/model filter. The workflow submits the selected matrix, waits for all
jobs, writes the aggregate table to the GitHub step summary, and uploads the
collected report directory as an artifact. It excludes long-running MMLU tasks
by default; explicitly enable `include_mmlu` in the manual workflow inputs when
that coverage is required.

The `yaml` input is `off` by default. Select one listed B200/GB200 CI YAML to
run that YAML independently of the bulk runner, type, match, trigger, and MMLU
filters. Every B200 or GB200 runner label declared by the selected YAML is
submitted as its own Slurm job.

The manual workflow keeps the dispatcher checkout on trusted `main` and merges
the requested PR only in the submitter's temporary worktree. The per-commit
workflow instead executes a same-repository PR's merge commit so dispatcher
changes can be validated before merge. Fork PRs must not use that path while
the coordinator pool is persistent.

For a YAML with multiple runner labels, select one or more explicitly with
repeated `--runner`.
Site-specific scheduler settings can be supplied with `--account`, `--qos`,
`--constraint`, `--time`, and `--gpu-type`. Additional host paths can be
mounted with repeated `--mount HOST:CONTAINER[:FLAGS]` options. Exported
`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` values are passed automatically;
`--pass-env NAME` passes other exported variables by name without writing their
values into the job script.

Submitted job snapshots, scripts, metadata, logs, and run results are written
below `.ci-artifacts/slurm` by default. `--render` only prints the command and
script; it does not create or submit them. The artifact root must be writable
from the compute node and should be on shared storage. Use `--artifact-root`
(or `TS_CI_ARTIFACT_ROOT`) to put artifacts elsewhere. `--cache-dir` mounts a
persistent host cache at `/home/runner/.cache`, matching the NVIDIA release
image, and points the Hugging Face and XDG caches there; the directory must
likewise be visible on the compute node.
