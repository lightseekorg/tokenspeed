# Ascend NPU ARM operator CI

## Triggers and change selection

The workflow is `.github/workflows/pr-test-ascendnpu-arm.yml`, displayed as `Ascend NPU Tests`.
It runs in `lightseekorg/tokenspeed` or the repository identified by the full `owner/repository` value of the `TOKENSPEED_NPU_CI_REPOSITORY` repository variable. Forks must explicitly set that variable.
Triggers include pushes to main, `pull_request` events targeting main, and `workflow_dispatch`.
The `select` job can run in an allowed repository. The hardware job additionally requires `TOKENSPEED_NPU_ARM_CI_ENABLED` to be set to `true`.
When that variable is absent or disabled, the hardware job is skipped without entering a runner queue. A positive selection result does not mean the operators have run.
The NPU allowlist uses its own `TOKENSPEED_NPU_CI_REPOSITORY` variable; it does not require the shared repository variable used by GPU workflows, so enabling NPU CI does not also enable other vendors' scheduling.

Supported PR events are `opened`, `synchronize`, `reopened`, and `ready_for_review`. Draft PRs do not start the NPU job.
Accuracy always precedes performance.
The independent `pr-test-npu-arm` concurrency group is keyed by PR or ref. Automatic events cancel older runs in the same group; manual events do not initiate cancellation.
Test code never runs through `pull_request_target`.

The GitHub-hosted `select` job checks out the event commit, runs the CPU-only API regression tests with `python3 -B -m unittest discover -s test/ci_system/npu -p test_changes.py`, then runs `test/ci_system/npu/changes.py`. Selection uses the same GitHub API change-list checks as the independent NPU workflow:

- PRs read every page of `/pulls/<number>/files` with 100 files per page. The current PR head and base must match the event before and after reading; the file count must remain stable and match the complete response. More than 3,000 files, duplicates, malformed paths, or partial responses fail selection.
- Pushes read `/compare/<before>...<after>`. Both SHAs must be nonzero. Responses with 300 or more files fail because the compare API may truncate the list. Creating a branch with an all-zero `before` fails selection instead of guessing missing changes.
- Manual runs select the full suite without an API file request.
- Renames check both `filename` and `previous_filename`; API failures fail the job instead of silently skipping tests.

The selector uses the event's PR identity and the complete API file list, without depending on the parents of GitHub's temporary test-merge commit. GitHub may supply an older test-merge SHA while refreshing that reference. The hardware job still checks out and verifies the event's exact `GITHUB_SHA`; selection does not substitute the PR head SHA. The selection step receives only the read-permission `GITHUB_TOKEN` through `GH_TOKEN`; no token is passed to operator test processes.

Path rules in `test/ci/npu/manifest.json` cover the NPU package, the shared `tokenspeed-kernel/python/` dependency, dependency configuration, NPU CI scripts, the task YAML and pipeline, and both the NPU caller and shared task workflow.
Documentation changes ending in `.md` or `.rst` are excluded. Matching is case-sensitive and includes additions, modifications, deletions, and both sides of a rename within the API completeness limits above.
Any matching path selects every registered operator. Unrelated changes emit `should_run=false` and `reason=no-watched-changes`.
The workflow has no top-level path filter, so unrelated changes still produce a selection result.
The caller passes the selected `test/ci/ut/npu-operators.yaml` task to `.github/workflows/run-ci-task-matrix.yml`, using the dedicated NPU runner label. This shared workflow invokes `test/ci_system/pipeline.py` in the existing NPU environment; the task has no dependency-install stage or Docker requirement.
NPU CI implementation and operator tests live under `test/ci_system/npu/` and `test/ci/npu/`; new operators do not require changes to GPU task lists.
The dedicated `npu` matrix group selects only `tokenspeed-npu-arm64`; AMD MI350, AMD MI450, and NVIDIA groups exclude that runner. GPU path selection also ignores NPU-only task YAML changes.

## Components and execution order

- `test/ci_system/npu/changes.py`: validates the manifest and event identity, then selects relevant changes.
- `test/ci/npu/manifest.json`: schema version 3; defines watched paths, operator order, one `script` path per operator, and expected counts.
- `test/ci/npu/expected-plans.json`: retains schema version 2; records accuracy cases, performance configurations, sampling parameters, and associated hashes.
- `test/ci_system/npu/run_step.sh`: checks ARM/Linux after checkout and prepares the host lock, environment, and caches.
- `test/ci_system/npu/run_phase.py`: executes one phase in order, supervises child processes and timeouts, saves evidence, and returns an exit code.
- `test/ci_system/npu/reporting.py`: validates structured logs and generates JSON reports and English Markdown summaries.
- `test/ci/npu/tests/<operator>-test.py`: combines numerical checks and performance sampling in one script with a required `--mode` argument.
- `test/ci/npu/operator_cli.py`: parses the explicit mode and runs the selected phase callbacks, stopping on the first failure.

The `tests/` directory contains the registered operator scripts. Each `<operator>-test.py` runs by file path through the manifest or an explicit command and requires an NPU for accuracy and performance modes. The hyphenated names are executable script paths, not Python module import names.

The task has exactly two sequential commands: `bash test/ci_system/npu/run_step.sh accuracy "$NPU_CI_WORK"`, then the same entry point with `performance`. Accuracy tests for all operators finish before performance tests begin.
The executor starts each manifest `script` with `--mode accuracy` or `--mode performance`; it has no operator-name branches.
The standalone `--mode all` option is a convenience for one operator only. CI does not use it: CI preserves the global ordering of all accuracy tests before any performance test.
Before invoking the pipeline, the shared workflow validates NPU deployment, platform, and event inputs, creates an isolated run/attempt directory, and checks out the requested SHA. `NPU_CI_WORK` identifies that directory; commands run from its `checkout/` directory.
Performance reuses the directory only after accuracy succeeds. Both commands retain the existing Linux, ARM64, SHA, and clean-checkout checks, host lock, fixed environment, and cache preparation in `run_step.sh`.
The shared workflow checks out the requested SHA with `actions/checkout` and does not persist checkout credentials for NPU tasks.
The interpreter path is fixed. Scripts use that interpreter with `-B -u`, and the two kernel package directories in the checkout precede other entries in `PYTHONPATH`.

| Operator | Accuracy cases | Performance configurations | Performance samples |
| --- | --- | --- | --- |
| scatter | 5 | 6 | 30 |
| virtual_slots_to_local | 48 | 8 | 40 |

The current suite has 53 accuracy cases, 14 performance configurations, and 70 samples.
Each configuration performs 10 warmup iterations, then collects 5 samples of 50 executions each.
Median latency is the median of the five per-execution sample averages, measured in milliseconds.
The `record_only` policy validates execution and sampling completeness only: `performance_verdict=not_evaluated` and `threshold_ms=null`.
There are no latency thresholds. A `completed` result does not mean performance passed a target.

## Failures, logs, and cleanup

Any case, configuration, script, checkout, environment, or required evidence-write failure returns nonzero and stops subsequent tests.
Later operators in the current phase are recorded as `not_run`; if accuracy fails, the performance command is skipped.
Only necessary evidence saving and resource cleanup continue after failure. Report or Actions summary write failures also fail the step.
Each script has a 600-second timeout, and the job has a 60-minute timeout. These limits are not performance thresholds.

Merged stdout/stderr is written to disk and flushed to the console as it arrives. Python output is unbuffered.
Each operator prints its start, finish, elapsed time, and exit code. A heartbeat is printed every 30 seconds while a script remains active.
Original exceptions are preserved. Error annotations escape `%`, CR, and LF; failure explanations and English summaries are generated within the same step.
Executor exceptions also include a traceback.

Test subprocesses run in private process groups. Completion, nonzero exit, timeout, SIGINT, and SIGTERM all attempt to terminate remaining processes in that group.
Test processes must not escape supervision by starting a new session. SIGKILL or host failure may leave partial logs without a final summary.
The host lock covers environment preparation and execution within each phase and is released when the executor exits after child-process cleanup.
Source recording, artifact upload, and work-directory cleanup use the shared workflow. Cleanup runs even after a task failure; phase process cleanup remains in the existing executor.

## Runner deployment contract and evidence locations

`runs-on` uses only the dedicated `tokenspeed-npu-arm64` label, without adding the default `self-hosted` label.
The hardware job is disabled by default. These are deployment requirements, not assertions that upstream already has a runner, NPU, or configured runtime.
From the repository home page, open **Settings > Secrets and variables > Actions > Variables** and configure these repository variables:

| Variable | Purpose |
| --- | --- |
| `TOKENSPEED_NPU_CI_REPOSITORY` | Forks set their full `owner/repository`; the official `lightseekorg/tokenspeed` repository is already allowlisted |
| `TOKENSPEED_NPU_ARM_RUNNER_NAME` | Must match the registered runner name and runtime `RUNNER_NAME`; there is no default |
| `TOKENSPEED_NPU_ARM_CI_ENABLED` | Set to `true` only after deployment requirements are met; leave unset to disable the hardware job |

Assign the dedicated label to an online Linux ARM64 NPU runner running under the `ci-npu` service account.
Once enabled, a missing or mismatched runner-name variable fails the step before checkout.
If no online runner has the required label, the job waits for scheduling and cannot execute runtime checks.
The environment entry point is `/srv/tokenspeed-npu/cann9-torch29-py312-root/env.sh`, and the interpreter is `venv/bin/python` under the same directory.
Working CANN, torch, torch_npu, and Triton Ascend dependencies and NPU device access for that account are required. Installing the runner alone is insufficient.
The account must have read/write access to `/srv/npu-ci/locks/host-all.lock`. An occupied lock fails immediately without starting tests.
The runner account must be able to create and remove job directories under `GITHUB_WORKSPACE`. The existing environment and lock paths remain deployment requirements.
PRs execute repository test code. Use a dedicated isolated runner, preserve GitHub's external-contributor approval policy, and do not inject deployment credentials into test processes.

The shared workflow creates `GITHUB_WORKSPACE/run-<run_id>-<attempt>-<task>-<runner>/`, with `checkout/`, separate `accuracy/` and `performance/` result directories, and caches.
Each phase directory contains `context.json`, `report.json`, and `summary.md`. Operator directories contain raw logs, `process.json`, `result.json`, and known exit codes.
Summaries are appended to `GITHUB_STEP_SUMMARY`. After execution, the workflow collects phase evidence under `checkout/.ci-artifacts/published/npu/` alongside the pipeline's `result.json` and the shared source reporter's `source.json` in `.ci-artifacts/`, then uploads the artifact.
The phase directories remain outside the checkout during execution so both commands can verify a clean source tree; `.ci-artifacts` is created only when the pipeline writes its final result.
Use the uploaded artifacts for verification after shared cleanup removes the entire run directory, including checkout and caches. Deploy from a clean baseline worktree rather than reusing a fault-injection branch.

## Adding cases, operators, and structured events

To add a case, update its script and test plan, then synchronize parameters, ordering, counts, and script hashes in `expected-plans.json`.
To add an operator, create `test/ci/npu/tests/<name>-test.py`, register its name, a single `script` field pointing to that canonical path, and counts in the manifest's `operators` list, and add its accuracy plan, performance plan, and timing parameters to the expected plan.
Each script must accept the required `--mode {accuracy,performance,all}` argument. Parse arguments before importing NPU dependencies so `--help`, missing arguments, and invalid values work on a CPU host.
Use the shared CLI with two callbacks returning integer exit codes; `all` calls accuracy first and starts performance only after accuracy returns zero. Exceptions, nonzero exits, and invalid callback results stop execution. Do not store the active mode in mutable global state.
Schema version 3 rejects the former per-phase `scripts` mapping. The expected plan keeps its existing case/configuration data and schema version 2, with one `source_sha256` entry for each registered script.
Update `watch_paths` for new source dependencies. Do not add operator-name branches to the common executor or additional workflow steps.

Each structured event occupies one line: `NPU_RESULT_JSON=<JSON object>`. Use `allow_nan=False` and flush output promptly.
Common fields are `schema_version=1`, `operator`, `phase`, and `event`; machine field names and values remain unchanged.
Other lines may contain diagnostics, but must not embed the structured marker. Duplicate keys, NaN/Infinity, out-of-order or missing events, and inconsistent counts fail validation.

- Accuracy: emit `started` with `expected_cases` and `case_names`; then `case_started` and `case_completed` for each case, including the complete `case`, planned metadata fields, and completion `status`; finally emit `completed` with expected, completed, passed, and failed counts.
- Performance: emit `started` with `expected_configs` and `config_names`; for each configuration, emit `config_started` with the complete `config`, metadata, `warmup`, `iterations`, and `expected_samples`. Emit `sample` events with a configuration-name string, zero-based `sample_index`, and positive finite `latency_ms`. Emit `config_completed` with the sample list, count, median, `threshold_ms=null`, and `performance_verdict=not_evaluated`; finally emit `completed` with expected and completed configuration counts.
- On the first failure, emit a terminal `failed` event and return nonzero. Include completed counts, `current_case` or `current_config`, and the original `error_type` and `error`. Do not continue to the next case or configuration. Uncaught import or process errors may omit the terminal event; the executor still reports failure.

See `validate_accuracy` and `validate_performance` in `reporting.py` for field validation, and the existing operator scripts for examples.
Any byte change to the manifest or registered scripts, including formatting, requires hash synchronization.
Hashes bind the plan to specific files; they do not replace review of expected values. After formatting, run from the repository root:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('test/ci/npu')
manifest = json.loads((root / 'manifest.json').read_text())
plan_path = root / 'expected-plans.json'
plan = json.loads(plan_path.read_text())
plan['manifest_sha256'] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
plan['source_sha256'] = {
    entry['script']: hashlib.sha256(Path(entry['script']).read_bytes()).hexdigest()
    for entry in manifest['operators']
}
plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
PY
```

Before committing, run the complete `pre-commit run --all-files` in the target environment, recheck the plan hashes, and sign off the commit. Validate operator changes through the ARM NPU workflow.
After loading the environment on an ARM runner, each operator has three standalone modes. For example, from the repository root:

```bash
export PYTHONPATH="$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python${PYTHONPATH:+:$PYTHONPATH}"
/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python -B -u \
  test/ci/npu/tests/scatter-test.py --mode accuracy
/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python -B -u \
  test/ci/npu/tests/scatter-test.py --mode performance
/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python -B -u \
  test/ci/npu/tests/scatter-test.py --mode all
```

Use `tests/virtual_slots_to_local-test.py` in place of `tests/scatter-test.py` for the other operator. There is no default mode.
`all` performs accuracy and then performance for that operator, preserving distinct `phase=accuracy` and `phase=performance` structured events. Any accuracy failure prevents performance from starting.
Standalone scripts print results but do not acquire the workflow host lock or save phase reports. They are not a replacement for the CI entry point.

To invoke the common phase executor directly with a new output directory:

```bash
/srv/tokenspeed-npu/cann9-torch29-py312-root/venv/bin/python -B -u \
  test/ci_system/npu/run_phase.py --repo-root "$PWD" --phase accuracy \
  --output-dir /tmp/npu-accuracy-unique --timeout-seconds 600
```

Direct executor invocation does not include the workflow's host lock or deployment checks. Prefer the workflow entry point for acceptance and use an output directory that does not already exist.
Final acceptance requires real Actions runs demonstrating success, unrelated-change skipping, accuracy failure with performance skipped, and agreement between Actions summaries and VM evidence.
