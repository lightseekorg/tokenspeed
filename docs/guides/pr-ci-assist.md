# PR CI assistance

On an open, same-repository PR, a repository writer can comment:

```text
@lightseek-bot watch
@lightseek-bot fix
@lightseek-bot fix <GitHub Actions job URL>
```

For a targeted repair, copy the failed job's URL from that PR's Actions run.
The optional `?pr=NUMBER` suffix is accepted. The job must belong to the PR's
current commit and latest run attempt, and must have failed or timed out.
The rest of its workflow may still be running. Links to other repositories,
PRs, or older commits cannot authorize a repair.

The repair agent receives the requested job's metadata and log alongside the
source and existing CI plan. It diagnoses the failure and proposes the source
change; the controller does not encode fixes for individual error types.
For explicit job requests, Python tests already changed by the PR can also be
repaired, while existing assertion checks still apply. Workflow and CI control
files remain protected.

The requested GPU task and exact runner are added to the existing validation
plan. Supported direct runners use the AMD, B200v2, GB200, or B300 K8s pools.
Scheduler tests, NVIDIA native library tests, and lint use their existing
verification entry points. Jobs without a supported validation task stop for
manual intervention.

Repairs are checked and tested on a candidate commit before updating the PR.
Passing another backend does not satisfy the requested task. Required PR CI
still applies after the update. Status comments are created at the start and
at completion or a blocker; intermediate progress updates the same comment.

If queueing exhausts the one-hour repair budget, let the candidate checks finish.
Then run **PR CI Assist** manually with the PR number and `repair_run` left empty.
This gives reconciliation a 15-minute window to verify the existing results and
publish the same candidate. It starts no new repair or GPU validation, and still
requires the original command, PR source, and validation branch to match.
The candidate keeps its original repair plan; later plan comments do not replace
its selected checks or start additional tasks.

A new main commit does not by itself invalidate a repair. Assistance reuses the
candidate when main advances without touching the PR or repair files or the
inputs of its selected checks. Documentation and independent components or
backends follow the existing CI path rules. Shared CI controls, build and
dependency inputs, unknown paths, rewritten history, and merge conflicts still
require fresh validation. Changes to the repair controller and its own tests do
not invalidate GPU results unless the PR or a selected native check covers them.
The original validation base and result source remain
unchanged; promotion checks compatibility again and publishes the same repair.
Explicit reconciliation also accepts a retained candidate previously marked stale
when these compatibility checks pass.
