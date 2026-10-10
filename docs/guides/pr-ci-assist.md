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
still applies after the update. A status comment is created when assistance
first engages and again at each completion or blocker — the only events that
notify subscribers. Progress, new pushes and new commands update the latest
comment in place, and terminal outcome comments are never edited.

Each repair and validation attempt has a one-hour budget. The overrun is
recognized the next time the workflow runs — on a completion event or a manual
trigger — and stops the attempt for manual intervention. Running **PR CI
Assist** manually with the PR number retries from the current state: with a
retained candidate it gets a 15-minute window to verify the existing results
and publish the same repair, starting no new repair or GPU validation; without
a candidate it starts a fresh repair. The candidate keeps its original repair
plan; later plan comments do not replace its selected checks or start
additional tasks.

A new main commit does not by itself invalidate a repair. The candidate stays
valid while a trial merge against the current main is clean; a conflict stops
assistance for manual intervention and requires a retry. Promotion repeats the
trial merge immediately before publishing the same repair. Required PR CI
still applies after the update.
