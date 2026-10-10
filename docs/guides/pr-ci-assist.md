# PR CI assistance

On an open, same-repository PR, a repository writer can comment:

```text
@lightseek-bot watch
@lightseek-bot fix
@lightseek-bot fix <GitHub Actions job URL>
@lightseek-bot rerun <GitHub Actions job URL>
```

For a targeted repair or rerun, copy the failed job's URL from that PR's
Actions run. The bot accepts the optional `?pr=NUMBER` suffix. The job must
belong to the PR's current commit and latest run attempt, and must have failed
or timed out. The rest of its workflow may still be running. Links to other
repositories, PRs, or older commits cannot authorize a repair or rerun.

A rerun re-dispatches the failed job's validation task through Slurm or K8s
dispatch and reports the outcome: a pass closes the request; a failure starts
the repair flow as if `fix` had been requested for that job. Reruns of native
checks, which have no independent dispatch channel, start the repair flow
directly.

The repair agent receives the requested job's metadata and log alongside the
source and existing CI plan. It diagnoses the failure and proposes the source
change. The controller does not encode fixes for individual error types.
For explicit job requests, the repair agent can also repair Python tests
already changed by the PR, while existing assertion checks still apply.
Workflow and CI control files remain protected.

The controller adds the requested GPU task and exact runner to the existing
validation plan. Supported direct runners use the AMD, B200v2, GB200, or
B300 K8s pools. Scheduler tests, NVIDIA native library tests, and lint use
their existing verification entry points. Jobs without a supported
validation task stop for manual intervention.

The controller checks and tests repairs on a candidate commit before
updating the PR. Passing another backend does not satisfy the requested
task. Required PR CI still applies after the update. The bot creates a
status comment when assistance first engages and again at each completion
or blocker — the only events that notify subscribers. Progress, new pushes
and new commands update the latest comment in place. The bot never edits
terminal outcome comments.

Each repair and validation attempt has a one-hour budget. The workflow
recognizes the overrun the next time it runs — on a completion event, a
manual trigger, or the twenty-minute sweep — and stops the attempt for
manual intervention. Running **PR CI Assist** manually with the PR number
retries from the current state. With a retained candidate, it gets a
15-minute window to verify the existing results and publish the same
repair, starting no new repair or GPU validation. Without a candidate, it
starts a fresh repair. The candidate keeps its original repair plan. Later
plan comments do not replace its selected checks or start additional
tasks.

A new main commit does not by itself invalidate a repair. The candidate stays
valid while a trial merge against the current main is clean. A conflict stops
assistance for manual intervention and requires a retry. Promotion repeats the
trial merge immediately before publishing the same repair. Required PR CI
still applies after the update.
