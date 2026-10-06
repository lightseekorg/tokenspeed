# Biweekly releases

The **Biweekly release** workflow starts every two weeks on Sunday at 20:00 in
`America/Los_Angeles`, including daylight saving changes. The weekly cron checks
the calendar-date distance from Sunday, January 2, 2000; alternate weeks finish
successfully without publishing. It can also be started
manually from `main`, with an optional stable TokenSpeed version. Otherwise,
the greater of the version on `main` and the published version advances by one
patch. A manually specified version must be greater than both. This ensures
the final version PR changes `version.py` and triggers its PyPI workflow.

The workflow completes these stages in order:

1. Lock the latest `main` commit and require its CI to pass before reserving
   versions. Wait for running jobs and retry failed jobs once. A failed second
   attempt stops the release, including retries already requested by another
   workflow. For path-filtered CI, require the push run for the latest first-parent
   commit affecting that workflow's declared inputs. An older success is reused
   only when those inputs are unchanged; a documentation-only head cannot hide
   an earlier failed GPU test. Disabled workflows and package publishers are
   excluded. Missing source runs, including commits inside a multi-commit push,
   require manual intervention. A change to `main` stops this release.
   Check that the latest stable MLA and scheduler releases contain all current
   changes in their component directories, using PyPI's published source
   provenance and Git history. Their versions on `main` and the MLA pin in the
   kernel and scheduler requirement in the runtime must match those releases.
   The two TokenSpeed version declarations must agree.
2. Create a version PR for `tokenspeed-kernel-amd`, verify its exact metadata diff, merge it,
   and publish its immutable `release/<version>` source to PyPI and the wheelhouse.
3. Update the kernel's AMD dependency and version in one PR, then publish CUDA
   12.9/13.0 variant wheels and ROCm 7.2 wheels. CUDA 13.0 supplies PyPI; all
   variants go to the wheelhouse.
4. Update TokenSpeed's kernel requirement and both version declarations in one
   PR. Keep the kernel's `.dev0` floor so CI retains matching in-tree builds.
   Wait for that exact merge's existing PyPI workflow, then publish its
   identical distributions to the wheelhouse.
5. Update the stable CUDA and ROCm pip indexes, preserving previous releases.
6. Publish the existing NVIDIA Docker image for `linux/amd64` and `linux/arm64`.
   The image installs and checks the exact released kernel, scheduler and MLA
   versions. Confirm both platforms are in the published manifest.
7. Create `v<version>` at the TokenSpeed release commit with generated release
   notes, a component version table and links to PyPI, wheelhouse releases,
   stable pip indexes, Docker and the publication runs. Notes use the previous
   stable release tag and stay below the page size limit; the full changelog
   link is always retained.
8. Verify the published sources and release page, then delete this run's three
   release branches with their expected commit leases. Tags, published assets
   and unrelated branches remain available. Cleanup is safe to retry; a moved
   branch stops cleanup for inspection.

Configure `LIGHTSEEK_BOT_TOKEN` for the `lightseek-bot` account with repository
and workflow access, and the existing `DOCKERHUB_USERNAME` variable and
`DOCKERHUB_TOKEN` secret. Component workflows continue to use the `pypi`
environment and their existing PyPI trusted publishers. Environment approvals
still apply. Generated version PRs run full non-C++ pre-commit before committing,
but do not wait for their repeated lint or GPU CI. The controller replays the
expected version and dependency-pin updates and compares the complete diff,
rejecting other changes even inside a metadata file. Each PR must contain one
commit whose parent is the validated `main` or the preceding release commit.
Every effective ruleset must explicitly allow the bot to bypass its requirements.
The controller fast-forwards that exact PR commit to `main` with an expected-SHA
lease, so a concurrent source commit prevents the merge without rewriting history.
It then waits for GitHub to report the PR merged at that same commit. Requested
changes, merge conflicts and missing bypass permission require manual intervention.
The workflow never changes repository rules. Publication builds and verification
remain mandatory for every released package and Docker image.

Source provenance checks read the repository and source claims that PyPI serves,
including the artifact digest. They do not perform independent signature
verification. Missing or conflicting provenance stops publication.

## Recovery

Any failed main CI retry, publication or wait stops downstream stages and fails the
weekly run. The summary and `weekly-state-<stage>` artifacts retain the reserved
versions, PRs, source commits and child run IDs. Each stage allows up to 340
minutes for checks, approvals, queueing and publication before requiring manual
intervention. No release page is created for an incomplete run.

Fix the first failed stage. If a child workflow failed, inspect and repair it,
then rerun the appropriate child jobs before rerunning **failed jobs** in the
weekly workflow. The controller reuses the same versions, immutable refs and
child runs. It never automatically retries a failed publisher or overwrites a
published PyPI version. Do not start a new weekly run to resume an interrupted
release. An ambiguous dispatch, conflicting source, edited version PR or expired
recovery artifact requires inspection rather than guessing a new version.

If a controller fix is needed after all publications succeeded, merge the fix
and start **Biweekly release** from `main` with **resume_run_id** set to the
original run ID and **version** empty. This uses the corrected controller and
runs only release notes and branch cleanup. Version planning, version PRs,
package uploads, index publication and Docker builds are skipped. The source
run must be completed, belong to this workflow on `main`, and have successful
stages through Docker. Its saved state must match the original run ID, and all
recorded publishers must still match their exact source and successful result.
Recovery and normal releases use the same concurrency group.

**Re-run failed jobs** uses the controller from the original run's commit. Use
that for repaired child publications; use **resume_run_id** when the controller
itself changed. Recovery state artifacts are retained for 90 days.

Stable pip indexes use ordinary pushes and reapply their changes on the latest
branch after a rejected push, preserving concurrent nightly updates. Three
rejected attempts stop the stage for manual recovery. Docker currently follows
the existing NVIDIA release workflow; AMD kernel wheels are published in the
ROCm index.
