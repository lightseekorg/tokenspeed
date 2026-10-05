# Weekly releases

The **Weekly release** workflow starts every Sunday at 20:00 in
`America/Los_Angeles`, including daylight saving changes. It can also be started
manually from `main`, with an optional stable TokenSpeed version. Otherwise,
the greater of the version on `main` and the published version advances by one
patch. A manually specified version must be greater than both. This ensures
the final version PR changes `version.py` and triggers its PyPI workflow.

The workflow completes these stages in order:

1. Check that the latest stable MLA and scheduler releases contain all current
   changes in their component directories, using PyPI's published source
   provenance and Git history. Their versions on `main` and the MLA pin in the
   kernel and scheduler requirement in the runtime must match those releases.
   The two TokenSpeed version declarations must agree.
2. Create a version PR for `tokenspeed-kernel-amd`, wait for checks, merge it,
   and publish its immutable `release/<version>` source to PyPI and the wheelhouse.
3. Update the kernel's AMD dependency and version in one PR, then publish CUDA
   12.9/13.0 variant wheels and ROCm 7.2 wheels. CUDA 13.0 supplies PyPI; all
   variants go to the wheelhouse.
4. Update TokenSpeed's kernel requirement and both version declarations in one
   PR. Wait for that exact merge's existing PyPI workflow, then publish its
   identical distributions to the wheelhouse.
5. Update the stable CUDA and ROCm pip indexes, preserving previous releases.
6. Publish the existing NVIDIA Docker image for `linux/amd64` and `linux/arm64`.
   The image installs and checks the exact released kernel, scheduler and MLA
   versions. Confirm both platforms are in the published manifest.
7. Create `v<version>` at the TokenSpeed release commit with generated release
   notes, a component version table and links to the publication runs.

Configure `LIGHTSEEK_BOT_TOKEN` for the `lightseek-bot` account with repository
and workflow access, and the existing `DOCKERHUB_USERNAME` variable and
`DOCKERHUB_TOKEN` secret. Component workflows continue to use the `pypi`
environment and their existing PyPI trusted publishers. Environment approvals
and branch rules still apply. Version PRs require successful lint and completed
checks; the controller does not bypass protection or use administrator merges.

Source provenance checks read the repository and source claims that PyPI serves,
including the artifact digest. They do not perform independent signature
verification. Missing or conflicting provenance stops publication.

## Recovery

Any failed check, publication or wait stops downstream stages and fails the
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

Stable pip indexes use ordinary pushes and reapply their changes on the latest
branch after a rejected push, preserving concurrent nightly updates. Three
rejected attempts stop the stage for manual recovery. Docker currently follows the existing NVIDIA release
workflow; AMD kernel wheels are published in the ROCm index.
