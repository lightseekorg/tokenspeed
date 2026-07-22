---
name: release-tokenspeed-kernel-amd
description: Prepare, publish, complete, check, or recover a tokenspeed-kernel-amd release with the repository's three manual release workflows.
---

# Release tokenspeed-kernel-amd

Run the three workflows in order. Treat each merged pull request and the PyPI
publication as a gate; do not start the next stage early.

## Establish the current state

* Work in `lightseekorg/tokenspeed` and use the local `gh` CLI for GitHub
  writes. Before the first write, run the following commands and require the
  second one to print `lightseek-bot`:

  ```shell
  gh auth status
  gh api user --jq .login
  ```

* Fetch `origin/main` and use it as the source of truth. Do not infer release
  state from a stale local branch.
* Normalize the target to a public version without a leading `v`, for example
  `0.1.4`. The workflows accept numeric release segments, optional
  `aN`/`bN`/`rcN`, and optional `.postN`; they reject development and local
  versions. Compare versions as PEP 440 versions, never as strings.
* Read all three version fields on `origin/main`: the AMD `project.version`,
  the `tokenspeed-kernel-amd==...` ROCm requirement, and tokenspeed-kernel's
  `BASE_VERSION`. If AMD `project.version` is newer, stop rather than asking
  the version workflow to downgrade it. If the ROCm pin or `BASE_VERSION` is
  newer, stop because the dependency workflow will refuse that downstream
  downgrade. Remember that changing `BASE_VERSION` affects the shared
  tokenspeed-kernel version used by both AMD and non-AMD builds.
* Inspect all release checkpoints before dispatching anything:
  * the three version fields on `origin/main`;
  * existing version-update and dependency-update pull requests for the target;
  * `release/<version>` on the remote;
  * `https://pypi.org/pypi/tokenspeed-kernel-amd/<version>/json`, including
    every filename and its yanked state;
  * wheelhouse tag `tokenspeed-kernel-amd-v<version>`, including its source
    note, prerelease setting, and assets.
* Resume at the first incomplete gate. Classify PyPI before doing so:
  * HTTP 404 means the version may be published after the earlier gates pass.
  * HTTP 200 consumes the version permanently. If every intended distribution
    is present and non-yanked, never republish; continue with dependency sync if
    needed.
  * HTTP 200 with missing intended files or any yanked file requires a
    maintainer decision and normally a new version. Never retry the upload.
  * Any other response or an in-flight release run is inconclusive; stop and
    recheck instead of starting a concurrent publication.
* Treat workflow-generated pull requests like any other change: require normal
  checks and review, and do not merge unless the caller authorized it.

Use these variables in the examples:

```shell
REPO=lightseekorg/tokenspeed
VERSION=0.1.4
```

For example, list recent release runs with:

```shell
gh run list --repo "$REPO" \
  --workflow release-tokenspeed-kernel-amd.yml \
  --event workflow_dispatch
```

After each dispatch, identify the newly created run and watch it:

```shell
RUN_ID=123456789
gh run watch "$RUN_ID" --repo "$REPO" --exit-status
```

Do not mistake an older run for the new dispatch, and never overlap release
attempts for the same version.

## 1. Update the AMD package version

Dispatch the version workflow from `main`:

```shell
gh workflow run update-tokenspeed-kernel-amd-version.yml \
  --repo "$REPO" \
  --ref main \
  --field version="$VERSION"
```

The workflow either reports a no-op or creates/updates
`bot/update-tokenspeed-kernel-amd-version-<version>`. Review its pull request and
confirm that it changes only `tokenspeed-kernel-amd/pyproject.toml`.

Do not continue until that pull request is merged and `origin/main` contains the
target version. If the workflow reports a no-op, verify the version on
`origin/main` before advancing.

## 2. Build and publish the AMD package

Use exactly one release-branch mode. Keep `publish_pypi=true` for a real
release. `publish_github` controls the optional `lightseekorg/whl` release, and
`prerelease` applies to that release; leave both false unless explicitly
requested.

### Let the workflow create the branch

Use this mode only when release-branch creation is explicitly requested. The
input defaults to false.

```shell
gh workflow run release-tokenspeed-kernel-amd.yml \
  --repo "$REPO" \
  --ref main \
  --field create_release_branch=true \
  --field publish_pypi=true \
  --field publish_github=false \
  --field prerelease=false
```

The main-branch run creates immutable `release/<version>` and dispatches a
second run on it. A green parent run means only that the child was dispatched.
Find the child by filtering recent runs to branch `release/<version>`, then
match its creation time and head SHA to the parent output and immutable branch.
Watch it through its `publish-pypi` job.

The parent refuses a wheelhouse tag that already has assets even when
`publish_github=false`. In that state, inspect the existing release source and
use the existing-branch mode; do not keep retrying automatic branch creation.

### Use an existing release branch

First verify that `release/<version>` contains the target AMD version and that
its tip is part of `main` history. If a release branch must be created manually,
create it from the reviewed `origin/main` commit without force-pushing. Then
dispatch:

```shell
gh workflow run release-tokenspeed-kernel-amd.yml \
  --repo "$REPO" \
  --ref "release/$VERSION" \
  --field create_release_branch=false \
  --field publish_pypi=true \
  --field publish_github=false \
  --field prerelease=false
```

Do not continue until every intended distribution is present and non-yanked on
PyPI. A successful run is the normal evidence, but PyPI is decisive after a
failed publish job because an upload may have partially succeeded.

## 3. Synchronize tokenspeed-kernel

Dispatch the dependency workflow from `main` only after publication:

```shell
gh workflow run update-tokenspeed-kernel-amd-dependency.yml \
  --repo "$REPO" \
  --ref main \
  --field amd_version="$VERSION"
```

The workflow verifies the version on `main`, waits for a non-yanked PyPI
distribution, rejects downgrades, and then either reports a no-op or
creates/updates `bot/update-tokenspeed-kernel-amd-dependency-<version>`.

Review its pull request and allow no files beyond these; one file may remain
unchanged if it already has the target value:

* pin `tokenspeed-kernel-amd==<version>` in
  `tokenspeed-kernel/python/requirements/rocm.txt`;
* set `BASE_VERSION = "<version>"` in
  `tokenspeed-kernel/python/setup.py`.

Verify both final values, then merge this pull request to complete the
lifecycle. If the workflow reports a no-op, verify both fields on
`origin/main`.

## Recover safely

* Inspect the existing run, pull request, release branch, PyPI state, and
  wheelhouse release before retrying. Deterministic bot branches may already
  contain review changes. Redispatching a PR workflow rebuilds from current
  `main` and force-updates its bot branch, so do not blindly replace those
  changes.
* If branch creation succeeded but dispatch or publication failed, dispatch the
  existing `release/<version>` with `create_release_branch=false`. Never ask the
  workflow to create the same branch again as a recovery strategy.
* Inspect failed logs and ensure no run is queued or in progress before choosing
  a retry. For a transient build failure before any publication, rerun failed
  jobs or redispatch the existing release branch. Recheck PyPI after every
  attempt and allow time for index propagation. Do not loop retries: diagnose
  and reassess the complete state before every additional attempt.
* A PyPI HTTP 200 blocks every later `publish_pypi=true` attempt, even when the
  original job failed or all files are yanked. Compare its filenames with the
  `tokenspeed-kernel-amd-dist` artifact, which contains the built sdist and
  wheel, and compare SHA-256 digests after any ambiguous upload. If the artifact
  is unavailable or any file is missing, yanked, or mismatched, stop for a
  maintainer decision.
* After an upload step starts, do not treat a single PyPI 404 as proof that no
  file was accepted. Wait for the run to become terminal and recheck; retry only
  when the logs prove publication never began. Otherwise stop for a maintainer
  decision instead of choosing an arbitrary timeout.
* If a requested wheelhouse publication partially succeeds before PyPI runs,
  verify its source and prerelease setting, then redispatch the existing branch
  with both `publish_github=true` and `publish_pypi=true`. The workflow resumes
  missing wheelhouse assets before publishing to PyPI.
* If only the optional wheelhouse publication remains, dispatch the existing
  release branch with `publish_pypi=false` and `publish_github=true`. Resume a
  partial wheelhouse release only when its source note and prerelease setting
  match; the workflow will preserve existing assets and upload missing ones.
  Stop on a foreign source or different prerelease setting.
* Never delete, force-push, or move `release/<version>`. If it points at the
  wrong commit, stop and obtain a maintainer decision; normally prepare a new
  patch or post-release version.
* If immutable source or workflow code fails deterministically, do not patch the
  release branch. Obtain a maintainer decision and normally prepare a new
  version from fixed `main`.
* Never overwrite a published version. Resolve a bad release with a new version
  and handle any yank separately with explicit maintainer approval.

## Report completion

Provide each workflow-generated pull request URL or explicit no-op evidence,
the release child run URL, the parent run URL when it created the branch, the
immutable release branch and commit, PyPI version URL, and final values on
`origin/main`. Clearly identify any optional wheelhouse publication that was
not requested.
