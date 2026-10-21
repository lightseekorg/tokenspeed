# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Select the NPU operator suite from a complete local Git comparison."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
    ):
        raise ValueError("Expected manifest schema_version=3")
    if manifest["schema_version"] != 3:
        raise ValueError("Expected manifest schema_version=3")
    patterns = manifest.get("watch_paths")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("watch_paths must be a nonempty list")
    if any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError("watch_paths must contain nonempty strings")
    phases = ("accuracy", "performance")
    if manifest.get("expected_plans") != "ci/npu_ops_test/expected-plans.json":
        raise ValueError("Invalid expected_plans path")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("phases") != list(phases):
        raise ValueError("execution.phases must be accuracy, performance in that order")
    if execution.get("fail_fast") is not True:
        raise ValueError("execution.fail_fast must be true")
    if manifest.get("performance_policy") != {"mode": "record_only"}:
        raise ValueError("Only record_only performance policy is currently supported")
    operators = manifest.get("operators")
    if not isinstance(operators, list) or not operators:
        raise ValueError("operators must be a nonempty list")
    names = set()
    for operator in operators:
        if not isinstance(operator, dict):
            raise ValueError("Each operator must be an object")
        name = operator.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid operator name")
        if name in names:
            raise ValueError(f"Duplicate operator: {name}")
        names.add(name)
        if "entry" in operator or "scripts" in operator:
            raise ValueError(f"{name}: legacy entry/scripts are not supported")
        if operator.get("script") != f"ci/npu_ops_test/tests/{name}-test.py":
            raise ValueError(f"{name}: invalid script path")
        for field in ("accuracy_cases", "performance_configs"):
            value = operator.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}: {field} must be a positive integer")
    return manifest


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, timeout=60
    ).stdout


def validate_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"Invalid commit SHA: {value!r}")
    return value


def validate_ref(repo, value):
    if not isinstance(value, str) or not value.startswith("refs/"):
        raise ValueError(f"Invalid ref: {value!r}")
    git(repo, "check-ref-format", value)
    return value


def require_commit(repo, sha):
    validate_sha(sha)
    git(repo, "cat-file", "-e", f"{sha}^{{commit}}")


def select_suite(
    repo: Path,
    manifest: dict,
    event_name: str,
    event: dict,
    checkout_sha: str,
    checkout_ref: str,
) -> dict:
    validate_sha(checkout_sha)
    validate_ref(repo, checkout_ref)
    require_commit(repo, checkout_sha)
    if git(repo, "rev-parse", "HEAD").decode().strip() != checkout_sha:
        raise ValueError("Checkout HEAD does not match the event SHA")
    changed, matched = None, []
    comparison = None
    if event_name == "workflow_dispatch":
        if not checkout_ref.startswith(("refs/heads/", "refs/tags/")):
            raise ValueError("Manual run requires a branch or tag ref")
        run, reason = True, "manual"
    elif event_name == "pull_request":
        pr = event["pull_request"]
        number = event["number"]
        if type(number) is not int or number <= 0:
            raise ValueError("Invalid pull request number")
        if checkout_ref != f"refs/pull/{number}/merge":
            raise ValueError("PR checkout must use its merge ref")
        base, head = validate_sha(pr["base"]["sha"]), validate_sha(pr["head"]["sha"])
        for side in ("base", "head"):
            validate_ref(repo, "refs/heads/" + pr[side]["ref"])
        for sha in (base, head):
            require_commit(repo, sha)
        parents = git(repo, "show", "-s", "--format=%P", checkout_sha).decode().split()
        if len(parents) != 2 or parents[1] != head:
            raise ValueError(
                "PR merge commit must have two parents and match event head"
            )
        # The event base can lag behind the merge parent as the target advances.
        # Accept only forward history and compare against the base actually merged.
        try:
            git(repo, "merge-base", "--is-ancestor", base, parents[0])
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 1:
                raise
            raise ValueError(
                "PR merge commit base parent must descend from event base"
            ) from exc
        merge_base = git(repo, "merge-base", parents[0], head).decode().strip()
        comparison = (validate_sha(merge_base), head)
    elif event_name == "push":
        ref = validate_ref(repo, event.get("ref"))
        before, after = validate_sha(event.get("before")), validate_sha(
            event.get("after")
        )
        if ref != checkout_ref:
            raise ValueError("Push ref differs from checkout ref")
        if event.get("deleted") is True or not ref.startswith("refs/heads/"):
            run, reason = False, (
                "branch-deleted" if event.get("deleted") else "not-a-branch"
            )
        else:
            if after != checkout_sha:
                raise ValueError("Push after SHA differs from checkout SHA")
            if before == "0" * 40:
                run, reason = True, "new-branch"
            else:
                require_commit(repo, before)
                comparison = (before, after)
    else:
        raise ValueError(f"Unsupported event: {event_name}")
    if comparison is not None:
        # No rename detection: both the deleted and added paths are checked.
        raw = git(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--name-only",
            "-z",
            *comparison,
            "--",
        )
        paths = [p for p in os.fsdecode(raw).split("\0") if p]
        changed = len(paths)
        matched = sorted(
            p
            for p in paths
            if not p.lower().endswith((".md", ".rst"))
            and any(fnmatchcase(p, pattern) for pattern in manifest["watch_paths"])
        )
        run = bool(matched)
        reason = "watched-path-changed" if run else "no-watched-changes"
    return {
        "should_run": run,
        "reason": reason,
        "changed_files": changed,
        "matched_files": len(matched),
        "matched_paths_sample": matched[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--event-name",
        choices=("push", "pull_request", "workflow_dispatch"),
        required=True,
    )
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--checkout-sha", required=True)
    parser.add_argument("--checkout-ref", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.repo_root / args.manifest)
        event = json.loads(args.event_path.read_text(encoding="utf-8"))
        if not isinstance(event, dict):
            raise ValueError("Event payload must be an object")
        result = select_suite(
            args.repo_root,
            manifest,
            args.event_name,
            event,
            args.checkout_sha,
            args.checkout_ref,
        )
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as output:
                output.write(f"should_run={str(result['should_run']).lower()}\n")
                output.write(f"reason={result['reason']}\n")
        print(json.dumps(result, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"NPU change scan failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
