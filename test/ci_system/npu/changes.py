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

"""Select the NPU operator suite from a complete GitHub API change list."""

from __future__ import annotations

import argparse
import json
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
    if manifest.get("expected_plans") != "test/ci/npu/expected-plans.json":
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
        if operator.get("script") != f"test/ci/npu/tests/{name}-test.py":
            raise ValueError(f"{name}: invalid script path")
        for field in ("accuracy_cases", "performance_configs"):
            value = operator.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}: {field} must be a positive integer")
    return manifest


def select_suite(manifest: dict, event_name: str, paths: set[str]) -> dict:
    if event_name not in ("push", "pull_request", "workflow_dispatch"):
        raise ValueError(f"Unsupported event: {event_name}")
    matched = sorted(
        path
        for path in paths
        if not path.lower().endswith((".md", ".rst"))
        and any(fnmatchcase(path, pattern) for pattern in manifest["watch_paths"])
    )
    manual = event_name == "workflow_dispatch"
    run = manual or bool(matched)
    return {
        "should_run": run,
        "reason": (
            "manual"
            if manual
            else ("watched-path-changed" if run else "no-watched-changes")
        ),
        "changed_files": len(paths),
        "matched_files": len(matched),
        "matched_paths_sample": matched[:20],
    }


def github_json(endpoint, *, paginate):
    command = ["gh", "api", endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120
    )
    return json.loads(result.stdout)


def changed_paths(repository, event_name, event, api=github_json):
    """Reject partial API responses rather than treating missing paths as unrelated."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    if event_name == "workflow_dispatch":
        return set()
    if event_name == "pull_request":
        pr = event["pull_request"]
        number = event["number"]
        if type(number) is not int or number <= 0:
            raise ValueError("Invalid pull request number")
        current = api(f"/repos/{repository}/pulls/{number}", paginate=False)
        if any(current[ref]["sha"] != pr[ref]["sha"] for ref in ("head", "base")):
            raise ValueError("Pull request identity changed since this event")
        count = current["changed_files"]
        if type(count) is not int or not 0 <= count <= 3000:
            raise ValueError(
                "Pull request file list exceeds the API completeness limit"
            )
        pages = api(
            f"/repos/{repository}/pulls/{number}/files?per_page=100", paginate=True
        )
        if not isinstance(pages, list) or not all(
            isinstance(page, list) for page in pages
        ):
            raise ValueError("Invalid paginated pull request files")
        files = [item for page in pages for item in page]
        after = api(f"/repos/{repository}/pulls/{number}", paginate=False)
        if (
            any(after[ref]["sha"] != current[ref]["sha"] for ref in ("head", "base"))
            or after["changed_files"] != count
        ):
            raise ValueError("Pull request changed while reading files")
        if len(files) != count:
            raise ValueError("Incomplete pull request file list")
    elif event_name == "push":
        before, after = event["before"], event["after"]
        if any(
            not isinstance(sha, str)
            or not re.fullmatch(r"[0-9a-f]{40}", sha)
            or sha == "0" * 40
            for sha in (before, after)
        ):
            raise ValueError("Push requires nonzero before and after SHAs")
        comparison = api(
            f"/repos/{repository}/compare/{before}...{after}", paginate=False
        )
        files = comparison["files"]
        if len(files) >= 300:
            raise ValueError("Comparison may be truncated at 300 files")
    else:
        raise ValueError(f"Unsupported event: {event_name}")
    if not isinstance(files, list):
        raise ValueError("Invalid changed file list")
    paths, seen = set(), set()
    for item in files:
        filename = item["filename"]
        if filename in seen:
            raise ValueError("Duplicate changed file")
        seen.add(filename)
        for path in (filename, item.get("previous_filename")):
            if path is None:
                continue
            if (
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or ".." in Path(path).parts
            ):
                raise ValueError("Invalid changed path")
            paths.add(path)
    return paths


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
    parser.add_argument("--repository", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.repo_root / args.manifest)
        event = json.loads(args.event_path.read_text(encoding="utf-8"))
        if not isinstance(event, dict):
            raise ValueError("Event payload must be an object")
        paths = changed_paths(args.repository, args.event_name, event)
        result = select_suite(manifest, args.event_name, paths)
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
