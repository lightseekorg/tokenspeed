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

"""Build and validate a source-bound CI coverage proposal."""

import json
import os
import subprocess
import sys
from pathlib import Path


def task_key(task: dict) -> str:
    return f"{task['config']}@{task['runner']}"


def context(source: Path, head: str, base: str) -> dict:
    # Only the established task loader and path classifier determine the floor.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/ci_system"))
    import yaml
    from ci_path_filter import (
        RUNNER_GROUPS,
        VENDOR_WORKFLOWS,
        path_requires_group,
        path_vendor,
        task_runner_labels,
    )
    from pipeline import KERNEL_BENCHMARK_SUITE_DIRECTORY, build_matrix

    paths = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--name-only",
            "--no-renames",
            f"{base}...{head}",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tasks = {}
    floor = set()
    broad = []
    policy_changes = []
    for path in paths:
        if not path.startswith("test/ci/") or not path.endswith(".yaml"):
            continue
        before = subprocess.run(
            ["git", "show", f"{base}:{path}"],
            cwd=source,
            capture_output=True,
            text=True,
        )
        old = yaml.safe_load(before.stdout) if before.returncode == 0 else {}
        new = (
            yaml.safe_load((source / path).read_text())
            if (source / path).is_file()
            else {}
        )
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise ValueError("Invalid task policy metadata.")
        if any(
            old.get(key) != new.get(key)
            for key in (
                "optional",
                "triggers",
                "runner",
                "workflow_stage",
                "type",
                "slurm",
                "score_threshold",
                "perf_threshold",
                "perf_reference",
                "retries",
            )
        ):
            policy_changes.append(path)
    for group in RUNNER_GROUPS:
        if (
            group == "nvidia-gb300-slurm"
            and os.environ.get("TOKENSPEED_CI_GB300_SLURM_PER_COMMIT_ENABLED") != "true"
        ):
            continue
        affected = [p for p in paths if path_requires_group(p, group, source)]
        if not affected:
            continue
        # Vendor-owned sources and task declarations are the only narrowable
        # classes. Shared or unclassified affected paths retain the full suite.
        full = any(
            path_vendor(p, source) is None and task_runner_labels(p, source) is None
            for p in affected
        )
        if full:
            broad.append(group)
        saved = os.environ.get("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", "")
        if group == "nvidia-x86":
            os.environ["TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS"] = f"h100,b300,{saved}"
        elif group.endswith("-slurm"):
            os.environ["TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS"] = ""
        try:
            entries = build_matrix(
                source / "test/ci",
                source,
                "per-commit",
                "nvidia-arm" if group.endswith("-slurm") else group,
                workflow_stage="model-test" if group == "nvidia-gb300-slurm" else None,
                multi_node="only" if group == "nvidia-gb300-slurm" else "exclude",
            )["include"]
        finally:
            os.environ["TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS"] = saved
        # Match pipeline scan's existing changed-task/suite-only selection.
        suite_changed = any(
            p.startswith(KERNEL_BENCHMARK_SUITE_DIRECTORY) for p in paths
        )
        if 0 < len(paths) < 300 and all(
            (p.startswith("test/ci/") and p.endswith(".yaml"))
            or p.startswith(KERNEL_BENCHMARK_SUITE_DIRECTORY)
            for p in paths
        ):
            entries = [
                t
                for t in entries
                if t["config"] in paths
                or (suite_changed and t.get("workflow_stage") == "kernel-benchmark")
            ]
        if group in {"nvidia-arm", "nvidia-x86"}:
            entries = [
                t
                for t in entries
                if t.get("workflow_stage") in {"unit-test", "model-test"}
            ]
        if group == "nvidia-arm":
            entries = [t for t in entries if not t["runner"].startswith("slurm-")]
        elif group == "nvidia-gb200-slurm":
            entries = [t for t in entries if t["runner"].startswith("slurm-gb200-")]
        for task in entries:
            # Optional checks remain optional, including on a broad diff.
            if task["optional"]:
                continue
            key = task_key(task)
            task["workflow"] = Path(VENDOR_WORKFLOWS[group]).name
            tasks[key] = task
            if (
                full
                or task.get("workflow_stage") == "unit-test"
                or task["config"] in paths
            ):
                floor.add(key)
    return {
        "version": 1,
        "repository": os.environ["GITHUB_REPOSITORY"],
        "pr": int(os.environ["PR_NUMBER"]),
        "head": head,
        "base": base,
        "paths": paths,
        "broad_groups": broad,
        "policy_changes": policy_changes,
        "catalog": list(tasks.values()),
        "floor": sorted(floor),
    }


def proposal(raw: str, data: dict) -> dict:
    response = json.loads(raw)
    if not isinstance(response, dict) or set(response) != {
        "summary",
        "tasks",
        "conflicts",
    }:
        raise ValueError("Expected the CI proposal schema.")
    if not all(
        isinstance(response[k], str) and len(response[k]) <= 2000
        for k in ("summary", "conflicts")
    ):
        raise ValueError("Invalid proposal summary.")
    if not isinstance(response["tasks"], list):
        raise ValueError("Invalid proposed task list.")
    catalog = {task_key(t): t for t in data["catalog"]}
    selected = {
        key: {**catalog[key], "reason": "Required by the deterministic coverage floor."}
        for key in data["floor"]
    }
    for choice in response["tasks"]:
        if not isinstance(choice, dict) or set(choice) != {
            "config",
            "runner",
            "reason",
        }:
            raise ValueError("Invalid proposed task.")
        if not all(isinstance(value, str) for value in choice.values()):
            raise ValueError("Invalid task values.")
        key = task_key(choice)
        if (
            key not in catalog
            or not choice["reason"].strip()
            or len(choice["reason"]) > 1000
        ):
            raise ValueError("Proposed task must belong to the coverage catalog.")
        selected[key] = {**catalog[key], "reason": choice["reason"]}
    return {
        "version": data["version"],
        "repository": data["repository"],
        "pr": data["pr"],
        "head": data["head"],
        "base": data["base"],
        "summary": response["summary"],
        "conflicts": response["conflicts"],
        "broad_groups": data["broad_groups"],
        "policy_changes": data.get("policy_changes", []),
        "tasks": list(selected.values()),
    }


def render(plan: dict) -> str:
    lines = [
        f"Reviewed commit: `{plan['head']}`",
        "",
        "### CI plan",
        "",
        plan["summary"],
        "",
        "CPU checks and review approval remain required. This coverage proposal needs explicit acceptance before any merge bypass.",
    ]
    if plan["broad_groups"]:
        lines += ["", "Shared changes retain the full affected test baseline."]
    if plan["policy_changes"]:
        lines += [
            "",
            "Task scheduling or acceptance metadata changed. This proposal cannot establish merge readiness until those policy changes are reviewed.",
        ]
        lines += [f"- `{path}`" for path in plan["policy_changes"]]
    if plan["tasks"]:
        lines += ["", "<details>", "<summary>Selected GPU tasks</summary>", ""]
        for task in plan["tasks"]:
            lines.append(
                f"- `{task['config']}` on `{task['runner']}`: {task['reason']}"
            )
        lines += ["", "</details>"]
    else:
        lines += [
            "",
            "The existing path classifier requires no GPU tasks for this diff.",
        ]
    if plan["conflicts"]:
        lines += ["", "### Conflict assistance", "", plan["conflicts"]]
    lines += [
        "",
        "Failed selected tasks can be diagnosed with the single-task dispatch workflow at this exact commit. A retry pass is a possible flake; validation on different hardware is diagnostic evidence.",
    ]
    return "\n".join(lines) + "\n"
