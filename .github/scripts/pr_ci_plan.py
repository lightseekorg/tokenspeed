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

"""Provide existing validation targets and validate a focused CI recommendation."""

import json
import os
import subprocess
import sys
from pathlib import Path


def task_key(task: dict) -> str:
    return f"{task['config']}@{task['runner']}"


def context(source: Path, head: str, base: str) -> dict:
    # Reuse task validation and target discovery, including manual-only tasks.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/ci_system"))
    from pipeline import (
        find_task_files,
        normalize_task,
        resolve_runner_labels,
        summarize_task_targets,
    )

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
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    tests = [
        p
        for p in tracked
        if {"test", "tests"}.intersection(Path(p).parts[:-1])
        and (Path(p).name.startswith("test_") or Path(p).name.endswith("_test.py"))
        and p.endswith(".py")
    ]
    tasks = []
    for path in find_task_files(source / "test/ci"):
        task = normalize_task(path, source)
        tasks.append(
            {
                "config": task["_source_path"],
                "name": task["name"],
                "type": task["type"],
                "runners": resolve_runner_labels(task["runner"]["labels"]),
                "triggers": task["triggers"],
                "targets": summarize_task_targets(task, source),
                "server_command": task.get("server", {}).get("command", ""),
            }
        )
    return {
        "version": 1,
        "repository": os.environ["GITHUB_REPOSITORY"],
        "pr": int(os.environ["PR_NUMBER"]),
        "head": head,
        "base": base,
        "paths": paths,
        "test_files": tests,
        "catalog": tasks,
    }


def proposal(raw: str, data: dict) -> dict:
    # The CLI can prepend an evidence summary or wrap its final JSON in a fence.
    # Publish only the validated final object; the raw text is screened separately.
    lines = raw.strip().splitlines()
    if lines and lines[-1] == "```":
        lines.pop()
    for index, line in enumerate(lines):
        if line.startswith("{"):
            try:
                response = json.loads("\n".join(lines[index:]))
                break
            except json.JSONDecodeError:
                continue
    else:
        raise ValueError("Expected a final JSON object.")
    if not isinstance(response, dict) or set(response) != {
        "summary",
        "tests",
        "tasks",
        "conflicts",
    }:
        raise ValueError("Expected the CI proposal schema.")
    if not all(
        isinstance(response[k], str) and len(response[k]) <= 2000
        for k in ("summary", "conflicts")
    ):
        raise ValueError("Invalid proposal summary.")
    if not all(isinstance(response[k], list) for k in ("tests", "tasks")):
        raise ValueError("Invalid proposed task list.")
    tests = {}
    for choice in response["tests"]:
        if (
            not isinstance(choice, dict)
            or set(choice) != {"path", "reason"}
            or not all(isinstance(v, str) for v in choice.values())
            or choice["path"] not in data["test_files"]
            or not choice["reason"].strip()
            or len(choice["reason"]) > 1000
        ):
            raise ValueError("Proposed test must be an existing test file.")
        tests[choice["path"]] = choice
    catalog = {
        task_key({"config": t["config"], "runner": runner}): t
        for t in data["catalog"]
        for runner in t["runners"]
    }
    selected = {}
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
        selected[key] = {**catalog[key], **choice}
    return {
        "version": data["version"],
        "repository": data["repository"],
        "pr": data["pr"],
        "head": data["head"],
        "base": data["base"],
        "summary": response["summary"],
        "conflicts": response["conflicts"],
        "tests": list(tests.values()),
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
        "Prioritize the checks below; existing required CI and merge policy remain unchanged.",
    ]
    if plan["tests"]:
        lines += ["", "**Focused tests, in priority order**", ""]
        for test in plan["tests"]:
            lines.append(f"- `{test['path']}`: {test['reason']}")
    if plan["tasks"]:
        lines += ["", "**Existing CI tasks, in priority order**", ""]
        for task in plan["tasks"]:
            lines.append(
                f"- `{task['config']}` on `{task['runner']}`: {task['reason']}"
            )
    else:
        lines += [
            "",
            "No GPU CI task is prioritized for this change.",
        ]
    if plan["conflicts"]:
        lines += ["", "### Conflict assistance", "", plan["conflicts"]]
    return "\n".join(lines) + "\n"
