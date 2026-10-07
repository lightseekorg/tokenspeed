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
import re
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.parse import quote

import yaml
from pr_ci_state import NATIVE_CHECKS


def _matches_path(path: str, patterns: list[str]) -> bool:
    matched = False
    # GitHub applies exclusions and subsequent inclusions in declaration order.
    for pattern in patterns:
        if fnmatchcase(path, pattern.removeprefix("!")):
            matched = not pattern.startswith("!")
    return matched


def native_checks(paths: list[str]) -> list[dict]:
    """Match the existing trusted workflows' PR path filters."""
    workflows = Path(__file__).resolve().parents[1] / "workflows"
    checks = []
    for workflow, details in NATIVE_CHECKS.items():
        config = yaml.safe_load(workflows.joinpath(workflow).read_text())
        # PyYAML's YAML 1.1 loader interprets the unquoted `on` key as True.
        trigger = config.get("on", config.get(True))["pull_request"]
        if "main" in trigger["branches"] and any(
            _matches_path(path, trigger["paths"]) for path in paths
        ):
            checks.append({"workflow": workflow, **details})
    return checks


def source_url(data: dict, path: str = "") -> str:
    root = f"https://github.com/{data['repository']}"
    if path:
        return f"{root}/blob/{data['head']}/{quote(path, safe='/')}"
    return f"{root}/commit/{data['head']}"


def _short_text(value: object, limit: int) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= limit
        and not re.search(r"(?<!\w)@\w", value)
    )


def _cell(value: str) -> str:
    # Keep model text literal inside a table cell or a generated link label.
    return re.sub(r"([\\`*\[\]|<>&_])", r"\\\1", " ".join(value.split()))


def task_key(task: dict) -> str:
    return f"{task['config']}@{task['runner']}@{task['cluster']}"


class CoverageError(ValueError):
    """Required test or serving coverage was omitted from the plan."""


def validate_test_coverage(
    tests: list[str], tasks: list[dict], catalog: list[dict], paths: list[str]
):
    selected = {task["config"] for task in tasks}
    mapped, covered = set(), set()
    for task in catalog:
        files = task.get("targets", {}).get("test_files", [])
        mapped.update(files)
        if task["config"] in selected:
            covered.update(files)
    missing = sorted(set(tests).intersection(mapped) - covered)
    if missing:
        raise CoverageError(
            f"Select a CI task covering each recommended test: {', '.join(missing)}"
        )
    if any(
        path.startswith("tokenspeed-mla/python/tokenspeed_mla/")
        and path.endswith(".py")
        for path in paths
    ) and not any(
        task["config"] in selected
        and re.search(
            r"--(?:drafter-)?attention-backend(?:=|\s+)tokenspeed_mla(?:\s|$)",
            task.get("server_command", ""),
        )
        for task in catalog
    ):
        raise CoverageError(
            "Select a serving CI task explicitly using tokenspeed_mla for in-tree MLA changes."
        )


def context(source: Path, head: str, base: str) -> dict:
    # Reuse task validation and target discovery, including manual-only tasks.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/ci_system"))
    from pipeline import (
        GB200_RUNNER_PREFIXES,
        find_task_files,
        normalize_task,
        resolve_runner_labels,
        summarize_task_targets,
        validate_gb300_runner_alias,
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
        labels = task["runner"]["labels"]
        slurm = {"gb200": [], "gb300": []}
        for label in labels:
            # Slurm Dispatch accepts original B200 labels on GB200 too.
            if label.startswith(("b200-", *(p + "-" for p in GB200_RUNNER_PREFIXES))):
                slurm["gb200"].append(label)
            prefix = "slurm-" if label.startswith("slurm-") else ""
            suffix = label.removeprefix(prefix).split("-", 1)[-1]
            try:
                validate_gb300_runner_alias(label, f"{prefix}gb300-{suffix}")
            except ValueError:
                continue
            slurm["gb300"].append(label)
        tasks.append(
            {
                "config": task["_source_path"],
                "name": task["name"],
                "type": task["type"],
                "runners": resolve_runner_labels(labels),
                "slurm_runners": slurm,
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
        "native_checks": native_checks(paths),
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
    if (
        not _short_text(response["summary"], 200)
        or not isinstance(response["conflicts"], str)
        or len(response["conflicts"]) > 2000
        or re.search(r"(?<!\w)@\w", response["conflicts"])
    ):
        raise ValueError("Invalid proposal summary.")
    if not all(isinstance(response[k], list) for k in ("tests", "tasks")):
        raise ValueError("Invalid proposed task list.")
    tests = {}
    for choice in response["tests"]:
        if (
            not isinstance(choice, dict)
            or set(choice) != {"path", "label", "reason"}
            or not all(isinstance(v, str) for v in choice.values())
            or choice["path"] not in data["test_files"]
            or not _short_text(choice["label"], 60)
            or not _short_text(choice["reason"], 120)
        ):
            raise ValueError("Proposed test must be an existing test file.")
        tests[choice["path"]] = choice
    catalog = {
        task_key({"config": t["config"], "runner": runner, "cluster": cluster}): t
        for t in data["catalog"]
        for cluster, runners in {"": t["runners"], **t.get("slurm_runners", {})}.items()
        for runner in runners
    }
    selected = {}
    for choice in response["tasks"]:
        if not isinstance(choice, dict) or set(choice) != {
            "config",
            "runner",
            "cluster",
            "label",
            "reason",
        }:
            raise ValueError("Invalid proposed task.")
        if not all(isinstance(value, str) for value in choice.values()):
            raise ValueError("Invalid task values.")
        key = task_key(choice)
        if (
            key not in catalog
            or not _short_text(choice["label"], 60)
            or not _short_text(choice["reason"], 120)
        ):
            raise ValueError("Proposed task must belong to the coverage catalog.")
        selected[key] = {**catalog[key], **choice}
    validate_test_coverage(
        list(tests), list(selected.values()), data["catalog"], data.get("paths", [])
    )
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
        "native_checks": data.get("native_checks", []),
    }


def render(plan: dict) -> str:
    lines = [
        "### CI plan",
        "",
        _cell(plan["summary"]),
    ]
    rows = [
        (
            check,
            f".github/workflows/{check['workflow']}",
            check.get("target", "Native CPU CI"),
        )
        for check in plan.get("native_checks", [])
    ]
    rows += [(t, t["path"], "Targeted UT") for t in plan["tests"]]
    for task in plan["tasks"]:
        if task["cluster"]:
            target = f"Slurm {task['cluster'].upper()}"
        elif task["runner"].startswith("amd-"):
            target = "K8s AMD"
        else:
            target = f"K8s {task['runner']}"
        gpus = re.search(r"(?:^|-)(\d+)gpu(?:-|$)", task["runner"])
        if gpus and (task["cluster"] or target == "K8s AMD"):
            target += f" / {gpus[1]} GPU"
        rows.append((task, task["config"], target))
    if rows:
        lines += [
            "",
            "| Order | Check | Verifies | Run on |",
            "| --- | --- | --- | --- |",
        ]
        for order, (item, path, target) in enumerate(rows, 1):
            label = f"[{_cell(item['label'])}]({source_url(plan, path)})"
            lines.append(
                f"| {order} | {label} | {_cell(item['reason'])} | {_cell(target)} |"
            )
    if not plan["tasks"]:
        lines += ["", "**Dispatch:** no task recommended."]
    if any(t["cluster"] == "gb200" for t in plan["tasks"]):
        lines += ["", "**Routing:** GB200 first; GB300 if full. One cluster per task."]
    if any(
        t["cluster"] and t["runner"].removeprefix("slurm-").startswith("b200-")
        for t in plan["tasks"]
    ):
        lines += [
            "",
            "**Hardware:** B200-labelled Slurm tasks are cross-hardware checks.",
        ]
    if plan["conflicts"]:
        lines += ["", f"**Conflicts:** {_cell(plan['conflicts'])}"]
    lines += [
        "",
        f"**Status:** recommendations only; tests not run; required CI unchanged. [Commit {plan['head'][:8]}]({source_url(plan)}).",
    ]
    return "\n".join(lines) + "\n"
