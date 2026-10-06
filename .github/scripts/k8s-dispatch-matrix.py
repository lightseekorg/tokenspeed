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

"""Select existing CI tasks for a manual Kubernetes dispatch."""

import json
import os
import subprocess
import sys
from pathlib import Path

repo = Path(os.environ.get("SOURCE_REPO", Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/ci_system"))
from pipeline import build_matrix


def terms(name):
    return [item.strip() for item in os.environ[name].split(",") if item.strip()]


selected_yaml = terms("YAML_SELECTION")
selected_types = set(terms("TASK_TYPES"))
matches = [item.lower() for item in terms("MATCH")]
trigger = os.environ["TRIGGER"]
matrix = build_matrix(
    repo / "test/ci",
    repo,
    None if trigger == "all" else trigger,
    runner_group="all",
)

if selected_yaml != ["all"]:
    invalid = [
        item
        for item in selected_yaml
        if not item.startswith("test/ci/") or not (repo / item).is_file()
    ]
    if invalid:
        raise SystemExit(f"Invalid CI YAML: {', '.join(invalid)}")
    selected_paths = set(selected_yaml)
else:
    selected_paths = None

include = []
for task in matrix["include"]:
    searchable = f"{task['name']} {task['config']}".lower()
    runner = task["runner"]
    if os.environ.get("EXACT_RUNNER") and runner != os.environ["EXACT_RUNNER"]:
        continue
    if runner.startswith("slurm-"):
        continue
    pool = os.environ["RUNNER_POOL"]
    runner_families = {
        "b200v2": runner.startswith("b200v2-"),
        "amd": runner.startswith("amd-"),
        "gb200": runner.startswith("gb200-"),
        "b300": runner.startswith("b300-"),
    }
    if pool != "all" and not runner_families[pool]:
        continue
    if selected_paths is not None and task["config"] not in selected_paths:
        continue
    if task["type"] not in selected_types:
        continue
    if matches and not any(term in searchable for term in matches):
        continue
    if os.environ["INCLUDE_MMLU"] != "true" and "mmlu" in searchable:
        continue
    include.append(task)

result = {"include": include}
if os.environ.get("EXACT_RUNNER"):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(os.environ["RUNNER_TEMP"], "dispatch-source.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "tasks": [
                    {"config": task["config"], "runner": task["runner"]}
                    for task in include
                ],
            }
        )
    )
output = os.environ["GITHUB_OUTPUT"]
with open(output, "a", encoding="utf-8") as stream:
    stream.write(f"matrix={json.dumps(result, separators=(',', ':'))}\n")
    stream.write(f"has_tasks={'true' if include else 'false'}\n")
if not include:
    raise SystemExit("No CI tasks matched the requested filters")
print(json.dumps(result, indent=2))
