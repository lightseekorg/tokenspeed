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

from __future__ import annotations

import json
from pathlib import Path

import pipeline
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGED_TASKS = [
    "test/ci/eval/deepseek-v4.1-flash-evalscope-gsm8k-amd.yaml",
    "test/ci/eval/deepseek-v4.1-flash-evalscope-gsm8k.yaml",
]


@pytest.fixture
def matrix():
    return {
        "include": [
            {"config": "test/ci/eval/changed.yaml", "runner": "b200-4gpu"},
            {"config": "test/ci/eval/changed.yaml", "runner": "gb200-4gpu"},
            {"config": "test/ci/ut/unchanged.yaml", "runner": "b200-1gpu"},
        ]
    }


def test_changed_task_keeps_every_existing_runner_entry(matrix):
    selected = pipeline.filter_matrix_for_changed_tasks(
        matrix, ["test/ci/eval/changed.yaml"]
    )
    assert selected["include"] == matrix["include"][:2]
    assert len(matrix["include"]) == 3


@pytest.mark.parametrize(
    "other",
    [
        "python/tokenspeed/runtime/model.py",
        "tokenspeed-kernel/python/requirements/cuda-thirdparty.txt",
        "test/ci_system/pipeline.py",
        ".github/workflows/pr-test-nvidia.yml",
        "test/ci/README.md",
        "test/ci/eval/other.yml",
        "test/ci/../shared.yaml",
        "test/ci/eval//other.yaml",
        "/test/ci/eval/other.yaml",
        "test/ci-other/task.yaml",
    ],
)
def test_mixed_or_noncanonical_diff_keeps_full_matrix(matrix, other):
    assert (
        pipeline.filter_matrix_for_changed_tasks(
            matrix, ["test/ci/eval/changed.yaml", other]
        )
        == matrix
    )


def test_empty_diff_keeps_manual_selection(matrix):
    assert pipeline.filter_matrix_for_changed_tasks(matrix, []) == matrix


@pytest.mark.parametrize(
    "other", [" test/ci/eval/other.yaml", "test/ci/eval/other.yaml "]
)
def test_scan_preserves_changed_path_whitespace(
    matrix, other, tmp_path, capsys, monkeypatch
):
    changed = tmp_path / "changed.txt"
    changed.write_text(f"test/ci/eval/changed.yaml\n{other}\n")
    monkeypatch.setattr(pipeline, "build_matrix", lambda *args: matrix)
    assert pipeline.main(["scan", "--changed-files", str(changed)]) == 0
    assert json.loads(capsys.readouterr().out) == matrix


def test_potentially_truncated_diff_keeps_full_matrix(matrix):
    paths = [f"test/ci/eval/task-{i}.yaml" for i in range(299)]
    assert pipeline.filter_matrix_for_changed_tasks(matrix, paths)["include"] == []
    paths.append("test/ci/eval/last.yaml")
    assert pipeline.filter_matrix_for_changed_tasks(matrix, paths) == matrix


def test_deleted_and_renamed_tasks(matrix):
    assert pipeline.filter_matrix_for_changed_tasks(
        matrix, ["test/ci/eval/deleted.yaml"]
    ) == {"include": []}
    assert (
        pipeline.filter_matrix_for_changed_tasks(
            matrix, ["test/ci/eval/old.yaml", "test/ci/eval/changed.yaml"]
        )["include"]
        == matrix["include"][:2]
    )


@pytest.mark.parametrize(
    ("group", "stage", "expected"),
    [
        ("amd", "model-test", [CHANGED_TASKS[0]]),
        ("nvidia-x86", "model-test", [CHANGED_TASKS[1]]),
        ("nvidia-arm", "model-test", []),
        ("amd", "unit-test", []),
        ("amd", "kernel-benchmark", []),
        ("nvidia-x86", "unit-test", []),
    ],
)
def test_scan_pr1748_selects_only_changed_tasks(
    group, stage, expected, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv(pipeline.EXCLUDED_RUNNER_LABELS_ENV, raising=False)
    monkeypatch.delenv(pipeline.B200_RUNNER_LABEL_ENV, raising=False)
    changed = tmp_path / "changed.txt"
    changed.write_text("\n".join(CHANGED_TASKS) + "\n")
    assert (
        pipeline.main(
            [
                "scan",
                "--repo-root",
                str(REPO_ROOT),
                "--root",
                "test/ci",
                "--trigger",
                "per-commit",
                "--runner-group",
                group,
                "--workflow-stage",
                stage,
                "--changed-files",
                str(changed),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert [item["config"] for item in result["include"]] == expected


def test_filter_does_not_hide_invalid_task_yaml(tmp_path):
    root = tmp_path / "test/ci"
    root.mkdir(parents=True)
    (root / "invalid.yaml").write_text("name: missing-required-fields\n")
    changed = tmp_path / "changed.txt"
    changed.write_text("test/ci/other.yaml\n")
    with pytest.raises(ValueError):
        pipeline.main(
            [
                "scan",
                "--repo-root",
                str(tmp_path),
                "--changed-files",
                str(changed),
            ]
        )


@pytest.mark.parametrize(
    "name",
    [
        "pr-test-amd",
        "pr-test-nvidia",
        "pr-test-nvidia-arm",
        "gb200-slurm-per-commit",
        "gb300-slurm-per-commit",
    ],
)
def test_per_commit_workflows_pass_the_collected_diff_to_scan(name):
    workflow = yaml.safe_load((REPO_ROOT / f".github/workflows/{name}.yml").read_text())
    steps = workflow["jobs"]["scan"]["steps"]
    classify = next(step["run"] for step in steps if step.get("id") == "changes")
    scan = next(
        step["run"] for step in steps if "pipeline.py scan" in step.get("run", "")
    )
    assert 'changed_files="$RUNNER_TEMP/tokenspeed-changed-files.txt"' in classify
    assert '--changed-files "$RUNNER_TEMP/tokenspeed-changed-files.txt"' in scan
    # Manual dispatches write an empty list rather than retaining a PR filter.
    if "workflow_dispatch" in workflow.get(True, workflow.get("on", {})):
        assert ': > "$changed_files"' in classify
