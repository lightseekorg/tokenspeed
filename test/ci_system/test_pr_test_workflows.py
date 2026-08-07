from pathlib import Path

import pipeline
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_VARIABLE = "${{ vars.TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS }}"


def test_nvidia_pr_workflow_keeps_fork_safe_runner_exclusions(monkeypatch):
    path = REPO_ROOT / ".github" / "workflows" / "pr-test-nvidia.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)
    assert "pull_request" in triggers

    exclusion_entries = [
        (key, value)
        for step in workflow["jobs"]["scan"]["steps"]
        for key, value in step.get("env", {}).items()
        if "EXCLUDED_RUNNER_LABELS" in key
    ]
    assert len(exclusion_entries) == 1
    env_key, configured_value = exclusion_entries[0]
    assert env_key == pipeline.EXCLUDED_RUNNER_LABELS_ENV
    assert REPOSITORY_VARIABLE in configured_value

    monkeypatch.setenv(env_key, configured_value.replace(REPOSITORY_VARIABLE, ""))
    fork_exclusions = set(pipeline.get_excluded_runner_labels())
    assert fork_exclusions >= {"h100", "b300"}

    monkeypatch.setenv(
        env_key, configured_value.replace(REPOSITORY_VARIABLE, "mi355")
    )
    configured_exclusions = set(pipeline.get_excluded_runner_labels())
    assert configured_exclusions >= {"h100", "b300", "mi355"}
