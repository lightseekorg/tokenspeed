import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_RUNNER_PREFIXES = ("b200-", "amd-", "gb200-", "b300-")
SLURM_RUNNER_PREFIXES = ("b200-", "gb200-", "slurm-gb200-")


def workflow_dispatch_inputs(name: str) -> dict:
    path = REPO_ROOT / ".github" / "workflows" / name
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)
    return triggers["workflow_dispatch"]["inputs"]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_slurm_dispatch_script(
    tmp_path: Path, **overrides: str
) -> subprocess.CompletedProcess[str]:
    workflow = load_yaml(REPO_ROOT / ".github/workflows/slurm-dispatch.yml")
    step = next(
        step
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Submit and wait for Slurm tasks"
    )
    original_script = step["run"]
    script = original_script.replace(
        'exec test/ci/run_slurm.sh "${args[@]}"',
        """printf 'arg=%s\\n' "${args[@]}"
printf 'artifact=%s\\n' "${TS_CI_ARTIFACT_ROOT-}"
printf 'cache=%s\\n' "${TS_CI_CACHE_DIR-}"
""",
    )
    script = script.replace(
        "repo = Path.cwd()",
        'repo = Path(__import__("os").environ.get("TOKENSPEED_TEST_REPO_ROOT", Path.cwd()))',
    )
    assert script != original_script
    env = {
        **os.environ,
        "PR": "",
        "CLUSTER": "gb200",
        "YAML_SELECTION": "off",
        "RUNNERS": "b200-4gpu,gb200-4gpu",
        "TASK_TYPES": "eval,perf",
        "MATCH": "",
        "INCLUDE_MMLU": "false",
        "TRIGGER": "all",
        "RUNNER_TEMP": str(tmp_path),
        "USER": "test-coordinator",
        **overrides,
    }
    env.pop("TS_CI_ARTIFACT_ROOT", None)
    env.pop("TS_CI_CACHE_DIR", None)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def eligible_config_paths(runner_prefixes: tuple[str, ...]) -> set[str]:
    paths = set()
    for path in (REPO_ROOT / "test" / "ci").rglob("*.yaml"):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = task["runner"]["labels"]
        if any(label.startswith(runner_prefixes) for label in labels):
            paths.add(path.relative_to(REPO_ROOT).as_posix())
    return paths


def eligible_slurm_config_paths() -> set[str]:
    paths = eligible_config_paths(SLURM_RUNNER_PREFIXES)
    for path in (REPO_ROOT / "test" / "ci").rglob("*.yaml"):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = task["runner"]["labels"]
        if task["type"] != "perf" and any(
            label.startswith("b300-") for label in labels
        ):
            paths.add(path.relative_to(REPO_ROOT).as_posix())
    return paths


def configured_yaml_choices(workflow_name: str) -> set[str]:
    choices = workflow_dispatch_inputs(workflow_name)["yaml"]["options"]
    return {choice for choice in choices if choice.startswith("test/ci/")}


def test_k8s_dispatch_lists_every_supported_ci_yaml():
    assert configured_yaml_choices("k8s-dispatch.yml") == eligible_config_paths(
        K8S_RUNNER_PREFIXES
    )


def test_slurm_dispatch_lists_every_supported_ci_yaml():
    assert (
        configured_yaml_choices("slurm-dispatch.yml") == eligible_slurm_config_paths()
    )


def test_slurm_dispatch_lists_every_supported_trigger():
    choices = workflow_dispatch_inputs("slurm-dispatch.yml")["trigger"]["options"]
    assert set(choices) == {
        "all",
        "per-commit",
        "manual",
        "nightly",
        "debug",
        "slurm",
    }


def test_slurm_dispatch_lists_every_supported_cluster():
    cluster = workflow_dispatch_inputs("slurm-dispatch.yml")["cluster"]

    assert cluster["default"] == "gb200"
    assert set(cluster["options"]) == {"gb200", "gb300"}


def test_slurm_dispatch_routes_gb300_to_its_coordinator():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/slurm-dispatch.yml")

    assert workflow["jobs"]["dispatch"]["runs-on"] == (
        "${{ inputs.cluster == 'gb300' && "
        "'slurm-dispatch-gb300' || 'slurm-dispatch' }}"
    )
    assert "${{ inputs.cluster }}" in workflow["concurrency"]["group"]


@pytest.mark.parametrize("runners", ["b200-4gpu,gb200-4gpu", "b200-4gpu, gb200-4gpu"])
def test_slurm_dispatch_uses_optional_gb300_alias_and_shared_paths(tmp_path, runners):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel.yaml",
        RUNNERS=runners,
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner-alias\narg=b300-1gpu=gb300-1gpu\n" in result.stdout
    assert "arg=b200-4gpu" not in result.stdout
    assert "arg=gb200-4gpu" not in result.stdout
    assert "artifact=/data/home/test-coordinator/tokenspeed-slurm" in result.stdout
    assert "cache=/data/home/test-coordinator/tokenspeed-cache" in result.stdout


def test_slurm_dispatch_preserves_gb200_defaults(tmp_path):
    result = run_slurm_dispatch_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "arg=--runner\narg=b200-4gpu\n" in result.stdout
    assert "arg=--runner\narg=gb200-4gpu\n" in result.stdout
    assert "artifact=\n" in result.stdout
    assert "cache=\n" in result.stdout


def test_slurm_dispatch_resolves_missing_coordinator_user(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel.yaml",
        USER="",
    )
    coordinator_user = subprocess.run(
        ["id", "-un"], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert result.returncode == 0, result.stderr
    assert f"artifact=/data/home/{coordinator_user}/tokenspeed-slurm" in result.stdout
    assert f"cache=/data/home/{coordinator_user}/tokenspeed-cache" in result.stdout


def test_slurm_dispatch_rejects_runner_for_another_cluster(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel.yaml",
        RUNNERS="b200-4gpu",
    )

    assert result.returncode == 2
    assert "Runner b200-4gpu is not supported by the GB300 cluster" in result.stderr


def test_slurm_dispatch_accepts_one_explicit_matching_gb300_runner(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel.yaml",
        RUNNERS="gb300-1gpu",
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner-alias\narg=b300-1gpu=gb300-1gpu\n" in result.stdout


@pytest.mark.parametrize(
    ("runners", "message"),
    [
        ("gb300-4gpu", "does not match"),
        ("gb300-1gpu,gb300-1gpu", "exactly one explicit runner"),
    ],
)
def test_slurm_dispatch_rejects_mismatched_or_multiple_gb300_runners(
    tmp_path, runners, message
):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel.yaml",
        RUNNERS=runners,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_slurm_dispatch_requires_explicit_yaml_for_gb300(tmp_path):
    result = run_slurm_dispatch_script(tmp_path, CLUSTER="gb300")

    assert result.returncode == 2
    assert "GB300 requires an explicit YAML selection" in result.stderr


def test_slurm_dispatch_rejects_ambiguous_b300_runner(tmp_path):
    task = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel.yaml")
    task["runner"]["labels"] = ["b300-1gpu", "b300-4gpu"]
    config = tmp_path / "ambiguous.yaml"
    config.write_text(yaml.safe_dump(task))

    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=str(config),
        TOKENSPEED_TEST_REPO_ROOT=str(tmp_path),
    )

    assert result.returncode == 2
    assert "exactly one declared b300-* runner" in result.stderr


def test_default_task_matrices_do_not_declare_gb300():
    labels = []
    for path in (REPO_ROOT / "test/ci").rglob("*.yaml"):
        labels.extend(load_yaml(path)["runner"]["labels"])

    assert not any(label.startswith("gb300-") for label in labels)


def test_qwen35_agentic_allows_declared_80k_context():
    task = load_yaml(
        REPO_ROOT / "test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml"
    )

    assert "--max-model-len 80000" in task["server"]["command"]
    assert task["env"]["TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"


def test_nvidia_arm_model_tests_allow_runner_wait_time():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/pr-test-nvidia-arm.yml")

    assert workflow["jobs"]["model-test"]["with"]["timeout_minutes"] >= 120
