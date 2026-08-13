from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_RUNNER_PREFIXES = ("b200-", "amd-", "gb200-", "b300-")
SLURM_RUNNER_PREFIXES = ("b200-", "gb200-", "slurm-")


def workflow_dispatch_inputs(name: str) -> dict:
    path = REPO_ROOT / ".github" / "workflows" / name
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on") or workflow.get(True)
    return triggers["workflow_dispatch"]["inputs"]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def eligible_config_paths(runner_prefixes: tuple[str, ...]) -> set[str]:
    paths = set()
    for path in (REPO_ROOT / "test" / "ci").rglob("*.yaml"):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = task["runner"]["labels"]
        if any(label.startswith(runner_prefixes) for label in labels):
            paths.add(path.relative_to(REPO_ROOT).as_posix())
    return paths


def configured_yaml_choices(workflow_name: str) -> set[str]:
    choices = workflow_dispatch_inputs(workflow_name)["yaml"]["options"]
    return {choice for choice in choices if choice.startswith("test/ci/")}


def test_k8s_dispatch_lists_every_supported_ci_yaml():
    assert configured_yaml_choices("k8s-dispatch.yml") == eligible_config_paths(
        K8S_RUNNER_PREFIXES
    )


def test_slurm_dispatch_lists_every_b200_and_gb200_ci_yaml():
    assert configured_yaml_choices("slurm-dispatch.yml") == eligible_config_paths(
        SLURM_RUNNER_PREFIXES
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


def test_qwen35_agentic_allows_declared_80k_context():
    task = load_yaml(
        REPO_ROOT / "test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml"
    )

    assert "--max-model-len 80000" in task["server"]["command"]
    assert task["env"]["TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"


def test_nvidia_arm_model_tests_allow_runner_wait_time():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/pr-test-nvidia-arm.yml")

    assert workflow["jobs"]["model-test"]["with"]["timeout_minutes"] >= 120
