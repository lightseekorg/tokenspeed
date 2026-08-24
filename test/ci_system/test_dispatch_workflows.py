import os
import subprocess
from pathlib import Path

import pytest
import yaml
from pipeline import build_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_RUNNER_PREFIXES = ("b200-", "amd-", "gb200-", "b300-")
SLURM_RUNNER_PREFIXES = (
    "b200-",
    "gb200-",
    "slurm-b200-",
    "slurm-gb200-",
    "slurm-gb300-",
)


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
    checkout = next(
        step
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Checkout trusted dispatcher"
    )
    dispatch_script = next(
        step["run"]
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Submit and wait for Slurm tasks"
    )

    assert workflow["jobs"]["dispatch"]["runs-on"] == (
        "${{ inputs.cluster == 'gb300' && "
        "'slurm-dispatch-gb300' || 'slurm-dispatch' }}"
    )
    assert "${{ inputs.cluster }}" in workflow["concurrency"]["group"]
    assert checkout["with"]["ref"] == "main"
    assert 'python3 - "$YAML_SELECTION" "$CLUSTER" "$PR"' in dispatch_script
    assert "from slurm_submit import pr_worktree" in dispatch_script
    assert "with pr_worktree(repo, pr) as checkout:" in dispatch_script


@pytest.mark.parametrize("runners", ["b200-4gpu,gb200-4gpu", "b200-4gpu, gb200-4gpu"])
def test_slurm_dispatch_uses_declared_gb300_runner_and_shared_paths(tmp_path, runners):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS=runners,
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner\narg=slurm-gb300-4gpu\n" in result.stdout
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
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
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
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS="b200-4gpu",
    )

    assert result.returncode == 2
    assert "Runner b200-4gpu is not supported by the GB300 cluster" in result.stderr


def test_slurm_dispatch_rejects_b300_yaml_for_gb300_cluster(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel.yaml",
    )

    assert result.returncode == 2
    assert "No supported Slurm runners declared" in result.stderr


def test_slurm_dispatch_accepts_one_explicit_matching_gb300_runner(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS="slurm-gb300-4gpu",
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner\narg=slurm-gb300-4gpu\n" in result.stdout


def test_slurm_dispatch_passes_multi_node_gb300_runner_unchanged(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner\narg=slurm-gb300-4gpu\n" in result.stdout
    assert "runner-alias" not in result.stdout


@pytest.mark.parametrize(
    ("runners", "message"),
    [
        ("gb300-4gpu", "is not declared"),
        ("slurm-gb300-4gpu,slurm-gb300-4gpu", "exactly one explicit runner"),
    ],
)
def test_slurm_dispatch_rejects_mismatched_or_multiple_gb300_runners(
    tmp_path, runners, message
):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=(
            "test/ci/eval/"
            "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
        ),
        RUNNERS=runners,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_slurm_dispatch_requires_explicit_yaml_for_gb300(tmp_path):
    result = run_slurm_dispatch_script(tmp_path, CLUSTER="gb300")

    assert result.returncode == 2
    assert "GB300 requires an explicit YAML selection" in result.stderr


def test_slurm_dispatch_rejects_ambiguous_gb300_runner(tmp_path):
    task = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel.yaml")
    task["runner"]["labels"] = ["gb300-1gpu", "gb300-4gpu"]
    config = tmp_path / "ambiguous.yaml"
    config.write_text(yaml.safe_dump(task))

    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION=str(config),
        TOKENSPEED_TEST_REPO_ROOT=str(tmp_path),
    )

    assert result.returncode == 2
    assert "exactly one declared [slurm-]gb300-* runner" in result.stderr


def test_only_dedicated_tasks_declare_gb300():
    configs = []
    for path in (REPO_ROOT / "test/ci").rglob("*.yaml"):
        if any("gb300-" in label for label in load_yaml(path)["runner"]["labels"]):
            configs.append(path.name)

    assert sorted(configs) == [
        "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-deepswe-gb300-slurm.yaml",
        "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml",
        "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml",
        "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
        "kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
        "kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
    ]


def test_kimi_k3_gb300_is_two_node_per_commit_tp8():
    task = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )

    assert task["triggers"] == ["per-commit"]
    assert task["runner"]["labels"] == ["slurm-gb300-4gpu"]
    assert task["slurm"] == {"nodes": 2, "gpus_per_node": 4}
    assert "--tensor-parallel-size 8" in task["server"]["command"]


def test_kimi_k3_nvfp4_gb300_uses_pinned_local_models():
    target_path = (
        "/models/nvidia--Kimi-K3-NVFP4/" "f8c5234a0a880bcc6cbf779a315e7ee2f405b812"
    )
    draft_path = (
        "/models/Inferact--Kimi-K3-DSpark/" "cf6b8244620e7ea4b0651d214f28e89eac75bed6"
    )
    plain = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )
    dspark = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )

    assert target_path in plain["server"]["command"]
    assert target_path in dspark["server"]["command"]
    assert draft_path in dspark["server"]["command"]
    assert dspark["env"]["TOKENSPEED_DFLASH_AUX_STREAM"] == "attn_res"


def test_gb300_slurm_per_commit_workflow_is_isolated_and_automatic():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/gb300-slurm-per-commit.yml")
    triggers = workflow.get("on") or workflow.get(True)
    submit = workflow["jobs"]["submit"]
    scan_steps = workflow["jobs"]["scan"]["steps"]
    gate = next(
        step for step in scan_steps if step.get("name") == "Check trusted source"
    )
    matrix_step = next(
        step
        for step in scan_steps
        if step.get("name") == "Build multi-node task matrix"
    )
    submit_script = next(
        step["run"]
        for step in submit["steps"]
        if step.get("name") == "Submit and wait for GB300 Slurm task"
    )
    checkout = next(
        step for step in submit["steps"] if step.get("name") == "Checkout dispatcher"
    )

    assert set(triggers) == {"push", "pull_request"}
    assert submit["name"] == "${{ matrix.name }}"
    assert submit["runs-on"] == "slurm-dispatch-gb300"
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert '--runner "$RUNNER"' in submit_script
    assert "--runner-alias" not in submit_script
    assert '--source-pr "$PR_NUMBER"' in submit_script
    assert '--pr "$PR_NUMBER"' not in submit_script
    assert "secrets.HF_TOKEN" not in str(submit)
    assert "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN" in submit_script
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False
    assert "github.repository == 'lightseekorg/tokenspeed'" in gate["env"]["ALLOWED"]
    assert "github.event.pull_request.draft == false" in gate["env"]["ALLOWED"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in gate["env"]["ALLOWED"]
    )
    assert gate["env"]["ENABLED"] == (
        "${{ vars.TOKENSPEED_CI_GB300_SLURM_PER_COMMIT_ENABLED == 'true' }}"
    )
    assert "needs.scan.outputs.enabled == 'true'" in submit["if"]
    assert "TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS" not in matrix_step.get("env", {})
    assert "--multi-node only" in matrix_step["run"]

    cancel_workflow = load_yaml(
        REPO_ROOT / ".github/workflows/cancel-pr-tests-on-close.yml"
    )
    cancel_groups = {
        item["group"]
        for item in cancel_workflow["jobs"]["cancel"]["strategy"]["matrix"]["include"]
    }
    assert "gb300-slurm-per-commit" in cancel_groups


def test_gb300_slurm_per_commit_matrix_selects_kimi_k3_tasks(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", raising=False)

    matrix = build_matrix(
        REPO_ROOT / "test/ci",
        REPO_ROOT,
        trigger="per-commit",
        runner_group="nvidia-arm",
        workflow_stage="model-test",
        multi_node="only",
    )

    assert matrix["include"] == [
        {
            "name": "eval-kimi-k3-mxfp4-tp8-two-node-aime26-gb300-slurm",
            "type": "eval",
            "config": (
                "test/ci/eval/"
                "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
            ),
            "runner": "slurm-gb300-4gpu",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        },
        {
            "name": "eval-kimi-k3-nvfp4-dspark-tp8-two-node-aime26-gb300-slurm",
            "type": "eval",
            "config": (
                "test/ci/eval/"
                "kimi-k3-nvfp4-dspark-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
            ),
            "runner": "slurm-gb300-4gpu",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        },
        {
            "name": "eval-kimi-k3-nvfp4-tp8-two-node-aime26-gb300-slurm",
            "type": "eval",
            "config": (
                "test/ci/eval/"
                "kimi-k3-nvfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
            ),
            "runner": "slurm-gb300-4gpu",
            "priority": "normal",
            "optional": False,
            "workflow_stage": "model-test",
        },
    ]


def test_nvidia_arm_workflow_excludes_multi_node_tasks():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/pr-test-nvidia-arm.yml")
    scan_script = next(
        step["run"]
        for step in workflow["jobs"]["scan"]["steps"]
        if step.get("name") == "Build task matrix"
    )

    assert "--multi-node exclude" in scan_script


def test_qwen35_agentic_allows_declared_80k_context():
    task = load_yaml(
        REPO_ROOT / "test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic.yaml"
    )

    assert "--max-model-len 80000" in task["server"]["command"]
    assert task["env"]["TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"


def test_nvidia_arm_model_tests_allow_runner_wait_time():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/pr-test-nvidia-arm.yml")

    assert workflow["jobs"]["model-test"]["with"]["timeout_minutes"] >= 120
