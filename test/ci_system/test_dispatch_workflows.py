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
printf 'image=%s\\n' "${TS_CI_CONTAINER_IMAGE-}"
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
        "CONTAINER_IMAGE": "",
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
    env.pop("TS_CI_CONTAINER_IMAGE", None)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_k8s_resolve_source_script(
    tmp_path: Path,
    *,
    pr: str = "",
    commit: str = "",
    pr_files: str = "README.md",
) -> tuple[subprocess.CompletedProcess[str], str, str, str]:
    workflow = load_yaml(REPO_ROOT / ".github/workflows/k8s-dispatch.yml")
    step = next(
        step
        for step in workflow["jobs"]["scan"]["steps"]
        if step.get("name") == "Resolve source"
    )
    script = step["run"].replace("${{ github.repository }}", "lightseekorg/tokenspeed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "gh-calls"
    output = tmp_path / "github-output"
    summary = tmp_path / "step-summary"
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$GH_CALLS"
endpoint=${2:-}
case "$endpoint" in
  repos/lightseekorg/tokenspeed/commits/main)
    printf '%s\\n' "$GH_MAIN_SHA"
    ;;
  repos/lightseekorg/tokenspeed/commits/*)
    printf '{"sha":"%s","html_url":"https://github.com/lightseekorg/tokenspeed/commit/%s"}\\n' \
      "$GH_COMMIT_SHA" "$GH_COMMIT_SHA"
    ;;
  repos/lightseekorg/tokenspeed/pulls/*/files)
    printf '%s\\n' "$GH_PR_FILES"
    ;;
  repos/lightseekorg/tokenspeed/pulls/*)
    printf '{"head":{"sha":"%s"},"html_url":"https://github.com/lightseekorg/tokenspeed/pull/123"}\\n' \
      "$GH_PR_SHA"
    ;;
  *)
    echo "Unexpected gh call: $*" >&2
    exit 1
    ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_TOKEN": "test-token",
        "GH_CALLS": str(calls),
        "GH_MAIN_SHA": "1" * 40,
        "GH_COMMIT_SHA": commit.strip().lower(),
        "GH_PR_SHA": "2" * 40,
        "GH_PR_FILES": pr_files,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
        "PR": pr,
        "COMMIT": commit,
    }
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        result,
        output.read_text(encoding="utf-8") if output.exists() else "",
        summary.read_text(encoding="utf-8") if summary.exists() else "",
        calls.read_text(encoding="utf-8") if calls.exists() else "",
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


def test_k8s_dispatch_accepts_full_commit_sha(tmp_path):
    requested = "A" * 40

    result, output, summary, calls = run_k8s_resolve_source_script(
        tmp_path, commit=f"  {requested}  "
    )

    assert result.returncode == 0, result.stderr
    assert f"sha={'a' * 40}" in output
    assert "install_mla=1" in output
    assert "- Mode: commit" in summary
    assert f"api repos/lightseekorg/tokenspeed/commits/{'a' * 40}" in calls


def test_k8s_dispatch_defaults_to_latest_main(tmp_path):
    result, output, summary, calls = run_k8s_resolve_source_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert f"sha={'1' * 40}" in output
    assert "install_mla=0" in output
    assert "- Mode: main" in summary
    assert "api repos/lightseekorg/tokenspeed/commits/main --jq .sha" in calls


@pytest.mark.parametrize(
    ("pr_files", "expected_install_mla"),
    [("README.md", "0"), ("tokenspeed-mla/src/kernel.py", "1")],
)
def test_k8s_dispatch_preserves_pr_resolution(tmp_path, pr_files, expected_install_mla):
    result, output, summary, calls = run_k8s_resolve_source_script(
        tmp_path,
        pr=" https://github.com/lightseekorg/tokenspeed/pull/123 ",
        pr_files=pr_files,
    )

    assert result.returncode == 0, result.stderr
    assert f"sha={'2' * 40}" in output
    assert f"install_mla={expected_install_mla}" in output
    assert "- Mode: pr" in summary
    assert "api repos/lightseekorg/tokenspeed/pulls/123" in calls


def test_k8s_dispatch_rejects_pr_and_commit_together(tmp_path):
    result, output, _, calls = run_k8s_resolve_source_script(
        tmp_path, pr="123", commit="a" * 40
    )

    assert result.returncode == 2
    assert "Set only one of pr or commit" in result.stderr
    assert output == ""
    assert calls == ""


@pytest.mark.parametrize("commit", ["a" * 39, "feature-branch", "a" * 41])
def test_k8s_dispatch_rejects_non_full_commit_sha(tmp_path, commit):
    result, output, _, calls = run_k8s_resolve_source_script(tmp_path, commit=commit)

    assert result.returncode == 2
    assert "expected exactly 40 hexadecimal characters" in result.stderr
    assert output == ""
    assert calls == ""


def test_k8s_dispatch_commit_input_is_optional():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/k8s-dispatch.yml")
    commit_input = workflow_dispatch_inputs("k8s-dispatch.yml")["commit"]

    assert commit_input["required"] is False
    assert commit_input["default"] == ""
    assert "full commit SHA" in commit_input["description"]
    assert (
        "${{ inputs.commit || inputs.pr || 'main' }}"
        in workflow["concurrency"]["group"]
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


def test_slurm_dispatch_accepts_immutable_tokenspeed_image_override(tmp_path):
    image = (
        "ghcr.io/lightseekorg/tokenspeed-runner:flashinfer-0.6.18@sha256:" + "d" * 64
    )

    result = run_slurm_dispatch_script(tmp_path, CONTAINER_IMAGE=image)

    assert result.returncode == 0, result.stderr
    assert f"image={image}" in result.stdout


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/lightseekorg/tokenspeed-runner:flashinfer-0.6.18",
        "ghcr.io/other/tokenspeed-runner:flashinfer-0.6.18@sha256:" + "d" * 64,
        "docker.io/lightseekorg/tokenspeed-runner:flashinfer-0.6.18@sha256:" + "d" * 64,
    ],
)
def test_slurm_dispatch_rejects_unsafe_image_override(tmp_path, image):
    result = run_slurm_dispatch_script(tmp_path, CONTAINER_IMAGE=image)

    assert result.returncode == 2
    assert "container_image must be an immutable" in result.stderr


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
    assert (
        "arg=--runner-alias\n"
        "arg=slurm-gb300-4gpu=slurm-gb300-4gpu\n" in result.stdout
    )
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
    assert "image=\n" in result.stdout


def test_slurm_dispatch_maps_gb300_defaults_without_changing_filters(tmp_path):
    result = run_slurm_dispatch_script(tmp_path, CLUSTER="gb300")

    assert result.returncode == 0, result.stderr
    assert "arg=--all\n" in result.stdout
    assert "arg=--runner-alias\narg=b200-4gpu=gb300-4gpu\n" in result.stdout
    assert "arg=--runner-alias\narg=gb200-4gpu=gb300-4gpu\n" in result.stdout
    assert "arg=--type\narg=eval\n" in result.stdout
    assert "arg=--type\narg=perf\n" in result.stdout
    assert "arg=--exclude-match\narg=mmlu\n" in result.stdout
    assert "artifact=/data/home/test-coordinator/tokenspeed-slurm" in result.stdout
    assert "cache=/data/home/test-coordinator/tokenspeed-cache" in result.stdout


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
    assert "does not identify exactly one runner declared" in result.stderr


def test_slurm_dispatch_maps_b200_yaml_to_gb300_runners(tmp_path):
    result = run_slurm_dispatch_script(
        tmp_path,
        CLUSTER="gb300",
        YAML_SELECTION="test/ci/ut/ut-tokenspeed-kernel.yaml",
    )

    assert result.returncode == 0, result.stderr
    assert "arg=--runner-alias\narg=b200-1gpu=gb300-1gpu\n" in result.stdout
    assert "arg=--runner-alias\narg=gb200-1gpu=gb300-1gpu\n" in result.stdout


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
    assert (
        "arg=--runner-alias\n"
        "arg=slurm-gb300-4gpu=slurm-gb300-4gpu\n" in result.stdout
    )


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
    assert (
        "arg=--runner-alias\n"
        "arg=slurm-gb300-4gpu=slurm-gb300-4gpu\n" in result.stdout
    )


@pytest.mark.parametrize(
    ("runners", "message"),
    [
        ("gb300-4gpu", "does not identify exactly one runner declared"),
        ("slurm-gb300-4gpu,slurm-gb300-4gpu", "more than once"),
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


def test_slurm_dispatch_accepts_multiple_native_gb300_runners(tmp_path):
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

    assert result.returncode == 0, result.stderr
    assert "arg=--runner-alias\narg=gb300-1gpu=gb300-1gpu\n" in result.stdout
    assert "arg=--runner-alias\narg=gb300-4gpu=gb300-4gpu\n" in result.stdout


def test_only_dedicated_tasks_declare_gb300():
    configs = []
    for path in (REPO_ROOT / "test/ci").rglob("*.yaml"):
        if any("gb300-" in label for label in load_yaml(path)["runner"]["labels"]):
            configs.append(path.name)

    assert sorted(configs) == [
        "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-gb300-slurm.yaml",
        "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml",
        "kimi-k3-mxfp4-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
        "kimi-k3-nvfp4-dflash2-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
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


def test_kimi_k3_dflash2_gb300_uses_a_window_aware_drafter_backend():
    task = load_yaml(
        REPO_ROOT / "test/ci/eval/"
        "kimi-k3-nvfp4-dflash2-tp8-two-node-evalscope-aime26-gb300-slurm.yaml"
    )
    command = task["server"]["command"]

    assert task["slurm"] == {"nodes": 2, "gpus_per_node": 4}
    assert "/models/nvidia--Kimi-K3-NVFP4/" in command
    assert "--speculative-draft-model-path lightseekorg/kimi-k3-dflash2" in command
    assert "--speculative-algorithm DFLASH" in command
    # The draft's sliding_attention layers need a backend that masks with them.
    assert "--drafter-attention-backend mla" in command


def test_gb300_slurm_nightly_workflow_is_scheduled_and_isolated():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/gb300-slurm-nightly.yml")
    triggers = workflow.get("on") or workflow.get(True)
    scan = workflow["jobs"]["scan"]
    submit = workflow["jobs"]["submit"]
    gate = next(
        step for step in scan["steps"] if step.get("name") == "Check trusted source"
    )
    matrix_step = next(
        step
        for step in scan["steps"]
        if step.get("name") == "Build nightly GB300 task matrix"
    )
    submit_script = next(
        step["run"]
        for step in submit["steps"]
        if step.get("name") == "Submit and wait for nightly GB300 Slurm task"
    )
    checkout = next(
        step for step in submit["steps"] if step.get("name") == "Checkout dispatcher"
    )

    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "17 18 * * *"}]
    assert "github.repository == 'lightseekorg/tokenspeed'" in gate["env"]["ALLOWED"]
    assert "vars.TOKENSPEED_CI_REPOSITORY" in gate["env"]["ALLOWED"]
    assert "github.ref == 'refs/heads/main'" in gate["env"]["ALLOWED"]
    assert gate["env"]["ENABLED"] == (
        "${{ vars.TOKENSPEED_CI_GB300_SLURM_NIGHTLY_ENABLED == 'true' }}"
    )
    gate_condition = (
        "steps.gate.outputs.allowed == 'true' && "
        "steps.gate.outputs.enabled == 'true'"
    )
    for step_name in (
        "Checkout code",
        "Install scan dependency",
        "Build nightly GB300 task matrix",
    ):
        step = next(step for step in scan["steps"] if step.get("name") == step_name)
        assert step["if"] == gate_condition
    assert workflow["concurrency"] == {
        "group": "gb300-slurm-nightly",
        "cancel-in-progress": False,
    }
    assert submit["name"] == "${{ matrix.name }}"
    assert submit["runs-on"] == "slurm-dispatch-gb300"
    assert "needs.scan.outputs.allowed == 'true'" in submit["if"]
    assert "needs.scan.outputs.enabled == 'true'" in submit["if"]
    assert "needs.scan.outputs.has_tasks == 'true'" in submit["if"]
    assert "--trigger nightly" in matrix_step["run"]
    assert "--runner-group nvidia-arm" in matrix_step["run"]
    assert "--workflow-stage model-test" in matrix_step["run"]
    assert "--multi-node only" in matrix_step["run"]
    assert "startswith('slurm-gb300-')" in matrix_step["run"]
    assert '--runner "$RUNNER"' in submit_script
    assert "--source-pr" not in submit_script
    assert "secrets.HF_TOKEN" not in str(submit)
    assert "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN" in submit_script
    scan_checkout = next(
        step for step in scan["steps"] if step.get("name") == "Checkout code"
    )
    assert scan_checkout["with"]["ref"] == "${{ github.sha }}"
    assert scan_checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False


def test_gb300_slurm_nightly_matrix_selects_the_nightly_kimi_k3_tasks(monkeypatch):
    monkeypatch.delenv("TOKENSPEED_CI_EXCLUDED_RUNNER_LABELS", raising=False)

    matrix = build_matrix(
        REPO_ROOT / "test/ci",
        REPO_ROOT,
        trigger="nightly",
        runner_group="nvidia-arm",
        workflow_stage="model-test",
        multi_node="only",
    )

    assert {(entry["config"], entry["runner"]) for entry in matrix["include"]} == {
        (
            "test/ci/eval/"
            "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-mmmu-pro-vision-"
            "gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
        (
            "test/ci/eval/"
            "kimi-k3-mxfp4-dspark-tp8-two-node-kvv-ocr-bench-gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
        (
            "test/ci/eval/"
            "kimi-k3-nvfp4-dflash2-tp8-two-node-evalscope-aime26-gb300-slurm.yaml",
            "slurm-gb300-4gpu",
        ),
    }


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


def test_mi450_sim_uses_direct_runner_and_bounded_timeout():
    workflow = load_yaml(REPO_ROOT / ".github/workflows/run-pr-test-stage.yml")
    job = workflow["jobs"]["test"]

    assert job["runs-on"] == "${{ matrix.runner }}"
    assert job["timeout-minutes"] == (
        "${{ matrix.runner == 'amd-mi45x-cpu-test'"
        " && 10 || inputs.timeout_minutes }}"
    )


def test_mi450_sim_runs_on_the_cpu_only_pool():
    task = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml")

    assert task["runner"]["labels"] == ["amd-mi45x-cpu-test"]


def test_mi450_sim_uses_bounded_smoke_suite():
    task = load_yaml(REPO_ROOT / "test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml")

    assert task["env"]["MI450_SIM_RUN_TIMEOUT"] == "330"
    assert task["env"]["MI450_SIM_TEST_ROOT"] != "tokenspeed-kernel/test"
    assert "tokenspeed-kernel/test/ops/attention" in task["env"]["MI450_SIM_TESTS"]


def test_mi450_sim_uses_stock_triton_compatible_libhip_path():
    script = (REPO_ROOT / "test/ci_system/run_mi450_rocjitsu.sh").read_text()

    assert 'libhip_path="${rocm_root}/lib/libamdhip64.so"' in script
    assert 'test -f "${libhip_path}"' in script
    assert 'export TRITON_LIBHIP_PATH="${libhip_path}"' in script
