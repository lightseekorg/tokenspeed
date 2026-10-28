import json
from pathlib import Path

import pipeline
import pytest
import yaml
from ci_path_filter import (
    RUNNER_GROUPS,
    VENDOR_WORKFLOWS,
    kernel_solution_vendor,
    path_vendor,
    should_run,
    should_run_import_check,
)
from pipeline import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NVIDIA_GROUPS = tuple(
    group for group in RUNNER_GROUPS if group not in ("amd-mi350", "amd-mi450")
)
KERNEL = "tokenspeed-kernel/python/tokenspeed_kernel"


def write_task(repo_root: Path, path: str, labels: list[str], stage: str) -> Path:
    target = repo_root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "api_version": "ci.tokenspeed.io/v1",
                "name": target.stem,
                "type": "ut",
                "workflow_stage": stage,
                "triggers": ["per-commit"],
                "runner": {"labels": labels},
            }
        )
    )
    return target


def required_groups(path: str, repo_root: Path = REPO_ROOT) -> set[str]:
    return {
        group
        for group in RUNNER_GROUPS
        if should_run({path}, group, "pull_request", repo_root)
    }


def test_gb300_slurm_path_filter_covers_shared_and_own_workflow_changes():
    group = "nvidia-gb300-slurm"

    assert should_run({"test/ci_system/pipeline.py"}, group, "pull_request", REPO_ROOT)
    assert should_run(
        {"tokenspeed-mla/src/kernel.cu"}, group, "pull_request", REPO_ROOT
    )
    assert should_run(
        {".github/workflows/nvidia-gb300-tests.yml"},
        group,
        "pull_request",
        REPO_ROOT,
    )


def test_gb300_slurm_path_filter_ignores_other_vendor_workflows():
    assert not should_run(
        {".github/workflows/pr-test-nvidia-arm.yml"},
        "nvidia-gb300-slurm",
        "pull_request",
        REPO_ROOT,
    )


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel/test/conftest.py",
        "tokenspeed-kernel/test/utils.py",
        "tokenspeed-kernel/test/ops/test_attention.py",
        "tokenspeed-kernel/test/ops/test_attention_gfx1250.py",
        "tokenspeed-kernel/test/benchmark/test_benchmark_harness.py",
        "tokenspeed-kernel/test/ops/attention/test_kda_mtp_verify.py",
        f"{KERNEL}/registry.py",
        f"{KERNEL}/ops/attention/mha/__init__.py",
        f"{KERNEL}/ops/attention/mha/triton.py",
        f"{KERNEL}/ops/attention/mha/_triton/decode.py",
        f"{KERNEL}/ops/gemm/triton.py",
        f"{KERNEL}/thirdparty/__init__.py",
        "python/tokenspeed/runtime/sampling/backends/flashinfer.py",
        "tokenspeed-engine/crates/ts-control/src/event_loop.rs",
    ],
)
def test_shared_paths_require_every_group(path):
    assert path_vendor(path, REPO_ROOT) is None
    assert required_groups(path) == set(RUNNER_GROUPS)


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsa_amd.py",
        "tokenspeed-kernel/test/amd/test_scheduling.py",
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/_triton.py",
        "tokenspeed-kernel-amd/pyproject.toml",
        "tokenspeed-kernel/python/requirements/rocm.txt",
        "tokenspeed-kernel/benchmarks/amd/unknown.json",
        "test/ci_system/install_deps_rocm.sh",
        f"{KERNEL}/ops/attention/mha/gluon.py",
        f"{KERNEL}/ops/attention/dsv41/_gluon/indexer.py",
        f"{KERNEL}/ops/moe/gluon.py",
        f"{KERNEL}/ops/communication/iris.py",
    ],
)
def test_amd_owned_paths_require_only_amd(path):
    assert path_vendor(path, REPO_ROOT) == "amd"
    assert required_groups(path) == {"amd-mi350", "amd-mi450"}


@pytest.mark.parametrize(
    ("arch", "group"), [("gfx950", "amd-mi350"), ("gfx1250", "amd-mi450")]
)
@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/{arch}/attention/mha/prefill.py",
        "tokenspeed-kernel/test/amd/{arch}/ops/attention/test_prefill.py",
        "tokenspeed-kernel/test/amd/{arch}/fixtures/input.json",
        "tokenspeed-kernel/benchmarks/amd/{arch}/attention.json",
        "tokenspeed-kernel/benchmarks/amd/{arch}.json",
    ],
)
def test_architecture_paths_select_one_full_suite(arch, group, path):
    assert required_groups(path.format(arch=arch)) == {group}


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1201/mhc.py",
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx12500/mhc.py",
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950_helpers.py",
        "tokenspeed-kernel/test/amd/test_gluon_gfx1250_helpers.py",
        "tokenspeed-kernel/test/amd/ops/attention/test_prefill_gfx1250.py",
        "tokenspeed-kernel/test/amd/gfx12500/test_prefill.py",
        "tokenspeed-kernel/test/amd/gfx1201/test_prefill.py",
    ],
)
def test_unclassified_amd_paths_keep_both_full_suites(path):
    assert required_groups(path) == {"amd-mi350", "amd-mi450"}


@pytest.mark.parametrize(
    "name",
    [
        "setup_mi450_sim.sh",
        "mi450_sim_env.sh",
        "run_mi450_rocjitsu.sh",
        "run_mi450_rocjitsu_parallel.sh",
        "mi450_rocjitsu_worker_python.sh",
        "mi450_sim_report_digest.py",
    ],
)
def test_simulator_scripts_select_only_mi450(name):
    assert required_groups(f"test/ci_system/{name}") == {"amd-mi450"}


@pytest.mark.parametrize(
    ("arch", "other"), [("gfx950", "amd-mi450"), ("gfx1250", "amd-mi350")]
)
def test_architecture_source_changes_keep_other_family_imports(arch, other, tmp_path):
    # Path ownership also applies to deleted files, without reading the source.
    paths = {f"tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/{arch}/mha.py"}
    for group in RUNNER_GROUPS:
        assert should_run_import_check(paths, group, "pull_request", tmp_path) == (
            group == other
        )


@pytest.mark.parametrize(
    "paths",
    [
        {"tokenspeed-kernel/test/amd/gfx1250/test_prefill.py"},
        {"tokenspeed-kernel/benchmarks/amd/gfx950/attention.json"},
        {"test/ci_system/setup_mi450_sim.sh"},
        {"tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/README.md"},
        {"tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/_triton.py"},
    ],
)
def test_non_source_or_shared_changes_need_no_separate_import_check(paths):
    assert not any(
        should_run_import_check(paths, group, "pull_request", REPO_ROOT)
        for group in RUNNER_GROUPS
    )


@pytest.mark.parametrize(
    "extra",
    [
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/mha.py",
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/_triton.py",
        f"{KERNEL}/ops/attention/mha/gluon.py",
    ],
)
def test_mixed_and_cross_architecture_renames_keep_full_amd_coverage(extra):
    paths = {
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/mha.py",
        extra,
    }
    for group in ("amd-mi350", "amd-mi450"):
        assert should_run(paths, group, "pull_request", REPO_ROOT)
        assert not should_run_import_check(paths, group, "pull_request", REPO_ROOT)


@pytest.mark.parametrize("event", ["workflow_dispatch", "schedule"])
def test_explicit_full_runs_bypass_architecture_filtering(event):
    paths = {"tokenspeed-kernel/test/amd/gfx1250/test_prefill.py"}
    for group in RUNNER_GROUPS:
        assert should_run(paths, group, event, REPO_ROOT)
        assert should_run_import_check(paths, group, event, REPO_ROOT) == (
            group in {"amd-mi350", "amd-mi450"}
        )


def test_potentially_truncated_diff_keeps_full_coverage():
    paths = {f"tokenspeed-kernel/test/amd/gfx1250/test_{i}.py" for i in range(300)}
    for group in RUNNER_GROUPS:
        assert should_run(paths, group, "push", REPO_ROOT)


@pytest.mark.parametrize(
    ("config", "owner"),
    [("mi350", "amd-mi350"), ("mi450-sim", "amd-mi450")],
)
def test_import_task_changes_run_the_declared_check_even_with_shared_changes(
    config, owner
):
    paths = {f"test/ci/ut/import-tokenspeed-kernel-{config}.yaml"}
    assert required_groups(next(iter(paths))) == set()
    for group in RUNNER_GROUPS:
        assert should_run_import_check(paths, group, "pull_request", REPO_ROOT) == (
            group == owner
        )
    paths.add(f"{KERNEL}/registry.py")
    assert should_run(paths, owner, "pull_request", REPO_ROOT)
    assert should_run_import_check(paths, owner, "pull_request", REPO_ROOT)


@pytest.mark.parametrize(
    ("label", "group"),
    [("amd-mi355-1gpu", "amd-mi350"), ("amd-mi45x-cpu-test", "amd-mi450")],
)
@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("shared_change", [False, True])
def test_added_and_renamed_import_tasks_reach_the_matrix(
    tmp_path, capsys, label, group, renamed, shared_change
):
    path = "test/ci/ut/adapter-compatibility.yaml"
    previous = "test/ci/ut/previous-compatibility.yaml"
    task = write_task(tmp_path, previous if renamed else path, [label], "import-check")
    paths = {path}
    if renamed:
        task.rename(tmp_path / path)
        paths.add(previous)
    unchanged = "test/ci/ut/unchanged-compatibility.yaml"
    write_task(tmp_path, unchanged, [label], "import-check")
    if shared_change:
        paths.add(f"{KERNEL}/registry.py")

    assert required_groups(path, tmp_path) == set()
    assert should_run_import_check(paths, group, "pull_request", tmp_path)
    assert should_run(paths, group, "pull_request", tmp_path) == (
        renamed or shared_change
    )
    if not renamed:
        other = "amd-mi450" if group == "amd-mi350" else "amd-mi350"
        assert not should_run_import_check(paths, other, "pull_request", tmp_path)

    changed = tmp_path / "changed.txt"
    changed.write_text("\n".join(sorted(paths)))
    assert (
        pipeline.main(
            [
                "scan",
                "--repo-root",
                str(tmp_path),
                "--changed-files",
                str(changed),
                "--trigger",
                "per-commit",
                "--runner-group",
                group,
                "--workflow-stage",
                "import-check",
            ]
        )
        == 0
    )
    matrix = json.loads(capsys.readouterr().out)
    assert {entry["config"] for entry in matrix["include"]} == (
        {path, unchanged} if shared_change else {path}
    )


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel/test/nvidia/ops/test_tokenspeed_mla.py",
        "tokenspeed-kernel/test/nvidia/thirdparty/test_cuda.py",
        "tokenspeed-mla/src/kernel.cu",
        "tokenspeed-kernel/python/requirements/cuda-thirdparty.txt",
        "test/ci_system/flashinfer_jit_cache_installer.py",
        f"{KERNEL}/ops/attention/mha/cuda.py",
        f"{KERNEL}/ops/attention/mha/flashinfer.py",
        f"{KERNEL}/ops/attention/qsa/flashinfer_prims.py",
        f"{KERNEL}/thirdparty/flashinfer/qsa_prims.py",
        f"{KERNEL}/ops/attention/rmha/_cute_dsl/rel_decode.py",
        f"{KERNEL}/ops/attention/gdn/_flashinfer/adapter.py",
        f"{KERNEL}/ops/moe/_deep_gemm/mega_moe_stage.py",
        f"{KERNEL}/ops/moe/flashinfer.py",
        f"{KERNEL}/ops/communication/_cuda/a2a_fp8.cuh",
        f"{KERNEL}/ops/communication/_cuda/lamport_a2a.py",
        f"{KERNEL}/ops/communication/_cuda/lamport_a2a.cu",
        f"{KERNEL}/thirdparty/cute_dsl/ll_bf16.py",
        f"{KERNEL}/thirdparty/msa/csrc/fmha_sm100_plan.cu",
    ],
)
def test_nvidia_owned_paths_require_only_nvidia_groups(path):
    assert path_vendor(path, REPO_ROOT) == "nvidia"
    assert required_groups(path) == set(NVIDIA_GROUPS)


@pytest.mark.parametrize(
    "path",
    [
        f"{KERNEL}/ops/layernorm/ascend.py",
        "test/ci_system/install_triton_ascend.sh",
        "test/ci_system/npu/changes.py",
        "test/ci/npu/manifest.json",
        "test/ci/npu/tests/scatter-test.py",
    ],
)
def test_npu_owned_paths_require_no_gpu_group(path):
    assert path_vendor(path, REPO_ROOT) == "npu"
    assert required_groups(path) == set()


def test_kernel_solution_names_only_classify_ops_and_thirdparty():
    # ``msa`` is an NVIDIA third-party package but also a shared attention
    # variant under ``ops/``.
    assert kernel_solution_vendor(f"{KERNEL}/thirdparty/msa/api.py") == "nvidia"
    assert kernel_solution_vendor(f"{KERNEL}/ops/attention/msa/triton.py") is None
    assert kernel_solution_vendor(f"{KERNEL}/benchmark/cuda.py") is None
    assert kernel_solution_vendor("tokenspeed-kernel/test/ops/cuda.py") is None


def test_kernel_solution_file_mentioning_another_vendor_is_shared(tmp_path):
    shared = f"{KERNEL}/ops/attention/mla/tokenspeed_mla.py"
    nvidia_only = f"{KERNEL}/ops/attention/mla/cuda.py"
    for path, text in (
        (shared, "if current_platform().is_cdna4:\n    pass\n"),
        (nvidia_only, 'vendors=frozenset({"nvidia"})\n'),
    ):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)

    assert kernel_solution_vendor(shared) == "nvidia"
    assert path_vendor(shared, tmp_path) is None
    assert required_groups(shared, tmp_path) == set(RUNNER_GROUPS)
    assert path_vendor(nvidia_only, tmp_path) == "nvidia"


def test_deleted_kernel_solution_file_keeps_its_vendor(tmp_path):
    path = f"{KERNEL}/ops/gemm/trtllm.py"

    assert path_vendor(path, tmp_path) == "nvidia"
    assert required_groups(path, tmp_path) == set(NVIDIA_GROUPS)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["amd-mi35x-1gpu-test", "amd-mi355-1gpu-bench"], {"amd-mi350"}),
        (["amd-mi450-1gpu", "amd-mi45x-cpu-test"], {"amd-mi450"}),
        (["tokenspeed-npu-arm64"], set()),
        (["tokenspeed-npu-arm64", "b200-1gpu"], {"nvidia-x86"}),
        (
            ["tokenspeed-npu-arm64", "amd-mi35x-1gpu-test", "amd-mi450-1gpu"],
            {"amd-mi350", "amd-mi450"},
        ),
        (["b200-4gpu", "h100-1gpu"], {"nvidia-x86"}),
        (["gb200-4gpu"], {"nvidia-arm"}),
        (["slurm-gb200-4gpu"], {"nvidia-gb200-slurm"}),
        (["slurm-gb300-4gpu"], {"nvidia-gb300-slurm"}),
        (["b200-1gpu", "amd-mi35x-1gpu-test"], {"nvidia-x86", "amd-mi350"}),
    ],
)
def test_task_yaml_requires_groups_of_its_runner_labels(tmp_path, labels, expected):
    path = "test/ci/eval/task.yaml"
    write_task(tmp_path, path, labels, "unit-test")

    assert required_groups(path, tmp_path) == expected


def test_existing_task_yaml_requires_only_its_runner_groups():
    assert required_groups("test/ci/perf/kernel-benchmark-amd-gfx950.yaml") == {
        "amd-mi350"
    }


@pytest.mark.parametrize(
    "text",
    [
        None,
        "runner: {}\n",
        "runner:\n  labels: [\n",
        "workflow_stage: import-check\nrunner:\n  labels: [amd-mi355-1gpu]\n",
    ],
)
def test_unreadable_task_yaml_requires_every_group(tmp_path, text):
    path = "test/ci/eval/task.yaml"
    if text is not None:
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(text)

    assert required_groups(path, tmp_path) == set(RUNNER_GROUPS)
    for group in ("amd-mi350", "amd-mi450"):
        assert should_run_import_check({path}, group, "pull_request", tmp_path)
    if text is not None:
        with pytest.raises((ValueError, yaml.YAMLError)):
            pipeline.build_matrix(tmp_path / "test/ci", tmp_path, "per-commit")


@pytest.mark.parametrize(
    "labels",
    [
        ["b200-1gpu"],
        ["gb200-4gpu"],
        ["tokenspeed-npu-arm64"],
        ["amd-mi355-1gpu", "b200-1gpu"],
        ["amd-mi45x-cpu-test", "tokenspeed-npu-arm64"],
    ],
)
def test_unsupported_import_runners_fail_matrix_validation(tmp_path, labels):
    path = "test/ci/ut/adapter-compatibility.yaml"
    write_task(tmp_path, path, labels, "import-check")

    # Invalid tasks must reach the scanner and fail, rather than select no jobs.
    assert required_groups(path, tmp_path) == set(RUNNER_GROUPS)
    with pytest.raises(ValueError, match="import-check.*AMD runner labels"):
        pipeline.build_matrix(tmp_path / "test/ci", tmp_path, "per-commit")


@pytest.mark.parametrize("directory", ["test/ci/npu", "test/ci/deepswe"])
def test_deleted_tasks_keep_their_directory_exclusions(tmp_path, directory):
    paths = {f"{directory}/deleted.yaml"}
    for group in ("amd-mi350", "amd-mi450"):
        assert not should_run(paths, group, "pull_request", tmp_path)
        assert not should_run_import_check(paths, group, "pull_request", tmp_path)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("test/ci/deepswe/run_deepswe.sh", set()),
        ("test/ci/run_slurm.sh", {"nvidia-gb200-slurm", "nvidia-gb300-slurm"}),
        (
            "test/ci_system/slurm_submit.py",
            {"nvidia-gb200-slurm", "nvidia-gb300-slurm"},
        ),
    ],
)
def test_runner_group_scoped_paths(path, expected):
    assert required_groups(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "docs/design/scheduler.md",
        "test/ci/README.md",
        f"{KERNEL}/ops/attention/dsa/README.md",
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/README.md",
    ],
)
def test_documentation_requires_no_group(path):
    assert required_groups(path) == set()


def test_vendor_subtree_prefix_does_not_leak_onto_sibling_paths():
    # ``.../test/amd`` must match the directory, not any path sharing the
    # prefix string, or a shared file like ``test/amd_helpers.py`` would skip
    # NVIDIA CI.
    assert path_vendor("tokenspeed-kernel/test/amd_helpers.py", REPO_ROOT) is None
    assert path_vendor("tokenspeed-kernel/test/nvidia_helpers.py", REPO_ROOT) is None
    assert path_vendor("tokenspeed-kernel-amd-docs/setup.py", REPO_ROOT) is None


def test_mixed_vendor_changes_require_both_vendors():
    paths = {
        "tokenspeed-kernel/test/amd/ops/test_mhc.py",
        f"{KERNEL}/ops/attention/mha/flashinfer.py",
    }
    for group in RUNNER_GROUPS:
        assert should_run(paths, group, "pull_request", REPO_ROOT), group


def test_workflow_dispatch_runs_vendor_groups_for_foreign_paths():
    assert should_run(
        {"tokenspeed-kernel/test/amd/gfx950/ops/test_mhc_gfx950.py"},
        "nvidia-x86",
        "workflow_dispatch",
        REPO_ROOT,
    )


def test_unrelated_paths_run_nothing():
    for group in RUNNER_GROUPS:
        assert not should_run({"assets/logo.png"}, group, "pull_request", REPO_ROOT)


VENDOR_WORKFLOW_CASES = [
    (group, workflow)
    for group, workflows in VENDOR_WORKFLOWS.items()
    for workflow in workflows
]


@pytest.mark.parametrize(("runner_group", "workflow_path"), VENDOR_WORKFLOW_CASES)
def test_workflow_yaml_requires_only_its_runner_group(runner_group, workflow_path):
    assert required_groups(workflow_path) == {runner_group}


@pytest.mark.parametrize(("runner_group", "workflow_path"), VENDOR_WORKFLOW_CASES)
def test_workflow_classifies_the_checkout_after_installing_pyyaml(
    runner_group, workflow_path
):
    workflow = load_yaml(REPO_ROOT / workflow_path)
    steps = next(
        job["steps"]
        for job in workflow["jobs"].values()
        if any(step.get("name") == "Classify changed paths" for step in job["steps"])
    )
    names = [step.get("name") for step in steps]
    classify = steps[names.index("Classify changed paths")]

    assert names.index("Install scan dependency") < names.index(
        "Classify changed paths"
    )
    assert f"--runner-group {runner_group}" in classify["run"]
    assert '--repo-root "$GITHUB_WORKSPACE"' in classify["run"]
