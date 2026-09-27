from pathlib import Path

import pytest
from ci_path_filter import (
    RUNNER_GROUPS,
    VENDOR_WORKFLOWS,
    kernel_solution_vendor,
    path_vendor,
    should_run,
)
from pipeline import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NVIDIA_GROUPS = tuple(group for group in RUNNER_GROUPS if group != "amd")
KERNEL = "tokenspeed-kernel/python/tokenspeed_kernel"


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
        {".github/workflows/gb300-slurm-per-commit.yml"},
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
        "tokenspeed-kernel/test/ops/attention/test_kda_mtp_verify.py",
        f"{KERNEL}/registry.py",
        f"{KERNEL}/ops/attention/mha/__init__.py",
        f"{KERNEL}/ops/attention/mha/triton.py",
        f"{KERNEL}/ops/attention/mha/_triton/decode.py",
        f"{KERNEL}/ops/gemm/routed_gemv.py",
        f"{KERNEL}/thirdparty/__init__.py",
        "python/tokenspeed/runtime/sampling/backends/flashinfer.py",
    ],
)
def test_shared_paths_require_every_group(path):
    assert path_vendor(path, REPO_ROOT) is None
    assert required_groups(path) == set(RUNNER_GROUPS)


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsa_amd.py",
        "tokenspeed-kernel/test/amd/test_kpool_gluon_integration.py",
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/mhc.py",
        "tokenspeed-kernel/python/requirements/rocm.txt",
        "tokenspeed-kernel/benchmarks/amd/gfx950.json",
        "test/ci_system/install_deps_rocm.sh",
        f"{KERNEL}/ops/attention/mha/gluon.py",
        f"{KERNEL}/ops/attention/dsv41/_gluon/indexer.py",
        f"{KERNEL}/ops/moe/gluon/mxfp4.py",
        f"{KERNEL}/ops/communication/iris.py",
    ],
)
def test_amd_owned_paths_require_only_amd(path):
    assert path_vendor(path, REPO_ROOT) == "amd"
    assert required_groups(path) == {"amd"}


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
        f"{KERNEL}/ops/attention/rmha/_cute_dsl/rel_decode.py",
        f"{KERNEL}/ops/attention/gdn/_flashinfer/adapter.py",
        f"{KERNEL}/ops/moe/deep_gemm/_triton/mega_moe_stage.py",
        f"{KERNEL}/ops/moe/flashinfer/tactics/kimi-k3,ep=8,tp=1.json",
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
        (["amd-mi35x-1gpu-test", "amd-mi355-1gpu-bench"], {"amd"}),
        (["b200-4gpu", "h100-1gpu"], {"nvidia-x86"}),
        (["gb200-4gpu"], {"nvidia-arm"}),
        (["slurm-gb200-4gpu"], {"nvidia-gb200-slurm"}),
        (["slurm-gb300-4gpu"], {"nvidia-gb300-slurm"}),
        (["b200-1gpu", "amd-mi35x-1gpu-test"], {"nvidia-x86", "amd"}),
    ],
)
def test_task_yaml_requires_groups_of_its_runner_labels(tmp_path, labels, expected):
    path = "test/ci/eval/task.yaml"
    (tmp_path / path).parent.mkdir(parents=True)
    (tmp_path / path).write_text(
        "name: task\nrunner:\n  labels:\n"
        + "".join(f"    - {label}\n" for label in labels)
    )

    assert required_groups(path, tmp_path) == expected


def test_existing_task_yaml_requires_only_its_runner_groups():
    assert required_groups("test/ci/perf/kernel-benchmark-amd-gfx950.yaml") == {"amd"}


@pytest.mark.parametrize(
    "text",
    [None, "runner: {}\n", "runner:\n  labels: [\n"],
)
def test_unreadable_task_yaml_requires_every_group(tmp_path, text):
    path = "test/ci/eval/task.yaml"
    if text is not None:
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(text)

    assert required_groups(path, tmp_path) == set(RUNNER_GROUPS)


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
        "tokenspeed-kernel/test/amd/ops/test_mhc_gfx950.py",
        f"{KERNEL}/ops/attention/mha/flashinfer.py",
    }
    for group in RUNNER_GROUPS:
        assert should_run(paths, group, "pull_request", REPO_ROOT), group


def test_workflow_dispatch_runs_vendor_groups_for_foreign_paths():
    assert should_run(
        {"tokenspeed-kernel/test/amd/ops/test_mhc_gfx950.py"},
        "nvidia-x86",
        "workflow_dispatch",
        REPO_ROOT,
    )


def test_unrelated_paths_run_nothing():
    for group in RUNNER_GROUPS:
        assert not should_run({"assets/logo.png"}, group, "pull_request", REPO_ROOT)


@pytest.mark.parametrize("runner_group", RUNNER_GROUPS)
def test_workflow_classifies_the_checkout_after_installing_pyyaml(runner_group):
    workflow = load_yaml(REPO_ROOT / VENDOR_WORKFLOWS[runner_group])
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
