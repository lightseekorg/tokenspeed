import pytest
from ci_path_filter import RUNNER_GROUPS, path_vendor, should_run

NVIDIA_GROUPS = tuple(group for group in RUNNER_GROUPS if group != "amd")


def test_gb300_slurm_path_filter_covers_shared_and_own_workflow_changes():
    group = "nvidia-gb300-slurm"

    assert should_run({"test/ci/eval/task.yaml"}, group, "pull_request")
    assert should_run({"tokenspeed-mla/src/kernel.cu"}, group, "pull_request")
    assert should_run(
        {".github/workflows/gb300-slurm-per-commit.yml"},
        group,
        "pull_request",
    )


def test_gb300_slurm_path_filter_ignores_other_vendor_workflows():
    assert not should_run(
        {".github/workflows/pr-test-nvidia-arm.yml"},
        "nvidia-gb300-slurm",
        "pull_request",
    )


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel/test/conftest.py",
        "tokenspeed-kernel/test/utils.py",
        "tokenspeed-kernel/test/ops/test_attention.py",
        "tokenspeed-kernel/test/thirdparty/test_kda_mtp_verify.py",
        "tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mha/gluon.py",
    ],
)
def test_shared_kernel_test_paths_require_every_group(path):
    assert path_vendor(path) is None
    for group in RUNNER_GROUPS:
        assert should_run({path}, group, "pull_request"), group


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel/test/amd/ops/attention/test_gluon_dsa_amd.py",
        "tokenspeed-kernel/test/amd/test_kpool_gluon_integration.py",
        "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/mhc.py",
    ],
)
def test_amd_owned_paths_require_only_amd(path):
    assert path_vendor(path) == "amd"
    assert should_run({path}, "amd", "pull_request")
    for group in NVIDIA_GROUPS:
        assert not should_run({path}, group, "pull_request"), group


@pytest.mark.parametrize(
    "path",
    [
        "tokenspeed-kernel/test/nvidia/ops/test_tokenspeed_mla.py",
        "tokenspeed-kernel/test/nvidia/thirdparty/test_cuda.py",
        "tokenspeed-mla/src/kernel.cu",
    ],
)
def test_nvidia_owned_paths_require_only_nvidia_groups(path):
    assert path_vendor(path) == "nvidia"
    assert not should_run({path}, "amd", "pull_request")
    for group in NVIDIA_GROUPS:
        assert should_run({path}, group, "pull_request"), group


def test_vendor_subtree_prefix_does_not_leak_onto_sibling_paths():
    # ``.../test/amd`` must match the directory, not any path sharing the
    # prefix string, or a shared file like ``test/amd_helpers.py`` would skip
    # NVIDIA CI.
    assert path_vendor("tokenspeed-kernel/test/amd_helpers.py") is None
    assert path_vendor("tokenspeed-kernel/test/nvidia_helpers.py") is None
    assert path_vendor("tokenspeed-kernel-amd-docs/README.md") is None


def test_mixed_vendor_changes_require_both_vendors():
    paths = {
        "tokenspeed-kernel/test/amd/ops/test_mhc_gfx950.py",
        "tokenspeed-kernel/test/nvidia/thirdparty/test_fa4.py",
    }
    for group in RUNNER_GROUPS:
        assert should_run(paths, group, "pull_request"), group


def test_workflow_dispatch_runs_vendor_groups_for_foreign_paths():
    assert should_run(
        {"tokenspeed-kernel/test/amd/ops/test_mhc_gfx950.py"},
        "nvidia-x86",
        "workflow_dispatch",
    )


def test_unrelated_paths_run_nothing():
    for group in RUNNER_GROUPS:
        assert not should_run({"docs/design/scheduler.md"}, group, "pull_request")
