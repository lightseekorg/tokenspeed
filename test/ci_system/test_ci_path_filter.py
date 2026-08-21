from ci_path_filter import should_run


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
