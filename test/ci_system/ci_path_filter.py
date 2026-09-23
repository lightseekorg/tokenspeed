"""Classify changed files for AMD and NVIDIA GPU CI."""

import argparse
import re
import sys
from pathlib import Path

import yaml
from pipeline import is_amd_runner, is_nvidia_arm_runner, load_yaml

RUNNER_GROUPS = (
    "amd",
    "nvidia-arm",
    "nvidia-gb200-slurm",
    "nvidia-gb300-slurm",
    "nvidia-x86",
)

# Every runner group belongs to one vendor; vendor-owned paths below are
# expressed per vendor so a new NVIDIA runner group needs no new path list.
RUNNER_GROUP_VENDORS = {
    "amd": "amd",
    "nvidia-arm": "nvidia",
    "nvidia-gb200-slurm": "nvidia",
    "nvidia-gb300-slurm": "nvidia",
    "nvidia-x86": "nvidia",
}

SHARED_DIRECTORIES = (
    "python",
    "test",
    "tokenspeed-kernel",
    "tokenspeed-scheduler",
)
SHARED_FILES = frozenset(
    {
        ".github/workflows/run-pr-test-stage.yml",
    }
)

# Documentation is not executed by any GPU job.
DOCUMENTATION_SUFFIXES = (".md",)

# A task YAML only runs on its declared runner labels, so a change to it
# requires only the runner groups that schedule one of those labels.
TASK_DIRECTORY = "test/ci"

# Directories and files owned by a single vendor. A change here requires only
# that vendor's runner groups, even when the path sits inside a shared
# directory.
VENDOR_PATHS = {
    "amd": (
        "test/ci_system/cleanup_amd_gpu_state.sh",
        "test/ci_system/install_deps_rocm.sh",
        "test/ci_system/install_kernel_benchmark_rocm.sh",
        "test/ci_system/kernel_benchmark_ci.py",
        "test/ci_system/mi450_rocjitsu_worker_python.sh",
        "test/ci_system/mi450_sim_report_digest.py",
        "test/ci_system/run_mi450_rocjitsu.sh",
        "test/ci_system/run_mi450_rocjitsu_parallel.sh",
        "test/ci_system/setup_mi450_sim.sh",
        "tokenspeed-kernel-amd",
        "tokenspeed-kernel/benchmarks/amd",
        "tokenspeed-kernel/python/requirements/rocm.txt",
        "tokenspeed-kernel/python/requirements/rocm-thirdparty.txt",
        "tokenspeed-kernel/test/amd",
    ),
    "npu": (
        "test/ci_system/install_triton_ascend.sh",
        "test/ci_system/verify_triton_ascend.py",
    ),
    "nvidia": (
        "test/ci_system/cleanup_nvidia_gpu_state.sh",
        "test/ci_system/cuda_coredump.py",
        "test/ci_system/diagnose_nvidia_state.sh",
        "test/ci_system/flashinfer_jit_cache_installer.py",
        "test/ci_system/install_deps_cu129.py",
        "tokenspeed-mla",
        "tokenspeed-kernel/python/requirements/cuda.txt",
        "tokenspeed-kernel/python/requirements/cuda-thirdparty.txt",
        "tokenspeed-kernel/test/nvidia",
    ),
}

# Paths used by only some runner groups' workflows. DeepSWE runs only from its
# manual B300 workflow, so it requires no per-commit group.
RUNNER_GROUP_PATHS = {
    "test/ci/deepswe": frozenset(),
    "test/ci/run_slurm.sh": frozenset({"nvidia-gb200-slurm", "nvidia-gb300-slurm"}),
    "test/ci_system/slurm_submit.py": frozenset(
        {"nvidia-gb200-slurm", "nvidia-gb300-slurm"}
    ),
}

# Kernel sources follow ``ops/<family>/[<variant>/]<solution>`` (with private
# helpers under ``_<solution>/``) and ``thirdparty/<package>``. These solution
# and package names only run on one vendor: their registrations declare that
# vendor and the family modules gate them on the current platform. Solutions
# that are not listed, such as ``triton``, are shared.
KERNEL_PACKAGE = "tokenspeed-kernel/python/tokenspeed_kernel"
VENDOR_KERNEL_SOLUTIONS = {
    "amd": frozenset({"gluon", "iris"}),
    "npu": frozenset({"ascend"}),
    "nvidia": frozenset(
        {
            "cuda",
            "cuda_lamport",
            "cute_dsl",
            "cute_fused",
            "deep_ep",
            "deep_gemm",
            "deep_select",
            "fabric",
            "faster_hadamard_transform",
            "flash_mla",
            "flashinfer",
            "marlin",
            "mega_moe",
            "tokenspeed_mla",
            "trtllm",
            "trtllm_cutedsl",
        }
    ),
}
VENDOR_KERNEL_THIRDPARTY = {
    "nvidia": frozenset(
        {
            "cuda",
            "cute_dsl",
            "cutedsl_kda",
            "deep_select",
            "flashinfer",
            "msa",
            "trtllm_blockwise",
        }
    ),
}

# A vendor-named kernel file that mentions another vendor's platform checks
# or registrations may change that vendor's behavior, so it stays shared.
VENDOR_CONTENT_MARKERS = {
    "amd": re.compile(
        r"\bis_amd\b|\bis_cdna\d*\b|\bis_rdna\d*\b|\"amd\"|tokenspeed_kernel_amd"
    ),
    "npu": re.compile(r"\bis_ascend\b|\bis_npu\b|\"ascend\""),
    "nvidia": re.compile(r"\bis_nvidia\b|\bis_hopper\w*|\bis_blackwell\w*|\"nvidia\""),
}
VENDOR_WORKFLOWS = {
    "amd": ".github/workflows/pr-test-amd.yml",
    "nvidia-arm": ".github/workflows/pr-test-nvidia-arm.yml",
    "nvidia-gb200-slurm": ".github/workflows/gb200-slurm-per-commit.yml",
    "nvidia-gb300-slurm": ".github/workflows/gb300-slurm-per-commit.yml",
    "nvidia-x86": ".github/workflows/pr-test-nvidia.yml",
}


def is_in_directory(path: str, directory: str) -> bool:
    return path == directory or path.startswith(f"{directory}/")


def touches_directory(paths: set[str], directory: str) -> bool:
    return any(is_in_directory(path, directory) for path in paths)


def solution_name(component: str) -> str:
    """Strip a private ``_`` prefix and file suffixes from a path component."""
    return component.lstrip("_").split(".", 1)[0]


def kernel_solution_vendor(path: str) -> str | None:
    """Return the vendor that a kernel solution path names, if exactly one."""
    if not path.startswith(f"{KERNEL_PACKAGE}/"):
        return None
    parts = path[len(KERNEL_PACKAGE) + 1 :].split("/")
    if len(parts) < 2:
        return None
    if parts[0] == "ops":
        names = {solution_name(part) for part in parts[1:]}
        vendors = {
            vendor
            for vendor, solutions in VENDOR_KERNEL_SOLUTIONS.items()
            if names & solutions
        }
    elif parts[0] == "thirdparty":
        package = solution_name(parts[1])
        vendors = {
            vendor
            for vendor, packages in VENDOR_KERNEL_THIRDPARTY.items()
            if package in packages
        }
    else:
        return None
    return vendors.pop() if len(vendors) == 1 else None


def mentions_other_vendor(path: str, vendor: str, repo_root: Path) -> bool:
    """Whether the checked-out ``path`` refers to a vendor other than ``vendor``.

    Deleted or non-text files have nothing to inspect and keep their path
    classification.
    """
    try:
        text = (repo_root / path).read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
        return False
    return any(
        marker.search(text)
        for other, marker in VENDOR_CONTENT_MARKERS.items()
        if other != vendor
    )


def path_vendor(path: str, repo_root: Path) -> str | None:
    """Return the vendor that owns ``path``, or ``None`` when it is shared."""
    for vendor, owned_paths in VENDOR_PATHS.items():
        if any(is_in_directory(path, owned) for owned in owned_paths):
            return vendor
    vendor = kernel_solution_vendor(path)
    if vendor is not None and not mentions_other_vendor(path, vendor, repo_root):
        return vendor
    return None


def runner_label_in_group(label: str, runner_group: str) -> bool:
    """Whether ``runner_group``'s workflow schedules tasks on ``label``.

    Mirrors the matrix filters in each workflow's scan job: the ARM workflow
    drops ``slurm-*`` labels, which only the Slurm workflows pick up.
    """
    if runner_group == "amd":
        return is_amd_runner(label)
    if is_amd_runner(label):
        return False
    if runner_group == "nvidia-x86":
        return not is_nvidia_arm_runner(label)
    if runner_group == "nvidia-arm":
        return is_nvidia_arm_runner(label) and not label.startswith("slurm-")
    if runner_group == "nvidia-gb200-slurm":
        return label.startswith("slurm-gb200-")
    if runner_group == "nvidia-gb300-slurm":
        return label.startswith("slurm-gb300-")
    raise ValueError(f"unsupported runner group: {runner_group!r}")


def task_runner_labels(path: str, repo_root: Path) -> list[str] | None:
    """Return the runner labels of a checked-out CI task YAML.

    Returns ``None`` for anything that is not a readable task declaration,
    including deleted files, so the caller falls back to the shared rule.
    """
    if not is_in_directory(path, TASK_DIRECTORY) or not path.endswith(".yaml"):
        return None
    try:
        data = load_yaml(repo_root / path)
    except (FileNotFoundError, IsADirectoryError, ValueError, yaml.YAMLError):
        return None
    runner = data.get("runner")
    labels = runner.get("labels") if isinstance(runner, dict) else None
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label for label in labels)
    ):
        return None
    return labels


def path_requires_group(path: str, runner_group: str, repo_root: Path) -> bool:
    if path.endswith(DOCUMENTATION_SUFFIXES):
        return False
    vendor = path_vendor(path, repo_root)
    if vendor is not None:
        return vendor == RUNNER_GROUP_VENDORS[runner_group]
    for group_path, runner_groups in RUNNER_GROUP_PATHS.items():
        if is_in_directory(path, group_path):
            return runner_group in runner_groups
    labels = task_runner_labels(path, repo_root)
    if labels is not None:
        return any(runner_label_in_group(label, runner_group) for label in labels)
    if path in SHARED_FILES:
        return True
    if any(is_in_directory(path, directory) for directory in SHARED_DIRECTORIES):
        return True
    return path == VENDOR_WORKFLOWS[runner_group]


def should_run(
    paths: set[str], runner_group: str, event_name: str, repo_root: Path
) -> bool:
    if event_name == "workflow_dispatch":
        return True
    return any(path_requires_group(path, runner_group, repo_root) for path in paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify changed files for a vendor PR test workflow."
    )
    parser.add_argument(
        "changed_files",
        type=Path,
        help="File containing one repository-relative changed path per line.",
    )
    parser.add_argument(
        "--runner-group",
        choices=RUNNER_GROUPS,
        required=True,
        help="Vendor runner group being considered.",
    )
    parser.add_argument(
        "--event-name",
        required=True,
        help="GitHub event name; workflow_dispatch always enables the workflow.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Checkout of the tested revision, used to inspect changed files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        line.strip()
        for line in args.changed_files.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    run_vendor_tests = should_run(
        paths, args.runner_group, args.event_name, args.repo_root
    )
    install_mla = args.runner_group.startswith("nvidia") and touches_directory(
        paths, "tokenspeed-mla"
    )

    print(f"should_run={str(run_vendor_tests).lower()}")
    print(f"install_tokenspeed_mla_from_source={int(install_mla)}")

    if run_vendor_tests:
        print(
            f"Changed paths require {args.runner_group} GPU tests.",
            file=sys.stderr,
        )
        for path in sorted(paths):
            if path_requires_group(path, args.runner_group, args.repo_root):
                print(f"  {path}", file=sys.stderr)
    else:
        print(
            f"Changed paths do not require {args.runner_group} GPU tests.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
