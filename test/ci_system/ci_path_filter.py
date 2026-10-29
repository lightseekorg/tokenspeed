"""Classify changed files for AMD and NVIDIA GPU CI."""

import argparse
import re
import sys
from pathlib import Path

import yaml
from pipeline import is_amd_runner, is_npu_runner, is_nvidia_arm_runner, normalize_task

RUNNER_GROUPS = (
    "amd-mi350",
    "amd-mi450",
    "nvidia-arm",
    "nvidia-gb200-slurm",
    "nvidia-gb300-slurm",
    "nvidia-x86",
)

# Every runner group belongs to one vendor; vendor-owned paths below are
# expressed per vendor so a new NVIDIA runner group needs no new path list.
RUNNER_GROUP_VENDORS = {
    "amd-mi350": "amd",
    "amd-mi450": "amd",
    "nvidia-arm": "nvidia",
    "nvidia-gb200-slurm": "nvidia",
    "nvidia-gb300-slurm": "nvidia",
    "nvidia-x86": "nvidia",
}

AMD_ARCH_RUNNER_GROUPS = {"gfx950": "amd-mi350", "gfx1250": "amd-mi450"}
AMD_KERNEL_OPS = "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops"
AMD_KERNEL_TESTS = "tokenspeed-kernel/test/amd"
AMD_BENCHMARKS = "tokenspeed-kernel/benchmarks/amd"

SHARED_DIRECTORIES = (
    "python",
    "test",
    "tokenspeed-engine",
    "tokenspeed-kernel",
    "tokenspeed-scheduler",
)
SHARED_FILES = frozenset(
    {
        ".github/workflows/run-ci-task-matrix.yml",
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
        "tokenspeed-kernel-amd",
        "tokenspeed-kernel/benchmarks/amd",
        "tokenspeed-kernel/python/requirements/rocm.txt",
        "tokenspeed-kernel/python/requirements/rocm-thirdparty.txt",
        "tokenspeed-kernel/test/amd",
    ),
    "npu": (
        "test/ci/npu",
        "test/ci_system/npu",
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
    **{
        f"test/ci_system/{name}": frozenset({"amd-mi450"})
        for name in (
            "mi450_rocjitsu_worker_python.sh",
            "mi450_sim_env.sh",
            "mi450_sim_report_digest.py",
            "run_mi450_rocjitsu.sh",
            "run_mi450_rocjitsu_parallel.sh",
            "setup_mi450_sim.sh",
        )
    },
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
            "flashinfer_prims",
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
    "amd-mi350": (
        ".github/workflows/amd-mi350-tests.yml",
        ".github/workflows/amd-kernel-benchmark.yml",
    ),
    "amd-mi450": (".github/workflows/amd-mi450-tests.yml",),
    "nvidia-arm": (".github/workflows/pr-test-nvidia-arm.yml",),
    "nvidia-gb200-slurm": (".github/workflows/nvidia-gb200-tests.yml",),
    "nvidia-gb300-slurm": (".github/workflows/nvidia-gb300-tests.yml",),
    "nvidia-x86": (".github/workflows/nvidia-b200-tests.yml",),
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


def amd_arch_runner_group(path: str) -> str | None:
    """Recognize explicit architecture ownership; unknown AMD paths stay shared."""
    for arch, group in AMD_ARCH_RUNNER_GROUPS.items():
        if is_in_directory(path, f"{AMD_KERNEL_OPS}/{arch}"):
            return group
        if is_in_directory(path, f"{AMD_BENCHMARKS}/{arch}") or path == (
            f"{AMD_BENCHMARKS}/{arch}.json"
        ):
            return group
        if is_in_directory(path, f"{AMD_KERNEL_TESTS}/{arch}"):
            return group
    return None


def runner_label_in_group(label: str, runner_group: str) -> bool:
    """Whether ``runner_group``'s workflow schedules tasks on ``label``.

    Mirrors the matrix filters in each workflow's scan job: the ARM workflow
    drops ``slurm-*`` labels, which only the Slurm workflows pick up.
    """
    if is_npu_runner(label):
        return False
    if runner_group == "amd-mi450":
        return label.startswith(("amd-mi450-", "amd-mi45x-"))
    if runner_group == "amd-mi350":
        return is_amd_runner(label) and not label.startswith(
            ("amd-mi450-", "amd-mi45x-")
        )
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


def task_config(path: str, repo_root: Path) -> dict | None:
    """Read a task using the same validation as matrix construction.

    Missing or invalid definitions keep conservative coverage. Matrix
    construction reports any invalid definitions still in the checkout.
    """
    if not is_in_directory(path, TASK_DIRECTORY) or not path.endswith(".yaml"):
        return None
    try:
        return normalize_task(repo_root / path, repo_root)
    except (OSError, ValueError, yaml.YAMLError):
        return None


def path_requires_group(path: str, runner_group: str, repo_root: Path) -> bool:
    if path.endswith(DOCUMENTATION_SUFFIXES):
        return False
    arch_group = amd_arch_runner_group(path)
    if arch_group is not None:
        return runner_group == arch_group
    vendor = path_vendor(path, repo_root)
    if vendor is not None:
        return vendor == RUNNER_GROUP_VENDORS[runner_group]
    for group_path, runner_groups in RUNNER_GROUP_PATHS.items():
        if is_in_directory(path, group_path):
            return runner_group in runner_groups
    task = task_config(path, repo_root)
    if task is not None:
        if task.get("workflow_stage") == "import-check":
            return False
        return any(
            runner_label_in_group(label, runner_group)
            for label in task["runner"]["labels"]
        )
    if path in SHARED_FILES:
        return True
    if any(is_in_directory(path, directory) for directory in SHARED_DIRECTORIES):
        return True
    return path in VENDOR_WORKFLOWS[runner_group]


def should_run(
    paths: set[str], runner_group: str, event_name: str, repo_root: Path
) -> bool:
    # A compare response can be truncated at 300 files. Do not narrow coverage
    # when the changed-path list might omit a shared file.
    if event_name in {"workflow_dispatch", "schedule"} or len(paths) >= 300:
        return True
    return any(path_requires_group(path, runner_group, repo_root) for path in paths)


def should_run_import_check(
    paths: set[str], runner_group: str, event_name: str, repo_root: Path
) -> bool:
    """Keep adapter imports covered on the other AMD family's software stack."""
    if runner_group not in AMD_ARCH_RUNNER_GROUPS.values():
        return False
    if event_name in {"workflow_dispatch", "schedule"}:
        return True
    # Both MI450 tasks consume the shared stack independently.
    if runner_group == "amd-mi450" and "test/ci_system/mi450_sim_env.sh" in paths:
        return True
    for path in paths:
        if not is_in_directory(path, TASK_DIRECTORY) or not path.endswith(".yaml"):
            continue
        task = task_config(path, repo_root)
        # A deleted or unreadable task might have been an import check.
        if task is None:
            if path_requires_group(path, runner_group, repo_root):
                return True
            continue
        if task.get("workflow_stage") == "import-check" and any(
            runner_label_in_group(label, runner_group)
            for label in task["runner"]["labels"]
        ):
            return True
    if should_run(paths, runner_group, event_name, repo_root):
        return False
    # AMD adapters import implementations for both architectures. Tests, suite
    # definitions and simulator scripts do not affect those runtime imports.
    return any(
        is_in_directory(path, AMD_KERNEL_OPS)
        and path.endswith(".py")
        and amd_arch_runner_group(path) is not None
        for path in paths
    )


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
    run_import_check = should_run_import_check(
        paths, args.runner_group, args.event_name, args.repo_root
    )
    install_mla = args.runner_group.startswith("nvidia") and touches_directory(
        paths, "tokenspeed-mla"
    )

    print(f"should_run={str(run_vendor_tests).lower()}")
    print(f"should_run_import_check={str(run_import_check).lower()}")
    print(f"install_tokenspeed_mla_from_source={int(install_mla)}")

    if run_vendor_tests:
        print(
            f"Changed paths require {args.runner_group} GPU tests.",
            file=sys.stderr,
        )
        for path in sorted(paths):
            if path_requires_group(path, args.runner_group, args.repo_root):
                print(f"  {path}", file=sys.stderr)
    elif run_import_check:
        print(
            f"Changed paths require only {args.runner_group} adapter import checks.",
            file=sys.stderr,
        )
    else:
        print(
            f"Changed paths do not require {args.runner_group} GPU tests.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
