"""Classify changed files for AMD and NVIDIA GPU CI."""

import argparse
import sys
from pathlib import Path

RUNNER_GROUPS = ("amd", "nvidia-arm", "nvidia-gb300-slurm", "nvidia-x86")

# Every runner group belongs to one vendor; vendor-owned paths below are
# expressed per vendor so a new NVIDIA runner group needs no new path list.
RUNNER_GROUP_VENDORS = {
    "amd": "amd",
    "nvidia-arm": "nvidia",
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

# Paths owned by a single vendor. A change here requires only that vendor's
# runner groups, even when the path sits inside a shared directory.
VENDOR_DIRECTORIES = {
    "amd": (
        "tokenspeed-kernel-amd",
        "tokenspeed-kernel/test/amd",
    ),
    "nvidia": (
        "tokenspeed-mla",
        "tokenspeed-kernel/test/nvidia",
    ),
}
VENDOR_WORKFLOWS = {
    "amd": ".github/workflows/pr-test-amd.yml",
    "nvidia-arm": ".github/workflows/pr-test-nvidia-arm.yml",
    "nvidia-gb300-slurm": ".github/workflows/gb300-slurm-per-commit.yml",
    "nvidia-x86": ".github/workflows/pr-test-nvidia.yml",
}


def is_in_directory(path: str, directory: str) -> bool:
    return path == directory or path.startswith(f"{directory}/")


def touches_directory(paths: set[str], directory: str) -> bool:
    return any(is_in_directory(path, directory) for path in paths)


def path_vendor(path: str) -> str | None:
    """Return the vendor that owns ``path``, or ``None`` when it is shared."""
    for vendor, directories in VENDOR_DIRECTORIES.items():
        if any(is_in_directory(path, directory) for directory in directories):
            return vendor
    return None


def path_requires_group(path: str, runner_group: str) -> bool:
    vendor = path_vendor(path)
    if vendor is not None:
        return vendor == RUNNER_GROUP_VENDORS[runner_group]
    if path in SHARED_FILES:
        return True
    if any(is_in_directory(path, directory) for directory in SHARED_DIRECTORIES):
        return True
    return path == VENDOR_WORKFLOWS[runner_group]


def should_run(paths: set[str], runner_group: str, event_name: str) -> bool:
    if event_name == "workflow_dispatch":
        return True
    return any(path_requires_group(path, runner_group) for path in paths)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        line.strip()
        for line in args.changed_files.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    run_vendor_tests = should_run(paths, args.runner_group, args.event_name)
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
    else:
        print(
            f"Changed paths do not require {args.runner_group} GPU tests.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
