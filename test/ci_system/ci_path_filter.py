"""Classify changed files for AMD and NVIDIA GPU CI."""

import argparse
import sys
from pathlib import Path

RUNNER_GROUPS = ("amd", "nvidia-arm", "nvidia-gb300-slurm", "nvidia-x86")

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

VENDOR_DIRECTORIES = {
    "amd": ("tokenspeed-kernel-amd",),
    "nvidia-arm": ("tokenspeed-mla",),
    "nvidia-gb300-slurm": ("tokenspeed-mla",),
    "nvidia-x86": ("tokenspeed-mla",),
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


def should_run(paths: set[str], runner_group: str, event_name: str) -> bool:
    if event_name == "workflow_dispatch":
        return True

    if paths & SHARED_FILES:
        return True
    if any(touches_directory(paths, directory) for directory in SHARED_DIRECTORIES):
        return True
    if VENDOR_WORKFLOWS[runner_group] in paths:
        return True
    return any(
        touches_directory(paths, directory)
        for directory in VENDOR_DIRECTORIES[runner_group]
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
