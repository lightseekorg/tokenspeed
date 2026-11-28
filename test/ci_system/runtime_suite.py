"""Plan which test files one ``run_ci_suite.py`` invocation runs.

``test/runtime/run_ci_suite.py`` collects the registered runtime test files,
keeps those of its suite that are enabled on the runner, narrows them to the
files a pull request's changed paths reach, and then takes its partition.
``pipeline.py scan`` makes the same plan for each unit-test task before any
runner starts, so it can drop a pull request's task whose invocations would run
no test file.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

from ci_system.ci_register import (
    CIRegistry,
    DeviceBackend,
    auto_partition,
    collect_tests,
)
from ci_system.runtime_test_selection import (
    RUNTIME_TEST_DIRECTORY,
    select_runtime_tests,
)

SUITE_SCRIPT = f"{RUNTIME_TEST_DIRECTORY}/run_ci_suite.py"
DEVICE_MAPPING = {
    "cuda": DeviceBackend.CUDA,
}

# Per-commit test suites (run on every PR)
PER_COMMIT_SUITES = {
    DeviceBackend.CUDA: [
        "runtime-1gpu",
        "runtime-2gpu",
        "runtime-prefix-cache-e2e",
    ],
}
# Shell operators, which make a command more than one suite invocation.
SHELL_OPERATOR_CHARACTERS = frozenset("();<>|&")
# Shell expansions, whose values only the task's shell knows.
SHELL_EXPANSION_CHARACTERS = frozenset("$`")


@dataclass(frozen=True)
class SuitePlan:
    """The test files one suite invocation runs.

    Args:
        tests: The files to run, in collection order.
        skipped: The suite's files disabled everywhere or on the runner.
        changed_selection: How many enabled files the changed paths reach,
            counted before partitioning; None when they do not narrow the suite.
    """

    tests: List[CIRegistry]
    skipped: List[CIRegistry]
    changed_selection: Optional[int]


def suite_argument_parser() -> argparse.ArgumentParser:
    """The command-line parser of ``run_ci_suite.py``."""
    parser = argparse.ArgumentParser(description="TokenSpeed Runtime CI Suite Runner")
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_MAPPING.keys(),
        required=True,
        help="Device backend to run tests on.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Test suite to run.",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
        help="The time limit for running one file in seconds (default: 1800).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails.",
    )
    parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    parser.add_argument(
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2).",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60).",
    )
    parser.add_argument(
        "--retry-timeout-increase",
        type=int,
        default=600,
        help="Additional timeout in seconds when retry is enabled (default: 600).",
    )
    return parser


def parse_suite_arguments(argv: List[str]) -> argparse.Namespace:
    """Parse and validate ``run_ci_suite.py`` arguments; exit on invalid ones."""
    parser = suite_argument_parser()
    args = parser.parse_args(argv)
    if (args.auto_partition_id is not None) != (args.auto_partition_size is not None):
        parser.error(
            "--auto-partition-id and --auto-partition-size must be specified together."
        )
    if args.auto_partition_size is not None:
        if args.auto_partition_size <= 0:
            parser.error("--auto-partition-size must be positive.")
        if not 0 <= args.auto_partition_id < args.auto_partition_size:
            parser.error(
                f"--auto-partition-id must be in range [0, {args.auto_partition_size}), "
                f"but got {args.auto_partition_id}"
            )
    return args


def suite_invocation(command: str) -> Optional[argparse.Namespace]:
    """The arguments of ``command`` when it only runs ``run_ci_suite.py``.

    A command qualifies when it is ``python``/``python3`` running the script
    by its repository path, with no variable assignments, shell operators or
    expansions. It is None for any other command. A qualifying command with
    invalid arguments exits, as the task itself would fail.
    """
    if not SHELL_EXPANSION_CHARACTERS.isdisjoint(command):
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
    if (
        len(tokens) < 2
        or Path(tokens[0]).name not in ("python", "python3")
        or tokens[1] != SUITE_SCRIPT
        or any(set(token) <= SHELL_OPERATOR_CHARACTERS for token in tokens)
    ):
        return None
    return parse_suite_arguments(tokens[2:])


def parse_changed_paths(text: str) -> List[str]:
    """The changed paths CI lists in ``text``, one per line."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def collect_runtime_tests(repo_root: Path) -> List[CIRegistry]:
    """Every CI registration under ``test/runtime``, with absolute file names.

    Files without a registration are skipped with a warning.
    """
    base_dir = repo_root / RUNTIME_TEST_DIRECTORY
    files = [
        path
        for path in sorted(glob.glob(str(base_dir / "**" / "*.py"), recursive=True))
        if not path.endswith(("/conftest.py", "/__init__.py", "/run_ci_suite.py"))
    ]
    return collect_tests(files, sanity_check=False)


def _repo_path(test: CIRegistry, repo_root: Path) -> str:
    return Path(os.path.relpath(test.filename, repo_root)).as_posix()


def tests_reached(
    registered: List[CIRegistry], changed_paths: List[str], repo_root: Path
) -> Optional[set[str]]:
    """The repository paths of the ``registered`` files the changes reach.

    It is None when ``changed_paths`` is empty or a change can reach any test,
    so that the suite does not narrow.
    """
    if not changed_paths:
        return None
    selected = select_runtime_tests(
        [_repo_path(test, repo_root) for test in registered], changed_paths, repo_root
    )
    return None if selected is None else set(selected)


def _runner_label_skip_reason(test: CIRegistry, runner_label: str) -> Optional[str]:
    """Return a non-None skip reason if `test` should be skipped on the
    runner identified by `runner_label`. None means the test is allowed.
    """
    if not test.disabled_on_runners or not runner_label:
        return None
    matched = [
        pat
        for pat in test.disabled_on_runners
        if fnmatch.fnmatchcase(runner_label, pat)
    ]
    if not matched:
        return None
    reason = test.disabled_on_runners_reason or (
        f"disabled on runner matching {matched[0]!r}"
    )
    return f"{reason} [runner={runner_label}, pattern={matched[0]}]"


def filter_tests(
    ci_tests: List[CIRegistry],
    device: DeviceBackend,
    suite: str,
    runner_label: str,
) -> tuple[List[CIRegistry], List[CIRegistry]]:
    """Filter tests by device backend, suite name, and runner label.

    Returns:
        tuple: (enabled_tests, skipped_tests)
    """
    ci_tests = [t for t in ci_tests if t.backend == device and t.suite == suite]

    valid_suites = PER_COMMIT_SUITES.get(device, [])

    if suite not in valid_suites:
        # stderr: the scan prints its matrix to stdout.
        print(
            f"Warning: Unknown suite {suite} for device {device.name}", file=sys.stderr
        )

    enabled_tests: List[CIRegistry] = []
    skipped_tests: List[CIRegistry] = []
    for t in ci_tests:
        if t.disabled is not None:
            skipped_tests.append(t)
            continue
        per_runner_reason = _runner_label_skip_reason(t, runner_label)
        if per_runner_reason is not None:
            # Surface the reason via the same `disabled` field so that
            # pretty_print_tests / step summary downstream do not need to
            # know about disabled_on_runners.
            skipped_tests.append(replace(t, disabled=per_runner_reason))
            continue
        enabled_tests.append(t)

    return enabled_tests, skipped_tests


def plan_suite(
    registered: List[CIRegistry],
    args: argparse.Namespace,
    runner_label: str,
    reached: Optional[set[str]],
    repo_root: Path,
) -> SuitePlan:
    """The test files that the invocation with ``args`` runs.

    Args:
        registered: Every runtime registration (``collect_runtime_tests``).
        args: The invocation's arguments (``parse_suite_arguments``).
        runner_label: The runner's label, matched against each file's
            ``disabled_on_runners`` (``CI_RUNNER_LABEL`` in a CI task).
        reached: The files the changed paths reach (``tests_reached``), or
            None to run the whole suite.
        repo_root: The checkout the file names lie in.
    """
    tests, skipped = filter_tests(
        registered, DEVICE_MAPPING[args.device], args.suite, runner_label
    )
    changed_selection = None
    if reached is not None:
        tests = [test for test in tests if _repo_path(test, repo_root) in reached]
        changed_selection = len(tests)
    if args.auto_partition_size:
        tests = auto_partition(tests, args.auto_partition_id, args.auto_partition_size)
    return SuitePlan(tests, skipped, changed_selection)
