#!/usr/bin/env python3
"""
TokenSpeed Runtime CI Suite Runner

Uses AST-based CI registration system (ci_system.ci_register) to collect, filter,
partition, and run test files. Test files register themselves via marker
functions like `register_cuda_ci(est_time=300, suite="runtime-1gpu")`.

Usage:
    python run_ci_suite.py --device cuda --suite runtime-1gpu
    python run_ci_suite.py --device cuda --suite runtime-1gpu --auto-partition-id 0 --auto-partition-size 2
"""

import argparse
import fnmatch
import glob
import os
import sys
from dataclasses import replace
from typing import List, Optional

# Add test directory to path for importing CI system helpers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import (
    CIRegistry,
    DeviceBackend,
    auto_partition,
    collect_tests,
)
from ci_system.ci_utils import run_unittest_files

DEVICE_MAPPING = {
    "cuda": DeviceBackend.CUDA,
}

# Per-commit test suites (run on every PR)
PER_COMMIT_SUITES = {
    DeviceBackend.CUDA: [
        "runtime-1gpu",
        "runtime-2gpu",
        "runtime-minimax-m2",
        "runtime-prefix-cache-e2e",
    ],
}


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
    runner_label: Optional[str] = None,
) -> tuple[List[CIRegistry], List[CIRegistry]]:
    """Filter tests by device backend, suite name, and (optionally) runner label.

    Returns:
        tuple: (enabled_tests, skipped_tests)
    """
    ci_tests = [t for t in ci_tests if t.backend == device and t.suite == suite]

    valid_suites = PER_COMMIT_SUITES.get(device, [])

    if suite not in valid_suites:
        print(f"Warning: Unknown suite {suite} for device {device.name}")

    if runner_label is None:
        runner_label = os.environ.get("CI_RUNNER_LABEL", "")

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


def pretty_print_tests(
    args, ci_tests: List[CIRegistry], skipped_tests: List[CIRegistry]
):
    """Print test information."""
    device = DEVICE_MAPPING[args.device]
    suite = args.suite
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} "
            f"(0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    msg = f"[Device={device.name}] [Suite={suite}] [Partition={partition_info}]\n"

    if skipped_tests:
        msg += f"⚠️  Skipped {len(skipped_tests)} test(s):\n"
        for t in skipped_tests:
            reason = t.disabled or "disabled"
            msg += f"  - {t.filename} (reason: {reason})\n"
        msg += "\n"

    if len(ci_tests) == 0:
        msg += f"No tests found for device={device.name}, suite={suite}\n"
        msg += "Skipping.\n"
    else:
        total_est_time = sum(t.est_time for t in ci_tests)
        msg += (
            f"✅ Enabled {len(ci_tests)} test(s) (est total {total_est_time:.1f}s):\n"
        )
        for t in ci_tests:
            msg += f"  - {t.filename} (est_time={t.est_time})\n"

    print(msg, flush=True)


def run_a_suite(args):
    """Collect, filter, partition, and run a test suite."""
    device = DEVICE_MAPPING[args.device]
    suite = args.suite
    auto_partition_id = args.auto_partition_id
    auto_partition_size = args.auto_partition_size

    # Scan all test files under test/runtime/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = [
        f
        for f in glob.glob(os.path.join(base_dir, "**", "*.py"), recursive=True)
        if not f.endswith("/conftest.py")
        and not f.endswith("/__init__.py")
        and not f.endswith("/run_ci_suite.py")
    ]

    # Non-strict: files without CI registration are skipped.
    sanity_check = False
    all_tests = collect_tests(files, sanity_check=sanity_check)
    ci_tests, skipped_tests = filter_tests(all_tests, device, suite)

    if auto_partition_size:
        ci_tests = auto_partition(ci_tests, auto_partition_id, auto_partition_size)

    pretty_print_tests(args, ci_tests, skipped_tests)

    if len(ci_tests) == 0:
        print("No tests to run. Exiting with success.", flush=True)
        return 0

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    return run_unittest_files(
        ci_tests,
        timeout_per_file=timeout,
        continue_on_error=args.continue_on_error,
        enable_retry=args.enable_retry,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
    )


def main():
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
    args = parser.parse_args()

    # Validate auto-partition arguments
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

    exit_code = run_a_suite(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
