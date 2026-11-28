#!/usr/bin/env python3
"""
TokenSpeed Runtime CI Suite Runner

Uses AST-based CI registration system (ci_system.ci_register) to collect, filter,
partition, and run test files. Test files register themselves via marker
functions like `register_cuda_ci(est_time=300, suite="runtime-1gpu")`. The
plan of which files run lives in ci_system/runtime_suite.py, which the CI scan
also uses to skip invocations that would run none.

Usage:
    python run_ci_suite.py --device cuda --suite runtime-1gpu
    python run_ci_suite.py --device cuda --suite runtime-1gpu --auto-partition-id 0 --auto-partition-size 2
"""

import os
import sys
from pathlib import Path
from typing import List

# Add test directory to path for importing CI system helpers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import CIRegistry
from ci_system.ci_utils import run_unittest_files
from ci_system.runtime_suite import (
    DEVICE_MAPPING,
    collect_runtime_tests,
    parse_changed_paths,
    parse_suite_arguments,
    plan_suite,
    tests_reached,
)
from ci_system.runtime_test_selection import CHANGED_PATHS_ENV


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
    repo_root = Path(__file__).resolve().parents[2]
    # Non-strict: files without CI registration are skipped.
    registered = collect_runtime_tests(repo_root)
    # On pull requests CI lists the changed paths; run only the files they reach.
    reached = tests_reached(
        registered,
        parse_changed_paths(os.environ.get(CHANGED_PATHS_ENV, "")),
        repo_root,
    )
    plan = plan_suite(
        registered, args, os.environ.get("CI_RUNNER_LABEL", ""), reached, repo_root
    )
    if plan.changed_selection is not None:
        print(f"Changed paths select {plan.changed_selection} test files.", flush=True)
    ci_tests = plan.tests
    skipped_tests = plan.skipped

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
    exit_code = run_a_suite(parse_suite_arguments(sys.argv[1:]))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
