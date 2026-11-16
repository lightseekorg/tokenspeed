# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU tests for the native L3 release validation runner.

These exercise the runner with real small subprocesses and temporary files;
no native Mooncake store, master or GPU is needed because the validated
command is injected explicitly, while the public CLI keeps the native suite
command fixed (verified by test_main_wires_the_fixed_native_command). The
process-group regression tests drive real process trees: a group member that
ignores SIGTERM, a leader that exits successfully leaving a child behind, a
repeated cancellation signal during cleanup, and a suite that cannot be
launched at all.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import l3_native_release_validation as runner
import pytest

ALL_TESTS = tuple(sorted(runner.EXPECTED_TEST_NAMES))

OK_BODY = "\n".join(
    [
        f"names = {list(ALL_TESTS)}",
        "print('running 5 tests')",
        "for name in names:",
        "    print(f'test {name} ... ok')",
        "print('')",
        "print('test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; '",
        "    '0 filtered out; finished in 0.02s')",
    ]
)

ZERO_TESTS_BODY = "\n".join(
    [
        "print('running 0 tests')",
        "print('test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; "
        "0 filtered out; finished in 0.00s')",
    ]
)

IGNORED_BODY = "\n".join(
    [
        "print('running 1 tests')",
        "print('test gamma ... ignored')",
        "print('test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; "
        "0 filtered out; finished in 0.01s')",
    ]
)

# The Rust tests print the skip marker yet still pass when the library is
# missing and MOONCAKE_REQUIRE_MASTER is unset; the runner must reject it.
SKIPPED_BODY = "\n".join(
    [
        "import sys",
        f"names = {list(ALL_TESTS)}",
        "print('skipped: TS_MOONCAKE_STORE_LIBRARY is unset', file=sys.stderr)",
        "print('running 5 tests')",
        "for name in names:",
        "    print(f'test {name} ... ok')",
        "print('test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; '",
        "    '0 filtered out; finished in 0.01s')",
    ]
)

# Passes on the first invocation, fails on every later one.
FAIL_AFTER_FIRST_BODY = "\n".join(
    [
        "import pathlib, sys",
        f"names = {list(ALL_TESTS)}",
        "counter = pathlib.Path(sys.argv[1])",
        "seen = int(counter.read_text()) if counter.exists() else 0",
        "counter.write_text(str(seen + 1))",
        "print('running 5 tests')",
        "if seen == 0:",
        "    for name in names:",
        "        print(f'test {name} ... ok')",
        "    print('test result: ok. 5 passed; 0 failed; 0 ignored; '",
        "        '0 measured; 0 filtered out; finished in 0.01s')",
        "else:",
        "    for name in names[:-1]:",
        "        print(f'test {name} ... ok')",
        "    print(f'test {names[-1]} ... FAILED')",
        "    print('test result: FAILED. 4 passed; 1 failed; 0 ignored; '",
        "        '0 measured; 0 filtered out; finished in 0.01s')",
        "    sys.exit(101)",
    ]
)

# Spawns a child in the same process group, records both PIDs, then sleeps.
SLOW_GROUP_BODY = "\n".join(
    [
        "import os, subprocess, sys, time",
        "open(sys.argv[1], 'w').write(str(os.getpid()))",
        "child = subprocess.Popen(['sleep', '1000'])",
        "open(sys.argv[2], 'w').write(str(child.pid))",
        "time.sleep(1000)",
    ]
)

# The child ignores SIGTERM: the stand-in for a master that outlives a
# SIGTERM to the group and must be SIGKILLed.
STUBBORN_GROUP_BODY = "\n".join(
    [
        "import os, subprocess, sys, time",
        "open(sys.argv[1], 'w').write(str(os.getpid()))",
        "child = subprocess.Popen([sys.executable, '-c'",
        "    'import signal, time'",
        "    '; signal.signal(signal.SIGTERM, signal.SIG_IGN)'",
        "    '; time.sleep(1000)'])",
        "open(sys.argv[2], 'w').write(str(child.pid))",
        "time.sleep(1000)",
    ]
)

# Spawns a plain child, records both PIDs, then exits successfully right
# away: the stand-in for a suite that leaks its master on a normal exit.
EARLY_EXIT_GROUP_BODY = "\n".join(
    [
        "import os, subprocess, sys",
        "open(sys.argv[1], 'w').write(str(os.getpid()))",
        "child = subprocess.Popen(['sleep', '1000'])",
        "open(sys.argv[2], 'w').write(str(child.pid))",
    ]
)

# Ignores SIGTERM itself, so cleanup only completes through SIGKILL after the
# grace wait; a second cancellation signal arrives during that wait.
TERM_PROOF_GROUP_BODY = "\n".join(
    [
        "import os, signal, subprocess, sys, time",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "open(sys.argv[1], 'w').write(str(os.getpid()))",
        "child = subprocess.Popen(['sleep', '1000'])",
        "open(sys.argv[2], 'w').write(str(child.pid))",
        "time.sleep(1000)",
    ]
)


def make_summary(
    summaries_seen: int,
    running_headers_seen: int,
    running: int | None,
    passed: int,
    failed: int,
    ignored: int,
    measured: int,
    filtered_out: int,
    test_names: tuple[str, ...],
    ok_test_names: tuple[str, ...],
    not_ok_tests: tuple[str, ...],
    skip_marker_seen: bool,
) -> runner.SuiteSummary:
    return runner.SuiteSummary(
        summaries_seen=summaries_seen,
        running_headers_seen=running_headers_seen,
        running=running,
        passed=passed,
        failed=failed,
        ignored=ignored,
        measured=measured,
        filtered_out=filtered_out,
        test_names=test_names,
        ok_test_names=ok_test_names,
        not_ok_tests=not_ok_tests,
        skip_marker_seen=skip_marker_seen,
    )


def make_prerequisites(tmp_path: Path) -> tuple[runner.Prerequisites, Path]:
    library = tmp_path / "libmooncake_store.so"
    library.write_bytes(b"\x7fELF fake store library")
    master = tmp_path / "mooncake_master"
    master.write_text("#!/bin/sh\nexit 0\n")
    master.chmod(0o755)
    prerequisites = runner.Prerequisites(
        library_path=str(library),
        master_mode="owned",
        master_binary=str(master),
        master_address=None,
        problems=[],
    )
    return prerequisites, library


def make_command(
    argv: list[str], cwd: Path, library: Path
) -> runner.NativeSuiteCommand:
    env = dict(os.environ)
    env[runner.REQUIRE_MASTER_ENV] = "1"
    env[runner.STORE_LIBRARY_ENV] = str(library)
    master = library.with_name(runner.MASTER_BINARY_NAME)
    env[runner.MASTER_BIN_ENV] = str(master)
    return runner.NativeSuiteCommand(
        argv=argv,
        cwd=str(cwd),
        env=env,
        env_overrides={
            runner.REQUIRE_MASTER_ENV: "1",
            runner.STORE_LIBRARY_ENV: str(library),
            runner.MASTER_BIN_ENV: str(master),
        },
    )


def make_config(
    tmp_path: Path,
    iterations: int,
    iteration_timeout_seconds: float,
    terminate_timeout_seconds: float,
    wall_clock_budget_seconds: float,
) -> runner.ValidationConfig:
    return runner.ValidationConfig(
        iterations=iterations,
        iteration_timeout_seconds=iteration_timeout_seconds,
        terminate_timeout_seconds=terminate_timeout_seconds,
        wall_clock_budget_seconds=wall_clock_budget_seconds,
        engine_dir=str(tmp_path),
        output_dir=str(tmp_path / "evidence"),
        source=runner.SourceIdentity(commit="abc123def456", dirty=False, label=None),
    )


def make_engine_dir(tmp_path: Path) -> Path:
    engine = tmp_path / "tokenspeed-engine"
    engine.mkdir()
    (engine / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
    return engine


def cli_args(
    tmp_path: Path,
    iterations: str,
    iteration_timeout: str,
    terminate_timeout: str,
    wall_clock_budget: str,
) -> list[str]:
    return [
        "--iterations",
        iterations,
        "--iteration-timeout",
        iteration_timeout,
        "--terminate-timeout",
        terminate_timeout,
        "--wall-clock-budget",
        wall_clock_budget,
        "--engine-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "evidence"),
        "--source-revision",
        "abc123def456",
    ]


def group_argv(body: str, parent_pidfile: Path, child_pidfile: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        body,
        str(parent_pidfile),
        str(child_pidfile),
    ]


def read_pid(path: Path) -> int:
    assert path.is_file(), f"pid file {path} was not written"
    return int(path.read_text().strip())


def assert_pid_gone(pid: int) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} still exists")


def read_report(output_dir: Path) -> dict:
    return json.loads((output_dir / "report.json").read_text())


def run_ok_validation(tmp_path: Path, iterations: int) -> tuple[str, Path]:
    prerequisites, library = make_prerequisites(tmp_path)
    command = make_command([sys.executable, "-c", OK_BODY], tmp_path, library)
    config = make_config(tmp_path, iterations, 30.0, 2.0, 100000.0)
    versions = {
        "tokenspeed-mooncake": "1.2.3",
        "cargo": "cargo 1.85.0",
        "rustc": runner.UNAVAILABLE,
    }
    overall = runner.run_validation(
        config, command, prerequisites, versions, runner.run_monitored
    )
    return overall, Path(config.output_dir)


def test_native_suite_argv_is_fixed():
    assert runner.native_suite_argv() == [
        "cargo",
        "test",
        "--locked",
        "-p",
        "tokenspeed-l3",
        "--test",
        "mooncake",
        "--",
        "--test-threads=1",
        "--nocapture",
    ]


def test_expected_test_names_match_the_native_suite():
    suite = (
        Path(__file__).resolve().parents[2]
        / "tokenspeed-engine"
        / "crates"
        / "ts-l3"
        / "tests"
        / "mooncake.rs"
    )
    if not suite.is_file():
        pytest.skip("the native suite source is not part of this checkout")
    declared = set(runner.EXPECTED_TEST_NAMES)
    assert declared, "the expected test set must not be empty"
    for name in declared:
        assert f"fn {name}(" in suite.read_text(), (
            f"expected test {name} is not in the native suite; update "
            "EXPECTED_TEST_NAMES with the suite"
        )


def test_parse_suite_summary_ok():
    output = subprocess.run(
        [sys.executable, "-c", OK_BODY],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout
    summary = runner.parse_suite_summary(output)
    assert summary.summaries_seen == 1
    assert summary.running_headers_seen == 1
    assert summary.running == 5
    assert (summary.passed, summary.failed, summary.ignored) == (5, 0, 0)
    assert summary.measured == 0
    assert summary.filtered_out == 0
    assert summary.test_names == ALL_TESTS
    assert summary.ok_test_names == ALL_TESTS
    assert summary.not_ok_tests == ()
    assert not summary.skip_marker_seen


@pytest.mark.parametrize(
    "body,summaries_seen,running,passed,ignored,skip_marker_seen",
    [
        (ZERO_TESTS_BODY, 1, 0, 0, 0, False),
        (IGNORED_BODY, 1, 1, 0, 1, False),
        (SKIPPED_BODY, 1, 5, 5, 0, True),
    ],
)
def test_parse_suite_summary_variants(
    body: str,
    summaries_seen: int,
    running: int,
    passed: int,
    ignored: int,
    skip_marker_seen: bool,
):
    output = subprocess.run(
        [sys.executable, "-c", body],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout
    summary = runner.parse_suite_summary(output)
    assert summary.summaries_seen == summaries_seen
    assert summary.running == running
    assert summary.passed == passed
    assert summary.ignored == ignored
    assert summary.skip_marker_seen == skip_marker_seen


def test_parse_suite_summary_incomplete_output():
    summary = runner.parse_suite_summary("error: could not compile `ts-l3`\n")
    assert summary.summaries_seen == 0
    assert summary.running_headers_seen == 0
    assert summary.running is None
    assert summary.test_names == ()


def test_parse_suite_summary_without_a_running_header():
    output = (
        "test alpha ... ok\n"
        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
        "0 filtered out; finished in 0.01s\n"
    )
    summary = runner.parse_suite_summary(output)
    assert summary.summaries_seen == 1
    assert summary.running_headers_seen == 0
    assert summary.running is None
    reasons = runner.iteration_rejections(0, False, summary)
    assert any("running-tests header" in reason for reason in reasons)


def test_parse_suite_summary_with_repeated_running_headers():
    output = (
        "running 5 tests\n"
        "running 5 tests\n"
        "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; "
        "0 filtered out; finished in 0.01s\n"
    )
    summary = runner.parse_suite_summary(output)
    assert summary.running_headers_seen == 2
    assert summary.running is None
    reasons = runner.iteration_rejections(0, False, summary)
    assert any("running-tests headers" in reason for reason in reasons)


def test_parse_suite_summary_detects_a_prefixed_skip_marker():
    # With --nocapture the tests' stderr can be glued to other output.
    output = subprocess.run(
        [sys.executable, "-c", SKIPPED_BODY],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout
    assert runner.parse_suite_summary(output).skip_marker_seen
    prefixed = f"partial line without newline\nxskipped: reason\n{output}"
    assert runner.parse_suite_summary(prefixed).skip_marker_seen


def test_parse_suite_summary_records_test_identities():
    output = (
        "running 5 tests\n"
        "test alpha ... ok\n"
        "test beta ... FAILED\n"
        "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; "
        "0 filtered out; finished in 0.01s\n"
    )
    summary = runner.parse_suite_summary(output)
    assert summary.test_names == ("alpha", "beta")
    assert summary.ok_test_names == ("alpha",)
    assert summary.not_ok_tests == ("beta (FAILED)",)


def test_iteration_rejections_accepts_a_clean_pass():
    reasons = runner.iteration_rejections(
        0, False, make_summary(1, 1, 5, 5, 0, 0, 0, 0, ALL_TESTS, ALL_TESTS, (), False)
    )
    assert reasons == []


@pytest.mark.parametrize(
    "returncode,timed_out,summary,expected_fragment",
    [
        (
            101,
            False,
            make_summary(
                1, 1, 1, 0, 1, 0, 0, 0, ("delta",), (), ("delta (FAILED)",), False
            ),
            "return code 101",
        ),
        (
            None,
            True,
            make_summary(0, 0, None, 0, 0, 0, 0, 0, (), (), (), False),
            "timeout",
        ),
        (
            0,
            False,
            make_summary(0, 1, 5, 0, 0, 0, 0, 0, (), (), (), False),
            "no test-result",
        ),
        (
            0,
            False,
            make_summary(2, 1, 5, 5, 0, 0, 0, 0, ALL_TESTS, ALL_TESTS, (), False),
            "summaries",
        ),
        (
            0,
            False,
            make_summary(1, 0, None, 5, 0, 0, 0, 0, ALL_TESTS, ALL_TESTS, (), False),
            "running-tests header",
        ),
        (
            0,
            False,
            make_summary(1, 2, None, 5, 0, 0, 0, 0, ALL_TESTS, ALL_TESTS, (), False),
            "running-tests headers",
        ),
        (
            0,
            False,
            make_summary(1, 1, 0, 0, 0, 0, 0, 0, (), (), (), False),
            "zero tests",
        ),
        (
            0,
            False,
            make_summary(
                1, 1, 1, 1, 1, 0, 0, 0, ("delta",), (), ("delta (FAILED)",), False
            ),
            "failed tests",
        ),
        (
            0,
            False,
            make_summary(
                1, 1, 1, 0, 0, 1, 0, 0, ("gamma",), (), ("gamma (ignored)",), False
            ),
            "ignored",
        ),
        (
            0,
            False,
            make_summary(1, 1, 1, 0, 0, 0, 1, 0, ("beta",), ("beta",), (), False),
            "measured",
        ),
        (
            0,
            False,
            make_summary(1, 1, 1, 0, 0, 0, 0, 1, (), (), (), False),
            "filtered out",
        ),
        (
            0,
            False,
            make_summary(1, 1, 3, 2, 0, 0, 0, 0, ALL_TESTS, ALL_TESTS, (), False),
            "incomplete",
        ),
        (
            0,
            False,
            make_summary(1, 1, 5, 5, 0, 0, 0, 0, ALL_TESTS, ALL_TESTS, (), True),
            "skip marker",
        ),
        (
            0,
            False,
            make_summary(
                1, 1, 5, 5, 0, 0, 0, 0, ALL_TESTS[:-1], ALL_TESTS[:-1], (), False
            ),
            "missing expected tests",
        ),
        (
            0,
            False,
            make_summary(
                1,
                1,
                5,
                5,
                0,
                0,
                0,
                0,
                ALL_TESTS + ("impostor",),
                ALL_TESTS + ("impostor",),
                (),
                False,
            ),
            "unexpected tests",
        ),
        (
            0,
            False,
            make_summary(
                1,
                1,
                5,
                4,
                1,
                0,
                0,
                0,
                ALL_TESTS,
                ALL_TESTS[:-1],
                (f"{ALL_TESTS[-1]} (FAILED)",),
                False,
            ),
            "did not pass",
        ),
    ],
)
def test_iteration_rejections_fail_closed(
    returncode: int | None,
    timed_out: bool,
    summary: runner.SuiteSummary,
    expected_fragment: str,
):
    reasons = runner.iteration_rejections(returncode, timed_out, summary)
    assert reasons, "a dubious run must be rejected"
    assert any(expected_fragment in reason for reason in reasons)


def test_run_monitored_captures_output_and_exit_code(tmp_path):
    log = tmp_path / "out.log"
    env = dict(os.environ)
    env["SUITE_MARKER"] = "marker-123"
    outcome = runner.run_monitored(
        [sys.executable, "-c", "import os; print(os.environ['SUITE_MARKER'])"],
        str(tmp_path),
        env,
        30.0,
        2.0,
        log,
    )
    assert outcome.returncode == 0
    assert not outcome.timed_out
    assert outcome.elapsed_seconds >= 0
    assert log.read_text().strip() == "marker-123"


def test_run_monitored_propagates_a_nonzero_exit_code(tmp_path):
    outcome = runner.run_monitored(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        str(tmp_path),
        dict(os.environ),
        30.0,
        2.0,
        tmp_path / "out.log",
    )
    assert outcome.returncode == 7
    assert not outcome.timed_out


def test_run_monitored_timeout_terminates_the_whole_group(tmp_path):
    parent_pidfile = tmp_path / "parent.pid"
    child_pidfile = tmp_path / "child.pid"
    started = time.monotonic()
    outcome = runner.run_monitored(
        group_argv(SLOW_GROUP_BODY, parent_pidfile, child_pidfile),
        str(tmp_path),
        dict(os.environ),
        1.0,
        2.0,
        tmp_path / "out.log",
    )
    elapsed = time.monotonic() - started
    assert outcome.timed_out
    assert elapsed < 30.0
    assert outcome.returncode is not None
    # The direct child is reaped; the grandchild in the same group (the
    # stand-in for a master/provider child) is signalled and reaped too.
    assert_pid_gone(read_pid(parent_pidfile))
    assert_pid_gone(read_pid(child_pidfile))


def test_run_monitored_kills_a_stubborn_group_member_on_timeout(tmp_path):
    # The leader dies on SIGTERM, but the group member ignores it; cleanup
    # must escalate to SIGKILL instead of returning after the leader's reap.
    parent_pidfile = tmp_path / "parent.pid"
    child_pidfile = tmp_path / "child.pid"
    outcome = runner.run_monitored(
        group_argv(STUBBORN_GROUP_BODY, parent_pidfile, child_pidfile),
        str(tmp_path),
        dict(os.environ),
        1.0,
        2.0,
        tmp_path / "out.log",
    )
    assert outcome.timed_out
    assert outcome.returncode is not None
    assert_pid_gone(read_pid(parent_pidfile))
    assert_pid_gone(read_pid(child_pidfile))


def test_run_monitored_sweeps_leftovers_after_a_successful_leader(tmp_path):
    # The leader exits 0 leaving a child behind (a suite that leaked its
    # master); the success path must sweep the group too.
    parent_pidfile = tmp_path / "parent.pid"
    child_pidfile = tmp_path / "child.pid"
    outcome = runner.run_monitored(
        group_argv(EARLY_EXIT_GROUP_BODY, parent_pidfile, child_pidfile),
        str(tmp_path),
        dict(os.environ),
        30.0,
        2.0,
        tmp_path / "out.log",
    )
    assert outcome.returncode == 0
    assert not outcome.timed_out
    assert outcome.elapsed_seconds < 30.0
    assert_pid_gone(read_pid(parent_pidfile))
    assert_pid_gone(read_pid(child_pidfile))


def test_prepare_output_dir_creates_and_refuses_overwrite(tmp_path):
    fresh = tmp_path / "new" / "nested"
    assert runner.prepare_output_dir(str(fresh)) == fresh
    assert fresh.is_dir()
    # An existing empty directory is reusable.
    assert runner.prepare_output_dir(str(fresh)) == fresh
    (fresh / "report.json").write_text("{}")
    with pytest.raises(runner.UsageError, match="not empty"):
        runner.prepare_output_dir(str(fresh))
    assert (fresh / "report.json").read_text() == "{}"
    file_path = tmp_path / "a-file"
    file_path.write_text("x")
    with pytest.raises(runner.UsageError, match="not a directory"):
        runner.prepare_output_dir(str(file_path))


def test_check_prerequisites_reports_every_gap(tmp_path):
    result = runner.check_prerequisites({"PATH": ""}, str(tmp_path))
    assert result.library_path is None
    assert result.master_mode == "owned"
    joined = "\n".join(result.problems)
    assert "TS_MOONCAKE_STORE_LIBRARY" in joined
    assert "mooncake_master" in joined
    assert "cargo" in joined
    assert "Cargo.toml" in joined


def test_check_prerequisites_owned_master_binary(tmp_path):
    engine = make_engine_dir(tmp_path)
    library = tmp_path / "libmooncake_store.so"
    library.write_bytes(b"\x7fELF fake store library")
    env = {
        "PATH": os.environ.get("PATH", ""),
        runner.STORE_LIBRARY_ENV: str(library),
    }
    # No master binary beside the library yet.
    result = runner.check_prerequisites(env, str(engine))
    assert result.master_mode == "owned"
    assert any("master binary" in p for p in result.problems)
    # A non-executable master binary is not usable.
    master = tmp_path / "mooncake_master"
    master.write_text("#!/bin/sh\nexit 0\n")
    result = runner.check_prerequisites(env, str(engine))
    assert any("not executable" in p for p in result.problems)
    master.chmod(0o755)
    result = runner.check_prerequisites(env, str(engine))
    assert result.problems == []
    assert result.master_binary == str(master)
    # TS_MOONCAKE_MASTER_BIN wins over the library sibling.
    other = tmp_path / "other-master"
    other.write_text("#!/bin/sh\nexit 0\n")
    other.chmod(0o755)
    env[runner.MASTER_BIN_ENV] = str(other)
    result = runner.check_prerequisites(env, str(engine))
    assert result.master_binary == str(other)
    # MOONCAKE_MASTER names a running master instead of a binary.
    env[runner.EXTERNAL_MASTER_ENV] = "10.0.0.9:50051"
    result = runner.check_prerequisites(env, str(engine))
    assert result.master_mode == "external"
    assert result.master_address == "10.0.0.9:50051"


def test_check_prerequisites_external_mode_still_needs_the_binary(tmp_path):
    # The suite's tenant and restart tests start their own master even when
    # MOONCAKE_MASTER names an external one.
    engine = make_engine_dir(tmp_path)
    library = tmp_path / "libmooncake_store.so"
    library.write_bytes(b"\x7fELF fake store library")
    env = {
        "PATH": os.environ.get("PATH", ""),
        runner.STORE_LIBRARY_ENV: str(library),
        runner.EXTERNAL_MASTER_ENV: "10.0.0.9:50051",
    }
    result = runner.check_prerequisites(env, str(engine))
    assert result.master_mode == "external"
    assert any("master binary" in p for p in result.problems)
    master = tmp_path / "mooncake_master"
    master.write_text("#!/bin/sh\nexit 0\n")
    master.chmod(0o755)
    result = runner.check_prerequisites(env, str(engine))
    assert result.problems == []
    assert result.master_mode == "external"
    assert result.master_binary == str(master)


def test_check_prerequisites_rejects_blank_master_settings(tmp_path):
    engine = make_engine_dir(tmp_path)
    library = tmp_path / "libmooncake_store.so"
    library.write_bytes(b"\x7fELF fake store library")
    master = tmp_path / "mooncake_master"
    master.write_text("#!/bin/sh\nexit 0\n")
    master.chmod(0o755)
    env = {
        "PATH": os.environ.get("PATH", ""),
        runner.STORE_LIBRARY_ENV: str(library),
        runner.EXTERNAL_MASTER_ENV: "   ",
    }
    result = runner.check_prerequisites(env, str(engine))
    assert any("set but empty" in p for p in result.problems)
    env = {
        "PATH": os.environ.get("PATH", ""),
        runner.STORE_LIBRARY_ENV: str(library),
        runner.MASTER_BIN_ENV: "  ",
    }
    result = runner.check_prerequisites(env, str(engine))
    assert any("set but empty" in p for p in result.problems)


def test_check_prerequisites_resolves_paths_to_absolute(tmp_path, monkeypatch):
    # Relative settings are resolved before the suite launches in engine_dir.
    engine = make_engine_dir(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    library = bundle / "libmooncake_store.so"
    library.write_bytes(b"\x7fELF fake store library")
    master = bundle / "mooncake_master"
    master.write_text("#!/bin/sh\nexit 0\n")
    master.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    env = {
        "PATH": os.environ.get("PATH", ""),
        runner.STORE_LIBRARY_ENV: "bundle/libmooncake_store.so",
    }
    result = runner.check_prerequisites(env, str(engine))
    assert result.problems == []
    assert result.library_path == str(library.resolve())
    assert result.master_binary == str(master.resolve())


def test_build_suite_command_strictly_sets_require_master(tmp_path):
    prerequisites, library = make_prerequisites(tmp_path)
    master = library.with_name(runner.MASTER_BINARY_NAME)
    base_env = {runner.REQUIRE_MASTER_ENV: "0", "UNRELATED": "kept"}
    command = runner.build_suite_command(base_env, prerequisites, "engine-dir")
    assert command.argv == runner.native_suite_argv()
    assert command.env[runner.REQUIRE_MASTER_ENV] == "1"
    assert command.env[runner.STORE_LIBRARY_ENV] == str(library)
    assert command.env[runner.MASTER_BIN_ENV] == str(master)
    assert command.env["UNRELATED"] == "kept"
    assert command.env_overrides == {
        runner.REQUIRE_MASTER_ENV: "1",
        runner.STORE_LIBRARY_ENV: str(library),
        runner.MASTER_BIN_ENV: str(master),
    }
    # The caller's mapping is never mutated.
    assert base_env[runner.REQUIRE_MASTER_ENV] == "0"


def test_run_validation_success_writes_durable_evidence(tmp_path):
    overall, output_dir = run_ok_validation(tmp_path, 2)
    assert overall == runner.STATUS_PASSED
    report = read_report(output_dir)
    assert report["schema_version"] == 2
    assert report["overall_status"] == "passed"
    assert report["stop_reason"] == "all 2 iterations passed"
    assert report["source"] == {
        "commit": "abc123def456",
        "dirty": False,
        "label": None,
    }
    assert report["versions"] == {
        "tokenspeed-mooncake": "1.2.3",
        "cargo": "cargo 1.85.0",
        "rustc": runner.UNAVAILABLE,
    }
    assert report["platform"]["system"]
    assert report["platform"]["python_version"]
    assert report["iterations_planned"] == 2
    assert report["iterations_completed"] == 2
    assert report["receipts"] == ["iteration-001.json", "iteration-002.json"]
    assert report["prerequisites"]["ok"] is True
    assert report["config"]["argv"] == [sys.executable, "-c", OK_BODY]
    assert report["config"]["master_mode"] == "owned"
    assert report["config"]["wall_clock_budget_seconds"] == 100000.0
    assert report["config"]["setup_margin_seconds"] == runner.SETUP_MARGIN_SECONDS
    assert set(report["config"]["env_overrides"]) == {
        runner.REQUIRE_MASTER_ENV,
        runner.STORE_LIBRARY_ENV,
        runner.MASTER_BIN_ENV,
    }
    receipt = json.loads((output_dir / "iteration-001.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["returncode"] == 0
    assert receipt["timed_out"] is False
    assert receipt["summary"]["passed"] == 5
    assert receipt["summary"]["ok_test_names"] == list(ALL_TESTS)
    assert receipt["rejection_reasons"] == []
    assert receipt["env_overrides"][runner.REQUIRE_MASTER_ENV] == "1"
    log = (output_dir / receipt["log_file"]).read_text()
    assert "test result: ok. 5 passed" in log


def test_run_validation_fails_fast_after_a_repeated_failure(tmp_path):
    prerequisites, library = make_prerequisites(tmp_path)
    counter = tmp_path / "counter"
    command = make_command(
        [sys.executable, "-c", FAIL_AFTER_FIRST_BODY, str(counter)],
        tmp_path,
        library,
    )
    config = make_config(tmp_path, 3, 30.0, 2.0, 100000.0)
    overall = runner.run_validation(
        config, command, prerequisites, {"cargo": "test"}, runner.run_monitored
    )
    assert overall == runner.STATUS_FAILED
    output_dir = Path(config.output_dir)
    report = read_report(output_dir)
    assert report["overall_status"] == "failed"
    assert report["iterations_completed"] == 1
    assert report["receipts"] == ["iteration-001.json", "iteration-002.json"]
    assert "iteration 2 failed" in report["stop_reason"]
    second = json.loads((output_dir / "iteration-002.json").read_text())
    assert second["status"] == "failed"
    assert second["returncode"] == 101
    assert any("return code 101" in r for r in second["rejection_reasons"])
    assert not (output_dir / "iteration-003.json").exists()
    # The failed iteration's raw output is still preserved.
    assert "FAILED" in (output_dir / "iteration-002.log").read_text()


@pytest.mark.parametrize(
    "body,expected_fragment",
    [
        (ZERO_TESTS_BODY, "zero tests"),
        (IGNORED_BODY, "ignored"),
        (SKIPPED_BODY, "skip marker"),
    ],
)
def test_run_validation_rejects_dubious_zero_exit_runs(
    tmp_path, body: str, expected_fragment: str
):
    prerequisites, library = make_prerequisites(tmp_path)
    command = make_command([sys.executable, "-c", body], tmp_path, library)
    config = make_config(tmp_path, 1, 30.0, 2.0, 100000.0)
    overall = runner.run_validation(
        config, command, prerequisites, {"cargo": "test"}, runner.run_monitored
    )
    assert overall == runner.STATUS_FAILED
    report = read_report(Path(config.output_dir))
    assert report["overall_status"] == "failed"
    assert report["iterations_completed"] == 0
    assert expected_fragment in report["stop_reason"]


def test_run_validation_launch_error_never_reports_a_pass(tmp_path):
    # The review's false-pass repro: the suite binary cannot be exec'ed, so
    # the launch raises OSError; the report must fail closed with a receipt.
    fake_cargo = tmp_path / "cargo"
    fake_cargo.write_text("definitely not an executable\n")
    fake_cargo.chmod(0o755)
    prerequisites, library = make_prerequisites(tmp_path)
    command = make_command([str(fake_cargo), "test"], tmp_path, library)
    config = make_config(tmp_path, 3, 30.0, 2.0, 100000.0)
    overall = runner.run_validation(
        config, command, prerequisites, {"cargo": "test"}, runner.run_monitored
    )
    assert overall == runner.STATUS_FAILED
    output_dir = Path(config.output_dir)
    report = read_report(output_dir)
    # The exact false pass from the review: overall "passed", stop reason
    # "all 3 iterations passed", zero completed, no receipt. None may recur.
    assert report["overall_status"] == "failed"
    assert report["stop_reason"] != "all 3 iterations passed"
    assert report["iterations_completed"] == 0
    assert report["receipts"] == ["iteration-001.json"]
    receipt = json.loads((output_dir / "iteration-001.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["returncode"] is None
    assert any("launched" in r for r in receipt["rejection_reasons"])
    assert not (output_dir / "iteration-002.json").exists()


def test_run_validation_unexpected_executor_error_fails_closed(tmp_path):
    def explode(
        argv: list[str],
        cwd: str,
        env: dict[str, str],
        timeout_seconds: float,
        terminate_timeout_seconds: float,
        log_path: Path,
    ) -> runner.ProcessOutcome:
        raise RuntimeError("boom")

    prerequisites, library = make_prerequisites(tmp_path)
    command = make_command([sys.executable, "-c", OK_BODY], tmp_path, library)
    config = make_config(tmp_path, 2, 30.0, 2.0, 100000.0)
    overall = runner.run_validation(
        config, command, prerequisites, {"cargo": "test"}, explode
    )
    assert overall == runner.STATUS_FAILED
    report = read_report(Path(config.output_dir))
    assert report["overall_status"] == "failed"
    assert report["stop_reason"] != "all 2 iterations passed"
    assert "unexpected error" in report["stop_reason"]
    assert "boom" in report["stop_reason"]
    assert report["iterations_completed"] == 0
    assert report["receipts"] == []


def test_run_validation_timeout_terminates_the_owned_group(tmp_path):
    prerequisites, library = make_prerequisites(tmp_path)
    parent_pidfile = tmp_path / "parent.pid"
    child_pidfile = tmp_path / "child.pid"
    command = make_command(
        group_argv(SLOW_GROUP_BODY, parent_pidfile, child_pidfile),
        tmp_path,
        library,
    )
    config = make_config(tmp_path, 2, 1.0, 2.0, 100000.0)
    overall = runner.run_validation(
        config, command, prerequisites, {"cargo": "test"}, runner.run_monitored
    )
    assert overall == runner.STATUS_TIMEOUT
    output_dir = Path(config.output_dir)
    report = read_report(output_dir)
    assert report["overall_status"] == "timeout"
    assert report["receipts"] == ["iteration-001.json"]
    receipt = json.loads((output_dir / "iteration-001.json").read_text())
    assert receipt["status"] == "timeout"
    assert receipt["timed_out"] is True
    assert any("timeout" in r for r in receipt["rejection_reasons"])
    assert_pid_gone(read_pid(parent_pidfile))
    assert_pid_gone(read_pid(child_pidfile))


def run_interrupted(
    tmp_path: Path,
    body: str,
    terminate_timeout_seconds: float,
    signals: list[tuple[signal.Signals, float]],
) -> tuple[str, Path, int, int]:
    prerequisites, library = make_prerequisites(tmp_path)
    parent_pidfile = tmp_path / "parent.pid"
    child_pidfile = tmp_path / "child.pid"
    command = make_command(
        group_argv(body, parent_pidfile, child_pidfile), tmp_path, library
    )
    config = make_config(tmp_path, 1, 60.0, terminate_timeout_seconds, 100000.0)

    def send_signals() -> None:
        deadline = time.monotonic() + 10.0
        while not parent_pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        for sig, delay in signals:
            time.sleep(delay)
            os.kill(os.getpid(), sig)

    thread = threading.Thread(target=send_signals, daemon=True)
    thread.start()
    overall = runner.run_validation(
        config, command, prerequisites, {"cargo": "test"}, runner.run_monitored
    )
    thread.join(timeout=5.0)
    return (
        overall,
        Path(config.output_dir),
        read_pid(parent_pidfile),
        read_pid(child_pidfile),
    )


def assert_interrupted_run_cleaned_up(
    overall: str, output_dir: Path, parent_pid: int, child_pid: int
) -> None:
    assert overall == runner.STATUS_INTERRUPTED
    report = read_report(output_dir)
    assert report["overall_status"] == "interrupted"
    assert report["receipts"] == ["iteration-001.json"]
    receipt = json.loads((output_dir / "iteration-001.json").read_text())
    assert receipt["status"] == "interrupted"
    assert receipt["summary"] is None
    # The interrupted iteration's process group was still terminated.
    assert_pid_gone(parent_pid)
    assert_pid_gone(child_pid)


def test_run_validation_interrupted_by_sigint(tmp_path):
    overall, output_dir, parent_pid, child_pid = run_interrupted(
        tmp_path, SLOW_GROUP_BODY, 2.0, [(signal.SIGINT, 0.0)]
    )
    assert_interrupted_run_cleaned_up(overall, output_dir, parent_pid, child_pid)


def test_run_validation_interrupted_by_sigterm(tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    overall, output_dir, parent_pid, child_pid = run_interrupted(
        tmp_path, SLOW_GROUP_BODY, 2.0, [(signal.SIGTERM, 0.0)]
    )
    assert_interrupted_run_cleaned_up(overall, output_dir, parent_pid, child_pid)
    # The runner restores the previous SIGTERM handler when it leaves.
    assert signal.getsignal(signal.SIGTERM) is previous


def test_run_validation_cleans_up_under_a_repeated_signal(tmp_path):
    # The leader ignores SIGTERM, so cleanup runs the full grace wait; the
    # second signal lands mid-cleanup and must not abort it: the group is
    # SIGKILLed and reaped, and the report is still written.
    overall, output_dir, parent_pid, child_pid = run_interrupted(
        tmp_path,
        TERM_PROOF_GROUP_BODY,
        3.0,
        [(signal.SIGINT, 0.0), (signal.SIGTERM, 0.4)],
    )
    assert_interrupted_run_cleaned_up(overall, output_dir, parent_pid, child_pid)


def test_run_validation_prerequisite_failure_still_writes_report(tmp_path):
    prerequisites = runner.Prerequisites(
        library_path=None,
        master_mode="owned",
        master_binary=None,
        master_address=None,
        problems=["TS_MOONCAKE_STORE_LIBRARY is not set"],
    )
    command = runner.NativeSuiteCommand(
        argv=runner.native_suite_argv(),
        cwd=str(tmp_path),
        env=dict(os.environ),
        env_overrides={runner.REQUIRE_MASTER_ENV: "1"},
    )
    config = make_config(tmp_path, 2, 30.0, 2.0, 100000.0)
    overall = runner.run_validation(
        config, command, prerequisites, {"cargo": "test"}, runner.run_monitored
    )
    assert overall == runner.STATUS_FAILED
    report = read_report(Path(config.output_dir))
    assert report["overall_status"] == "failed"
    assert report["prerequisites"]["ok"] is False
    assert report["prerequisites"]["problems"] == [
        "TS_MOONCAKE_STORE_LIBRARY is not set"
    ]
    assert report["iterations_completed"] == 0
    assert report["receipts"] == []
    assert "prerequisite" in report["stop_reason"]


def test_run_validation_refuses_structurally_invalid_config(tmp_path):
    config = make_config(tmp_path, 0, 30.0, 2.0, 100000.0)
    with pytest.raises(runner.UsageError, match="iterations"):
        runner.run_validation(
            config,
            make_command([sys.executable, "-c", OK_BODY], tmp_path, tmp_path),
            runner.Prerequisites(None, None, None, None, []),
            {},
            runner.run_monitored,
        )
    config = make_config(tmp_path, 1, 0.0, 2.0, 100000.0)
    with pytest.raises(runner.UsageError, match="iteration timeout"):
        runner.run_validation(
            config,
            make_command([sys.executable, "-c", OK_BODY], tmp_path, tmp_path),
            runner.Prerequisites(None, None, None, None, []),
            {},
            runner.run_monitored,
        )
    config = make_config(tmp_path, 1, 30.0, 2.0, 0.0)
    with pytest.raises(runner.UsageError, match="wall-clock budget"):
        runner.run_validation(
            config,
            make_command([sys.executable, "-c", OK_BODY], tmp_path, tmp_path),
            runner.Prerequisites(None, None, None, None, []),
            {},
            runner.run_monitored,
        )


def test_run_validation_never_overwrites_earlier_evidence(tmp_path):
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    (output_dir / "report.json").write_text('{"overall_status": "passed"}')
    config = make_config(tmp_path, 1, 30.0, 2.0, 100000.0)
    with pytest.raises(runner.UsageError, match="not empty"):
        runner.run_validation(
            config,
            make_command([sys.executable, "-c", OK_BODY], tmp_path, tmp_path),
            runner.Prerequisites(None, None, None, None, []),
            {},
            runner.run_monitored,
        )
    assert json.loads((output_dir / "report.json").read_text()) == {
        "overall_status": "passed"
    }


def test_report_records_only_explicit_environment(tmp_path, monkeypatch):
    marker = "s3cr3t-marker-that-must-not-leak"
    monkeypatch.setenv("TOKENSPEED_FAKE_SECRET", marker)
    overall, output_dir = run_ok_validation(tmp_path, 1)
    assert overall == runner.STATUS_PASSED
    report_text = (output_dir / "report.json").read_text()
    receipt_text = (output_dir / "iteration-001.json").read_text()
    assert marker not in report_text
    assert marker not in receipt_text
    report = json.loads(report_text)
    assert set(report["config"]["env_overrides"]) == {
        runner.REQUIRE_MASTER_ENV,
        runner.STORE_LIBRARY_ENV,
        runner.MASTER_BIN_ENV,
    }


def test_main_wires_the_fixed_native_command(tmp_path, monkeypatch):
    engine = make_engine_dir(tmp_path)
    prerequisites, library = make_prerequisites(tmp_path)
    monkeypatch.setenv(runner.STORE_LIBRARY_ENV, str(library))
    monkeypatch.delenv(runner.MASTER_BIN_ENV, raising=False)
    monkeypatch.delenv(runner.EXTERNAL_MASTER_ENV, raising=False)
    captured = {}

    def capture_run_validation(
        config: runner.ValidationConfig,
        command: runner.NativeSuiteCommand,
        prerequisites: runner.Prerequisites,
        versions: dict[str, str],
        executor: runner.ProcessExecutor,
    ) -> str:
        captured["config"] = config
        captured["command"] = command
        captured["prerequisites"] = prerequisites
        captured["versions"] = versions
        captured["executor"] = executor
        return runner.STATUS_PASSED

    monkeypatch.setattr(runner, "run_validation", capture_run_validation)
    exit_code = runner.main(
        [
            "--iterations",
            "1",
            "--iteration-timeout",
            "900",
            "--terminate-timeout",
            "20",
            "--wall-clock-budget",
            "1800",
            "--engine-dir",
            str(engine),
            "--output-dir",
            str(tmp_path / "evidence"),
            "--source-revision",
            "cafef00d",
        ]
    )
    assert exit_code == 0
    command = captured["command"]
    assert command.argv == runner.native_suite_argv()
    assert command.cwd == str(engine)
    assert command.env[runner.REQUIRE_MASTER_ENV] == "1"
    assert command.env[runner.STORE_LIBRARY_ENV] == str(library)
    master = library.with_name(runner.MASTER_BINARY_NAME)
    assert command.env[runner.MASTER_BIN_ENV] == str(master)
    assert captured["config"].iterations == 1
    assert captured["config"].wall_clock_budget_seconds == 1800.0
    # The label is recorded, but the tested commit comes from the checkout.
    assert captured["config"].source.label == "cafef00d"
    assert captured["config"].source.commit == runner.UNAVAILABLE
    assert captured["config"].source.dirty is None
    assert captured["prerequisites"].master_binary == str(master)
    assert captured["prerequisites"].master_mode == "owned"
    assert captured["executor"] is runner.run_monitored
    assert "cargo" in captured["versions"]
    assert "rustc" in captured["versions"]
    assert "tokenspeed-mooncake" in captured["versions"]


def test_main_fails_closed_without_native_prerequisites(tmp_path, monkeypatch):
    monkeypatch.delenv(runner.STORE_LIBRARY_ENV, raising=False)
    monkeypatch.delenv(runner.MASTER_BIN_ENV, raising=False)
    monkeypatch.delenv(runner.EXTERNAL_MASTER_ENV, raising=False)
    exit_code = runner.main(cli_args(tmp_path, "1", "900", "20", "1800"))
    assert exit_code == 1
    report = read_report(tmp_path / "evidence")
    assert report["overall_status"] == "failed"
    assert report["prerequisites"]["ok"] is False
    assert any(
        "TS_MOONCAKE_STORE_LIBRARY" in problem
        for problem in report["prerequisites"]["problems"]
    )
    assert report["receipts"] == []


def test_main_reports_an_unexpected_error_as_a_failed_run(tmp_path, monkeypatch):
    monkeypatch.delenv(runner.STORE_LIBRARY_ENV, raising=False)
    monkeypatch.delenv(runner.MASTER_BIN_ENV, raising=False)
    monkeypatch.delenv(runner.EXTERNAL_MASTER_ENV, raising=False)

    def explode(
        config: runner.ValidationConfig,
        command: runner.NativeSuiteCommand,
        prerequisites: runner.Prerequisites,
        versions: dict[str, str],
        executor: runner.ProcessExecutor,
    ) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "run_validation", explode)
    exit_code = runner.main(cli_args(tmp_path, "1", "900", "20", "1800"))
    assert exit_code == 1


@pytest.mark.parametrize(
    "iterations,iteration_timeout,terminate_timeout,wall_clock_budget",
    [
        ("0", "900", "20", "100000"),
        ("21", "900", "20", "100000"),
        ("1", "5", "20", "100000"),
        ("1", "3601", "20", "100000"),
        ("1", "900", "0.5", "100000"),
        ("1", "900", "301", "100000"),
        ("1", "900", "20", "0"),
        ("1", "900", "20", "nan"),
        ("1", "900", "20", "inf"),
        # Include all three bounded cleanup waits as well as setup.
        ("1", "900", "20", "1500"),
        ("1", "900", "20", "1559"),
        # 20 x 3600s cannot fit any plausible dispatch budget.
        ("20", "3600", "20", "10800"),
        # 12 x 900s plus the margin exceeds the 180-minute dispatch budget.
        ("12", "900", "20", "10800"),
    ],
)
def test_main_rejects_out_of_bounds_values(
    tmp_path,
    iterations: str,
    iteration_timeout: str,
    terminate_timeout: str,
    wall_clock_budget: str,
):
    assert (
        runner.main(
            cli_args(
                tmp_path,
                iterations,
                iteration_timeout,
                terminate_timeout,
                wall_clock_budget,
            )
        )
        == 2
    )


def test_main_accepts_a_budget_that_exactly_fits(tmp_path, monkeypatch):
    # 900s + 3 x 20s cleanup + the 600s margin is exactly 1560s; only missing
    # native prerequisites may fail the run, not the bounds check.
    monkeypatch.delenv(runner.STORE_LIBRARY_ENV, raising=False)
    monkeypatch.delenv(runner.MASTER_BIN_ENV, raising=False)
    monkeypatch.delenv(runner.EXTERNAL_MASTER_ENV, raising=False)
    assert runner.main(cli_args(tmp_path, "1", "900", "20", "1560")) == 1


def test_main_rejects_malformed_and_missing_arguments(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(cli_args(tmp_path, "three", "900", "20", "1800"))
    with pytest.raises(SystemExit):
        runner.main(["--iterations", "1"])


def test_resolve_package_version():
    assert runner.resolve_package_version("pip") != runner.UNAVAILABLE
    assert (
        runner.resolve_package_version("tokenspeed-no-such-package")
        == runner.UNAVAILABLE
    )


def test_resolve_tool_version():
    version = runner.resolve_tool_version(
        [sys.executable, "--version"], dict(os.environ)
    )
    assert version.startswith("Python")
    assert (
        runner.resolve_tool_version(
            ["tokenspeed-no-such-tool", "--version"], dict(os.environ)
        )
        == runner.UNAVAILABLE
    )


def test_resolve_source_identity_without_git(tmp_path):
    identity = runner.resolve_source_identity(str(tmp_path), " deadbeef ")
    assert identity.label == "deadbeef"
    assert identity.commit == runner.UNAVAILABLE
    assert identity.dirty is None
    identity = runner.resolve_source_identity(str(tmp_path), None)
    assert identity.label is None
    assert identity.commit == runner.UNAVAILABLE
    with pytest.raises(runner.UsageError, match="source-revision"):
        runner.resolve_source_identity(str(tmp_path), "  ")


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_resolve_source_identity_reports_commit_and_dirty_state(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("one\n")
    git(tmp_path, "add", "tracked.txt")
    git(
        tmp_path,
        "-c",
        "user.email=ci@example.invalid",
        "-c",
        "user.name=ci",
        "commit",
        "-q",
        "-m",
        "initial",
    )
    head = git(tmp_path, "rev-parse", "HEAD")
    identity = runner.resolve_source_identity(str(tmp_path), None)
    assert identity.commit == head
    assert identity.dirty is False
    assert identity.label is None
    # An untracked file already makes the tree dirty.
    (tmp_path / "untracked.txt").write_text("two\n")
    identity = runner.resolve_source_identity(str(tmp_path), "pr-7-head-cafe")
    assert identity.commit == head
    assert identity.dirty is True
    assert identity.label == "pr-7-head-cafe"


@pytest.mark.parametrize(
    "native_log", ["native store log", "ok", "skipped: no native store"]
)
def test_nocapture_interleaving_preserves_test_identities(native_log):
    # Real libtest --nocapture output places stderr after the name prefix
    # and the terminal status on a later line. Native logs must not hide
    # either the selected test identities or a skip marker.
    output = (
        "running 5 tests\n"
        + "".join(f"test {name} ... {native_log}\nok\n" for name in ALL_TESTS)
        + "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    summary = runner.parse_suite_summary(output)
    assert summary.test_names == ALL_TESTS
    reasons = runner.iteration_rejections(0, False, summary)
    if "skipped:" in native_log:
        assert reasons == ["suite output contains a skip marker"]
    else:
        assert reasons == []


def test_repeated_identity_cannot_satisfy_the_fixed_suite():
    output = (
        "running 5 tests\n"
        + "".join(f"test {name} ... ok\n" for name in (*ALL_TESTS, ALL_TESTS[0]))
        + "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    assert "suite output repeats a test identity" in runner.iteration_rejections(
        0, False, runner.parse_suite_summary(output)
    )
