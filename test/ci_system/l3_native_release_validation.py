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

"""Repeatable, fail-closed native L3 release validation.

Runs the ts-l3 live Mooncake master suite (``cargo test --locked -p
tokenspeed-l3 --test mooncake``) ``--iterations`` times. Every iteration is an
isolated subprocess in its own process group with a hard timeout. Cleanup is
identical on every exit path — normal leader exit, timeout, interruption or a
repeated cancellation signal: the owned group is swept (a SIGTERM grace on
the timeout and interrupt paths, then SIGKILL), remaining members are killed
before the leader is reaped, and SIGINT/SIGTERM are deferred until cleanup
finishes. Only the group the runner itself created is ever signalled: the
unreaped leader pins the group id until the sweep is done, so no unrelated
process can be hit. Linux and macOS are supported.

An iteration passes only when the suite exits zero AND its output is exactly
the expected run: one ``running N tests`` header, one ``test result`` summary
with no failed, ignored, measured or filtered-out tests, every expected test
of the fixed suite present and ok, and no skip marker. The suite runs with
``--nocapture`` so the skip marker the tests print when the native store is
missing actually reaches the raw log.

Evidence is durable: the raw Rust output of every iteration, one structured
receipt per iteration and an overall report with the source identity (the
checked-out commit and its dirty flag resolved from git; an operator label is
recorded for cross-checking but never trusted as the tested revision), the
exact command and configuration, resolved package/tool versions, platform and
timing. Only the environment overrides the runner sets are recorded; the
process environment is never collected.

The runner fails closed. It refuses to start without the native store
library and a usable master binary — the suite's tenant and restart tests
always start their own master, even when MOONCAKE_MASTER names an external
one — and resolves both paths to absolute before launching in the engine
workspace. Launch and evidence-write errors fail the run, and the report
never says "passed" before every requested iteration completed. The
requested repetitions must fit the wall-clock budget including cleanup and
a setup margin.
Repetition is repeated protocol validation; it is not a long-duration serving
soak and proves nothing about GPU bytes or RDMA transfers. See
docs/operations/l3-release-validation.md.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import math
import os
import platform
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Callable

REQUIRE_MASTER_ENV = "MOONCAKE_REQUIRE_MASTER"
STORE_LIBRARY_ENV = "TS_MOONCAKE_STORE_LIBRARY"
MASTER_BIN_ENV = "TS_MOONCAKE_MASTER_BIN"
EXTERNAL_MASTER_ENV = "MOONCAKE_MASTER"
MOONCAKE_PACKAGE = "tokenspeed-mooncake"
MASTER_BINARY_NAME = "mooncake_master"

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_INTERRUPTED = "interrupted"

UNAVAILABLE = "unavailable"

# CLI bounds keep workflow_dispatch inputs bounded; the library entry points
# accept any structurally valid values so tests can use short timeouts.
MAX_ITERATIONS = 20
MIN_ITERATION_TIMEOUT_SECONDS = 10.0
MAX_ITERATION_TIMEOUT_SECONDS = 3600.0
MIN_TERMINATE_TIMEOUT_SECONDS = 1.0
MAX_TERMINATE_TIMEOUT_SECONDS = 300.0

# Reserved share of the wall-clock budget for checkout, dependency install
# and toolchain setup before the first iteration can start; iterations x
# iteration-timeout and cleanup waits must fit together with this margin.
SETUP_MARGIN_SECONDS = 600.0

VERSION_COMMAND_TIMEOUT_SECONDS = 10.0
SCHEMA_VERSION = 2

# The fixed suite this runner validates (tokenspeed-engine
# crates/ts-l3/tests/mooncake.rs). A deleted or renamed test is a failure,
# never a smaller pass.
EXPECTED_TEST_NAMES = (
    "a_sharded_block_restores_on_another_owner",
    "a_tenant_sees_only_its_own_objects",
    "objects_outlive_their_writer",
    "objects_round_trip_through_the_master",
    "the_backend_recovers_once_a_restarted_master_serves",
)

_RUNNING_RE = re.compile(r"^running (\d+) tests?$", re.MULTILINE)
_RESULT_RE = re.compile(
    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out",
    re.MULTILINE,
)
_TEST_NAME_RE = re.compile(r"^test (\S+) \.\.\.(?: |$)", re.MULTILINE)
_TEST_LINE_RE = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)$", re.MULTILINE)
# The Rust tests print "skipped: ..." to stderr when they skip the native
# store. With --nocapture the marker reaches the log, possibly glued to other
# output, so it is matched anywhere on a line.
_SKIP_MARKER_RE = re.compile(r"skipped:")

# Signals a cancellation can arrive on; held back while a group's cleanup
# runs so a repeated cancellation cannot interrupt it.
_CLEANUP_SIGNALS = frozenset({signal.SIGINT, signal.SIGTERM})
_CLEANUP_POLL_SECONDS = 0.01


class UsageError(Exception):
    """An operator error that must abort before any iteration runs."""


@dataclass(frozen=True)
class NativeSuiteCommand:
    """The fixed suite invocation plus the environment the runner sets.

    argv: the suite command; never a shell string.
    cwd: the engine workspace the suite runs in.
    env: the full child environment (inherited plus env_overrides).
    env_overrides: only the variables this runner sets; recorded as evidence.
    """

    argv: list[str]
    cwd: str
    env: dict[str, str]
    env_overrides: dict[str, str]


@dataclass(frozen=True)
class Prerequisites:
    """Resolved native prerequisites; any problem fails the run closed.

    library_path: the store library from TS_MOONCAKE_STORE_LIBRARY, resolved
        to an absolute path, if usable.
    master_mode: "external" (MOONCAKE_MASTER names a running master) or
        "owned" (the suite starts the master binary itself).
    master_binary: the master executable, resolved to an absolute path.
        Required in both modes: the tenant and restart tests always start
        their own master.
    master_address: the external master address, when master_mode is
        "external".
    problems: every unmet prerequisite; empty means the run may start.
    """

    library_path: str | None
    master_mode: str | None
    master_binary: str | None
    master_address: str | None
    problems: list[str]


@dataclass(frozen=True)
class SuiteSummary:
    """The cargo test summary parsed from one iteration's raw output.

    summaries_seen: how many test-result lines the output carries; exactly
        one is expected from this single-target suite.
    running_headers_seen: how many "running N tests" headers the output
        carries; exactly one is expected.
    running: the announced test count when exactly one header is present.
    passed/failed/ignored/measured/filtered_out: the single summary's counts.
    test_names: every distinct test name the output reports, sorted.
    ok_test_names: the distinct tests reported ok, sorted.
    not_ok_tests: "name (status)" entries for tests not reported ok.
    skip_marker_seen: a test reported skipping the native store.
    """

    summaries_seen: int
    running_headers_seen: int
    running: int | None
    passed: int
    failed: int
    ignored: int
    measured: int
    filtered_out: int
    test_names: tuple[str, ...]
    ok_test_names: tuple[str, ...]
    not_ok_tests: tuple[str, ...]
    skip_marker_seen: bool


@dataclass(frozen=True)
class ProcessOutcome:
    """The result of one monitored subprocess.

    returncode: the child's exit status; negative for a signal, None only if
        the child could not be reaped after SIGKILL.
    timed_out: the child exceeded its timeout and its group was terminated.
    elapsed_seconds: wall time from spawn to reap.
    """

    returncode: int | None
    timed_out: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class SourceIdentity:
    """What the report may claim about the tested source.

    commit: the checked-out HEAD resolved from git, or "unavailable". On
        pull_request runs this is the merge commit the checkout produced,
        not the PR head.
    dirty: True when the working tree has modifications or untracked files,
        False when clean, None when git could not determine it.
    label: the operator's --source-revision label (for example a PR head
        SHA). Recorded for cross-checking; never trusted as the tested
        revision.
    """

    commit: str
    dirty: bool | None
    label: str | None


@dataclass(frozen=True)
class ValidationConfig:
    """Operator-supplied run parameters; every behavior choice is explicit.

    iterations: how many isolated suite runs to require.
    iteration_timeout_seconds: hard bound on one suite run.
    terminate_timeout_seconds: grace between SIGTERM and SIGKILL of a
        timed-out or interrupted iteration's process group.
    wall_clock_budget_seconds: total budget the repetitions must fit with
        the setup margin.
    engine_dir: the tokenspeed-engine workspace the suite runs in.
    output_dir: evidence directory; must be absent or empty.
    source: the resolved source identity recorded in the report.
    """

    iterations: int
    iteration_timeout_seconds: float
    terminate_timeout_seconds: float
    wall_clock_budget_seconds: float
    engine_dir: str
    output_dir: str
    source: SourceIdentity


# run_monitored's call signature; the runner logic takes it as a parameter so
# tests drive real small subprocesses without a native store.
ProcessExecutor = Callable[
    [list[str], str, dict[str, str], float, float, Path], ProcessOutcome
]


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def native_suite_argv() -> list[str]:
    """The one suite invocation this runner validates; fixed and shell-free.

    --nocapture lets the tests' own stderr (the skip marker, panic output)
    reach the raw log; --test-threads=1 keeps that output serialized.
    """
    return [
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


def build_suite_command(
    base_env: dict[str, str],
    prerequisites: Prerequisites,
    engine_dir: str,
) -> NativeSuiteCommand:
    """Assemble the suite command, strictly setting MOONCAKE_REQUIRE_MASTER=1.

    Args:
        base_env: the environment the child inherits (usually os.environ).
        prerequisites: the checked native setup; its resolved absolute
            library and master binary paths are exported so the suite uses
            exactly what was checked no matter which directory it runs in.
        engine_dir: the tokenspeed-engine workspace the suite runs in.

    Returns:
        The fixed command with the runner's overrides applied on top of
        base_env, so a caller-provided MOONCAKE_REQUIRE_MASTER cannot weaken
        the gate and a relative or blank TS_MOONCAKE_MASTER_BIN cannot
        misresolve in the suite's working directory.
    """
    overrides = {REQUIRE_MASTER_ENV: "1"}
    if prerequisites.library_path is not None:
        overrides[STORE_LIBRARY_ENV] = prerequisites.library_path
    if prerequisites.master_binary is not None:
        overrides[MASTER_BIN_ENV] = prerequisites.master_binary
    env = dict(base_env)
    env.update(overrides)
    return NativeSuiteCommand(
        argv=native_suite_argv(),
        cwd=engine_dir,
        env=env,
        env_overrides=overrides,
    )


def check_prerequisites(env: dict[str, str], engine_dir: str) -> Prerequisites:
    """Resolve the native store library and master setup, failing closed.

    Args:
        env: the environment to inspect (TS_MOONCAKE_STORE_LIBRARY,
            TS_MOONCAKE_MASTER_BIN, MOONCAKE_MASTER, PATH). Paths are
            resolved to absolute so the suite, launched in engine_dir, uses
            exactly what was checked. A variable that is set but blank is a
            problem, not a silent default: the suite reads these variables
            as-is.
        engine_dir: must contain the engine workspace's Cargo.toml.

    Returns:
        The resolved setup with every unmet prerequisite in problems; the
        runner must not start any iteration while problems is non-empty. The
        master binary is required in both modes because the suite's tenant
        and restart tests always start their own master.
    """
    problems: list[str] = []
    library_path: str | None = None
    library_raw = env.get(STORE_LIBRARY_ENV)
    if library_raw is None or not library_raw.strip():
        problems.append(
            f"{STORE_LIBRARY_ENV} is not set; install {MOONCAKE_PACKAGE} and "
            "point it at libmooncake_store.so"
        )
    else:
        candidate = Path(library_raw.strip())
        if not candidate.is_file():
            problems.append(
                f"{STORE_LIBRARY_ENV} names {candidate}, which is not a file"
            )
        else:
            library_path = str(candidate.resolve())

    master_mode: str | None = None
    master_binary: str | None = None
    master_address: str | None = None
    external = env.get(EXTERNAL_MASTER_ENV)
    if external is not None and not external.strip():
        problems.append(
            f"{EXTERNAL_MASTER_ENV} is set but empty; unset it or set it to "
            "the running master's host:port"
        )
    elif external is not None:
        master_mode = "external"
        master_address = external.strip()
    else:
        master_mode = "owned"

    binary_raw = env.get(MASTER_BIN_ENV)
    if binary_raw is not None and not binary_raw.strip():
        problems.append(
            f"{MASTER_BIN_ENV} is set but empty; unset it or point it at "
            f"the {MASTER_BINARY_NAME} binary"
        )
        binary: Path | None = None
    elif binary_raw is not None:
        binary = Path(binary_raw.strip())
    elif library_path is not None:
        binary = Path(library_path).with_name(MASTER_BINARY_NAME)
    else:
        binary = None
    # The tenant and restart tests always start their own master, so the
    # binary is a prerequisite even in external-master mode.
    if binary is None and binary_raw is None:
        problems.append(
            f"cannot locate the {MASTER_BINARY_NAME} binary without the "
            f"store library; set {MASTER_BIN_ENV}"
        )
    elif binary is not None:
        if not binary.is_file():
            problems.append(
                f"master binary {binary} does not exist; set {MASTER_BIN_ENV}"
            )
        elif not os.access(binary, os.X_OK):
            problems.append(f"master binary {binary} is not executable")
        else:
            master_binary = str(binary.resolve())

    if shutil.which("cargo", path=env.get("PATH")) is None:
        problems.append("cargo is not on PATH; install the Rust toolchain")
    if not (Path(engine_dir) / "Cargo.toml").is_file():
        problems.append(f"engine directory {engine_dir} has no Cargo.toml")

    return Prerequisites(
        library_path=library_path,
        master_mode=master_mode,
        master_binary=master_binary,
        master_address=master_address,
        problems=problems,
    )


def parse_suite_summary(output: str) -> SuiteSummary:
    """Parse the cargo test summary from one iteration's raw output.

    Args:
        output: the iteration's combined stdout/stderr.

    Returns:
        The parsed summary; counts are zero when no single test-result line
        is present, which the acceptance check rejects as incomplete.
    """
    running_matches = _RUNNING_RE.findall(output)
    result_matches = _RESULT_RE.findall(output)
    passed = failed = ignored = measured = filtered_out = 0
    if len(result_matches) == 1:
        passed, failed, ignored, measured, filtered_out = (
            int(group) for group in result_matches[0]
        )
    entries = _TEST_LINE_RE.findall(output)
    # --nocapture lets native logs appear between libtest's test-name
    # prefix and its terminal status. The summary proves the pass count;
    # collect identities independently of that interleaved output.
    test_names = tuple(sorted(_TEST_NAME_RE.findall(output)))
    ok_test_names = tuple(sorted({name for name, status in entries if status == "ok"}))
    not_ok_tests = tuple(
        sorted({f"{name} ({status})" for name, status in entries if status != "ok"})
    )
    return SuiteSummary(
        summaries_seen=len(result_matches),
        running_headers_seen=len(running_matches),
        running=int(running_matches[0]) if len(running_matches) == 1 else None,
        passed=passed,
        failed=failed,
        ignored=ignored,
        measured=measured,
        filtered_out=filtered_out,
        test_names=test_names,
        ok_test_names=ok_test_names,
        not_ok_tests=not_ok_tests,
        skip_marker_seen=_SKIP_MARKER_RE.search(output) is not None,
    )


def iteration_rejections(
    returncode: int | None, timed_out: bool, summary: SuiteSummary
) -> list[str]:
    """Every reason one iteration's outcome is not a pass; empty means pass.

    Args:
        returncode: the suite's exit status.
        timed_out: whether the iteration exceeded its timeout.
        summary: the parsed suite summary.

    Returns:
        Rejection reasons. A zero exit code never suffices on its own:
        zero-test, skipped, ignored, measured, filtered, incomplete,
        misheadered and wrongly-scoped runs all reject, as does a suite
        whose tests are not exactly the expected ones.
    """
    if timed_out:
        return ["iteration exceeded its timeout"]
    reasons: list[str] = []
    if returncode != 0:
        reasons.append(f"suite exited with return code {returncode}")
    if summary.skip_marker_seen:
        reasons.append("suite output contains a skip marker")
    if summary.summaries_seen == 0:
        reasons.append("suite output has no test-result summary")
    elif summary.summaries_seen > 1:
        reasons.append(
            f"suite output has {summary.summaries_seen} test-result "
            "summaries; exactly one is expected"
        )
    if summary.running_headers_seen == 0:
        reasons.append("suite output has no running-tests header")
    elif summary.running_headers_seen > 1:
        reasons.append(
            f"suite output has {summary.running_headers_seen} running-tests "
            "headers; exactly one is expected"
        )
    if summary.summaries_seen == 1:
        if summary.passed == 0:
            reasons.append("suite passed zero tests")
        if summary.failed != 0:
            reasons.append(
                f"suite reported {summary.failed} failed tests with a zero " "exit code"
            )
        if summary.ignored != 0:
            reasons.append(f"suite ignored {summary.ignored} tests")
        if summary.measured != 0:
            reasons.append(f"suite measured {summary.measured} tests")
        if summary.filtered_out != 0:
            reasons.append(f"suite filtered out {summary.filtered_out} tests")
        if summary.running is not None:
            accounted = (
                summary.passed
                + summary.failed
                + summary.ignored
                + summary.measured
                + summary.filtered_out
            )
            if accounted != summary.running:
                reasons.append(
                    f"suite announced {summary.running} tests but accounted "
                    f"for {accounted}; the run is incomplete"
                )
        expected = set(EXPECTED_TEST_NAMES)
        seen = set(summary.test_names)
        if summary.running != len(expected) or summary.passed != len(expected):
            reasons.append(
                f"the fixed suite must run and pass exactly {len(expected)} tests"
            )
        if len(summary.test_names) != len(seen):
            reasons.append("suite output repeats a test identity")
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        if missing:
            reasons.append("suite is missing expected tests: " + ", ".join(missing))
        if unexpected:
            reasons.append("suite ran unexpected tests: " + ", ".join(unexpected))
        if summary.not_ok_tests:
            reasons.append("tests did not pass: " + ", ".join(summary.not_ok_tests))
    return reasons


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class _ExitWatch:
    """Wait for a child's exit without reaping it.

    While the child stays unreaped, its zombie keeps the process group id
    allocated, so a killpg cannot hit a recycled id. Uses waitid with
    WNOWAIT where available (Linux) and a kqueue NOTE_EXIT watch elsewhere
    (macOS, where Python has no waitid). Both wait briefly at a time so a
    cancellation signal still lands promptly.
    """

    def __init__(self, pid: int):
        self._pid = pid
        self._gone = False
        self._kq: select.kqueue | None = None
        if not hasattr(os, "waitid"):
            self._kq = select.kqueue()
            event = select.kevent(
                pid,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD,
                fflags=select.KQ_NOTE_EXIT,
            )
            try:
                self._kq.control([event], 0, 0)
            except OSError:
                # The child is already gone; treat every wait as exited.
                self._gone = True

    def wait(self, deadline: float) -> bool:
        """True once the child has exited, by deadline (monotonic)."""
        if self._gone:
            return True
        if self._kq is not None:
            return self._wait_kqueue(deadline)
        return self._wait_waitid(deadline)

    def _wait_waitid(self, deadline: float) -> bool:
        while True:
            try:
                info = os.waitid(
                    os.P_PID,
                    self._pid,
                    os.WEXITED | os.WNOWAIT | os.WNOHANG,
                )
            except ChildProcessError:
                return True
            if info is not None and info.si_pid == self._pid:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_CLEANUP_POLL_SECONDS)

    def _wait_kqueue(self, deadline: float) -> bool:
        while True:
            remaining = deadline - time.monotonic()
            events = self._kq.control(None, 1, max(remaining, 0.0))
            if events:
                return True
            if remaining <= 0:
                return False

    def close(self) -> None:
        if self._kq is not None:
            self._kq.close()


def _group_gone(pgid: int, deadline: float) -> None:
    """Best-effort bounded wait until no process remains in the group."""
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(_CLEANUP_POLL_SECONDS)


def _cleanup_group(
    proc: subprocess.Popen,
    watch: _ExitWatch,
    terminate_timeout_seconds: float,
    terminate_first: bool,
) -> None:
    """Kill every process in proc's owned group, then reap proc.

    SIGINT/SIGTERM handlers defer cancellation for the whole cleanup, even
    if another Python thread receives the signal. A thread-local signal
    mask alone cannot stop Python from dispatching that pending handler on
    the main thread. The group is signalled before the
    leader is reaped: while the leader's zombie pins the group id, killpg
    can never reach a recycled id, and no process outside this group is
    ever touched.
    """
    # start_new_session made the child its process group leader.
    pgid = proc.pid
    deferred: list[int] = []

    def defer(signum: int, frame: FrameType | None) -> None:
        del frame
        deferred.append(signum)

    previous_handlers = {sig: signal.signal(sig, defer) for sig in _CLEANUP_SIGNALS}
    try:
        if terminate_first:
            _signal_group(pgid, signal.SIGTERM)
            watch.wait(time.monotonic() + terminate_timeout_seconds)
        # Sweep members that outlived the leader or ignored SIGTERM, then
        # reap the leader last.
        _signal_group(pgid, signal.SIGKILL)
        try:
            proc.wait(timeout=terminate_timeout_seconds)
        except subprocess.TimeoutExpired:
            # SIGKILL cannot be ignored; an unreaped leader here means the
            # OS has not delivered the status yet. Recorded through
            # returncode=None.
            pass
        _group_gone(pgid, time.monotonic() + terminate_timeout_seconds)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if deferred:
        raise KeyboardInterrupt(f"received signal {deferred[0]} during cleanup")


def run_monitored(
    argv: list[str],
    cwd: str,
    env: dict[str, str],
    timeout_seconds: float,
    terminate_timeout_seconds: float,
    log_path: Path,
) -> ProcessOutcome:
    """Run argv in its own process group under a hard timeout.

    Args:
        argv: the command; passed to exec without a shell.
        cwd: working directory.
        env: full child environment.
        timeout_seconds: hard bound on the child's lifetime.
        terminate_timeout_seconds: grace between SIGTERM and SIGKILL.
        log_path: receives the child's combined stdout/stderr verbatim.

    Returns:
        The exit status, timeout flag and elapsed wall time. The owned
        process group is cleaned up on every exit path — normal leader
        exit, timeout, or interruption: leftover members (such as a master
        the suite leaked) are SIGKILLed before the leader is reaped, and
        only that group is ever signalled.

    Raises:
        OSError: the child could not be launched at all.
    """
    started = time.monotonic()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        watch = _ExitWatch(proc.pid)
        try:
            try:
                exited = watch.wait(started + timeout_seconds)
            except BaseException:
                _cleanup_group(proc, watch, terminate_timeout_seconds, True)
                raise
            timed_out = not exited
            _cleanup_group(proc, watch, terminate_timeout_seconds, timed_out)
        finally:
            watch.close()
    return ProcessOutcome(
        returncode=proc.returncode,
        timed_out=timed_out,
        elapsed_seconds=time.monotonic() - started,
    )


def prepare_output_dir(output_dir: str) -> Path:
    """Create the evidence directory, refusing to overwrite earlier evidence.

    Args:
        output_dir: must be absent or an empty directory.

    Returns:
        The directory's path.

    Raises:
        UsageError: the path exists and is not an empty directory.
    """
    path = Path(output_dir)
    if path.exists():
        if not path.is_dir():
            raise UsageError(f"output path {path} exists and is not a directory")
        if any(path.iterdir()):
            raise UsageError(
                f"output directory {path} is not empty; refusing to "
                "overwrite earlier evidence"
            )
    else:
        path.mkdir(parents=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    """Write one evidence file as stable, human-readable JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_package_version(package: str) -> str:
    """The installed distribution's version, or "unavailable"."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return UNAVAILABLE


def resolve_tool_version(argv: list[str], env: dict[str, str]) -> str:
    """A tool's first --version line, or "unavailable"; bounded and quiet.

    Args:
        argv: e.g. ["cargo", "--version"]; executed without a shell.
        env: the environment to run it in.
    """
    try:
        result = subprocess.run(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=VERSION_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return UNAVAILABLE
    lines = result.stdout.splitlines()
    first = lines[0].strip() if lines else ""
    if result.returncode != 0 or not first:
        return UNAVAILABLE
    return first


def _git_output(argv: list[str], cwd: str) -> str | None:
    """A git command's stdout, or None on any failure; bounded and quiet."""
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=VERSION_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def resolve_source_identity(engine_dir: str, label: str | None) -> SourceIdentity:
    """The source identity for the report.

    Args:
        engine_dir: any path inside the checkout; git resolves the root.
        label: an operator-provided label (for example a PR head SHA). The
            tested commit is always resolved from the checkout itself, never
            from this value.

    Raises:
        UsageError: an explicit label that is empty.
    """
    stripped: str | None = None
    if label is not None:
        stripped = label.strip()
        if not stripped:
            raise UsageError("--source-revision must not be empty")
    head = _git_output(["git", "rev-parse", "HEAD"], engine_dir)
    if head is None or not head.strip():
        return SourceIdentity(commit=UNAVAILABLE, dirty=None, label=stripped)
    status = _git_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], engine_dir
    )
    return SourceIdentity(
        commit=head.strip(),
        dirty=None if status is None else bool(status.strip()),
        label=stripped,
    )


def platform_facts() -> dict[str, str]:
    """The platform facts recorded in the report; never the environment."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def _interrupted_by_signal(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def run_iteration(
    iteration: int,
    config: ValidationConfig,
    command: NativeSuiteCommand,
    output_dir: Path,
    executor: ProcessExecutor,
) -> tuple[str, str, list[str]]:
    """Run one suite iteration, write its receipt, and classify the outcome.

    Args:
        iteration: the 1-based iteration index; names the receipt and log.
        config: the run's timeouts and parameters.
        command: the fixed suite command.
        output_dir: the prepared evidence directory.
        executor: the monitored subprocess runner.

    Returns:
        (receipt file name, iteration status, rejection reasons).

    Raises:
        KeyboardInterrupt: re-raised after the in-flight iteration's receipt
            is written with status "interrupted".
    """
    log_name = f"iteration-{iteration:03d}.log"
    receipt_name = f"iteration-{iteration:03d}.json"
    log_path = output_dir / log_name
    print(
        f"[l3-validation] iteration {iteration}/{config.iterations}: "
        f"{' '.join(command.argv)} "
        f"(timeout {config.iteration_timeout_seconds}s)",
        flush=True,
    )
    iteration_started_utc = utc_now()
    iteration_started = time.monotonic()
    interrupted: KeyboardInterrupt | None = None
    outcome: ProcessOutcome | None = None
    launch_error: str | None = None
    try:
        outcome = executor(
            command.argv,
            command.cwd,
            command.env,
            config.iteration_timeout_seconds,
            config.terminate_timeout_seconds,
            log_path,
        )
    except KeyboardInterrupt as exc:
        interrupted = exc
    except OSError as exc:
        # The suite could not be launched at all (missing or non-executable
        # cargo, ...); the iteration fails closed with its receipt written.
        launch_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - iteration_started
    output = ""
    read_error: str | None = None
    if interrupted is None and launch_error is None:
        try:
            output = log_path.read_text(errors="replace") if log_path.is_file() else ""
        except Exception as exc:
            # e.g. OSError or MemoryError on a huge log; the iteration fails
            # closed rather than judging partial evidence.
            read_error = f"{type(exc).__name__}: {exc}"

    summary: SuiteSummary | None = None
    returncode: int | None = None
    timed_out = False
    if interrupted is not None:
        status = STATUS_INTERRUPTED
        reasons = [f"interrupted: {str(interrupted) or 'SIGINT'}"]
    elif launch_error is not None:
        status = STATUS_FAILED
        reasons = [f"suite could not be launched: {launch_error}"]
    elif outcome is not None:
        summary = parse_suite_summary(output)
        returncode = outcome.returncode
        timed_out = outcome.timed_out
        reasons = iteration_rejections(returncode, timed_out, summary)
        if read_error is not None:
            reasons.append(f"iteration log could not be read: {read_error}")
        if not reasons:
            status = STATUS_PASSED
        elif timed_out:
            status = STATUS_TIMEOUT
        else:
            status = STATUS_FAILED
    else:
        # The executor contract guarantees an outcome or an exception.
        raise RuntimeError("executor returned without an outcome")

    receipt = {
        "iteration": iteration,
        "status": status,
        "started_utc": iteration_started_utc,
        "elapsed_seconds": round(elapsed, 3),
        "argv": command.argv,
        "cwd": command.cwd,
        "env_overrides": command.env_overrides,
        "returncode": returncode,
        "timed_out": timed_out,
        "summary": asdict(summary) if summary is not None else None,
        "rejection_reasons": reasons,
        "log_file": log_name,
    }
    write_json(output_dir / receipt_name, receipt)
    detail = f": {'; '.join(reasons)}" if reasons else ""
    print(
        f"[l3-validation] iteration {iteration}: {status} ({elapsed:.1f}s){detail}",
        flush=True,
    )
    if interrupted is not None:
        raise interrupted
    return receipt_name, status, reasons


def run_validation(
    config: ValidationConfig,
    command: NativeSuiteCommand,
    prerequisites: Prerequisites,
    versions: dict[str, str],
    executor: ProcessExecutor,
) -> str:
    """Run the validation loop and write all evidence.

    Args:
        config: iteration count, timeouts, budget and paths; must be
            structurally valid (positive counts and timeouts).
        command: the fixed suite command.
        prerequisites: the checked native setup; any problem fails the run
            before the first iteration, with the report still written.
        versions: resolved package/tool versions for the report.
        executor: the monitored subprocess runner.

    Returns:
        The overall status: "passed", "failed", "timeout" or "interrupted".
        The run stops at the first non-passing iteration; report.json is
        written in every outcome, including failure, timeout, interrupt and
        unexpected errors. "passed" is only ever returned and recorded after
        every requested iteration completed and passed.

    Raises:
        UsageError: the config is structurally invalid or the output
            directory would overwrite earlier evidence.
    """
    if config.iterations < 1:
        raise UsageError(f"iterations must be at least 1, got {config.iterations}")
    if (
        not math.isfinite(config.iteration_timeout_seconds)
        or config.iteration_timeout_seconds <= 0
    ):
        raise UsageError(
            f"iteration timeout must be positive, got "
            f"{config.iteration_timeout_seconds}"
        )
    if (
        not math.isfinite(config.terminate_timeout_seconds)
        or config.terminate_timeout_seconds <= 0
    ):
        raise UsageError(
            f"terminate timeout must be positive, got "
            f"{config.terminate_timeout_seconds}"
        )
    if (
        not math.isfinite(config.wall_clock_budget_seconds)
        or config.wall_clock_budget_seconds <= 0
    ):
        raise UsageError(
            f"wall-clock budget must be positive, got "
            f"{config.wall_clock_budget_seconds}"
        )
    output_dir = prepare_output_dir(config.output_dir)
    started_utc = utc_now()
    started = time.monotonic()
    receipts: list[str] = []
    completed = 0
    # Fail closed: the report says "passed" only after every iteration did.
    overall = STATUS_FAILED
    stop_reason = "validation did not complete"
    # Translate SIGTERM so cancellation lands on the same interrupted path as
    # SIGINT: the in-flight child group is terminated and the report written.
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupted_by_signal)
    try:
        if prerequisites.problems:
            overall = STATUS_FAILED
            stop_reason = "prerequisite check failed: " + "; ".join(
                prerequisites.problems
            )
            print(f"[l3-validation] {stop_reason}", flush=True)
        else:
            for iteration in range(1, config.iterations + 1):
                try:
                    receipt_name, status, reasons = run_iteration(
                        iteration, config, command, output_dir, executor
                    )
                except KeyboardInterrupt:
                    # The interrupted iteration's receipt is already on disk.
                    receipts.append(f"iteration-{iteration:03d}.json")
                    raise
                receipts.append(receipt_name)
                if status == STATUS_PASSED:
                    completed += 1
                    continue
                overall = status
                stop_reason = f"iteration {iteration} {status}: " + "; ".join(reasons)
                break
            if completed == config.iterations:
                overall = STATUS_PASSED
                stop_reason = f"all {config.iterations} iterations passed"
    except KeyboardInterrupt as exc:
        overall = STATUS_INTERRUPTED
        stop_reason = f"interrupted: {str(exc) or 'SIGINT'}"
    except Exception as exc:
        # An unexpected error (evidence write failure, a violated executor
        # contract, ...) must never read as a pass.
        overall = STATUS_FAILED
        stop_reason = (
            f"validation aborted on an unexpected error: "
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        report = {
            "schema_version": SCHEMA_VERSION,
            "tool": "l3_native_release_validation",
            "overall_status": overall,
            "stop_reason": stop_reason,
            "source": {
                "commit": config.source.commit,
                "dirty": config.source.dirty,
                "label": config.source.label,
            },
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "config": {
                "iterations": config.iterations,
                "iteration_timeout_seconds": config.iteration_timeout_seconds,
                "terminate_timeout_seconds": config.terminate_timeout_seconds,
                "wall_clock_budget_seconds": config.wall_clock_budget_seconds,
                "setup_margin_seconds": SETUP_MARGIN_SECONDS,
                "engine_dir": config.engine_dir,
                "output_dir": config.output_dir,
                "argv": command.argv,
                "env_overrides": command.env_overrides,
                "master_mode": prerequisites.master_mode,
                "master_binary": prerequisites.master_binary,
                "master_address": prerequisites.master_address,
            },
            "prerequisites": {
                "ok": not prerequisites.problems,
                "library_path": prerequisites.library_path,
                "problems": prerequisites.problems,
            },
            "platform": platform_facts(),
            "versions": versions,
            "iterations_planned": config.iterations,
            "iterations_completed": completed,
            "receipts": receipts,
        }
        write_json(output_dir / "report.json", report)
        print(
            f"[l3-validation] overall={overall} completed={completed}/"
            f"{config.iterations} report={output_dir / 'report.json'}",
            flush=True,
        )
    return overall


def validate_bounds(
    iterations: int,
    iteration_timeout: float,
    terminate_timeout: float,
    wall_clock_budget: float,
) -> None:
    """Enforce the CLI bounds that keep workflow_dispatch inputs bounded.

    The requested repetitions must fit the wall-clock budget together with
    the setup margin and cleanup waits, so a run cannot request more than the job it runs in.

    Raises:
        UsageError: a value outside its documented range.
    """
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise UsageError(
            f"--iterations must be within 1..{MAX_ITERATIONS}, got {iterations}"
        )
    if not (
        MIN_ITERATION_TIMEOUT_SECONDS
        <= iteration_timeout
        <= MAX_ITERATION_TIMEOUT_SECONDS
    ):
        raise UsageError(
            f"--iteration-timeout must be within "
            f"{MIN_ITERATION_TIMEOUT_SECONDS}..{MAX_ITERATION_TIMEOUT_SECONDS} "
            f"seconds, got {iteration_timeout}"
        )
    if not (
        MIN_TERMINATE_TIMEOUT_SECONDS
        <= terminate_timeout
        <= MAX_TERMINATE_TIMEOUT_SECONDS
    ):
        raise UsageError(
            f"--terminate-timeout must be within "
            f"{MIN_TERMINATE_TIMEOUT_SECONDS}..{MAX_TERMINATE_TIMEOUT_SECONDS} "
            f"seconds, got {terminate_timeout}"
        )
    if not math.isfinite(wall_clock_budget) or wall_clock_budget <= 0:
        raise UsageError(
            f"--wall-clock-budget must be positive, got {wall_clock_budget}"
        )
    # Each cleanup can wait once for TERM, once for the direct child's
    # reap, and once for remaining group members to disappear.
    required = (
        iterations * (iteration_timeout + 3 * terminate_timeout) + SETUP_MARGIN_SECONDS
    )
    if required > wall_clock_budget:
        raise UsageError(
            f"--iterations x (--iteration-timeout + 3 x --terminate-timeout) plus the "
            f"{SETUP_MARGIN_SECONDS:.0f}s setup margin needs {required:.0f}s, "
            f"over --wall-clock-budget {wall_clock_budget}s; request fewer "
            "iterations or a shorter timeout"
        )


def build_parser() -> argparse.ArgumentParser:
    """The CLI; every behavior choice is an explicit required argument."""
    parser = argparse.ArgumentParser(
        description=(
            "Repeatable, fail-closed native L3 release validation: re-run "
            "the ts-l3 live Mooncake master suite in isolated, timed-out "
            "subprocesses and keep per-iteration receipts plus an overall "
            "report. Repetition is repeated protocol validation, not a "
            "long-duration serving soak."
        )
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help=f"suite repetitions (1..{MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--iteration-timeout",
        type=float,
        required=True,
        help=(
            f"hard timeout of one suite run, seconds "
            f"({MIN_ITERATION_TIMEOUT_SECONDS}..{MAX_ITERATION_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--terminate-timeout",
        type=float,
        required=True,
        help=(
            f"grace between SIGTERM and SIGKILL of a timed-out iteration's "
            f"process group, seconds "
            f"({MIN_TERMINATE_TIMEOUT_SECONDS}..{MAX_TERMINATE_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--wall-clock-budget",
        type=float,
        required=True,
        help=(
            "total wall-clock budget of the run, seconds; iterations x "
            f"(iteration-timeout + 3 x terminate-timeout) plus the {SETUP_MARGIN_SECONDS:.0f}s setup "
            "margin must fit it"
        ),
    )
    parser.add_argument(
        "--engine-dir",
        required=True,
        help="path of the tokenspeed-engine crate workspace",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="evidence directory; must be absent or empty",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="label recorded in the report for cross-checking (for example "
        "a PR head SHA); the tested commit is always resolved from the "
        "checkout itself, never from this value",
    )
    return parser


def main(argv: list[str]) -> int:
    """CLI entry point.

    Args:
        argv: command-line arguments without the program name.

    Returns:
        0 passed, 1 failed, timed out or errored, 130 interrupted, 2 usage
        error. The native suite command is fixed here; only run_validation's
        orchestration is parameterizable by tests.
    """
    args = build_parser().parse_args(argv)
    try:
        validate_bounds(
            args.iterations,
            args.iteration_timeout,
            args.terminate_timeout,
            args.wall_clock_budget,
        )
        env = dict(os.environ)
        prerequisites = check_prerequisites(env, args.engine_dir)
        source = resolve_source_identity(args.engine_dir, args.source_revision)
        versions = {
            MOONCAKE_PACKAGE: resolve_package_version(MOONCAKE_PACKAGE),
            "cargo": resolve_tool_version(["cargo", "--version"], env),
            "rustc": resolve_tool_version(["rustc", "--version"], env),
        }
        command = build_suite_command(env, prerequisites, args.engine_dir)
        config = ValidationConfig(
            iterations=args.iterations,
            iteration_timeout_seconds=args.iteration_timeout,
            terminate_timeout_seconds=args.terminate_timeout,
            wall_clock_budget_seconds=args.wall_clock_budget,
            engine_dir=args.engine_dir,
            output_dir=args.output_dir,
            source=source,
        )
        overall = run_validation(
            config, command, prerequisites, versions, run_monitored
        )
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Fail closed: an unexpected error is a failed run, never a pass.
        print(
            f"error: unexpected failure: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    if overall == STATUS_PASSED:
        return 0
    if overall == STATUS_INTERRUPTED:
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
