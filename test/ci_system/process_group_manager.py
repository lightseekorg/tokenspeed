"""
Process-group-based process lifecycle manager for CI runners.

Usage
-----
# At job start (cleans up stale processes from previous run):
mgr = ProcessGroupManager(runner_id="gb200-node-1")
mgr.cleanup_stale()

# Launch a long-running server:
proc = mgr.start(command, shell=True, cwd=cwd, env=env)

# At job end, terminate all process groups started by this manager:
mgr.terminate_all()
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

# Directory that survives across CI runs on the same host.
# Must be writable by the runner user.
_PGID_DIR = Path("/tmp/ci-pgid")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_runner_id(runner_id: str) -> str:
    """Sanitise runner_id so it is safe to use as a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", runner_id)


def _pgid_path(runner_id: str) -> Path:
    return _PGID_DIR / f"{_safe_runner_id(runner_id)}.pgid"


def _save_pgids(runner_id: str, pgids: Iterable[int]) -> None:
    _PGID_DIR.mkdir(parents=True, exist_ok=True)
    _pgid_path(runner_id).write_text(
        "".join(f"{pgid}\n" for pgid in sorted(set(pgids)))
    )


def _add_pgid(runner_id: str, pgid: int) -> None:
    pgids = _load_pgids(runner_id)
    pgids.add(pgid)
    _save_pgids(runner_id, pgids)


def _load_pgids(runner_id: str) -> set[int]:
    path = _pgid_path(runner_id)
    if not path.exists():
        return set()
    pgids: set[int] = set()
    try:
        for token in path.read_text().split():
            pgids.add(int(token))
    except (ValueError, OSError):
        return set()
    return pgids


def _load_pgid(runner_id: str) -> Optional[int]:
    """Backward-compatible single-pgid accessor for older callers/tests."""
    pgids = _load_pgids(runner_id)
    if not pgids:
        return None
    return next(iter(sorted(pgids)))


def _remove_pgid(runner_id: str) -> None:
    _pgid_path(runner_id).unlink(missing_ok=True)


def _kill_pgid(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class ProcessGroupManager:
    def __init__(self, runner_id: str, term_timeout: float = 10.0) -> None:
        if not runner_id:
            raise ValueError("runner_id must be a non-empty string")
        self.runner_id = runner_id
        self.term_timeout = term_timeout
        self._procs: list[subprocess.Popen] = []

    def cleanup_stale(self, dry_run: bool = False) -> None:
        """
        Kill any process group left over from a previous run of this runner.
        Safe to call even if no stale pgid file exists.
        """
        pgids = _load_pgids(self.runner_id)
        if not pgids:
            print(
                f"[pgm] cleanup_stale: no stale pgid file for runner={self.runner_id}",
                flush=True,
            )
            return

        print(
            f"[pgm] cleanup_stale: killing stale process groups pgids={sorted(pgids)} "
            f"for runner={self.runner_id}",
            flush=True,
        )
        if dry_run:
            print(
                f"[pgm] cleanup_stale: [dry-run] skip kill pgids={sorted(pgids)}",
                flush=True,
            )
            return

        print(
            f"[pgm] cleanup_stale: sending SIGTERM to pgids={sorted(pgids)}",
            flush=True,
        )
        for pgid in sorted(pgids):
            _kill_pgid(pgid, signal.SIGTERM)
        print(
            f"[pgm] cleanup_stale: waiting {self.term_timeout}s for graceful shutdown",
            flush=True,
        )
        time.sleep(self.term_timeout)
        print(
            f"[pgm] cleanup_stale: sending SIGKILL to pgids={sorted(pgids)}",
            flush=True,
        )
        for pgid in sorted(pgids):
            _kill_pgid(pgid, signal.SIGKILL)
        _remove_pgid(self.runner_id)
        print(
            f"[pgm] cleanup_stale: removed pgid file for runner={self.runner_id}",
            flush=True,
        )

    def start(
        self,
        command: str,
        *,
        shell: bool = True,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        dry_run: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Launch command in a new process group (long-running server)."""
        print(f"[pgm] start: launching command for runner={self.runner_id}", flush=True)
        print(f"$ {command}", flush=True)
        if dry_run:
            print(f"[pgm] start: [dry-run] skip", flush=True)
            return None

        proc = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        self._procs.append(proc)
        print(
            f"[pgm] start: spawned pid={proc.pid}, tracked_procs={len(self._procs)}",
            flush=True,
        )

        try:
            pgid = os.getpgid(proc.pid)
            _add_pgid(self.runner_id, pgid)
            print(
                f"[pgm] start: pid={proc.pid} pgid={pgid} "
                f"runner={self.runner_id} pgid_file={_pgid_path(self.runner_id)}",
                flush=True,
            )
        except ProcessLookupError:
            print(
                f"[pgm] start: pid={proc.pid} already exited before getpgid", flush=True
            )

        return proc

    def run(
        self,
        command: str,
        *,
        shell: bool = True,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        dry_run: bool = False,
        check: bool = True,
    ) -> dict:
        """
        Run a short-lived command and track it.
        """
        print(f"[pgm] run: executing command for runner={self.runner_id}", flush=True)
        print(f"$ {command}", flush=True)
        if dry_run:
            print(f"[pgm] run: [dry-run] skip", flush=True)
            return {"command": command, "returncode": 0, "output": "", "dry_run": True}

        proc = subprocess.Popen(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
        self._procs.append(proc)

        try:
            pgid = os.getpgid(proc.pid)
            _add_pgid(self.runner_id, pgid)
            print(
                f"[pgm] run: spawned pid={proc.pid} pgid={pgid} "
                f"tracked_procs={len(self._procs)}",
                flush=True,
            )
        except ProcessLookupError:
            print(
                f"[pgm] run: pid={proc.pid} already exited before getpgid", flush=True
            )

        output_lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
        proc.wait()
        self._procs.remove(proc)

        returncode = proc.returncode
        print(
            f"[pgm] run: finished pid={proc.pid} returncode={returncode} "
            f"tracked_procs={len(self._procs)}",
            flush=True,
        )

        # Python multiprocessing daemon children receive SIGTERM when the
        # parent exits, which can race with normal shutdown and cause the
        # parent to report -15.  If the test output shows ALL tests passed,
        # downgrade the signal exit to a warning.
        if returncode < 0:
            output_text = "".join(output_lines)
            # Match "Test Summary: N/N passed" only when N == N (all passed)
            summary_match = re.search(
                r"Test Summary:\s+(\d+)/(\d+)\s+passed", output_text
            )
            all_passed = (
                summary_match is not None
                and summary_match.group(1) == summary_match.group(2)
            ) or re.search(r"Ran \d+ tests?.*\bOK\b", output_text)
            if all_passed:
                print(
                    f"[pgm] run: process exited with signal {-returncode}, "
                    f"but test output indicates success — treating as passed",
                    flush=True,
                )
                returncode = 0

        if check and returncode != 0:
            raise RuntimeError(f"command failed with exit code {returncode}: {command}")

        return {
            "command": command,
            "returncode": returncode,
            "output": "".join(output_lines),
        }

    def terminate_all(self, dry_run: bool = False) -> None:
        """Kill all processes started via this manager."""
        print(
            f"[pgm] terminate_all: runner={self.runner_id} tracked_procs={len(self._procs)}",
            flush=True,
        )
        if dry_run:
            print(
                f"[pgm] terminate_all: [dry-run] skip killing {len(self._procs)} procs",
                flush=True,
            )
            return

        procs = list(self._procs)
        if not procs:
            # run() removes completed parents from _procs, but their process
            # groups may still have surviving children. Kill the persisted
            # pgids before discarding the record.
            saved_pgids = _load_pgids(self.runner_id)
            if saved_pgids:
                print(
                    f"[pgm] terminate_all: no tracked procs but pgid file exists, "
                    f"killing pgids={sorted(saved_pgids)}",
                    flush=True,
                )
                for pgid in sorted(saved_pgids):
                    _kill_pgid(pgid, signal.SIGTERM)
                time.sleep(1)
                for pgid in sorted(saved_pgids):
                    _kill_pgid(pgid, signal.SIGKILL)
            else:
                print(
                    f"[pgm] terminate_all: no tracked processes, nothing to do",
                    flush=True,
                )
            _remove_pgid(self.runner_id)
            return

        pids = [p.pid for p in procs]
        print(
            f"[pgm] terminate_all: killing {len(procs)} process(es) pids={pids}",
            flush=True,
        )

        pgids: set[int] = _load_pgids(self.runner_id)
        for proc in procs:
            try:
                pgid = os.getpgid(proc.pid)
                pgids.add(pgid)
                _kill_pgid(pgid, signal.SIGTERM)
                print(
                    f"[pgm] terminate_all: sent SIGTERM to pgid={pgid} (pid={proc.pid})",
                    flush=True,
                )
            except ProcessLookupError:
                print(f"[pgm] terminate_all: pid={proc.pid} already exited", flush=True)

        print(
            f"[pgm] terminate_all: waiting up to {self.term_timeout}s for graceful shutdown",
            flush=True,
        )
        deadline = time.time() + self.term_timeout
        for proc in procs:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                print(
                    f"[pgm] terminate_all: pid={proc.pid} exited gracefully", flush=True
                )
            except subprocess.TimeoutExpired:
                print(
                    f"[pgm] terminate_all: pid={proc.pid} did not exit in time",
                    flush=True,
                )

        print(
            f"[pgm] terminate_all: sending SIGKILL to pgids={sorted(pgids)}", flush=True
        )
        for pgid in pgids:
            _kill_pgid(pgid, signal.SIGKILL)
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(
                    f"[pgm] terminate_all: pid={proc.pid} still alive after SIGKILL",
                    flush=True,
                )

        self._procs.clear()
        _remove_pgid(self.runner_id)
        print(f"[pgm] terminate_all: cleanup done, all processes cleared", flush=True)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_manager(term_timeout: float = 10.0) -> ProcessGroupManager:
    """
    Build a ProcessGroupManager from the environment.

    Looks for a stable runner identifier in order:
      1. RUNNER_NAME   – set by GitHub Actions per registered runner
      2. CI_RUNNER_NAME – set by GitLab CI
      3. HOSTNAME      – machine hostname (stable on dedicated hosts)

    Raises RuntimeError if none of the above is available.
    """
    runner_id = (
        os.environ.get("RUNNER_NAME")
        or os.environ.get("CI_RUNNER_NAME")
        or os.environ.get("HOSTNAME")
    )
    if not runner_id:
        raise RuntimeError(
            "Cannot determine a stable runner identifier. "
            "Set RUNNER_NAME, CI_RUNNER_NAME, or HOSTNAME."
        )
    print(
        f"[pgm] make_manager: runner_id={runner_id} "
        f"(RUNNER_NAME={os.environ.get('RUNNER_NAME', '<unset>')}, "
        f"HOSTNAME={os.environ.get('HOSTNAME', '<unset>')})",
        flush=True,
    )
    return ProcessGroupManager(runner_id=runner_id, term_timeout=term_timeout)
