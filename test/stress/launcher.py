"""Optional server launcher.

Spawns `tokenspeed serve` (or any other command) in its own process group,
tees logs to a file, polls /health until 200 or timeout, and tears the
whole group down at the end. This is a convenience wrapper -- if you're
already running the server by hand, just don't pass --launch-cmd.

Design notes:
  * Process *group*, not just process. `tokenspeed serve` forks TP workers
    and a C++ scheduler; a bare SIGTERM to the python wrapper can orphan
    them and leave GPUs stuck. We `os.setsid()` the child so one signal
    to the group takes the whole tree down.
  * We poll /health rather than scraping the log for a "ready" string --
    health is the authoritative signal the server itself exposes.
  * Startup can take minutes (model weight load + CUDA graph capture).
    The default readiness timeout is generous; bump it if needed.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp


@dataclass
class LaunchConfig:
    cmd: str  # shell command, e.g. "bash run-mm25.sh"
    base_url: str
    log_path: str
    readiness_timeout_s: float = 900.0
    readiness_poll_s: float = 2.0
    shutdown_grace_s: float = 30.0


class ServerProcess:
    def __init__(self, cfg: LaunchConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._log_fp = None

    @property
    def pid(self) -> Optional[int]:
        """PID of the launched process group leader, or None if not started."""
        return self._proc.pid if self._proc is not None else None

    @property
    def running(self) -> bool:
        """True once start() has spawned a process (regardless of liveness)."""
        return self._proc is not None

    def _already_up(self) -> bool:
        """Cheap sync probe before we bother spawning anything."""
        try:
            import urllib.request

            req = urllib.request.Request(self.cfg.base_url.rstrip("/") + "/health")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start(self) -> None:
        if self._already_up():
            raise RuntimeError(
                f"Something is already serving on {self.cfg.base_url}. "
                "Refusing to launch a second one -- stop it first, or drop --launch-cmd."
            )
        os.makedirs(
            os.path.dirname(os.path.abspath(self.cfg.log_path)) or ".", exist_ok=True
        )
        self._log_fp = open(self.cfg.log_path, "w", buffering=1)
        print(f"[stress] launching: {self.cfg.cmd}", flush=True)
        print(f"[stress] server log: {self.cfg.log_path}", flush=True)
        # start_new_session=True => new process group (os.setsid under the hood).
        self._proc = subprocess.Popen(
            self.cfg.cmd,
            shell=True,
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def wait_ready(self) -> None:
        assert self._proc is not None, "call start() first"
        deadline = time.time() + self.cfg.readiness_timeout_s
        url = self.cfg.base_url.rstrip("/") + "/health"
        last_err = "(no probe yet)"
        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"server process exited with code {self._proc.returncode} "
                        f"before becoming ready; see {self.cfg.log_path}"
                    )
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=5.0)
                    ) as resp:
                        if resp.status == 200:
                            print(f"[stress] /health is 200; server ready", flush=True)
                            return
                        last_err = f"status={resp.status}"
                except Exception as e:  # noqa: BLE001
                    last_err = f"{type(e).__name__}: {e}"[:200]
                await asyncio.sleep(self.cfg.readiness_poll_s)
        raise TimeoutError(
            f"server not ready after {self.cfg.readiness_timeout_s:.0f}s "
            f"(last probe: {last_err}); see {self.cfg.log_path}"
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            # Already exited on its own.
            self._close_log()
            return
        pgid = os.getpgid(self._proc.pid)
        print(f"[stress] stopping server (pgid={pgid})", flush=True)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self._close_log()
            return
        try:
            self._proc.wait(timeout=self.cfg.shutdown_grace_s)
        except subprocess.TimeoutExpired:
            print(
                f"[stress] server did not exit in {self.cfg.shutdown_grace_s}s; "
                "escalating to SIGKILL",
                flush=True,
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                print("[stress] SIGKILL did not reap the group; giving up", flush=True)
        self._close_log()

    def _close_log(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None

    # Context-manager sugar so callers can't forget to stop().
    def __enter__(self) -> "ServerProcess":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
