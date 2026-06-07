"""Background poller for server RSS + process-tree memory.

Reads ``/proc/<pid>/status`` (VmRSS) for a root PID and all its
descendants at a fixed cadence, and emits one ``rss_probe`` event per
poll. Useful for catching slow CPU memory growth that unfolds over
hours of stress traffic.

The tokenspeed server forks TP workers and a C++ scheduler; each has
its own Python process. We walk the full tree so the summary reflects
every Python process under the server's root.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List, Optional

from ..events import RSS_PROBE, JsonlSink


def _read_vm_rss_kb(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _read_comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", "r") as f:
            return f.read().strip()
    except OSError:
        return "?"


def _walk_children(root_pid: int) -> List[int]:
    """Return root_pid and all descendants via procfs ``children``.

    Uses the per-thread ``/proc/<pid>/task/<tid>/children`` interface
    (CONFIG_PROC_CHILDREN, default-on on modern Linux). We silently drop
    PIDs that die mid-walk — descendants vanish all the time during a
    stress run.
    """
    seen = {root_pid}
    frontier = [root_pid]
    while frontier:
        nxt: List[int] = []
        for pid in frontier:
            try:
                tids = os.listdir(f"/proc/{pid}/task")
            except OSError:
                continue
            for tid in tids:
                try:
                    with open(f"/proc/{pid}/task/{tid}/children", "r") as f:
                        for c in f.read().split():
                            try:
                                c_int = int(c)
                            except ValueError:
                                continue
                            if c_int not in seen:
                                seen.add(c_int)
                                nxt.append(c_int)
                except OSError:
                    continue
        frontier = nxt
    return sorted(seen)


class RssMonitor:
    """Polls VmRSS for a process tree and emits rss_probe events.

    The monitor records per-PID RSS + aggregate total. Post-run tooling
    can compute the growth slope (kB/min) from the first and last
    samples to quantify the leak.
    """

    def __init__(
        self,
        root_pid: int,
        sink: JsonlSink,
        interval_s: float = 2.0,
    ) -> None:
        self.root_pid = root_pid
        self.sink = sink
        self.interval_s = interval_s
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def _probe_once(self) -> None:
        pids = _walk_children(self.root_pid)
        total_kb = 0
        per_pid: Dict[str, Dict[str, int]] = {}
        for pid in pids:
            rss = _read_vm_rss_kb(pid)
            if rss is None:
                continue
            per_pid[str(pid)] = {"rss_kb": rss, "comm": _read_comm(pid)}
            total_kb += rss
        self.sink.emit(
            RSS_PROBE,
            root_pid=self.root_pid,
            total_kb=total_kb,
            num_pids=len(per_pid),
            per_pid=per_pid,
        )

    async def _loop(self) -> None:
        while not self._stop.is_set():
            tick = time.time()
            try:
                await self._probe_once()
            except Exception as e:  # noqa: BLE001
                # Don't let a transient /proc read error take down the monitor.
                self.sink.emit(
                    RSS_PROBE,
                    root_pid=self.root_pid,
                    error=f"{type(e).__name__}: {e}"[:200],
                )
            elapsed = time.time() - tick
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=max(0.0, self.interval_s - elapsed),
                )
            except asyncio.TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
