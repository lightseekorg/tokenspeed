"""Background poller for /health and /health_generate.

Both endpoints in tokenspeed currently share one handler that waits for any
detokenizer heartbeat within HEALTH_CHECK_TIMEOUT (see
python/tokenspeed/runtime/entrypoints/http_server.py:281). We still poll both
paths so that:

  1. If the handler is ever split (cheap liveness vs. deep readiness), the
     monitor transparently picks up the new behaviour.
  2. We cross-check that both paths agree under load / during injected faults.

Emits one `health_probe` event per poll and one `health_transition` event
whenever the ok/fail verdict flips for an endpoint.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterable, Optional

import aiohttp

from ..events import HEALTH_PROBE, HEALTH_TRANSITION, JsonlSink

DEFAULT_PATHS = ("/health", "/health_generate")


class HealthMonitor:
    def __init__(
        self,
        base_url: str,
        sink: JsonlSink,
        interval_s: float = 1.0,
        request_timeout_s: float = 30.0,
        paths: Iterable[str] = DEFAULT_PATHS,
    ):
        self.base_url = base_url.rstrip("/")
        self.sink = sink
        self.interval_s = interval_s
        self.request_timeout_s = request_timeout_s
        self.paths = tuple(paths)
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._last_ok: dict[str, Optional[bool]] = {p: None for p in self.paths}

    async def _probe_once(self, session: aiohttp.ClientSession, path: str) -> None:
        url = f"{self.base_url}{path}"
        start = time.time()
        ok = False
        status: Optional[int] = None
        detail = ""
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=self.request_timeout_s)
            ) as resp:
                status = resp.status
                # 200 = healthy; anything else (notably 503) = not.
                ok = resp.status == 200
                # Drain body so the connection can be reused.
                await resp.read()
        except asyncio.TimeoutError:
            detail = "timeout"
        except aiohttp.ClientError as e:
            detail = f"{type(e).__name__}: {e}"[:200]
        except Exception as e:  # noqa: BLE001
            detail = f"{type(e).__name__}: {e}"[:200]
        latency = time.time() - start
        self.sink.emit(
            HEALTH_PROBE,
            endpoint=path,
            ok=ok,
            status=status,
            latency_s=latency,
            detail=detail,
        )
        prev = self._last_ok[path]
        cur_label = "ok" if ok else "fail"
        if prev is None:
            self._last_ok[path] = ok
        elif prev != ok:
            self.sink.emit(
                HEALTH_TRANSITION,
                endpoint=path,
                **{"from": "ok" if prev else "fail", "to": cur_label},
            )
            self._last_ok[path] = ok

    async def _loop(self) -> None:
        connector = aiohttp.TCPConnector(limit=len(self.paths) * 2, force_close=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            while not self._stop.is_set():
                tick = time.time()
                await asyncio.gather(
                    *(self._probe_once(session, p) for p in self.paths),
                    return_exceptions=True,
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
