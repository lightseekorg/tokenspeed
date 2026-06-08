"""Background poller for server-wide Prometheus metrics.

Some quality signals aren't attributable to a single response — spec-decode
acceptance, for instance, is only exposed as global counters
(``tokenspeed:spec_decode_num_accepted_tokens`` / ``...num_drafts``), scraped
from the server's ``/metrics`` endpoint. This monitor samples them on a fixed
cadence, derives the windowed acceptance length, and raises a FATAL
``audit_finding`` when it drops below a floor — the spec-decode equivalent of
the per-response auditors. The counters only emit a series once spec decode has
actually run, so this check is inert unless spec decode is enabled.

accept_len = Δaccepted_draft_tokens / Δdrafts + 1   (the +1 is the always-
sampled bonus token). A value near 1.0 means almost no draft tokens are being
accepted, i.e. speculation is buying nothing — usually a correctness bug (e.g.
corrupted speculative coins), a regression, or a misconfigured drafter.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Dict, Optional, Tuple

import aiohttp

from ..audits import SEVERITY_FATAL
from ..events import AUDIT_FINDING, METRICS_PROBE, JsonlSink

# Counter base names we care about. prometheus_client appends `_total` to
# counters in the exposition format, so match either form.
_ACCEPTED = "tokenspeed:spec_decode_num_accepted_tokens"
_DRAFTS = "tokenspeed:spec_decode_num_drafts"

# `name{labels} value` or `name value`; skip `#` comment/HELP/TYPE lines.
_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)$")


def _sum_counter(text: str, base: str) -> Optional[float]:
    """Sum all label series for a counter, matching `base` or `base_total`.

    Returns None if the counter is absent (e.g. spec decode disabled), so the
    caller can stay silent rather than reporting a bogus 0.
    """
    total = 0.0
    seen = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if m is None:
            continue
        name = m.group(1)
        if name == base or name == base + "_total":
            try:
                total += float(m.group(3))
                seen = True
            except ValueError:
                continue
    return total if seen else None


class MetricsMonitor:
    """Scrapes /metrics and flags spec-decode acceptance below a floor."""

    def __init__(
        self,
        base_url: str,
        sink: JsonlSink,
        interval_s: float = 5.0,
        accept_len_min: float = 1.1,
        min_window_drafts: int = 200,
        consecutive_below: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.sink = sink
        self.interval_s = interval_s
        self.accept_len_min = accept_len_min
        self.min_window_drafts = min_window_drafts
        self.consecutive_below = consecutive_below
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._prev: Optional[Tuple[float, float]] = None  # (accepted, drafts)
        self._below_streak = 0
        self._fired = False

    async def _scrape(self, session: aiohttp.ClientSession) -> Optional[str]:
        url = f"{self.base_url}/metrics"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10.0)
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

    async def _probe_once(self, session: aiohttp.ClientSession) -> None:
        text = await self._scrape(session)
        if text is None:
            return
        accepted = _sum_counter(text, _ACCEPTED)
        drafts = _sum_counter(text, _DRAFTS)
        if accepted is None or drafts is None:
            return  # spec decode not enabled / counters absent — stay silent

        fields: Dict[str, float] = {
            "spec_accepted_total": accepted,
            "spec_drafts_total": drafts,
        }
        accept_len: Optional[float] = None
        if self._prev is None:
            self._prev = (accepted, drafts)
        else:
            d_acc = accepted - self._prev[0]
            d_draft = drafts - self._prev[1]
            if d_draft < 0 or d_acc < 0:
                # Counters went backwards => server restart (Prometheus counters
                # reset to 0). Re-baseline; do NOT emit a spurious negative.
                self._prev = (accepted, drafts)
            elif d_draft >= self.min_window_drafts:
                # Window closed: compute and re-baseline. Only advance _prev here
                # so that under frequent scrapes / low traffic the window keeps
                # accumulating until it has enough drafts (otherwise d_draft stays
                # below threshold forever and accept_len is never computed).
                accept_len = d_acc / d_draft + 1.0
                fields["accept_len"] = round(accept_len, 4)
                self._prev = (accepted, drafts)

        self.sink.emit(METRICS_PROBE, **fields)

        # accept_len is non-None only once a draft window has closed, which can
        # only happen when spec decode is actually running (the counters emit no
        # series otherwise and we already returned above). So the floor check
        # below inherently fires only when spec decode is enabled.
        if accept_len is None:
            return

        if accept_len < self.accept_len_min:
            self._below_streak += 1
            # A sustained drop toward ~1.0 means spec decode is accepting (close
            # to) zero draft tokens — a correctness signature (e.g. corrupted
            # speculative coins), not a tuning dip. Fire FATAL once so it lands
            # in the loud top-of-summary banner rather than the quiet audit block.
            if self._below_streak >= self.consecutive_below and not self._fired:
                self._fired = True
                self.sink.emit(
                    AUDIT_FINDING,
                    rid="",
                    workload="__server__",
                    check="spec_acceptance",
                    severity=SEVERITY_FATAL,
                    detail=(
                        f"accept_len={accept_len:.3f} below {self.accept_len_min} "
                        f"for {self._below_streak} consecutive windows — spec "
                        "decode accepting ~zero draft tokens"
                    ),
                    value=round(accept_len, 4),
                )
        else:
            self._below_streak = 0

    async def _loop(self) -> None:
        async with aiohttp.ClientSession() as session:
            while not self._stop.is_set():
                tick = time.time()
                try:
                    await self._probe_once(session)
                except Exception:  # noqa: BLE001 — never let the monitor die
                    pass
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
