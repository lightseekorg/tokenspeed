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

"""The event loop's side of a shared L2 pool: lookups, commands and completions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.shared_l2.client import SharedL2Client

# How often a forward's wait checks for arrivals to look up in the pool.
_ARRIVAL_POLL_S = 0.002


class SharedL2Hooks:
    """Glue between the event loop and the rank's shared L2 client; every call
    is a no-op without a pool."""

    def __init__(self, loop: Any, client: SharedL2Client | None):
        self._loop = loop
        self._client = client

    @property
    def active(self) -> bool:
        return self._client is not None

    @property
    def scheduler_config(self):
        """The scheduler's shared L2 limits, or None without a pool."""
        return self._client.config.scheduler if self._client is not None else None

    def poll_ready_events(self) -> list:
        """Completed pool writes, as scheduler events."""
        return self._client.poll_events() if self._client is not None else []

    def lookup(self) -> None:
        """Mirror the pool's hits of the waiting prompts not yet looked up into
        the scheduler's proxy Host tier, in one owner request."""
        if self._client is None:
            return
        scheduler = self._loop.scheduler
        lookups = scheduler.shared_l2_lookups()
        if lookups:
            results = self._client.lookup(lookups)
            self._client.map_proxy_pages(scheduler.apply_shared_l2_lookups(results))

    def submit(self, operations) -> None:
        """Hand the plan's pool commands to the client."""
        if not operations:
            return
        if self._client is None:
            raise RuntimeError("shared L2 commands require a shared L2 client")
        self._client.submit(operations)

    def wait_for_forward(self, result) -> None:
        """Until a forward's result is ready, take in arriving prompts and look
        them up, so the next plan admits them without a lookup of its own."""
        if self._client is None:
            return
        while not result.ready():
            if self._loop._process_new_requests():
                self.lookup()
            else:
                time.sleep(_ARRIVAL_POLL_S)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
