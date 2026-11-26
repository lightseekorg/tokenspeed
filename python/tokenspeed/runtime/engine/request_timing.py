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

"""Role-local engine timing, encoded as typed maps on the output wire."""

from __future__ import annotations

import msgspec


class RequestClock(msgspec.Struct, kw_only=True):
    """Clock context shared by every boundary in one request timing record."""

    host_id: str
    process_instance_id: str
    clock_sync: str
    clock_uncertainty_ns: str | None


class RequestTimestamp(msgspec.Struct, kw_only=True):
    """Nanosecond decimal strings in the enclosing record's wall/monotonic clocks."""

    ts_unix_ns: str
    ts_mono_ns: str | None


class RequestTimingEvents(msgspec.Struct, kw_only=True):
    """Output-processor registration, first KV readiness and local completion."""

    accepted: RequestTimestamp | None
    kv_ready: RequestTimestamp | None
    finished: RequestTimestamp


class RequestTiming(msgspec.Struct, kw_only=True):
    """One local engine request; identity and final token counts stay on its output.

    ``kv_output_tokens`` counts output tokens at the first readiness boundary.
    ``e2e_latency_s`` retains the native latency from frontend ``created_time``;
    registration-to-finish and client-observed latency use different boundaries.
    """

    role: str
    dp_rank: int | None
    tp_rank: int | None
    clock: RequestClock
    events: RequestTimingEvents
    kv_output_tokens: int | None
    e2e_latency_s: float | None
