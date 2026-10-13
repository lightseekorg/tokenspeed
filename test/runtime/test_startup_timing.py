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

import json
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.utils import startup_timing as timing


def test_disabled_does_not_read_clocks_or_compile_stats(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_STARTUP_TIMING", "0")

    def unexpected():
        pytest.fail("disabled timing must not collect telemetry")

    monkeypatch.setattr(timing.time, "perf_counter", unexpected)
    monkeypatch.setattr(timing, "compile_stats", unexpected)
    with timing.startup_phase("disabled", rank=2):
        pass
    assert timing._current.get() is None


def test_nested_failure_preserves_exception_and_metadata(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_STARTUP_TIMING", "1")
    monkeypatch.setattr(timing, "compile_stats", lambda: None)
    lines = []
    monkeypatch.setattr(timing.logger, "info", lines.append)
    failure = ValueError("original failure")
    with pytest.raises(ValueError) as raised:
        with timing.startup_phase("scheduler.init", rank=3, role="decode"):
            with timing.startup_phase("weights.target"):
                raise failure
    assert raised.value is failure
    assert timing._current.get() is None
    events = [json.loads(line.split("startup_timing ", 1)[1]) for line in lines]
    outer, inner, inner_end, outer_end = events
    assert [e["event"] for e in events] == ["start", "start", "end", "end"]
    assert outer["parent_id"] is None
    assert inner["parent_id"] == outer["span_id"]
    assert inner_end["span_id"] == inner["span_id"]
    assert outer_end["span_id"] == outer["span_id"]
    assert all(e["rank"] == 3 and e["role"] == "decode" for e in events)
    for end in (inner_end, outer_end):
        assert end["status"] == "error"
        assert end["error_type"] == "ValueError"
        assert end["duration_s"] >= 0
        assert isinstance(end["wall_time_ns"], int)
        assert end["triton_compiles"] is None
        assert end["triton_compile_s"] is None


def test_compile_deltas_and_decorator_reuse(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_STARTUP_TIMING", "1")
    stats = iter(
        SimpleNamespace(startup_compiles=n, startup_seconds=s)
        for n, s in [(4, 1.0), (6, 1.5), (6, 1.5), (6, 1.5)]
    )
    monkeypatch.setattr(timing, "compile_stats", lambda: next(stats))
    lines = []
    monkeypatch.setattr(timing.logger, "info", lines.append)

    @timing.startup_phase("load")
    def load(value):
        return value

    assert load(7) == 7
    assert load(8) == 8
    events = [json.loads(line.split("startup_timing ", 1)[1]) for line in lines]
    assert events[1]["triton_compiles"] == 2
    assert events[1]["triton_compile_s"] == 0.5
    assert events[3]["triton_compiles"] == 0
    assert events[1]["status"] == events[3]["status"] == "ok"
    assert events[0]["span_id"] != events[2]["span_id"]
    assert events[2]["parent_id"] is None


def test_logging_failure_cannot_mask_startup_failure(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_STARTUP_TIMING", "1")
    monkeypatch.setattr(timing, "compile_stats", lambda: None)

    def broken_handler(message):
        raise OSError("log unavailable")

    monkeypatch.setattr(timing.logger, "info", broken_handler)
    failure = RuntimeError("model failed")
    with pytest.raises(RuntimeError) as raised:
        with timing.startup_phase("load"):
            raise failure
    assert raised.value is failure
    assert timing._current.get() is None
