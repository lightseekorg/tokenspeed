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

"""Tests for the per-request timeout guards in :mod:`tokenspeed.bench`.

These tests don't talk to a real server; they exercise the timeout helper
directly. The point is to lock in the behaviour that one stuck
stream-response future cannot deadlock the outer ``asyncio.gather``: instead
it surfaces as a normal ``RequestFuncOutput`` marked failed.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tokenspeed import bench


def test_aiohttp_timeout_has_sock_subtimeouts():
    """Regression guard: a future refactor must keep ``sock_read`` set.

    Without ``sock_read``, ``aiohttp.StreamReader._wait`` awaits forever on
    a silent socket and one stuck stream-response future blocks the entire
    benchmark's ``asyncio.gather`` at high concurrency. The exact value is
    user-tunable via env; we only assert it's bounded and strictly tighter
    than ``total`` so an indefinitely silent socket still surfaces.
    """
    timeout = bench.AIOHTTP_TIMEOUT
    assert timeout.sock_read is not None, "AIOHTTP_TIMEOUT must set sock_read"
    assert timeout.total is not None
    assert 0 < timeout.sock_read < timeout.total, (
        f"sock_read {timeout.sock_read} must be smaller than total "
        f"{timeout.total} so it actually fires before the umbrella"
    )
    assert timeout.sock_connect is not None, (
        "AIOHTTP_TIMEOUT must also set sock_connect"
    )
    assert 0 < timeout.sock_connect <= timeout.sock_read


def test_per_request_timeout_constant_is_positive():
    assert bench.PER_REQUEST_TIMEOUT_SEC > 0


@pytest.mark.asyncio
async def test_await_with_per_request_timeout_returns_failed_output_on_hang(
    monkeypatch,
):
    """A stuck request must time out instead of blocking forever."""
    monkeypatch.setattr(bench, "PER_REQUEST_TIMEOUT_SEC", 0.1)

    async def stuck_request() -> bench.RequestFuncOutput:
        await asyncio.sleep(60)  # would block past the gather without the wrap
        return bench.RequestFuncOutput()  # pragma: no cover

    start = time.perf_counter()
    output = await bench.await_with_per_request_timeout(
        stuck_request(), prompt_len=42
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"expected sub-second timeout, took {elapsed:.2f}s"
    assert output.success is False
    assert "per-request timeout" in output.error
    assert output.prompt_len == 42


@pytest.mark.asyncio
async def test_await_with_per_request_timeout_passes_through_success(monkeypatch):
    """The wrap must not perturb a request that completes normally."""
    monkeypatch.setattr(bench, "PER_REQUEST_TIMEOUT_SEC", 5.0)

    async def fast_request() -> bench.RequestFuncOutput:
        output = bench.RequestFuncOutput()
        output.success = True
        output.prompt_len = 13
        output.generated_text = "hello"
        return output

    result = await bench.await_with_per_request_timeout(
        fast_request(), prompt_len=13
    )

    assert result.success is True
    assert result.generated_text == "hello"


@pytest.mark.asyncio
async def test_concurrent_stuck_request_does_not_block_gather(monkeypatch):
    """End-to-end shape: stuck + healthy requests gather together cleanly."""
    monkeypatch.setattr(bench, "PER_REQUEST_TIMEOUT_SEC", 0.2)

    async def stuck() -> bench.RequestFuncOutput:
        await asyncio.sleep(60)
        return bench.RequestFuncOutput()  # pragma: no cover

    async def healthy(latency: float) -> bench.RequestFuncOutput:
        await asyncio.sleep(latency)
        out = bench.RequestFuncOutput()
        out.success = True
        return out

    start = time.perf_counter()
    results = await asyncio.gather(
        bench.await_with_per_request_timeout(stuck(), prompt_len=1),
        bench.await_with_per_request_timeout(healthy(0.05), prompt_len=2),
        bench.await_with_per_request_timeout(stuck(), prompt_len=3),
        bench.await_with_per_request_timeout(healthy(0.05), prompt_len=4),
    )
    elapsed = time.perf_counter() - start

    # Without the timeout wrap this gather would block on the two stuck
    # requests forever; with it the gather returns in roughly the timeout.
    assert elapsed < 1.5, f"gather elapsed {elapsed:.2f}s, expected ~0.2s"
    assert [r.success for r in results] == [False, True, False, True]
    assert all("per-request timeout" in r.error for r in results if not r.success)
