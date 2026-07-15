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

import argparse
import asyncio
import time

import pytest

from tokenspeed import bench


def _parse_serving_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bench.add_serving_cli_args(parser)
    return parser.parse_args(argv)


def test_aiohttp_timeout_defaults_have_sock_subtimeouts():
    """Regression guard: a future refactor must keep ``sock_read`` set.

    Without ``sock_read``, ``aiohttp.StreamReader._wait`` awaits forever on
    a silent socket and one stuck stream-response future blocks the entire
    benchmark's ``asyncio.gather`` at high concurrency. We assert that the
    explicit CLI defaults are bounded and strictly tighter than ``total`` so
    an indefinitely silent socket still surfaces.
    """
    timeout = bench.make_http_timeout(
        bench.DEFAULT_HTTP_TOTAL_TIMEOUT_SEC,
        bench.DEFAULT_HTTP_SOCK_CONNECT_TIMEOUT_SEC,
        bench.DEFAULT_HTTP_SOCK_READ_TIMEOUT_SEC,
    )
    assert timeout.sock_read is not None
    assert timeout.total is not None
    assert 0 < timeout.sock_read < timeout.total, (
        f"sock_read {timeout.sock_read} must be smaller than total "
        f"{timeout.total} so it actually fires before the umbrella"
    )
    assert (
        timeout.sock_connect is not None
    ), "benchmark HTTP timeout must also set sock_connect"
    assert 0 < timeout.sock_connect <= timeout.sock_read


def test_timeout_cli_defaults_do_not_read_legacy_environment(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_BENCH_TOTAL_TIMEOUT_SEC", "1")
    monkeypatch.setenv("TOKENSPEED_BENCH_SOCK_CONNECT_TIMEOUT_SEC", "2")
    monkeypatch.setenv("TOKENSPEED_BENCH_SOCK_READ_TIMEOUT_SEC", "3")
    monkeypatch.setenv("TOKENSPEED_BENCH_PER_REQUEST_TIMEOUT_SEC", "4")

    args = _parse_serving_args([])

    assert args.http_total_timeout_sec == bench.DEFAULT_HTTP_TOTAL_TIMEOUT_SEC
    assert (
        args.http_sock_connect_timeout_sec
        == bench.DEFAULT_HTTP_SOCK_CONNECT_TIMEOUT_SEC
    )
    assert args.http_sock_read_timeout_sec == bench.DEFAULT_HTTP_SOCK_READ_TIMEOUT_SEC
    assert args.per_request_timeout_sec == bench.DEFAULT_PER_REQUEST_TIMEOUT_SEC


def test_timeout_cli_accepts_explicit_values():
    args = _parse_serving_args(
        [
            "--http-total-timeout-sec",
            "120",
            "--http-sock-connect-timeout-sec",
            "4",
            "--http-sock-read-timeout-sec",
            "30",
            "--per-request-timeout-sec",
            "60",
        ]
    )

    assert args.http_total_timeout_sec == 120
    assert args.http_sock_connect_timeout_sec == 4
    assert args.http_sock_read_timeout_sec == 30
    assert args.per_request_timeout_sec == 60


def test_timeout_cli_rejects_non_positive_values():
    with pytest.raises(SystemExit):
        _parse_serving_args(["--per-request-timeout-sec", "0"])


@pytest.mark.asyncio
async def test_await_with_per_request_timeout_returns_failed_output_on_hang():
    """A stuck request must time out instead of blocking forever."""

    async def stuck_request() -> bench.RequestFuncOutput:
        await asyncio.sleep(60)  # would block past the gather without the wrap
        return bench.RequestFuncOutput()  # pragma: no cover

    start = time.perf_counter()
    output = await bench.await_with_per_request_timeout(
        stuck_request(), prompt_len=42, timeout_sec=0.1
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"expected sub-second timeout, took {elapsed:.2f}s"
    assert output.success is False
    assert "per-request timeout" in output.error
    assert output.prompt_len == 42


@pytest.mark.asyncio
async def test_await_with_per_request_timeout_passes_through_success():
    """The wrap must not perturb a request that completes normally."""

    async def fast_request() -> bench.RequestFuncOutput:
        output = bench.RequestFuncOutput()
        output.success = True
        output.prompt_len = 13
        output.generated_text = "hello"
        return output

    result = await bench.await_with_per_request_timeout(
        fast_request(), prompt_len=13, timeout_sec=5.0
    )

    assert result.success is True
    assert result.generated_text == "hello"


@pytest.mark.asyncio
async def test_concurrent_stuck_request_does_not_block_gather():
    """End-to-end shape: stuck + healthy requests gather together cleanly."""

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
        bench.await_with_per_request_timeout(stuck(), prompt_len=1, timeout_sec=0.2),
        bench.await_with_per_request_timeout(
            healthy(0.05), prompt_len=2, timeout_sec=0.2
        ),
        bench.await_with_per_request_timeout(stuck(), prompt_len=3, timeout_sec=0.2),
        bench.await_with_per_request_timeout(
            healthy(0.05), prompt_len=4, timeout_sec=0.2
        ),
    )
    elapsed = time.perf_counter() - start

    # Without the timeout wrap this gather would block on the two stuck
    # requests forever; with it the gather returns in roughly the timeout.
    assert elapsed < 1.5, f"gather elapsed {elapsed:.2f}s, expected ~0.2s"
    assert [r.success for r in results] == [False, True, False, True]
    assert all("per-request timeout" in r.error for r in results if not r.success)
