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

from __future__ import annotations

import asyncio
import importlib.util
import json
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from socket import socket

import pytest

pytest.importorskip("aiohttp")
from aiohttp import ClientSession, web

_SPEC = importlib.util.spec_from_file_location(
    "gdn_serving_benchmark",
    Path(__file__).resolve().parents[2]
    / "benchmarks/nvidia/bench_gdn_deferred_serving.py",
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def _metrics(*ranks: tuple[str, float, float, float]) -> str:
    return "\n".join(
        f'{name}{{dp_rank="{rank}",model_name="model with spaces"}} {value}'
        for rank, running, waiting, active in ranks
        for name, value in [
            ("tokenspeed:num_requests_running", running),
            ("tokenspeed:num_requests_waiting", waiting),
            ("tokenspeed:kv_cache_usage_perc", active),
        ]
    )


def test_idle_ignores_resident_prefix_cache():
    metrics = _metrics(("0", 0, 0, 0), ("1", 0, 0, 0))
    metrics += '\ntokenspeed:kv_token_usage_perc{dp_rank="0"} 0.75\n'
    assert benchmark.engine_is_idle(metrics)


@pytest.mark.parametrize("activity", [(1, 0, 0), (0, 1, 0), (0, 0, 0.1)])
def test_idle_requires_all_ranks_to_release_requests_and_pages(activity):
    # A stopped overlap forward may hold pages after its last response.
    metrics = _metrics(("0", 0, 0, 0), ("1", *activity))
    assert not benchmark.engine_is_idle(metrics)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_idle_rejects_invalid_metrics(value):
    with pytest.raises(ValueError, match="invalid engine activity metric"):
        benchmark.engine_is_idle(_metrics(("0", 0, 0, value)))


@pytest.mark.parametrize(
    "missing",
    [
        "tokenspeed:num_requests_running",
        "tokenspeed:num_requests_waiting",
        "tokenspeed:kv_cache_usage_perc",
    ],
)
def test_idle_rejects_missing_metrics(missing):
    metrics = "\n".join(
        line
        for line in _metrics(("0", 0, 0, 0)).splitlines()
        if not line.startswith(missing)
    )
    with pytest.raises(ValueError, match="engine metrics are missing"):
        benchmark.engine_is_idle(metrics)


@asynccontextmanager
async def _metrics_server(handler, generate_handler=None):
    app = web.Application()
    app.router.add_get("/metrics", handler)
    if generate_handler is not None:
        app.router.add_post("/generate", generate_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket() as listener:
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.setblocking(False)
            port = listener.getsockname()[1]
            site = web.SockSite(runner, listener)
            await site.start()
            yield f"http://127.0.0.1:{port}/metrics"
        finally:
            await runner.cleanup()


def test_wait_until_idle_drains_stopped_forward_pages():
    async def run():
        active_pages = [0.1]
        observed = asyncio.Event()

        async def metrics(_request):
            observed.set()
            return web.Response(text=_metrics(("0", 0, 0, active_pages[0])))

        async def finish_forward():
            await observed.wait()
            await asyncio.sleep(0.02)
            active_pages[0] = 0

        async with _metrics_server(metrics) as url, ClientSession() as session:
            forward = asyncio.create_task(finish_forward())
            try:
                drain_ms = await benchmark.wait_until_idle(session, url, timeout_s=1)
                assert active_pages[0] == 0
                assert drain_ms >= 0
            finally:
                forward.cancel()
                await asyncio.gather(forward, return_exceptions=True)

    asyncio.run(run())


def test_wait_until_idle_times_out_on_busy_engine():
    async def run():
        async def metrics(_request):
            return web.Response(text=_metrics(("0", 0, 0, 0.1)))

        async with _metrics_server(metrics) as url, ClientSession() as session:
            with pytest.raises(TimeoutError, match="did not become idle"):
                await benchmark.wait_until_idle(session, url, timeout_s=0.02)

    asyncio.run(run())


def test_wait_until_idle_rejects_wrong_endpoint():
    async def run():
        async def metrics(_request):
            return web.Response(status=404)

        async with _metrics_server(metrics) as url, ClientSession() as session:
            with pytest.raises(RuntimeError, match="HTTP 404"):
                await benchmark.wait_until_idle(session, url, timeout_s=1)

    asyncio.run(run())


def test_wait_until_idle_timeout_covers_metrics_response():
    async def run():
        async def metrics(_request):
            await asyncio.sleep(0.05)
            return web.Response(text=_metrics(("0", 0, 0, 0)))

        async with _metrics_server(metrics) as url, ClientSession() as session:
            with pytest.raises(TimeoutError, match="did not become idle"):
                await benchmark.wait_until_idle(session, url, timeout_s=0.01)

    asyncio.run(run())


@pytest.mark.parametrize("structured_finish", [False, True])
def test_measure_starts_requests_after_warmup_pages_are_released(
    tmp_path, structured_finish
):
    async def run():
        active_pages = [0.0]
        observed = asyncio.Event()

        async def metrics(_request):
            if active_pages[0] > 0:
                observed.set()
            return web.Response(text=_metrics(("0", 0, 0, active_pages[0])))

        async def generate(request):
            body = await request.json()
            tokens = body["sampling_params"]["max_new_tokens"]
            if tokens == 32:
                active_pages[0] = 0.1
            elif active_pages[0] > 0:
                raise web.HTTPConflict(text="warmup forward still owns pages")
            finish = (
                {"type": "length", "length": tokens} if structured_finish else "length"
            )
            updates = [
                {"meta_info": {"completion_tokens": 1, "finish_reason": None}},
                {"meta_info": {"completion_tokens": tokens, "finish_reason": finish}},
            ]
            return web.Response(
                text="".join(f"data: {json.dumps(item)}\n\n" for item in updates),
                content_type="text/event-stream",
            )

        async def finish_forward():
            await observed.wait()
            await asyncio.sleep(0.02)
            active_pages[0] = 0

        async with _metrics_server(metrics, generate) as metrics_url:
            output = tmp_path / "result.json"
            args = Namespace(
                url=metrics_url.removesuffix("/metrics"),
                metrics_url=metrics_url,
                model="test-model",
                concurrency=1,
                requests=2,
                output_tokens=4,
                output=output,
            )
            forward = asyncio.create_task(finish_forward())
            try:
                await benchmark.measure(args)
                result = json.loads(output.read_text())
                assert result["output_tokens"] == 8
                assert result["warmup_completion"] == "engine_idle"
                assert result["warmup_drain_ms"] > 0
                assert result["ttft_p50_ms"] > 0
                assert result["ttft_p95_ms"] <= result["ttft_p99_ms"]
            finally:
                forward.cancel()
                await asyncio.gather(forward, return_exceptions=True)

    asyncio.run(run())
