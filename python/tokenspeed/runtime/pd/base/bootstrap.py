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
import logging
import threading
from dataclasses import dataclass

from aiohttp import web

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class BootstrapInfo:
    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int


class DisaggBootstrapServerBase:
    """HTTP rendezvous server shared by both PD and EPD transfer roles.

    A data-source rank PUTs its (ip, port) keyed by (dp_group, tp_rank_in_dp);
    the peer GETs that back to open a Mooncake session, and uses the sentinel
    ``engine_rank == -1`` query to sync the source's parallel sizes. The routing,
    dp-group sharding, and server lifecycle are role-neutral and live here;
    role-specific parallel-info fields (e.g. the KV path's MLA / kv-page lengths)
    are layered in by subclasses via :meth:`_ingest_put_extra` (record extra PUT
    fields) and :meth:`_extra_parallel_info` (add them to the GET sync response).
    """

    _STARTUP_TIMEOUT_SECONDS = 10.0
    _SHUTDOWN_TIMEOUT_SECONDS = 5.0

    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.lock = asyncio.Lock()
        self.world_size = None
        self.dp_size = None
        self.tp_size_per_dp_rank = None
        # Prefill chunk-pipeline stage count (1 when the source has no PP).
        self.pp_size = 1
        # Optional explicit per-stage layer counts; the Decode side must plan
        # its transfer routes over the SAME stage windows the Prefill used.
        self.pp_layer_partition: list[int] | None = None
        self.prefill_port_table: dict[int, dict[int, dict[str, str | int]]] = {}

        # Initialize every field shared with the server thread before starting
        # it.  In particular, close() may race with startup when construction
        # times out, so it must never observe partially-created attributes.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.BaseSite | None = None
        self._startup_error: BaseException | None = None
        self._runtime_error: BaseException | None = None
        self._startup_complete = threading.Event()
        self._stop_requested = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._thread_started = False

        self._setup_routes()
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        with self._lifecycle_lock:
            if self._thread_started:
                if (
                    self._startup_complete.is_set()
                    and self._startup_error is None
                    and self.thread.is_alive()
                ):
                    return
                raise RuntimeError("Bootstrap server thread cannot be restarted")
            self._thread_started = True
            self.thread.start()

        if not self._startup_complete.wait(self._STARTUP_TIMEOUT_SECONDS):
            self._request_stop()
            self.thread.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
            thread_state = (
                " and its thread is still running" if self.thread.is_alive() else ""
            )
            raise TimeoutError(
                "Bootstrap server did not bind within "
                f"{self._STARTUP_TIMEOUT_SECONDS:.1f}s{thread_state}"
            )

        if self._startup_error is not None:
            self.thread.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
            raise RuntimeError(
                f"Bootstrap server failed to bind on port {self.port}"
            ) from self._startup_error

        # _startup_complete is published only after TCPSite.start() returns.
        # A dead thread here means the loop failed between publishing readiness
        # and the constructor observing it; do not return a broken server.
        if not self.thread.is_alive():
            raise RuntimeError(
                f"Bootstrap server stopped while starting on port {self.port}"
            ) from self._runtime_error

    def _setup_routes(self):
        self.app.router.add_route("*", "/route", self._handle_route)
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request):
        return web.Response(text="OK", status=200)

    async def _handle_route(self, request: web.Request):
        method = request.method
        if method == "PUT":
            return await self._handle_route_put(request)
        elif method == "GET":
            return await self._handle_route_get(request)
        else:
            return web.Response(
                text="Method not allowed", status=405, content_type="application/json"
            )

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        role = data["role"]
        world_size = data["world_size"]
        dp_size = data["dp_size"]
        rank_ip = data["rank_ip"]
        rank_port = int(data["rank_port"])
        engine_rank = int(data["engine_rank"])
        self._ingest_put_extra(data)

        if self.world_size is None:
            self.world_size = world_size

        if self.dp_size is None:
            self.dp_size = dp_size

        self.pp_size = int(data.get("pp_size", self.pp_size or 1))
        if data.get("pp_layer_partition") is not None:
            self.pp_layer_partition = [
                int(count) for count in data["pp_layer_partition"]
            ]

        # With PP the per-dp world spans pp stages; the port table keys by the
        # dense stage-major rank (pp_rank * tp + tp_rank == global rank at
        # dp_size == 1, which PP requires).
        tp_size_per_dp_rank = world_size // dp_size
        if self.tp_size_per_dp_rank is None:
            self.tp_size_per_dp_rank = tp_size_per_dp_rank

        if role == "Prefill":
            dp_group = engine_rank // tp_size_per_dp_rank
            tp_rank_in_dp_group = engine_rank % tp_size_per_dp_rank

            # Add lock to make sure thread-safe
            async with self.lock:
                if dp_group not in self.prefill_port_table:
                    self.prefill_port_table[dp_group] = {}

            self.prefill_port_table[dp_group][tp_rank_in_dp_group] = {
                "rank_ip": rank_ip,
                "rank_port": rank_port,
            }
            logger.debug(
                "Register prefill bootstrap: %s with rank_ip: %s and rank_port: %s",
                engine_rank,
                rank_ip,
                rank_port,
            )

        return web.Response(text="OK", status=200)

    async def _handle_route_get(self, request: web.Request):
        engine_rank = request.query.get("engine_rank")
        target_dp_group = request.query.get("target_dp_group")
        if not engine_rank or not target_dp_group:
            return web.Response(text="Missing inputs for bootstrap server.", status=400)

        # Currently we use engine_rank == -1 and target_dp_group == -1 to sync dp size
        if int(engine_rank) == -1 and int(target_dp_group) == -1:
            prefill_parallel_info = {
                "prefill_tp_size": self.world_size,
                "prefill_dp_size": self.dp_size,
                "prefill_pp_size": self.pp_size,
                "prefill_pp_layer_partition": self.pp_layer_partition,
            }
            prefill_parallel_info.update(self._extra_parallel_info())
            return web.json_response(prefill_parallel_info, status=200)

        # Find corresponding prefill info
        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(target_dp_group)][
                int(engine_rank)
            ]

        if bootstrap_info is not None:
            return web.json_response(bootstrap_info, status=200)
        else:
            return web.Response(text="Bootstrap info not Found", status=404)

    def _ingest_put_extra(self, data: dict) -> None:
        """Record role-specific fields off a register PUT. Default: none."""

    def _extra_parallel_info(self) -> dict:
        """Role-specific fields to merge into the GET parallel-info sync response.
        Default: none."""
        return {}

    def _request_stop(self) -> None:
        self._stop_requested.set()
        loop = self._loop
        if loop is None or loop.is_closed() or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            # The loop can close between is_closed() and the thread-safe call.
            # In that case the server thread is already on its way out.
            return

    def _run_server(self):
        loop: asyncio.AbstractEventLoop | None = None
        runner: web.AppRunner | None = None
        startup_succeeded = False
        try:
            # Event Loop
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            runner = web.AppRunner(self.app, access_log=access_log)
            self._runner = runner
            loop.run_until_complete(runner.setup())

            site = web.TCPSite(runner, port=self.port)
            self._site = site
            loop.run_until_complete(site.start())
            startup_succeeded = True
            self._startup_complete.set()

            if not self._stop_requested.is_set():
                loop.run_forever()
        except BaseException as exc:
            if startup_succeeded:
                self._runtime_error = exc
                logger.exception("Bootstrap server failed after startup")
            else:
                self._startup_error = exc
                logger.error("Bootstrap server startup failed: %s", str(exc))
        finally:
            # Wake a constructor waiting for startup even when loop creation,
            # runner setup, or TCP bind failed.
            self._startup_complete.set()

            if loop is not None and not loop.is_closed():
                if runner is not None:
                    try:
                        loop.run_until_complete(runner.cleanup())
                    except BaseException as exc:
                        if self._runtime_error is None:
                            self._runtime_error = exc
                        logger.exception("Bootstrap server cleanup failed")
                loop.close()

    def close(self):
        """Stop the rendezvous server and wait for all socket cleanup.

        The operation is idempotent.  A bounded join prevents shutdown from
        hanging indefinitely, but a live thread after that bound is surfaced to
        the caller instead of being logged as if shutdown had succeeded.
        """

        with self._lifecycle_lock:
            if not self._thread_started or not self.thread.is_alive():
                return
            if threading.current_thread() is self.thread:
                self._request_stop()
                raise RuntimeError(
                    "Bootstrap server cannot synchronously close from its own thread"
                )

            self._request_stop()
            logger.info("Stopping bootstrap server loop...")
            self.thread.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
            if self.thread.is_alive():
                raise RuntimeError(
                    "Bootstrap server thread did not stop within "
                    f"{self._SHUTDOWN_TIMEOUT_SECONDS:.1f}s"
                )
            logger.info("Bootstrap server thread stopped")
