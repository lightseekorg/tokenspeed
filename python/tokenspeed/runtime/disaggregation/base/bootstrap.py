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

from aiohttp import web

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class DisaggBootstrapServer:
    """HTTP rendezvous server shared by both disaggregation roles.

    A data-source rank PUTs its (ip, port) keyed by (dp_group, tp_rank_in_dp);
    the peer GETs that back to open a Mooncake session, and uses the sentinel
    ``engine_rank == -1`` query to sync the source's parallel sizes. The routing,
    dp-group sharding, and server lifecycle are role-neutral and live here;
    role-specific parallel-info fields (e.g. the KV path's MLA / kv-page lengths)
    are layered in by subclasses via :meth:`_ingest_put_extra` (record extra PUT
    fields) and :meth:`_extra_parallel_info` (add them to the GET sync response).
    """

    def __init__(self, port: int):
        self.port = port
        self.app = web.Application()
        self.lock = asyncio.Lock()
        self._setup_routes()
        self.world_size = None
        self.dp_size = None
        self.tp_size_per_dp_rank = None
        self.prefill_port_table: dict[int, dict[int, dict[str, str | int]]] = {}

        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.run()

    def run(self):
        self.thread.start()

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

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as exc:
            logger.error("Server error: %s", str(exc))
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")
