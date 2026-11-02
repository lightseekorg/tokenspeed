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

"""The pool's owner: one process serving the cache index to every client.

It runs ``tokenspeed_scheduler.SharedL2Pool`` over every provider's memory and
holds no bytes. The pool is built when every expected provider has joined and
the first client reports its cache geometry; later clients must match it and
the namespace (model and weights) it serves. A client silent for
``client_timeout_s`` loses its pins and reservations.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class SharedL2Owner:
    def __init__(
        self,
        *,
        address: str,
        lcm_bytes: int,
        num_providers: int,
        headroom_lcm_blocks: int,
        client_timeout_s: float,
    ):
        import zmq

        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(address)
        self._lcm_bytes = int(lcm_bytes)
        self._num_providers = int(num_providers)
        self._headroom_lcm_blocks = int(headroom_lcm_blocks)
        self._client_timeout_s = float(client_timeout_s)
        self._providers: list[dict[str, Any]] = []
        self._geometry: dict[str, Any] | None = None
        self._namespace: str | None = None
        self._pool: Any = None
        self._last_seen: dict[int, float] = {}

    def serve(self, stop: threading.Event) -> None:
        import msgpack

        poller = self._zmq.Poller()
        poller.register(self._socket, self._zmq.POLLIN)
        next_sweep = time.monotonic() + self._client_timeout_s / 4
        while not stop.is_set():
            if poller.poll(250):
                # The envelope (identity, a correlated REQ's request id, the
                # empty delimiter) goes back unchanged.
                *envelope, payload = self._socket.recv_multipart()
                try:
                    reply = self.handle(msgpack.unpackb(payload, raw=False))
                except Exception as exc:
                    logger.exception("shared L2 owner request failed")
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self._socket.send_multipart(
                    [*envelope, msgpack.packb(reply, use_bin_type=True)]
                )
            now = time.monotonic()
            if now >= next_sweep:
                next_sweep = now + self._client_timeout_s / 4
                self.disconnect_silent(now)

    def disconnect_silent(self, now: float) -> None:
        for client, seen in list(self._last_seen.items()):
            if now - seen > self._client_timeout_s:
                logger.warning(f"shared L2 client {client} timed out")
                del self._last_seen[client]
                if self._pool is not None:
                    self._pool.disconnect(client)

    def close(self) -> None:
        self._socket.close(0)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request["op"]
        client = request.get("client")
        if client is not None:
            self._last_seen[int(client)] = time.monotonic()
        if op == "provider":
            return self._add_provider(request)
        if op == "hello":
            return self._hello(request)
        if op == "ping":
            return {"ok": True}
        if self._pool is None:
            raise RuntimeError("the shared L2 pool is not ready")
        pool = self._pool
        if op == "probe_many":
            # One prompt per waiting request: [prefix_hashes, tokens as int32
            # bytes, max_tokens]; each hit is pinned under its own ticket.
            hits = []
            for prefix_hashes, tokens, max_tokens in request["prompts"]:
                ticket, num_tokens, kv_tokens, tail, slots = pool.probe(
                    int(client), prefix_hashes, tokens, int(max_tokens)
                )
                hits.append(
                    [
                        ticket,
                        num_tokens,
                        kv_tokens,
                        None if tail is None else list(tail),
                        [list(slot) for slot in slots],
                    ]
                )
            return {"ok": True, "hits": hits}
        if op == "release":
            pool.release(int(client), int(request["ticket"]))
            return {"ok": True}
        if op == "reserve":
            ticket, pages = pool.reserve(
                int(client), [tuple(key) for key in request["keys"]]
            )
            return {"ok": True, "ticket": ticket, "pages": pages}
        if op == "commit":
            publications = [
                (tokens, hashes, None if tail is None else tuple(tail))
                for tokens, hashes, tail in request["publications"]
            ]
            pool.commit(int(client), int(request["ticket"]), publications)
            return {"ok": True}
        if op == "touch":
            pool.touch([tuple(key) for key in request["keys"]])
            return {"ok": True}
        if op == "clear":
            return {"ok": True, "cleared": pool.clear()}
        if op == "stats":
            return {"ok": True, "stats": pool.stats(), "providers": self._providers}
        raise ValueError(f"unknown shared L2 request {op!r}")

    def _add_provider(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._pool is not None:
            raise RuntimeError("providers cannot join a running pool")
        if len(self._providers) == self._num_providers:
            raise RuntimeError("every expected provider has already joined")
        blocks = int(request["bytes"]) // self._lcm_bytes
        if blocks <= 0:
            raise ValueError("a provider must hold at least one LCM block")
        start = sum(provider["lcm_blocks"] for provider in self._providers)
        self._providers.append(
            {
                "session": str(request["session"]),
                "base": int(request["base"]),
                "lcm_blocks": blocks,
                "start": start,
            }
        )
        logger.info(
            f"shared L2 provider {request['session']} joined: {blocks} LCM blocks"
        )
        return {"ok": True}

    def _hello(self, request: dict[str, Any]) -> dict[str, Any]:
        if len(self._providers) < self._num_providers:
            return {"ok": True, "ready": False}
        geometry, namespace = request["geometry"], str(request["namespace"])
        if geometry["lcm_bytes"] != self._lcm_bytes:
            raise ValueError(
                f"the client's LCM block holds {geometry['lcm_bytes']} bytes, "
                f"a pool slot {self._lcm_bytes}"
            )
        if self._pool is None:
            self._pool = _make_pool(
                geometry,
                sum(provider["lcm_blocks"] for provider in self._providers),
                self._headroom_lcm_blocks,
            )
            self._geometry, self._namespace = geometry, namespace
            logger.info(f"shared L2 pool ready for {namespace}: {self._pool.stats()}")
        elif geometry != self._geometry or namespace != self._namespace:
            raise ValueError(
                "a client's cache geometry or namespace differs from the pool's"
            )
        return {
            "ok": True,
            "ready": True,
            "providers": self._providers,
            "lcm_bytes": self._lcm_bytes,
        }


def _make_pool(geometry: dict[str, Any], num_lcm_blocks: int, headroom_lcm_blocks: int):
    import tokenspeed_scheduler as ts

    groups = []
    for group in geometry["groups"]:
        config = ts.CacheGroupConfig()
        config.group_id = str(group["group_id"])
        config.block_granularity = int(group["block_granularity"])
        config.cache_blocks_per_lcm_block = int(group["cache_blocks_per_lcm_block"])
        config.total_pages = num_lcm_blocks * config.cache_blocks_per_lcm_block
        config.attn_kind = getattr(ts.AttnKind, group["attn_kind"])
        config.sliding_window_tokens = group["sliding_window_tokens"]
        config.replayable = bool(group["replayable"])
        groups.append(config)
    scheduler = ts.SchedulerConfig()
    scheduler.prefix_granularity = int(geometry["prefix_granularity"])
    scheduler.cache_groups = groups
    return ts.SharedL2Pool(scheduler, num_lcm_blocks, headroom_lcm_blocks)
