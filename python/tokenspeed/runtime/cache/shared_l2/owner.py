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

It runs the scheduler's ``SharedL2Pool`` over every provider's memory and
holds no bytes. The pool is built when every expected provider has joined and
the first client reports its cache geometry; later clients must match that
geometry and the deployment the pool caches for. A provider joins with its
bytes; its share of the pool, in LCM blocks, is settled when the pool is built,
from the first client's LCM block. Every announcement is one provider: one
that is started again counts as one more, or is refused once every expected
provider has joined. A client silent for ``client_timeout_s`` loses its pins
and reservations, and its next request is answered with an error.

Requests (``wire``), by ``op``; every reply also carries ``ok``:

* ``add_provider``: ``session`` (the provider's ``host:port``), ``base_address``,
  ``bytes``.
* ``hello``: ``client``, ``geometry``, ``deployment_id`` -> ``ready``, and once
  ready ``providers``, ``lcm_block_bytes``.
* ``ping``: ``client``.
* ``lookup``: ``client``, ``lookups`` ``[[prefix_hashes, tokens,
  max_tokens]]`` (``tokens`` are the prompt's token ids, not a count) ->
  ``hits`` ``[[ticket, num_tokens, kv_tokens, tail, [[key, bucket, page]]]]``,
  one per lookup. Each hit is pinned under its ticket; ticket 0 pins nothing.
* ``release``: ``client``, ``ticket``.
* ``reserve``: ``client``, ``keys``, ``buckets`` -> ``ticket``, ``pages`` (one
  per key; negative: no page for the key now, -1 the pool holds it, -2 no
  page is free, -3 another reservation writes it).
* ``commit``: ``client``, ``ticket``, ``checkpoints`` ``[[tokens,
  prefix_hashes, tail, access_epoch]]`` (``tokens`` are a checkpoint's token
  ids, not a count).
* ``touch``: ``client``, ``keys``.
* ``clear`` -> ``cleared``; ``stats`` -> ``stats``, ``providers``, ``num_clients``.

A client's geometry is its scheduler's view of the cache: ``prefix_granularity``,
``lcm_block_bytes`` and ``groups``, which holds, per cache group in the scheduler's
order, ``group_id``, ``attn_kind``, ``block_granularity``,
``cache_blocks_per_lcm_block``, ``host_group_packing``, ``shard_count``,
``sliding_window_tokens``, ``sink_tokens``, ``replayable`` and
``checkpoint_granularity``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import msgspec
import zmq

logger = logging.getLogger(__name__)

# How long the serve loop waits for a request before it looks at its stop
# event and the client sweep again.
_POLL_MS = 50


class SharedL2Owner:
    def __init__(
        self,
        *,
        address: str,
        num_providers: int,
        client_timeout_s: float,
        make_shared_l2_pool: Callable[[dict[str, Any], int], Any],
    ):
        """
        Args:
            address: ZMQ bind address.
            num_providers: Providers to wait for before the pool is built.
            client_timeout_s: Silence after which a client is dropped.
            make_shared_l2_pool: ``(geometry, num_lcm_blocks)`` to an object with
                the scheduler's ``SharedL2Pool``'s methods; called once,
                with the first client's geometry.
        """
        self._socket = zmq.Context.instance().socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(address)
        self._num_providers = int(num_providers)
        self._client_timeout_s = float(client_timeout_s)
        self._make_shared_l2_pool = make_shared_l2_pool
        # The providers that joined, with their bytes, and once the pool is
        # built each one's share of it: its LCM blocks and where they start.
        self._announced_providers: list[dict[str, Any]] = []
        self._providers: list[dict[str, Any]] = []
        self._geometry: dict[str, Any] | None = None
        self._deployment_id: str | None = None
        self._pool: Any = None
        # Every client that joined the pool, and when it was last heard.
        self._clients: dict[int, float] = {}

    def serve(self, stop: threading.Event) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        next_sweep = time.monotonic() + self._client_timeout_s / 4
        while not stop.is_set():
            if poller.poll(_POLL_MS):
                # The envelope (identity, a correlated REQ's request id, the
                # empty delimiter) goes back unchanged.
                *envelope, payload = self._socket.recv_multipart()
                try:
                    reply = msgspec.msgpack.encode(
                        self.handle(msgspec.msgpack.decode(payload))
                    )
                except Exception as exc:
                    logger.exception("shared L2 owner request failed")
                    reply = msgspec.msgpack.encode(
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    )
                self._socket.send_multipart([*envelope, reply])
            now = time.monotonic()
            if now >= next_sweep:
                next_sweep = now + self._client_timeout_s / 4
                self._disconnect_timed_out_clients(now)

    def _disconnect_timed_out_clients(self, now: float) -> None:
        for client, heard in list(self._clients.items()):
            if now - heard > self._client_timeout_s:
                logger.warning(f"shared L2 client {client} timed out")
                del self._clients[client]
                self._pool.disconnect(client)

    def close(self) -> None:
        self._socket.close(0)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request["op"]
        if op == "add_provider":
            return self._add_provider(request)
        if op == "hello":
            return self._hello(request)
        if op in ("clear", "stats"):
            if self._pool is None:
                raise RuntimeError("the shared L2 pool is not ready")
            if op == "clear":
                return {"ok": True, "cleared": self._pool.clear_cache()}
            return {
                "ok": True,
                "stats": self._pool.stats(),
                "providers": self._providers,
                "num_clients": len(self._clients),
            }
        client = int(request["client"])
        if client not in self._clients:
            # It timed out here, or this owner started after its hello: what
            # it held is gone, and it must not go on as if it were not.
            return {
                "ok": False,
                "error": f"client {client} has not joined the pool",
            }
        self._clients[client] = time.monotonic()
        pool = self._pool
        if op == "ping":
            return {"ok": True}
        if op == "lookup":
            hits = [
                pool.lookup(client, prefix_hashes, tokens, int(max_tokens))
                for prefix_hashes, tokens, max_tokens in request["lookups"]
            ]
            return {"ok": True, "hits": hits}
        if op == "release":
            pool.release(client, int(request["ticket"]))
            return {"ok": True}
        if op == "reserve":
            ticket, pages = pool.reserve(
                client, [tuple(key) for key in request["keys"]], request["buckets"]
            )
            return {"ok": True, "ticket": ticket, "pages": pages}
        if op == "commit":
            checkpoints = [
                (
                    tokens,
                    prefix_hashes,
                    None if tail is None else tuple(tail),
                    access_epoch,
                )
                for tokens, prefix_hashes, tail, access_epoch in request["checkpoints"]
            ]
            pool.commit(client, int(request["ticket"]), checkpoints)
            return {"ok": True}
        if op == "touch":
            pool.touch([tuple(key) for key in request["keys"]])
            return {"ok": True}
        raise ValueError(f"unknown shared L2 request {op!r}")

    def _add_provider(self, request: dict[str, Any]) -> dict[str, Any]:
        session, size = str(request["session"]), int(request["bytes"])
        if len(self._announced_providers) == self._num_providers:
            raise RuntimeError("every expected provider has already joined")
        self._announced_providers.append(
            {
                "session": session,
                "base_address": int(request["base_address"]),
                "bytes": size,
            }
        )
        logger.info(f"shared L2 provider {session} joined: {size} bytes")
        return {"ok": True}

    def _hello(self, request: dict[str, Any]) -> dict[str, Any]:
        if len(self._announced_providers) < self._num_providers:
            return {"ok": True, "ready": False}
        geometry, deployment_id = request["geometry"], str(request["deployment_id"])
        if self._pool is None:
            lcm_block_bytes = int(geometry["lcm_block_bytes"])
            providers: list[dict[str, Any]] = []
            num_lcm_blocks = 0
            for provider in self._announced_providers:
                provider_lcm_blocks = provider["bytes"] // lcm_block_bytes
                if provider_lcm_blocks <= 0:
                    raise ValueError(
                        f"provider {provider['session']} holds {provider['bytes']} "
                        f"bytes, less than one LCM block of {lcm_block_bytes} bytes"
                    )
                providers.append(
                    {
                        "session": provider["session"],
                        "base_address": provider["base_address"],
                        "num_lcm_blocks": provider_lcm_blocks,
                        "first_lcm_block": num_lcm_blocks,
                    }
                )
                num_lcm_blocks += provider_lcm_blocks
            self._pool = self._make_shared_l2_pool(geometry, num_lcm_blocks)
            self._providers = providers
            self._geometry, self._deployment_id = geometry, deployment_id
            logger.info(
                f"shared L2 pool ready for deployment {deployment_id}: "
                f"{self._pool.stats()}"
            )
        elif geometry != self._geometry:
            raise ValueError("the client's cache geometry differs from the pool's")
        elif deployment_id != self._deployment_id:
            # Same layout, other model, weights or runtime: its KV is not
            # this pool's.
            raise ValueError("the client's deployment differs from the pool's")
        self._clients[int(request["client"])] = time.monotonic()
        return {
            "ok": True,
            "ready": True,
            "providers": self._providers,
            "lcm_block_bytes": int(self._geometry["lcm_block_bytes"]),
        }


def make_shared_l2_pool(
    geometry: dict[str, Any],
    num_lcm_blocks: int,
    *,
    headroom_lcm_blocks: int,
    checkpoint_retention_requests: int,
    checkpoint_retention_leaf_distance: int,
):
    """The pool's cache index, laid out as a client's Host tier is.

    Args:
        geometry: The first client's cache geometry.
        num_lcm_blocks: LCM blocks of every provider together.
        headroom_lcm_blocks: Empty LCM blocks a reservation evicts back up to.
        checkpoint_retention_requests: The scheduler's setting of that name,
            counted over the lookups of every client, hit or miss; 0 ranks
            every entry by its own use.
        checkpoint_retention_leaf_distance: The scheduler's setting of that
            name.
    """
    import tokenspeed_engine.scheduler as ts

    groups = []
    for group in geometry["groups"]:
        config = ts.CacheGroupConfig()
        config.group_id = str(group["group_id"])
        config.attn_kind = ts.AttnKind[group["attn_kind"]]
        config.block_granularity = int(group["block_granularity"])
        config.cache_blocks_per_lcm_block = int(group["cache_blocks_per_lcm_block"])
        config.shard_count = int(group["shard_count"])
        config.sliding_window_tokens = group["sliding_window_tokens"]
        config.sink_tokens = int(group["sink_tokens"])
        config.replayable = bool(group["replayable"])
        config.checkpoint_granularity = int(group["checkpoint_granularity"])
        groups.append(config)
    scheduler_config = ts.SchedulerConfig()
    scheduler_config.prefix_granularity = int(geometry["prefix_granularity"])
    scheduler_config.cache_groups = groups
    scheduler_config.host_group_packing = [
        int(group["host_group_packing"]) for group in geometry["groups"]
    ]
    scheduler_config.checkpoint_retention_requests = checkpoint_retention_requests
    scheduler_config.checkpoint_retention_leaf_distance = (
        checkpoint_retention_leaf_distance
    )
    return ts.SharedL2Pool(scheduler_config, num_lcm_blocks, headroom_lcm_blocks)
