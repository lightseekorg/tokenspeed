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
from ``lcm_bytes`` or else the first client's LCM block. One provider runs per
node: until the pool is built, a node that announces again has restarted its
provider, and the new one takes the earlier one's place. A client silent for
``client_timeout_s`` loses its pins and reservations, and its next request is
answered with an error.

Requests (``wire``), by ``op``; every reply also carries ``ok``:

* ``provider``: ``session`` (the provider's ``host:port``), ``base``,
  ``bytes``.
* ``hello``: ``client``, ``geometry``, ``deployment_id`` -> ``ready``, and once
  ready ``providers``, ``lcm_bytes``.
* ``ping``: ``client``.
* ``probe_many``: ``client``, ``lookups`` ``[[prefix_hashes, tokens,
  max_tokens]]`` -> ``hits`` ``[[ticket, num_tokens, kv_tokens, tail,
  [[key, bucket, page]]]]``, one per lookup. Each hit is pinned under its
  ticket; ticket 0 pins nothing.
* ``release``: ``client``, ``ticket``.
* ``reserve``: ``client``, ``keys``, ``buckets`` -> ``ticket``, ``pages`` (one
  per key; negative: nothing to write).
* ``commit``: ``client``, ``ticket``, ``publications`` ``[[tokens,
  prefix_hashes, tail, publisher]]`` -> ``committed``.
* ``touch``: ``client``, ``keys``, ``buckets``.
* ``clear`` -> ``cleared``; ``stats`` -> ``stats``, ``providers``, ``clients``.

A client's geometry is its scheduler's view of the cache: ``prefix_granularity``,
``lcm_bytes`` and, per cache group in the scheduler's order, ``group_id``,
``attn_kind``, ``block_granularity``, ``cache_blocks_per_lcm_block``,
``host_packing``, ``shard_count``, ``sliding_window_tokens``, ``sink_tokens``,
``replayable`` and ``checkpoint_granularity``.
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
        lcm_bytes: int | None,
        num_providers: int,
        client_timeout_s: float,
        make_pool: Callable[[dict[str, Any], int], Any],
    ):
        """
        Args:
            address: ZMQ bind address.
            lcm_bytes: Bytes of one pool LCM block, which every client's
                must equal; None takes the first client's.
            num_providers: Providers to wait for before the pool is built.
            client_timeout_s: Silence after which a client is dropped.
            make_pool: ``(geometry, num_lcm_blocks)`` to an object with
                the scheduler's ``SharedL2Pool``'s methods; called once,
                with the first client's geometry.
        """
        self._socket = zmq.Context.instance().socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(address)
        self._lcm_bytes = None if lcm_bytes is None else int(lcm_bytes)
        self._num_providers = int(num_providers)
        self._client_timeout_s = float(client_timeout_s)
        self._make_pool = make_pool
        # The providers that joined, with their bytes, and once the pool is
        # built each one's share of it: its LCM blocks and where they start.
        self._joined: list[dict[str, Any]] = []
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
                self.disconnect_silent(now)

    def disconnect_silent(self, now: float) -> None:
        for client, heard in list(self._clients.items()):
            if now - heard > self._client_timeout_s:
                logger.warning(f"shared L2 client {client} timed out")
                del self._clients[client]
                self._pool.disconnect(client)

    def close(self) -> None:
        self._socket.close(0)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request["op"]
        if op == "provider":
            return self._add_provider(request)
        if op == "hello":
            return self._hello(request)
        if op in ("clear", "stats"):
            if self._pool is None:
                raise RuntimeError("the shared L2 pool is not ready")
            if op == "clear":
                return {"ok": True, "cleared": self._pool.clear()}
            return {
                "ok": True,
                "stats": self._pool.stats(),
                "providers": self._providers,
                "clients": len(self._clients),
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
        if op == "probe_many":
            hits = [
                pool.probe(client, prefix_hashes, tokens, int(max_tokens))
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
            publications = [
                (
                    tokens,
                    prefix_hashes,
                    None if tail is None else tuple(tail),
                    publisher,
                )
                for tokens, prefix_hashes, tail, publisher in request["publications"]
            ]
            committed = pool.commit(client, int(request["ticket"]), publications)
            return {"ok": True, "committed": committed}
        if op == "touch":
            # The pool's touch takes no bucket: it holds one per group so far.
            if any(request["buckets"]):
                raise ValueError("the shared L2 pool holds bucket 0 only")
            pool.touch([tuple(key) for key in request["keys"]])
            return {"ok": True}
        raise ValueError(f"unknown shared L2 request {op!r}")

    def _add_provider(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._pool is not None:
            raise RuntimeError("providers cannot join a running pool")
        session, size = str(request["session"]), int(request["bytes"])
        # One provider runs per node, so a session of a node that has joined
        # is that node's provider started again: the memory the earlier one
        # announced is gone.
        node = session.rsplit(":", 1)[0]
        for earlier in self._joined:
            if earlier["session"].rsplit(":", 1)[0] == node:
                logger.warning(
                    f"shared L2 provider {session} takes the place of "
                    f"{earlier['session']}, which its node announced before"
                )
                self._joined.remove(earlier)
                break
        if len(self._joined) == self._num_providers:
            raise RuntimeError("every expected provider has already joined")
        # Without a block size yet, the pool's creation checks the provider.
        if self._lcm_bytes is not None and size < self._lcm_bytes:
            raise ValueError("a provider must hold at least one LCM block")
        self._joined.append(
            {"session": session, "base": int(request["base"]), "bytes": size}
        )
        logger.info(f"shared L2 provider {session} joined: {size} bytes")
        return {"ok": True}

    def _hello(self, request: dict[str, Any]) -> dict[str, Any]:
        if len(self._joined) < self._num_providers:
            return {"ok": True, "ready": False}
        geometry, deployment_id = request["geometry"], str(request["deployment_id"])
        lcm_bytes = int(geometry["lcm_bytes"])
        if self._lcm_bytes is not None and lcm_bytes != self._lcm_bytes:
            raise ValueError(
                f"the client's LCM block holds {lcm_bytes} bytes, "
                f"a pool slot {self._lcm_bytes}"
            )
        if self._pool is None:
            providers: list[dict[str, Any]] = []
            start = 0
            for provider in self._joined:
                blocks = provider["bytes"] // lcm_bytes
                if blocks <= 0:
                    raise ValueError(
                        f"provider {provider['session']} holds {provider['bytes']} "
                        f"bytes, less than one LCM block of {lcm_bytes} bytes"
                    )
                providers.append(
                    {
                        "session": provider["session"],
                        "base": provider["base"],
                        "lcm_blocks": blocks,
                        "start": start,
                    }
                )
                start += blocks
            self._pool = self._make_pool(geometry, start)
            self._providers, self._lcm_bytes = providers, lcm_bytes
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
            "lcm_bytes": self._lcm_bytes,
        }


def make_pool(
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
    scheduler = ts.SchedulerConfig()
    scheduler.prefix_granularity = int(geometry["prefix_granularity"])
    scheduler.cache_groups = groups
    scheduler.host_group_packing = [
        int(group["host_packing"]) for group in geometry["groups"]
    ]
    scheduler.checkpoint_retention_requests = checkpoint_retention_requests
    scheduler.checkpoint_retention_leaf_distance = checkpoint_retention_leaf_distance
    return ts.SharedL2Pool(scheduler, num_lcm_blocks, headroom_lcm_blocks)
