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

"""A prefill rank's control side of the shared L2 pool, which is its Host L2.

The scheduler keeps a proxy Host tier that mirrors the pool; this client
carries its lookups, writes and notices to the owner.

* ``lookup`` asks the owner, in one request, for the pool's hit of every
  waiting prompt, and waits for the answer. The owner pins each hit's pages,
  which the scheduler mirrors into proxy Host blocks (``map_proxy_pages``
  records the pool page behind each).
* ``submit`` takes a plan's pool commands, each kind in plan order, to the
  calls below.
* ``submit_write`` queues the write of newly published Device blocks: a write
  thread reserves pool pages for them, has the transfer copy them, and commits
  them. ``poll_writes`` returns the writes that finished.
* ``release`` and ``touch`` are notices, which one thread sends in order; it
  pings the owner when it has had nothing to send for ``_KEEPALIVE_S``.

The host's control thread makes every call. ``lookup`` waits for the owner,
up to the request timeout, with the interpreter lock released; the other calls
hand work to the client's threads or read what they finished.

Nothing is sent twice, and every failure stops the rank's engine: a request
the owner does not answer within ``request_timeout_ms`` or answers with an
error (an owner that does not know the rank answers so), an answer that is not
one hit of the wire's shape, on pages the pool has, for each lookup, a copy
that fails, a commit the owner does not carry out. ``lookup`` raises its
failure; a write's or a notice's is kept, and so is a load's, which the load
executor hands over (``record_failure``). ``check_failure`` raises what is
kept, and so does ``poll_writes``.
"""

from __future__ import annotations

import json
import logging
import queue
import secrets
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import msgspec

from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection, OwnerError

logger = logging.getLogger(__name__)

# Between hellos while the pool is not ready.
_HELLO_RETRY_S = 1.0

# An idle rank pings the owner at this pace: the owner drops a client it has
# not heard for its client timeout.
_KEEPALIVE_S = 5.0

# How long close waits for each of the client's threads.
_CLOSE_JOIN_S = 10.0

# A lookup's hit as the owner answers it: ``(ticket, num_tokens, kv_tokens,
# tail, [(key, bucket, page)])``, the tail ``(tokens, hash)`` and a key
# ``(namespace, group, content hash, page offset)``. The bounds are the widths
# the event loop reads the numbers with.
_U32 = Annotated[int, msgspec.Meta(ge=0, lt=1 << 32)]
_I32 = Annotated[int, msgspec.Meta(ge=-(1 << 31), lt=1 << 31)]
_Hit = tuple[
    Annotated[int, msgspec.Meta(ge=0)],
    _I32,
    _I32,
    tuple[_I32, str] | None,
    list[tuple[tuple[_U32, _U32, str, _I32], _I32, _I32]],
]


@dataclass(frozen=True, slots=True)
class SharedL2ClientConfig:
    """``--shared-l2-config``: every field is required."""

    owner: str
    # RDMA devices for each local GPU index, as Mooncake names them.
    device_names: tuple[str, ...]
    # How long a request waits for the owner's answer. One not answered by
    # then has failed.
    request_timeout_ms: int
    connect_timeout_s: float
    # Threads reading one load's providers in parallel.
    load_workers: int
    write_workers: int
    # The scheduler's limits: writes in flight, and keys in one write.
    max_writes: int
    max_write_keys: int

    @classmethod
    def from_json(cls, raw: str) -> SharedL2ClientConfig:
        values = json.loads(raw)
        fields = {
            "owner",
            "device_names",
            "request_timeout_ms",
            "connect_timeout_s",
            "load_workers",
            "write_workers",
            "scheduler",
        }
        if not isinstance(values, dict) or values.keys() != fields:
            raise ValueError(f"--shared-l2-config needs exactly {sorted(fields)}")
        limits = values["scheduler"]
        limit_fields = {"max_writes", "max_write_keys"}
        if not isinstance(limits, dict) or limits.keys() != limit_fields:
            raise ValueError(
                f"--shared-l2-config scheduler needs exactly {sorted(limit_fields)}"
            )
        device_names = values["device_names"]
        if not isinstance(device_names, list) or not all(
            isinstance(name, str) for name in device_names
        ):
            raise ValueError(
                "--shared-l2-config device_names must be a list of strings"
            )
        config = cls(
            owner=str(values["owner"]),
            device_names=tuple(device_names),
            request_timeout_ms=int(values["request_timeout_ms"]),
            connect_timeout_s=float(values["connect_timeout_s"]),
            load_workers=int(values["load_workers"]),
            write_workers=int(values["write_workers"]),
            max_writes=int(limits["max_writes"]),
            max_write_keys=int(limits["max_write_keys"]),
        )
        if min(config.max_writes, config.max_write_keys) < 1:
            raise ValueError("--shared-l2-config scheduler needs positive write limits")
        if min(config.load_workers, config.write_workers) < 1:
            raise ValueError(
                "--shared-l2-config needs at least one load and one write worker"
            )
        if min(config.request_timeout_ms, config.connect_timeout_s) <= 0:
            raise ValueError("--shared-l2-config needs positive timeouts")
        return config


@dataclass(frozen=True, slots=True)
class SharedL2Provider:
    """One node's share of the pool: ``lcm_blocks`` LCM blocks at ``base`` of
    the RDMA session, the pool's LCM blocks from ``start`` on."""

    session: str
    base: int
    lcm_blocks: int
    start: int


@dataclass(frozen=True, slots=True)
class SharedL2PoolView:
    """The pool as the rank's hello described it: the rank's client id there
    and where the bytes are."""

    client: int
    providers: tuple[SharedL2Provider, ...]
    lcm_bytes: int

    @property
    def num_lcm_blocks(self) -> int:
        return sum(provider.lcm_blocks for provider in self.providers)


class SharedL2Client:
    def __init__(
        self,
        config: SharedL2ClientConfig,
        *,
        geometry: dict[str, Any],
        deployment_id: str,
        transfer: Any,
        connection: OwnerConnection,
    ):
        """Join the pool, waiting up to ``connect_timeout_s`` for it.

        Args:
            config: ``--shared-l2-config``.
            geometry: The rank's cache geometry, as ``owner`` describes it.
            deployment_id: This rank's ``storage_deployment_id``; the owner
                serves one.
            transfer: Moves the bytes. ``write(pool, placed)`` copies the
                Device block of every ``(key, bucket, page, block)`` to its
                page of ``pool`` and raises when it cannot; ``close()``.
            connection: To the owner, with the config's request timeout;
                closed with the client.

        Raises:
            TimeoutError: The pool did not become ready.
            OwnerError: The owner refused the geometry or the deployment.
        """
        self.config = config
        self.transfer = transfer
        self._geometry = geometry
        self._owner = connection
        client = secrets.randbits(63)
        deadline = time.monotonic() + config.connect_timeout_s
        while True:
            try:
                reply = connection.request(
                    "hello",
                    client=client,
                    geometry=geometry,
                    deployment_id=deployment_id,
                )
            except TimeoutError:
                # The owner is not there yet.
                reply = {"ready": False}
            # Not ready while the owner still waits for providers.
            if reply["ready"]:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("the shared L2 pool did not become ready")
            time.sleep(_HELLO_RETRY_S)
        self.pool = SharedL2PoolView(
            client=client,
            providers=tuple(
                SharedL2Provider(
                    session=provider["session"],
                    base=int(provider["base"]),
                    lcm_blocks=int(provider["lcm_blocks"]),
                    start=int(provider["start"]),
                )
                for provider in reply["providers"]
            ),
            lcm_bytes=int(reply["lcm_bytes"]),
        )
        logger.info(
            f"shared L2 client {client} joined: {len(self.pool.providers)} "
            f"providers, {self.pool.num_lcm_blocks} LCM blocks of "
            f"{self.pool.lcm_bytes} bytes"
        )
        # (group, proxy Host block) -> the (bucket, pool page) it stands for.
        # Written by the event loop before the plan that loads from it; read
        # by the load executor when that plan is submitted.
        self._proxy_pages: dict[tuple[int, int], tuple[int, int]] = {}
        self._writes: queue.SimpleQueue = queue.SimpleQueue()
        self._notices: queue.SimpleQueue = queue.SimpleQueue()
        self._completions: queue.SimpleQueue = queue.SimpleQueue()
        # The first failure of a write, a notice or a load; check_failure
        # raises it.
        self._failure: BaseException | None = None
        self._threads = [
            *(
                threading.Thread(
                    target=self._work,
                    args=(self._writes, self._write, None),
                    name=f"shared-l2-write-{index}",
                    daemon=True,
                )
                for index in range(config.write_workers)
            ),
            threading.Thread(
                target=self._work,
                args=(self._notices, self._notice, ("ping", {})),
                name="shared-l2-notice",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    # Scheduler side.

    def lookup(self, lookups: Sequence[tuple]) -> list[tuple]:
        """Ask the owner, in one request, for the pool's hit of every prompt,
        and wait for the answer.

        Args:
            lookups: ``Scheduler.take_shared_l2_lookups()`` entries,
                ``(request_id, prefix_hashes, tokens, max_tokens)``; the
                tokens are int32 bytes.

        Returns:
            ``Scheduler.apply_shared_l2_lookups`` entries, one per lookup in
            their order: ``(request_id, ticket, num_tokens, kv_tokens, tail,
            slots)``. A prompt the pool holds no prefix of is a miss: ticket
            0 and no slot.

        Raises:
            TimeoutError: The owner did not answer within
                ``request_timeout_ms``.
            OwnerError: The owner answered with an error, or with hits that
                are not one for each lookup on pages the pool has.
            msgspec.ValidationError: A hit is not of the wire's shape.
        """
        if not lookups:
            return []
        prompts = []
        for _, prefix_hashes, tokens, max_tokens in lookups:
            if not isinstance(tokens, bytes):
                raise TypeError("shared L2 tokens travel as int32 bytes")
            prompts.append((prefix_hashes, tokens, max_tokens))
        reply = self._owner.request(
            "probe_many", client=self.pool.client, lookups=prompts
        )
        # Checked and converted here: the event loop takes an answer of
        # exactly this shape.
        hits = msgspec.convert(reply["hits"], type=list[_Hit])
        if len(hits) != len(lookups):
            raise OwnerError(f"{len(hits)} hits for {len(lookups)} lookups")
        # A load reads a hit's pages by their numbers.
        groups, blocks = self._geometry["groups"], self.pool.num_lcm_blocks
        for *_, slots in hits:
            for (_, group, _, _), _, page in slots:
                if not (
                    group < len(groups)
                    and 0 < page <= blocks * groups[group]["host_packing"]
                ):
                    raise OwnerError(
                        f"a hit names page {page} of group {group}, which "
                        "the pool does not have"
                    )
        return [(lookup[0], *hit) for lookup, hit in zip(lookups, hits)]

    def map_proxy_pages(self, pages: Sequence[tuple]) -> None:
        """Record the pool page behind each proxy Host block the scheduler
        made for one answer.

        Args:
            pages: ``Scheduler.apply_shared_l2_lookups`` of that answer:
                ``(group, proxy_block, bucket, page)``.
        """
        for group, proxy_block, bucket, page in pages:
            self._proxy_pages[group, proxy_block] = (bucket, page)

    def proxy_page(self, group: int, proxy_block: int) -> tuple[int, int]:
        """The ``(bucket, page)`` a proxy Host block stands for."""
        return self._proxy_pages[group, proxy_block]

    def submit(
        self,
        writes: Sequence[tuple],
        releases: Sequence[int],
        touches: Sequence[Sequence[tuple]],
    ) -> None:
        """Queue a plan's pool commands, each kind in plan order.

        Args:
            writes: Each write as ``submit_write`` takes it: ``(write_id,
                keys, buckets, blocks, publications)``.
            releases: The tickets to release.
            touches: The keys of each touch.
        """
        for write in writes:
            self.submit_write(*write)
        for ticket in releases:
            self.release(ticket)
        for keys in touches:
            # The command carries no bucket: an unsharded group has one.
            self.touch(keys, [0] * len(keys))

    def submit_write(
        self,
        write_id: int,
        keys: Sequence[tuple],
        buckets: Sequence[int],
        blocks: Sequence[int],
        publications: Sequence[tuple],
    ) -> None:
        """Queue a ``SharedL2WriteOp``: ``blocks`` are the Device blocks to
        copy for ``keys``, ``publications`` the checkpoints among them,
        ``(tokens, prefix_hashes, tail, publisher)`` with int32-byte tokens."""
        for publication in publications:
            if not isinstance(publication[0], bytes):
                raise TypeError("shared L2 tokens travel as int32 bytes")
        self._writes.put(
            (write_id, list(keys), list(buckets), list(blocks), list(publications))
        )

    def record_failure(self, failure: BaseException) -> None:
        """Keep the first failure of the rank's pool work: a write's or a
        notice's, from the client's threads, or a load's, from the load
        executor. The rank's engine is to stop on it."""
        if self._failure is None:
            self._failure = failure

    def check_failure(self) -> None:
        """Raise once a write, a notice or a load has failed.

        The event loop calls it when a forward has landed, before it commits
        the forward's results: a load that failed let its forward go over
        pages it never filled.

        Raises:
            RuntimeError: At this call and at every later one.
        """
        if self._failure is not None:
            raise RuntimeError("shared L2 operation failed") from self._failure

    def poll_writes(self) -> list[int]:
        """The ids of the writes finished since the last poll.

        Raises:
            RuntimeError: A write, a notice or a load failed
                (``check_failure``).
        """
        self.check_failure()
        finished = []
        while True:
            try:
                finished.append(self._completions.get_nowait())
            except queue.Empty:
                return finished

    def release(self, ticket: int) -> None:
        """Unpin what a lookup hit."""
        self._notices.put(("release", {"ticket": ticket}))

    def touch(self, keys: Sequence[tuple], buckets: Sequence[int]) -> None:
        """Tell the owner that the rank used its own copies of these blocks."""
        self._notices.put(("touch", {"keys": list(keys), "buckets": list(buckets)}))

    def close(self) -> None:
        for _ in range(self.config.write_workers):
            self._writes.put(None)
        self._notices.put(None)
        for thread in self._threads:
            thread.join(timeout=_CLOSE_JOIN_S)
        running = [thread.name for thread in self._threads if thread.is_alive()]
        if running:
            # A thread that still waits for the owner or for a copy uses its
            # socket and the registered memory: both stay as they are, for
            # the process's exit to take.
            logger.warning(
                f"shared L2 client: {', '.join(running)} did not stop in "
                f"{_CLOSE_JOIN_S}s; the transfer and the owner's sockets stay open"
            )
            return
        self.transfer.close()
        self._owner.close()

    # Threads.

    def _work(
        self,
        work: queue.SimpleQueue,
        handler: Callable[..., None],
        idle: tuple | None,
    ) -> None:
        """Run ``handler`` on every item of ``work`` until its None; with
        ``idle``, also on that item whenever none came for ``_KEEPALIVE_S``."""
        while True:
            try:
                item = work.get(timeout=None if idle is None else _KEEPALIVE_S)
            except queue.Empty:
                item = idle
            if item is None:
                return
            try:
                handler(*item)
            except Exception as exc:
                logger.exception("shared L2 operation failed")
                self.record_failure(exc)

    def _write(
        self,
        write_id: int,
        keys: list[tuple],
        buckets: list[int],
        blocks: list[int],
        publications: list[tuple],
    ) -> None:
        reply = self._owner.request(
            "reserve", client=self.pool.client, keys=keys, buckets=buckets
        )
        # Ticket 0: every block is in the pool, is being written by another
        # rank, or has no room.
        if reply["ticket"]:
            # A key without a page (negative) is one of those: skipped.
            placed = [
                (key, bucket, page, block)
                for key, bucket, page, block in zip(
                    keys, buckets, reply["pages"], blocks, strict=True
                )
                if page > 0
            ]
            self.transfer.write(self.pool, placed)
            committed = self._owner.request(
                "commit",
                client=self.pool.client,
                ticket=reply["ticket"],
                publications=publications,
            )["committed"]
            if not committed:
                raise OwnerError("the owner no longer holds the reservation")
        self._completions.put(write_id)

    def _notice(self, op: str, fields: dict[str, Any]) -> None:
        self._owner.request(op, client=self.pool.client, **fields)
