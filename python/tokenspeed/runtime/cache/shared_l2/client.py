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
carries its lookups, writes and notices to the owner. Only ``wait_lookups``
waits for the owner, for as long as its caller allows. An owner that is slow,
wrong or gone costs the rank its pool hits and no request, but for the load
the last point names:

* ``submit_lookups`` sends one owner request for every waiting prompt and
  ``poll_lookups`` returns the answers. A lookup not answered by its
  deadline, or answered with anything but one hit of the wire's shape, on
  pages the pool has, for each lookup, is answered as a miss, and the owner
  is told to abandon it.
* ``submit_write`` reserves pool pages for newly published Device blocks, has
  the transfer copy them, and commits them. A write that fails at any step is
  reported at once, as failed, by ``poll_writes``, so the scheduler releases
  its Device pins, and is abandoned at the owner: at once too, unless its
  copy failed, which keeps the reservation for ``_WRITE_QUARANTINE_S``.
* ``release``, ``abandon_lookup`` and ``touch`` are notices, sent again until
  the owner answers them. A notice the owner refuses, or answers with what is
  no reply of the wire, is dropped with a warning.
* A rank the owner no longer knows, or that has had no answer of the owner
  for ``no_pool_after_s``, serves without the pool: lookups miss at once and
  writes are dropped, until a hello as a new client succeeds. What the earlier
  client held is forgotten.
* The rank holds a lease on what a lookup pinned, which every answer of the
  owner renews, with two bounds. A load is planned from a hit only within
  ``lease_s`` of the owner's last answer (``loads_allowed``); past it the
  scheduler treats the hit as a miss. A load already planned starts up to
  ``load_start_margin_s`` later, also when the rank has left the pool
  meanwhile (``load_pool``). A load that could not start by then fails, which
  stops the engine (``SharedL2LoadExecutor``).
"""

from __future__ import annotations

import collections
import itertools
import json
import logging
import math
import queue
import secrets
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import msgspec

from tokenspeed.runtime.cache.shared_l2.wire import (
    MalformedReplyError,
    OwnerConnection,
    OwnerError,
    UnknownClientError,
)

logger = logging.getLogger(__name__)

# Between hellos while the pool is not there.
_HELLO_RETRY_S = 1.0

# ``load_start_margin_s`` of a config that does not name it.
_LOAD_START_MARGIN_S = 5.0

# How long close waits for each of the client's threads.
_CLOSE_JOIN_S = 10.0

# How long the reservation of a write whose copy failed stays at the owner.
# Mooncake's synchronous write returns on a failure, or at its own timeout,
# without waiting for the RDMA slices it has posted; one that lands later must
# not find its page given to another rank. As long as Mooncake waits for a
# transfer by default (MC_TRANSFER_TIMEOUT).
_WRITE_QUARANTINE_S = 30.0

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
    """``--shared-l2-config``: every field but ``load_start_margin_s`` is
    required."""

    owner: str
    # RDMA devices for each local GPU index, as Mooncake names them.
    device_names: tuple[str, ...]
    # For a hello, a reservation, a commit and the abandon after a failure;
    # at most ``no_pool_after_s``.
    request_timeout_ms: int
    connect_timeout_s: float
    # A lookup the owner has not answered by then is a miss.
    lookup_deadline_ms: int
    # Without an answer from the owner for this long, the rank serves without
    # the pool.
    no_pool_after_s: float
    # Without an answer from the owner for this long, the rank plans no load
    # from the pool.
    lease_s: float
    # A load planned within the lease may still start this much later. The two
    # together stay below the owner's client timeout, at which the owner drops
    # the rank's pins, by more than the longest load takes.
    load_start_margin_s: float
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
            "lookup_deadline_ms",
            "no_pool_after_s",
            "lease_s",
            "load_workers",
            "write_workers",
            "scheduler",
        }
        if (
            not isinstance(values, dict)
            or values.keys() - {"load_start_margin_s"} != fields
        ):
            raise ValueError(
                f"--shared-l2-config needs exactly {sorted(fields)}, and may name "
                "load_start_margin_s"
            )
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
            lookup_deadline_ms=int(values["lookup_deadline_ms"]),
            no_pool_after_s=float(values["no_pool_after_s"]),
            lease_s=float(values["lease_s"]),
            load_start_margin_s=float(
                values.get("load_start_margin_s", _LOAD_START_MARGIN_S)
            ),
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
        if (
            min(
                config.request_timeout_ms,
                config.connect_timeout_s,
                config.lookup_deadline_ms,
                config.no_pool_after_s,
                config.lease_s,
                config.load_start_margin_s,
            )
            <= 0
        ):
            raise ValueError("--shared-l2-config needs positive timeouts")
        if config.request_timeout_ms > config.no_pool_after_s * 1000:
            raise ValueError(
                "--shared-l2-config request_timeout_ms must not exceed "
                "no_pool_after_s"
            )
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
    """The pool as one hello described it: the rank's client id there and
    where the bytes are."""

    client: int
    providers: tuple[SharedL2Provider, ...]
    lcm_bytes: int

    @property
    def num_lcm_blocks(self) -> int:
        return sum(provider.lcm_blocks for provider in self.providers)


def _miss(lookup_id: int) -> tuple:
    return (lookup_id, 0, 0, 0, None, [])


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
            connection: To the owner; closed with the client.

        Raises:
            TimeoutError: The pool did not become ready.
            OwnerError: The owner refused the geometry or the deployment.
            ValueError: ``lease_s`` plus ``load_start_margin_s`` is not below
                the owner's client timeout.
        """
        self.config = config
        self.transfer = transfer
        self._geometry = geometry
        self._deployment_id = deployment_id
        self._owner = connection
        # A ping when the owner has not answered for this long, so two more
        # fit before the lease or the no-pool threshold runs out.
        self._heartbeat_s = min(config.lease_s, config.no_pool_after_s) / 3
        self._retry_s = min(_HELLO_RETRY_S, self._heartbeat_s)
        self._lock = threading.Lock()
        # Wakes the notice thread.
        self._wake = threading.Condition(self._lock)
        # Wakes wait_lookups when lookups stop waiting for the owner.
        self._answered = threading.Condition(self._lock)
        self._closed = False
        self._pool: SharedL2PoolView | None = None
        # The id of the next hello. One id for every hello of an outage: the
        # owner then sees the ones that reach it late as one client.
        self._joining = secrets.randbits(63)
        # When the last request the owner answered was sent.
        self._contact = 0.0
        # Lookup id -> deadline of every lookup not answered yet, in
        # submission order.
        self._pending: dict[int, float] = {}
        self._answers: list[tuple] = []
        # The ticket the scheduler sees -> the owner's, for the pool joined
        # now. An owner that restarts numbers its tickets from the start.
        self._tickets: dict[int, int] = {}
        # A ticket of a pool the rank left -> that pool, and until when a load
        # planned from the ticket may still start.
        self._left_tickets: dict[int, tuple[SharedL2PoolView, float]] = {}
        self._ticket_ids = itertools.count(1)
        # (group, proxy Host block) -> the (ticket, bucket, pool page) it
        # stands for. Written by the event loop before the plan that loads
        # from it; read by the load executor when that plan is submitted.
        self._proxy_pages: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._notices: collections.deque[tuple[str, dict[str, Any]]] = (
            collections.deque()
        )
        # (When its abandon is due, write id) of every write whose copy
        # failed, in that order.
        self._quarantined: collections.deque[tuple[float, int]] = collections.deque()
        self._lookups: queue.SimpleQueue = queue.SimpleQueue()
        self._writes: queue.SimpleQueue = queue.SimpleQueue()
        self._completions: queue.SimpleQueue = queue.SimpleQueue()
        # Lookups answered as a miss because their deadline passed.
        self.lookup_timeouts = 0
        # Blocks of the writes that failed or were dropped without the pool.
        self.dropped_write_blocks = 0
        deadline = time.monotonic() + config.connect_timeout_s
        while not self._hello():
            if time.monotonic() > deadline:
                raise TimeoutError("the shared L2 pool did not become ready")
            time.sleep(self._retry_s)
        self._threads = [
            threading.Thread(
                target=self._work,
                args=(self._lookups, self._look_up),
                name="shared-l2-lookup",
                daemon=True,
            ),
            *(
                threading.Thread(
                    target=self._work,
                    args=(self._writes, self._write),
                    name=f"shared-l2-write-{index}",
                    daemon=True,
                )
                for index in range(config.write_workers)
            ),
            threading.Thread(
                target=self._send_notices, name="shared-l2-notice", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    # Scheduler side: only wait_lookups waits for the owner, at most as long
    # as its caller says.

    @property
    def pool(self) -> SharedL2PoolView | None:
        """The pool joined now; None while the rank serves without one."""
        return self._pool

    @property
    def no_pool(self) -> bool:
        return self._pool is None

    def submit_lookups(self, lookups: Sequence[tuple]) -> None:
        """Ask the owner, in one request, for the pool's hit of every prompt.

        Args:
            lookups: ``Scheduler.shared_l2_lookups()`` entries, ``(lookup_id,
                request_id, prefix_hashes, tokens, max_tokens)``; the tokens
                are int32 bytes.
        """
        if not lookups:
            return
        batch = []
        for lookup_id, _, prefix_hashes, tokens, max_tokens in lookups:
            if not isinstance(tokens, bytes):
                raise TypeError("shared L2 tokens travel as int32 bytes")
            batch.append((lookup_id, prefix_hashes, tokens, max_tokens))
        deadline = time.monotonic() + self.config.lookup_deadline_ms / 1000
        with self._lock:
            pool = self._pool
            if pool is None:
                self._answers.extend(_miss(lookup[0]) for lookup in batch)
                return
            for lookup in batch:
                self._pending[lookup[0]] = deadline
        self._lookups.put((pool, deadline, batch))

    def poll_lookups(self) -> list[tuple]:
        """The answers since the last poll.

        Returns:
            ``Scheduler.apply_shared_l2_lookups`` entries, ``(lookup_id,
            ticket, num_tokens, kv_tokens, tail, slots)``. A lookup past its
            deadline, or made without the pool, is a miss (ticket 0).
        """
        now = time.monotonic()
        with self._lock:
            answers, self._answers = self._answers, []
            # Deadlines grow in submission order: the passed ones are first.
            while self._pending:
                lookup_id, deadline = next(iter(self._pending.items()))
                if deadline > now:
                    break
                del self._pending[lookup_id]
                answers.append(_miss(lookup_id))
                self.lookup_timeouts += 1
        return answers

    def wait_lookups(self, lookup_ids: Sequence[int], timeout_s: float) -> None:
        """Wait until none of these lookups waits for the owner any more, so
        that ``poll_lookups`` has their answers; at most ``timeout_s``."""
        end = time.monotonic() + timeout_s
        with self._answered:
            while True:
                deadlines = [
                    self._pending[lookup_id]
                    for lookup_id in lookup_ids
                    if lookup_id in self._pending
                ]
                if not deadlines:
                    return
                # A lookup past its deadline is answered by the next poll.
                remaining = min(end, *deadlines) - time.monotonic()
                if remaining <= 0:
                    return
                self._answered.wait(remaining)

    def map_proxy_pages(self, ticket: int, pages: Sequence[tuple]) -> None:
        """Record the pool page behind each proxy Host block the scheduler
        made for one answer.

        Args:
            ticket: The answer's ticket.
            pages: ``Scheduler.apply_shared_l2_lookups`` of that answer:
                ``(group, proxy_block, bucket, page)``.
        """
        for group, proxy_block, bucket, page in pages:
            self._proxy_pages[group, proxy_block] = (ticket, bucket, page)

    def proxy_page(self, group: int, proxy_block: int) -> tuple[int, int, int]:
        """The ``(ticket, bucket, page)`` a proxy Host block stands for."""
        return self._proxy_pages[group, proxy_block]

    def loads_allowed(self, ticket: int) -> bool:
        """Whether a load from the pages ``ticket`` pins may be planned now.

        False once the rank left the pool the ticket is from, or the owner has
        not answered for ``lease_s``: the scheduler then treats the hit as a
        miss. A load already planned asks ``load_pool``.
        """
        with self._lock:
            return (
                ticket in self._tickets
                and time.monotonic() - self._contact <= self.config.lease_s
            )

    def load_pool(self, ticket: int) -> SharedL2PoolView | None:
        """The pool a load planned from ``ticket`` reads, while it may start.

        A load starts later than it was planned, behind the forwards in
        flight, so it may start ``load_start_margin_s`` past the lease, also
        from a pool the rank has left meanwhile. Until then the owner that
        granted the ticket has not dropped its pins; an owner that restarted
        has no provider of that pool, so it gives those pages to nobody.

        Returns:
            The pool that granted ``ticket``; None once the owner may have
            dropped the pins, or for a released ticket.
        """
        now = time.monotonic()
        with self._lock:
            if ticket in self._tickets:
                pool = self._pool
                start_by = self._contact + (
                    self.config.lease_s + self.config.load_start_margin_s
                )
            else:
                pool, start_by = self._left_tickets.get(ticket, (None, 0.0))
        return pool if now <= start_by else None

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

    def poll_writes(self) -> list[tuple[int, bool]]:
        """``(write_id, success)`` of every write finished since the last
        poll. A failed write has put nothing in the pool, with one exception:
        a commit the owner carried out and did not answer in time. The pool
        then holds that write's blocks, complete, and the owner answers a
        later offer of them as already there."""
        finished = []
        while True:
            try:
                finished.append(self._completions.get_nowait())
            except queue.Empty:
                return finished

    def release(self, ticket: int) -> None:
        """Unpin what a lookup hit. A ticket of a pool the rank has left is
        gone there, or will be: it is only forgotten here."""
        with self._wake:
            owner_ticket = self._tickets.pop(ticket, None)
            if owner_ticket is not None:
                self._notices.append(("release", {"ticket": owner_ticket}))
                self._wake.notify()
            else:
                self._left_tickets.pop(ticket, None)

    def abandon_lookup(self, lookup_id: int) -> None:
        """Have the owner drop what a lookup the scheduler gave up on pinned,
        or will pin."""
        self._notice("abandon_lookup", lookup_ids=[lookup_id])

    def touch(self, keys: Sequence[tuple], buckets: Sequence[int]) -> None:
        """Tell the owner that the rank used its own copies of these blocks."""
        self._notice("touch", keys=list(keys), buckets=list(buckets))

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify()
        self._lookups.put(None)
        for _ in range(self.config.write_workers):
            self._writes.put(None)
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

    # Owner side.

    def _hello(self) -> bool:
        """Join the pool as a new client. False while the owner is not there
        or still waits for providers."""
        sent = time.monotonic()
        try:
            reply = self._owner.request(
                "hello",
                timeout_ms=self.config.request_timeout_ms,
                client=self._joining,
                geometry=self._geometry,
                deployment_id=self._deployment_id,
            )
        except TimeoutError:
            return False
        if not reply["ready"]:
            return False
        if (
            self.config.lease_s + self.config.load_start_margin_s
            >= reply["client_timeout_s"]
        ):
            raise ValueError(
                f"--shared-l2-config lease_s {self.config.lease_s} plus "
                f"load_start_margin_s {self.config.load_start_margin_s} must be "
                f"below the owner's client timeout {reply['client_timeout_s']}"
            )
        pool = SharedL2PoolView(
            client=self._joining,
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
        with self._lock:
            self._pool, self._contact = pool, sent
        logger.info(
            f"shared L2 client {pool.client} joined: {len(pool.providers)} "
            f"providers, {pool.num_lcm_blocks} LCM blocks of {pool.lcm_bytes} bytes"
        )
        return True

    def _request(
        self,
        pool: SharedL2PoolView,
        op: str,
        timeout_ms: int,
        **fields: Any,
    ) -> dict[str, Any]:
        """One request as ``pool``'s client. Its outcome is what the rank
        knows of the owner: an answer renews the lease; an owner that does not
        know the client, or gives no answer for too long, ends the membership.
        What is no reply of the wire is no answer."""
        sent = time.monotonic()
        refused: OwnerError | None = None
        try:
            reply = self._owner.request(
                op, timeout_ms=timeout_ms, client=pool.client, **fields
            )
        except UnknownClientError:
            # No answer to this client: it renews nothing.
            self._leave_pool(pool, "the owner dropped this client")
            raise
        except (TimeoutError, MalformedReplyError):
            if time.monotonic() - self._contact > self.config.no_pool_after_s:
                self._leave_pool(pool, f"no answer for {self.config.no_pool_after_s}s")
            raise
        except OwnerError as exc:
            # A refusal is an answer too: the owner heard this client.
            refused = exc
        with self._lock:
            if pool is self._pool:
                # The owner heard the client no earlier than the request left.
                self._contact = max(self._contact, sent)
        if refused is not None:
            raise refused
        return reply

    def _leave_pool(self, pool: SharedL2PoolView, reason: str) -> None:
        with self._wake:
            if pool is not self._pool:
                return
            self._pool = None
            self._joining = secrets.randbits(63)
            # The owner drops, or dropped, all this client held: nothing is
            # left to release or to abandon. A load already planned from one
            # of its tickets may still start, until the start bound.
            now = time.monotonic()
            start_by = self._contact + (
                self.config.lease_s + self.config.load_start_margin_s
            )
            self._left_tickets = {
                ticket: left
                for ticket, left in self._left_tickets.items()
                if left[1] >= now
            }
            self._left_tickets.update(
                (ticket, (pool, start_by)) for ticket in self._tickets
            )
            self._tickets.clear()
            self._notices.clear()
            self._quarantined.clear()
            self._answers.extend(_miss(lookup_id) for lookup_id in self._pending)
            self._pending.clear()
            self._answered.notify_all()
            self._wake.notify()
        logger.warning(
            f"shared L2 client {pool.client} serves without the pool: {reason}"
        )

    def _notice(self, op: str, **fields: Any) -> None:
        with self._wake:
            if self._pool is not None:
                self._notices.append((op, fields))
                self._wake.notify()

    def _abandon(self, pool: SharedL2PoolView, op: str, **fields: Any) -> None:
        """Have the owner drop what a lookup or a write the rank gave up on
        holds there.

        The thread that sent the request sends this too, so the owner reads
        the two in that order however late; the notice thread repeats it when
        it is not acknowledged.
        """
        if pool is not self._pool:
            return
        try:
            self._request(pool, op, self.config.request_timeout_ms, **fields)
        except OwnerError as exc:
            logger.warning(f"shared L2 {op} was refused: {exc}")
        except Exception:
            # Not answered: no reply in time, or the request broke.
            self._notice(op, **fields)

    # Threads.

    def _work(self, work: queue.SimpleQueue, handler: Callable[..., None]) -> None:
        while (item := work.get()) is not None:
            try:
                handler(*item)
            except Exception:
                # Nothing the owner or the wire does may stop the rank.
                logger.exception("shared L2 worker failed")

    def _look_up(
        self, pool: SharedL2PoolView, deadline: float, lookups: list[tuple]
    ) -> None:
        if pool is not self._pool:
            # The rank left the pool these lookups were made in, which
            # answered them as misses: not sent, so the owner pins nothing
            # for a client that is gone here.
            return
        # Rounded up: the wait for the answer ends no earlier than the deadline.
        timeout_ms = math.ceil((deadline - time.monotonic()) * 1000)
        if timeout_ms <= 0:
            # Not sent, so there is nothing to abandon; poll_lookups answers.
            return
        hits = None
        refused = False
        try:
            # Checked and converted here, before any lookup is settled: the
            # event loop takes an answer of exactly this shape.
            hits = msgspec.convert(
                self._request(pool, "probe_many", timeout_ms, lookups=lookups)["hits"],
                type=list[_Hit],
            )
            if len(hits) != len(lookups):
                raise OwnerError(f"{len(hits)} hits for {len(lookups)} lookups")
            # A load reads a hit's pages by their numbers: one the pool does
            # not have would fail the load, and so stop the engine.
            groups, blocks = self._geometry["groups"], pool.num_lcm_blocks
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
        except TimeoutError:
            # poll_lookups answers them when the deadline has passed.
            pass
        except Exception as exc:
            # A refusal, or an answer that is not one hit for each lookup, on
            # pages of the pool.
            hits, refused = None, True
            logger.warning(f"shared L2 lookup failed: {exc!r}")
        abandoned = []
        with self._lock:
            for position, (lookup_id, *_) in enumerate(lookups):
                if hits is None:
                    # The owner may have pinned a hit, or may yet.
                    abandoned.append(lookup_id)
                    if refused and self._pending.pop(lookup_id, None) is not None:
                        self._answers.append(_miss(lookup_id))
                    continue
                if self._pending.pop(lookup_id, None) is None:
                    # poll_lookups gave up on it before this answer came, so
                    # the hit it pinned is not used.
                    abandoned.append(lookup_id)
                    continue
                ticket, num_tokens, kv_tokens, tail, slots = hits[position]
                if ticket:
                    local = next(self._ticket_ids)
                    self._tickets[local] = ticket
                    ticket = local
                self._answers.append(
                    (
                        lookup_id,
                        ticket,
                        num_tokens,
                        kv_tokens,
                        tail,
                        slots,
                    )
                )
            self._answered.notify_all()
        if abandoned:
            self._abandon(pool, "abandon_lookup", lookup_ids=abandoned)

    def _write(
        self,
        write_id: int,
        keys: list[tuple],
        buckets: list[int],
        blocks: list[int],
        publications: list[tuple],
    ) -> None:
        pool = self._pool
        success = False
        # Whether the copy started and has not returned.
        copying = False
        try:
            if pool is not None:
                reply = self._request(
                    pool,
                    "reserve",
                    self.config.request_timeout_ms,
                    write_id=write_id,
                    keys=keys,
                    buckets=buckets,
                )
                # Ticket 0: every block is in the pool, is being written by
                # another rank, or has no room.
                if reply["ticket"]:
                    placed = [
                        (key, bucket, page, block)
                        for key, bucket, page, block in zip(
                            keys, buckets, reply["pages"], blocks, strict=True
                        )
                        if page > 0
                    ]
                    copying = True
                    self.transfer.write(pool, placed)
                    copying = False
                    committed = self._request(
                        pool,
                        "commit",
                        self.config.request_timeout_ms,
                        ticket=reply["ticket"],
                        publications=publications,
                    )["committed"]
                    if not committed:
                        raise OwnerError("the owner no longer holds the reservation")
                success = True
        except Exception as exc:
            logger.warning(
                f"shared L2 write {write_id} of {len(keys)} blocks failed: {exc!r}"
            )
        if not success:
            with self._lock:
                self.dropped_write_blocks += len(keys)
        # Reported first: the abandon may wait for the owner.
        self._completions.put((write_id, success))
        if success or pool is None:
            return
        if not copying:
            self._abandon(pool, "abandon_write", write_id=write_id)
            return
        # The copy failed: the owner keeps the reservation, and so the pages,
        # until no slice of the copy can still land.
        with self._wake:
            if pool is self._pool:
                self._quarantined.append(
                    (time.monotonic() + _WRITE_QUARANTINE_S, write_id)
                )
                self._wake.notify()

    def _send_notices(self) -> None:
        """Send every notice until the owner answers it, the abandon of a
        write whose copy failed once its quarantine is over, ping an owner
        the rank has not heard, and rejoin a pool it left."""
        # A notice is repeated and a ping is due at the same pace.
        timeout_ms = max(1, int(self._heartbeat_s * 1000))
        while True:
            with self._wake:
                if self._closed:
                    return
                pool = self._pool
                now = time.monotonic()
                while self._quarantined and self._quarantined[0][0] <= now:
                    _, write_id = self._quarantined.popleft()
                    self._notices.append(("abandon_write", {"write_id": write_id}))
                notice = self._notices[0] if self._notices else None
                if pool is not None and notice is None:
                    idle_s = self._contact + self._heartbeat_s - now
                    if self._quarantined:
                        idle_s = min(idle_s, self._quarantined[0][0] - now)
                    if idle_s > 0:
                        self._wake.wait(idle_s)
                        continue
            if pool is None:
                joined = False
                try:
                    joined = self._hello()
                except Exception as exc:
                    logger.warning(f"shared L2: the pool refused this rank: {exc}")
                if not joined:
                    with self._wake:
                        if not self._closed:
                            self._wake.wait(self._retry_s)
                continue
            op, fields = ("ping", {}) if notice is None else notice
            try:
                self._request(pool, op, timeout_ms, **fields)
            except TimeoutError:
                # Not acknowledged: it stays first and is sent again.
                continue
            except MalformedReplyError as exc:
                # Refused, so dropped. The owner was not heard either: what
                # comes next, a ping or a notice, waits as a retry does.
                logger.warning(f"shared L2 {op} was refused: {exc}")
                with self._wake:
                    if not self._closed:
                        self._wake.wait(self._retry_s)
            except OwnerError as exc:
                logger.warning(f"shared L2 {op} was refused: {exc}")
            except Exception:
                logger.exception(f"shared L2 {op} failed")
                with self._wake:
                    if not self._closed:
                        self._wake.wait(self._retry_s)
                try:
                    # The wire carries msgpack.
                    msgspec.msgpack.encode(fields)
                except Exception:
                    # It can never be sent: dropped.
                    pass
                else:
                    # Not acknowledged either: it stays first too.
                    continue
            with self._lock:
                if self._notices and self._notices[0] is notice:
                    self._notices.popleft()
