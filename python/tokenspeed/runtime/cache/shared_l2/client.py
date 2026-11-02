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

"""A prefill rank's side of the shared L2 pool, which is its Host L2.

The scheduler keeps a proxy Host tier that mirrors the pool: this client
carries the scheduler's questions and commands to the owner and moves bytes.

* ``lookup`` asks the owner, in one request, for the pool's hit of each
  waiting prompt; the owner pins the hit's pages, which the scheduler mirrors
  into proxy Host entries (``map_proxy_pages`` records the page behind each).
* A load-back from the proxy reads the pool pages by RDMA
  (``SharedL2LoadExecutor``).
* Write reserves pages for newly published Device blocks, copies them and
  commits them with the lineage of the State checkpoints among them.
* Release and Touch are fire-and-forget notices.

Write completions become scheduler events on the event-loop thread.
"""

from __future__ import annotations

import bisect
import json
import logging
import queue
import secrets
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import tokenspeed_scheduler as ts

from tokenspeed.runtime.cache.shared_l2.layout import SharedL2Layout
from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection

logger = logging.getLogger(__name__)

_KEEPALIVE_S = 5.0


@dataclass(frozen=True, slots=True)
class SharedL2ClientConfig:
    """``--shared-l2-config``: every field is required."""

    owner: str
    # Names the model and weights the pool caches; the owner serves one.
    namespace: str
    # RDMA devices for each local GPU index, as Mooncake names them.
    device_names: tuple[str, ...]
    request_timeout_ms: int
    connect_timeout_s: float
    # Threads reading one load's providers in parallel.
    load_workers: int
    write_workers: int
    scheduler: ts.SharedL2Config

    @classmethod
    def from_json(cls, raw: str) -> SharedL2ClientConfig:
        values = json.loads(raw)
        fields = {
            "owner",
            "namespace",
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
        scheduler = ts.SharedL2Config(
            max_writes=int(limits["max_writes"]),
            max_write_keys=int(limits["max_write_keys"]),
        )
        scheduler.validate()
        config = cls(
            owner=str(values["owner"]),
            namespace=str(values["namespace"]),
            device_names=tuple(str(name) for name in values["device_names"]),
            request_timeout_ms=int(values["request_timeout_ms"]),
            connect_timeout_s=float(values["connect_timeout_s"]),
            load_workers=int(values["load_workers"]),
            write_workers=int(values["write_workers"]),
            scheduler=scheduler,
        )
        if min(config.load_workers, config.write_workers) < 1:
            raise ValueError(
                "--shared-l2-config needs at least one load and one write worker"
            )
        return config


class RdmaTransfer:
    """The rank's TransferEngine over its registered Device cache."""

    def __init__(self, *, device_name: str, regions: Sequence[tuple[int, int]]):
        from mooncake.engine import TransferEngine

        from tokenspeed.runtime.utils.network import get_local_ip_by_remote

        hostname = get_local_ip_by_remote()
        if hostname is None:
            raise RuntimeError("shared L2 could not resolve this node's address")
        self._engine = TransferEngine()
        if self._engine.initialize(hostname, "P2PHANDSHAKE", "rdma", device_name) != 0:
            raise RuntimeError("shared L2: TransferEngine initialization failed")
        self._regions = [(int(address), int(size)) for address, size in regions]
        for address, size in self._regions:
            if self._engine.register_memory(address, size) != 0:
                raise RuntimeError("shared L2: Device cache registration failed")

    def read(
        self, session: str, local: list[int], remote: list[int], lengths: list[int]
    ) -> bool:
        return (
            self._engine.batch_transfer_sync_read(session, local, remote, lengths) >= 0
        )

    def write(
        self, session: str, local: list[int], remote: list[int], lengths: list[int]
    ) -> bool:
        return (
            self._engine.batch_transfer_sync_write(session, local, remote, lengths) >= 0
        )

    def close(self) -> None:
        for address, _ in self._regions:
            self._engine.unregister_memory(address)


@dataclass(frozen=True, slots=True)
class _Provider:
    session: str
    base: int
    lcm_blocks: int
    start: int


class SharedL2Client:
    def __init__(
        self,
        config: SharedL2ClientConfig,
        *,
        layout: SharedL2Layout,
        transfer: Any,
        connection: OwnerConnection,
    ):
        self.config = config
        self.layout = layout
        self.transfer = transfer
        self._owner = connection
        self._client = secrets.randbits(63)
        deadline = time.monotonic() + config.connect_timeout_s
        while True:
            reply = self._owner.request(
                "hello",
                client=self._client,
                geometry=layout.geometry(),
                namespace=config.namespace,
            )
            if reply["ready"]:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("the shared L2 pool did not become ready")
            time.sleep(1)
        self._providers = [
            _Provider(
                p["session"], int(p["base"]), int(p["lcm_blocks"]), int(p["start"])
            )
            for p in reply["providers"]
        ]
        self._starts = [provider.start for provider in self._providers]
        self._lcm_bytes = int(reply["lcm_bytes"])
        self.num_lcm_blocks = sum(provider.lcm_blocks for provider in self._providers)
        # (group, proxy Host block) -> the pool page it stands for. Written
        # by the event loop before the plan that loads from it; read by
        # the load executor on the forward thread.
        self._proxy_pages: dict[tuple[int, int], int] = {}
        self._completions: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._failure: Exception | None = None
        self._queues = {kind: queue.Queue() for kind in ("write", "notice")}
        self._threads = []
        workers = {
            "write": (config.write_workers, self._write),
            "notice": (1, self._notice),
        }
        for kind, (count, handler) in workers.items():
            for index in range(count):
                thread = threading.Thread(
                    target=self._work,
                    args=(kind, handler),
                    name=f"shared-l2-{kind}-{index}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        logger.info(
            f"shared L2 client ready: {len(self._providers)} providers, "
            f"{self.num_lcm_blocks} LCM blocks of {self._lcm_bytes} bytes"
        )

    # Scheduler side.

    def lookup(self, lookups: Sequence[tuple]) -> list[tuple]:
        """Ask the owner for the pool's hit of every prompt in one request.

        Args:
            lookups: ``Scheduler.shared_l2_lookups()`` entries, ``(request_id,
                prefix_hashes, tokens, max_tokens)``.

        Returns:
            ``Scheduler.apply_shared_l2_lookups`` entries, one per lookup.
        """
        if not lookups:
            return []
        reply = self._owner.request(
            "probe_many",
            client=self._client,
            prompts=[
                [hashes, tokens, max_tokens]
                for _, hashes, tokens, max_tokens in lookups
            ],
        )
        return [
            (
                lookup[0],
                ticket,
                num_tokens,
                kv_tokens,
                None if tail is None else tuple(tail),
                [tuple(slot) for slot in slots],
            )
            for lookup, (ticket, num_tokens, kv_tokens, tail, slots) in zip(
                lookups, reply["hits"], strict=True
            )
        ]

    def map_proxy_pages(self, pages: Sequence[tuple[int, int, int]]) -> None:
        """Record the pool page behind each new proxy Host block."""
        for group, block, page in pages:
            self._proxy_pages[(group, block)] = page

    def pool_page(self, group: int, block: int) -> int:
        return self._proxy_pages[(group, block)]

    def submit(self, operations) -> None:
        """Snapshot the plan's shared L2 commands and queue them."""
        cache = ts.Cache
        for op in operations:
            if isinstance(op, cache.SharedL2WriteOp):
                self._queues["write"].put(
                    (
                        op.op_id,
                        [tuple(k) for k in op.keys],
                        list(op.blocks),
                        list(op.publications),
                    )
                )
            elif isinstance(op, cache.SharedL2ReleaseOp):
                self._queues["notice"].put(("release", op.ticket))
            elif isinstance(op, cache.SharedL2TouchOp):
                self._queues["notice"].put(("touch", [list(k) for k in op.keys]))
            else:
                raise TypeError(f"not a shared L2 command: {type(op).__name__}")

    def poll_events(self) -> list:
        """Every write completion since the last poll.

        A failed write stops the engine here.
        """
        if self._failure is not None:
            raise RuntimeError("shared L2 operation failed") from self._failure
        events = []
        while True:
            try:
                events.append(
                    ts.Cache.SharedL2WriteDone(self._completions.get_nowait())
                )
            except queue.Empty:
                return events

    def close(self) -> None:
        for kind, work in self._queues.items():
            count = 1 if kind == "notice" else self.config.write_workers
            for _ in range(count):
                work.put(None)
        for thread in self._threads:
            thread.join(timeout=10)
        self.transfer.close()
        self._owner.close()

    # Workers.

    def _work(self, kind: str, handler) -> None:
        work = self._queues[kind]
        while True:
            try:
                item = work.get(timeout=_KEEPALIVE_S if kind == "notice" else None)
            except queue.Empty:
                item = ("ping",)
            if item is None:
                return
            try:
                handler(*item)
            except Exception as exc:
                logger.exception(f"shared L2 {kind} failed")
                if self._failure is None:
                    self._failure = exc

    def _write(self, op_id, keys, blocks, publications) -> None:
        ticket = 0
        if keys:
            reply = self._owner.request(
                "reserve", client=self._client, keys=[list(key) for key in keys]
            )
            ticket, pages = reply["ticket"], reply["pages"]
            self._write_pages(keys, pages, blocks)
        if ticket or publications:
            self._owner.request(
                "commit",
                client=self._client,
                ticket=ticket,
                publications=[[list(t), list(h), tail] for t, h, tail in publications],
            )
        self._completions.put(op_id)

    def _notice(self, kind, payload=None) -> None:
        if kind == "release":
            self._owner.request("release", client=self._client, ticket=payload)
        elif kind == "touch":
            self._owner.request("touch", client=self._client, keys=payload)
        else:
            self._owner.request("ping", client=self._client)

    # Data path.

    def address(self, group: int, page: int) -> tuple[str, int]:
        """The provider session and remote address of a group's pool page."""
        lcm, slot = divmod(page - 1, self.layout.groups[group].packing)
        provider = self._providers[bisect.bisect_right(self._starts, lcm) - 1]
        if not 0 <= lcm - provider.start < provider.lcm_blocks:
            raise ValueError(f"shared L2 page {page} of group {group} is out of range")
        return provider.session, (
            provider.base
            + (lcm - provider.start) * self._lcm_bytes
            + slot * self.layout.slot_stride(group)
        )

    def _write_pages(self, keys, pages, blocks) -> None:
        """Write every key that has a page; a failed transfer raises."""
        by_session: dict[str, tuple[list[int], list[int], list[int]]] = {}
        for (group, _, _), page, block in zip(keys, pages, blocks, strict=True):
            if page < 0:
                continue
            session, address = self.address(group, page)
            local, remote, lengths = by_session.setdefault(session, ([], [], []))
            cursor = 0
            for local_address, size in self.layout.block_segments(group, block):
                local.append(local_address)
                remote.append(address + cursor)
                lengths.append(size)
                cursor += size
        for session, (local, remote, lengths) in by_session.items():
            if not self.transfer.write(session, local, remote, lengths):
                raise RuntimeError(f"shared L2 write to {session} failed")


def create_shared_l2_client(
    config_json: str, *, layout: SharedL2Layout, gpu_id: int
) -> SharedL2Client:
    config = SharedL2ClientConfig.from_json(config_json)
    if not 0 <= gpu_id < len(config.device_names):
        raise ValueError(
            f"--shared-l2-config device_names has no entry for GPU {gpu_id}"
        )
    transfer = RdmaTransfer(
        device_name=config.device_names[gpu_id], regions=layout.registration_regions
    )
    connection = OwnerConnection(config.owner, timeout_ms=config.request_timeout_ms)
    return SharedL2Client(
        config, layout=layout, transfer=transfer, connection=connection
    )
