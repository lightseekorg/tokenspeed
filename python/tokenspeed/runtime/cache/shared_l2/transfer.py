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

"""The bytes of a shared L2 pool: RDMA between a rank's Device cache and the
providers' memory, in either direction, never through the rank's host memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tokenspeed.runtime.cache.shared_l2.client import SharedL2PoolView
from tokenspeed.runtime.cache.shared_l2.layout import SharedL2Layout


class RdmaEngine:
    """Mooncake's TransferEngine over the rank's registered Device cache."""

    def __init__(self, *, device_name: str, regions: Sequence[tuple[int, int]]):
        """
        Args:
            device_name: The RDMA device of this rank's GPU, as Mooncake names
                it.
            regions: ``(address, bytes)`` of the Device cache buffers.
        """
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
    ) -> None:
        """Copy every ``remote`` range of a provider into its ``local`` one."""
        if self._engine.batch_transfer_sync_read(session, local, remote, lengths) != 0:
            raise RuntimeError(f"shared L2 read from {session} failed")

    def write(
        self, session: str, local: list[int], remote: list[int], lengths: list[int]
    ) -> None:
        """Copy every ``local`` range into its ``remote`` one of a provider."""
        if self._engine.batch_transfer_sync_write(session, local, remote, lengths) != 0:
            raise RuntimeError(f"shared L2 write to {session} failed")

    def close(self) -> None:
        # Once: the rank's writes and loads share the engine.
        for address, _ in self._regions:
            self._engine.unregister_memory(address)
        self._regions = []


class SharedL2Transfer:
    """The copies of a rank's pool writes (``SharedL2Client``'s transfer)."""

    def __init__(self, layout: SharedL2Layout, engine: Any):
        """
        Args:
            layout: Where the rank's Device blocks and the pool's pages are.
            engine: Moves byte ranges: ``write(session, local, remote,
                lengths)`` and ``read`` of the same shape, raising when a copy
                fails, and ``close()``. ``RdmaEngine`` in a server.
        """
        self._layout = layout
        self.engine = engine

    def write(self, pool: SharedL2PoolView, placed: Sequence[tuple]) -> None:
        """Copy the Device block of every ``(key, bucket, page, block)`` to its
        pool page; raises when a copy fails."""
        by_session: dict[str, tuple[list[int], list[int], list[int]]] = {}
        for key, bucket, page, block in placed:
            # A key names its group second.
            group = key[1]
            session, address = self._layout.page_address(pool, group, bucket, page)
            local, remote, lengths = by_session.setdefault(session, ([], [], []))
            for (device_address, stride, size), offset in zip(
                self._layout.device_fields(group),
                self._layout.field_offsets[group],
                strict=True,
            ):
                local.append(device_address + block * stride)
                remote.append(address + offset)
                lengths.append(size)
        for session, (local, remote, lengths) in by_session.items():
            self.engine.write(session, local, remote, lengths)

    def close(self) -> None:
        self.engine.close()
