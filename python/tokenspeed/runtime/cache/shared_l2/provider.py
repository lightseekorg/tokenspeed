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

"""One node's share of the pool: allocated, registered for RDMA, announced."""

from __future__ import annotations

import ctypes
import logging
import mmap
import threading
import time

from tokenspeed.runtime.cache.shared_l2.wire import OwnerConnection

logger = logging.getLogger(__name__)


def run_provider(
    *,
    owner: str,
    hostname: str,
    device_name: str,
    size_bytes: int,
    stop: threading.Event,
) -> None:
    """Host ``size_bytes`` of pool memory until ``stop``."""
    from mooncake.engine import TransferEngine

    from tokenspeed.runtime.utils.network import join_host_port

    engine = TransferEngine()
    if engine.initialize(hostname, "P2PHANDSHAKE", "rdma", device_name) != 0:
        raise RuntimeError("shared L2 provider: TransferEngine initialization failed")
    memory = mmap.mmap(-1, size_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    memory.madvise(mmap.MADV_HUGEPAGE)
    base = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    started = time.monotonic()
    if engine.register_memory(base, size_bytes) != 0:
        raise RuntimeError("shared L2 provider: memory registration failed")
    session = join_host_port(hostname, engine.get_rpc_port())
    logger.info(
        f"shared L2 provider {session} registered {size_bytes} bytes in "
        f"{time.monotonic() - started:.1f}s"
    )
    connection = OwnerConnection(owner, timeout_ms=60_000)
    connection.request("provider", session=session, base=base, bytes=size_bytes)
    stop.wait()
    connection.close()
    engine.unregister_memory(base)
