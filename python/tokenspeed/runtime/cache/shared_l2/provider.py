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
import importlib.util
import logging
import mmap
import os
import threading
import time

import tokenspeed_engine

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
    # The Mooncake library the engine hosts load. The environment is read
    # directly: a provider runs outside the engine and must stay importable
    # without a device, which the engine's environment registry is not.
    library = os.environ.get("TOKENSPEED_MOONCAKE_LIBRARY")
    if not library:
        spec = importlib.util.find_spec("mooncake")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("a shared L2 provider needs tokenspeed-mooncake")
        library = os.path.join(
            spec.submodule_search_locations[0], "libtokenspeed_transfer_engine.so"
        )
        if not os.path.exists(library):
            raise RuntimeError(
                f"{library} is missing: a shared L2 provider needs a "
                "tokenspeed-mooncake build with the TokenSpeed C entry points"
            )
    # The engine host's transfer engine (``ts-mooncake``).
    engine = tokenspeed_engine.TransferEngine(
        library=library,
        hostname=hostname,
        protocol="rdma",
        device_filter=device_name,
    )
    memory = mmap.mmap(-1, size_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    memory.madvise(mmap.MADV_HUGEPAGE)
    base = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    started = time.monotonic()
    engine.register(base, size_bytes)
    session = engine.session_id
    logger.info(
        f"shared L2 provider {session} registered {size_bytes} bytes in "
        f"{time.monotonic() - started:.1f}s"
    )
    connection = OwnerConnection(owner, timeout_ms=60_000)
    connection.request("provider", session=session, base=base, bytes=size_bytes)
    stop.wait()
    connection.close()
    engine.unregister(base)
