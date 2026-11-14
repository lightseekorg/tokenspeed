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

"""Map runtime allocation owners to shared pools and memory-saver lifecycle tags."""

from abc import ABC
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

import torch

_NAMED_REGION_POOLS = {
    "executor_inputs": "persistent_buffers",
    "runtime_states": "persistent_buffers",
    "attention_state": "persistent_buffers",
    "workspace": "persistent_buffers",
}


@cache
def _gpu_memory_saver():
    from torch_memory_saver import TorchMemorySaver

    return TorchMemorySaver()


@cache
def _gpu_buffer_memory_savers():
    from torch_memory_saver import TorchMemorySaver

    # Separate pools retain persistent values when parameters use discard.
    return {"persistent_buffers": TorchMemorySaver()}


_active_buffer_pool: ContextVar[object | None] = ContextVar(
    "tokenspeed_active_buffer_pool", default=None
)


_communication_initializers: ContextVar[list[Callable[[], None]] | None] = ContextVar(
    "tokenspeed_communication_initializers", default=None
)


def initialize_communication(operation: Callable[[], None]) -> None:
    """Allocate persistent communication storage after a managed region ends.

    Deferring avoids nested CUDA MemPools: destroying an inner pool while a
    managed outer pool is active aborts some supported Torch builds.
    """
    pending = _communication_initializers.get()
    if pending is None:
        operation()
    else:
        pending.append(operation)


@contextmanager
def _defer_communication_initializers():
    pending: list[Callable[[], None]] = []
    token = _communication_initializers.set(pending)
    try:
        yield
    finally:
        _communication_initializers.reset(token)
    # A failed load never initializes communication. Callbacks run only after
    # the managed allocator scope has exited.
    for operation in pending:
        operation()


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool, *, device: str) -> "TorchMemorySaverAdapter":
        if not enable:
            return _TorchMemorySaverAdapterNoop()
        if device.split(":", 1)[0] == "npu":
            from tokenspeed.runtime.utils.ascend_memory_saver_adapter import (
                AscendMemorySaverAdapter,
            )

            return AscendMemorySaverAdapter()
        return _TorchMemorySaverAdapterReal()

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def named_region(self, name: str):
        """Allocate an owner's persistent buffers in its named region.

        Logical names may alias the same CPU-backed pool. These buffers keep
        their addresses and contents across the existing ``weights`` lifecycle.

        Args:
            name: The allocation owner's registered region name.

        Raises:
            KeyError: The region name is not registered.
        """
        raise NotImplementedError

    def cuda_graph(self, graph, *, pool, stream, tag: str):
        """Capture context for ``graph``: ``torch.cuda.graph`` whose private
        memory pool, when the saver is enabled, lands in the pausable region
        ``tag`` (discarded on pause; every replay rewrites it before reading).
        """
        raise NotImplementedError

    def pause(self, tag: str | None = None):
        raise NotImplementedError

    def resume(self, tag: str | None = None):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    @contextmanager
    def named_region(self, name: str):
        """Resolve the owner's pool and reuse it across nested allocation scopes."""
        pool = _gpu_buffer_memory_savers()[_NAMED_REGION_POOLS[name]]
        # The native pause/resume API operates on process-wide tags. Restore
        # before weight updates, even when KV is still paused. Back up values:
        # some buffers contain constant tables, pointers and kernel semaphores.
        # A backend can construct or grow the shared WorkspacePool. TMS disallows
        # nested regions; allocations in this case already have the right owner.
        if _active_buffer_pool.get() is pool:
            yield
            return
        token = _active_buffer_pool.set(pool)
        try:
            with pool.region(tag="weights", enable_cpu_backup=True):
                yield
        finally:
            _active_buffer_pool.reset(token)

    def configure_subprocess(self):
        from torch_memory_saver import configure_subprocess

        return configure_subprocess()

    @contextmanager
    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        # tag defaults to "default" in the library; pass it through explicitly so
        # weights vs kv_cache regions can be paused/resumed independently.
        # enable_cpu_backup=True offloads (and restores) contents to CPU on
        # pause (weights); False discards them (kv_cache).
        with _defer_communication_initializers(), _gpu_memory_saver().region(
            tag=tag or "default", enable_cpu_backup=enable_cpu_backup
        ):
            yield

    def cuda_graph(self, graph, *, pool, stream, tag: str):
        # Requires the library's preload hook mode, which configure_subprocess
        # sets up for the scheduler processes.
        return _gpu_memory_saver().cuda_graph(
            cuda_graph=graph,
            pool=pool,
            stream=stream,
            tag=tag,
            enable_cpu_backup=False,
        )

    def pause(self, tag: str | None = None):
        return _gpu_memory_saver().pause(tag=tag)

    def resume(self, tag: str | None = None):
        return _gpu_memory_saver().resume(tag=tag)


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def named_region(self, name: str):
        """Validate the owner name while leaving its allocations unmanaged."""
        if name not in _NAMED_REGION_POOLS:
            raise KeyError(name)
        yield

    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        yield

    def cuda_graph(self, graph, *, pool, stream, tag: str):
        return torch.cuda.graph(graph, pool=pool, stream=stream)

    def pause(self, tag: str | None = None):
        pass

    def resume(self, tag: str | None = None):
        pass
