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

"""POSIX SHM handle for cross-process multimodal feature tensors.

The lifecycle keeps the unlink race-free for tensor-parallel ranks while still
allowing the model-side multimodal planner to deduplicate requests before any
large payload copy happens:

``publish`` (producer) -> ``attach`` (every rank, before barrier) ->
``consume`` (only encoder misses) or ``release`` (deduplicated aliases).
"""

from __future__ import annotations

import logging
import time
from multiprocessing import shared_memory

import msgspec
import torch

from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)
LOG_MM_TIMING = envs.TOKENSPEED_LOG_MM_TIMING.get()


class ShmTensorHandle(msgspec.Struct, eq=False, dict=True):
    """msgpack/pickle-safe handle to a CPU tensor in a POSIX SHM segment.

    A ``msgspec.Struct`` so the handle rides engine msgpack IPC natively
    (``dtype`` uses the shared torch.dtype enc/dec hooks in io_struct).
    ``dict=True`` allows the non-wire ``_segment`` instance attribute that
    caches this rank's open SHM mapping between ``attach`` and ``consume``.
    """

    shm_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    # Per-process open segment; never serialized (class-level default, the
    # instance attribute is only created by attach()).
    _segment = None
    # Payload received over the CPU group when the producer's POSIX segment
    # lives on another host; also non-wire.
    _remote = None

    @classmethod
    def publish(cls, tensor: torch.Tensor) -> ShmTensorHandle:
        nbytes = tensor.numel() * tensor.element_size()
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            shm_bytes = torch.frombuffer(shm.buf, dtype=torch.uint8)
            shm_bytes.copy_(tensor.contiguous().view(torch.uint8).reshape(-1))
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        name = shm.name
        shm.close()
        return cls(shm_name=name, shape=tuple(tensor.shape), dtype=tensor.dtype)

    def attach(self) -> None:
        """Open the SHM segment on this rank. Must run before the cross-rank
        barrier so unlink in ``consume()`` cannot race another rank's open.
        """
        if self._segment is None:
            self._segment = shared_memory.SharedMemory(name=self.shm_name)

    def try_attach(self) -> bool:
        """Attach if the segment exists on this host; False when the
        producer lives on another node."""
        if self._segment is not None or self._remote is not None:
            return True
        try:
            self._segment = shared_memory.SharedMemory(name=self.shm_name)
            return True
        except FileNotFoundError:
            return False

    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * torch.empty((), dtype=self.dtype).element_size()

    def peek_bytes(self) -> torch.Tensor:
        """Copy the attached segment as flat bytes without consuming it."""
        assert self._segment is not None
        return torch.frombuffer(self._segment.buf, dtype=torch.uint8).clone()

    def set_remote(self, flat_bytes: torch.Tensor) -> None:
        self._remote = flat_bytes

    def consume(self) -> torch.Tensor:
        """Copy into a pinned tensor (so downstream non_blocking H2D is real),
        close this rank's FD, and unlink. ``attach()`` must have run.
        """
        if self._remote is not None:
            flat, self._remote = self._remote, None
            return flat.view(self.dtype).reshape(self.shape)
        started = time.perf_counter() if LOG_MM_TIMING else None
        try:
            dst = self._copy_to_pinned()
        finally:
            self._close_and_unlink()
        if LOG_MM_TIMING and started is not None:
            logger.info(
                "mm_timing shm_consume_ms name=%s elapsed=%.3f shape=%s dtype=%s",
                self.shm_name,
                (time.perf_counter() - started) * 1000,
                list(self.shape),
                self.dtype,
            )
        return dst

    def copy_to_pinned(self) -> torch.Tensor:
        """Copy into pinned memory while retaining this rank's SHM ownership.

        The caller must subsequently call :meth:`release`. This allows an
        asynchronous H2D copy to be enqueued before close/unlink cleanup.
        """
        started = time.perf_counter() if LOG_MM_TIMING else None
        if self._remote is not None:
            dst = self._remote.view(self.dtype).reshape(self.shape)
        else:
            dst = self._copy_to_pinned()
        if LOG_MM_TIMING and started is not None:
            logger.info(
                "mm_timing shm_copy_to_pinned_ms name=%s elapsed=%.3f shape=%s dtype=%s",
                self.shm_name,
                (time.perf_counter() - started) * 1000,
                list(self.shape),
                self.dtype,
            )
        return dst

    def copy_into(self, destination: torch.Tensor) -> None:
        """Synchronously copy into an existing tensor and release the SHM segment."""
        if self._remote is not None:
            source = self._remote.view(self.dtype).reshape(self.shape)
        elif self._segment is not None:
            source = torch.frombuffer(self._segment.buf, dtype=self.dtype).reshape(
                self.shape
            )
        else:
            raise RuntimeError(
                f"ShmTensorHandle({self.shm_name!r}) must be attach()'d "
                "before copying (or has already been released on this rank)"
            )
        started = time.perf_counter() if LOG_MM_TIMING else None
        try:
            if source.dtype != destination.dtype:
                raise ValueError(
                    "SHM source and destination dtypes differ: "
                    f"{source.dtype} != {destination.dtype}"
                )
            if source.shape != destination.shape:
                if source.numel() != destination.numel():
                    raise ValueError(
                        "SHM source and destination element counts differ: "
                        f"{source.numel()} != {destination.numel()}"
                    )
                source = source.reshape(destination.shape)
            destination.copy_(source)
        finally:
            del source
            self._close_and_unlink()
        if LOG_MM_TIMING and started is not None:
            logger.info(
                "mm_timing shm_copy_into_ms name=%s elapsed=%.3f shape=%s dtype=%s",
                self.shm_name,
                (time.perf_counter() - started) * 1000,
                list(self.shape),
                self.dtype,
            )

    def _copy_to_pinned(self) -> torch.Tensor:
        if self._segment is None:
            raise RuntimeError(
                f"ShmTensorHandle({self.shm_name!r}) must be attach()'d "
                "before copying (or has already been released on this rank)"
            )
        dst = torch.empty(self.shape, dtype=self.dtype, pin_memory=True)
        src = torch.frombuffer(self._segment.buf, dtype=self.dtype).reshape(self.shape)
        dst.copy_(src)
        return dst

    def _close_and_unlink(self) -> None:
        if self._remote is not None:
            self._remote = None
            return
        segment = self._segment
        self._segment = None
        try:
            if segment is None:
                segment = shared_memory.SharedMemory(name=self.shm_name)
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                # Another rank already won the unlink race; benign.
                pass
        except FileNotFoundError:
            pass

    def release(self) -> None:
        """Close and unlink a SHM segment without materializing the tensor."""
        started = time.perf_counter() if LOG_MM_TIMING else None
        self._close_and_unlink()
        if LOG_MM_TIMING and started is not None:
            logger.info(
                "mm_timing shm_release_ms name=%s elapsed=%.3f shape=%s dtype=%s",
                self.shm_name,
                (time.perf_counter() - started) * 1000,
                list(self.shape),
                self.dtype,
            )


def sync_shm_features(reqs, group, group_size: int) -> None:
    """Attach SHM-backed features in ``reqs`` on every rank.

    The barrier makes later consume/release unlink race-free in multi-rank
    setups. Actual materialization is intentionally deferred until the
    multimodal encoder planner has deduplicated the batch.
    """
    pending = [
        mm
        for req in reqs
        if (mm := getattr(req, "multimodal_inputs", None)) is not None
        and mm.has_pending_shm_features()
    ]
    if not pending:
        return
    started = time.perf_counter() if LOG_MM_TIMING else None
    handles = [
        item.feature_shm
        for mm in pending
        for item in mm.mm_items
        if item.feature_shm is not None
    ]
    attached = [h.try_attach() for h in handles]
    if group_size > 1:
        # Ship payloads whose POSIX segment lives on another host once over
        # the CPU group; same-host ranks keep the zero-copy shm path.
        flags = torch.zeros((group_size, len(handles)), dtype=torch.uint8)
        flags[torch.distributed.get_rank(group)] = torch.tensor(
            attached, dtype=torch.uint8
        )
        torch.distributed.all_reduce(flags, group=group)
        for i, handle in enumerate(handles):
            owners = flags[:, i].nonzero().flatten()
            if owners.numel() == 0:
                raise RuntimeError(
                    f"multimodal shm segment {handle.shm_name!r} is not "
                    "reachable from any rank in the group"
                )
            if owners.numel() == group_size:
                continue
            src = torch.distributed.get_global_rank(group, int(owners[0]))
            if attached[i]:
                payload = handle.peek_bytes()
            else:
                # Pinned: consume() keeps its non_blocking H2D contract.
                payload = torch.empty(
                    handle.nbytes(), dtype=torch.uint8, pin_memory=True
                )
            torch.distributed.broadcast(payload, src=src, group=group)
            if not attached[i]:
                handle.set_remote(payload)
        torch.distributed.barrier(group)
    if LOG_MM_TIMING and started is not None:
        item_count = sum(len(mm.mm_items) for mm in pending)
        logger.info(
            "mm_timing shm_attach_ms requests=%d items=%d elapsed=%.3f",
            len(pending),
            item_count,
            (time.perf_counter() - started) * 1000,
        )
