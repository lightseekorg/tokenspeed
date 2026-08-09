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

"""POSIX SHM representation and reachability for multimodal feature tensors.

The request path discovers which ranks can open each producer-side SHM segment.
Later encoder transport stages use that process-local reachability information
to select a source without coupling this low-level module to encoder ownership
or device-transfer policy.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
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
    # Populated by ``prepare_shm_features`` after every rank has attempted a
    # local POSIX attach. These are process-local and never serialized.
    _reachable_ranks = None
    _group_size = None

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
        """Open the segment before peers may consume and unlink it."""
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

    def copy_bytes_into(self, destination: torch.Tensor) -> None:
        """Copy this attached segment into a flat CPU uint8 destination."""
        if self._segment is None:
            raise RuntimeError(
                f"ShmTensorHandle({self.shm_name!r}) is not attached on the "
                "selected transport source rank"
            )
        if (
            destination.device.type != "cpu"
            or destination.dtype != torch.uint8
            or destination.numel() != self.nbytes()
        ):
            raise ValueError(
                "SHM byte destination must be a flat CPU uint8 tensor with "
                f"{self.nbytes()} elements"
            )
        source = torch.frombuffer(self._segment.buf, dtype=torch.uint8)
        destination.copy_(source)

    def _set_reachability(
        self, reachable_ranks: tuple[int, ...], group_size: int
    ) -> None:
        """Record process-local SHM reachability after discovery."""
        if (
            not reachable_ranks
            or group_size < len(reachable_ranks)
            or len(set(reachable_ranks)) != len(reachable_ranks)
        ):
            raise ValueError(
                "Invalid SHM reachability: "
                f"reachable={reachable_ranks}, group_size={group_size}"
            )
        self._reachable_ranks = reachable_ranks
        self._group_size = group_size

    @property
    def is_cross_node(self) -> bool:
        reachable_ranks = self._reachable_ranks
        group_size = self._group_size
        return (
            reachable_ranks is not None
            and group_size is not None
            and len(reachable_ranks) != group_size
        )

    @property
    def source_rank(self) -> int | None:
        """Return the selected rank that can directly open this SHM."""
        reachable_ranks = self._reachable_ranks
        return reachable_ranks[0] if reachable_ranks else None

    def is_reachable_from(self, rank: int) -> bool:
        """Return whether ``rank`` can directly open this SHM."""
        reachable_ranks = self._reachable_ranks
        return reachable_ranks is not None and rank in reachable_ranks

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
        # A deferred non-owner on another host never received the payload and
        # cannot unlink the producer's POSIX segment. Avoid a guaranteed failed
        # SharedMemory open for every queued/aliased item on every remote rank.
        if self._reachable_ranks is not None and self._segment is None:
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


def _discover_shm_reachability(handles: Sequence[ShmTensorHandle], group) -> None:
    """Record which group ranks can attach each producer-side segment."""
    group_size = torch.distributed.get_world_size(group)
    attached = [handle.try_attach() for handle in handles]
    local_flags = torch.tensor(attached, dtype=torch.uint8)
    if group_size > 1:
        gathered_flags = [torch.empty_like(local_flags) for _ in range(group_size)]
        # Each rank contributes H bytes instead of all-reducing a P x H matrix.
        torch.distributed.all_gather(gathered_flags, local_flags, group=group)
        flags = torch.stack(gathered_flags)
    else:
        flags = local_flags.unsqueeze(0)

    global_ranks = tuple(
        torch.distributed.get_global_rank(group, group_rank)
        for group_rank in range(group_size)
    )
    for index, handle in enumerate(handles):
        local_group_ranks = flags[:, index].nonzero().flatten().tolist()
        if not local_group_ranks:
            raise RuntimeError(
                f"multimodal shm segment {handle.shm_name!r} is not "
                "reachable from any rank in the group"
            )
        reachable_ranks = tuple(global_ranks[int(rank)] for rank in local_group_ranks)
        handle._set_reachability(reachable_ranks, group_size)


def prepare_shm_features(
    reqs: Sequence[object],
    group,
) -> None:
    """Attach local SHM segments and record their rank reachability.

    Args:
        reqs: Requests that may carry SHM-backed multimodal features.
        group: CPU process group whose ranks will later execute the request.

    Returns:
        None.
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
    _discover_shm_reachability(handles, group)

    cross_node_handles = 0
    cross_node_bytes = 0
    for handle in handles:
        if handle.is_cross_node:
            cross_node_handles += 1
            cross_node_bytes += handle.nbytes()

    if LOG_MM_TIMING and started is not None:
        item_count = sum(len(mm.mm_items) for mm in pending)
        logger.info(
            "mm_timing shm_attach_ms requests=%d items=%d elapsed=%.3f "
            "cross_node_handles=%d cross_node_bytes=%d",
            len(pending),
            item_count,
            (time.perf_counter() - started) * 1000,
            cross_node_handles,
            cross_node_bytes,
        )
