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

Three-step lifecycle so a cross-rank barrier can sit between FD-open
and FD-unlink: ``publish`` (producer) -> ``attach`` (every rank, before
barrier) -> ``consume`` (every rank, after barrier). Driver methods live on
``MultimodalInputs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import shared_memory

import torch


@dataclass
class ShmTensorHandle:
    """Pickle-safe handle to a CPU tensor in a POSIX SHM segment."""

    shm_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    _segment: shared_memory.SharedMemory | None = field(
        default=None, init=False, repr=False, compare=False
    )

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

    def consume(self) -> torch.Tensor:
        """Copy into a pinned tensor (so downstream non_blocking H2D is real),
        close this rank's FD, and unlink. ``attach()`` must have run.
        """
        if self._segment is None:
            raise RuntimeError(
                f"ShmTensorHandle({self.shm_name!r}) must be attach()'d "
                "before consume() (or has already been consumed on this rank)"
            )
        segment = self._segment
        try:
            dst = torch.empty(self.shape, dtype=self.dtype, pin_memory=True)
            src = torch.frombuffer(segment.buf, dtype=self.dtype).reshape(self.shape)
            dst.copy_(src)
        finally:
            self._segment = None
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                # Another rank already won the unlink race; benign.
                pass
        return dst


def sync_shm_features(reqs, group, group_size: int) -> None:
    """Consumer-side three-phase pipeline for SHM-backed features in
    ``reqs``: attach FDs on this rank, optional cross-rank barrier, then
    consume (memcpy out + close + unlink). No-op when no request carries
    a pending handle.

    The barrier is what makes the open-vs-unlink race-free in multi-rank
    setups; ``reqs`` must be identical across ranks (broadcast output)
    so the predicate is consistent everywhere and the barrier is
    deadlock-free.
    """
    pending = [
        mm
        for req in reqs
        if (mm := getattr(req, "multimodal_inputs", None)) is not None
        and mm.has_pending_shm_features()
    ]
    if not pending:
        return
    for mm in pending:
        mm.attach_shm_features()
    if group_size > 1:
        torch.distributed.barrier(group)
    for mm in pending:
        mm.consume_shm_features()
