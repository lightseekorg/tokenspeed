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

"""Pack strided Prefill KV rows into contiguous SGEs before Mooncake WRITE.

Heterogeneous TP partitions heads, so a page is ``rows_per_page`` rows of
``bytes_per_row`` with a larger source pitch. Decode's local layout is often
already contiguous (``dst_stride == bytes_per_row``). Expanding each row into
its own 1KB SGE dominates RDMA submit time; a device-side 2D copy plus one
SGE matches equal-TP descriptor counts.

Destination-strided fragments still expand to per-row SGEs: a contiguous
WRITE would land on the wrong Decode layout without a remote unpack.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PACK_ALIGN = 256

Sge = tuple[int, int, int]


@dataclass(frozen=True)
class PackedCopy:
    """One dest-contiguous, src-strided page fragment to pack before WRITE."""

    src: int
    dst: int
    width: int
    src_pitch: int
    rows: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.src_pitch < self.width or self.rows <= 0:
            raise ValueError(
                f"invalid PackedCopy width={self.width} src_pitch={self.src_pitch} "
                f"rows={self.rows}"
            )

    @property
    def nbytes(self) -> int:
        return self.width * self.rows


def expand_packed_copy(copy: PackedCopy) -> list[Sge]:
    """Row-wise SGEs used when 2D pack is unavailable. Dest is contiguous."""
    return [
        (
            copy.src + row * copy.src_pitch,
            copy.dst + row * copy.width,
            copy.width,
        )
        for row in range(copy.rows)
    ]


def flatten_transfer_blocks(blocks: Iterable[object]) -> Iterator[Sge]:
    """Yield row SGEs for PackedCopy entries; pass 3-tuples through.

    Expansion is lazy so the WRITE path can flush every 4096 descriptors
    even when a packed page becomes many rows.
    """
    for block in blocks:
        if isinstance(block, PackedCopy):
            yield from expand_packed_copy(block)
            continue
        src, dst, length = block
        yield (int(src), int(dst), int(length))


def _aligned(nbytes: int) -> int:
    return (nbytes + _PACK_ALIGN - 1) & ~(_PACK_ALIGN - 1)


def scratch_device(gpu_id: int | None):
    """CUDA device for pack staging. Index-less ``"cuda"`` is device 0 on new threads."""
    import torch

    if gpu_id is None:
        return torch.device("cuda")
    return torch.device("cuda", int(gpu_id))


class PrefillPackScratch:
    """Reusable registered CUDA buffer for dest-contiguous 2D packs."""

    def __init__(self, engine, gpu_id: int | None = None) -> None:
        self.engine = engine
        self._gpu_id = None if gpu_id is None else int(gpu_id)
        self._tensor = None
        self._nbytes = 0
        self._ptr = 0
        self._stream = None

    def materialize(self, blocks: Iterable[object]) -> Iterable[Sge]:
        items = list(blocks)
        if not any(isinstance(item, PackedCopy) for item in items):
            return flatten_transfer_blocks(items)
        try:
            return self._pack_cuda(items)
        except Exception:
            logger.exception("CachePD 2D pack failed; falling back to per-row SGEs")
            return flatten_transfer_blocks(items)

    def _pack_cuda(self, items: list[object]) -> Iterable[Sge]:
        import torch
        from tokenspeed_kernel.platform import current_platform

        if not torch.cuda.is_available() or not current_platform().is_nvidia:
            return flatten_transfer_blocks(items)

        from tokenspeed_kernel.ops.copy.cuda import memcpy_2d_async

        packed_bytes = 0
        for item in items:
            if isinstance(item, PackedCopy):
                packed_bytes += _aligned(item.nbytes)
        self._ensure(packed_bytes)

        stream = self._stream
        assert stream is not None
        sges: list[Sge] = []
        offset = 0
        for item in items:
            if not isinstance(item, PackedCopy):
                src, dst, length = item
                sges.append((int(src), int(dst), int(length)))
                continue
            scratch = self._ptr + offset
            memcpy_2d_async(
                dst=scratch,
                dst_pitch=item.width,
                src=item.src,
                src_pitch=item.src_pitch,
                width=item.width,
                height=item.rows,
                stream_ptr=int(stream.cuda_stream),
            )
            sges.append((scratch, item.dst, item.nbytes))
            offset += _aligned(item.nbytes)
        stream.synchronize()
        return sges

    def _ensure(self, nbytes: int) -> None:
        import torch

        if nbytes <= self._nbytes and self._tensor is not None:
            return
        alloc = max(nbytes, 1)
        device = scratch_device(self._gpu_id)
        tensor = torch.empty(alloc, dtype=torch.uint8, device=device)
        ptr = int(tensor.data_ptr())
        ret = self.engine.register(ptr, alloc)
        if ret:
            raise RuntimeError(
                f"Mooncake scratch registration failed: ptr={ptr} nbytes={alloc} "
                f"ret={ret}"
            )
        old_ptr = self._ptr
        self._tensor = tensor
        self._ptr = ptr
        self._nbytes = alloc
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        if old_ptr:
            self.engine.deregister(old_ptr)
