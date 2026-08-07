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

"""Coarse-grained, peer-accessible symmetric buffers via HIP IPC.

torch symm_mem on ROCm hands back **fine-grained** memory (HIP VMM,
``hipMemCreate`` with ``hipMemAllocationTypePinned``, no coherence knob), which
bypasses L2 and delivers only ~105 GB/s for bulk local access vs. ~3200 GB/s for
coarse-grained HBM. That is the dominant cost in the ``triton_shmem`` fused
AR+RMSNorm backend: copy-in, copy-out, and the kernel's local reads/writes all
pay the ~30x penalty.

This module provides the rocSHMEM-style alternative used by every high-perf AMD
P2P library (rocSHMEM, MSCCL++, vLLM custom all-reduce): allocate the *data*
buffers as ordinary **coarse-grained** ``torch.empty`` tensors (cached HBM, full
bandwidth) and expose them peer-to-peer through **HIP IPC**
(``hipIpcGetMemHandle`` / ``hipIpcOpenMemHandle``), building the same
``buffer_ptrs_dev``-style uint64 peer-pointer table the embedded Triton kernels
already consume. The signal pad stays a (small) fine-grained symm_mem
allocation, because barrier atomics genuinely need fine-grained coherence.

Coherence: peer writes and reads are ordered by the existing symm_mem
signal-pad barrier (``sem="release"/"acquire"``, ``scope="sys"``), which emits
system-scope cache operations that flush and invalidate coarse L2 state across
the barrier.

Requirements: the torch caching allocator must NOT be in expandable-segments
(VMM) mode, since ``hipIpcGetMemHandle`` only works on ``hipMalloc``-backed
memory. If IPC export fails, callers fall back to the symm_mem path.
"""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.hip.hip_ipc import get_hip_ipc_library


def _export_handle(ptr: int) -> tuple[bytes, int]:
    """Return (64-byte IPC handle for the containing allocation, offset of ptr)."""
    hip = get_hip_ipc_library()
    base, _size = hip.hipMemGetAddressRange(ptr)
    return hip.hipIpcGetMemHandle(base), ptr - base


def _open_base(raw: bytes, opened: dict[bytes, int]) -> int:
    """Open a peer allocation's IPC handle, deduped per process (opening the
    same handle twice is an error), returning its mapped base pointer."""
    if raw in opened:
        return opened[raw]
    out = get_hip_ipc_library().hipIpcOpenMemHandle(raw)
    opened[raw] = out
    return out


@dataclass
class CoarseSymmBuffer:
    """A coarse-grained local tensor plus its peer-pointer table.

    ``tensor`` is ordinary coarse-grained device memory (full HBM bandwidth).
    ``peer_ptrs_dev`` is a ``(world_size,)`` uint64 device tensor of per-peer
    pointers to the same logical buffer (this rank's entry is its own pointer),
    matching the ``buffer_ptrs_dev`` contract the embedded kernels consume.
    """

    tensor: torch.Tensor
    peer_ptrs_dev: torch.Tensor
    _opened_bases: list[int] = field(default_factory=list)

    def close(self) -> None:
        if not self._opened_bases:
            return
        hip = get_hip_ipc_library()
        for base in self._opened_bases:
            hip.hipIpcCloseMemHandle(base)
        self._opened_bases = []


def alloc_coarse_symm(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    *,
    _opened_cache: dict[bytes, int] | None = None,
) -> CoarseSymmBuffer:
    """Allocate a coarse-grained buffer and build its peer-pointer table via IPC.

    ``_opened_cache`` lets a caller share one process-level dedup map across
    several buffers (needed when two buffers may land in the same caching-allocator
    segment, hence share one IPC handle). Raises on IPC failure so callers can
    fall back to the fine-grained symm_mem path.
    """
    world_size = group.size()
    rank = dist.get_rank(group)
    tensor = torch.empty(shape, dtype=dtype, device=device)

    try:
        raw, offset = _export_handle(tensor.data_ptr())
        local_export = {"handle": raw, "offset": offset, "error": None}
    except Exception as exc:  # noqa: BLE001 - synchronize fallback across ranks
        local_export = {
            "handle": None,
            "offset": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    gathered: list = [None] * world_size
    dist.all_gather_object(gathered, local_export, group=group)
    export_errors = [
        f"rank {rank}: {item['error']}"
        for rank, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if export_errors:
        raise RuntimeError(
            "coarse HIP-IPC export failed on one or more ranks: "
            + "; ".join(export_errors)
        )

    opened = _opened_cache if _opened_cache is not None else {}
    opened_keys_before = set(opened)
    peer_ptrs: list[int] = []
    local_open_error = None
    try:
        for peer in range(world_size):
            if peer == rank:
                peer_ptrs.append(tensor.data_ptr())
                continue
            item = gathered[peer]
            peer_ptrs.append(_open_base(item["handle"], opened) + item["offset"])
    except Exception as exc:  # noqa: BLE001 - synchronize fallback across ranks
        local_open_error = f"{type(exc).__name__}: {exc}"

    open_errors: list = [None] * world_size
    dist.all_gather_object(open_errors, local_open_error, group=group)
    failed_opens = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(open_errors)
        if error is not None
    ]
    newly_opened_keys = set(opened) - opened_keys_before
    if failed_opens:
        hip = get_hip_ipc_library()
        for key in newly_opened_keys:
            hip.hipIpcCloseMemHandle(opened.pop(key))
        raise RuntimeError(
            "coarse HIP-IPC open failed on one or more ranks: "
            + "; ".join(failed_opens)
        )

    newly_opened = [opened[key] for key in newly_opened_keys]
    return CoarseSymmBuffer(
        tensor=tensor,
        peer_ptrs_dev=torch.tensor(peer_ptrs, dtype=torch.uint64, device=device),
        _opened_bases=newly_opened,
    )
