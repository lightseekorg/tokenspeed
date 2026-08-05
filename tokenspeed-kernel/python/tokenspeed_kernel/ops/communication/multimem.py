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

"""NVLS multimem all-reduce over torch symmetric memory.

The in-switch ``ld_reduce`` collective measures ~2x under NCCL at latent-tail
widths on GB200 and stays flat down to tens of tokens, where ring all-reduce
pays its full latency floor. Tensors are staged through cached
symmetric-memory buffers (one per width) so callers keep ordinary allocators.
"""

from __future__ import annotations

import torch

_BUFFERS: dict[tuple[int, int], torch.Tensor] = {}
_SUPPORTED: bool | None = None
_MIN_BUFFER_ROWS = 2048


def multimem_available() -> bool:
    """Whether staged multimem all-reduce can be used on this rank's device."""
    global _SUPPORTED
    if _SUPPORTED is not None:
        return _SUPPORTED
    supported = False
    try:
        import torch.distributed._symmetric_memory  # noqa: F401

        if torch.cuda.is_available() and hasattr(
            torch.ops.symm_mem, "multimem_all_reduce_"
        ):
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory

            supported = bool(
                _SymmetricMemory.has_multicast_support(
                    DeviceType.CUDA, torch.cuda.current_device()
                )
            )
    except Exception:
        supported = False
    _SUPPORTED = supported
    return _SUPPORTED


def _ensure_buffer(
    rows: int, width: int, device: torch.device, group_name: str
) -> torch.Tensor | None:
    """Return a symmetric buffer with >= rows capacity (collective on growth).

    Returns None when capacity is missing during CUDA-graph capture, where
    allocation and rendezvous are impossible.
    """
    import torch.distributed._symmetric_memory as symm_mem

    key = (device.index, width)
    buf = _BUFFERS.get(key)
    if buf is None or buf.shape[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            return None
        cap = max(rows, _MIN_BUFFER_ROWS)
        buf = symm_mem.empty((cap, width), dtype=torch.bfloat16, device=device)
        symm_mem.rendezvous(buf, group_name)
        _BUFFERS[key] = buf
    return buf


def multimem_stage(tensor: torch.Tensor, group_name: str) -> torch.Tensor | None:
    """Copy ``tensor`` into its width's symmetric buffer.

    Args:
        tensor: Contiguous 2-D BF16 CUDA tensor with rank-identical shape.
        group_name: Process-group name covering every rank of the reduction.

    Returns:
        A ``[rows, width]`` view of the symmetric buffer (valid until the next
        stage of the same width), or None when multimem is unavailable — the
        caller falls back to its ordinary reduction path.
    """
    if (
        not multimem_available()
        or tensor.ndim != 2
        or tensor.dtype != torch.bfloat16
        or not tensor.is_cuda
    ):
        return None
    rows, width = tensor.shape
    buf = _ensure_buffer(rows, width, tensor.device, group_name)
    if buf is None:
        return None
    view = buf[:rows]
    view.copy_(tensor)
    return view


def multimem_all_reduce_staged(view: torch.Tensor, group_name: str) -> torch.Tensor:
    """In-place ld_reduce all-reduce of a view returned by ``multimem_stage``."""
    torch.ops.symm_mem.multimem_all_reduce_(view, "sum", group_name)
    return view


__all__ = [
    "multimem_available",
    "multimem_stage",
    "multimem_all_reduce_staged",
]
