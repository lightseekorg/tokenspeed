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

The in-switch ``ld_reduce`` collective takes about half NCCL's latency at
latent-tail widths on GB200 and stays flat down to tens of tokens, where ring all-reduce
pays its full latency floor. Tensors are staged through cached
symmetric-memory buffers (one per width) so callers keep ordinary allocators.
"""

from __future__ import annotations

import torch

_BUFFERS: dict[tuple[int, int, str], torch.Tensor] = {}
# Replaced buffers stay referenced: in-flight kernels may still read them.
_RETIRED: list[torch.Tensor] = []
_SUPPORTED: bool | None = None
_AGREED: bool | None = None
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


def multimem_available_all_ranks() -> bool:
    """Collectively-agreed availability (min over ranks); call in lockstep.

    Returns:
        True only when every rank's local probe succeeded, so no rank can
        privately fall back while its peers wait in a multimem barrier.
    """
    global _AGREED
    if _AGREED is None:
        import torch.distributed as dist

        flag = torch.tensor(
            [int(multimem_available())], dtype=torch.int32, device="cuda"
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        _AGREED = bool(flag.item())
    return _AGREED


def _ensure_buffer(
    rows: int,
    width: int,
    device: torch.device,
    group_name: str,
    max_rows: int | None,
) -> torch.Tensor | None:
    """Return a symmetric buffer with >= rows capacity (collective on growth).

    Returns None when capacity is missing during CUDA-graph capture, where
    allocation and rendezvous are impossible.
    """
    import torch.distributed._symmetric_memory as symm_mem

    key = (device.index, width, group_name)
    buf = _BUFFERS.get(key)
    if buf is None or buf.shape[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            return None
        grown = _MIN_BUFFER_ROWS if buf is None else 2 * buf.shape[0]
        if max_rows is not None:
            grown = min(grown, max_rows)
        cap = max(rows, grown)
        if buf is not None:
            _RETIRED.append(buf)
        buf = symm_mem.empty((cap, width), dtype=torch.bfloat16, device=device)
        symm_mem.rendezvous(buf, group_name)
        _BUFFERS[key] = buf
    return buf


def multimem_prealloc(rows: int, widths: tuple[int, ...], group_name: str) -> bool:
    """Allocate and rendezvous the staging buffers up front, collectively.

    Call once at init on every rank in lockstep. With capacity pre-sized to
    the dispatch ceiling, serving-time growth (a collective rendezvous inside
    a forward, where one rank's allocation failure strands its peers) can
    never happen.

    Args:
        rows: Capacity to reserve, normally the caller's dispatch ceiling.
        widths: The tensor widths that will be staged.
        group_name: Process-group name covering every rank.

    Returns:
        True when every buffer is ready.
    """
    if not multimem_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    for width in widths:
        _ensure_buffer(rows, width, device, group_name, rows)
    return True


def multimem_stage(
    tensor: torch.Tensor, group_name: str, max_rows: int | None = None
) -> torch.Tensor | None:
    """Copy ``tensor`` into its width's symmetric buffer.

    Args:
        tensor: 2-D BF16 CUDA tensor, rank-identical shape, width % 8 == 0.
        group_name: Process-group name covering every rank of the reduction.
        max_rows: Caps the speculative doubling only; an explicit request
            larger than this still allocates to fit.

    Returns:
        A ``[rows, width]`` view of the symmetric buffer (valid until the next
        stage of the same width), or None when multimem is unavailable — the
        caller falls back to its ordinary reduction path. Callers must keep
        their capacity history rank-lockstep: growth is a collective.
    """
    if (
        not multimem_available()
        or tensor.ndim != 2
        or tensor.dtype != torch.bfloat16
        or not tensor.is_cuda
        or tensor.shape[1] % 8 != 0
    ):
        return None
    rows, width = tensor.shape
    buf = _ensure_buffer(rows, width, tensor.device, group_name, max_rows)
    if buf is None:
        return None
    view = buf[:rows]
    view.copy_(tensor)
    return view


def multimem_all_reduce_staged(view: torch.Tensor, group_name: str) -> torch.Tensor:
    """In-switch (ld_reduce) sum all-reduce of a staged view, in place.

    Args:
        view: A view previously returned by ``multimem_stage``; every rank
            must call with its own staged view of the same shape, in lockstep.
        group_name: The same group name the view was staged with.

    Returns:
        The reduced view (same storage as the input).
    """
    torch.ops.symm_mem.multimem_all_reduce_(view, "sum", group_name)
    return view


__all__ = [
    "multimem_available",
    "multimem_prealloc",
    "multimem_available_all_ranks",
    "multimem_stage",
    "multimem_all_reduce_staged",
]
