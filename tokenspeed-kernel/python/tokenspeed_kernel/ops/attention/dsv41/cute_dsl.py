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

"""CuTe DSL entry point for Hopper sparse index scoring.

This is an implementation detail of the registered ``deep_gemm_hopper`` indexer
rather than a separately selectable solution: it replaces the dense score of a
Reindex pass, which reads the whole history only to discard everything outside
the candidate pool.
"""

import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform

__all__ = ["sparse_index_scores", "sparse_index_scores_supported"]

_BLOCKS_PER_TILE = 16
_SUPPORTED = None
_COMPILED = {}


class _GraphSafeDLPack:
    """DLPack view that does not synchronize with the producer stream."""

    def __init__(self, tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _to_cute(tensor, dynamic, align):
    """Wrap a torch tensor for CuTe with the requested specialisation.

    ``dynamic`` is one of:

    * ``"static"`` -- every extent and stride is compiled in (the cache planes,
      whose geometry is fixed for the life of the process).
    * ``"rows"`` -- only the row count is a kernel argument; the remaining
      extents become compile-time tile counts and the strides are compiled in
      (queries, weights, visible: head count and head width are fixed).
    * ``"layout"`` -- every extent and every stride but the unit one is a
      kernel argument. The page table is sliced to the visible history and
      the candidate width follows it, so a prefill chunk sees a new width
      every time; specialising on it would recompile per chunk.

    ``align`` is a promise about the data pointer, so it has to hold for the
    row slices the caller chunks the batch into, not just for whole tensors.
    """
    from cutlass.cute.runtime import from_dlpack

    wrapped = from_dlpack(_GraphSafeDLPack(tensor.detach()), assumed_align=align)
    if dynamic == "rows":
        wrapped = wrapped.mark_compact_shape_dynamic(
            mode=0, stride_order=tuple(range(tensor.dim()))
        )
    elif dynamic == "layout":
        wrapped = wrapped.mark_layout_dynamic(leading_dim=tensor.dim() - 1)
    elif dynamic != "static":
        raise ValueError(f"unknown specialisation {dynamic!r}")
    return wrapped


def _specialised(tensor, dynamic):
    """The extents and strides ``_to_cute`` compiles in for ``dynamic``."""
    if dynamic == "static":
        return (tuple(tensor.shape), tensor.stride())
    if dynamic == "rows":
        return (tuple(tensor.shape[1:]), tensor.stride())
    if dynamic != "layout":
        raise ValueError(f"unknown specialisation {dynamic!r}")
    # A dynamic layout still compiles a zero stride in as the constant 0, and
    # a one-row broadcast view has one with a unit last stride, so the
    # broadcast pattern is a specialisation even though the stride values are not.
    return (tuple(stride == 0 for stride in tensor.stride()),)


def _kernel():
    from tokenspeed_kernel.ops.attention.dsv41._cute_dsl.sparse_index_scores import (
        SparseIndexScoreKernel,
    )

    return SparseIndexScoreKernel


def sparse_index_scores_supported(queries, weights, table, candidates) -> bool:
    """Whether the CuTe DSL sparse scorer can serve this call.

    Args:
        queries: ``[tokens, heads, 128]`` FP8-E4M3 index queries.
        weights: ``[tokens, heads]`` FP32 per-head weights.
        table: ``[tokens, table_width]`` page ids.
        candidates: ``[tokens, blocks]`` int32 candidate block ids.

    Returns:
        True when the platform is Hopper, the shapes tile evenly, the query
        tensors are compact and the table and candidates have unit-stride
        rows. Those two are handed to CuTe with a dynamic layout, so a view
        sliced to fewer columns is served; a view striding along the row is
        not, and the dense scorer, which honours any layout, takes it.
    """
    global _SUPPORTED
    if _SUPPORTED is None:
        platform = current_platform()
        _SUPPORTED = (
            platform.is_nvidia
            and platform.arch_version == ArchVersion(9, 0)
            and _import_ok()
        )
    return (
        _SUPPORTED
        and queries.dtype == torch.float8_e4m3fn
        and queries.shape[-1] == 128
        and queries.shape[1] % 8 == 0
        and candidates is not None
        and candidates.shape[1] % _BLOCKS_PER_TILE == 0
        and queries.is_contiguous()
        and weights.is_contiguous()
        and table.stride(-1) == 1
        and candidates.stride(-1) == 1
    )


def _import_ok() -> bool:
    try:
        _kernel()
    except ImportError:
        return False
    return True


# CTAs to aim for per SM. The kernel is bound by how fast its gather warp can
# issue 1 KiB bulk copies, not by bandwidth, so oversubscribing the machine
# keeps copies in flight. Measured at the production shape (8 queries, 2048
# candidate blocks): 64 CTAs 24.2 us, 128 CTAs 21.0, 512 CTAs 19.4, 1024 CTAs
# 25.4 -- past that the per-CTA setup outweighs the work each one does.
_CTAS_PER_SM = 8


def _split_k(tokens: int, tiles: int) -> int:
    """Fill the machine without splitting a query into more tiles than it has.

    Powers of two only: ``tiles`` is itself a power of two in practice, and a
    split that does not divide it leaves the last wave ragged, which measured
    worse than a smaller even split.
    """
    target = max(1, _CTAS_PER_SM * current_platform().sm_count // max(1, tokens))
    split = 1
    while split * 2 <= min(target, tiles):
        split *= 2
    return split


def sparse_index_scores(
    queries,
    weights,
    values,
    scales,
    table,
    visible,
    candidates,
    enable_pdl,
):
    """Score each query's candidate pool, compacted in candidate order.

    Args:
        queries: ``[tokens, heads, 128]`` FP8-E4M3 index queries; the query
            scale is already folded into ``weights``.
        weights: ``[tokens, heads]`` FP32 per-head weights.
        values: ``[pages, 64, 128]`` FP8-E4M3 index-key value plane.
        scales: ``[pages, 64]`` FP32 index-key scale plane.
        table: ``[tokens, table_width]`` int32 page ids; a negative or
            out-of-range entry scores its rows ``-inf``.
        visible: ``[tokens]`` int32 visible row count per query.
        candidates: ``[tokens, blocks]`` int32 request-local block ids, ``-1``
            padded.
        enable_pdl: Request Programmatic Dependent Launch.

    Returns:
        FP32 ``[tokens, blocks * 8]``. Column ``c`` holds the score of row
        ``candidates[:, c // 8] * 8 + c % 8``, or ``-inf`` when that row is
        null, past ``visible``, or on an unmapped page.
    """
    import cuda.bindings.driver as cuda
    from cutlass import Int32, cute

    # The gather resolves a whole tile of candidates per lane and the MMA reads
    # whole eight-head groups, so a ragged tail would read past either buffer.
    # Checked here rather than left to callers: getting it wrong is silent.
    blocks = candidates.shape[1]
    if blocks % _BLOCKS_PER_TILE or queries.shape[1] % 8:
        raise ValueError(
            f"sparse index scoring needs {_BLOCKS_PER_TILE}-block tiles and "
            f"8-head groups, got {blocks} blocks and {queries.shape[1]} heads"
        )
    if values.shape[0] == 0:
        raise ValueError("sparse index scoring needs at least one cache page")
    tokens = queries.shape[0]
    out = torch.empty((tokens, blocks * 8), dtype=torch.float32, device=queries.device)
    if tokens == 0 or blocks == 0:
        return out
    # The cache planes are views of a page-planar field: their rows are dense
    # but the page stride is the field's, so they stay fully static and both
    # their page count and their strides are part of the compile key.
    # An int32 row is four bytes wide, so a page table of, say, 255 pages puts
    # every chunk after the first on a 1020-byte offset. The index tensors are
    # read one element at a time, so four bytes is the honest promise for them.
    operands = (
        (queries, "rows", 16),
        (weights, "rows", 16),
        (values, "static", 16),
        (scales, "static", 16),
        (table, "layout", 4),
        (visible, "rows", 4),
        (candidates, "layout", 4),
        (out, "layout", 16),
    )
    args = tuple(_to_cute(*operand) for operand in operands)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    heads = queries.shape[1]
    # The split is a launch parameter, not a compile-time constant, so batch
    # sizes do not multiply the compiled variants.
    split = Int32(_split_k(tokens, blocks // _BLOCKS_PER_TILE))
    # cute.compile specialises on element type, extent and stride, so the key
    # is derived from exactly those rather than listed by hand: a hand-written
    # key has already been found short twice, and a missing entry silently
    # hands one caller another's compiled binary. Whatever _to_cute marked
    # dynamic is a kernel argument rather than a specialisation, so it stays out.
    key = (bool(enable_pdl),) + tuple(
        (tensor.dtype,) + _specialised(tensor, dynamic)
        for tensor, dynamic, _ in operands
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _kernel()(heads, bool(enable_pdl)), *args, split, stream
        )
        _COMPILED[key] = compiled
    compiled(*args, split, stream)
    return out
