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

"""DSA indexer prefill top-k, the TensorRT-LLM kernel built into tokenspeed-kernel."""

from __future__ import annotations

import functools
from pathlib import Path

import torch


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_indexer_topk_module():
    """Load the pre-compiled indexer_topk shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "indexer_topk" / "indexer_topk.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel indexer_topk library not found at {so_path}. "
            "Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def indexer_topk_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    index_topk: int = 2048,
) -> None:
    """Write each row's top-``index_topk`` logit columns into ``indices``.

    Row ``i`` selects from ``logits[i, row_starts[i]:row_ends[i]]``. The
    indices are relative to ``row_starts[i]`` and unsorted; a row with at most
    ``index_topk`` candidates lists them in order and pads with -1.

    Args:
        logits: FP32 ``[rows, columns]`` CUDA scores.
        row_starts: Contiguous int32 ``[rows]`` first candidate column.
        row_ends: Contiguous int32 ``[rows]`` end (exclusive) of the candidates.
        indices: int32 ``[rows, >= index_topk]`` output. Rows are written
            ``index_topk`` elements apart, so pass a contiguous
            ``[rows, index_topk]`` buffer.
        index_topk: Number of columns selected per row.
    """
    _load_indexer_topk_module().indexer_topk_prefill(
        logits, row_starts, row_ends, indices, int(index_topk)
    )
