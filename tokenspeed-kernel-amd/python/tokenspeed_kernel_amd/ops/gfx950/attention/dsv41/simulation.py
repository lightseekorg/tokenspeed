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

"""Final-writeback sampling for synthetic sparse-attention histories."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.attention._simulation import stratified_index


@dataclass(frozen=True)
class IndexSelectionSimulation:
    """Synthetic history contract for the CSA2 selector.

    Every visible history row must be allocated and finite. Candidate blocks
    must be sorted, unique, prefix-packed and contain the newest visible block.
    Only that final block may be partial. These are the benchmark allocator's
    invariants, not a replacement for the normal selector's missing-page rules.
    """

    seed: int
    visible_lens: torch.Tensor
    force_latest: bool


@gluon.jit
def simulated_index(
    original,
    rank,
    count,
    extent,
    query,
    seed,
    candidates,
    cand_stride,
    live,
    HAS_CANDIDATES: gl.constexpr,
    FORCE_LATEST: gl.constexpr,
):
    column = stratified_index(original, rank, count, extent, query, seed)
    if FORCE_LATEST:
        column = gl.where(rank == count - 1, extent - 1, column)
    if HAS_CANDIDATES:
        block = gl.load(
            candidates + query.to(gl.int64) * cand_stride + column // 8,
            mask=live,
            other=-1,
        ).to(gl.int64)
        column = (block * 8 + column % 8).to(gl.int32)
    return column
