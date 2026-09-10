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

"""Off-device contract for gfx1250 MXFP4 index ownership, width and narrowing."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest
from utils import is_amd

if not is_amd():
    pytest.skip(
        "tokenspeed-kernel-amd is installed on AMD CI only",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd._triton import gl  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4 import _common  # noqa: E402

# Read sources from the tree the import resolved to, not the repo layout.
MXFP4_ROOT = Path(_common.__file__).parent

# ---------------------------------------------------------------------------
# Index ownership: which warp holds which row
# ---------------------------------------------------------------------------

INDEX_LAYOUT_CONSUMERS = (
    MXFP4_ROOT / "_common.py",
    MXFP4_ROOT / "decode.py",
    MXFP4_ROOT / "fused.py",
)

# (NUM_INDICES, NUM_WARPS); these kernels run 4 or 8 warps per CTA.
INDEX_SHAPES = ((16, 4), (16, 8), (32, 4), (32, 8), (64, 8))


class IndexOwnership(NamedTuple):
    """Warps over the index dimension, and the rows they cover between them."""

    warps: int
    rows_covered: int


def index_ownership(base: gl.BlockedLayout, slice_dim: int) -> IndexOwnership:
    """Report how ``base`` spreads rows once ``slice_dim`` is sliced away."""
    assert slice_dim in (0, 1), f"index layouts are rank 2, got slice dim {slice_dim}"
    index_dim = 1 - slice_dim
    warps = base.warps_per_cta[index_dim]
    rows_per_warp = base.size_per_thread[index_dim] * base.threads_per_warp[index_dim]
    return IndexOwnership(warps=warps, rows_covered=rows_per_warp * warps)


def index_layout_slice_dim(path: Path) -> int:
    """Return the dimension ``path`` slices away from the shared index layout."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    slice_dims = [
        node.args[0].value
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "SliceLayout"
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "IDX_BASE_LAYOUT"
    ]
    assert len(slice_dims) == 1, f"{path} must slice the index layout exactly once"
    return slice_dims[0]


@pytest.mark.parametrize(("num_indices", "num_warps"), INDEX_SHAPES)
def test_index_layout_partitions_rows_across_all_warps(
    num_indices: int, num_warps: int
) -> None:
    base = _common.get_tdm_gather_scatter_idx_layout(num_indices, num_warps)
    assert index_ownership(base, 0) == IndexOwnership(
        warps=num_warps, rows_covered=num_indices
    )


@pytest.mark.parametrize(("num_indices", "num_warps"), [(16, 0), (16, 32), (12, 8)])
def test_index_layout_rejects_unpartitionable_warp_counts(
    num_indices: int, num_warps: int
) -> None:
    with pytest.raises(AssertionError):
        _common.get_tdm_gather_scatter_idx_layout(num_indices, num_warps)


@pytest.mark.parametrize("path", INDEX_LAYOUT_CONSUMERS, ids=lambda p: p.name)
def test_consumers_slice_the_dimension_that_leaves_rows_distributed(
    path: Path,
) -> None:
    slice_dim = index_layout_slice_dim(path)

    num_indices, num_warps = 16, 8
    base = _common.get_tdm_gather_scatter_idx_layout(num_indices, num_warps)
    assert index_ownership(base, slice_dim) == IndexOwnership(
        warps=num_warps, rows_covered=num_indices
    )
