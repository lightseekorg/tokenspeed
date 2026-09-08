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
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4 import _common, fused  # noqa: E402

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


# ---------------------------------------------------------------------------
# Index width: how many row indices one TDM instruction carries
# ---------------------------------------------------------------------------


# A gather's largest index is source rows minus one, so 65536 rows still fit.
@pytest.mark.parametrize(
    ("gather_input_rows", "expected_bits"),
    [(1, 16), (65_536, 16), (65_537, 32)],
)
def test_gather_narrows_up_to_the_last_representable_source_row(
    gather_input_rows: int, expected_bits: int
) -> None:
    bits = fused.select_tdm_index_width_bits(
        gather_input_rows=gather_input_rows, scatter_writeback_rows=None
    )
    assert bits == expected_bits


# Masked-off scatter lanes store the row count itself as an out-of-bounds
# sentinel, so the count must fit, not the count minus one.
@pytest.mark.parametrize(
    ("scatter_writeback_rows", "expected_bits"),
    [(1, 16), (65_535, 16), (65_536, 32)],
)
def test_scatter_reserves_room_for_the_masked_off_sentinel(
    scatter_writeback_rows: int, expected_bits: int
) -> None:
    bits = fused.select_tdm_index_width_bits(
        gather_input_rows=None, scatter_writeback_rows=scatter_writeback_rows
    )
    assert bits == expected_bits


@pytest.mark.parametrize(
    ("gather_input_rows", "scatter_writeback_rows", "expected_bits"),
    [
        (65_536, 65_535, 16),
        (65_537, 65_535, 32),
        (65_536, 65_536, 32),
        (65_537, 65_536, 32),
        # No index is emitted at all, so keep the conservative width.
        (None, None, 32),
    ],
)
def test_either_direction_alone_can_force_the_wide_index(
    gather_input_rows: int | None,
    scatter_writeback_rows: int | None,
    expected_bits: int,
) -> None:
    bits = fused.select_tdm_index_width_bits(
        gather_input_rows=gather_input_rows,
        scatter_writeback_rows=scatter_writeback_rows,
    )
    assert bits == expected_bits


# ---------------------------------------------------------------------------
# Index narrowing: which values narrow at a 32-bit operand
# ---------------------------------------------------------------------------
#
# The kernels address expert weight slabs that can exceed the signed 32-bit
# range, so their pointer arithmetic stays wide. Three values are nonetheless
# handed to instructions whose operand is 32-bit: the TDM descriptor row
# extent, the TDM descriptor column offset and the buffer_store element offset.
# Each narrows at the instruction boundary and nowhere earlier. Those are
# compile-time properties of kernels that need a gfx1250 device to run, so they
# are pinned structurally, and only to the shape of the invariant: which value
# a narrowing cast wraps, and which arithmetic is left symbolic. Where a name
# matters it is recovered from the kernel, so a rename cannot break the test.


DECODE_SOURCE = MXFP4_ROOT / "decode.py"
FUSED_SOURCE = MXFP4_ROOT / "fused.py"
KERNEL_SOURCES = (DECODE_SOURCE, FUSED_SOURCE)

# Concrete integer widths, as opposed to a symbolic type name that a build
# resolves at launch.
FIXED_WIDTHS = frozenset({"int8", "int16", "int32", "int64"})


def calls_named(scope: ast.AST, attr: str) -> list[ast.Call]:
    """Return every ``<obj>.attr(...)`` call reachable from ``scope``."""
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def cast_target_id(node: ast.AST) -> str | None:
    """Return the trailing identifier of a ``.to(<target>)`` cast, if any.

    ``.to(gl.int32)``, ``.to(cfg.index_type)`` and ``.to(address_index_type)``
    all reduce to their last name component, so callers do not care how a type
    is spelled or which object it is reached through.
    """
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "to":
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    target = node.args[0]
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return None


def cast_width(node: ast.AST) -> str | None:
    """Return the width name if ``node`` casts to a fixed width, else ``None``."""
    target_id = cast_target_id(node)
    return target_id if target_id in FIXED_WIDTHS else None


def last_binding(scope: ast.AST, name: str) -> ast.expr:
    """Return the value last bound to ``name`` inside ``scope``."""
    bindings = [
        node.value
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    assert bindings, f"{name} is never assigned"
    return bindings[-1]


def matmul_kernel(path: Path) -> ast.FunctionDef:
    """Return the matmul kernel in ``path``, identified by the ops it issues."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    kernels = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and calls_named(node, "buffer_store")
        and calls_named(node, "update_tensor_descriptor")
    ]
    assert len(kernels) == 1, f"expected one matmul kernel in {path}"
    return kernels[0]


def store_offset_expr(kernel: ast.FunctionDef) -> ast.expr:
    """Return the expression bound to the ``buffer_store`` offset operand."""
    stores = calls_named(kernel, "buffer_store")
    assert len(stores) == 1, f"expected one buffer store in {kernel.name}"
    offset = stores[0].args[2]
    assert isinstance(offset, ast.Name)
    return last_binding(kernel, offset.id)


def descriptor_offset_exprs(kernel: ast.FunctionDef) -> list[ast.expr]:
    """Return the expressions bound to the descriptor ``add_offsets`` operands."""
    updates = calls_named(kernel, "update_tensor_descriptor")
    assert len(updates) == 1, f"expected one descriptor update in {kernel.name}"
    offsets = [
        keyword.value for keyword in updates[0].keywords if keyword.arg == "add_offsets"
    ]
    assert len(offsets) == 1
    assert isinstance(offsets[0], ast.List)
    return [
        last_binding(kernel, element.id)
        for element in offsets[0].elts
        if isinstance(element, ast.Name)
    ]


def descriptor_extent_expr(kernel: ast.FunctionDef) -> ast.expr:
    """Return the expression bound to the descriptor row extent."""
    # The extent is the argument the kernel recomputes for the no-gather path;
    # it is the only expert-local rebinding of the descriptor row count.
    return last_binding(kernel, "descriptor_m")


@pytest.mark.parametrize("path", KERNEL_SOURCES, ids=lambda path: path.name)
def test_narrows_every_32_bit_instruction_operand(path: Path) -> None:
    """Each value fed to a 32-bit operand is narrowed where it is handed over."""
    kernel = matmul_kernel(path)

    operands = [
        descriptor_extent_expr(kernel),
        store_offset_expr(kernel),
        *descriptor_offset_exprs(kernel),
    ]
    assert len(operands) == 3, "expected extent, store offset and column offset"
    for operand in operands:
        assert cast_width(operand) == "int32", ast.dump(operand)
