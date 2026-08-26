"""Write-location kernels: the sliding-window ring and the page-table width.

The SWA draft's page table is a ring: absolute position ``p`` lives in column
``(p // P) % window_pages``. The first half pins ring mapping (including
wrap-around), null-hole routing to the dummy slot, and agreement with the
full-history kernel before any wrap.

The last test pins that table's width as a runtime argument: ``tl.constexpr``
there recompiles the kernel once per distinct width.
"""

from __future__ import annotations

import inspect
import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from test.runtime.conftest import requires_cuda

from tokenspeed.runtime.execution.cache_loc_kernel import (
    compute_out_cache_loc_kernel,
    compute_out_cache_loc_sliding,
    compute_out_cache_loc_uniform,
    dflash_prepare_decode_kernel,
    fused_decode_input_prep_kernel,
)
from tokenspeed.runtime.execution.draft_page_staging import CacheView

_P = 16  # kernel page size for the tests
_W_PAGES = 4  # ring width: window of 64 tokens


def _sliding(table, cache_start, num_tokens):
    bs = table.shape[0]
    out = torch.zeros(bs * num_tokens, dtype=torch.int64, device=table.device)
    compute_out_cache_loc_sliding(
        out_cache_loc_ptr=out,
        uniform_input_length=num_tokens,
        cache_start=cache_start,
        page_table=table,
        page_size=_P,
    )
    return out.view(bs, num_tokens)


def _reference(table_row, position):
    ring_col = (position // _P) % table_row.shape[0]
    page = int(table_row[ring_col])
    if page <= 0:
        return 0
    return page * _P + position % _P


@requires_cuda
def test_ring_mapping_matches_reference() -> None:
    table = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]], device="cuda", dtype=torch.int32
    )
    # Request 0 writes at 5 (inside page 0); request 1 at 35 (page 2 of its ring).
    cache_start = torch.tensor([5, 35], device="cuda", dtype=torch.int32)
    locs = _sliding(table, cache_start, num_tokens=3)
    for req in range(2):
        for k in range(3):
            expected = _reference(table[req].cpu(), int(cache_start[req]) + k)
            assert int(locs[req, k]) == expected, (req, k)


@requires_cuda
def test_wraparound_reuses_ring_columns() -> None:
    """Positions past the window wrap onto the ring's first columns.

    Position 64 with a 4-page ring of 16-token pages maps to column
    (64 // 16) % 4 == 0 — the scheduler has by then punched/reused that
    column for the new page, so the kernel must read column 0, not a
    nonexistent column 4.
    """
    table = torch.tensor([[91, 12, 13, 14]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([64], device="cuda", dtype=torch.int32)
    locs = _sliding(table, cache_start, num_tokens=1)
    assert int(locs[0, 0]) == 91 * _P + 0


@requires_cuda
def test_null_hole_routes_to_dummy_slot() -> None:
    """SwaManager punches slid-out columns to the null page; a write that
    lands on one must route to slot 0, never to page 0's real bytes."""
    table = torch.tensor([[0, 12, 13, 14]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([2], device="cuda", dtype=torch.int32)  # column 0
    locs = _sliding(table, cache_start, num_tokens=1)
    assert int(locs[0, 0]) == 0


@requires_cuda
def test_agrees_with_full_history_before_wrap() -> None:
    """Inside the first window the ring is the identity mapping: both
    kernels must produce identical slots from the same table."""
    table = torch.tensor([[7, 8, 9, 10]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([10], device="cuda", dtype=torch.int32)
    num = 20  # spans pages 0..1 of the window, no wrap
    got = _sliding(table, cache_start, num)
    ref = torch.zeros(num, dtype=torch.int64, device="cuda")
    compute_out_cache_loc_uniform(
        out_cache_loc_ptr=ref,
        uniform_input_length=num,
        cache_start=cache_start,
        page_table=table,
        page_size=_P,
    )
    assert torch.equal(got[0], ref)


@requires_cuda
def test_cache_view_dispatches_by_retention() -> None:
    table = torch.tensor([[91, 12, 13, 14]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([64], device="cuda", dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int64, device="cuda")

    sliding_view = CacheView(table, kernel_page_size=_P, retention="sliding_window")
    sliding_view.out_cache_loc_uniform(out, cache_start, num_tokens=1)
    assert int(out[0]) == 91 * _P  # ring-wrapped onto column 0

    full_view = CacheView(table, kernel_page_size=_P, retention="full_history")
    full_view.out_cache_loc_uniform(out, cache_start, num_tokens=1)
    # Full-history clamps the out-of-table page index and routes overflow
    # to slot 0 — a different, non-ring behavior.
    assert int(out[0]) == 0


def test_cache_view_rejects_unknown_retention() -> None:
    with pytest.raises(ValueError, match="retention"):
        CacheView(
            torch.zeros(1, 1, dtype=torch.int32),
            kernel_page_size=16,
            retention="state",
        )


@requires_cuda
def test_capture_replay_address_stability() -> None:
    """The staged table's address is recorded at capture; replays must see
    in-place updates. Mirrors how the drafter's multi-step loop uses it."""
    table = torch.tensor([[11, 12, 13, 14]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([0], device="cuda", dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int64, device="cuda")

    # Warmup (Triton JIT) outside capture.
    compute_out_cache_loc_sliding(
        out_cache_loc_ptr=out,
        uniform_input_length=1,
        cache_start=cache_start,
        page_table=table,
        page_size=_P,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compute_out_cache_loc_sliding(
            out_cache_loc_ptr=out,
            uniform_input_length=1,
            cache_start=cache_start,
            page_table=table,
            page_size=_P,
        )
    table[0, 0] = 77  # in-place update, same address
    graph.replay()
    torch.cuda.synchronize()
    assert int(out[0]) == 77 * _P


# --------------------------------------------------------------------------
# Page-table width stays a runtime argument
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel",
    [
        compute_out_cache_loc_kernel,
        fused_decode_input_prep_kernel,
        dflash_prepare_decode_kernel,
    ],
    ids=["out_cache_loc", "fused_decode_input_prep", "dflash_prepare_decode"],
)
def test_max_pages_is_not_constexpr(kernel) -> None:
    """Re-annotating it ``tl.constexpr`` is the regression to catch."""
    parameter = inspect.signature(kernel.fn).parameters["max_pages"]
    assert parameter.annotation is inspect.Parameter.empty
