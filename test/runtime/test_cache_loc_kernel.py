"""Write-location kernel: hole routing, overflow, and the page-table width.

The first tests pin the safety behaviors of the slot math: null/hole pages
and past-the-table positions must route to the dummy slot 0, never to a real
request's bytes, and the staged table's address recorded at capture must see
in-place updates on replay.

The last test pins the table's width as a runtime argument: ``tl.constexpr``
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
    compute_out_cache_loc_uniform,
    dflash_prepare_decode_kernel,
    fused_decode_input_prep_kernel,
)

_P = 16  # kernel page size for the tests


def _uniform(table, cache_start, num_tokens):
    bs = table.shape[0]
    out = torch.zeros(bs * num_tokens, dtype=torch.int64, device=table.device)
    compute_out_cache_loc_uniform(
        out_cache_loc_ptr=out,
        uniform_input_length=num_tokens,
        cache_start=cache_start,
        page_table=table,
        page_size=_P,
    )
    return out.view(bs, num_tokens)


@requires_cuda
def test_mapping_matches_reference() -> None:
    table = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]], device="cuda", dtype=torch.int32
    )
    # Request 0 writes at 5 (inside page 0); request 1 at 35 (its page 2).
    cache_start = torch.tensor([5, 35], device="cuda", dtype=torch.int32)
    locs = _uniform(table, cache_start, num_tokens=3)
    for req in range(2):
        for k in range(3):
            position = int(cache_start[req]) + k
            page = int(table[req, position // _P])
            assert int(locs[req, k]) == page * _P + position % _P, (req, k)


@requires_cuda
def test_null_hole_routes_to_dummy_slot() -> None:
    """A write that lands on a null (id 0) column must route to slot 0,
    never to page 0's real bytes."""
    table = torch.tensor([[0, 12, 13, 14]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([2], device="cuda", dtype=torch.int32)  # column 0
    locs = _uniform(table, cache_start, num_tokens=1)
    assert int(locs[0, 0]) == 0


@requires_cuda
def test_overflow_routes_to_dummy_slot() -> None:
    """A position past the table's last column clamps and routes to slot 0
    instead of reading out of bounds."""
    table = torch.tensor([[91, 12, 13, 14]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([64], device="cuda", dtype=torch.int32)  # column 4
    locs = _uniform(table, cache_start, num_tokens=1)
    assert int(locs[0, 0]) == 0


@requires_cuda
def test_capture_replay_address_stability() -> None:
    """The staged table's address is recorded at capture; replays must see
    in-place updates. Mirrors how the drafter's multi-step loop uses it."""
    table = torch.tensor([[11, 12, 13, 14]], device="cuda", dtype=torch.int32)
    cache_start = torch.tensor([0], device="cuda", dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int64, device="cuda")

    # Warmup (Triton JIT) outside capture.
    compute_out_cache_loc_uniform(
        out_cache_loc_ptr=out,
        uniform_input_length=1,
        cache_start=cache_start,
        page_table=table,
        page_size=_P,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compute_out_cache_loc_uniform(
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
