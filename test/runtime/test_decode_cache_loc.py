"""Decode write-location safety for null, boundary, and overflow pages."""

from __future__ import annotations

import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from test.runtime.conftest import requires_cuda

from tokenspeed.runtime.execution.cache_loc_kernel import (
    dflash_prepare_decode,
    fused_decode_input_prep,
)

_PAGE_SIZE = 16


@requires_cuda
def test_fused_decode_routes_null_negative_and_overflow_pages_to_zero() -> None:
    req_pool_indices = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.int64)
    valid_cache_lengths = torch.tensor(
        [0, 15, 17, 2, 48], device="cuda", dtype=torch.int32
    )
    page_table = torch.tensor(
        [
            [7, 8, 9],
            [11, 0, 13],
            [-3, 12, 13],
            [21, 22, 23],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    out_cache_loc = torch.full((8,), -1, device="cuda", dtype=torch.int32)
    positions = torch.full((8,), -1, device="cuda", dtype=torch.int64)
    seq_lens = torch.full((4,), -1, device="cuda", dtype=torch.int32)

    fused_decode_input_prep(
        out_cache_loc,
        positions,
        seq_lens,
        req_pool_indices,
        valid_cache_lengths,
        uniform_input_length=2,
        page_table=page_table,
        page_size=_PAGE_SIZE,
    )

    assert out_cache_loc.view(4, 2).tolist() == [
        [7 * _PAGE_SIZE + 15, 8 * _PAGE_SIZE],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert positions.view(4, 2).tolist() == [[15, 16], [17, 18], [2, 3], [48, 49]]
    assert seq_lens.tolist() == [17, 19, 4, 50]


def _run_dflash(
    page_table: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    *,
    max_draft_prefix: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = page_table.shape[0]
    verify_width = 8
    draft_query_width = 3
    output_tokens = torch.arange(
        100,
        100 + batch_size * verify_width,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, verify_width)
    accept_lengths = torch.ones(batch_size, device="cuda", dtype=torch.int32)
    req_pool_indices = torch.arange(1, batch_size + 1, device="cuda", dtype=torch.int64)
    draft_seq_lens = torch.full((batch_size,), -1, device="cuda", dtype=torch.int32)
    block_ids = torch.full(
        (batch_size, draft_query_width), -1, device="cuda", dtype=torch.int32
    )
    block_positions = torch.full(
        (batch_size, draft_query_width), -1, device="cuda", dtype=torch.int64
    )
    out_cache_loc = torch.full(
        (batch_size * draft_query_width,), -1, device="cuda", dtype=torch.int32
    )

    dflash_prepare_decode(
        output_tokens=output_tokens,
        accept_lengths=accept_lengths,
        req_pool_indices=req_pool_indices,
        valid_cache_lengths=valid_cache_lengths,
        page_table=page_table,
        draft_seq_lens=draft_seq_lens,
        block_ids=block_ids,
        block_positions=block_positions,
        out_cache_loc=out_cache_loc,
        verify_width=verify_width,
        draft_query_width=draft_query_width,
        page_size=_PAGE_SIZE,
        max_draft_prefix=max_draft_prefix,
    )
    return draft_seq_lens, block_ids, block_positions, out_cache_loc


@requires_cuda
def test_dflash_routes_null_and_negative_pages_to_zero_across_boundary() -> None:
    page_table = torch.tensor(
        [[7, 0, 9], [-2, 11, 12]], device="cuda", dtype=torch.int32
    )
    valid_cache_lengths = torch.tensor([0, 14, 0], device="cuda", dtype=torch.int32)

    seq_lens, block_ids, positions, cache_locs = _run_dflash(
        page_table,
        valid_cache_lengths,
        max_draft_prefix=45,
    )

    assert seq_lens.tolist() == [15, 1]
    assert block_ids[:, 0].tolist() == [100, 108]
    assert positions.tolist() == [[15, 16, 17], [1, 2, 3]]
    assert cache_locs.view(2, 3).tolist() == [
        [7 * _PAGE_SIZE + 15, 0, 0],
        [0, 0, 0],
    ]


@requires_cuda
def test_dflash_routes_out_of_table_positions_to_zero() -> None:
    page_table = torch.tensor([[7, 8, 9]], device="cuda", dtype=torch.int32)
    valid_cache_lengths = torch.tensor([0, 47], device="cuda", dtype=torch.int32)

    _, _, positions, cache_locs = _run_dflash(
        page_table,
        valid_cache_lengths,
        max_draft_prefix=64,
    )

    assert positions.tolist() == [[48, 49, 50]]
    assert cache_locs.tolist() == [0, 0, 0]


@requires_cuda
def test_dflash_capture_replay_observes_null_page_update() -> None:
    page_table = torch.tensor([[7, 8]], device="cuda", dtype=torch.int32)
    valid_cache_lengths = torch.tensor([0, 0], device="cuda", dtype=torch.int32)

    # Warm up Triton before capture.
    _, _, _, cache_locs = _run_dflash(
        page_table,
        valid_cache_lengths,
        max_draft_prefix=29,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, _, _, cache_locs = _run_dflash(
            page_table,
            valid_cache_lengths,
            max_draft_prefix=29,
        )

    page_table[0, 0] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert cache_locs.tolist() == [0, 0, 0]
