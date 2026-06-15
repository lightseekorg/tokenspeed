"""Regression coverage for GLM DSA prefill page-table snapshots."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend  # noqa: E402


class _DenseStub:
    def __init__(self) -> None:
        self.chunked_prefill_metadata = None

    def init_forward_metadata(self, *, req_pool_indices, **kwargs):
        self.chunked_prefill_metadata = type(
            "_ChunkMeta", (), {"req_pool_indices": req_pool_indices}
        )()
        return None


def _make_backend() -> DSABackend:
    be = object.__new__(DSABackend)
    be._dense_backend = _DenseStub()
    be._prefill_block_tables = None
    return be


def _init(be, *, num_extends, req_pool_indices, req_to_page, mode) -> None:
    be.init_forward_metadata(
        bs=int(req_pool_indices.numel()),
        num_extends=num_extends,
        req_pool_indices=req_pool_indices,
        seq_lens=torch.ones(req_pool_indices.numel(), dtype=torch.int32),
        forward_mode=mode,
        req_to_page=req_to_page,
    )


def test_extend_snapshots_page_table_in_chunk_order() -> None:
    be = _make_backend()
    req_to_page = torch.arange(5 * 4, dtype=torch.int32).reshape(5, 4)
    req_pool_indices = torch.tensor([2, 0, 3], dtype=torch.int64)

    _init(
        be,
        num_extends=3,
        req_pool_indices=req_pool_indices,
        req_to_page=req_to_page,
        mode=ForwardMode.EXTEND,
    )

    assert be._prefill_block_tables is not None
    assert torch.equal(be._prefill_block_tables, req_to_page[req_pool_indices])


def test_snapshot_is_decoupled_from_mutable_req_to_page() -> None:
    be = _make_backend()
    req_to_page = torch.arange(5 * 4, dtype=torch.int32).reshape(5, 4)
    req_pool_indices = torch.tensor([1, 4], dtype=torch.int64)

    _init(
        be,
        num_extends=2,
        req_pool_indices=req_pool_indices,
        req_to_page=req_to_page,
        mode=ForwardMode.EXTEND,
    )
    snapshot = be._prefill_block_tables.clone()
    req_to_page.fill_(-999)

    assert torch.equal(be._prefill_block_tables, snapshot)


def test_decode_clears_stale_prefill_snapshot() -> None:
    be = _make_backend()
    req_to_page = torch.arange(5 * 4, dtype=torch.int32).reshape(5, 4)

    _init(
        be,
        num_extends=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        req_to_page=req_to_page,
        mode=ForwardMode.EXTEND,
    )
    assert be._prefill_block_tables is not None

    _init(
        be,
        num_extends=0,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        req_to_page=req_to_page,
        mode=ForwardMode.DECODE,
    )

    assert be._prefill_block_tables is None
