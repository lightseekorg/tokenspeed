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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""CacheBatchMetadata.kernel_table: the ONE logical->kernel page expansion.

Attention backends read physical kernel pages only; the scheduler exports
logical CacheBlock ids. This accessor owns the conversion, so its contract is
load-bearing for every MLA backend:

* token-location invariant: expanded[i, t//p] * p + t%p == table[i, t//P] * P + t%P
* memoization per (group, page_size, max_pages) within one forward operation
* -1 holes and the null page 0 expand inside the physical null page
* freshness and divisibility checks fail loudly
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import full_attention_metadata_for as _metadata_for
from test.runtime.conftest import make_kimi_pool as _make_pool
from test.runtime.conftest import requires_cuda

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

_KERNEL_PAGE = 64


def _metadata(pool, rows):
    return _metadata_for(pool, np.array(rows, dtype=np.int32), "cuda")


@requires_cuda
def test_kernel_table_expansion_preserves_token_locations() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    P = pool.arena.prefix_granularity
    metadata, op = _metadata(pool, [[3, 5], [1, 4]])

    logical = metadata.require_full_attention_table(active_forward_op=op)
    kernel = metadata.kernel_table(
        kernel_page_size=_KERNEL_PAGE, max_pages=None, active_forward_op=op
    )
    ratio = P // _KERNEL_PAGE
    assert kernel.shape == (2, logical.shape[1] * ratio)
    # Invariant: same absolute slot for every live token position.
    for req in range(2):
        for t in range(0, 2 * P, 17):
            logical_slot = int(logical[req, t // P]) * P + t % P
            kernel_slot = int(kernel[req, t // _KERNEL_PAGE]) * _KERNEL_PAGE + (
                t % _KERNEL_PAGE
            )
            assert kernel_slot == logical_slot, (req, t)


@requires_cuda
def test_kernel_table_is_memoized_per_geometry() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    metadata, op = _metadata(pool, [[3, 5]])

    a = metadata.kernel_table(
        kernel_page_size=_KERNEL_PAGE, max_pages=8, active_forward_op=op
    )
    b = metadata.kernel_table(
        kernel_page_size=_KERNEL_PAGE, max_pages=8, active_forward_op=op
    )
    assert a is b
    # A different width is a different view, not the cached one.
    c = metadata.kernel_table(
        kernel_page_size=_KERNEL_PAGE, max_pages=4, active_forward_op=op
    )
    assert c is not a
    assert c.shape[1] == 4


@requires_cuda
def test_kernel_table_identity_when_sizes_match() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    metadata, op = _metadata(pool, [[3, 5]])

    table = metadata.kernel_table(
        kernel_page_size=pool.arena.prefix_granularity,
        max_pages=None,
        active_forward_op=op,
    )
    logical = metadata.require_full_attention_table(active_forward_op=op)
    assert table.data_ptr() == logical.data_ptr()


@requires_cuda
def test_kernel_table_max_pages_pads_with_null_page() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    P = pool.arena.prefix_granularity
    ratio = P // _KERNEL_PAGE
    metadata, op = _metadata(pool, [[3, 5]])

    wide = metadata.kernel_table(
        kernel_page_size=_KERNEL_PAGE, max_pages=10 * ratio, active_forward_op=op
    )
    assert wide.shape[1] == 10 * ratio
    # Columns past the source width are the null page 0 (always dereferenceable).
    assert wide[0, 2 * ratio :].eq(0).all()


@requires_cuda
def test_kernel_table_rejects_non_divisible_page_size() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    metadata, op = _metadata(pool, [[3, 5]])
    with pytest.raises(RuntimeError, match="not a positive multiple"):
        metadata.kernel_table(kernel_page_size=48, max_pages=None, active_forward_op=op)


@requires_cuda
def test_kernel_table_rejects_stale_forward_op() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    metadata, _ = _metadata(pool, [[3, 5]])
    with pytest.raises(RuntimeError, match="stale"):
        metadata.kernel_table(
            kernel_page_size=_KERNEL_PAGE, max_pages=None, active_forward_op=object()
        )


@requires_cuda
def test_validate_live_pages_flags_null_page_in_live_range() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    P = pool.arena.prefix_granularity
    # Row 0 is fine; row 1 has the null page 0 inside its live range.
    metadata, op = _metadata(pool, [[3, 5], [0, 4]])
    seq_lens = torch.tensor([2 * P, P], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="inside a live range"):
        metadata.validate_live_pages(seq_lens, active_forward_op=op)


@requires_cuda
def test_validate_live_pages_ignores_pages_past_seq_len() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    P = pool.arena.prefix_granularity
    # -1 hole in the tail column is legal padding while seq stays on page 0.
    metadata, op = _metadata(pool, [[3, -1]])
    seq_lens = torch.tensor([P], device="cuda", dtype=torch.int32)
    metadata.validate_live_pages(seq_lens, active_forward_op=op)
