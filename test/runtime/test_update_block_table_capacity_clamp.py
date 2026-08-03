"""Regression: page-overflow in ``update_block_table`` must NOT crash engine.

Reproduces the engine-killing crash analyzed in dashllm1.log:

    RuntimeError: page copy would exceed req_to_page capacity:
      row size=514 > req_to_page.shape[1]=513

Root cause (per-iter): when an MTP request approaches ``context_len`` with
accept_rate collapsed to 0, the scheduler still reserves spec lookahead pages
each iter. Eventually a request reaches the per-request page cap
(``req_to_page.shape[1]``) and the next allocation goes past it, raising a
``RuntimeError`` that tears down the **entire engine** (all in-flight
requests die with the gloo cascade visible in the log).

The fix in ``update_block_table`` clamps the offending request's row to the
table capacity, logs a warning, and lets the other requests proceed.
The offending request's KV becomes incomplete from that iter onward, but it
is past its ``max_new_tokens`` clamp and will be naturally marked
``FINISH_LENGTH`` shortly.

Tests use a lightweight ``SimpleNamespace`` stand-in for ``forward_op`` so we
don't depend on the C++ scheduler binding. The ``write_req_to_page_rows`` kernel
is itself stubbed (we assert what arguments it receives), keeping the test
CPU-only and GPU-free.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch


def _make_forward_op(
    rows: list[list[int]],
    request_ids: list[str] | None = None,
    request_pool_indices: list[int] | None = None,
    group_id: str = "full_attention",
) -> SimpleNamespace:
    """Build a minimal forward op exporting a complete per-group block table."""
    if request_ids is None:
        request_ids = [f"req-{i}" for i in range(len(rows))]
    if request_pool_indices is None:
        request_pool_indices = list(range(len(rows)))
    width = max((len(row) for row in rows), default=0)
    table = np.full((len(rows), width), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        table[index, : len(row)] = row
    return SimpleNamespace(
        request_ids=request_ids,
        request_pool_indices=request_pool_indices,
        block_tables_arrays=lambda: {group_id: table},
    )


def test_update_block_table_does_not_raise_on_overflow(monkeypatch):
    """Per-request overflow used to ``raise RuntimeError`` and kill the engine.

    Now it must clamp the offending request's ``size`` and proceed without
    raising, so the rest of the batch survives.
    """
    from tokenspeed.runtime.execution import cache_loc_kernel

    # max_pages=513 (the value from the real crash). req[1] is the offender.
    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(
        rows=[list(range(2)), list(range(514)), list(range(3))],
    )

    captured: dict = {}

    def fake_write_req_to_page_rows(
        req_to_page,
        req_pool_indices,
        page_ids,
        page_counts,
    ):
        captured["num"] = page_counts.tolist()
        captured["pages"] = page_ids.tolist()

    monkeypatch.setattr(
        cache_loc_kernel, "write_req_to_page_rows", fake_write_req_to_page_rows
    )

    # Must not raise.
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    assert captured["num"] == [2, 513, 3]
    assert len(captured["pages"]) == 518


def test_update_block_table_passthrough_when_no_overflow(monkeypatch):
    """When no request overflows, behavior must be identical to the old path."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(rows=[[1], [2, 3], [4]])

    captured: dict = {}

    def fake_write_req_to_page_rows(
        req_to_page,
        req_pool_indices,
        page_ids,
        page_counts,
    ):
        captured["num"] = page_counts.tolist()

    monkeypatch.setattr(
        cache_loc_kernel, "write_req_to_page_rows", fake_write_req_to_page_rows
    )
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    # Sizes survive untouched.
    assert captured["num"] == [1, 2, 1]


def test_update_block_table_clamps_row_to_capacity(monkeypatch):
    """A representative row longer than req_to_page is truncated."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(rows=[list(range(516))])

    captured: dict = {}

    def fake_write_req_to_page_rows(
        req_to_page,
        req_pool_indices,
        page_ids,
        page_counts,
    ):
        captured["num"] = page_counts.tolist()
        captured["pages"] = page_ids.tolist()

    monkeypatch.setattr(
        cache_loc_kernel, "write_req_to_page_rows", fake_write_req_to_page_rows
    )
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    assert captured["num"] == [513]
    assert captured["pages"] == list(range(513))


def test_update_block_table_zero_total_returns_early(monkeypatch):
    """If every size is 0 the function must short-circuit (no kernel call)."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(rows=[[], []])

    called = {"v": False}

    def fake_write_req_to_page_rows(**kwargs):
        called["v"] = True

    monkeypatch.setattr(
        cache_loc_kernel, "write_req_to_page_rows", fake_write_req_to_page_rows
    )
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )
    assert called["v"] is False


def test_update_block_table_logs_warning_on_clamp():
    """Engine survives, but the clamp must be loud (logger.warning) so the
    upstream length-bound bug remains visible. cache_loc_kernel uses a
    non-propagating colorful logger, so caplog can't see it; attach a direct
    capturing handler instead."""
    import logging

    from tokenspeed.runtime.execution import cache_loc_kernel

    captured_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = _Capture(level=logging.WARNING)
    cache_loc_kernel.logger.addHandler(handler)
    try:
        req_to_page = torch.zeros(8, 513, dtype=torch.int32)
        forward_op = _make_forward_op(
            rows=[list(range(514))],
            request_ids=["my-bad-req"],
        )
        with mock.patch.object(
            cache_loc_kernel, "write_req_to_page_rows", lambda **kw: None
        ):
            cache_loc_kernel.update_block_table(
                forward_op, device="cpu", req_to_page=req_to_page
            )
    finally:
        cache_loc_kernel.logger.removeHandler(handler)

    msgs = [r.getMessage() for r in captured_records]
    assert any("my-bad-req" in m for m in msgs), msgs
    assert any("page copy would exceed req_to_page capacity" in m for m in msgs), msgs


def test_update_block_table_selects_explicit_history_group(monkeypatch):
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(4, 8, dtype=torch.int32)
    forward_op = _make_forward_op(rows=[[91, 92]], group_id="linear_attention_0")
    history = np.array([[11, 12, -1]], dtype=np.int32)
    original_arrays = forward_op.block_tables_arrays
    forward_op.block_tables_arrays = lambda: {
        **original_arrays(),
        "full_attention": history,
    }
    captured: dict = {}

    def fake_write_req_to_page_rows(
        req_to_page,
        req_pool_indices,
        page_ids,
        page_counts,
    ):
        captured["pages"] = page_ids.tolist()
        captured["num"] = page_counts.tolist()

    monkeypatch.setattr(
        cache_loc_kernel, "write_req_to_page_rows", fake_write_req_to_page_rows
    )
    cache_loc_kernel.update_block_table(
        forward_op,
        device="cpu",
        req_to_page=req_to_page,
        history_group_id="full_attention",
    )

    assert captured == {"pages": [11, 12], "num": [2]}


def test_update_block_table_rejects_ambiguous_group_selection():
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(4, 8, dtype=torch.int32)
    forward_op = _make_forward_op(rows=[[91]], group_id="linear_attention_0")
    original_arrays = forward_op.block_tables_arrays
    forward_op.block_tables_arrays = lambda: {
        **original_arrays(),
        "full_attention": np.array([[11]], dtype=np.int32),
    }

    with pytest.raises(ValueError, match="history_group_id"):
        cache_loc_kernel.update_block_table(
            forward_op,
            device="cpu",
            req_to_page=req_to_page,
        )
