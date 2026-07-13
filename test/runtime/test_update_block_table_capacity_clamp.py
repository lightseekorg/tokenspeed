"""Regression: page-overflow in ``update_block_table`` must NOT crash engine.

Reproduces the engine-killing crash analyzed in dashllm1.log:

    RuntimeError: page copy would exceed req_to_page capacity:
      begin=513 + size=1 = 514 > req_to_page.shape[1]=513

Root cause (per-iter): when an MTP request approaches ``context_len`` with
accept_rate collapsed to 0, the scheduler still reserves spec lookahead pages
each iter. Eventually a request reaches the per-request page cap
(``req_to_page.shape[1]``) and the next allocation goes past it, raising a
``RuntimeError`` that tears down the **entire engine** (all in-flight
requests die with the gloo cascade visible in the log).

The fix in ``update_block_table`` clamps the offending request's ``size`` to
the remaining capacity, logs a warning, and lets the other requests proceed.
The offending request's KV becomes incomplete from that iter onward, but it
is past its ``max_new_tokens`` clamp and will be naturally marked
``FINISH_LENGTH`` shortly.

Tests use a lightweight ``SimpleNamespace`` stand-in for ``forward_op`` so we
don't depend on the C++ scheduler binding. The ``update_req_to_page`` kernel
is itself stubbed (we assert what arguments it receives), keeping the test
CPU-only and GPU-free.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def _make_forward_op(
    begins: list[int],
    sizes: list[int],
    new_occupied_pages: list[list[int]] | None = None,
    request_ids: list[str] | None = None,
    request_pool_indices: list[int] | None = None,
) -> SimpleNamespace:
    """Build a minimal forward_op stand-in with just the fields the function reads."""
    if new_occupied_pages is None:
        new_occupied_pages = [list(range(s)) for s in sizes]
    if request_ids is None:
        request_ids = [f"req-{i}" for i in range(len(begins))]
    if request_pool_indices is None:
        request_pool_indices = list(range(len(begins)))
    return SimpleNamespace(
        begins=list(begins),
        sizes=list(sizes),
        new_occupied_pages=new_occupied_pages,
        request_ids=request_ids,
        request_pool_indices=request_pool_indices,
    )


def test_update_block_table_does_not_raise_on_overflow(monkeypatch):
    """Per-request overflow used to ``raise RuntimeError`` and kill the engine.

    Now it must clamp the offending request's ``size`` and proceed without
    raising, so the rest of the batch survives.
    """
    from tokenspeed.runtime.execution import cache_loc_kernel

    # max_pages=513 (the value from the real crash). req[1] is the offender:
    # begin=513 + size=1 = 514 > 513.
    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(
        begins=[400, 513, 100],
        sizes=[2, 1, 3],
    )

    captured: dict = {}

    def fake_update_req_to_page(
        req_to_page,
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
    ):
        captured["num"] = new_occupied_pages_num.tolist()
        captured["starts"] = pages_copy_starts.tolist()
        captured["pages"] = new_occupied_pages.tolist()

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)

    # Must not raise.
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    # The offender (req[1]) got clamped to 0, others unchanged.
    assert captured["num"] == [2, 0, 3]
    # begins are unchanged.
    assert captured["starts"] == [400, 513, 100]
    # And the flattened pages array dropped the offending request's entry,
    # so the cumsum-based offsets the kernel uses stay consistent.
    # req[0] contributes 2 pages, req[1] contributes 0, req[2] contributes 3 → 5 total.
    assert len(captured["pages"]) == 5


def test_update_block_table_passthrough_when_no_overflow(monkeypatch):
    """When no request overflows, behavior must be identical to the old path."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(
        begins=[100, 200, 0],
        sizes=[1, 2, 1],
    )

    captured: dict = {}

    def fake_update_req_to_page(
        req_to_page,
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
    ):
        captured["num"] = new_occupied_pages_num.tolist()

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    # Sizes survive untouched.
    assert captured["num"] == [1, 2, 1]


def test_update_block_table_clamp_partial_overflow(monkeypatch):
    """If begin < max_pages and begin+size > max_pages, clamp to (max - begin)."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    # begin=512 + size=4 = 516 > 513, but begin < 513 → clamp to size=1.
    forward_op = _make_forward_op(
        begins=[512],
        sizes=[4],
        new_occupied_pages=[[700, 701, 702, 703]],
    )

    captured: dict = {}

    def fake_update_req_to_page(
        req_to_page,
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
    ):
        captured["num"] = new_occupied_pages_num.tolist()
        captured["pages"] = new_occupied_pages.tolist()

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    assert captured["num"] == [1]
    # Only the first page from new_occupied_pages survives (the one that fits).
    assert captured["pages"] == [700]


def test_update_block_table_zero_total_returns_early(monkeypatch):
    """If every size is 0 the function must short-circuit (no kernel call)."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(begins=[100, 200], sizes=[0, 0])

    called = {"v": False}

    def fake_update_req_to_page(**kwargs):
        called["v"] = True

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)
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
            begins=[513],
            sizes=[1],
            request_ids=["my-bad-req"],
        )
        with mock.patch.object(
            cache_loc_kernel, "update_req_to_page", lambda **kw: None
        ):
            cache_loc_kernel.update_block_table(
                forward_op, device="cpu", req_to_page=req_to_page
            )
    finally:
        cache_loc_kernel.logger.removeHandler(handler)

    msgs = [r.getMessage() for r in captured_records]
    assert any("my-bad-req" in m for m in msgs), msgs
    assert any("page copy would exceed req_to_page capacity" in m for m in msgs), msgs
