# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""V4 prefix-cache metadata tests for per-group block-table /
base-offset wiring exposed by the scheduler when two same-prefix requests
land on the V4 paged-cache adjunct.

The scheduler is stood up via the nanobind bindings and the assertions read
the V4 metadata observable by Python runtime / backend code. No GPU forwards.
"""

from __future__ import annotations

import unittest

import pytest

try:
    from tokenspeed_scheduler import (  # type: ignore
        ExecutionEvent,
        ForwardEvent,
        PagedCacheGroupConfig,
        PagedCacheGroupFamily,
        PagedCacheRetention,
        PrefixCacheAdjunctSpec,
        RequestSpec,
        Scheduler,
        SchedulerConfig,
    )

    _BINDING_AVAILABLE = True
except ImportError:  # pragma: no cover - bindings unbuilt
    _BINDING_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _BINDING_AVAILABLE,
    reason="tokenspeed_scheduler nanobind bindings unavailable in this env",
)


def _make_two_group_config() -> "SchedulerConfig":
    cfg = SchedulerConfig()
    cfg.block_size = 2
    cfg.num_device_pages = 64
    cfg.num_host_pages = 64
    cfg.max_scheduled_tokens = 16
    cfg.max_batch_size = 8
    cfg.paged_cache_groups = [
        PagedCacheGroupConfig(
            group_id="fh",
            rows_per_page=4,
            entry_stride_tokens=1,
            total_pages=32,
            retention=PagedCacheRetention.FullHistory,
        ),
        PagedCacheGroupConfig(
            group_id="swa",
            rows_per_page=2,
            entry_stride_tokens=1,
            total_pages=32,
            retention=PagedCacheRetention.SlidingWindow,
            sliding_window_tokens=8,
        ),
    ]
    # Opt into the prefix-cache adjunct so the scheduler actually builds the
    # snapshot chain; without this the attach loop is a no-op and the
    # borrowed-prefix path under test never runs. The C++ side derives
    # lcm_raw_tokens and sliding_window_per_group from each required group's
    # PagedCacheGroupConfig; the Python ABI only declares the group ids.
    adjunct = PrefixCacheAdjunctSpec()
    adjunct.required_groups = ["fh", "swa"]
    cfg.prefix_cache_adjunct = adjunct
    return cfg


def _make_l2_binding_config() -> "SchedulerConfig":
    cfg = _make_two_group_config()
    cfg.disable_l2_cache = False
    cfg.paged_cache_host_group_pages = {"fh": 32, "swa": 32}
    groups = list(cfg.paged_cache_groups)
    for group in groups:
        group.family = (
            PagedCacheGroupFamily.History
            if group.group_id == "fh"
            else PagedCacheGroupFamily.State
        )
    cfg.paged_cache_groups = groups
    cfg.prefix_cache_adjunct.required_groups = ["fh"]
    return cfg


def _request(request_id: str, tokens) -> "RequestSpec":
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = list(tokens)
    return spec


def _post(sched: "Scheduler", event_type, request_id: str, tokens=None) -> None:
    payload = event_type()
    payload.request_id = request_id
    if tokens is not None:
        payload.tokens = tokens
    ev = ExecutionEvent()
    ev.add_event(payload)
    sched.advance(ev)


def _prime_r1(sched: "Scheduler") -> tuple[list[int], list[int]]:
    """Prime r1 and capture page ids before finish releases its tables."""
    sched.submit_requests([_request("r1", range(1, 13))])
    sched.next_execution_plan()
    _post(sched, ForwardEvent.ExtendResult, "r1", [99])
    sched.next_execution_plan()
    r1_fh = list(sched.get_request_paged_cache_page_ids("r1", "fh"))
    r1_swa = list(sched.get_request_paged_cache_page_ids("r1", "swa"))
    _post(sched, ForwardEvent.Finish, "r1")
    sched.next_execution_plan()
    return r1_fh, r1_swa


def _submit_r2_same_prefix(sched: "Scheduler") -> int:
    sched.submit_requests([_request("r2", range(1, 13))])
    plan = sched.next_execution_plan()
    assert len(plan.forward) == 1
    forward = plan.forward[0]
    assert forward.request_ids == ["r2"]
    assert len(forward.extend_prefix_lens) == 1
    return int(forward.extend_prefix_lens[0])


def test_paged_l2_transfer_binding_shape() -> None:
    sched = Scheduler(_make_l2_binding_config())
    sched.submit_requests([_request("r1", range(1, 13))])
    assert len(sched.next_execution_plan().forward) == 1

    _post(sched, ForwardEvent.ExtendResult, "r1", [99])
    sched.next_execution_plan()

    _post(sched, ForwardEvent.Finish, "r1")
    writeback_plan = sched.next_execution_plan()
    writeback = next(
        (op for op in writeback_plan.cache if hasattr(op, "is_retract")),
        None,
    )
    assert writeback is not None
    assert len(writeback.op_ids) == len(writeback.paged_cache_transfers) == 1
    assert writeback.paged_cache_transfers[0]

    writeback_groups = {
        transfer.group_id for transfer in writeback.paged_cache_transfers[0]
    }
    assert writeback_groups == {"fh", "swa"}
    for transfer in writeback.paged_cache_transfers[0]:
        assert isinstance(transfer.group_id, str)
        assert len(transfer.src_pages) == len(transfer.dst_pages) > 0


def test_scheduler_abort_binding_shape() -> None:
    config = _make_l2_binding_config()
    groups = list(config.paged_cache_groups)
    for group in groups:
        group.total_pages = 3 if group.group_id == "fh" else 5
    config.paged_cache_groups = groups
    sched = Scheduler(config)
    specs = [
        _request(request_id, range(start, start + 4))
        for request_id, start in (("z-request", 1), ("a-request", 101))
    ]
    sched.submit_requests(specs)
    sched.next_execution_plan()

    for request_id, token in (("z-request", 42), ("a-request", 43)):
        _post(sched, ForwardEvent.ExtendResult, request_id, [token])

    plan = sched.next_execution_plan()

    ((request_id, message),) = plan.scheduler_aborts
    assert request_id == "a-request"
    assert "capacity" in message


class TestV4PrefixCacheMetadata(unittest.TestCase):
    def setUp(self) -> None:
        config = _make_two_group_config()
        fh_config = next(
            group for group in config.paged_cache_groups if group.group_id == "fh"
        )
        self.fh_raw_tokens_per_page = (
            fh_config.rows_per_page * fh_config.entry_stride_tokens
        )
        self.sched = Scheduler(config)
        self.r1_fh, self.r1_swa = _prime_r1(self.sched)
        self.assertNotEqual(
            self.r1_fh,
            [],
            "r1 fh page ids must be captured before finish releases the request table",
        )
        self.assertNotEqual(
            self.r1_swa,
            [],
            "r1 swa page ids must be captured before finish releases the request table",
        )
        self.r2_prefix_len = _submit_r2_same_prefix(self.sched)

    def test_block_table_borrowed_plus_suffix(self) -> None:
        """R2's table starts with r1's borrowed prefix page ids."""
        r2_fh = list(self.sched.get_request_paged_cache_page_ids("r2", "fh"))
        self.assertEqual(self.r2_prefix_len % self.fh_raw_tokens_per_page, 0)
        borrowed_pages = self.r2_prefix_len // self.fh_raw_tokens_per_page
        self.assertGreater(
            borrowed_pages,
            0,
            "r2 must borrow at least one fh page from r1's prefix snapshot",
        )
        self.assertGreaterEqual(len(self.r1_fh), borrowed_pages)
        self.assertGreater(
            len(r2_fh),
            borrowed_pages,
            "r2 must allocate a suffix after its borrowed prefix pages",
        )
        self.assertEqual(r2_fh[:borrowed_pages], self.r1_fh[:borrowed_pages])

    def test_base_offsets_sliding_correct(self) -> None:
        """For sliding-window groups the per-request base_logical_page
        matches the snapshot's base offset; full-history is always 0."""
        swa_base = self.sched.get_request_paged_cache_base_logical_page("r2", "swa")
        self.assertGreaterEqual(swa_base, 0)
        swa_ids = list(self.sched.get_request_paged_cache_page_ids("r2", "swa"))
        self.assertLessEqual(swa_base, len(self.r1_swa) + len(swa_ids))
        fh_base = self.sched.get_request_paged_cache_base_logical_page("r2", "fh")
        self.assertEqual(fh_base, 0)


if __name__ == "__main__":
    unittest.main()
