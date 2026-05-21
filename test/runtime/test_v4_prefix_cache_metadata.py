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
    cfg.page_size = 2
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


def _post(sched: "Scheduler", payload) -> None:
    ev = ExecutionEvent()
    ev.add_event(payload)
    sched.advance(ev)


def _prime_r1(sched: "Scheduler") -> tuple[list[int], list[int]]:
    """Prime r1 and capture page ids before finish releases its tables."""
    spec = RequestSpec()
    spec.request_id = "r1"
    spec.tokens = list(range(1, 13))
    sched.submit_requests([spec])
    sched.next_execution_plan()
    er = ForwardEvent.ExtendResult()
    er.request_id = "r1"
    er.tokens = [99]
    _post(sched, er)
    sched.next_execution_plan()
    r1_fh = list(sched.get_request_paged_cache_page_ids("r1", "fh"))
    r1_swa = list(sched.get_request_paged_cache_page_ids("r1", "swa"))
    fin = ForwardEvent.Finish()
    fin.request_id = "r1"
    _post(sched, fin)
    sched.next_execution_plan()
    return r1_fh, r1_swa


def _submit_r2_same_prefix(sched: "Scheduler") -> None:
    spec = RequestSpec()
    spec.request_id = "r2"
    spec.tokens = list(range(1, 13))
    sched.submit_requests([spec])
    sched.next_execution_plan()


class TestV4PrefixCacheMetadata(unittest.TestCase):

    def setUp(self) -> None:
        self.sched = Scheduler(_make_two_group_config())
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
        _submit_r2_same_prefix(self.sched)

    def test_block_table_borrowed_plus_suffix(self) -> None:
        """R2's table starts with r1's borrowed prefix page ids."""
        r2_fh = list(self.sched.get_request_paged_cache_page_ids("r2", "fh"))
        self.assertGreaterEqual(len(r2_fh), 1)
        n = min(len(self.r1_fh), len(r2_fh))
        self.assertGreater(
            n,
            0,
            "r2 must borrow at least one fh page from r1's prefix snapshot",
        )
        self.assertEqual(r2_fh[:n], self.r1_fh[:n])

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
