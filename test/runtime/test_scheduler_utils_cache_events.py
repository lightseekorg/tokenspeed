from __future__ import annotations

from tokenspeed.runtime.engine.scheduler_utils import pop_common_cache_event_payloads


def test_cache_event_payloads_commit_only_common_success():
    rank0 = [
        {"kind": "WriteBackDoneEvent", "op_id": 3, "success": True},
        {"kind": "WriteBackDoneEvent", "op_id": 5, "success": True},
    ]
    rank1 = [
        {"kind": "WriteBackDoneEvent", "op_id": 3, "success": False},
    ]

    ready = pop_common_cache_event_payloads([rank0, rank1])

    assert ready == [{"kind": "WriteBackDoneEvent", "op_id": 3, "success": False}]
    assert pop_common_cache_event_payloads([rank0, []]) == []
