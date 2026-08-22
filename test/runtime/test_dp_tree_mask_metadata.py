from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402


def _forward_op(custom_mask):
    return SimpleNamespace(
        input_lengths=[2],
        request_ids=["request-0"],
        spec_info=SimpleNamespace(custom_mask=custom_mask),
        num_extends=lambda: 0,
    )


def _event_loop_stub():
    return SimpleNamespace(
        kv_transfer=None,
        world_cpu_group=object(),
        _dp_local_info=torch.zeros(1, 4, dtype=torch.int32),
        _dp_global_info=torch.zeros(2, 4, dtype=torch.int32),
    )


def test_dp_sync_publishes_local_tree_mask(monkeypatch):
    loop = _event_loop_stub()
    global_info = torch.tensor(
        [
            [2, 1, int(ForwardMode.DECODE), 1],
            [0, 0, int(ForwardMode.IDLE), 0],
        ],
        dtype=torch.int32,
    )

    def fake_all_gather(output, local, *, group):
        assert group is loop.world_cpu_group
        assert local[0, 3].item() == 1
        output.copy_(global_info)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather)

    metadata = EventLoop._dp_sync_and_check(
        loop, _forward_op(torch.ones(1, dtype=torch.int8))
    )

    assert metadata.any_custom_tree_mask
    assert not metadata.need_idle_forward


def test_dp_sync_propagates_remote_tree_mask_to_idle_rank(monkeypatch):
    loop = _event_loop_stub()
    global_info = torch.tensor(
        [
            [0, 0, int(ForwardMode.IDLE), 0],
            [2, 1, int(ForwardMode.DECODE), 1],
        ],
        dtype=torch.int32,
    )

    def fake_all_gather(output, local, *, group):
        assert group is loop.world_cpu_group
        assert local[0, 3].item() == 0
        output.copy_(global_info)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather)

    metadata = EventLoop._dp_sync_and_check(loop, None)

    assert metadata.any_custom_tree_mask
    assert metadata.need_idle_forward
    assert metadata.all_decode_or_idle
