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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Startup drives real scheduler plans without consuming the live scheduler."""

from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch
from tokenspeed_scheduler import AttnKind, CacheGroupConfig, Scheduler, SchedulerConfig

from tokenspeed.runtime.execution.startup_warmup import warmup_request_path
from tokenspeed.runtime.execution.types import ModelExecutionResult

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _config(decode_width):
    config = SchedulerConfig()
    config.prefix_granularity = 16
    config.max_scheduled_tokens = 16
    config.max_batch_size = 3
    config.num_device_pages = 256
    config.num_snapshot_pages = 1
    config.decode_input_tokens = decode_width
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full",
            block_granularity=16,
            total_pages=256,
            attn_kind=AttnKind.Full,
        )
    ]
    return config


class _Device:
    def __init__(self, decode_width, *, fail_decode):
        self.width = decode_width
        self.fail_decode = fail_decode
        self.batches = []
        self.prefixes = []
        self.zeroed = False
        self.synced = 0
        self.settled = 0
        self.drained = 0

    def execute(self, plan, planned, *, cache_ops):
        assert not cache_ops
        self.zeroed |= any(plan.pages_to_zero.values())
        if planned is None:
            return None
        batch = planned.forward_op
        extends = batch.num_extends()
        count = len(batch.request_ids)
        self.batches.append((count, extends))
        self.prefixes.extend(batch.extend_prefix_lens)
        assert len(planned.stop_criteria) == count
        assert all(not value.stop_token_ids for value in planned.stop_criteria)
        if extends == 0 and self.fail_decode:
            raise RuntimeError("device failed before startup completion")
        tokens, lengths = [], []
        for row in range(count):
            if row < extends:
                complete = (
                    batch.extend_prefix_lens[row] + batch.input_lengths[row]
                    == batch.prefill_lengths[row]
                )
                kept = int(complete)
                tokens.extend([7] * kept)
            else:
                kept = 1 + row % self.width
                tokens.extend([7] * self.width)
            lengths.append(kept)
        device = self

        class CopyEvent:
            def synchronize(self):
                device.synced += 1

        return ModelExecutionResult(
            output_tokens=torch.tensor(tokens, dtype=torch.int32),
            output_lengths=torch.tensor(lengths, dtype=torch.int32),
            copy_event=CopyEvent(),
        )

    def settle(self):
        self.settled += 1

    def synchronize(self):
        self.drained += 1


@pytest.mark.parametrize("width", [1, 4])
def test_startup_covers_cold_prefix_and_full_decode_without_live_admissions(width):
    config = _config(width)
    live = Scheduler(config)
    available = live.available_lcm_blocks()
    device = _Device(width, fail_decode=False)
    forwards = warmup_request_path(
        device,
        config,
        phases=((1, 5), (3, 31)),
        decode_width=width,
        decode_steps=2,
    )
    assert forwards == device.synced == len(device.batches)
    assert device.zeroed and max(device.prefixes) > 0
    assert device.batches.count((1, 0)) == 2
    assert device.batches.count((3, 0)) == 2
    assert device.settled == device.drained == 2
    assert live.available_lcm_blocks() == available
    assert live.waiting_size() == live.decoding_size() == 0


def test_startup_failure_drains_pending_device_work_and_propagates():
    device = _Device(4, fail_decode=True)
    with pytest.raises(RuntimeError, match="device failed before startup completion"):
        warmup_request_path(
            device,
            _config(4),
            phases=((1, 5),),
            decode_width=4,
            decode_steps=2,
        )
    assert device.settled == device.drained == 1


def test_startup_reaches_sixteen_requests_after_chunked_admission():
    config = _config(4)
    config.max_batch_size = 16
    config.max_scheduled_tokens = 4096
    config.num_device_pages = 8192
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full",
            block_granularity=16,
            total_pages=8192,
            attn_kind=AttnKind.Full,
        )
    ]
    config.enable_mixed_prefill_decode = False
    config.overlap_schedule_depth = 1
    device = _Device(4, fail_decode=False)
    warmup_request_path(
        device,
        config,
        phases=((16, 4160),),
        decode_width=4,
        decode_steps=2,
    )
    assert device.batches.count((16, 0)) == 2
