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

"""Remote admission covers the whole suffix, even with a local promotion boundary."""

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _finish, _spec

GROUPS = (
    ("full", 64, 1),
    ("swa.0", 32, 1),
    ("swa.1", 32, 1),
    ("swa.2", 32, 5),
    ("draft.swa", 32, 15),
)
PROBE_TOKENS = list(range(577)) + [9001] * 62


def _scheduler(*, lookahead, width, overlap):
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.D
    cfg.prefix_hash_lookahead_tokens = lookahead
    cfg.prefix_granularity = 64
    cfg.num_device_pages = 257
    cfg.num_host_pages = 0
    cfg.max_scheduled_tokens = 1024
    cfg.max_batch_size = 8
    cfg.decode_input_tokens = width
    cfg.overlap_schedule_depth = overlap
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=group,
            block_granularity=grain,
            total_pages=1 + 256 * packing,
            cache_blocks_per_lcm_block=packing,
            retention=(
                ts.CacheRetention.FullHistory
                if group == "full"
                else ts.CacheRetention.SlidingWindow
            ),
            sliding_window_tokens=None if group == "full" else 513,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
        for group, grain, packing in GROUPS
    ]
    return ts.Scheduler(cfg)


def _event(scheduler, event):
    scheduler.advance(ts.ExecutionEvent().add_event(event))


def _admit_remote(scheduler, request_id, tokens):
    spec = _spec(request_id, tokens)
    spec.max_new_tokens = 256
    scheduler.submit_requests([spec])
    _event(scheduler, ts.PD.BootstrappedEvent(request_id))
    op = scheduler.next_execution_plan().remote_prefill
    assert op is not None
    assert scheduler.pd_transfer_pinned(request_id)
    return op


def _seed_prefix(scheduler, prompt):
    op = _admit_remote(scheduler, "seed", list(range(prompt)))
    assert op.extend_prefix_lens == [0]
    assert op.input_lengths == [prompt]
    _event(scheduler, ts.PD.RemotePrefillDoneEvent("seed", 10000))
    scheduler.next_execution_plan()
    _advance(scheduler, "seed", [10001])
    _finish(scheduler, "seed")
    scheduler.next_execution_plan()
    assert scheduler.active_lcm_blocks() == 0


@pytest.mark.parametrize("lookahead", [0, 1])
@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("overlap", [0, 1])
@pytest.mark.parametrize("seed_prompt, expected_hit", [(634, 0), (600, 576)])
def test_remote_admission_preserves_prompt_extent_and_decode_reserve(
    lookahead, width, overlap, seed_prompt, expected_hit
):
    scheduler = _scheduler(lookahead=lookahead, width=width, overlap=overlap)
    # Both seeds publish Full through 576. SWA starts at slot 3 for 634, missing
    # slot 2 needed to resume 576; at 600 it starts at slot 2 and the hit is usable.
    _seed_prefix(scheduler, seed_prompt)
    remote = _admit_remote(scheduler, "probe", PROBE_TOKENS)
    prompt = len(PROBE_TOKENS)
    assert remote.extend_prefix_lens == [expected_hit]
    assert remote.prefill_lengths == [prompt]
    assert remote.input_lengths == [prompt - expected_hit]

    _event(scheduler, ts.PD.RemotePrefillDoneEvent("probe", 11000))
    assert not scheduler.pd_transfer_pinned("probe")
    # The initial position 639 and subsequent forwards cross a SWA page boundary
    # at every width, including width 1 where the first token alone still fits.
    for step in range(4):
        op = scheduler.next_execution_plan().forward[0]
        assert op.request_ids == ["probe"]
        assert op.num_extends() == 0
        assert op.input_lengths == [width]
        if step == 0:
            assert op.decode_input_ids == [11000]
        computed = prompt + step * width
        for group, grain, _ in GROUPS:
            row = op.block_tables[group][0]
            begin = 0 if group == "full" else max(0, computed - 512)
            for slot in range(begin // grain, (computed + width + grain - 1) // grain):
                assert slot < len(row), (group, computed, slot, row)
                assert row[slot] > 0, (group, computed, slot, row)
        _advance(
            scheduler,
            "probe",
            list(range(11001 + step * width, 11001 + (step + 1) * width)),
        )
    _finish(scheduler, "probe")
    assert scheduler.active_lcm_blocks() == 0


@pytest.mark.parametrize("lookahead", [0, 1])
def test_decode_role_local_recovery_still_chunks_at_promotion(lookahead):
    scheduler = _scheduler(lookahead=lookahead, width=4, overlap=1)
    spec = _spec("probe", list(range(634)))
    spec.max_new_tokens = 8192  # Exceeds admission headroom: eligible for retraction.
    scheduler.submit_requests([spec])
    _event(scheduler, ts.PD.BootstrappedEvent("probe"))
    assert scheduler.next_execution_plan().remote_prefill.input_lengths == [634]
    _event(scheduler, ts.PD.RemotePrefillDoneEvent("probe", 10000))
    scheduler.next_execution_plan()
    _advance(scheduler, "probe", [10001])

    # Force a capacity retraction after publication, without a pending forward.
    # D then recovers locally even without L2; Full's 576-token hit lacks SWA.
    reserve = ts.ForwardEvent.UpdateReserveNumTokens()
    reserve.request_id = "probe"
    reserve.reserve_num_tokens_in_next_schedule_event = 8192
    _event(scheduler, reserve)
    retraction = scheduler.next_execution_plan()
    assert all(not op.request_ids for op in retraction.forward)
    assert scheduler.active_lcm_blocks() == 0
    assert not scheduler.pd_transfer_pinned("probe")
    for prefix, length in ((0, 576), (576, 60)):
        plan = scheduler.next_execution_plan()
        assert plan.remote_prefill is None
        assert plan.remote_decode is None
        op = plan.forward[0]
        assert op.request_ids == ["probe"]
        assert op.extend_prefix_lens == [prefix]
        assert op.input_lengths == [length]
        assert op.prefill_lengths == [636]  # Original prompt plus committed outputs.
        _advance(scheduler, "probe", [] if prefix == 0 else [11000])
    _finish(scheduler, "probe")
    assert scheduler.active_lcm_blocks() == 0
