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

"""Exercise final-extent scheduling through the runtime configuration bridge."""

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.engine.scheduler_utils import make_config


@pytest.mark.parametrize("disable_prefix_cache", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("token_budget, lengths", [(1024, [868]), (768, [768, 100])])
def test_final_extent_and_decode_through_runtime_config(
    disable_prefix_cache: bool,
    with_state: bool,
    token_budget: int,
    lengths: list[int],
) -> None:
    """Cover whole/chunked prefill and decode across cache and model families.

    There is no prefix hit or promotion here: only the token budget can shorten
    the 868-token extent. This checks scheduling, not GPU checkpoint contents.
    """
    groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=128,
            total_pages=64,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
        )
    ]
    if with_state:
        groups.append(
            ts.CacheGroupConfig(
                group_id="state",
                block_granularity=128,
                total_pages=64,
                retention=ts.CacheRetention.FullHistory,
                family=ts.CacheGroupFamily.State,
            )
        )
    cfg = make_config(
        num_device_pages=64,
        max_scheduled_tokens=token_budget,
        max_batch_size=1,
        prefix_granularity=128,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        enable_kv_cache_events=False,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        disable_prefix_cache=disable_prefix_cache,
        cache_groups=groups,
        enable_mixed_prefill_decode=False,
        prefix_replay_tokens=0,
    )
    scheduler = ts.Scheduler(cfg)
    request = ts.RequestSpec()
    request.request_id = "prefill"
    request.tokens = list(range(868))
    request.max_new_tokens = 2
    scheduler.submit_requests([request])

    before = 0
    for length in lengths:
        plan = scheduler.next_execution_plan()
        forwards = [op for op in plan.forward if op.request_ids]
        assert len(forwards) == 1
        batch = forwards[0]
        assert list(batch.input_lengths) == [length]
        assert list(batch.extend_prefix_lens) == [before]
        before += length
        result = ts.ForwardEvent.ExtendResult()
        result.request_id = request.request_id
        result.tokens = [900] if before == 868 else []
        event = ts.ExecutionEvent()
        event.add_event(result)
        scheduler.advance(event)

    plan = scheduler.next_execution_plan()
    forwards = [op for op in plan.forward if op.request_ids]
    assert len(forwards) == 1
    # Local decode obtains its token from the runtime's device-side buffer.
    assert list(forwards[0].decode_input_ids) == [-1]
    assert list(forwards[0].input_lengths) == [1]
    assert forwards[0].num_extends() == 0
