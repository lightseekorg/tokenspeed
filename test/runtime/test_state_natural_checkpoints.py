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

"""Exercise chunked prefill and decode through the runtime configuration bridge."""

from test.ci_system.ci_register import register_cuda_ci

import pytest

register_cuda_ci(est_time=10, suite="runtime-1gpu")

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.engine.scheduler_utils import make_config


def _groups(with_state: bool) -> list:
    groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=128,
            total_pages=64,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    if with_state:
        # A state group covers no token span; the bridge sends 0 for it.
        groups.append(
            ts.CacheGroupConfig(
                group_id="state",
                block_granularity=0,
                total_pages=64,
                attn_kind=ts.AttnKind.State,
            )
        )
    return groups


def _scheduler(*, token_budget: int, with_state: bool, disable_prefix_cache: bool):
    cfg = make_config(
        num_device_pages=64,
        max_scheduled_tokens=token_budget,
        max_batch_size=1,
        prefix_granularity=128,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=1,
        host_group_packing=(),
        max_retracted_requests=0,
        host_write_policy="write_through",
        storage_config=None,
        enable_kv_cache_events=False,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        disable_prefix_cache=disable_prefix_cache,
        cache_groups=_groups(with_state),
        enable_mixed_prefill_decode=False,
        prefix_replay_tokens=0,
    )
    return ts.Scheduler(cfg)


def _state_row(batch) -> list[int]:
    return list(batch.block_tables_arrays()["state"][0, :2])


@pytest.mark.parametrize("disable_prefix_cache", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize(
    "token_budget, lengths", [(1024, [868]), (768, [768, 100]), (300, [300, 300, 268])]
)
def test_chunks_follow_the_budget_alone(
    disable_prefix_cache: bool,
    with_state: bool,
    token_budget: int,
    lengths: list[int],
) -> None:
    """A state group does not shape the prefill chunk: chunks end where the
    token budget says, on a page boundary or not, and decode follows."""
    scheduler = _scheduler(
        token_budget=token_budget,
        with_state=with_state,
        disable_prefix_cache=disable_prefix_cache,
    )
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


@pytest.mark.parametrize("disable_prefix_cache", [False, True])
def test_state_rows_rotate_between_chunks_and_into_the_first_decode(
    disable_prefix_cache: bool,
) -> None:
    """A state group's row is two slots: the first chunk reads the zero state
    from a fresh zeroed block in slot 0 (zeroed by the plan like any fresh
    page) and writes slot 1; each later chunk reads the previous live block,
    rotated into slot 0, and writes a fresh slot 1. With the prefix cache on,
    the prompt-end block is published, so the first decode reads it from slot
    0 and writes a fresh live block; without it nothing is published and
    decode evolves the live block in place from the first step. Every later
    decode evolves the live block in place with slot 0 null."""
    scheduler = _scheduler(
        token_budget=300, with_state=True, disable_prefix_cache=disable_prefix_cache
    )
    request = ts.RequestSpec()
    request.request_id = "r"
    request.tokens = list(range(700))
    request.max_new_tokens = 4
    scheduler.submit_requests([request])

    rows = []
    for computed in (300, 600, 700):
        plan = scheduler.next_execution_plan()
        (batch,) = [op for op in plan.forward if op.request_ids]
        rows.append(_state_row(batch))
        if computed == 300:
            assert all(block in plan.pages_to_zero["state"] for block in rows[0])
        result = ts.ForwardEvent.ExtendResult()
        result.request_id = "r"
        result.tokens = [900] if computed == 700 else []
        event = ts.ExecutionEvent()
        event.add_event(result)
        scheduler.advance(event)
    first, second, third = rows
    assert first[0] > 0 and first[1] > 0 and first[0] != first[1]
    assert second[0] == first[1] and second[1] > 0 and second[1] != second[0]
    assert third[0] == second[1] and third[1] > 0 and third[1] != third[0]

    plan = scheduler.next_execution_plan()
    (decode,) = [op for op in plan.forward if op.request_ids]
    decode_row = _state_row(decode)
    if disable_prefix_cache:
        assert decode_row == [0, third[1]]
    else:
        assert decode_row[0] == third[1] and decode_row[1] > 0
        assert decode_row[1] != decode_row[0]
    # A state block is never copied; the prompt-end copy-in is the KV page's.
    assert plan.pages_to_copy_arrays()["state"].shape == (0, 2)
    result = ts.ForwardEvent.ExtendResult()
    result.request_id = "r"
    result.tokens = [901]
    event = ts.ExecutionEvent()
    event.add_event(result)
    scheduler.advance(event)

    plan = scheduler.next_execution_plan()
    (decode,) = [op for op in plan.forward if op.request_ids]
    assert _state_row(decode) == [0, decode_row[1]]
