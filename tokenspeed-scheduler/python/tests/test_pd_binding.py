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

"""Tests for PD Python bindings."""

import pytest
from tokenspeed_scheduler import (
    PD,
    AttnKind,
    CacheGroupConfig,
    CacheTransferPolicy,
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def make_scheduler() -> Scheduler:
    cfg = SchedulerConfig()
    cfg.prefix_granularity = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 1024
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="full_attention",
            block_granularity=cfg.prefix_granularity,
            total_pages=cfg.num_device_pages,
            attn_kind=AttnKind.Full,
        )
    ]
    return Scheduler(cfg)


def make_spec(request_id: str, tokens: list[int]) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    return spec


def test_pd_event_fields_are_bound():
    """PD event objects require request_id constructor arg and expose it as read-only."""
    event = PD.BootstrappedEvent("req-0")

    assert event.request_id == "req-0"


@pytest.mark.parametrize(
    "terminal", ["success", "failure", "cancel_success", "cancel_failure"]
)
@pytest.mark.parametrize("pool_pages", [8, 12])
@pytest.mark.parametrize("transfer_stops_first", [True, False])
def test_prefill_duplicate_sources_survive_until_pd_terminal(
    terminal, pool_pages, transfer_stops_first
):
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.P
    cfg.decode_allocation_policy = SchedulerConfig.DecodeAllocationPolicy.Early
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 8
    cfg.max_batch_size = 4
    cfg.num_device_pages = pool_pages
    cfg.num_snapshot_pages = 1
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=pool_pages,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([make_spec(rid, [1, 2, 3, 4]) for rid in ("a", "b")])
    scheduler.advance(
        ExecutionEvent()
        .add_event(PD.BootstrappedEvent("a"))
        .add_event(PD.BootstrappedEvent("b"))
    )
    first = scheduler.next_execution_plan().forward[0]
    sources = {
        rid: [int(p) for p in row if p > 0]
        for rid, row in zip(first.request_ids, dict(first.block_tables)["history"])
    }
    assert set(sources["a"]).isdisjoint(sources["b"])
    for rid in ("a", "b"):
        result = ForwardEvent.ExtendResult()
        result.request_id, result.tokens = rid, [5]
        scheduler.advance(ExecutionEvent().add_event(result))
    handoff = scheduler.next_execution_plan().remote_decode
    actual = {
        rid: [int(p) for p in row if p > 0]
        for rid, row in zip(handoff.request_ids, dict(handoff.block_tables)["history"])
    }
    assert (
        actual == sources
    ), "prefix publication must not replace transfer source pages"
    if terminal.startswith("cancel"):
        abort = ForwardEvent.Abort()
        abort.request_id = "b"
        scheduler.advance(ExecutionEvent().add_event(abort))
        assert scheduler.pd_transfer_size() == 2
    scheduler.submit_requests([make_spec("c", [10, 11])])
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("c")))
    later = scheduler.next_execution_plan().forward[0]
    if pool_pages == 8:
        assert (
            not later.request_ids
        ), "capacity pressure must wait instead of recycling a PD source"
    else:
        allocated = {
            int(p) for row in dict(later.block_tables)["history"] for p in row if p > 0
        }
        assert allocated.isdisjoint(set(sources["a"] + sources["b"]))
    pinned_before_terminal = scheduler.active_lcm_blocks()
    transfers_before_terminal = scheduler.pd_transfer_size()
    for rid in ("a", "b"):
        event = (
            PD.SucceededEvent(rid)
            if terminal.endswith("success")
            else PD.FailedEvent(rid)
        )
        scheduler.advance(ExecutionEvent().add_event(event))
    if terminal.endswith("failure"):
        assert scheduler.pd_transfer_size() == transfers_before_terminal
        assert (
            scheduler.active_lcm_blocks() == pinned_before_terminal
        ), "failure alone must not release transfer sources"
        if pool_pages == 8:
            waiting = scheduler.next_execution_plan().forward[0]
            assert not waiting.request_ids, "failed transfers still own their pages"
            assert scheduler.active_lcm_blocks() == pinned_before_terminal
        for rid in ("a", "b"):
            scheduler.advance(ExecutionEvent().add_event(PD.TransferStoppedEvent(rid)))
    if pool_pages == 8:
        later = scheduler.next_execution_plan().forward[0]
        assert list(later.request_ids) == ["c"], "terminal ACK must unblock allocation"
    scheduler.advance(ExecutionEvent().add_event(PD.FailedEvent("c")))
    if transfer_stops_first:
        scheduler.advance(ExecutionEvent().add_event(PD.TransferStoppedEvent("c")))
    assert scheduler.active_lcm_blocks() > 0, "the local forward has not landed yet"
    # Repeated cancellation must preserve any transfer retirement already received.
    abort = ForwardEvent.Abort()
    abort.request_id = "c"
    scheduler.advance(ExecutionEvent().add_event(abort))
    scheduler.advance(ExecutionEvent().add_event(abort))
    result = ForwardEvent.ExtendResult()
    result.request_id, result.tokens = "c", [12]
    scheduler.advance(ExecutionEvent().add_event(result))
    if not transfer_stops_first:
        assert scheduler.active_lcm_blocks() > 0, "the PD transfer has not stopped yet"
        scheduler.advance(ExecutionEvent().add_event(PD.TransferStoppedEvent("c")))
    scheduler.next_execution_plan()
    assert (
        scheduler.active_lcm_blocks() == 0
    ), "terminal requests must release every pin"
    assert scheduler.available_lcm_blocks() == cfg.num_device_pages - 1


def test_independent_prefill_counts_prompts_waiting_for_their_destination():
    """Under independent decode allocation computed prompts wait for their
    decode destination with their pages; the scheduler counts them and the
    pages in their block tables."""
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.P
    cfg.decode_allocation_policy = SchedulerConfig.DecodeAllocationPolicy.Independent
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 8
    cfg.max_batch_size = 3
    cfg.num_device_pages = 64
    cfg.num_snapshot_pages = 1
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=64,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(cfg)
    rids = ["oldest", "older", "newest"]
    scheduler.submit_requests(
        [make_spec(rid, [10 * i + 1, 10 * i + 2]) for i, rid in enumerate(rids)]
    )
    admitted = ExecutionEvent()
    for rid in rids:
        admitted.add_event(PD.BootstrappedEvent(rid))
    scheduler.advance(admitted)
    assert scheduler.next_execution_plan().forward[0].request_ids == rids
    for rid in rids:
        result = ForwardEvent.ExtendResult()
        result.request_id, result.tokens = rid, [5]
        scheduler.advance(ExecutionEvent().add_event(result))
    assert scheduler.awaiting_destination_size() == 3
    assert scheduler.awaiting_destination_lcm_blocks() == scheduler.active_lcm_blocks()

    scheduler.advance(ExecutionEvent().add_event(PD.DestinationReadyEvent("older")))
    handoff = scheduler.next_execution_plan().remote_decode
    assert handoff is not None and list(handoff.request_ids) == ["older"]
    assert scheduler.awaiting_destination_size() == 2
    assert (
        0 < scheduler.awaiting_destination_lcm_blocks() < scheduler.active_lcm_blocks()
    )


def test_execution_event_accepts_pd_events():
    """ExecutionEvent.add_event accepts PD events and returns self for chaining."""
    execution_event = ExecutionEvent()
    event = PD.SucceededEvent("req-0")

    assert execution_event.add_event(event) is execution_event


def test_execution_plan_exposes_forward():
    scheduler = make_scheduler()
    scheduler.submit_requests([make_spec("r0", [1, 2, 3, 4])])

    plan = scheduler.next_execution_plan()

    assert len(plan.forward) == 1
    assert plan.forward[0].request_ids == ["r0"]


def test_pd_counters_follow_request_state():
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.D
    cfg.prefix_granularity = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 64
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=16,
            total_pages=64,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([make_spec("remote", [1, 2, 3, 4])])
    assert scheduler.bootstrapping_size() == 1
    assert scheduler.remote_prefilling_size() == scheduler.pd_transfer_size() == 0
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("remote")))
    assert scheduler.bootstrapping_size() == 0
    scheduler.next_execution_plan()
    assert scheduler.remote_prefilling_size() == scheduler.pd_transfer_size() == 1
    scheduler.advance(
        ExecutionEvent().add_event(PD.RemotePrefillDoneEvent("remote", 5))
    )
    assert scheduler.remote_prefilling_size() == scheduler.pd_transfer_size() == 0
    finish = ForwardEvent.Finish()
    finish.request_id = "remote"
    scheduler.advance(ExecutionEvent().add_event(finish))
    assert scheduler.active_lcm_blocks() == 0


def test_prefill_role_reserves_the_decode_window_on_the_completing_chunk():
    """The P role never decodes, but the chunk that completes a prompt drafts
    the first candidate window, so it reserves ``decode_input_tokens`` exactly
    like a decoding role; intermediate chunks hold only their own tokens."""
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.P
    cfg.decode_allocation_policy = SchedulerConfig.DecodeAllocationPolicy.Early
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 4
    cfg.max_batch_size = 1
    cfg.num_device_pages = 17
    cfg.num_snapshot_pages = 1  # the P role never retracts
    cfg.disable_l2_cache = True
    cfg.decode_input_tokens = 3
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=17,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(cfg)
    scheduler.submit_requests([make_spec("chunked", list(range(8)))])
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("chunked")))

    def held_pages(batch) -> int:
        return len(
            [page for page in dict(batch.block_tables)["history"][0] if page > 0]
        )

    first_chunk = scheduler.next_execution_plan().forward[0]
    assert list(first_chunk.input_lengths) == [4]
    assert held_pages(first_chunk) == 2, "tokens 0..3 only, no reserve yet"

    completing_chunk = scheduler.next_execution_plan().forward[0]
    assert list(completing_chunk.input_lengths) == [4]
    assert held_pages(completing_chunk) == 6, "tokens 0..7 plus a 3-token window"


def test_prefill_role_hands_off_without_the_draft_window():
    """The P role never verifies locally, so its handoff holds only the first
    verify window: a pool sized for the prompt and that window still hands off."""
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.P
    cfg.decode_allocation_policy = SchedulerConfig.DecodeAllocationPolicy.Early
    cfg.prefix_granularity = 1
    cfg.max_scheduled_tokens = 8
    cfg.max_batch_size = 1
    # Page 0 plus the two-token prompt and its four-token window.
    cfg.num_device_pages = 7
    cfg.num_snapshot_pages = 1  # the P role never retracts
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.decode_input_tokens = 4
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=1,
            total_pages=7,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(cfg)
    spec = make_spec("p", [1, 2])
    spec.max_new_tokens = 8
    scheduler.submit_requests([spec])
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("p")))
    assert scheduler.next_execution_plan().forward[0].request_ids == ["p"]
    result = ForwardEvent.ExtendResult()
    result.request_id = "p"
    result.tokens = [3]
    scheduler.advance(ExecutionEvent().add_event(result))
    handoff = scheduler.next_execution_plan().remote_decode
    assert handoff is not None and list(handoff.request_ids) == ["p"]


def test_independent_prefill_holds_its_kv_until_the_destination_is_ready():
    """Under independent decode allocation ``PD.BootstrappedEvent`` only admits
    a prompt; its KV is handed off once ``PD.DestinationReadyEvent`` reports
    that the Decode destination is allocated."""
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.P
    cfg.decode_allocation_policy = SchedulerConfig.DecodeAllocationPolicy.Independent
    cfg.prefix_granularity = 1
    cfg.max_scheduled_tokens = 8
    cfg.max_batch_size = 1
    cfg.num_device_pages = 7
    cfg.num_snapshot_pages = 1  # the P role never retracts
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.decode_input_tokens = 4
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=1,
            total_pages=7,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(cfg)
    spec = make_spec("p", [1, 2])
    spec.max_new_tokens = 8
    scheduler.submit_requests([spec])
    assert not scheduler.next_execution_plan().forward[0].request_ids
    assert scheduler.bootstrapping_size() == 1

    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent("p")))
    assert scheduler.next_execution_plan().forward[0].request_ids == ["p"]
    result = ForwardEvent.ExtendResult()
    result.request_id = "p"
    result.tokens = [3]
    scheduler.advance(ExecutionEvent().add_event(result))
    assert scheduler.next_execution_plan().remote_decode is None

    ready = PD.DestinationReadyEvent("p")
    assert ready.request_id == "p"
    scheduler.advance(ExecutionEvent().add_event(ready))
    handoff = scheduler.next_execution_plan().remote_decode
    assert handoff is not None and list(handoff.request_ids) == ["p"]


@pytest.mark.parametrize("decode_input_tokens", [1, 4])
def test_remote_admission_lands_the_state_in_slot_one_and_decodes_in_place(
    decode_input_tokens,
):
    """A History hit without its sliding lookback cannot split a remote landing;
    the landed state is slot 1 of a two-slot row and every decode writes it in
    place."""
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.D
    cfg.prefix_granularity = 4
    cfg.max_scheduled_tokens = 16
    cfg.max_batch_size = 1
    cfg.num_device_pages = 128
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.decode_input_tokens = decode_input_tokens
    cfg.overlap_schedule_depth = 1
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=4,
            total_pages=128,
            attn_kind=AttnKind.Full,
            transfer_policy=CacheTransferPolicy.FullSuffix,
        ),
        CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=128,
            attn_kind=AttnKind.State,
            transfer_policy=CacheTransferPolicy.LatestSnapshot,
        ),
        CacheGroupConfig(
            group_id="recent",
            block_granularity=1,
            total_pages=128,
            attn_kind=AttnKind.SlidingWindow,
            sliding_window_tokens=2,
            transfer_policy=CacheTransferPolicy.FullSuffix,
        ),
    ]
    scheduler = Scheduler(cfg)

    def submit_remote(request_id, tokens):
        spec = make_spec(request_id, tokens)
        spec.max_new_tokens = 32
        scheduler.submit_requests([spec])
        scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent(request_id)))
        return scheduler.next_execution_plan().remote_prefill

    def complete_decode(request_id, token):
        result = ForwardEvent.ExtendResult()
        result.request_id = request_id
        result.tokens = [token]
        reserve = ForwardEvent.UpdateReserveNumTokens()
        reserve.request_id = request_id
        reserve.reserve_num_tokens_in_next_schedule_event = 1
        scheduler.advance(ExecutionEvent().add_event(result).add_event(reserve))

    seed = list(range(11))
    assert list(submit_remote("seed", seed).input_lengths) == [11]
    scheduler.advance(
        ExecutionEvent().add_event(PD.RemotePrefillDoneEvent("seed", 100))
    )
    scheduler.next_execution_plan()
    complete_decode("seed", 101)
    finish = ForwardEvent.Finish()
    finish.request_id = "seed"
    scheduler.advance(ExecutionEvent().add_event(finish))
    scheduler.next_execution_plan()

    # History covers the shared prefix through token 8, but the remote seed
    # only landed the recent prompt tail, so the common reusable prefix is 0.
    landing = submit_remote("next", seed[:8] + [20, 21, 22])
    assert list(landing.extend_prefix_lens) == [0]
    assert list(landing.prefill_lengths) == [11]
    assert list(landing.input_lengths) == [11]
    assert list(landing.state_prefix_cached) == [False]
    [slot0, slot1] = dict(landing.block_tables)["state"][0]
    assert slot0 == 0, "no prompt-end checkpoint: nothing to read"
    assert slot1 > 0, "the peer's state lands in slot 1"
    scheduler.advance(
        ExecutionEvent().add_event(PD.RemotePrefillDoneEvent("next", 200))
    )

    # Dispatch the next decode before committing the previous one, as the
    # runtime does. Every decode step writes the landed slot in place.
    for step in range(10):
        batch = scheduler.next_execution_plan().forward[0]
        assert list(dict(batch.block_tables)["state"][0]) == [0, slot1], step
        if step > 0:
            complete_decode("next", 200 + step)
    complete_decode("next", 210)


def make_prefill_state_config(
    drop_short_prompt_end_checkpoints: bool,
) -> SchedulerConfig:
    """A prefill server for a hybrid model (one History and one State group,
    2-token pages) with the in-forward State checkpoint under an 8-token chunk
    budget, so a prompt of 8 tokens or fewer has none."""
    cfg = SchedulerConfig()
    cfg.role = SchedulerConfig.Role.P
    cfg.decode_allocation_policy = SchedulerConfig.DecodeAllocationPolicy.Early
    cfg.prefix_granularity = 2
    cfg.max_scheduled_tokens = 8
    cfg.max_batch_size = 1
    cfg.num_device_pages = 65
    cfg.num_snapshot_pages = 1  # the P role never retracts
    cfg.disable_l2_cache = True
    cfg.state_checkpoint_in_forward = True
    cfg.drop_short_prompt_end_checkpoints = drop_short_prompt_end_checkpoints
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=65,
            attn_kind=AttnKind.Full,
            transfer_policy=CacheTransferPolicy.FullSuffix,
        ),
        CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=65,
            attn_kind=AttnKind.State,
            transfer_policy=CacheTransferPolicy.LatestSnapshot,
        ),
    ]
    return cfg


def prefill_and_hand_off(
    scheduler: Scheduler, request_id: str, tokens: list[int]
) -> int:
    """A prefill-server turn: the prompt's forwards, the hand-off to the decode
    peer and the peer's ACK, which finishes the request. Returns where the
    first forward resumed."""
    scheduler.submit_requests([make_spec(request_id, tokens)])
    scheduler.advance(ExecutionEvent().add_event(PD.BootstrappedEvent(request_id)))
    first_prefix = None
    scheduled = 0
    while scheduled < len(tokens):
        [batch] = [
            op for op in scheduler.next_execution_plan().forward if op.request_ids
        ]
        prefix = list(batch.extend_prefix_lens)[0]
        first_prefix = prefix if first_prefix is None else first_prefix
        scheduled = prefix + list(batch.input_lengths)[0]
        result = ForwardEvent.ExtendResult()
        result.request_id = request_id
        result.tokens = [100] if scheduled == len(tokens) else []
        scheduler.advance(ExecutionEvent().add_event(result))
    handoff = scheduler.next_execution_plan().remote_decode
    assert handoff is not None and list(handoff.request_ids) == [request_id]
    scheduler.advance(ExecutionEvent().add_event(PD.SucceededEvent(request_id)))
    return first_prefix


@pytest.mark.parametrize("drop", [False, True])
def test_a_prefill_server_can_drop_a_short_prompt_end(drop):
    """``drop_short_prompt_end_checkpoints``: the 5-token prompt ends inside
    its third page. Without the flag its finish publishes that end, where a
    longer prompt resumes; with it only the full pages, which hold no state,
    so the longer prompt starts over."""
    scheduler = Scheduler(make_prefill_state_config(drop))
    assert prefill_and_hand_off(scheduler, "short", [1, 2, 3, 4, 5]) == 0
    resumed = prefill_and_hand_off(scheduler, "longer", [1, 2, 3, 4, 5, 6, 7])
    assert resumed == (0 if drop else 5)


def test_a_page_aligned_short_prompt_end_keeps_its_state():
    scheduler = Scheduler(make_prefill_state_config(True))
    prefill_and_hand_off(scheduler, "aligned", [1, 2, 3, 4])
    assert prefill_and_hand_off(scheduler, "longer", [1, 2, 3, 4, 5, 6]) == 4


def test_dropping_short_prompt_ends_takes_the_prefill_role_and_the_in_forward_checkpoint():
    cfg = make_prefill_state_config(True)
    cfg.state_checkpoint_in_forward = False
    with pytest.raises(ValueError, match="requires state_checkpoint_in_forward"):
        Scheduler(cfg)
    cfg.state_checkpoint_in_forward = True
    cfg.role = SchedulerConfig.Role.Fused
    cfg.decode_allocation_policy = None
    with pytest.raises(ValueError, match="P role only"):
        Scheduler(cfg)


def test_prefill_chunk_yield_round_trips_and_is_validated():
    """The P-role knob is off by default, reaches the scheduler, and a value
    above half the chunk budget is refused at construction."""
    cfg = SchedulerConfig()
    assert cfg.prefill_chunk_yield_tokens == 0
    cfg.role = SchedulerConfig.Role.P
    cfg.decode_allocation_policy = SchedulerConfig.DecodeAllocationPolicy.Early
    cfg.prefix_granularity = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 64
    cfg.num_snapshot_pages = 1  # the P role never retracts
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="history",
            block_granularity=16,
            total_pages=64,
            transfer_policy=CacheTransferPolicy.FullSuffix,
            attn_kind=AttnKind.Full,
        )
    ]
    cfg.prefill_chunk_yield_tokens = 16
    assert cfg.prefill_chunk_yield_tokens == 16
    Scheduler(cfg)
    cfg.prefill_chunk_yield_tokens = 17
    with pytest.raises(ValueError, match="at most half of max_scheduled_tokens"):
        Scheduler(cfg)
    cfg.prefill_chunk_yield_tokens = 16
    cfg.role = SchedulerConfig.Role.D
    cfg.decode_allocation_policy = None
    with pytest.raises(ValueError, match="P role only"):
        Scheduler(cfg)
