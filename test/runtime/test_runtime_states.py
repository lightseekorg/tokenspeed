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

import os
import sys

import pytest
import tokenspeed_scheduler as ts
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.engine.scheduler_utils import make_config
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import StopCriteria
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [[], [2], [1, 3]])
def test_reset_states_preserves_unselected_rows(device, rows):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    states = RuntimeStates(4, 32, 32, 2, device)
    states.valid_cache_lengths.fill_(7)
    states.spec_candidates_ready.fill_(True)
    states.future_input_map.fill_(11)
    indices = torch.tensor(rows, dtype=torch.int64, device=device)
    lengths = torch.full((len(rows),), 100, dtype=torch.int32, device=device)

    states.reset_states(indices, lengths)

    assert states.valid_cache_lengths.tolist() == [
        100 if i in rows else 7 for i in range(5)
    ]
    assert states.spec_candidates_ready.tolist() == [i not in rows for i in range(5)]
    assert states.future_input_map.tolist() == [[11, 11]] * 5


def test_draft_probs_start_as_sentinel_with_a_zero_bonus_slot():
    states = RuntimeStates(4, 32, 32, 3, "cpu")
    assert states.draft_probs is None
    # reset_draft_probs is a no-op until the buffer exists.
    states.reset_draft_probs(torch.tensor([1], dtype=torch.int64))

    states.init_draft_probs(spec_num_tokens=3, reject_threshold=2.0)
    probs = states.draft_probs
    assert probs.shape == (5, 3, 32) and probs.dtype == torch.float32
    assert states.draft_probs_sentinel == 3.0
    # Step slots 0..N-2 hold the sentinel; the bonus slot is never read.
    assert torch.equal(probs[:, :-1], torch.full((5, 2, 32), 3.0))
    assert torch.equal(probs[:, -1], torch.zeros(5, 32))


def test_draft_tree_parents_are_declared_at_construction():
    # A PD decode destination writes its bootstrap token through
    # write_decode_input_ids, which reads future_parent_map, before any reset
    # or drafter has touched the slot: the fields exist from construction.
    states = RuntimeStates(4, 32, 32, 3, "cpu")
    assert states.chain_parents is None
    assert states.future_parent_map is None

    states.init_draft_trees(num_nodes=3)
    assert torch.equal(
        states.chain_parents, torch.tensor([-1, 0, 1], dtype=torch.int32)
    )
    assert states.future_parent_map.shape == (5, 3)
    assert torch.equal(states.future_parent_map, states.chain_parents.repeat(5, 1))


def _recorded_states() -> RuntimeStates:
    states = RuntimeStates(4, 32, 32, 3, "cpu")
    states.init_draft_probs(spec_num_tokens=3, reject_threshold=2.0)
    # Pretend the drafter recorded every slot's two steps.
    states.draft_probs[:, :-1] = 0.25
    return states


def test_reset_states_restores_the_sentinel_rows_only():
    states = _recorded_states()
    probs = states.draft_probs

    states.reset_states(
        torch.tensor([1, 3], dtype=torch.int64),
        torch.tensor([10, 20], dtype=torch.int32),
    )
    for slot in range(5):
        expected = 3.0 if slot in (1, 3) else 0.25
        assert torch.equal(probs[slot, :-1], torch.full((2, 32), expected))
    assert torch.equal(probs[:, -1], torch.zeros(5, 32))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the candidate upload pins host memory"
)
def test_remote_landing_relies_on_the_receive_time_reset():
    # A PD decode destination seeds the landing slot's cache length through
    # reset_states when it receives the prompt (the one sentinel reset site);
    # the later candidate landing carries no draft distribution and leaves
    # draft_probs alone.
    states = _recorded_states()
    probs = states.draft_probs
    states.reset_states(
        torch.tensor([2], dtype=torch.int64), torch.tensor([40], dtype=torch.int32)
    )
    before = probs.clone()
    states.write_remote_spec_candidate_ids(2, [5, 6, 7])
    assert states.future_input_map[2].tolist() == [5, 6, 7]
    assert bool(states.spec_candidates_ready[2])
    assert torch.equal(probs, before)
    for slot in range(5):
        expected = 3.0 if slot == 2 else 0.25
        assert torch.equal(probs[slot, :-1], torch.full((2, 32), expected))
    assert torch.equal(probs[:, -1], torch.zeros(5, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reset_states_does_not_synchronize_cuda():
    states = RuntimeStates(4, 32, 32, 2, "cuda")
    indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    lengths = torch.tensor([128, 256], dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        states.reset_states(indices, lengths)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.cuda.synchronize()
    assert states.valid_cache_lengths.tolist() == [0, 128, 0, 256, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_remote_candidate_landing_does_not_synchronize_cuda():
    # A PD decode destination lands each remote prefill's candidates on the
    # forward thread, while a step may be in flight.
    states = RuntimeStates(4, 32, 32, 3, "cuda")
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        states.write_remote_spec_candidate_ids(2, [5, 6, 7])
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.cuda.synchronize()
    assert states.future_input_map[2].tolist() == [5, 6, 7]
    assert states.spec_candidates_ready.tolist() == [i == 2 for i in range(5)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reset_states_graph_replay_uses_current_inputs():
    states = RuntimeStates(4, 32, 32, 2, "cuda")
    indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    lengths = torch.tensor([128, 256], dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        states.reset_states(indices, lengths)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        states.reset_states(indices, lengths)

    for offset in (0, 1):
        indices.copy_(torch.tensor([1 + offset, 3 + offset], device="cuda"))
        lengths.fill_(512 + offset)
        states.valid_cache_lengths.zero_()
        states.spec_candidates_ready.fill_(True)
        graph.replay()
        selected = (1 + offset, 3 + offset)
        assert states.valid_cache_lengths.tolist() == [
            512 + offset if i in selected else 0 for i in range(5)
        ]
        assert states.spec_candidates_ready.tolist() == [
            i not in selected for i in range(5)
        ]


# A history group between two state groups: the bridge packs the state groups last.
_GROUP_KINDS = (("conv_a", "state"), ("history", "full_history"), ("conv_b", "state"))
_STATE_GROUPS = ("conv_a", "conv_b")
_PAGES = 17


def _scheduled_batch():
    """Two prompts' first forward, as the scheduler plans it."""
    kinds = {"state": ts.AttnKind.State, "full_history": ts.AttnKind.Full}
    config = make_config(
        num_device_pages=_PAGES,
        max_scheduled_tokens=64,
        max_batch_size=2,
        prefix_granularity=4,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=1,
        host_group_packing=(),
        max_retracted_requests=0,
        host_write_policy="write_through",
        prefill_decode_interval=0,
        decode_allocation_policy=None,
        storage_config=None,
        enable_kv_cache_events=False,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        disable_prefix_cache=True,
        cache_groups=[
            ts.CacheGroupConfig(
                group_id=group_id,
                block_granularity=0 if kind == "state" else 4,
                total_pages=_PAGES,
                attn_kind=kinds[kind],
            )
            for group_id, kind in _GROUP_KINDS
        ],
        enable_mixed_prefill_decode=False,
        prefix_replay_tokens=0,
    )
    scheduler = ts.Scheduler(config)
    for i in range(2):
        request = ts.RequestSpec()
        request.request_id = f"r{i}"
        request.tokens = list(range(6 + i))
        request.max_new_tokens = 2
        scheduler.submit_requests([request])
    (batch,) = scheduler.next_execution_plan().forward
    assert len(batch.request_ids) == 2
    return batch


def _contract() -> CacheRuntimeContract:
    specs = tuple(
        CacheGroupSpec(
            group_id=group_id,
            attn_kind=kind,
            rows_per_page=None if kind == "state" else 4,
            entry_stride_tokens=None if kind == "state" else 1,
            replayable=False,
        )
        for group_id, kind in _GROUP_KINDS
    )
    return CacheRuntimeContract(
        prefix_granularity=4,
        num_lcm_blocks=_PAGES - 1,
        token_capacity=(_PAGES - 1) * 4,
        group_specs=specs,
        group_page_counts={spec.group_id: _PAGES for spec in specs},
        group_packing={spec.group_id: 1 for spec in specs},
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("layout", ["bridge", "separate"])
def test_masked_state_tables_pads_only_the_stopped_rows_of_state_groups(device, layout):
    """A stopped request's rows read -1 in every state group; live rows and
    the history group keep their blocks, whether the tables are the bridge's
    one upload or tensors of their own."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _scheduled_batch()
    expected = {
        group_id: torch.from_numpy(array.copy())
        for group_id, array in batch.block_tables_arrays().items()
    }
    metadata = CacheBatchMetadata.from_forward_op(
        batch,
        device=device,
        contract=_contract(),
        num_requests=2,
        validate_page_ids=True,
        host_tables=False,
    )
    tables = dict(metadata.tables(active_forward_op=batch))
    assert tuple(tables) == tuple(group_id for group_id, _ in _GROUP_KINDS)
    if layout == "separate":
        tables = {group_id: table.clone() for group_id, table in tables.items()}
    else:
        # The one-launch padding needs the bridge to pack the state groups end to end.
        conv_a, conv_b = (tables[group_id] for group_id in _STATE_GROUPS)
        assert (
            conv_b.untyped_storage().data_ptr() == conv_a.untyped_storage().data_ptr()
        )
        assert conv_b.storage_offset() == conv_a.storage_offset() + conv_a.numel()
    for group_id, table in tables.items():
        assert torch.equal(table.cpu(), expected[group_id])

    states = RuntimeStates(8, 32, 32, 1, device)
    slots = torch.tensor(batch.request_pool_indices, dtype=torch.int64, device=device)
    states.stopped[slots[1]] = True
    masked = states.masked_state_tables(tables, slots, _STATE_GROUPS)

    for group_id in _STATE_GROUPS:
        assert masked[group_id] is tables[group_id]
        assert torch.equal(masked[group_id][0].cpu(), expected[group_id][0])
        assert (masked[group_id][1] == -1).all()
    assert torch.equal(masked["history"].cpu(), expected["history"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "old_criteria,new_criteria,valid_length,token,expected_length,expected_stopped",
    [
        (StopCriteria((), 9), StopCriteria((), 53), 21, 5, 1, False),
        (StopCriteria((), 53), StopCriteria((), 9), 8, 5, 0, True),
        (StopCriteria((5,), 53), StopCriteria((6,), 53), 21, 5, 1, False),
        (StopCriteria((6,), 53), StopCriteria((5,), 53), 21, 5, 1, True),
    ],
    ids=["longer-budget", "shorter-budget", "removed-stop-token", "added-stop-token"],
)
def test_reused_rid_updates_stop_criteria(
    device,
    old_criteria,
    new_criteria,
    valid_length,
    token,
    expected_length,
    expected_stopped,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    states = RuntimeStates(2, 32, 32, 1, device)
    states.bind_stop_criteria(["reused", "other"], [0, 1], [old_criteria, old_criteria])
    slots = torch.tensor([0], dtype=torch.int64, device=device)
    lengths = torch.tensor([valid_length], dtype=torch.int32, device=device)
    # A later request may reuse both the public RID and its retired slot.
    states.stopped[0] = True
    states.reset_states(slots, lengths)
    states.bind_stop_criteria(["reused"], [0], [new_criteria])
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    states.apply_stop_conditions(
        slots,
        torch.tensor([token], dtype=torch.int32, device=device),
        accepted,
        torch.ones(1, dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        num_extends=0,
        num_prefill_outputs=0,
        decode_width=1,
        padding_index=2,
    )
    assert accepted.tolist() == [expected_length]
    assert states.stopped[0].item() == expected_stopped
    assert states.token_limits[:2].tolist() == [
        new_criteria.token_limit,
        old_criteria.token_limit,
    ]
    assert states.stop_token_ids[
        1, : len(old_criteria.stop_token_ids)
    ].tolist() == list(old_criteria.stop_token_ids)
    # Binding an unchanged criterion must not clear the device's stop decision
    # while a later overlapped forward is still draining.
    states.bind_stop_criteria(["reused"], [0], [new_criteria])
    assert states.stopped[0].item() == expected_stopped
