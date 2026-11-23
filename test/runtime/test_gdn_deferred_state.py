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

"""GDN pending ownership through real verify, acceptance and graph replay."""

import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from test.ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=45,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="Deferred WY requires NVIDIA SM90+",
)

if not torch.cuda.is_available() or not current_platform().is_hopper_plus:
    pytest.skip("deferred GDN requires NVIDIA SM90+", allow_module_level=True)
pytest.importorskip("cutlass")

from test.runtime.test_gdn_cute_dsl_replay import make_case

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode


def case(defer):
    return make_case(
        tokens=4,
        head_dim=128,
        value_head_dim=128,
        vheads=4,
        state_dtype=torch.bfloat16,
        replay=True,
        tree=False,
        defer_recurrent_state=defer,
    )


def refresh(case, slots, *, pages=None, block_table=None):
    count = len(slots)
    reads = [5 if slot == 1 else 6 for slot in slots] if pages is None else pages
    rows = [[page, 5 if slot == 1 else 6] for page, slot in zip(reads, slots)]
    rows += [[0, 0]] * (2 - count)
    rows = torch.tensor(rows, device="cuda", dtype=torch.int32)
    if block_table is None:
        block_table = rows
    else:
        block_table.copy_(rows)
    case.backend.refresh_decode_metadata(
        2,
        count,
        torch.tensor([*slots, *([0] * (2 - count))], device="cuda", dtype=torch.int32),
        torch.full((2,), 16, device="cuda", dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        for_graph_replay=False,
        block_tables={"linear_attention": block_table},
    )


def run(case, slots, frontier, stopped, accepted):
    case.backend.prepare_pending_recurrent_state(slots, frontier, allow_defer=True)
    output = case.forward()
    case.backend.prepare_recurrent_state_acceptance(stopped, frontier, allow_defer=True)
    case.backend.commit_verified_state(accepted, accepted_path=None)
    return output


@pytest.mark.parametrize("graph", [False, True])
def test_repeated_steps_reorder_dropout_stop_and_slot_reuse(graph):
    regular, deferred = case(False), case(True)
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    frontiers = [torch.zeros(4, device="cuda", dtype=torch.int32) for _ in range(2)]
    stopped = torch.zeros(4, device="cuda", dtype=torch.bool)
    accepted = torch.tensor([4, 4], device="cuda", dtype=torch.int32)
    for value in (regular, deferred):
        refresh(value, [1, 2])
    tape = None
    graph_block_table = None
    if graph:
        for _ in range(2):
            for value, frontier in zip((regular, deferred), frontiers):
                refresh(value, [1, 2])
                run(value, slots, frontier, stopped, accepted)
                frontier.index_add_(0, slots.long(), accepted)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        tape = torch.cuda.CUDAGraph()
        # Acceptance reads the block table. Update the captured storage in place
        # as the executor does when a batch changes.
        graph_block_table = torch.empty((2, 2), device="cuda", dtype=torch.int32)
        refresh(deferred, [1, 2], block_table=graph_block_table)
        with torch.cuda.graph(tape, stream=stream):
            graphed_output = run(deferred, slots, frontiers[1], stopped, accepted)
        torch.cuda.current_stream().wait_stream(stream)
    schedule = [
        ([1, 2], [0, 4]),
        ([2, 1], [1, 2]),
        ([1], [3, 0]),
        ([2], [4, 0]),
        ([1, 2], [2, 1]),
        ([2, 1], [4, 4]),
    ]
    for step, (request_slots, lengths) in enumerate(schedule):
        slots.copy_(
            torch.tensor(
                [*request_slots, *([0] * (2 - len(request_slots)))],
                device="cuda",
                dtype=torch.int32,
            )
        )
        accepted.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        if step == len(schedule) - 1:
            stopped[1] = True
        # Vary each round's actual convolution inputs rather than rereading one payload.
        for name in ("mixed_qkv", "a", "b"):
            data = torch.randn_like(regular.inputs[name]) * 0.1
            regular.inputs[name].copy_(data)
            deferred.inputs[name].copy_(data)
        refresh(regular, request_slots)
        refresh(deferred, request_slots, block_table=graph_block_table)
        expected = run(regular, slots, frontiers[0], stopped, accepted)
        if tape is None:
            actual = run(deferred, slots, frontiers[1], stopped, accepted)
        else:
            tape.replay()
            actual = graphed_output
        torch.testing.assert_close(
            actual[:, : len(request_slots) * 4],
            expected[:, : len(request_slots) * 4],
            rtol=2e-2,
            atol=8e-4,
        )
        for frontier in frontiers:
            frontier.index_add_(0, slots.long(), accepted)
    deferred.backend.materialize_recurrent_state(None)
    torch.testing.assert_close(
        deferred.state[5:7], regular.state[5:7], rtol=2e-2, atol=1e-3
    )
    assert deferred.backend._pending_state.records[1, 1].item() == -1


def test_first_decode_commits_distinct_source_before_pending_and_masked_materialization():
    deferred = case(True)
    frontier = torch.zeros(4, device="cuda", dtype=torch.int32)
    stopped = torch.zeros(4, device="cuda", dtype=torch.bool)
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    refresh(deferred, [1, 2], pages=[2, 3])
    run(deferred, slots, frontier, stopped, accepted)
    pending = deferred.backend._pending_state
    assert not pending.valid.any().item()
    # A subsequent decode reads the live page in place and can defer replay.
    frontier.index_add_(0, slots.long(), accepted)
    refresh(deferred, [1, 2])
    run(deferred, slots, frontier, stopped, accepted)
    assert (pending.records[1:3, 1] >= 0).all().item()
    deferred.backend.materialize_recurrent_state(torch.tensor([1], device="cuda"))
    assert pending.records[1, 1].item() == -1
    assert pending.records[2, 1].item() >= 0
    deferred.backend.materialize_recurrent_state(None)
    assert not pending.valid.any().item()


def test_deferred_workspace_budget_matches_owned_storage():
    from dataclasses import replace
    from test.runtime.test_gdn_state_paging import _mamba_config_pair

    from transformers import PretrainedConfig

    from tokenspeed.runtime.layers.attention.backends.state.gdn_pending import (
        deferred_gdn_metadata_bytes,
    )
    from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.qwen35 import (
        QwenGDNRecipe,
    )
    from tokenspeed.runtime.utils.server_args import ServerArgs

    deferred = case(True)
    pending = deferred.backend._pending_state
    metadata = deferred_gdn_metadata_bytes(
        layers=1, capacity=2, slot_count=pending.records.shape[0]
    )
    assert pending.nbytes == pending.payload.nbytes + metadata
    config, spec = _mamba_config_pair(
        torch,
        heads=2,
        head_dim=128,
        spec_tokens=4,
        max_bs=2,
        device="cuda",
        replay_ssm=True,
        draft_tree=False,
    )
    linear = replace(config.component(LinearAttnConfig), num_v_heads=4)
    spec = replace(spec, cache_layer_types=("linear_attention", "full_attention"))
    config = replace(config, components=(spec, linear), max_resident_requests=8)
    assert pending.records.shape[0] == 10  # eight resident slots plus two reserved
    args = ServerArgs(
        model="unused",
        enable_deferred_gdn_state=True,
        mamba_ssm_dtype="bfloat16",
        speculative_num_draft_tokens=4,
    )
    recipe = QwenGDNRecipe(
        server_args=args,
        model_config=None,
        attn_config=config,
        draft_model_config=PretrainedConfig(num_attention_layers=1),
        draft_attn_config=config,
        cache_budget_bytes=1 << 28,
        probe_batch_rows=None,
        decode_input_tokens=4,
        overlap_schedule_depth=1,
    )
    assert recipe._replay_payload_bytes() == pending.nbytes + pending.parameters.nbytes


def test_graph_target_with_eager_acceptance_reenables_deferral_after_prefill():
    regular, deferred = case(False), case(True)
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    frontiers = [torch.zeros(10, device="cuda", dtype=torch.int32) for _ in range(2)]
    stopped = torch.zeros(10, device="cuda", dtype=torch.bool)
    accepted = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    for _ in range(2):
        for value, frontier in zip((regular, deferred), frontiers):
            refresh(value, [1, 2])
            run(value, slots, frontier, stopped, accepted)
            frontier.index_add_(0, slots.long(), accepted)
    refresh(deferred, [1, 2])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    tape = torch.cuda.CUDAGraph()
    # The production runner replays target/draft work, then accepts eagerly.
    with torch.cuda.graph(tape, stream=stream):
        deferred.backend.prepare_pending_recurrent_state(
            slots, frontiers[1], allow_defer=True
        )
        output = deferred.forward()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        deferred.backend.prepare_pending_recurrent_state(
            slots, frontiers[1], allow_defer=False
        )
        for value in (regular, deferred):
            refresh(value, [1, 2])
        expected = run(regular, slots, frontiers[0], stopped, accepted)
        tape.replay()
        deferred.backend.prepare_recurrent_state_acceptance(
            stopped, frontiers[1], allow_defer=True
        )
        deferred.backend.commit_verified_state(accepted, accepted_path=None)
        assert deferred.backend._pending_state.valid.any().item()
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=8e-4)
        for frontier in frontiers:
            frontier.index_add_(0, slots.long(), accepted)
    deferred.backend.materialize_recurrent_state(None)
    torch.testing.assert_close(
        deferred.state[5:7], regular.state[5:7], rtol=2e-2, atol=1e-3
    )


def test_device_retirement_batches_preserve_and_discard_without_kvstore():
    from dataclasses import dataclass

    from tokenspeed_engine.scheduler import (
        AttnKind,
        CacheGroupConfig,
        ExecutionEvent,
        ForwardEvent,
        RequestSpec,
        Scheduler,
        SchedulerConfig,
    )

    from tokenspeed.runtime.execution.device import DeviceHandle
    from tokenspeed.runtime.layers.attention.backends.state.gdn import GdnAttnBackend

    config = SchedulerConfig()
    config.prefix_granularity = 16
    config.max_scheduled_tokens = 16
    config.max_batch_size = 2
    config.max_resident_requests = 2
    config.num_device_pages = 64
    config.num_snapshot_pages = 1
    config.overlap_schedule_depth = 0
    config.decode_input_tokens = 4
    config.defer_recurrent_state = True
    config.disable_l2_cache = True
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full",
            block_granularity=16,
            total_pages=64,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(config)
    preserve = RequestSpec()
    preserve.request_id = "preserve"
    preserve.tokens = [1, 2, 3]
    preserve.max_new_tokens = 32
    discard = RequestSpec()
    discard.request_id = "discard"
    discard.tokens = [7, 8, 9]
    discard.max_new_tokens = 32
    requests = (preserve, discard)
    scheduler.submit_requests(list(requests))
    scheduler.next_execution_plan()
    prefill_results = ExecutionEvent()
    for request in requests:
        result = ForwardEvent.ExtendResult()
        result.request_id, result.tokens = request.request_id, [4]
        prefill_results.add_event(result)
    scheduler.advance(prefill_results)
    scheduler.next_execution_plan()
    decode_results = ExecutionEvent()
    for request in requests:
        result = ForwardEvent.ExtendResult()
        result.request_id, result.tokens = request.request_id, [5, 6, 7, 8]
        decode_results.add_event(result)
    finish = ForwardEvent.Finish()
    finish.request_id = preserve.request_id
    abort = ForwardEvent.Abort()
    abort.request_id = discard.request_id
    scheduler.advance(decode_results.add_event(finish).add_event(abort))
    plan = scheduler.next_execution_plan()
    retirement = plan.state_materialization
    assert retirement.materialize_slots == [1]
    assert retirement.discard_slots == [2]
    assert scheduler.active_lcm_blocks() == 0
    assert scheduler.has_state_retirements()

    value, reference = case(True), case(False)
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    frontier = torch.zeros(10, device="cuda", dtype=torch.int32)
    stopped = torch.zeros(10, device="cuda", dtype=torch.bool)
    accepted = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    before = value.state.clone()
    for current in (value, reference):
        refresh(current, [1, 2])
        run(current, slots, frontier, stopped, accepted)

    @dataclass
    class Executor:
        # Model weights/drafting are absent; all state and CUDA work is real.
        attn_backend: GdnAttnBackend
        execution_stream: torch.cuda.Stream
        default_stream: torch.cuda.Stream
        device: str = "cuda"
        device_module: object = torch.cuda

        def complete_pending_drafting(self):
            pass

    executor = Executor(value.backend, torch.cuda.Stream(), torch.cuda.current_stream())
    executor.execution_stream.wait_stream(executor.default_stream)
    handle = DeviceHandle(executor)
    previous_debug = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        handle.execute(plan, None, cache_ops=())
    finally:
        torch.cuda.set_sync_debug_mode(previous_debug)
    executor.default_stream.synchronize()
    torch.testing.assert_close(value.state[5], reference.state[5], rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(value.state[6], before[6], rtol=0, atol=0)
    assert not value.backend._pending_state.valid.any().item()
    (ack,) = handle.poll_cache_results()
    assert ack.op_id == retirement.op_id
    scheduler.advance(ExecutionEvent().add_event(ack))
    assert not scheduler.has_state_retirements()
    assert handle.poll_cache_results() == []


def test_pending_metadata_and_mode_switch_materialization_do_not_recompile():
    import importlib.util
    from pathlib import Path

    from tokenspeed_kernel.ops.attention.gdn._triton.pending import pending_metadata
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        _gdn_replay_commit_kernel,
        _gdn_replay_commit_ring_kernel,
        _gdn_selected_requests,
    )

    path = Path(__file__).resolve().parents[2] / "tokenspeed-kernel/test/utils.py"
    spec = importlib.util.spec_from_file_location("gdn_kernel_test_utils", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    value = case(True)
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    retiring_slots = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)[1:]
    frontier = torch.zeros(10, device="cuda", dtype=torch.int32)
    stopped = torch.zeros(10, device="cuda", dtype=torch.bool)
    accepted = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    # Startup capture prepares the target's metadata; acceptance is warmed
    # by workspace construction and must compile nothing on its first use.
    refresh(value, [1, 2])
    value.backend.prepare_pending_recurrent_state(
        torch.zeros_like(slots), frontier, allow_defer=True
    )
    with helper.assert_no_triton_compile(
        pending_metadata,
        _gdn_replay_commit_kernel,
        _gdn_replay_commit_ring_kernel,
        _gdn_selected_requests,
    ):
        refresh(value, [1, 2])
        run(value, slots, frontier, stopped, accepted)
        frontier.index_add_(0, slots.long(), accepted)
        value.backend.materialize_recurrent_state(retiring_slots[:1])
        value.backend.discard_recurrent_state(retiring_slots)
        value.backend.materialize_recurrent_state(retiring_slots[:0])
        refresh(value, [1])
        run(value, slots[:1], frontier, stopped, accepted[:1])
        value.backend.prepare_pending_recurrent_state(
            slots, frontier, allow_defer=False
        )


def test_discarded_replay_cannot_write_to_the_next_owner_of_the_same_slot_and_page():
    value, reference = case(True), case(False)
    slots = torch.tensor([1, 0], device="cuda", dtype=torch.int64)
    frontier = torch.zeros(10, device="cuda", dtype=torch.int32)
    stopped = torch.zeros(10, device="cuda", dtype=torch.bool)
    accepted = torch.tensor([4, 0], device="cuda", dtype=torch.int32)
    refresh(value, [1])
    run(value, slots, frontier, stopped, accepted)
    state_before_discard = value.state[5].clone()
    value.backend.discard_recurrent_state(slots[:1])
    value.backend.materialize_recurrent_state(slots[:1])
    torch.testing.assert_close(value.state[5], state_before_discard, rtol=0, atol=0)

    # The new owner starts at the same frontier on the same page. Only the
    # discarded replay would distinguish it from a fresh reference forward.
    frontier[1] = 4
    value.state[5].copy_(reference.state[5])
    conv = value.pool.get_component(0, "conv_state")
    conv[5].copy_(reference.pool.get_component(0, "conv_state")[5])
    refresh(value, [1])
    refresh(reference, [1])
    actual = run(value, slots, frontier, stopped, accepted)
    expected = run(reference, slots, frontier, stopped, accepted)
    torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=2e-2, atol=8e-4)
