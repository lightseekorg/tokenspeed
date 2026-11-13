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

"""Distributed adaptive verification agreement and CED count exchange."""

import os
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import DeepseekV41DSpark
from tokenspeed.runtime.execution.drafter.verify_budget import VerifyBudgetPolicy
from tokenspeed.runtime.execution.drafter.verify_calibration import (
    VerifyCalibration,
)
from tokenspeed.runtime.execution.drafter.verify_feedback import VerifyFeedback
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout


class DraftModel(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping


def _make_drafter(mapping):
    drafter = object.__new__(DeepseekV41DSpark)
    drafter.draft_model = DraftModel(mapping)
    drafter.tp_group = mapping.attn.tp_group
    drafter.verify_feedback = VerifyFeedback(draft_width=5)
    drafter._pending_verify_feedback = []
    drafter._verify_cache_initialized = True
    drafter._verify_cache = None
    drafter._cached_verify_profiles = ()
    drafter._verify_calibrations = ()
    drafter._verify_policy_ms = None
    drafter.device = "cuda"
    drafter.spec_num_tokens = 6
    drafter.block_size = 5
    drafter.adaptive_min_batch_size = 4
    drafter.verify_budget_policy = VerifyBudgetPolicy(True, 4, 6, 0.02)
    from tokenspeed.runtime.execution.input_buffer import InputBuffers

    drafter.confidence_scores = torch.full((4, 5), float("inf"), device="cuda")
    drafter.input_buffers = InputBuffers.__new__(InputBuffers)
    drafter.input_buffers.req_pool_indices_buf = torch.arange(4, device="cuda")
    drafter.verify_graph_token_step = 4
    drafter._verify_rows = torch.empty(24, dtype=torch.int64, device="cuda")
    drafter._verify_indptr = torch.empty(5, dtype=torch.int64, device="cuda")
    drafter._verify_layout_counts = None
    drafter._verify_budget_graphs = {}
    drafter._verify_budget_buffers = {}
    return drafter


def _serving_budget(drafter, world):
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=4,
        num_extends=0,
        input_num_tokens=24,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 4, 6),
        global_bs=[4] * world,
        global_num_decodes=[4] * world,
    )
    return drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=4)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires torchrun with multiple GPUs",
)
@torch.inference_mode()
def test_dp_budget_agreement_and_native_fallback():
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    mapping = Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=1,
        attn_dp_size=world,
        moe_ep_size=world,
    )
    process_group_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=120,
        device_id=torch.device("cuda", local_rank),
    )
    process_group_manager.init_process_group(tuple(range(world)), backend="nccl")
    drafter = _make_drafter(mapping)
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    drafter.capture_round_graphs(2, stream=stream, pool=pool)
    assert not drafter._verify_budget_graphs
    drafter.capture_round_graphs(4, stream=stream, pool=pool)
    graphs = dict(drafter._verify_budget_graphs)
    drafter.capture_round_graphs(4, stream=stream, pool=pool)
    assert drafter._verify_budget_graphs == graphs

    profile = VerifyCalibration(4, 1, 16384, 6, 100.0, ((12, 1.0),), 8)
    drafter.install_verify_calibration((profile,), policy_ms=0.0)
    ids, slots = [f"{rank}-{i}" for i in range(4)], list(range(4))
    drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    local_scores = [[float((i + rank) % 4 - 2)] * 5 for i in range(4)]
    # One rank's missing feedback forces all ranks to retain native execution.
    if rank != world - 1:
        drafter.verify_feedback.publish(
            drafter.verify_feedback.snapshot(),
            local_scores,
            context_lengths=[120 + rank] * 4,
        )
    for _ in range(2):
        drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    assert _serving_budget(drafter, world) == 24
    drafter.verify_feedback.publish(
        drafter.verify_feedback.snapshot(),
        local_scores,
        context_lengths=[120 + rank] * 4,
    )
    for _ in range(2):
        drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    assert _serving_budget(drafter, world) == 12
    assert drafter._verify_indptr[-1].item() == 12
    assert drafter._verify_budget_buffers[4].valid.item()
    # Repeated allocations must retain fresh per-rank scores and layouts.
    # Snapshot results on the same stream without synchronizing each iteration.
    pending_layouts = []
    for iteration in range(24):
        favored = (rank + iteration) % 4
        changing_scores = [[20.0 if i == favored else -20.0] * 5 for i in range(4)]
        drafter.verify_feedback.publish(
            drafter.verify_feedback.snapshot(),
            changing_scores,
            context_lengths=[120 + rank] * 4,
        )
        for _ in range(2):
            drafter.verify_feedback.prepare(ids, slots, num_extends=0)
        assert _serving_budget(drafter, world) == 12
        pending_layouts.append(
            (
                favored,
                drafter._verify_indptr.clone(),
                drafter._verify_budget_buffers[4].scores.clone(),
                changing_scores,
            )
        )
    for favored, indptr, scores, expected in pending_layouts:
        counts = indptr.diff().cpu().tolist()
        assert counts[favored] == 6 and sum(counts) == 12
        torch.testing.assert_close(scores.cpu(), torch.tensor(expected), rtol=0, atol=0)
    # The allocation graph reads current GPU confidence without recapture;
    # its budget still comes from the completed host feedback.
    for offset in range(3):
        favored = (rank + offset) % 4
        drafter.confidence_scores.fill_(-20)
        drafter.confidence_scores[favored].fill_(20)
        assert _serving_budget(drafter, world) == 12
        counts = drafter._verify_indptr.diff().tolist()
        assert counts[favored] == 6 and sum(counts) == 12
    drafter.confidence_scores.fill_(float("inf"))
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=4,
        num_extends=0,
        input_num_tokens=24,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 4, 6),
        global_bs=[4] * world,
        global_num_decodes=[4] * world,
    )
    for peer_bs in (0, 3):
        ctx.global_bs[-1] = peer_bs
        ctx.global_num_decodes[-1] = peer_bs
        assert (
            drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=4)
            == 24
        )
    # Idle peers need no feedback. A shortened active batch still has stale
    # four-request feedback here and must fall back.
    for peer_bs in (0, 3):
        ctx.global_bs[-1] = ctx.global_num_decodes[-1] = peer_bs
        actual_bs = peer_bs if rank == world - 1 else 4
        assert drafter.prepare_decode_graph_token_layout(
            ctx, actual_bs=actual_bs, padded_bs=4
        ) == (12 if peer_bs == 0 else 24)
    # Equal graph padding and batches below the local threshold stay native.
    for actual_bs, padded_bs in ((3, 4), (2, 2)):
        ctx.global_bs = [actual_bs] * world
        ctx.global_num_decodes = [actual_bs] * world
        assert (
            drafter.prepare_decode_graph_token_layout(
                ctx, actual_bs=actual_bs, padded_bs=padded_bs
            )
            == padded_bs * 6
        )
    ctx.global_bs = ctx.global_num_decodes = [4] * world
    assert (
        drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=4) == 12
    )
    # Startup owns profiling: an uncovered serving shape must remain native,
    # without starting online collection or allocating step timing events.
    profiles = drafter._verify_calibrations
    drafter._verify_calibrations = ()
    for _ in range(3):
        assert (
            drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=4)
            == 24
        )
    drafter._verify_calibrations = profiles
    # CED prefill may narrow each rank to a different count, including zero.
    from tokenspeed.runtime.models.deepseek_v41 import _exchange_decoder_counts

    process_group_manager.init_process_group(
        tuple(range(world)), backend="gloo", name="v41_decoder_counts"
    )
    assert _exchange_decoder_counts(rank * 3, world) == [i * 3 for i in range(world)]
    assert _exchange_decoder_counts(0, world) == [0] * world
    torch.distributed.barrier()


@contextmanager
def _strict_serving_sync():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        # Test-side tensor inspection may synchronize; serving calls may not.
        torch.cuda.set_sync_debug_mode(previous)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires torchrun with multiple GPUs",
)
@pytest.mark.parametrize("topology", ["dp", "tp", "mixed"])
@torch.inference_mode()
def test_budget_without_forward_thread_sync(topology):
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if topology == "mixed" and (world < 4 or world % 2):
        pytest.skip("mixed TP/DP requires an even world of at least four ranks")
    tp = world if topology == "tp" else (2 if topology == "mixed" else 1)
    torch.cuda.set_device(local_rank)
    mapping = Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=tp,
        attn_dp_size=world // tp,
        moe_ep_size=world,
    )
    process_group_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=120,
        device_id=torch.device("cuda", local_rank),
    )
    process_group_manager.init_process_group(tuple(range(world)), backend="nccl")
    drafter = _make_drafter(mapping)
    drafter.capture_round_graphs(
        4, stream=torch.cuda.Stream(), pool=torch.cuda.graph_pool_handle()
    )
    drafter.install_verify_calibration(
        (VerifyCalibration(4, 1, 16384, 6, 100.0, ((12, 1.0),), 8),),
        policy_ms=0.0,
    )
    ids = [f"{mapping.attn.dp_rank}-{i}" for i in range(4)]
    slots = list(range(4))
    drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    favored = mapping.attn.dp_rank % 4
    scores = [[20.0 if i == favored else -20.0] * 5 for i in range(4)]
    # Missing non-root feedback is allowed only for authoritative TP broadcast.
    if rank != world - 1:
        drafter.verify_feedback.publish(
            drafter.verify_feedback.snapshot(), scores, context_lengths=[120] * 4
        )
    for _ in range(2):
        drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    with _strict_serving_sync():
        selected = _serving_budget(drafter, world)
    assert selected == (12 if topology == "tp" else 24)

    drafter.verify_feedback.publish(
        drafter.verify_feedback.snapshot(), scores, context_lengths=[120] * 4
    )
    for _ in range(2):
        drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    with _strict_serving_sync():
        selected = _serving_budget(drafter, world)
    assert selected == 12
    counts = drafter._verify_indptr.diff().cpu().tolist()
    assert counts[favored] == 6 and sum(counts) == 12
    torch.testing.assert_close(
        drafter._verify_budget_buffers[4].scores.cpu(),
        torch.tensor(scores),
        rtol=0,
        atol=0,
    )
    # Startup preparation broadcasts only inside each TP group.
    with _strict_serving_sync():
        selected = drafter._prepare_calibrated_budget(actual_bs=4, padded_bs=4)
    assert selected == 12
    torch.testing.assert_close(
        drafter._verify_budget_buffers[4].scores.cpu(),
        torch.tensor(scores),
        rtol=0,
        atol=0,
    )
    torch.distributed.barrier()
