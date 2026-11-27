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

"""Independent DP graph buckets and adaptive verification share one scope."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import (
    DeepseekV41DSpark,
    _VerifyBudgetBuffers,
)
from tokenspeed.runtime.execution.drafter.verify_budget import VerifyBudgetPolicy
from tokenspeed.runtime.execution.drafter.verify_calibration import VerifyCalibration
from tokenspeed.runtime.execution.drafter.verify_feedback import VerifyFeedback
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout


def _drafter(rank: int, *, coordinated: bool) -> DeepseekV41DSpark:
    mapping = Mapping(
        rank=rank,
        world_size=2,
        attn_tp_size=1,
        attn_dp_size=2,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=2 if coordinated else 1,
    )
    # Model weights are unnecessary for the real host policy and GPU allocator.
    drafter = object.__new__(DeepseekV41DSpark)
    drafter.draft_model = SimpleNamespace(mapping=mapping)
    drafter.tp_group = mapping.attn.tp_group
    drafter.dp_local_graph = False
    return drafter


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("coordinated", [False, True])
def test_verification_scope_follows_forward_coordination(rank, coordinated):
    drafter = _drafter(rank, coordinated=coordinated)
    assert drafter._verification_group == ((0, 1) if coordinated else (rank,))
    assert drafter._verification_rank == (rank if coordinated else 0)


@pytest.fixture
def independent_budget(request):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA graph allocation")
    drafter = _drafter(request.param, coordinated=False)
    drafter.device = "cuda"
    drafter.spec_num_tokens = 4
    drafter.block_size = 3
    drafter.adaptive_min_batch_size = 32
    drafter.verify_budget_policy = VerifyBudgetPolicy(True, 32, 4, 0.02)
    drafter.verify_feedback = VerifyFeedback(draft_width=3)
    drafter._pending_verify_feedback = []
    drafter._verify_cache_initialized = True
    drafter._verify_cache = None
    drafter._cached_verify_profiles = ()
    drafter._verify_calibrations = ()
    drafter._verify_policy_ms = None
    drafter._verify_budget_counts = {}
    drafter._verify_budget_context_bounds = None
    drafter._verify_budget_buffers = {}
    drafter._verify_budget_graphs = {}
    drafter.confidence_scores = torch.zeros((40, 3), device="cuda")
    drafter.verify_graph_token_step = 40
    drafter._verify_rows = torch.empty(160, dtype=torch.int64, device="cuda")
    drafter._verify_indptr = torch.empty(41, dtype=torch.int64, device="cuda")
    drafter._verify_layout_counts = None
    drafter.input_buffers = InputBuffers(40, 160, 0, device="cuda")
    drafter.input_buffers.req_pool_indices_buf.copy_(torch.arange(40, device="cuda"))
    drafter.input_buffers.seq_lens_buf.fill_(120)
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    for bs in (32, 40):
        drafter.capture_round_graphs(bs, stream=stream, pool=pool)
    drafter.install_verify_calibration(
        tuple(
            VerifyCalibration(bs, 100, 200, 4, 10.0, ((bs * 2, 1.0),), 8)
            for bs in (32, 40)
        ),
        policy_ms=0.0,
    )
    yield drafter
    torch.cuda.synchronize()


def _context(drafter, bs: int, *, metadata: str) -> ForwardContext:
    rank = drafter.draft_model.mapping.rank
    batches = [32, 33]
    batches[rank] = bs
    decodes = batches.copy()
    if metadata == "mixed":
        batches[1 - rank] = 1000
        decodes[1 - rank] = 0
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=bs * 4,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, bs, 4),
        global_num_tokens=None if metadata == "absent" else [n * 4 for n in batches],
        global_bs=None if metadata == "absent" else batches,
        global_num_decodes=None if metadata == "absent" else decodes,
        all_decode_or_idle=metadata == "decode",
    )


@pytest.mark.parametrize("independent_budget", [0, 1], indirect=True)
def test_independent_graph_and_budget_ignore_peer_batches(independent_budget):
    drafter = independent_budget
    runner = object.__new__(ForwardStepRunner)
    runner.disable = False
    runner.dp_size = 2
    runner.requires_forward_coordination = (
        drafter.draft_model.mapping.requires_forward_coordination
    )
    runner.dp_local_graph = False
    runner.disable_padding = False
    runner.capture_bs = [32, 40]
    runner.max_capture_bs = 40
    runner.max_tokens_per_req = 4
    # Uneven replicas select different graph buckets. The padded replica keeps
    # native width; an exact local batch may pack independently of its peers.
    for bs in (32 + drafter.draft_model.mapping.rank, 40):
        for metadata in ("absent", "decode", "mixed"):
            ctx = _context(drafter, bs, metadata=metadata)
            assert runner.can_run(bs, ctx)
            padded = runner.padded_bs(bs, ctx)
            assert padded == (32 if bs == 32 else 40)
            drafter.verify_feedback = VerifyFeedback(draft_width=3)
            slots = list(range(bs))
            drafter.verify_feedback.prepare(slots, slots, num_extends=0)
            drafter.verify_feedback.publish(
                drafter.verify_feedback.snapshot(),
                [[0.0] * 3] * bs,
                context_lengths=[120] * bs,
            )
            for _ in range(2):
                drafter.verify_feedback.prepare(slots, slots, num_extends=0)
            selected = drafter.prepare_decode_graph_token_layout(
                ctx, actual_bs=bs, padded_bs=padded
            )
            assert selected == (bs * 2 if bs == padded else padded * 4)
            if selected != padded * 4:
                assert drafter._verify_indptr[padded].item() == selected
                assert drafter._verify_budget_buffers[padded].valid.item()


@pytest.mark.parametrize("independent_budget", [0, 1], indirect=True)
def test_independent_feedback_uses_local_admission(independent_budget):
    drafter = independent_budget
    for bs, graph_bs in ((16, 32), (33, 40)):
        ctx = _context(drafter, bs, metadata="mixed")
        drafter.finish_round(ctx, graph_bs=graph_bs)
        assert not drafter._pending_verify_feedback
    ctx = _context(drafter, 32, metadata="mixed")
    slots = list(range(32))
    drafter.verify_feedback.prepare(slots, slots, num_extends=0)
    drafter.finish_round(ctx, graph_bs=32)
    torch.cuda.synchronize()
    drafter._poll_verify_feedback()
    for _ in range(2):
        drafter.verify_feedback.prepare(slots, slots, num_extends=0)
    assert drafter.verify_feedback.scores() == ((0.0,) * 3,) * 32


def _budget_round(rank: int, rendezvous: str, coordinated: bool) -> None:
    drafter = _drafter(rank, coordinated=coordinated)
    process_group_manager.init_distributed(
        drafter.draft_model.mapping,
        distributed_init_method=rendezvous,
        backend="gloo",
        timeout=30,
    )
    process_group_manager.init_process_group(
        (0, 1), backend="gloo", name="dsv41_verify_metadata"
    )
    try:
        # Cache control may keep a peer group even when execution is local.
        # Different local buckets must never enter a verification collective.
        padded = 40 if coordinated or rank == 1 else 32
        packet_size = 3 + padded * 3
        drafter.spec_num_tokens = 4
        drafter.block_size = 3
        drafter.adaptive_min_batch_size = 32
        drafter.verify_budget_policy = VerifyBudgetPolicy(True, 32, 4, 0.02)
        drafter.verify_feedback = VerifyFeedback(draft_width=3)
        drafter._pending_verify_feedback = []
        drafter._verify_calibrations = ()
        drafter._verify_policy_ms = 0.0
        drafter._verify_budget_counts = {}
        drafter._verify_budget_context_bounds = None
        drafter._verify_budget_buffers = {
            padded: _VerifyBudgetBuffers(
                scores=torch.empty(padded, 3),
                counts=torch.empty(padded, dtype=torch.int64),
                valid=torch.empty((), dtype=torch.bool),
                packet=torch.empty(packet_size),
                peers=torch.empty(2, packet_size),
            )
        }
        ctx = _context(drafter, 32 + rank, metadata="decode")
        assert (
            drafter.prepare_decode_graph_token_layout(
                ctx, actual_bs=32 + rank, padded_bs=padded
            )
            == padded * 4
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned packets")
@pytest.mark.parametrize("coordinated", [False, True])
def test_dp_budget_collectives_follow_forward_contract(tmp_path, coordinated):
    torch.multiprocessing.spawn(
        _budget_round,
        args=(f"file://{tmp_path / 'verify-budget'}", coordinated),
        nprocs=2,
    )
