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

"""TP8 GB300 regression for a K3 HT tail after an eager attention break.

Launch on two four-GPU GB300 Slurm segments with::

    REQUIRE_MNNVL_CUTEDSL=1 torchrun --nnodes=2 --nproc-per-node=4 ... \
      -m pytest -q tokenspeed-kernel/test/nvidia/thirdparty/\
test_k3_mnnvl_breakable_prefill_graph.py

The exact failure this guards is structural.  A full-graph ``capture_mode``
activates K3's attention ``StreamFork``.  Its eager attention call then ends a
``BreakableCapture`` segment before the auxiliary branch rejoins the origin
stream, invalidating the event/capture.  A prefill-specific phase must instead
leave both generic graph flags clear while still allowing the HT candidate to
be captured in the segment after attention.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
    MNNVLCuteDSLHTFinalizeAllReduceRMSNorm,
    MNNVLCuteDSLHTFinalizeTuning,
    mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported,
)

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    break_point,
)
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
    get_is_prefill_graph_phase,
    prefill_graph_phase,
)
from tokenspeed.runtime.utils.cuda_stream import StreamFork

HIDDEN = 3584
TOP_K = 16
TP_SIZE = 8
TOKENS = 2048
RMS_EPS = 1e-5


def _unavailable(message: str) -> None:
    if os.environ.get("REQUIRE_MNNVL_CUTEDSL") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


@pytest.fixture(scope="module")
def distributed_group():
    if int(os.environ.get("WORLD_SIZE", "1")) != TP_SIZE:
        _unavailable("breakable-prefill HT validation requires exactly eight ranks")
    if not torch.cuda.is_available():
        _unavailable("breakable-prefill HT validation requires CUDA")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] < 10:
        _unavailable("breakable-prefill HT validation requires Blackwell")

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        if dist.get_world_size() != TP_SIZE:
            pytest.fail("initialized process group is not TP8", pytrace=False)
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.barrier()
            dist.destroy_process_group()


def _collective_support_vote(local: bool, group: dist.ProcessGroup) -> bool:
    vote = torch.tensor(
        [int(local)],
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    return bool(vote.item())


def _workspace(group: dist.ProcessGroup):
    local_support = mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported(
        group=group,
        tp_size=TP_SIZE,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        dtype=torch.bfloat16,
        candidate_min_tokens=1280,
        candidate_max_tokens=TOKENS,
    )
    if not _collective_support_vote(local_support, group):
        _unavailable("rank-agreed native-H3584 HT support is unavailable")

    return MNNVLCuteDSLHTFinalizeAllReduceRMSNorm.initialize(
        group=group,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        candidate_min_tokens=1280,
        candidate_max_tokens=TOKENS,
        tuning_routes=(
            MNNVLCuteDSLHTFinalizeTuning(
                max_tokens=TOKENS,
                persistent_ctas=None,
                consumer_threads=448,
                vectors_per_thread=1,
                stages=7,
                reduction_warps=2,
                reduction_cta_groups=None,
                rms_token_groups=2,
                rms_pipeline_stages=3,
                rms_shard_major=False,
                enable_pdl=True,
            ),
        ),
    )


def _ordered_bf16(bits: torch.Tensor) -> torch.Tensor:
    unsigned = bits.to(torch.int32) & 0xFFFF
    negative = (unsigned & 0x8000) != 0
    return torch.where(negative, 0x8000 - (unsigned & 0x7FFF), 0x8000 + unsigned)


def _max_bf16_ulp(actual: torch.Tensor, reference: torch.Tensor) -> int:
    actual_bits = actual.view(torch.int16)
    reference_bits = reference.to(torch.bfloat16).view(torch.int16)
    ulp = (_ordered_bf16(actual_bits) - _ordered_bf16(reference_bits)).abs()
    both_zero = (actual == 0) & (reference == 0)
    return int(torch.where(both_zero, 0, ulp).max().item())


def _oracle(
    routed: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    gamma: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    local = torch.zeros((TOKENS, HIDDEN), dtype=torch.float32, device=routed.device)
    for route in range(TOP_K):
        rows = indices[:, route].to(torch.int64)
        selected = routed.index_select(0, rows).float()
        local.add_(selected * weights[:, route, None].float())
    local = local.to(torch.bfloat16)

    peers = [torch.empty_like(local) for _ in range(TP_SIZE)]
    dist.all_gather(peers, local, group=group)
    reduced = torch.zeros_like(local, dtype=torch.float32)
    for peer in peers:
        reduced.add_(peer.float())
    reduced = reduced.to(torch.bfloat16).float()
    inv_rms = torch.rsqrt(reduced.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    return (reduced * inv_rms * gamma.float()).to(torch.bfloat16)


class _AttentionBreak:
    @break_point
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value)


@torch.inference_mode()
def test_ht_candidate_captures_after_attention_break(distributed_group) -> None:
    group = distributed_group
    workspace = _workspace(group)
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(17_000 + rank)

    routed = (
        torch.randn(
            TOKENS * TOP_K,
            HIDDEN,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.125
    ).to(torch.bfloat16)
    weights = (
        torch.rand(
            TOKENS,
            TOP_K,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        / TOP_K
    ).to(torch.bfloat16)
    indices = torch.arange(TOKENS * TOP_K, dtype=torch.int32, device=device).reshape(
        TOKENS, TOP_K
    )
    gamma = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
    attention_input = torch.randn(32, 32, generator=generator, device=device)
    branch_output = torch.empty_like(attention_input)
    attention_output = torch.empty_like(attention_input)
    output = torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    attention = _AttentionBreak()
    fork = StreamFork(torch.cuda.Stream(device=device))
    observed_states: list[tuple[bool, bool, bool, bool]] = []

    def forward() -> None:
        observed_states.append(
            (
                get_is_prefill_graph_phase(),
                get_is_cuda_graph_phase(),
                get_is_capture_mode(),
                fork._active,
            )
        )
        with fork.scope(enable=get_is_capture_mode()) as active_fork:
            with active_fork.branch():
                branch_output.copy_(attention_input.square())
            attention_output.copy_(attention(branch_output))
        output.copy_(workspace(routed, weights, indices, gamma))

    with prefill_graph_phase():
        for _ in range(3):
            forward()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    observed_states.clear()
    capture = BreakableCapture()
    with prefill_graph_phase(), capture:
        forward()
    assert observed_states == [(True, False, False, False)]
    assert capture.num_segments == 3
    assert not get_is_prefill_graph_phase()
    assert not get_is_cuda_graph_phase()
    assert not get_is_capture_mode()

    gamma.copy_(
        torch.linspace(0.5, 1.5, HIDDEN, dtype=torch.float32, device=device).to(
            torch.bfloat16
        )
    )
    expected = _oracle(routed, weights, indices, gamma, group)
    dist.barrier(group=group)
    capture.replay()
    torch.cuda.synchronize()

    local_ulp = _max_bf16_ulp(output, expected)
    rank_max_ulp = torch.tensor([local_ulp], dtype=torch.int32, device=device)
    dist.all_reduce(rank_max_ulp, op=dist.ReduceOp.MAX, group=group)
    assert torch.isfinite(output).all()
    assert int(rank_max_ulp.item()) <= 3
    torch.testing.assert_close(
        attention_output,
        torch.tanh(attention_input.square()),
        rtol=1e-5,
        atol=1e-6,
    )
    dist.barrier(group=group)
