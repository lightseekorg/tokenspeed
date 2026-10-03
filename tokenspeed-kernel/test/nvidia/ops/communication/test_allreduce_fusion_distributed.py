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

"""TP4/TP8/TP16 correctness and graph lifetime for the unified routed stage.

Run under torchrun with 4, 8, or 16 ranks. Snapshot copies are test instrumentation;
serving consumes the first-stage view before reusing its workspace.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication import (
    AllReduceFusionPattern,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)

H, K, EPS = 3584, 16, 1e-5
pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8, 16),
    reason="requires torchrun TP4, TP8, or TP16",
)


@pytest.fixture(scope="module")
def group():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    owns = not dist.is_initialized()
    if owns:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=20))
    yield dist.group.WORLD
    torch.cuda.synchronize()
    if owns:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def workspace(group):
    return create_allreduce_fusion_workspace(
        group=group, hidden_size=H, top_k=K, max_num_tokens=16384, rms_eps=EPS
    )


def make_inputs(m, seed):
    device = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    gen = torch.Generator(device=device).manual_seed(seed + rank)
    rows = (torch.randn((m * K, H), generator=gen, device=device) * 0.125).to(
        torch.bfloat16
    )
    common = torch.Generator(device=device).manual_seed(seed + 1000)
    weights = torch.rand((m, K), generator=common, device=device)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    indices = (
        torch.randperm(m * K, generator=common, device=device, dtype=torch.int64)
        .to(torch.int32)
        .view(m, K)
    )
    gamma = (0.5 + torch.rand((H,), generator=common, device=device)).to(torch.bfloat16)
    return rows, weights, indices, gamma


def local_finalize(rows, weights, indices):
    m = weights.shape[0]
    result = torch.zeros((m, H), device=weights.device, dtype=torch.float32)
    if rows.shape[0]:
        for k in range(K):
            index = indices[:, k].long()
            value = rows.index_select(0, index.clamp_min(0)).float()
            value.masked_fill_(index[:, None] < 0, 0.0)
            result.add_(value * weights[:, k, None].float())
    return result.to(torch.bfloat16)


def rank_sum(value, group):
    peers = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, value.contiguous(), group=group)
    result = torch.zeros_like(value, dtype=torch.float32)
    for peer in peers:
        result.add_(peer.float())
    return result.to(torch.bfloat16)


def norm_reference(local, gamma, group):
    reduced = rank_sum(local, group).float()
    return (
        reduced
        * torch.rsqrt(reduced.square().mean(-1, keepdim=True) + EPS)
        * gamma.float()
    ).to(torch.bfloat16)


def assert_close_all_ranks(actual, expected, group):
    diff = actual.float() - expected.float()
    absmax = diff.abs().max()
    l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(
        expected.float()
    ).clamp_min(1e-12)
    bounds = 0.03125 + 0.02 * expected.float().abs()
    valid = torch.isfinite(actual).all() & (diff.abs() <= bounds).all() & (l2 <= 0.01)
    flag = valid.to(torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    assert flag.item(), f"absmax={absmax.item()} relative_l2={l2.item()}"


def call(workspace, data, local, finalize):
    rows, weights, indices, gamma = data
    return allreduce_fusion(
        rows if finalize else local,
        workspace,
        pattern=(
            AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
            if finalize
            else AllReduceFusionPattern.ALLREDUCE_RMSNORM
        ),
        rms_gamma=gamma,
        num_tokens=weights.shape[0],
        expert_weights=weights if finalize else None,
        expanded_idx_to_permuted_idx=indices if finalize else None,
    )


@pytest.mark.parametrize("m", [1, 32, 33, 1024, 1025, 8193])
@pytest.mark.parametrize("finalize", [False, True])
def test_both_input_forms_match_independent_reference(group, workspace, m, finalize):
    data = make_inputs(m, 1900 + m)
    local = local_finalize(*data[:3])
    expected = norm_reference(local, data[3], group)
    actual = call(workspace, data, local, finalize)
    assert_close_all_ranks(actual, expected, group)


@pytest.mark.parametrize("m", [4, 64, 1025])
def test_independent_graphs_with_changed_inputs_and_modes(group, workspace, m):
    cases = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for seed in [2300, 2700]:
            data = make_inputs(m, seed)
            local = local_finalize(*data[:3])
            for _ in range(2):
                call(workspace, data, local, True)
                call(workspace, data, local, False)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                deferred = call(workspace, data, local, True).clone()
                finalized = call(workspace, data, local, False).clone()
            cases.append((graph, data, local, deferred, finalized))
        try:
            for generation in range(3):
                for graph, data, local, deferred, finalized in cases:
                    data[0].add_(0.01 * (generation + 1))
                    data[2][0, 0] = -1
                    local.copy_(local_finalize(*data[:3]))
                    expected = norm_reference(local, data[3], group)
                    graph.replay()
                    assert_close_all_ranks(deferred, expected, group)
                    assert_close_all_ranks(finalized, expected, group)
        finally:
            torch.cuda.synchronize()
            for graph, *_ in cases:
                graph.reset()


@pytest.mark.parametrize("m", [4, 64, 1025])
def test_empty_local_experts_and_negative_zero(group, workspace, m):
    data = list(make_inputs(m, 4200))
    if dist.get_rank() == 0:
        data[0] = data[0][:0]
        data[2].fill_(-1)
    else:
        data[0].fill_(-0.0)
    local = local_finalize(*data[:3])
    expected = norm_reference(local, data[3], group)
    for finalize in [False, True]:
        actual = call(workspace, data, local, finalize)
        assert_close_all_ranks(actual, expected, group)
        assert torch.count_nonzero(actual).item() == 0
