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

"""Distributed K3 TP4/TP8/TP16 stage-2 dispatch and stream-ordering checks.

The LL/BT/HT contracts are covered by the allreduce_fusion kernel tests.
Run this file under torchrun with 4, 8, or 16 ranks.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

H, L, EPS = 7168, 3584, 1e-6


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _setup() -> tuple[int, torch.device]:
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    # A server gathers this at distributed init; the gates refuse to, because
    # they are also asked from dispatch, where the world is not all present.
    gather_fabric_map()
    # The runtime's collectives look groups up by rank tuple; serving registers
    # them during distributed init, which this test does not run. Register the
    # already-built world group rather than calling init_process_group: that
    # would also build gloo groups, which this test never uses and which stall
    # on a multi-node launch.
    group = tuple(range(_world_size()))
    if not process_group_manager.has_process_group("nccl", group):
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    return dist.get_rank(), torch.device("cuda", local)


def _mapping_stub(world: int):
    """The MoE group used by the tail collectives."""
    return SimpleNamespace(
        moe=SimpleNamespace(
            # tokenspeed groups are tuples of global ranks, not ProcessGroups.
            tp_ep_group=tuple(range(world)),
            tp_ep_size=world,
            has_tp_ep=world > 1,
        )
    )


def _build_comm(device: torch.device):
    """Build the separate-reduce tail with real norm and projection modules."""
    from tokenspeed.runtime.distributed.comm_backend import get_global_backend
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.layers.moe.latent import Kimi3LatentProjection
    from tokenspeed.runtime.models.kimi_k3_comm import K3MoeTailComm

    world = _world_size()
    comm = object.__new__(K3MoeTailComm)
    comm.hidden_size = H
    comm.routed_hidden = L

    with torch.device(device):
        up_proj = Kimi3LatentProjection(
            L,
            H,
            params_dtype=torch.bfloat16,
            shard_group=tuple(range(world)),
            shard_rank=dist.get_rank(),
            shard_size=world,
        )
        # RMSNorm takes its dtype from the default; the checkpoint's is bf16,
        # and its kernel rejects a weight that does not match the activation.
        norm = RMSNorm(L, eps=EPS).to(torch.bfloat16)
    gw = torch.Generator(device="cpu").manual_seed(7)
    up_weight = (torch.randn(H, L, generator=gw) * 0.02).to(device, torch.bfloat16)
    start, width = up_proj.shard_slice
    up_proj.weight.data.copy_(up_weight[start : start + width])
    norm.weight.data.copy_((torch.randn(L, generator=gw) * 0.1).to(device))
    comm.up_proj = up_proj
    comm.routed_norm = norm
    comm.mapping = _mapping_stub(world)
    comm.use_allreduce_fusion = False
    comm.prepare(8193)
    assert get_global_backend().trtllm_ar.configure_group(
        dist.get_rank(),
        comm.mapping.moe.tp_ep_group,
        max_token_num=8193,
        hidden_dim=H + L,
    ), "Ordinary all-reduce must be armed before testing the stage-2 crossover"
    return comm, up_weight


def _inputs(rank: int, device: torch.device, m: int, seed: int):
    """Per-rank partials plus the rank-identical residual stream."""
    g = torch.Generator(device="cpu").manual_seed(seed + rank)
    routed = (torch.randn(m, L, generator=g) * 0.1).to(device, torch.bfloat16)
    shared = (torch.randn(m, H, generator=g) * 0.1).to(device, torch.bfloat16)
    gw = torch.Generator(device="cpu").manual_seed(seed)
    prefix = (torch.randn(m, H, generator=gw) * 0.1).to(device, torch.bfloat16)
    return routed, shared, prefix


def _reference(comm, up_weight, routed, shared, prefix):
    """All-reduce both partials, RMS-norm the latent, up-project, accumulate.

    Compute in FP32 from the full projection weight and per-rank partials.
    """
    routed_sum = routed.float().clone()
    shared_sum = shared.float().clone()
    dist.all_reduce(routed_sum)
    dist.all_reduce(shared_sum)
    var = routed_sum.pow(2).mean(dim=-1, keepdim=True)
    normed = routed_sum * torch.rsqrt(var + EPS) * comm.routed_norm.weight.float()
    up_w = up_weight.float()
    return prefix.float() + normed @ up_w.T + shared_sum


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    scale = ref.abs().max().item()
    return (out.float() - ref).abs().max().item() / max(scale, 1e-6)


def _run_tail(comm, routed, shared, prefix, m, aux):
    main = torch.cuda.current_stream()
    aux.wait_stream(main)
    with torch.cuda.stream(aux):
        shared_partial = shared.clone()
        if m <= 32:
            shared_shard = comm.shared_rs(shared_partial)
    routed_latent = comm.routed_ar_fusion(routed.clone(), m)
    main.wait_stream(aux)
    if m <= 32:
        return comm.up_proj_ag(routed_latent, shared_shard, prefix)
    return comm.up_proj_inject_ar(routed_latent, shared_partial, prefix)


collective = pytest.mark.skipif(
    _world_size() not in (4, 8, 16),
    reason="launch with torchrun world size 4, 8, or 16",
)


@collective
@pytest.mark.parametrize("m", [1, 5, 6, 32, 33, 256, 512, 1024, 1025, 8193])
def test_stage2_matches_reference_in_eager_and_graph(m):
    rank, dev = _setup()
    comm, up_weight = _build_comm(dev)
    routed, shared, prefix = _inputs(rank, dev, m, seed=33)
    aux = torch.cuda.Stream()
    ref = _reference(comm, up_weight, routed, shared, prefix)
    for _ in range(2):
        out = _run_tail(comm, routed, shared, prefix, m, aux)
        torch.cuda.synchronize()
        assert _rel_err(out, ref) < 0.05

    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture):
        output = _run_tail(comm, routed, shared, prefix, m, aux)
        next_output = _run_tail(comm, routed, shared * 2, prefix, m, aux)
    try:
        for generation in range(2):
            new_inputs = _inputs(rank, dev, m, seed=100 + generation)
            for target, value in zip((routed, shared, prefix), new_inputs):
                target.copy_(value)
            ref = _reference(comm, up_weight, routed, shared, prefix)
            next_ref = _reference(comm, up_weight, routed, shared * 2, prefix)
            graph.replay()
            torch.cuda.synchronize()
            assert _rel_err(output, ref) < 0.05
            assert _rel_err(next_output, next_ref) < 0.05
    finally:
        torch.cuda.synchronize()
        graph.reset()
