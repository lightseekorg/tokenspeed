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

"""Verify row-sharded K3 MoE inputs against the replicated projection."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
        dist.barrier(group=group, device_ids=[rank])
        state = _prepare(rank, device, group)
        _check_row_inputs(rank, device, group, state)
    finally:
        dist.destroy_process_group()


def _prepare(rank: int, device: torch.device, group):
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )

    return prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=4096 * 10752,
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=False,
        sharded_max_rows=4096,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )


def _check_row_inputs(rank: int, device: torch.device, group, state) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_largem import (
        launch_gluon_latent_input_largem_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
        _quantize_mxfp4_activation,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        invoke_sigmoid_bias_topk_route_prefill_gluon,
    )

    # Every rank holds the same rows and weights, as after the attention gather.
    generator = torch.Generator(device=device).manual_seed(1030)
    weight = (torch.randn(6016, 7168, device=device, generator=generator) / 40).to(
        torch.bfloat16
    )
    bias = torch.randn(896, device=device, generator=generator) / 100
    calls = []

    def route(rows, logits):
        calls.append(rows.shape[0])
        ids, weights = invoke_sigmoid_bias_topk_route_prefill_gluon(
            logits,
            bias,
            16,
            routed_scaling_factor=2.5,
            normalize_topk_weights=True,
            simulation_seed=None,
        )
        return weights, ids

    assert state.sharded.can_gather_moe_inputs(4096)
    assert not state.sharded.can_gather_moe_inputs(4096 - 8)
    assert not state.sharded.can_gather_moe_inputs(8192)
    for rows in (4096, 2048, 256):
        for layer in range(2):
            hidden = (torch.randn(rows, 7168, device=device, generator=generator)).to(
                torch.bfloat16
            )
            router, latent, shared = launch_gluon_latent_input_largem_gfx950(
                hidden,
                weight[:896],
                weight[896:4480],
                weight[4480:],
                weight,
                beta=4.0,
                linear_beta=25.0,
            )
            ids, weights = invoke_sigmoid_bias_topk_route_prefill_gluon(
                router,
                bias,
                16,
                routed_scaling_factor=2.5,
                normalize_topk_weights=True,
                simulation_seed=None,
            )
            packed, scales = _quantize_mxfp4_activation(latent)
            if rank == (rows // 256 + layer) % 8:
                # A late rank must still see every peer's shard.
                torch.cuda._sleep(2_000_000)
            result = state.moe_row_sharded_inputs(
                hidden,
                weight,
                beta=4.0,
                linear_beta=25.0,
                route=route,
                attention_gather=None,
            )
            assert result is not None
            (got_packed, got_scales), got_weights, got_ids, got_shared = result
            torch.cuda.synchronize()
            assert torch.equal(got_packed, packed)
            assert got_scales.shape == scales.shape
            assert torch.equal(got_scales, scales)
            assert torch.equal(got_ids, ids)
            assert torch.equal(got_weights, weights)
            assert torch.equal(got_shared, shared)
            dist.barrier(group=group, device_ids=[rank])
    assert calls == [4096 // 8] * 2 + [2048 // 8] * 2 + [256 // 8] * 2
    unsupported = torch.zeros(4096 - 256 + 8, 7168, device=device, dtype=torch.bfloat16)
    assert (
        state.moe_row_sharded_inputs(
            unsupported,
            weight,
            beta=4.0,
            linear_beta=25.0,
            route=route,
            attention_gather=None,
        )
        is None
    )
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="Row-sharded MoE inputs gather across eight CDNA4 GPUs",
)
def test_iris_moe_row_sharded_inputs() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=8, join=True)
