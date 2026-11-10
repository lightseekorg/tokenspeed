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

"""Verify the K3 MoE all-reduces (one-shot and two-shot) that sum the decode
MoE's top-k slot rows."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd, reason="Iris requires AMD ROCm"
)

_ROUTED = 3584
_SHARED = 7168
_TOPK = 16


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _topk_reduce_worker(rank: int, port: int) -> None:
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
        _check_topk_reduce(rank, device, group)
    finally:
        dist.destroy_process_group()


def _check_topk_reduce(
    rank: int, device: torch.device, group: dist.ProcessGroup
) -> None:
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        prepare_k3_communication,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.prefill_stage2 import (
        gluon_mxfp4_moe_stage2_reduce_kernel,
    )

    state = prepare_k3_communication(
        group=group,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=64 * (_ROUTED + _SHARED),
        attnres_max_numel=0,
        attnres_max_rows=0,
        enable_lamport=True,
        sharded_max_rows=0,
        dtype=torch.bfloat16,
        device=device,
        all_reduce=None,
    )
    producer = state.all_reduce
    inputs = {}

    def partials(rows, seed):
        if seed not in inputs:
            source = torch.Generator(device=device).manual_seed(seed * 8 + rank)
            slots = torch.randn(
                (rows * _TOPK, _ROUTED), generator=source, device=device
            ).to(torch.bfloat16)
            # Negative zeros reach the mailbox only as sanitized zeros.
            slots[:, :8] = -0.0
            shared = torch.randn((rows, _SHARED), generator=source, device=device).to(
                torch.bfloat16
            )
            inputs[seed] = slots, shared
        return inputs[seed]

    def separate(rows, seed):
        """The MoE's own top-k reduce into the producer, then the all-reduce."""
        slots, shared = partials(rows, seed)
        outputs = producer.acquire_outputs(((rows, _ROUTED), (rows, _SHARED)))
        gluon_mxfp4_moe_stage2_reduce_kernel[(rows * (_ROUTED // 256),)](
            slots,
            outputs[0],
            rows,
            _ROUTED,
            _TOPK * slots.stride(0),
            slots.stride(0),
            slots.stride(1),
            outputs[0].stride(0),
            outputs[0].stride(1),
            BLOCK_M=1,
            BLOCK_N=256,
            TOP_K=_TOPK,
            MASK_INVALID_ROUTES=False,
            num_warps=1,
        )
        outputs[1].copy_(shared)
        return state.reduce_moe(outputs)

    def fused(rows, seed):
        slots, shared = partials(rows, seed)
        outputs = producer.acquire_outputs(((rows, _ROUTED), (rows, _SHARED)))
        # Whatever the routed destination holds is ignored.
        outputs[0].fill_(float("nan"))
        outputs[1].copy_(shared)
        assert state.can_reduce_moe_topk_partials(tuple(t.shape for t in outputs))
        return state.reduce_moe(outputs, routed_partials=slots)

    # Plain and partial-sum reduces share mailbox tiles: interleave one- and two-shot.
    for seed, rows in enumerate((4, 1, 8, 7, 64, 6, 16, 3, 2, 32, 5, 10, 4, 48)):
        expected = [t.clone() for t in separate(rows, seed)]
        result = fused(rows, seed)
        for got, want in zip(result, expected, strict=True):
            assert torch.equal(got, want)

    # Odd rows past the one-shot window are not two-shot aligned.
    assert not state.can_reduce_moe_topk_partials(((9, _ROUTED), (9, _SHARED)))
    assert not state.can_reduce_moe_topk_partials(((66, _ROUTED), (66, _SHARED)))
    assert not state.can_reduce_moe_topk_partials(((4, _SHARED), (4, _ROUTED)))
    assert not state.can_reduce_moe_topk_partials(((16, _SHARED), (16, _ROUTED)))
    outputs = producer.acquire_outputs(((4, _ROUTED), (4, _SHARED)))
    with pytest.raises(ValueError):
        state.reduce_moe(outputs, routed_partials=partials(4, 0)[0][:, : _SHARED // 4])
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])

    # Captured decode layers (one-shot and two-shot rows) with one rank late.
    layer_rows = (4, 32, 4, 8, 64, 4)
    expected = [
        [t.clone() for t in separate(rows, 100 + layer)]
        for layer, rows in enumerate(layer_rows)
    ]
    snapshots = [[torch.empty_like(t) for t in layer] for layer in expected]
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for layer in range(6):
                if rank == (layer * 3 + 1) % 8:
                    torch.cuda._sleep(100_000)
                for snapshot, result in zip(
                    snapshots[layer], fused(layer_rows[layer], 100 + layer), strict=True
                ):
                    snapshot.copy_(result)
    torch.cuda.synchronize()
    dist.barrier(group=group, device_ids=[rank])
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for layer_snapshots, layer_expected in zip(snapshots, expected, strict=True):
            for got, want in zip(layer_snapshots, layer_expected, strict=True):
                assert torch.equal(got, want)
    dist.barrier(group=group, device_ids=[rank])


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="the K3 Lamport MoE all-reduce runs across eight CDNA4 GPUs",
)
def test_iris_moe_topk_partials_reduce_bitwise() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_topk_reduce_worker, args=(port,), nprocs=8, join=True)
