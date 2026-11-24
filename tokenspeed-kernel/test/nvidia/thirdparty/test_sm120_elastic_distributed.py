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

"""Elastic FP8 dispatch/BF16 combine correctness, including graph replay.

Run under torchrun with 2, 4, 8, 16 or 32 processes; ordinary pytest skips it.
The reference is weighted identity experts with exactly FP8-representable input.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(
    _WORLD_SIZE not in (2, 4, 8, 16, 32),
    reason="requires torchrun with 2, 4, 8, 16 or 32 CUDA processes",
)


@pytest.fixture(scope="module")
def transport():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ["TOKENSPEED_SM120_DEEPEP_WIRE"] = "elastic_fp8_bf16"
    os.environ["TOKENSPEED_SM120_DEEPEP_CAPACITY"] = "128"
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment import buffer

    buffer.configure_device_nic("elastic_fp8_bf16")
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    group = dist.group.WORLD
    marker = torch.ones(1, device="cuda")
    dist.all_reduce(marker)
    assert marker.item() == _WORLD_SIZE

    values = {
        (experts, topk): buffer.get_buffer(group, 5120, topk, experts)
        for experts, topk in [(384, 6), (128, 3)]
    }
    yield values
    torch.cuda.synchronize()
    dist.barrier()
    for value in values.values():
        value.destroy()
    buffer._BUFFERS.clear()
    dist.destroy_process_group()


def _input(rows, rank):
    columns = torch.arange(5120, device="cuda")
    powers = torch.tensor([0.25, 0.5, 1.0, 2.0], device="cuda")
    value = powers[(columns // 128) % 4].repeat(rows, 1)
    if rows:
        row_ids = torch.arange(rows, device="cuda")[:, None]
        signs = ((columns[None, :] + row_ids + rank) % 2) * 2 - 1
        value *= signs
        # Encode the source rank independently of receive ordering.
        bits = (rank >> torch.arange(5, device="cuda")) & 1
        value[:, :5] = bits * 2 - 1
    return value.to(torch.bfloat16)


def _routes(rows, rank, mode, shift, experts, topk):
    row = torch.arange(rows, device="cuda")[:, None]
    slot = torch.arange(topk, device="cuda")[None, :]
    destination = (rank + row + slot * 7 + shift) % _WORLD_SIZE
    if mode == "skew":
        destination = torch.zeros_like(destination)
    ids = destination * (experts // _WORLD_SIZE) + slot
    if mode == "masked":
        ids = torch.where((row + slot) % 3 == 0, -1, ids)
    coefficients = (
        [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125] if topk == 6 else [0.5, 0.25, 0.25]
    )
    weights = torch.tensor(coefficients, device="cuda").repeat(rows, 1)
    return ids.to(torch.int64), weights


def _forward(transport, x, ids, weights, experts):
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.nvfp4_receive import (
        decode_elastic_fp8,
    )
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.q0_pack import pack_q0
    from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.route_mask import (
        mask_routes,
    )

    received, rx_ids, rx_weights, handle, _ = transport.dispatch(
        pack_q0(x),
        ids,
        weights,
        num_experts=experts,
        num_sms=4,
        num_qps=4,
        do_cpu_sync=False,
        num_max_tokens_per_rank=128,
    )
    local_ids, safe_weights = mask_routes(
        rx_ids,
        rx_weights,
        handle.psum_num_recv_tokens_per_scaleup_rank,
        experts // _WORLD_SIZE,
    )
    decoded = decode_elastic_fp8(
        *received, handle.psum_num_recv_tokens_per_scaleup_rank
    )
    coefficient = torch.where(local_ids >= 0, safe_weights, 0).sum(1)
    result = (decoded.float() * coefficient[:, None]).to(torch.bfloat16)
    output, _, _ = transport.combine(result, handle, num_sms=4, num_qps=4)
    return output


@pytest.mark.parametrize(
    "mode,rows", [("balanced", 128), ("skew", 8), ("idle", 17), ("masked", 16)]
)
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
def test_elastic_weighted_identity_eager_and_graph(
    transport, mode, rows, experts, topk
):
    rank = dist.get_rank()
    if mode == "idle" and rank % 4:
        rows = 0
    x = _input(rows, rank)
    ids, weights = _routes(rows, rank, mode, 0, experts, topk)
    value = transport[experts, topk]

    def check(output):
        coefficient = torch.where(ids >= 0, weights, 0).sum(1)
        expected = (x.float() * coefficient[:, None]).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    for _ in range(2):
        check(_forward(value, x, ids, weights, experts))
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _forward(value, x, ids, weights, experts)
    for shift in (3, 11):
        next_ids, _ = _routes(rows, rank, mode, shift, experts, topk)
        ids.copy_(next_ids)
        x.mul_(2)
        graph.replay()
        check(output)
    torch.cuda.synchronize()
    dist.barrier()


def test_elastic_target_draft_target(transport):
    rank = dist.get_rank()
    x = _input(8, rank)
    for experts, topk in [(384, 6), (128, 3), (384, 6)]:
        ids, weights = _routes(8, rank, "balanced", 3, experts, topk)
        output = _forward(transport[experts, topk], x, ids, weights, experts)
        torch.testing.assert_close(output, x, rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()
