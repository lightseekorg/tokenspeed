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

"""Check the fused GEMM + reduce-scatter against an exact reference.

Small integer operands make every product and FP32 sum exact, so each rank's
BF16 partial is known bit for bit and the switch's FP32 sum of the partials
rounds once: within one BF16 unit of the reference, since the switch may break
rounding ties either way. Covers blocks that fill and that pad the 128-row
tile, row counts that change between calls without recompiling, graph replays
with fresh inputs, and an owner whose rows lie past 2^31 elements of the
symmetric storage.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.cute_dsl import (
    create_gemm_reduce_scatter_state,
    cute_dsl_gemm_reduce_scatter,
)
from utils import assert_no_cute_compile, kernel_supported

N, K = 512, 256
BLOCKS = (128, 384, 100, 256)


def _init(rank, size, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    return device


def _operands(size, block, n, k, seed, device):
    """Every rank's ``[size * block, k]`` input and ``[n, k]`` weight."""
    generator = torch.Generator().manual_seed(seed)
    shape = (size, size * block, k)
    xs = torch.randint(-2, 3, shape, generator=generator, dtype=torch.int8)
    ws = torch.randint(-2, 3, (size, n, k), generator=generator, dtype=torch.int8)
    return xs.to(device, torch.bfloat16), ws.to(device, torch.bfloat16)


def _assert_matches(got, xs, ws, rows, block):
    partials = [(x[rows].float() @ w.float().T).bfloat16() for x, w in zip(xs, ws)]
    expected = torch.stack(partials).float().sum(0).bfloat16()
    # One BF16 unit: a tie the switch rounds the other way.
    torch.testing.assert_close(got, expected, rtol=2**-7, atol=0, msg=str(block))


def _matches_reference(rank, size, rendezvous):
    device = _init(rank, size, rendezvous)
    state = create_gemm_reduce_scatter_state(
        dist.group.WORLD, device, size * max(BLOCKS), N, K
    )
    # The state compiled the kernel: no row count compiles another.
    with assert_no_cute_compile():
        for seed, block in enumerate(BLOCKS):
            xs, ws = _operands(size, block, N, K, seed, device)
            got = cute_dsl_gemm_reduce_scatter(state, xs[rank], ws[rank])
            rows = slice(rank * block, (rank + 1) * block)
            _assert_matches(got, xs, ws, rows, block)
    dist.destroy_process_group()


def _graph_replays_fresh_inputs(rank, size, rendezvous):
    device = _init(rank, size, rendezvous)
    state = create_gemm_reduce_scatter_state(
        dist.group.WORLD, device, size * max(BLOCKS), N, K
    )
    blocks = (256, 100)
    inputs = [
        _operands(size, block, N, K, 10 + i, device) for i, block in enumerate(blocks)
    ]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        outputs = [
            cute_dsl_gemm_reduce_scatter(state, xs[rank], ws[rank]) for xs, ws in inputs
        ]
    for replay in range(3):
        for i, (block, (xs, ws)) in enumerate(zip(blocks, inputs)):
            fresh, _ = _operands(size, block, N, K, 20 + 2 * replay + i, device)
            xs.copy_(fresh)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        for block, (xs, ws), out in zip(blocks, inputs, outputs):
            rows = slice(rank * block, (rank + 1) * block)
            _assert_matches(out, xs, ws, rows, block)
    dist.destroy_process_group()


def _offsets_past_int32(rank, size, rendezvous):
    device = _init(rank, size, rendezvous)
    n, k = 16384, 64
    # The last owner's rows start past 2^31 elements of the symmetric storage.
    block = 2**31 // n // size + 128
    state = create_gemm_reduce_scatter_state(
        dist.group.WORLD, device, size * block, n, k
    )
    xs, ws = _operands(size, block, n, k, 30, device)
    got = cute_dsl_gemm_reduce_scatter(state, xs[rank], ws[rank])
    end = (rank + 1) * block
    _assert_matches(got[-128:], xs, ws, slice(end - 128, end), block)
    dist.destroy_process_group()


def _spawn(target, tmp_path):
    size = min(torch.cuda.device_count(), 8)
    size = 8 if size >= 8 else 4 if size >= 4 else 2
    mp.spawn(target, args=(size, f"file://{tmp_path}/rendezvous"), nprocs=size, join=True)  # fmt: skip


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
    or not kernel_supported("cute_dsl_gemm_reduce_scatter"),
    reason="requires two SM100 or SM103 GPUs",
)


def test_matches_reference(tmp_path):
    _spawn(_matches_reference, tmp_path)


def test_graph_replays_fresh_inputs(tmp_path):
    _spawn(_graph_replays_fresh_inputs, tmp_path)


def test_offsets_past_int32(tmp_path):
    _spawn(_offsets_past_int32, tmp_path)
