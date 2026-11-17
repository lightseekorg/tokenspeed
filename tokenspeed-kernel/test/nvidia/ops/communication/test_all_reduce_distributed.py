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

"""The plain BT all-reduce and its place in the workspace all-reduce, at TP8 H7168.

Sums must equal the rank-ordered fp32 fold rounded once, on every rank, around
the payload threshold and the preset split; the workspace all-reduce must send
payloads at or above the threshold to it and smaller ones, or early-release
calls, to the Lamport kernels (which sum in BF16, an observable difference);
serving's row counts must compile nothing after arming; and a captured graph
interleaving both must replay correctly. Run with 8 ranks, for example
``torchrun --nproc-per-node=8`` or one srun task per GPU with WORLD_SIZE, RANK,
LOCAL_RANK, MASTER_ADDR and MASTER_PORT set.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication import trtllm
from tokenspeed_kernel.ops.communication.allreduce_fusion import (
    all_reduce,
    all_reduce_supported,
    create_all_reduce_workspace,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    ensure_workspace_initialized,
    trtllm_workspace_allreduce,
)
from tokenspeed_kernel.thirdparty.flashinfer.plain_all_reduce import (
    PLAIN_ALL_REDUCE_MIN_BYTES,
)
from utils import assert_no_cute_compile

H = 7168
CAPACITY = 2048
# Smallest served row count, the preset split (256) on both sides, and capacity.
THRESHOLD_ROWS = -(-PLAIN_ALL_REDUCE_MIN_BYTES // (2 * H))
SERVED_ROWS = [THRESHOLD_ROWS, 128, 256, 257, 512, 1000, CAPACITY]

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="requires eight ranks"
)


@pytest.fixture(scope="module")
def group():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    owns = not dist.is_initialized()
    if owns:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=20))
    supported = torch.tensor(
        [
            int(
                all_reduce_supported(
                    group=dist.group.WORLD,
                    hidden_size=H,
                    max_num_tokens=CAPACITY,
                    dtype=torch.bfloat16,
                )
            )
        ],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(supported, op=dist.ReduceOp.MIN)
    if not supported.item():
        pytest.skip("the plain BT all-reduce is unavailable on some rank")
    yield dist.group.WORLD
    torch.cuda.synchronize()
    if owns:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def workspace(group):
    return create_all_reduce_workspace(
        group=group, hidden_size=H, max_num_tokens=CAPACITY
    )


def _inputs(rows, seed):
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(seed + dist.get_rank())
    return (torch.randn((rows, H), generator=gen, device=device) * 0.5).to(
        torch.bfloat16
    )


def _fold32(x):
    """The ranks' rows summed in rank order in fp32, rounded once."""
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x)
    total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for part in parts:
        total += part.float()
    return total.to(torch.bfloat16)


def _same_on_every_rank(x):
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x)
    return all(torch.equal(parts[0], part) for part in parts)


def _agree(ok, what):
    flag = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert flag.item() == 1, f"{what} failed on some rank"


@pytest.mark.parametrize("rows", SERVED_ROWS)
def test_plain_all_reduce_is_the_rank_ordered_fp32_fold(workspace, rows):
    x = _inputs(rows, seed=rows)
    out = all_reduce(x, workspace, torch.empty_like(x))
    torch.cuda.synchronize()
    _agree(torch.equal(out, _fold32(x)), f"fold at {rows} rows")
    _agree(_same_on_every_rank(out), f"agreement at {rows} rows")


def test_served_row_counts_compile_nothing(workspace):
    # Arming compiled the kernels; the row counts serving varies reuse them.
    warm = _inputs(THRESHOLD_ROWS, seed=5)
    all_reduce(warm, workspace, torch.empty_like(warm))
    try:
        with assert_no_cute_compile():
            for rows in SERVED_ROWS:
                x = _inputs(rows, seed=rows + 11)
                all_reduce(x, workspace, torch.empty_like(x))
        compiled = ""
    except AssertionError as error:
        compiled = str(error)
    torch.cuda.synchronize()
    _agree(not compiled, f"compile-free served rows ({compiled})")


def test_plain_all_reduce_refuses_small_payloads(workspace):
    x = _inputs(THRESHOLD_ROWS - 1, seed=3)
    with pytest.raises(ValueError):
        all_reduce(x, workspace, torch.empty_like(x))


def _rounding_probe(rows):
    """Inputs whose BF16 and fp32 folds over the eight ranks round apart.

    Rank 0 holds 1.0, the others 2**-9: a BF16 fold stays at 1.0, while the
    fp32 fold rounds 1 + 7 * 2**-9 to 1.015625.
    """
    value = 1.0 if dist.get_rank() == 0 else 2.0**-9
    return torch.full((rows, H), value, dtype=torch.bfloat16, device="cuda")


@pytest.fixture(scope="module")
def armed(group):
    assert ensure_workspace_initialized(
        rank=dist.get_rank(), group=group, max_token_num=CAPACITY, hidden_dim=H
    )
    return group


@pytest.mark.parametrize("rows", [8, 64, THRESHOLD_ROWS - 1, *SERVED_ROWS])
def test_workspace_all_reduce_routes_by_payload(armed, rows):
    out = trtllm_workspace_allreduce(_rounding_probe(rows), armed, fp32_acc=False)
    torch.cuda.synchronize()
    want = 1.015625 if rows * H * 2 >= PLAIN_ALL_REDUCE_MIN_BYTES else 1.0
    _agree(bool((out == want).all()), f"route at {rows} rows")


def test_an_early_release_call_keeps_the_lamport_kernels(armed):
    # Defined on NVIDIA only, so it is looked up here rather than imported.
    out = trtllm._workspace_allreduce(
        _rounding_probe(512),
        armed,
        fp32_acc=False,
        out=None,
        trigger_completion_at_end=False,
    )
    torch.cuda.synchronize()
    _agree(bool((out == 1.0).all()), "early-release route")


def test_captured_mixed_sizes_replay(armed):
    shapes = [8, 512, 128, CAPACITY, 64, THRESHOLD_ROWS]
    inputs = [_inputs(rows, seed=rows + 7) for rows in shapes]
    outputs = [torch.empty_like(x) for x in inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x, out in zip(inputs, outputs):
            trtllm_workspace_allreduce(x, armed, fp32_acc=False, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x, out in zip(inputs, outputs):
            trtllm_workspace_allreduce(x, armed, fp32_acc=False, out=out)
    for step in range(3):
        for i, x in enumerate(inputs):
            x.copy_(_inputs(x.shape[0], seed=100 * step + i))
        graph.replay()
        torch.cuda.synchronize()
        for x, out in zip(inputs, outputs):
            rows = x.shape[0]
            if rows * H * 2 >= PLAIN_ALL_REDUCE_MIN_BYTES:
                ok = torch.equal(out, _fold32(x))
            else:
                # The Lamport kernels sum in BF16: compare with a tolerance.
                ok = torch.allclose(out.float(), _fold32(x).float(), atol=0.1)
            _agree(ok, f"replay {step} at {rows} rows")
    del graph
