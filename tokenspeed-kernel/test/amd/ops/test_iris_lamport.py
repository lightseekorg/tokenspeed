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

import re
import socket
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd, reason="Iris requires AMD ROCm"
)


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication.iris")


@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize("reverse", (False, True))
def test_lamport_boundary(rows, reverse):
    from tokenspeed_kernel.ops.communication.iris import (
        _kimi_k3_moe_producer_direct_protocol,
    )

    shapes = ((rows, 3584), (rows, 7168))
    if reverse:
        shapes = shapes[::-1]
    expected = "lamport" if rows <= 6 else None
    assert _kimi_k3_moe_producer_direct_protocol(8, shapes, torch.bfloat16) == expected


@pytest.mark.parametrize(
    "world,shapes,dtype",
    [
        (4, ((1, 3584), (1, 7168)), torch.bfloat16),
        (8, ((1, 3584), (1, 7168)), torch.float16),
        (8, ((1, 3584), (1, 7168)), torch.float32),
        (8, ((0, 3584), (0, 7168)), torch.bfloat16),
        (8, ((1, 3584), (2, 7168)), torch.bfloat16),
        (8, ((1, 10752),), torch.bfloat16),
        (8, ((3584,), (7168,)), torch.bfloat16),
        (8, ((1, 3584), (1, 7168), (1, 512)), torch.bfloat16),
        (8, ((1, 4096), (1, 6656)), torch.bfloat16),
    ],
)
def test_lamport_rejects_other_payloads(world, shapes, dtype):
    from tokenspeed_kernel.ops.communication.iris import (
        _kimi_k3_moe_producer_direct_protocol,
    )

    assert _kimi_k3_moe_producer_direct_protocol(world, shapes, dtype) is None


@pytest.mark.parametrize("enable_lamport", [False, True])
@pytest.mark.parametrize("rows", [1, 6, 7])
@pytest.mark.parametrize("reverse", [False, True])
def test_lamport_dispatch_requires_opt_in(monkeypatch, enable_lamport, rows, reverse):
    from tokenspeed_kernel.ops.communication import iris as iris_ops

    monkeypatch.setattr(iris_ops, "_platform", SimpleNamespace(is_cdna4=True))
    state = iris_ops.IrisAllReduce.__new__(iris_ops.IrisAllReduce)
    state.world_size = 8
    state.dtype = torch.bfloat16
    state.device = torch.device("cpu")
    state.enable_lamport = enable_lamport
    state._kernel_config = iris_ops.IRIS_ALL_REDUCE_KERNEL_CONFIG
    state._elements_per_word = (
        state._kernel_config.packed_word_bytes // state.dtype.itemsize
    )
    state.producer_direct_max_numel = rows * 10752
    state._input_buf = torch.empty(rows * 10752, dtype=state.dtype)
    state._reduced_output_buf = torch.empty_like(state._input_buf)
    state._all_reduce_symmetric_lamport = Mock()
    state._all_reduce_symmetric_pull = Mock()
    shapes = ((rows, 3584), (rows, 7168))
    if reverse:
        shapes = shapes[::-1]
    inputs = iris_ops.iris_acquire_outputs(state, shapes)

    outputs = iris_ops.iris_all_reduce_symmetric(state, inputs)

    assert tuple(tuple(tensor.shape) for tensor in outputs) == shapes
    if enable_lamport and rows <= 6:
        state._all_reduce_symmetric_lamport.assert_called_once_with(rows * 10752)
        state._all_reduce_symmetric_pull.assert_not_called()
    else:
        state._all_reduce_symmetric_pull.assert_called_once_with(rows * 10752)
        state._all_reduce_symmetric_lamport.assert_not_called()


@pytest.mark.parametrize("rank", range(8))
def test_lamport_buffer_polling_codegen(rank, tmp_path):
    """Cache flags alone do not prevent a future compiler from hoisting loads.

    Compile without launching and require all seven 16-byte peer reads to
    remain on the retry backedge. This intentionally guards the tested machine
    behavior in addition to the distributed numerical tests.
    """
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication.iris import (
        lamport_all_reduce_bf16,
    )

    fn = lamport_all_reduce_bf16
    signature = {
        name: "*i32" if name == "epochs" else "*bf16" for name in fn.arg_names[:4]
    }
    signature.update({name: "i64" for name in fn.arg_names[4:12]})
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs={
            "RANK": rank,
            "WORLD_SIZE": 8,
            "TOTAL_ELEMENTS": 6 * 10752,
            "MAX_ELEMENTS": 6 * 10752,
            "NUM_STAGES": 3,
        },
        attrs={(index,): [["tt.divisibility", 16]] for index in range(12)},
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 1},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"lamport-rank{rank}.amdgcn").write_text(assembly)
    (tmp_path / f"lamport-rank{rank}.llir").write_text(kernel.asm["llir"])
    lines = assembly.splitlines()
    labels = {
        match[1]: index
        for index, line in enumerate(lines)
        if (match := re.match(r"^(\.LBB\w+):", line))
    }
    cycles = []
    for index, line in enumerate(lines):
        match = re.search(r"\bs_(?:cbranch_\w+|branch)\s+(\.LBB\w+)", line)
        if match and labels[match[1]] < index:
            cycles.append(lines[labels[match[1]] : index + 1])
    assert cycles, "polling loop was removed"
    assert any(
        sum("buffer_load_dwordx4" in line and "sc0 sc1" in line for line in cycle) == 7
        for cycle in cycles
    ), "all seven system-cache reads must execute again on a retry"
    assert (
        sum("buffer_store_dwordx4" in line and "sc0 sc1" in line for line in lines)
        == 14
    ), "publication and reset must use 16-byte system-cache stores"
    assert kernel.metadata.num_warps == 1
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def _new_state(rank, device, capacity, dtype):
    from tokenspeed_kernel.ops.communication.iris import create_iris_state

    return create_iris_state(
        enable_lamport=True,
        moe_tail_max_rows=0,
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=capacity,
        attnres_max_numel=0,
        attnres_max_rows=0,
        dtype=dtype,
        heap_size=None,
        device=device,
    )


def _check_lamport_state(rank, device):
    from tokenspeed_kernel.ops.communication.iris import (
        iris_acquire_outputs,
        iris_all_reduce_symmetric,
    )

    state = _new_state(rank, device, 8 * 10752, torch.bfloat16)
    region = state._kimi_k3_moe_lamport_region
    epochs = state._kimi_k3_moe_lamport_epochs
    assert region.shape == (3, 8, 6 * 10752)
    assert epochs.shape == (126,)
    assert state._producer_direct_max_programs == 84
    torch.testing.assert_close(
        region.view(torch.int32),
        torch.full_like(region.view(torch.int32), -2147483648),
        atol=0,
        rtol=0,
    )
    expected_epochs = torch.zeros_like(epochs)
    for rows in range(1, 9):
        for reverse in (False, True):
            shapes = ((rows, 3584), (rows, 7168))
            if reverse:
                shapes = shapes[::-1]
            inputs = iris_acquire_outputs(state, shapes)
            # Distinct element/rank values, cancellation, and sentinel-valued
            # inputs. Reference uses the kernel's ascending-rank FP32 order.
            generator = torch.Generator().manual_seed(100 + rows)
            peers = torch.randn(8, rows * 10752, generator=generator).bfloat16()
            peers[:, ::31] = -0.0
            peers[1::2, 1::31] = -peers[::2, 1::31]
            peers[:, 2::31] = float("inf")
            peers[:, 3::31] = float("nan")
            expected = peers[0].float()
            for peer in range(1, 8):
                expected += peers[peer].float()
            offset = 0
            for tensor in inputs:
                tensor.copy_(
                    peers[rank, offset : offset + tensor.numel()].view(tensor.shape)
                )
                offset += tensor.numel()
            actual = iris_all_reduce_symmetric(state, inputs)
            actual = torch.cat([tensor.flatten() for tensor in actual])
            torch.testing.assert_close(
                actual.cpu(), expected.bfloat16(), atol=0, rtol=0, equal_nan=True
            )
            if rows <= 6:
                expected_epochs[: rows * 21] = (expected_epochs[: rows * 21] + 1) % 3
            torch.testing.assert_close(epochs, expected_epochs, atol=0, rtol=0)

    # Capture every row count, then alternate Lamport and pull without host
    # synchronization between replays. Skipped tiles retain independent epochs.
    captures = []
    for rows in (6, 1, 7, 3, 8, 2, 5, 4):
        inputs = iris_acquire_outputs(state, ((rows, 7168), (rows, 3584)))
        sources = tuple(torch.full_like(tensor, rank + 1) for tensor in inputs)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph):
            for tensor, source in zip(inputs, sources, strict=True):
                tensor.copy_(source)
            outputs = iris_all_reduce_symmetric(state, inputs)
        captures.append((rows, graph, sources, outputs))
    snapshots = []
    before_epochs = epochs.clone()
    pointers = (region.data_ptr(), epochs.data_ptr(), state._input_buf.data_ptr())
    for iteration in range(12):
        for rows, graph, sources, outputs in captures:
            scale = iteration + 1
            for index, source in enumerate(sources, start=1):
                source.fill_(scale * index * (rank + 1))
            if rank == iteration % 8:
                torch.cuda._sleep(100_000)
            graph.replay()
            snapshots.append((scale, tuple(output.clone() for output in outputs)))
    torch.cuda.synchronize()
    assert pointers == (
        region.data_ptr(),
        epochs.data_ptr(),
        state._input_buf.data_ptr(),
    )
    torch.testing.assert_close(epochs, before_epochs, atol=0, rtol=0)
    for scale, outputs in snapshots:
        for index, output in enumerate(outputs, start=1):
            torch.testing.assert_close(
                output, torch.full_like(output, 36 * scale * index), atol=0, rtol=0
            )

    for capacity, expected_capacity in (
        (0, 0),
        (10751, 0),
        (10752, 10752),
        (32257, 32256),
    ):
        smaller = _new_state(rank, device, capacity, torch.bfloat16)
        assert smaller._kimi_k3_moe_lamport_max_numel == expected_capacity
        if expected_capacity:
            inputs = iris_acquire_outputs(smaller, ((1, 3584), (1, 7168)))
            for tensor in inputs:
                tensor.fill_(rank + 1)
            for output in iris_all_reduce_symmetric(smaller, inputs):
                torch.testing.assert_close(
                    output, torch.full_like(output, 36), atol=0, rtol=0
                )
        else:
            assert smaller._kimi_k3_moe_lamport_region is None
            assert smaller._kimi_k3_moe_lamport_epochs is None
    for dtype in (torch.float16, torch.float32):
        other = _new_state(rank, device, 10752, dtype)
        assert other._kimi_k3_moe_lamport_region is None
        inputs = iris_acquire_outputs(other, ((1, 3584), (1, 7168)))
        for tensor in inputs:
            tensor.fill_(rank + 1)
        for output in iris_all_reduce_symmetric(other, inputs):
            torch.testing.assert_close(
                output, torch.full_like(output, 36), atol=0, rtol=0
            )


def _lamport_worker(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        _check_lamport_state(rank, torch.device(f"cuda:{rank}"))
    finally:
        dist.destroy_process_group()


def test_lamport_world8():
    if not current_platform().is_cdna4 or torch.cuda.device_count() < 8:
        pytest.skip("requires eight CDNA4 GPUs")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    mp.spawn(_lamport_worker, args=(port,), nprocs=8, join=True)
