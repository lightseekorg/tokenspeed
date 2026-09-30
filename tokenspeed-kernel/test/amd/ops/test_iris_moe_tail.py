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

"""Verify the MoE tail when each rank handles consecutive token rows."""

import re
import socket
from contextlib import ExitStack
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile, is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication.iris")


@pytest.mark.parametrize("rank", (0, 7))
@pytest.mark.parametrize("prefix_is_sharded", (False, True))
def test_moe_gather_vector_codegen(rank, prefix_is_sharded, tmp_path):
    """Runtime row counts must retain vectorized payload loads and peer stores."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication.iris import (
        iris_moe_add_push_gather_gluon_kernel,
    )

    fn = iris_moe_add_push_gather_gluon_kernel
    constants = {
        "RANK": rank,
        "BLOCK_ELEMENTS": 2048,
        "NUM_PROGRAMS": 128,
        "NUM_WARPS": 4,
        "PREFIX_IS_SHARDED": prefix_is_sharded,
    }
    signature = {name: "*bf16" for name in fn.arg_names[:4]}
    signature["ready_flags"] = "*i32"
    signature.update({name: "i64" for name in fn.arg_names[5:13]})
    signature.update(
        {name: "i32" for name in fn.arg_names[13:] if name not in constants}
    )
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs=constants,
        # Payloads and heap bases are aligned; the runtime row count has no
        # divisibility annotation, including for an odd number of local rows.
        attrs={(index,): [["tt.divisibility", 16]] for index in range(13)},
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4},
    )
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"moe-gather-rank{rank}.amdgcn").write_text(assembly)
    lines = assembly.splitlines()
    assert sum("buffer_load_dwordx4" in line for line in lines) == 3
    assert (
        sum("buffer_store_dwordx4" in line and "sc0 sc1" in line for line in lines) == 8
    )
    assert not re.search(r"\bbuffer_(?:load_ushort|store_short)\b", assembly)


def _moe_tail_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Use Gloo for host coordination and RCCL for the device group.
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
        _check_moe_tail(rank, device, group)
    finally:
        dist.destroy_process_group()


def _check_moe_tail(rank: int, device: torch.device, group: dist.ProcessGroup) -> None:
    from tokenspeed_kernel.ops.communication import triton as comm
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_ar_rmsnorm_state,
        iris_kimi3_moe_tail,
        iris_moe_add_push_gather_gluon_kernel,
        iris_moe_reduce_scatter_gluon_kernel,
    )
    from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_latent_projection_add3
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

    backing = comm.TritonCommState(
        group=group,
        rank_in_group=rank,
        world_size=8,
        device=device,
        attnres_max_numel=0,
        enable_lamport=True,
        moe_tail_max_rows=8192,
        max_numel=0,
        max_bytes=8192 * 10752 * 2,
        max_token_num=0,
        hidden_dim=0,
        comm_buff=None,
        symm_mem_hdl=None,
    )
    comm.initialize_all_reduce_state(backing, torch.bfloat16)
    state = comm._get_or_create_iris_state(backing, torch.bfloat16)
    # Check that the shared heap still has room for EAGLE3's reduction state.
    rmsnorm_state = create_iris_ar_rmsnorm_state(
        group=group,
        rank_in_group=rank,
        max_token_num=2048,
        hidden_dim=7168,
        dtype=torch.bfloat16,
        heap_size=None,
        device=device,
        persistent=False,
    )
    assert rmsnorm_state._ctx is state._ctx
    allocations = (
        state._input_buf,
        state._producer_direct_scratch_buf,
        state._producer_direct_ready_flags,
        state._moe_tail_output_buf,
        state._moe_tail_ready_flags,
    )
    allocation_pointers = tuple(t.data_ptr() for t in allocations)
    generator = torch.Generator(device=device).manual_seed(72391)
    weight = torch.randn(
        (7168, 3584), dtype=torch.bfloat16, device=device, generator=generator
    ) / (3584**0.5)
    norm = (
        torch.rand((3584,), dtype=torch.bfloat16, device=device, generator=generator)
        + 0.5
    )

    def acquire(rows):
        return comm.acquire_symm_outputs(
            backing, ((rows, 3584), (rows, 7168)), torch.bfloat16
        )

    def restore(inputs, sources):
        for destination, source in zip(inputs, sources, strict=True):
            destination.copy_(source)

    def ordinary(inputs, prefix, norm_weight):
        routed, shared = comm.all_reduce_symmetric(backing, inputs)
        if norm_weight is not None:
            routed = rmsnorm(routed, norm_weight, 1e-5, residual=None, out=None)
        return kimi3_latent_projection_add3(
            routed,
            weight,
            prefix,
            shared,
            norm_weight=None,
            eps=None,
            solution="auto",
        )

    tail_kernels_warmed = set()

    def projected(inputs, prefix, norm_weight, *, prefix_is_sharded):
        # Warm once, then exercise changing row counts without recompiling
        # either collective. Projection and normalization have separate kernels.
        with ExitStack() as stack:
            if prefix_is_sharded in tail_kernels_warmed:
                for kernel in (
                    iris_moe_reduce_scatter_gluon_kernel,
                    iris_moe_add_push_gather_gluon_kernel,
                ):
                    stack.enter_context(assert_no_triton_compile(kernel))
            output = iris_kimi3_moe_tail(
                *inputs,
                prefix,
                weight,
                prefix_is_sharded=prefix_is_sharded,
                norm_weight=norm_weight,
                eps=1e-5 if norm_weight is not None else None,
                group=group,
            )
        assert output is not None
        tail_kernels_warmed.add(prefix_is_sharded)
        return output

    held = None
    held_expected = None
    for rows in (40, 48, 56, 64, 128, 256, 512, 520, 848, 4096, 8144, 8192):
        inputs = acquire(rows)
        generator.manual_seed(89103 + rank)
        sources = tuple(
            torch.randn(
                tensor.shape, dtype=tensor.dtype, device=device, generator=generator
            )
            / 8
            for tensor in inputs
        )
        generator.manual_seed(781)
        prefix = torch.randn(
            (rows, 7168), dtype=torch.bfloat16, device=device, generator=generator
        )
        for norm_weight in (None, norm):
            restore(inputs, sources)
            expected = ordinary(inputs, prefix, norm_weight)
            restore(inputs, sources)
            if rank == rows % 8:
                torch.cuda._sleep(100_000)
            result = projected(inputs, prefix, norm_weight, prefix_is_sharded=False)
            assert result.data_ptr() == state._moe_tail_output_buf.data_ptr()
            torch.testing.assert_close(result, expected, atol=0.03125, rtol=0.015625)
            output = result.clone()
            for tensor, source in zip(inputs, sources, strict=True):
                torch.testing.assert_close(tensor, source, atol=0, rtol=0)
            # Ordinary collectives preserve the reusable result.
            ordinary(inputs, prefix, norm_weight)
            torch.testing.assert_close(result, output, atol=0, rtol=0)
            # Delay one rank before reusing the output as its prefix.
            restore(inputs, sources)
            if rank == rows % 8:
                torch.cuda._sleep(100_000)
            result.copy_(prefix)
            inplace = projected(inputs, result, norm_weight, prefix_is_sharded=False)
            assert inplace is not None and inplace.data_ptr() == result.data_ptr()
            torch.testing.assert_close(
                inplace.view(torch.int16), output.view(torch.int16), atol=0, rtol=0
            )
            # The local prefix must survive projection into the borrowed result.
            # Include normalization, odd local rows, and a delayed peer.
            first_row = rank * (rows // 8)
            local_prefix = prefix[first_row : first_row + rows // 8].clone()
            restore(inputs, sources)
            if rank == rows % 8:
                torch.cuda._sleep(100_000)
            local_result = projected(
                inputs, local_prefix, norm_weight, prefix_is_sharded=True
            )
            torch.testing.assert_close(local_result, output, atol=0, rtol=0)
            torch.testing.assert_close(
                local_prefix, prefix[first_row : first_row + rows // 8], atol=0, rtol=0
            )
            if rows in (40, 64):
                # The lower dispatch boundary also reaches decode graphs.
                torch.cuda.synchronize()
                dist.barrier()
                small_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(small_graph):
                    restore(inputs, sources)
                    captured = projected(
                        inputs, prefix, norm_weight, prefix_is_sharded=False
                    )
                for _ in range(2):
                    for source in sources:
                        source.neg_()
                    restore(inputs, sources)
                    reference = ordinary(inputs, prefix, norm_weight)
                    small_graph.replay()
                    torch.testing.assert_close(
                        captured, reference, atol=0.03125, rtol=0.015625
                    )
            # Cloned outputs survive later calls.
            if held is not None:
                torch.testing.assert_close(
                    held, held_expected, atol=0.03125, rtol=0.015625
                )
            held, held_expected = output, expected

    # Rejected inputs must leave buffers and flags untouched.
    torch.cuda.synchronize()
    dist.barrier()
    inputs = acquire(848)
    restore(inputs, tuple(t[:848] for t in sources))
    prefix = torch.zeros((848, 7168), dtype=torch.bfloat16, device=device)
    flags_before = state._producer_direct_ready_flags.clone()
    gather_flags_before = state._moe_tail_ready_flags.clone()
    inputs_before = tuple(t.clone() for t in inputs)
    unsupported = [
        (inputs[0].clone(), inputs[1], prefix, norm, 1e-5, group),
        (inputs[0], inputs[1].clone(), prefix, norm, 1e-5, group),
        (inputs[0], inputs[1], prefix, norm, 1e-5, dist.group.WORLD),
        (inputs[0], inputs[1], prefix, norm.float(), 1e-5, group),
        (inputs[0], inputs[1], prefix, norm, float("nan"), group),
        (inputs[0], inputs[1], prefix, norm, 0.0, group),
        (inputs[0], inputs[1], prefix, None, 1e-5, group),
        (inputs[0], inputs[1], prefix.T.contiguous().T, norm, 1e-5, group),
        (
            inputs[0],
            inputs[1],
            state._input_buf[: 848 * 7168].view(848, 7168),
            norm,
            1e-5,
            group,
        ),
        (inputs[0][:-1], inputs[1][:-1], prefix[:-1], norm, 1e-5, group),
    ]
    for routed, shared, residual, norm_weight, eps, owner in unsupported:
        assert (
            iris_kimi3_moe_tail(
                routed,
                shared,
                residual,
                weight,
                prefix_is_sharded=False,
                norm_weight=norm_weight,
                eps=eps,
                group=owner,
            )
            is None
        )
    torch.testing.assert_close(state._producer_direct_ready_flags, flags_before)
    # Reject shifted prefixes and weights that overlap the output.
    result_buffer = state._moe_tail_output_buf
    invalid_aliases = (
        (result_buffer[1:849], weight, norm),
        (result_buffer.flatten()[1 : 848 * 7168 + 1].view(848, 7168), weight, norm),
        (prefix, result_buffer.flatten()[: 7168 * 3584].view_as(weight), norm),
        (prefix, weight, result_buffer.flatten()[:3584]),
    )
    for residual, projection, norm_weight in invalid_aliases:
        assert (
            iris_kimi3_moe_tail(
                *inputs,
                residual,
                projection,
                prefix_is_sharded=False,
                norm_weight=norm_weight,
                eps=1e-5,
                group=group,
            )
            is None
        )
    # A local prefix in the replicated result can be overwritten by another
    # owner's push before this rank consumes it, even at the exact base pointer.
    for first_row in (0, rank * (848 // 8)):
        assert (
            iris_kimi3_moe_tail(
                *inputs,
                result_buffer[first_row : first_row + 848 // 8],
                weight,
                prefix_is_sharded=True,
                norm_weight=norm,
                eps=1e-5,
                group=group,
            )
            is None
        )
    torch.testing.assert_close(state._producer_direct_ready_flags, flags_before)
    torch.testing.assert_close(state._moe_tail_ready_flags, gather_flags_before)
    for tensor, before in zip(inputs, inputs_before, strict=True):
        torch.testing.assert_close(tensor, before, atol=0, rtol=0)
    # Finish checking flags on every rank before the next collective changes them.
    dist.barrier()

    # Power-of-two scaling preserves BF16 rounding without a norm or prefix.
    base_sources = tuple(t.clone() for t in inputs)
    sources = tuple(t.clone() for t in base_sources)
    expected = ordinary(inputs, prefix, None)
    restore(inputs, sources)
    projected(inputs, prefix, None, prefix_is_sharded=False)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        restore(inputs, sources)
        graph_output = projected(inputs, prefix, None, prefix_is_sharded=False)
    snapshots = []
    ordinary_snapshots = []
    for iteration in range(6):
        scale = 2**iteration
        restore(sources, tuple(t * scale for t in base_sources))
        if rank == iteration:
            torch.cuda._sleep(100_000)
        graph.replay()
        snapshots.append((scale, graph_output.clone()))
        # Mix Lamport and pull collectives with different grids and epoch steps.
        other = acquire((1, 48, 520)[iteration % 3])
        for tensor in other:
            tensor.fill_(rank + 1)
        ordinary_snapshots.extend(comm.all_reduce_symmetric(backing, other))
    torch.cuda.synchronize()
    for scale, output in snapshots:
        torch.testing.assert_close(
            output, expected * scale, atol=0.03125, rtol=0.015625
        )
    for output in ordinary_snapshots:
        torch.testing.assert_close(output, torch.full_like(output, 36), atol=0, rtol=0)
    torch.testing.assert_close(held, held_expected, atol=0.03125, rtol=0.015625)

    # Consume each changed output immediately to check visibility across replays.
    restore(inputs, base_sources)
    exact = projected(inputs, prefix, None, prefix_is_sharded=False).clone()
    restore(inputs, tuple(-t for t in base_sources))
    negative = projected(inputs, prefix, None, prefix_is_sharded=False).clone()
    restore(sources, base_sources)
    result_snapshots = [torch.empty_like(prefix) for _ in range(4)]
    torch.cuda.synchronize()
    dist.barrier()
    result_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(result_graph):
        for iteration in range(4):
            for source in sources:
                source.neg_()
            restore(inputs, sources)
            if rank == (iteration * 3 + 1) % 8:
                torch.cuda._sleep(100_000)
            result = projected(inputs, prefix, None, prefix_is_sharded=False)
            assert result is not None
            result_snapshots[iteration].copy_(result)
    for _ in range(3):
        result_graph.replay()
        for iteration, result in enumerate(result_snapshots):
            oracle = negative if iteration % 2 == 0 else exact
            torch.testing.assert_close(
                result.view(torch.int16), oracle.view(torch.int16), atol=0, rtol=0
            )
    torch.cuda.synchronize()
    dist.barrier()
    state._producer_direct_ready_flags.fill_(-2)
    state._moe_tail_ready_flags.fill_(-2)
    torch.cuda.synchronize()
    dist.barrier()
    result_graph.replay()
    torch.testing.assert_close(result_snapshots[-1], exact, atol=0, rtol=0)

    # Replay in-place residual updates against a disjoint-prefix reference.
    initial_prefix = exact.clone()
    expected_chain = []
    residual = initial_prefix.clone()
    for iteration in range(4):
        residual.add_(0.125 * (iteration + 1))
        restore(inputs, base_sources)
        residual = projected(inputs, residual, norm, prefix_is_sharded=False).clone()
        expected_chain.append(residual.clone())
    inplace_prefix = state._moe_tail_output_buf[:848]
    chain_snapshots = [torch.empty_like(prefix) for _ in range(4)]
    torch.cuda.synchronize()
    dist.barrier()
    inplace_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(inplace_graph):
        inplace_prefix.copy_(initial_prefix)
        for iteration in range(4):
            if rank == (iteration * 3 + 1) % 8:
                torch.cuda._sleep(100_000)
            inplace_prefix.add_(0.125 * (iteration + 1))
            restore(inputs, base_sources)
            result = projected(inputs, inplace_prefix, norm, prefix_is_sharded=False)
            assert result is not None and result.data_ptr() == inplace_prefix.data_ptr()
            chain_snapshots[iteration].copy_(result)
    for _ in range(3):
        inplace_graph.replay()
        for result, oracle in zip(chain_snapshots, expected_chain, strict=True):
            torch.testing.assert_close(
                result.view(torch.int16), oracle.view(torch.int16), atol=0, rtol=0
            )
    assert tuple(t.data_ptr() for t in allocations) == allocation_pointers
    torch.cuda.synchronize()
    dist.barrier()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="Iris MoE tail assigns token rows across eight CDNA4 GPUs",
)
def test_iris_moe_tail() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_moe_tail_worker, args=(port,), nprocs=8, join=True)
