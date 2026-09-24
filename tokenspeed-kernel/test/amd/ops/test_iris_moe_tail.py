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

"""Producer ownership, lifetime and ordering of the token-sharded K3 tail."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform


def _moe_tail_worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    dist.barrier(group=group, device_ids=[rank])
    from tokenspeed_kernel.ops.communication import triton as comm
    from tokenspeed_kernel.ops.communication.iris import create_iris_ar_rmsnorm_state
    from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_latent_projection_add3
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm
    from tokenspeed_kernel.ops.moe.iris import iris_kimi3_moe_tail

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
    # EAGLE3 allocates this shared-heap buffer after K3's prefill state.
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
        state._reduced_output_buf,
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

    def projected(inputs, prefix, norm_weight):
        output = iris_kimi3_moe_tail(
            *inputs,
            prefix,
            weight,
            norm_weight=norm_weight,
            eps=1e-5 if norm_weight is not None else None,
            group=group,
        )
        assert output is not None
        return output

    held = None
    held_expected = None
    for rows in (512, 520, 848, 4096, 8144, 8192):
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
            borrowed = projected(inputs, prefix, norm_weight)
            assert borrowed.data_ptr() == state._moe_tail_output_buf.data_ptr()
            torch.testing.assert_close(borrowed, expected, atol=0.03125, rtol=0.015625)
            output = borrowed.clone()
            for tensor, source in zip(inputs, sources, strict=True):
                torch.testing.assert_close(tensor, source, atol=0, rtol=0)
            # Ordinary collectives preserve the borrowed result.
            ordinary(inputs, prefix, norm_weight)
            torch.testing.assert_close(borrowed, output, atol=0, rtol=0)
            # Exercise exact in-place reuse with a delayed prefix writer.
            restore(inputs, sources)
            if rank == rows % 8:
                torch.cuda._sleep(100_000)
            borrowed.copy_(prefix)
            inplace = projected(inputs, borrowed, norm_weight)
            assert inplace is not None and inplace.data_ptr() == borrowed.data_ptr()
            torch.testing.assert_close(
                inplace.view(torch.int16), output.view(torch.int16), atol=0, rtol=0
            )
            # An explicitly retained copy survives later tails.
            if held is not None:
                torch.testing.assert_close(
                    held, held_expected, atol=0.03125, rtol=0.015625
                )
            held, held_expected = output, expected

    # Rejection must occur before publishing any flags or consuming inputs.
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
                norm_weight=norm_weight,
                eps=eps,
                group=owner,
            )
            is None
        )
    torch.testing.assert_close(state._producer_direct_ready_flags, flags_before)
    # Only exact prefix/output aliasing preserves each tile's read/write
    # ownership. Reject shifted prefixes and aliased weights before launch.
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
                norm_weight=norm_weight,
                eps=1e-5,
                group=group,
            )
            is None
        )
    torch.testing.assert_close(state._producer_direct_ready_flags, flags_before)
    torch.testing.assert_close(state._moe_tail_ready_flags, gather_flags_before)
    for tensor, before in zip(inputs, inputs_before, strict=True):
        torch.testing.assert_close(tensor, before, atol=0, rtol=0)
    # A faster rank must not publish the next ordinary reduction while a
    # slower rank is still checking that these calls left its flags untouched.
    dist.barrier()

    # Scale by powers of two, with no norm or prefix, so the expected result
    # scales identically at every BF16 rounding boundary.
    base_sources = tuple(t.clone() for t in inputs)
    sources = tuple(t.clone() for t in base_sources)
    expected = ordinary(inputs, prefix, None)
    restore(inputs, sources)
    projected(inputs, prefix, None)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        restore(inputs, sources)
        graph_output = projected(inputs, prefix, None)
    snapshots = []
    ordinary_snapshots = []
    for iteration in range(6):
        scale = 2**iteration
        restore(sources, tuple(t * scale for t in base_sources))
        if rank == iteration:
            torch.cuda._sleep(100_000)
        graph.replay()
        snapshots.append((scale, graph_output.clone()))
        # Alternate Lamport, one-shot pull and two-stage pull without a host
        # rendezvous. Their program counts and epoch increments differ.
        other = acquire((1, 48, 520)[iteration % 3])
        for tensor in other:
            tensor.fill_(rank + 1)
        ordinary_snapshots.extend(
            tensor.clone() for tensor in comm.all_reduce_symmetric(backing, other)
        )
    torch.cuda.synchronize()
    for scale, output in snapshots:
        torch.testing.assert_close(
            output, expected * scale, atol=0.03125, rtol=0.015625
        )
    for output in ordinary_snapshots:
        torch.testing.assert_close(output, torch.full_like(output, 36), atol=0, rtol=0)
    torch.testing.assert_close(held, held_expected, atol=0.03125, rtol=0.015625)

    # Every captured invocation has changed data and an immediate consumer.
    # This warms output caches and checks receive acquire, not just the final
    # output of a fixed-input graph.
    restore(inputs, base_sources)
    exact = projected(inputs, prefix, None).clone()
    restore(inputs, tuple(-t for t in base_sources))
    negative = projected(inputs, prefix, None).clone()
    restore(sources, base_sources)
    borrowed_snapshots = [torch.empty_like(prefix) for _ in range(4)]
    torch.cuda.synchronize()
    dist.barrier()
    borrowed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(borrowed_graph):
        for iteration in range(4):
            for source in sources:
                source.neg_()
            restore(inputs, sources)
            if rank == (iteration * 3 + 1) % 8:
                torch.cuda._sleep(100_000)
            result = projected(inputs, prefix, None)
            assert result is not None
            borrowed_snapshots[iteration].copy_(result)
    for _ in range(3):
        borrowed_graph.replay()
        for iteration, result in enumerate(borrowed_snapshots):
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
    borrowed_graph.replay()
    torch.testing.assert_close(borrowed_snapshots[-1], exact, atol=0, rtol=0)

    # Replay in-place residual updates against a disjoint-prefix reference.
    initial_prefix = exact.clone()
    expected_chain = []
    residual = initial_prefix.clone()
    for iteration in range(4):
        residual.add_(0.125 * (iteration + 1))
        restore(inputs, base_sources)
        residual = projected(inputs, residual, norm).clone()
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
            result = projected(inputs, inplace_prefix, norm)
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
    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < 8,
    reason="Token-sharded Iris MoE requires eight CDNA4 GPUs",
)
def test_iris_moe_tail() -> None:
    pytest.importorskip("tokenspeed_kernel.ops.communication.iris")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_moe_tail_worker, args=(port,), nprocs=8, join=True)
