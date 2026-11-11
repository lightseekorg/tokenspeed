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


"""Muse Spark protocol selection, ownership, and changed-input graph replay."""

import re
import socket
import traceback
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import ArchVersion, current_platform
from utils import polling_cycles, votes_with_subgroup_mask

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="Muse Iris fusions require gfx950"
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int, reason_prefix: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"CUDA/ROCm is required for {reason_prefix}")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not current_platform().is_amd:
        pytest.skip(f"{reason_prefix} only targets AMD ROCm")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=args + (error_dict,),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _communication_worker(rank, world_size, port, check, backend, error_dict):
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend,
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        check(rank, world_size, device)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _column_all_gather_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        iris_pull_allgather_columns,
    )
    from tokenspeed_kernel.ops.quantization import rmsnorm_mxfp4_quant
    from utils import assert_no_triton_compile

    state = _muse_state(rank, device, 8, False)
    weight = torch.linspace(0.75, 1.25, 4096, dtype=torch.bfloat16, device=device)
    overwrite = torch.full(
        (1, (4096 // world_size)), -7 * (rank + 1), dtype=torch.bfloat16, device=device
    )
    # Published M1, ordinary rows, and three chunks exceeding the prepared capacity.
    for rows in (1, 4, 16, 33):
        if rows == 1:
            published = state.all_reduce.acquire_outputs(((1, 4096),))[0]
            local = published[
                :, rank * (4096 // world_size) : (rank + 1) * (4096 // world_size)
            ]
        else:
            local = torch.empty(
                (rows, (4096 // world_size)), dtype=torch.bfloat16, device=device
            )
        row_values = torch.arange(rows, dtype=torch.bfloat16, device=device)[:, None]
        expected = torch.cat(
            [
                (row_values + peer + 1).expand(-1, (4096 // world_size))
                for peer in range(world_size)
            ],
            dim=-1,
        )
        fused = rows <= 16 and current_platform().is_cdna4

        def run():
            gathered = state.gather.run(local)
            if rows <= 16:
                gathered = gathered.clone()  # Small outputs borrow shared storage.
            quantized = (
                state.all_gather_rmsnorm_mxfp4_quant(local, weight, 1e-6, 1.0)
                if fused
                else None
            )
            state.gather.run(overwrite)
            return gathered, quantized

        local.copy_(row_values + rank + 1)
        run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gathered, quantized = run()
        for scale in (2, -1, 3):
            local.copy_((row_values + rank + 1) * scale)
            graph.replay()
            torch.testing.assert_close(gathered, expected * scale, atol=0, rtol=0)
            if fused:
                reference = rmsnorm_mxfp4_quant(expected * scale, weight, 1e-6, 1.0)
                torch.testing.assert_close(quantized[0], reference[0], atol=0, rtol=0)
                torch.testing.assert_close(
                    _logical_cdna4_scales(quantized[1], rows, 128),
                    _logical_cdna4_scales(reference[1], rows, 128),
                    atol=0,
                    rtol=0,
                )
    with assert_no_triton_compile(iris_pull_allgather_columns):
        for rows in (3, 7):
            local = torch.full(
                (rows, (4096 // world_size)),
                rank + 1,
                dtype=torch.bfloat16,
                device=device,
            )
            expected = torch.cat(
                [torch.full_like(local, peer + 1) for peer in range(world_size)], dim=-1
            )
            torch.testing.assert_close(
                state.gather.run(local), expected, atol=0, rtol=0
            )

    # Packed fp32 words ride the BF16 gather as raw 16-bit halves: negative
    # zero, NaN payloads and halves equal to the mailbox's empty value (0x8000)
    # arrive bit for bit.
    def words(peer):
        pattern = torch.tensor(
            # 0x80000000, 0x7FC00001, 0x00008000, 0x80008000, 0x3F800000, ~0.
            [-(1 << 31), 0x7FC00001, 0x8000, 0x8000 - (1 << 31), 0x3F800000, -1],
            dtype=torch.int32,
            device=device,
        )
        return (pattern.repeat(7, 6)[:, :32] ^ (peer << 20)).contiguous()

    gathered = state.gather.run(words(rank).view(torch.bfloat16))
    expected = torch.cat([words(peer) for peer in range(world_size)], dim=-1)
    assert torch.equal(gathered.view(torch.int32), expected)
    if current_platform().is_cdna4:
        _check_all_reduce_routed_rmsnorm_sum(state, rank, world_size, device)


def _logical_cdna4_scales(
    scales: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Read the logical values from a padded CDNA4 scale allocation."""
    logical_m = torch.arange(rows, device=scales.device)[:, None]
    logical_k = torch.arange(cols, device=scales.device)[None, :]
    m_in_block = logical_m % 32
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = logical_k // 8
    k_hi = (logical_k % 8) // 4
    k_lo = logical_k % 4
    swizzled_k = (((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi
    offsets = swizzled_k + (logical_m // 32) * (cols * 32)
    return scales.view(torch.uint8).flatten()[offsets]


def _check_all_reduce_routed_rmsnorm_sum(state, rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum,
    )
    from tokenspeed_kernel.ops.moe import routed_input_scale_rmsnorm_sum_gathered
    from utils import assert_no_triton_compile

    rows, topk, width = 4, 4, 2048
    indices = torch.arange(rows * topk, dtype=torch.int32, device=device)
    columns = torch.arange(width, device=device)
    values = torch.sin(
        columns[None, :] * 0.003
        + (indices // topk)[:, None] * 0.1
        + (indices % topk)[:, None] * 0.2
    ).bfloat16()
    routes = torch.empty_like(values)
    mapping = torch.empty_like(indices)
    weights = (
        torch.tensor([0.1, 0.2, 0.3, 0.4], device=device)
        .expand(rows, topk)
        .contiguous()
    )
    norm = torch.linspace(0.75, 1.25, width, dtype=torch.bfloat16, device=device)

    def update(scale, empty_last_rank):
        owners = indices % topk
        if empty_last_rank:
            owners = torch.where(owners == 3, 0, owners)
        mapping.copy_(torch.where(owners == rank, indices, -1))
        routes.copy_(torch.where((owners == rank)[:, None], values * scale, torch.nan))
        partials = [
            routed_input_scale_rmsnorm_sum_gathered(
                values * scale,
                torch.where(owners == peer, indices, -1),
                rows,
                topk,
                weights,
                norm,
                1e-6,
                1.0,
                False,
                None,
            )
            for peer in range(world_size)
        ]
        reduced = partials[rank].float()
        for peer, value in enumerate(partials):
            if peer != rank:
                reduced.add_(value.float())
        return reduced.bfloat16()

    def run():
        return state.all_reduce_routed_rmsnorm_sum(
            routes, mapping, rows, topk, weights, norm, 1e-6, 1.0
        )

    state.all_reduce.producer_pull.ready_flags.fill_(torch.iinfo(torch.int32).max - 1)
    expected = update(1, False)
    torch.cuda.synchronize()
    dist.barrier()
    torch.testing.assert_close(run(), expected, atol=2e-2, rtol=2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for scale, empty in ((-1, False), (2, True)):
        expected = update(scale, empty)
        torch.cuda._sleep((rank + 1) * 20000)
        with assert_no_triton_compile(
            iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum
        ):
            graph.replay()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("world_size", [4, 8])
def test_iris_column_all_gather(world_size):
    """Concatenate model-shaped TP4/TP8 hidden shards with the Gluon transport."""
    _skip_if_unsupported(world_size, "Iris inner all-gather tests")
    port = _get_open_port()
    _spawn_and_collect(
        _communication_worker,
        (world_size, port, _column_all_gather_worker, "gloo"),
        world_size,
    )


def _lamport_gather_rmsnorm_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.layernorm import affine_rmsnorm

    state = _muse_state(rank, device, 64, True)
    width = 4096
    local_width = width // world_size
    weight = torch.linspace(-0.5, 0.5, width, device=device).to(torch.bfloat16)
    for rows in (1, 8, 16, 64):
        gen = torch.Generator(device=device).manual_seed(rows)
        full = torch.randn(rows, width, generator=gen, device=device).to(torch.bfloat16)
        local = torch.empty(rows, local_width, dtype=torch.bfloat16, device=device)

        def run():
            # The MXFP4 sibling shares the mailboxes and stage counters.
            state.all_gather_rmsnorm_mxfp4_quant(local, weight, 1e-5, 1.0)
            return state.all_gather_rmsnorm(local, weight, 1e-5, 1.0)

        assert state.can_all_gather_rmsnorm(local, weight)
        local.copy_(full[:, rank * local_width : (rank + 1) * local_width])
        run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        # Four replays cycle every mailbox stage with changed inputs.
        for scale in (2.0, -1.0, 0.5, 3.0):
            scaled = (full.float() * scale).to(torch.bfloat16)
            local.copy_(scaled[:, rank * local_width : (rank + 1) * local_width])
            graph.replay()
            reference = affine_rmsnorm(scaled, weight, 1e-5, 1.0, torch.bfloat16)
            # Another in-thread addition order: a rare element rounds the
            # other way, at most one BF16 ulp (adjacent bit patterns).
            # Adding zero turns a negative zero into zero on both sides.
            got, want = output + 0.0, reference + 0.0
            assert torch.equal(got.sign(), want.sign())
            ulps = got.view(torch.int16).int() - want.view(torch.int16).int()
            assert ulps.abs().max().item() <= 1


@pytest.mark.parametrize("world_size", [4, 8])
def test_iris_lamport_gather_rmsnorm_matches_affine_rmsnorm(world_size):
    """The fused gather + RMSNorm is affine_rmsnorm of the gathered rows."""
    _skip_if_unsupported(world_size, "Iris gather + RMSNorm tests")
    port = _get_open_port()
    _spawn_and_collect(
        _communication_worker,
        (world_size, port, _lamport_gather_rmsnorm_worker, "gloo"),
        world_size,
    )


def _muse_state(rank, device, max_rows, enable_lamport):
    from tokenspeed_kernel.ops.communication._iris.muse_spark import (
        prepare_muse_communication,
    )

    return prepare_muse_communication(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=max_rows * 8192,
        enable_lamport=enable_lamport,
        owned_route_max_rows=0,
        intermediate_gather_max_rows=0,
        all_reduce=None,
        device=device,
    )


def _boundary_reference(stream, reduced, post, alpha, beta, pre):
    from tokenspeed_kernel.ops.residual._triton.gated_boundary import (
        triton_post_norm_residual_pre_norm,
    )

    return triton_post_norm_residual_pre_norm(
        stream,
        reduced,
        post,
        1e-6,
        1.0,
        alpha,
        beta,
        pre,
        1e-6,
        1.0,
        False,
        None,
        1,
        None,
    )


def _muse_boundary_worker(rank, world_size, device, sharded):
    from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
        iris_fused_lamport_allreduce_muse_attntail,
        iris_fused_pull_allgather_muse_attntail,
    )
    from tokenspeed_kernel.ops.communication._iris.reduce_scatter import (
        iris_pull_reduce_scatter,
    )
    from utils import assert_no_triton_compile

    state = _muse_state(rank, device, 8192 if sharded else 32, sharded)
    torch.manual_seed(101)
    pre = torch.randn(8192, device=device).bfloat16()
    post = torch.randn_like(pre)
    beta = torch.sigmoid(torch.randn(8192, device=device))
    alpha = torch.sqrt(1 - beta.square())
    rows_sequence = (512, 1, 8191, 21, 513, 8192) if sharded else (26, 1, 25, 2)
    bases = [torch.randn(rows, 8192, device=device) for rows in rows_sequence]
    # Every rank sends negative zero in some columns; the push path must not
    # mistake it for its empty mailbox marker.
    negative_zero = torch.zeros(8192, dtype=torch.bool, device=device)
    negative_zero[::97] = True

    def contribution(base, peer, step):
        values = (base + (peer + 1) * (step + 1) / 8).bfloat16()
        values[:, negative_zero] = -0.0
        return values

    inputs = [contribution(base, rank, 0) for base in bases]
    streams = [torch.randn_like(base) for base in bases]
    posts = [
        None if sharded or index % 2 == 0 else post for index in range(len(inputs))
    ]

    def boundary(values, stream, post_weight):
        branch = state.all_reduce.acquire_outputs((tuple(values.shape),))[0]
        branch.copy_(values)
        return state.all_reduce_symmetric_boundary(
            branch,
            stream,
            post_weight,
            alpha,
            beta,
            pre,
            1e-6,
            1.0,
            1e-6,
            1.0,
        )

    def sequence():
        outputs = []
        for index, (values, stream, weight) in enumerate(zip(inputs, streams, posts)):
            outputs.append(boundary(values, stream, weight))
            if sharded:
                # Alternate Lamport and pull reductions over the boundary's scratch.
                n = 173 * 8192 if index % 2 else 43 * 4096
                generic = state.all_reduce.acquire_outputs(((n,),))[0]
                generic.fill_(rank + 1)
                state.reduce_outputs((generic,))
        return outputs

    sequence()
    if sharded:
        state.row_scatter_flags.fill_(2**32 - 2)
        state.row_gather_flags.fill_(2**32 - 2)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = sequence()
    for step in range(4):
        for base, values in zip(bases, inputs):
            values.copy_(contribution(base, rank, step))
        if rank == step:
            torch.cuda._sleep(10000)
        graph.replay()
        for base, stream, weight, actual in zip(bases, streams, posts, outputs):
            reduced = contribution(base, 0, step).float()
            for peer in range(1, world_size):
                reduced.add_(contribution(base, peer, step).float())
            expected = _boundary_reference(
                stream, reduced.bfloat16(), weight, alpha, beta, pre
            )
            for got, reference in zip(actual, expected):
                torch.testing.assert_close(
                    got,
                    reference,
                    atol=(
                        0
                        if sharded
                        else (1e-5 if got.dtype == torch.float32 else 0.015625)
                    ),
                    rtol=(
                        0
                        if sharded
                        else (2e-5 if got.dtype == torch.float32 else 0.008)
                    ),
                )
    if sharded:
        with assert_no_triton_compile(
            iris_pull_reduce_scatter, iris_fused_pull_allgather_muse_attntail
        ):
            for rows in (1025, 4096):
                values = torch.ones((rows, 8192), device=device, dtype=torch.bfloat16)
                boundary(values, torch.zeros_like(values, dtype=torch.float32), None)
    else:
        # Rows reach the kernel only through the grid and the two-valued
        # peer-rotation bucket. The sequence above compiled rows 1/2 with a
        # post-norm weight and rows 25/26 without one.
        with assert_no_triton_compile(iris_fused_lamport_allreduce_muse_attntail):
            for rows, weight in ((3, post), (7, None)):
                values = torch.ones((rows, 8192), device=device, dtype=torch.bfloat16)
                boundary(values, torch.zeros_like(values, dtype=torch.float32), weight)
    torch.cuda.synchronize()
    dist.barrier()


def _muse_collective_policy_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_lamport_bf16_allreduce,
        iris_lamport_twoshot_bf16_allreduce,
        iris_pull_twoshot_allreduce,
        iris_push_twoshot_allreduce,
    )
    from utils import assert_no_triton_compile

    state = _muse_state(rank, device, 8192, True)
    counts = (
        512,
        43 * 4096,
        44 * 4096,
        21 * 8192,
        22 * 8192,
        26 * 8192,
        27 * 8192,
        128 * 8192,
        129 * 8192,
        172 * 8192,
        173 * 8192,
        8191 * 8192,
        8192 * 8192,
        32,
    )
    boundary_base = (
        torch.arange(26 * 8192, device=device).view(26, 8192) % 31
    ).float() / 32
    boundary_values = (boundary_base + (rank + 1) / 8).bfloat16()
    boundary_stream = torch.zeros_like(boundary_base)
    coefficients = torch.ones(8192, device=device)
    pre_weight = torch.zeros(8192, device=device, dtype=torch.bfloat16)

    inputs = [
        torch.full((n,), rank + 1, dtype=torch.bfloat16, device=device) for n in counts
    ]

    def sequence():
        outputs = []
        for index, values in enumerate(inputs):
            branch = state.all_reduce.acquire_outputs((tuple(values.shape),))[0]
            branch.copy_(values)
            outputs.append(state.reduce_outputs((branch,))[0].clone())
            if index == 5:
                boundary_branch = state.all_reduce.acquire_outputs(((26, 8192),))[0]
                boundary_branch.copy_(boundary_values)
                fused = tuple(
                    output.clone()
                    for output in state.all_reduce_symmetric_boundary(
                        boundary_branch,
                        boundary_stream,
                        None,
                        coefficients,
                        coefficients,
                        pre_weight,
                        1e-6,
                        1.0,
                        1e-6,
                        1.0,
                    )
                )
        return outputs, fused

    sequence()
    state.reduction.epochs.fill_(2**32 - 4)
    state.reduction.ready_flags.fill_(2**32 - 4)
    state.reduction.lamport_epochs.fill_(2)
    if state.reduction.lamport_two_shot_epochs is not None:
        state.reduction.lamport_two_shot_epochs.fill_(2)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs, fused = sequence()
    for step in range(7):
        for values in inputs:
            values.fill_((rank + 1) * (step + 1) / 8)
        boundary_values.copy_(boundary_base + (rank + 1) * (step + 1) / 8)
        if rank == step % world_size:
            torch.cuda._sleep(10000)
        graph.replay()
        for output in outputs:
            torch.testing.assert_close(
                output,
                torch.full_like(
                    output, (world_size * (world_size + 1) // 2) * (step + 1) / 8
                ),
                rtol=0,
                atol=0,
            )
        reduced = (
            world_size * boundary_base
            + (world_size * (world_size + 1) // 2) * (step + 1) / 8
        ).bfloat16()
        expected = _boundary_reference(
            boundary_stream, reduced, None, coefficients, coefficients, pre_weight
        )
        for actual, reference in zip(fused, expected):
            torch.testing.assert_close(
                actual,
                reference,
                atol=1e-5 if actual.dtype == torch.float32 else 0.015625,
                rtol=2e-5 if actual.dtype == torch.float32 else 0.008,
            )
    with assert_no_triton_compile(
        iris_lamport_bf16_allreduce,
        iris_lamport_twoshot_bf16_allreduce,
        iris_push_twoshot_allreduce,
        iris_pull_twoshot_allreduce,
    ):
        for rows in (3, 17, 127, 130):
            ordinary = torch.full(
                (rows, 8192), rank + 1, device=device, dtype=torch.bfloat16
            )
            result = state.reduce(ordinary)
            torch.testing.assert_close(
                result,
                torch.full_like(result, world_size * (world_size + 1) // 2),
                rtol=0,
                atol=0,
            )
            branch = state.all_reduce.acquire_outputs(((rows, 8192),))[0]
            branch.fill_(rank + 1)
            output = state.reduce_outputs((branch,))[0]
            torch.testing.assert_close(
                output,
                torch.full_like(output, world_size * (world_size + 1) // 2),
                rtol=0,
                atol=0,
            )
    for numel in (512, 44 * 4096, 129 * 8192):
        ordinary = torch.full((numel,), rank + 1, dtype=torch.bfloat16, device=device)
        result = state.reduce(ordinary)
        assert result.data_ptr() == ordinary.data_ptr()
        torch.testing.assert_close(
            ordinary,
            torch.full_like(ordinary, world_size * (world_size + 1) // 2),
            rtol=0,
            atol=0,
        )


def _muse_pull_warmup_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_pull_twoshot_allreduce,
    )
    from utils import assert_no_triton_compile

    # Past the push and Lamport two-shot windows, below the owned-row path:
    # the producer pull reduces these rows.
    rows = 179
    state = _muse_state(rank, device, 192, True)
    state.warmup_reduce_outputs()
    with assert_no_triton_compile(iris_pull_twoshot_allreduce):
        branch = state.all_reduce.acquire_outputs(((rows, 8192),))[0]
        branch.fill_(rank + 1)
        output = state.reduce_outputs((branch,))[0]
    torch.testing.assert_close(
        output,
        torch.full_like(output, world_size * (world_size + 1) // 2),
        rtol=0,
        atol=0,
    )


def _muse_owned_packet_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.muse_spark.all_gather import (
        iris_pack_moe_inputs,
        iris_pull_allgather_moe_inputs,
    )
    from tokenspeed_kernel.ops.quantization import rmsnorm_mxfp4_quant
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
        swizzle_cdna4_mxfp4_scale,
    )
    from utils import assert_no_triton_compile

    state = _muse_state(rank, device, 8192, True)
    norm_weight = torch.linspace(-0.25, 0.25, 4096, device=device).bfloat16()

    def make_inputs(rows):
        local_rows = (rows + world_size - 1) // world_size
        first = rank * local_rows
        valid_rows = min(local_rows, max(0, rows - first))
        local = torch.empty((local_rows, 4096), device=device, dtype=torch.bfloat16)
        ids = torch.empty((local_rows, 8), device=device, dtype=torch.int32)
        weights = torch.empty((local_rows, 8), device=device, dtype=torch.float32)
        branch = torch.empty((rows, 8192), device=device, dtype=torch.bfloat16)
        return rows, first, valid_rows, local, ids, weights, branch

    inputs = [make_inputs(rows) for rows in (193, 8192)]

    def update(item, step):
        rows, first, valid_rows, local, ids, weights, branch = item
        global_row = torch.arange(rows, device=device)[:, None]
        column = torch.arange(4096, device=device)[None, :]
        full = (
            ((global_row * 37 + column * 11 + step * 13) % 257 - 128) / 64
        ).bfloat16()
        full_ids = (
            (global_row + torch.arange(8, device=device)[None, :] + step) % 256
        ).int()
        full_weights = (
            (global_row % 31 + torch.arange(8, device=device)[None, :] + step) / 64
        ).float()
        local.fill_(float("nan"))
        ids.fill_(-2147483648)
        weights.fill_(float("nan"))
        local[:valid_rows].copy_(full[first : first + valid_rows])
        ids[:valid_rows].copy_(full_ids[first : first + valid_rows])
        weights[:valid_rows].copy_(full_weights[first : first + valid_rows])
        branch.fill_((rank + 1) * (step + 1))
        return full, full_ids, full_weights

    def one(item):
        rows, _, valid_rows, local, ids, weights, branch = item
        reduced = state.reduce_scatter_owned_rows(branch)
        retained = state.all_gather_owned_rows(reduced, rows, True)
        quantized = rmsnorm_mxfp4_quant(local, norm_weight, 1e-8, 1.0)
        # Pull peers may still read scratch when this rank starts packet packing.
        generic = state.all_reduce.acquire_outputs(((129, 8192),))[0]
        generic.copy_(branch[:1])
        generic_output = state.reduce_outputs((generic,))[0]
        packed_result = state.gather_owned_moe_inputs(
            quantized,
            ids,
            weights,
            rows,
            valid_rows,
        )
        bf16_result = state.gather_owned_moe_inputs(
            local,
            ids,
            weights,
            rows,
            valid_rows,
        )
        generic_snapshot = generic_output.clone()
        # The next scatter overwrites packet staging before outputs are checked.
        state.reduce_scatter_owned_rows(branch)
        return retained, packed_result, bf16_result, generic_snapshot

    for item in inputs:
        update(item, 0)
        one(item)
    state.row_gather_flags.fill_(2**32 - 2)
    state.row_scatter_flags.fill_(2**32 - 2)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [one(item) for item in inputs]
    for step in range(4):
        expected = [update(item, step) for item in inputs]
        if rank == step:
            torch.cuda._sleep(10000)
        graph.replay()
        for item, result, reference in zip(inputs, outputs, expected):
            rows = item[0]
            retained, packed_result, bf16_result, generic = result
            full, full_ids, full_weights = reference
            torch.testing.assert_close(
                generic,
                torch.full_like(
                    generic, (world_size * (world_size + 1) // 2) * (step + 1)
                ),
                atol=0,
                rtol=0,
            )
            expected_packed, expected_scales = rmsnorm_mxfp4_quant(
                full, norm_weight, 1e-8, 1.0
            )
            valid_scale_mask = swizzle_cdna4_mxfp4_scale(
                torch.ones((rows, 128), device=device, dtype=torch.uint8)
            )[0].bool()
            actual_packed, actual_scales = packed_result[0]
            torch.testing.assert_close(actual_packed, expected_packed, atol=0, rtol=0)
            torch.testing.assert_close(
                actual_scales[valid_scale_mask],
                expected_scales[valid_scale_mask],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(bf16_result[0], full, atol=0, rtol=0)
            for _, actual_ids, actual_weights in (packed_result, bf16_result):
                assert actual_ids.is_contiguous() and actual_weights.is_contiguous()
                torch.testing.assert_close(actual_ids, full_ids, atol=0, rtol=0)
                torch.testing.assert_close(actual_weights, full_weights, atol=0, rtol=0)
            torch.testing.assert_close(
                retained,
                torch.full_like(
                    retained, (world_size * (world_size + 1) // 2) * (step + 1)
                ),
                atol=0,
                rtol=0,
            )
    with assert_no_triton_compile(iris_pack_moe_inputs, iris_pull_allgather_moe_inputs):
        item = make_inputs(2049)
        update(item, 7)
        one(item)
    torch.cuda.synchronize()


def _compile_gfx950(fn, types, constexprs, num_warps, unspecialized=(), options=None):
    """Compile a Gluon kernel for gfx950 without launching it.

    Arguments named in unspecialized get no 16-byte divisibility, as when a
    caller passes them in do_not_specialize or compiles ahead of time.
    Pointers the kernel annotates ``tl.const`` compile as const pointers, as
    at launch. ``options`` adds compile options. Every compile links the
    extern intrinsic libraries.
    """
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel_amd._extern import extern_compile_options

    types = dict(types)
    for param in fn.params:
        kind = types.get(param.name)
        if param.is_const and isinstance(kind, str) and kind.startswith("*"):
            types[param.name] = "*k" + kind[1:]
    attrs = {}
    for index, name in enumerate(fn.arg_names):
        if isinstance(types.get(name), tuple):
            attrs.update(
                {
                    (index, peer): [["tt.divisibility", 16]]
                    for peer in range(len(types[name]))
                }
            )
        elif name in types and name not in unspecialized:
            attrs[(index,)] = [["tt.divisibility", 16]]
    source = gluon._runtime.GluonASTSource(
        fn,
        {name: types[name] for name in fn.arg_names if name in types},
        constexprs=constexprs,
        attrs=attrs,
    )
    return triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={
            "num_warps": num_warps,
            **extern_compile_options(),
            **(options or {}),
        },
    )


def _batched_peer_reads(cycle, reads):
    """Return whether a cycle issues all system-scope peer reads before waiting."""
    loads = [
        index
        for index, line in enumerate(cycle)
        if "buffer_load_dwordx4" in line and "sc0 sc1" in line
    ]
    return len(loads) == reads and not any(
        "s_waitcnt vmcnt(0)" in line for line in cycle[loads[0] : loads[-1]]
    )


@pytest.mark.parametrize("rank", [0, 5])
def test_routed_lamport_polling_codegen(rank):
    """Keep the mailbox reads batched and on the retry path.

    Compile without launching. Each retry must issue all seven 16-byte
    cache-bypassing peer reads before waiting for any of them; one wait per
    read would serialize seven memory round trips. The retry decision must
    be the subgroup's lane mask rather than a cross-lane reduction.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_lamport_allreduce_routed_rmsnorm_sum as fn,
    )

    policy = iris_arch_policy(ArchVersion(9, 5))
    types = {
        "routes": "*bf16",
        "route_map": "*i32",
        "route_weights": "*fp32",
        "norm_weight": "*bf16",
        "output": "*bf16",
        "region": "*bf16",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "row_stride": "i32",
    }
    kernel = _compile_gfx950(
        fn,
        types,
        {
            "RANK": rank,
            "WIDTH": 4096,
            "EPS": 1e-5,
            "MAX_ELEMENTS": 43 * 4096,
            "PORTABLE_MATH": False,
            "PEER_SIGNAL_STORE_CACHE": policy.peer_signal_store,
            "WAIT_STORES": policy.wait_stores,
        },
        num_warps=1,
    )
    assembly = kernel.asm["amdgcn"]
    lines = assembly.splitlines()
    cycles = polling_cycles(assembly)
    assert cycles, "polling loop was removed"
    polls = [cycle for cycle in cycles if _batched_peer_reads(cycle, 7)]
    assert polls, "all seven system-cache reads must issue together on a retry"
    assert all(
        votes_with_subgroup_mask(cycle) for cycle in polls
    ), "a retry must branch on one subgroup lane mask"
    assert (
        sum("buffer_store_dwordx4" in line and "sc0 sc1" in line for line in lines)
        == 14
    ), "publication and reset must use 16-byte system-cache stores"
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def test_routed_pull_stores_reach_l2_before_release():
    """Every subgroup's routed sum reaches L2 before the readiness release.

    All eight subgroups store the token sum into the symmetric input, and the
    release run by one thread waits only for its own subgroup's stores.
    Compile without launching and check that, on every path, each subgroup
    runs s_waitcnt vmcnt(0) before the barrier preceding the release. The
    later completion release only reports returned peer reads, so it needs
    just a barrier after them.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum as fn,
    )
    from utils import vmcnt_zero_before_barrier

    types = {
        "routes": "*bf16",
        "route_map": "*i32",
        "route_weights": "*fp32",
        "norm_weight": "*bf16",
        "symmetric_input": "*bf16",
        "output": "*bf16",
        "ready_flags": "*i32",
        "heaps": ("i64",) * 8,
        "row_stride": "i32",
    }
    kernel = _compile_gfx950(
        fn,
        types,
        {
            "rank": 3,
            "width": 4096,
            "topk": 8,
            "eps": 1e-5,
            "weight_offset": 1.0,
            "block": 4096,
            "num_warps": 8,
            "WAIT_STORES": iris_arch_policy(ArchVersion(9, 5)).wait_stores,
        },
        num_warps=8,
    )
    releases = vmcnt_zero_before_barrier(
        kernel.asm["amdgcn"], fn.fn.__name__, r"\bglobal_atomic_add\b.*sc0 sc1"
    )
    assert len(releases) == 2, releases
    assert releases[0][1], releases
    assert releases[1][2], releases


def test_route_sum_gather_lamport_polling_codegen():
    """Keep the expert TP8 route-sum gather's polls batched on the retry path.

    Each subgroup polls its own 16-byte packs: a retry issues all seven peer
    reads before waiting and branches on one subgroup lane mask, with no
    workgroup barrier inside the loop.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_routed_rmsnorm_sum_lamport_allgather,
    )

    policy = iris_arch_policy(ArchVersion(9, 5))
    types = {
        "routes": "*bf16",
        "route_weights": "*fp32",
        "norm_weight": "*bf16",
        "output": "*bf16",
        "region": "*bf16",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "rows": "i32",
    }
    constexprs = {
        "RANK": 5,
        "WIDTH": 4096,
        "EPS": 1e-5,
        "WEIGHT_OFFSET": 1.0,
        "MAX_ELEMENTS": 43 * 4096,
        "PEER_SIGNAL_STORE_CACHE": policy.peer_signal_store,
        "WAIT_STORES": policy.wait_stores,
    }
    kernel = _compile_gfx950(
        iris_fused_routed_rmsnorm_sum_lamport_allgather,
        types,
        constexprs,
        8,
        unspecialized=("rows",),
    )
    assembly = kernel.asm["amdgcn"]
    polls = [
        cycle for cycle in polling_cycles(assembly) if _batched_peer_reads(cycle, 7)
    ]
    assert polls, "all seven system-cache reads must issue together on a retry"
    assert all(votes_with_subgroup_mask(cycle) for cycle in polls)
    assert not any("s_barrier" in line for cycle in polls for line in cycle)
    # Seven pushes and seven mailbox resets.
    assert (
        sum(
            bool(re.search(r"\bbuffer_store_dwordx4\b", line)) and "sc0 sc1" in line
            for line in assembly.splitlines()
        )
        == 14
    ), "publication and reset must use system-cache stores"
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def _attntail_lamport_assembly() -> str:
    """gfx950 assembly of the 16-subgroup Lamport attention tail at TP8."""
    rank = 5
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
        iris_fused_lamport_allreduce_muse_attntail as fn,
    )

    pointers = {
        "branch_ptr": "*bf16",
        "mailbox_ptr": "*bf16",
        "stream_ptr": "*fp32",
        "post_weight_ptr": "*bf16",
        "alpha_ptr": "*fp32",
        "beta_ptr": "*fp32",
        "pre_weight_ptr": "*bf16",
        "out_stream_ptr": "*fp32",
        "pre_fp32_ptr": "*fp32",
        "pre_bf16_ptr": "*bf16",
        "epochs": "*i32",
        "heaps": ("i64",) * 8,
    }
    kernel = _compile_gfx950(
        fn,
        pointers,
        {
            "RANK": rank,
            "WORLD_SIZE": 8,
            "WIDTH": 8192,
            "MAX_ROWS": 26,
            "POST_EPS": 1e-6,
            "POST_WEIGHT_OFFSET": 1.0,
            "PRE_EPS": 1e-6,
            "PRE_WEIGHT_OFFSET": 1.0,
            "POST_HAS_WEIGHT": True,
            "SENDERS": 8,
            "ROTATE_PEERS": False,
            "ELEMENTS_PER_THREAD": 8,
            "NUM_WARPS": 16,
            "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
                ArchVersion(9, 5)
            ).peer_signal_store,
        },
        num_warps=16,
    )
    return kernel.asm["amdgcn"]


def test_attntail_lamport_polling_codegen():
    """Each subgroup of the 16-subgroup poller retries on its own.

    A workgroup-wide readiness reduction would put shared-memory traffic and
    barriers into every retry; the subgroup ballot keeps them out.
    """
    assembly = _attntail_lamport_assembly()
    polls = [
        cycle for cycle in polling_cycles(assembly) if _batched_peer_reads(cycle, 7)
    ]
    assert polls, "all seven system-cache reads must issue together on a retry"
    assert all(
        votes_with_subgroup_mask(cycle) for cycle in polls
    ), "a retry must branch on one subgroup lane mask"
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def test_attntail_stage_store_follows_a_barrier():
    """Every wave loads its workgroup's stage; one lane stores the next.

    A wave that loaded after that store would use the next stage's mailbox
    slot, so a workgroup barrier must precede the store on every path: it
    sits in the block that ends in the store.
    """
    lines = _attntail_lamport_assembly().splitlines()
    stores = [i for i, line in enumerate(lines) if "global_store_dword " in line]
    assert len(stores) == 1, "expected one global store (the stage)"
    index = stores[0]
    # Walk back through fall-through-only blocks ("; %bb.N:") to a block
    # that branches may enter (".LBB").
    while index > 0 and not lines[index].startswith(".LBB"):
        if "s_barrier" in lines[index]:
            break
        index -= 1
    assert "s_barrier" in lines[index], "the stage store does not follow a barrier"


def _sliding_tail_assembly() -> str:
    """gfx950 assembly of the fused sliding tail at 8 rows, TP8."""
    from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
        iris_fused_sliding_attention_twoshot_muse_attntail as fn,
    )
    from tokenspeed_kernel_amd._extern import extern_compile_options

    pointers = {
        name: kind
        for name, kind in {
            "q_ptr": "*fp8e4nv",
            "k_cache_ptr": "*fp8e4nv",
            "v_cache_ptr": "*fp8e4nv",
            "page_table_ptr": "*i32",
            "cache_seqlens_ptr": "*i32",
            "mid_o_ptr": "*fp32",
            "mid_lse_ptr": "*fp32",
            "counters_ptr": "*i32",
            "gate_ptr": "*bf16",
            "weight_ptr": "*bf16",
            "att_buffer": "*i32",
            "epochs": "*i32",
            "region": "*bf16",
            "rs_peers": ("i64",) * 8,
            "sums": "*i64",
            "sums_peers": ("i64",) * 8,
            "stream_ptr": "*fp32",
            "post_weight_ptr": "*bf16",
            "alpha_ptr": "*fp32",
            "beta_ptr": "*fp32",
            "pre_weight_ptr": "*bf16",
            "out_stream_ptr": "*fp32",
            "pre_fp32_ptr": "*fp32",
            "pre_bf16_ptr": "*bf16",
        }.items()
    }
    pointers.update({"gate_stride": "i32", "rows": "i32"})
    kernel = _compile_gfx950(
        fn,
        pointers,
        dict(
            Q_STRIDE_B=1024,
            Q_STRIDE_H=64,
            Q_STRIDE_D=1,
            SM_SCALE=0.125 * 1.4426950408889634,
            PAGE_TABLE_STRIDE=4096,
            PAGE_SIZE=64,
            NUM_KV_SPLITS=8,
            MAX_SEQLEN_Q=8,
            NUM_Q_HEADS=16,
            NUM_KV_HEADS=2,
            HEAD_DIM=64,
            BLOCK_M=64,
            BLOCK_N=64,
            WINDOW_LEFT=2047,
            IS_FP8=True,
            NUM_REQUESTS=1,
            WIDTH=8192,
            RANK=3,
            WORLD_SIZE=8,
            MAX_SHARD_ELEMENTS=16 * 1024,
            MAX_ROWS=16,
            PARTS=8,
            GATE_EPS=1e-6,
            POST_EPS=1e-6,
            POST_WEIGHT_OFFSET=1.0,
            PRE_EPS=1e-6,
            PRE_WEIGHT_OFFSET=1.0,
            POST_HAS_WEIGHT=True,
            RESERVED_LDS=128 * 1024,
            NUM_STAGES=2,
            PEER_SIGNAL_STORE_CACHE=".wt",
        ),
        num_warps=4,
        options=extern_compile_options(),
    )
    return kernel.asm["amdgcn"]


def test_sliding_attention_tail_epoch_store_follows_a_barrier():
    """Every subgroup of a fused sliding-tail workgroup loads its epoch; one
    lane stores the next. A subgroup that loaded after that store would push
    with the next stage and tag (or clear the current stage's all-gather
    slots), so a workgroup barrier must precede the store on every path: it
    sits in the block that ends in the store."""
    lines = _sliding_tail_assembly().splitlines()
    stores = [i for i, line in enumerate(lines) if "global_store_dword " in line]
    assert stores, "expected the epoch's global store"
    index = stores[-1]
    # Walk back through fall-through-only blocks to a block that branches may
    # enter (".LBB").
    while index > 0 and not lines[index].startswith(".LBB"):
        if "s_barrier" in lines[index]:
            break
        index -= 1
    assert "s_barrier" in lines[index], "the epoch store does not follow a barrier"


def test_sliding_attention_tail_hand_offs_bypass_every_l2():
    """No load or store of the fused sliding tail uses agent scope alone.

    Its hand-offs cross XCDs (and GPUs), and each XCD has its own L2 that
    does not snoop the others: a hand-off word read or written with SC1 but
    not SC0 may hit a stale line in the reader's L2. Hand-offs use SC0 SC1,
    which bypasses every L2 coherently (CDNA4 ISA 9.1.10.2); everything else
    uses neither bit.
    """
    agent_only = [
        line.strip()
        for line in _sliding_tail_assembly().splitlines()
        if re.match(r"\s*(global|buffer)_(load|store)", line)
        and re.search(r"\bsc1\b", line)
        and not re.search(r"\bsc0\b", line)
    ]
    assert not agent_only, agent_only


def test_sliding_attention_tail_stage_is_unsigned():
    """The fused sliding tail takes its stage (epoch % 3) as unsigned.

    A signed remainder of an epoch word past 2**31 - 1 is negative, which
    moves the two-shot slots below the region. LLVM divides by three with a
    multiply-high by a magic constant: 0x55555556 (s/v_mul_hi_i32) for a
    signed remainder, 0xaaaaaaab (s/v_mul_hi_u32) for an unsigned one.
    """
    assembly = _sliding_tail_assembly()
    signed = re.findall(r"_mul_hi_i32\b.*0x55555556", assembly)
    assert not signed, signed
    assert re.search(r"_mul_hi_u32\b.*0xaaaaaaab", assembly), "no unsigned % 3"


@pytest.mark.parametrize("rows", [8, 16])
def test_sliding_attntail_page_loads_keep_copies_in_flight(rows):
    """The fused sliding tail's page-index loads do not drain the KV copies.

    The split loads each page index two pages ahead with a volatile load, so
    the next pages' K/V copies stay in flight. LLVM follows a volatile load
    from global memory with `s_waitcnt vmcnt(0)`, which would wait for every
    copy issued before it; from a const page table it does not.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
        iris_fused_sliding_attention_twoshot_muse_attntail as fn,
    )
    from tokenspeed_kernel_amd._extern import extern_compile_options

    types = {
        "q_ptr": "*fp8e4nv",
        "k_cache_ptr": "*fp8e4nv",
        "v_cache_ptr": "*fp8e4nv",
        "page_table_ptr": "*i32",
        "cache_seqlens_ptr": "*i32",
        "mid_o_ptr": "*fp32",
        "mid_lse_ptr": "*fp32",
        "counters_ptr": "*i32",
        "gate_ptr": "*bf16",
        "gate_stride": "i32",
        "weight_ptr": "*bf16",
        "att_buffer": "*bf16",
        "epochs": "*i32",
        "region": "*bf16",
        "rs_peers": ("i64",) * 8,
        "sums": "*i64",
        "sums_peers": ("i64",) * 8,
        "stream_ptr": "*fp32",
        "post_weight_ptr": "*bf16",
        "alpha_ptr": "*fp32",
        "beta_ptr": "*fp32",
        "pre_weight_ptr": "*bf16",
        "out_stream_ptr": "*fp32",
        "pre_fp32_ptr": "*fp32",
        "pre_bf16_ptr": "*bf16",
        "rows": "i32",
    }
    # The 4-wave sliding split of 8 or 16 verify rows over a 2048-token
    # window of 64-token FP8 pages, as sliding_attention_boundary selects it.
    kernel = _compile_gfx950(
        fn,
        types,
        {
            "Q_STRIDE_B": 1024,
            "Q_STRIDE_H": 64,
            "Q_STRIDE_D": 1,
            "SM_SCALE": 0.125,
            "PAGE_TABLE_STRIDE": 4096,
            "PAGE_SIZE": 64,
            "NUM_KV_SPLITS": 8,
            "MAX_SEQLEN_Q": 8,
            "NUM_Q_HEADS": 16,
            "NUM_KV_HEADS": 2,
            "HEAD_DIM": 64,
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "WINDOW_LEFT": 2047,
            "IS_FP8": True,
            "NUM_REQUESTS": rows // 8,
            "WIDTH": 8192,
            "RANK": 5,
            "WORLD_SIZE": 8,
            "MAX_SHARD_ELEMENTS": 16 * 1024,
            "MAX_ROWS": 16,
            "PARTS": 8 if rows <= 8 else 4,
            "GATE_EPS": 1e-6,
            "POST_EPS": 1e-6,
            "POST_WEIGHT_OFFSET": 1.0,
            "PRE_EPS": 1e-6,
            "PRE_WEIGHT_OFFSET": 1.0,
            "POST_HAS_WEIGHT": True,
            "RESERVED_LDS": 128 * 1024,
            "NUM_STAGES": 2,
            "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
                ArchVersion(9, 5)
            ).peer_signal_store,
        },
        num_warps=4,
        unspecialized=("rows", "gate_stride"),
        options=extern_compile_options(),
    )
    lines = [
        line.strip()
        for line in kernel.asm["amdgcn"].splitlines()
        if line.strip() and not line.strip().startswith((".", ";"))
    ]
    memory = re.compile(r"^(?:global|buffer)_(?:load|store|atomic)")
    # Poll loops (the finisher's sums) also load with SC0 SC1 and wait for
    # each load; they are not page-index loads. The page loop holds MFMAs.
    polled = {
        line.strip()
        for cycle in polling_cycles(kernel.asm["amdgcn"])
        if not any("v_mfma" in line for line in cycle)
        for line in cycle
    }
    page_loads = 0
    for index, line in enumerate(lines):
        if not (line.startswith("global_load_dword ") and "sc0 sc1" in line):
            continue
        if line in polled:
            continue
        page_loads += 1
        for after in lines[index + 1 :]:
            if memory.match(after):
                break
            assert not (
                after.startswith("s_waitcnt") and "vmcnt(0)" in after
            ), f"a page-index load drains the KV copies in flight: {line}"
    assert page_loads >= 2, "the split no longer loads page indices ahead"


def test_gather_lamport_polling_codegen():
    """The Lamport gather rereads all remote packs and votes once per retry."""
    rank = 5
    from tokenspeed_kernel.ops.communication._iris.all_gather import (
        iris_fused_lamport_allgather_rmsnorm_mxfp4 as fn,
    )
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy

    policy = iris_arch_policy(ArchVersion(9, 5))
    types = {
        "local_input": "*bf16",
        "weight": "*bf16",
        "packed_output": "*u8",
        "scale_output": "*u8",
        "region": "*bf16",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "scale_stride_linear": "i32",
        "scale_stride_mblock": "i32",
    }
    kernel = _compile_gfx950(
        fn,
        types,
        {
            "rank": rank,
            "local_width": 512,
            "total_width": 4096,
            "eps": 1e-6,
            "weight_offset": 1.0,
            "max_elements": 43 * 4096,
            "PEER_SIGNAL_STORE_CACHE": policy.peer_signal_store,
            "WAIT_STORES": policy.wait_stores,
        },
        num_warps=1,
    )
    assembly = kernel.asm["amdgcn"]
    # Each lane polls eight 16-byte packs; packs of its own shard are masked.
    polls = [
        cycle for cycle in polling_cycles(assembly) if _batched_peer_reads(cycle, 8)
    ]
    assert polls, "all remote system-cache reads must issue together on a retry"
    assert all(
        votes_with_subgroup_mask(cycle) for cycle in polls
    ), "a retry must branch on one subgroup lane mask"
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


def test_column_routes_lamport_codegen():
    """Every poll of the column-owned route completion rereads its words per retry.

    Each retry issues all of its cache-bypassing mailbox reads before waiting
    and branches on the subgroup's lane mask.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_lamport_column_routed_rmsnorm_sum as fn,
    )

    types = {
        "routes": "*bf16",
        "route_weights": "*fp32",
        "norm_weight": "*bf16",
        "output": "*bf16",
        "region": "*i32",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "rows": "i32",
    }
    kernel = _compile_gfx950(
        fn,
        types,
        {
            "RANK": 5,
            "MAX_ROWS": 43,
            "EPS": 1e-6,
            "WEIGHT_OFFSET": 1.0,
            "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
                ArchVersion(9, 5)
            ).peer_signal_store,
        },
        num_warps=1,
        unspecialized=("rows",),
    )
    assembly = kernel.asm["amdgcn"]
    cycles = polling_cycles(assembly)
    # The completer polls seven peers' 16-byte packs; the finisher its
    # eight routes' packs plus the sums of squares.
    assert any(_batched_peer_reads(cycle, 7) for cycle in cycles)
    assert any(
        _batched_peer_reads(cycle, 8)
        and any(
            re.search(r"\bbuffer_load_dword\b", line) and "sc0 sc1" in line
            for line in cycle
        )
        for cycle in cycles
    )
    assert all(votes_with_subgroup_mask(cycle) for cycle in cycles)
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


@pytest.mark.parametrize("block_m", [1, 8])
def test_moetail_lamport_codegen(block_m):
    """The fused tail pushes without waiting and its reducer votes per subgroup.

    Each sender's eight 8-byte packets per row (seven peers and its own slot)
    must issue back to back, one store instruction per peer for all rows; a
    wait after each would expose seven remote write latencies. The reducer
    rereads its 16-byte mailbox words on every retry and branches on the
    subgroup's lane mask. The mailbox pointer is compiled without alignment
    specialization: no 32-bit word may be split into 16-bit accesses.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (
        iris_fused_gemv_lamport_allgather_muse_moetail as fn,
    )
    from tokenspeed_kernel_amd._extern import extern_compile_options

    types = {
        "x_ptr": "*bf16",
        "weight_ptr": "*bf16",
        "region": "*bf16",
        "epochs": "*i32",
        "regions": ("i64",) * 8,
        "stream_ptr": "*fp32",
        "post_weight_ptr": "*bf16",
        "alpha_ptr": "*fp32",
        "beta_ptr": "*fp32",
        "pre_weight_ptr": "*bf16",
        "out_stream_ptr": "*fp32",
        "pre_bf16_ptr": "*bf16",
        "M": "i32",
    }
    kernel = _compile_gfx950(
        fn,
        types,
        {
            "RANK": 5,
            "WORLD_SIZE": 8,
            "K": 4096,
            "LOCAL_WIDTH": 1024,
            "MAX_ELEMENTS": 43 * 4096,
            "BLOCK_M": block_m,
            "BLOCK_K": 512,
            "WAVES_PER_OUTPUT": 4,
            "POST_EPS": 1e-5,
            "POST_WEIGHT_OFFSET": 1.0,
            "PRE_EPS": 1e-5,
            "PRE_WEIGHT_OFFSET": 1.0,
            "HAS_PRE": True,
            "PEER_SIGNAL_STORE_CACHE": iris_arch_policy(
                ArchVersion(9, 5)
            ).peer_signal_store,
        },
        num_warps=16,
        unspecialized=("region", "M"),
        options=extern_compile_options(),
    )
    assembly = kernel.asm["amdgcn"]
    lines = assembly.splitlines()
    pushes = [
        index
        for index, line in enumerate(lines)
        if "buffer_store_dwordx2" in line and "sc0 sc1" in line
    ]
    # Seven peer orders of seven pushes each, plus the own slot.
    assert len(pushes) == 7 * 7 + 1
    for order in range(7):
        first, last = pushes[7 * order], pushes[7 * order + 6]
        assert not any("s_waitcnt vmcnt" in line for line in lines[first:last])
    # The reducer restores its polled words with one 16-byte write-through store.
    # (Duplicate pusher lanes are dropped by an out-of-range buffer offset, not
    # an exec mask, so this test does not pin that pattern.)
    clears = [
        line for line in lines if "buffer_store_dwordx4" in line and "sc0 sc1" in line
    ]
    assert len(clears) == 1, clears
    polls = [
        cycle for cycle in polling_cycles(assembly) if _batched_peer_reads(cycle, 1)
    ]
    assert polls, "the reducer must reread its mailbox words on every retry"
    assert all(
        votes_with_subgroup_mask(cycle) for cycle in polls
    ), "a retry must branch on one subgroup lane mask"
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


@pytest.mark.parametrize("rows", [9, 12, 16])
def test_moetail_column_senders_push_whole_lines(rows):
    """Each column sender's push instruction writes whole 128-byte lines.

    A sender pushes its OUTPUTS columns of every row with one 16-byte store
    per lane; a store instruction that covers only part of each line it
    touches makes the peer link carry partial-line writes. Lanes map to
    elements through the send layout, as the store lowering maps them (one
    instruction per wave; replica waves store nothing).
    """
    from tokenspeed_kernel.ops.communication._iris.muse_spark import (
        _MUSE_BOUNDARY_WIDTH,
        _MUSE_MOETAIL_COLUMN_OUTPUTS,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (
        _send_layout,
    )

    outputs = _MUSE_MOETAIL_COLUMN_OUTPUTS
    block_m = 1 << (rows - 1).bit_length()
    row_bytes = 2 * _MUSE_BOUNDARY_WIDTH // 8
    layout = _send_layout(outputs, 16)
    per_lane, lanes_per_row = layout.size_per_thread[1], layout.threads_per_warp[1]
    # One store instruction per wave: each lane's elements are one 16-byte store.
    assert layout.size_per_thread[0] == 1 and layout.order == [1, 0]
    assert 2 * per_lane == 16 and per_lane * lanes_per_row == outputs
    rows_per_wave = layout.threads_per_warp[0]
    stored = set()
    for first_output in range(0, _MUSE_BOUNDARY_WIDTH // 8, outputs):
        for wave in range(16):
            lines = {}
            for lane in range(64):
                row = (wave * rows_per_wave + lane // lanes_per_row) % block_m
                column = first_output + lane % lanes_per_row * per_lane
                if row >= rows or (row, column) in stored:
                    continue  # masked, or a replica's duplicate
                stored.add((row, column))
                start = row * row_bytes + 2 * column
                lines.setdefault(start // 128, set()).update(range(start, start + 16))
            for line, written in lines.items():
                assert written == set(range(128 * line, 128 * (line + 1))), (
                    f"rows={rows}: sender {first_output // outputs}, wave "
                    f"{wave} writes {len(written)} of line {line}'s 128 bytes"
                )
    assert len(stored) == rows * (_MUSE_BOUNDARY_WIDTH // 8) // per_lane


def _muse_lamport_gather_mxfp4_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
        iris_lamport_bf16_allreduce,
    )
    from tokenspeed_kernel.ops.quantization import rmsnorm_mxfp4_quant

    state = _muse_state(rank, device, 8192, True)
    width, local_width = 4096, 4096 // world_size
    weight = (torch.randn(width, device=device) * 0.1).bfloat16()
    # Mixed ranks start from the last stage, so replays wrap the stage counter.
    state.reduction.lamport_epochs.fill_(2)
    raw_input = torch.empty(3 * 4096, dtype=torch.bfloat16, device=device)
    raw_output = torch.empty_like(raw_input)

    def raw():
        # The gather shares the reduction's mailbox tiles and stage counters.
        iris_lamport_bf16_allreduce[(raw_input.numel() // 512,)](
            raw_input,
            state.reduction.lamport_region,
            raw_output,
            state.reduction.lamport_epochs,
            state.reduction.lamport_peer_addresses[:world_size],
            RANK=rank,
            WORLD_SIZE=world_size,
            total_elements=raw_input.numel(),
            MAX_ELEMENTS=state.reduction.lamport_max_numel,
            NUM_STAGES=3,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=1,
        )

    # 128 rows exceed the pull variant's 84 ready flags; Lamport needs none.
    for rows in (1, 3, 16, 128):
        if rows == 1:
            published = state.all_reduce.acquire_outputs(((1, width),))[0]
            local = published[:, rank * local_width : (rank + 1) * local_width]
        else:
            local = torch.empty(
                (rows, local_width), dtype=torch.bfloat16, device=device
            )
        generator = torch.Generator(device=device)

        def update(phase):
            generator.manual_seed(100 * rows + phase)
            full = torch.randn((rows, width), generator=generator, device=device)
            full = full.bfloat16()
            # Negative zero is exchanged as zero; the empty marker stays unique.
            full[:, ::7] = -0.0
            local.copy_(full[:, rank * local_width : (rank + 1) * local_width])
            raw_input.fill_((rank + 1) * (phase + 1) / 16)
            return full.abs().where(full == 0, full)

        def run():
            raw()
            if rank == 0:
                torch.cuda._sleep(20000)
            packed, scales = state.all_gather_rmsnorm_mxfp4_quant(
                local, weight, 1e-6, 1.0
            )
            raw()
            return packed, scales

        update(0)
        run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            packed, scales = run()
        for phase in range(4):
            expected = update(phase)
            if rank == phase % world_size:
                torch.cuda._sleep(20000)
            before = state.reduction.lamport_epochs.clone()
            graph.replay()
            # Two reductions advance their 24 tiles; the gather advances its own.
            tiles = torch.arange(before.numel(), device=device)
            advanced = 2 * (tiles < 24) + (tiles < rows * local_width // 512)
            torch.testing.assert_close(
                state.reduction.lamport_epochs,
                ((before + advanced) % 3).to(before.dtype),
                atol=0,
                rtol=0,
            )
            reference = rmsnorm_mxfp4_quant(expected, weight, 1e-6, 1.0)
            torch.testing.assert_close(packed, reference[0], atol=0, rtol=0)
            torch.testing.assert_close(
                _logical_cdna4_scales(scales, rows, width // 32),
                _logical_cdna4_scales(reference[1], rows, width // 32),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                raw_output,
                torch.full_like(
                    raw_output, world_size * (world_size + 1) // 2 * (phase + 1) / 16
                ),
                atol=0,
                rtol=0,
            )
    torch.cuda.synchronize()
    dist.barrier()
    # Every consumed mailbox pack was returned to the empty marker.
    region = state.reduction.lamport_region.view(torch.int32)
    assert bool((region == -(2**31)).all())


def _muse_lamport_gather_mxfp4_topk_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.muse_spark.all_gather import (
        iris_fused_lamport_allgather_rmsnorm_mxfp4_topk,
    )
    from tokenspeed_kernel.ops.moe import moe_topk
    from tokenspeed_kernel.ops.quantization import rmsnorm_mxfp4_quant
    from utils import assert_no_triton_compile

    state = _muse_state(rank, device, 8192, True)
    width, local_width, experts, topk, scale = 4096, 4096 // world_size, 256, 8, 2.5
    weight = (torch.randn(width, device=device) * 0.1).bfloat16()
    bias = torch.randn(experts, device=device) * 0.01
    for rows in (2, 8, 16):
        local = torch.empty((rows, local_width), dtype=torch.bfloat16, device=device)
        logits = torch.empty((rows, experts), dtype=torch.float32, device=device)
        generator = torch.Generator(device=device)

        def update(phase):
            # Every rank draws the same rows and logits.
            generator.manual_seed(1000 * rows + phase)
            full = torch.randn((rows, width), generator=generator, device=device)
            full = full.bfloat16()
            local.copy_(full[:, rank * local_width : (rank + 1) * local_width])
            logits.copy_(
                torch.randn((rows, experts), generator=generator, device=device)
            )
            return full

        update(0)
        state.all_gather_rmsnorm_mxfp4_quant_topk(
            local, weight, 1e-6, 1.0, logits, bias, topk, scale
        )
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            (packed, scales), weights, ids = state.all_gather_rmsnorm_mxfp4_quant_topk(
                local, weight, 1e-6, 1.0, logits, bias, topk, scale
            )
        for phase in range(3):
            expected = update(phase)
            if rank == phase % world_size:
                torch.cuda._sleep(20000)
            graph.replay()
            reference = rmsnorm_mxfp4_quant(expected, weight, 1e-6, 1.0)
            torch.testing.assert_close(packed, reference[0], atol=0, rtol=0)
            torch.testing.assert_close(
                _logical_cdna4_scales(scales, rows, width // 32),
                _logical_cdna4_scales(reference[1], rows, width // 32),
                atol=0,
                rtol=0,
            )
            reference_weights, reference_ids = moe_topk(
                router_logits=logits,
                top_k=topk,
                score_function="sigmoid",
                selection_method="topk",
                renormalize=True,
                routed_scaling_factor=scale,
                correction_bias=bias,
                hash_indices_table=None,
                input_ids=None,
                logical_to_physical_map=None,
                topk_indices_dtype=torch.int32,
                topk_weights_dtype=torch.float32,
                override=None,
                solution=None,
            )
            assert torch.equal(ids, reference_ids)
            assert torch.equal(weights, reference_weights)
    # The row count is a runtime argument: rows of an already compiled
    # specialization class reuse its binary.
    with assert_no_triton_compile(iris_fused_lamport_allgather_rmsnorm_mxfp4_topk):
        for rows in (3, 5, 12):
            local = torch.zeros(
                (rows, local_width), dtype=torch.bfloat16, device=device
            )
            logits = torch.zeros((rows, experts), dtype=torch.float32, device=device)
            state.all_gather_rmsnorm_mxfp4_quant_topk(
                local, weight, 1e-6, 1.0, logits, bias, topk, scale
            )
    torch.cuda.synchronize()
    dist.barrier()
    region = state.reduction.lamport_region.view(torch.int32)
    assert bool((region == -(2**31)).all())


def _muse_routed_lamport_worker(rank, world_size, device):
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
        iris_lamport_bf16_allreduce,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_lamport_allreduce_routed_rmsnorm_sum,
        iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum,
    )
    from tokenspeed_kernel.ops.moe._triton.routed_norm import (
        _launch_routed_input_scale_sum,
    )
    from utils import assert_no_triton_compile

    state = _muse_state(rank, device, 8192, True)
    raw_inputs = [
        torch.empty(count, dtype=torch.bfloat16, device=device)
        for count in (43 * 4096, 512, 3 * 4096)
    ]
    raw_outputs = [torch.empty_like(value) for value in raw_inputs]

    def raw(index):
        value = raw_inputs[index]
        iris_lamport_bf16_allreduce[(value.numel() // 512,)](
            value,
            state.reduction.lamport_region,
            raw_outputs[index],
            state.reduction.lamport_epochs,
            state.reduction.lamport_peer_addresses[:world_size],
            RANK=rank,
            WORLD_SIZE=world_size,
            total_elements=value.numel(),
            MAX_ELEMENTS=state.reduction.lamport_max_numel,
            NUM_STAGES=3,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=1,
        )

    # The fused path ends at the one-shot mailbox capacity, or earlier where
    # TP8's two-shot plus the separate routed norm is faster; above it the
    # model normalizes locally and reduces through the generic policy.
    max_rows = state.routed_lamport_max_rows
    # This heap prepares the two-shot on 8 ranks only; pin the policy value.
    assert max_rows == (32 if world_size == 8 else 43)
    probe = torch.zeros((max_rows + 1) * 8, 4096, dtype=torch.bfloat16, device=device)
    probe_map = torch.zeros((max_rows + 1) * 8, dtype=torch.int32, device=device)
    probe_weights = torch.zeros(max_rows + 1, 8, device=device)
    probe_norm = torch.zeros(4096, dtype=torch.bfloat16, device=device)
    for count, eligible in ((max_rows, True), (max_rows + 1, False)):
        assert (
            state.can_all_reduce_routed_rmsnorm_sum(
                probe[: count * 8],
                probe_map[: count * 8],
                count,
                8,
                probe_weights[:count],
                probe_norm,
                1.0,
            )
            == eligible
        )

    state.all_reduce.producer_pull.ready_flags.fill_(-4)
    state.all_reduce.peers.ctx.device_barrier()
    for rows in (16, max_rows, 1, 17):
        torch.manual_seed(1300 + rows)
        values = torch.randn(rows * 8, 4096, device=device).bfloat16()
        routes = torch.empty_like(values)
        mapping = torch.empty(rows * 8, dtype=torch.int32, device=device)
        weights = torch.rand(rows, 8, device=device)
        weights /= weights.sum(-1, keepdim=True)
        norm = torch.randn(4096, device=device).bfloat16() * 0.1
        indices = torch.arange(rows * 8, dtype=torch.int32, device=device)

        def update(phase):
            owner = (indices + phase) % (world_size - 1 if phase == 3 else world_size)
            mapping.copy_(torch.where(owner == rank, indices, -1))
            routes.copy_(
                torch.where((owner == rank)[:, None], values * (phase + 1), torch.nan)
            )
            for value in raw_inputs:
                value.fill_((rank + 1) * (phase + 1) / 16)
                value[0] = -0.0

        def run():
            raw(0)
            if rank == 0:
                torch.cuda._sleep(20000)
            actual = state.all_reduce_routed_rmsnorm_sum(
                routes, mapping, rows, 8, weights, norm, 1e-5, 1.0
            ).clone()
            raw(1)
            if rows > 16:
                local = state.all_reduce.producer.data[: rows * 4096].view(rows, 4096)
                # The fused kernel keeps the two-launch routed sum's order; CDNA4
                # dispatches the one-launch sum, within one BF16 ULP of it.
                _launch_routed_input_scale_sum(
                    routes,
                    mapping,
                    row_stride=routes.stride(0),
                    route_weights=weights,
                    norm_weight=norm,
                    output=local,
                    num_tokens=rows,
                    topk=8,
                    eps=1e-5,
                    weight_offset=1.0,
                    block=4096,
                    single_pass=False,
                    enable_pdl=False,
                )
                expected = state.reduction.run(
                    local,
                    state.gather.output[: rows * 4096].view(rows, 4096),
                    rows * 4096,
                ).clone()
            else:
                iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum[(rows,)](
                    routes,
                    mapping,
                    weights,
                    norm,
                    state.all_reduce.producer.data,
                    state.gather.output,
                    state.all_reduce.producer_pull.ready_flags,
                    state.all_reduce.peers.addresses[:world_size],
                    rank=rank,
                    width=4096,
                    row_stride=routes.stride(0),
                    topk=8,
                    eps=1e-5,
                    weight_offset=1.0,
                    block=4096,
                    num_warps=8,
                    WAIT_STORES=iris_arch_policy().wait_stores,
                )
                expected = state.gather.output[: rows * 4096].view(rows, 4096).clone()
            raw(2)
            return actual, expected

        update(0)
        if rows in (16, max_rows):
            run()
        else:
            with assert_no_triton_compile(
                iris_fused_lamport_allreduce_routed_rmsnorm_sum
            ):
                run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, expected = run()
        for phase in range(4):
            update(phase)
            if rank == phase % world_size:
                torch.cuda._sleep(20000)
            graph.replay()
            torch.cuda.synchronize()
            assert bool(torch.isfinite(actual).all())
            torch.testing.assert_close(
                actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            for output in raw_outputs:
                reference = torch.full_like(
                    output, (world_size * (world_size + 1) // 2) * (phase + 1) / 16
                )
                reference[0] = 0.0
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
        dist.barrier()


def _muse_moetail_reference(x, weight, stream, post, alpha, beta, pre):
    from tokenspeed_kernel.ops.residual._triton.gated_boundary import (
        _launch_post_norm_residual_pre_norm,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        gluon_rowcta_bf16_gemv_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        launch_gluon_mm_a16w16_decode_gfx950,
    )

    # The replicated projection (single-row GEMVs up to 8 rows, the decode
    # GEMM beyond), then the unfused boundary.
    branch = x.new_empty((x.shape[0], weight.shape[0]))
    if x.shape[0] > 8:
        counters = torch.zeros(1, dtype=torch.int32, device=x.device)
        launch_gluon_mm_a16w16_decode_gfx950(
            x, weight, torch.bfloat16, counters=counters, out=branch
        )
    else:
        for row in range(x.shape[0]):
            gluon_rowcta_bf16_gemv_gfx950(
                x[row : row + 1], weight, branch[row : row + 1]
            )
    out_stream, _, pre_bf16 = _launch_post_norm_residual_pre_norm(
        stream,
        branch,
        post,
        1e-5,
        1.0,
        alpha,
        beta,
        pre,
        1e-6,
        1.0,
        False,
        None,
        1,
        None,
        return_pre_fp32=False,
    )
    return out_stream, pre_bf16


def _muse_moetail_worker(rank, world_size, device):
    """The fused tail matches the replicated projection plus the boundary.

    Up to 8 rows the senders project (column GEMV) and match per-row
    single-row GEMVs; 9-16 rows take a column-shard decode GEMM first and match
    the replicated decode GEMM. Row counts cover every power-of-two bucket at
    TP8 and odd counts inside them (TP4 takes one row). Two layers alternate weights and next norms, one
    rank lags, row counts interleave in one graph, and the stage counters start
    at the last stage so eager calls and replays wrap them.
    """
    state = _muse_state(rank, device, 2, True)
    reduction = state.reduction
    reduction.lamport_epochs.fill_(2)
    generator = torch.Generator(device=device).manual_seed(2610)
    weights = [
        torch.randn(8192, 4096, device=device, generator=generator).bfloat16() / 64
        for _ in range(2)
    ]
    # A zero row yields an exact zero, which travels as data, not as empty.
    weights[0][5] = 0
    post = (torch.randn(8192, device=device, generator=generator) * 0.1).bfloat16()
    pres = [
        (torch.randn(8192, device=device, generator=generator) * 0.1).bfloat16(),
        None,
    ]
    beta = torch.sigmoid(torch.randn(8192, device=device, generator=generator))
    alpha = torch.sqrt(1 - beta.square())
    counts = (1, 2, 3, 4, 5, 8, 9, 13, 16) if world_size == 8 else (1,)
    xs = {
        rows: [
            torch.empty(rows, 4096, dtype=torch.bfloat16, device=device)
            for _ in range(2)
        ]
        for rows in counts
    }
    streams = {
        rows: [torch.empty(rows, 8192, device=device) for _ in range(2)]
        for rows in counts
    }

    def update(phase):
        g = torch.Generator(device=device).manual_seed(2700 + phase)
        for rows in counts:
            for x, stream in zip(xs[rows], streams[rows]):
                # Negative inputs make the zero row's products negative zeros.
                x.copy_(-torch.rand(rows, 4096, device=device, generator=g).bfloat16())
                stream.copy_(torch.randn(rows, 8192, device=device, generator=g))

    def step(rows):
        outputs = []
        for layer in range(2):
            assert state.can_gemv_all_gather_boundary(
                rows,
                weights[layer],
                streams[rows][layer],
                post,
                alpha,
                beta,
                pres[layer],
            )
            if rank == layer:
                torch.cuda._sleep(20000)
            outputs.append(
                state.gemv_all_gather_boundary(
                    xs[rows][layer],
                    weights[layer],
                    streams[rows][layer],
                    post,
                    alpha,
                    beta,
                    pres[layer],
                    1e-5,
                    1.0,
                    1e-6,
                    1.0,
                )
            )
        return outputs

    def check(rows, outputs):
        for layer, (out_stream, pre_bf16) in enumerate(outputs):
            expected = _muse_moetail_reference(
                xs[rows][layer],
                weights[layer],
                streams[rows][layer],
                post,
                alpha,
                beta,
                pres[layer],
            )
            assert torch.equal(out_stream, expected[0])
            assert (pre_bf16 is None) == (expected[1] is None)
            if pre_bf16 is not None:
                assert torch.equal(pre_bf16, expected[1])

    tiles = torch.arange(reduction.lamport_epochs.numel(), device=device)
    for phase in range(4):
        update(phase)
        for rows in counts:
            advanced = (tiles < rows * 8192 // world_size // 512).int() * 2
            before = reduction.lamport_epochs.clone()
            check(rows, step(rows))
            torch.cuda.synchronize()
            assert torch.equal(reduction.lamport_epochs, (before + advanced) % 3)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = {rows: step(rows) for rows in counts}
    for phase in range(4, 8):
        update(phase)
        graph.replay()
        for rows in counts:
            check(rows, outputs[rows])
    torch.cuda.synchronize()
    dist.barrier()
    assert bool((reduction.lamport_region.view(torch.int32) == -(2**31)).all())
    epochs = [torch.empty_like(reduction.lamport_epochs) for _ in range(world_size)]
    dist.all_gather(epochs, reduction.lamport_epochs)
    assert all(torch.equal(e, reduction.lamport_epochs) for e in epochs)


def _muse_sliding_attention_tail_worker(rank, world_size, device, first_epoch):
    """The fused sliding attention tail matches the four separate kernels.

    Today: sliding split attention, the gated per-head norm, the o_proj GEMV
    into the producer buffer and the attention boundary. The tail differs only
    in reduction orders (GEMV, per-head and per-chunk sums, the norm-sum
    expansion), so it agrees within rounding while ranks agree bit for bit.
    Row counts alternate 8 -> 16 -> 8 -> 16 eagerly and under graph replay
    with fresh inputs, one rank lags, and both post-norm forms run; afterwards
    the next launch's mailboxes are empty. The epoch words start at
    first_epoch; 2**31 - 2 runs the launches across the signed 32-bit limit.
    """
    from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
        _SLIDING_TAIL_EPOCH_PERIOD,
    )
    from tokenspeed_kernel.ops.gemm import mv
    from tokenspeed_kernel.ops.layernorm.fused_norm import (
        weightless_rmsnorm_gated_sigmoid,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        launch_gluon_mha_decode_gfx950,
    )

    state = _muse_state(rank, device, 16, True)
    if world_size != 8:
        assert state.attntail_region is None
        return
    state.attntail_epochs.fill_(first_epoch)
    generator = torch.Generator(device=device).manual_seed(4100 + rank)
    shared = torch.Generator(device=device).manual_seed(4200)
    # 4104 cached tokens per request: a 2048-key window over 33 pages of 64.
    context, window, page = 4104, 2047, 64
    pages = context // page + 1
    k_cache = (
        torch.randn(2 * pages + 8, page, 2, 64, device=device, generator=generator) / 2
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn_like(k_cache, dtype=torch.float32).to(torch.float8_e4m3fn)
    page_table = (
        torch.randperm(2 * pages + 8, device=device, generator=generator)[: 2 * pages]
        .view(2, pages)
        .to(torch.int32)
    )
    seqlens = torch.full((2,), context, dtype=torch.int32, device=device)
    weight = torch.randn(8192, 1024, device=device, generator=generator).bfloat16() / 32
    pre = (torch.randn(8192, device=device, generator=shared) * 0.1).bfloat16()
    post = (torch.randn(8192, device=device, generator=shared) * 0.1).bfloat16()
    beta = torch.sigmoid(torch.randn(8192, device=device, generator=shared))
    alpha = torch.sqrt(1 - beta.square())
    q = torch.empty(16, 16, 64, dtype=torch.float8_e4m3fn, device=device)
    qkvg = torch.empty(16, 2304, dtype=torch.bfloat16, device=device)
    stream = torch.empty(16, 8192, device=device)

    # Plain decode (one row per request) declines; 8 verify rows per request serve.
    for rows in (8, 16):
        for requests, served in ((rows, False), (rows // 8, True)):
            assert (
                state.can_sliding_attention_boundary(
                    rows,
                    requests,
                    k_cache,
                    context,
                    window,
                    0.125,
                    weight,
                    stream[:rows],
                )
                is served
            )

    def update(phase):
        g = torch.Generator(device=device).manual_seed(4300 + 8 * phase + rank)
        q.copy_((torch.randn(16, 16, 64, device=device, generator=g) / 3).to(q.dtype))
        qkvg.copy_(torch.randn(16, 2304, device=device, generator=g).bfloat16())
        # The residual is replicated across ranks.
        h = torch.Generator(device=device).manual_seed(4400 + phase)
        stream.copy_(torch.randn(16, 8192, device=device, generator=h))

    def tail(rows, post_weight, lag):
        if lag:
            torch.cuda._sleep(20000)
        return state.sliding_attention_boundary(
            q[:rows],
            k_cache,
            v_cache,
            page_table[: rows // 8],
            seqlens[: rows // 8],
            context,
            window,
            0.125,
            qkvg[:rows, 1280:],
            weight,
            stream[:rows],
            post_weight,
            alpha,
            beta,
            pre,
            1e-6,
            1e-6,
            1.0,
            1e-6,
            1.0,
        )

    def today(rows, post_weight):
        attended = launch_gluon_mha_decode_gfx950(
            q[:rows],
            k_cache,
            v_cache,
            page_table[: rows // 8],
            seqlens[: rows // 8],
            context,
            max_seqlen_q=8,
            window_left=window,
            softmax_scale=0.125,
        )
        normed = weightless_rmsnorm_gated_sigmoid(
            attended.view(rows, 1024), qkvg[:rows, 1280:], 16, 64, 1e-6, False
        )
        branch = state.all_reduce.producer.acquire(((rows, 8192),))[0]
        mv(normed, weight, out=branch)
        return state.all_reduce_symmetric_boundary(
            branch, stream[:rows], post_weight, alpha, beta, pre, 1e-6, 1.0, 1e-6, 1.0
        )

    def check(rows, got, post_weight):
        expected = today(rows, post_weight)
        for a, b in zip(got, expected, strict=True):
            a, b = a.float(), b.float()
            # Measured relative RMS 5e-5 to 1e-4 (reduction order only).
            assert (a - b).pow(2).mean().sqrt() < 1e-3 * b.pow(2).mean().sqrt()
            copies = [torch.empty_like(a) for _ in range(world_size)]
            dist.all_gather(copies, a)
            assert all(torch.equal(c, copies[0]) for c in copies)

    order = (8, 16, 8, 16)
    for phase in range(2):
        update(phase)
        for step, rows in enumerate(order):
            post_weight = post if step % 2 else None
            check(rows, tail(rows, post_weight, rank == step), post_weight)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [tail(rows, None, False) for rows in order]
    for phase in range(2, 4):
        update(phase)
        graph.replay()
        for rows, got in zip(order, outputs):
            check(rows, got, None)
    torch.cuda.synchronize()
    dist.barrier()
    # The next launch's stage is empty everywhere (cleared at an entry two
    # launches back and unused since), and every attention-buffer word is re-armed.
    # Eight eager launches and eight replayed ones advanced every epoch word.
    period = _SLIDING_TAIL_EPOCH_PERIOD.value
    epochs = state.attntail_epochs
    assert bool((epochs == (first_epoch + 16) % period).all())
    stage = int(epochs[0]) % 3
    assert bool((state.attntail_region[stage].view(torch.int32) == -(2**31)).all())
    assert bool((state.attention_buffer == -2147450880).all())


def _muse_decode_step_lamport_worker(rank, world_size, device):
    """Interleave every Lamport user of one decode layer under graph replay.

    The routed-normalization, gather-norm-MXFP4, post-expert tail and
    generic one-shot BF16 reductions share one sentinel mailbox and its
    per-512-element stage counters while covering different tile prefixes, so
    the counters diverge per tile. TP8's generic two-shot (route payloads from
    two tokens) and the attention tail keep their own mailboxes but run in the
    same step.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import iris_arch_policy
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum,
    )
    from tokenspeed_kernel.ops.quantization import rmsnorm_mxfp4_quant

    state = _muse_state(rank, device, 32, True)
    width, local_width, layers = 4096, 4096 // world_size, 2
    torch.manual_seed(1700)
    pre = torch.randn(8192, device=device).bfloat16()
    beta = torch.sigmoid(torch.randn(8192, device=device))
    alpha = torch.sqrt(1 - beta.square())
    gather_weight = (torch.randn(width, device=device) * 0.1).bfloat16()
    norm = (torch.randn(width, device=device) * 0.1).bfloat16()
    # The post-expert tail projects the routed rows and gathers 8192 columns.
    tail_weight = (torch.randn(8192, width, device=device) / 64).bfloat16()
    tail_post = (torch.randn(8192, device=device) * 0.1).bfloat16()
    # Start every tile at the last stage so the first replays wrap the counters.
    reduction = state.reduction
    reduction.lamport_epochs.fill_(2)
    if reduction.lamport_two_shot_epochs is not None:
        reduction.lamport_two_shot_epochs.fill_(2)
    two_shot_tile = world_size * reduction.LAMPORT_TWO_SHOT_TILE

    def two_shot_programs(count):
        """Programs of the generic reduction's two-shot launch, or zero."""
        selected = (
            reduction.LAMPORT_TWO_SHOT_MIN_ELEMENTS
            <= count
            <= reduction.lamport_two_shot_max_numel
            and count % two_shot_tile == 0
        )
        return count // two_shot_tile if selected else 0

    # Eight rows (the post-expert tail's largest bucket) are out of reach here:
    # the routed one-shot Lamport that feeds the tail would need 8 * 8 * 4096
    # elements per slot, beyond lamport_max_numel (43 * 4096). The moetail
    # worker covers eight rows on its own.
    for rows in (1, 3, 4, 5):
        assert rows * 8 * width <= state.reduction.lamport_max_numel
        torch.manual_seed(1800 + rows)
        attn_base = torch.randn(rows, 8192, device=device)
        attn_inputs = [
            torch.empty(rows, 8192, dtype=torch.bfloat16, device=device)
            for _ in range(layers)
        ]
        attn_streams = [torch.randn(rows, 8192, device=device) for _ in range(layers)]
        gather_full = torch.randn(rows, width, device=device)
        gather_locals = [
            torch.empty(rows, local_width, dtype=torch.bfloat16, device=device)
            for _ in range(layers)
        ]
        route_values = torch.randn(rows * 8, width, device=device).bfloat16()
        routes = torch.empty_like(route_values)
        mapping = torch.empty(rows * 8, dtype=torch.int32, device=device)
        route_weights = torch.rand(rows, 8, device=device)
        route_weights /= route_weights.sum(-1, keepdim=True)
        indices = torch.arange(rows * 8, dtype=torch.int32, device=device)
        # EP1's raw-route reduction plus a short one touching only three tiles.
        raw_inputs = [
            torch.empty(count, dtype=torch.bfloat16, device=device)
            for count in (rows * 8 * width, 3 * 512)
        ]

        def contribution(layer, peer, phase):
            return (attn_base + (peer + 1) * (phase + 1) / 8 + layer).bfloat16()

        def gather_rows(layer, phase):
            full = (gather_full * (phase + 1) + layer).bfloat16()
            # Negative zero is exchanged as zero; the empty marker stays unique.
            full[:, ::7] = -0.0
            return full

        def update(phase):
            for layer in range(layers):
                attn_inputs[layer].copy_(contribution(layer, rank, phase))
                gather_locals[layer].copy_(
                    gather_rows(layer, phase)[
                        :, rank * local_width : (rank + 1) * local_width
                    ]
                )
            owner = (indices + phase) % world_size
            mapping.copy_(torch.where(owner == rank, indices, -1))
            routes.copy_(
                torch.where(
                    (owner == rank)[:, None], route_values * (phase + 1), torch.nan
                )
            )
            for value in raw_inputs:
                value.fill_((rank + 1) * (phase + 1) / 16)
                value[0] = -0.0

        def step():
            outputs = []
            for layer in range(layers):
                branch = state.all_reduce.acquire_outputs(((rows, 8192),))[0]
                branch.copy_(attn_inputs[layer])
                boundary = state.all_reduce_symmetric_boundary(
                    branch,
                    attn_streams[layer],
                    None,
                    alpha,
                    beta,
                    pre,
                    1e-6,
                    1.0,
                    1e-6,
                    1.0,
                )
                if rank == layer:
                    torch.cuda._sleep(20000)
                packed, scales = state.all_gather_rmsnorm_mxfp4_quant(
                    gather_locals[layer], gather_weight, 1e-6, 1.0
                )
                routed = state.all_reduce_routed_rmsnorm_sum(
                    routes, mapping, rows, 8, route_weights, norm, 1e-5, 1.0
                ).clone()
                raw = []
                for value in raw_inputs:
                    branch = state.all_reduce.acquire_outputs((tuple(value.shape),))[0]
                    branch.copy_(value)
                    raw.append(state.reduce_outputs((branch,))[0].clone())
                tail = None
                if state.can_gemv_all_gather_boundary(
                    rows, tail_weight, boundary[0], tail_post, alpha, beta, pre
                ):
                    tail = state.gemv_all_gather_boundary(
                        routed,
                        tail_weight,
                        boundary[0],
                        tail_post,
                        alpha,
                        beta,
                        pre,
                        1e-5,
                        1.0,
                        1e-6,
                        1.0,
                    )
                outputs.append((boundary, packed, scales, routed, raw, tail))
            return outputs

        update(0)
        step()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = step()
        # Stage counters each Lamport call advances per layer: gather, routed,
        # and each generic reduction on whichever protocol it selects.
        tiles = torch.arange(reduction.lamport_epochs.numel(), device=device)
        advanced = (tiles < rows * local_width // 512).int() + (tiles < rows * 8).int()
        if state.can_gemv_all_gather_boundary(
            rows, tail_weight, attn_streams[0], tail_post, alpha, beta, pre
        ):
            advanced += (tiles < rows * 8192 // world_size // 512).int()
        two_shot_advanced = None
        if reduction.lamport_two_shot_epochs is not None:
            programs = torch.arange(
                reduction.lamport_two_shot_epochs.numel(), device=device
            )
            two_shot_advanced = torch.zeros_like(programs)
        for value in raw_inputs:
            count = value.numel()
            if two_shot_programs(count):
                two_shot_advanced += layers * (programs < two_shot_programs(count))
            else:
                advanced += (tiles < count // 512).int()
        advanced *= layers
        if world_size == 8 and rows > 1:
            # Route payloads from two tokens take the TP8 two-shot.
            assert two_shot_programs(rows * 8 * width)
        for phase in range(4):
            update(phase)
            if rank == phase % world_size:
                torch.cuda._sleep(20000)
            counters = [(reduction.lamport_epochs, advanced)]
            if two_shot_advanced is not None:
                counters.append((reduction.lamport_two_shot_epochs, two_shot_advanced))
            before = [epochs.clone() for epochs, _ in counters]
            graph.replay()
            torch.cuda.synchronize()
            for (epochs, step_advance), previous in zip(counters, before):
                torch.testing.assert_close(
                    epochs,
                    ((previous + step_advance) % 3).to(epochs.dtype),
                    atol=0,
                    rtol=0,
                )
                gathered = [torch.empty_like(epochs) for _ in range(world_size)]
                dist.all_gather(gathered, epochs)
                assert all(torch.equal(epochs, other) for other in gathered)

            # Reference for the routed reduction: the pull one-shot kernel,
            # which shares no state with the Lamport mailbox.
            iris_fused_pull_oneshot_allreduce_routed_rmsnorm_sum[(rows,)](
                routes,
                mapping,
                route_weights,
                norm,
                state.all_reduce.producer.data,
                state.gather.output,
                state.all_reduce.producer_pull.ready_flags,
                state.all_reduce.peers.addresses[:world_size],
                rank=rank,
                width=width,
                row_stride=routes.stride(0),
                topk=8,
                eps=1e-5,
                weight_offset=1.0,
                block=width,
                num_warps=8,
                WAIT_STORES=iris_arch_policy().wait_stores,
            )
            routed_expected = state.gather.output[: rows * width].view(rows, width)
            for layer, (boundary, packed, scales, routed, raw, tail) in enumerate(
                outputs
            ):
                reduced = contribution(layer, 0, phase).float()
                for peer in range(1, world_size):
                    reduced.add_(contribution(layer, peer, phase).float())
                expected = _boundary_reference(
                    attn_streams[layer], reduced.bfloat16(), None, alpha, beta, pre
                )
                for got, reference in zip(boundary, expected):
                    torch.testing.assert_close(
                        got,
                        reference,
                        atol=1e-5 if got.dtype == torch.float32 else 0.015625,
                        rtol=2e-5 if got.dtype == torch.float32 else 0.008,
                    )
                full = gather_rows(layer, phase)
                reference = rmsnorm_mxfp4_quant(
                    full.abs().where(full == 0, full), gather_weight, 1e-6, 1.0
                )
                torch.testing.assert_close(packed, reference[0], atol=0, rtol=0)
                torch.testing.assert_close(
                    _logical_cdna4_scales(scales, rows, width // 32),
                    _logical_cdna4_scales(reference[1], rows, width // 32),
                    atol=0,
                    rtol=0,
                )
                assert bool(torch.isfinite(routed).all())
                torch.testing.assert_close(
                    routed.view(torch.int16),
                    routed_expected.view(torch.int16),
                    atol=0,
                    rtol=0,
                )
                for output in raw:
                    reference = torch.full_like(
                        output, world_size * (world_size + 1) // 2 * (phase + 1) / 16
                    )
                    reference[0] = 0.0
                    torch.testing.assert_close(output, reference, atol=0, rtol=0)
                if tail is not None:
                    expected = _muse_moetail_reference(
                        routed, tail_weight, boundary[0], tail_post, alpha, beta, pre
                    )
                    assert torch.equal(tail[0], expected[0])
                    assert torch.equal(tail[1], expected[1])
            dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    # Every consumed mailbox pack was returned to the empty marker.
    for region in (reduction.lamport_region, reduction.lamport_two_shot_region):
        if region is not None:
            assert bool((region.view(torch.int32) == -(2**31)).all())


def _muse_lamport_payload_worker(rank, world_size, device):
    """Distinct per-rank data through each Lamport protocol, grid, and stage."""
    state = _muse_state(rank, device, 128, True)
    reduction = state.reduction
    two_shot = world_size == 8
    if two_shot:
        assert reduction.lamport_two_shot_max_numel == 128 * 8192
    else:
        assert reduction.lamport_two_shot_epochs is None
    # (elements, in place, protocol). One-shot: one and eight tiles, one route
    # token, and a two-shot-sized payload of partial two-shot tiles. TP8
    # two-shot: the smallest payload, route tokens 2..32 including the largest
    # one-shot payload, and the full capacity. TP4 has no two-shot; its larger
    # payloads use push.
    cases = (
        (512, False, "one"),
        (8 * 512, False, "one"),
        (32768, False, "one"),
        (32768, True, "one"),
        (48 * 1024 + 512, False, "one"),
        (48 * 1024, False, "two"),
        (65536, False, "two"),
        (98304, True, "two"),
        (131072, False, "two"),
        (43 * 4096, False, "two"),
        (262144, False, "two"),
        (524288, True, "two"),
        (1048576, False, "two"),
    )
    if not two_shot:
        cases = tuple(
            (count, in_place, "one" if count <= 43 * 4096 else "push")
            for count, in_place, _ in cases
        )
    generator = torch.Generator().manual_seed(4096 + world_size)
    payloads = []
    for count, _, _ in cases:
        values = torch.randint(-512, 512, (world_size, count), generator=generator)
        values = (values.float() / 64).bfloat16()
        values[:, 1::97] = -0.0
        values[:, 2::97] = float("inf")
        values[:, 3::97] = float("nan")
        payloads.append(values)
    inputs = [
        torch.empty(count, dtype=torch.bfloat16, device=device) for count, _, _ in cases
    ]
    outputs = [
        value if in_place else torch.empty_like(value)
        for value, (_, in_place, _) in zip(inputs, cases, strict=True)
    ]

    def load(scale):
        for values, value in zip(payloads, inputs, strict=True):
            value.copy_(values[rank] * scale)

    def expected(values, scale):
        # These FP32 sums are exact, so every protocol's addition order agrees.
        reduced = (values[0] * scale).float()
        for peer in range(1, world_size):
            reduced += (values[peer] * scale).float()
        return reduced.bfloat16().to(device)

    def run():
        # Push returns borrowed storage that the next reduction overwrites.
        return [
            reduction.run(value, output, value.numel())[: value.numel()].clone()
            for value, output in zip(inputs, outputs, strict=True)
        ]

    def check(results, scale):
        for values, actual in zip(payloads, results, strict=True):
            torch.testing.assert_close(
                actual, expected(values, scale), atol=0, rtol=0, equal_nan=True
            )

    # Each Lamport call advances the stage of exactly the programs it launched.
    for (count, _, protocol), value, output in zip(cases, inputs, outputs, strict=True):
        one_shot = reduction.lamport_epochs.clone()
        two = two_shot and reduction.lamport_two_shot_epochs.clone()
        value.copy_(payloads[0][rank, :1].expand(count))
        reduction.run(value, output, count)
        advanced = (reduction.lamport_epochs != one_shot).sum().item()
        assert advanced == (count // 512 if protocol == "one" else 0)
        if two_shot:
            advanced = (reduction.lamport_two_shot_epochs != two).sum().item()
            assert advanced == (count // 1024 if protocol == "two" else 0)
    load(1)
    check(run(), 1)
    # Start the capture from the last stage so replays wrap the rotation.
    reduction.lamport_epochs.fill_(2)
    if two_shot:
        reduction.lamport_two_shot_epochs.fill_(2)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = run()
    for step in range(5):
        scale = float(2**step)
        load(scale)
        if rank == step % world_size:
            torch.cuda._sleep(20000)
        graph.replay()
        torch.cuda.synchronize()
        check(results, scale)
    dist.barrier()


def _muse_tp_route_scatter_worker(rank, world_size, device):
    """Expert TP8 completes decode routes on token owners and gathers the sums.

    Owners sum routes in rank order, as the TP8 Lamport two-shot does, so the
    completed routes match that all-reduce bit for bit; the owner's fused
    weighted accumulation keeps the sums within BF16 rounding of the portable
    routed norm. One graph interleaves row counts with and without padding
    owners, a route source already in the staging buffer and one that is
    copied there, an owned prefill reduce-scatter that shares the staging
    buffer and flags, and a Lamport one-shot all-reduce that shares the
    gather's mailbox; one rank is delayed in each replay.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
        iris_lamport_bf16_allreduce,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark import (
        prepare_muse_communication,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_routed_rmsnorm_sum_lamport_allgather,
    )
    from tokenspeed_kernel.ops.moe import routed_input_scale_rmsnorm_sum
    from utils import assert_no_triton_compile

    torch.set_grad_enabled(False)
    capacity = 256
    state = prepare_muse_communication(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=capacity * 8192,
        enable_lamport=True,
        owned_route_max_rows=capacity,
        intermediate_gather_max_rows=0,
        all_reduce=None,
        device=device,
    )
    if world_size != 8:
        # Only the TP8 crossover was measured.
        return
    # The window ends where owned rows take over.
    assert [
        state.can_reduce_scatter_routed_rmsnorm_sum_all_gather(rows)
        for rows in (43, 44, 49, 191, 192)
    ] == [False, True, True, True, False]
    norm = torch.linspace(-0.25, 0.25, 4096, device=device).bfloat16()
    state.warmup_reduce_scatter_routed_rmsnorm_sum_all_gather(norm, 1e-5, 1.0)

    def partial_routes(rows, peer, phase):
        generator = torch.Generator(device=device).manual_seed(
            (rows * 64 + peer) * 4 + phase
        )
        return torch.randn(
            rows * 8, 4096, generator=generator, device=device
        ).bfloat16()

    def complete_routes(rows, phase):
        # FP32 in rank order, rounded once, as the owners sum them.
        total = partial_routes(rows, 0, phase).float()
        for peer in range(1, world_size):
            total += partial_routes(rows, peer, phase).float()
        return total.bfloat16()

    def route_weights(rows, phase):
        generator = torch.Generator(device=device).manual_seed(rows * 4 + phase)
        weights = torch.rand(rows, 8, generator=generator, device=device)
        return weights / weights.sum(-1, keepdim=True)

    def reference(rows, phase):
        return routed_input_scale_rmsnorm_sum(
            complete_routes(rows, phase).view(rows, 8, 4096),
            route_weights(rows, phase),
            norm,
            1e-5,
            1.0,
            False,
            None,
        )

    raw_input = torch.empty(3 * 4096, dtype=torch.bfloat16, device=device)
    raw_output = torch.empty_like(raw_input)

    def raw():
        iris_lamport_bf16_allreduce[(raw_input.numel() // 512,)](
            raw_input,
            state.reduction.lamport_region,
            raw_output,
            state.reduction.lamport_epochs,
            state.reduction.lamport_peer_addresses[:world_size],
            RANK=rank,
            WORLD_SIZE=world_size,
            total_elements=raw_input.numel(),
            MAX_ELEMENTS=state.reduction.lamport_max_numel,
            NUM_STAGES=3,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=1,
        )

    # Owned prefill rows; their staging copy and flags are shared.
    prefill_rows = 192
    prefill = torch.empty(prefill_rows * 8, 4096, dtype=torch.bfloat16, device=device)
    prefill_local = prefill_rows // world_size
    plain = torch.empty(capacity * 8, 4096, dtype=torch.bfloat16, device=device)
    weights = [torch.empty(capacity, 8, device=device) for _ in range(2)]
    # The first call reads the staging buffer in place, the second copies in.
    # At 49 rows the last owner has no token; two such calls run back to back.
    for index, counts in enumerate(((64, 44), (128, 191), (49, 49))):
        staged_rows, copied_rows = counts

        def update(phase):
            state.acquire_owned_route_input(staged_rows).copy_(
                partial_routes(staged_rows, rank, phase)
            )
            plain.fill_(torch.nan)
            plain[: copied_rows * 8].copy_(partial_routes(copied_rows, rank, phase))
            for weight, rows in zip(weights, counts):
                weight[:rows].copy_(route_weights(rows, phase))
            prefill.copy_(partial_routes(prefill_rows, rank, phase))
            raw_input.fill_((rank + 1) * (phase + 1) / 16)

        def run():
            outputs = [
                state.reduce_scatter_routed_rmsnorm_sum_all_gather(
                    state.acquire_owned_route_input(staged_rows),
                    staged_rows,
                    weights[0][:staged_rows],
                    norm,
                    1e-5,
                    1.0,
                )
            ]
            raw()
            owned = state.reduce_scatter_owned_routes(prefill, None, prefill_rows)
            # The next call restages over the owned result's source.
            owned_copy = owned.clone()
            outputs.append(
                state.reduce_scatter_routed_rmsnorm_sum_all_gather(
                    plain[: copied_rows * 8],
                    copied_rows,
                    weights[1][:copied_rows],
                    norm,
                    1e-5,
                    1.0,
                )
            )
            return outputs, owned_copy

        def check(results, phase):
            outputs, owned = results
            for output, rows in zip(outputs, counts):
                # One BF16 ULP, with an absolute floor for sums that cancel.
                torch.testing.assert_close(
                    output, reference(rows, phase), atol=2**-8, rtol=2**-7
                )
            first = rank * prefill_local
            expected = complete_routes(prefill_rows, phase).view(prefill_rows, 8, 4096)
            torch.testing.assert_close(
                owned.view(torch.int16),
                expected[first : first + prefill_local].view(torch.int16),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                raw_output,
                torch.full_like(
                    raw_output,
                    (world_size * (world_size + 1) // 2) * (phase + 1) / 16,
                ),
                rtol=0,
                atol=0,
            )

        update(0)
        if index == 0:
            # Startup's compile-only warmup covers the gather.
            with assert_no_triton_compile(
                iris_fused_routed_rmsnorm_sum_lamport_allgather
            ):
                check(run(), 0)
        else:
            check(run(), 0)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for phase in range(1, 4):
            update(phase)
            if rank == (phase * 3 + index) % world_size:
                torch.cuda._sleep(20000)
            graph.replay()
            torch.cuda.synchronize()
            check(captured, phase)
        dist.barrier()


def _muse_tp_column_routes_worker(rank, world_size, device):
    """Expert TP8 decode tokens complete their routes column-owned in one launch.

    Every rank writes the same sums, within one BF16 unit of the Lamport
    two-shot all-reduce plus the portable routed norm, and bit for bit when
    each token weights a single route. One graph runs two layers back to back
    at each of 1, 8, 9, 16, 32, and 43 tokens, interleaved with the generic
    Lamport one-shot and two-shot reductions; a zero route and negative zeros
    travel as data, one rank is delayed in each replay, and the stage counters
    start at the last stage so the replays wrap them.
    """
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
        iris_lamport_bf16_allreduce,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark import (
        prepare_muse_communication,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_fused_lamport_column_routed_rmsnorm_sum,
    )
    from tokenspeed_kernel.ops.moe import routed_input_scale_rmsnorm_sum
    from utils import assert_no_triton_compile

    torch.set_grad_enabled(False)
    capacity = 256
    state = prepare_muse_communication(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=capacity * 8192,
        enable_lamport=True,
        owned_route_max_rows=capacity,
        intermediate_gather_max_rows=0,
        all_reduce=None,
        device=device,
    )
    if world_size != 8:
        assert not state.can_reduce_column_routed_rmsnorm_sum(8)
        return
    # It takes over every token count the two-shot route all-reduce served.
    assert [
        state.can_reduce_column_routed_rmsnorm_sum(rows) for rows in (0, 1, 43, 44)
    ] == [False, True, True, False]
    assert state.route_scatter_min_rows == 44
    state.column_routes_epochs.fill_(2)
    counts = (1, 8, 9, 16, 32, 43)
    layers = 2
    norms = [
        (torch.randn(4096, generator=torch.Generator().manual_seed(layer)) * 0.1)
        .bfloat16()
        .to(device)
        for layer in range(layers)
    ]

    def partial_routes(rows, peer, phase, layer):
        generator = torch.Generator(device=device).manual_seed(
            ((rows * 64 + peer) * 8 + phase) * 2 + layer
        )
        routes = torch.randn(rows * 8, 4096, generator=generator, device=device)
        routes = (routes / 8).bfloat16()
        # An all-zero route normalizes to zeros; negative zeros travel as zeros.
        routes[3 % (rows * 8)] = 0
        routes[-1, :64] = -0.0
        return routes

    def route_weights(rows, phase, layer):
        generator = torch.Generator(device=device).manual_seed(
            (rows * 8 + phase) * 2 + layer
        )
        weights = torch.rand(rows, 8, generator=generator, device=device)
        if layer == 1:
            # One unit weight per token: the sum is one normalized route, which
            # any multiply-add order rounds alike, so the completed routes and
            # inverse RMS must match bit for bit.
            hot = weights.argmax(-1, keepdim=True)
            return torch.zeros_like(weights).scatter_(-1, hot, 1.0)
        return weights / weights.sum(-1, keepdim=True)

    def reference(rows, phase, layer):
        # FP32 in rank order, rounded once, as the owners sum the routes.
        total = partial_routes(rows, 0, phase, layer).float()
        for peer in range(1, world_size):
            total += partial_routes(rows, peer, phase, layer).float()
        return routed_input_scale_rmsnorm_sum(
            total.bfloat16().view(rows, 8, 4096),
            route_weights(rows, phase, layer),
            norms[layer],
            1e-6,
            1.0,
            False,
            None,
        )

    routes = {
        (rows, layer): torch.empty(rows * 8, 4096, dtype=torch.bfloat16, device=device)
        for rows in counts
        for layer in range(layers)
    }
    weights = {key: torch.empty(key[0], 8, device=device) for key in routes}
    one_shot_input = torch.empty(3 * 4096, dtype=torch.bfloat16, device=device)
    one_shot_output = torch.empty_like(one_shot_input)
    two_shot = torch.empty(8 * 8 * 4096, dtype=torch.bfloat16, device=device)

    def update(phase):
        for (rows, layer), value in routes.items():
            value.copy_(partial_routes(rows, rank, phase, layer))
            weights[rows, layer].copy_(route_weights(rows, phase, layer))
        one_shot_input.fill_((rank + 1) * (phase + 1) / 16)
        two_shot.fill_((rank + 1) * (phase + 1) / 64)

    def run(run_counts):
        outputs = {}
        for rows in run_counts:
            for layer in range(layers):
                outputs[rows, layer] = state.reduce_column_routed_rmsnorm_sum(
                    routes[rows, layer],
                    weights[rows, layer],
                    norms[layer],
                    1e-6,
                    1.0,
                )
            iris_lamport_bf16_allreduce[(one_shot_input.numel() // 512,)](
                one_shot_input,
                state.reduction.lamport_region,
                one_shot_output,
                state.reduction.lamport_epochs,
                state.reduction.lamport_peer_addresses[:world_size],
                RANK=rank,
                WORLD_SIZE=world_size,
                total_elements=one_shot_input.numel(),
                MAX_ELEMENTS=state.reduction.lamport_max_numel,
                NUM_STAGES=3,
                PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
                num_warps=1,
            )
        state.reduce(two_shot)
        return outputs

    def check(outputs, phase):
        for (rows, layer), output in outputs.items():
            # Every rank writes every column the same way.
            gathered = [torch.empty_like(output) for _ in range(world_size)]
            dist.all_gather(gathered, output)
            assert all(torch.equal(value, output) for value in gathered)
            expected = reference(rows, phase, layer)
            if layer == 1:
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
            else:
                # Within one BF16 unit: the portable sum contracts its
                # multiply-add in some specializations only. The floor covers
                # sums that cancel.
                torch.testing.assert_close(output, expected, atol=2**-14, rtol=2**-7)
        sum_of_ranks = world_size * (world_size + 1) // 2
        assert bool((one_shot_output == sum_of_ranks * (phase + 1) / 16).all())
        assert bool((two_shot == sum_of_ranks * (phase + 1) / 64).all())

    update(0)
    check(run((8,)), 0)
    # Token counts are runtime arguments: after one warm count, counts of one,
    # multiples of 16 and others all reuse its variant.
    update(1)
    with assert_no_triton_compile(iris_fused_lamport_column_routed_rmsnorm_sum):
        check(run(counts), 1)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(counts)
    for phase in range(2, 6):
        update(phase)
        if rank == (phase * 3) % world_size:
            torch.cuda._sleep(20000)
        graph.replay()
        torch.cuda.synchronize()
        check(captured, phase)
    dist.barrier()
    assert bool((state.column_routes_region == -(2**31)).all())
    epochs = [torch.empty_like(state.column_routes_epochs) for _ in range(world_size)]
    dist.all_gather(epochs, state.column_routes_epochs)
    assert all(torch.equal(e, state.column_routes_epochs) for e in epochs)


def _muse_intermediate_exchange_worker(rank, world_size, device):
    """Expert TP prefill gathers sorted intermediates and returns column-split rows.

    Every rank sorts the same top-k IDs in their fixed order, so the gather
    must place each rank's slice at its K range of the same sorted row; the
    routed norm adds the ranks' sums of squares in rank order and must match
    the portable norm of the assembled routes within BF16 rounding, with zero
    padding rows and bitwise repeats. Row counts change between calls. One
    graph replays three back-to-back layers that each rewrite the slot, with a
    route reduce-scatter between them, while one rank is delayed: this guards
    the exit waits that keep a rank from overwriting buffers peers still read.
    """
    from tokenspeed_kernel.ops.communication._iris.muse_spark import (
        prepare_muse_communication,
    )
    from tokenspeed_kernel.ops.moe import routed_input_scale_rmsnorm_sum
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
        MXFP4SortedRoutes,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import (
        gluon_moe_sorting,
    )

    torch.set_grad_enabled(False)
    capacity = 8192
    state = prepare_muse_communication(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=0,
        producer_direct_max_numel=capacity * 8192,
        enable_lamport=True,
        owned_route_max_rows=2049,
        intermediate_gather_max_rows=capacity,
        all_reduce=None,
        device=device,
    )
    assert not state.can_gather_expert_intermediate(191)
    assert not state.can_gather_expert_intermediate(capacity + 1)
    slice_bytes = 4096 // world_size // 2
    width = 4096 // world_size
    norm = torch.linspace(0.5, 1.5, 4096, device=device).bfloat16()

    def sorted_routes(rows):
        generator = torch.Generator(device=device).manual_seed(rows)
        ids = torch.rand(rows, 256, device=device, generator=generator).topk(8).indices
        weights = torch.rand(rows, 8, device=device, generator=generator)
        block = 64
        sorted_ids, sorted_weights, expert_ids, valid, _ = gluon_moe_sorting(
            ids.to(torch.int32).contiguous(),
            weights,
            256,
            4096,
            torch.bfloat16,
            block,
            use_minimum_route_chunks=False,
            runtime_rows=False,
            deterministic_order=True,
        )
        routes = MXFP4SortedRoutes(
            sorted_ids, sorted_weights, expert_ids, valid, block, rows, 8
        )
        return routes, weights

    def fill(rows, routes, seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        data, scales = state.acquire_expert_intermediate(rows)
        sorted_rows = routes.sorted_ids.shape[0]
        panels = (sorted_rows + 31) // 32 * 32
        data[:sorted_rows].copy_(
            torch.randint(
                0, 256, (sorted_rows, slice_bytes), device=device, generator=generator
            )
        )
        scales[:panels].copy_(
            torch.randint(
                118,
                124,
                (panels, slice_bytes // 16),
                device=device,
                generator=generator,
            )
        )
        columns = torch.randn(
            rows * 8, width, device=device, generator=generator
        ).bfloat16()
        return columns

    def expected_gather(rows, routes):
        data, scales = state.acquire_expert_intermediate(rows)
        sorted_rows = routes.sorted_ids.shape[0]
        panels = (sorted_rows + 31) // 32 * 32
        parts = [torch.empty_like(data[:sorted_rows]) for _ in range(world_size)]
        dist.all_gather(parts, data[:sorted_rows].contiguous())
        panel_parts = [torch.empty_like(scales[:panels]) for _ in range(world_size)]
        dist.all_gather(panel_parts, scales[:panels].contiguous())
        rows_data = torch.cat(parts, dim=1)
        rows_scales = torch.cat(
            [part.view(panels // 32, -1) for part in panel_parts], dim=1
        )
        return rows_data, rows_scales

    def check_gather(rows, routes, full, full_scales, expected):
        valid = int(routes.num_valid_ids[0])
        real = (routes.sorted_ids[:valid] & 0xFFFFFF) < rows
        assert torch.equal(full[:valid][real], expected[0][:valid][real])
        panels = (valid + 31) // 32
        assert torch.equal(full_scales.view(-1, 4096)[:panels], expected[1][:panels])

    def expected_rows(rows, columns, weights):
        parts = [torch.empty_like(columns) for _ in range(world_size)]
        dist.all_gather(parts, columns)
        complete = torch.cat(parts, dim=1).view(rows, 8, 4096)
        return routed_input_scale_rmsnorm_sum(
            complete, weights, norm, 1e-6, 1.0, False, None
        )

    def check_rows(rows, actual, reference):
        local_rows = (rows + world_size - 1) // world_size
        first = rank * local_rows
        valid = min(local_rows, rows - first)
        assert actual.shape == (local_rows, 4096)
        torch.testing.assert_close(
            actual[:valid], reference[first : first + valid], atol=2**-7, rtol=2**-7
        )
        assert not actual[valid:].any()

    for rows in (193, 2049, capacity - 1, 192, capacity):
        routes, weights = sorted_routes(rows)
        columns = fill(rows, routes, 1000 * rank + rows)
        expected = expected_gather(rows, routes)
        full, full_scales = state.gather_expert_intermediate(routes, rows)
        check_gather(rows, routes, full, full_scales, expected)
        actual = state.column_routed_rmsnorm_sum(
            columns, weights, norm, 1e-6, 1.0, rows
        )
        check_rows(rows, actual, expected_rows(rows, columns, weights))
        again = state.column_routed_rmsnorm_sum(columns, weights, norm, 1e-6, 1.0, rows)
        assert torch.equal(again, actual)

    # Back-to-back layers in one graph: each overwrites the slot before its
    # gather, so a missing exit wait lets a fast rank clobber a slow peer's
    # reads. A route reduce-scatter (as on the BF16 expert layers) sits between.
    rows = 2049
    routes, weights = sorted_routes(rows)
    data, scales = state.acquire_expert_intermediate(rows)
    sorted_rows = routes.sorted_ids.shape[0]
    panels = (sorted_rows + 31) // 32 * 32
    layers = 3
    sources = [
        (
            torch.empty_like(data[:sorted_rows]),
            torch.empty_like(scales[:panels]),
            torch.empty((rows * 8, width), device=device, dtype=torch.bfloat16),
        )
        for _ in range(layers)
    ]
    staged = state.acquire_owned_route_input(rows)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    outputs = []
    with torch.cuda.graph(graph):
        for layer, (source, source_scales, columns) in enumerate(sources):
            data[:sorted_rows].copy_(source)
            scales[:panels].copy_(source_scales)
            full, full_scales = state.gather_expert_intermediate(routes, rows)
            summed = state.column_routed_rmsnorm_sum(
                columns, weights, norm, 1e-6, 1.0, rows
            )
            outputs.append((full, full_scales, summed))
            if layer == 1:
                reduced = state.reduce_scatter_owned_routes(staged, None, rows).clone()
    for replay in range(3):
        expected = []
        for layer, (source, source_scales, columns) in enumerate(sources):
            columns.copy_(fill(rows, routes, 7 * replay + 3 * layer + rank))
            source.copy_(data[:sorted_rows])
            source_scales.copy_(scales[:panels])
            expected.append(
                (
                    expected_gather(rows, routes),
                    expected_rows(rows, columns, weights),
                )
            )
        staged.copy_(torch.randn(staged.shape, device=device).bfloat16())
        complete = staged.float()
        dist.all_reduce(complete)
        local_rows = (rows + world_size - 1) // world_size
        torch.cuda.synchronize()
        dist.barrier()
        if rank == replay % world_size:
            torch.cuda._sleep(2_000_000)
        graph.replay()
        torch.cuda.synchronize()
        for (full, full_scales, summed), (gathered, reference) in zip(
            outputs, expected
        ):
            check_gather(rows, routes, full, full_scales, gathered)
            check_rows(rows, summed, reference)
        first = rank * local_rows
        valid = min(local_rows, rows - first)
        torch.testing.assert_close(
            reduced[:valid].float(),
            complete.view(rows, 8, 4096)[first : first + valid],
            atol=2**-5,
            rtol=2**-7,
        )
    dist.barrier()


def _muse_owned_stream_tap_worker(rank, world_size, device):
    """Gather BF16-rounded owned FP32 residual rows between other collectives
    that use pull scratch."""
    torch.set_grad_enabled(False)
    state = _muse_state(rank, device, 8192, False)

    def make_inputs(rows):
        local_rows = (rows + world_size - 1) // world_size
        stream = torch.empty((local_rows, 8192), device=device, dtype=torch.float32)
        branch = torch.empty((rows, 8192), device=device, dtype=torch.bfloat16)
        return rows, stream, branch

    def update(item, step):
        rows, stream, branch = item
        local_rows = stream.shape[0]
        # Every rank derives every owner's rows; random FP32 exercises rounding.
        generator = torch.Generator(device=device).manual_seed(rows * 31 + step)
        full = torch.randn(
            (world_size * local_rows, 8192),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        full[rows:] = float("nan")
        stream.copy_(full[rank * local_rows : (rank + 1) * local_rows])
        branch.fill_((rank + 1) * (step + 1))
        return full[:rows].bfloat16()

    def one(item):
        rows, stream, branch = item
        # A reduce-scatter result occupies pull scratch first, as attention
        # output does.
        reduced = state.reduce_scatter_owned_rows(branch)
        retained = state.all_gather_owned_rows(reduced, rows, True)
        if rank == rows % world_size:
            torch.cuda._sleep(20000)
        tap = state.all_gather_owned_stream_bf16(stream, rows)
        # Staging different data right away fails the check below if a peer
        # is still reading the gathered rows.
        staged = state.all_gather_owned_rows((-stream).bfloat16(), rows, False)
        return retained, tap, staged

    inputs = [make_inputs(rows) for rows in (192, 193, 8191)]
    for item in inputs:
        update(item, 0)
        one(item)
    state.row_gather_flags.fill_(2**32 - 2)
    state.row_scatter_flags.fill_(2**32 - 2)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [one(item) for item in inputs]
    for step in range(4):
        expected = [update(item, step) for item in inputs]
        graph.replay()
        for (retained, tap, staged), full in zip(outputs, expected):
            torch.testing.assert_close(
                tap.view(torch.int16), full.view(torch.int16), atol=0, rtol=0
            )
            torch.testing.assert_close(
                staged.view(torch.int16), (-full).view(torch.int16), atol=0, rtol=0
            )
            torch.testing.assert_close(
                retained,
                torch.full_like(
                    retained, (world_size * (world_size + 1) // 2) * (step + 1)
                ),
                atol=0,
                rtol=0,
            )
    torch.cuda.synchronize()
    dist.barrier()


@pytest.mark.parametrize(
    "check,backend",
    [
        pytest.param(
            partial(_muse_boundary_worker, sharded=False), "nccl", id="boundary_push"
        ),
        pytest.param(
            partial(_muse_boundary_worker, sharded=True), "nccl", id="boundary_scatter"
        ),
        pytest.param(_muse_collective_policy_worker, "nccl", id="collective_policy"),
        pytest.param(_muse_pull_warmup_worker, "nccl", id="pull_warmup"),
        pytest.param(_muse_owned_packet_worker, "gloo", id="owned_packets"),
        pytest.param(_muse_routed_lamport_worker, "nccl", id="routed_lamport"),
        pytest.param(
            _muse_lamport_gather_mxfp4_worker, "nccl", id="lamport_gather_mxfp4"
        ),
        pytest.param(
            _muse_lamport_gather_mxfp4_topk_worker,
            "nccl",
            id="lamport_gather_mxfp4_topk",
        ),
        pytest.param(
            _muse_decode_step_lamport_worker, "nccl", id="decode_step_lamport"
        ),
        pytest.param(_muse_moetail_worker, "nccl", id="moetail"),
        pytest.param(
            partial(_muse_sliding_attention_tail_worker, first_epoch=0),
            "nccl",
            id="sliding_attention_tail",
        ),
        pytest.param(
            partial(_muse_sliding_attention_tail_worker, first_epoch=2**31 - 2),
            "nccl",
            id="sliding_attention_tail_epoch_wrap",
        ),
        pytest.param(_muse_lamport_payload_worker, "nccl", id="lamport_payloads"),
        pytest.param(_muse_owned_stream_tap_worker, "nccl", id="owned_stream_taps"),
        pytest.param(_muse_tp_route_scatter_worker, "nccl", id="tp_route_scatter"),
        pytest.param(_muse_tp_column_routes_worker, "nccl", id="tp_column_routes"),
        pytest.param(
            _muse_intermediate_exchange_worker, "nccl", id="intermediate_exchange"
        ),
    ],
)
@pytest.mark.parametrize("world_size", [4, 8])
def test_muse_collective_replay(check, backend, world_size):
    _skip_if_unsupported(world_size, "Muse collective replay")
    if not current_platform().is_cdna4:
        pytest.skip("Muse collectives require CDNA4")
    _spawn_and_collect(
        _communication_worker,
        (world_size, _get_open_port(), check, backend),
        world_size,
    )
