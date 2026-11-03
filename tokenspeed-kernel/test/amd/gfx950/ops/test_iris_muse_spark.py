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
from types import SimpleNamespace

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
                n = 129 * 8192 if index % 2 else 43 * 4096
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


def _prepare_muse_backend(world_size, projection, moe, *, expert_tp_size):
    from tokenspeed.runtime.distributed import process_group_manager as pgm
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.models.muse_spark import MuseSparkForConditionalGeneration

    group = tuple(range(world_size))
    pgm.process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    backend = initialize_comm_backend()
    if projection is None:
        projection = SimpleNamespace(iris_state=None)
    if moe is None:
        moe = SimpleNamespace(
            iris_state=None, pre_expert_proj=None, post_expert_proj=None
        )
    layer = SimpleNamespace(
        iris_state=None,
        self_attn=SimpleNamespace(iris_state=None, qkv_proj=projection),
        mlp=moe,
    )
    owner = SimpleNamespace(
        mapping=SimpleNamespace(
            attn=SimpleNamespace(tp_size=world_size, tp_group=group),
            moe=SimpleNamespace(
                tp_ep_group=group,
                tp_size=expert_tp_size,
                ep_size=world_size // expert_tp_size,
            ),
        ),
        config=SimpleNamespace(
            hidden_size=8192, moe_hidden_size=4096, num_experts_per_tok=8
        ),
        model=SimpleNamespace(iris_state=None, layers=[layer]),
    )
    assert MuseSparkForConditionalGeneration.prepare_communication_runtime(owner, 8192)
    state = owner.model.iris_state
    assert MuseSparkForConditionalGeneration.prepare_communication_runtime(owner, 8192)
    assert owner.model.iris_state is state
    assert projection.iris_state is state and moe.iris_state is state
    assert backend.get_prepared_all_gather(group, torch.bfloat16) is state.gather
    return group, state


def _muse_owned_qkvg_worker(rank, world_size, device):
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    from tokenspeed_kernel.ops.attention.prologue import (
        HeadKVCache,
        HeadNorm,
        RopeStyle,
        Rotary,
        gqa_prologue,
    )
    from tokenspeed_kernel.ops.communication._iris.muse_spark.all_to_all import (
        iris_a2a_transpose_qkvg,
        iris_fused_a2a_transpose_qkvg_norm_rope,
    )
    from utils import assert_no_triton_compile

    from tokenspeed.runtime.models.muse_spark import (
        MUSE_OWNED_QKVG_MIN_TOKENS,
        QKVOGateParallelLinear,
    )

    group = tuple(range(world_size))
    with torch.device(device):
        projection = QKVOGateParallelLinear(
            hidden_size=8192,
            head_size=64,
            total_num_heads=128,
            total_num_kv_heads=16,
            quant_config=None,
            prefix="model.layers.1.self_attn.qkv_proj",
            tp_rank=rank,
            tp_size=world_size,
            tp_group=group,
            replicate_for_owned_rows=True,
        )
    group, state = _prepare_muse_backend(world_size, projection, None, expert_tp_size=1)
    full_weights = {}
    expected_columns = []
    for index, (name, width) in enumerate(
        (("q", 8192), ("k", 1024), ("v", 1024), ("g", 8192))
    ):
        weight = torch.zeros(width, 8192, device=device)
        columns = (torch.arange(width, device=device) * 13 + index * 17) % 8192
        weight[torch.arange(width, device=device), columns] = 1
        full_weights[name] = weight
        projection.weight_loader(projection.weight, weight, name)
        expected_columns.append(
            columns[rank * (width // world_size) : (rank + 1) * (width // world_size)]
        )
    expected_columns = torch.cat(expected_columns)
    parameter_pointer = projection.weight.data_ptr()
    local_pointer = projection._local_weight().data_ptr()

    angles = torch.arange(16384, device=device, dtype=torch.float32)[:, None] / (
        torch.arange(32, device=device, dtype=torch.float32)[None, :] + 1
    )
    rope = torch.cat((angles.cos(), angles.sin()), dim=1)
    nope = torch.empty((1, 0), device=device, dtype=torch.float32)
    eps, q_scale = 1e-5, 5.4730064863875
    norm = HeadNorm(
        torch.full((64,), q_scale, device=device, dtype=torch.float32),
        torch.ones((64,), device=device, dtype=torch.float32),
        0.0,
        eps,
    )

    def make_inputs(rows):
        local_rows = (rows + world_size - 1) // world_size
        first = rank * local_rows
        valid = min(local_rows, rows - first)
        full = (
            (
                torch.arange(rows, device=device)[:, None] % 13
                + torch.arange(8192, device=device)[None, :] % 7
            )
            / 16
        ).bfloat16()
        local = torch.full((local_rows, 8192), float("nan"), device=device)
        local[:valid].copy_(full[first : first + valid])
        poison = torch.full((rows, 8192), rank + 1, dtype=torch.bfloat16, device=device)
        positions = (
            torch.arange(rows, device=device, dtype=torch.int64) * 17 + 31
        ) % 16384
        k_cache = torch.empty(
            (rows, 16 // world_size, 64), dtype=torch.bfloat16, device=device
        )
        cache = HeadKVCache(
            k_cache,
            torch.empty_like(k_cache),
            None,
            torch.arange(rows, device=device, dtype=torch.int64),
        )
        return full, local, poison, positions, cache

    inputs = [make_inputs(rows) for rows in (2047, 2048, 8191)]

    def run(item):
        full, local, poison, positions, cache = item
        generic = state.all_reduce.acquire_outputs(
            ((state.reduction.push_max_numel + 8192,),)
        )[0]
        generic.fill_(rank + 1)
        previous = state.reduce_outputs((generic,))[0]
        fused = reference = None
        if full.shape[0] >= MUSE_OWNED_QKVG_MIN_TOKENS:
            projected = projection.project_owned_rows(local, full.shape[0])
            output = state.all_to_all_transpose_qkvg(projected, full.shape[0])
            table = rope if full.shape[0] < 4096 else nope
            q, k, v, gate = output.split(
                tuple(width // world_size for width in (8192, 1024, 1024, 8192)), dim=-1
            )
            prepared = gqa_prologue(
                q,
                k,
                v,
                norm=norm,
                rotary=(
                    Rotary(table, positions, RopeStyle.GPTJ, None)
                    if table.shape[1]
                    else None
                ),
                cache=cache,
                return_kv=True,
                solution="triton",
                override=None,
            )
            reference = (prepared.q, prepared.k, prepared.v, gate)
            projected = projection.project_owned_rows(local, full.shape[0])
            if rank == full.shape[0] % world_size:
                torch.cuda._sleep(20000)
            fused = state.all_to_all_transpose_qkvg_norm_rope(
                projected, full.shape[0], positions, table, eps, q_scale
            )
        else:
            normalized = state.all_gather_owned_rows(local, full.shape[0], True)
            output, _ = projection(normalized)
        # O's next producer can overwrite every old QKVG source address.
        reduced = state.reduce_scatter_owned_rows(poison)
        gathered = state.all_gather_owned_rows(reduced, full.shape[0], True)
        return output, gathered, previous, fused, reference

    def sequence():
        return [run(item) for item in inputs]

    sequence()
    state.row_gather_flags.fill_(2**32 - 2)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = sequence()
    for replay in range(4):
        gain = 2 ** (replay + 1)
        for name, weight in full_weights.items():
            weight.mul_(2)
            projection.weight_loader(projection.weight, weight, name)
        assert projection.weight.data_ptr() == parameter_pointer
        assert projection._local_weight().data_ptr() == local_pointer
        for full, local, _, positions, _ in inputs:
            positions.add_(97).remainder_(16384)
            full.add_(1 / 16)
            first = rank * local.shape[0]
            valid = min(local.shape[0], full.shape[0] - first)
            local[:valid].copy_(full[first : first + valid])
        if rank == replay:
            torch.cuda._sleep(10000)
        graph.replay()
        for (full, *_), (output, gathered, previous, fused, reference) in zip(
            inputs, outputs
        ):
            if fused is not None:
                for actual, expected in zip(fused, reference):
                    assert actual.is_contiguous() and bool(torch.isfinite(actual).all())
                    torch.testing.assert_close(
                        actual.view(torch.int16),
                        expected.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
            expected = full.index_select(1, expected_columns) * gain
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                gathered,
                torch.full_like(gathered, world_size * (world_size + 1) // 2),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                previous,
                torch.full_like(previous, world_size * (world_size + 1) // 2),
                rtol=0,
                atol=0,
            )
        small = inputs[0][0][:4].contiguous()
        small_output, _ = projection(small)
        torch.testing.assert_close(
            small_output,
            small.index_select(1, expected_columns) * gain,
            rtol=0,
            atol=0,
        )
    with assert_no_triton_compile(
        iris_a2a_transpose_qkvg, iris_fused_a2a_transpose_qkvg_norm_rope
    ):
        for rows in (2051, 8190):
            output, _, _, fused, reference = run(make_inputs(rows))
            for actual, expected in zip(fused, reference):
                torch.testing.assert_close(
                    actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
                )
    torch.cuda.synchronize()
    dist.barrier()


def _compile_gfx950(fn, types, constexprs, num_warps, unspecialized=()):
    """Compile a Gluon kernel for gfx950 without launching it.

    Arguments named in unspecialized get no 16-byte divisibility, as when a
    caller passes them in do_not_specialize or compiles ahead of time.
    """
    from tokenspeed_kernel._triton import gluon, triton

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
        options={"num_warps": num_warps},
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


def test_attntail_lamport_polling_codegen():
    """Each subgroup of the 16-subgroup poller retries on its own.

    A workgroup-wide readiness reduction would put shared-memory traffic and
    barriers into every retry; the subgroup ballot keeps them out.
    """
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
    assembly = kernel.asm["amdgcn"]
    polls = [
        cycle for cycle in polling_cycles(assembly) if _batched_peer_reads(cycle, 7)
    ]
    assert polls, "all seven system-cache reads must issue together on a retry"
    assert all(
        votes_with_subgroup_mask(cycle) for cycle in polls
    ), "a retry must branch on one subgroup lane mask"
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)


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

    for rows in (1, 3, 16):
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
        triton_routed_input_scale_rmsnorm_sum_gathered,
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
                triton_routed_input_scale_rmsnorm_sum_gathered(
                    routes,
                    mapping,
                    rows,
                    8,
                    weights,
                    norm,
                    1e-5,
                    1.0,
                    False,
                    local,
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

    # The replicated single-row projection of each row, then the unfused boundary.
    branch = x.new_empty((x.shape[0], weight.shape[0]))
    for row in range(x.shape[0]):
        gluon_rowcta_bf16_gemv_gfx950(x[row : row + 1], weight, branch[row : row + 1])
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
        return_pre_fp32=False,
    )
    return out_stream, pre_bf16


def _muse_moetail_worker(rank, world_size, device):
    """The fused tail matches per-row replicated single-row GEMVs plus the boundary.

    Row counts cover every power-of-two bucket at TP8 and odd counts inside
    them (TP4 takes one row). Two layers alternate weights and next norms, one
    rank lags, row counts interleave in one graph, and the stage counters start
    at the last stage so eager calls and replays wrap them.
    """
    state = _muse_state(rank, device, 1, True)
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
    counts = (1, 2, 3, 4, 5, 8) if world_size == 8 else (1,)
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


def _muse_decode_step_lamport_worker(rank, world_size, device):
    """Interleave every Lamport user of one decode layer under graph replay.

    The routed-normalization, gather-norm-MXFP4, post-expert tail and
    generic one-shot BF16 reductions share one sentinel mailbox and its
    per-512-element stage counters while covering different tile prefixes, so
    the counters diverge per tile. TP8's generic two-shot (route payloads from
    two tokens) and the attention tail keep their own mailboxes but run in the
    same step.
    """
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


def _muse_tp_routes_worker(rank, world_size, device):
    """Expert TP sums aligned partial routes before nonlinear normalization."""

    from tokenspeed.runtime.models.muse_spark import MuseSparkMoE, MuseSparkRMSNorm

    torch.set_grad_enabled(False)
    width, topk = 4096, 8
    group = tuple(range(world_size))
    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(moe)
    moe.iris_state = None
    moe.pre_expert_proj = None
    moe.post_expert_proj = None
    moe.tp_group, moe.tp_size, moe.topk = group, world_size, topk
    moe.ep_group, moe.ep_size = (rank,), 1
    moe.post_expert_norm = MuseSparkRMSNorm(width, 1e-8, True, 1.0)
    norm_weight = torch.linspace(-0.25, 0.25, width, device=device).bfloat16()
    moe.post_expert_norm.weight = torch.nn.Parameter(norm_weight)
    _, state = _prepare_muse_backend(world_size, None, moe, expert_tp_size=world_size)
    dims = torch.arange(width, device=device)[None, :]

    # Full-row inputs exceed generic staging; owned inputs keep complete token
    # route groups together, including an uneven last owner and mapped outputs.
    cases = [
        (n, c, False) for n, c in ((1, False), (3, False), (16, False), (2049, False))
    ]
    cases += [(n, True, False) for n in (3, 6, 33)]
    cases += [
        (192, True, True),
        (193, True, True),
        (193, False, True),
        (2049, True, True),
        (8191, False, True),
        (8192, True, True),
        (2051, True, True),
    ]
    for tokens, canonical, owned in cases:
        indices = torch.arange(tokens * topk, device=device)
        mapping = None if canonical else ((indices + rank) % indices.numel()).int()
        weights = (
            torch.linspace(0.01, 1, topk, device=device).expand(tokens, -1).contiguous()
        )
        routes = (
            state.acquire_owned_route_input(tokens)
            if owned and canonical
            else torch.empty(
                (indices.numel(), width if canonical else width + 16),
                device=device,
                dtype=torch.bfloat16,
            )
        )

        def update(step):
            partial = (
                (indices[:, None] * 3 + dims * 5 + rank * 7 + step) % 9 - 4
            ).bfloat16() / 32
            routes[:, :width].copy_(
                partial if canonical else partial.roll(rank, dims=0)
            )
            routes[:, width:].fill_(torch.nan)

        def reference(step):
            local_rows = (tokens + world_size - 1) // world_size
            first = rank * local_rows if owned else 0
            count = min(local_rows, tokens - first) if owned else tokens
            route_indices = indices[first * topk : (first + count) * topk]
            full = sum(
                (
                    (route_indices[:, None] * 3 + dims * 5 + peer * 7 + step) % 9 - 4
                ).float()
                / 32
                for peer in range(world_size)
            ).reshape(count, topk, width)
            scaled = full * (norm_weight + 1.0).float()
            normalized = scaled * torch.rsqrt(
                scaled.square().mean(-1, keepdim=True) + 1e-8
            )
            expected = (
                (normalized * weights[first : first + count, :, None]).sum(1).bfloat16()
            )
            if owned and count < local_rows:
                expected = torch.nn.functional.pad(
                    expected, (0, 0, 0, local_rows - count)
                )
            return expected

        def run():
            return moe._reduce_expert_routes(
                routes.view(tokens, topk, width) if canonical else routes,
                mapping,
                tokens,
                weights,
                owned_rows=owned,
            )

        update(0)
        actual = run()
        torch.testing.assert_close(actual, reference(0), rtol=0, atol=2**-8)
        if tokens == 192 and owned and canonical:
            before = routes.clone()
            rotated = ((indices + 1) % indices.numel()).int()
            with pytest.raises(ValueError, match="must not overlap"):
                state.reduce_scatter_owned_routes(routes, rotated, tokens)
            shifted = state.owned_route_input.flatten()[
                width : width + routes.numel()
            ].view_as(routes)
            with pytest.raises(ValueError, match="must not overlap"):
                state.reduce_scatter_owned_routes(shifted, None, tokens)
            torch.testing.assert_close(routes, before, rtol=0, atol=0)
        if tokens <= 33 or owned:
            if owned:
                state.owned_route_scatter_flags.view(torch.int32).fill_(-2)
                state.all_reduce.peers.ctx.device_barrier()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for step in (1, 2):
                update(step)
                if rank == step % world_size:
                    torch.cuda._sleep(20000)
                graph.replay()
                if owned:
                    # A second reduction overwrites borrowed raw-route storage;
                    # the normalization result must retain its own contents.
                    state.reduce_scatter_owned_routes(routes, mapping, tokens)
                torch.testing.assert_close(
                    captured, reference(step), rtol=0, atol=2**-8
                )
        dist.barrier()


def _muse_tp_owned_forward_worker(rank, world_size, device, *, quantized: bool):
    """Exercise the real router/pool/gather/expert/unpool caller on uneven rows."""
    from tokenspeed.runtime.configs.muse_spark_config import MuseSparkConfig
    from tokenspeed.runtime.distributed import process_group_manager as pgm
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        initialize_comm_backend,
    )
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
    from tokenspeed.runtime.models.muse_spark import MuseSparkMoE

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=world_size,
        dense_tp_size=world_size,
        moe_tp_size=world_size,
        moe_ep_size=1,
        vision_tp_size=world_size,
        linear_attn_tp_size=world_size,
        nprocs_per_node=world_size,
        nnodes=1,
    )
    group = mapping.attn.tp_group
    pgm.process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    initialize_comm_backend()
    config = MuseSparkConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=8192,
            head_dim=64,
            num_attention_heads=128,
            num_key_value_heads=16,
            num_hidden_layers=1,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            final_logit_softcapping=20.0,
            qk_scale_factor=5.47,
            output_multiplier=0.125,
            residual_gate_temperature=0.3,
            moe_hidden_size=4096,
            moe_intermediate_size=4096 if quantized else 256,
            num_local_experts=256 if quantized else 8,
            num_experts_per_tok=8,
            num_shared_experts=0,
            routed_scaling_factor=1.0,
            layer_types=["full_attention"],
            layer_rope_theta=[0.0],
            mlp_layer_types=["sparse"],
            sliding_window=128,
            rope_parameters=dict(rope_theta=500000.0, rope_type="default"),
        ),
        vision_config={},
        audio_config={},
        image_token_id=200,
        video_token_id=201,
        audio_token_id=202,
        out_hidden_size=32,
        projector_hidden_size=32,
        projector_hidden_act="gelu",
    )
    quant_config = (
        Mxfp4Config(
            is_checkpoint_mxfp4_serialized=True,
            use_dynamic_mxfp4_activations=True,
        )
        if quantized
        else None
    )
    with torch.device(device):
        moe = MuseSparkMoE(config, mapping, quant_config, 0, "model.layers.0.mlp", None)
    torch.manual_seed(117)
    for parameter in moe.parameters():
        if parameter.is_floating_point():
            parameter.normal_(0, 0.02)
    # Sparse pool/unpool weights make the reference insensitive to GEMM tiling.
    for projection in (moe.pre_expert_proj, moe.post_expert_proj):
        projection.weight.zero_()
        index = torch.arange(projection.output_size, device=device)
        projection.weight[index, (index * 3) % projection.input_size] = 1
    torch.manual_seed(117 + rank)
    if quantized:
        moe.experts.w13_weight.random_(0, 256)
        moe.experts.w2_weight.random_(0, 256)
        moe.experts.w13_weight_scale.random_(118, 124)
        moe.experts.w2_weight_scale.random_(118, 124)
        moe.experts.process_weights_after_loading(moe.experts)
    else:
        moe.experts.w13_weight.normal_(0, 0.02)
        moe.experts.w2_weight.normal_(0, 0.02)
    moe.gate.process_weights_after_loading(moe.gate)
    _, state = _prepare_muse_backend(world_size, None, moe, expert_tp_size=world_size)
    for rows in (192, 193):
        local_rows = (rows + world_size - 1) // world_size
        first = rank * local_rows
        valid = min(local_rows, rows - first)
        for step in (0, 1):
            torch.manual_seed(rows + step)
            full = torch.randn(rows, 8192, device=device)
            local = torch.zeros(local_rows, 8192, device=device)
            local[:valid].copy_(full[first : first + valid])
            expected, _ = moe(full, full.float(), rows, rows, None, project=True)
            expected = expected[first : first + valid].clone()
            actual, _ = moe(local, local.float(), rows, rows, rows, project=True)
            actual = actual.clone()
            # Post projection borrows the generic producer; retain it before reuse.
            state.all_reduce.producer.data.fill_(torch.nan)
            torch.testing.assert_close(actual[:valid], expected, atol=2**-7, rtol=2**-7)
            torch.testing.assert_close(
                actual[valid:], torch.zeros_like(actual[valid:]), atol=0, rtol=0
            )
        dist.barrier()


def _muse_owned_routed_wt_worker(rank, world_size, device):
    torch.set_grad_enabled(False)
    from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
        iris_routed_rmsnorm_weighted_sum_write_through,
    )
    from tokenspeed_kernel.ops.communication._iris.reduce_scatter import (
        iris_pull_reduce_scatter,
    )
    from tokenspeed_kernel.ops.moe import (
        routed_input_scale_rmsnorm_sum,
        routed_input_scale_rmsnorm_sum_gathered,
    )
    from tokenspeed_kernel.platform import pdl_enabled
    from utils import assert_no_triton_compile

    from tokenspeed.runtime.models.muse_spark import MuseSparkMoE, MuseSparkRMSNorm

    group = tuple(range(world_size))
    torch.manual_seed(181)
    norm = torch.randn(4096, device=device, dtype=torch.bfloat16) / 8
    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(moe)
    moe.iris_state = None
    moe.pre_expert_proj = None
    moe.post_expert_proj = None
    moe.ep_group, moe.ep_size, moe.topk = group, world_size, 8
    moe.tp_size = 1
    moe.post_expert_norm = MuseSparkRMSNorm(4096, 1e-8, True, 1.0)
    moe.post_expert_norm.weight = torch.nn.Parameter(norm)
    group, state = _prepare_muse_backend(world_size, None, moe, expert_tp_size=1)

    def inputs(rows):
        torch.manual_seed(rows + rank * 17)
        original = torch.randn(rows * 8, 4096, device=device, dtype=torch.bfloat16)
        local_rows = (rows + world_size - 1) // world_size
        return dict(
            rows=rows,
            original=original,
            routes=torch.empty_like(original),
            index=torch.arange(rows * 8, device=device, dtype=torch.int32),
            mapping=torch.empty(rows * 8, device=device, dtype=torch.int32),
            weights=torch.full((rows, 8), 0.125, device=device),
            packet=torch.empty(local_rows, 4096, device=device, dtype=torch.bfloat16),
            ids=torch.zeros(local_rows, 8, device=device, dtype=torch.int32),
            scores=torch.ones(local_rows, 8, device=device),
            attention=torch.empty(rows, 8192, device=device, dtype=torch.bfloat16),
        )

    def update(case, step):
        owner = (case["index"] + step) % (world_size - 1 if step == 2 else world_size)
        case["mapping"].copy_(torch.where(owner == rank, case["index"], -1))
        case["routes"].copy_(
            torch.where(
                (owner == rank)[:, None],
                case["original"] * (1 + step / 8),
                torch.nan,
            )
        )
        case["packet"].fill_((rank + 1) * (step + 1) / 8)
        case["attention"].fill_((rank + 1) * (step + 1) / 16)

    def reference(routes, mapping, rows, weights):
        norm = moe.post_expert_norm
        if mapping is None:
            combined = routed_input_scale_rmsnorm_sum(
                routes,
                weights,
                norm.weight,
                norm.eps,
                norm.weight_offset,
                pdl_enabled(),
                None,
            )
            output = state.all_reduce.acquire_outputs((tuple(combined.shape),))[0]
            output.copy_(combined)
        else:
            output = state.all_reduce.acquire_outputs(((rows, 4096),))[0]
            routed_input_scale_rmsnorm_sum_gathered(
                routes,
                mapping,
                rows,
                8,
                weights,
                norm.weight,
                norm.eps,
                norm.weight_offset,
                pdl_enabled(),
                output,
            )
        return state.reduce_scatter_owned_rows(output).clone()

    def run(case):
        rows = case["rows"]
        expected = reference(case["routes"], case["mapping"], rows, case["weights"])
        producer = state.all_reduce.acquire_outputs(((rows, 8192),))[0]
        producer.copy_(case["attention"])
        state.reduce_scatter_owned_rows(producer)
        local_rows = (rows + world_size - 1) // world_size
        valid = max(0, min(local_rows, rows - rank * local_rows))
        packet, _, _ = state.gather_owned_moe_inputs(
            case["packet"], case["ids"], case["scores"], rows, valid
        )
        if rank == rows % world_size:
            torch.cuda._sleep(20000)
        actual = moe._reduce_expert_routes(
            case["routes"], case["mapping"], rows, case["weights"], owned_rows=True
        ).clone()
        upper = state.acquire_owned_post_expert_output(rows, 8192)
        upper[:, :4096].copy_(actual)
        upper[:, 4096:].copy_(actual * 2)
        full = state.all_gather_owned_rows(upper, rows, False)
        return expected, actual, packet, full

    cases = [inputs(rows) for rows in (192, 193, 8191)]
    for case in cases:
        update(case, 0)
        run(case)
    for flags in (state.row_scatter_flags, state.row_gather_flags):
        flags.view(torch.int32).fill_(-2)
    state.all_reduce.peers.ctx.device_barrier()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        saved = [run(case) for case in cases]
    for step in range(4):
        for case in cases:
            update(case, step)
        graph.replay()
        for case, (expected, actual, packet, full) in zip(cases, saved):
            torch.testing.assert_close(
                actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            rows = case["rows"]
            local_rows = (rows + world_size - 1) // world_size
            valid = max(0, min(local_rows, rows - rank * local_rows))
            torch.testing.assert_close(
                actual[valid:], torch.zeros_like(actual[valid:]), rtol=0, atol=0
            )
            owner = torch.arange(rows, device=device) // local_rows
            torch.testing.assert_close(
                packet,
                ((owner + 1) * (step + 1) / 8)
                .to(torch.bfloat16)[:, None]
                .expand_as(packet),
                rtol=0,
                atol=0,
            )
            first = rank * local_rows
            torch.testing.assert_close(
                full[first : first + valid, :4096], actual[:valid], rtol=0, atol=0
            )
            torch.testing.assert_close(
                full[first : first + valid, 4096:],
                (actual * 2)[:valid],
                rtol=0,
                atol=0,
            )
    unseen = inputs(2051)
    update(unseen, 0)
    with assert_no_triton_compile(
        iris_routed_rmsnorm_weighted_sum_write_through, iris_pull_reduce_scatter
    ):
        expected, actual, _, _ = run(unseen)
        torch.testing.assert_close(
            actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
    moe.post_expert_norm.weight_offset = 0.0
    expected, actual, _, _ = run(unseen)
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )
    # The unmapped BF16 layer retains the portable producer and release path.
    dense = cases[1]["original"].view(193, 8, 4096)
    expected = reference(dense, None, 193, cases[1]["weights"])
    actual = moe._reduce_expert_routes(
        dense, None, 193, cases[1]["weights"], owned_rows=True
    )
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
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
        pytest.param(_muse_owned_packet_worker, "gloo", id="owned_packets"),
        pytest.param(_muse_owned_qkvg_worker, "nccl", id="qkvg_reload_and_receiver"),
        pytest.param(_muse_routed_lamport_worker, "nccl", id="routed_lamport"),
        pytest.param(
            _muse_lamport_gather_mxfp4_worker, "nccl", id="lamport_gather_mxfp4"
        ),
        pytest.param(
            _muse_decode_step_lamport_worker, "nccl", id="decode_step_lamport"
        ),
        pytest.param(_muse_moetail_worker, "nccl", id="moetail"),
        pytest.param(_muse_lamport_payload_worker, "nccl", id="lamport_payloads"),
        pytest.param(_muse_owned_routed_wt_worker, "nccl", id="owned_routed_wt"),
        pytest.param(_muse_tp_routes_worker, "nccl", id="tp_routes"),
        pytest.param(
            partial(_muse_tp_owned_forward_worker, quantized=False),
            "nccl",
            id="tp_owned_forward",
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


def test_muse_tp8_owned_a4w4_forward():
    _skip_if_unsupported(8, "Muse A4W4 owned forward")
    _spawn_and_collect(
        _communication_worker,
        (
            8,
            _get_open_port(),
            partial(_muse_tp_owned_forward_worker, quantized=True),
            "nccl",
        ),
        8,
    )
