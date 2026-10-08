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


"""Checkpoint-free numerical checks of DP projections with real collectives."""

import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-2gpu")


def _context(rank, counts):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=counts[rank],
        num_extends=0,
        input_num_tokens=counts[rank],
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, counts[rank], 1),
        global_num_tokens=counts,
    )


def _worker(rank, size, tp_size, rendezvous, backend_name):
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
        TrtllmAllReduceBackend,
    )
    from tokenspeed.runtime.distributed.comm_ops import (
        acquire_projection_output,
        prepare_projection_collectives,
        projection_all_gather,
        projection_reduce_scatter,
    )
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.execution.context import report_collective_sizing
    from tokenspeed.runtime.layers.linear import (
        DPColumnParallelLinear,
        DPRowParallelLinear,
        prepare_dp_linear_communication,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=240),
        device_id=device,
    )
    # This mapping shards projection weights; each rank owns independent tokens,
    # including when one TP group spans the world (projection dp_size == 1).
    parallel = DenseLayerMapping(
        rank=rank, world_size=size, tp_size=tp_size, dp_size=size // tp_size
    )
    width, stored, logical, capacity = 512, 256, 239, 513
    with torch.device(device):
        columns = [
            DPColumnParallelLinear(
                width,
                logical,
                padded_output_size=stored,
                parallel=parallel,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix="qkv_proj",
            )
            for _ in range(2)
        ]
        row = DPRowParallelLinear(
            width,
            stored,
            parallel=parallel,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix="o_proj",
        )
    # Every channel shard contributes, and distinct token/channel values expose
    # permutations. These sparse, power-of-two products are exact in BF16.
    weight = torch.zeros(stored, width, device=device, dtype=torch.bfloat16)
    indices = torch.arange(stored, device=device)
    for shard in range(tp_size):
        weight[
            indices, (indices * 7) % (width // tp_size) + shard * (width // tp_size)
        ] = 1
    column_weight = weight.clone()
    column_weight[logical:] = 0
    for column in columns:
        column.weight.weight_loader(column.weight, column_weight)
        column.quant_method.process_weights_after_loading(column)
    row.weight.weight_loader(row.weight, weight)
    row.quant_method.process_weights_after_loading(row)
    model = torch.nn.ModuleList([*columns, row])
    backend = AutoBackend() if backend_name == "auto" else NcclBackend()
    prepare_dp_linear_communication(model, capacity, torch.bfloat16, backend)

    # Separate preparations keep borrowed activations and producer-direct
    # destinations independent, even for identical specs. The NCCL case also
    # exercises the generic projection path inherited by an AllReduce wrapper.
    other_backend = (
        TrtllmAllReduceBackend(fallback=backend) if backend_name == "nccl" else backend
    )
    other_column, other_row = [
        prepare_projection_collectives(linear.projection_workspace.spec, other_backend)
        for linear in (columns[0], row)
    ]
    local = torch.full((32, width), rank + 1, dtype=torch.bfloat16, device=device)
    expected = torch.cat(
        [torch.full_like(local, peer + 1) for peer in parallel.tp_group]
    )
    gathered, _ = projection_all_gather(
        local, 32, False, columns[0].projection_workspace, backend
    )
    other_gathered, _ = projection_all_gather(
        -local, 32, False, other_column, other_backend
    )
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    torch.testing.assert_close(other_gathered, -expected, rtol=0, atol=0)
    partial = acquire_projection_output(32, row.projection_workspace, backend)
    partial.fill_(rank + 1)
    other_partial = acquire_projection_output(32, other_row, other_backend)
    other_partial.fill_(-(rank + 1))
    reduced = projection_reduce_scatter(partial, 32, row.projection_workspace, backend)
    other_reduced = projection_reduce_scatter(
        other_partial, 32, other_row, other_backend
    )
    expected = torch.full_like(reduced, sum(peer + 1 for peer in parallel.tp_group))
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
    torch.testing.assert_close(other_reduced, -expected, rtol=0, atol=0)
    other_column.close()
    other_row.close()

    # Same-shape layers share communication, but previous forward results must
    # remain valid when the next layer overwrites its borrowed intermediate.
    # These calls also exercise the original workspaces after the others close.
    cases = (
        [32] * size,
        [64] * size,
        [128] * size,
        [0] + [32] * (size - 1),
        [1] + [0] * (size - 1),
        [0] * size,
        [129] * size,
        [513] * size,
    )
    for counts in cases:
        values = torch.arange(counts[rank] * width, device=device).view(
            counts[rank], width
        )
        x = ((values % 11 - 5 + rank) / 8).to(torch.bfloat16)
        ctx = _context(rank, counts)
        for linear, following, full_weight, output_width in (
            (columns[0], columns[1], column_weight, logical),
            (row, row, weight, stored),
        ):
            expected = (x.float() @ full_weight.float().T)[:, :output_width].to(x.dtype)
            actual, bias = linear(x, ctx=ctx)
            assert bias is None
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            following(-x, ctx=ctx)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            # Models narrowing their row axis override scheduler counts without
            # changing Linear's call site or constructing a different path.
            wide_ctx = _context(rank, [capacity] * size)
            with report_collective_sizing(wide_ctx, counts[rank], counts):
                narrowed, _ = linear(x, ctx=wide_ctx)
                torch.testing.assert_close(narrowed, expected, rtol=0, atol=0)
            if not max(counts):
                continue
            for _ in range(2):
                linear(x, ctx=ctx)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, _ = linear(x, ctx=ctx)
            original = x.clone()
            for factor in (0.5, -1, 0):
                x.copy_(original * factor)
                graph.replay()
                torch.testing.assert_close(captured, expected * factor, rtol=0, atol=0)
            x.copy_(original)
            torch.cuda.synchronize()
            del graph, captured
    columns[0].projection_workspace.close()
    row.projection_workspace.close()
    if backend_name == "auto" and tp_size == 4:
        from tokenspeed.runtime.distributed.comm_backend.projection import (
            ProjectionSpec,
        )
        from tokenspeed.runtime.utils.env import global_server_args_dict

        # Cancellation distinguishes the required rank-ordered FP32 fold from
        # a shape-dependent BF16 reduction tree. Resolve the global backend for
        # both preparation and execution, matching the ordinary comm_ops API.
        global_server_args_dict["batch_invariant_collectives"] = True
        workspace = prepare_projection_collectives(
            ProjectionSpec(
                parallel.tp_group, "row", width, stored, 128, torch.bfloat16, device
            ),
            None,
        )
        for rows in (32, 128):
            partial = acquire_projection_output(rows, workspace, None)
            partial.fill_((1e8, 1, -1e8, 1)[rank])
            reduced = projection_reduce_scatter(partial, rows, workspace, None)
            torch.testing.assert_close(
                reduced, torch.ones_like(reduced), rtol=0, atol=0
            )
        workspace.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4], ids=["tp2", "tp4"])
@pytest.mark.parametrize("backend_name", ["nccl", "auto"])
def test_dp_linears(size, backend_name, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} GPUs")
    # With four GPUs, TP2 also exercises two isolated subgroups. The
    # one-active-owner case leaves a whole subgroup idle.
    world_size = 4 if torch.cuda.device_count() >= 4 else 2
    mp.spawn(
        _worker,
        args=(world_size, size, f"file://{tmp_path / 'rendezvous'}", backend_name),
        nprocs=world_size,
        join=True,
    )


def _fp8_worker(rank, rendezvous):
    from tokenspeed_kernel.ops.gemm.fp8_utils import per_block_quant_fp8

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.layers.linear import (
        DPColumnParallelLinear,
        DPRowParallelLinear,
        prepare_dp_linear_communication,
    )
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=240),
        device_id=device,
    )
    # TP4 projection over four independent token owners, not attention TP4.
    parallel = DenseLayerMapping(rank=rank, world_size=4, tp_size=4, dp_size=1)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )
    # Production KDA widths, without a checkpoint, attention or KV cache.
    for kind, k, n in (("column", 7168, 66048), ("row", 16384, 7168)):
        torch.manual_seed(19)
        weight = torch.randn(n, k, device=device, dtype=torch.bfloat16) / k**0.5
        quantized, scales = per_block_quant_fp8(weight, (128, 128), 1e-10)
        with torch.device(device):

            def make_linear():
                kwargs = dict(
                    parallel=parallel,
                    params_dtype=torch.bfloat16,
                    quant_config=config,
                    prefix="projection",
                )
                if kind == "column":
                    return DPColumnParallelLinear(k, n, padded_output_size=n, **kwargs)
                return DPRowParallelLinear(k, n, **kwargs)

            baseline, optimized = make_linear(), make_linear()
        for linear, backend in ((baseline, NcclBackend()), (optimized, AutoBackend())):
            linear.weight.weight_loader(linear.weight, quantized)
            linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, scales)
            linear.quant_method.process_weights_after_loading(linear)
            prepare_dp_linear_communication(linear, 128, torch.bfloat16, backend)
        for counts in ([32] * 4, [0, 64, 32, 1], [128] * 4):
            torch.manual_seed(20 + rank)
            x = torch.randn(counts[rank], k, device=device, dtype=torch.bfloat16)
            ctx = _context(rank, counts)
            expected, _ = baseline(x, ctx=ctx)
            actual, _ = optimized(x, ctx=ctx)
            if x.numel():
                reference = x.float() @ weight.float().T
                # Quantization error is not a redistribution error. NCCL and
                # Lamport may round row partials differently; both must stay
                # within the much smaller BF16 accumulation error budget.
                magnitude = reference.square().mean().sqrt()
                assert (
                    expected.float() - reference
                ).square().mean().sqrt() < 0.06 * magnitude
                assert (
                    actual.float() - expected.float()
                ).square().mean().sqrt() < 0.007 * magnitude
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, _ = optimized(x, ctx=ctx)
            x.mul_(0.5).add_(0.125)
            refreshed, _ = optimized(x, ctx=ctx)
            graph.replay()
            torch.testing.assert_close(captured, refreshed, rtol=0, atol=0)
            torch.cuda.synchronize()
            del graph, captured
        baseline.projection_workspace.close()
        optimized.projection_workspace.close()
    dist.destroy_process_group()


def test_dp_linears_fp8(tmp_path):
    from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale

    if torch.cuda.device_count() < 4 or not has_flashinfer_fp8_blockscale():
        pytest.skip("requires four Blackwell GPUs and FlashInfer block-FP8 GEMM")
    mp.spawn(
        _fp8_worker,
        args=(f"file://{tmp_path / 'fp8-rendezvous'}",),
        nprocs=4,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
