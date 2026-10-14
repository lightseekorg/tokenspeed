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

"""Numerical correctness of fused DP column-linear and CUDA graph replay."""

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


@torch.no_grad()
def _worker(rank, world_size, rendezvous):
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        create_fused_tp_projection_state,
        cute_dsl_fused_fp8_dp_column_linear,
    )
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.linear import (
        DPColumnParallelLinear,
        prepare_dp_linear_communication,
        release_dp_linear_communication,
    )
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    parallel = DenseLayerMapping(
        rank=rank, world_size=world_size, tp_size=world_size, dp_size=1
    )
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )
    models = []
    for backend in (NcclBackend(), AutoBackend()):
        with torch.device(device):
            model = torch.nn.ModuleList(
                [
                    DPColumnParallelLinear(
                        7168,
                        49376,
                        padded_output_size=(
                            (49376 + world_size * 128 - 1) // (world_size * 128)
                        )
                        * world_size
                        * 128,
                        parallel=parallel,
                        params_dtype=torch.bfloat16,
                        quant_config=config,
                        prefix=f"projection_{index}",
                    )
                    for index in range(2)
                ]
            )
        for index, linear in enumerate(model):
            torch.manual_seed(917 + rank + 4 * index)
            linear.weight.copy_(
                torch.randn_like(linear.weight, dtype=torch.bfloat16).to(
                    torch.float8_e4m3fn
                )
            )
            linear.weight_scale_inv.copy_(
                torch.rand_like(linear.weight_scale_inv) * 0.005 + 0.005
            )
        prepare_dp_linear_communication(model, 4097, torch.bfloat16, backend)
        models.append(model)
    baseline, candidate = models
    workspace = candidate[0].projection_workspace
    assert workspace is candidate[1].projection_workspace
    state = workspace.fused_column
    if world_size == 8:
        # TP8 is available only through an explicitly prepared native state.
        state = create_fused_tp_projection_state(
            dist.group.WORLD, device, 4097, 7168, candidate[0].output_size
        )
    else:
        assert state is not None

    def context(counts, physical):
        return ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=physical[rank],
            num_extends=0,
            input_num_tokens=physical[rank],
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, physical[rank], 1),
            global_num_tokens=counts,
            collective_global_num_tokens=physical,
        )

    def counts_for_group(pattern):
        return (tuple(pattern) * ((world_size + len(pattern) - 1) // len(pattern)))[
            :world_size
        ]

    def uses_fusion(rows):
        return world_size in (2, 4) and rows >= 64

    # Cover bucket boundaries, the shared dispatch cutoff, and small native GEMMs.
    equal_counts = [
        1,
        9,
        32,
        63,
        64,
        65,
        87,
        88,
        128,
        129,
        256,
        257,
        383,
        384,
        385,
        512,
        513,
        639,
        640,
        641,
        1024,
        1025,
        2048,
        2049,
        2304,
        2305,
        4096,
        4097,
    ]
    count_cases = [tuple([rows] * world_size) for rows in equal_counts]
    count_cases += [
        counts_for_group(pattern)
        for pattern in (
            (127, 128, 1, 0),
            (0, 65),
            (2048, 1536, 0, 1),
            (0, 0),
        )
    ]
    for counts in count_cases:
        fused = uses_fusion(max(counts))
        torch.manual_seed(712 + rank)
        x = torch.randn((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        ctx = context(counts, counts)
        epoch = int(state.ready[0])
        for left, right in zip(baseline, candidate):
            expected = left(x, ctx=ctx)[0]
            actual = right(x, ctx=ctx)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Automatic fusion already checks the native kernel at these rows.
            if max(counts) > 0 and not fused:
                native_expected = expected
                if 17 <= max(counts) * world_size <= 32:
                    # The ordinary path excludes FlashInfer at these total M
                    # values. Compare the native kernel with the canonical
                    # padded GEMM arithmetic, without relaxing output tolerance.
                    padded_inputs = x.new_zeros((32, x.shape[1]))
                    padded_inputs[: x.shape[0]].copy_(x)
                    padded_counts = (32,) * world_size
                    native_expected = left(
                        padded_inputs, ctx=context(padded_counts, padded_counts)
                    )[0][: x.shape[0]]
                out = x.new_empty((counts[rank], right.output_size))
                cute_dsl_fused_fp8_dp_column_linear(
                    state, x, right.weight, right.weight_scale_inv, counts, out
                )
                torch.testing.assert_close(
                    out[:, : right.logical_output_size], native_expected, rtol=0, atol=0
                )
        assert int(state.ready[0]) - epoch == (2 if max(counts) > 0 else 0)
        if max(counts) > 0:
            rows = max(counts)
            step = 128
            while rows > 16 * step:
                step *= 2
            bucket = (
                32
                if rows <= 32
                else 64 if rows <= 64 else (rows + step - 1) // step * step
            )
            gathered, _ = baseline[0].comm_backend.projection_all_gather(
                x, rows, False, baseline[0].projection_workspace
            )
            padded = gathered.new_zeros((world_size, bucket, 7168))
            padded[:, :rows].copy_(gathered.view(world_size, rows, 7168))
            values, scales = quantize_fp8(
                padded.view(world_size * bucket, 7168),
                granularity="token_group",
                group_size=128,
                scale_encoding="float32",
            )
            torch.testing.assert_close(
                state.a[: bucket * world_size].view(torch.uint8),
                values.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                state.sa[: bucket * world_size], scales, rtol=0, atol=0
            )

    # Physical graph counts take precedence over smaller logical counts.
    graph_cases = (
        ((32,) * 4, False),
        ((32,) * 4, True),
        ((64, 17, 0, 63), False),
        ((64,) * 4, False),
        ((0, 128), False),
        ((129, 255, 128, 0), False),
        ((257, 384, 33, 0), False),
        ((385, 512, 1, 0), False),
        ((513, 640, 0, 129), False),
        ((641, 768, 129, 0), False),
        ((1025, 1024, 0, 1), False),
        ((2048, 1536, 0, 1), False),
        ((2049, 2304, 0, 129), False),
        ((2305, 2048, 1, 0), False),
        ((4097, 4096, 0, 1), False),
    )
    graph_cases = tuple(
        (counts_for_group(counts), native) for counts, native in graph_cases
    )
    contexts = {
        counts: context([int(n > 0) for n in counts], list(counts))
        for counts, _ in graph_cases
    }
    inputs = {
        counts: torch.randn((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        for counts in contexts
    }
    source = {
        counts: [torch.randn_like(value) for _ in range(3)]
        for counts, value in inputs.items()
    }
    expected = {}
    for counts in contexts:
        source[counts][0].zero_()
        expected[counts] = [
            [layer(value, ctx=contexts[counts])[0] for layer in baseline]
            for value in source[counts]
        ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for counts in contexts:
            for layer in candidate:
                layer(inputs[counts], ctx=contexts[counts])
    stream.synchronize()
    # Exercise device epoch width and successive layers without host joins.
    for flags in (state.ready, state.ack, state.input_flags):
        flags.fill_(2**32 - 4)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    # Interleave automatic 32-row fallback with padded and unequal fused calls.
    # Explicit 32-row fusion remains available for kernel experiments.
    calls = graph_cases * 6
    with torch.cuda.graph(graph, stream=stream):
        outputs = []
        for index, (counts, native) in enumerate(calls):
            repeat = index // len(graph_cases)
            inputs[counts].copy_(source[counts][repeat % 3])
            torch.cuda._sleep(25000 * ((rank + index) % world_size))
            layer = candidate[repeat % 2]
            if native:
                out = inputs[counts].new_empty((counts[rank], layer.output_size))
                cute_dsl_fused_fp8_dp_column_linear(
                    state,
                    inputs[counts],
                    layer.weight,
                    layer.weight_scale_inv,
                    counts,
                    out,
                )
                outputs.append(out[:, : layer.logical_output_size])
            else:
                outputs.append(layer(inputs[counts], ctx=contexts[counts])[0])
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for index, actual in enumerate(outputs):
        counts, _ = calls[index]
        repeat = index // len(graph_cases)
        torch.testing.assert_close(
            actual, expected[counts][repeat % 3][repeat % 2], rtol=0, atol=0
        )
    fused_calls = sum(native or uses_fusion(max(counts)) for counts, native in calls)
    assert int(state.ready[0]) == 2**32 - 4 + 3 * fused_calls
    retained = outputs[0].clone()
    del graph, outputs
    for model in models:
        release_dp_linear_communication(model)
    if world_size == 8:
        state.close()
    assert state.closed
    torch.testing.assert_close(
        retained, expected[(32,) * world_size][0][0], rtol=0, atol=0
    )
    # Reprepare below/at the dispatch cutoff and at a non-power-of-two capacity.
    for capacity, counts, sample, layer in (
        (32, (32,) * world_size, 1, 1),
        (64, (64,) * world_size, 2, 0),
        (640, counts_for_group((513, 640, 0, 129)), 2, 0),
        (2305, counts_for_group((2305, 2048, 1, 0)), 1, 1),
    ):
        prepare_dp_linear_communication(
            candidate, capacity, torch.bfloat16, AutoBackend()
        )
        inputs[counts].copy_(source[counts][sample])
        torch.testing.assert_close(
            candidate[layer](inputs[counts], ctx=contexts[counts])[0],
            expected[counts][sample][layer],
            rtol=0,
            atol=0,
        )
        release_dp_linear_communication(candidate)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_registered_tp_projection(tmp_path, world_size):
    if torch.cuda.device_count() < world_size or torch.cuda.get_device_capability() != (
        10,
        3,
    ):
        pytest.skip(f"requires {world_size} node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=world_size,
        join=True,
    )
