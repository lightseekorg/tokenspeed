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

"""Representative numerical checks of fused projections and CUDA graph reuse."""

import gc
import os
import sys
from contextlib import contextmanager
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=360, suite="runtime-2gpu")


@contextmanager
def _capture_graph(graph, stream):
    # Destructors of discarded CUDA objects can invalidate an active capture.
    # Match the runtime's capture guard and restore the caller's GC state.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(graph, stream=stream):
            yield
    finally:
        if gc_was_enabled:
            gc.enable()


def _context(rank, counts, physical):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=physical[rank],
        num_extends=0,
        input_num_tokens=physical[rank],
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, physical[rank], 1),
        global_num_tokens=list(counts),
        collective_global_num_tokens=list(physical),
    )


def _initialize_weights(layers, rank):
    for index, linear in enumerate(layers):
        torch.manual_seed(917 + rank + 4 * index)
        linear.weight.copy_(
            torch.randn_like(linear.weight, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
        )
        linear.weight_scale_inv.copy_(
            torch.rand_like(linear.weight_scale_inv) * 0.005 + 0.005
        )


def _reset_epochs(state):
    # Replay across the 32-bit boundary to exercise persistent 64-bit epochs.
    for flags in (state.ready, state.ack, state.input_flags):
        flags.fill_(2**32 - 4)
    torch.cuda.synchronize()
    dist.barrier()


@torch.no_grad()
def _worker(rank, world_size, rendezvous, kind):
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
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
    try:
        check = {"column": _check_column_projection, "row": _check_row_projection}[kind]
        check(rank, world_size, device, parallel, config)
    finally:
        dist.destroy_process_group()


def _check_column_projection(rank, world_size, device, parallel, config):
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        create_fused_tp_column_projection_state,
        cute_dsl_fused_fp8_dp_column_linear,
    )
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.layers.linear import (
        DPColumnParallelLinear,
        prepare_dp_linear_communication,
        release_dp_linear_communication,
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
        _initialize_weights(model, rank)
        # Adjacent K blocks need distinct scales when vector loads are prefetched.
        for linear in model:
            blocks = torch.arange(linear.weight_scale_inv.shape[1], device=device)
            linear.weight_scale_inv.mul_(torch.exp2((blocks % 7 - 3).float()))
        prepare_dp_linear_communication(model, 4097, torch.bfloat16, backend)
        models.append(model)
    baseline, candidate = models
    workspace = candidate[0].projection_workspace
    assert workspace is candidate[1].projection_workspace
    state = workspace.fused_column
    if world_size == 8:
        # TP8 is available only through an explicitly prepared native state.
        state = create_fused_tp_column_projection_state(
            dist.group.WORLD, device, 4097, 7168, candidate[0].output_size
        )
    else:
        assert state is not None

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
        96,
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
        ctx = _context(rank, counts, counts)
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
                        padded_inputs, ctx=_context(rank, padded_counts, padded_counts)
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
        # Verify padded split-accumulator tiles with unequal and empty owners.
        ((96, 65, 1, 0), False),
        ((0, 128), False),
        ((129, 255, 128, 0), False),
        # Exercise output-copy tails for tiny and empty owners in a larger bucket.
        ((255, 32, 1, 0), False),
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
        counts: _context(rank, [int(n > 0) for n in counts], counts)
        for counts, _ in graph_cases
    }
    inputs = {
        counts: torch.randn((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        for counts in contexts
    }
    source = {}
    # Vary adjacent K-block scales to catch reuse of a stale pipeline stage.
    block_scales = torch.exp2(
        (torch.arange(7168 // 128, device=device) % 7 - 3).float()
    ).to(torch.bfloat16)
    for counts, value in inputs.items():
        sample = torch.randn_like(value)
        sample.view(value.shape[0], 7168 // 128, 128).mul_(block_scales.view(1, -1, 1))
        # Change FP8 values and scales between epochs so operand prefetch cannot
        # reuse stale data, with opposite scale shifts across owners.
        source[counts] = [
            torch.zeros_like(value),
            sample,
            sample.flip(1) * (16.0 if rank % 2 else 0.0625),
        ]
    expected = {}
    for counts in contexts:
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
    _reset_epochs(state)
    graph = torch.cuda.CUDAGraph()
    # Interleave automatic 32-row fallback with padded and unequal fused calls.
    # Explicit 32-row fusion remains available for kernel experiments.
    calls = graph_cases * 6
    with _capture_graph(graph, stream):
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


def _row_projection_graph_counts(world_size):
    """Exercise dispatch, owner changes, and reuse across representative buckets."""
    return [
        tuple(counts[:world_size])
        for counts in (
            # Fallback, fusion, and a padded input with unequal owners.
            (32, 32, 32, 32),
            (64, 64, 64, 64),
            (128, 65, 1, 0),
            # Keep one bucket while the empty owner changes, then change stride.
            (192, 192, 192, 192),
            (0, 161, 192, 1),
            (161, 192, 0, 1),
            (224, 224, 224, 224),
            (193, 0, 1, 208),
            (0, 224, 193, 1),
            (256, 256, 256, 256),
            # Clear receive tails before input publication, including empty owners.
            (384, 257, 1, 0),
            (512, 512, 512, 512),
            (257, 0, 512, 1),
            (0, 512, 257, 1),
            (256, 256, 256, 256),
            # Cross-owner tiles and receive-tail clearing above the small prefix.
            (576, 576, 576, 576),
            (513, 0, 1, 545),
            (0, 576, 513, 1),
            (960, 897, 0, 1),
            # Clustered TP4 output and dynamic TP2 scheduling, including padding.
            (2048, 2048, 2048, 2048),
            (4608, 4608, 4608, 4608),
            (4097, 0, 128, 1),
            (0, 4097, 128, 1),
            # Non-clustered large output, skipped padding, then a smaller layout.
            (5120, 5120, 5120, 5120),
            (4865, 0, 128, 1),
            (64, 64, 64, 64),
        )
    ]


def _check_row_projection(rank, world_size, device, parallel, config):
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.layers.linear import (
        DPRowParallelLinear,
        RowParallelLinear,
        prepare_dp_linear_communication,
        release_dp_linear_communication,
    )

    models = []
    for _ in range(2):
        with torch.device(device):
            model = DPRowParallelLinear(
                12288,
                7168,
                parallel=parallel,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix="o_proj",
            )
        _initialize_weights((model,), rank)
        prepare_dp_linear_communication(model, 5120, torch.bfloat16, AutoBackend())
        models.append(model)
    baseline, candidate = models
    # Retain the real AutoBackend transport and GEMM for the unfused reference.
    if baseline.projection_workspace.fused_row is not None:
        baseline.projection_workspace.fused_row.close()
    baseline.projection_workspace.fused_row = None

    from tokenspeed_kernel.ops.communication.cute_dsl import (
        cute_dsl_fused_fp8_dp_row_linear,
    )

    def fused(x, ctx, scale=None):
        out = torch.empty((x.shape[0], 7168), device=device, dtype=torch.bfloat16)
        return cute_dsl_fused_fp8_dp_row_linear(
            candidate.projection_workspace.fused_row,
            x,
            candidate.weight,
            candidate.weight_scale_inv,
            tuple(ctx.collective_global_num_tokens),
            out,
            input_scales=scale,
        )

    def reference(x, ctx, scale=None):
        rows = max(ctx.collective_global_num_tokens)
        if rows == 0:
            return x.new_empty((0, 7168), dtype=torch.bfloat16)
        if rows <= 128 and scale is None:
            return baseline(x, ctx=ctx)[0].clone()
        # Above 128 rows, the ordinary path uses NCCL's BF16 reduction.
        # Verify the fused contract with ordinary GEMM partials and an FP32 sum.
        from tokenspeed.runtime.distributed.comm_backend.projection import (
            projection_all_to_all,
        )

        values, scales = projection_all_to_all(
            x,
            rows,
            False,
            False,
            None,
            baseline.projection_workspace,
            baseline.comm_backend,
            input_scales=scale,
        )
        if scale is not None:
            # Independently gather owner rows, then take this producer's channel
            # shard. Check the real fallback exchanges FP8 bytes and scales.
            gathered = []
            for source in (x.view(torch.uint8), scale):
                padded = source.new_zeros((rows, source.shape[1]))
                padded[: source.shape[0]].copy_(source)
                parts = [torch.empty_like(padded) for _ in range(world_size)]
                dist.all_gather(parts, padded)
                full = torch.cat(parts)
                shard = source.shape[1] // world_size
                gathered.append(full[:, rank * shard : (rank + 1) * shard].contiguous())
            torch.testing.assert_close(
                values.view(torch.uint8), gathered[0], rtol=0, atol=0
            )
            torch.testing.assert_close(scales, gathered[1], rtol=0, atol=0)
            values, scales = gathered[0].view(x.dtype), gathered[1]
        partial = RowParallelLinear.forward(baseline, values, scale=scales, out=None)[0]
        parts = [torch.empty_like(partial) for _ in range(world_size)]
        dist.all_gather(parts, partial)
        total = torch.zeros_like(partial, dtype=torch.float32)
        for part in parts:
            total.add_(part.float())
        return total[rank * rows : rank * rows + x.shape[0]].to(torch.bfloat16)

    # Sample execution behaviors instead of sweeping every tuning bucket.
    patterns = [
        (rows,) * world_size
        for rows in (0, 32, 63, 64, 128, 129, 192, 256, 512, 576, 2048, 5120)
    ]
    patterns += [
        tuple(counts[:world_size])
        for counts in (
            (128, 65, 1, 0),
            (224, 0, 129, 1),
            (0, 960, 897, 1),
            (0, 4097, 129, 1),
        )
    ]
    for counts in patterns:
        x = torch.randn((counts[rank], 12288), device=device, dtype=torch.bfloat16)
        ctx = _context(rank, counts, counts)
        expected = reference(x, ctx)
        actual = candidate(x, ctx=ctx)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if max(counts) > 0:
            torch.testing.assert_close(fused(x, ctx), expected, rtol=0, atol=0)

    # Neighboring quantization groups have different maxima at different lanes.
    # This catches accidental scale sharing between the eight-lane groups.
    counts = (128,) * world_size
    x = torch.zeros((128, 12288), device=device, dtype=torch.bfloat16)
    row_ids = torch.arange(128, device=device)[:, None]
    group_ids = torch.arange(96, device=device)[None, :]
    columns = group_ids * 128 + (row_ids + 17 * group_ids + rank) % 128
    x.scatter_(
        1, columns, torch.exp2((group_ids % 17 - 8).float()).expand(128, -1).to(x.dtype)
    )
    ctx = _context(rank, counts, counts)
    expected = reference(x, ctx)
    torch.testing.assert_close(candidate(x, ctx=ctx)[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(fused(x, ctx), expected, rtol=0, atol=0)
    # Small weight scales produce signed-zero and subnormal BF16 partials.
    saved_weight = candidate.weight.clone()
    saved_scales = candidate.weight_scale_inv.clone()
    signed_weight = torch.ones_like(candidate.weight, dtype=torch.bfloat16)
    signed_weight[::2].neg_()
    scale_exponents = torch.tensor((-145, -143, -140, -126, 0), device=device)
    scale_rows = torch.arange(candidate.weight_scale_inv.shape[0], device=device)
    tiny_scales = torch.exp2(scale_exponents[scale_rows % 5].float())[
        :, None
    ].expand_as(candidate.weight_scale_inv)
    for model in models:
        model.weight.copy_(signed_weight)
        model.weight_scale_inv.copy_(tiny_scales)
    # Small stores, split accumulators, cross-owner vector stores, and both
    # clustered and non-clustered reduction must preserve the zero sentinel.
    for rows in (32, 128, 256, 4608, 5120):
        counts = (rows,) * world_size
        x = torch.zeros((rows, 12288), device=device, dtype=torch.bfloat16)
        x[:, :: 12288 // world_size] = 448
        x[1::4].mul_(-1)
        x[2::4].mul_(8)
        x[3::4].zero_()
        ctx = _context(rank, counts, counts)
        # One active input per weight shard gives the same exact partial on
        # every producer; use the FP32-to-BF16 contract directly. The ordinary
        # library GEMM flushes subnormal scales and is not an oracle here.
        partial = (
            x[:, :1].float()
            * signed_weight[:, 0][None, :].float()
            * tiny_scales[:, 0].repeat_interleave(128)[None, :]
        ).to(torch.bfloat16)
        total = torch.zeros_like(partial, dtype=torch.float32)
        for _ in range(world_size):
            total.add_(partial.float())
        expected = total.to(torch.bfloat16)
        torch.testing.assert_close(fused(x, ctx), expected, rtol=0, atol=0)
    for model in models:
        model.weight.copy_(saved_weight)
        model.weight_scale_inv.copy_(saved_scales)
    # Reuse the prepared kernel with different weight and scale allocations.
    # Prefetch must use each call's descriptors rather than a previous layer's.
    alternate_weight = (-candidate.weight.to(torch.bfloat16)).to(candidate.weight.dtype)
    alternate_scales = candidate.weight_scale_inv * 2
    baseline.weight.copy_(alternate_weight)
    baseline.weight_scale_inv.copy_(alternate_scales)
    # Rebind descriptors for scalar scales, TMA scales, and clustered loads.
    for rows in (128, 576, 4608):
        counts = (rows,) * world_size
        x = torch.randn((rows, 12288), device=device, dtype=torch.bfloat16)
        ctx = _context(rank, counts, counts)
        expected = reference(x, ctx)
        actual = cute_dsl_fused_fp8_dp_row_linear(
            candidate.projection_workspace.fused_row,
            x,
            alternate_weight,
            alternate_scales,
            counts,
            torch.empty_like(expected),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    baseline.weight.copy_(candidate.weight)
    baseline.weight_scale_inv.copy_(candidate.weight_scale_inv)
    counts_list = _row_projection_graph_counts(world_size)
    xs = [
        torch.randn((c[rank], 12288), device=device, dtype=torch.bfloat16)
        for c in counts_list
    ]
    ctxs = [_context(rank, c, c) for c in counts_list]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x, ctx in zip(xs, ctxs):
            candidate(x, ctx=ctx)
            fused(x, ctx)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with _capture_graph(graph, stream):
        outputs = [
            (candidate(x, ctx=ctx)[0], fused(x, ctx)) for x, ctx in zip(xs, ctxs)
        ]

    # Reset after capture so replay itself crosses the 32-bit epoch boundary.
    _reset_epochs(candidate.projection_workspace.fused_row)
    for replay in range(8):
        for index, x in enumerate(xs):
            if (replay + index) % 3 == 0:
                x.zero_()
            else:
                x.copy_(torch.randn_like(x))
        expected = [reference(x, ctx) for x, ctx in zip(xs, ctxs)]
        torch.cuda._sleep(100000 * ((rank + replay) % world_size))
        graph.replay()
        for pair, ref in zip(outputs, expected):
            for actual in pair:
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    graph.reset()

    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_gate

    # Real KDA norm/gate/quant outputs: fallback, small fusion, vector stores,
    # pipelined input exchange with an empty owner, and clustered GEMM.
    fp8_counts = [
        tuple(counts[:world_size])
        for counts in (
            (32, 17, 1, 0),
            (128, 65, 1, 0),
            (256, 241, 1, 0),
            (512, 481, 1, 0),
            (0, 513, 576, 1),
            (0, 161, 192, 1),
            (161, 192, 0, 1),
            (640, 609, 1, 0),
            (1024, 0, 961, 1),
            (4608, 4608, 4608, 4608),
        )
    ]
    norm_weight = torch.randn(128, dtype=torch.bfloat16, device=device)

    def quantized(counts):
        x = torch.randn((counts[rank], 12288), dtype=torch.bfloat16, device=device)
        gate = torch.randn_like(x)
        values, scales = quantize_fp8_with_gate(
            x, gate, norm_weight=norm_weight, eps=1e-6, enable_pdl=False
        )
        # Distinct row and K-group scales expose transposed-index errors.
        row_exponents = torch.arange(x.shape[0], device=device)[:, None] % 5 - 2
        group_exponents = torch.arange(scales.shape[1], device=device)[None, :] % 7 - 3
        scales.mul_(torch.exp2((row_exponents + group_exponents).float()))
        return values, scales

    empty = torch.empty((0, 12288), dtype=torch.float8_e4m3fn, device=device)
    empty_scales = torch.empty((0, 96), dtype=torch.float32, device=device)
    empty_counts = (0,) * world_size
    empty_output = candidate(
        empty, scale=empty_scales, ctx=_context(rank, empty_counts, empty_counts)
    )[0]
    torch.testing.assert_close(
        empty_output, empty.new_empty((0, 7168), dtype=torch.bfloat16), rtol=0, atol=0
    )
    pairs = [quantized(counts) for counts in fp8_counts]
    contexts = [_context(rank, counts, counts) for counts in fp8_counts]
    for (x, scale), ctx in zip(pairs, contexts):
        expected = reference(x, ctx, scale)
        destination = torch.full_like(expected, float("nan"))
        actual = candidate(x, scale=scale, ctx=ctx, out=destination)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)
        torch.testing.assert_close(fused(x, ctx, scale), expected, rtol=0, atol=0)

    # Only one owner has an unaligned value pointer and strided scales. Local
    # layout normalization must preserve the collective selected by its peers.
    if rank == 0:
        x, scale = pairs[0]
        unaligned = torch.empty(x.numel() + 1, dtype=x.dtype, device=device)[
            1:
        ].view_as(x)
        unaligned.copy_(x)
        strided = torch.empty(
            (scale.shape[0], scale.shape[1] * 2), dtype=scale.dtype, device=device
        )[:, ::2]
        strided.copy_(scale)
        pairs[0] = unaligned, strided

    # Reuse FP8 input and scale pointers while their contents and peer timing
    # change. Interleave BF16 to check both compiled variants share epochs safely.
    mixed_indices = (2, 3)
    bf16_inputs = [
        torch.randn((fp8_counts[i][rank], 12288), dtype=torch.bfloat16, device=device)
        for i in mixed_indices
    ]
    torch.cuda.synchronize()
    with _capture_graph(graph, stream):
        fp8_outputs = [
            candidate(x, scale=scale, ctx=ctx)[0]
            for (x, scale), ctx in zip(pairs[:-1], contexts[:-1])
        ]
        mixed_outputs = [
            (
                candidate(bf16_x, ctx=contexts[i])[0],
                candidate(pairs[i][0], scale=pairs[i][1], ctx=contexts[i])[0],
                candidate(bf16_x, ctx=contexts[i])[0],
            )
            for i, bf16_x in zip(mixed_indices, bf16_inputs)
        ]
    for replay in range(3):
        for counts, (x, scale) in zip(fp8_counts[:-1], pairs[:-1]):
            new_x, new_scale = quantized(counts)
            x.copy_(new_x)
            scale.copy_(new_scale)
        expected = [
            reference(x, ctx, scale)
            for (x, scale), ctx in zip(pairs[:-1], contexts[:-1])
        ]
        bf16_expected = [
            reference(bf16_x, contexts[i])
            for i, bf16_x in zip(mixed_indices, bf16_inputs)
        ]
        torch.cuda._sleep(100000 * ((rank + replay) % world_size))
        graph.replay()
        for actual, ref in zip(fp8_outputs, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        for i, (bf16_output, same_bucket_fp8, same_bucket_bf16), ref in zip(
            mixed_indices, mixed_outputs, bf16_expected
        ):
            torch.testing.assert_close(bf16_output, ref, rtol=0, atol=0)
            torch.testing.assert_close(same_bucket_fp8, expected[i], rtol=0, atol=0)
            torch.testing.assert_close(same_bucket_bf16, ref, rtol=0, atol=0)
    graph.reset()
    for model in models:
        release_dp_linear_communication(model)


@pytest.mark.parametrize(
    ("kind", "world_size"),
    (("column", 2), ("column", 4), ("column", 8), ("row", 2), ("row", 4)),
)
def test_registered_tp_projection(tmp_path, kind, world_size):
    if torch.cuda.device_count() < world_size or torch.cuda.get_device_capability() != (
        10,
        3,
    ):
        pytest.skip(f"requires {world_size} node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}", kind),
        nprocs=world_size,
        join=True,
    )
