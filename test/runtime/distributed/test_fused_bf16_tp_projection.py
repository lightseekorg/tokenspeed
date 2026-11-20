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

"""BF16 TP projection checks and a bounded native block-FP8 regression smoke.

To exercise BF16 over a multi-node NVLink fabric, run the BF16 cases with
torchrun and select pytest node IDs matching the torchrun world size.
"""

import gc
import importlib.util
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@torch.no_grad()
def _worker(
    rank, world_size, rendezvous, kind, quantized, input_size, output_size, capacity
):
    # Keep optional CUDA implementation imports out of test discovery.
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        create_fused_tp_column_projection_state,
        create_fused_tp_row_projection_state,
        cute_dsl_fused_bf16_dp_column_linear,
        cute_dsl_fused_bf16_dp_row_linear,
        cute_dsl_fused_fp8_dp_column_linear,
        cute_dsl_fused_fp8_dp_row_linear,
    )

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_ops import (
        ProjectionSpec,
        prepare_projection_collectives,
    )

    helper_spec = importlib.util.spec_from_file_location(
        "bf16_projection_kernel_test_utils",
        Path(__file__).resolve().parents[3] / "tokenspeed-kernel/test/utils.py",
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    launched_with_torchrun = rendezvous == "env://"
    local_rank = int(os.environ["LOCAL_RANK"]) if launched_with_torchrun else rank
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    try:
        if launched_with_torchrun:
            from tokenspeed_kernel.ops.communication.fabric import (
                gather_fabric_map,
                group_has_fabric,
                group_host_span,
            )

            gather_fabric_map()
            ranks = tuple(range(world_size))
            assert group_host_span(ranks) > 1, "requires ranks on multiple hosts"
            assert group_has_fabric(ranks), "requires NVLink fabric on every rank"

        if kind == "column":
            create = create_fused_tp_column_projection_state
            run = (
                cute_dsl_fused_fp8_dp_column_linear
                if quantized
                else cute_dsl_fused_bf16_dp_column_linear
            )
        else:
            create = create_fused_tp_row_projection_state
            run = (
                cute_dsl_fused_fp8_dp_row_linear
                if quantized
                else cute_dsl_fused_bf16_dp_row_linear
            )
        weight_dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
        weight_shape = (
            (output_size // world_size, input_size)
            if kind == "column"
            else (output_size, input_size // world_size)
        )
        torch.manual_seed(917 + rank)

        def values(shape):
            if quantized:
                result = torch.randint(-4, 5, shape, device=device).to(torch.bfloat16)
                # Every quantization group has amax=448 and hence scale=1.
                result[:, ::128] = 448
                return result
            # Bound integer products so every reduction is exact in FP32.
            # Dyadic values still expose accidental block-FP8 rounding.
            limit = 127 if input_size <= 1024 else 31
            return (
                torch.randint(-limit, limit + 1, shape, device=device).to(
                    torch.bfloat16
                )
                / 128
            )

        weight = (
            torch.randint(-2, 3, weight_shape, device=device).to(weight_dtype)
            if quantized
            else values(weight_shape)
        )
        weight_scales = (
            torch.ones(
                (weight.shape[0] // 128, weight.shape[1] // 128),
                device=device,
                dtype=torch.float32,
            )
            if quantized
            else None
        )
        reference_weight = weight.to(torch.bfloat16)
        weight_parts = [torch.empty_like(reference_weight) for _ in range(world_size)]
        dist.all_gather(weight_parts, reference_weight)
        full_weight = torch.cat(weight_parts) if kind == "column" else None

        def reference(x, counts):
            if kind == "column":
                return x @ full_weight.T
            rows = max(counts)
            if rows == 0:
                return x.new_empty((0, output_size))
            padded = x.new_zeros((rows, input_size))
            padded[: x.shape[0]].copy_(x)
            owners = [torch.empty_like(padded) for _ in range(world_size)]
            dist.all_gather(owners, padded)
            gathered = torch.cat(owners)
            width = input_size // world_size
            partial = (
                gathered[:, rank * width : (rank + 1) * width] @ reference_weight.T
            )
            partials = [torch.empty_like(partial) for _ in range(world_size)]
            dist.all_gather(partials, partial)
            # The fused contract rounds each GEMM partial to BF16, then sums
            # in FP32 rank order before the final BF16 rounding.
            total = torch.zeros_like(partial, dtype=torch.float32)
            for part in partials:
                total.add_(part.float())
            return total[rank * rows : rank * rows + x.shape[0]].to(torch.bfloat16)

        if quantized:
            counts_list = [(32,) * world_size, (64,) * world_size, (64, 0)]
        else:
            counts_list = [(rows,) * world_size for rows in (30, 32, 63, 64, 65, 128)]
            counts_list += [
                (rows,) * world_size
                for rows in (256, 384, 512, 1024, 2048)
                if rows <= capacity
            ]
            counts_list += [
                tuple(counts[:world_size])
                for counts in ((128, 65, 1, 0), (0, 65, 32, 1), (0, 0, 0, 0))
            ]
        if not quantized and capacity >= 512:
            counts_list += [
                counts[:world_size]
                for counts in (
                    (512, 0, 257, 1),
                    (0, 512, 65, 1),
                    (256, 256, 256, 256),
                    (128, 128, 128, 128),
                    (512, 1, 0, 128),
                )
            ]
        if not quantized and kind == "column" and capacity >= 512:
            # Alternate the cooperative 512-row bucket with an ordinary
            # column bucket while moving the full input to a different owner.
            counts_list += [
                (1, 0, 512, 257),
                (384, 1, 0, 193),
                (0, 1, 257, 512),
                (385, 257, 511, 0),
                (512, 256, 257, 0),
            ]
        if not quantized and kind == "row" and capacity >= 128:
            # Exercise descending strides without a host synchronization between
            # projections, including empty owners entering and leaving a bucket.
            counts_list += [
                (128,) * world_size,
                (32, 0, 30, 1)[:world_size],
                (0, 128, 65, 1)[:world_size],
                (64,) * world_size,
                (128,) * world_size,
                (0, 32, 1, 30)[:world_size],
            ]
        backend = AutoBackend()
        spec = ProjectionSpec(
            group=tuple(range(world_size)),
            kind=kind,
            input_size=input_size,
            output_size=output_size,
            max_tokens=capacity,
            dtype=torch.bfloat16,
            weight_dtype=weight_dtype,
            device=device,
        )
        workspace = prepare_projection_collectives(spec, backend)
        state = workspace.fused_column if kind == "column" else workspace.fused_row
        assert state is not None, (
            kind,
            world_size,
            weight_dtype,
            input_size,
            output_size,
        )
        assert state.quantized == quantized
        dispatch_rows = 64 if quantized or kind == "row" else 32

        def project(x, counts, out):
            if kind == "column":
                fused = backend.try_fused_dp_column_linear(
                    x, weight, weight_scales, counts, out, workspace
                )
            else:
                fused = backend.try_fused_dp_row_linear(
                    x, None, weight, weight_scales, counts, out, workspace
                )
            assert fused == (max(counts) >= dispatch_rows), (kind, counts)
            if not fused:
                # Keep native coverage for batches below the runtime cutoff.
                run(state, x, weight, weight_scales, counts, out)

        # Give every rank invalid operands so the test can catch the error
        # without leaving a peer inside a collective.
        counts = (dispatch_rows,) * world_size
        invalid_input = torch.zeros(
            (input_size, dispatch_rows), dtype=torch.bfloat16, device=device
        ).T
        out = invalid_input.new_empty((dispatch_rows, output_size))
        with pytest.raises(ValueError, match="inputs must be contiguous"):
            project(invalid_input, counts, out)

        if quantized and kind == "row":
            fp8_input = invalid_input.contiguous().to(torch.float8_e4m3fn)
            scales = torch.ones(
                (dispatch_rows, input_size // 64), dtype=torch.float32, device=device
            )[:, ::2]
            with pytest.raises(ValueError, match="input_scales must be contiguous"):
                backend.try_fused_dp_row_linear(
                    fp8_input, scales, weight, weight_scales, counts, out, workspace
                )

        inputs = [values((counts[rank], input_size)) for counts in counts_list]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        skew_ranks = not quantized and (kind == "row" or capacity >= 512)

        retained = []
        # Check eager outputs on the stream used for graph warmup.
        with torch.cuda.stream(stream), helper.assert_no_cute_compile():
            for case_index, (x, counts) in enumerate(zip(inputs, counts_list)):
                expected = reference(x, counts)
                if skew_ranks:
                    torch.cuda._sleep(25000 * ((rank + case_index) % world_size))
                out = torch.full_like(expected, float("nan"))
                project(x, counts, out)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                retained.append((out, expected))

        stream.synchronize()
        graph_outputs = [
            x.new_full((x.shape[0], output_size), float("nan")) for x in inputs
        ]
        graph = torch.cuda.CUDAGraph()
        # Match the runtime capture guard: CUDA object destructors must not
        # run while the stream is being captured.
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                helper.assert_no_cute_compile(),
                torch.cuda.graph(graph, stream=stream),
            ):
                for case_index, (x, counts, out) in enumerate(
                    zip(inputs, counts_list, graph_outputs)
                ):
                    if skew_ranks:
                        torch.cuda._sleep(25000 * ((rank + case_index) % world_size))
                    project(x, counts, out)
        finally:
            if gc_was_enabled:
                gc.enable()
        for replay in range(3):
            for x in inputs:
                if replay == 0:
                    x.zero_()
                else:
                    x.copy_(values(x.shape))
            expected = [reference(x, c) for x, c in zip(inputs, counts_list)]
            torch.cuda._sleep(25000 * ((rank + replay) % world_size))
            graph.replay()
            for actual, ref in zip(graph_outputs, expected):
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        graph.reset()
        workspace.close()

        if kind == "row" and not quantized:
            small_workspace = prepare_projection_collectives(
                replace(spec, max_tokens=dispatch_rows - 1), backend
            )
            prepared_fusion = small_workspace.fused_row is not None
            small_workspace.close()
            assert not prepared_fusion

        # Reprepare a smaller capacity after calls that changed row buckets.
        state = create(
            dist.group.WORLD,
            device,
            32,
            input_size,
            output_size,
            weight_dtype=weight_dtype,
        )
        counts = (32, 0, 30, 1)[:world_size]
        x = values((counts[rank], input_size))
        ref = reference(x, counts)
        out = torch.full_like(ref, float("nan"))
        torch.testing.assert_close(
            run(state, x, weight, weight_scales, counts, out), ref, rtol=0, atol=0
        )
        state.close()
        # Caller-owned outputs survive later calls and workspace replacement.
        for actual, ref in retained + list(zip(graph_outputs, expected)):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _launch_bf16_projection(
    tmp_path, kind, world_size, input_size, output_size, capacity, supported_arches
):
    args = (kind, False, input_size, output_size, capacity)
    if "TORCHELASTIC_RUN_ID" in os.environ:
        assert (
            int(os.environ["WORLD_SIZE"]) == world_size
        ), "select BF16 cases matching the torchrun world size"
        _worker(int(os.environ["RANK"]), world_size, "env://", *args)
        return
    if (
        torch.cuda.device_count() < world_size
        or torch.cuda.get_device_capability() not in supported_arches
    ):
        pytest.skip(f"requires {world_size} node-local GPUs in {supported_arches}")
    mp.spawn(
        _worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}", *args),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize(
    ("kind", "world_size"),
    (("column", 2), ("column", 4), ("row", 2)),
)
def test_native_bf16_tp_projection(tmp_path, kind, world_size):
    _launch_bf16_projection(
        tmp_path, kind, world_size, 1024, 1024, 128, ((10, 0), (10, 3))
    )


@pytest.mark.parametrize("kind", ("column", "row"))
def test_native_fp8_tp_projection_smoke(tmp_path, kind):
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires 2 node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(2, f"file://{tmp_path / 'rendezvous'}", kind, True, 1024, 1024, 64),
        nprocs=2,
        join=True,
    )


def test_native_bf16_dsa_column_projection(tmp_path):
    _launch_bf16_projection(tmp_path, "column", 4, 7168, 14592, 2048, ((10, 3),))
