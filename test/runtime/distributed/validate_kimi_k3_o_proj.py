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

"""Standalone multi-GPU projection correctness runner; never collected by pytest.

Run from the repository root with torchrun --module
test.runtime.distributed.validate_kimi_k3_o_proj.
"""

import argparse
import json
import os
from test.runtime.distributed.kimi_k3_o_proj_helpers import (
    ENV_NAME,
    dep_mapping,
    load_projection,
    make_linears,
)
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.cuda import (
    tokenspeed_a2a_lamport,
)

from tokenspeed.runtime.distributed.comm_ops import reduce_scatter
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.dp_linear_communication import (
    DPRowParallelCommunication,
    initialize_projection_group,
    validate_projection_settings,
)
from tokenspeed.runtime.layers.linear import RowParallelLinear
from tokenspeed.runtime.utils.env import envs


def validate_backend_transitions(linear, baseline, k, world):
    """Replay mixed backend boundaries with delayed peers and retained outputs."""
    rank = dist.get_rank()
    patterns = [
        [128] * world,
        [129] * world,
        [512] * world,
        [513] * world,
        [8192] * world,
        [8193] * world,
        [0] * world,
        [512] * world,
        [513 if r % 4 == 0 else 0 for r in range(world)],
        [512] * world,
    ]
    inputs = [
        torch.randn(counts[rank], k, device="cuda", dtype=torch.bfloat16)
        for counts in patterns
    ]
    # Different progress within each subgroup stresses cross-backend reuse.
    delay = torch.ones(262144, device="cuda")
    expected = [
        baseline(x)[0] if x.shape[0] else x.new_empty((0, linear.output_size))
        for x in inputs
    ]
    for x, counts in zip(inputs, patterns):
        linear(x, counts=counts)[0]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = []
        for x, counts in zip(inputs, patterns):
            if rank % 4 == 0:
                for _ in range(4):
                    delay.mul_(1.0001)
            outputs.append(linear(x, counts=counts)[0])
    for _ in range(5):
        graph.replay()
        for actual, reference in zip(outputs, expected):
            if actual.numel():
                relative = (
                    actual.float() - reference.float()
                ).norm() / reference.float().norm()
                assert relative < 0.015
    torch.cuda.synchronize()
    del outputs, graph


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--large-tokens", action="store_true")
    parser.add_argument("--layer", type=int, choices=(0, 3))
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_dtype(torch.bfloat16)
    mapping = dep_mapping(rank, world)
    pg_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=600,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    os.environ[ENV_NAME] = "4"
    # Reject rank disagreement collectively before creating subgroups. This is
    # a deadlock-safety check with real collectives, not a mocked settings table.
    for local_value in ("1", "invalid"):
        rejected = False
        try:
            validate_projection_settings(
                mapping, local_value if rank == 0 else "4", "nccl", "nccl"
            )
        except ValueError as exc:
            rejected = "differ across ranks" in str(exc)
        rejected_on_all_ranks = torch.tensor(int(rejected), device="cuda")
        dist.all_reduce(rejected_on_all_ranks, op=dist.ReduceOp.MIN)
        assert rejected_on_all_ranks.item() == 1
    parallel = validate_projection_settings(
        mapping,
        envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.get(),
        envs.TOKENSPEED_PROJ_A2A_BACKEND.get(),
        envs.TOKENSPEED_O_PROJ_RS_BACKEND.get(),
    )
    initialize_projection_group(parallel)
    shapes = [(128, 256, None), (7168, 12288, None)]
    if args.model:
        shapes = [
            (0, 0, layer)
            for layer in ((args.layer,) if args.layer is not None else (0, 3))
        ]
    for n, k, layer in shapes:
        if layer is None:
            generator = torch.Generator().manual_seed(42)
            weight = (
                torch.randn(n, k, generator=generator, dtype=torch.float32).to(
                    torch.bfloat16
                )
                / k**0.5
            )
            scale, quant = None, None
        else:
            weight, scale, quant = load_projection(args.model, layer)
            n, k = weight.shape
        baseline, linear = make_linears(mapping, weight, scale, quant)
        # A quant method without apply_into must still populate the caller's
        # destination. Keep the real GPU GEMM, hiding only its optional out API.
        inputs = torch.randn(2, k // parallel.tp_size, device="cuda")
        expected, expected_bias = RowParallelLinear.forward(linear, inputs, scale=None)
        destination = torch.full_like(expected, float("nan"))
        with patch.object(
            linear, "quant_method", SimpleNamespace(apply=linear.quant_method.apply)
        ):
            actual, output_bias = linear.forward_into(inputs, None, destination)
        assert actual is destination and output_bias is expected_bias
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reference_weight = weight.float()
        if scale is not None:
            reference_weight *= scale.repeat_interleave(128, 0).repeat_interleave(
                128, 1
            )
        reference_weight = reference_weight.cuda()
        linear.communication = DPRowParallelCommunication(
            8193 if args.large_tokens else 512, k, torch.bfloat16, torch.device("cuda")
        )
        linear.communication.initialize_a2a(
            linear.parallel,
            [k],
            backend=envs.TOKENSPEED_PROJ_A2A_BACKEND.get(),
        )
        linear.communication.initialize_reduce_scatter(
            linear.parallel,
            [n],
            backend=envs.TOKENSPEED_O_PROJ_RS_BACKEND.get(),
        )
        patterns = [
            [1] * world,
            [8] * world,
            [2] * world,
            [4] * world,
            [16] * world,
            [16 if r < 4 else 0 for r in range(world)],
            [17] * world,
            [32] * world,
            [64] * world,
            [65] * world,
            [128] * world,
            [129] * world,
            [257] * world,
            [7 if r == 0 else 0 for r in range(world)],
            [r % 4 for r in range(world)],
            [0] * world,
        ]
        if args.large_tokens:
            patterns += [[rows] * world for rows in (512, 513, 8192, 8193, 512, 513)]
        for counts in patterns:
            generator = torch.Generator(device="cuda").manual_seed(100 + rank)
            x = torch.randn(
                counts[rank],
                k,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            expected = baseline(x)[0] if counts[rank] else x.new_empty((0, n))
            actual = linear(x, counts=counts)[0]
            # Compare against the same TP4 GEMM/reduction with its original
            # separate quantizer, isolating fusion from TP1 rounding differences.
            with patch(
                "tokenspeed.runtime.layers.linear.fp8_linear_accepts_prepacked_input",
                return_value=False,
            ):
                unfused = linear(x, counts=counts)[0]
            torch.testing.assert_close(actual, unfused, rtol=0, atol=0)
            max_rows = max(counts[r] for r in linear.parallel.tp_group)
            lamport_a2a = linear.communication.lamport_a2a_state(k, max_rows)
            check_reduction = lamport_a2a is not None and all(
                counts[r] == max_rows for r in linear.parallel.tp_group
            )
            using_peer = (
                linear.communication.peer_state(
                    n, max(counts[r] for r in linear.parallel.tp_group)
                )
                is not None
            )
            using_lamport = (
                linear.communication.lamport_state(
                    n, max(counts[r] for r in linear.parallel.tp_group)
                )
                is not None
            )
            custom_reduction = using_peer or using_lamport
            reduction_errors = torch.zeros(2, device="cuda", dtype=torch.float32)
            if custom_reduction and check_reduction:
                # Same quantized GEMM partials: isolate reduction rounding from
                # weight/activation quantization and TP1 accumulation changes.
                redistributed = tokenspeed_a2a_lamport(
                    lamport_a2a, x.contiguous(), inverse=False, out=None
                )
                partial, _ = RowParallelLinear.forward(
                    linear, redistributed, scale=None
                )
                reduction_reference = partial.float()
                dist.all_reduce(
                    reduction_reference,
                    group=pg_manager.get_process_group(
                        "nccl", linear.parallel.tp_group
                    ),
                )
                rows = counts[rank]
                offset = linear.parallel.tp_rank * rows
                reduction_reference = reduction_reference[offset : offset + rows]
                nccl_output = reduce_scatter(
                    partial, linear.parallel.tp_group, backend=None
                )
                norm = reduction_reference.norm().clamp_min(1e-8)
                reduction_errors[0] = (
                    actual.float() - reduction_reference
                ).norm() / norm
                reduction_errors[1] = (
                    nccl_output.float() - reduction_reference
                ).norm() / norm
                assert reduction_errors[0] <= reduction_errors[1] + 0.001
            dist.all_reduce(reduction_errors, op=dist.ReduceOp.MAX)
            # Outputs must survive reuse of symmetric scratch by a later layer.
            preserved = actual.clone()
            linear(x * 0.5, counts=counts)[0]
            torch.testing.assert_close(actual, preserved, rtol=0, atol=0)
            if x.shape[0]:
                reference = x.float() @ reference_weight.T
                delta = actual.float() - expected.float()
                absolute_max = delta.abs().max()
                relative_l2 = delta.norm() / expected.float().norm().clamp_min(1e-8)
                max_scaled = delta.abs().max() / expected.float().abs().max().clamp_min(
                    1e-8
                )
                assert relative_l2 < 0.015, (rank, counts, relative_l2.item())
                assert max_scaled < 0.03, (rank, counts, max_scaled.item())
                norm = reference.norm().clamp_min(1e-8)
                baseline_error = (expected.float() - reference).norm() / norm
                feature_error = (actual.float() - reference).norm() / norm
                assert feature_error < baseline_error + 0.01
            else:
                relative_l2 = torch.tensor(0.0, device="cuda", dtype=torch.float32)
                absolute_max = torch.zeros_like(relative_l2)
                baseline_error = torch.zeros_like(relative_l2)
                feature_error = torch.zeros_like(relative_l2)
                assert actual.shape == (0, n)
            dist.all_reduce(relative_l2, op=dist.ReduceOp.MAX)
            dist.all_reduce(absolute_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(baseline_error, op=dist.ReduceOp.MAX)
            dist.all_reduce(feature_error, op=dist.ReduceOp.MAX)
            # Fixed collective shapes under graphs; change input contents on
            # replay to detect stale packing and graph-owned scratch aliases.
            if max(counts):
                for _ in range(3):
                    linear(x, counts=counts)[0]
                torch.cuda.synchronize()
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture):
                    graphed = linear(x, counts=counts)[0]
                for multiplier in (0.5, -1.0, 1e-10, 0.0):
                    x.mul_(multiplier)
                    capture.replay()
                    eager = linear(x, counts=counts)[0]
                    torch.testing.assert_close(graphed, eager, rtol=0, atol=0)
                # Captured collective counts describe physical rows. Valid
                # token counts may change within that envelope between replays.
                for valid in (counts[rank] // 2, counts[rank], 0):
                    x.zero_()
                    x[:valid].normal_(generator=generator)
                    capture.replay()
                    eager = linear(x, counts=counts)[0]
                    torch.testing.assert_close(graphed, eager, rtol=0, atol=0)
                torch.cuda.synchronize()
                # NCCL communicators cannot be destroyed while a live graph
                # still holds collective nodes referencing them.
                del capture, graphed
            record = {
                "shape": [n, k],
                "layer": layer,
                "counts": counts,
                "reduction_l2_vs_fp32": (
                    reduction_errors.tolist()
                    if custom_reduction and check_reduction
                    else None
                ),
                "rs_backend": (
                    "trtllm_lamport"
                    if using_lamport
                    else "triton_peer" if using_peer else "nccl"
                ),
                "a2a_backend": (
                    "tokenspeed_a2a_lamport" if lamport_a2a is not None else "nccl"
                ),
                "relative_l2": relative_l2.item(),
                "absolute_max": absolute_max.item(),
                "baseline_reference_l2": baseline_error.item(),
                "feature_reference_l2": feature_error.item(),
            }
            if rank == 0:
                print(json.dumps(record), flush=True)
        if args.large_tokens:
            validate_backend_transitions(linear, baseline, k, world)
        linear.communication.close()
        del baseline, linear, reference_weight
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        print("PROJECTION_VALIDATION_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
