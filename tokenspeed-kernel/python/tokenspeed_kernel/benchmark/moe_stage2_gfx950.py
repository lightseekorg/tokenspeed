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

"""Benchmark the gfx950 exact-BF16 MoE stage-2 production decode shape."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch
from tokenspeed_kernel_amd._triton import triton
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
    moe_align_block_size_device,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage2_kernel import (
    gluon_bf16_moe_reduce_kernel,
    gluon_bf16_moe_stage2_kernel,
)


@dataclass(frozen=True)
class Schedule:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int

    @classmethod
    def parse(cls, value: str) -> Schedule:
        values = tuple(int(part) for part in value.split("x"))
        if len(values) != 4:
            raise argparse.ArgumentTypeError(
                "schedule must be BLOCK_MxBLOCK_NxBLOCK_KxNUM_WARPS"
            )
        return cls(*values)


def _launch(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_valid_ids: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    schedule: Schedule,
) -> None:
    _, hidden_size, intermediate_size = w2.shape
    num_tokens = out.shape[0]
    em = sorted_token_ids.shape[0]
    grid = triton.cdiv(em, schedule.block_m) * triton.cdiv(
        hidden_size, schedule.block_n
    )
    gluon_bf16_moe_stage2_kernel[(grid,)](
        inter_states,
        w2.view(torch.uint8),
        w2_scale,
        partials,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        hidden_size,
        intermediate_size,
        em,
        num_tokens,
        topk,
        inter_states.stride(0),
        inter_states.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        BLOCK_M=schedule.block_m,
        BLOCK_N=schedule.block_n,
        BLOCK_K=schedule.block_k,
        NUM_WARPS=schedule.num_warps,
        WEIGHT_FP8=True,
        num_warps=schedule.num_warps,
    )

    reduce_block_m = 64
    reduce_block_n = 256
    reduce_grid = (
        triton.cdiv(num_tokens, reduce_block_m)
        * triton.cdiv(hidden_size, reduce_block_n),
    )
    gluon_bf16_moe_reduce_kernel[reduce_grid](
        partials,
        out,
        num_tokens,
        hidden_size,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        out.stride(0),
        out.stride(1),
        TOP_K=topk,
        BLOCK_M=reduce_block_m,
        BLOCK_N=reduce_block_n,
        num_warps=4,
    )


def _measure(
    schedule: Schedule,
    inputs: tuple[torch.Tensor, ...],
    *,
    topk: int,
    warmup: int,
    replays: int,
    rounds: int,
    calls_per_graph: int,
) -> tuple[torch.Tensor, list[float]]:
    inter_states, w2 = inputs[:2]
    num_tokens = inter_states.shape[0] // topk
    hidden_size = w2.shape[1]
    partials = torch.empty(
        num_tokens, topk, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    out = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

    _launch(*inputs, partials, out, topk, schedule)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            _launch(*inputs, partials, out, topk, schedule)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / (replays * calls_per_graph))
    graph.replay()
    torch.cuda.synchronize()
    return out.clone(), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule",
        action="append",
        type=Schedule.parse,
        default=[],
        help="BLOCK_MxBLOCK_NxBLOCK_KxNUM_WARPS; may be repeated",
    )
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--replays", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--calls-per-graph", type=int, default=42)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    schedules = args.schedule or [Schedule(16, 64, 128, 4)]

    if torch.version.hip is None or torch.cuda.get_device_capability() != (9, 5):
        raise RuntimeError("this benchmark requires a gfx950 ROCm GPU")

    num_tokens, topk = 64, 8
    num_experts, hidden_size, intermediate_size = 288, 4096, 512
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    inter_states = torch.randn(
        num_tokens * topk,
        intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    weight_bytes = torch.empty(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.uint8,
        device="cuda",
    ).random_(0, 120, generator=generator)
    w2 = weight_bytes.view(torch.float8_e4m3fn)
    w2_scale = torch.full(
        (num_experts, hidden_size // 128, intermediate_size // 128),
        0.001,
        dtype=torch.float32,
        device="cuda",
    )
    route_logits = torch.randn(
        num_tokens, num_experts, device="cuda", generator=generator
    )
    topk_ids = route_logits.topk(topk, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", generator=generator), dim=-1
    )
    sorted_ids, sorted_experts, sorted_weights, num_valid = moe_align_block_size_device(
        topk_ids, topk_weights, num_experts, schedules[0].block_m
    )
    torch.cuda.synchronize()
    inputs = (
        inter_states,
        w2,
        w2_scale,
        sorted_ids,
        sorted_experts,
        sorted_weights,
        num_valid,
    )

    results = []
    reference = None
    for schedule in schedules:
        if schedule.block_m != schedules[0].block_m:
            raise ValueError("all schedules must use the same BLOCK_M")
        output, samples = _measure(
            schedule,
            inputs,
            topk=topk,
            warmup=args.warmup,
            replays=args.replays,
            rounds=args.rounds,
            calls_per_graph=args.calls_per_graph,
        )
        if reference is None:
            reference = output
        else:
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        results.append(
            {
                "schedule": asdict(schedule),
                "median_us": statistics.median(samples),
                "min_us": min(samples),
                "max_us": max(samples),
                "samples_us": samples,
            }
        )

    print(
        json.dumps(
            {
                "shape": {
                    "num_tokens": num_tokens,
                    "topk": topk,
                    "num_experts": num_experts,
                    "hidden_size": hidden_size,
                    "intermediate_size_per_rank": intermediate_size,
                    "route_buffer_rows": sorted_ids.numel(),
                    "valid_route_rows": int(num_valid.item()),
                },
                "graph_replays_per_sample": args.replays,
                "stage2_calls_per_graph": args.calls_per_graph,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
