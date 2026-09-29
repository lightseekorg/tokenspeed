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

"""Benchmark GPT OSS 120B and DSV4 MegaMoE profiles with EP8 and graph timing.

Each timed iteration includes top-k routing, activation quantization, dispatch,
both expert projections, return, and combine, from BF16 inputs to BF16 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel

petit_kernel = load_petit_kernel()

MXFP4_SCALE_MIN = 118
MXFP4_SCALE_MAX = 122


@dataclass(frozen=True)
class Profile:
    model_name: str
    global_experts: int
    topk: int
    hidden_size: int
    padded_hidden_size: int
    intermediate_size: int
    activation_function: str
    bias: bool


_PROFILES = {
    "gpt_oss_120b": Profile(
        model_name="GPT-OSS-120B",
        global_experts=128,
        topk=4,
        hidden_size=2880,
        padded_hidden_size=3072,
        intermediate_size=3072,
        activation_function="swiglu",
        bias=True,
    ),
    "dsv4": Profile(
        model_name="DSV4",
        global_experts=384,
        topk=6,
        hidden_size=7168,
        padded_hidden_size=7168,
        intermediate_size=3072,
        activation_function="silu",
        bias=False,
    ),
}


@dataclass(frozen=True)
class Topology:
    world_size: int
    rank: int
    local_rank: int
    local_experts: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", choices=tuple(_PROFILES), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=("graph",), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--graph-iters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--stage-breakdown", action="store_true")
    args = parser.parse_args(argv)
    if any(m <= 0 or m > 1024 for m in args.tokens):
        parser.error("--tokens values must be between 1 and 1024")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    if args.graph_iters <= 0:
        parser.error("--graph-iters must be positive")
    return args


def init_topology(profile: Profile) -> Topology:
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"this benchmark requires EP8; got world_size={world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return Topology(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        local_experts=profile.global_experts // world_size,
    )


def mask_negative_zero_native_fp4(words: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(words)
    for i in range(8):
        nibble = (words >> (i * 4)) & 0xF
        nibble = torch.where(nibble == 0x8, torch.zeros_like(nibble), nibble)
        out |= nibble << (i * 4)
    return out


def build_native_mxfp4_weights(
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_words = torch.randint(
        0,
        1 << 32,
        (local_experts, intermediate_size * 2, hidden_size // 8),
        dtype=torch.int64,
        device=device,
    ).to(torch.int32)
    w2_words = torch.randint(
        0,
        1 << 32,
        (local_experts, hidden_size, intermediate_size // 8),
        dtype=torch.int64,
        device=device,
    ).to(torch.int32)
    w1_q = (
        mask_negative_zero_native_fp4(w1_words)
        .view(torch.uint8)
        .reshape(local_experts, intermediate_size * 2, hidden_size // 2)
    )
    w2_q = (
        mask_negative_zero_native_fp4(w2_words)
        .view(torch.uint8)
        .reshape(local_experts, hidden_size, intermediate_size // 2)
    )
    fc1_scale = torch.randint(
        MXFP4_SCALE_MIN,
        MXFP4_SCALE_MAX + 1,
        (local_experts, intermediate_size * 2, hidden_size // 32),
        dtype=torch.uint8,
        device=device,
    )
    fc2_scale = torch.randint(
        MXFP4_SCALE_MIN,
        MXFP4_SCALE_MAX + 1,
        (local_experts, hidden_size, intermediate_size // 32),
        dtype=torch.uint8,
        device=device,
    )
    return (
        w1_q.contiguous(),
        w2_q.contiguous(),
        fc1_scale.contiguous(),
        fc2_scale.contiguous(),
    )


def build_petit_weights(
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    w1_q, w2_q, fc1_scale, fc2_scale = build_native_mxfp4_weights(
        local_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    w1, fc1 = petit_kernel.repack_moe_kernel_layout(
        w1_q, fc1_scale, layout=petit_kernel.MoeKernelLayout.native_mxfp4
    )
    w2, fc2 = petit_kernel.repack_moe_kernel_layout(
        w2_q, fc2_scale, layout=petit_kernel.MoeKernelLayout.native_mxfp4
    )
    b1 = torch.zeros(
        (local_experts, 2, intermediate_size), dtype=torch.bfloat16, device=device
    )
    b2 = torch.zeros((local_experts, hidden_size), dtype=torch.bfloat16, device=device)
    b1 = petit_kernel.repack_moe_kernel_layout(
        b1,
        layout=petit_kernel.MoeKernelLayout.native_mxfp4,
    )
    b2 = petit_kernel.repack_moe_kernel_layout(
        b2,
        layout=petit_kernel.MoeKernelLayout.native_mxfp4,
    )
    return (
        w1.contiguous(),
        w2.contiguous(),
        fc1.contiguous(),
        fc2.contiguous(),
        b1.contiguous(),
        b2.contiguous(),
    )


def make_inputs(
    *,
    m: int,
    hidden_size: int,
    padded_hidden_size: int,
    global_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.randn(
        (m, padded_hidden_size), dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    hidden[:, hidden_size:] = 0
    router_logits = torch.empty((m, global_experts), dtype=torch.float32, device=device)
    return hidden.contiguous(), router_logits


def make_topk_buffers(
    *, m: int, topk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.empty((m, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((m, topk), dtype=torch.float32, device=device)
    return topk_ids, topk_weights


def import_aiter_topk() -> Callable:
    """Load the same routing kernel used by the upstream Petit benchmark."""
    from aiter.fused_moe import fused_topk

    return fused_topk


def stage_topk(
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
    fused_topk: Callable,
) -> None:
    router_logits.copy_(hidden[:, : router_logits.size(1)].float())
    fused_topk(
        hidden,
        router_logits,
        topk,
        True,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


def make_petit_backend(
    *,
    topo: Topology,
    profile: Profile,
    m: int,
    device: torch.device,
) -> tuple[
    Callable[[], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    dict[str, Callable[[], None]],
]:
    w1, w2, fc1_scale, fc2_scale, w1_bias, w2_bias = build_petit_weights(
        local_experts=topo.local_experts,
        hidden_size=profile.padded_hidden_size,
        intermediate_size=profile.intermediate_size,
        device=device,
    )
    config = petit_kernel.MegaMoeConfig(
        world_size=topo.world_size,
        num_experts=profile.global_experts,
        topk=profile.topk,
        model_dim=profile.hidden_size,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=petit_kernel.MegaMoeActivationFunction(
            profile.activation_function
        ),
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=profile.intermediate_size,
        has_bias=profile.bias,
    )
    heap = petit_kernel.create_vmm_symmetric_heap(topo.world_size)
    views = config.input_views(heap, m)
    if views.scales is None:
        raise RuntimeError("MXFP4 MegaMoE workspace is missing activation scales")
    out = torch.empty(
        (m, profile.padded_hidden_size), dtype=torch.bfloat16, device=device
    )
    state: dict[str, torch.Tensor] = {}

    def input_views() -> petit_kernel.MegaMoeInputViews:
        return petit_kernel.MegaMoeInputViews(
            views.tokens,
            views.scales,
            state["topk_ids"],
            state["topk_weights"],
        )

    def quantize() -> None:
        config.quantize(
            state["hidden"][:, : profile.hidden_size],
            out=input_views(),
        )

    def prepare(
        hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        state["hidden"] = hidden
        state["topk_ids"] = topk_ids
        state["topk_weights"] = topk_weights
        quantize()

    def compute() -> torch.Tensor:
        return config.run(
            heap,
            w1,
            w2,
            fc1_scale,
            fc2_scale,
            m,
            w13_bias=w1_bias if profile.bias else None,
            w2_bias=w2_bias if profile.bias else None,
            out=out,
            inputs=input_views(),
        )

    return compute, prepare, {"prepare_quantize": quantize}


def event_ms(fn: Callable[[], object], repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def distributed_stats(
    value: float, device: torch.device
) -> tuple[float, float, float, float]:
    local = torch.tensor([value], dtype=torch.float32, device=device)
    max_value = local.clone()
    dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = torch.stack(gathered).flatten().sort().values
    count = values.numel()

    def percentile(p: float) -> float:
        idx = min(count - 1, max(0, int(math.ceil(p * count) - 1)))
        return float(values[idx].item())

    return float(max_value.item()), percentile(0.5), percentile(0.9), percentile(0.99)


def distributed_rank_values(value: float, device: torch.device) -> list[float]:
    local = torch.tensor([value], dtype=torch.float32, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [float(item.item()) for item in gathered]


def benchmark_with_graph(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    graph_iters: int,
) -> tuple[float, int, bool]:
    # Eager warmup completes shape-specific Gluon compilation before capture.
    # Synchronize before capture so none of that one-time work can leak into the
    # graph or its timing.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    try:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                fn()
            with torch.cuda.graph(graph):
                for _ in range(graph_iters):
                    fn()
        torch.cuda.current_stream().wait_stream(capture_stream)

        # A graph's first replay can instantiate/upload kernels and register
        # collective resources. Warm up the captured path before measuring.
        warmup_replays = math.ceil(warmup / graph_iters)
        for _ in range(warmup_replays):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        num_replays = math.ceil(repeat / graph_iters)
        total_iters = num_replays * graph_iters
        ms = event_ms(lambda: graph.replay(), num_replays) / graph_iters
        return ms, total_iters, True
    except RuntimeError as exc:
        torch.cuda.synchronize()
        raise RuntimeError(
            "CUDA graph capture/replay failed; this benchmark requires CUDA graph timing"
        ) from exc


def route_histogram(topk_ids: torch.Tensor, global_experts: int) -> list[int]:
    counts = torch.bincount(
        topk_ids.flatten().to(torch.long), minlength=global_experts
    ).to(torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return [int(v) for v in counts.detach().cpu().tolist()]


def output_is_valid(tensor: torch.Tensor, device: torch.device) -> bool:
    ok = torch.tensor(
        [int(torch.isfinite(tensor.float()).all().item())],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


def write_results(row: dict[str, object], args: argparse.Namespace) -> None:
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv.exists()
        with args.csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def run_one(
    args: argparse.Namespace,
    topo: Topology,
    profile: Profile,
    m: int,
) -> dict[str, object]:
    device = torch.device("cuda", topo.local_rank)
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    torch.manual_seed(args.seed + topo.rank * 17 + m)
    torch.cuda.manual_seed_all(args.seed + topo.rank * 17 + m)

    hidden, router_logits = make_inputs(
        m=m,
        hidden_size=profile.hidden_size,
        padded_hidden_size=profile.padded_hidden_size,
        global_experts=profile.global_experts,
        device=device,
    )
    topk_ids, topk_weights = make_topk_buffers(m=m, topk=profile.topk, device=device)
    fused_topk = import_aiter_topk()

    stage_topk(
        hidden,
        router_logits,
        topk_ids,
        topk_weights,
        profile.topk,
        fused_topk,
    )
    torch.cuda.synchronize()
    dist.barrier()

    (
        petit_compute,
        petit_prepare,
        prepare_components,
    ) = make_petit_backend(topo=topo, profile=profile, m=m, device=device)

    def topk_fn() -> None:
        stage_topk(
            hidden, router_logits, topk_ids, topk_weights, profile.topk, fused_topk
        )

    def prepare_fn() -> None:
        petit_prepare(hidden, topk_ids, topk_weights)

    def total_fn() -> torch.Tensor:
        topk_fn()
        prepare_fn()
        return petit_compute()

    total_local, total_iters, graph = benchmark_with_graph(
        total_fn,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
    )

    stage_times: dict[str, float | str] = {}
    if args.stage_breakdown:
        for stage_name, stage_fn in (
            ("topk", topk_fn),
            ("prepare", prepare_fn),
            ("moe_compute_combine", petit_compute),
        ):
            dist.barrier()
            local_ms, _, _ = benchmark_with_graph(
                stage_fn,
                warmup=args.warmup,
                repeat=args.repeat,
                graph_iters=args.graph_iters,
            )
            rank_values = distributed_rank_values(local_ms, device)
            stage_times[f"{stage_name}_ms"] = max(rank_values)
            stage_times[f"{stage_name}_rank_ms"] = json.dumps(rank_values)
        for stage_name, stage_fn in prepare_components.items():
            dist.barrier()
            local_ms, _, _ = benchmark_with_graph(
                stage_fn,
                warmup=args.warmup,
                repeat=args.repeat,
                graph_iters=args.graph_iters,
            )
            rank_values = distributed_rank_values(local_ms, device)
            stage_times[f"{stage_name}_ms"] = max(rank_values)
            stage_times[f"{stage_name}_rank_ms"] = json.dumps(rank_values)

    output = total_fn()
    # MegaMoE only defines the logical GPT-OSS hidden columns; the 192 padding
    # columns are transport/compute padding and are intentionally unspecified.
    valid_output = output_is_valid(output[:, : profile.hidden_size], device)
    hist = route_histogram(topk_ids, profile.global_experts)
    local_m_tensor = torch.tensor([m], dtype=torch.int64, device=device)
    gathered_local_m = [
        torch.empty_like(local_m_tensor) for _ in range(topo.world_size)
    ]
    dist.all_gather(gathered_local_m, local_m_tensor)
    local_m_by_rank = [int(value.item()) for value in gathered_local_m]
    global_m = sum(local_m_by_rank)

    total_max, total_p50, total_p90, total_p99 = distributed_stats(total_local, device)
    total_rank_ms = distributed_rank_values(total_local, device)

    routes = global_m * profile.topk
    tps = global_m / (total_max * 1.0e-3) if total_max > 0 else float("inf")
    route_tps = routes / (total_max * 1.0e-3) if total_max > 0 else float("inf")
    row: dict[str, object] = {
        "model": profile.model_name,
        "backend": "gluon_petit",
        "batch_size": m,
        "m": m,
        "global_m": global_m,
        "local_m_by_rank": json.dumps(local_m_by_rank),
        "dp_size": 8,
        "tp_size": 1,
        "ep_size": 8,
        "world_size": topo.world_size,
        "comparison_mode": "two_stage_full_path_bf16_boundary",
        "physical_world_size": topo.world_size,
        "logical_dp_size": 8,
        "logical_ep_size": 8,
        "dispatch_semantics": "fused_megamoe_ep",
        "global_experts": profile.global_experts,
        "local_experts": topo.local_experts,
        "topk": profile.topk,
        "hidden_size": profile.hidden_size,
        "padded_hidden_size": profile.padded_hidden_size,
        "intermediate_size": profile.intermediate_size,
        "sm_count": sm_count,
        "stages": 2,
        "activation": profile.activation_function,
        "has_bias": int(profile.bias),
        "expert_compute": "petit_a4w4_two_stage",
        "timed_boundary": "bf16_hidden_to_bf16_combined_output",
        "routing_metadata_timed": 1,
        "graph": int(graph),
        "graph_iters": args.graph_iters,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "total_iters": total_iters,
        "valid_output": int(valid_output),
        "tokens_per_s": tps,
        "routes_per_s": route_tps,
        "total_ms": total_max,
        "total_p50_ms": total_p50,
        "total_p90_ms": total_p90,
        "total_p99_ms": total_p99,
        "total_rank_ms": json.dumps(total_rank_ms),
        "route_histogram": json.dumps(hist),
        "routing_replay": "uniform_random",
        "routing_trace_ordinal": None,
    }
    row.update(stage_times)
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA/HIP device is not available.", file=sys.stderr)
        return 2
    profile = _PROFILES[args.profile]
    try:
        topo = init_topology(profile)
        for m in args.tokens:
            row = run_one(args, topo, profile, m)
            if topo.rank == 0:
                write_results(row, args)
                print(
                    f"backend={row['backend']} dp={row['dp_size']} "
                    f"tp={row['tp_size']} ep={row['ep_size']} "
                    f"world={row['world_size']} batch={row['batch_size']} "
                    f"m={row['m']} graph={row['graph']} "
                    f"total_iters={row['total_iters']} "
                    f"total_ms={row['total_ms']:.4f} "
                    f"tokens_per_s={row['tokens_per_s']:.2f} "
                    f"valid_output={row['valid_output']}",
                    flush=True,
                )
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
