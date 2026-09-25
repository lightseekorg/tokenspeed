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

"""Benchmark TokenSpeed's Gluon Petit adapter on its registered EP8 profiles."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import tokenspeed_kernel
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _BenchmarkProfile:
    name: str
    experts: int
    top_k: int
    hidden: int
    logical_intermediate: int
    compute_intermediate: int
    activation: str
    has_bias: bool


_PROFILES = {
    "gpt_oss_120b": _BenchmarkProfile(
        name="gpt_oss_120b",
        experts=128,
        top_k=4,
        hidden=2880,
        logical_intermediate=2880,
        compute_intermediate=3072,
        activation="swiglu",
        has_bias=True,
    ),
    "dsv4": _BenchmarkProfile(
        name="dsv4",
        experts=384,
        top_k=6,
        hidden=7168,
        logical_intermediate=3072,
        compute_intermediate=3072,
        activation="swiglu",
        has_bias=False,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(_PROFILES), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("eager", "graph", "both"),
        default="both",
    )
    return parser.parse_args()


def _initialize_distributed() -> tuple[int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    if dist.get_world_size() != 8:
        raise RuntimeError("Gluon Petit benchmark requires exactly eight ranks")
    return dist.get_rank(), torch.device("cuda", local_rank)


def _parameter(shape: tuple[int, ...], device: torch.device) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.zeros(shape, dtype=torch.uint8, device=device),
        requires_grad=False,
    )


def _build_layer(
    profile: _BenchmarkProfile,
    device: torch.device,
) -> torch.nn.Module:
    local_experts = profile.experts // 8
    layer = torch.nn.Module()
    layer.num_experts = profile.experts
    layer.top_k = profile.top_k
    layer.hidden_size = profile.hidden
    layer.intermediate_size = profile.logical_intermediate
    layer.num_local_experts = local_experts
    layer.ep_size = 8
    layer.tp_size = 1
    layer.activation = profile.activation
    layer.swiglu_beta = 1.0 if profile.has_bias else None
    layer.swiglu_arg = (
        argparse.Namespace(alpha=1.702, limit=7.0)
        if profile.has_bias
        else argparse.Namespace(alpha=None, limit=10.0)
    )
    layer.w13_input_layout = "interleaved" if profile.has_bias else "concatenated"
    layer.register_parameter(
        "w13_weight",
        _parameter(
            (
                local_experts,
                2 * profile.logical_intermediate,
                profile.hidden // 2,
            ),
            device,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        _parameter(
            (
                local_experts,
                2 * profile.logical_intermediate,
                profile.hidden // 32,
            ),
            device,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        _parameter(
            (
                local_experts,
                profile.hidden,
                profile.logical_intermediate // 2,
            ),
            device,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        _parameter(
            (
                local_experts,
                profile.hidden,
                profile.logical_intermediate // 32,
            ),
            device,
        ),
    )
    if profile.has_bias:
        layer.register_parameter(
            "w13_weight_bias",
            torch.nn.Parameter(
                torch.zeros(
                    (local_experts, 2 * profile.logical_intermediate),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_weight_bias",
            torch.nn.Parameter(
                torch.zeros(
                    (local_experts, profile.hidden),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                requires_grad=False,
            ),
        )
    else:
        layer.register_parameter("w13_weight_bias", None)
        layer.register_parameter("w2_weight_bias", None)
    return layer


def _make_plan(profile: _BenchmarkProfile) -> dict:
    return tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation=profile.activation,
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="petit_gluon",
        ep_size=8,
        ispp=profile.logical_intermediate,
        fp8_scale_block_shape=None,
        internal_activation_dtype="mxfp4",
        with_bias=profile.has_bias,
        deepep_group=None,
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution="petit_gluon",
    )


def _routing(
    tokens: int,
    rank: int,
    profile: _BenchmarkProfile,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.arange(tokens, dtype=torch.int32, device=device)[:, None]
    route = torch.arange(profile.top_k, dtype=torch.int32, device=device)[None, :]
    local_experts = profile.experts // 8
    destination = (rank + token + token // 16 + route) % 8
    local = (token * profile.top_k + route) % local_experts
    ids = (destination * local_experts + local).to(torch.int32).contiguous()
    raw_weights = 1.0 + ((token + 2 * route) % 7).to(torch.float32)
    weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).contiguous()
    return ids, weights


def _apply(
    plan: dict,
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        layer,
        x.new_empty((x.shape[0], 0)),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=x.shape[0] * 8,
        max_num_tokens_per_gpu=x.shape[0],
        do_finalize=True,
        low_latency=None,
        overlap_fn=None,
        shared_input=None,
        shared_weight=None,
        shared_out=None,
    )


def _time(
    fn,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(repeat):
        output = fn()
    end.record()
    end.synchronize()
    if output is None or not torch.isfinite(output).all():
        raise RuntimeError("Gluon Petit benchmark produced non-finite output")
    latency = torch.tensor(
        start.elapsed_time(end) / repeat,
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.item())


def _capture(fn) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be nonnegative and --repeat must be positive")
    if any(tokens < 0 or tokens > 1024 for tokens in args.tokens):
        raise ValueError("--tokens values must be in [0, 1024]")

    rank, device = _initialize_distributed()
    profile = _PROFILES[args.profile]
    plan = _make_plan(profile)
    layer = _build_layer(profile, device)
    tokenspeed_kernel.moe_process_weights(plan, layer)

    if rank == 0:
        print("profile,mode,tokens,latency_ms")
    for tokens in args.tokens:
        generator = torch.Generator(device=device).manual_seed(42 + rank)
        x = torch.randn(
            (tokens, profile.hidden),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        topk_ids, topk_weights = _routing(tokens, rank, profile, device)

        def eager_call() -> torch.Tensor:
            return _apply(plan, layer, x, topk_ids, topk_weights)

        if args.mode in {"eager", "both"}:
            latency = _time(
                eager_call,
                warmup=args.warmup,
                repeat=args.repeat,
                device=device,
            )
            if rank == 0:
                print(f"{profile.name},eager,{tokens},{latency:.6f}")

        if args.mode in {"graph", "both"}:
            dist.barrier()
            graph, graph_output = _capture(eager_call)

            def graph_call() -> torch.Tensor:
                graph.replay()
                return graph_output

            latency = _time(
                graph_call,
                warmup=args.warmup,
                repeat=args.repeat,
                device=device,
            )
            if rank == 0:
                print(f"{profile.name},graph,{tokens},{latency:.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
