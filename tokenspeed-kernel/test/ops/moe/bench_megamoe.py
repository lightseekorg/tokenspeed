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

"""Benchmark the Gluon Petit two-stage MegaMoE full path.

This benchmark uses the same serving-oriented scope as the native Petit
benchmark. Each iteration starts from per-rank hidden states and the timed
region includes top-k, live routing metadata, activation quantization,
dispatch, both expert stages, return, and combine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.petit_gluon import load_petit_kernel
from torch.cuda import nvtx

petit_kernel = load_petit_kernel()

MXFP4_SCALE_MIN = 118
MXFP4_SCALE_MAX = 122
GSM8K_C128_TRACE_PREFIX = Path("/tmp/gsm8k_c128_aligned_decode_trace_20260829")
GSM8K_C128_DECODE_STEP_ORDINALS = tuple(range(360, 396))


@dataclass(frozen=True)
class Topology:
    dp_size: int
    tp_size: int
    ep_size: int
    ranks_per_dp: int
    world_size: int
    rank: int
    local_rank: int
    dp_rank: int
    rank_in_dp: int
    local_experts: int
    local_expert_start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Gluon MegaMoE full path."
    )
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--m", type=int, nargs="+", default=[256])
    parser.add_argument(
        "--rank-m",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Run one rank-asymmetric workload with one local token count per "
            "physical rank. This models continuous batches where one DP rank "
            "prefills while its peers decode."
        ),
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=[256])
    parser.add_argument("--global-experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=2880)
    parser.add_argument("--padded-hidden-size", type=int, default=3072)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--model-name", default="GPT-OSS-120B")
    parser.add_argument(
        "--activation-function",
        choices=("swiglu", "silu"),
        default="swiglu",
    )
    parser.add_argument(
        "--bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--graph-iters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--print-summary", action="store_true", default=True)
    parser.add_argument(
        "--stage-breakdown",
        action="store_true",
        help="Also time top-k, input preparation, and MoE/combine separately.",
    )
    parser.add_argument(
        "--gsm8k-skew-routing",
        action="store_true",
        help=(
            "Replay the observed GPT-OSS-120B GSM8K destination-rank skew. "
            "The rank ratio is scaled to each M and top-k; EP=8 is required."
        ),
    )
    parser.add_argument(
        "--routing-trace-prefix",
        type=Path,
        default=None,
        help=(
            "Replay buffered per-rank routing records written as "
            "<prefix>.rank<R>.pt. This preserves the real per-source expert "
            "IDs and variable local token counts."
        ),
    )
    parser.add_argument(
        "--routing-trace-ordinals",
        type=int,
        nargs="+",
        default=None,
        help="Global MoE call ordinals to replay from --routing-trace-prefix.",
    )
    parser.add_argument(
        "--gsm8k-c128-decode-step",
        type=Path,
        nargs="?",
        const=GSM8K_C128_TRACE_PREFIX,
        default=None,
        metavar="TRACE_PREFIX",
        help=(
            "Replay the 36 MoE layers (ordinals 360-395) from one captured "
            "c=128 GSM8K decode step. TRACE_PREFIX defaults to "
            f"{GSM8K_C128_TRACE_PREFIX}. This mode enables --stage-breakdown "
            "and reports the serving-critical sum of moe_compute_combine_ms."
        ),
    )
    parser.add_argument(
        "--petit-profile-samples",
        type=int,
        default=0,
        help="Collect this many per-CTA kernel phase samples after normal Petit timing.",
    )
    parser.add_argument(
        "--profile-ranges",
        action="store_true",
        help="Emit NVTX/ROCTx ranges around benchmark stages. Off by default because it can perturb ROCm timings.",
    )
    return parser.parse_args()


def configure_trace_suite(args: argparse.Namespace) -> None:
    if args.gsm8k_c128_decode_step is None:
        return
    if args.routing_trace_prefix is not None or args.routing_trace_ordinals is not None:
        raise ValueError(
            "--gsm8k-c128-decode-step cannot be combined with the generic "
            "--routing-trace-prefix/--routing-trace-ordinals options"
        )
    if args.m != [256] or args.batch_size != [256]:
        raise ValueError(
            "--gsm8k-c128-decode-step supplies captured token counts; omit "
            "--m and --batch-size"
        )
    args.routing_trace_prefix = args.gsm8k_c128_decode_step
    args.routing_trace_ordinals = list(GSM8K_C128_DECODE_STEP_ORDINALS)
    args.stage_breakdown = True


def validate_args(args: argparse.Namespace) -> None:
    for name in ("dp_size", "tp_size", "ep_size", "global_experts", "topk"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.global_experts < args.topk:
        raise ValueError("--global-experts must be >= --topk")
    if args.hidden_size <= 0 or args.padded_hidden_size <= 0:
        raise ValueError("hidden sizes must be positive")
    if args.padded_hidden_size < args.hidden_size:
        raise ValueError("--padded-hidden-size must be >= --hidden-size")
    if args.padded_hidden_size % 256 != 0:
        raise ValueError("--padded-hidden-size must be divisible by 256")
    if args.intermediate_size <= 0 or args.intermediate_size % 256 != 0:
        raise ValueError("--intermediate-size must be positive and divisible by 256")
    if any(m <= 0 for m in args.m):
        raise ValueError("--m values must be positive")
    if args.rank_m is not None:
        expected_ranks = args.dp_size * args.tp_size
        if len(args.rank_m) != expected_ranks:
            raise ValueError(
                f"--rank-m requires {expected_ranks} values for this topology"
            )
        if any(m <= 0 for m in args.rank_m):
            raise ValueError("--rank-m values must be positive")
    if any(b <= 0 for b in args.batch_size):
        raise ValueError("--batch-size values must be positive")
    if len(args.batch_size) not in (1, len(args.m)):
        raise ValueError(
            "--batch-size must have length 1 or match the number of --m values"
        )
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be >= 0 and --repeat must be > 0")
    if args.graph_iters <= 0:
        raise ValueError("--graph-iters must be positive")
    if args.petit_profile_samples < 0:
        raise ValueError("--petit-profile-samples must be nonnegative")
    if args.gsm8k_skew_routing and args.ep_size != 8:
        raise ValueError("--gsm8k-skew-routing requires EP=8")
    if (args.routing_trace_prefix is None) != (args.routing_trace_ordinals is None):
        raise ValueError(
            "--routing-trace-prefix and --routing-trace-ordinals are required together"
        )
    if args.routing_trace_prefix is not None:
        if args.rank_m is not None:
            raise ValueError(
                "--rank-m and --routing-trace-prefix are mutually exclusive"
            )
        if args.gsm8k_skew_routing:
            raise ValueError(
                "--routing-trace-prefix and --gsm8k-skew-routing are mutually exclusive"
            )
        if (args.ep_size, args.global_experts, args.topk) != (8, 128, 4):
            raise ValueError("routing-trace replay requires EP=8, E=128, and top-k=4")
        if any(ordinal < 0 for ordinal in args.routing_trace_ordinals):
            raise ValueError("routing trace ordinals must be nonnegative")
    if int(args.tp_size) != 1:
        raise ValueError("vLLM-equivalent MegaMoE profiling requires --tp-size=1")
    actual_config = (
        args.ep_size,
        args.global_experts,
        args.topk,
        args.hidden_size,
        args.padded_hidden_size,
        args.intermediate_size,
        args.activation_function,
        args.bias,
    )
    registered_configs = {
        (1, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (2, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (4, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (8, 32, 4, 2880, 3072, 3072, "swiglu", True),
        (8, 128, 4, 2880, 3072, 3072, "swiglu", True),
        (8, 256, 8, 7168, 7168, 2048, "silu", False),
        (8, 384, 6, 7168, 7168, 3072, "silu", False),
    }
    if actual_config not in registered_configs:
        raise ValueError(
            f"unsupported registered two-stage MegaMoE configuration: {actual_config}"
        )
    if any(m > 1024 for m in args.m):
        raise ValueError(
            "the registered MegaMoE workspace supports at most 1024 tokens per rank"
        )
    if args.rank_m is not None and any(m > 1024 for m in args.rank_m):
        raise ValueError(
            "the registered MegaMoE workspace supports at most 1024 tokens per rank"
        )


def init_topology(args: argparse.Namespace) -> Topology:
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    if args.dp_size != args.ep_size:
        raise RuntimeError(
            "vLLM-equivalent MegaMoE profiling requires logical dp_size == ep_size; "
            f"got dp_size={args.dp_size}, ep_size={args.ep_size}"
        )
    expected_world = args.ep_size
    if world_size != expected_world:
        raise RuntimeError(
            f"torchrun world_size={world_size} does not match "
            f"the logical EP size = {expected_world}"
        )
    if args.global_experts % args.ep_size != 0:
        raise RuntimeError(
            f"global_experts={args.global_experts} must be divisible by ep_size={args.ep_size}"
        )

    ranks_per_dp = world_size
    dp_rank = rank
    rank_in_dp = rank
    local_experts = args.global_experts // args.ep_size
    return Topology(
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        ranks_per_dp=ranks_per_dp,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        dp_rank=dp_rank,
        rank_in_dp=rank_in_dp,
        local_experts=local_experts,
        local_expert_start=rank * local_experts,
    )


def make_tp_group(topo: Topology):
    if topo.tp_size != 1:
        raise RuntimeError("this benchmark only supports tp_size=1")
    return dist.group.WORLD


def workload_pairs(args: argparse.Namespace) -> Iterable[tuple[int, int]]:
    if len(args.batch_size) == 1:
        for m in args.m:
            yield int(args.batch_size[0]), int(m)
        return
    for batch, m in zip(args.batch_size, args.m, strict=True):
        yield int(batch), int(m)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(
        (m, padded_hidden_size), dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    hidden[:, hidden_size:] = 0
    reduced_hidden = torch.empty_like(hidden)
    router_logits = torch.empty((m, global_experts), dtype=torch.float32, device=device)
    return hidden.contiguous(), reduced_hidden, router_logits


def make_topk_buffers(
    *, m: int, topk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.empty((m, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((m, topk), dtype=torch.float32, device=device)
    return topk_ids, topk_weights


def apply_gsm8k_skew_routing(
    hidden: torch.Tensor, global_experts: int, topk: int
) -> None:
    """Encode the observed rank skew into the router-logit input columns."""
    observed_rank_counts = (35, 39, 333, 23, 53, 10, 13, 6)
    routes = hidden.size(0) * topk
    observed_routes = sum(observed_rank_counts)
    scaled = [routes * count // observed_routes for count in observed_rank_counts]
    remainder_order = sorted(
        range(len(observed_rank_counts)),
        key=lambda rank: routes * observed_rank_counts[rank] % observed_routes,
        reverse=True,
    )
    for rank in remainder_order[: routes - sum(scaled)]:
        scaled[rank] += 1
    rank_counts = tuple(scaled)
    local_experts = global_experts // len(rank_counts)
    expert_ids: list[int] = []
    for owner_rank, count in enumerate(rank_counts):
        expert_ids.extend(
            owner_rank * local_experts + route % local_experts for route in range(count)
        )
    if len(expert_ids) != routes:
        raise RuntimeError(
            f"GSM8K replay generated {len(expert_ids)} routes, expected {routes}"
        )
    hidden[:, :global_experts] = -10
    rows = torch.arange(hidden.size(0), device=hidden.device).repeat_interleave(topk)
    cols = torch.tensor(expert_ids, dtype=torch.long, device=hidden.device)
    hidden[rows, cols] = 10


def load_routing_trace_record(
    prefix: Path, rank: int, ordinal: int, topk: int
) -> torch.Tensor:
    path = Path(f"{prefix}.rank{rank}.pt")
    if not path.is_file():
        raise RuntimeError(f"routing trace rank file does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("rank", -1)) != rank:
        raise RuntimeError(f"routing trace rank mismatch in {path}")
    if int(payload.get("topk", -1)) != topk:
        raise RuntimeError(f"routing trace top-k mismatch in {path}")
    records = payload.get("records")
    if not isinstance(records, torch.Tensor) or records.ndim != 2:
        raise RuntimeError(f"routing trace records are malformed in {path}")
    matches = torch.nonzero(records[:, 0] == ordinal).flatten()
    if matches.numel() != 1:
        raise RuntimeError(
            f"routing trace {path} contains {matches.numel()} records for ordinal {ordinal}"
        )
    row = records[int(matches.item())]
    num_tokens = int(row[1].item())
    ids = row[2 : 2 + num_tokens * topk].to(torch.int64).view(num_tokens, topk)
    if ids.numel() and int(ids.min()) < 0:
        raise RuntimeError(f"routing trace {path} has missing IDs at ordinal {ordinal}")
    return ids


def apply_routing_trace(hidden: torch.Tensor, expert_ids: torch.Tensor) -> None:
    if hidden.size(0) != expert_ids.size(0):
        raise RuntimeError("routing trace token count does not match hidden input")
    hidden[:, :128] = -10
    rows = torch.arange(hidden.size(0), device=hidden.device).repeat_interleave(
        expert_ids.size(1)
    )
    hidden[rows, expert_ids.flatten().to(hidden.device)] = 10


def import_aiter_topk() -> Callable:
    """Load the same routing kernel used by the upstream Petit benchmark."""
    from aiter.fused_moe import fused_topk

    return fused_topk


def range_call(enabled: bool, name: str, fn: Callable[[], object]) -> object:
    if not enabled:
        return fn()
    nvtx.range_push(name)
    try:
        return fn()
    finally:
        nvtx.range_pop()


def stage_tp_reduce(
    hidden: torch.Tensor,
    reduced_hidden: torch.Tensor,
    tp_group,
    tp_size: int,
) -> None:
    if tp_size <= 1:
        return
    reduced_hidden.copy_(hidden)
    dist.all_reduce(reduced_hidden, op=dist.ReduceOp.SUM, group=tp_group)


def stage_topk(
    reduced_hidden: torch.Tensor,
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    topk: int,
    fused_topk: Callable,
) -> None:
    router_logits.copy_(reduced_hidden[:, : router_logits.size(1)].float())
    fused_topk(
        reduced_hidden,
        router_logits,
        topk,
        True,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


def make_petit_backend(
    *,
    topo: Topology,
    args: argparse.Namespace,
    m: int,
    device: torch.device,
) -> tuple[
    Callable[[], torch.Tensor],
    Callable[[], tuple[torch.Tensor, torch.Tensor]],
    Callable[[], None],
    dict[str, Callable[[], None]],
]:
    w1, w2, fc1_scale, fc2_scale, w1_bias, w2_bias = build_petit_weights(
        local_experts=topo.local_experts,
        hidden_size=args.padded_hidden_size,
        intermediate_size=args.intermediate_size,
        device=device,
    )
    config = petit_kernel.MegaMoeConfig(
        world_size=topo.ep_size,
        num_experts=args.global_experts,
        topk=args.topk,
        model_dim=args.hidden_size,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=petit_kernel.MegaMoeActivationFunction(
            args.activation_function
        ),
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=args.intermediate_size,
        has_bias=args.bias,
    )
    heap = petit_kernel.create_vmm_symmetric_heap(topo.world_size)
    views = config.input_views(heap, m)
    if views.scales is None:
        raise RuntimeError("MXFP4 MegaMoE workspace is missing activation scales")
    out = torch.empty((m, args.padded_hidden_size), dtype=torch.bfloat16, device=device)
    profile = torch.zeros(
        (
            len(petit_kernel._MEGA_MOE_PROFILE_COUNTER_NAMES),
            petit_kernel._MEGA_MOE_PROFILE_CTAS,
        ),
        dtype=torch.int64,
        device=device,
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
            state["reduced_hidden"][:, : args.hidden_size],
            out=input_views(),
        )

    def copy_expert_ids() -> None:
        views.expert_ids.copy_(state["topk_ids"])

    def copy_expert_weights() -> None:
        views.expert_weights.copy_(state["topk_weights"])

    def prepare(
        reduced_hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        state["reduced_hidden"] = reduced_hidden
        state["topk_ids"] = topk_ids
        state["topk_weights"] = topk_weights
        quantize()
        if topo.ep_size == 1:
            copy_expert_ids()
            copy_expert_weights()

    def run(profile_output: torch.Tensor | None = None) -> torch.Tensor:
        return config.run(
            heap,
            w1,
            w2,
            fc1_scale,
            fc2_scale,
            m,
            w13_bias=w1_bias if args.bias else None,
            w2_bias=w2_bias if args.bias else None,
            out=out,
            inputs=input_views() if topo.ep_size > 1 else None,
            profile=profile_output,
        )

    def compute() -> torch.Tensor:
        return run()

    def profile_compute() -> tuple[torch.Tensor, torch.Tensor]:
        profile.zero_()
        return run(profile), profile

    components = {"prepare_quantize": quantize}
    if topo.ep_size == 1:
        components.update(
            {
                "prepare_copy_expert_ids": copy_expert_ids,
                "prepare_copy_expert_weights": copy_expert_weights,
            }
        )
    return compute, profile_compute, prepare, components


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


def summarize_trace_replay(
    rows: list[dict[str, object]], args: argparse.Namespace
) -> dict[str, object] | None:
    if not rows or args.routing_trace_prefix is None:
        return None
    if not all("moe_compute_combine_ms" in row for row in rows):
        return None

    critical = [float(row["moe_compute_combine_ms"]) for row in rows]
    full_path = [float(row["total_ms"]) for row in rows]
    ordered = sorted(critical)

    def percentile(p: float) -> float:
        index = min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)
        return ordered[index]

    return {
        "record_type": "routing_trace_aggregate",
        "backend": "petit_gluon",
        "routing_replay": (
            "gsm8k_c128_decode_step"
            if args.gsm8k_c128_decode_step is not None
            else "gsm8k_trace"
        ),
        "routing_trace_prefix": str(args.routing_trace_prefix),
        "routing_trace_ordinals": [int(row["routing_trace_ordinal"]) for row in rows],
        "layers": len(rows),
        "primary_metric": "serving_critical_moe_compute_combine_sum_ms",
        "serving_critical_moe_compute_combine_sum_ms": sum(critical),
        "serving_critical_moe_compute_combine_mean_ms": sum(critical) / len(critical),
        "serving_critical_moe_compute_combine_p50_ms": percentile(0.50),
        "serving_critical_moe_compute_combine_p90_ms": percentile(0.90),
        "serving_critical_moe_compute_combine_max_ms": max(critical),
        "full_path_sum_ms": sum(full_path),
        "full_path_mean_ms": sum(full_path) / len(full_path),
        "valid_output": int(all(int(row["valid_output"]) for row in rows)),
    }


def write_trace_summary(summary: dict[str, object], args: argparse.Namespace) -> None:
    # A trace aggregate has a different schema from the per-layer CSV rows, so
    # keep it in JSONL and stdout. This also avoids silently padding a CSV with
    # backend-specific stage/profile columns.
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
    if args.print_summary:
        print(
            f"backend={summary['backend']} trace_layers={summary['layers']} "
            f"primary_metric={summary['primary_metric']} "
            "serving_critical_sum_ms="
            f"{summary['serving_critical_moe_compute_combine_sum_ms']:.4f} "
            "serving_critical_mean_ms="
            f"{summary['serving_critical_moe_compute_combine_mean_ms']:.4f} "
            "serving_critical_p90_ms="
            f"{summary['serving_critical_moe_compute_combine_p90_ms']:.4f} "
            f"full_path_sum_ms={summary['full_path_sum_ms']:.4f} "
            f"valid_output={summary['valid_output']}",
            flush=True,
        )


def collect_petit_profile(
    profile_compute: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    samples: int,
    topo: Topology,
) -> dict[str, object]:
    if samples == 0:
        return {}
    names = petit_kernel._MEGA_MOE_PROFILE_COUNTER_NAMES
    ctas = petit_kernel._MEGA_MOE_PROFILE_CTAS
    gathered_samples: list[torch.Tensor] = []
    dist.barrier()
    for _ in range(samples):
        # Direct push begins with a cross-rank launch-admission handshake.
        # Align every eager profiling launch so Python scheduling skew is not
        # misreported as device-side admission cost.
        dist.barrier()
        _, local = profile_compute()
        gathered = torch.empty(
            (topo.world_size * len(names), ctas),
            dtype=torch.int64,
            device=local.device,
        )
        dist.all_gather_into_tensor(gathered, local)
        torch.cuda.synchronize()
        gathered_samples.append(gathered.view(topo.world_size, len(names), ctas).cpu())

    values = torch.stack(gathered_samples).to(torch.float64)
    combine_ctas = 128
    count_names = {"stage1_work_count", "stage2_work_count"}
    combine_names = {
        "combine_epoch_load",
        "combine_grid_sync",
        "combine_xgpu_handoff",
        "combine_reduce",
        "combine_total",
    }
    summary: dict[str, object] = {"petit_profile_samples": samples}
    for counter, name in enumerate(names):
        active_ctas = combine_ctas if name in combine_names else ctas
        counter_values = values[:, :, counter, :active_ctas]
        if name in count_names:
            per_sample = counter_values.sum(dim=2).amax(dim=1)
            suffix = "max_rank_sum"
        else:
            per_sample = counter_values.amax(dim=(1, 2))
            suffix = "max_rank_cta_cycles"
        summary[f"profile_{name}_{suffix}_p50"] = float(
            torch.quantile(per_sample, 0.5).item()
        )
        summary[f"profile_{name}_{suffix}_p90"] = float(
            torch.quantile(per_sample, 0.9).item()
        )

        if topo.world_size > 2:
            rank2_values = counter_values[:, 2, :]
            if name in count_names:
                rank2_per_sample = rank2_values.sum(dim=1)
                rank2_suffix = "sum"
            else:
                rank2_per_sample = rank2_values.amax(dim=1)
                rank2_suffix = "max_cta_cycles"
            summary[f"profile_rank2_{name}_{rank2_suffix}_p50"] = float(
                torch.quantile(rank2_per_sample, 0.5).item()
            )
            summary[f"profile_rank2_{name}_{rank2_suffix}_p90"] = float(
                torch.quantile(rank2_per_sample, 0.9).item()
            )

    # Preserve a coherent phase decomposition: select one median sample and
    # the exact rank/CTA that maximizes each kernel's total, rather than
    # adding independently selected maxima from unrelated CTAs.
    main_total = values[:, :, names.index("stage1_kernel_total"), :]
    main_max = main_total.amax(dim=(1, 2))
    median_sample = int(torch.argsort(main_max)[len(main_max) // 2].item())
    flat_main = int(main_total[median_sample].argmax().item())
    main_rank, main_cta = divmod(flat_main, ctas)
    main_detail = {
        name: int(values[median_sample, main_rank, i, main_cta].item())
        for i, name in enumerate(names)
        if i < 13 or name.startswith(("stage1_", "dispatch_"))
    }
    main_detail["stage1_kernel_total"] = int(
        values[
            median_sample,
            main_rank,
            names.index("stage1_kernel_total"),
            main_cta,
        ].item()
    )
    main_detail["rank"] = main_rank
    main_detail["cta"] = main_cta
    summary["profile_main_critical_cta_p50_json"] = json.dumps(main_detail)

    if topo.world_size > 2:
        rank2_total = main_total[:, 2, :]
        rank2_max = rank2_total.amax(dim=1)
        rank2_median_sample = int(torch.argsort(rank2_max)[len(rank2_max) // 2].item())
        rank2_cta = int(rank2_total[rank2_median_sample].argmax().item())
        rank2_detail = {
            name: int(values[rank2_median_sample, 2, i, rank2_cta].item())
            for i, name in enumerate(names)
            if i < 13 or name.startswith(("stage1_", "dispatch_"))
        }
        rank2_detail["stage1_kernel_total"] = int(
            rank2_total[rank2_median_sample, rank2_cta].item()
        )
        rank2_detail["rank"] = 2
        rank2_detail["cta"] = rank2_cta
        summary["profile_rank2_main_critical_cta_p50_json"] = json.dumps(rank2_detail)

    stage2_total = values[:, :, names.index("stage2_kernel_total"), :]
    stage2_max = stage2_total.amax(dim=(1, 2))
    median_stage2_sample = int(torch.argsort(stage2_max)[len(stage2_max) // 2].item())
    flat_stage2 = int(stage2_total[median_stage2_sample].argmax().item())
    stage2_rank, stage2_cta = divmod(flat_stage2, ctas)
    stage2_detail = {
        name: int(values[median_stage2_sample, stage2_rank, i, stage2_cta].item())
        for i, name in enumerate(names[8:13], start=8)
    }
    stage2_detail["stage2_kernel_total"] = int(
        values[
            median_stage2_sample,
            stage2_rank,
            names.index("stage2_kernel_total"),
            stage2_cta,
        ].item()
    )
    stage2_detail["rank"] = stage2_rank
    stage2_detail["cta"] = stage2_cta
    summary["profile_stage2_critical_cta_p50_json"] = json.dumps(stage2_detail)

    combine_total = values[:, :, names.index("combine_total"), :combine_ctas]
    combine_max = combine_total.amax(dim=(1, 2))
    median_combine_sample = int(
        torch.argsort(combine_max)[len(combine_max) // 2].item()
    )
    flat_combine = int(combine_total[median_combine_sample].argmax().item())
    combine_rank, combine_cta = divmod(flat_combine, combine_ctas)
    combine_detail = {
        name: int(values[median_combine_sample, combine_rank, i, combine_cta].item())
        for i, name in enumerate(names)
        if name in combine_names
    }
    combine_detail["rank"] = combine_rank
    combine_detail["cta"] = combine_cta
    summary["profile_combine_critical_cta_p50_json"] = json.dumps(combine_detail)
    return summary


def run_one(
    args: argparse.Namespace,
    topo: Topology,
    tp_group,
    batch_size: int,
    m: int,
    trace_expert_ids: torch.Tensor | None = None,
    trace_ordinal: int | None = None,
) -> dict[str, object]:
    device = torch.device("cuda", topo.local_rank)
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    torch.manual_seed(args.seed + topo.rank * 17 + m)
    torch.cuda.manual_seed_all(args.seed + topo.rank * 17 + m)

    hidden, reduced_hidden, router_logits = make_inputs(
        m=m,
        hidden_size=args.hidden_size,
        padded_hidden_size=args.padded_hidden_size,
        global_experts=args.global_experts,
        device=device,
    )
    if args.gsm8k_skew_routing:
        apply_gsm8k_skew_routing(hidden, args.global_experts, args.topk)
    if trace_expert_ids is not None:
        apply_routing_trace(hidden, trace_expert_ids)
    if topo.tp_size == 1:
        reduced_hidden = hidden
    topk_ids, topk_weights = make_topk_buffers(m=m, topk=args.topk, device=device)
    fused_topk = import_aiter_topk()

    stage_tp_reduce(hidden, reduced_hidden, tp_group, topo.tp_size)
    stage_topk(
        reduced_hidden,
        router_logits,
        topk_ids,
        topk_weights,
        args.topk,
        fused_topk,
    )
    torch.cuda.synchronize()
    dist.barrier()

    (
        petit_compute,
        petit_profile_compute,
        petit_prepare,
        prepare_components,
    ) = make_petit_backend(topo=topo, args=args, m=m, device=device)

    def backend_prepare() -> None:
        petit_prepare(reduced_hidden, topk_ids, topk_weights)

    def backend_compute() -> torch.Tensor:
        return petit_compute()

    def reduce_fn() -> None:
        range_call(
            args.profile_ranges,
            "petit_gluon:tp_reduce_for_topk",
            lambda: stage_tp_reduce(hidden, reduced_hidden, tp_group, topo.tp_size),
        )

    def topk_fn() -> None:
        range_call(
            args.profile_ranges,
            "petit_gluon:topk",
            lambda: stage_topk(
                reduced_hidden,
                router_logits,
                topk_ids,
                topk_weights,
                args.topk,
                fused_topk,
            ),
        )

    def prepare_fn() -> None:
        range_call(args.profile_ranges, "petit_gluon:prep_dispatch", backend_prepare)

    def compute_fn() -> torch.Tensor:
        return range_call(
            args.profile_ranges,
            "petit_gluon:moe_compute_combine",
            backend_compute,
        )

    def total_fn() -> torch.Tensor:
        return range_call(
            args.profile_ranges,
            "petit_gluon:total",
            lambda: (
                reduce_fn(),
                topk_fn(),
                prepare_fn(),
                compute_fn(),
            )[-1],
        )

    total_local, total_iters, graph = benchmark_with_graph(
        total_fn,
        warmup=args.warmup,
        repeat=args.repeat,
        graph_iters=args.graph_iters,
    )

    stage_times: dict[str, float | str] = {}
    stage_rank_times: dict[str, list[float]] = {}
    if args.stage_breakdown:
        for stage_name, stage_fn in (
            ("topk", topk_fn),
            ("prepare", prepare_fn),
            ("moe_compute_combine", compute_fn),
        ):
            dist.barrier()
            local_ms, _, _ = benchmark_with_graph(
                stage_fn,
                warmup=args.warmup,
                repeat=args.repeat,
                graph_iters=args.graph_iters,
            )
            rank_values = distributed_rank_values(local_ms, device)
            stage_rank_times[stage_name] = rank_values
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
            stage_rank_times[stage_name] = rank_values
            stage_times[f"{stage_name}_ms"] = max(rank_values)
            stage_times[f"{stage_name}_rank_ms"] = json.dumps(rank_values)

    output = total_fn()
    # MegaMoE only defines the logical GPT-OSS hidden columns; the 192 padding
    # columns are transport/compute padding and are intentionally unspecified.
    valid_output = output_is_valid(output[:, : args.hidden_size], device)
    hist = route_histogram(topk_ids, args.global_experts)
    local_m_tensor = torch.tensor([m], dtype=torch.int64, device=device)
    gathered_local_m = [
        torch.empty_like(local_m_tensor) for _ in range(topo.world_size)
    ]
    dist.all_gather(gathered_local_m, local_m_tensor)
    local_m_by_rank = [int(value.item()) for value in gathered_local_m]
    global_m = sum(local_m_by_rank)

    total_max, total_p50, total_p90, total_p99 = distributed_stats(total_local, device)
    total_rank_ms = distributed_rank_values(total_local, device)

    routes = global_m * args.topk
    tps = global_m / (total_max * 1.0e-3) if total_max > 0 else float("inf")
    route_tps = routes / (total_max * 1.0e-3) if total_max > 0 else float("inf")
    row: dict[str, object] = {
        "model": args.model_name,
        "backend": "petit_gluon",
        "batch_size": batch_size,
        "m": m,
        "global_m": global_m,
        "local_m_by_rank": json.dumps(local_m_by_rank),
        "dp_size": topo.dp_size,
        "tp_size": topo.tp_size,
        "ep_size": topo.ep_size,
        "world_size": topo.world_size,
        "comparison_mode": "two_stage_full_path_bf16_boundary",
        "physical_world_size": topo.world_size,
        "logical_dp_size": topo.dp_size,
        "logical_ep_size": topo.ep_size,
        "dispatch_semantics": "fused_megamoe_ep",
        "global_experts": args.global_experts,
        "local_experts": topo.local_experts,
        "topk": args.topk,
        "hidden_size": args.hidden_size,
        "padded_hidden_size": args.padded_hidden_size,
        "intermediate_size": args.intermediate_size,
        "sm_count": sm_count,
        "stages": 2,
        "activation": args.activation_function,
        "has_bias": int(args.bias),
        "expert_compute": "petit_a4w4_two_stage",
        "timed_boundary": "bf16_hidden_to_bf16_combined_output",
        "routing_metadata_timed": 1,
        "graph": int(graph),
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
        "routing_replay": (
            "gsm8k_trace"
            if trace_expert_ids is not None
            else ("gsm8k_rank_skew" if args.gsm8k_skew_routing else "uniform_random")
        ),
        "routing_trace_ordinal": trace_ordinal,
    }
    row.update(stage_times)
    if args.petit_profile_samples:
        row.update(
            collect_petit_profile(
                petit_profile_compute, args.petit_profile_samples, topo
            )
        )
    return row


def main() -> int:
    args = parse_args()
    try:
        configure_trace_suite(args)
        validate_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA/HIP device is not available.", file=sys.stderr)
        return 2

    topo = init_topology(args)
    tp_group = make_tp_group(topo)

    try:
        if args.rank_m is not None:
            workloads = ((max(args.rank_m), args.rank_m[topo.rank], None, None),)
        elif args.routing_trace_prefix is None:
            workloads: Iterable[tuple[int, int, torch.Tensor | None, int | None]] = (
                (batch_size, m, None, None) for batch_size, m in workload_pairs(args)
            )
        else:
            workloads = (
                (
                    ordinal,
                    ids.size(0),
                    ids,
                    ordinal,
                )
                for ordinal in args.routing_trace_ordinals
                for ids in (
                    load_routing_trace_record(
                        args.routing_trace_prefix,
                        topo.rank,
                        ordinal,
                        args.topk,
                    ),
                )
            )
        rows: list[dict[str, object]] = []
        for batch_size, m, trace_expert_ids, trace_ordinal in workloads:
            row = run_one(
                args,
                topo,
                tp_group,
                batch_size,
                m,
                trace_expert_ids,
                trace_ordinal,
            )
            if topo.rank == 0:
                rows.append(row)
                write_results(row, args)
                if args.print_summary:
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
        if topo.rank == 0:
            trace_summary = summarize_trace_replay(rows, args)
            if trace_summary is not None:
                write_trace_summary(trace_summary, args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
