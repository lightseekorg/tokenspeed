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

"""Benchmark Kimi K3's production-shaped NVFP4 SiTU MoE and latent tail.

This TP8/EP1 GB300 benchmark includes the registered FlashInfer TRT-LLM NVFP4
SiTU routed-expert apply in both measured paths.  It uses K3's real routed
hidden width (3584), model hidden width (7168), top-k (16), RMS epsilon, and
TP8 intermediate shard (3072 / 8 = 384), and all 896 routed experts.
Synthetic NVFP4 values use the same loader layout and TokenSpeed weight
preprocessor as the model.

The baseline measures::

    moe_apply(do_finalize=True)
      -> staged multimem all-reduce -> RMSNorm
      -> sharded routed up-projection + prefix injection
      -> staged multimem all-reduce -> clone

The candidate changes only the first line after expert GEMMs::

    moe_apply(do_finalize=False)
      -> BT/HT fused finalize + staged MNNVL all-reduce + RMSNorm
      -> identical sharded projection/prefix/second-all-reduce/clone

Each CUDA Graph contains multiple sequential synthetic K3 layers.  Eager and
captured paths are checked before timing, inputs are then changed in place,
and both graphs must reproduce fresh eager results.  Latencies are reduced
with MAX across ranks before p50/p90 calculation and reported per layer.

The deliberately bounded synthetic deviation is explicit in the output:
checkpoint values, the router distribution, the shared-expert partial, and
the prefix are synthetic.  Expert count, top-k sparsity, TP layout, packed
weight working set, and the measured registered kernels match production.
Shared-expert compute is represented by a synthetic rank-local partial,
matching the input boundary of the measured tail on both sides.

Example::

    BENCH_PROTOCOL=bt BENCH_TOKENS=256:384:512:768:1024 \
      torchrun --nnodes=2 --nproc-per-node=4 \
      tokenspeed-kernel/test/nvidia/thirdparty/\
bench_k3_mnnvl_cutedsl_situ_moe_tail.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

# Kimi K3 production geometry.
HIDDEN = 3584
LATENT = HIDDEN
MODEL_HIDDEN = 7168
MODEL_INTERMEDIATE = 3072
TP_SIZE = 8
EP_SIZE = 1
INTERMEDIATE_PER_RANK = MODEL_INTERMEDIATE // TP_SIZE
MODEL_NUM_EXPERTS = 896
NUM_EXPERTS = MODEL_NUM_EXPERTS
TOP_K = 16
MODEL_SHARD = MODEL_HIDDEN // TP_SIZE
RMS_EPS = 1e-5
BT_MIN_TOKENS = 256
BT_CAPACITY = 1024
HT_MIN_TOKENS = 1280
HT_CAPACITY = 8192
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_EXPECTED_APPLY_KERNEL = "flashinfer_trtllm_nvfp4_situ_routed_moe_apply"


def _csv_ints(name: str, fallback: str) -> list[int]:
    # Slurm --export consumes commas, so accept a colon spelling as well.
    raw = os.environ.get(name, fallback).replace(":", ",")
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicate values")
    return values


PROTOCOL = os.environ.get("BENCH_PROTOCOL", "bt").strip().lower()
DEFAULT_TOKENS = (
    "256,384,512,768,1024" if PROTOCOL == "bt" else "1280,2048,4096,6144,8192"
)
TOKENS = _csv_ints("BENCH_TOKENS", DEFAULT_TOKENS)
MAX_TOKENS = max(TOKENS)
GRAPH_LAYERS = int(os.environ.get("BENCH_GRAPH_LAYERS", "2"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("BENCH_ITERS", "10"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "5"))
MIN_SPEEDUP = float(
    os.environ.get(
        "BENCH_MIN_SPEEDUP",
        os.environ.get("BENCH_MIN_QUALIFIED_SPEEDUP", "1.0"),
    )
)
GATE_P90 = os.environ.get("BENCH_GATE_P90", "1") == "1"
OUTPUT = os.environ.get("BENCH_OUTPUT")

# The production finalized path and fused deferred path can use different
# BF16 reduction orders.  These bounds match the full-tail qualification and
# still reject a missing rank, route, projection shard, or prefix.
MATCH_ATOL = float(os.environ.get("BENCH_MATCH_ATOL", "0.046875"))
MATCH_RTOL = float(os.environ.get("BENCH_MATCH_RTOL", "0.005"))
MATCH_L2_RTOL = float(os.environ.get("BENCH_MATCH_L2_RTOL", "0.006"))
GRAPH_ATOL = float(os.environ.get("BENCH_GRAPH_ATOL", "0.0"))
GRAPH_RTOL = float(os.environ.get("BENCH_GRAPH_RTOL", "0.0"))
GRAPH_L2_RTOL = float(os.environ.get("BENCH_GRAPH_L2_RTOL", "0.0"))


def _bt_tuning_spec() -> dict[str, int | bool]:
    raw = os.environ.get(
        "BENCH_BT_TUNING",
        os.environ.get("BENCH_BT_MEDIUM_TUNING", "2,256,1,224,448,1"),
    )
    fields = [int(item.strip()) for item in raw.split(",")]
    if len(fields) != 6 or fields[-1] not in (0, 1):
        raise ValueError(
            "BENCH_BT_TUNING must be "
            "elements,threads,prefetch,reduction,rms,pdl(0|1)"
        )
    elements, threads, prefetch, reduction, rms, pdl = fields
    return {
        "max_tokens": BT_CAPACITY,
        "elements_per_thread": elements,
        "threads": threads,
        "prefetch_group": prefetch,
        "reduction_threads": reduction,
        "rms_threads": rms,
        "enable_pdl": bool(pdl),
    }


def _optional_int(value: str) -> int | None:
    return None if value.strip().lower() == "auto" else int(value)


def _ht_tuning_spec() -> dict[str, int | bool | None]:
    raw = os.environ.get("BENCH_HT_TUNING", "auto,448,1,7,2,auto,2,3,0,1")
    fields = [item.strip() for item in raw.split(",")]
    if (
        len(fields) != 10
        or fields[-2] not in ("0", "1")
        or fields[-1]
        not in (
            "0",
            "1",
        )
    ):
        raise ValueError(
            "BENCH_HT_TUNING must be "
            "ctas,consumer,vpt,stages,rw,groups,rmsg,rp,shard,pdl"
        )
    return {
        "max_tokens": HT_CAPACITY,
        "persistent_ctas": _optional_int(fields[0]),
        "consumer_threads": int(fields[1]),
        "vectors_per_thread": int(fields[2]),
        "stages": int(fields[3]),
        "reduction_warps": int(fields[4]),
        "reduction_cta_groups": _optional_int(fields[5]),
        "rms_token_groups": int(fields[6]),
        "rms_pipeline_stages": int(fields[7]),
        "rms_shard_major": bool(int(fields[8])),
        "enable_pdl": bool(int(fields[9])),
    }


BT_TUNING_SPEC = _bt_tuning_spec()
HT_TUNING_SPEC = _ht_tuning_spec()


class _MoEWeights(torch.nn.Module):
    """Minimal K3 NVFP4 weight module consumed by the registered kernel."""

    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            self.register_parameter(
                name,
                torch.nn.Parameter(raw[name], requires_grad=False),
            )
        self.activation_situ_beta = SITU_BETA
        self.activation_situ_linear_beta = SITU_LINEAR_BETA
        self._spec = SimpleNamespace(
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_EXPERTS,
            top_k=TOP_K,
            ep_rank=0,
            ep_size=EP_SIZE,
            tp_size=TP_SIZE,
        )


@dataclass(slots=True)
class LayerInputs:
    """Pointer-stable inputs for one synthetic K3 routed-expert layer."""

    weights: _MoEWeights
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    shared_partial: torch.Tensor
    prefix: torch.Tensor
    gamma: torch.Tensor
    up_weight: torch.Tensor


def _validate_options() -> None:
    if PROTOCOL not in ("bt", "ht"):
        raise ValueError("BENCH_PROTOCOL must be bt or ht")
    if TOKENS != sorted(TOKENS):
        raise ValueError("BENCH_TOKENS must be strictly increasing")
    lower, upper = (
        (BT_MIN_TOKENS, BT_CAPACITY)
        if PROTOCOL == "bt"
        else (HT_MIN_TOKENS, HT_CAPACITY)
    )
    if TOKENS[0] < lower or TOKENS[-1] > upper:
        raise ValueError(f"{PROTOCOL} BENCH_TOKENS must remain in [{lower}, {upper}]")
    if GRAPH_LAYERS < 2:
        raise ValueError("BENCH_GRAPH_LAYERS must be at least two")
    if WARMUP < 0 or ITERS <= 0 or ROUNDS <= 0:
        raise ValueError("BENCH_WARMUP >= 0 and positive ITERS/ROUNDS are required")
    if not math.isfinite(MIN_SPEEDUP) or MIN_SPEEDUP <= 0:
        raise ValueError("BENCH_MIN_SPEEDUP must be finite and positive")
    for name, value in (
        ("MATCH_ATOL", MATCH_ATOL),
        ("MATCH_RTOL", MATCH_RTOL),
        ("MATCH_L2_RTOL", MATCH_L2_RTOL),
        ("GRAPH_ATOL", GRAPH_ATOL),
        ("GRAPH_RTOL", GRAPH_RTOL),
        ("GRAPH_L2_RTOL", GRAPH_L2_RTOL),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if MODEL_INTERMEDIATE % TP_SIZE or MODEL_HIDDEN % TP_SIZE:
        raise AssertionError("K3 model geometry must divide evenly across TP8")
    if NUM_EXPERTS < TOP_K:
        raise AssertionError("synthetic expert count must cover top-k")


def _setup() -> tuple[int, torch.device]:
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as error:
        raise RuntimeError("launch this benchmark with torchrun") from error
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != TP_SIZE:
        raise RuntimeError("this benchmark requires exactly eight TP ranks")
    return dist.get_rank(), torch.device("cuda", local_rank)


def _collective_errors(
    label: str,
    local_error: str | None,
    group: dist.ProcessGroup,
) -> None:
    errors: list[str | None] = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(errors, local_error, group=group)
    if any(error is not None for error in errors):
        details = "; ".join(
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        )
        raise RuntimeError(f"{label} failed closed: {details}")


def _runtime_error() -> str | None:
    try:
        if not torch.cuda.is_available():
            return "CUDA is unavailable"
        capability = torch.cuda.get_device_capability()
        if not (10, 0) <= capability <= (10, 3):
            return f"expected sm_100-family GPU, got capability {capability}"
        if find_spec("flashinfer") is None:
            return "flashinfer is not installed"
        from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
            situ_moe_unavailable_reason,
        )

        return situ_moe_unavailable_reason()
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def _build_plan_collectively(group: dist.ProcessGroup) -> tuple[Any, dict[str, Any]]:
    plan: dict[str, Any] | None = None
    module: Any = None
    local_error: str | None = None
    try:
        import tokenspeed_kernel

        module = tokenspeed_kernel
        plan = tokenspeed_kernel.moe_plan(
            "nvfp4",
            input_dtype=torch.bfloat16,
            activation="situ",
            requires_deferred_finalize=True,
            routing_mode="precomputed_topk",
            a2a_backend=None,
            ep_size=EP_SIZE,
            ispp=INTERMEDIATE_PER_RANK,
            fp8_scale_block_shape=None,
            internal_activation_dtype="input",
            with_bias=False,
            deepep_group=None,
            deepep_mode=None,
            deepep_low_latency_max_num_tokens_per_gpu=None,
            solution="flashinfer_trtllm",
        )
        if plan["apply_kernel_name"] != _EXPECTED_APPLY_KERNEL:
            raise RuntimeError(
                f"selected {plan['apply_kernel_name']!r}, expected "
                f"{_EXPECTED_APPLY_KERNEL!r}"
            )
        if not plan.get("supports_deferred_finalize", False):
            raise RuntimeError(
                "selected SiTU plan does not advertise deferred finalize"
            )
        if not plan.get("supports_precomputed_topk", False):
            raise RuntimeError("selected SiTU plan does not consume precomputed top-k")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _collective_errors("registered NVFP4 SiTU plan selection", local_error, group)
    if module is None or plan is None:
        raise AssertionError("collective plan vote succeeded without a local plan")

    fingerprint = (
        plan["apply_kernel_name"],
        plan.get("solution"),
        bool(plan.get("supports_deferred_finalize")),
        bool(plan.get("supports_precomputed_topk")),
    )
    fingerprints: list[tuple[Any, ...] | None] = [
        None for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather_object(fingerprints, fingerprint, group=group)
    if any(peer != fingerprint for peer in fingerprints):
        raise RuntimeError(f"MoE plan differs across ranks: {fingerprints}")
    return module, plan


def _nvfp4_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one 2-D weight with K3's NVFP4 global/group scale layout."""

    from flashinfer import fp4_quantize

    global_scale = (
        _FP8_E4M3_MAX * _FP4_E2M1_MAX / weight.abs().amax().clamp(min=1e-12)
    ).to(torch.float32)
    packed, scale = fp4_quantize(
        weight,
        global_scale,
        is_sf_swizzled_layout=False,
    )
    rows, columns = weight.shape
    scale = scale.reshape(-1)[: rows * (columns // 16)].view(rows, columns // 16)
    return (
        packed,
        scale.view(torch.float8_e4m3fn),
        (1.0 / global_scale).reshape(()),
    )


def _make_processed_weights(
    kernel_module: Any,
    plan: dict[str, Any],
    rank: int,
    layer_id: int,
    device: torch.device,
) -> _MoEWeights:
    """Create valid synthetic K3 loader-layout weights and preprocess them."""

    generator = torch.Generator(device=device).manual_seed(
        810_000 + rank * 1_003 + layer_id * 97
    )
    w13_weight: list[torch.Tensor] = []
    w13_scale: list[torch.Tensor] = []
    w13_scale_2: list[torch.Tensor] = []
    w2_weight: list[torch.Tensor] = []
    w2_scale: list[torch.Tensor] = []
    w2_scale_2: list[torch.Tensor] = []
    for _ in range(NUM_EXPERTS):
        # Loader layout is concatenated [gate | up].  Quantizing the halves
        # together intentionally gives them the one global scale SiTU requires.
        w13_bf16 = (
            torch.randn(
                (2 * INTERMEDIATE_PER_RANK, HIDDEN),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.5
        ).contiguous()
        w2_bf16 = (
            torch.randn(
                (HIDDEN, INTERMEDIATE_PER_RANK),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.5
        ).contiguous()
        packed, scale, scale_2 = _nvfp4_quantize(w13_bf16)
        w13_weight.append(packed)
        w13_scale.append(scale)
        # Loader layout carries distinct gate/up slots.  The synthetic halves
        # intentionally share one scale because SiTU requires them to match.
        w13_scale_2.append(scale_2.repeat(2))
        packed, scale, scale_2 = _nvfp4_quantize(w2_bf16)
        w2_weight.append(packed)
        w2_scale.append(scale)
        w2_scale_2.append(scale_2)

    raw = {
        "w13_weight": torch.stack(w13_weight),
        "w13_weight_scale": torch.stack(w13_scale),
        "w13_weight_scale_2": torch.stack(w13_scale_2),
        "w2_weight": torch.stack(w2_weight),
        "w2_weight_scale": torch.stack(w2_scale),
        "w2_weight_scale_2": torch.stack(w2_scale_2),
        "w13_input_scale": torch.ones(
            (NUM_EXPERTS, 2), dtype=torch.float32, device=device
        ),
        "w2_input_scale": torch.ones(NUM_EXPERTS, dtype=torch.float32, device=device),
    }
    weights = _MoEWeights(raw)
    kernel_module.moe_process_weights(plan, weights)
    if weights.intermediate_size_per_partition != INTERMEDIATE_PER_RANK:
        raise AssertionError(
            "SiTU preprocessor changed the real TP8 intermediate shard: "
            f"{weights.intermediate_size_per_partition}"
        )
    return weights


def _make_layer_inputs(
    weights: _MoEWeights,
    rank: int,
    m: int,
    layer_id: int,
    device: torch.device,
) -> LayerInputs:
    replicated_generator = torch.Generator(device=device).manual_seed(
        910_000 + m * 17 + layer_id * 101
    )
    sharded_generator = torch.Generator(device=device).manual_seed(
        920_000 + rank * 1_009 + m * 17 + layer_id * 101
    )
    hidden_states = (
        torch.randn(
            (m, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=replicated_generator,
        )
        * 0.2
    ).contiguous()

    # Keep the production [M, top-k] routing shape while spreading work evenly
    # over the real 896-expert domain.  Every token selects 16 distinct experts.
    token_offsets = (
        torch.arange(m, dtype=torch.int32, device=device).unsqueeze(1) * TOP_K
    )
    expert_offsets = torch.arange(TOP_K, dtype=torch.int32, device=device).unsqueeze(0)
    topk_ids = (
        (token_offsets + expert_offsets + layer_id * (TOP_K + 1)) % NUM_EXPERTS
    ).contiguous()
    topk_weights = (
        torch.softmax(
            torch.randn(
                (m, TOP_K),
                dtype=torch.float32,
                device=device,
                generator=replicated_generator,
            ),
            dim=-1,
        )
        .to(torch.bfloat16)
        .contiguous()
    )
    shared_partial = (
        torch.randn(
            (m, MODEL_HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=sharded_generator,
        )
        * 0.02
    ).contiguous()
    prefix = (
        torch.randn(
            (m, MODEL_HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=replicated_generator,
        )
        * 0.02
    ).contiguous()
    gamma = (
        1.0
        + torch.randn(
            (HIDDEN,),
            dtype=torch.bfloat16,
            device=device,
            generator=replicated_generator,
        )
        * 0.02
    ).contiguous()
    up_weight = (
        torch.randn(
            (MODEL_SHARD, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
            generator=sharded_generator,
        )
        * 0.015
    ).contiguous()

    # The finalized FlashInfer lane is caller-owned, matching K3's zero-copy
    # production join.  Deferred calls intentionally ignore this buffer.
    weights._situ_output_buffer = torch.empty(
        (m, HIDDEN), dtype=torch.bfloat16, device=device
    )
    return LayerInputs(
        weights=weights,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_partial=shared_partial,
        prefix=prefix,
        gamma=gamma,
        up_weight=up_weight,
    )


def _rank_max(value: float, device: torch.device) -> float:
    reduced = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return float(reduced.item())


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _error_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    difference = actual.float() - expected.float()
    finite = torch.tensor(
        [int(torch.isfinite(actual).all() and torch.isfinite(expected).all())],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        return {
            "max_abs": math.inf,
            "reference_max_abs": 0.0,
            "relative_l2": math.inf,
        }
    return {
        "max_abs": _rank_max(float(difference.abs().amax()), device),
        "reference_max_abs": _rank_max(float(expected.float().abs().amax()), device),
        "relative_l2": _rank_max(
            float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
            ),
            device,
        ),
    }


def _assert_correct(
    actual: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
    label: str,
    *,
    atol: float,
    rtol: float,
    l2_rtol: float,
) -> dict[str, float]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: got {actual.shape}/{actual.dtype}, expected "
            f"{expected.shape}/{expected.dtype}"
        )
    stats = _error_stats(actual, expected, device)
    allowed_max = atol + rtol * stats["reference_max_abs"]
    if stats["max_abs"] > allowed_max or stats["relative_l2"] > l2_rtol:
        raise AssertionError(
            f"{label}: max_abs={stats['max_abs']:.8g} > {allowed_max:.8g} "
            f"or relative_l2={stats['relative_l2']:.8g} > {l2_rtol:.8g}"
        )
    stats["allowed_max_abs"] = allowed_max
    stats["allowed_relative_l2"] = l2_rtol
    return stats


def _assert_outputs_changed(
    before: Sequence[torch.Tensor],
    after: Sequence[torch.Tensor],
    device: torch.device,
    label: str,
) -> float:
    finite = torch.tensor(
        [
            int(
                all(torch.isfinite(value).all() for value in before)
                and all(torch.isfinite(value).all() for value in after)
            )
        ],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise AssertionError(f"{label}: changed-input outputs are not finite")
    local_change = max(
        float((new.float() - old.float()).abs().amax())
        for old, new in zip(before, after, strict=True)
    )
    change = _rank_max(local_change, device)
    if change <= 0.01:
        raise AssertionError(f"{label}: changed inputs did not change outputs")
    return change


def _capture_layers(
    fn: Callable[[], tuple[torch.Tensor, ...]],
    expected_outputs: int,
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...]]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    torch.cuda.synchronize()
    dist.barrier()
    if len(outputs) != expected_outputs:
        raise AssertionError(
            f"captured graph returned {len(outputs)} outputs, expected "
            f"{expected_outputs}"
        )
    return graph, outputs


def _split_diagnostic_outputs(
    outputs: Sequence[torch.Tensor],
) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """Split per-layer full-tail outputs from immediate routed-norm snapshots."""

    if len(outputs) != 2 * GRAPH_LAYERS:
        raise AssertionError(
            "diagnostic execution must return full-tail and routed-norm outputs"
        )
    return outputs[:GRAPH_LAYERS], outputs[GRAPH_LAYERS:]


def _change_graph_inputs(layers: Sequence[LayerInputs]) -> None:
    """Change every caller-owned dynamic graph input in place."""

    for layer_id, layer in enumerate(layers):
        layer.hidden_states.mul_(0.9375).add_(0.0009765625 * (layer_id + 1))
        layer.topk_weights.mul_(0.875)
        layer.topk_weights[:, 0].add_(0.125)
        layer.topk_ids.copy_(
            torch.roll(layer.topk_ids, shifts=1 + layer_id % 2, dims=-1)
        )
        layer.shared_partial.mul_(0.875).add_(0.00048828125 * (layer_id + 1))
        layer.prefix.mul_(1.0625).add_(0.000244140625 * (layer_id + 1))
        layer.gamma.mul_(0.984375).add_(0.001953125 * (layer_id + 1))
        layer.up_weight.mul_(1.015625)


def _time_graph_pair_us(
    baseline_graph: torch.cuda.CUDAGraph,
    candidate_graph: torch.cuda.CUDAGraph,
    device: torch.device,
) -> tuple[
    tuple[float, float, list[float]],
    tuple[float, float, list[float]],
]:
    for _ in range(WARMUP):
        baseline_graph.replay()
        candidate_graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    graph_by_name = {
        "baseline": baseline_graph,
        "candidate": candidate_graph,
    }
    for round_index in range(ROUNDS):
        order = (
            ("baseline", "candidate")
            if round_index % 2 == 0
            else ("candidate", "baseline")
        )
        for name in order:
            torch.cuda.synchronize()
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(ITERS):
                graph_by_name[name].replay()
            end.record()
            torch.cuda.synchronize()
            local_us = start.elapsed_time(end) * 1000.0 / (ITERS * GRAPH_LAYERS)
            samples[name].append(_rank_max(local_us, device))
        dist.barrier()
    summaries = []
    for name in ("baseline", "candidate"):
        values = samples[name]
        summaries.append((statistics.median(values), _percentile(values, 0.9), values))
    return summaries[0], summaries[1]


def _build_candidate(
    group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]]]:
    from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
        MNNVLCuteDSLBTFinalizeTuning,
        MNNVLCuteDSLFinalizeAllReduceRMSNorm,
        MNNVLCuteDSLHTFinalizeAllReduceRMSNorm,
        MNNVLCuteDSLHTFinalizeTuning,
        mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported,
        mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported,
    )

    if PROTOCOL == "bt":
        routes = (MNNVLCuteDSLBTFinalizeTuning(**BT_TUNING_SPEC),)
        support_fn = mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported
        candidate_cls = MNNVLCuteDSLFinalizeAllReduceRMSNorm
        candidate_min = BT_MIN_TOKENS
        candidate_max = BT_CAPACITY
        tuning_metadata = [BT_TUNING_SPEC]
    else:
        routes = (MNNVLCuteDSLHTFinalizeTuning(**HT_TUNING_SPEC),)
        support_fn = mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported
        candidate_cls = MNNVLCuteDSLHTFinalizeAllReduceRMSNorm
        candidate_min = HT_MIN_TOKENS
        candidate_max = HT_CAPACITY
        tuning_metadata = [HT_TUNING_SPEC]

    local_support = support_fn(
        group=group,
        tp_size=TP_SIZE,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        dtype=torch.bfloat16,
        candidate_min_tokens=candidate_min,
        candidate_max_tokens=candidate_max,
    )
    support = torch.tensor([int(local_support)], dtype=torch.int32, device=device)
    dist.all_reduce(support, op=dist.ReduceOp.MIN)
    if not bool(support.item()):
        raise RuntimeError(
            f"{PROTOCOL} MNNVL CuTe DSL support vote failed on at least one rank"
        )
    candidate = candidate_cls.initialize(
        group=group,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        candidate_min_tokens=candidate_min,
        candidate_max_tokens=candidate_max,
        tuning_routes=routes,
    )
    return candidate, tuning_metadata


def _moe_apply(
    kernel_module: Any,
    plan: dict[str, Any],
    layer: LayerInputs,
    do_finalize: bool,
) -> Any:
    return kernel_module.moe_apply(
        plan,
        layer.hidden_states,
        layer.weights,
        router_logits=None,
        topk_weights=layer.topk_weights,
        topk_ids=layer.topk_ids,
        num_tokens_global=layer.hidden_states.shape[0],
        max_num_tokens_per_gpu=MAX_TOKENS,
        do_finalize=do_finalize,
        low_latency=None,
        overlap_fn=None,
        shared_input=None,
        shared_weight=None,
        shared_out=None,
    )


def main() -> None:
    _validate_options()
    rank, device = _setup()
    group = dist.group.WORLD
    _collective_errors("optional SiTU runtime check", _runtime_error(), group)

    import flashinfer
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_all_reduce_staged,
        multimem_available,
        multimem_prealloc,
        multimem_stage,
    )
    from tokenspeed_kernel.ops.layernorm import rmsnorm
    from tokenspeed_kernel.ops.tuning import (
        load_packaged_flashinfer_tuning_cache,
        set_autotune_max_num_tokens,
    )

    # Match production startup.  A heuristic tactic is not a valid performance
    # baseline because finalized and deferred SiTU have distinct cache keys.
    tuning_cache_loaded = False
    tuning_cache_error: str | None = None
    try:
        set_autotune_max_num_tokens(HT_CAPACITY)
        tuning_cache_loaded = load_packaged_flashinfer_tuning_cache(
            "kimi-k3", EP_SIZE, TP_SIZE
        )
        if not tuning_cache_loaded:
            tuning_cache_error = (
                "the exact Kimi K3 TP8/EP1 packaged FlashInfer tuning cache "
                "did not load"
            )
    except Exception as error:
        tuning_cache_error = f"{type(error).__name__}: {error}"
    _collective_errors(
        "packaged FlashInfer tuning-cache load", tuning_cache_error, group
    )

    kernel_module, plan = _build_plan_collectively(group)
    group_name = group.group_name
    multimem_support = torch.tensor(
        [int(multimem_available())], dtype=torch.int32, device=device
    )
    dist.all_reduce(multimem_support, op=dist.ReduceOp.MIN)
    if not bool(multimem_support.item()):
        raise RuntimeError("staged multimem is unavailable on at least one rank")
    local_multimem = multimem_prealloc(MAX_TOKENS, (HIDDEN, MODEL_HIDDEN), group_name)
    multimem_ready = torch.tensor(
        [int(local_multimem)], dtype=torch.int32, device=device
    )
    dist.all_reduce(multimem_ready, op=dist.ReduceOp.MIN)
    if not bool(multimem_ready.item()):
        raise RuntimeError(
            "staged multimem baseline is unavailable on at least one rank"
        )
    candidate, tuning_metadata = _build_candidate(group, device)

    weights: list[_MoEWeights] = []
    local_weight_error: str | None = None
    try:
        for layer_id in range(GRAPH_LAYERS):
            weights.append(
                _make_processed_weights(
                    kernel_module,
                    plan,
                    rank,
                    layer_id,
                    device,
                )
            )
        torch.cuda.synchronize()
    except Exception as error:
        local_weight_error = f"{type(error).__name__}: {error}"
    _collective_errors("synthetic NVFP4 weight preparation", local_weight_error, group)
    if len(weights) != GRAPH_LAYERS:
        raise AssertionError("weight preparation vote succeeded with missing layers")
    torch.cuda.synchronize()
    dist.barrier()

    rows: list[dict[str, Any]] = []
    if rank == 0:
        print(
            f"# protocol={PROTOCOL} TP={TP_SIZE}/EP={EP_SIZE} "
            f"H={HIDDEN} model_H={MODEL_HIDDEN} top_k={TOP_K} "
            f"ispp={INTERMEDIATE_PER_RANK} layers/graph={GRAPH_LAYERS}; "
            "latency=full routed-expert+tail per-layer rank-max",
            flush=True,
        )
        print(
            f"{'M':>6} {'old_p50_us':>12} {'new_p50_us':>12} {'p50_x':>8} "
            f"{'old_p90_us':>12} {'new_p90_us':>12} {'p90_x':>8}",
            flush=True,
        )

    for m in TOKENS:
        layers = [
            _make_layer_inputs(weights[layer_id], rank, m, layer_id, device)
            for layer_id in range(GRAPH_LAYERS)
        ]
        input_pointers = {layer.hidden_states.data_ptr() for layer in layers}
        output_pointers = {
            layer.weights._situ_output_buffer.data_ptr() for layer in layers
        }
        if len(input_pointers) != GRAPH_LAYERS or len(output_pointers) != GRAPH_LAYERS:
            raise AssertionError("every synthetic graph layer needs distinct storage")
        shard_start = rank * MODEL_SHARD

        def stage_shared(layer: LayerInputs) -> torch.Tensor:
            shared_stage = multimem_stage(layer.shared_partial, group_name, MAX_TOKENS)
            if shared_stage is None:
                raise RuntimeError("shared-output multimem staging failed")
            return shared_stage

        def finish_tail(
            routed_norm: torch.Tensor,
            layer: LayerInputs,
            shared_stage: torch.Tensor,
        ) -> torch.Tensor:
            target = shared_stage[:, shard_start : shard_start + MODEL_SHARD]
            target.add_(layer.prefix[:, shard_start : shard_start + MODEL_SHARD])
            target.addmm_(routed_norm, layer.up_weight.t())
            reduced = multimem_all_reduce_staged(shared_stage, group_name)
            return reduced.view(m, MODEL_HIDDEN).clone()

        def baseline_impl(diagnostic: bool) -> tuple[torch.Tensor, ...]:
            full_outputs: list[torch.Tensor] = []
            routed_outputs: list[torch.Tensor] = []
            for layer in layers:
                finalized = _moe_apply(kernel_module, plan, layer, True)
                if finalized.data_ptr() != layer.weights._situ_output_buffer.data_ptr():
                    raise AssertionError(
                        "finalized SiTU output missed its stable buffer"
                    )
                routed_stage = multimem_stage(finalized, group_name, MAX_TOKENS)
                if routed_stage is None:
                    raise RuntimeError("routed-output multimem staging failed")
                # Production starts both independent stage copies before AR1.
                shared_stage = stage_shared(layer)
                routed_reduced = multimem_all_reduce_staged(routed_stage, group_name)
                routed_norm = rmsnorm(routed_reduced, layer.gamma, RMS_EPS)
                if diagnostic:
                    routed_outputs.append(routed_norm.clone())
                full_outputs.append(finish_tail(routed_norm, layer, shared_stage))
            return tuple(full_outputs + routed_outputs)

        def baseline() -> tuple[torch.Tensor, ...]:
            return baseline_impl(False)

        def baseline_diagnostic() -> tuple[torch.Tensor, ...]:
            return baseline_impl(True)

        def optimized_impl(diagnostic: bool) -> tuple[torch.Tensor, ...]:
            full_outputs: list[torch.Tensor] = []
            routed_outputs: list[torch.Tensor] = []
            for layer in layers:
                deferred = _moe_apply(kernel_module, plan, layer, False)
                if not isinstance(deferred, tuple) or len(deferred) != 3:
                    raise AssertionError(
                        "deferred SiTU apply did not return its triple"
                    )
                gemm2_output, expert_weights, expanded_idx = deferred
                # Match K3MoeTailComm: hide the shared copy under the persistent
                # candidate rather than issuing it after routed reduction.
                shared_stage = stage_shared(layer)
                routed_norm = candidate(
                    gemm2_output,
                    expert_weights,
                    expanded_idx,
                    layer.gamma,
                )
                if diagnostic:
                    # The candidate's routed output is persistent scratch reused
                    # by the next layer, so snapshot it only in correctness graphs.
                    routed_outputs.append(routed_norm.clone())
                full_outputs.append(finish_tail(routed_norm, layer, shared_stage))
            return tuple(full_outputs + routed_outputs)

        def optimized() -> tuple[torch.Tensor, ...]:
            return optimized_impl(False)

        def optimized_diagnostic() -> tuple[torch.Tensor, ...]:
            return optimized_impl(True)

        # Prime lazy wrappers and persistent buffers.  The exact production
        # tactic table was required above; no autotune search occurs here.
        for _ in range(3):
            baseline()
            optimized()
        torch.cuda.synchronize()
        dist.barrier()

        baseline_eager = baseline_diagnostic()
        candidate_eager = optimized_diagnostic()
        baseline_full, baseline_routed = _split_diagnostic_outputs(baseline_eager)
        candidate_full, candidate_routed = _split_diagnostic_outputs(candidate_eager)
        correctness: dict[str, Any] = {"layers": []}
        for layer_id, (
            baseline_output,
            candidate_output,
            baseline_routed_output,
            candidate_routed_output,
        ) in enumerate(
            zip(
                baseline_full,
                candidate_full,
                baseline_routed,
                candidate_routed,
                strict=True,
            )
        ):
            correctness["layers"].append(
                {
                    "layer": layer_id,
                    "candidate_eager_vs_baseline_full_tail": _assert_correct(
                        candidate_output,
                        baseline_output,
                        device,
                        f"candidate vs baseline eager M={m} layer={layer_id}",
                        atol=MATCH_ATOL,
                        rtol=MATCH_RTOL,
                        l2_rtol=MATCH_L2_RTOL,
                    ),
                    "candidate_eager_vs_baseline_routed_norm": _assert_correct(
                        candidate_routed_output,
                        baseline_routed_output,
                        device,
                        f"candidate routed norm vs baseline eager M={m} "
                        f"layer={layer_id}",
                        atol=MATCH_ATOL,
                        rtol=MATCH_RTOL,
                        l2_rtol=MATCH_L2_RTOL,
                    ),
                }
            )
        torch.cuda.synchronize()
        dist.barrier()

        baseline_graph, baseline_graph_outputs = _capture_layers(
            baseline_diagnostic, 2 * GRAPH_LAYERS
        )
        candidate_graph, candidate_graph_outputs = _capture_layers(
            optimized_diagnostic, 2 * GRAPH_LAYERS
        )
        pre_change_baseline = tuple(output.clone() for output in baseline_eager)
        _change_graph_inputs(layers)
        changed_baseline_eager = baseline_diagnostic()
        changed_candidate_eager = optimized_diagnostic()
        correctness["changed_output_max_abs"] = _assert_outputs_changed(
            pre_change_baseline,
            changed_baseline_eager,
            device,
            f"M={m}",
        )

        baseline_graph.replay()
        torch.cuda.synchronize()
        candidate_graph.replay()
        torch.cuda.synchronize()
        baseline_graph_full, baseline_graph_routed = _split_diagnostic_outputs(
            baseline_graph_outputs
        )
        candidate_graph_full, candidate_graph_routed = _split_diagnostic_outputs(
            candidate_graph_outputs
        )
        changed_baseline_full, changed_baseline_routed = _split_diagnostic_outputs(
            changed_baseline_eager
        )
        changed_candidate_full, changed_candidate_routed = _split_diagnostic_outputs(
            changed_candidate_eager
        )
        for layer_id, (
            baseline_graph_full_output,
            candidate_graph_full_output,
            baseline_graph_routed_output,
            candidate_graph_routed_output,
            baseline_eager_full_output,
            candidate_eager_full_output,
            baseline_eager_routed_output,
            candidate_eager_routed_output,
        ) in enumerate(
            zip(
                baseline_graph_full,
                candidate_graph_full,
                baseline_graph_routed,
                candidate_graph_routed,
                changed_baseline_full,
                changed_candidate_full,
                changed_baseline_routed,
                changed_candidate_routed,
                strict=True,
            )
        ):
            layer_correctness = correctness["layers"][layer_id]
            layer_correctness["baseline_changed_graph_vs_eager_full"] = _assert_correct(
                baseline_graph_full_output,
                baseline_eager_full_output,
                device,
                f"baseline changed graph M={m} layer={layer_id}",
                atol=GRAPH_ATOL,
                rtol=GRAPH_RTOL,
                l2_rtol=GRAPH_L2_RTOL,
            )
            layer_correctness["candidate_changed_graph_vs_eager_full"] = (
                _assert_correct(
                    candidate_graph_full_output,
                    candidate_eager_full_output,
                    device,
                    f"candidate changed graph M={m} layer={layer_id}",
                    atol=GRAPH_ATOL,
                    rtol=GRAPH_RTOL,
                    l2_rtol=GRAPH_L2_RTOL,
                )
            )
            layer_correctness["candidate_changed_graph_vs_baseline_full"] = (
                _assert_correct(
                    candidate_graph_full_output,
                    baseline_graph_full_output,
                    device,
                    f"candidate vs baseline changed graph M={m} layer={layer_id}",
                    atol=MATCH_ATOL,
                    rtol=MATCH_RTOL,
                    l2_rtol=MATCH_L2_RTOL,
                )
            )
            layer_correctness["baseline_changed_graph_vs_eager_routed_norm"] = (
                _assert_correct(
                    baseline_graph_routed_output,
                    baseline_eager_routed_output,
                    device,
                    f"baseline routed norm changed graph M={m} layer={layer_id}",
                    atol=GRAPH_ATOL,
                    rtol=GRAPH_RTOL,
                    l2_rtol=GRAPH_L2_RTOL,
                )
            )
            layer_correctness["candidate_changed_graph_vs_eager_routed_norm"] = (
                _assert_correct(
                    candidate_graph_routed_output,
                    candidate_eager_routed_output,
                    device,
                    f"candidate routed norm changed graph M={m} layer={layer_id}",
                    atol=GRAPH_ATOL,
                    rtol=GRAPH_RTOL,
                    l2_rtol=GRAPH_L2_RTOL,
                )
            )
            layer_correctness["candidate_changed_graph_vs_baseline_routed_norm"] = (
                _assert_correct(
                    candidate_graph_routed_output,
                    baseline_graph_routed_output,
                    device,
                    f"candidate routed norm vs baseline changed graph M={m} "
                    f"layer={layer_id}",
                    atol=MATCH_ATOL,
                    rtol=MATCH_RTOL,
                    l2_rtol=MATCH_L2_RTOL,
                )
            )
        dist.barrier()

        del baseline_graph, candidate_graph
        del baseline_graph_outputs, candidate_graph_outputs
        del baseline_eager, candidate_eager
        del changed_baseline_eager, changed_candidate_eager
        del pre_change_baseline
        torch.cuda.synchronize()
        dist.barrier()

        # Timed graphs contain the production path only; routed-norm snapshots
        # above are deliberately excluded from both sides' latency.
        baseline_graph, baseline_graph_outputs = _capture_layers(baseline, GRAPH_LAYERS)
        candidate_graph, candidate_graph_outputs = _capture_layers(
            optimized, GRAPH_LAYERS
        )
        baseline_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize()
        for layer_id, (
            baseline_graph_output,
            candidate_graph_output,
            baseline_eager_output,
            candidate_eager_output,
        ) in enumerate(
            zip(
                baseline_graph_outputs,
                candidate_graph_outputs,
                changed_baseline_full,
                changed_candidate_full,
                strict=True,
            )
        ):
            layer_correctness = correctness["layers"][layer_id]
            layer_correctness["baseline_timed_graph_vs_eager"] = _assert_correct(
                baseline_graph_output,
                baseline_eager_output,
                device,
                f"baseline timed graph M={m} layer={layer_id}",
                atol=GRAPH_ATOL,
                rtol=GRAPH_RTOL,
                l2_rtol=GRAPH_L2_RTOL,
            )
            layer_correctness["candidate_timed_graph_vs_eager"] = _assert_correct(
                candidate_graph_output,
                candidate_eager_output,
                device,
                f"candidate timed graph M={m} layer={layer_id}",
                atol=GRAPH_ATOL,
                rtol=GRAPH_RTOL,
                l2_rtol=GRAPH_L2_RTOL,
            )
        dist.barrier()

        (old_p50, old_p90, old_samples), (
            new_p50,
            new_p90,
            new_samples,
        ) = _time_graph_pair_us(baseline_graph, candidate_graph, device)

        row: dict[str, Any] = {
            "tokens": m,
            "candidate_protocol": PROTOCOL,
            "graph_layers": GRAPH_LAYERS,
            "correctness": correctness,
            "old_p50_us_per_layer": old_p50,
            "new_p50_us_per_layer": new_p50,
            "old_p90_us_per_layer": old_p90,
            "new_p90_us_per_layer": new_p90,
            "p50_speedup": old_p50 / new_p50,
            "p90_speedup": old_p90 / new_p90,
            "old_samples_us_per_layer": old_samples,
            "new_samples_us_per_layer": new_samples,
        }
        rows.append(row)
        if rank == 0:
            print(
                f"{m:6d} {old_p50:12.2f} {new_p50:12.2f} "
                f"{row['p50_speedup']:8.4f} {old_p90:12.2f} "
                f"{new_p90:12.2f} {row['p90_speedup']:8.4f}",
                flush=True,
            )

        del baseline_graph, candidate_graph
        del baseline_graph_outputs, candidate_graph_outputs
        del layers
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()

    minimum_p50 = min(float(row["p50_speedup"]) for row in rows)
    minimum_p90 = min(float(row["p90_speedup"]) for row in rows)
    summary: dict[str, Any] = {
        "metadata": {
            "candidate_protocol": PROTOCOL,
            "world_size": dist.get_world_size(),
            "tensor_parallel_size": TP_SIZE,
            "expert_parallel_size": EP_SIZE,
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "flashinfer_version": getattr(flashinfer, "__version__", None),
            "moe_apply_kernel": plan["apply_kernel_name"],
            "moe_solution": plan.get("solution"),
            "packaged_tuning_cache_loaded": tuning_cache_loaded,
            "packaged_tuning_cache_model": "kimi-k3",
            "autotune_max_num_tokens": HT_CAPACITY,
            "weight_dtype": "nvfp4",
            "activation": "situ",
            "routed_hidden_size": HIDDEN,
            "latent_size": LATENT,
            "model_hidden_size": MODEL_HIDDEN,
            "model_intermediate_size": MODEL_INTERMEDIATE,
            "intermediate_size_per_tp_rank": INTERMEDIATE_PER_RANK,
            "top_k": TOP_K,
            "rms_eps": RMS_EPS,
            "projection_shard_size": MODEL_SHARD,
            "num_experts": NUM_EXPERTS,
            "production_num_experts": MODEL_NUM_EXPERTS,
            "graph_layers": GRAPH_LAYERS,
            "warmup_replays": WARMUP,
            "timed_replays_per_round": ITERS,
            "rounds": ROUNDS,
            "latency_unit": "microseconds per complete routed-expert+tail layer",
            "latency_rank_reduction": "max",
            "timing_order": "paired and alternating baseline/candidate each round",
            "baseline_collectives": "staged multimem",
            "candidate_tuning_routes": tuning_metadata,
            "synthetic_deviations": [
                "synthetic NVFP4 values instead of checkpoint tensors",
                "deterministic balanced routing over the production 896-expert "
                "domain instead of checkpoint router output",
                "synthetic shared-expert partial and prefix; shared-expert GEMMs "
                "are outside both measured paths",
                f"{GRAPH_LAYERS} sequential synthetic layers instead of the full model",
            ],
        },
        "correctness_thresholds": {
            "candidate_baseline_atol": MATCH_ATOL,
            "candidate_baseline_rtol": MATCH_RTOL,
            "candidate_baseline_l2_rtol": MATCH_L2_RTOL,
            "graph_eager_atol": GRAPH_ATOL,
            "graph_eager_rtol": GRAPH_RTOL,
            "graph_eager_l2_rtol": GRAPH_L2_RTOL,
        },
        "minimum_p50_speedup": minimum_p50,
        "minimum_p90_speedup": minimum_p90,
        "minimum_required_every_m_speedup": MIN_SPEEDUP,
        "p90_gate_enabled": GATE_P90,
        "rows": rows,
    }
    if rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if OUTPUT:
            with open(OUTPUT, "w", encoding="utf-8") as output_file:
                json.dump(summary, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
    dist.barrier()

    failures = [
        row
        for row in rows
        if float(row["p50_speedup"]) <= MIN_SPEEDUP
        or (GATE_P90 and float(row["p90_speedup"]) <= MIN_SPEEDUP)
    ]
    if failures:
        details = ", ".join(
            f"M={row['tokens']}:p50={row['p50_speedup']:.4f},"
            f"p90={row['p90_speedup']:.4f}"
            for row in failures
        )
        raise RuntimeError(
            "SiTU MoE+tail gate failed: every-M p50"
            + ("/p90" if GATE_P90 else "")
            + f" speedup must exceed {MIN_SPEEDUP}; {details}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
