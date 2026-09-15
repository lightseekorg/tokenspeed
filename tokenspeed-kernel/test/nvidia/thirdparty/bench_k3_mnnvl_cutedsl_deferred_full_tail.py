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

"""Benchmark Kimi K3's deferred-finalize TP8 sharded tail on GB300.

The baseline starts at the exact deferred triple produced by TRT-LLM MoE:
``gemm2_output``, BF16 routing weights, and an int32 expanded-to-permuted map
whose ``-1`` entries contribute zero.  It runs TokenSpeed's standalone
``moe_finalize_fuse_shared`` and then mirrors the production communication
tier on both sides of the sharded up projection:

* M < 256: NCCL all-reduce, RMSNorm, local 896-column projection/prefix
  injection, then NCCL all-reduce;
* 256 <= M <= 1024: staged multimem all-reduce, RMSNorm, the same local
  injection, then staged multimem all-reduce and the production clone.

The candidate changes only finalize plus the first all-reduce/RMSNorm to the
selected CuTe DSL protocol: balanced-tree (``BENCH_PROTOCOL=bt``) for medium M
or native-H3584 persistent HT (``BENCH_PROTOCOL=ht``) for large M.  The
shared-expert side, prefix, projection GEMM, second collective, and output
materialization remain matched.

Each CUDA Graph contains several independent, sequential K3 tails with
different gamma and up-projection weights.  Reported CUDA-event time is per
tail and is reduced with MAX across TP ranks.  Before timing, every layer is
checked against an independent FP32-finalize/rank-ordered reference, and both
graphs are replayed after their input buffers are changed.  This makes stale
pointer/input capture and cross-layer workspace reuse correctness failures.

Example TP8 launch on two four-GPU GB300 nodes::

    BENCH_OUTPUT=/tmp/k3-deferred-full-tail.json \
      torchrun --nnodes=2 --nproc-per-node=4 \
      tokenspeed-kernel/test/nvidia/thirdparty/\
bench_k3_mnnvl_cutedsl_deferred_full_tail.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

LATENT = 3584
HIDDEN = 7168
TOP_K = 16
TP_SIZE = 8
SHARD = HIDDEN // TP_SIZE
MULTIMEM_MIN_TOKENS = 256
BT_CAPACITY = 1024
HT_CAPACITY = 8192
EPS = 1e-5


def _csv_ints(name: str, fallback: str) -> list[int]:
    # A colon form is accepted because Slurm's --export uses commas as its own
    # field separator; direct torchrun invocations may keep the CSV spelling.
    raw = os.environ.get(name, fallback).replace(":", ",")
    values = [int(item.strip()) for item in raw.split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")
    return values


PROTOCOL = os.environ.get("BENCH_PROTOCOL", "bt").strip().lower()
DEFAULT_TOKENS = (
    "33,64,128,192,255,256,384,512,768,1024"
    if PROTOCOL == "bt"
    else "1280,2048,4096,6144,8192"
)
TOKENS = _csv_ints("BENCH_TOKENS", DEFAULT_TOKENS)
MAX_TOKENS = max(TOKENS)
GRAPH_LAYERS = int(os.environ.get("BENCH_GRAPH_LAYERS", "4"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))
ITERS = int(os.environ.get("BENCH_ITERS", "20"))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "5"))
MIN_QUALIFIED_SPEEDUP = float(
    os.environ.get(
        "BENCH_MIN_QUALIFIED_SPEEDUP",
        os.environ.get("BENCH_MIN_MEDIUM_SPEEDUP", "1.0"),
    )
)
GATE_P90 = os.environ.get("BENCH_GATE_P90", "1") == "1"
OUTPUT = os.environ.get("BENCH_OUTPUT")

# Final-output bounds permit BF16 reduction-order differences but are small
# enough to reject a missing rank, a non-zero -1 route, a duplicated prefix,
# the wrong 896-column shard, or use of another layer's gamma/weight.
REFERENCE_ATOL = float(os.environ.get("BENCH_REFERENCE_ATOL", "0.03125"))
REFERENCE_RTOL = float(os.environ.get("BENCH_REFERENCE_RTOL", "0.01"))
# The independent reference uses a deterministic rank-ordered second sum,
# whereas both production NCCL and multimem may choose another BF16 tree.  Its
# max-error gate remains tight; this L2 allowance covers the measured 0.50%
# baseline/reference tree delta without relaxing candidate-vs-baseline below.
REFERENCE_L2_RTOL = float(os.environ.get("BENCH_REFERENCE_L2_RTOL", "0.006"))
# Two BF16 output quanta were observed after the first-reduction tree delta is
# amplified by the 3584x896 projection; the core protocol itself remains under
# its separately enforced two-ULP contract.
BASELINE_ATOL = float(os.environ.get("BENCH_BASELINE_ATOL", "0.046875"))
BASELINE_RTOL = float(os.environ.get("BENCH_BASELINE_RTOL", "0.005"))
# BT deliberately uses a deterministic rank-ordered first reduction while the
# production baseline uses the backend's BF16 tree.  The strict core ULP suite
# validates that boundary directly; after the 3584x896 projection, its benign
# 1--2 ULP differences measure about 0.50% L2.  Keep the output max-error gate
# and allow only that measured propagation here.
BASELINE_L2_RTOL = float(os.environ.get("BENCH_BASELINE_L2_RTOL", "0.006"))
LOCAL_FINALIZE_ATOL = float(os.environ.get("BENCH_LOCAL_FINALIZE_ATOL", "0.0078125"))
LOCAL_FINALIZE_RTOL = float(os.environ.get("BENCH_LOCAL_FINALIZE_RTOL", "0.002"))
LOCAL_FINALIZE_L2_RTOL = float(
    os.environ.get("BENCH_LOCAL_FINALIZE_L2_RTOL", "0.00075")
)


def _tuning_spec(name: str, max_tokens: int) -> dict[str, int | bool]:
    """Parse ept,threads,prefetch,reduce,rms,pdl for a bounded BT route."""

    raw = os.environ.get(name, "2,256,1,224,448,1")
    fields = [int(item.strip()) for item in raw.split(",")]
    if len(fields) != 6 or fields[-1] not in (0, 1):
        raise ValueError(f"{name} must be ept,threads,prefetch,reduction,rms,pdl(0|1)")
    ept, threads, prefetch, reduction, rms, pdl = fields
    return {
        "max_tokens": max_tokens,
        "elements_per_thread": ept,
        "threads": threads,
        "prefetch_group": prefetch,
        "reduction_threads": reduction,
        "rms_threads": rms,
        "enable_pdl": bool(pdl),
    }


# Explicit routes make the benchmark result self-describing and prevent a
# FlashInfer default change from silently changing the experiment.  The default
# is the AR-only CUDA-Graph winner at M=512/1024.  Small-M qualification can
# sweep alternatives by setting BENCH_BT_SMALL_TUNING; the medium route has an
# independent BENCH_BT_MEDIUM_TUNING override.
BT_TUNING_SPECS: tuple[dict[str, int | bool], ...] = (
    _tuning_spec("BENCH_BT_SMALL_TUNING", 255),
    _tuning_spec("BENCH_BT_MEDIUM_TUNING", 1024),
)


def _optional_int(value: str) -> int | None:
    return None if value.strip().lower() == "auto" else int(value)


def _ht_tuning_spec() -> dict[str, int | bool | None]:
    """Parse ctas,consumer,vpt,stages,rw,groups,rmsg,rp,shard,pdl."""

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


HT_TUNING_SPEC = _ht_tuning_spec()


@dataclass(slots=True)
class LayerInputs:
    """Pointer-stable inputs and small-M workspaces for one synthetic layer."""

    gemm2_output: torch.Tensor
    expert_weights: torch.Tensor
    expanded_idx: torch.Tensor
    shared_partial: torch.Tensor
    prefix: torch.Tensor
    gamma: torch.Tensor
    up_weight: torch.Tensor
    baseline_nccl_work: torch.Tensor
    candidate_nccl_work: torch.Tensor


def _validate_options() -> None:
    if PROTOCOL not in ("bt", "ht"):
        raise ValueError("BENCH_PROTOCOL must be bt or ht")
    if TOKENS != sorted(TOKENS):
        raise ValueError("BENCH_TOKENS must be strictly increasing")
    if PROTOCOL == "bt":
        if MAX_TOKENS > BT_CAPACITY:
            raise ValueError("the BT full-tail benchmark only covers M <= 1024")
        if not any(value >= MULTIMEM_MIN_TOKENS for value in TOKENS):
            raise ValueError("BT BENCH_TOKENS must contain a medium multimem point")
    elif TOKENS[0] < 1280 or MAX_TOKENS > HT_CAPACITY:
        raise ValueError("the HT full-tail benchmark covers 1280 <= M <= 8192")
    if GRAPH_LAYERS < 2:
        raise ValueError("BENCH_GRAPH_LAYERS must be at least two")
    if WARMUP < 0 or ITERS <= 0 or ROUNDS <= 0:
        raise ValueError("BENCH_WARMUP >= 0 and BENCH_ITERS/ROUNDS > 0 are required")
    if not math.isfinite(MIN_QUALIFIED_SPEEDUP) or MIN_QUALIFIED_SPEEDUP <= 0:
        raise ValueError("the minimum qualified speedup must be finite and positive")


def _setup() -> tuple[int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != TP_SIZE:
        raise RuntimeError("this benchmark requires exactly eight ranks (TP8)")
    return dist.get_rank(), torch.device("cuda", local_rank)


def _rank_max(value: float, device: torch.device) -> float:
    reduced = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return float(reduced.item())


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


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


def _rmsnorm_reference(value: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    value_float = value.float()
    variance = value_float.square().mean(dim=-1, keepdim=True)
    return (value_float * torch.rsqrt(variance + EPS) * gamma.float()).to(
        torch.bfloat16
    )


def _local_finalize_reference(layer: LayerInputs) -> torch.Tensor:
    """FP32 top-k accumulation with explicit zero semantics for ``-1``."""

    m = layer.expert_weights.shape[0]
    flat_idx = layer.expanded_idx.view(-1)
    valid = flat_idx >= 0
    safe_idx = flat_idx.clamp_min(0).to(torch.int64)
    gathered = layer.gemm2_output.index_select(0, safe_idx).view(m, TOP_K, LATENT)
    weighted = gathered.float() * layer.expert_weights.float().unsqueeze(-1)
    weighted.masked_fill_(~valid.view(m, TOP_K, 1), 0.0)
    return weighted.sum(dim=1).to(torch.bfloat16)


def _rank_ordered_sum(local: torch.Tensor) -> torch.Tensor:
    """Render BT's rank-ordered BF16 contribution reduction."""

    gathered = [torch.empty_like(local) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, local)
    total = gathered[0].float()
    for peer in gathered[1:]:
        total.add_(peer.float())
    return total.to(torch.bfloat16)


def _semantic_reference(layer: LayerInputs, rank: int) -> torch.Tensor:
    local_finalized = _local_finalize_reference(layer)
    routed_sum = _rank_ordered_sum(local_finalized)
    routed_norm = _rmsnorm_reference(routed_sum, layer.gamma)

    result = layer.shared_partial.clone()
    start = rank * SHARD
    target = result[:, start : start + SHARD]
    target.add_(layer.prefix[:, start : start + SHARD])
    target.addmm_(routed_norm, layer.up_weight.t())
    # The independent reference intentionally uses NCCL for this last sum at
    # every M.  At medium M the production multimem result may differ only by
    # BF16 reduction ordering, which the strict error bounds cover.
    dist.all_reduce(result)
    return result


def _max_bf16_ulp(actual: torch.Tensor, expected: torch.Tensor) -> int:
    actual_bits = actual.view(torch.int16).to(torch.int32)
    expected_bits = expected.view(torch.int16).to(torch.int32)
    actual_ordered = torch.where(actual_bits < 0, 0x8000 - actual_bits, actual_bits)
    expected_ordered = torch.where(
        expected_bits < 0, 0x8000 - expected_bits, expected_bits
    )
    return int((actual_ordered - expected_ordered).abs().amax().item())


def _error_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int]:
    difference = actual.float() - expected.float()
    finite = torch.tensor(
        [int(torch.isfinite(actual).all())], dtype=torch.int32, device=device
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        return {
            "max_abs": math.inf,
            "reference_max_abs": 0.0,
            "relative_l2": math.inf,
            "max_bf16_ulp": 65535,
        }
    max_abs = _rank_max(float(difference.abs().amax()), device)
    reference_max_abs = _rank_max(float(expected.float().abs().amax()), device)
    relative_l2 = _rank_max(
        float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ),
        device,
    )
    max_ulp = int(_rank_max(float(_max_bf16_ulp(actual, expected)), device))
    return {
        "max_abs": max_abs,
        "reference_max_abs": reference_max_abs,
        "relative_l2": relative_l2,
        # Recorded for diagnostics rather than used as a gate: values close to
        # zero can have a large ULP distance despite a tiny absolute error.
        "max_bf16_ulp": max_ulp,
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
) -> dict[str, float | int]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: got {actual.shape}/{actual.dtype}, "
            f"expected {expected.shape}/{expected.dtype}"
        )
    stats = _error_stats(actual, expected, device)
    allowed_max = atol + rtol * float(stats["reference_max_abs"])
    if float(stats["max_abs"]) > allowed_max or float(stats["relative_l2"]) > l2_rtol:
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
    local_change = max(
        float((new.float() - old.float()).abs().amax())
        for old, new in zip(before, after, strict=True)
    )
    change = _rank_max(local_change, device)
    if change <= 0.01:
        raise AssertionError(f"{label}: changed inputs did not change the reference")
    return change


def _capture_layers(
    fn: Callable[[], tuple[torch.Tensor, ...]],
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
    if len(outputs) != GRAPH_LAYERS:
        raise AssertionError("the captured graph does not contain every layer")
    return graph, outputs


def _change_graph_inputs(layers: Sequence[LayerInputs], rank: int) -> None:
    # Exact BF16-friendly factors avoid injecting comparison noise.  Routing
    # weights are rotated so graph correctness covers every deferred argument,
    # not only the large GEMM2/shared buffers.
    for layer_id, layer in enumerate(layers):
        layer.gemm2_output.mul_(0.9375).add_(0.0009765625 * (rank + layer_id + 1))
        layer.expert_weights.copy_(
            torch.roll(layer.expert_weights, shifts=1 + layer_id % 2, dims=-1)
        )
        layer.shared_partial.mul_(0.875).add_(0.00048828125 * (rank - layer_id))
        layer.prefix.mul_(1.0625).add_(0.000244140625 * (layer_id + 1))


def _geomean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _make_layer_inputs(
    rank: int,
    m: int,
    layer_id: int,
    device: torch.device,
) -> LayerInputs:
    rows = m * TOP_K
    routed_generator = torch.Generator(device=device).manual_seed(
        10_000 + rank * 101 + m * 7 + layer_id
    )
    weights_generator = torch.Generator(device=device).manual_seed(
        20_000 + m * 7 + layer_id
    )
    indices_generator = torch.Generator(device=device).manual_seed(
        30_000 + m * 7 + layer_id
    )
    shared_generator = torch.Generator(device=device).manual_seed(
        40_000 + rank * 101 + m * 7 + layer_id
    )
    prefix_generator = torch.Generator(device=device).manual_seed(
        50_000 + m * 7 + layer_id
    )
    gamma_generator = torch.Generator(device=device).manual_seed(60_000 + layer_id)
    weight_generator = torch.Generator(device=device).manual_seed(
        70_000 + rank * 101 + layer_id
    )

    gemm2_output = (
        torch.randn(
            rows,
            LATENT,
            dtype=torch.bfloat16,
            device=device,
            generator=routed_generator,
        )
        * 0.08
    ).contiguous()
    expert_weights = torch.softmax(
        torch.randn(
            m,
            TOP_K,
            dtype=torch.float32,
            device=device,
            generator=weights_generator,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    expanded_idx = torch.randperm(
        rows, dtype=torch.int64, device=device, generator=indices_generator
    ).to(torch.int32)
    slot_ids = torch.arange(rows, dtype=torch.int32, device=device)
    sentinel_mask = (slot_ids + 3 * layer_id + m) % 11 == 0
    expanded_idx.masked_fill_(sentinel_mask, -1)
    if not bool((expanded_idx == -1).any()):
        raise AssertionError("the synthetic map must exercise -1 sentinel slots")

    shared_partial = (
        torch.randn(
            m,
            HIDDEN,
            dtype=torch.bfloat16,
            device=device,
            generator=shared_generator,
        )
        * 0.02
    ).contiguous()
    prefix = (
        torch.randn(
            m,
            HIDDEN,
            dtype=torch.bfloat16,
            device=device,
            generator=prefix_generator,
        )
        * 0.02
    ).contiguous()
    gamma = (
        1.0
        + torch.randn(
            LATENT,
            dtype=torch.bfloat16,
            device=device,
            generator=gamma_generator,
        )
        * 0.02
    ).contiguous()
    up_weight = (
        torch.randn(
            SHARD,
            LATENT,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_generator,
        )
        * 0.015
    ).contiguous()
    return LayerInputs(
        gemm2_output=gemm2_output,
        expert_weights=expert_weights.contiguous(),
        expanded_idx=expanded_idx.contiguous(),
        shared_partial=shared_partial,
        prefix=prefix,
        gamma=gamma,
        up_weight=up_weight,
        baseline_nccl_work=torch.empty_like(shared_partial),
        candidate_nccl_work=torch.empty_like(shared_partial),
    )


def main() -> None:
    _validate_options()
    from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_finalize import (
        MNNVLCuteDSLBTFinalizeTuning,
        MNNVLCuteDSLFinalizeAllReduceRMSNorm,
        MNNVLCuteDSLHTFinalizeAllReduceRMSNorm,
        MNNVLCuteDSLHTFinalizeTuning,
        mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported,
        mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported,
    )
    from tokenspeed_kernel.ops.communication.multimem import (
        multimem_all_reduce_staged,
        multimem_prealloc,
        multimem_stage,
    )
    from tokenspeed_kernel.ops.layernorm import rmsnorm
    from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared

    rank, device = _setup()
    group = dist.group.WORLD
    group_name = group.group_name
    if not multimem_prealloc(MAX_TOKENS, (LATENT, HIDDEN), group_name):
        raise RuntimeError("production multimem staging is unavailable")

    if PROTOCOL == "bt":
        tuning_routes = tuple(
            MNNVLCuteDSLBTFinalizeTuning(**spec) for spec in BT_TUNING_SPECS
        )
        support_fn = mnnvl_cutedsl_finalize_allreduce_rmsnorm_supported
        candidate_cls = MNNVLCuteDSLFinalizeAllReduceRMSNorm
        candidate_min_tokens = 1
        candidate_capacity = BT_CAPACITY
    else:
        tuning_routes = (MNNVLCuteDSLHTFinalizeTuning(**HT_TUNING_SPEC),)
        support_fn = mnnvl_cutedsl_ht_finalize_allreduce_rmsnorm_supported
        candidate_cls = MNNVLCuteDSLHTFinalizeAllReduceRMSNorm
        candidate_min_tokens = 1280
        candidate_capacity = HT_CAPACITY
    local_support = support_fn(
        group=group,
        tp_size=TP_SIZE,
        hidden_size=LATENT,
        top_k=TOP_K,
        dtype=torch.bfloat16,
        candidate_min_tokens=candidate_min_tokens,
        candidate_max_tokens=candidate_capacity,
    )
    support = torch.tensor([int(local_support)], dtype=torch.int32, device=device)
    dist.all_reduce(support, op=dist.ReduceOp.MIN)
    if not bool(support.item()):
        raise RuntimeError("deferred MNNVL CuTe DSL is not supported on every rank")
    candidate = candidate_cls.initialize(
        group=group,
        hidden_size=LATENT,
        top_k=TOP_K,
        rms_eps=EPS,
        candidate_min_tokens=candidate_min_tokens,
        candidate_max_tokens=candidate_capacity,
        tuning_routes=tuning_routes,
    )

    rows: list[dict[str, Any]] = []
    if rank == 0:
        print(
            f"# protocol={PROTOCOL} TP={TP_SIZE} layers/graph={GRAPH_LAYERS} "
            f"warmup={WARMUP} "
            f"iters={ITERS} rounds={ROUNDS}; latency=per-tail rank-max",
            flush=True,
        )
        print(
            f"{'M':>6} {'tier':>9} {'old_p50_us':>12} {'new_p50_us':>12} "
            f"{'p50_x':>8} {'old_p90_us':>12} {'new_p90_us':>12} {'p90_x':>8}",
            flush=True,
        )

    for m in TOKENS:
        layers = [
            _make_layer_inputs(rank, m, layer_id, device)
            for layer_id in range(GRAPH_LAYERS)
        ]
        gamma_pointers = {layer.gamma.data_ptr() for layer in layers}
        weight_pointers = {layer.up_weight.data_ptr() for layer in layers}
        if len(gamma_pointers) != GRAPH_LAYERS or len(weight_pointers) != GRAPH_LAYERS:
            raise AssertionError(
                "graph layers must have distinct gamma/up-weight storage"
            )
        start = rank * SHARD
        use_multimem = m >= MULTIMEM_MIN_TOKENS

        def finish_tail(
            routed_norm: torch.Tensor,
            layer: LayerInputs,
            nccl_work: torch.Tensor,
            shared_stage: torch.Tensor | None,
        ) -> torch.Tensor:
            if use_multimem:
                if shared_stage is None:
                    raise RuntimeError("shared-output multimem stage was not prepared")
                target = shared_stage[:, start : start + SHARD]
                target.add_(layer.prefix[:, start : start + SHARD])
                target.addmm_(routed_norm, layer.up_weight.t())
                shared_reduced = multimem_all_reduce_staged(shared_stage, group_name)
                return shared_reduced.view(m, HIDDEN).clone()

            # A real shared-expert GEMM produces a fresh tensor each layer.  A
            # replayed synthetic graph instead restores a pointer-stable work
            # buffer; the same copy is present in both measured paths.
            nccl_work.copy_(layer.shared_partial)
            target = nccl_work[:, start : start + SHARD]
            target.add_(layer.prefix[:, start : start + SHARD])
            target.addmm_(routed_norm, layer.up_weight.t())
            dist.all_reduce(nccl_work)
            return nccl_work

        def baseline() -> tuple[torch.Tensor, ...]:
            outputs = []
            for layer in layers:
                finalized = moe_finalize_fuse_shared(
                    layer.gemm2_output,
                    layer.expanded_idx,
                    layer.expert_weights,
                    None,
                    TOP_K,
                )
                if use_multimem:
                    routed_stage = multimem_stage(finalized, group_name, MAX_TOKENS)
                    if routed_stage is None:
                        raise RuntimeError("routed-output multimem staging failed")
                    shared_stage = multimem_stage(
                        layer.shared_partial, group_name, MAX_TOKENS
                    )
                    if shared_stage is None:
                        raise RuntimeError("shared-output multimem staging failed")
                    routed_reduced = multimem_all_reduce_staged(
                        routed_stage, group_name
                    )
                else:
                    shared_stage = None
                    routed_reduced = finalized
                    dist.all_reduce(routed_reduced)
                routed_norm = rmsnorm(routed_reduced, layer.gamma, EPS)
                outputs.append(
                    finish_tail(
                        routed_norm,
                        layer,
                        layer.baseline_nccl_work,
                        shared_stage,
                    )
                )
            return tuple(outputs)

        def optimized() -> tuple[torch.Tensor, ...]:
            outputs = []
            for layer in layers:
                shared_stage = (
                    multimem_stage(layer.shared_partial, group_name, MAX_TOKENS)
                    if use_multimem
                    else None
                )
                if use_multimem and shared_stage is None:
                    raise RuntimeError("shared-output multimem staging failed")
                routed_norm = candidate(
                    layer.gemm2_output,
                    layer.expert_weights,
                    layer.expanded_idx,
                    layer.gamma,
                )
                outputs.append(
                    finish_tail(
                        routed_norm,
                        layer,
                        layer.candidate_nccl_work,
                        shared_stage,
                    )
                )
            return tuple(outputs)

        references = tuple(_semantic_reference(layer, rank) for layer in layers)
        correctness: dict[str, Any] = {"layers": []}
        for layer_id, layer in enumerate(layers):
            local_expected = _local_finalize_reference(layer)
            local_actual = moe_finalize_fuse_shared(
                layer.gemm2_output,
                layer.expanded_idx,
                layer.expert_weights,
                None,
                TOP_K,
            )
            local_stats = _assert_correct(
                local_actual,
                local_expected,
                device,
                f"local finalize M={m} layer={layer_id}",
                atol=LOCAL_FINALIZE_ATOL,
                rtol=LOCAL_FINALIZE_RTOL,
                l2_rtol=LOCAL_FINALIZE_L2_RTOL,
            )
            correctness["layers"].append(
                {
                    "layer": layer_id,
                    "sentinel_slots": int((layer.expanded_idx == -1).sum().item()),
                    "local_finalize_vs_reference": local_stats,
                }
            )

        baseline_eager = baseline()
        candidate_eager = optimized()
        for layer_id, (baseline_out, candidate_out, reference) in enumerate(
            zip(baseline_eager, candidate_eager, references, strict=True)
        ):
            layer_stats = correctness["layers"][layer_id]
            layer_stats["baseline_eager_vs_reference"] = _assert_correct(
                baseline_out,
                reference,
                device,
                f"baseline eager M={m} layer={layer_id}",
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_rtol=REFERENCE_L2_RTOL,
            )
            layer_stats["candidate_eager_vs_reference"] = _assert_correct(
                candidate_out,
                reference,
                device,
                f"candidate eager M={m} layer={layer_id}",
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_rtol=REFERENCE_L2_RTOL,
            )
            layer_stats["candidate_eager_vs_baseline"] = _assert_correct(
                candidate_out,
                baseline_out,
                device,
                f"candidate vs baseline eager M={m} layer={layer_id}",
                atol=BASELINE_ATOL,
                rtol=BASELINE_RTOL,
                l2_rtol=BASELINE_L2_RTOL,
            )
        torch.cuda.synchronize()
        dist.barrier()

        baseline_graph, baseline_graph_outputs = _capture_layers(baseline)
        candidate_graph, candidate_graph_outputs = _capture_layers(optimized)
        pre_change_references = tuple(reference.clone() for reference in references)
        _change_graph_inputs(layers, rank)
        changed_references = tuple(_semantic_reference(layer, rank) for layer in layers)
        correctness["changed_reference_max_abs"] = _assert_outputs_changed(
            pre_change_references,
            changed_references,
            device,
            f"M={m}",
        )

        baseline_graph.replay()
        torch.cuda.synchronize()
        candidate_graph.replay()
        torch.cuda.synchronize()
        for layer_id, (baseline_out, candidate_out, reference) in enumerate(
            zip(
                baseline_graph_outputs,
                candidate_graph_outputs,
                changed_references,
                strict=True,
            )
        ):
            layer_stats = correctness["layers"][layer_id]
            layer_stats["baseline_changed_graph_vs_reference"] = _assert_correct(
                baseline_out,
                reference,
                device,
                f"baseline changed graph M={m} layer={layer_id}",
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_rtol=REFERENCE_L2_RTOL,
            )
            layer_stats["candidate_changed_graph_vs_reference"] = _assert_correct(
                candidate_out,
                reference,
                device,
                f"candidate changed graph M={m} layer={layer_id}",
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
                l2_rtol=REFERENCE_L2_RTOL,
            )
            layer_stats["candidate_changed_graph_vs_baseline"] = _assert_correct(
                candidate_out,
                baseline_out,
                device,
                f"candidate vs baseline changed graph M={m} layer={layer_id}",
                atol=BASELINE_ATOL,
                rtol=BASELINE_RTOL,
                l2_rtol=BASELINE_L2_RTOL,
            )
        dist.barrier()

        (old_p50, old_p90, old_samples), (
            new_p50,
            new_p90,
            new_samples,
        ) = _time_graph_pair_us(baseline_graph, candidate_graph, device)

        tier = "multimem" if use_multimem else "nccl"
        batch_class = (
            "large" if PROTOCOL == "ht" else ("medium" if use_multimem else "small")
        )
        row: dict[str, Any] = {
            "tokens": m,
            "batch_class": batch_class,
            "baseline_tier": tier,
            "candidate_protocol": f"{PROTOCOL}-finalize",
            "graph_layers": GRAPH_LAYERS,
            "correctness": correctness,
            "old_p50_us_per_tail": old_p50,
            "new_p50_us_per_tail": new_p50,
            "old_p90_us_per_tail": old_p90,
            "new_p90_us_per_tail": new_p90,
            "p50_speedup": old_p50 / new_p50,
            "p90_speedup": old_p90 / new_p90,
            "old_samples_us_per_tail": old_samples,
            "new_samples_us_per_tail": new_samples,
        }
        rows.append(row)
        if rank == 0:
            print(
                f"{m:6d} {tier:>9} {old_p50:12.2f} {new_p50:12.2f} "
                f"{row['p50_speedup']:8.4f} {old_p90:12.2f} {new_p90:12.2f} "
                f"{row['p90_speedup']:8.4f}",
                flush=True,
            )

        del baseline_graph, candidate_graph
        del baseline_graph_outputs, candidate_graph_outputs
        del baseline_eager, candidate_eager
        del references, changed_references, pre_change_references
        del layers
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.empty_cache()

    qualified_class = "medium" if PROTOCOL == "bt" else "large"
    qualified = [row for row in rows if row["batch_class"] == qualified_class]
    qualified_p50 = _geomean([float(row["p50_speedup"]) for row in qualified])
    qualified_p90 = _geomean([float(row["p90_speedup"]) for row in qualified])
    qualified_min_p50 = min(float(row["p50_speedup"]) for row in qualified)
    qualified_min_p90 = min(float(row["p90_speedup"]) for row in qualified)
    summary: dict[str, Any] = {
        "metadata": {
            "candidate_protocol": PROTOCOL,
            "world_size": dist.get_world_size(),
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "latent_size": LATENT,
            "hidden_size": HIDDEN,
            "top_k": TOP_K,
            "projection_shard_size": SHARD,
            "graph_layers": GRAPH_LAYERS,
            "warmup_replays": WARMUP,
            "timed_replays_per_round": ITERS,
            "rounds": ROUNDS,
            "latency_unit": "microseconds per complete tail",
            "latency_rank_reduction": "max",
            "timing_order": "paired and alternating baseline/candidate each round",
            "baseline_threshold_tokens": MULTIMEM_MIN_TOKENS,
            "candidate_tuning_routes": (
                list(BT_TUNING_SPECS) if PROTOCOL == "bt" else [HT_TUNING_SPEC]
            ),
        },
        "correctness_thresholds": {
            "reference_atol": REFERENCE_ATOL,
            "reference_rtol": REFERENCE_RTOL,
            "reference_l2_rtol": REFERENCE_L2_RTOL,
            "baseline_atol": BASELINE_ATOL,
            "baseline_rtol": BASELINE_RTOL,
            "baseline_l2_rtol": BASELINE_L2_RTOL,
            "local_finalize_atol": LOCAL_FINALIZE_ATOL,
            "local_finalize_rtol": LOCAL_FINALIZE_RTOL,
            "local_finalize_l2_rtol": LOCAL_FINALIZE_L2_RTOL,
        },
        "qualified_batch_class": qualified_class,
        "qualified_p50_geomean_speedup": qualified_p50,
        "qualified_p90_geomean_speedup": qualified_p90,
        "qualified_min_p50_speedup": qualified_min_p50,
        "qualified_min_p90_speedup": qualified_min_p90,
        "minimum_required_qualified_speedup": MIN_QUALIFIED_SPEEDUP,
        "p90_gate_enabled": GATE_P90,
        "rows": rows,
    }
    if PROTOCOL == "bt":
        # Preserve the original keys for downstream consumers of BT artifacts.
        summary["medium_p50_geomean_speedup"] = qualified_p50
        summary["medium_p90_geomean_speedup"] = qualified_p90
        summary["minimum_required_medium_speedup"] = MIN_QUALIFIED_SPEEDUP
    if rank == 0:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if OUTPUT:
            with open(OUTPUT, "w", encoding="utf-8") as output_file:
                json.dump(summary, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
    dist.barrier()

    failing_rows = [
        row
        for row in qualified
        if float(row["p50_speedup"]) <= MIN_QUALIFIED_SPEEDUP
        or (GATE_P90 and float(row["p90_speedup"]) <= MIN_QUALIFIED_SPEEDUP)
    ]
    failed = bool(failing_rows)
    if failed:
        details = ", ".join(
            f"M={row['tokens']}:p50={row['p50_speedup']:.4f},"
            f"p90={row['p90_speedup']:.4f}"
            for row in failing_rows
        )
        raise RuntimeError(
            f"deferred full-tail performance gate failed: every {qualified_class} p50"
            + ("/p90" if GATE_P90 else "")
            + f" speedup must exceed {MIN_QUALIFIED_SPEEDUP}; {details}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
