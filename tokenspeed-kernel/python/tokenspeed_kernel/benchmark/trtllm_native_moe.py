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

"""Offline tuner for the source-vendored TensorRT-LLM native routed-MoE.

The bucketing, candidate enumeration, and CUDA Graph timing semantics follow
NVIDIA/TensorRT-LLM commit
``54efe57e41f9742d915da977e7701c5cff2b7d6c``.  This focused tool calls the
TokenSpeed DSO directly because importing the full TensorRT-LLM runtime would
both add an unwanted package dependency and register a conflicting C++ custom
class.  The JSON report records every candidate, not only the winner, so an
offline table remains auditable when the embedded cubins change.  Synthetic
routing remains available for smoke tests; publishable V4-Pro measurements can
instead replay exact precomputed routing tensors captured from a real request.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import tokenspeed_kernel.thirdparty.trtllm_native_moe as native_moe
import torch

_UPSTREAM_TRTLLM_COMMIT = "54efe57e41f9742d915da977e7701c5cff2b7d6c"
_GLOBAL_NUM_EXPERTS = 384
_HIDDEN_SIZE = 7168
_TOP_K = 6
_MAX_TUNED_TOKENS = 8192
_DEFAULT_BUCKETS = tuple(1 << exponent for exponent in range(14))
_DEFAULT_TOPOLOGIES = ("ep8",)
_EP8_LOCAL_EXPERTS = 48
_EP8_RANKS = tuple(range(8))
_INPUT_GENERATOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class V4MoeShape:
    """Physical native-runner shape for one V4-Pro rank."""

    topology: str
    ep_rank: int
    num_experts: int
    local_num_experts: int
    local_expert_offset: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    valid_hidden_size: int
    valid_intermediate_size: int


@dataclass
class _Weights:
    gemm1: torch.Tensor
    gemm1_scale: torch.Tensor
    gemm1_bias: torch.Tensor
    gemm1_clamp_limit: torch.Tensor
    gemm2: torch.Tensor
    gemm2_scale: torch.Tensor
    gemm2_bias: torch.Tensor


@dataclass
class _Case:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class _RoutingTraceRecord:
    num_tokens: int
    layer_index: int
    role: str
    path: Path
    file_sha256: str
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


@dataclass(frozen=True)
class _RoutingTraceManifest:
    path: Path
    sha256: str
    model_config_sha256: str
    source: dict[str, object]
    records: dict[int, _RoutingTraceRecord]


def _load_routing_trace_manifest(
    path: Path,
    *,
    required_m_values: tuple[int, ...] = (),
) -> _RoutingTraceManifest:
    """Load and validate exact V4 precomputed-routing tensors."""

    path = path.resolve()
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "tokenspeed_deepseek_v4_routing_trace_manifest"
        or manifest.get("num_experts") != _GLOBAL_NUM_EXPERTS
        or manifest.get("top_k") != _TOP_K
    ):
        raise ValueError(f"unsupported V4 routing trace manifest: {path}")

    records = {}
    for raw_num_tokens, metadata in manifest.get("records", {}).items():
        num_tokens = int(raw_num_tokens)
        record_path = (path.parent / metadata["path"]).resolve()
        try:
            record_path.relative_to(path.parent)
        except ValueError as error:
            raise ValueError(
                f"routing trace record escapes the manifest directory: {record_path}"
            ) from error
        expected_sha256 = str(metadata["sha256"])
        record_bytes = record_path.read_bytes()
        actual_sha256 = hashlib.sha256(record_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"routing trace SHA256 mismatch for M={num_tokens}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        payload = torch.load(
            io.BytesIO(record_bytes), map_location="cpu", weights_only=True
        )
        topk_weights = payload.get("topk_weights")
        topk_ids = payload.get("topk_ids")
        if (
            payload.get("format") != "tokenspeed-deepseek-v4-topk-v1"
            or int(payload.get("num_tokens", -1)) != num_tokens
            or int(payload.get("num_experts", -1)) != _GLOBAL_NUM_EXPERTS
            or int(payload.get("top_k", -1)) != _TOP_K
            or int(payload.get("layer_index", -1)) != int(metadata["layer_index"])
            or not isinstance(topk_weights, torch.Tensor)
            or not isinstance(topk_ids, torch.Tensor)
            or topk_weights.shape != (num_tokens, _TOP_K)
            or topk_ids.shape != (num_tokens, _TOP_K)
            or topk_weights.dtype != torch.float32
            or topk_ids.dtype != torch.int32
        ):
            raise ValueError(f"invalid V4 routing trace record: {record_path}")
        if (
            not torch.isfinite(topk_weights).all()
            or not (topk_weights > 0).all()
            or not torch.allclose(
                topk_weights.sum(dim=-1),
                torch.ones(num_tokens),
                rtol=1.0e-5,
                atol=1.0e-5,
            )
            or int(topk_ids.min()) < 0
            or int(topk_ids.max()) >= _GLOBAL_NUM_EXPERTS
            or not torch.all(torch.sort(topk_ids, dim=-1).values.diff(dim=-1) != 0)
        ):
            raise ValueError(f"invalid values in V4 routing trace: {record_path}")
        if num_tokens in records:
            raise ValueError(f"duplicate V4 routing trace record for M={num_tokens}")
        records[num_tokens] = _RoutingTraceRecord(
            num_tokens=num_tokens,
            layer_index=int(metadata["layer_index"]),
            role=str(metadata["role"]),
            path=record_path,
            file_sha256=actual_sha256,
            topk_weights=topk_weights.contiguous(),
            topk_ids=topk_ids.contiguous(),
        )

    missing = sorted(set(required_m_values) - set(records))
    if missing:
        raise ValueError(f"routing trace manifest is missing M values: {missing}")
    return _RoutingTraceManifest(
        path=path,
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        model_config_sha256=str(manifest["model_config_sha256"]),
        source={
            key: manifest.get(key)
            for key in (
                "dataset_sha256",
                "request_partition_sha256",
                "capture_hook_sha256",
                "capture_run_id",
                "capture_curve",
                "selection_method",
            )
            if manifest.get(key) is not None
        },
        records=records,
    )


def _routing_trace_config(manifest: _RoutingTraceManifest) -> dict[str, object]:
    return {
        "routing_trace_manifest_path": str(manifest.path),
        "routing_trace_manifest_sha256": manifest.sha256,
        "routing_trace_source": manifest.source,
        "routing_trace_scope": "one_exact_route_per_m_exploratory_only",
        "routing_trace_records": {
            str(num_tokens): {
                "sha256": record.file_sha256,
                "layer_index": record.layer_index,
                "role": record.role,
            }
            for num_tokens, record in sorted(manifest.records.items())
        },
    }


def power_of_two_buckets(max_tokens: int = _MAX_TUNED_TOKENS) -> tuple[int, ...]:
    """Return the same ascending floor-power-of-two ladder used upstream."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    largest = 1 << (max_tokens.bit_length() - 1)
    return tuple(1 << exponent for exponent in range(largest.bit_length()))


def runtime_token_bucket(num_tokens: int, max_tokens: int = _MAX_TUNED_TOKENS) -> int:
    """Map a runtime M to TRT-LLM's floor-power-of-two tuning bucket."""

    return native_moe._tactic_token_bucket(num_tokens, max_tokens)


def v4_moe_shape(topology: str, ep_rank: int = 0) -> V4MoeShape:
    """Return the V4-Pro TP8 or EP8 native-runner shape for one rank."""

    if topology == "tp8":
        if ep_rank != 0:
            raise ValueError("TP8 has only ep_rank=0")
        local_num_experts = _GLOBAL_NUM_EXPERTS
        intermediate_size = 384
    elif topology == "ep8":
        if ep_rank not in _EP8_RANKS:
            raise ValueError(f"EP8 rank must be in {_EP8_RANKS}, got {ep_rank}")
        local_num_experts = _EP8_LOCAL_EXPERTS
        intermediate_size = 3072
    else:
        raise ValueError(f"unsupported topology {topology!r}; expected tp8 or ep8")

    return V4MoeShape(
        topology=topology,
        ep_rank=ep_rank,
        num_experts=_GLOBAL_NUM_EXPERTS,
        local_num_experts=local_num_experts,
        local_expert_offset=ep_rank * local_num_experts,
        top_k=_TOP_K,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=intermediate_size,
        valid_hidden_size=_HIDDEN_SIZE,
        valid_intermediate_size=intermediate_size,
    )


def _input_generator_contract(
    shape: V4MoeShape,
    num_tokens: int,
    config: dict[str, object],
) -> dict[str, object]:
    """Describe the deterministic tensors generated for one tuning profile."""

    routing_distribution = str(config["routing_distribution"])
    routing_trace = None
    if routing_distribution == "trace":
        routing_trace = config.get("routing_trace_records", {}).get(str(num_tokens))
        if routing_trace is None:
            raise ValueError(f"routing trace has no record for M={num_tokens}")
    return {
        "schema_version": _INPUT_GENERATOR_SCHEMA_VERSION,
        "routing_distribution": routing_distribution,
        "case_seed": int(config["seed"]) + num_tokens,
        "weight_seed": int(config["seed"]) + shape.ep_rank,
        "weight_data": config["weight_data"],
        "hidden_states": {
            "shape": [num_tokens, shape.hidden_size],
            "source_dtype": "bfloat16",
            "quantization": "mxfp8_quantize",
            "alignment": shape.hidden_size,
        },
        "routing": {
            "top_k": shape.top_k,
            "weights_dtype": "bfloat16",
            "ids_dtype": "int32",
            "global_num_experts": shape.num_experts,
            "trace_manifest_sha256": config.get("routing_trace_manifest_sha256"),
            "trace_record": routing_trace,
        },
    }


def _rank_profile_contract(
    topology: str,
    ep_rank: int,
    num_tokens: int,
    config: dict[str, object],
) -> dict[str, object]:
    shape = v4_moe_shape(topology, ep_rank)
    return {
        "shape": asdict(shape),
        "local_expert_offset": shape.local_expert_offset,
        "input_generator": _input_generator_contract(shape, num_tokens, config),
    }


def make_dummy_topk(
    num_tokens: int,
    shape: V4MoeShape,
    *,
    distribution: str,
    seed: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic, duplicate-free top-k weights and global ids.

    ``balanced`` is the upstream round-robin dummy for pure EP
    (``use_dp=False``). ``random`` uses seeded logits followed by ``topk``;
    unlike the old scratch tuner, it cannot select the same expert twice in a
    row.
    """

    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    device = torch.device(device)
    if distribution == "balanced":
        stride = max(
            1,
            min(shape.local_num_experts, shape.num_experts // shape.top_k),
        )
        token_index = torch.arange(
            num_tokens, dtype=torch.int32, device=device
        ).unsqueeze(1)
        expert_index = (
            torch.arange(shape.top_k, dtype=torch.int32, device=device) * stride
        ).unsqueeze(0)
        topk_ids = (token_index + expert_index) % shape.num_experts
        topk_weights = torch.ones(
            (num_tokens, shape.top_k), dtype=torch.bfloat16, device=device
        )
    elif distribution == "random":
        generator = torch.Generator(device=device).manual_seed(seed)
        logits = torch.randn(
            (num_tokens, shape.num_experts),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        selected_logits, topk_ids = torch.topk(logits, shape.top_k, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = torch.softmax(selected_logits.float(), dim=-1).to(torch.bfloat16)
    else:
        raise ValueError(
            f"unsupported routing distribution {distribution!r}; "
            "expected balanced or random"
        )
    return topk_weights.contiguous(), topk_ids.contiguous()


def _cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _allocate_weights(
    shape: V4MoeShape,
    device: torch.device,
    *,
    seed: int,
    weight_data: str,
) -> _Weights:
    generator = _cuda_generator(device, seed)

    def packed_weight(tensor_shape: tuple[int, ...]) -> torch.Tensor:
        if weight_data == "zero":
            return torch.zeros(tensor_shape, dtype=torch.uint8, device=device)
        if weight_data != "random":
            raise ValueError("weight_data must be random or zero")
        return torch.randint(
            0,
            256,
            tensor_shape,
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    local_experts = shape.local_num_experts
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size
    # E8M0 value 121 keeps random packed FP4 values comfortably finite while
    # retaining incompressible weight payloads for representative memory traffic.
    scale_value = 121
    return _Weights(
        gemm1=packed_weight((local_experts, 2 * intermediate, hidden // 2)),
        gemm1_scale=torch.full(
            (local_experts, 2 * intermediate, hidden // 32),
            scale_value,
            dtype=torch.uint8,
            device=device,
        ),
        gemm1_bias=torch.zeros(
            (local_experts, 2 * intermediate),
            dtype=torch.float32,
            device=device,
        ),
        # V4-Pro uses standard SwiGLU (alpha=None, beta=None) with clamp=10.
        gemm1_clamp_limit=torch.full(
            (local_experts,), 10.0, dtype=torch.float32, device=device
        ),
        gemm2=packed_weight((local_experts, hidden, intermediate // 2)),
        gemm2_scale=torch.full(
            (local_experts, hidden, intermediate // 32),
            scale_value,
            dtype=torch.uint8,
            device=device,
        ),
        gemm2_bias=torch.zeros(
            (local_experts, hidden), dtype=torch.float32, device=device
        ),
    )


def _allocate_case(
    num_tokens: int,
    shape: V4MoeShape,
    device: torch.device,
    *,
    distribution: str,
    seed: int,
    routing_record: _RoutingTraceRecord | None = None,
) -> _Case:
    # Optional and heavyweight: keep FlashInfer out of CPU-only report tests.
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import mxfp8_quantize

    generator = _cuda_generator(device, seed)
    hidden_states = torch.randn(
        (num_tokens, shape.hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    hidden_states, hidden_states_scale = mxfp8_quantize(
        hidden_states, False, alignment=shape.hidden_size
    )
    if distribution == "trace":
        if routing_record is None or routing_record.num_tokens != num_tokens:
            raise ValueError(f"routing trace record does not match M={num_tokens}")
        topk_weights = routing_record.topk_weights.to(
            device=device, dtype=torch.bfloat16
        )
        topk_ids = routing_record.topk_ids.to(device=device, dtype=torch.int32)
    else:
        if routing_record is not None:
            raise ValueError("routing record requires distribution='trace'")
        topk_weights, topk_ids = make_dummy_topk(
            num_tokens,
            shape,
            distribution=distribution,
            seed=seed,
            device=device,
        )
    return _Case(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale.view(torch.uint8).flatten(),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty(
            (num_tokens, shape.valid_hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ),
    )


def _run_tactic(
    shape: V4MoeShape,
    weights: _Weights,
    case: _Case,
    tactic: tuple[int, int],
) -> None:
    native_moe.run_native_mxfp4_moe(
        hidden_states=case.hidden_states,
        hidden_states_scale=case.hidden_states_scale,
        gemm1_weights=weights.gemm1,
        gemm1_weights_scale=weights.gemm1_scale,
        gemm1_bias=weights.gemm1_bias,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=weights.gemm1_clamp_limit,
        gemm2_weights=weights.gemm2,
        gemm2_weights_scale=weights.gemm2_scale,
        gemm2_bias=weights.gemm2_bias,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        intermediate_size=shape.intermediate_size,
        valid_hidden_size=shape.valid_hidden_size,
        valid_intermediate_size=shape.valid_intermediate_size,
        local_expert_offset=shape.local_expert_offset,
        local_num_experts=shape.local_num_experts,
        topk_weights=case.topk_weights,
        topk_ids=case.topk_ids,
        output=case.output,
        tactic=tactic,
    )


def _profile_cuda_graph(
    call: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    samples: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeat):
                call()

        # An untimed replay raises clocks immediately before the event-timed
        # samples without requiring TRT-LLM's private delay_kernel binding.
        graph.replay()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize()
        timings = [
            start.elapsed_time(end) / repeat
            for start, end in zip(starts, ends, strict=True)
        ]
        del graph
    return timings


def _profile_eager(
    call: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    samples: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        timings = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                call()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) / repeat)
    return timings


def _timing_summary(samples_ms: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _raw_median(samples: object, *, context: str) -> float:
    """Recompute a median from persisted raw samples.

    Summaries in input reports are intentionally ignored so a confirmation can
    be audited from the event timings alone.
    """

    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{context} has no raw samples")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{context} has invalid raw samples")
    return statistics.median(values)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _compatibility_identity(
    provenance: dict[str, object], config: dict[str, object]
) -> dict[str, object]:
    """Return the build/workload identity that confirmation shards must share.

    The tuner source hash is deliberately absent: phase two is implemented by
    a newer tuner than an already-collected phase-one report.  Kernel binary,
    vendored source, ABI, software, hardware class, model, and workload timing
    settings must remain identical.
    """

    hardware = provenance["hardware"]
    software = provenance["software"]
    model_config = provenance.get("model_config") or {}
    return {
        "upstream_trtllm_commit": provenance["upstream_trtllm_commit"],
        "tactic_abi": provenance["tactic_abi"],
        "dso_sha256": provenance["dso"]["sha256"],
        "vendored_source_sha256": provenance["vendored_source_sha256"],
        "hardware": {
            "name": hardware["name"],
            "compute_capability": hardware["compute_capability"],
            "driver_version": hardware["driver_version"],
        },
        "software": {
            "torch": software["torch"],
            "torch_cuda": software["torch_cuda"],
            "flashinfer": software["flashinfer"],
        },
        "model_config_sha256": model_config.get("sha256"),
        "tuning_config": {
            "routing_distribution": config["routing_distribution"],
            "routing_trace_manifest_sha256": config.get(
                "routing_trace_manifest_sha256"
            ),
            "routing_trace_records": config.get("routing_trace_records"),
            "seed": config["seed"],
            "weight_data": config["weight_data"],
            "warmup": config["warmup"],
            "repeat": config["repeat"],
            "use_cuda_graph": config["use_cuda_graph"],
        },
    }


def _profile_index(
    profiles: list[dict[str, object]],
) -> dict[tuple[str, int, int], dict[str, object]]:
    indexed = {}
    for profile in profiles:
        key = (
            str(profile["topology"]),
            int(profile["ep_rank"]),
            int(profile["num_tokens"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate topology/ep_rank/M profile {key}")
        indexed[key] = profile
    return indexed


def _result_index(
    profile: dict[str, object],
) -> dict[tuple[int, int], dict[str, object]]:
    indexed = {}
    for result in profile["results"]:
        tactic = tuple(int(value) for value in result["tactic"])
        if len(tactic) != 2:
            raise ValueError(f"invalid tactic {tactic}")
        if tactic in indexed:
            raise ValueError(
                f"duplicate tactic {tactic} in "
                f"{profile['topology']}/ep{profile['ep_rank']} "
                f"M={profile['num_tokens']}"
            )
        indexed[tactic] = result
    return indexed


def build_shared_confirmation_plan(
    initial_report: dict[str, object],
    *,
    initial_report_sha256: str,
    shared_top_k: int,
) -> dict[str, object]:
    """Build a deterministic cross-rank minimax confirmation plan."""

    if shared_top_k < 1:
        raise ValueError("shared_top_k must be positive")
    config = initial_report["config"]
    indexed = _profile_index(initial_report["profiles"])
    entries = []
    for topology in config["topologies"]:
        expected_ranks = (0,) if topology == "tp8" else _EP8_RANKS
        for num_tokens in sorted(int(value) for value in config["m_values"]):
            rank_contracts = {
                str(ep_rank): _rank_profile_contract(
                    str(topology), ep_rank, num_tokens, config
                )
                for ep_rank in expected_ranks
            }
            rows = [
                indexed.get((str(topology), ep_rank, num_tokens))
                for ep_rank in expected_ranks
            ]
            initial_device_uuids = {
                str(row["device"]["uuid"]) for row in rows if row is not None
            }
            initial_contracts_match = all(
                row is not None
                and row.get("shape") == rank_contracts[str(ep_rank)]["shape"]
                and int(row.get("local_expert_offset", -1))
                == rank_contracts[str(ep_rank)]["local_expert_offset"]
                and (
                    config["routing_distribution"] != "trace"
                    or row.get("input_generator")
                    == rank_contracts[str(ep_rank)]["input_generator"]
                )
                for ep_rank, row in zip(expected_ranks, rows, strict=True)
            )
            initial_complete = (
                all(row is not None for row in rows)
                and len(initial_device_uuids) == len(expected_ranks)
                and initial_contracts_match
            )
            ranked_candidates = []
            if initial_complete:
                result_maps = [_result_index(row) for row in rows]
                common = set(result_maps[0])
                for result_map in result_maps[1:]:
                    common.intersection_update(result_map)
                for tactic in common:
                    per_rank = {
                        str(ep_rank): _raw_median(
                            result_map[tactic]["samples_ms"],
                            context=(
                                f"initial {topology}/ep{ep_rank} M={num_tokens} "
                                f"tactic={tactic}"
                            ),
                        )
                        for ep_rank, result_map in zip(
                            expected_ranks, result_maps, strict=True
                        )
                    }
                    ranked_candidates.append(
                        {
                            "tactic": list(tactic),
                            "per_rank_initial_median_ms": per_rank,
                            "worst_rank_initial_median_ms": max(per_rank.values()),
                        }
                    )
                ranked_candidates.sort(
                    key=lambda candidate: (
                        float(candidate["worst_rank_initial_median_ms"]),
                        tuple(candidate["tactic"]),
                    )
                )
            entries.append(
                {
                    "topology": topology,
                    "num_tokens": num_tokens,
                    "expected_ep_ranks": list(expected_ranks),
                    "initial_complete": initial_complete,
                    "initial_device_uuids": sorted(initial_device_uuids),
                    "initial_contracts_match": initial_contracts_match,
                    "rank_contracts": rank_contracts,
                    "candidates": ranked_candidates[:shared_top_k],
                }
            )

    plan = {
        "schema_version": 1,
        "kind": "trtllm_native_moe_shared_confirmation_plan",
        "initial_report_sha256": initial_report_sha256,
        "compatibility_identity": _compatibility_identity(
            initial_report["provenance"], config
        ),
        "shared_top_k": shared_top_k,
        "entries": entries,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _validate_plan(plan: dict[str, object]) -> None:
    expected = plan.get("plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    actual = _canonical_sha256(unhashed)
    if expected != actual:
        raise ValueError(
            f"confirmation plan hash mismatch: expected={expected}, actual={actual}"
        )


def _valid_tactics(shape: V4MoeShape, num_tokens: int) -> list[tuple[int, int]]:
    return [
        tuple(int(value) for value in config)
        for config in native_moe._runner().get_valid_configs(
            shape.top_k,
            shape.hidden_size,
            shape.intermediate_size,
            shape.local_num_experts,
            num_tokens,
            shape.valid_hidden_size,
            shape.valid_intermediate_size,
        )
    ]


def _local_assignment_count(shape: V4MoeShape, case: _Case) -> int:
    local_end = shape.local_expert_offset + shape.local_num_experts
    is_local = (case.topk_ids >= shape.local_expert_offset) & (
        case.topk_ids < local_end
    )
    return int(is_local.sum().item())


def _verification_reference(
    shape: V4MoeShape,
    weights: _Weights,
    case: _Case,
) -> tuple[torch.Tensor, int]:
    _run_tactic(shape, weights, case, (-1, -1))
    torch.cuda.synchronize()
    return case.output.clone(), _local_assignment_count(shape, case)


def _verify_tactic(
    shape: V4MoeShape,
    weights: _Weights,
    case: _Case,
    tactic: tuple[int, int],
    reference: torch.Tensor,
    local_assignments: int,
) -> dict[str, object]:
    _run_tactic(shape, weights, case, tactic)
    torch.cuda.synchronize()

    actual = case.output.float()
    reference_float = reference.float()
    absolute = (actual - reference_float).abs()
    relative = absolute / reference_float.abs().clamp_min(1.0e-5)
    finite = bool(torch.isfinite(actual).all().item())
    informative = bool(torch.count_nonzero(reference).item())
    close = bool(torch.allclose(actual, reference_float, rtol=2.0e-2, atol=2.0e-2))
    informative_for_work = informative if local_assignments else not informative
    return {
        "passed": finite and close and informative_for_work,
        "finite": finite,
        "close_to_fallback": close,
        "local_assignments": local_assignments,
        "no_local_assignments": local_assignments == 0,
        "informative_nonzero_reference": informative,
        "max_abs_diff": float(absolute.max().item()),
        "max_rel_diff": float(relative.max().item()),
    }


def tune_profile(
    shape: V4MoeShape,
    num_tokens: int,
    weights: _Weights,
    device: torch.device,
    *,
    distribution: str,
    seed: int,
    warmup: int,
    repeat: int,
    samples: int,
    confirm_top_k: int,
    confirm_rounds: int,
    verify_output: bool,
    use_cuda_graph: bool,
    input_generator: dict[str, object],
    routing_record: _RoutingTraceRecord | None = None,
) -> dict[str, object]:
    """Profile every valid tactic for one M/rank and return an auditable row."""

    case = _allocate_case(
        num_tokens,
        shape,
        device,
        distribution=distribution,
        seed=seed + num_tokens,
        routing_record=routing_record,
    )
    candidates = _valid_tactics(shape, num_tokens)
    if not candidates:
        raise RuntimeError(f"no valid native MoE tactics for {shape} M={num_tokens}")

    order = list(candidates)
    random.Random(seed + num_tokens + shape.local_expert_offset).shuffle(order)
    profile = _profile_cuda_graph if use_cuda_graph else _profile_eager
    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    wall_start = time.monotonic()
    for index, tactic in enumerate(order, start=1):
        try:
            timings = profile(
                lambda tactic=tactic: _run_tactic(shape, weights, case, tactic),
                warmup=warmup,
                repeat=repeat,
                samples=samples,
            )
            results.append(
                {
                    "tactic": list(tactic),
                    "confirmed": False,
                    **_timing_summary(timings),
                }
            )
        except Exception as error:  # Keep the remaining valid candidates auditable.
            torch.cuda.synchronize()
            errors.append({"tactic": list(tactic), "error": repr(error)})
        if (index % 25 == 0 or index == len(order)) and results:
            best = min(results, key=lambda row: float(row["median_ms"]))
            print(
                f"{shape.topology}/ep{shape.ep_rank} M={num_tokens} "
                f"{index}/{len(order)} best={best['tactic']} "
                f"{float(best['median_ms']):.6f}ms",
                flush=True,
            )

    if not results:
        raise RuntimeError(f"all native MoE tactics failed for {shape} M={num_tokens}")
    results.sort(key=lambda row: float(row["median_ms"]))
    finalists = results[: min(confirm_top_k, len(results))]
    for result in finalists:
        result["confirmed"] = True
        result["initial_samples_ms"] = list(result["samples_ms"])
        result["confirmation_samples_ms"] = []
    confirmation_order: list[list[list[int]]] = []
    for round_index in range(confirm_rounds):
        round_finalists = list(finalists)
        random.Random(
            seed + num_tokens + shape.local_expert_offset + round_index + 1
        ).shuffle(round_finalists)
        confirmation_order.append([result["tactic"] for result in round_finalists])
        for result in round_finalists:
            tactic = tuple(int(value) for value in result["tactic"])
            extra = profile(
                lambda tactic=tactic: _run_tactic(shape, weights, case, tactic),
                warmup=warmup,
                repeat=repeat,
                samples=1,
            )
            result["confirmation_samples_ms"].extend(extra)
    for result in finalists:
        combined = result["initial_samples_ms"] + result["confirmation_samples_ms"]
        result.update(_timing_summary(combined))
    finalists.sort(key=lambda row: float(row["median_ms"]))
    results.sort(key=lambda row: float(row["median_ms"]))

    verification_attempts = []
    winner_result = finalists[0]
    publishable = False
    if verify_output:
        reference, local_assignments = _verification_reference(shape, weights, case)
        for result in finalists:
            tactic = tuple(int(value) for value in result["tactic"])
            verification = _verify_tactic(
                shape,
                weights,
                case,
                tactic,
                reference,
                local_assignments,
            )
            result["verification"] = verification
            verification_attempts.append({"tactic": list(tactic), **verification})
            if verification["passed"] and not publishable:
                winner_result = result
                publishable = True
    else:
        local_assignments = _local_assignment_count(shape, case)
    winner = tuple(int(value) for value in winner_result["tactic"])
    fallback_samples = profile(
        lambda: _run_tactic(shape, weights, case, (-1, -1)),
        warmup=warmup,
        repeat=repeat,
        samples=samples,
    )
    gap_percent = None
    verified_results = [
        result
        for result in finalists
        if not verify_output or result.get("verification", {}).get("passed", False)
    ]
    if len(verified_results) > 1:
        first = float(verified_results[0]["median_ms"])
        second = float(verified_results[1]["median_ms"])
        gap_percent = 100.0 * (second - first) / first

    properties = torch.cuda.get_device_properties(device)

    return {
        "topology": shape.topology,
        "ep_rank": shape.ep_rank,
        "local_expert_offset": shape.local_expert_offset,
        "device": {
            "index": device.index,
            "name": properties.name,
            "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
        },
        "num_tokens": num_tokens,
        "shape": asdict(shape),
        "input_generator": input_generator,
        "candidate_count": len(candidates),
        "profile_order": [list(tactic) for tactic in order],
        "confirmation_order": confirmation_order,
        "winner": list(winner),
        "publishable": publishable,
        "local_assignments": local_assignments,
        "winner_gap_percent": gap_percent,
        "ambiguous_below_one_percent": gap_percent is not None and gap_percent < 1.0,
        "fallback": {"tactic": [-1, -1], **_timing_summary(fallback_samples)},
        "verification_attempts": verification_attempts,
        "results": results,
        "errors": errors,
        "wall_seconds": time.monotonic() - wall_start,
    }


def _confirm_shared_profile(
    shape: V4MoeShape,
    num_tokens: int,
    tactics: list[tuple[int, int]],
    weights: _Weights,
    device: torch.device,
    *,
    distribution: str,
    seed: int,
    warmup: int,
    repeat: int,
    samples: int,
    rounds: int,
    use_cuda_graph: bool,
    input_generator: dict[str, object],
    routing_record: _RoutingTraceRecord | None = None,
) -> dict[str, object]:
    """Reprofile only a shared plan's tactics on one logical rank."""

    case = _allocate_case(
        num_tokens,
        shape,
        device,
        distribution=distribution,
        seed=seed + num_tokens,
        routing_record=routing_record,
    )
    profile = _profile_cuda_graph if use_cuda_graph else _profile_eager
    results = {
        tactic: {
            "tactic": list(tactic),
            "round_samples_ms": [],
        }
        for tactic in tactics
    }
    errors: list[dict[str, object]] = []
    failed: set[tuple[int, int]] = set()
    fallback_tactic = (-1, -1)
    fallback_round_samples_ms = []
    fallback_failed = False
    confirmation_order = []
    wall_start = time.monotonic()
    for round_index in range(rounds):
        # Interleave fallback with candidates so it sees the same warmup,
        # repeat, sample count, and round-level clock conditions.
        order = [*tactics, fallback_tactic]
        random.Random(
            seed + num_tokens + shape.local_expert_offset + 10_000 + round_index
        ).shuffle(order)
        confirmation_order.append([list(tactic) for tactic in order])
        for tactic in order:
            if tactic == fallback_tactic:
                if fallback_failed:
                    continue
                try:
                    raw_samples = profile(
                        lambda: _run_tactic(shape, weights, case, fallback_tactic),
                        warmup=warmup,
                        repeat=repeat,
                        samples=samples,
                    )
                    fallback_round_samples_ms.append(raw_samples)
                except Exception as error:
                    torch.cuda.synchronize()
                    fallback_failed = True
                    errors.append(
                        {
                            "tactic": list(fallback_tactic),
                            "round": round_index,
                            "stage": "fallback_timing",
                            "error": repr(error),
                        }
                    )
                continue
            if tactic in failed:
                continue
            try:
                raw_samples = profile(
                    lambda tactic=tactic: _run_tactic(shape, weights, case, tactic),
                    warmup=warmup,
                    repeat=repeat,
                    samples=samples,
                )
                results[tactic]["round_samples_ms"].append(raw_samples)
            except Exception as error:
                torch.cuda.synchronize()
                failed.add(tactic)
                results[tactic]["error"] = repr(error)
                errors.append(
                    {
                        "tactic": list(tactic),
                        "round": round_index,
                        "error": repr(error),
                    }
                )

    reference, local_assignments = _verification_reference(shape, weights, case)
    for tactic in tactics:
        result = results[tactic]
        if tactic in failed or len(result["round_samples_ms"]) != rounds:
            continue
        try:
            result["verification"] = _verify_tactic(
                shape,
                weights,
                case,
                tactic,
                reference,
                local_assignments,
            )
        except Exception as error:
            torch.cuda.synchronize()
            result["verification_error"] = repr(error)
            errors.append(
                {
                    "tactic": list(tactic),
                    "stage": "verification",
                    "error": repr(error),
                }
            )

    properties = torch.cuda.get_device_properties(device)
    return {
        "topology": shape.topology,
        "ep_rank": shape.ep_rank,
        "local_expert_offset": shape.local_expert_offset,
        "device": {
            "index": device.index,
            "name": properties.name,
            "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
        },
        "num_tokens": num_tokens,
        "shape": asdict(shape),
        "input_generator": input_generator,
        "planned_tactics": [list(tactic) for tactic in tactics],
        "confirmation_order": confirmation_order,
        "fallback_round_samples_ms": fallback_round_samples_ms,
        "fallback_error": fallback_failed,
        "local_assignments": local_assignments,
        "results": [results[tactic] for tactic in tactics],
        "errors": errors,
        "wall_seconds": time.monotonic() - wall_start,
    }


def _routing_trace_from_config(
    config: dict[str, object],
    *,
    override_path: Path | None = None,
) -> _RoutingTraceManifest | None:
    distribution = str(config["routing_distribution"])
    if distribution != "trace":
        if override_path is not None:
            raise ValueError("--routing-trace-manifest requires trace routing")
        return None
    raw_path = override_path or Path(str(config["routing_trace_manifest_path"]))
    manifest = _load_routing_trace_manifest(
        raw_path,
        required_m_values=tuple(int(value) for value in config["m_values"]),
    )
    actual = _routing_trace_config(manifest)
    if actual["routing_trace_manifest_sha256"] != config.get(
        "routing_trace_manifest_sha256"
    ) or actual["routing_trace_records"] != config.get("routing_trace_records"):
        raise ValueError("routing trace manifest does not match the initial report")
    return manifest


def _confirmation_model_config(initial_report: dict[str, object]) -> Path:
    record = initial_report["provenance"].get("model_config") or {}
    raw_path = record.get("resolved_path") or record.get("path")
    if not raw_path:
        raise ValueError("initial report has no model-config path")
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"initial model-config path is unavailable: {path}")
    return path


def _run_shared_confirmation_rank(
    initial_path: Path,
    output_path: Path,
    *,
    ep_rank: int,
    device: torch.device,
    shared_top_k: int,
    confirmation_rounds: int,
    confirmation_samples: int,
    model_config: Path | None,
    routing_trace_manifest_path: Path | None,
    argv: list[str],
) -> None:
    """Run one physical GPU's rank-only shard and persist after every M."""

    initial_report = json.loads(initial_path.read_text())
    initial_sha256 = _sha256_file(initial_path)
    plan = build_shared_confirmation_plan(
        initial_report,
        initial_report_sha256=initial_sha256,
        shared_top_k=shared_top_k,
    )
    config = initial_report["config"]
    routing_trace = _routing_trace_from_config(
        config, override_path=routing_trace_manifest_path
    )
    provenance = build_provenance(
        device,
        argv=argv,
        model_config=model_config or _confirmation_model_config(initial_report),
    )
    expected_identity = plan["compatibility_identity"]
    if (
        routing_trace is not None
        and routing_trace.model_config_sha256
        != expected_identity["model_config_sha256"]
    ):
        raise ValueError("routing trace model config does not match the initial report")
    current_identity = _compatibility_identity(provenance, config)
    if current_identity != expected_identity:
        raise ValueError(
            "confirmation runtime is incompatible with the initial report: "
            f"expected={expected_identity}, actual={current_identity}"
        )

    applicable_entries = [
        entry for entry in plan["entries"] if ep_rank in entry["expected_ep_ranks"]
    ]
    if not applicable_entries:
        raise ValueError(f"initial report has no work for logical ep_rank={ep_rank}")
    confirmation_config: dict[str, object] = {
        "shared_top_k": shared_top_k,
        "rounds": confirmation_rounds,
        "samples_per_round": confirmation_samples,
        "warmup": config["warmup"],
        "repeat": config["repeat"],
        "use_cuda_graph": config["use_cuda_graph"],
    }
    if config.get("routing_trace_manifest_sha256") is not None:
        confirmation_config["routing_trace_manifest_sha256"] = config[
            "routing_trace_manifest_sha256"
        ]
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "trtllm_native_moe_shared_confirmation_rank",
        "initial_report": str(initial_path.resolve()),
        "initial_report_sha256": initial_sha256,
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "compatibility_identity": expected_identity,
        "confirmation_config": confirmation_config,
        "ep_rank": ep_rank,
        "provenance": provenance,
        "profiles": [],
    }
    profiles: list[dict[str, object]] = report["profiles"]
    by_topology: dict[str, list[dict[str, object]]] = {}
    for entry in applicable_entries:
        by_topology.setdefault(str(entry["topology"]), []).append(entry)
    for topology, entries in by_topology.items():
        shape = v4_moe_shape(topology, ep_rank)
        weights = _allocate_weights(
            shape,
            device,
            seed=int(config["seed"]) + ep_rank,
            weight_data=str(config["weight_data"]),
        )
        for entry in sorted(entries, key=lambda value: int(value["num_tokens"])):
            rank_contract = entry["rank_contracts"][str(ep_rank)]
            tactics = [
                tuple(int(value) for value in candidate["tactic"])
                for candidate in entry["candidates"]
            ]
            if tactics:
                profile = _confirm_shared_profile(
                    shape,
                    int(entry["num_tokens"]),
                    tactics,
                    weights,
                    device,
                    distribution=str(config["routing_distribution"]),
                    seed=int(config["seed"]),
                    warmup=int(config["warmup"]),
                    repeat=int(config["repeat"]),
                    samples=confirmation_samples,
                    rounds=confirmation_rounds,
                    use_cuda_graph=bool(config["use_cuda_graph"]),
                    input_generator=rank_contract["input_generator"],
                    routing_record=(
                        routing_trace.records[int(entry["num_tokens"])]
                        if routing_trace is not None
                        else None
                    ),
                )
            else:
                profile = {
                    "topology": topology,
                    "ep_rank": ep_rank,
                    "local_expert_offset": shape.local_expert_offset,
                    "num_tokens": int(entry["num_tokens"]),
                    "shape": asdict(shape),
                    "input_generator": rank_contract["input_generator"],
                    "device": {
                        "index": device.index,
                        "name": provenance["hardware"]["name"],
                        "uuid": provenance["hardware"]["uuid"],
                        "compute_capability": provenance["hardware"][
                            "compute_capability"
                        ],
                    },
                    "planned_tactics": [],
                    "confirmation_order": [],
                    "fallback_round_samples_ms": [],
                    "fallback_error": False,
                    "local_assignments": None,
                    "results": [],
                    "errors": [{"error": "initial plan has no common candidates"}],
                }
            profiles.append(profile)
            _write_report(output_path, report)
        del weights
        torch.cuda.empty_cache()


def aggregate_tables(profiles: list[dict[str, object]]) -> dict[str, object]:
    """Build per-rank and worst-rank shared tables from profile rows."""

    per_rank: dict[str, dict[str, dict[str, object]]] = {}
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for profile in profiles:
        topology = str(profile["topology"])
        ep_rank = int(profile["ep_rank"])
        num_tokens = int(profile["num_tokens"])
        rank_key = f"{topology}:ep{ep_rank}"
        winner = tuple(int(value) for value in profile["winner"])
        winner_result = next(
            result
            for result in profile["results"]
            if tuple(int(value) for value in result["tactic"]) == winner
        )
        per_rank.setdefault(rank_key, {})[str(num_tokens)] = {
            "tactic": profile["winner"],
            "median_ms": winner_result["median_ms"],
            "gap_percent": profile["winner_gap_percent"],
            "publishable": profile["publishable"],
            "local_assignments": profile["local_assignments"],
            "device_uuid": profile["device"]["uuid"],
        }
        groups.setdefault((topology, num_tokens), []).append(profile)

    shared: dict[str, dict[str, dict[str, object]]] = {}
    for (topology, num_tokens), rows in sorted(groups.items()):
        expected_ranks = {0} if topology == "tp8" else set(_EP8_RANKS)
        seen_ranks = {int(row["ep_rank"]) for row in rows}
        if len(seen_ranks) != len(rows):
            raise ValueError(f"duplicate logical rank for {topology} M={num_tokens}")
        physical_devices = {str(row["device"]["uuid"]) for row in rows}
        score_maps = []
        for row in rows:
            score_maps.append(
                {
                    tuple(int(value) for value in result["tactic"]): float(
                        result["median_ms"]
                    )
                    for result in row["results"]
                    if result.get("confirmed", False)
                    and result.get("verification", {}).get("passed", False)
                }
            )
        common = set(score_maps[0])
        for scores in score_maps[1:]:
            common.intersection_update(scores)
        ranked = sorted(
            (
                max(scores[tactic] for scores in score_maps),
                tactic,
            )
            for tactic in common
        )
        complete = seen_ranks == expected_ranks and len(physical_devices) == len(
            expected_ranks
        )
        if not ranked:
            shared.setdefault(topology, {})[str(num_tokens)] = {
                "tactic": None,
                "seen_ep_ranks": sorted(seen_ranks),
                "expected_ep_ranks": sorted(expected_ranks),
                "physical_device_uuids": sorted(physical_devices),
                "complete": complete,
                "verified_on_all_ranks": False,
                "publishable": False,
                "reason": "no tactic was confirmed and verified on every rank",
            }
            continue
        best_score, best_tactic = ranked[0]
        gap_percent = None
        if len(ranked) > 1:
            gap_percent = 100.0 * (ranked[1][0] - best_score) / best_score
        shared_verified = True
        for row in rows:
            result = next(
                (
                    result
                    for result in row["results"]
                    if tuple(int(value) for value in result["tactic"]) == best_tactic
                ),
                None,
            )
            if result is None or not result.get("verification", {}).get(
                "passed", False
            ):
                shared_verified = False
                break
        shared.setdefault(topology, {})[str(num_tokens)] = {
            "tactic": list(best_tactic),
            "worst_rank_median_ms": best_score,
            "gap_percent": gap_percent,
            "ambiguous_below_one_percent": (
                gap_percent is not None and gap_percent < 1.0
            ),
            "seen_ep_ranks": sorted(seen_ranks),
            "expected_ep_ranks": sorted(expected_ranks),
            "physical_device_uuids": sorted(physical_devices),
            "complete": complete,
            "verified_on_all_ranks": shared_verified,
            "publishable": complete
            and shared_verified
            and gap_percent is not None
            and gap_percent >= 1.0,
        }
    return {"per_rank": per_rank, "shared_worst_rank": shared}


def _confirmation_raw_median(
    result: dict[str, object],
    *,
    rounds: int,
    samples_per_round: int,
    context: str,
) -> float:
    raw_rounds = result.get("round_samples_ms")
    if not isinstance(raw_rounds, list) or len(raw_rounds) != rounds:
        raise ValueError(f"{context} has incomplete confirmation rounds")
    flattened = []
    for round_index, raw_samples in enumerate(raw_rounds):
        if not isinstance(raw_samples, list) or len(raw_samples) != samples_per_round:
            raise ValueError(
                f"{context} round {round_index} has incomplete raw samples"
            )
        flattened.extend(raw_samples)
    return _raw_median(flattened, context=context)


def finalize_shared_confirmation(
    initial_report: dict[str, object],
    confirmation_reports: list[dict[str, object]],
    *,
    initial_report_sha256: str,
) -> dict[str, object]:
    """Finalize shared tactics on CPU using confirmation raw samples only."""

    if not confirmation_reports:
        raise ValueError("at least one confirmation report is required")
    first_plan = confirmation_reports[0].get("plan")
    if not isinstance(first_plan, dict):
        raise ValueError("confirmation report has no embedded plan")
    _validate_plan(first_plan)
    shared_top_k = int(first_plan["shared_top_k"])
    expected_plan = build_shared_confirmation_plan(
        initial_report,
        initial_report_sha256=initial_report_sha256,
        shared_top_k=shared_top_k,
    )
    expected_identity = expected_plan["compatibility_identity"]

    confirmation_configs = {
        _canonical_sha256(report.get("confirmation_config"))
        for report in confirmation_reports
    }
    if len(confirmation_configs) != 1:
        raise ValueError("confirmation shards use different confirmation settings")
    confirmation_config = confirmation_reports[0]["confirmation_config"]
    required_config = {
        "shared_top_k": shared_top_k,
        "rounds": int(confirmation_config["rounds"]),
        "samples_per_round": int(confirmation_config["samples_per_round"]),
        "warmup": initial_report["config"]["warmup"],
        "repeat": initial_report["config"]["repeat"],
        "use_cuda_graph": initial_report["config"]["use_cuda_graph"],
    }
    routing_trace_sha256 = initial_report["config"].get("routing_trace_manifest_sha256")
    if routing_trace_sha256 is not None:
        required_config["routing_trace_manifest_sha256"] = routing_trace_sha256
    if confirmation_config != required_config:
        raise ValueError("confirmation settings do not match the initial report")

    tuner_hashes = set()
    reports_by_rank = {}
    profile_indexes = {}
    valid_ranks = {
        int(rank)
        for entry in expected_plan["entries"]
        for rank in entry["expected_ep_ranks"]
    }
    for report in confirmation_reports:
        if report.get("kind") != "trtllm_native_moe_shared_confirmation_rank":
            raise ValueError("input is not a shared-confirmation rank report")
        plan = report.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("confirmation report has no embedded plan")
        _validate_plan(plan)
        if (
            plan != expected_plan
            or report.get("plan_sha256") != expected_plan["plan_sha256"]
        ):
            raise ValueError(
                "confirmation shard plan does not match the initial report"
            )
        if report.get("initial_report_sha256") != initial_report_sha256:
            raise ValueError("confirmation shard references a different initial report")
        if report.get("compatibility_identity") != expected_identity:
            raise ValueError("confirmation shard compatibility identity was modified")
        actual_identity = _compatibility_identity(
            report["provenance"], initial_report["config"]
        )
        if actual_identity != expected_identity:
            raise ValueError(
                "confirmation shard is incompatible with the initial report"
            )
        tuner_hashes.add(report["provenance"]["tuner_sha256"])
        ep_rank = int(report["ep_rank"])
        if ep_rank not in valid_ranks:
            raise ValueError(f"unexpected confirmation ep_rank={ep_rank}")
        if ep_rank in reports_by_rank:
            raise ValueError(f"duplicate confirmation shard for ep_rank={ep_rank}")
        reports_by_rank[ep_rank] = report
        profile_index = _profile_index(report["profiles"])
        expected_profile_keys = {
            (str(entry["topology"]), ep_rank, int(entry["num_tokens"]))
            for entry in expected_plan["entries"]
            if ep_rank in entry["expected_ep_ranks"]
        }
        unexpected_profile_keys = set(profile_index) - expected_profile_keys
        if unexpected_profile_keys:
            raise ValueError(
                f"unexpected confirmation profiles for ep_rank={ep_rank}: "
                f"{sorted(unexpected_profile_keys)}"
            )
        profile_indexes[ep_rank] = profile_index
    if len(tuner_hashes) != 1:
        raise ValueError("confirmation shards use different tuner SHA256 values")

    entries = []
    for plan_entry in expected_plan["entries"]:
        topology = str(plan_entry["topology"])
        num_tokens = int(plan_entry["num_tokens"])
        expected_ranks = [int(rank) for rank in plan_entry["expected_ep_ranks"]]
        present_ranks = [rank for rank in expected_ranks if rank in reports_by_rank]
        missing_ranks = sorted(set(expected_ranks) - set(present_ranks))
        rows = {}
        result_maps = {}
        missing_profiles = []
        physical_devices = set()
        per_rank_fallback_median = {}
        fallback_failures = {}
        for ep_rank in present_ranks:
            report = reports_by_rank[ep_rank]
            profile = profile_indexes[ep_rank].get((topology, ep_rank, num_tokens))
            if profile is None:
                missing_profiles.append(ep_rank)
                continue
            report_uuid = str(report["provenance"]["hardware"]["uuid"])
            profile_uuid = str(profile["device"]["uuid"])
            if profile_uuid != report_uuid:
                raise ValueError(
                    f"profile GPU UUID does not match shard provenance for ep_rank={ep_rank}"
                )
            rank_contract = plan_entry["rank_contracts"][str(ep_rank)]
            if (
                profile.get("shape") != rank_contract["shape"]
                or int(profile.get("local_expert_offset", -1))
                != rank_contract["local_expert_offset"]
                or profile.get("input_generator") != rank_contract["input_generator"]
            ):
                raise ValueError(
                    f"profile shape/input contract does not match the plan for "
                    f"{topology}/ep{ep_rank} M={num_tokens}"
                )
            expected_tactics = [
                candidate["tactic"] for candidate in plan_entry["candidates"]
            ]
            if profile.get("planned_tactics") != expected_tactics:
                raise ValueError(
                    f"profile tactics do not match the plan for "
                    f"{topology}/ep{ep_rank} M={num_tokens}"
                )
            if expected_tactics:
                expected_round_tactics = {
                    tuple(int(value) for value in tactic) for tactic in expected_tactics
                }
                expected_round_tactics.add((-1, -1))
                confirmation_order = profile.get("confirmation_order")
                if not isinstance(confirmation_order, list) or len(
                    confirmation_order
                ) != int(confirmation_config["rounds"]):
                    raise ValueError(
                        f"profile confirmation order is incomplete for "
                        f"{topology}/ep{ep_rank} M={num_tokens}"
                    )
                for round_order in confirmation_order:
                    ordered_tactics = [
                        tuple(int(value) for value in tactic) for tactic in round_order
                    ]
                    if (
                        len(ordered_tactics) != len(expected_round_tactics)
                        or set(ordered_tactics) != expected_round_tactics
                    ):
                        raise ValueError(
                            f"profile confirmation round does not include every "
                            f"planned tactic and fallback for "
                            f"{topology}/ep{ep_rank} M={num_tokens}"
                        )
            result_map = _result_index(profile)
            expected_tactic_set = {
                tuple(int(value) for value in tactic) for tactic in expected_tactics
            }
            if set(result_map) != expected_tactic_set:
                raise ValueError(
                    f"profile result tactics do not exactly match the plan for "
                    f"{topology}/ep{ep_rank} M={num_tokens}"
                )
            rows[ep_rank] = profile
            result_maps[ep_rank] = result_map
            physical_devices.add(report_uuid)
            if profile.get("fallback_error"):
                fallback_failures[str(ep_rank)] = "fallback timing failed"
            else:
                try:
                    per_rank_fallback_median[str(ep_rank)] = _confirmation_raw_median(
                        {"round_samples_ms": profile.get("fallback_round_samples_ms")},
                        rounds=int(confirmation_config["rounds"]),
                        samples_per_round=int(confirmation_config["samples_per_round"]),
                        context=(
                            f"confirmation fallback {topology}/ep{ep_rank} "
                            f"M={num_tokens}"
                        ),
                    )
                except ValueError as error:
                    fallback_failures[str(ep_rank)] = str(error)
        for ep_rank in expected_ranks:
            if ep_rank not in rows:
                fallback_failures[str(ep_rank)] = "missing confirmation profile"

        candidate_evaluations = []
        for candidate in plan_entry["candidates"]:
            tactic = tuple(int(value) for value in candidate["tactic"])
            per_rank_median = {}
            failures = {}
            for ep_rank in expected_ranks:
                row = rows.get(ep_rank)
                if row is None:
                    failures[str(ep_rank)] = "missing confirmation profile"
                    continue
                result = result_maps[ep_rank].get(tactic)
                if result is None:
                    failures[str(ep_rank)] = "missing tactic result"
                    continue
                if result.get("error") or result.get("verification_error"):
                    failures[str(ep_rank)] = "tactic execution failed"
                    continue
                verification = result.get("verification") or {}
                if not verification.get("passed", False):
                    failures[str(ep_rank)] = "output verification failed"
                    continue
                try:
                    median = _confirmation_raw_median(
                        result,
                        rounds=int(confirmation_config["rounds"]),
                        samples_per_round=int(confirmation_config["samples_per_round"]),
                        context=(
                            f"confirmation {topology}/ep{ep_rank} M={num_tokens} "
                            f"tactic={tactic}"
                        ),
                    )
                except ValueError as error:
                    failures[str(ep_rank)] = str(error)
                    continue
                per_rank_median[str(ep_rank)] = median
            successful = not failures and len(per_rank_median) == len(expected_ranks)
            candidate_evaluations.append(
                {
                    "tactic": list(tactic),
                    "successful_on_all_ranks": successful,
                    "verified_on_all_ranks": successful,
                    "per_rank_confirmation_median_ms": per_rank_median,
                    "worst_rank_confirmation_median_ms": (
                        max(per_rank_median.values()) if successful else None
                    ),
                    "failures": failures,
                }
            )

        ranked = sorted(
            (
                candidate
                for candidate in candidate_evaluations
                if candidate["successful_on_all_ranks"]
            ),
            key=lambda candidate: (
                float(candidate["worst_rank_confirmation_median_ms"]),
                tuple(candidate["tactic"]),
            ),
        )
        winner = ranked[0] if ranked else None
        gap_percent = None
        if len(ranked) > 1:
            first = float(winner["worst_rank_confirmation_median_ms"])
            second = float(ranked[1]["worst_rank_confirmation_median_ms"])
            gap_percent = 100.0 * (second - first) / first
        fallback_complete = not fallback_failures and len(
            per_rank_fallback_median
        ) == len(expected_ranks)
        fallback_worst_rank_median = (
            max(per_rank_fallback_median.values()) if fallback_complete else None
        )
        winner_vs_fallback_speedup_percent = None
        if winner is not None and fallback_worst_rank_median is not None:
            winner_worst = float(winner["worst_rank_confirmation_median_ms"])
            winner_vs_fallback_speedup_percent = (
                100.0
                * (fallback_worst_rank_median - winner_worst)
                / fallback_worst_rank_median
            )
        sampling_sufficient = (
            int(confirmation_config["rounds"]) >= 3
            and int(confirmation_config["samples_per_round"]) >= 3
        )
        complete = (
            bool(plan_entry["initial_complete"])
            and not missing_ranks
            and not missing_profiles
            and len(physical_devices) == len(expected_ranks)
        )
        publishable = (
            complete
            and winner is not None
            and bool(winner["verified_on_all_ranks"])
            and gap_percent is not None
            and gap_percent >= 1.0
            and fallback_complete
            and winner_vs_fallback_speedup_percent is not None
            and winner_vs_fallback_speedup_percent >= 1.0
            and sampling_sufficient
        )
        diagnostics = []
        if not plan_entry["initial_complete"]:
            diagnostics.append("initial report does not contain every expected rank")
        if missing_ranks:
            diagnostics.append(f"missing confirmation ranks: {missing_ranks}")
        if missing_profiles:
            diagnostics.append(f"missing confirmation profiles: {missing_profiles}")
        if len(physical_devices) != len(expected_ranks):
            diagnostics.append(
                "confirmation ranks were not collected on distinct physical GPUs"
            )
        if fallback_failures:
            diagnostics.append(f"fallback timing incomplete: {fallback_failures}")
        if winner is None:
            diagnostics.append("no candidate succeeded and verified on every rank")
        elif gap_percent is None:
            diagnostics.append("fewer than two candidates can establish a winner gap")
        elif gap_percent < 1.0:
            diagnostics.append(f"winner gap {gap_percent:.6f}% is below 1%")
        if (
            winner_vs_fallback_speedup_percent is not None
            and winner_vs_fallback_speedup_percent < 1.0
        ):
            diagnostics.append(
                "winner vs 8-rank worst fallback speedup "
                f"{winner_vs_fallback_speedup_percent:.6f}% is below 1%"
            )
        if not sampling_sufficient:
            diagnostics.append(
                "confirmation is smoke-only: publication requires at least "
                "3 rounds and 3 samples per round"
            )
        entries.append(
            {
                "topology": topology,
                "num_tokens": num_tokens,
                "expected_ep_ranks": expected_ranks,
                "seen_ep_ranks": sorted(rows),
                "physical_device_uuids": sorted(physical_devices),
                "complete": complete,
                "winner": winner["tactic"] if winner is not None else None,
                "winner_worst_rank_median_ms": (
                    winner["worst_rank_confirmation_median_ms"]
                    if winner is not None
                    else None
                ),
                "per_rank_fallback_median_ms": per_rank_fallback_median,
                "fallback_worst_rank_median_ms": fallback_worst_rank_median,
                "fallback_failures": fallback_failures,
                "winner_vs_fallback_speedup_percent": (
                    winner_vs_fallback_speedup_percent
                ),
                "gap_percent": gap_percent,
                "verified_on_all_ranks": (
                    bool(winner["verified_on_all_ranks"])
                    if winner is not None
                    else False
                ),
                "publishable": publishable,
                "sampling_sufficient_for_publication": sampling_sufficient,
                "diagnostics": diagnostics,
                "candidates": candidate_evaluations,
            }
        )

    runtime_tables = {}
    runtime_table_status = {}
    runtime_table_eligible = initial_report["config"]["routing_distribution"] != "trace"
    for topology in sorted({str(entry["topology"]) for entry in entries}):
        topology_entries = [entry for entry in entries if entry["topology"] == topology]
        full_bucket_set = {
            int(entry["num_tokens"]) for entry in topology_entries
        } == set(_DEFAULT_BUCKETS)
        complete = (
            runtime_table_eligible
            and full_bucket_set
            and all(bool(entry["publishable"]) for entry in topology_entries)
        )
        runtime_table_status[topology] = {
            "eligible_for_generic_runtime_table": runtime_table_eligible,
            "has_all_14_buckets": full_bucket_set,
            "all_buckets_publishable": complete,
            "reason": (
                None
                if runtime_table_eligible
                else "one exact routing trace per M is exploratory-only"
            ),
        }
        if complete:
            runtime_tables[topology] = {
                str(entry["num_tokens"]): entry["winner"]
                for entry in sorted(
                    topology_entries, key=lambda value: int(value["num_tokens"])
                )
            }
    runtime_table_publishable = bool(runtime_table_status) and all(
        bool(status["all_buckets_publishable"])
        for status in runtime_table_status.values()
    )

    return {
        "schema_version": 1,
        "kind": "trtllm_native_moe_shared_confirmation_final",
        "initial_report_sha256": initial_report_sha256,
        "plan": expected_plan,
        "plan_sha256": expected_plan["plan_sha256"],
        "compatibility_identity": expected_identity,
        "confirmation_config": confirmation_config,
        "confirmation_tuner_sha256": next(iter(tuner_hashes)),
        "entries": entries,
        "all_entries_publishable": all(bool(entry["publishable"]) for entry in entries),
        "runtime_table_status": runtime_table_status,
        "runtime_table_publishable": runtime_table_publishable,
        "runtime_tables": runtime_tables,
    }


def _finalize_shared_from_paths(
    initial_path: Path, confirmation_paths: list[Path]
) -> dict[str, object]:
    initial_report = json.loads(initial_path.read_text())
    confirmation_reports = [json.loads(path.read_text()) for path in confirmation_paths]
    finalized = finalize_shared_confirmation(
        initial_report,
        confirmation_reports,
        initial_report_sha256=_sha256_file(initial_path),
    )
    finalized["initial_report"] = str(initial_path.resolve())
    finalized["confirmation_reports"] = [
        str(path.resolve()) for path in confirmation_paths
    ]
    return finalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "objs" in path.parts or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_provenance(
    device: torch.device,
    *,
    argv: list[str],
    model_config: Path | None,
) -> dict[str, object]:
    """Record the build, hardware, software, and invocation identity."""

    dso = native_moe._LIBRARY.resolve()
    source_root = dso.parent.parent
    repository = Path(__file__).resolve().parents[4]
    properties = torch.cuda.get_device_properties(device)
    model_config_record = None
    if model_config is not None:
        requested_model_config = model_config.absolute()
        resolved_model_config = model_config.resolve()
        model_config_record = {
            "path": str(requested_model_config),
            "resolved_path": str(resolved_model_config),
            "sha256": _sha256_file(resolved_model_config),
        }
    try:
        import flashinfer

        flashinfer_record = {
            "runtime_version": getattr(flashinfer, "__version__", None),
            "module_path": str(Path(flashinfer.__file__).resolve()),
            "distribution_version": _package_version("flashinfer-python")
            or _package_version("flashinfer"),
        }
    except ImportError:
        flashinfer_record = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "argv": argv,
        "upstream_trtllm_commit": _UPSTREAM_TRTLLM_COMMIT,
        "tactic_abi": native_moe._TACTIC_ABI,
        "dso": {"path": str(dso), "sha256": _sha256_file(dso)},
        "vendored_source_sha256": _sha256_tree(source_root),
        "tuner_sha256": _sha256_file(Path(__file__).resolve()),
        "repository": {
            "head": _command_output(["git", "rev-parse", "HEAD"], repository),
            "status": _command_output(
                ["git", "status", "--short", "--untracked-files=all"], repository
            ),
        },
        "hardware": {
            "device_index": device.index,
            "name": properties.name,
            "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "driver_version": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    f"--id={device.index}",
                ]
            ),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": flashinfer_record,
        },
        "model_config": model_config_record,
    }


def _parse_csv_ints(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw.split(",") if value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated ints"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def _parse_topologies(raw: str) -> tuple[str, ...]:
    values = tuple(value for value in raw.split(",") if value)
    if not values or any(value not in {"tp8", "ep8"} for value in values):
        raise argparse.ArgumentTypeError("topologies must contain tp8 and/or ep8")
    return values


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_report_coverage(path: Path, report: dict[str, object]) -> None:
    config = report["config"]
    expected = set()
    for topology in config["topologies"]:
        ranks = (0,) if topology == "tp8" else config["ep_ranks"]
        expected.update(
            (topology, int(rank), int(num_tokens))
            for rank in ranks
            for num_tokens in config["m_values"]
        )
    actual = {
        (
            profile["topology"],
            int(profile["ep_rank"]),
            int(profile["num_tokens"]),
        )
        for profile in report["profiles"]
    }
    if actual != expected or len(actual) != len(report["profiles"]):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"incomplete profile matrix in {path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _merge_reports(paths: list[Path]) -> dict[str, object]:
    reports = [json.loads(path.read_text()) for path in paths]
    for path, report in zip(paths, reports, strict=True):
        _validate_report_coverage(path, report)
    profiles = []
    for path, report in zip(paths, reports, strict=True):
        report_uuid = str(report["provenance"]["hardware"]["uuid"])
        for original in report["profiles"]:
            profile_uuid = str(original["device"]["uuid"])
            if profile_uuid != report_uuid:
                raise ValueError(
                    f"profile GPU UUID {profile_uuid} does not match source "
                    f"report {path} provenance UUID {report_uuid}"
                )
            profile = dict(original)
            profile["report_source"] = str(path.resolve())
            profiles.append(profile)
    identities = {
        (
            report["provenance"]["upstream_trtllm_commit"],
            report["provenance"]["tactic_abi"],
            report["provenance"]["dso"]["sha256"],
            report["provenance"]["vendored_source_sha256"],
            report["provenance"]["tuner_sha256"],
            report["provenance"]["hardware"]["name"],
            tuple(report["provenance"]["hardware"]["compute_capability"]),
            report["provenance"]["hardware"]["driver_version"],
            json.dumps(report["provenance"]["software"], sort_keys=True),
            (report["provenance"].get("model_config") or {}).get("sha256"),
            report["config"]["routing_distribution"],
            report["config"].get("routing_trace_manifest_sha256"),
            json.dumps(report["config"].get("routing_trace_records"), sort_keys=True),
            report["config"]["seed"],
            report["config"]["weight_data"],
            report["config"]["warmup"],
            report["config"]["repeat"],
            report["config"]["samples"],
            report["config"]["confirm_top_k"],
            report["config"]["confirm_rounds"],
            report["config"]["verify_output"],
            report["config"]["use_cuda_graph"],
        )
        for report in reports
    }
    if len(identities) != 1:
        raise ValueError("cannot merge reports with different DSO or tuning settings")
    profile_keys = [
        (profile["topology"], profile["ep_rank"], profile["num_tokens"])
        for profile in profiles
    ]
    if len(set(profile_keys)) != len(profile_keys):
        raise ValueError("cannot merge duplicate topology/ep_rank/M profiles")
    config = dict(reports[0]["config"])
    config["topologies"] = sorted(
        {topology for report in reports for topology in report["config"]["topologies"]}
    )
    config["m_values"] = sorted(
        {value for report in reports for value in report["config"]["m_values"]}
    )
    config["ep_ranks"] = sorted(
        {rank for report in reports for rank in report["config"]["ep_ranks"]}
    )
    return {
        "schema_version": 1,
        "provenance": reports[0]["provenance"],
        "merged_provenance": [report["provenance"] for report in reports],
        "merged_inputs": [str(path.resolve()) for path in paths],
        "config": config,
        "profiles": profiles,
        "tables": aggregate_tables(profiles),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tune the source-vendored TRT-LLM native V4-Pro MoE runner"
    )
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        help="Merge existing per-device reports without running CUDA",
    )
    modes.add_argument(
        "--confirm-shared-from",
        type=Path,
        help="Reprofile a shared minimax plan from an initial merged report",
    )
    modes.add_argument(
        "--finalize-shared-from",
        type=Path,
        help="Finalize rank confirmation reports on CPU",
    )
    parser.add_argument(
        "--confirmation-reports",
        type=Path,
        nargs="+",
        help="Rank reports consumed by --finalize-shared-from",
    )
    parser.add_argument(
        "--topologies", type=_parse_topologies, default=_DEFAULT_TOPOLOGIES
    )
    parser.add_argument(
        "--m-values",
        type=lambda value: _parse_csv_ints(value, name="m-values"),
        default=_DEFAULT_BUCKETS,
    )
    parser.add_argument(
        "--ep-ranks",
        type=lambda value: _parse_csv_ints(value, name="ep-ranks"),
        default=(0,),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--routing", choices=("balanced", "random", "trace"), default="balanced"
    )
    parser.add_argument(
        "--routing-trace-manifest",
        type=Path,
        help=(
            "Replay exact precomputed V4 routing tensors. Supplying this option "
            "selects trace routing; --routing trace may be used explicitly."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--confirm-top-k", type=int, default=8)
    parser.add_argument("--confirm-rounds", type=int, default=5)
    parser.add_argument("--shared-top-k", type=int, default=8)
    parser.add_argument("--shared-confirm-rounds", type=int, default=5)
    parser.add_argument("--shared-confirm-samples", type=int, default=3)
    parser.add_argument("--no-verify-output", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--weight-data", choices=("random", "zero"), default="random")
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args(argv)

    input_paths = []
    input_paths.extend(args.merge or [])
    if args.confirm_shared_from:
        input_paths.append(args.confirm_shared_from)
    if args.finalize_shared_from:
        input_paths.append(args.finalize_shared_from)
    input_paths.extend(args.confirmation_reports or [])
    if args.model_config:
        input_paths.append(args.model_config)
    if args.routing_trace_manifest:
        input_paths.append(args.routing_trace_manifest)
    output_resolved = args.output.resolve()
    conflicting_inputs = [
        path for path in input_paths if path.resolve() == output_resolved
    ]
    if conflicting_inputs:
        parser.error(
            f"--output must not overwrite an input path: {conflicting_inputs[0]}"
        )

    if args.merge:
        if args.confirmation_reports:
            parser.error("--confirmation-reports requires --finalize-shared-from")
        _write_report(args.output, _merge_reports(args.merge))
        return 0
    if args.finalize_shared_from:
        if not args.confirmation_reports:
            parser.error("--finalize-shared-from requires --confirmation-reports")
        _write_report(
            args.output,
            _finalize_shared_from_paths(
                args.finalize_shared_from, args.confirmation_reports
            ),
        )
        return 0
    if args.confirmation_reports:
        parser.error("--confirmation-reports requires --finalize-shared-from")
    if not args.confirm_shared_from and args.model_config is None:
        parser.error("phase-one tuning requires --model-config")
    if (
        not args.confirm_shared_from
        and args.routing == "trace"
        and args.routing_trace_manifest is None
    ):
        parser.error("--routing trace requires --routing-trace-manifest")
    if (
        not args.confirm_shared_from
        and args.routing_trace_manifest is not None
        and args.routing == "random"
    ):
        parser.error(
            "--routing-trace-manifest cannot be combined with --routing random"
        )
    if (
        args.warmup < 0
        or args.repeat < 1
        or args.samples < 1
        or args.confirm_top_k < 1
        or args.confirm_rounds < 0
        or args.shared_top_k < 1
        or args.shared_confirm_rounds < 1
        or args.shared_confirm_samples < 1
    ):
        parser.error(
            "warmup/confirm-rounds must be >=0 and all other sampling "
            "counts must be >=1"
        )
    if any(value not in _DEFAULT_BUCKETS for value in args.m_values):
        parser.error(f"m-values must be selected from {_DEFAULT_BUCKETS}")
    if any(rank not in _EP8_RANKS for rank in args.ep_ranks):
        parser.error(f"ep-ranks must be selected from {_EP8_RANKS}")
    if args.confirm_shared_from and len(args.ep_ranks) != 1:
        parser.error("--confirm-shared-from requires exactly one --ep-ranks value")
    if (
        not args.confirm_shared_from
        and "ep8" in args.topologies
        and len(args.ep_ranks) != 1
    ):
        parser.error(
            "tune exactly one EP rank per report, then merge reports from "
            "eight physical GPUs"
        )

    routing_trace_input = None
    if args.routing_trace_manifest is not None:
        try:
            routing_trace_input = _load_routing_trace_manifest(
                args.routing_trace_manifest,
                required_m_values=(
                    () if args.confirm_shared_from else tuple(args.m_values)
                ),
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            parser.error(str(error))
        conflicting_records = [
            record.path
            for record in routing_trace_input.records.values()
            if record.path == output_resolved
        ]
        if conflicting_records:
            parser.error(
                "--output must not overwrite a routing trace record: "
                f"{conflicting_records[0]}"
            )

    routing_trace = None
    if not args.confirm_shared_from and routing_trace_input is not None:
        routing_trace = routing_trace_input
        try:
            model_config_sha256 = _sha256_file(args.model_config.resolve())
        except OSError as error:
            parser.error(str(error))
        if routing_trace.model_config_sha256 != model_config_sha256:
            parser.error(
                "routing trace model config does not match --model-config: "
                f"{routing_trace.model_config_sha256} != {model_config_sha256}"
            )
    if not torch.cuda.is_available():
        parser.error("CUDA is required for tuning")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if torch.cuda.get_device_capability(device) != (10, 0):
        parser.error("the current offline table is scoped to SM100/B200")
    if not native_moe.has_native_mxfp4_moe():
        raise RuntimeError("the native TRT-LLM MoE DSO is not built or loadable")

    if args.confirm_shared_from:
        _run_shared_confirmation_rank(
            args.confirm_shared_from,
            args.output,
            ep_rank=int(args.ep_ranks[0]),
            device=device,
            shared_top_k=args.shared_top_k,
            confirmation_rounds=args.shared_confirm_rounds,
            confirmation_samples=args.shared_confirm_samples,
            model_config=args.model_config,
            routing_trace_manifest_path=args.routing_trace_manifest,
            argv=sys.argv if argv is None else argv,
        )
        return 0

    routing_distribution = "trace" if routing_trace is not None else args.routing
    config = {
        "topologies": list(args.topologies),
        "m_values": list(args.m_values),
        "ep_ranks": list(args.ep_ranks),
        "routing_distribution": routing_distribution,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "samples": args.samples,
        "confirm_top_k": args.confirm_top_k,
        "confirm_rounds": args.confirm_rounds,
        "verify_output": not args.no_verify_output,
        "use_cuda_graph": not args.eager,
        "ranking_statistic": "median_of_per-replay_average_ms",
        "weight_data": args.weight_data,
        "upstream_deviations": [
            "untimed graph replay replaces private delay_kernel(100us)",
            "all candidates use repeat rather than the >1ms short-profile shortcut",
            "multiple replay samples are ranked by median instead of one average",
            (
                "exact precomputed routing is replayed from one curated SWE-smith "
                "trace per M"
                if routing_distribution == "trace"
                else "synthetic routing is not a captured V4 production top-k trace"
            ),
            *(
                ["random routing uses seeded global top-k logits"]
                if routing_distribution == "random"
                else (
                    ["balanced routing is upstream's synthetic pure-EP dummy"]
                    if routing_distribution == "balanced"
                    else []
                )
            ),
        ],
    }
    if routing_trace is not None:
        config.update(_routing_trace_config(routing_trace))
    report: dict[str, object] = {
        "schema_version": 1,
        "provenance": build_provenance(
            device,
            argv=sys.argv if argv is None else argv,
            model_config=args.model_config,
        ),
        "config": config,
        "profiles": [],
        "tables": {},
    }
    profiles: list[dict[str, object]] = report["profiles"]
    for topology in args.topologies:
        ranks = (0,) if topology == "tp8" else args.ep_ranks
        for ep_rank in ranks:
            shape = v4_moe_shape(topology, ep_rank)
            print(f"allocating {topology}/ep{ep_rank} weights on {device}", flush=True)
            weights = _allocate_weights(
                shape,
                device,
                seed=args.seed + ep_rank,
                weight_data=args.weight_data,
            )
            for num_tokens in args.m_values:
                profiles.append(
                    tune_profile(
                        shape,
                        num_tokens,
                        weights,
                        device,
                        distribution=routing_distribution,
                        seed=args.seed,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        samples=args.samples,
                        confirm_top_k=args.confirm_top_k,
                        confirm_rounds=args.confirm_rounds,
                        verify_output=not args.no_verify_output,
                        use_cuda_graph=not args.eager,
                        input_generator=_input_generator_contract(
                            shape, num_tokens, config
                        ),
                        routing_record=(
                            routing_trace.records[num_tokens]
                            if routing_trace is not None
                            else None
                        ),
                    )
                )
                report["tables"] = aggregate_tables(profiles)
                _write_report(args.output, report)
            del weights
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
