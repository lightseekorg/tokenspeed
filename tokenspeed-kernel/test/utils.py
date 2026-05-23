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

from __future__ import annotations

from typing import Callable

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import FormatSignature, format_signatures

SampleRegistration = tuple[dict, Callable]


def dummy_impl(name: str) -> Callable:
    def impl(*args, **kwargs):
        return name

    impl.__name__ = name
    return impl


def _sample_registration(
    name: str,
    family: str,
    mode: str,
    solution: str,
    signatures: frozenset[FormatSignature],
    *,
    features: frozenset[str] | None = None,
    capability: CapabilityRequirement | None = None,
    priority: int = 10,
    tags: frozenset[str] | None = None,
) -> SampleRegistration:
    return (
        {
            "family": family,
            "mode": mode,
            "name": name,
            "solution": solution,
            "features": features,
            "capability": capability,
            "signatures": signatures,
            "priority": priority,
            "tags": tags,
        },
        dummy_impl(name),
    )


def make_sample_specs() -> dict[str, SampleRegistration]:
    return {
        "flashinfer_decode": _sample_registration(
            "flashinfer_decode",
            "attention",
            "decode",
            "flashinfer",
            format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            features=frozenset({"paged"}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=18,
            tags=frozenset({"latency"}),
        ),
        "triton_decode": _sample_registration(
            "triton_decode",
            "attention",
            "decode",
            "triton",
            format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            features=frozenset({"paged"}),
            priority=10,
            tags=frozenset({"portability"}),
        ),
        "cutlass_prefill": _sample_registration(
            "cutlass_prefill",
            "attention",
            "prefill",
            "cutlass",
            format_signatures(
                ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=16,
            tags=frozenset({"throughput"}),
        ),
        "reference_decode": _sample_registration(
            "reference_decode",
            "attention",
            "decode",
            "reference",
            format_signatures(
                ("q", "k_cache", "v_cache"),
                "dense",
                {torch.float16, torch.bfloat16, torch.float32},
            ),
            features=frozenset({"paged"}),
            capability=CapabilityRequirement(),
            priority=10,
            tags=frozenset({"determinism", "portability"}),
        ),
        "aiter_decode": _sample_registration(
            "aiter_decode",
            "attention",
            "decode",
            "aiter",
            format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            features=frozenset({"paged"}),
            capability=CapabilityRequirement(vendors=frozenset({"amd"})),
            priority=16,
            tags=frozenset({"latency", "portability"}),
        ),
        "cutlass_gemm": _sample_registration(
            "cutlass_gemm",
            "gemm",
            "mm",
            "cutlass",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=15,
            tags=frozenset({"throughput", "latency"}),
        ),
        "triton_gemm": _sample_registration(
            "triton_gemm",
            "gemm",
            "mm",
            "triton",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            priority=10,
            tags=frozenset({"portability"}),
        ),
        "cutlass_grouped_gemm": _sample_registration(
            "cutlass_grouped_gemm",
            "gemm",
            "grouped_mm",
            "cutlass",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=16,
            tags=frozenset({"throughput"}),
        ),
        "triton_grouped_gemm": _sample_registration(
            "triton_grouped_gemm",
            "gemm",
            "grouped_mm",
            "triton",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            priority=10,
            tags=frozenset({"portability"}),
        ),
        "triton_fused_moe": _sample_registration(
            "triton_fused_moe",
            "moe",
            "fused",
            "triton",
            format_signatures(
                ("x", "weight"), "dense", {torch.float16, torch.bfloat16}
            ),
            priority=12,
            tags=frozenset({"throughput", "portability"}),
        ),
        "cutlass_fused_moe": _sample_registration(
            "cutlass_fused_moe",
            "moe",
            "fused",
            "cutlass",
            format_signatures(
                ("x", "weight"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=15,
            tags=frozenset({"latency", "throughput"}),
        ),
        "triton_modular_moe": _sample_registration(
            "triton_modular_moe",
            "moe",
            "modular",
            "triton",
            format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
            priority=10,
            tags=frozenset({"determinism", "portability"}),
        ),
        "cutlass_modular_moe": _sample_registration(
            "cutlass_modular_moe",
            "moe",
            "modular",
            "cutlass",
            format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=14,
            tags=frozenset({"throughput"}),
        ),
    }


def register_all_samples(
    registry: KernelRegistry, samples: dict[str, SampleRegistration]
) -> None:
    if registry is not KernelRegistry.get():
        raise ValueError("sample registrations must target the active KernelRegistry")
    for options, impl in samples.values():
        register_kernel(**options)(impl)
