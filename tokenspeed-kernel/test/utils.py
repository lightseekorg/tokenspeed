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
from tokenspeed_kernel.registry import (
    KernelRegistry,
    KernelSpec,
)
from tokenspeed_kernel.signature import format_signatures


def dummy_impl(name: str) -> Callable:
    def impl(*args, **kwargs):
        return name

    impl.__name__ = name
    return impl


def make_sample_specs() -> dict[str, tuple[KernelSpec, Callable]]:
    specs: dict[str, tuple[KernelSpec, Callable]] = {}

    specs["flashinfer_decode"] = (
        KernelSpec(
            name="flashinfer_decode",
            family="attention",
            mode="decode",
            solution="flashinfer",
            features=frozenset({"paged"}),
            format_signatures=format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=18,
            tags=frozenset({"latency"}),
        ),
        dummy_impl("flashinfer_decode"),
    )

    specs["triton_decode"] = (
        KernelSpec(
            name="triton_decode",
            family="attention",
            mode="decode",
            solution="triton",
            features=frozenset({"paged"}),
            format_signatures=format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            priority=10,
            tags=frozenset({"portability"}),
        ),
        dummy_impl("triton_decode"),
    )

    specs["cutlass_prefill"] = (
        KernelSpec(
            name="cutlass_prefill",
            family="attention",
            mode="prefill",
            solution="cutlass",
            format_signatures=format_signatures(
                ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=16,
            tags=frozenset({"throughput"}),
        ),
        dummy_impl("cutlass_prefill"),
    )

    specs["reference_decode"] = (
        KernelSpec(
            name="reference_decode",
            family="attention",
            mode="decode",
            solution="reference",
            features=frozenset({"paged"}),
            format_signatures=format_signatures(
                ("q", "k_cache", "v_cache"),
                "dense",
                {torch.float16, torch.bfloat16, torch.float32},
            ),
            capability=CapabilityRequirement(),
            priority=10,
            tags=frozenset({"determinism", "portability"}),
        ),
        dummy_impl("reference_decode"),
    )

    specs["aiter_decode"] = (
        KernelSpec(
            name="aiter_decode",
            family="attention",
            mode="decode",
            solution="aiter",
            features=frozenset({"paged"}),
            format_signatures=format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"amd"}),
            ),
            priority=16,
            tags=frozenset({"latency", "portability"}),
        ),
        dummy_impl("aiter_decode"),
    )

    specs["cutlass_gemm"] = (
        KernelSpec(
            name="cutlass_gemm",
            family="gemm",
            mode="mm",
            solution="cutlass",
            format_signatures=format_signatures(
                ("a", "b"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=15,
            tags=frozenset({"throughput", "latency"}),
        ),
        dummy_impl("cutlass_gemm"),
    )

    specs["triton_gemm"] = (
        KernelSpec(
            name="triton_gemm",
            family="gemm",
            mode="mm",
            solution="triton",
            format_signatures=format_signatures(
                ("a", "b"), "dense", {torch.float16, torch.bfloat16}
            ),
            priority=10,
            tags=frozenset({"portability"}),
        ),
        dummy_impl("triton_gemm"),
    )

    specs["cutlass_grouped_gemm"] = (
        KernelSpec(
            name="cutlass_grouped_gemm",
            family="gemm",
            mode="grouped_mm",
            solution="cutlass",
            format_signatures=format_signatures(
                ("a", "b"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=16,
            tags=frozenset({"throughput"}),
        ),
        dummy_impl("cutlass_grouped_gemm"),
    )

    specs["triton_grouped_gemm"] = (
        KernelSpec(
            name="triton_grouped_gemm",
            family="gemm",
            mode="grouped_mm",
            solution="triton",
            format_signatures=format_signatures(
                ("a", "b"), "dense", {torch.float16, torch.bfloat16}
            ),
            priority=10,
            tags=frozenset({"portability"}),
        ),
        dummy_impl("triton_grouped_gemm"),
    )

    specs["triton_fused_moe"] = (
        KernelSpec(
            name="triton_fused_moe",
            family="moe",
            mode="fused",
            solution="triton",
            format_signatures=format_signatures(
                ("x", "weight"), "dense", {torch.float16, torch.bfloat16}
            ),
            priority=12,
            tags=frozenset({"throughput", "portability"}),
        ),
        dummy_impl("triton_fused_moe"),
    )

    specs["cutlass_fused_moe"] = (
        KernelSpec(
            name="cutlass_fused_moe",
            family="moe",
            mode="fused",
            solution="cutlass",
            format_signatures=format_signatures(
                ("x", "weight"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=15,
            tags=frozenset({"latency", "throughput"}),
        ),
        dummy_impl("cutlass_fused_moe"),
    )

    specs["triton_modular_moe"] = (
        KernelSpec(
            name="triton_modular_moe",
            family="moe",
            mode="modular",
            solution="triton",
            format_signatures=format_signatures(
                "x", "dense", {torch.float16, torch.bfloat16}
            ),
            priority=10,
            tags=frozenset({"determinism", "portability"}),
        ),
        dummy_impl("triton_modular_moe"),
    )

    specs["cutlass_modular_moe"] = (
        KernelSpec(
            name="cutlass_modular_moe",
            family="moe",
            mode="modular",
            solution="cutlass",
            format_signatures=format_signatures(
                "x", "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=14,
            tags=frozenset({"throughput"}),
        ),
        dummy_impl("cutlass_modular_moe"),
    )

    return specs


def register_all_samples(registry: KernelRegistry, specs: dict) -> None:
    for spec, impl in specs.values():
        registry.register(spec, impl)
