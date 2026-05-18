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

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import (
    KernelRegistry,
    KernelSpec,
    describe_kernel,
    register_kernel,
)
from utils import dummy_impl, register_all_samples

pytestmark = pytest.mark.usefixtures("fresh_registry")


class TestKernelSpec:
    def test_frozen_dataclass(self):
        spec = KernelSpec(name="k1", family="attention", mode="decode")
        with pytest.raises(AttributeError):
            spec.name = "k2"

    def test_default_values(self):
        spec = KernelSpec(name="k1", family="attention", mode="decode")
        assert spec.features == frozenset()
        assert spec.solution == ""
        assert spec.priority == 10
        assert spec.tags == frozenset()
        assert spec.dtypes == frozenset()

    def test_hashable_without_dict_traits(self):
        spec = KernelSpec(name="k1", family="attention", mode="decode", traits={})
        with pytest.raises(TypeError):
            hash(spec)

    def test_equality(self):
        spec1 = KernelSpec(name="k1", family="attention", mode="decode")
        spec2 = KernelSpec(name="k1", family="attention", mode="decode")
        assert spec1 == spec2


class TestRegistrySingleton:
    def test_get_returns_same_instance(self):
        r1 = KernelRegistry.get()
        r2 = KernelRegistry.get()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        r1 = KernelRegistry.get()
        KernelRegistry.reset()
        r2 = KernelRegistry.get()
        assert r1 is not r2


class TestRegistryRegister:
    def test_register_and_retrieve(self):
        reg = KernelRegistry.get()
        spec = KernelSpec(name="test_k", family="attention", mode="decode")
        impl = dummy_impl("test_k")
        reg.register(spec, impl)

        assert reg.get_by_name("test_k") is spec
        assert reg.get_impl("test_k") is impl

    def test_register_multiple_kernels(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        assert reg.get_by_name("flashinfer_decode") is not None
        assert reg.get_by_name("triton_decode") is not None
        assert reg.get_by_name("cutlass_prefill") is not None
        assert reg.get_by_name("nonexistent") is None

    def test_reregister_replaces_old(self):
        reg = KernelRegistry.get()
        spec1 = KernelSpec(name="k", family="attention", mode="decode", priority=5)
        spec2 = KernelSpec(name="k", family="attention", mode="decode", priority=15)
        impl1 = dummy_impl("old")
        impl2 = dummy_impl("new")

        reg.register(spec1, impl1)
        reg.register(spec2, impl2)

        assert reg.get_by_name("k") is spec2
        assert reg.get_impl("k") is impl2
        assert len(reg.get_for_operator("attention", "decode")) == 1

    def test_sorted_by_priority_descending(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        decode_specs = reg.get_for_operator("attention", "decode")
        priorities = [s.priority for s in decode_specs]
        assert priorities == sorted(priorities, reverse=True)


class TestRegistryQueries:
    def test_get_for_operator_basic(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        decode = reg.get_for_operator("attention", "decode")
        assert len(decode) >= 3
        for s in decode:
            assert s.family == "attention"
            assert s.mode == "decode"

    def test_get_for_operator_empty(self):
        reg = KernelRegistry.get()
        assert reg.get_for_operator("nonexistent", "op") == []

    def test_filter_by_features(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        paged = reg.get_for_operator(
            "attention", "decode", features=frozenset({"paged"})
        )
        for s in paged:
            assert "paged" in s.features

    def test_filter_by_platform(
        self, sample_specs, h100_platform, mi300_platform, mi350_platform
    ):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        nvidia_kernels = reg.get_for_operator(
            "attention", "decode", platform=h100_platform
        )
        nvidia_names = {s.name for s in nvidia_kernels}
        assert "aiter_decode" not in nvidia_names
        assert "flashinfer_decode" in nvidia_names

        amd_kernels = reg.get_for_operator(
            "attention", "decode", platform=mi300_platform
        )
        amd_names = {s.name for s in amd_kernels}
        assert "flashinfer_decode" not in amd_names
        assert "aiter_decode" in amd_names

        mi350_kernels = reg.get_for_operator(
            "attention", "decode", platform=mi350_platform
        )
        mi350_names = {s.name for s in mi350_kernels}
        assert "flashinfer_decode" not in mi350_names
        assert "aiter_decode" in mi350_names
        assert "triton_decode" in mi350_names

    def test_filter_by_dtype(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        fp32 = reg.get_for_operator("attention", "decode", dtype=torch.float32)
        names = {s.name for s in fp32}
        assert "reference_decode" in names
        assert "flashinfer_decode" not in names

    def test_filter_by_tags(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        latency = reg.get_for_operator("attention", "decode", tags={"latency"})
        for s in latency:
            assert "latency" in s.tags

    def test_filter_by_solution(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        triton = reg.get_for_operator("attention", "decode", solution="triton")
        assert all(s.solution == "triton" for s in triton)
        assert len(triton) == 1

    def test_list_operators(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        ops = reg.list_operators()
        assert ("attention", "decode") in ops
        assert ("attention", "prefill") in ops
        assert ("gemm", "mm") in ops

    def test_list_kernels_all(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        all_kernels = reg.list_kernels()
        assert len(all_kernels) == len(sample_specs)

    def test_list_kernels_by_family(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        attn = reg.list_kernels(family="attention")
        assert all(s.family == "attention" for s in attn)

    def test_list_kernels_by_mode(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        decode = reg.list_kernels(mode="decode")
        assert all(s.mode == "decode" for s in decode)

    def test_list_kernels_by_family_and_mode(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        decode = reg.list_kernels(family="attention", mode="decode")
        assert all(s.family == "attention" and s.mode == "decode" for s in decode)

    def test_list_solutions(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        solutions = reg.list_solutions("attention", "decode")
        assert "flashinfer" in solutions
        assert "triton" in solutions
        assert "reference" in solutions


class TestRegistryCache:
    def test_cache_put_and_get(self):
        reg = KernelRegistry.get()
        key = ("attention", "decode", torch.bfloat16, "sm_90")
        impl = dummy_impl("cached")

        assert reg.cache_get(key) is None
        reg.cache_put(key, impl)
        assert reg.cache_get(key) is impl

    def test_clear_cache(self):
        reg = KernelRegistry.get()
        key = ("attention", "decode", torch.bfloat16, "sm_90")
        reg.cache_put(key, dummy_impl("cached"))

        reg.clear_cache()
        assert reg.cache_get(key) is None

    def test_invalidate_cache_on_register(self):
        reg = KernelRegistry.get()
        key = ("attention", "decode", torch.bfloat16, "sm_90")
        reg.cache_put(key, dummy_impl("cached"))

        spec = KernelSpec(name="new_k", family="attention", mode="decode")
        reg.register(spec, dummy_impl("new_k"))

        assert reg.cache_get(key) is None

    def test_invalidate_preserves_other_ops(self):
        reg = KernelRegistry.get()
        attn_key = ("attention", "decode", torch.bfloat16, "sm_90")
        gemm_key = ("gemm", "mm", torch.bfloat16, "sm_90")
        reg.cache_put(attn_key, dummy_impl("attn"))
        reg.cache_put(gemm_key, dummy_impl("gemm"))

        spec = KernelSpec(name="new_attn", family="attention", mode="decode")
        reg.register(spec, dummy_impl("new_attn"))

        assert reg.cache_get(attn_key) is None
        assert reg.cache_get(gemm_key) is not None


class TestRegisterKernelDecorator:
    def test_basic_decorator(self):
        @register_kernel(
            "gemm",
            "mm",
            solution="reference",
            dtypes={torch.bfloat16},
            priority=12,
        )
        def my_torch_gemm(a, b):
            return a @ b

        reg = KernelRegistry.get()
        spec = reg.get_by_name("reference_gemm_mm")
        assert spec is not None
        assert spec.solution == "reference"
        assert spec.priority == 12
        assert torch.bfloat16 in spec.dtypes

        impl = reg.get_impl("reference_gemm_mm")
        assert impl is my_torch_gemm

    def test_custom_name(self):
        @register_kernel(
            "attention",
            "decode",
            name="my_custom_kernel",
            solution="custom",
            dtypes={torch.float16},
        )
        def some_func():
            pass

        reg = KernelRegistry.get()
        assert reg.get_by_name("my_custom_kernel") is not None

    def test_decorator_with_features_and_tags(self):
        @register_kernel(
            "attention",
            "decode",
            features={"paged", "rope"},
            solution="triton",
            capability=CapabilityRequirement(
                min_arch_version=ArchVersion(8, 0),
            ),
            dtypes={torch.float16, torch.bfloat16},
            tags={"determinism", "latency"},
        )
        def decorated_kernel():
            pass

        reg = KernelRegistry.get()
        spec = reg.get_by_name("triton_attention_decode")
        assert spec is not None
        assert spec.features == frozenset({"paged", "rope"})
        assert spec.tags == frozenset({"determinism", "latency"})
        assert spec.capability.min_arch_version == ArchVersion(8, 0)

    def test_decorator_returns_original_function(self):
        @register_kernel(
            "gemm",
            "mm",
            solution="test",
            dtypes={torch.float16},
        )
        def original(x):
            return x * 2

        assert original(5) == 10


class TestDescribeKernel:
    def test_describe_existing(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        desc = describe_kernel("flashinfer_decode")
        assert "flashinfer_decode" in desc
        assert "attention" in desc
        assert "flashinfer" in desc

    def test_describe_not_found(self):
        desc = describe_kernel("nonexistent_kernel")
        assert "not found" in desc.lower()


class TestUnregister:
    def test_unregister_removes_from_all_lookups(self, sample_specs):
        reg = KernelRegistry.get()
        register_all_samples(reg, sample_specs)

        assert reg.get_by_name("triton_decode") is not None
        reg._unregister("triton_decode")

        assert reg.get_by_name("triton_decode") is None
        assert reg.get_impl("triton_decode") is None
        names = {s.name for s in reg.get_for_operator("attention", "decode")}
        assert "triton_decode" not in names

    def test_unregister_nonexistent_is_noop(self):
        reg = KernelRegistry.get()
        reg._unregister("does_not_exist")
