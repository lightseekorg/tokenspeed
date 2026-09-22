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
import tokenspeed_kernel.benchmark.generators.kda as kda_generator
import torch
from tokenspeed_kernel.benchmark.generators.kda import prepare_kda_paged_prefill
from tokenspeed_kernel.benchmark.harness import BenchmarkCaseError, BenchmarkRequest
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelSpec


def test_kda_generator_rejects_unsupported_validation_before_allocating_inputs(
    mi350_platform: PlatformInfo,
) -> None:
    request = BenchmarkRequest(
        family="attention",
        mode="kda_paged_prefill",
        parameters={
            "model_profile": "glm53_flash_tp4",
            "batch": 1,
            "tokens_per_sequence": 64,
            "validation": {"runs": 1},
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )

    with pytest.raises(BenchmarkCaseError, match="not implemented"):
        prepare_kda_paged_prefill(request, mi350_platform)


def test_kda_generator_rejects_unimplemented_model_profile(
    mi350_platform: PlatformInfo,
) -> None:
    request = BenchmarkRequest(
        family="attention",
        mode="kda_paged_prefill",
        parameters={
            "model_profile": "unimplemented",
            "batch": 1,
            "tokens_per_sequence": 64,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )

    with pytest.raises(BenchmarkCaseError, match="Implemented KDA model_profile"):
        prepare_kda_paged_prefill(request, mi350_platform)


def test_kda_generator_builds_glm_decode_input_strides(monkeypatch) -> None:
    def cpu_randn(shape, *, dtype, generator):
        _ = generator
        return torch.randn(shape, dtype=dtype)

    monkeypatch.setattr(kda_generator, "_randn", cpu_randn)
    q, k, v = kda_generator._packed_decode_qkv(
        4,
        16,
        128,
        128,
        dtype=torch.bfloat16,
        generator=None,
    )
    beta = kda_generator._packed_beta_logits(
        4,
        16,
        128,
        128,
        dtype=torch.bfloat16,
        generator=None,
    )

    assert q.stride() == (24576, 6144, 128, 1)
    assert k.stride() == q.stride()
    assert v.stride() == q.stride()
    assert beta.stride() == (25664, 6416, 1)


def test_kda_generator_builds_glm_prefill_input_layouts(monkeypatch) -> None:
    def cpu_randn(shape, *, dtype, generator):
        _ = generator
        return torch.randn(shape, dtype=dtype)

    monkeypatch.setattr(kda_generator, "_randn", cpu_randn)
    q, k, v, g_raw, beta = kda_generator._packed_prefill_inputs(
        4,
        16,
        128,
        128,
        dtype=torch.bfloat16,
        generator=None,
    )

    assert q.stride() == (8192, 2048, 128, 1)
    assert k.stride() == q.stride()
    assert v.stride() == q.stride()
    assert g_raw.stride() == q.stride()
    assert beta.stride() == (25664, 6416, 1)


def test_kda_generator_builds_int64_prefill_boundaries() -> None:
    device_boundaries, host_boundaries = kda_generator._cu_seqlens(
        4,
        64,
        device="cpu",
    )

    assert device_boundaries.tolist() == [0, 64, 128, 192, 256]
    assert device_boundaries.dtype == torch.int64
    assert host_boundaries.dtype == torch.int64


def test_kda_prefill_binds_chunk_hint_to_converted_boundaries(
    monkeypatch,
    mi350_platform: PlatformInfo,
) -> None:
    from tokenspeed_kernel.ops.attention import kda as kda_ops
    from tokenspeed_kernel.ops.attention.gdn import triton as gdn_triton

    original_randn = torch.randn

    def cpu_randn(*size, **kwargs):
        kwargs.pop("device", None)
        return original_randn(*size, **kwargs)

    source_boundaries = torch.tensor([0, 64], dtype=torch.int64)
    host_boundaries = source_boundaries.clone()
    seen: dict[str, object] = {}

    def record_hint(_batch, _tokens, boundaries, _chunk_sizes):
        seen["hint"] = boundaries

    def record_operation(
        *_args,
        cu_seqlens,
        capacity,
        inputs_packed,
        **_kwargs,
    ):
        seen["operation"] = cu_seqlens
        seen["capacity"] = capacity
        seen["inputs_packed"] = inputs_packed
        return object()

    monkeypatch.setattr(kda_generator.torch, "randn", cpu_randn)
    monkeypatch.setattr(
        kda_generator,
        "_generator",
        lambda _seed: torch.Generator(device="cpu"),
    )
    monkeypatch.setattr(
        kda_generator,
        "_cu_seqlens",
        lambda _batch, _tokens: (source_boundaries, host_boundaries),
    )
    monkeypatch.setattr(kda_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(
        kda_generator,
        "_select_registration",
        lambda _request, _platform, traits: KernelSpec(
            name="test_kda_prefill",
            family="attention",
            mode="kda_paged_prefill",
            solution="test",
        ),
    )
    monkeypatch.setattr(gdn_triton, "set_total_chunks_hint_uniform", record_hint)
    monkeypatch.setattr(kda_ops, "kda_paged_prefill", record_operation)

    prepared = prepare_kda_paged_prefill(
        BenchmarkRequest(
            family="attention",
            mode="kda_paged_prefill",
            parameters={
                "model_profile": "glm53_flash_tp4",
                "heads": 16,
                "key_dim": 128,
                "value_dim": 128,
                "dtype": "bfloat16",
                "lower_bound": -5.0,
                "recurrent_layout": "v_major",
                "batch": 1,
                "tokens_per_sequence": 64,
            },
            solution=None,
            registration=None,
            cold_cache=True,
            seed=42,
        ),
        mi350_platform,
    )
    prepared.invocation.invoke()

    assert source_boundaries.dtype == torch.int64
    operation_boundaries = seen["operation"]
    assert isinstance(operation_boundaries, torch.Tensor)
    assert operation_boundaries.dtype == torch.int32
    assert seen["hint"] is seen["operation"]
    assert seen["capacity"] is None
    assert seen["inputs_packed"] is False


def test_kda_generator_builds_flat_decay_bias() -> None:
    generator = torch.Generator(device="cpu")
    a_log, dt_bias = kda_generator._decay_parameters(
        16,
        128,
        generator=generator,
        device="cpu",
    )

    assert a_log.shape == (16,)
    assert dt_bias.shape == (2048,)


def test_kda_generator_builds_arena_strided_state_pool() -> None:
    state_pool = kda_generator._strided_state_pool(
        3,
        2,
        4,
        4,
        48,
        device="cpu",
    )

    assert state_pool.shape == (3, 2, 4, 4)
    assert state_pool.stride() == (48, 16, 4, 1)
    assert state_pool.is_contiguous() is False


def test_kda_generator_builds_in_place_and_boundary_state_indices() -> None:
    read_indices, in_place = kda_generator._state_page_indices(
        4,
        "in_place",
        device="cpu",
    )
    _, distinct = kda_generator._state_page_indices(
        4,
        "distinct",
        device="cpu",
    )

    assert read_indices.tolist() == [1, 2, 3, 4]
    assert in_place.tolist() == [1, 2, 3, 4]
    assert distinct.tolist() == [17, 18, 19, 20]
