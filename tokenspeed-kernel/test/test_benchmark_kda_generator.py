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

from functools import partial

import pytest
import tokenspeed_kernel.benchmark.generators.kda as kda_generator
import torch
from tokenspeed_kernel.benchmark.generators.kda import prepare_kda_paged_prefill
from tokenspeed_kernel.benchmark.harness import BenchmarkCaseError, BenchmarkRequest
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


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
        kda_generator._MODEL_PROFILES["glm53_flash_tp4"],
        dtype=torch.bfloat16,
        generator=None,
    )

    assert q.stride() == (24576, 6144, 128, 1)
    assert k.stride() == q.stride()
    assert v.stride() == q.stride()
    assert beta.stride() == (25664, 6416, 1)


@pytest.mark.parametrize(
    ("model_profile", "head_stride", "projection_width"),
    [("glm53_flash_tp4", 2048, 6416), ("kimi_k3_tp8", 1536, 6288)],
)
def test_kda_generator_builds_prefill_input_layouts(
    monkeypatch, model_profile, head_stride, projection_width
) -> None:
    def cpu_randn(shape, *, dtype, generator):
        _ = generator
        return torch.randn(shape, dtype=dtype)

    monkeypatch.setattr(kda_generator, "_randn", cpu_randn)
    q, k, v, g_raw, beta = kda_generator._packed_prefill_inputs(
        4,
        kda_generator._MODEL_PROFILES[model_profile],
        dtype=torch.bfloat16,
        generator=None,
    )

    assert q.stride() == (4 * head_stride, head_stride, 128, 1)
    assert k.stride() == q.stride()
    assert v.stride() == q.stride()
    assert g_raw.stride() == q.stride()
    assert beta.stride() == (4 * projection_width, projection_width, 1)


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
        **_kwargs,
    ):
        seen["operation"] = cu_seqlens
        seen["capacity"] = capacity
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


def test_kda_generator_builds_kimi_k3_cache_arena_state_pages() -> None:
    # One 884736-byte page of the TP8 cache arena: the FP32 recurrent state
    # followed by the BF16 convolution history.
    conv_pool, state_pool = kda_generator._state_arena(
        3,
        884736,
        kda_generator._MODEL_PROFILES["kimi_k3_tp8"],
        4,
        device="cpu",
    )

    assert conv_pool.shape == (3, 4608, 3)
    assert conv_pool.stride() == (442368, 3, 1)
    assert state_pool.shape == (3, 12, 128, 128)
    assert state_pool.stride() == (221184, 16384, 128, 1)
    assert conv_pool.data_ptr() - state_pool.data_ptr() == 12 * 128 * 128 * 4


@pytest.fixture
def cpu_fused_kda_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def cpu_factory(factory, *args, **kwargs):
        kwargs["device"] = "cpu"
        return factory(*args, **kwargs)

    for name in ("arange", "empty", "full", "ones", "randn", "tensor", "zeros"):
        monkeypatch.setattr(torch, name, partial(cpu_factory, getattr(torch, name)))
    monkeypatch.setattr(
        kda_generator, "_generator", lambda seed: torch.Generator().manual_seed(seed)
    )
    monkeypatch.setattr(kda_generator, "load_builtin_kernels", lambda: None)


@pytest.mark.usefixtures("cpu_fused_kda_inputs", "fresh_registry")
@pytest.mark.parametrize(
    ("mode", "capture_replay"),
    [
        ("kda_fused_paged_decode", False),
        ("kda_fused_paged_verify", False),
        ("kda_fused_paged_verify", True),
        ("kda_replay_commit", False),
    ],
)
def test_fused_kda_benchmarks_dispatch_with_their_fp32_arena(
    monkeypatch: pytest.MonkeyPatch,
    mi350_platform: PlatformInfo,
    mode: str,
    capture_replay: bool,
) -> None:
    from tokenspeed_kernel.ops.attention import kda as kda_ops

    seen: dict[str, object] = {}

    def record_kernel(**kwargs):
        seen["invocation"] = kwargs
        return object()

    signature = format_signature(
        **{role: dense_tensor_format(torch.bfloat16) for role in ("q", "k", "v")}
    )
    for state_dtype, priority in ((torch.float32, 10), (torch.bfloat16, 11)):
        KernelRegistry.get().register(
            KernelSpec(
                name=f"test_{mode}_{state_dtype}",
                family="attention",
                mode=mode,
                solution="test",
                format_signatures=frozenset({signature}),
                traits={
                    "fused_output_norm": frozenset({True}),
                    "state_dtype": frozenset({state_dtype}),
                },
                priority=priority,
            ),
            record_kernel,
        )

    def record_selection(stage, *args, traits, **kwargs):
        seen[stage] = traits
        kwargs["platform"] = mi350_platform
        return select_kernel(*args, traits=traits, **kwargs)

    monkeypatch.setattr(
        kda_generator, "select_kernel", partial(record_selection, "benchmark")
    )
    monkeypatch.setattr(
        kda_ops, "select_kernel", partial(record_selection, "operation")
    )
    request = BenchmarkRequest(
        family="attention",
        mode=mode,
        parameters={
            "model_profile": "kimi_k3_tp8",
            "heads": 12,
            "key_dim": 128,
            "value_dim": 128,
            "dtype": "bfloat16",
            "lower_bound": -5.0,
            "recurrent_layout": "v_major",
            "batch": 1,
            "conv_kernel_size": 4,
            "state_pages": 3,
            "state_page_bytes": 884736,
            "state_page_relation": "in_place",
            "norm_eps": 1e-5,
            "draft_token_num": 4,
            "capture_replay": capture_replay,
            "store_states": False,
            "accepted_length": 2,
            "layers": 2,
            "state_groups": 1,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )
    prepare = {
        "kda_fused_paged_decode": kda_generator.prepare_kda_fused_paged_decode,
        "kda_fused_paged_verify": kda_generator.prepare_kda_fused_paged_verify,
        "kda_replay_commit": kda_generator.prepare_kda_replay_commit,
    }[mode]
    prepared = prepare(request, mi350_platform)
    prepared.invocation.invoke()

    assert prepared.registration.traits["state_dtype"] == frozenset({torch.float32})
    invocation = seen["invocation"]
    if mode == "kda_replay_commit":
        assert invocation["state_dtype"] == torch.float32
        assert invocation["state_stride"] == request.parameters["state_page_bytes"] // 4
        assert invocation["descriptors"].shape == (2, 10)
    else:
        assert seen["benchmark"] == seen["operation"]
        assert invocation["state_pool"].dtype == torch.float32
        if mode == "kda_fused_paged_verify":
            assert ("replay_gate" in invocation) is capture_replay
            if capture_replay:
                assert invocation["replay_gate"].shape == (4, 12 * 128)
