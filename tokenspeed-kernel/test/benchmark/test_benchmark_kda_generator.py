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
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
)
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

    def record_hint(_batch, _tokens, boundaries):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    ("mode", "window", "history_length"),
    [
        ("kda_fused_paged_verify", 1, 0),
        ("kda_fused_paged_verify", 4, 8),
        ("kda_retained_commit", 1, 0),
        ("kda_retained_commit", 4, 0),
    ],
)
def test_retained_benchmark_repetitions_restore_mutable_state(
    monkeypatch, mode, window, history_length
):
    from tokenspeed_kernel.platform import current_platform

    parameters = dict(
        model_profile="kimi_k3_tp8",
        heads=12,
        key_dim=128,
        value_dim=128,
        dtype="bfloat16",
        lower_bound=-5.0,
        recurrent_layout="v_major",
        batch=1,
        draft_token_num=window,
        conv_kernel_size=4,
        state_pages=3,
        state_page_bytes=884736,
    )
    if mode == "kda_retained_commit":
        parameters.update(accepted_length=window, layers=2, state_groups=1)
    else:
        parameters["history_length"] = history_length
    request = BenchmarkRequest(
        family="attention",
        mode=mode,
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=False,
        seed=47,
    )
    layers = []
    original_prepare = kda_generator._retained_benchmark_layer

    def capture_layer(*args):
        layer = original_prepare(*args)
        layers.append(layer)
        return layer

    monkeypatch.setattr(kda_generator, "_retained_benchmark_layer", capture_layer)
    prepare = (
        kda_generator.prepare_kda_fused_paged_verify
        if mode == "kda_fused_paged_verify"
        else kda_generator.prepare_kda_retained_commit
    )
    prepared = prepare(request, current_platform())
    assert prepared.invocation.reset is not None
    pages = [(layer["conv"].clone(), layer["state"].clone()) for layer in layers]
    snapshots = None
    for _ in range(3):
        prepared.invocation.reset()
        output = prepared.invocation.invoke()
        current = []
        if isinstance(output, torch.Tensor):
            current.append(output)
        for layer in layers:
            history = layer["records"].history
            current.extend(
                [
                    layer["state"],
                    layer["conv"],
                    layer["replay"],
                    layer["records"].preconv.rows,
                    history.length,
                    history.correction,
                    history.key,
                    history.decay,
                ]
            )
            expected_length = (
                window if mode == "kda_retained_commit" else history_length
            )
            slots = history.record_slots.long()
            # The commit advances the count and ring state every layer
            # shares, past the window that started at ring row 3; the verify
            # reads the round's start.
            assert layer["replay"][slots].tolist() == [
                [expected_length, 1 if mode == "kda_retained_commit" else 0]
            ]
            assert history.length[slots, 0].tolist() == [
                0 if mode == "kda_retained_commit" else history_length
            ]
            if history_length:
                assert torch.count_nonzero(
                    history.correction[history.record_slots.long(), :, :history_length]
                )
        if snapshots is None:
            snapshots = [value.clone() for value in current]
        else:
            for value, saved in zip(current, snapshots):
                torch.testing.assert_close(value, saved, atol=0, rtol=0)
    if mode == "kda_retained_commit":
        # No request ends: the ring keeps the convolution inputs, so the
        # commit writes no page and runs as a single launch.
        for layer, (conv, state) in zip(layers, pages):
            torch.testing.assert_close(layer["conv"], conv, atol=0, rtol=0)
            torch.testing.assert_close(layer["state"], state, atol=0, rtol=0)
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile

        prepared.invocation.reset()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            prepared.invocation.invoke()
            torch.cuda.synchronize()
        launched = [
            event.name
            for event in trace.events()
            if event.device_type == DeviceType.CUDA
        ]
        assert len(launched) == 1 and "retained_batched_commit" in launched[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("batch", [1, 17])
def test_retained_verify_benchmark_keeps_the_raw_qkv_rows(monkeypatch, batch):
    """Each producer arrangement leaves the window's pre-convolution rows.

    One request fuses the producers into the window kernel; 17 requests run
    the split producers, whose convolution keeps the rows as serving does.
    """
    from tokenspeed_kernel.platform import current_platform

    request = BenchmarkRequest(
        family="attention",
        mode="kda_fused_paged_verify",
        parameters=dict(
            model_profile="kimi_k3_tp8",
            heads=12,
            key_dim=128,
            value_dim=128,
            dtype="bfloat16",
            lower_bound=-5.0,
            recurrent_layout="v_major",
            batch=batch,
            draft_token_num=4,
            conv_kernel_size=4,
            state_pages=batch + 2,
            state_page_bytes=884736,
            history_length=4,
        ),
        solution=None,
        registration=None,
        cold_cache=False,
        seed=47,
    )
    layers = []
    original_prepare = kda_generator._retained_benchmark_layer

    def capture_layer(*args):
        layer = original_prepare(*args)
        layers.append(layer)
        return layer

    monkeypatch.setattr(kda_generator, "_retained_benchmark_layer", capture_layer)
    prepared = kda_generator.prepare_kda_fused_paged_verify(request, current_platform())
    (layer,) = layers
    # Each request's ring holds its page's taps in rows 0..2 and takes its
    # window's raw rows from row 3.
    ring = layer["records"].preconv.rows
    ring[:, 3:].fill_(float("nan"))
    prepared.invocation.reset()
    prepared.invocation.invoke()
    torch.testing.assert_close(
        ring[:, 3:],
        layer["projection"].mixed_qkv.view(batch, 4, -1),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        ring[:, :3].transpose(1, 2),
        layer["conv"][layer["reads"].long()],
        atol=0,
        rtol=0,
    )


def test_history_capacity_defaults_to_16_and_validates_against_window():
    assert kda_generator._history_capacity({}, 1) == 16
    assert kda_generator._history_capacity({}, 16) == 16
    assert kda_generator._history_capacity({"history_capacity": None}, 4) == 16
    assert kda_generator._history_capacity({"history_capacity": 20}, 20) == 20
    assert kda_generator._history_capacity({"history_capacity": 8}, 4) == 8

    for parameters, window in [
        ({}, 20),
        ({"history_capacity": None}, 20),
        ({"history_capacity": 3}, 4),
        ({"history_capacity": True}, 1),
        ({"history_capacity": "16"}, 1),
    ]:
        with pytest.raises(BenchmarkCaseError) as exc_info:
            kda_generator._history_capacity(parameters, window)
        assert exc_info.value.status == BenchmarkStatus.INVALID_CASE
