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

import sys
from types import SimpleNamespace

import pytest
import tokenspeed_kernel
import tokenspeed_kernel.ops.gemm.deep_gemm as deep_gemm_backend
import tokenspeed_kernel.ops.gemm.deepseek_v4 as deepseek_v4_gemm
import torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel


def _portable_plan() -> object:
    return tokenspeed_kernel.deepseek_v4_grouped_output_projection_plan(
        input_dtype=torch.bfloat16,
        weight_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=torch.float32,
        num_groups=2,
        heads_per_group=1,
        head_dim=128,
        nope_dim=64,
        rope_dim=64,
        output_dim=128,
        block_size=(128, 128),
        scale_format="ue8m0",
        solution="triton",
    )


def test_public_plan_pins_portable_preprocessing_and_execution(
    monkeypatch: pytest.MonkeyPatch,
    mi350_platform,
) -> None:
    original_platform = Platform.get()
    calls = []
    sentinel = object()

    def fake_call(self, *args, **kwargs):
        calls.append((self.name, kwargs))
        return sentinel

    try:
        Platform.override(mi350_platform)
        plan = _portable_plan()
        weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
        scales = torch.ones((2, 1), dtype=torch.float32)
        assert (
            tokenspeed_kernel.deepseek_v4_grouped_output_projection_process_weights(
                plan, weight, scales
            )
            is scales
        )

        monkeypatch.setattr(SelectedKernel, "__call__", fake_call)
        result = tokenspeed_kernel.deepseek_v4_grouped_output_projection(
            plan,
            torch.empty((3, 2, 128), dtype=torch.bfloat16),
            torch.arange(3),
            torch.empty((3, 64), dtype=torch.float32),
            weight,
            scales,
        )
    finally:
        Platform.override(original_platform)

    assert result is sentinel
    assert calls[0][0] == "triton_deepseek_v4_grouped_output_projection"
    assert calls[0][1]["tma_aligned_scales"] is False
    assert calls[0][1]["recipe"] == (1, 128, 128)


@pytest.mark.parametrize(
    ("head_dim", "nope_dim", "rope_dim"),
    [(192, 128, 64), (256, 64, 192), (384, 320, 64)],
)
def test_public_plan_rejects_unsupported_per_head_quantization_geometry(
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
) -> None:
    with pytest.raises(ValueError):
        tokenspeed_kernel.deepseek_v4_grouped_output_projection_plan(
            input_dtype=torch.bfloat16,
            weight_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=torch.float32,
            num_groups=2,
            heads_per_group=2,
            head_dim=head_dim,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            output_dim=128,
            block_size=(128, 128),
            scale_format="ue8m0",
            solution="triton",
        )


def test_deep_gemm_plan_pins_hopper_and_blackwell_scale_layouts(
    monkeypatch: pytest.MonkeyPatch,
    h100_platform,
    b200_platform,
) -> None:
    if (
        KernelRegistry.get().get_by_name(
            "deep_gemm_deepseek_v4_grouped_output_projection"
        )
        is None
    ):
        pytest.skip("DeepGEMM grouped output projection is unavailable")

    original_platform = Platform.get()
    calls = []

    def fake_call(self, *args, **kwargs):
        calls.append(kwargs)
        return torch.empty((2, 2, 128), dtype=torch.bfloat16)

    monkeypatch.setattr(SelectedKernel, "__call__", fake_call)
    try:
        for platform in (h100_platform, b200_platform):
            Platform.override(platform)
            plan = tokenspeed_kernel.deepseek_v4_grouped_output_projection_plan(
                input_dtype=torch.bfloat16,
                weight_dtype=torch.float8_e4m3fn,
                weight_scale_dtype=torch.float32,
                num_groups=2,
                heads_per_group=1,
                head_dim=128,
                nope_dim=64,
                rope_dim=64,
                output_dim=128,
                block_size=(128, 128),
                scale_format="ue8m0",
                solution="deep_gemm",
            )
            tokenspeed_kernel.deepseek_v4_grouped_output_projection(
                plan,
                torch.empty((2, 2, 128), dtype=torch.bfloat16),
                torch.arange(2),
                torch.empty((2, 64)),
                torch.empty((256, 128), dtype=torch.float8_e4m3fn),
                torch.empty((2, 1)),
            )
    finally:
        Platform.override(original_platform)

    assert calls[0]["tma_aligned_scales"] is False
    assert calls[0]["recipe"] == (1, 128, 128)
    assert calls[1]["tma_aligned_scales"] is True
    assert calls[1]["recipe"] == (1, 1, 128)


def test_deep_gemm_plan_owns_grouped_scale_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
    b200_platform,
) -> None:
    if (
        KernelRegistry.get().get_by_name(
            "deep_gemm_deepseek_v4_grouped_output_projection"
        )
        is None
    ):
        pytest.skip("DeepGEMM grouped output projection is unavailable")

    original_platform = Platform.get()
    transformed = torch.empty((11,), dtype=torch.int32)
    calls = []

    def ceil_to_ue8m0(scales):
        calls.append(("ceil", scales))
        return scales

    def transform(**kwargs):
        calls.append(("transform", kwargs))
        return transformed

    monkeypatch.setattr(deep_gemm_backend, "ceil_to_ue8m0", ceil_to_ue8m0)
    monkeypatch.setattr(
        deep_gemm_backend, "transform_sf_into_required_layout", transform
    )
    try:
        Platform.override(b200_platform)
        plan = tokenspeed_kernel.deepseek_v4_grouped_output_projection_plan(
            input_dtype=torch.bfloat16,
            weight_dtype=torch.float8_e4m3fn,
            weight_scale_dtype=torch.float32,
            num_groups=2,
            heads_per_group=1,
            head_dim=128,
            nope_dim=64,
            rope_dim=64,
            output_dim=128,
            block_size=(128, 128),
            scale_format="ue8m0",
            solution="deep_gemm",
        )
        result = (
            tokenspeed_kernel.deepseek_v4_grouped_output_projection_process_weights(
                plan,
                torch.empty((256, 128), dtype=torch.float8_e4m3fn),
                torch.ones((2, 1)),
            )
        )
    finally:
        Platform.override(original_platform)

    assert result is transformed
    assert calls[0][0] == "ceil"
    assert calls[1][1]["num_groups"] == 2
    assert calls[1][1]["recipe"] == (1, 128, 128)


@pytest.mark.parametrize(
    ("tma_aligned", "expected_dtype"),
    [(False, torch.float32), (True, torch.int32)],
)
def test_deep_gemm_warmup_matches_activation_scale_layout(
    monkeypatch: pytest.MonkeyPatch,
    tma_aligned: bool,
    expected_dtype: torch.dtype,
) -> None:
    calls = []

    def fake_einsum(_expr, activation, _weight, _output, *, recipe):
        calls.append((activation[1], recipe))

    monkeypatch.setattr(deep_gemm_backend, "fp8_einsum", fake_einsum)
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel.thirdparty.deep_gemm.warmup",
        SimpleNamespace(_warmup_m_values=lambda _max: [3]),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    deep_gemm_backend._warmup_deep_gemm_deepseek_v4_grouped_output_projection(
        weight=torch.empty((256, 128), dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones((2, 1)),
        num_groups=2,
        output_dim=128,
        input_dim=128,
        block_size=(128, 128),
        tma_aligned_scales=tma_aligned,
        recipe=(1, 1, 128) if tma_aligned else (1, 128, 128),
        max_tokens=3,
    )

    scales, _ = calls[0]
    assert scales.dtype == expected_dtype
    assert scales.shape == (3, 2, 1)
    assert scales.stride() == (1, 4, 4)


def test_grouped_projection_model_warmup_deduplicates_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _portable_plan()
    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    for layer in layers:
        layer._deepseek_v4_grouped_output_projection_plan = plan
        layer.weight_scale_inv = torch.nn.Parameter(torch.ones((2, 1)))
    calls = []
    monkeypatch.setattr(
        deepseek_v4_gemm,
        "deepseek_v4_grouped_output_projection_warmup",
        lambda *args: calls.append(args),
    )

    deepseek_v4_gemm.deepseek_v4_grouped_output_projection_warmup_model(
        layers, max_tokens=17
    )

    assert len(calls) == 1
    assert calls[0][0] is plan
    assert calls[0][-1] == 17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_portable_projection_matches_grouped_reference() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    plan = _portable_plan()
    tokens, groups, head_dim, output_dim = 5, 2, 128, 128
    attention = torch.randn(
        (tokens, groups, head_dim), device=device, dtype=torch.bfloat16
    )
    positions = torch.arange(tokens, device=device)
    cos_sin_cache = torch.cat(
        (
            torch.ones((tokens, 32), device=device),
            torch.zeros((tokens, 32), device=device),
        ),
        dim=-1,
    )
    source_weight = torch.randn(
        (groups, output_dim, head_dim), device=device, dtype=torch.float32
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scales = source_weight.abs().amax(dim=(1, 2)).clamp_min(1e-10) / fp8_max
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    weight = (
        (source_weight / scales[:, None, None])
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    flat_weight = weight.flatten(0, 1)
    flat_scales = scales[:, None]
    prepared_scales = (
        tokenspeed_kernel.deepseek_v4_grouped_output_projection_process_weights(
            plan, flat_weight, flat_scales
        )
    )

    actual = tokenspeed_kernel.deepseek_v4_grouped_output_projection(
        plan,
        attention,
        positions,
        cos_sin_cache,
        flat_weight,
        prepared_scales,
    )
    dequantized_weight = weight.float() * scales[:, None, None]
    expected = torch.einsum("tgd,gnd->tgn", attention.float(), dequantized_weight)

    assert actual.shape == expected.shape
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.flatten(), dim=0
    )
    assert cosine > 0.99
