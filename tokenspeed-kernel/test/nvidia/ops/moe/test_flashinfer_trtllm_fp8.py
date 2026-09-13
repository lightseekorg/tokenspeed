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

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import current_platform


def _runtime_reason() -> str | None:
    if find_spec("flashinfer") is None:
        return "requires flashinfer"
    if not current_platform().is_nvidia:
        return "TRT-LLM MoE is registered only on NVIDIA"
    return None


_reason = _runtime_reason()
requires_flashinfer = pytest.mark.skipif(_reason is not None, reason=str(_reason))


class _MoEWeights(torch.nn.Module):
    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        self.w13_weight_scale_inv = torch.nn.Parameter(w13_scale, requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(w2_scale, requires_grad=False)


def _fp8_arange(shape: tuple[int, ...]) -> torch.Tensor:
    size = 1
    for dim in shape:
        size *= dim
    return (
        torch.arange(size, dtype=torch.int64)
        .remainder(120)
        .to(torch.uint8)
        .reshape(shape)
        .view(torch.float8_e4m3fn)
    )


@requires_flashinfer
def test_weight_preprocessor_uses_shuffled_block_major_k() -> None:
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import convert_to_block_layout
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_fp8 import (
        flashinfer_trtllm_fp8_moe_process_weights,
    )

    num_experts = 2
    hidden = intermediate = 128
    raw_w13 = _fp8_arange((num_experts, 2 * intermediate, hidden))
    raw_w2 = _fp8_arange((num_experts, hidden, intermediate))
    raw_s13 = torch.arange(
        num_experts * 2 * intermediate // 128 * hidden // 128,
        dtype=torch.float32,
    ).reshape(num_experts, 2 * intermediate // 128, hidden // 128)
    raw_s2 = torch.zeros(
        num_experts, hidden // 128, intermediate // 128, dtype=torch.float32
    )

    logical_w13 = torch.cat(
        (raw_w13[:, intermediate:], raw_w13[:, :intermediate]), dim=1
    )
    expected_w13 = torch.stack(
        [
            convert_to_block_layout(shuffle_matrix_a(expert.view(torch.uint8), 64), 128)
            for expert in logical_w13
        ]
    ).view(torch.float8_e4m3fn)
    expected_w2 = torch.stack(
        [
            convert_to_block_layout(shuffle_matrix_a(expert.view(torch.uint8), 64), 128)
            for expert in raw_w2
        ]
    ).view(torch.float8_e4m3fn)
    expected_s13 = torch.cat((raw_s13[:, 1:], raw_s13[:, :1]), dim=1).clamp(min=1e-10)

    weights = _MoEWeights(
        raw_w13.clone(), raw_w2.clone(), raw_s13.clone(), raw_s2.clone()
    )
    weights.swiglu_arg = SimpleNamespace(alpha=1.25, limit=10.0)
    weights.swiglu_beta = 0.5
    flashinfer_trtllm_fp8_moe_process_weights({}, weights)

    assert weights.intermediate_size_per_partition == intermediate
    assert weights.w13_weight.shape == (num_experts, 1, 2 * intermediate, 128)
    assert weights.w2_weight.shape == (num_experts, 1, hidden, 128)
    assert torch.equal(
        weights.w13_weight.view(torch.uint8), expected_w13.view(torch.uint8)
    )
    assert torch.equal(
        weights.w2_weight.view(torch.uint8), expected_w2.view(torch.uint8)
    )
    torch.testing.assert_close(weights.w13_weight_scale_inv, expected_s13)
    torch.testing.assert_close(weights.w2_weight_scale_inv, raw_s2.clamp(min=1e-10))
    torch.testing.assert_close(
        weights.gemm1_alpha,
        torch.full((num_experts,), 1.25, dtype=torch.float32),
    )
    torch.testing.assert_close(
        weights.gemm1_beta,
        torch.full((num_experts,), 0.5, dtype=torch.float32),
    )
    torch.testing.assert_close(
        weights.gemm1_clamp_limit,
        torch.full((num_experts,), 10.0, dtype=torch.float32),
    )


@requires_flashinfer
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_apply_forwards_prepared_layout_and_pdl(monkeypatch, enable_pdl: bool) -> None:
    from flashinfer.tllm_enums import WeightLayout
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_fp8

    captured: dict[str, object] = {}

    def fake_quantize(x, *args, **kwargs):
        del args, kwargs
        return (
            torch.zeros_like(x, dtype=torch.float8_e4m3fn),
            torch.ones((x.shape[1] // 128, x.shape[0]), dtype=torch.float32),
        )

    def fake_moe(**kwargs):
        captured.update(kwargs)
        return torch.zeros(
            kwargs["hidden_states"].shape,
            dtype=torch.bfloat16,
            device=kwargs["hidden_states"].device,
        )

    monkeypatch.setattr(trtllm_fp8, "per_token_group_quant_fp8", fake_quantize)
    monkeypatch.setattr(trtllm_fp8, "trtllm_fp8_block_scale_routed_moe", fake_moe)

    weights = _MoEWeights(
        torch.empty((2, 1, 256, 128), dtype=torch.float8_e4m3fn),
        torch.empty((2, 1, 128, 128), dtype=torch.float8_e4m3fn),
        torch.ones((2, 2, 1), dtype=torch.float32),
        torch.ones((2, 1, 1), dtype=torch.float32),
    )
    weights.intermediate_size_per_partition = 128
    weights.num_experts = 2
    weights.num_local_experts = 2
    weights.top_k = 1
    weights.ep_rank = 0
    weights.gemm1_alpha = torch.nn.Parameter(
        torch.full((2,), 1.25, dtype=torch.float32), requires_grad=False
    )
    weights.gemm1_beta = torch.nn.Parameter(
        torch.full((2,), 0.5, dtype=torch.float32), requires_grad=False
    )
    weights.gemm1_clamp_limit = torch.nn.Parameter(
        torch.full((2,), 10.0, dtype=torch.float32), requires_grad=False
    )

    trtllm_fp8.flashinfer_trtllm_fp8_moe_apply(
        {},
        torch.zeros((2, 128), dtype=torch.bfloat16),
        weights,
        router_logits=torch.empty((2, 2), dtype=torch.float32),
        topk_weights=torch.ones((2, 1), dtype=torch.bfloat16),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        enable_pdl=enable_pdl,
    )

    assert captured["intermediate_size"] == 128
    assert captured["use_shuffled_weight"] is True
    assert captured["weight_layout"] == int(WeightLayout.BlockMajorK)
    assert captured["enable_pdl"] is enable_pdl
    assert captured["do_finalize"] is True
    assert captured["gemm1_alpha"] is weights.gemm1_alpha
    assert captured["gemm1_beta"] is weights.gemm1_beta
    assert captured["gemm1_clamp_limit"] is weights.gemm1_clamp_limit


@requires_flashinfer
def test_apply_supports_deferred_finalize(monkeypatch) -> None:
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_fp8

    captured: dict[str, object] = {}

    def fake_quantize(x, *args, **kwargs):
        del args, kwargs
        return (
            torch.zeros_like(x, dtype=torch.float8_e4m3fn),
            torch.ones((x.shape[1] // 128, x.shape[0]), dtype=torch.float32),
        )

    gemm2_out = torch.randn((4, 128), dtype=torch.bfloat16)
    expanded_idx = torch.arange(2, dtype=torch.int32)

    def fake_moe(**kwargs):
        captured.update(kwargs)
        return gemm2_out, kwargs["topk_ids"][1], expanded_idx

    monkeypatch.setattr(trtllm_fp8, "per_token_group_quant_fp8", fake_quantize)
    monkeypatch.setattr(trtllm_fp8, "trtllm_fp8_block_scale_routed_moe", fake_moe)

    weights = _MoEWeights(
        torch.empty((2, 1, 256, 128), dtype=torch.float8_e4m3fn),
        torch.empty((2, 1, 128, 128), dtype=torch.float8_e4m3fn),
        torch.ones((2, 2, 1), dtype=torch.float32),
        torch.ones((2, 1, 1), dtype=torch.float32),
    )
    weights.intermediate_size_per_partition = 128
    weights.num_experts = 2
    weights.num_local_experts = 2
    weights.top_k = 1
    weights.ep_rank = 0
    topk_weights = torch.ones((2, 1), dtype=torch.float32)

    result = trtllm_fp8.flashinfer_trtllm_fp8_moe_apply(
        {},
        torch.zeros((2, 128), dtype=torch.bfloat16),
        weights,
        router_logits=torch.empty((2, 2), dtype=torch.float32),
        topk_weights=topk_weights,
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        do_finalize=False,
        enable_pdl=True,
    )

    assert captured["do_finalize"] is False
    assert result[0] is gemm2_out
    assert result[1] is topk_weights
    assert result[2] is expanded_idx


@requires_flashinfer
def test_empty_apply_preserves_deferred_finalize_contract() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer import trtllm_fp8

    weights = _MoEWeights(
        torch.empty((2, 1, 256, 128), dtype=torch.float8_e4m3fn),
        torch.empty((2, 1, 128, 128), dtype=torch.float8_e4m3fn),
        torch.ones((2, 2, 1), dtype=torch.float32),
        torch.ones((2, 1, 1), dtype=torch.float32),
    )
    weights.top_k = 1
    topk_weights = torch.empty((0, 1), dtype=torch.float32)

    gemm2_out, expert_weights, expanded_idx = (
        trtllm_fp8.flashinfer_trtllm_fp8_moe_apply(
            {},
            torch.empty((0, 128), dtype=torch.bfloat16),
            weights,
            router_logits=torch.empty((0, 2), dtype=torch.float32),
            topk_weights=topk_weights,
            topk_ids=torch.empty((0, 1), dtype=torch.int32),
            do_finalize=False,
        )
    )

    assert gemm2_out.shape == (0, 128)
    assert expert_weights is topk_weights
    assert expanded_idx.shape == (0,)
    assert expanded_idx.dtype == torch.int32
