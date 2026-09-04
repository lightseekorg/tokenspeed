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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _requires_gfx950() -> None:
    if torch.version.hip is None or torch.cuda.get_device_capability() != (9, 5):
        pytest.skip("requires gfx950")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gluon_block_fp8_dequantize_matches_torch() -> None:
    _requires_gfx950()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_dequantize,
    )

    generator = torch.Generator(device="cuda").manual_seed(3)
    weight = torch.randn(
        3, 256, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    ).to(torch.float8_e4m3fn)
    scale = torch.rand(3, 2, 4, device="cuda", generator=generator)

    actual = gluon_fp8_block_dequantize(weight, scale)
    expected = weight.float() * scale.repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    torch.testing.assert_close(actual.float(), expected, rtol=0.005, atol=0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gluon_block_fp8_large_batch_uses_bf16_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _requires_gfx950()
    from tokenspeed_kernel.ops.moe.gluon import bf16 as bf16_moe
    from tokenspeed_kernel.ops.moe.gluon.fp8 import (
        gluon_fp8_block_precomputed_moe_apply,
    )

    num_tokens = 4097
    weights = torch.nn.Module()
    weights.w13_weight = torch.empty(
        1, 1024, 512, device="cuda", dtype=torch.float8_e4m3fn
    )
    weights.w2_weight = torch.empty(
        1, 512, 512, device="cuda", dtype=torch.float8_e4m3fn
    )
    weights.w13_weight_scale_inv = torch.ones(1, 8, 4, device="cuda")
    weights.w2_weight_scale_inv = torch.ones(1, 4, 4, device="cuda")
    weights.w13_weight_prefill_bf16 = torch.empty(
        1, 1024, 512, device="cuda", dtype=torch.bfloat16
    )
    weights.w2_weight_prefill_bf16 = torch.empty(
        1, 512, 512, device="cuda", dtype=torch.bfloat16
    )
    weights.ep_size = 1
    weights.activation = "swiglu"
    weights.swiglu_arg = SimpleNamespace(alpha=1.0, limit=7.0)
    weights.swiglu_beta = 0.0
    weights.w13_input_layout = "concatenated"
    x = torch.empty(num_tokens, 512, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.zeros(num_tokens, 1, device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(num_tokens, 1, device="cuda")
    sentinel = torch.full_like(x, 3.0)

    def fake_bf16_apply(*args, **kwargs):
        assert args[2].w13_weight is weights.w13_weight_prefill_bf16
        assert args[2].w2_weight is weights.w2_weight_prefill_bf16
        return sentinel

    monkeypatch.setattr(bf16_moe, "gluon_bf16_precomputed_moe_apply", fake_bf16_apply)
    actual = gluon_fp8_block_precomputed_moe_apply(
        {"activation": "swiglu"},
        x,
        weights,
        torch.empty(num_tokens, 1, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    assert actual is sentinel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gluon_block_fp8_medium_batch_uses_exact_mfma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _requires_gfx950()
    from tokenspeed_kernel.ops.moe.gluon import fp8 as fp8_moe

    num_tokens = 64
    weights = torch.nn.Module()
    weights.w13_weight = torch.empty(
        1, 1024, 512, device="cuda", dtype=torch.float8_e4m3fn
    )
    weights.w2_weight = torch.empty(
        1, 512, 512, device="cuda", dtype=torch.float8_e4m3fn
    )
    weights.w13_weight_scale_inv = torch.ones(1, 8, 4, device="cuda")
    weights.w2_weight_scale_inv = torch.ones(1, 4, 4, device="cuda")
    weights.ep_size = 1
    weights.activation = "swiglu"
    weights.swiglu_arg = SimpleNamespace(alpha=1.0, limit=7.0)
    weights.swiglu_beta = 0.0
    weights.w13_input_layout = "concatenated"
    x = torch.empty(num_tokens, 512, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.zeros(num_tokens, 1, device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(num_tokens, 1, device="cuda")
    sentinel = torch.full_like(x, 3.0)

    monkeypatch.setattr(
        fp8_moe,
        "gluon_fp8_block_exact_mfma_moe",
        lambda *args, **kwargs: sentinel,
    )
    actual = fp8_moe.gluon_fp8_block_precomputed_moe_apply(
        {"activation": "swiglu"},
        x,
        weights,
        torch.empty(num_tokens, 1, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    assert actual is sentinel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("num_tokens", "expected_block_m"),
    ((16, 16), (64, 16), (128, 32), (256, 32), (512, 64)),
)
def test_gluon_block_fp8_exact_mfma_route_block_size(
    num_tokens: int, expected_block_m: int
) -> None:
    _requires_gfx950()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.exact_mfma import (
        _select_route_block_size,
    )

    assert _select_route_block_size(num_tokens, topk=8, num_experts=288) == (
        expected_block_m
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gluon_block_fp8_moe_rejects_bad_scale_shape() -> None:
    from tokenspeed_kernel.ops.moe.gluon.fp8 import _validate

    weights = torch.nn.Module()
    weights.w13_weight = torch.empty(
        2, 1024, 512, device="cuda", dtype=torch.float8_e4m3fn
    )
    weights.w2_weight = torch.empty(
        2, 512, 512, device="cuda", dtype=torch.float8_e4m3fn
    )
    weights.w13_weight_scale_inv = torch.empty(2, 1, 1, device="cuda")
    weights.w2_weight_scale_inv = torch.empty(2, 1, 1, device="cuda")
    weights.ep_size = 1
    weights.activation = "swiglu"
    weights.swiglu_arg = SimpleNamespace(alpha=1.0, limit=None)
    weights.swiglu_beta = 0.0
    weights.w13_input_layout = "concatenated"
    x = torch.empty(1, 512, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    topk_weights = torch.ones(1, 1, device="cuda")

    with pytest.raises(ValueError, match="scale tensors have incompatible shapes"):
        _validate({}, x, weights, topk_weights, topk_ids, True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 64])
def test_gluon_block_fp8_decode_matches_torch(num_tokens: int) -> None:
    _requires_gfx950()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_warp_decode_moe,
    )

    generator = torch.Generator(device="cuda").manual_seed(11)
    hidden_size = 512
    intermediate_size = 512
    num_experts = 4
    top_k = 2
    x = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w13_scale = (
        torch.rand(
            num_experts,
            2 * intermediate_size // 128,
            hidden_size // 128,
            device="cuda",
            generator=generator,
        )
        * 0.002
    )
    w2_scale = (
        torch.rand(
            num_experts,
            hidden_size // 128,
            intermediate_size // 128,
            device="cuda",
            generator=generator,
        )
        * 0.002
    )
    w13 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    topk_ids = (
        torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", generator=generator), dim=-1
    )

    actual = gluon_fp8_block_warp_decode_moe(
        x, w13, w2, w13_scale, w2_scale, topk_ids, topk_weights, 7.0
    )
    dequant_w13 = w13.float() * w13_scale.repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    dequant_w2 = w2.float() * w2_scale.repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert_id in range(num_experts):
        token_ids, slots = torch.where(topk_ids == expert_id)
        gate, up = F.linear(x[token_ids].float(), dequant_w13[expert_id]).chunk(
            2, dim=-1
        )
        intermediate = (F.silu(gate.clamp(max=7.0)) * up.clamp(-7.0, 7.0)).to(
            torch.bfloat16
        )
        expert_output = F.linear(intermediate.float(), dequant_w2[expert_id]).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output.float() * topk_weights[token_ids, slots, None],
        )
    torch.testing.assert_close(actual.float(), expected, rtol=0.08, atol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gluon_block_fp8_exact_mfma_matches_materialized_bf16() -> None:
    _requires_gfx950()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_dequantize,
        gluon_fp8_block_exact_mfma_moe,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import gluon_bf16_moe

    generator = torch.Generator(device="cuda").manual_seed(17)
    tokens, hidden, intermediate = 64, 512, 512
    experts, topk = 4, 2
    x = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    w13 = torch.randn(
        experts,
        2 * intermediate,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    w2 = torch.randn(
        experts,
        hidden,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    s13 = torch.rand(experts, 8, 4, device="cuda", generator=generator) * 0.002
    s2 = torch.rand(experts, 4, 4, device="cuda", generator=generator) * 0.002
    topk_ids = torch.randint(
        0,
        experts,
        (tokens, topk),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device="cuda", generator=generator), dim=-1
    )

    actual = gluon_fp8_block_exact_mfma_moe(x, w13, w2, s13, s2, topk_ids, topk_weights)
    expected = gluon_bf16_moe(
        x,
        gluon_fp8_block_dequantize(w13, s13),
        gluon_fp8_block_dequantize(w2, s2),
        topk_ids,
        topk_weights,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("block_n", "block_k"), ((64, 64), (64, 128), (128, 64), (128, 128))
)
def test_gluon_block_fp8_stage2_aligned_scale_matches_materialized_bf16(
    block_n: int, block_k: int
) -> None:
    """Aligned scale tiles preserve exact BF16 stage-2 arithmetic."""
    _requires_gfx950()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_dequantize,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage2_kernel import invoke_stage2

    generator = torch.Generator(device="cuda").manual_seed(23)
    tokens, hidden, intermediate = 64, 512, 512
    experts, topk, block_m = 4, 2, 16
    inter = torch.randn(
        tokens * topk,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        experts,
        hidden,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    scale = (
        torch.rand(
            experts,
            hidden // 128,
            intermediate // 128,
            device="cuda",
            generator=generator,
        )
        * 0.002
    )
    topk_ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
        .reshape(tokens, topk)
        .remainder(experts)
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device="cuda", generator=generator), dim=-1
    )
    sorted_ids, sorted_experts, sorted_weights, num_valid = moe_align_block_size_device(
        topk_ids, topk_weights, experts, block_m=block_m
    )
    actual = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(actual)

    invoke_stage2(
        inter,
        weight,
        sorted_ids,
        sorted_experts,
        sorted_weights,
        num_valid,
        actual,
        topk,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        atomic=False,
        w2_scale=scale,
    )
    invoke_stage2(
        inter,
        gluon_fp8_block_dequantize(weight, scale),
        sorted_ids,
        sorted_experts,
        sorted_weights,
        num_valid,
        expected,
        topk,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        atomic=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gluon_block_fp8_decode_ep_filters_nonlocal_routes() -> None:
    """EP decode remaps local experts and leaves remote routes for all-reduce."""
    _requires_gfx950()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_warp_decode_moe,
    )

    generator = torch.Generator(device="cuda").manual_seed(19)
    num_tokens = 8
    hidden_size = intermediate_size = 512
    num_local_experts = 2
    expert_start = 4
    top_k = 2
    x = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w13_scale = (
        torch.rand(num_local_experts, 8, 4, device="cuda", generator=generator) * 0.002
    )
    w2_scale = (
        torch.rand(num_local_experts, 4, 4, device="cuda", generator=generator) * 0.002
    )
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    w2 = torch.randn(
        num_local_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    topk_ids = torch.tensor(
        [[4, 0], [1, 5], [4, 5], [7, 2], [5, 3], [4, 6], [0, 1], [5, 4]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", generator=generator), dim=-1
    )

    actual = gluon_fp8_block_warp_decode_moe(
        x,
        w13,
        w2,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        7.0,
        expert_start,
        True,
    )
    dequant_w13 = w13.float() * w13_scale.repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    dequant_w2 = w2.float() * w2_scale.repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    expected = torch.zeros_like(x, dtype=torch.float32)
    for local_expert in range(num_local_experts):
        global_expert = expert_start + local_expert
        token_ids, slots = torch.where(topk_ids == global_expert)
        gate, up = F.linear(x[token_ids].float(), dequant_w13[local_expert]).chunk(
            2, dim=-1
        )
        intermediate = (F.silu(gate.clamp(max=7.0)) * up.clamp(-7.0, 7.0)).to(
            torch.bfloat16
        )
        expert_output = F.linear(intermediate.float(), dequant_w2[local_expert]).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output.float() * topk_weights[token_ids, slots, None],
        )
    torch.testing.assert_close(actual.float(), expected, rtol=0.08, atol=0.04)
