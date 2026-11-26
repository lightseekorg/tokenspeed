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
from utils import is_cdna4

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="requires AMD gfx950 GPU")


def test_gluon_block_fp8_dequantize_matches_torch() -> None:
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


def test_gluon_block_fp8_large_batch_uses_bf16_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed_kernel.ops.moe.gluon as gluon_moe
    from tokenspeed_kernel.ops.moe.gluon import (
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

    monkeypatch.setattr(gluon_moe, "gluon_bf16_precomputed_moe_apply", fake_bf16_apply)
    actual = gluon_fp8_block_precomputed_moe_apply(
        {"activation": "swiglu"},
        x,
        weights,
        torch.empty(num_tokens, 1, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    assert actual is sentinel


def test_gluon_block_fp8_medium_batch_uses_exact_mfma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed_kernel.ops.moe.gluon as gluon_moe

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
        gluon_moe,
        "gluon_fp8_block_exact_mfma_moe",
        lambda *args, **kwargs: sentinel,
    )
    actual = gluon_moe.gluon_fp8_block_precomputed_moe_apply(
        {"activation": "swiglu"},
        x,
        weights,
        torch.empty(num_tokens, 1, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    assert actual is sentinel


@pytest.mark.parametrize(
    ("num_tokens", "expected_block_m"),
    ((16, 16), (64, 16), (128, 32), (256, 32), (512, 64)),
)
def test_gluon_block_fp8_exact_mfma_route_block_size(
    num_tokens: int, expected_block_m: int
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.exact_mfma import (
        _select_route_block_size,
    )

    assert _select_route_block_size(num_tokens, topk=8, num_experts=288) == (
        expected_block_m
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 64])
@pytest.mark.parametrize("packed_weights", [False, True], ids=["linear", "packed"])
def test_gluon_block_fp8_decode_matches_torch(
    num_tokens: int, packed_weights: bool
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_warp_decode_moe,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
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
        x,
        pack_fp8_weight_n16k32(w13) if packed_weights else w13,
        pack_fp8_weight_n16k32(w2) if packed_weights else w2,
        w13_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        7.0,
        packed_weights=packed_weights,
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


def test_gluon_block_fp8_exact_mfma_matches_materialized_bf16() -> None:
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


@pytest.mark.parametrize(
    ("block_n", "block_k"), ((64, 64), (64, 128), (128, 64), (128, 128))
)
def test_gluon_block_fp8_stage2_aligned_scale_matches_materialized_bf16(
    block_n: int, block_k: int
) -> None:
    """Aligned scale tiles preserve exact BF16 stage-2 arithmetic."""
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


def test_gluon_block_fp8_decode_ep_filters_nonlocal_routes() -> None:
    """EP decode remaps local experts and leaves remote routes for all-reduce."""
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
        packed_weights=False,
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


@pytest.mark.parametrize("packed_weights", [False, True], ids=["linear", "packed"])
@pytest.mark.parametrize(
    ("rows", "ep_size", "rank", "intermediate"),
    [(1, 1, 0, 512), (4, 4, 3, 2048), (3, 4, 1, 1536)],
)
def test_gluon_block_fp8_warp_routed_reference(
    rows: int,
    ep_size: int,
    rank: int,
    intermediate: int,
    packed_weights: bool,
    record_property,
) -> None:
    from test_gluon_fp8_packed_small import (
        _assert_numerics,
        _fixture,
        _reference,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_warp_decode_moe,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )

    x, w13, w2, s13, s2, ids, weights = _fixture(
        rows, ep_size, rank, True, 4096, intermediate
    )
    expert_start = rank * w13.shape[0]
    expected = _reference(
        x,
        w13,
        w2,
        s13,
        s2,
        ids,
        weights,
        expert_start,
        7.0,
        quantized_activations=False,
    )[0]
    actual = gluon_fp8_block_warp_decode_moe(
        x,
        pack_fp8_weight_n16k32(w13) if packed_weights else w13,
        pack_fp8_weight_n16k32(w2) if packed_weights else w2,
        s13,
        s2,
        ids,
        weights,
        7.0,
        expert_start,
        ep_size > 1,
        packed_weights=packed_weights,
    )
    _assert_numerics(actual, expected, record_property)
    if ep_size > 1:
        ids.fill_(-1)
        empty = gluon_fp8_block_warp_decode_moe(
            x,
            pack_fp8_weight_n16k32(w13) if packed_weights else w13,
            pack_fp8_weight_n16k32(w2) if packed_weights else w2,
            s13,
            s2,
            ids,
            weights,
            7.0,
            expert_start,
            True,
            packed_weights=packed_weights,
        )
        assert not empty.any()


def _block_fp8_module(
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
    ep_size: int,
    ep_rank: int,
) -> torch.nn.Module:
    from tokenspeed_kernel.ops.moe.gluon import gluon_fp8_moe_weights

    module = torch.nn.Module()
    module.w13_weight = w13.contiguous()
    module.w2_weight = w2.contiguous()
    module.w13_weight_scale_inv = s13.contiguous()
    module.w2_weight_scale_inv = s2.contiguous()
    module.ep_size = ep_size
    module.ep_rank = ep_rank
    module.num_local_experts = w13.shape[0]
    module.activation = "swiglu"
    module.swiglu_arg = SimpleNamespace(alpha=1.0, limit=7.0)
    module.swiglu_beta = 0.0
    module.w13_input_layout = "concatenated"
    gluon_fp8_moe_weights({"activation": "swiglu"}, module)
    return module


@pytest.mark.parametrize(
    "num_tokens", [8, 64, 4097], ids=["warp-decode", "exact-mfma", "bf16-prefill"]
)
@pytest.mark.parametrize(
    ("num_experts", "hidden", "intermediate"),
    [(16, 512, 512), (288, 4096, 2048)],
    ids=["small", "glm53-flash"],
)
def test_gluon_block_fp8_apply_ep8_ranks_match_reference(
    num_tokens: int,
    num_experts: int,
    hidden: int,
    intermediate: int,
    record_property,
) -> None:
    """EP8 on one GPU: each of eight ranks holds its slice of the expert bank.

    Each rank's output matches the reference over its local experts, and the
    eight partial outputs (what the EP all-reduce sums) match the reference
    over the whole bank, on all three paths of the block-FP8 apply: warp decode
    (<= 32 tokens), exact MFMA (<= 4096) and the BF16 prefill copies (> 4096).
    The second shape is GLM-5.3 Flash at MoE TP1 (288 experts, 36 per rank,
    hidden 4096, intermediate 2048, top-8).
    """
    from test_gluon_fp8_packed_small import _assert_numerics, _fixture, _reference
    from tokenspeed_kernel.ops.moe.gluon import gluon_fp8_block_precomputed_moe_apply

    ep_size = 8
    x, w13, w2, s13, s2, ids, weights = _fixture(
        num_tokens, 1, 0, False, hidden, intermediate, local_experts=num_experts
    )
    local = num_experts // ep_size
    total = torch.zeros(num_tokens, hidden, device="cuda", dtype=torch.float32)
    for rank in range(ep_size):
        experts = slice(rank * local, (rank + 1) * local)
        module = _block_fp8_module(
            w13[experts], w2[experts], s13[experts], s2[experts], ep_size, rank
        )
        actual = gluon_fp8_block_precomputed_moe_apply(
            {"activation": "swiglu"},
            x,
            module,
            torch.empty(num_tokens, num_experts, device="cuda"),
            topk_weights=weights,
            topk_ids=ids,
        )
        expected = _reference(
            x,
            w13[experts],
            w2[experts],
            s13[experts],
            s2[experts],
            ids,
            weights,
            rank * local,
            7.0,
            quantized_activations=False,
        )[0]
        if expected.float().norm() > 0:
            _assert_numerics(actual, expected, record_property)
        else:
            assert not actual.any()
        total += actual.float()
        del module
    full = _reference(
        x, w13, w2, s13, s2, ids, weights, 0, 7.0, quantized_activations=False
    )[0]
    _assert_numerics(total.to(torch.bfloat16), full, record_property)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("stress", ["normal", "outlier", "cancellation"])
def test_gluon_block_fp8_packed_warp_stress(
    rows: int, stress: str, record_property
) -> None:
    from test_gluon_fp8_packed_small import (
        _assert_numerics,
        _fixture,
        _reference,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        gluon_fp8_block_warp_decode_moe,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )

    x, w13, w2, s13, s2, ids, weights = _fixture(
        rows, 1, 0, stress == "outlier", 4096, 512
    )
    if stress == "cancellation":
        x[:, 1::2] = x[:, ::2]
        x[:, 0] += 0.125
        w13[:, :, 1::2] = (-w13[:, :, ::2].float()).to(w13.dtype)
    expected = _reference(
        x,
        w13,
        w2,
        s13,
        s2,
        ids,
        weights,
        0,
        7.0,
        quantized_activations=False,
    )[0]
    raw = gluon_fp8_block_warp_decode_moe(
        x, w13, w2, s13, s2, ids, weights, 7.0, packed_weights=False
    )
    actual = gluon_fp8_block_warp_decode_moe(
        x,
        pack_fp8_weight_n16k32(w13),
        pack_fp8_weight_n16k32(w2),
        s13,
        s2,
        ids,
        weights,
        7.0,
        packed_weights=True,
    )
    _assert_numerics(actual, expected, record_property)
    delta = (actual.float() - raw.float()).flatten()
    record_property("versus_raw_max_abs", delta.abs().max().item())
    record_property(
        "versus_raw_relative_l2",
        (delta.norm() / raw.float().norm().clamp_min(1.0e-12)).item(),
    )


# GLM-5.3 Flash EP8, one rank (the last: global experts 252..287 of 288).
_SERVED_EP_SIZE = 8
_SERVED_RANK = 7
_SERVED_EXPERTS = 288
_SERVED_HIDDEN = 4096
_SERVED_INTERMEDIATE = 2048
_SERVED_REFERENCE_SLICE = 8192


@pytest.fixture(scope="module")
def _served_ep8_rank():
    from test_gluon_fp8_packed_small import _fixture

    local = _SERVED_EXPERTS // _SERVED_EP_SIZE
    _, w13, w2, s13, s2, _, _ = _fixture(
        1, 1, 0, False, _SERVED_HIDDEN, _SERVED_INTERMEDIATE, local_experts=local
    )
    module = _block_fp8_module(w13, w2, s13, s2, _SERVED_EP_SIZE, _SERVED_RANK)
    yield module, (w13, w2, s13, s2)
    del module
    torch.cuda.empty_cache()


def _served_routes(num_tokens: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Random top-8 over all 288 experts; the first and last 64 tokens route
    every slot to a local expert, so the (token, slot) rows at both ends of the
    intermediate and partial buffers are written and read."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (
        torch.randn(num_tokens, _SERVED_HIDDEN, device="cuda", generator=generator)
        * 0.5
    ).to(torch.bfloat16)
    scores = torch.rand(
        num_tokens, _SERVED_EXPERTS, device="cuda", generator=generator
    )
    ids = scores.topk(8, dim=-1).indices.to(torch.int32)
    local = _SERVED_EXPERTS // _SERVED_EP_SIZE
    start = _SERVED_RANK * local
    edge = torch.arange(8, device="cuda", dtype=torch.int32)
    for rows in (slice(0, 64), slice(num_tokens - 64, num_tokens)):
        count = ids[rows].shape[0]
        offsets = torch.arange(count, device="cuda", dtype=torch.int32)[:, None]
        ids[rows] = start + (offsets + edge * 4) % local
    weights = torch.softmax(
        torch.randn(num_tokens, 8, device="cuda", generator=generator), dim=-1
    )
    return x, ids, weights


def _apply(module, x, ids, weights) -> torch.Tensor:
    from tokenspeed_kernel.ops.moe.gluon import gluon_fp8_block_precomputed_moe_apply

    return gluon_fp8_block_precomputed_moe_apply(
        {"activation": "swiglu"},
        x,
        module,
        torch.empty(x.shape[0], _SERVED_EXPERTS, device="cuda"),
        topk_weights=weights,
        topk_ids=ids,
    )


def _time_ms(fn, iterations: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def _decode_timings(module) -> dict[str, float]:
    timings = {}
    for rows in (4, 64):
        x, ids, weights = _served_routes(rows, seed=rows)
        timings[f"eager_{rows}"] = _time_ms(lambda: _apply(module, x, ids, weights))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _apply(module, x, ids, weights)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _apply(module, x, ids, weights)
        timings[f"graph_{rows}"] = _time_ms(graph.replay)
        del graph
    return timings


@pytest.mark.parametrize("num_tokens", [8192, 16385, 32768, 50000, 65536])
def test_gluon_block_fp8_ep8_served_prefill_matches_sliced_and_reference(
    num_tokens: int, _served_ep8_rank, record_property
) -> None:
    """GLM-5.3 Flash EP8 prefill at served chunk sizes (one rank of eight).

    At 65,536 rows the unsliced BF16 copies would allocate an intermediate of
    exactly 2 GiB ([524288, 2048] bf16) and EP partials of 4 GiB. The output
    must be bit-identical to the same apply on slices of at most 8,192 rows
    (whose buffers are far below 2 GiB), match the float reference on
    sampled rows (including both ends), leave its inputs, the weights and
    allocator canaries untouched, and leave decode timing unchanged.
    """
    from test_gluon_fp8_packed_small import _reference

    module, (w13, w2, s13, s2) = _served_ep8_rank
    x, ids, weights = _served_routes(num_tokens, seed=num_tokens)
    inputs = (x.clone(), ids.clone(), weights.clone())
    weight_sums = [
        t.view(torch.int16).sum(dtype=torch.int64).item()
        for t in (module.w13_weight_prefill_bf16, module.w2_weight_prefill_bf16)
    ]
    canaries = [
        torch.full((64 << 20,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
        for _ in range(4)
    ]
    before = _decode_timings(module)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    actual = _apply(module, x, ids, weights)
    end.record()
    end.synchronize()
    peak_gib = (torch.cuda.max_memory_allocated() - base_alloc) / 2**30
    record_property("apply_ms", start.elapsed_time(end))
    record_property("transient_peak_gib", peak_gib)

    warm_ms = []
    for _ in range(3):
        start_warm = torch.cuda.Event(enable_timing=True)
        end_warm = torch.cuda.Event(enable_timing=True)
        start_warm.record()
        warm = _apply(module, x, ids, weights)
        end_warm.record()
        end_warm.synchronize()
        warm_ms.append(start_warm.elapsed_time(end_warm))
        assert torch.equal(warm, actual), "repeated apply is not deterministic"
        del warm
    record_property("apply_warm_ms", sorted(warm_ms)[1])

    # Balanced slices of at most 8,192 rows, every one above the exact-MFMA
    # bound (4,096), so the reference runs the same BF16 path on small buffers.
    slices = -(-num_tokens // _SERVED_REFERENCE_SLICE)
    step = -(-num_tokens // slices)
    assert num_tokens - (slices - 1) * step > 4096
    sliced = torch.cat(
        [
            _apply(
                module,
                x[row : row + step],
                ids[row : row + step],
                weights[row : row + step],
            )
            for row in range(0, num_tokens, step)
        ]
    )
    difference = (actual.float() - sliced.float()).abs()
    bad_rows = torch.nonzero(difference.amax(dim=1) > 0).flatten()
    mismatched = int(bad_rows.numel())
    bad_columns = (
        torch.nonzero(difference[bad_rows[0]] > 0).flatten().tolist()
        if mismatched
        else []
    )
    record_property("versus_sliced_max_abs", difference.max().item())
    record_property("versus_sliced_mismatched_rows", mismatched)

    sample = torch.cat(
        [
            torch.arange(0, 64, device="cuda"),
            torch.randperm(num_tokens, device="cuda")[:128],
            torch.arange(num_tokens - 64, num_tokens, device="cuda"),
        ]
    )
    expected = _reference(
        x[sample],
        w13,
        w2,
        s13,
        s2,
        ids[sample],
        weights[sample],
        _SERVED_RANK * (_SERVED_EXPERTS // _SERVED_EP_SIZE),
        7.0,
        quantized_activations=False,
    )[0]
    reference_error = (actual[sample].float() - expected.float()).abs()
    relative_l2 = (
        (actual[sample].float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1.0e-12)
    ).item()
    record_property("versus_reference_max_abs", reference_error.max().item())
    record_property("versus_reference_relative_l2", relative_l2)

    after = _decode_timings(module)
    for key, value in before.items():
        record_property(f"decode_{key}_ms_before", value)
        record_property(f"decode_{key}_ms_after", after[key])
    inputs_intact = all(
        torch.equal(a, b) for a, b in zip(inputs, (x, ids, weights))
    )
    weights_intact = weight_sums == [
        t.view(torch.int16).sum(dtype=torch.int64).item()
        for t in (module.w13_weight_prefill_bf16, module.w2_weight_prefill_bf16)
    ]
    canaries_intact = all(bool((c == 0x5A5A5A5A).all().item()) for c in canaries)
    print(
        f"served-ep8 rows={num_tokens} apply_ms={start.elapsed_time(end):.2f} "
        f"peak_gib={peak_gib:.3f} sliced_max_abs={difference.max().item():.6g} "
        f"sliced_mismatched_rows={mismatched} rows={bad_rows[:8].tolist()} "
        f"first_row_columns={bad_columns[:4]}..{bad_columns[-4:]} "
        f"({len(bad_columns)}) warm_ms={[round(t, 2) for t in warm_ms]} ref_max_abs="
        f"{reference_error.max().item():.6g} ref_rel_l2={relative_l2:.6g} "
        f"inputs_intact={inputs_intact} weights_intact={weights_intact} "
        f"canaries_intact={canaries_intact} decode_before={before} "
        f"decode_after={after}"
    )
    assert mismatched == 0, f"{mismatched} rows differ from the sliced apply"
    # Sliced EP prefill keeps the per-call transient near 2 GiB at 65,536 rows
    # (unsliced: 2 GiB intermediate + 4 GiB zero-filled partials, 6.6 GiB).
    assert peak_gib < 3.0, f"transient peak {peak_gib:.3f} GiB"
    assert relative_l2 <= 0.03, f"relative L2 {relative_l2:.6g} vs reference"
    assert inputs_intact and weights_intact and canaries_intact
    for key, value in before.items():
        assert after[key] <= 2 * value + 0.05, f"{key}: {value} -> {after[key]} ms"
