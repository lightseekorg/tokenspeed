# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe.gluon import (
    _direct_topk_small_m,
    _gluon_mxfp4_fp8_warp_decode_moe,
)
from tokenspeed_kernel.ops.moe.triton_kernels import (
    FlexCtx,
    InFlexData,
    PrecisionConfig,
)
from tokenspeed_kernel.platform import current_platform


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
@pytest.mark.skipif(
    not current_platform().is_cdna4, reason="Gluon warp-decode helpers are gfx950-only"
)
@pytest.mark.parametrize("M", [1, 4, 8, 16])
@pytest.mark.parametrize("E", [16, 128])
@pytest.mark.parametrize("topk", [1, 4])
def test_direct_topk_small_m_matches_selected_softmax(M: int, E: int, topk: int):
    torch.manual_seed(M * 1000 + E * 10 + topk)
    logits = torch.randn((M, E), device="cuda", dtype=torch.float32)

    ids, weights = _direct_topk_small_m(logits, topk)
    torch.cuda.synchronize()

    ref_vals, ref_ids = torch.topk(logits, topk, dim=-1)
    ref_weights = torch.softmax(ref_vals, dim=-1)

    assert torch.equal(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
@pytest.mark.skipif(
    not current_platform().is_cdna4, reason="Gluon warp-decode helpers are gfx950-only"
)
def test_direct_topk_small_m_rejects_large_m():
    logits = torch.randn((17, 16), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="requires M <="):
        _direct_topk_small_m(logits, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
@pytest.mark.skipif(
    not current_platform().is_cdna4, reason="Gluon warp-decode helpers are gfx950-only"
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_fp8_mxfp4_warp_decode_moe_matches_torch_reference(use_bias: bool):
    pytest.importorskip("aiter")
    from aiter.utility.fp4_utils import mxfp4_to_f32

    from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel import swizzle_mxfp4

    torch.manual_seed(123)
    M, E, D, I, topk = 2, 4, 128, 128, 2
    device = "cuda"
    hidden = torch.randn((M, D), device=device, dtype=torch.bfloat16)
    router = torch.randn((M, E), device=device, dtype=torch.float32)
    w13 = torch.randint(0, 256, (E, 2 * I, D // 2), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, D, I // 2), device=device, dtype=torch.uint8)
    s13 = torch.full((E, 2 * I, D // 32), 127, device=device, dtype=torch.uint8)
    s2 = torch.full((E, D, I // 32), 127, device=device, dtype=torch.uint8)
    # N (w2 output dim) equals D for this square config.
    w13_bias = torch.randn((E, 2 * I), device=device, dtype=torch.float32) if use_bias else None
    w2_bias = torch.randn((E, D), device=device, dtype=torch.float32) if use_bias else None
    wt13, flex13, st13 = swizzle_mxfp4(w13, s13, 8)
    wt2, flex2, st2 = swizzle_mxfp4(w2, s2, 8)
    scale1 = torch.ones((1,), device=device, dtype=torch.float32)
    scale2 = torch.ones((1,), device=device, dtype=torch.float32)
    fp8_dtype = torch.float8_e4m3fn
    pc1 = PrecisionConfig(
        flex_ctx=FlexCtx(
            lhs_data=InFlexData(dtype=fp8_dtype, scale=scale1), rhs_data=flex13
        ),
        b_mx_scale=st13,
        b_microblock_size=32,
        out_dtype=torch.bfloat16,
    )
    pc2 = PrecisionConfig(
        flex_ctx=FlexCtx(
            lhs_data=InFlexData(dtype=fp8_dtype, scale=scale2), rhs_data=flex2
        ),
        b_mx_scale=st2,
        b_microblock_size=32,
        out_dtype=torch.bfloat16,
    )

    out = _gluon_mxfp4_fp8_warp_decode_moe(
        hidden,
        router,
        wt13,
        wt2,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        w13_precision_config=pc1,
        w2_precision_config=pc2,
        w13_act_scale=scale1,
        w2_act_scale=scale2,
        top_k=topk,
    )
    assert out is not None
    torch.cuda.synchronize()

    topk_vals, topk_ids = torch.topk(router, topk, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    hidden_fp8 = hidden.to(fp8_dtype).to(torch.float32)
    w13_f = mxfp4_to_f32(w13)
    w2_f = mxfp4_to_f32(w2)
    ref = torch.zeros((M, D), device=device, dtype=torch.float32)
    for m in range(M):
        for slot in range(topk):
            expert = int(topk_ids[m, slot])
            gate_up = hidden_fp8[m : m + 1] @ w13_f[expert].T
            if use_bias:
                # Bias is added before the swiglu clamp, matching the kernel.
                gate_up = gate_up + w13_bias[expert][None, :]
            gate = torch.minimum(gate_up[:, :I], torch.tensor(7.0, device=device))
            linear = torch.clamp(gate_up[:, I:], -7.0, 7.0)
            inter = (gate / (1.0 + torch.exp(-1.702 * gate))) * (linear + 1.0)
            inter_fp8 = inter.to(fp8_dtype).to(torch.float32)
            second = inter_fp8 @ w2_f[expert].T
            if use_bias:
                second = second + w2_bias[expert][None, :]
            ref[m] += topk_weights[m, slot] * second.squeeze(0)
    torch.testing.assert_close(
        out.float(), ref.to(torch.bfloat16).float(), rtol=5e-2, atol=2.0
    )
