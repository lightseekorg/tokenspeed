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

"""Numerics of the native decoder layer's front bodies on gfx950: the o_proj
sender and the router / pooler K-part consumers, each as a test kernel
(native_body_helpers.py) against an FP32 reference."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel_amd._triton import gl, triton
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from native_body_helpers import (  # noqa: E402
    kpart_pooler_kernel,
    kpart_pooler_steps_kernel,
    kpart_router_kernel,
    kpart_router_steps_kernel,
    qkv_task_kernel,
    qkv_task_steps_kernel,
    sender_kernel,
    topk_keys_kernel,
)

WIDTH = 8192
PRODUCERS = 64
PARTS = 8
POST_EPS = 1e-6
PRE_EPS = 1e-5


@pytest.mark.parametrize("rows", [8, 16])
def test_sender_matches_fp32_reference(rows: int) -> None:
    """Each sender block's BF16 o_proj columns equal the FP32 product rounded
    once, and repeat bitwise."""
    torch.manual_seed(rows)
    k, n, outputs = 1024, 8192, 64
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03
    blocks = n // outputs
    out = torch.empty(blocks * 16, outputs, device="cuda", dtype=torch.bfloat16)
    sender_kernel[(blocks,)](
        w, x, out, rows, K=k, OUTPUTS=outputs, VECTOR=2, num_warps=8
    )
    actual = out.view(blocks, 16, outputs)[:, :rows].permute(1, 0, 2).reshape(rows, n)
    expected = x.float() @ w.float().T
    # FP32 accumulation then one BF16 rounding: one BF16 ULP (2**-8 relative,
    # rtol 2**-7 for the rounding plus FP32 order); atol for sums near zero,
    # whose ~1e-6 FP32 order noise is far below a BF16 ULP of the |out| ~ 1
    # terms (K 1024 products of N(0,1) x 0.03).
    torch.testing.assert_close(actual.float(), expected, atol=2**-10, rtol=2**-7)
    again = torch.empty_like(out)
    sender_kernel[(blocks,)](
        w, x, again, rows, K=k, OUTPUTS=outputs, VECTOR=2, num_warps=8
    )
    assert torch.equal(again, out)


def _boundary_inputs(rows: int, seed: int) -> dict:
    """The attention boundary as the K-part consumers see it: the branch b,
    each producer's four partial sums of its column block, h and the norms'
    vectors."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    b = torch.randn(rows, WIDTH, device="cuda", generator=g).to(torch.bfloat16)
    h = torch.randn(rows, WIDTH, device="cuda", generator=g) * 4
    alpha = torch.rand(WIDTH, device="cuda", generator=g) + 0.5
    beta = torch.rand(WIDTH, device="cuda", generator=g) + 0.5
    post = (torch.rand(WIDTH, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    pre = (torch.rand(WIDTH, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    bp = b.float() * post.float()
    blocks = lambda v: v.view(rows, PRODUCERS, WIDTH // PRODUCERS).sum(-1)  # noqa: E731
    ah, bb = alpha * h, beta * bp
    sums = torch.stack(
        [blocks(bp * bp), blocks(ah * bb), blocks(ah * ah), blocks(bb * bb)], -1
    ).contiguous()
    inv1 = torch.rsqrt((bp * bp).sum(-1, keepdim=True) / WIDTH + POST_EPS)
    hp = beta * bp * inv1 + alpha * h
    inv2 = torch.rsqrt((hp * hp).sum(-1, keepdim=True) / WIDTH + PRE_EPS)
    x = hp * inv2 * pre.float()
    return dict(args=(b, sums, h.contiguous(), alpha, beta, post, pre), x=x, rows=rows)


@pytest.mark.parametrize("rows", [8, 16])
def test_kpart_pooler_matches_fp32_reference(rows: int) -> None:
    """The pooler K-part partials, summed over the parts, equal the pooler
    applied to the deferred-norm x (BF16-rounded), within the K split's
    rounding."""
    inputs = _boundary_inputs(rows, 3 + rows)
    outputs, tile_rows = 512, 64
    w = torch.randn(outputs, WIDTH, device="cuda", dtype=torch.bfloat16) * 0.02
    parts = torch.zeros(PARTS * rows * outputs, device="cuda", dtype=torch.int64)
    kpart_pooler_kernel[(outputs // tile_rows * PARTS,)](
        *inputs["args"], w, parts, 5, rows, WIDTH=WIDTH, OUTPUTS=outputs,
        ROWS=tile_rows, PRODUCERS=PRODUCERS, PARTS=PARTS, POST_EPS=POST_EPS,
        PRE_EPS=PRE_EPS, num_warps=8,
    )  # fmt: skip
    words = parts.view(PARTS, rows, outputs)
    assert bool(((words >> 32) == 5).all())
    actual = (words & 0xFFFFFFFF).to(torch.int32).view(torch.float32).sum(0)
    # x rounds to BF16 on both sides, but the kernel's x differs from the
    # reference's in the last FP32 bits (block-summed norms), which flips a
    # BF16 rounding now and then: one BF16 ULP (2**-8) of an x element times
    # its weight (0.02) is ~1e-4 per flip; the FP32 sums agree to 1e-6.
    expected = inputs["x"].to(torch.bfloat16).float() @ w.float().T
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("rows", [8, 16])
def test_kpart_router_matches_fp32_reference(rows: int) -> None:
    """The router K-part logits (FP32 weight as BF16 hi + lo, FP32 x as hi +
    lo) match the FP32 router on the FP32 x to ~1e-5 relative."""
    inputs = _boundary_inputs(rows, 30 + rows)
    experts = 256
    router = torch.randn(experts, WIDTH, device="cuda") * 0.02
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    parts = torch.zeros(PARTS * rows * experts, device="cuda", dtype=torch.int64)
    kpart_router_kernel[(experts // 16 * PARTS,)](
        *inputs["args"], hi, lo, parts, 9, rows, WIDTH=WIDTH, EXPERTS=experts,
        PRODUCERS=PRODUCERS, PARTS=PARTS, POST_EPS=POST_EPS, PRE_EPS=PRE_EPS,
        num_warps=8,
    )  # fmt: skip
    words = parts.view(PARTS, rows, experts)
    assert bool(((words >> 32) == 9).all())
    actual = (words & 0xFFFFFFFF).to(torch.int32).view(torch.float32).sum(0)
    # hi + lo carries ~16 bits of the weight and of x: each product of
    # |x w| ~ 0.02 is off by up to ~2**-15 relative (the dropped lo x lo
    # term and both residuals), ~6e-7. Over K = 8192 terms of random sign
    # the sum's sigma is ~1e-5 (GPU: max 3.5e-5 over 2048 logits, ~3.5
    # sigma), so 6e-5 is ~6 sigma. FP32 accumulation adds ~1e-6.
    expected = (inputs["x"].double() @ router.double().T).float()
    torch.testing.assert_close(actual, expected, atol=6e-5, rtol=1e-5)


@pytest.mark.parametrize("rows", [8, 16])
@pytest.mark.parametrize("role", ["router", "pooler", "qkv"])
def test_kpart_steps_are_bitwise_the_full_x_build(role: str, rows: int) -> None:
    """Building x one K step at a time (norm vectors and b from LDS) runs the
    same elementwise operations and each tile's MFMAs in the same K order as
    the full-x build, so the partial words are bitwise equal."""
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (
        persistent_compile_options,
    )

    inputs = _boundary_inputs(rows, 50 + rows)
    common = dict(
        WIDTH=WIDTH, PRODUCERS=PRODUCERS, PARTS=PARTS, POST_EPS=POST_EPS,
        PRE_EPS=PRE_EPS, num_warps=8,
    )  # fmt: skip
    if role == "qkv":
        # 144-row tasks over 16 K parts, weights paced 3 tiles ahead.
        common.pop("PARTS")
        weights = (
            torch.randn(2304, WIDTH, device="cuda", dtype=torch.bfloat16) * 0.02,
        )
        kernels, grid, outputs = (qkv_task_kernel, qkv_task_steps_kernel), 256, 2304
        extra = dict(N=outputs, WINDOW=3)
        parts_count = 16
    elif role == "router":
        router = torch.randn(256, WIDTH, device="cuda") * 0.02
        hi = router.to(torch.bfloat16)
        weights = (hi, (router - hi.float()).to(torch.bfloat16))
        kernels, grid, outputs = (
            (kpart_router_kernel, kpart_router_steps_kernel),
            16 * PARTS,
            256,
        )
        extra = dict(EXPERTS=outputs)
        parts_count = PARTS
    else:
        weights = (torch.randn(512, WIDTH, device="cuda", dtype=torch.bfloat16) * 0.02,)
        kernels, grid, outputs = (
            (kpart_pooler_kernel, kpart_pooler_steps_kernel),
            8 * PARTS,
            512,
        )
        extra = dict(OUTPUTS=outputs, ROWS=64)
        parts_count = PARTS
    words = []
    for kernel in kernels:
        parts = torch.zeros(
            parts_count * rows * outputs, device="cuda", dtype=torch.int64
        )
        kernel[(grid,)](
            *inputs["args"], *weights, parts, 5, rows, **extra, **common,
            **persistent_compile_options(),
        )  # fmt: skip
        words.append(parts)
    assert torch.equal(words[1], words[0])


@pytest.mark.parametrize("rows", [8, 16])
def test_topk_keys_match_the_row_topk_bitwise(rows: int) -> None:
    """The one-wave-per-row 64-bit-key top-8 picks the same experts in the
    same order with the same weights as the per-row top-k the layer uses
    today, ties (equal logits) included."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (
        gluon_sigmoid_bias_topk_gfx950,
    )

    torch.manual_seed(rows)
    experts, topk, scale = 256, 8, 2.5
    logits = torch.randn(rows, experts, device="cuda")
    logits[:, 100:110] = logits[:, 3:4]  # equal scores: the lower expert wins
    bias = torch.randn(experts, device="cuda") * 0.1
    bias[100:110] = bias[3]
    ids = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    weights = torch.empty(rows, topk, device="cuda")
    gluon_sigmoid_bias_topk_gfx950[(rows,)](
        logits, bias, ids, weights, experts, 1, 1, topk, 1, topk, 1, E=experts,
        TOPK=topk, EP=experts, TKP=topk, NORMALIZE_TOPK_WEIGHTS=True,
        ROUTED_SCALING_FACTOR=scale, X_DTYPE=gl.float32, NUM_WARPS=1, num_warps=1,
    )  # fmt: skip
    keys_ids = torch.empty_like(ids)
    keys_weights = torch.empty_like(weights)
    topk_keys_kernel[(triton.cdiv(rows, 8),)](
        logits, bias, keys_ids, keys_weights, rows, E=experts, TOPK=topk,
        ROUTED_SCALING_FACTOR=scale, num_warps=8,
    )  # fmt: skip
    assert torch.equal(keys_ids, ids)
    assert torch.equal(keys_weights.view(torch.int32), weights.view(torch.int32))
