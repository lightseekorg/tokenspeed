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


"""HC4 coefficient epilogue with all Sinkhorn rounds inside one NPU kernel."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _hc4_coefficients(
    M, Scale, Base, Pre, Post, Comb, MS0, MS1, EPS: tl.constexpr, ITERS: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, 4)
    pre = tl.load(M + row * MS0 + i * MS1)
    post = tl.load(M + row * MS0 + (4 + i) * MS1)
    pre = tl.sigmoid(pre * tl.load(Scale) + tl.load(Base + i)) + EPS
    post = 2.0 * tl.sigmoid(post * tl.load(Scale + 1) + tl.load(Base + 4 + i))
    # Each writer owns a 64-byte row. Narrow adjacent rows caused corrupted
    # batched output with this Ascend compiler's sub-block stores.
    tl.store(Pre + row * 16 + i, pre)
    tl.store(Post + row * 16 + i, post)
    j = tl.arange(0, 16)
    comb = tl.load(M + row * MS0 + (8 + j) * MS1)
    comb = comb * tl.load(Scale + 2) + tl.load(Base + 8 + j)
    comb = tl.reshape(comb, (4, 4))
    comb = tl.exp(comb - tl.max(comb, 1)[:, None])
    comb = comb / tl.sum(comb, 1)[:, None] + EPS
    comb = comb / (tl.sum(comb, 0)[None, :] + EPS)
    for _ in tl.static_range(1, ITERS):
        comb = comb / (tl.sum(comb, 1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, 0)[None, :] + EPS)
    tl.store(Comb + row * 16 + j, tl.reshape(comb, (16,)))


def coefficients(mixes, scale, base, hc_eps, sinkhorn_iters):
    """FP32 [T,24] projected mixes -> FP32 pre/post [T,4], comb [T,4,4]."""
    n = mixes.shape[0]
    pre = torch.empty((n, 16), device=mixes.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty_like(pre)
    if n:
        _hc4_coefficients[(n,)](
            mixes,
            scale,
            base,
            pre,
            post,
            comb,
            *mixes.stride(),
            hc_eps,
            sinkhorn_iters,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return pre[:, :4], post[:, :4], comb.reshape(n, 4, 4)


def mhc_mixes(x, weight, scale, base, norm_eps, hc_eps, sinkhorn_iters):
    """Preserve FP32 projection and fuse RMS with the coefficient epilogue."""
    from tokenspeed_kernel_npu.ops.dsv41_mhc_mixes import mhc_mixes as fused_mixes

    return fused_mixes(x, weight, scale, base, norm_eps, hc_eps, sinkhorn_iters)


@triton.jit
def _hc4_pre_norm(
    X,
    Pre,
    Weight,
    Out,
    XS0,
    XS1,
    XS2,
    PS0,
    PS1,
    WS,
    OS,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for stream in tl.static_range(4):
        x = tl.load(X + row * XS0 + stream * XS1 + d * XS2, d < H, other=0).to(
            tl.float32
        )
        pre = tl.load(Pre + row * PS0 + stream * PS1)
        value = value + x * pre
    # The original pre-mix materializes BF16 before FP32 RMSNorm reads it.
    value = value.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / H + EPS)
    weight = tl.load(Weight + d * WS, d < H, other=0).to(tl.float32)
    result = (value * inv_rms) * weight
    tl.store(Out + row * OS + d, result, d < H)


def mhc_pre_norm(residual, pre, weight, eps):
    """Mix BF16 [T,4,H] with FP32 [T,4], round to BF16, then RMSNorm."""
    # A strided hidden axis lowers to large gather temporaries on Ascend.
    # Serving residuals already have a contiguous hidden axis; only callers
    # with sliced inputs need these copies, which are also captured by NPUGraph.
    if residual.stride(-1) != 1:
        residual = residual.contiguous()
    if weight.stride(0) != 1:
        weight = weight.contiguous()
    n, _, h = residual.shape
    # Align rows for sub-block stores, including small test hidden dimensions.
    output = torch.empty(
        (n, triton.cdiv(h, 32) * 32), device=residual.device, dtype=residual.dtype
    )[:, :h]
    if n:
        _hc4_pre_norm[(n,)](
            residual,
            pre,
            weight,
            output,
            *residual.stride(),
            *pre.stride(),
            weight.stride(0),
            output.stride(0),
            h,
            eps,
            triton.next_power_of_2(h),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return output


@triton.jit
def _hc4_post(
    X,
    Residual,
    Post,
    Comb,
    Out,
    XS0,
    XS1,
    RS0,
    RS1,
    RS2,
    PS0,
    PS1,
    CS0,
    CS1,
    CS2,
    OS0,
    OS1,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * XS0 + d * XS1, d < H, other=0).to(tl.float32)
    r0 = tl.load(Residual + row * RS0 + d * RS2, d < H, other=0).to(tl.float32)
    r1 = tl.load(Residual + row * RS0 + RS1 + d * RS2, d < H, other=0).to(tl.float32)
    r2 = tl.load(Residual + row * RS0 + 2 * RS1 + d * RS2, d < H, other=0).to(
        tl.float32
    )
    r3 = tl.load(Residual + row * RS0 + 3 * RS1 + d * RS2, d < H, other=0).to(
        tl.float32
    )
    for output_stream in tl.static_range(4):
        c = Comb + row * CS0 + output_stream * CS2
        mixed = tl.full((BLOCK,), 0, tl.float32)
        mixed = mixed + r0 * tl.load(c)
        mixed = mixed + r1 * tl.load(c + CS1)
        mixed = mixed + r2 * tl.load(c + 2 * CS1)
        mixed = mixed + r3 * tl.load(c + 3 * CS1)
        post = tl.load(Post + row * PS0 + output_stream * PS1)
        tl.store(Out + row * OS0 + output_stream * OS1 + d, x * post + mixed, d < H)


def mhc_post(hidden_states, residual, post, comb):
    """Apply FP32 comb[input,output] and post[T,4,1], returning BF16 [T,4,H]."""
    n, _, h = residual.shape
    output = torch.empty(
        (n, 4, triton.cdiv(h, 32) * 32), device=residual.device, dtype=residual.dtype
    )[:, :, :h]
    if n:
        # Larger tiles amortize per-program overhead; load each residual stream
        # once and reuse it for all four outputs. Bound the live Vector buffers.
        tile_limit = 1024 if n <= 32 else 4096
        if hidden_states.stride(-1) != 1 or residual.stride(-1) != 1:
            # Strided Vector loads need extra gather buffers on Ascend.
            tile_limit = 512
        block = min(tile_limit, triton.next_power_of_2(h))
        _hc4_post[(n, triton.cdiv(h, block))](
            hidden_states,
            residual,
            post,
            comb,
            output,
            *hidden_states.stride(),
            *residual.stride(),
            *post.stride()[:2],
            *comb.stride(),
            *output.stride()[:2],
            h,
            block,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return output
