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

"""Gluon device helpers shared by staged and fused MXFP4 MoE kernels."""

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def _mxfp4_quantize_tile(out):
    max_normal: gl.constexpr = 6.0
    min_normal: gl.constexpr = 1.0
    BLOCK_M: gl.constexpr = out.shape[0]
    OUT_BLOCK_N: gl.constexpr = out.shape[1]
    Q_GROUPS: gl.constexpr = OUT_BLOCK_N // 32
    gl.static_assert(OUT_BLOCK_N % 32 == 0)

    vals = out.to(gl.bfloat16).to(gl.float32).reshape((BLOCK_M, Q_GROUPS, 32))
    raw_abs = vals.to(gl.uint32, bitcast=True) & 0x7FFFFFFF
    abs_vals = raw_abs.to(gl.float32, bitcast=True)
    amax = gl.max(abs_vals, axis=2, keep_dims=True)
    amax_bits = amax.to(gl.uint32, bitcast=True)
    rounded_bits = (amax_bits + 0x200000) & 0x7F800000
    exp_biased = (rounded_bits >> 23).to(gl.int32)
    scale_i = gl.minimum(gl.maximum(exp_biased - 2, 0), 254)
    scale_byte = scale_i.to(gl.uint8).reshape((BLOCK_M, Q_GROUPS))

    inv_scale_bits = ((254 - scale_i) << 23).to(gl.uint32)
    inv_scale = inv_scale_bits.to(gl.float32, bitcast=True)
    qx = vals * inv_scale
    qx_bits = qx.to(gl.uint32, bitcast=True)

    sign = qx_bits & 0x80000000
    qx_mag = qx_bits ^ sign
    qx_fp32 = qx_mag.to(gl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_mask_int: gl.constexpr = ((127 - 1) + (23 - 1) + 1) << 23
    denorm_mask_float: gl.constexpr = gl.cast(
        denorm_mask_int,
        gl.float32,
        bitcast=True,
    )
    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(gl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(gl.uint8)

    normal_x = qx_mag
    mant_odd = (normal_x >> (23 - 1)) & 1
    normal_x += 0xC11FFFFF
    normal_x += mant_odd
    normal_x = normal_x >> (23 - 1)
    normal_x = normal_x.to(gl.uint8)

    e2m1 = gl.full(vals.shape, 0x7, gl.uint8, layout=vals.type.layout)
    e2m1 = gl.where(normal_mask, normal_x, e2m1)
    e2m1 = gl.where(denormal_mask, denormal_x, e2m1)
    sign_lp = (sign >> (23 + 8 - 1 - 2)).to(gl.uint8)
    e2m1 = e2m1 | sign_lp
    e2m1 = e2m1.reshape((BLOCK_M, Q_GROUPS, 16, 2))
    evens, odds = gl.split(e2m1)
    packed = evens | (odds << 4)
    return packed, scale_byte


@gluon.jit
def _mxfp4_store_cdna4_scale(
    scale_ptr,
    scale_byte,
    scale_m,
    scale_k,
    stride_kswizzled,
    stride_mblock,
    mask,
    M_SWIZZLE: gl.constexpr,
    K_SWIZZLE: gl.constexpr,
):
    m_in_block = scale_m % M_SWIZZLE
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = scale_k // K_SWIZZLE
    k_in_block = scale_k % K_SWIZZLE
    k_hi = k_in_block // 4
    k_lo = k_in_block % 4
    swizzled_k = (((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi
    m_block = scale_m // M_SWIZZLE
    gl.store(
        scale_ptr
        + swizzled_k.to(gl.int64) * stride_kswizzled
        + m_block.to(gl.int64) * stride_mblock,
        scale_byte,
        mask=mask,
    )
