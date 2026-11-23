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

"""Activation quantization of a row for the expert GEMMs, as they read it on
gfx950: the exact BF16 values of the quantized row and, for NVFP4, the row's
``amax``.

NVFP4 (TokenSpeed ``_nvfp4_chunk_codes_and_values``):

Per row: ``enc = 1536 / amax``, ``dec = 1 / enc``, ``outer = amax / 1536``.
Per group of 16 with ``gm = max |x|``, for target ``t`` in 6 and 4:
``sc = e4m3(min(gm enc / t, 448))``, ``v = e2m1(x / (sc dec))``, error
``sum |x - v sc outer|`` (a pairwise tree over the 16). Target 4 wins only on
a strictly smaller error. The published value is ``bf16(v sc)``: the GEMM
applies ``outer`` (through ``amax``) after its dot products.

MXFP8 (TokenSpeed ``_mxfp8_chunk_codes`` and ``_mxfp8_chunk_values``), per
group of 32 with ``r = max |x| (1 / 448)``: the UE8M0 exponent ``e`` rounds
``r`` up to a power of two (0 at ``r <= 2^-127``, at most 254, 255 for NaN),
``s = max(2^(e - 127), 2^-127)``, ``v = e4m3(clamp(x / s, +-448))`` with the
quotient as a multiply by ``1 / s`` (``2^127`` at ``e = 0``). The published
value is ``bf16(v s)``, exact: no outer factor.

A row of ``N`` values (the plan's ``dims.hidden``) is ``N / 16`` groups of
16 (288 for Inkling 2 Small, 144 at half its width), each quantized on its
own; only the row's ``amax`` (a max, exact in any order) spans the row. The
stages hold one 16-group a thread, the groups padded to the next of 512 or
1024 (padding groups are +0.0 and store nothing).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._extern import arithmetic_fence
from tokenspeed_kernel_amd._triton import gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950.monokernel import trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import mix

FLT_MAX = gl.constexpr(3.4028234663852886e38)
GROUP = gl.constexpr(16)
MX_GROUP = gl.constexpr(32)


@gluon.jit
def e2m1(y):
    """``y`` rounded to the nearest E2M1 value, ties to even, saturating,
    with the sign bit of ``y`` (``-0.0`` stays ``-0.0``, as TokenSpeed's
    ``_fp4_encode`` keeps it)."""
    a = gl.abs(y)
    v = gl.where(a <= 0.25, 0.0, 0.5)
    v = gl.where(a >= 0.75, 1.0, v)
    v = gl.where(a > 1.25, 1.5, v)
    v = gl.where(a >= 1.75, 2.0, v)
    v = gl.where(a > 2.5, 3.0, v)
    v = gl.where(a >= 3.5, 4.0, v)
    v = gl.where(a > 5.0, 6.0, v)
    return gl.where(y.to(gl.int32, bitcast=True) < 0, -v, v)


@gluon.jit
def _tree_sum(e):
    """Pairwise sum over the last axis (16 values) of ``e [G, 16]``."""
    G: gl.constexpr = e.shape[0]
    for _ in gl.static_range(4):
        e = gl.sum(gl.reshape(e, [G, e.shape[1] // 2, 2]), axis=2)
    return gl.reshape(e, [G])


@gluon.jit
def _candidate(x, gm, enc, dec, outer, target: gl.constexpr):
    """``(values, error)`` of quantizing ``x [G, 16]`` toward ``target``."""
    sc = gl.minimum(gm * (enc * (1.0 / target)), 448.0)
    sc = sc.to(gl.float8e4nv).to(gl.float32)
    inv = gl.minimum(gl.div_rn(1.0, sc * dec), FLT_MAX)
    value = e2m1(x * inv[:, None]) * sc[:, None]
    # The reconstruction rounds before the subtraction, as TokenSpeed's
    # quantizer (a launch without FP fusion) does: no FMA.
    reconstructed = arithmetic_fence(value * outer)
    return value, gl.convert_layout(
        _tree_sum(gl.abs(x - reconstructed)), gm.type.layout
    )


@gluon.jit
def quantize_groups(x, enc, dec, outer):
    """BF16-exact quantized values ``bf16(v sc)`` (FP32) of ``x [G, 16]``."""
    gm = gl.max(gl.abs(x), axis=1)
    six, err6 = _candidate(x, gm, enc, dec, outer, 6.0)
    four, err4 = _candidate(x, gm, enc, dec, outer, 4.0)
    # Target 4 replaces 6 only on a strictly smaller error.
    best = gl.where((err4 < err6)[:, None], four, six)
    return best.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def mxfp8_groups(x):
    """BF16-exact quantized values ``bf16(v s)`` (FP32) of ``x [G, 32]``,
    each row an MXFP8 group."""
    raw = gl.max(gl.abs(x), axis=1) * (1.0 / 448.0)
    exponent = ((raw.to(gl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = gl.where(raw <= 2.0**-127, 0, gl.minimum(exponent, 254))
    exponent = gl.where(raw != raw, 255, exponent)
    scale = gl.maximum((exponent << 23).to(gl.float32, bitcast=True), 2.0**-127)
    inverse = gl.where(exponent == 0, 2.0**127, gl.div_rn(1.0, scale))
    inverse = gl.where(exponent == 255, 0.0, inverse)
    values = x * inverse[:, None]
    values = gl.minimum(
        gl.maximum(values, -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    # The decode of the expert GEMMs: NaN scales for exponent 255.
    decoded = gl.where(exponent == 255, float("nan"), scale)
    values = values.to(gl.float8e4nv).to(gl.float32) * decoded[:, None]
    return values.to(gl.bfloat16).to(gl.float32)


@gluon.constexpr_function
def _group_layout(num_warps):
    return gl.BlockedLayout([1, GROUP // 2], [64, 1], [num_warps, 1], [1, 0])


@gluon.jit
def quant_stage(
    mb,
    x: gl.constexpr,
    xq: gl.constexpr,
    xa: gl.constexpr,
    xq_flag: gl.constexpr,
    N: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Quantize the row that ``mb`` views: ``N`` BF16 values of region ``x``
    to ``xq`` (plain words of BF16 pairs of ``bf16(v sc)``, for the expert
    GEMM's operand loads) and ``xa`` (the FP32 amax), then raise flag 0 of
    ``xq_flag``."""
    mb.stamp(STAGE, trace.START)
    G: gl.constexpr = N // GROUP
    gl.static_assert(G * GROUP == N and G <= 1024)
    GP: gl.constexpr = 512 if G <= 512 else 1024
    layout: gl.constexpr = _group_layout(gl.num_warps())
    g = gl.arange(0, GP, gl.SliceLayout(1, layout))
    c = gl.arange(0, GROUP // 2, gl.SliceLayout(0, layout))
    live = (g < G)[:, None]
    pair = gl.minimum(g, G - 1)[:, None] * (GROUP // 2) + c[None, :]
    word = mb.poll(x, pair)
    mb.stamp(STAGE, trace.READY)
    lo, hi = mix.unpack(word)
    lo = gl.where(live, lo, 0.0)
    hi = gl.where(live, hi, 0.0)
    values = gl.reshape(gl.join(lo, hi), [GP, GROUP])
    amax = gl.max(gl.max(gl.abs(values), axis=1), axis=0)
    enc = gl.minimum(gl.div_rn(1536.0, amax), FLT_MAX)
    dec = gl.div_rn(1.0, enc)
    outer = amax * (1.0 / 1536.0)
    q = quantize_groups(values, enc, dec, outer)
    q_lo, q_hi = gl.split(gl.reshape(q, [GP, GROUP // 2, 2]))
    q_lo = gl.convert_layout(q_lo, layout)
    q_hi = gl.convert_layout(q_hi, layout)
    mb.store_words(xq, pair, mix.pack(q_lo, q_hi), mask=live)
    one = gl.arange(0, 1, gl.SliceLayout(1, layout))
    mb.put(xa, one, amax.to(gl.int32, bitcast=True) + one * 0)
    mb.raise_flag(xq_flag, 0)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def mxfp8_quant_stage(
    mb,
    x: gl.constexpr,
    xq: gl.constexpr,
    xq_flag: gl.constexpr,
    N: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Quantize the row that ``mb`` views to MXFP8: ``N`` BF16 values of
    region ``x`` to ``xq`` (plain words of BF16 pairs of ``bf16(v s)``, for
    the expert GEMM's operand loads), then raise flag 0 of ``xq_flag``."""
    mb.stamp(STAGE, trace.START)
    G: gl.constexpr = N // GROUP
    gl.static_assert(G * GROUP == N and G % 2 == 0 and G <= 1024)
    GP: gl.constexpr = 512 if G <= 512 else 1024
    layout: gl.constexpr = _group_layout(gl.num_warps())
    g = gl.arange(0, GP, gl.SliceLayout(1, layout))
    c = gl.arange(0, GROUP // 2, gl.SliceLayout(0, layout))
    live = (g < G)[:, None]
    pair = gl.minimum(g, G - 1)[:, None] * (GROUP // 2) + c[None, :]
    word = mb.poll(x, pair)
    mb.stamp(STAGE, trace.READY)
    lo, hi = mix.unpack(word)
    lo = gl.where(live, lo, 0.0)
    hi = gl.where(live, hi, 0.0)
    # Two 16-value rows of ``values`` are one MXFP8 group.
    values = gl.reshape(gl.join(lo, hi), [GP // 2, MX_GROUP])
    q = gl.reshape(mxfp8_groups(values), [GP, GROUP // 2, 2])
    q_lo, q_hi = gl.split(q)
    q_lo = gl.convert_layout(q_lo, layout)
    q_hi = gl.convert_layout(q_hi, layout)
    mb.store_words(xq, pair, mix.pack(q_lo, q_hi), mask=live)
    mb.raise_flag(xq_flag, 0)
    mb.stamp(STAGE, trace.END)
