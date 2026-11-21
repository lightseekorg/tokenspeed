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

"""FP32 libdevice ``log1p`` with the bits of a launch without FP fusion
(``enable_fp_fusion=False``) on gfx950.

A monokernel launches with FP fusion on. The backend then contracts some of
ocml ``log1p``'s products and the adds that use them into FMAs, which changes
the last bits of about 8% of its results; libdevice's ``exp``, ``log`` and
``tanh`` keep their bits. A regular kernel launched with fusion off computes
the plain steps. ``log1p`` here spells out ocml's steps with each product
rounded on its own (``_mul``), so a stage calling it equals such a regular
kernel bit for bit. Launch with ``extern_compile_options()``.

The steps and constants follow ROCm ocml's FP32 ``log1p`` (Apache-2.0 with
LLVM Exceptions); see ``THIRDPARTYNOTICES``.
"""

from __future__ import annotations

import struct

from tokenspeed_kernel_amd._extern import arithmetic_fence, hw_rcp
from tokenspeed_kernel_amd._triton import gl, gluon


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


INF = gl.constexpr(float("inf"))
QNAN_BITS = gl.constexpr(0x7FC00000)  # ocml QNAN_F32
# ocml lnep: ln(2) as head and tail, the log polynomial, 2/3.
LNEP_LN2 = gl.constexpr(_f32(0x3F317218))
LNEP_LN2_LO = gl.constexpr(_f32(0xB102E308))
LNEP_C0 = gl.constexpr(_f32(0x3E9B6DAC))
LNEP_C1 = gl.constexpr(_f32(0x3ECC95A3))
LNEP_C2 = gl.constexpr(_f32(0x3F2AAADA))
TWO_THIRDS = gl.constexpr(_f32(0x3F2AAAAB))


@gluon.jit
def _mul(a, b):
    """``a * b`` rounded on its own: no add that uses it fuses it into an
    FMA."""
    return arithmetic_fence(a * b)


# ocml's double-float (head, tail) steps that log1p uses, with ocml's names.


@gluon.jit
def _quick_two_sum(a, b):
    """``fadd(a, b)``: ``a + b`` and its error, for ``|a| >= |b|``."""
    s = a + b
    return s, b - (s - a)


@gluon.jit
def _two_sum(a, b):
    """``add(a, b)``: ``a + b`` and its error."""
    s = a + b
    d = s - a
    return s, (a - (s - d)) + (b - d)


@gluon.jit
def _two_prod(a, b):
    """``mul(a, b)``: ``a b`` and its error."""
    p = _mul(a, b)
    return p, gl.fma(a, b, -p)


@gluon.jit
def _sub_dd(ah, al, bh, bl):
    """``fsub(a, b)`` of two (head, tail) pairs."""
    d = ah - bh
    dl = ((ah - d) - bh) + al - bl
    return _quick_two_sum(d, dl)


@gluon.jit
def _mul_f_dd(a, bh, bl):
    """``mul(a, b)`` of a float and a (head, tail) pair."""
    ph, pl = _two_prod(a, bh)
    return _quick_two_sum(ph, gl.fma(a, bl, pl))


@gluon.jit
def _div_dd(ah, al, bh, bl):
    """``div(a, b)`` of two (head, tail) pairs."""
    y = hw_rcp(bh)
    qhi = _mul(ah, y)
    ph, pl = _mul_f_dd(qhi, bh, bl)
    rh, rl = _sub_dd(ah, al, ph, pl)
    qmi = _mul(rh, y)
    ph, pl = _mul_f_dd(qmi, bh, bl)
    rh, rl = _sub_dd(rh, rl, ph, pl)
    qlo = _mul(rh, y)
    qh, ql = _quick_two_sum(qhi, qmi)
    return _quick_two_sum(qh, ql + qlo)


@gluon.jit
def _fadd_f_dd(a, bh, bl):
    """``fadd(a, b)`` of a float and a (head, tail) pair."""
    s, sl = _quick_two_sum(a, bh)
    return _quick_two_sum(s, sl + bl)


@gluon.jit
def log1p(x):
    """libdevice ``log1p`` (ocml): ``ln`` of the double-float ``1 + x``,
    reduced to ``m 2^e`` with ``m`` in [2/3, 4/3), as
    ``2 atanh((m - 1) / (m + 1)) + e ln(2)``."""
    ah, al = _two_sum(1.0, x)
    # frexp of the head (a normal, or inf / NaN that the selects below
    # replace): mantissa in [0.5, 1) and exponent.
    bits = ah.to(gl.int32, bitcast=True)
    mant = ((bits & -2139095041) | 0x3F000000).to(gl.float32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 126 - (mant < TWO_THIRDS).to(gl.int32)
    mh = gl.extra.libdevice.ldexp(ah, -e)
    ml = gl.extra.libdevice.ldexp(al, -e)
    nh, nl = _fadd_f_dd(-1.0, mh, ml)
    dh, dl = _fadd_f_dd(1.0, mh, ml)
    th, tl_ = _div_dd(nh, nl, dh, dl)
    s = _mul(th, th)
    p = gl.fma(s, gl.fma(s, LNEP_C0, LNEP_C1), LNEP_C2)
    # fadd(2 t, s t p)
    uh, ul = _quick_two_sum(_mul(th, 2.0), _mul(_mul(s, th), p))
    uh, ul = _quick_two_sum(uh, ul + _mul(tl_, 2.0))
    # mul((ln 2 head, tail), e)
    fe = e.to(gl.float32)
    vh, vl = _two_prod(LNEP_LN2, fe)
    vh, vl = _quick_two_sum(vh, gl.fma(LNEP_LN2_LO, fe, vl))
    # add(v, u)
    sh, sl = _two_sum(vh, uh)
    th2, tl2 = _two_sum(vl, ul)
    sh, sl = _quick_two_sum(sh, sl + th2)
    z, _ = _quick_two_sum(sh, sl + tl2)
    z = gl.where(ah != ah, ah, z)
    z = gl.where(x == INF, x, z)
    qnan = (bits * 0 + QNAN_BITS).to(gl.float32, bitcast=True)
    z = gl.where(x < -1.0, qnan, z)
    z = gl.where(x == -1.0, -INF, z)
    return gl.where(gl.abs(x) < 5.960464477539063e-08, x, z)
