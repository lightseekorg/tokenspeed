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

"""The residual boundary's two RMSNorms folded into their consumers (gfx950).

At a residual boundary, the post-norm of the branch output ``b``, the gated
residual ``h' = alpha * h + beta * post_norm(b)`` and the next layer's
pre-norm of ``h'`` need two row-wide sums of squares. Instead of the
normalized rows, the producers of ``b`` publish four partial sums per row
(with ``b' = b * post``: ``sum b'^2``, ``B = sum (alpha h)(beta b')``,
``A = sum (alpha h)^2``, ``C = sum (beta b')^2``). Each consumer adds every
producer's sums and builds its own columns of the normalized input ``x``,
so no consumer waits for a second exchange:

    inv1 = rsqrt(sum b'^2 / WIDTH + POST_EPS)
    inv2 = rsqrt((A + 2 inv1 B + inv1^2 C) / WIDTH + PRE_EPS)
    h'   = fma(beta, b * post * inv1, alpha * h)
    x    = h' * inv2 * pre

The operations and their order are the regular residual-boundary kernel's.
"""

from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = [
    "deferred_norm_rows_gfx950",
    "deferred_norm_scales_gfx950",
    "deferred_norm_terms_gfx950",
]


@gluon.jit
def deferred_norm_terms_gfx950(b, h, alpha, beta, post):
    """A producer's per-element terms of the four sums, elementwise:
    ``[..., 4]`` = ``(b'^2, (alpha h)(beta b'), (alpha h)^2, (beta b')^2)``
    with ``b' = b * post`` (FP32). The producer sums them over its columns
    in a fixed order and publishes ``sum b'^2, B, A, C``."""
    weighted = b.to(gl.float32) * post.to(gl.float32)
    alpha_h = alpha * h
    beta_b = beta * weighted
    # join(join(p, q), join(r, s)) orders the trailing 2 x 2 as p, r, q, s.
    terms = gl.join(
        gl.join(weighted * weighted, alpha_h * alpha_h),
        gl.join(alpha_h * beta_b, beta_b * beta_b),
    )
    return gl.reshape(terms, list(weighted.shape) + list([4]))


@gluon.jit
def deferred_norm_scales_gfx950(
    sums,
    WIDTH: gl.constexpr,
    POST_EPS: gl.constexpr,
    PRE_EPS: gl.constexpr,
):
    """Each row's two inverse RMS scales from its producers' sums.

    ``sums`` is FP32 ``[rows, producers, 4]`` (``sum b'^2, B, A, C`` last),
    in a layout whose producers lie across one wave's lanes and whose four
    sums lie in one thread, so each row's producers add in one wave, in the
    same lane order for every row. Returns ``(inv1, inv2)``, ``[rows]`` in
    ``SliceLayout(1, SliceLayout(1, sums.layout))``.
    """
    totals = gl.sum(sums, 1)  # [rows, 4]
    which = gl.arange(0, 4, gl.SliceLayout(0, totals.type.layout))[None, :]
    t0 = gl.sum(gl.where(which == 0, totals, 0.0), 1)
    t1 = gl.sum(gl.where(which == 1, totals, 0.0), 1)
    t2 = gl.sum(gl.where(which == 2, totals, 0.0), 1)
    t3 = gl.sum(gl.where(which == 3, totals, 0.0), 1)
    inv1 = gl.rsqrt(t0 / WIDTH + POST_EPS)
    inv2 = gl.rsqrt((t2 + 2.0 * inv1 * t1 + inv1 * inv1 * t3) / WIDTH + PRE_EPS)
    return inv1, inv2


@gluon.jit
def deferred_norm_rows_gfx950(b, h, alpha, beta, post, pre, inv1, inv2):
    """Columns of the residual ``h'`` and the normalized input ``x`` (FP32),
    elementwise in the operands' common layout (scales broadcast by the
    caller): ``h' = fma(beta, b * post * inv1, alpha * h)``,
    ``x = h' * inv2 * pre``. ``b`` is the branch value (BF16 or FP32);
    ``post`` and ``pre`` are the norms' weights with their offsets added
    (``weight + offset`` rounded to BF16, as the regular boundary rounds
    it; BF16 operands halve their registers and convert exactly). Returns
    ``(h', x)``."""
    hp = gl.fma(beta, b.to(gl.float32) * post.to(gl.float32) * inv1, alpha * h)
    return hp, hp * inv2 * pre.to(gl.float32)
