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

"""Single source of truth for the gfx950 A4W4 CDNA4 MXFP4 scale layout.

Both scale producers and consumers must agree on the CDNA4 MXFP4 scale-swizzle
parameters and byte permutation:

* B-scales (weights): swizzled at load time by the weight preprocessor
  (``mxfp4_gfx950_preprocess._swizzle_mxfp4`` via
  :func:`swizzle_cdna4_mxfp4_scale`).
* A-scales (activations): emitted in token order by the MXFP4 activation
  quantizer and re-gathered into sorted-route order by
  ``gluon_a4w4_gfx950.scale.gather_package_cdna4_scale``.
* Consumers: the package stage kernels (``prefill_stage1/2``,
  ``decode_stage1/2``) address the scales with the matching CDNA4 MFMA scale
  layout (``gl.amd.cdna4.get_mfma_scale_layout``).

Historically these constants were duplicated (under different names) across the
preprocessor and the activation-scale gather. Centralizing them here makes a
scale-layout change a single-file edit: update the constants / permutation
below, keep the stage-kernel scale addressing in sync, and the bit-exact
package tests gate correctness.

This module is intentionally a leaf (only depends on ``torch``) so every
producer/consumer can import it without pulling in the Triton stage kernels.


Why a swizzle at all
--------------------
An MXFP4 tensor carries one e8m0 block scale per 32 elements along K. The
gfx950 ``v_mfma_scale_f32_*`` instructions consume those block scales from a
*fixed distribution across the 64 lanes and their VGPRs* -- each lane must
already hold the scale bytes for the (row, K-block) coordinates it is
responsible for. A plain row-major ``(N, K//32)`` scale array does not land in
those positions under a coalesced load, so we permute the bytes at load time
(the "swizzle") such that a single direct global->VGPR gather drops every byte
where the MFMA expects it, with no in-kernel shuffle.

The permutation is dictated entirely by the MFMA instruction shape the stage
kernels issue. We support exactly one shape, ``mfma16`` (16x16x128), so there
is a single scale swizzle (:func:`swizzle_cdna4_mxfp4_scale`). Producer and
consumer are hard-coupled: the byte order this swizzle bakes into memory must
match the ``get_mfma_scale_layout`` the stage kernels address with, so any
change here must be mirrored in the kernels' scale addressing.

The swizzled scales feed a direct global->VGPR gather in the stage kernels, so
the scale bytes arrive in registers already distributed for the MFMA with no
in-kernel shuffle.
"""

from __future__ import annotations

import torch

# MXFP4 microblock: one e8m0 block scale per 32 elements along the K (reduction)
# axis. Fixed by the OCP MXFP4 format.
MXFP4_BLOCK = 32

# CDNA4 scale-swizzle block sizes.
#   N (non-K / row) preshuffle block: rows are grouped/padded to multiples of 32.
#   K (scale) swizzle alignment: the per-32-element scale columns are grouped
#   and padded to multiples of 8.
CDNA4_SCALE_N_BLOCK = 32
CDNA4_SCALE_K_BLOCK = 8

# MFMA non-K dimension used by the a4w4 stage kernels
# (``v_mfma_scale_f32_16x16x128_f8f6f4``). The scale swizzle below is fixed to
# this shape.
MFMA_NONK_DIM = 16

__all__ = [
    "MXFP4_BLOCK",
    "CDNA4_SCALE_N_BLOCK",
    "CDNA4_SCALE_K_BLOCK",
    "MFMA_NONK_DIM",
    "swizzle_cdna4_mxfp4_scale",
]


def swizzle_cdna4_mxfp4_scale(scale: torch.Tensor) -> torch.Tensor:
    """CDNA4 MXFP4 block-scale swizzle for the 16x16x128 MFMA (canonical).

    This is the single canonical definition of the B-scale (and, structurally,
    the A-scale) CDNA4 swizzle used by the runtime. Any change here must be
    mirrored in the stage kernels' scale addressing.

    Input:  MXFP4 block scales with the logical ``(..., N, K // 32)`` layout
            (last two dims are rows and per-block scale columns).
    Output: CDNA4-swizzled scales with ``stride(-2) == 1`` (the byte order the
            CDNA4 16x16x128 MFMA scaled-dot path expects for the B operand).

    Lane 0 access block (what one lane reads)
    -----------------------------------------
    For a 32-row x 8-K-block tile, the swizzle interleaves rows in 16-apart
    pairs ``(n, n + 16)`` and K-blocks in 4-apart pairs ``(kb, kb + 4)`` so the
    first eight bytes a coalesced load anchored at lane 0 sees are::

        mem[0] = (n=0,  kb=0)     mem[4] = (n=1,  kb=0)
        mem[1] = (n=16, kb=0)     mem[5] = (n=17, kb=0)
        mem[2] = (n=0,  kb=4)     mem[6] = (n=1,  kb=4)
        mem[3] = (n=16, kb=4)     mem[7] = (n=17, kb=4)

    i.e. lane 0 ends up holding the scale bytes for the row/K-block quarters the
    16x16x128 ``mfma_scaled`` op reduces over -- two 16-row halves and two
    K-quarters spaced 4 blocks apart -- with no in-kernel shuffle. That
    (row-pair, K-quarter-pair) grouping is exactly the permutation encoded by
    ``view(N//32, 2, 16, K//8, 2, 4, 1).permute(N//32, K//8, 4, 16, 2, 2, 1)``
    below.
    """
    if scale.ndim < 2:
        raise ValueError("MXFP4 scale tensor must have at least 2 dimensions")
    scale = scale.transpose(-2, -1).contiguous()
    *leading_shape, k_scale, n = scale.shape
    leading = 1
    for dim in leading_shape:
        leading *= dim
    k_scale_pad = (k_scale + CDNA4_SCALE_K_BLOCK - 1) // CDNA4_SCALE_K_BLOCK
    k_scale_pad *= CDNA4_SCALE_K_BLOCK
    n_pad = (n + CDNA4_SCALE_N_BLOCK - 1) // CDNA4_SCALE_N_BLOCK
    n_pad *= CDNA4_SCALE_N_BLOCK

    scale = scale.mT.contiguous().mT
    scale = torch.nn.functional.pad(scale, (0, n_pad - n, 0, k_scale_pad - k_scale))
    scale = scale.transpose(-1, -2)
    scale = scale.reshape(leading, n_pad, k_scale_pad)
    scale = scale.view(leading, n_pad // 32, 2, 16, k_scale_pad // 8, 2, 4, 1)
    scale = scale.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    scale = scale.reshape(leading, n_pad // 32, k_scale_pad * 32)
    scale = scale.transpose(-1, -2)
    assert scale.stride(-2) == 1
    return scale
