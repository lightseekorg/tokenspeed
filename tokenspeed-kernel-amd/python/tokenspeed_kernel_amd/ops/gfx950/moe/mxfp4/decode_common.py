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


"""Direct CDNA4 MFMA primitives shared by the gfx950 A4W4 decode stages.

Tile addressing, MXFP4 operand loads, and the scaled-MFMA op used by both
production decode stages (``decode_stage1.py`` / ``decode_stage2.py``). They
assume the intermediate rows are in (token, topk-slot) order, so no ragged
metadata / scatter indices are consumed. ``situ_decode.py`` reuses the CDNA4
scale-offset helper for its in-situ dequantization path.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def _gluon_dot_preshuffled_w_offset(w_expert_off, k_pack, n_col, n_phys):
    """Return flat byte offset for the 128x128 Gluon-dot W layout."""
    k_in_block = k_pack % 128
    n_in_block = n_col % 128

    k_within = k_in_block % 16
    k_quad = (k_in_block // 16) % 4
    k_block = k_in_block // 64
    n_in_sub = n_in_block % 16
    n_block = n_in_block // 16

    in_tile = (
        n_block.to(gl.int64) * 2048
        + k_block.to(gl.int64) * 1024
        + k_quad.to(gl.int64) * 256
        + n_in_sub.to(gl.int64) * 16
        + k_within.to(gl.int64)
    )
    n_tiles = n_phys // 128
    tile_id = (k_pack // 128).to(gl.int64) * n_tiles + (n_col // 128).to(gl.int64)
    return w_expert_off + tile_id * (128 * 128) + in_tile


@gluon.jit
def _cdna4_swizzled_mxfp4_scale_offset(
    scale_expert_off,
    n_col,
    k_scale,
    stride_slin,
    stride_snb,
):
    """Return byte offset for CDNA4-swizzled e8m0 scales.

    The preprocessor stores scales as ``(E, K_scale_padded * 32, N_padded / 32)``
    with ``stride(-2) == 1``.  This is the scalar form of the unswizzle used by
    the reference MFMA path.
    """
    n_block = n_col // 32
    n_mix = (n_col % 16) * 4 + ((n_col % 32) // 16)
    k_lin = (
        (k_scale // 8).to(gl.int64) * 256
        + (k_scale % 4).to(gl.int64) * 64
        + ((k_scale % 8) // 4).to(gl.int64) * 2
        + n_mix.to(gl.int64)
    )
    return scale_expert_off + k_lin * stride_slin + n_block.to(gl.int64) * stride_snb


@gluon.constexpr_function
def _direct_mxfp4_mfma_layouts(m_dup, block_n, block_k_scale):
    mfma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1]
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [m_dup, block_k_scale])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [block_n, block_k_scale])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _direct_mxfp4_load_tile(
    kt,
    ak,
    bk,
    ask,
    bsk,
    am,
    asm,
    x_ptr,
    x_scale_ptr,
    w_ptr,
    w_scale_ptr,
    x_row_off,
    w_expert_off,
    s_expert_off,
    n_cols,
    n_cols_s,
    x_scale_row,
    stride_xk,
    stride_xslin,
    stride_xsnb,
    stride_slin,
    stride_snb,
    N_PHYS,
    K_DIM,
    N_DIM,
    K_PACKED: gl.constexpr,
    BLOCK_K_PACKED: gl.constexpr,
    BLOCK_K_SCALE: gl.constexpr,
):
    """Load one direct MXFP4xMXFP4 K tile into MFMA operand layouts."""
    k_pack_a = kt * BLOCK_K_PACKED + ak
    k_pack_b = kt * BLOCK_K_PACKED + bk
    k_scale_a = kt * BLOCK_K_SCALE + ask
    k_scale_b = kt * BLOCK_K_SCALE + bsk
    a_off = x_row_off + k_pack_a.to(gl.int64) * stride_xk + am.to(gl.int64) * 0
    b_off = _gluon_dot_preshuffled_w_offset(w_expert_off, k_pack_b, n_cols, N_PHYS)
    a_scale_off = _cdna4_swizzled_mxfp4_scale_offset(
        0,
        x_scale_row + asm * 0,
        k_scale_a,
        stride_xslin,
        stride_xsnb,
    )
    b_scale_off = _cdna4_swizzled_mxfp4_scale_offset(
        s_expert_off,
        n_cols_s,
        k_scale_b,
        stride_slin,
        stride_snb,
    )
    a = gl.amd.cdna4.buffer_load(
        ptr=x_ptr,
        offsets=a_off.to(gl.int32),
        mask=k_pack_a < K_PACKED,
        other=0,
    )
    b = gl.amd.cdna4.buffer_load(
        ptr=w_ptr,
        offsets=b_off.to(gl.int32),
        mask=(n_cols < N_DIM) & (k_pack_b < K_PACKED),
        other=0,
    )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=x_scale_ptr,
        offsets=a_scale_off.to(gl.int32),
        mask=k_scale_a < (K_DIM // 32),
        other=127,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=w_scale_ptr,
        offsets=b_scale_off.to(gl.int32),
        mask=(n_cols_s < N_DIM) & (k_scale_b < (K_DIM // 32)),
        other=127,
    )
    return a, b, a_scale, b_scale


@gluon.jit
def _direct_mxfp4_mfma(acc, a, b, a_scale, b_scale):
    return gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_scale,
        a_format="e2m1",
        b=b,
        b_scale=b_scale,
        b_format="e2m1",
        acc=acc,
    )
