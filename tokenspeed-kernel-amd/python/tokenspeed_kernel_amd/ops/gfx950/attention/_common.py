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

from tokenspeed_kernel_amd._triton import gl, gluon, tl

_INV_LN2_VALUE = 1.4426950408889634
_INV_LN2 = tl.constexpr(_INV_LN2_VALUE)
_LN2_VALUE = 0.6931471805599453
_LN2 = tl.constexpr(_LN2_VALUE)


def select_kv_splits(*, base_ctas: int, num_pages: int, sm_count: int) -> int:
    """Pick num_kv_splits to balance occupancy against reduce overhead.

    The launch grid is base_ctas * num_kv_splits work-groups. Too few splits
    under-fill the machine at low batch; too many leave each split with a
    handful of pages, so the reduce kernel dominates.

    Return the smaller of two candidate counts: splits_for_occupancy (enough to
    fill ~wave_target waves of CUs) and splits_for_pages (~min_pages_per_split
    pages per split), with the pages candidate clamped to [min_page_splits,
    max_page_splits] so a short context still splits without launching empty work
    and a long one does not over-split where reduce cost outgrows the decode win.
    """
    wave_target = 2
    min_pages_per_split = 2
    min_page_splits = 8
    max_page_splits = 32

    target_ctas = sm_count * wave_target
    splits_for_occupancy = (target_ctas + base_ctas - 1) // base_ctas

    splits_for_pages = num_pages // min_pages_per_split
    min_page_splits = min(min_page_splits, num_pages)
    if splits_for_pages < min_page_splits:
        splits_for_pages = min_page_splits
    if splits_for_pages > max_page_splits:
        splits_for_pages = max_page_splits
    return min(splits_for_occupancy, splits_for_pages)


@gluon.jit
def maximum(a, b, propagate_nan: gl.constexpr = tl.PropagateNan.ALL):
    return gl.maximum(a, b, propagate_nan=propagate_nan)


@gluon.jit
def max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, maximum, keep_dims=keep_dims)


@gluon.constexpr_function
def padded_shared_layout(operand_layout, shape, dtype, is_k_contig):
    # Take only the built-in's padding, not its swizzle: the swizzle scatters the
    # contraction dim across banks, making the LDS write stride non-constant, so it
    # can't lower through async_copy's affine [1, 0] load. Padding-only is affine
    # and DMA-legal.
    # TODO(perf): to also use the swizzle, co-design a matched load layout so the
    # DMA stays legal.
    api = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        operand_layout, shape, dtype, is_k_contig=is_k_contig
    )
    assert api is not None, "no CDNA4 padded shared layout for this operand/dtype"
    pairs = list(api.interval_padding_pairs)
    assert len(pairs) == 1, "expected a single interval padding pair from the built-in"
    return gl.PaddedSharedLayout.with_identity_for(
        [[int(pairs[0][0]), int(pairs[0][1])]], shape, [1, 0]
    )


@gluon.constexpr_function
def attention_layouts(head_dim, block_n, is_fp8, dtype, num_warps, instr_shape):
    mfma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=instr_shape,
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    qk_layout = mfma
    pv_layout = mfma
    # qk_kw is derived from a 128-bit load / dtype bitwidth; pv_kw is tuned.
    qk_kw = 16 if is_fp8 else 8
    pv_kw = 8 if is_fp8 else 4
    q_layout = gl.DotOperandLayout(0, qk_layout, k_width=qk_kw)
    k_layout = gl.DotOperandLayout(1, qk_layout, k_width=qk_kw)
    p_layout = gl.DotOperandLayout(0, pv_layout, k_width=pv_kw)
    v_layout = gl.DotOperandLayout(1, pv_layout, k_width=pv_kw)
    # load_vec = elems/lane (dtype-dependent, == qk_kw); load_threads span HEAD_DIM.
    load_vec = 16 if is_fp8 else 8
    load_threads = head_dim // load_vec
    load_layout = gl.BlockedLayout(
        [1, load_vec], [64 // load_threads, load_threads], [num_warps, 1], [1, 0]
    )
    # store_vec is always 16-bit (128 / 16 = 8) regardless of input dtype.
    store_vec = 8
    store_threads = head_dim // store_vec
    store_layout = gl.BlockedLayout(
        [1, store_vec], [64 // store_threads, store_threads], [num_warps, 1], [1, 0]
    )
    k_smem_layout = padded_shared_layout(
        k_layout, [block_n, head_dim], dtype, is_k_contig=True
    )
    v_smem_layout = padded_shared_layout(
        v_layout, [block_n, head_dim], dtype, is_k_contig=False
    )
    return (
        qk_layout,
        pv_layout,
        q_layout,
        k_layout,
        p_layout,
        v_layout,
        load_layout,
        store_layout,
        k_smem_layout,
        v_smem_layout,
    )


@gluon.aggregate
class InputStrides:
    stride_t: gl.constexpr
    stride_h: gl.constexpr
    stride_d: gl.constexpr

    @gluon.jit
    def offsets(self, token, head, dim):
        return (token * self.stride_t + head * self.stride_h + dim * self.stride_d).to(
            gl.int32
        )
