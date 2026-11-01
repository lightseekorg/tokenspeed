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

from tokenspeed_kernel_amd._triton import gl, gluon, gluon_builtin, tl

_INV_LN2_VALUE = 1.4426950408889634
_INV_LN2 = tl.constexpr(_INV_LN2_VALUE)
_LN2_VALUE = 0.6931471805599453
_LN2 = tl.constexpr(_LN2_VALUE)

# An FP8 attention kernel converts its softmax probabilities to FP8 for the
# P @ V MFMA. Probabilities lie in [0, 1], where e4m3 has 3 mantissa bits only
# down to 2^-6 and rounds anything at or below 2^-10 to zero, while the row
# sum that normalizes the output keeps them in fp32. So the kernels subtract
# this from the running max before exponentiating, making each base-2 exponent
# 8 larger and each probability 2^8 times larger: the largest is 256, and the
# row sum carries the same factor, so the normalized output is unchanged.
# Base-2 LSE outputs and sinks account for it (fp8_lse_shift). e5m2 has more
# range, so the same scale is safe for it.
FP8_PROBABILITY_LOG2_SCALE = tl.constexpr(8.0)

# Upper bound of select_kv_splits. Reduce kernels take the split count at
# runtime and must handle any value up to this bound.
MAX_KV_SPLITS = 32


@gluon_builtin
def _mfma_unscaled_fp8(a, b, acc, *, _semantic):
    # None scales select K=64 FP8 MFMA without block-scale operands.
    # cdna4.mfma selects K=16 for this layout.
    fmt = "e4m3" if a.dtype == gl.float8e4nv else "e5m2"
    output = _semantic.dot_scaled(
        a,
        None,
        fmt,
        b,
        None,
        fmt,
        acc,
        fast_math=False,
        lhs_k_pack=True,
        rhs_k_pack=True,
        out_dtype=gl.float32,
    )
    return gl.tensor(output.handle, acc.type)


def reject_block_scales(kernel: str, **scales) -> None:
    """Raise if block scales reach a kernel that reads FP8 at unit scale.

    The registry never routes a block-scaled cache to these kernels; this
    keeps a direct caller from having its scales ignored.
    """
    names = [name for name, scale in scales.items() if scale is not None]
    if names:
        raise ValueError(f"{kernel} reads FP8 at unit scale, got {', '.join(names)}")


def select_kv_splits(
    *, base_ctas: int, num_pages: int, sm_count: int, max_splits: int = MAX_KV_SPLITS
) -> int:
    """Pick num_kv_splits to balance occupancy against reduce overhead.

    The launch grid is base_ctas * num_kv_splits work-groups. Too few splits
    under-fill the machine at low batch; too many leave each split with a
    handful of pages, so the reduce kernel dominates.

    Return the smaller of two candidate counts: splits_for_occupancy (enough to
    fill ~wave_target waves of CUs) and splits_for_pages (~min_pages_per_split
    pages per split), with the pages candidate clamped to [min_page_splits,
    max_page_splits] so a short context still splits without launching empty work
    and a long one does not over-split where reduce cost outgrows the decode win.

    max_page_splits is max_splits, which defaults to MAX_KV_SPLITS, the bound
    the runtime-count reduce kernels support; a caller whose reduce takes the
    split count as a constexpr may raise it.
    """
    wave_target = 2
    min_pages_per_split = 2
    min_page_splits = 8
    max_page_splits = max_splits

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
def fp8_probability_shift(m_scaled, IS_FP8: gl.constexpr):
    """What to subtract from the base-2 logits before exp2: the running max
    m_scaled (base-2 units), less FP8_PROBABILITY_LOG2_SCALE for FP8 so the
    probabilities come out 2^8 larger."""
    if IS_FP8:
        return m_scaled - FP8_PROBABILITY_LOG2_SCALE
    return m_scaled


@gluon.jit
def fp8_lse_shift(m_i, SM_SCALE: gl.constexpr, IS_FP8: gl.constexpr):
    """The running max m_i in base-2 units, less the FP8 probability scale
    the row sum l_i carries, so that fp8_lse_shift(m_i) + log2(l_i) is the
    base-2 LSE."""
    return fp8_probability_shift(m_i * SM_SCALE, IS_FP8)


@gluon.jit
def maximum(a, b, propagate_nan: gl.constexpr = tl.PropagateNan.ALL):
    return gl.maximum(a, b, propagate_nan=propagate_nan)


@gluon.jit
def max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, maximum, keep_dims=keep_dims)


# ===-----------------------------------------------------------------------===#
# Warp-pipeline helpers
#
# Shared by the 8-wave kernels that split their loop into
# gl.amd.warp_pipeline_stage clusters.
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _keep_in_cluster(x):
    # An empty side-effecting asm that reads and redefines x. Neither the IR
    # optimizer nor MachineSink moves it across the cluster barriers, so the
    # ops producing x stay in the cluster that computes them instead of drifting
    # toward their consumer in a later cluster. "=v,0" ties the result to the
    # input register, so no instruction is emitted; narrower values go in
    # groups that fill one 32-bit VGPR per asm operand.
    pack: gl.constexpr = 32 // x.dtype.primitive_bitwidth
    return gl.inline_asm_elementwise(
        asm="",
        constraints="=v,0",
        args=[x],
        dtype=x.dtype,
        is_pure=False,
        pack=pack,
    )


@gluon.jit
def _split_columns(x):
    # Register-only column halves: each half keeps x's MFMA layout.
    half: gl.constexpr = x.shape[1] // 2
    lo = gl.amd.slice(x, [x.shape[0], half], [0, 0])
    hi = gl.amd.slice(x, [x.shape[0], half], [0, half])
    return lo, hi


@gluon.jit
def _join_columns(lo, hi):
    # Inverse of _split_columns, also register-only. join/permute/reshape
    # places lo's columns before hi's; assert_trivial fails the build if the
    # result would need data movement.
    layout: gl.constexpr = lo.type.layout
    shape: gl.constexpr = [lo.shape[0], lo.shape[1] + hi.shape[1]]
    x = gl.join(lo, hi).permute([0, 2, 1]).reshape(shape)
    return gl.convert_layout(x, layout, assert_trivial=True)


@gluon.jit
def _wave_maxima_unchanged(m_old, m_new):
    # Returns 1 when no row of the wave moved its maximum, else 0, as one
    # wave-uniform SGPR value, so a branch on it is a scalar branch that skips
    # the rescale for the whole wave:
    #   v_cmp_eq_f32  VCC bit set for each lane whose maximum is unchanged
    #   s_cmp_eq_u64  SCC = every active lane set its bit (VCC == EXEC)
    #   s_cselect     SGPR result = SCC ? 1 : 0
    # "=s" asks for an SGPR result; VCC and SCC are declared clobbered.
    return gl.inline_asm_elementwise(
        asm="v_cmp_eq_f32_e64 vcc, $1, $2\n"
        "s_cmp_eq_u64 vcc, exec\n"
        "s_cselect_b32 $0, 1, 0",
        constraints="=s,v,v,~{vcc},~{scc}",
        args=[m_old, m_new],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


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
