"""Kimi K3 Triton AttnRes split kernels."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _attnres_partial_kernel(
    blocks_ptr,  # [KB, T, H]
    wp_ptr,  # [H] precomputed rms_w * res_w
    m_ptr,  # [T]
    s_ptr,  # [T]
    acc_ptr,  # [T, H] fp32
    n_blocks,
    n_cols: tl.constexpr,
    stride_bk,
    stride_bt,
    eps,
    BLOCK: tl.constexpr,
):
    """Online-softmax partial over the static block candidates (aux stream).

    Score weights and the running accumulator stay in registers; each block
    row is read from global memory exactly once.
    """
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n_iters: tl.constexpr = (n_cols + BLOCK - 1) // BLOCK
    tl.static_assert(n_iters == 2)

    col0 = offs
    col1 = BLOCK + offs
    mask0 = col0 < n_cols
    mask1 = col1 < n_cols
    wp0 = tl.load(wp_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wp1 = tl.load(wp_ptr + col1, mask=mask1, other=0.0).to(tl.float32)

    acc0 = tl.zeros([BLOCK], tl.float32)
    acc1 = tl.zeros([BLOCK], tl.float32)
    m_run = -float("inf")
    s_run = 0.0
    b = 0
    while b < n_blocks:
        base = blocks_ptr + b * stride_bk + t * stride_bt
        v0 = tl.load(base + col0, mask=mask0, other=0.0).to(tl.float32)
        v1 = tl.load(base + col1, mask=mask1, other=0.0).to(tl.float32)
        sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
        dot = tl.sum(v0 * wp0) + tl.sum(v1 * wp1)
        rsig = tl.math.rsqrt(sq / n_cols + eps)
        logit = dot * rsig
        m_new = tl.maximum(m_run, logit)
        corr = tl.exp(m_run - m_new)
        wgt = tl.exp(logit - m_new)
        acc0 = acc0 * corr + wgt * v0
        acc1 = acc1 * corr + wgt * v1
        s_run = s_run * corr + wgt
        m_run = m_new
        b += 1
    tl.store(acc_ptr + t * n_cols + col0, acc0, mask=mask0)
    tl.store(acc_ptr + t * n_cols + col1, acc1, mask=mask1)
    tl.store(m_ptr + t, m_run)
    tl.store(s_ptr + t, s_run)


@triton.jit
def _attnres_partial_dual_kernel(
    blocks_ptr,  # [KB, T, H]
    wp_a_ptr,  # [H] precomputed side-A rms_w * res_w
    wp_b_ptr,  # [H] side-B product
    m_a_ptr,
    s_a_ptr,
    acc_a_ptr,
    m_b_ptr,
    s_b_ptr,
    acc_b_ptr,
    n_blocks,
    n_cols: tl.constexpr,
    stride_bk,
    stride_bt,
    eps,
    BLOCK: tl.constexpr,
):
    """Two online-softmax partials over the same block sweep (aux stream).

    Side A is this layer's mlp-side mix, side B the next layer's attn-side
    mix. Both consume the identical block-snapshot set, so one kernel pays
    the global reads and the single-CTA latency once for both.
    """
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n_iters: tl.constexpr = (n_cols + BLOCK - 1) // BLOCK
    tl.static_assert(n_iters == 2)

    col0 = offs
    col1 = BLOCK + offs
    mask0 = col0 < n_cols
    mask1 = col1 < n_cols
    wa0 = tl.load(wp_a_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wa1 = tl.load(wp_a_ptr + col1, mask=mask1, other=0.0).to(tl.float32)
    wb0 = tl.load(wp_b_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wb1 = tl.load(wp_b_ptr + col1, mask=mask1, other=0.0).to(tl.float32)

    acc_a0 = tl.zeros([BLOCK], tl.float32)
    acc_a1 = tl.zeros([BLOCK], tl.float32)
    acc_b0 = tl.zeros([BLOCK], tl.float32)
    acc_b1 = tl.zeros([BLOCK], tl.float32)
    m_a = -float("inf")
    s_a = 0.0
    m_b = -float("inf")
    s_b = 0.0
    b = 0
    while b < n_blocks:
        base = blocks_ptr + b * stride_bk + t * stride_bt
        v0 = tl.load(base + col0, mask=mask0, other=0.0).to(tl.float32)
        v1 = tl.load(base + col1, mask=mask1, other=0.0).to(tl.float32)
        sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
        rsig = tl.math.rsqrt(sq / n_cols + eps)
        dot_a = tl.sum(v0 * wa0) + tl.sum(v1 * wa1)
        logit_a = dot_a * rsig
        m_an = tl.maximum(m_a, logit_a)
        corr_a = tl.exp(m_a - m_an)
        wgt_a = tl.exp(logit_a - m_an)
        acc_a0 = acc_a0 * corr_a + wgt_a * v0
        acc_a1 = acc_a1 * corr_a + wgt_a * v1
        s_a = s_a * corr_a + wgt_a
        m_a = m_an
        dot_b = tl.sum(v0 * wb0) + tl.sum(v1 * wb1)
        logit_b = dot_b * rsig
        m_bn = tl.maximum(m_b, logit_b)
        corr_b = tl.exp(m_b - m_bn)
        wgt_b = tl.exp(logit_b - m_bn)
        acc_b0 = acc_b0 * corr_b + wgt_b * v0
        acc_b1 = acc_b1 * corr_b + wgt_b * v1
        s_b = s_b * corr_b + wgt_b
        m_b = m_bn
        b += 1
    tl.store(acc_a_ptr + t * n_cols + col0, acc_a0, mask=mask0)
    tl.store(acc_a_ptr + t * n_cols + col1, acc_a1, mask=mask1)
    tl.store(m_a_ptr + t, m_a)
    tl.store(s_a_ptr + t, s_a)
    tl.store(acc_b_ptr + t * n_cols + col0, acc_b0, mask=mask0)
    tl.store(acc_b_ptr + t * n_cols + col1, acc_b1, mask=mask1)
    tl.store(m_b_ptr + t, m_b)
    tl.store(s_b_ptr + t, s_b)


@triton.jit
def _attnres_combine_kernel(
    prefix_ptr,  # [T, H]
    wp_ptr,  # [H] precomputed rms_w * res_w
    outw_ptr,  # out-norm weight or dummy
    m_ptr,
    s_ptr,
    acc_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    stride_p,
    stride_o,
    eps,
    HAS_OUTNORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold the prefix candidate into the block partial; optional out-norm.

    All operands are read once and stay register-resident.
    """
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    n_iters: tl.constexpr = (n_cols + BLOCK - 1) // BLOCK
    tl.static_assert(n_iters == 2)

    col0 = offs
    col1 = BLOCK + offs
    mask0 = col0 < n_cols
    mask1 = col1 < n_cols

    v0 = tl.load(prefix_ptr + t * stride_p + col0, mask=mask0, other=0.0).to(tl.float32)
    v1 = tl.load(prefix_ptr + t * stride_p + col1, mask=mask1, other=0.0).to(tl.float32)
    wp0 = tl.load(wp_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
    wp1 = tl.load(wp_ptr + col1, mask=mask1, other=0.0).to(tl.float32)
    sq = tl.sum(v0 * v0) + tl.sum(v1 * v1)
    dot = tl.sum(v0 * wp0) + tl.sum(v1 * wp1)
    rsig = tl.math.rsqrt(sq / n_cols + eps)
    logit_p = dot * rsig
    m_b = tl.load(m_ptr + t)
    s_b = tl.load(s_ptr + t)
    m = tl.maximum(m_b, logit_p)
    corr = tl.exp(m_b - m)
    w_p = tl.exp(logit_p - m)
    inv_s = 1.0 / (s_b * corr + w_p)

    a0 = tl.load(acc_ptr + t * n_cols + col0, mask=mask0, other=0.0)
    a1 = tl.load(acc_ptr + t * n_cols + col1, mask=mask1, other=0.0)
    mix0 = ((a0 * corr + w_p * v0) * inv_s).to(tl.bfloat16).to(tl.float32)
    mix1 = ((a1 * corr + w_p * v1) * inv_s).to(tl.bfloat16).to(tl.float32)

    if HAS_OUTNORM:
        mix_sq = tl.sum(mix0 * mix0) + tl.sum(mix1 * mix1)
        rsig_mix = tl.math.rsqrt(mix_sq / n_cols + eps)
        ow0 = tl.load(outw_ptr + col0, mask=mask0, other=0.0).to(tl.float32)
        ow1 = tl.load(outw_ptr + col1, mask=mask1, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + t * stride_o + col0,
            (mix0 * rsig_mix * ow0).to(out_ptr.dtype.element_ty),
            mask=mask0,
        )
        tl.store(
            out_ptr + t * stride_o + col1,
            (mix1 * rsig_mix * ow1).to(out_ptr.dtype.element_ty),
            mask=mask1,
        )
    else:
        tl.store(
            out_ptr + t * stride_o + col0,
            mix0.to(out_ptr.dtype.element_ty),
            mask=mask0,
        )
        tl.store(
            out_ptr + t * stride_o + col1,
            mix1.to(out_ptr.dtype.element_ty),
            mask=mask1,
        )


def attnres_partial(blocks, wp, eps, scratch):
    """Blocks-side online-softmax partial. scratch = (m [T], s [T], acc [T,H] fp32)."""
    KB, T, H = blocks.shape
    m, s_, acc = scratch
    _attnres_partial_kernel[(T,)](
        blocks,
        wp,
        m,
        s_,
        acc,
        KB,
        n_cols=H,
        stride_bk=blocks.stride(0),
        stride_bt=blocks.stride(1),
        eps=eps,
        BLOCK=4096,
        num_warps=8,
    )


def attnres_partial_dual(blocks, wp_a, wp_b, eps, scratch_a, scratch_b):
    """Both mix partials (mlp-side A, next-layer attn-side B) in one sweep.

    Args:
        blocks: ``[KB, T, H]`` block snapshots shared by both sides.
        wp_a/wp_b: precomputed ``rms_w * res_w`` products per side (``[H]``).
        eps: shared RMS epsilon.
        scratch_a/scratch_b: (m [T], s [T], acc [T, H] fp32) per side.
    """
    KB, T, H = blocks.shape
    m_a, s_a, acc_a = scratch_a
    m_b, s_b, acc_b = scratch_b
    _attnres_partial_dual_kernel[(T,)](
        blocks,
        wp_a,
        wp_b,
        m_a,
        s_a,
        acc_a,
        m_b,
        s_b,
        acc_b,
        KB,
        n_cols=H,
        stride_bk=blocks.stride(0),
        stride_bt=blocks.stride(1),
        eps=eps,
        BLOCK=4096,
        num_warps=8,
    )


def attnres_combine(prefix, wp, out_norm_w, eps, scratch, out):
    """Merge the prefix candidate into the partial; optional fused out-norm.

    Args:
        prefix: ``[T, H]`` residual stream.
        scratch: (m, s, acc) from :func:`attnres_partial`.
        out: ``[T, H]`` mixed (and out-normed) hidden destination.

    Returns:
        ``out``.
    """
    T, H = prefix.shape
    m, s_, acc = scratch
    _attnres_combine_kernel[(T,)](
        prefix,
        wp,
        out_norm_w if out_norm_w is not None else wp,
        m,
        s_,
        acc,
        out,
        n_cols=H,
        stride_p=prefix.stride(0),
        stride_o=out.stride(0),
        eps=eps,
        HAS_OUTNORM=out_norm_w is not None,
        BLOCK=4096,
        num_warps=8,
    )
    return out
