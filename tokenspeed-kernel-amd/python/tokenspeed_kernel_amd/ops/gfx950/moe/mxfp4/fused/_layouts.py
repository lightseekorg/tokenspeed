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

"""Constexpr layout factories and small shared Gluon device helpers used
across the fused MXFP4 MoE kernels."""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon

# ---------------------------------------------------------------------------
# Layout factories (gluon constexpr functions)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _store_layout(
    num_warps: int,
    block_m: int = 0,
    w_via_vgpr: bool = False,
    use_narrow_n_layout: bool = False,
):
    # Mirrors the warps_m policy in get_mfma_layout so the MFMA acc
    # and store layouts stay convert-compatible.
    if w_via_vgpr and num_warps >= 4:
        warps_m = 2
    elif block_m and block_m <= 32 and num_warps >= 4:
        warps_m = 1
    else:
        warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    # Selected W-via-VGPR combine routes store 16 contiguous N values per
    # thread instead of 32; profiling found this lower-resource store layout
    # faster for those shapes.
    if use_narrow_n_layout and w_via_vgpr and block_m >= 64 and num_warps >= 4:
        return gl.BlockedLayout([1, 16], [8, 8], [warps_m, warps_n], [1, 0])
    if w_via_vgpr and block_m >= 128 and num_warps >= 4:
        return gl.BlockedLayout([1, 16], [4, 16], [warps_m, warps_n], [1, 0])
    return gl.BlockedLayout([1, 32], [8, 8], [warps_m, warps_n], [1, 0])


@gluon.constexpr_function
def _load_layout(
    block_k: int,
    block_nonk: int,
    num_warps: int,
    order: list[int] = [1, 0],
    elem_bits: int = 8,
):
    # CDNA4 direct-to-LDS coalesce: K_PER_THREAD * elem_bits <= 128.
    max_vec = max(1, 128 // elem_bits)
    K_PER_THREAD: gl.constexpr = min(max_vec, block_k)
    LANES_K = block_k // K_PER_THREAD
    LANES_NONK = 64 // LANES_K
    NONK_PER_WARP = LANES_NONK
    if block_nonk >= NONK_PER_WARP:
        WARPS_NONK = block_nonk // NONK_PER_WARP
        if WARPS_NONK > num_warps:
            WARPS_NONK = num_warps
        WARPS_K = num_warps // WARPS_NONK
    else:
        # Narrow tile: more lanes on K so warps_K * warps_NONK == num_warps.
        WARPS_NONK = 1
        WARPS_K = num_warps
    if order == [1, 0]:
        regs = [1, K_PER_THREAD]
        lanes = [LANES_NONK, LANES_K]
        warps = [WARPS_NONK, WARPS_K]
    else:
        regs = [K_PER_THREAD, 1]
        lanes = [LANES_K, LANES_NONK]
        warps = [WARPS_K, WARPS_NONK]
    return gl.BlockedLayout(regs, lanes, warps, order)


# ---------------------------------------------------------------------------
# Software-pipelined Gluon MoE kernel
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _swiglu_split_layout(
    block_m: int, block_n_full: int, num_warps: int
) -> gl.constexpr:
    THREADS_PER_WARP = 64  # CDNA4 wavefront size.
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, THREADS_PER_WARP // 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


@gluon.jit
def _swiglu_reduce(
    acc,
    alpha: gl.constexpr,
    limit: gl.constexpr,
    beta: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    MMA: gl.constexpr,
):
    BLOCK_M: gl.constexpr = acc.shape[0]
    BLOCK_N_FULL: gl.constexpr = acc.shape[1]
    SPLIT_LAYOUT: gl.constexpr = _swiglu_split_layout(
        BLOCK_M, BLOCK_N_FULL, gl.num_warps()
    )
    acc = gl.convert_layout(acc, SPLIT_LAYOUT)
    reshaped = acc.reshape((BLOCK_M, OUT_BLOCK_N, 2))
    gate, linear = gl.split(reshaped)
    if limit > 0.0:
        gate = gl.minimum(gate, limit)
        linear = gl.clamp(linear, -limit, limit)
    s = gate / (1.0 + gl.exp(-alpha * gate))
    return s * (linear + beta)


@gluon.jit
def _situ_reduce(
    acc,
    beta: gl.constexpr,
    linear_beta: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
):
    block_m: gl.constexpr = acc.shape[0]
    block_n_full: gl.constexpr = acc.shape[1]
    split_layout: gl.constexpr = _swiglu_split_layout(
        block_m, block_n_full, gl.num_warps()
    )
    acc = gl.convert_layout(acc, split_layout).to(gl.bfloat16).to(gl.float32)
    gate, linear = gl.split(acc.reshape((block_m, OUT_BLOCK_N, 2)))
    gate = beta * gl.extra.libdevice.tanh(gate / beta) / (1.0 + gl.exp(-gate))
    linear = linear_beta * gl.extra.libdevice.tanh(linear / linear_beta)
    return gate * linear


# ---------------------------------------------------------------------------
# Scaled MFMA MoE kernel (mxfp4 / fp8 + e8m0 block scales)
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def get_mfma_layout(
    num_warps: int,
    use_mfma_scaled: bool,
    scale_preshuffle: bool = False,
    block_m: int = 0,
    w_via_vgpr: bool = False,
) -> gl.constexpr:
    # CDNA4 (gfx950): scaled MFMA = 16x16x128 (mxfp/fp8); regular = 16x16x32.
    # ``[2, 2]`` warps_per_cta split keeps W DotOperand per warp at
    # half the ``[num_warps, 1]`` footprint -- the latter spills VGPRs
    # at BN=256. ``w_via_vgpr`` forces ``warps_m=2`` because the host-
    # preshuffled ``LOAD_W_LAYOUT`` assumes that split for the
    # ``assert_trivial=True`` convert; BM<=32 small-tile decode prefers
    # ``warps_m=1`` to keep the fundamental block from over-filling M.
    assert num_warps in (4, 8), "MI355 MoE kernel currently supports 4 or 8 warps."
    if w_via_vgpr and num_warps >= 4:
        warps_m = 2
    elif block_m and block_m <= 32 and num_warps >= 4:
        warps_m = 1
    else:
        warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    instr_shape = [16, 16, 128] if use_mfma_scaled else [16, 16, 32]
    # tpw=[2,2] required when scales preshuffle through LDS (the 5-D
    # unswizzle view absorbs one 2x2 MFMA block per warp per K-iter).
    tiles_per_warp = [2, 2] if scale_preshuffle else [1, 1]
    return gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=instr_shape,
        transposed=True,
        warps_per_cta=[warps_m, warps_n],
        tiles_per_warp=tiles_per_warp,
    )


@gluon.constexpr_function
def _scale_async_blocked_layout(
    BLOCK_NONK_PS: gl.constexpr, BLOCK_K_PS: gl.constexpr, NUM_WARPS: gl.constexpr
):
    vec = 4
    lanes_k = max(1, min(64, BLOCK_K_PS // vec))
    lanes_nonk = max(1, 64 // lanes_k)
    warps_nonk = max(1, min(NUM_WARPS, BLOCK_NONK_PS // lanes_nonk))
    warps_k = max(1, NUM_WARPS // warps_nonk)
    return gl.BlockedLayout(
        [1, vec],
        [lanes_nonk, lanes_k],
        [warps_nonk, warps_k],
        [1, 0],
    )


@gluon.jit
def _xcd_chiplet_swizzle(pid, num_pids, XCD_SWIZZLE: gl.constexpr):
    if XCD_SWIZZLE == 1:
        return pid
    pids_per_xcd = num_pids // XCD_SWIZZLE
    extra = num_pids % XCD_SWIZZLE
    xcd = pid % XCD_SWIZZLE
    local = pid // XCD_SWIZZLE
    return xcd * pids_per_xcd + gl.minimum(xcd, extra) + local


@gluon.jit
def _group_m_swizzle(
    pid_mn,
    grid_m,
    grid_n,
    GROUP_M: gl.constexpr,
):
    if GROUP_M == 1:
        pid_m = pid_mn // grid_n
        pid_n = pid_mn % grid_n
    else:
        width = GROUP_M * grid_n
        group_id = pid_mn // width
        group_size = gl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        intra = pid_mn % width
        pid_m = group_id * GROUP_M + (intra % group_size)
        pid_n = intra // group_size
    return pid_m, pid_n


@gluon.jit
def _load_w_scale_tile_direct_cdna4(
    WScale,
    expert,
    kt,
    off_n,
    stride_wse,
    stride_wsk,
    stride_wsn,
    cfg,
):
    """Load W e8m0 scales in AITER's physical CDNA4-swizzled layout."""
    BLOCK_N: gl.constexpr = cfg.BLOCK_N
    BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
    BLOCK_N_PS: gl.constexpr = cfg.BLOCK_N_PRESHUFFLED
    BLOCK_K_S_PS: gl.constexpr = cfg.BLOCK_K_SCALE_PRESHUFFLED
    LW_S: gl.constexpr = cfg.load_layout_w_scale

    offs_ws_n = gl.arange(0, BLOCK_N_PS, layout=gl.SliceLayout(1, LW_S))[:, None]
    offs_ws_k = gl.arange(0, BLOCK_K_S_PS, layout=gl.SliceLayout(0, LW_S))[None, :]
    rows_n_scale = off_n // cfg.PRESHUFFLE_FACTOR + offs_ws_n
    scale_k_base = kt * BLOCK_K_S_PS
    raw_off = (
        expert.to(gl.int64) * stride_wse
        + (scale_k_base + offs_ws_k).to(gl.int64) * stride_wsk
        + rows_n_scale.to(gl.int64) * stride_wsn
    )
    raw = gl.amd.cdna4.buffer_load(
        ptr=WScale, offsets=raw_off.to(gl.int32), cache=".cg"
    )

    raw_7d = raw.reshape((BLOCK_N_PS, BLOCK_K_SCALE // 8, 4, 16, 2, 2, 1))
    raw_perm = raw_7d.permute((0, 5, 3, 1, 4, 2, 6))
    logical = raw_perm.reshape((BLOCK_N, BLOCK_K_SCALE))
    return gl.convert_layout(logical, cfg.layout_w_scale)


@gluon.jit
def _moe_partial_reduce(
    Partial,
    Out,
    M,
    N,
    stride_pk,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    SPLIT_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Sum SPLIT_K partials per (m, n) into Out in one launch.

    Shared by the warp-decode split-K stage2 ([SPLIT_K, M, N] partials) and the
    medium-decode top-k combine (consecutive-row partials, mapped by passing
    stride_pk = row stride and stride_pm = TOPK * row stride). The float32 cast
    is a no-op for f32 partials and upcasts bf16 combine partials.
    """
    pid = gl.program_id(axis=0)
    num_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n
    LAYOUT: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=LAYOUT)
    bound = (pid_m < M) & (n < N)
    acc = gl.zeros([BLOCK_N], gl.float32, layout=LAYOUT)
    for k in gl.static_range(SPLIT_K):
        acc += gl.load(
            Partial
            + k * stride_pk
            + pid_m.to(gl.int64) * stride_pm
            + n.to(gl.int64) * stride_pn,
            mask=bound,
            other=0.0,
        ).to(gl.float32)
    gl.store(
        Out + pid_m.to(gl.int64) * stride_om + n.to(gl.int64) * stride_on,
        acc.to(Out.dtype.element_ty),
        mask=bound,
    )


@gluon.jit
def _moe_partial_reduce_shared(
    Partial,
    Out,
    SharedInput,
    SharedWeight,
    SharedOut,
    M,
    N,
    stride_pk,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    stride_sim,
    stride_sik,
    stride_som,
    stride_son,
    SPLIT_K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_REDUCE_PROGRAMS: gl.constexpr,
    NUM_SHARED_PID_N: gl.constexpr,
    SHARED_BLOCK_N: gl.constexpr,
):
    """Combine routed experts and compute K3's shared down projection."""
    pid = gl.program_id(axis=0)
    if pid < NUM_REDUCE_PROGRAMS:
        num_n = gl.cdiv(N, BLOCK_N)
        pid_m = pid // num_n
        pid_n = pid % num_n
        layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
        n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
        bound = (pid_m < M) & (n < N)
        acc = gl.zeros([BLOCK_N], gl.float32, layout=layout)
        for k in gl.static_range(SPLIT_K):
            acc += gl.load(
                Partial
                + k * stride_pk
                + pid_m.to(gl.int64) * stride_pm
                + n.to(gl.int64) * stride_pn,
                mask=bound,
                other=0.0,
            ).to(gl.float32)
        gl.store(
            Out + pid_m.to(gl.int64) * stride_om + n.to(gl.int64) * stride_on,
            acc.to(Out.dtype.element_ty),
            mask=bound,
        )
    else:
        shared_pid = pid - NUM_REDUCE_PROGRAMS
        shared_token = shared_pid // NUM_SHARED_PID_N
        shared_pid_n = shared_pid % NUM_SHARED_PID_N
        shared_layout: gl.constexpr = gl.BlockedLayout(
            [SHARED_BLOCK_N, 8], [1, 64], [1, 1], [1, 0]
        )
        shared_n_layout: gl.constexpr = gl.SliceLayout(1, shared_layout)
        shared_k_layout: gl.constexpr = gl.SliceLayout(0, shared_layout)
        shared_n = shared_pid_n * SHARED_BLOCK_N + gl.arange(
            0, SHARED_BLOCK_N, layout=shared_n_layout
        )
        shared_acc = gl.zeros([SHARED_BLOCK_N], gl.float32, shared_n_layout)
        for k0 in range(0, 768, 512):
            shared_k = k0 + gl.arange(0, 512, layout=shared_k_layout)
            valid_k = shared_k < 768
            shared_input = gl.amd.cdna4.buffer_load(
                SharedInput,
                (shared_token * stride_sim + shared_k * stride_sik).to(gl.int32),
                mask=valid_k,
                other=0.0,
            ).to(gl.float32)
            shared_weight = gl.amd.cdna4.buffer_load(
                SharedWeight,
                (
                    shared_n[:, None].to(gl.int64) * 768
                    + shared_k[None, :].to(gl.int64)
                ).to(gl.int32),
                mask=valid_k[None, :],
                other=0.0,
            ).to(gl.float32)
            shared_input = gl.convert_layout(shared_input[None, :], shared_layout)
            shared_acc += gl.sum(shared_weight * shared_input, axis=1)
        gl.store(
            SharedOut + shared_token * stride_som + shared_n * stride_son,
            shared_acc.to(SharedOut.dtype.element_ty),
        )
