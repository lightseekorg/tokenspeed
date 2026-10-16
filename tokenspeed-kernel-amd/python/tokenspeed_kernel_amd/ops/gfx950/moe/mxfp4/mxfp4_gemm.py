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

"""Dynamic-MXFP4-activation (A4W4) SiLU MoE stages on N16 weights for gfx950.

Serves checkpoints whose experts quantize activations to MXFP4 at runtime
(for example Quark MXFP4 with dynamic activations): both GEMMs are
``e2m1 x e2m1`` scaled MFMAs with E8M0 block scales on both operands.

Pipeline (all launches on the caller's stream, no host synchronization):

1. ``..._prep`` quantizes the BF16 tokens to MXFP4. For decode-sized route
   sets one extra program groups routes by expert into ``BLOCK_M`` blocks;
   larger sets use ``gluon_moe_sorting`` before it.
2. ``..._stage1`` gathers each block's tokens, runs the gate/up GEMM, applies
   ``SiLU(gate) * up`` and requantizes the intermediate to MXFP4 in
   sorted-route order.
3. ``..._stage2`` runs the down GEMM and writes BF16 per-route outputs.
4. ``..._combine`` reduces each token's routes in FP32 top-k slot order.

Numerics match the portable Triton path: per-32 E8M0 scale
``2**(floor(log2(amax rounded to a power of two)) - 2)`` (the checkpoint's
Quark "even" mode) with saturating round-to-nearest-even E2M1, gate and up
rounded to BF16, ``SiLU(gate)`` rounded to BF16, the product rounded to BF16
before requantization, and BF16 per-route outputs summed in FP32.

Weights use the N16 bank of ``n16_weights.py`` (shared with the MXFP8
SiTU pipelines), so every wave reads its MFMA weight fragment as one
contiguous 1 KiB load. This is the E2M1-activation counterpart of
``mxfp8_gemm.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe_sorting import gluon_moe_sorting
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (
    _n16_weight_offset,
    _n32_scale_offset,
    n16_mxfp4_shape,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile_impl,
    scaled_downcast_layout,
)

_MXFP4_BLOCK = gl.constexpr(32)
# gfx950 (MI350/MI355) dispatches workgroups round-robin over 8 XCDs.
_NUM_XCDS = 8
# Route sets up to this size are sorted inside the quantization launch.
_FUSED_SORT_MAX_ROUTES = 1024


# ---------------------------------------------------------------------------
# Layouts and device helpers
# ---------------------------------------------------------------------------


@gluon.constexpr_function
def _mfma_layouts(warps_m, warps_n, block_m, block_n):
    """MFMA 16x16x128 layouts for one 128-K step of a ``[BM, BN]`` tile."""
    mfma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[warps_m, warps_n],
    )
    dot_a = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [block_m, 4])
    b_scale = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [block_n, 4])
    return mfma, dot_a, dot_b, a_scale, b_scale


@gluon.jit
def _store_mxfp4_tile(
    packed,
    scale,
    q_ptr,
    s_ptr,
    row_start,
    col_start,
    num_rows,
    stride_q,
    stride_s,
    QUAD_MAJOR: gl.constexpr,
):
    """Store packed E2M1 values and their E8M0 scales for one tile.

    With ``QUAD_MAJOR`` a row's scales are stored as ``[4, groups / 4]``: the
    scales one MFMA lane quad reads over consecutive 128-K steps are
    contiguous, so a lane fetches four steps with one 32-bit load.
    """
    BM: gl.constexpr = packed.shape[0]
    BN_PK: gl.constexpr = packed.shape[1]
    q_layout: gl.constexpr = packed.type.layout
    s_layout: gl.constexpr = scale.type.layout
    q_rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, q_layout))
    q_cols = col_start // 2 + gl.arange(0, BN_PK, gl.SliceLayout(0, q_layout))
    gl.store(
        q_ptr + q_rows[:, None].to(gl.int64) * stride_q + q_cols[None, :],
        packed,
        mask=q_rows[:, None] < num_rows,
    )
    s_rows = row_start + gl.arange(0, BM, gl.SliceLayout(1, s_layout))
    s_cols = col_start // _MXFP4_BLOCK + gl.arange(
        0, BN_PK // 16, gl.SliceLayout(0, s_layout)
    )
    if QUAD_MAJOR:
        s_cols = (s_cols % 4) * (stride_s // 4) + s_cols // 4
    gl.store(
        s_ptr + s_rows[:, None].to(gl.int64) * stride_s + s_cols[None, :],
        scale,
        mask=s_rows[:, None] < num_rows,
    )


@gluon.jit
def _xcd_local_pid(NUM_XCDS: gl.constexpr):
    """Renumber programs so consecutive ids run on one XCD (and its L2).

    Workgroups are dispatched round-robin over the XCDs; renumbering keeps
    the programs that share a route block's activations and an expert's
    weights on the same L2.
    """
    pid = gl.program_id(0)
    if NUM_XCDS == 1:
        return pid
    num_pids = gl.num_programs(0)
    per_xcd = num_pids // NUM_XCDS
    xcd = pid % NUM_XCDS
    # The first ``num_pids % NUM_XCDS`` XCDs run one extra program.
    extra = num_pids - per_xcd * NUM_XCDS
    return xcd * per_xcd + gl.minimum(xcd, extra) + pid // NUM_XCDS


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@gluon.jit
def _sort_routes_one_program(
    topk_ids_ptr,
    sorted_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    cursor_ptr,
    M,
    E: gl.constexpr,
    TOP_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    ROUTE_BLOCK: gl.constexpr,
    EXPERT_PAD: gl.constexpr,
):
    """Block-aligned expert sort of up to ``ROUTE_BLOCK`` routes in one program.

    Same output contract as ``gluon_moe_sorting`` (padding id
    ``(TOP_K << 24) | M``), except that routes of one expert keep no
    particular order, which no consumer depends on.
    """
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    route = gl.arange(0, ROUTE_BLOCK, layout=layout)
    valid = route < M * TOP_K
    expert_id = gl.load(topk_ids_ptr + route, mask=valid, other=E).to(gl.int32)
    valid &= (expert_id >= 0) & (expert_id < E)
    expert_id = gl.where(valid, expert_id, E)
    count = gl.histogram(expert_id, EXPERT_PAD, mask=valid, layout=layout)
    count = count.to(gl.int32)
    expert = gl.arange(0, EXPERT_PAD, layout=layout)
    count = gl.where(expert < E, count, 0)
    padded = (count + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    start = gl.cumsum(padded, 0) - padded
    rows = gl.sum(padded, 0)
    gl.store(num_valid_ids_ptr, rows)
    gl.store(num_valid_ids_ptr + 1, M)
    gl.store(cursor_ptr + expert, start, mask=expert < E)
    for j in range(0, gl.max(padded, 0) // BLOCK_M):
        gl.store(
            sorted_expert_ids_ptr + start // BLOCK_M + j,
            expert,
            mask=j * BLOCK_M < padded,
        )
    # Every routed row first holds the padding id; the routes overwrite
    # theirs below. One store per ROUTE_BLOCK rows: per-expert padding stores
    # (BLOCK_M of them over every expert) fill the store queue and stall on
    # its acknowledgements.
    for row0 in range(0, rows, ROUTE_BLOCK):
        row = row0 + route
        gl.store(
            sorted_ids_ptr + row,
            gl.zeros_like(row) + ((TOP_K << 24) | M),
            mask=row < rows,
        )
    # The padding and the cursors must be in place before any wave claims a
    # slot: every wave drains its stores, then all meet.
    gl.inline_asm_elementwise(
        asm="s_waitcnt vmcnt(0)",
        constraints="=v,0",
        args=[route],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()
    slot = gl.atomic_add(
        cursor_ptr + expert_id,
        gl.full([ROUTE_BLOCK], 1, gl.int32, layout=layout),
        mask=valid,
        sem="relaxed",
        scope="gpu",
    )
    gl.store(
        sorted_ids_ptr + slot, ((route % TOP_K) << 24) | (route // TOP_K), mask=valid
    )


def _prep_launch_metadata(grid, kernel, args):
    m = args["M"]
    k = args["K"]
    return {"name": kernel.name, "bytes": m * k * 2 + m * (k // 2 + k // 32)}


@gluon.jit(launch_metadata=_prep_launch_metadata, do_not_specialize=("M",))
def gluon_mxfp4_a4w4_moe_apply_prep_gfx950(
    x_ptr,
    xq_ptr,
    xs_ptr,
    topk_ids_ptr,
    sorted_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    cursor_ptr,
    M,
    stride_xm,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    FUSE_SORT: gl.constexpr,
    E: gl.constexpr,
    TOP_K: gl.constexpr,
    SORT_BLOCK_M: gl.constexpr,
    ROUTE_BLOCK: gl.constexpr,
    EXPERT_PAD: gl.constexpr,
):
    """Quantize BF16 rows ``[M, K]`` to packed E2M1 and quad-major scales.

    With ``FUSE_SORT`` the last program instead sorts the routes into
    ``SORT_BLOCK_M`` expert blocks, saving decode two launches.
    """
    pid = gl.program_id(0)
    if FUSE_SORT:
        if pid == gl.num_programs(0) - 1:
            _sort_routes_one_program(
                topk_ids_ptr,
                sorted_ids_ptr,
                sorted_expert_ids_ptr,
                num_valid_ids_ptr,
                cursor_ptr,
                M,
                E,
                TOP_K,
                SORT_BLOCK_M,
                ROUTE_BLOCK,
                EXPERT_PAD,
            )
            return
    pid_m = pid // (K // BLOCK_K)
    pid_k = pid % (K // BLOCK_K)
    layout: gl.constexpr = scaled_downcast_layout(BLOCK_M, BLOCK_K, gl.num_warps())
    rows = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    cols = pid_k * BLOCK_K + gl.arange(0, BLOCK_K, gl.SliceLayout(0, layout))
    vals = gl.load(
        x_ptr + rows[:, None].to(gl.int64) * stride_xm + cols[None, :],
        mask=rows[:, None] < M,
        other=0.0,
    ).to(gl.float32)
    packed, scale = _mxfp4_quantize_tile_impl(vals)
    _store_mxfp4_tile(
        packed,
        scale,
        xq_ptr,
        xs_ptr,
        pid_m * BLOCK_M,
        pid_k * BLOCK_K,
        M,
        K // 2,
        K // _MXFP4_BLOCK,
        True,
    )


@gluon.jit
def _stage1_load_b(kt, w_expert, ws_expert, b_off, bs_off, UP_OFF: gl.constexpr):
    """Load W13 gate/up fragments and scale words of 256-wide K step ``kt``."""
    g0 = gl.amd.cdna4.buffer_load(w_expert, b_off + kt * 2048)
    g1 = gl.amd.cdna4.buffer_load(w_expert, b_off + (kt * 2048 + 1024))
    u0 = gl.amd.cdna4.buffer_load(w_expert, b_off + (UP_OFF + kt * 2048))
    u1 = gl.amd.cdna4.buffer_load(w_expert, b_off + (UP_OFF + kt * 2048 + 1024))
    # One word holds this step's gate/up scales for both MFMA K steps.
    words = gl.amd.cdna4.buffer_load(ws_expert, bs_off + kt * 64)
    return g0, g1, u0, u1, words


@gluon.jit
def _stage1_issue_a(kt, xq_ptr, a_off, smem, A_VIA_LDS: gl.constexpr):
    """Start fetching the activations of 256-wide K step ``kt``.

    Via LDS this starts an async copy into ``smem`` and returns nothing;
    otherwise it returns both 128-K MFMA operands, loaded into registers.
    """
    if A_VIA_LDS:
        cdna4_async_copy.buffer_load_to_shared(smem, xq_ptr, a_off + kt * 128)
        cdna4_async_copy.commit_group()
        return ()
    else:
        a0 = gl.amd.cdna4.buffer_load(xq_ptr, a_off + kt * 128)
        a1 = gl.amd.cdna4.buffer_load(xq_ptr, a_off + (kt * 128 + 64))
        return ((a0, a1),)


@gluon.jit
def _stage1_step(gate, up, a0, a1, a_word, b, kt):
    """Accumulate one 256-wide K step (two scaled MFMAs each for gate, up)."""
    g0, g1, u0, u1, words = b
    shift = (kt % 2) * 16
    as0 = (a_word >> shift).to(gl.uint8)
    as1 = (a_word >> (shift + 8)).to(gl.uint8)
    gate = gl.amd.cdna4.mfma_scaled(
        a0, as0, "e2m1", g0, words.to(gl.uint8), "e2m1", gate
    )
    up = gl.amd.cdna4.mfma_scaled(
        a0, as0, "e2m1", u0, (words >> 8).to(gl.uint8), "e2m1", up
    )
    gate = gl.amd.cdna4.mfma_scaled(
        a1, as1, "e2m1", g1, (words >> 16).to(gl.uint8), "e2m1", gate
    )
    up = gl.amd.cdna4.mfma_scaled(
        a1, as1, "e2m1", u1, (words >> 24).to(gl.uint8), "e2m1", up
    )
    return gate, up


def _stage1_launch_metadata(grid, kernel, args):
    routes = args["M"] * args["TOP_K"]
    h = args["H"]
    i = args["I"]
    experts = min(args["E"], routes)
    weight_bytes = experts * 2 * i * (h // 2 + h // 32)
    act_bytes = routes * (h // 2 + h // 32 + i // 2 + i // 32)
    return {
        "name": kernel.name,
        "flops": 2 * routes * h * 2 * i,
        "bytes": weight_bytes + act_bytes,
    }


@gluon.jit(launch_metadata=_stage1_launch_metadata, do_not_specialize=("M",))
def gluon_mxfp4_a4w4_moe_apply_stage1_gfx950(
    xq_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    sorted_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    out_q_ptr,
    out_s_ptr,
    M,
    E: gl.constexpr,
    TOP_K: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    A_VIA_LDS: gl.constexpr,
    PREFETCH: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Gate/up GEMM + SiLU-and-mul + MXFP4 requant for one route block.

    Program ``(block, n)`` computes intermediate columns
    ``[n * BLOCK_N, (n + 1) * BLOCK_N)`` for the ``BLOCK_M`` sorted routes of
    one expert and writes them in sorted-route order. Waves load their W13
    fragments straight into registers. ``A_VIA_LDS`` stages the gathered
    activations through an LDS ring (async copies) so the ``WARPS_N`` waves
    share them; otherwise each wave loads its activation fragments directly,
    which is cheaper for the few rows of a decode block. ``PREFETCH`` 256-K
    steps stay in flight.
    """
    # Quad-major activation scale words cover 512 K values.
    gl.static_assert(H % 512 == 0)
    gl.static_assert(I % BLOCK_N == 0)
    gl.static_assert(BLOCK_N % _MXFP4_BLOCK == 0)
    NUM_N: gl.constexpr = I // BLOCK_N
    H_PK: gl.constexpr = H // 2
    H_SC: gl.constexpr = H // _MXFP4_BLOCK
    NUM_K: gl.constexpr = H // 256

    pid = _xcd_local_pid(NUM_XCDS)
    pid_m = pid // NUM_N
    pid_n = pid % NUM_N
    num_valid = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return
    expert = gl.load(sorted_expert_ids_ptr + pid_m)
    row0 = pid_m * BLOCK_M

    layouts: gl.constexpr = _mfma_layouts(WARPS_M, WARPS_N, BLOCK_M, BLOCK_N)
    mfma: gl.constexpr = layouts[0]
    dot_a: gl.constexpr = layouts[1]
    dot_b: gl.constexpr = layouts[2]
    a_scale_l: gl.constexpr = layouts[3]
    b_scale_l: gl.constexpr = layouts[4]

    # Padding routes read token 0; their results are never stored.
    if A_VIA_LDS:
        # Each wave copies whole 128-byte rows, 8 rows per instruction.
        a_l: gl.constexpr = gl.BlockedLayout(
            [1, 16], [8, 8], [WARPS_M * WARPS_N, 1], [1, 0]
        )
        A_COLS: gl.constexpr = 128
        smem = gl.allocate_shared_memory(
            gl.uint8,
            [PREFETCH + 1, BLOCK_M, 128],
            gl.SwizzledSharedLayout(16, 1, 8, [1, 0]),
        )
    else:
        a_l: gl.constexpr = dot_a
        A_COLS: gl.constexpr = 64
        smem = 0
    tok_a = gl.load(
        sorted_ids_ptr + row0 + gl.arange(0, BLOCK_M, gl.SliceLayout(1, a_l))
    )
    tok_a = (tok_a & 0xFFFFFF).to(gl.int32)
    a_off = (
        gl.where(tok_a < M, tok_a, 0)[:, None] * H_PK
        + gl.arange(0, A_COLS, gl.SliceLayout(0, a_l))[None, :]
    )
    # Quad-major activation scales: word (token, quad, j) holds the scales
    # of 128-K steps 4j .. 4j + 3, i.e. of 256-K steps 2j and 2j + 1.
    tok_s = gl.load(
        sorted_ids_ptr + row0 + gl.arange(0, BLOCK_M, gl.SliceLayout(1, a_scale_l))
    )
    tok_s = (tok_s & 0xFFFFFF).to(gl.int32)
    xs_words = xs_ptr.to(gl.pointer_type(gl.uint32))
    as_off = gl.where(tok_s < M, tok_s, 0)[:, None] * (H_SC // 4) + gl.arange(
        0, 4, gl.SliceLayout(0, a_scale_l)
    )[None, :] * (H_SC // 16)

    # N16 W13: 16-row group 2j holds gate rows [16j, 16j + 16), 2j + 1 the
    # matching up rows.
    w_expert = w_ptr + expert.to(gl.int64) * (2 * I * H_PK)
    bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, dot_b))
    bk = gl.arange(0, 64, gl.SliceLayout(1, dot_b))
    b_off = _n16_weight_offset(
        bn[None, :] + (bn[None, :] // 16) * 16, bk[:, None], H_PK
    )
    # N16 W13 scale words: [i16, K / 256, lane quad, n % 16] holding bytes
    # (gate, up) x (first, second MFMA K step).
    ws_expert = ws_ptr.to(gl.pointer_type(gl.uint32)) + expert.to(gl.int64) * (
        2 * I * H_SC // 4
    )
    bsn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(1, b_scale_l))
    bsq = gl.arange(0, 4, gl.SliceLayout(0, b_scale_l))
    bs_off = (bsn[:, None] // 16) * (NUM_K * 64) + bsq[None, :] * 16 + bsn[:, None] % 16

    gate = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    up = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    SLOTS: gl.constexpr = PREFETCH + 1
    bufs_a = ()
    bufs_b = ()
    bufs_s = ()
    for kt in gl.static_range(PREFETCH):
        slot = smem.index(kt) if A_VIA_LDS else smem
        bufs_a += _stage1_issue_a(kt, xq_ptr, a_off, slot, A_VIA_LDS)
        bufs_b += (_stage1_load_b(kt, w_expert, ws_expert, b_off, bs_off, H_PK * 16),)
        if kt % 2 == 0:
            bufs_s += (gl.amd.cdna4.buffer_load(xs_words, as_off + kt // 2),)
    for kt in gl.static_range(NUM_K):
        if kt + PREFETCH < NUM_K:
            slot = smem.index((kt + PREFETCH) % SLOTS) if A_VIA_LDS else smem
            bufs_a += _stage1_issue_a(kt + PREFETCH, xq_ptr, a_off, slot, A_VIA_LDS)
            bufs_b += (
                _stage1_load_b(
                    kt + PREFETCH, w_expert, ws_expert, b_off, bs_off, H_PK * 16
                ),
            )
            if (kt + PREFETCH) % 2 == 0:
                bufs_s += (
                    gl.amd.cdna4.buffer_load(xs_words, as_off + (kt + PREFETCH) // 2),
                )
        if A_VIA_LDS:
            # Copies issued after step kt's may stay outstanding.
            cdna4_async_copy.wait_group(min(PREFETCH, NUM_K - 1 - kt))
            stage = smem.index(kt % SLOTS)
            a0 = stage.slice(0, 64, 1).load(dot_a)
            a1 = stage.slice(64, 64, 1).load(dot_a)
        else:
            a0, a1 = bufs_a[0]
            bufs_a = bufs_a[1:]
        gate, up = _stage1_step(gate, up, a0, a1, bufs_s[0], bufs_b[0], kt)
        bufs_b = bufs_b[1:]
        if kt % 2 == 1:
            bufs_s = bufs_s[1:]

    # SiLU(gate) * up with the BF16 roundings of the portable reference.
    gate = gate.to(gl.bfloat16).to(gl.float32)
    up = up.to(gl.bfloat16).to(gl.float32)
    silu = (gate / (1.0 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
    act = (silu * up).to(gl.bfloat16).to(gl.float32)
    act = gl.convert_layout(
        act, scaled_downcast_layout(BLOCK_M, BLOCK_N, gl.num_warps())
    )
    packed, scale = _mxfp4_quantize_tile_impl(act)
    _store_mxfp4_tile(
        packed,
        scale,
        out_q_ptr,
        out_s_ptr,
        row0,
        pid_n * BLOCK_N,
        num_valid,
        I // 2,
        I // _MXFP4_BLOCK,
        False,
    )


def _stage2_launch_metadata(grid, kernel, args):
    routes = args["M"] * args["TOP_K"]
    h = args["H"]
    i = args["I"]
    experts = min(args["E"], routes)
    weight_bytes = experts * h * (i // 2 + i // 32)
    act_bytes = routes * (i // 2 + i // 32 + 2 * h)
    return {
        "name": kernel.name,
        "flops": 2 * routes * h * i,
        "bytes": weight_bytes + act_bytes,
    }


@gluon.jit(launch_metadata=_stage2_launch_metadata, do_not_specialize=("M",))
def gluon_mxfp4_a4w4_moe_apply_stage2_gfx950(
    iq_ptr,
    is_ptr,
    w_ptr,
    ws_ptr,
    sorted_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    route_out_ptr,
    M,
    E: gl.constexpr,
    TOP_K: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    COALESCE_STORE: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Down GEMM for one route block; writes BF16 per-route outputs.

    ``COALESCE_STORE`` redistributes the output tile through LDS so each
    scattered route row leaves as one contiguous run.
    """
    gl.static_assert(H % BLOCK_N == 0)
    gl.static_assert(I % 64 == 0)
    NUM_N: gl.constexpr = H // BLOCK_N
    I_PK: gl.constexpr = I // 2
    I_SC: gl.constexpr = I // _MXFP4_BLOCK
    # W2 scales pad K to whole 8-group cells.
    I_SC_PAD: gl.constexpr = (I_SC + 7) // 8 * 8

    pid = _xcd_local_pid(NUM_XCDS)
    pid_m = pid // NUM_N
    pid_n = pid % NUM_N
    num_valid = gl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return
    expert = gl.load(sorted_expert_ids_ptr + pid_m)
    row0 = pid_m * BLOCK_M

    # Two accumulators split each 32-row group of W2: ``NI = 0`` takes rows
    # n % 32 < 16, ``NI = 1`` the rest. One N16 scale word holds both halves'
    # scales for two 128-K steps, so every MFMA picks a constant byte.
    BN_HALF: gl.constexpr = BLOCK_N // 2
    layouts: gl.constexpr = _mfma_layouts(WARPS_M, WARPS_N, BLOCK_M, BN_HALF)
    mfma: gl.constexpr = layouts[0]
    dot_a: gl.constexpr = layouts[1]
    dot_b: gl.constexpr = layouts[2]
    a_scale_l: gl.constexpr = layouts[3]
    b_scale_l: gl.constexpr = layouts[4]

    # The intermediate rows of a block are contiguous (sorted order).
    ak = gl.arange(0, 64, gl.SliceLayout(0, dot_a))
    am = row0 + gl.arange(0, BLOCK_M, gl.SliceLayout(1, dot_a))
    a_off = am[:, None] * I_PK + ak[None, :]
    ask = gl.arange(0, 4, gl.SliceLayout(0, a_scale_l))
    asm = row0 + gl.arange(0, BLOCK_M, gl.SliceLayout(1, a_scale_l))
    as_off = asm[:, None] * I_SC + ask[None, :]

    # N16 W2: a K width that is not a multiple of 128 ends in a half chunk
    # whose upper lane quads are masked, so they fetch nothing and add zero.
    w_expert = w_ptr + expert.to(gl.int64) * (H * I_PK)
    ws_words = (ws_ptr + expert.to(gl.int64) * (H * I_SC_PAD)).to(
        gl.pointer_type(gl.uint32)
    )
    bk = gl.arange(0, 64, gl.SliceLayout(1, dot_b))
    bn = gl.arange(0, BN_HALF, gl.SliceLayout(0, dot_b))
    bn = pid_n * BLOCK_N + bn // 16 * 32 + bn % 16
    b_off = _n16_weight_offset(bn[None, :], bk[:, None], I_PK)
    # Word of (rows n, n + 16) x (K groups j, j + 4); padded groups of the
    # last 8-group cell hold unit scales.
    bsk = gl.arange(0, 4, gl.SliceLayout(0, b_scale_l))
    bsn = gl.arange(0, BN_HALF, gl.SliceLayout(1, b_scale_l))
    bsn = pid_n * BLOCK_N + bsn // 16 * 32 + bsn % 16
    bs_off = _n32_scale_offset(bsn[:, None], bsk[None, :], I_SC_PAD) // 4

    acc0 = gl.zeros((BLOCK_M, BN_HALF), gl.float32, mfma)
    acc1 = gl.zeros((BLOCK_M, BN_HALF), gl.float32, mfma)
    for kt in gl.static_range(gl.cdiv(I, 128)):
        a = gl.amd.cdna4.buffer_load(
            iq_ptr, a_off + kt * 64, mask=(ak + kt * 64 < I_PK)[None, :], other=0
        )
        a_s = gl.amd.cdna4.buffer_load(
            is_ptr,
            as_off + kt * 4,
            mask=(ask + kt * 4 < I_SC)[None, :],
            other=127,
        )
        b_mask = (bk + kt * 64 < I_PK)[:, None]
        b0 = gl.amd.cdna4.buffer_load(w_expert, b_off + kt * 1024, mask=b_mask, other=0)
        b1 = gl.amd.cdna4.buffer_load(
            w_expert, b_off + (16 * I_PK + kt * 1024), mask=b_mask, other=0
        )
        if kt % 2 == 0:
            words = gl.amd.cdna4.buffer_load(ws_words, bs_off + kt // 2 * 64)
        acc0 = gl.amd.cdna4.mfma_scaled(
            a, a_s, "e2m1", b0, (words >> (kt % 2 * 16)).to(gl.uint8), "e2m1", acc0
        )
        acc1 = gl.amd.cdna4.mfma_scaled(
            a,
            a_s,
            "e2m1",
            b1,
            (words >> (kt % 2 * 16 + 8)).to(gl.uint8),
            "e2m1",
            acc1,
        )
    # Interleave the halves' 16-column groups back into row order.
    acc = (
        gl.join(
            acc0.reshape(BLOCK_M, BN_HALF // 16, 16),
            acc1.reshape(BLOCK_M, BN_HALF // 16, 16),
        )
        .permute(0, 1, 3, 2)
        .reshape(BLOCK_M, BLOCK_N)
    )

    if COALESCE_STORE:
        store_l: gl.constexpr = gl.BlockedLayout(
            [1, 8], [64 // (BLOCK_N // 8), BLOCK_N // 8], [gl.num_warps(), 1], [1, 0]
        )
    else:
        store_l: gl.constexpr = mfma
    out = gl.convert_layout(acc.to(gl.bfloat16), store_l)
    cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, store_l))
    cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, store_l))
    ids = gl.load(sorted_ids_ptr + row0 + cm)
    tok = ids & 0xFFFFFF
    route = tok * TOP_K + (ids >> 24)
    gl.store(
        route_out_ptr + route[:, None].to(gl.int64) * H + cn[None, :],
        out,
        mask=(tok < M)[:, None],
    )


def _combine_launch_metadata(grid, kernel, args):
    m = args["M"]
    h = args["H"]
    k = args["TOP_K"]
    return {"name": kernel.name, "bytes": m * k * (2 * h + 8) + 2 * m * h}


@gluon.jit(launch_metadata=_combine_launch_metadata, do_not_specialize=("M",))
def gluon_mxfp4_a4w4_moe_apply_combine_gfx950(
    route_out_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    out_ptr,
    M,
    E: gl.constexpr,
    TOP_K: gl.constexpr,
    H: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Sum each token's BF16 route outputs in FP32 top-k slot order."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 64], [gl.num_warps(), 1], [1, 0]
    )
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    cols = gl.program_id(1) * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    row_mask = rows < M
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, layout)
    for slot in gl.static_range(TOP_K):
        route = rows * TOP_K + slot
        expert = gl.load(topk_ids_ptr + route, mask=row_mask, other=-1)
        weight = gl.load(topk_weights_ptr + route, mask=row_mask, other=0.0)
        # Routes to experts outside [0, E) were never computed: they add 0.
        live = row_mask & (expert >= 0) & (expert < E)
        vals = gl.load(
            route_out_ptr + route[:, None].to(gl.int64) * H + cols[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc += vals.to(gl.float32) * weight[:, None]
    gl.store(
        out_ptr + rows[:, None].to(gl.int64) * H + cols[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Host launcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _A4W4Config:
    """Launch shape for one route-block height (rocprofv3-tuned on MI350X)."""

    block_m: int
    s1_block_n: int
    s1_warps_m: int
    s1_warps_n: int
    s1_a_via_lds: bool
    s1_prefetch: int
    s2_block_n: int
    s2_warps_m: int
    s2_warps_n: int
    s2_coalesce_store: bool
    num_xcds: int


# Decode blocks stream each expert's weights once and gain nothing from XCD
# grouping; larger blocks share activations across N tiles and weights
# across the blocks of one expert, so they keep those programs on one L2.
_CONFIGS = {
    16: _A4W4Config(16, 64, 1, 4, False, 1, 128, 1, 4, False, 1),
    32: _A4W4Config(32, 64, 1, 2, True, 1, 512, 1, 4, False, _NUM_XCDS),
    64: _A4W4Config(64, 64, 1, 2, True, 1, 512, 1, 4, True, _NUM_XCDS),
}


def _select_config(num_routes: int, num_experts: int) -> _A4W4Config:
    """Pick the route-block height from the average routes per expert."""
    if num_routes <= 4 * num_experts:
        return _CONFIGS[16]
    if num_routes <= 48 * num_experts:
        return _CONFIGS[32]
    return _CONFIGS[64]


def launch_gluon_mxfp4_a4w4_moe_apply_gfx950(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the dynamic-MXFP4 A4W4 SiLU MoE on N16-bank weights.

    Args:
        x: BF16 tokens ``[M, H]`` with unit stride along ``H``.
        w13, w13_scale, w2, w2_scale: N16 banks from
            ``n16_weights.preprocess_n16_mxfp4_weights``; ``H`` must be a
            multiple of 512 and ``I`` of 64.
        topk_weights: Route weights ``[M, top_k]``.
        topk_ids: Expert ids ``[M, top_k]``; ids outside ``[0, E)`` add zero.

    Returns:
        ``[M, H]`` BF16 output.
    """
    num_tokens, hidden = x.shape
    num_experts, weight_hidden, inter = n16_mxfp4_shape(w13, w13_scale, w2, w2_scale)
    # The kernels index routes as token * top_k + slot without bounds checks.
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != num_tokens
        or topk_weights.shape != topk_ids.shape
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk_weights.is_floating_point()
    ):
        raise ValueError(
            "A4W4 MoE requires integer topk_ids and floating topk_weights shaped "
            f"[{num_tokens}, top_k], got {topk_ids.dtype} "
            f"{tuple(topk_ids.shape)} and {topk_weights.dtype} "
            f"{tuple(topk_weights.shape)}"
        )
    top_k = int(topk_ids.shape[1])
    device = x.device
    if x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError("A4W4 MoE requires BF16 activations with unit inner stride")
    # Quad-major activation scale words cover 512 K values.
    if weight_hidden != hidden or hidden % 512:
        raise ValueError("A4W4 MoE weights do not match the activation width")
    out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=device)
    if num_tokens == 0:
        return out
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    num_routes = num_tokens * top_k
    cfg = _select_config(num_routes, num_experts)

    xq = torch.empty((num_tokens, hidden // 2), dtype=torch.uint8, device=device)
    xs = torch.empty((num_tokens, hidden // 32), dtype=torch.uint8, device=device)
    quant_block_m = 16
    quant_block_k = 512
    quant_programs = triton.cdiv(num_tokens, quant_block_m) * (hidden // quant_block_k)
    # Decode-sized route sets are sorted by one extra program of the
    # quantization launch.
    fuse_sort = num_routes <= _FUSED_SORT_MAX_ROUTES
    if fuse_sort:
        padded_rows = (
            (num_routes + min(num_routes, num_experts) * (cfg.block_m - 1))
            // cfg.block_m
            * cfg.block_m
        )
        sorted_ids = torch.empty((padded_rows,), dtype=torch.int32, device=device)
        sorted_expert_ids = torch.empty(
            (triton.cdiv(padded_rows, cfg.block_m),), dtype=torch.int32, device=device
        )
        num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
        cursor = torch.empty((num_experts,), dtype=torch.int32, device=device)
    else:
        sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = gluon_moe_sorting(
            topk_ids.to(torch.int32).contiguous(),
            topk_weights,
            num_experts,
            hidden,
            x.dtype,
            cfg.block_m,
            use_minimum_route_chunks=False,
            runtime_rows=False,
            out=out,
        )
        padded_rows = int(sorted_ids.shape[0])
        cursor = num_valid_ids
    num_blocks = triton.cdiv(padded_rows, cfg.block_m)
    gluon_mxfp4_a4w4_moe_apply_prep_gfx950[(quant_programs + int(fuse_sort),)](
        x,
        xq,
        xs,
        topk_ids,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        cursor,
        num_tokens,
        x.stride(0),
        K=hidden,
        BLOCK_M=quant_block_m,
        BLOCK_K=quant_block_k,
        FUSE_SORT=fuse_sort,
        E=num_experts,
        TOP_K=top_k,
        SORT_BLOCK_M=cfg.block_m,
        ROUTE_BLOCK=(
            max(256, triton.next_power_of_2(num_routes))
            if fuse_sort
            else _FUSED_SORT_MAX_ROUTES
        ),
        EXPERT_PAD=triton.next_power_of_2(num_experts + 1),
        num_warps=4,
    )

    iq = torch.empty((padded_rows, inter // 2), dtype=torch.uint8, device=device)
    i_s = torch.empty((padded_rows, inter // 32), dtype=torch.uint8, device=device)
    gluon_mxfp4_a4w4_moe_apply_stage1_gfx950[(num_blocks * (inter // cfg.s1_block_n),)](
        xq,
        xs,
        w13,
        w13_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        iq,
        i_s,
        num_tokens,
        E=num_experts,
        TOP_K=top_k,
        H=hidden,
        I=inter,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.s1_block_n,
        WARPS_M=cfg.s1_warps_m,
        WARPS_N=cfg.s1_warps_n,
        A_VIA_LDS=cfg.s1_a_via_lds,
        PREFETCH=cfg.s1_prefetch,
        NUM_XCDS=cfg.num_xcds,
        num_warps=cfg.s1_warps_m * cfg.s1_warps_n,
    )

    route_out = torch.empty((num_routes, hidden), dtype=x.dtype, device=device)
    gluon_mxfp4_a4w4_moe_apply_stage2_gfx950[
        (num_blocks * (hidden // cfg.s2_block_n),)
    ](
        iq,
        i_s,
        w2,
        w2_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        route_out,
        num_tokens,
        E=num_experts,
        TOP_K=top_k,
        H=hidden,
        I=inter,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.s2_block_n,
        WARPS_M=cfg.s2_warps_m,
        WARPS_N=cfg.s2_warps_n,
        COALESCE_STORE=cfg.s2_coalesce_store,
        NUM_XCDS=cfg.num_xcds,
        num_warps=cfg.s2_warps_m * cfg.s2_warps_n,
    )

    combine_block_m = 4
    combine_block_n = 512
    gluon_mxfp4_a4w4_moe_apply_combine_gfx950[
        (triton.cdiv(num_tokens, combine_block_m), hidden // combine_block_n)
    ](
        route_out,
        topk_weights,
        topk_ids,
        out,
        num_tokens,
        E=num_experts,
        TOP_K=top_k,
        H=hidden,
        BLOCK_M=combine_block_m,
        BLOCK_N=combine_block_n,
        num_warps=4,
    )
    return out
