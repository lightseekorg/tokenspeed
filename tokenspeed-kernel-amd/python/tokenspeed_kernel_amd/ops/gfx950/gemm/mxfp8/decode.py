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

"""Small-M (decode) MXFP8 GEMM for gfx950.

Computes ``A @ B.T`` for E4M3 ``A [M, K]`` and ``B [N, K]`` with native
UE8M0 K32 scales or arbitrary FP32 K128 block scales. Both use FP32
accumulation; the FP32-scale contract produces BF16 output. Decode GEMMs
stream the weight once while the activation stays in L2,
so the kernel is organized around weight bandwidth: small N tiles, optional
split-K to occupy every CU, and a multi-buffered async global-to-LDS pipeline.

For native UE8M0, gfx950 FP8 MFMAs sum their products at reduced internal
precision, so small
results of large cancelling terms miss the FP8 GEMM contract. Instead, each
32-value scale group runs one BF16 MFMA on the exactly upcast E4M3 values
(products exact, sums in FP32), and the group's power-of-two row and column
scales multiply its partial exactly before it joins the FP32 accumulator.
The FP32-scale specialization instead retains the existing FP8 K128 matrix
arithmetic, applying activation then weight scales to each unscaled partial.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.splitk import check_splitk_counters

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy

MXFP8_SCALE_GROUP = 32
SCALE_GROUP = gl.constexpr(MXFP8_SCALE_GROUP)
_SUPPORTED_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}

_partial_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _mxfp8_decode_launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    scale_values = (
        4 * (m + triton.cdiv(n, 128)) * triton.cdiv(k, 128)
        if args.get("FP32_SCALES", False)
        else (m + n) * (k // MXFP8_SCALE_GROUP)
    )
    split_k = args["SPLIT_K"]
    partial_bytes = 0
    if split_k > 1:
        # Every split writes its FP32 tile; the last split reads them back.
        partial_bytes = 2 * split_k * m * n * 4
    return {
        "name": kernel.name,
        "flops8": 2 * m * n * k,
        "bytes": m * k
        + n * k
        + scale_values
        + m * n * args["c_ptr"].element_size()
        + partial_bytes,
    }


@gluon.constexpr_function
def _operand_shared_layout(dot_layout, shape, dtype, k_dim):
    """K-contiguous padded LDS layout for one MFMA operand tile.

    Prefer the compiler's conflict-free padded layout. It does not cover
    16-wide tiles; those use an identity layout padded every 1 KiB, the
    smallest interval a 64-lane x 16-byte direct-to-LDS copy accepts.
    """
    layout = cdna4.compute_efficient_padded_shared_layout(
        dot_layout, shape, dtype, True
    )
    if layout is None:
        order = [1, 0] if k_dim == 1 else [0, 1]
        layout = gl.PaddedSharedLayout.with_identity_for([[1024, 16]], shape, order)
    return layout


@gluon.constexpr_function
def _copy_layout(shared_layout, shape, num_warps):
    """Global-load layout whose warps each fill one contiguous 1 KiB LDS run.

    Direct-to-LDS copies on gfx950 write one contiguous run per warp, so the
    distributed layout follows the shared layout's own address bits: four
    register bits form each lane's 16-byte vector, the next six bits the
    lanes, then the warps; the rest become extra registers. A tile smaller
    than all warps leaves the surplus warps replicated (predicated off).
    """
    bases = [list(b) for b in shared_layout.offset_bases]
    warp_bits = num_warps.bit_length() - 1
    reg_bases = bases[:4]
    lane_bases = bases[4:10]
    warp_bases = bases[10 : 10 + warp_bits]
    reg_bases += bases[10 + warp_bits :]
    while len(warp_bases) < warp_bits:
        warp_bases.append([0, 0])
    return gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=lane_bases,
        warp_bases=warp_bases,
        block_bases=[],
        shape=shape,
    )


@gluon.jit
def _remap_xcd(pid, num_programs, NUM_XCDS: gl.constexpr):
    # Hardware dispatches program i to XCD i % NUM_XCDS; give each XCD a
    # contiguous range of program ids instead.
    pids_per_xcd = (num_programs + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = num_programs % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@gluon.jit
def _scale_value(scales, g, PACKED: gl.constexpr):
    """FP32 scale 2^(code - 127) of group ``g`` from a byte or four-byte load.

    The code becomes the FP32 exponent field directly: exact for codes 1..254,
    infinity for 255 (as ``exp2``), and 0 for code 0, which the activation
    quantizer never emits.
    """
    if PACKED:
        code = (scales[g // 4] >> (8 * (g % 4))) & 0xFF
    else:
        code = scales[g].to(gl.int32)
    return (code << 23).to(gl.float32, bitcast=True)


@gluon.jit
def _issue_stage(
    smem_a,
    smem_b,
    a_ptr,
    b_ptr,
    a_scales_ptr,
    b_scales_ptr,
    a_offsets,
    b_offsets,
    as_offsets,
    bs_offsets,
    a_k_offs,
    b_k_offs,
    a_row_mask,
    k0,
    k_begin,
    K: gl.constexpr,
    K_TAIL: gl.constexpr,
    GROUPS_PER_TILE: gl.constexpr,
    PACKED: gl.constexpr,
    FP32_SCALES: gl.constexpr,
    ALIGNED_VALUES: gl.constexpr,
    stride_ask,
    stride_bsk,
):
    """Start loading one K tile: scales into registers, values into LDS.

    Each 32-value scale group gets a row-scale vector for A and a column-scale
    vector for B, laid out like the accumulator's rows and columns. Scales go
    first: loads retire in issue order, so the wait for this tile's values
    also covers its scales.
    """
    # Advance the scalar base pointers rather than the per-lane offsets, so a
    # runtime K loop adds nothing per lane and keeps no extra offset vectors.
    GROUP_SIZE: gl.constexpr = 128 if FP32_SCALES else SCALE_GROUP
    kg = k0 // GROUP_SIZE
    a_s = ()
    b_s = ()
    if FP32_SCALES:
        for g in gl.static_range(GROUPS_PER_TILE):
            group = kg + g
            if K_TAIL:
                group = gl.minimum(group, (K - 1) // 128 - k_begin // 128)
            a_s = a_s + (
                cdna4.buffer_load(a_scales_ptr, as_offsets + group * stride_ask),
            )
            b_s = b_s + (
                cdna4.buffer_load(b_scales_ptr, bs_offsets + group * stride_bsk),
            )
    elif PACKED:
        # Four consecutive groups are one dword of a scale row.
        for w in gl.static_range(GROUPS_PER_TILE // 4):
            a_s = a_s + (cdna4.buffer_load(a_scales_ptr + kg // 4 + w, as_offsets),)
            b_s = b_s + (cdna4.buffer_load(b_scales_ptr + kg // 4 + w, bs_offsets),)
    for g in gl.static_range(0 if PACKED or FP32_SCALES else GROUPS_PER_TILE):
        group = kg + g
        if K_TAIL:
            # A group past K re-reads the last real group: its finite scale
            # meets zero-filled values, so it adds exactly 0 without a mask.
            group = gl.minimum(group, K // SCALE_GROUP - 1 - k_begin // SCALE_GROUP)
        a_s = a_s + (cdna4.buffer_load(a_scales_ptr + group, as_offsets),)
        b_s = b_s + (cdna4.buffer_load(b_scales_ptr + group, bs_offsets),)
    if not ALIGNED_VALUES:
        smem_a.store(
            cdna4.buffer_load(
                a_ptr + k0,
                a_offsets,
                mask=a_row_mask & (a_k_offs + k0 < K)[None, :],
                other=0.0,
            )
        )
        smem_b.store(
            cdna4.buffer_load(
                b_ptr + k0,
                b_offsets,
                mask=(b_k_offs + k0 < K)[:, None],
                other=0.0,
            )
        )
    elif K_TAIL:
        # Zero both operands past K: an unmasked B tail would read the next
        # row, and zero times a NaN code would poison the accumulator.
        async_copy.buffer_load_to_shared(
            smem_a,
            a_ptr + k0,
            a_offsets,
            mask=a_row_mask & (a_k_offs + k0 < K)[None, :],
        )
        async_copy.buffer_load_to_shared(
            smem_b, b_ptr + k0, b_offsets, mask=(b_k_offs + k0 < K)[:, None]
        )
    else:
        async_copy.buffer_load_to_shared(smem_a, a_ptr + k0, a_offsets, mask=a_row_mask)
        async_copy.buffer_load_to_shared(smem_b, b_ptr + k0, b_offsets)
    async_copy.commit_group()
    return a_s, b_s


@gluon.jit
def _mma_tile(
    buf_a,
    buf_b,
    a_s,
    b_s,
    acc,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    mfma: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GROUPS_PER_TILE: gl.constexpr,
    PACKED: gl.constexpr,
    FP32_SCALES: gl.constexpr,
):
    """Accumulate one K tile with the arithmetic of its scale contract."""
    # Issue every group's MFMA before scaling any partial, so MFMA latency
    # overlaps instead of stalling each group's scaling.
    if FP32_SCALES:
        for g in gl.static_range(GROUPS_PER_TILE):
            a = buf_a.slice(g * 128, 128, dim=1).load(dot_a)
            b = buf_b.slice(g * 128, 128, dim=0).load(dot_b)
            partial = cdna4.mfma_scaled(
                a,
                None,
                "e4m3",
                b,
                None,
                "e4m3",
                gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma),
            )
            acc = gl.fma(partial * a_s[g][:, None], b_s[g][None, :], acc)
        return acc
    parts = ()
    for g in gl.static_range(GROUPS_PER_TILE):
        # E4M3 values are exact in BF16, so the MFMA's products are exact and
        # only its FP32 sum rounds.
        a = buf_a.slice(g * SCALE_GROUP, SCALE_GROUP, dim=1).load(dot_a)
        b = buf_b.slice(g * SCALE_GROUP, SCALE_GROUP, dim=0).load(dot_b)
        parts = parts + (
            cdna4.mfma(
                a.to(gl.bfloat16),
                b.to(gl.bfloat16),
                gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma),
            ),
        )
    for g in gl.static_range(GROUPS_PER_TILE):
        # Power-of-two scales multiply exactly while the product stays a
        # normal FP32 number, so only the accumulation rounds.
        scale = (
            _scale_value(a_s, g, PACKED)[:, None]
            * _scale_value(b_s, g, PACKED)[None, :]
        )
        acc += parts[g] * scale
    return acc


# M follows the batch; without do_not_specialize its value-1 and
# divisible-by-16 classes would each compile another binary.
@gluon.jit(launch_metadata=_mxfp8_decode_launch_metadata, do_not_specialize=("M",))
def gluon_mm_mxfp8_decode_gfx950(
    a_ptr,
    b_ptr,
    a_scales_ptr,
    b_scales_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_bn,
    stride_asm,
    stride_bsn,
    stride_cm,
    stride_ask,
    stride_bsk,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    PACKED_SCALES: gl.constexpr,
    FP32_SCALES: gl.constexpr,
    ALIGNED_VALUES: gl.constexpr,
):
    """Multi-buffered FP8 GEMM for decode-sized ``M``.

    Program ids enumerate (split, n tile, m tile) with the M tile fastest, so
    programs that share a weight tile are adjacent and, after the XCD remap,
    share one XCD's L2. Each program streams its K slice through a
    ``NUM_BUFFERS``-deep ring of LDS buffers filled by async copies; the
    scale bytes of each K tile are loaded into registers just ahead of its
    values. Only the ring's fill and drain are unrolled; the steady state is a
    runtime loop, so code size and compile time do not grow with ``K``.

    With ``SPLIT_K > 1`` each split stores its FP32 partial tile and bumps the
    tile's arrival counter; the last arriving split sums all partials in split
    order (deterministic), writes C, and resets the counter for the next
    launch.
    """
    gl.static_assert(NUM_BUFFERS >= 2)
    gl.static_assert(BLOCK_K % 128 == 0)
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    K_TILES: gl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    gl.static_assert(K_TILES % SPLIT_K == 0)
    TILES_PER_SPLIT: gl.constexpr = K_TILES // SPLIT_K
    gl.static_assert(TILES_PER_SPLIT >= NUM_BUFFERS - 1)
    K_TAIL: gl.constexpr = K % BLOCK_K != 0

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    if NUM_XCDS > 1:
        pid = _remap_xcd(pid, num_tiles * SPLIT_K, NUM_XCDS)
    tile = pid % num_tiles
    split = pid // num_tiles
    pid_m = tile % num_pid_m
    pid_n = tile // num_pid_m
    k_begin = split * (TILES_PER_SPLIT * BLOCK_K)

    # One 16x16x32 BF16 MFMA consumes exactly one 32-value scale group.
    GROUPS_PER_TILE: gl.constexpr = BLOCK_K // (128 if FP32_SCALES else SCALE_GROUP)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128] if FP32_SCALES else [16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 32 if FP32_SCALES else 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma, 32 if FP32_SCALES else 8)
    row_layout: gl.constexpr = gl.SliceLayout(1, mfma)
    col_layout: gl.constexpr = gl.SliceLayout(0, mfma)

    shared_a: gl.constexpr = _operand_shared_layout(
        dot_a, [BLOCK_M, BLOCK_K], a_ptr.dtype.element_ty, 1
    )
    shared_b: gl.constexpr = _operand_shared_layout(
        dot_b, [BLOCK_K, BLOCK_N], b_ptr.dtype.element_ty, 0
    )
    gload_a: gl.constexpr = _copy_layout(shared_a, [BLOCK_M, BLOCK_K], NUM_WARPS)
    gload_b: gl.constexpr = _copy_layout(shared_b, [BLOCK_K, BLOCK_N], NUM_WARPS)
    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_K, BLOCK_N], shared_b
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gload_a))
    a_k_offs = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gload_a))
    # Activation rows past M are masked (no memory traffic, zero fill); weight
    # rows past N are clamped to the last row. The store mask drops both.
    a_rows = pid_m * BLOCK_M + offs_am
    a_row_mask = (a_rows < M)[:, None]
    a_offsets = a_rows[:, None] * stride_am + (k_begin + a_k_offs)[None, :]
    if ALIGNED_VALUES:
        a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 16]), [1, 16])

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gload_b))
    b_k_offs = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gload_b))
    b_rows = gl.minimum(pid_n * BLOCK_N + offs_bn, N - 1)
    b_offsets = b_rows[None, :] * stride_bn + (k_begin + b_k_offs)[:, None]
    if ALIGNED_VALUES:
        b_offsets = gl.max_contiguous(gl.multiple_of(b_offsets, [16, 1]), [16, 1])
    if K_TAIL or not ALIGNED_VALUES:
        a_k_offs = a_k_offs + k_begin
        b_k_offs = b_k_offs + k_begin

    as_rows = gl.minimum(pid_m * BLOCK_M + gl.arange(0, BLOCK_M, row_layout), M - 1)
    bs_rows = gl.minimum(pid_n * BLOCK_N + gl.arange(0, BLOCK_N, col_layout), N - 1)
    if FP32_SCALES:
        as_offsets = as_rows * stride_asm + (k_begin // 128) * stride_ask
        bs_offsets = (bs_rows // 128) * stride_bsn + (k_begin // 128) * stride_bsk
    elif PACKED_SCALES:
        # Scale pointers are int32 views and strides count dwords.
        as_offsets = as_rows * stride_asm + k_begin // (4 * SCALE_GROUP)
        bs_offsets = bs_rows * stride_bsn + k_begin // (4 * SCALE_GROUP)
    else:
        as_offsets = as_rows * stride_asm + k_begin // SCALE_GROUP
        bs_offsets = bs_rows * stride_bsn + k_begin // SCALE_GROUP

    # Software pipeline over the K tiles: fill NUM_BUFFERS - 1 stages, run a
    # runtime loop that consumes one stage and issues the next, then drain.
    # Only the fill and drain are unrolled, so code size does not grow with K.
    # The in-flight stages' scales ride in a tuple rotated every iteration.
    scales = ()
    for t in gl.static_range(NUM_BUFFERS - 1):
        scales = scales + (
            _issue_stage(
                smem_a.index(t),
                smem_b.index(t),
                a_ptr,
                b_ptr,
                a_scales_ptr,
                b_scales_ptr,
                a_offsets,
                b_offsets,
                as_offsets,
                bs_offsets,
                a_k_offs,
                b_k_offs,
                a_row_mask,
                t * BLOCK_K,
                k_begin,
                K,
                K_TAIL,
                GROUPS_PER_TILE,
                PACKED_SCALES,
                FP32_SCALES,
                ALIGNED_VALUES,
                stride_ask,
                stride_bsk,
            ),
        )

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    for t in range(TILES_PER_SPLIT - (NUM_BUFFERS - 1)):
        # Every iteration issues one stage, so NUM_BUFFERS - 2 stay in flight.
        async_copy.wait_group(NUM_BUFFERS - 2)
        a_s, b_s = scales[0]
        issue = t + NUM_BUFFERS - 1
        next_scales = _issue_stage(
            smem_a.index(issue % NUM_BUFFERS),
            smem_b.index(issue % NUM_BUFFERS),
            a_ptr,
            b_ptr,
            a_scales_ptr,
            b_scales_ptr,
            a_offsets,
            b_offsets,
            as_offsets,
            bs_offsets,
            a_k_offs,
            b_k_offs,
            a_row_mask,
            issue * BLOCK_K,
            k_begin,
            K,
            K_TAIL,
            GROUPS_PER_TILE,
            PACKED_SCALES,
            FP32_SCALES,
            ALIGNED_VALUES,
            stride_ask,
            stride_bsk,
        )
        acc = _mma_tile(
            smem_a.index(t % NUM_BUFFERS),
            smem_b.index(t % NUM_BUFFERS),
            a_s,
            b_s,
            acc,
            dot_a,
            dot_b,
            mfma,
            BLOCK_M,
            BLOCK_N,
            GROUPS_PER_TILE,
            PACKED_SCALES,
            FP32_SCALES,
        )
        scales = scales[1:] + (next_scales,)
    for e in gl.static_range(NUM_BUFFERS - 1):
        t = TILES_PER_SPLIT - (NUM_BUFFERS - 1) + e
        async_copy.wait_group(NUM_BUFFERS - 2 - e)
        a_s, b_s = scales[e]
        acc = _mma_tile(
            smem_a.index(t % NUM_BUFFERS),
            smem_b.index(t % NUM_BUFFERS),
            a_s,
            b_s,
            acc,
            dot_a,
            dot_b,
            mfma,
            BLOCK_M,
            BLOCK_N,
            GROUPS_PER_TILE,
            PACKED_SCALES,
            FP32_SCALES,
        )

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    cm = pid_m * BLOCK_M + offs_cm
    cn = pid_n * BLOCK_N + offs_cn
    if SPLIT_K > 1:
        tile_elems: gl.constexpr = BLOCK_M * BLOCK_N
        partial_offsets = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]
        cdna4.buffer_store(
            ptr=partial_ptr + (split * num_tiles + tile) * tile_elems,
            offsets=partial_offsets,
            stored_value=acc,
            cache=".wt",
        )
        # Publish the partial without an acq_rel atomic, whose L2 writeback and
        # invalidate cost more than the GEMM: ".wt" writes through to memory,
        # every wave waits for its stores to complete (vmcnt), and the barrier
        # orders all waves' stores before the counter bump.
        gl.inline_asm_elementwise(
            asm="s_waitcnt vmcnt(0)",
            constraints="=v,0",
            args=[offs_cm],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.barrier()
        old = gl.atomic_add(counter_ptr + tile, 1, sem="relaxed", scope="gpu")
        if old != SPLIT_K - 1:
            return
        # Sum in split order, not arrival order, so results are deterministic.
        # ".cv" bypasses caches so partials from other XCDs are not stale.
        acc = cdna4.buffer_load(
            ptr=partial_ptr + tile * tile_elems,
            offsets=partial_offsets,
            cache=".cv",
        )
        for s in gl.static_range(1, SPLIT_K):
            acc += cdna4.buffer_load(
                ptr=partial_ptr + (s * num_tiles + tile) * tile_elems,
                offsets=partial_offsets,
                cache=".cv",
            )
        gl.store(counter_ptr + tile, 0)
    c_offsets = cm[:, None] * stride_cm + cn[None, :]
    c_mask = (cm[:, None] < M) & (cn[None, :] < N)
    cdna4.buffer_store(
        ptr=c_ptr,
        offsets=c_offsets,
        stored_value=acc.to(c_ptr.dtype.element_ty),
        mask=c_mask,
    )


def _get_partial_scratch(device: torch.device, numel: int) -> torch.Tensor:
    """Return stream-local FP32 split-K partials.

    Same-stream launches are ordered and can share scratch. During HIP graph
    capture, allocate from the graph-aware allocator instead of caching.
    """
    if torch.cuda.is_current_stream_capturing():
        return torch.empty((numel,), device=device, dtype=torch.float32)
    device_index = torch.cuda.current_device() if device.index is None else device.index
    stream_id = torch.cuda.current_stream(device_index).cuda_stream
    key = (device_index, stream_id, numel)
    cached = _partial_cache.get(key)
    if cached is None:
        cached = torch.empty((numel,), device=device, dtype=torch.float32)
        _partial_cache[key] = cached
    return cached


def _launch(A, B, A_scales, B_scales, C, config, counters):
    block_m, block_n, block_k, warps_m, warps_n, num_buffers, split_k, num_xcds = config
    M, K = A.shape
    N = B.shape[0]
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    if split_k > 1:
        check_splitk_counters(counters, num_tiles, A.device, "gfx950 MXFP8 decode GEMM")
        partial = _get_partial_scratch(
            A.device, num_tiles * split_k * block_m * block_n
        )
    else:
        partial = counters = C
    # Four-group scale words need 4-byte aligned rows and whole words per tile.
    # A K tail would read the next row's scale bytes into the padded groups.
    fp32_scales = A_scales.dtype == torch.float32
    # K128 FP8 operands need more per-wave state than the native BF16 K32
    # arithmetic; distribute a full 64-square tile over four waves.
    if fp32_scales and block_m == 64 and block_n == 64 and warps_m * warps_n < 4:
        warps_m, warps_n = 2, 2
    packed = (
        not fp32_scales and K % block_k == 0 and block_k % (4 * MXFP8_SCALE_GROUP) == 0
    ) and all(
        scale.stride(0) % 4 == 0 and scale.data_ptr() % 4 == 0
        for scale in (A_scales, B_scales)
    )
    if packed:
        A_scales = A_scales.view(torch.int32)
        B_scales = B_scales.view(torch.int32)
    gluon_mm_mxfp8_decode_gfx950[(num_tiles * split_k,)](
        A,
        B,
        A_scales,
        B_scales,
        C,
        partial,
        counters,
        M,
        N,
        K,
        A.stride(0),
        B.stride(0),
        A_scales.stride(0),
        B_scales.stride(0),
        C.stride(0),
        A_scales.stride(1),
        B_scales.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        NUM_BUFFERS=num_buffers,
        SPLIT_K=split_k,
        NUM_XCDS=num_xcds,
        PACKED_SCALES=packed,
        FP32_SCALES=fp32_scales,
        ALIGNED_VALUES=all(
            value.stride(0) % 16 == 0 and value.data_ptr() % 16 == 0 for value in (A, B)
        ),
        num_warps=warps_m * warps_n,
    )
    return C


# Tile configuration: (BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N,
# NUM_BUFFERS, SPLIT_K, NUM_XCDS). A call uses the config of the smallest M
# bucket covering its M, so M stays a runtime argument within a bucket.
DECODE_M_BUCKETS = (16, 32, 64, 128, 256)
MAX_M = DECODE_M_BUCKETS[-1]
_Config = tuple[int, int, int, int, int, int, int, int]

# Sweeps on gfx950 at the DeepSeek V4.1 TP4 decode shapes (graph-replayed,
# rotating buffers), keyed by (N, K) and then by M bucket (measured at
# M = 6, 24, 48, 96, 192). The 2**26-element weights (25600x6144, 5120x15360)
# keep no 128/256 buckets: mm() routes them to the large-tile kernel from 64
# rows on.
# Other shapes use _heuristic_config.
_DECODE_CONFIGS: dict[tuple[int, int], dict[int, _Config]] = {
    (8192, 1280): {
        16: (16, 16, 256, 1, 1, 6, 1, 8),
        32: (32, 32, 256, 2, 2, 2, 1, 8),
        64: (32, 64, 256, 2, 4, 4, 1, 8),
        128: (16, 64, 256, 1, 4, 3, 1, 8),
        256: (64, 64, 128, 1, 4, 2, 2, 8),
    },
    (5120, 2048): {
        16: (16, 32, 256, 1, 2, 2, 1, 8),
        32: (32, 32, 256, 2, 2, 6, 1, 8),
        64: (16, 128, 128, 1, 4, 3, 2, 8),
        128: (16, 64, 256, 1, 4, 3, 1, 8),
        256: (64, 32, 256, 2, 2, 3, 1, 8),
    },
    (1152, 5120): {
        16: (16, 32, 256, 1, 2, 3, 5, 8),
        32: (16, 64, 256, 1, 4, 2, 5, 8),
        64: (16, 16, 128, 1, 1, 3, 5, 8),
        128: (32, 16, 256, 2, 1, 3, 4, 8),
        256: (64, 64, 128, 2, 4, 2, 4, 8),
    },
    (5120, 576): {
        16: (16, 32, 256, 1, 2, 2, 1, 8),
        32: (32, 32, 128, 2, 2, 2, 1, 8),
        64: (16, 16, 128, 1, 1, 6, 1, 8),
        128: (32, 16, 128, 1, 1, 3, 1, 8),
        256: (64, 16, 128, 2, 1, 2, 1, 8),
    },
    (1792, 5120): {
        16: (16, 16, 256, 1, 1, 2, 5, 8),
        32: (32, 32, 256, 2, 2, 2, 5, 8),
        64: (16, 128, 256, 1, 4, 3, 4, 8),
        128: (32, 32, 256, 2, 2, 3, 5, 8),
        256: (64, 32, 256, 4, 2, 2, 4, 8),
    },
    (4096, 1280): {
        16: (16, 16, 128, 1, 1, 6, 1, 8),
        32: (32, 32, 256, 2, 2, 4, 1, 8),
        64: (16, 16, 128, 1, 1, 4, 1, 8),
        128: (64, 32, 256, 4, 1, 4, 1, 8),
        256: (64, 64, 128, 2, 4, 2, 1, 8),
    },
    (1536, 5120): {
        16: (16, 16, 256, 1, 1, 3, 4, 8),
        32: (16, 64, 256, 1, 4, 2, 5, 8),
        64: (64, 32, 256, 2, 2, 6, 4, 8),
        128: (64, 32, 128, 4, 1, 2, 5, 8),
        256: (32, 32, 128, 1, 1, 2, 5, 8),
    },
    (25600, 6144): {
        16: (16, 64, 256, 1, 4, 2, 1, 8),
        32: (32, 64, 256, 1, 4, 2, 4, 8),
        64: (64, 128, 128, 2, 2, 2, 2, 8),
    },
    (5120, 15360): {
        16: (16, 64, 128, 1, 2, 3, 8, 8),
        32: (32, 32, 128, 1, 1, 3, 4, 8),
        64: (64, 32, 128, 2, 2, 3, 5, 8),
    },
}

# Untuned fallback per M bucket: (BLOCK_M, BLOCK_N, WARPS_M, WARPS_N,
# NUM_BUFFERS) with BLOCK_K = 128, the most common winners above.
_HEURISTIC_TILES: dict[int, tuple[int, int, int, int, int]] = {
    16: (16, 32, 1, 2, 8),
    32: (32, 32, 1, 2, 6),
    64: (32, 64, 2, 2, 4),
    128: (64, 64, 2, 2, 4),
    256: (64, 64, 1, 2, 3),
}
# Split K until the grid reaches this many programs.
_MIN_PROGRAMS = 128


def _heuristic_config(M: int, N: int, K: int, bucket: int) -> _Config:
    block_m, block_n, warps_m, warps_n, num_buffers = _HEURISTIC_TILES[bucket]
    block_k = 128
    k_tiles = triton.cdiv(K, block_k)
    tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    split_k = 1
    for candidate in (2, 4, 5, 8):
        if tiles * split_k >= _MIN_PROGRAMS:
            break
        if k_tiles % candidate == 0 and k_tiles // candidate >= 4:
            split_k = candidate
    num_buffers = min(num_buffers, max(2, k_tiles // split_k))
    return (block_m, block_n, block_k, warps_m, warps_n, num_buffers, split_k, 8)


def _choose_config(M: int, N: int, K: int) -> _Config:
    bucket = next(bound for bound in DECODE_M_BUCKETS if M <= bound)
    tuned = _DECODE_CONFIGS.get((N, K), {}).get(bucket)
    if tuned is not None:
        return tuned
    return _heuristic_config(M, N, K, bucket)


def gluon_mm_mxfp8_decode_num_counters_gfx950(m: int, n: int, k: int) -> int:
    """Split-K arrival counters a decode MXFP8 launch needs; 0 without split-K."""
    block_m, block_n, *_, split_k, _ = _choose_config(m, n, k)
    if split_k == 1:
        return 0
    return triton.cdiv(m, block_m) * triton.cdiv(n, block_n)


def supports_gluon_mm_mxfp8_decode_gfx950(m: int, n: int, k: int) -> bool:
    """Whether the decode MXFP8 GEMM covers an ``[m, k] x [n, k]`` problem."""
    return (
        1 <= m <= MAX_M
        and n >= 16
        and k >= 128
        and k % MXFP8_SCALE_GROUP == 0
        # Buffer offsets are 32-bit.
        and n * k < 2**31
    )


def _validate_scale(name: str, scale: torch.Tensor, rows: int, k: int) -> None:
    expected = (rows, k // MXFP8_SCALE_GROUP)
    if scale.dtype != torch.uint8 or tuple(scale.shape) != expected:
        raise ValueError(
            f"{name} must be uint8 E8M0 with shape {expected}, "
            f"got dtype={scale.dtype}, shape={tuple(scale.shape)}"
        )
    if scale.stride(1) != 1:
        raise ValueError(f"{name} must have unit inner stride")


def launch_gluon_mm_mxfp8_decode_gfx950(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None,
    block_size: list[int],
    counters: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a decode-sized MXFP8 ``A @ B.T`` on gfx950.

    Args:
        A: E4M3 activation ``[M, K]`` with ``1 <= M <= 256`` and unit inner
            stride. Native UE8M0 additionally requires 16-byte-aligned rows.
        B: E4M3 weight ``[N, K]`` with the same value-stride contract;
            ``K`` must be a multiple of 32.
        A_scales: uint8 UE8M0 ``[M, K/32]`` with unit inner stride, or FP32
            ``[M, ceil(K/128)]`` with arbitrary strides.
        B_scales: uint8 UE8M0 ``[N, K/32]`` with unit inner stride, or FP32
            ``[ceil(N/128), ceil(K/128)]`` with arbitrary strides.
        out_dtype: BF16 or FP16 output element type.
        alpha: Optional output multiplier applied after the GEMM.
        block_size: ``[1, 32]`` for UE8M0 or ``[128, 128]`` for FP32 scales;
            the latter requires BF16 output.
        counters: Zeroed int32 split-K arrival counters, at least
            ``gluon_mm_mxfp8_decode_num_counters_gfx950(M, N, K)`` of them;
            the launch leaves them zeroed (see ``splitk.py``).
        out: Optional ``[M, N]`` output with unit inner stride.

    Returns:
        The supplied ``out`` tensor or a newly allocated ``[M, N]`` tensor.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("gfx950 MXFP8 decode GEMM requires rank-2 A and B")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise TypeError("gfx950 MXFP8 decode GEMM requires E4M3 A and B")
    if not A.is_cuda or B.device != A.device:
        raise ValueError("gfx950 MXFP8 decode GEMM requires colocated GPU operands")
    fp32_scales = list(block_size) == [128, 128]
    if list(block_size) not in ([1, MXFP8_SCALE_GROUP], [128, 128]):
        raise ValueError("gfx950 decode GEMM requires block_size=[1, 32] or [128, 128]")
    if out_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise TypeError(f"gfx950 MXFP8 decode GEMM does not support {out_dtype}")
    if fp32_scales and out_dtype != torch.bfloat16:
        raise TypeError("FP32 block-scale decode GEMM requires BF16 output")
    M, K = A.shape
    N, b_k = B.shape
    if b_k != K:
        raise ValueError(f"gfx950 MXFP8 decode GEMM K mismatch: A={K}, B={b_k}")
    if not supports_gluon_mm_mxfp8_decode_gfx950(M, N, K):
        raise ValueError(f"gfx950 MXFP8 decode GEMM does not cover M={M}, N={N}, K={K}")
    for name, tensor in (("A", A), ("B", B)):
        if (
            tensor.stride(1) != 1
            or (not fp32_scales and tensor.stride(0) % 16 != 0)
            or (not fp32_scales and tensor.data_ptr() % 16 != 0)
        ):
            raise ValueError(
                f"gfx950 MXFP8 decode GEMM requires {name} with unit inner "
                "stride and 16-byte-aligned rows"
            )
    if A_scales.device != A.device or B_scales.device != A.device:
        raise ValueError("gfx950 MXFP8 decode scales must share the operand device")
    if fp32_scales:
        for name, scale, rows in (
            ("A_scales", A_scales, M),
            ("B_scales", B_scales, triton.cdiv(N, 128)),
        ):
            if scale.dtype != torch.float32 or tuple(scale.shape) != (
                rows,
                triton.cdiv(K, 128),
            ):
                raise ValueError(
                    f"{name} must be FP32 with shape {(rows, triton.cdiv(K, 128))}, "
                    f"got dtype={scale.dtype}, shape={tuple(scale.shape)}"
                )
    else:
        _validate_scale("A_scales", A_scales, M, K)
        _validate_scale("B_scales", B_scales, N, K)

    if out is None:
        output = torch.empty((M, N), device=A.device, dtype=out_dtype)
    else:
        output = out
        if (
            tuple(output.shape) != (M, N)
            or output.dtype != out_dtype
            or output.device != A.device
            or output.stride(1) != 1
        ):
            raise ValueError(
                "gfx950 MXFP8 decode out must have matching shape/device/dtype "
                "and unit inner stride"
            )
    _launch(A, B, A_scales, B_scales, output, _choose_config(M, N, K), counters)
    if alpha is not None:
        output.mul_(alpha.to(device=output.device, dtype=output.dtype))
    return output


__all__ = [
    "gluon_mm_mxfp8_decode_num_counters_gfx950",
    "launch_gluon_mm_mxfp8_decode_gfx950",
    "supports_gluon_mm_mxfp8_decode_gfx950",
]
