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

Computes ``A @ B.T`` for E4M3 ``A [M, K]`` and ``B [N, K]`` with one uint8
UE8M0 scale per 32 K values on each row, FP32 accumulation, and BF16/FP16
output. Decode GEMMs stream the weight once while the activation stays in L2,
so the kernel is organized around weight bandwidth: small N tiles, optional
split-K to occupy every CU, and a multi-buffered async global-to-LDS pipeline.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy

MXFP8_SCALE_GROUP = 32
SCALE_GROUP = gl.constexpr(MXFP8_SCALE_GROUP)
_SUPPORTED_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}

_partial_cache: dict[tuple[int, int, int], torch.Tensor] = {}
_counter_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _mxfp8_decode_launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    scale_values = (m + n) * (k // MXFP8_SCALE_GROUP)
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
    as_k_offs,
    bs_k_offs,
    a_row_mask,
    k0,
    K: gl.constexpr,
    K_TAIL: gl.constexpr,
):
    """Start loading one K tile: scales into registers, values into LDS.

    Scales go first: loads retire in issue order, so the wait for this tile's
    values also covers its scales.
    """
    kg = k0 // SCALE_GROUP
    if K_TAIL:
        groups: gl.constexpr = K // SCALE_GROUP
        a_s = cdna4.buffer_load(
            a_scales_ptr,
            as_offsets + kg,
            mask=(as_k_offs + kg < groups)[None, :],
            other=127,
        )
        b_s = cdna4.buffer_load(
            b_scales_ptr,
            bs_offsets + kg,
            mask=(bs_k_offs + kg < groups)[None, :],
            other=127,
        )
        # Zero both operands past K: an unmasked B tail would read the next
        # row, and zero times a NaN code would poison the accumulator.
        async_copy.buffer_load_to_shared(
            smem_a,
            a_ptr,
            a_offsets + k0,
            mask=a_row_mask & (a_k_offs + k0 < K)[None, :],
        )
        async_copy.buffer_load_to_shared(
            smem_b, b_ptr, b_offsets + k0, mask=(b_k_offs + k0 < K)[:, None]
        )
    else:
        a_s = cdna4.buffer_load(a_scales_ptr, as_offsets + kg)
        b_s = cdna4.buffer_load(b_scales_ptr, bs_offsets + kg)
        async_copy.buffer_load_to_shared(smem_a, a_ptr, a_offsets + k0, mask=a_row_mask)
        async_copy.buffer_load_to_shared(smem_b, b_ptr, b_offsets + k0)
    async_copy.commit_group()
    return a_s, b_s


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
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """Multi-buffered scaled-MFMA GEMM for decode-sized ``M``.

    Program ids enumerate (split, n tile, m tile) with the M tile fastest, so
    programs that share a weight tile are adjacent and, after the XCD remap,
    share one XCD's L2. Each program streams its K slice through a
    ``NUM_BUFFERS``-deep ring of LDS buffers filled by async copies; the
    scale bytes of each K tile are loaded into registers just ahead of its
    values. ``K`` is a model constant, so the K loop is fully unrolled.

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

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma, 16)
    scale_a: gl.constexpr = cdna4.get_mfma_scale_layout(
        dot_a, [BLOCK_M, BLOCK_K // SCALE_GROUP]
    )
    scale_b: gl.constexpr = cdna4.get_mfma_scale_layout(
        dot_b, [BLOCK_N, BLOCK_K // SCALE_GROUP]
    )

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
    a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 16]), [1, 16])

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gload_b))
    b_k_offs = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gload_b))
    b_rows = gl.minimum(pid_n * BLOCK_N + offs_bn, N - 1)
    b_offsets = b_rows[None, :] * stride_bn + (k_begin + b_k_offs)[:, None]
    b_offsets = gl.max_contiguous(gl.multiple_of(b_offsets, [16, 1]), [16, 1])
    if K_TAIL:
        a_k_offs = a_k_offs + k_begin
        b_k_offs = b_k_offs + k_begin

    offs_asm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, scale_a))
    as_k_offs = gl.arange(0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, scale_a))
    as_rows = gl.minimum(pid_m * BLOCK_M + offs_asm, M - 1)
    as_offsets = (
        as_rows[:, None] * stride_asm + (k_begin // SCALE_GROUP + as_k_offs)[None, :]
    )
    offs_bsn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, scale_b))
    bs_k_offs = gl.arange(0, BLOCK_K // SCALE_GROUP, gl.SliceLayout(0, scale_b))
    bs_rows = gl.minimum(pid_n * BLOCK_N + offs_bsn, N - 1)
    bs_offsets = (
        bs_rows[:, None] * stride_bsn + (k_begin // SCALE_GROUP + bs_k_offs)[None, :]
    )
    if K_TAIL:
        as_k_offs = as_k_offs + k_begin // SCALE_GROUP
        bs_k_offs = bs_k_offs + k_begin // SCALE_GROUP

    # The K loop is unrolled at compile time (K is a model constant), so the
    # stage scales are kept in a tuple indexed by the static tile number.
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
                as_k_offs,
                bs_k_offs,
                a_row_mask,
                t * BLOCK_K,
                K,
                K_TAIL,
            ),
        )

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    for t in gl.static_range(TILES_PER_SPLIT):
        # Tiles t + 1 .. t + NUM_BUFFERS - 2 may stay in flight.
        async_copy.wait_group(min(NUM_BUFFERS - 2, TILES_PER_SPLIT - 1 - t))
        a = smem_a.index(t % NUM_BUFFERS).load(dot_a)
        b = smem_b.index(t % NUM_BUFFERS).load(dot_b)
        if t + NUM_BUFFERS - 1 < TILES_PER_SPLIT:
            scales = scales + (
                _issue_stage(
                    smem_a.index((t + NUM_BUFFERS - 1) % NUM_BUFFERS),
                    smem_b.index((t + NUM_BUFFERS - 1) % NUM_BUFFERS),
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
                    as_k_offs,
                    bs_k_offs,
                    a_row_mask,
                    (t + NUM_BUFFERS - 1) * BLOCK_K,
                    K,
                    K_TAIL,
                ),
            )
        a_s, b_s = scales[t]
        acc = cdna4.mfma_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)

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


def _get_splitk_counters(device: torch.device, num_tiles: int) -> torch.Tensor:
    """Return stream-local zeroed per-tile arrival counters.

    The last split of each tile resets its counter, so the buffer is zero
    again whenever a launch completes and same-stream launches can share it.
    """
    if torch.cuda.is_current_stream_capturing():
        # Keep captured launches on graph-private counters.
        return torch.zeros((num_tiles,), dtype=torch.int32, device=device)
    device_index = torch.cuda.current_device() if device.index is None else device.index
    stream_id = torch.cuda.current_stream(device_index).cuda_stream
    key = (device_index, stream_id, num_tiles)
    counters = _counter_cache.get(key)
    if counters is None:
        counters = torch.zeros((num_tiles,), dtype=torch.int32, device=device)
        _counter_cache[key] = counters
    return counters


def _launch(A, B, A_scales, B_scales, C, config):
    block_m, block_n, block_k, warps_m, warps_n, num_buffers, split_k, num_xcds = config
    M, K = A.shape
    N = B.shape[0]
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    if split_k > 1:
        partial = _get_partial_scratch(
            A.device, num_tiles * split_k * block_m * block_n
        )
        counters = _get_splitk_counters(A.device, num_tiles)
    else:
        partial = counters = C
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        NUM_BUFFERS=num_buffers,
        SPLIT_K=split_k,
        NUM_XCDS=num_xcds,
        num_warps=warps_m * warps_n,
    )
    return C


# Tile configuration: (BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N,
# NUM_BUFFERS, SPLIT_K, NUM_XCDS). A call uses the config of the smallest M
# bucket covering its M, so M stays a runtime argument within a bucket.
DECODE_M_BUCKETS = (16, 32, 64, 128, 256)
MAX_M = DECODE_M_BUCKETS[-1]
_Config = tuple[int, int, int, int, int, int, int, int]

# Cold-cache rocprofv3 sweeps on gfx950 at the DeepSeek V4.1 TP4 decode shapes,
# keyed by (N, K) and then by M bucket (measured at M = 6, 24, 48, 96, 192).
# Other shapes use _heuristic_config.
_DECODE_CONFIGS: dict[tuple[int, int], dict[int, _Config]] = {
    (8192, 1280): {
        16: (16, 32, 128, 1, 2, 10, 1, 8),
        32: (32, 32, 128, 1, 2, 6, 1, 8),
        64: (32, 64, 128, 2, 2, 4, 1, 8),
        128: (64, 64, 128, 2, 2, 4, 1, 8),
        256: (128, 64, 128, 4, 1, 3, 1, 8),
    },
    (5120, 2048): {
        16: (16, 32, 128, 1, 1, 8, 1, 8),
        32: (32, 32, 128, 1, 2, 6, 1, 8),
        64: (16, 64, 128, 1, 2, 6, 1, 8),
        128: (32, 64, 128, 2, 2, 4, 1, 8),
        256: (64, 64, 128, 1, 2, 3, 1, 8),
    },
    (1152, 5120): {
        16: (16, 32, 128, 1, 1, 8, 4, 8),
        32: (16, 16, 128, 1, 1, 8, 1, 8),
        64: (16, 16, 256, 1, 1, 6, 1, 8),
        128: (32, 32, 128, 2, 1, 6, 2, 8),
        256: (64, 32, 128, 2, 1, 3, 2, 8),
    },
    (5120, 576): {
        16: (16, 32, 128, 1, 2, 5, 1, 8),
        32: (16, 16, 128, 1, 1, 5, 1, 8),
        64: (16, 64, 128, 1, 2, 5, 1, 8),
        128: (32, 64, 128, 2, 2, 4, 1, 8),
        256: (64, 64, 128, 1, 4, 3, 1, 8),
    },
    (1792, 5120): {
        16: (16, 32, 128, 1, 1, 8, 4, 8),
        32: (16, 16, 128, 1, 1, 8, 1, 8),
        64: (16, 32, 128, 1, 1, 6, 1, 8),
        128: (64, 64, 128, 2, 2, 4, 4, 8),
        256: (64, 64, 256, 2, 1, 3, 2, 8),
    },
    (4096, 1280): {
        16: (16, 32, 128, 1, 2, 8, 1, 8),
        32: (32, 16, 128, 2, 1, 8, 1, 8),
        64: (32, 32, 128, 2, 1, 6, 1, 8),
        128: (64, 32, 128, 2, 1, 6, 1, 8),
        256: (64, 64, 128, 1, 2, 3, 1, 8),
    },
    (1536, 5120): {
        16: (16, 32, 128, 1, 1, 8, 5, 8),
        32: (16, 16, 128, 1, 1, 8, 1, 8),
        64: (64, 32, 128, 2, 1, 6, 5, 8),
        128: (64, 32, 128, 2, 1, 6, 5, 8),
        256: (64, 64, 256, 2, 1, 3, 2, 8),
    },
    (25600, 6144): {
        16: (16, 32, 128, 1, 2, 8, 1, 8),
        32: (32, 32, 256, 2, 1, 3, 1, 8),
        64: (64, 64, 128, 2, 2, 4, 1, 8),
        128: (64, 128, 128, 1, 2, 3, 1, 8),
        256: (64, 128, 128, 1, 2, 3, 1, 8),
    },
    (5120, 15360): {
        16: (16, 32, 128, 1, 1, 8, 1, 8),
        32: (32, 64, 128, 2, 2, 4, 5, 8),
        64: (16, 64, 256, 1, 2, 3, 1, 8),
        128: (64, 128, 128, 1, 2, 3, 5, 8),
        256: (64, 128, 128, 1, 2, 3, 2, 8),
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
    tuned = _DECODE_CONFIGS.get((N, K))
    if tuned is not None:
        return tuned[bucket]
    return _heuristic_config(M, N, K, bucket)


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
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a decode-sized MXFP8 ``A @ B.T`` on gfx950.

    Args:
        A: E4M3 activation ``[M, K]`` with ``1 <= M <= 256``, unit inner
            stride and a 16-byte-aligned row stride.
        B: E4M3 weight ``[N, K]`` with unit inner stride and a 16-byte-aligned
            row stride; ``K`` must be a multiple of 32.
        A_scales: uint8 UE8M0 scales ``[M, K/32]`` with unit inner stride.
        B_scales: uint8 UE8M0 scales ``[N, K/32]`` with unit inner stride.
        out_dtype: BF16 or FP16 output element type.
        alpha: Optional output multiplier applied after the GEMM.
        block_size: Logical scale block; must be ``[1, 32]``.
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
    if list(block_size) != [1, MXFP8_SCALE_GROUP]:
        raise ValueError(
            f"gfx950 MXFP8 decode GEMM requires block_size=[1, {MXFP8_SCALE_GROUP}]"
        )
    if out_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise TypeError(f"gfx950 MXFP8 decode GEMM does not support {out_dtype}")
    M, K = A.shape
    N, b_k = B.shape
    if b_k != K:
        raise ValueError(f"gfx950 MXFP8 decode GEMM K mismatch: A={K}, B={b_k}")
    if not supports_gluon_mm_mxfp8_decode_gfx950(M, N, K):
        raise ValueError(f"gfx950 MXFP8 decode GEMM does not cover M={M}, N={N}, K={K}")
    for name, tensor in (("A", A), ("B", B)):
        if (
            tensor.stride(1) != 1
            or tensor.stride(0) % 16 != 0
            or tensor.data_ptr() % 16 != 0
        ):
            raise ValueError(
                f"gfx950 MXFP8 decode GEMM requires {name} with unit inner "
                "stride and 16-byte-aligned rows"
            )
    if A_scales.device != A.device or B_scales.device != A.device:
        raise ValueError("gfx950 MXFP8 decode scales must share the operand device")
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
    _launch(A, B, A_scales, B_scales, output, _choose_config(M, N, K))
    if alpha is not None:
        output.mul_(alpha.to(device=output.device, dtype=output.dtype))
    return output


__all__ = [
    "launch_gluon_mm_mxfp8_decode_gfx950",
    "supports_gluon_mm_mxfp8_decode_gfx950",
]
