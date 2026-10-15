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

"""Dense MXFP4 linears with in-kernel dynamic activation quantization.

``y = mxfp4(x) @ W.T`` for BF16 ``x [M, K]`` and a packed MXFP4 weight
``W [N, K/2]`` (two E2M1 values per byte, low nibble first, one UE8M0 scale
per 32 K values). Each BF16 K tile is quantized inside the kernel with the
dynamic per-32 recipe of ``mxfp4_quantize`` (power-of-two scale from the
rounded group amax, E2M1 round-to-nearest-even with saturation), bit-identical
to quantizing first, so there is no separate quantization launch or HBM round
trip. Products run on CDNA4 scaled MFMAs (``16x16x128``) with FP32
accumulation. The weight is stored preshuffled (``preshuffle_mxfp4_weight``)
so each MFMA operand load reads 1 KiB contiguously.

Two kernels share the main loop and differ in the epilogue:

* ``gluon_mxfp4_linear_gfx950`` writes ``y [M, N]`` in BF16.
* ``gluon_mxfp4_swiglu_linear_gfx950`` reads a ``W`` that stacks the gate
  rows ``[0, N/2)`` over the up rows ``[N/2, N)``. A tile's first
  ``BLOCK_N / 2`` columns are gate rows and the rest the matching up rows.
  The epilogue rounds both to BF16 (the unfused GEMM's output), applies
  ``silu(gate) * up`` in FP32 and writes the BF16 ``[M, N/2]`` result; the
  next projection quantizes it on load, where the unfused path re-quantized
  it.

Small batches split K across programs; the last arriving split of each tile
sums the FP32 partials in split order and runs the epilogue.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.decode import _remap_xcd
from tokenspeed_kernel_amd.ops.gfx950.gemm.splitk import check_splitk_counters

cdna4 = gl.amd.cdna4

MXFP4_GROUP = 32
GROUP = gl.constexpr(MXFP4_GROUP)

_partial_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _launch_metadata(grid, kernel, args):
    """Expose algorithmic FLOPs and tensor traffic to Proton."""
    m, n, k = args["M"], args["N"], args["K"]
    out_cols = n // 2 if kernel.name.startswith("gluon_mxfp4_swiglu") else n
    split_k = args["SPLIT_K"]
    partial_bytes = 2 * split_k * m * n * 4 if split_k > 1 else 0
    return {
        "name": kernel.name,
        "flops4": 2 * m * n * k,
        "bytes": m * k * 2
        + n * k // 2
        + n * k // MXFP4_GROUP
        + m * out_cols * 2
        + partial_bytes,
    }


@gluon.constexpr_function
def _quant_layout(block_m, block_k, num_warps):
    """Coalesced BF16 A-tile layout that splits quantization across warps.

    The MFMA A operand is replicated across the warps that tile N, so
    quantizing in that layout would repeat the work on every such warp and
    its loads would not coalesce. Here each lane holds 8 consecutive values
    (one 16-byte load, the minimum run of CDNA4's scaled downcast) and
    consecutive lanes hold consecutive runs, so every load instruction reads
    whole row segments contiguously. A 32-value group spans four lanes, whose
    amax is a short cross-lane reduction. Rows, then K, spread over the warps
    without replication; the quantized tile moves to the MFMA operand layout
    through LDS.
    """
    lanes_k = min(16, block_k // 8)
    lanes_m = 64 // lanes_k
    warps_m = max(1, min(num_warps, block_m // lanes_m))
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[lanes_m, lanes_k],
        warps_per_cta=[warps_m, num_warps // warps_m],
        order=[1, 0],
    )


@gluon.constexpr_function
def _gate_up_pair_layout(block_m, half, num_warps):
    """``[rows, HALF, 2]`` layout holding each gate/up pair in one lane."""
    lanes_n = min(16, half)
    return gl.BlockedLayout(
        size_per_thread=[1, 1, 2],
        threads_per_warp=[64 // lanes_n, lanes_n, 1],
        warps_per_cta=[num_warps, 1, 1],
        order=[2, 1, 0],
    )


@gluon.jit
def _quantize_operand(x, a_layout: gl.constexpr, a_scale_layout: gl.constexpr):
    """Quantize a BF16 ``[rows, BLOCK_K]`` tile to MFMA-ready E2M1 + UE8M0.

    Matches ``mxfp4_quantize``: round the group amax up at the third mantissa
    bit to a power of two, back off two exponents (so the largest value lands
    in E2M1's 4.0 binade), clamp the biased exponent to ``[0, 254]``, then
    convert with round-to-nearest-even and saturation.
    """
    ROWS: gl.constexpr = x.shape[0]
    COLS: gl.constexpr = x.shape[1]
    # |x| ordered as an integer: clearing the BF16 sign bit leaves magnitudes
    # whose bit patterns sort like the values, so the group amax is a packed
    # 16-bit integer max, and shifting it up 16 bits gives its FP32 bits.
    magnitude = x.to(gl.int16, bitcast=True) & 0x7FFF
    grouped = magnitude.reshape((ROWS, COLS // GROUP, GROUP))
    amax_bits = gl.max(grouped, axis=2).to(gl.int32) << 16
    rounded_bits = (amax_bits + 0x200000) & 0x7F800000
    exp_biased = (rounded_bits >> 23).to(gl.int32)
    scale = gl.minimum(gl.maximum(exp_biased - 2, 0), 254).to(gl.uint8)
    packed = cdna4.scaled_downcast(x, scale, "e2m1", axis=1)
    return (
        gl.convert_layout(packed, a_layout),
        gl.convert_layout(scale, a_scale_layout),
    )


@gluon.jit
def _load_stage(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    a_offsets,
    b_offsets,
    bs_offsets,
    a_k_offs,
    b_k_offs,
    bs_k_offs,
    k0,
    K: gl.constexpr,
    K_TAIL: gl.constexpr,
):
    """Issue every load of one K tile: BF16 A, preshuffled E2M1 B, B scales.

    The scalar base pointers advance with ``k0`` so the per-lane offsets stay
    loop invariant. In the preshuffled weight, ``k0`` K values are ``k0 / 64``
    whole 512-byte chunks, i.e. ``8 * k0`` bytes. Activation rows past ``M``
    were clamped to the last row by the caller (the epilogue drops them), so
    only a K tail needs masks: zero values with finite scales add exactly
    zero.
    """
    if K_TAIL:
        a = cdna4.buffer_load(
            a_ptr + k0, a_offsets, mask=(a_k_offs + k0 < K)[None, :], other=0.0
        )
        b = cdna4.buffer_load(
            b_ptr + 8 * k0,
            b_offsets,
            mask=(b_k_offs + k0 // 2 < K // 2)[:, None],
            other=0,
        )
        b_s = cdna4.buffer_load(
            b_scale_ptr + k0 // GROUP,
            bs_offsets,
            mask=(bs_k_offs + k0 // GROUP < K // GROUP)[None, :],
            other=127,
        )
    else:
        a = cdna4.buffer_load(a_ptr + k0, a_offsets)
        b = cdna4.buffer_load(b_ptr + 8 * k0, b_offsets)
        b_s = cdna4.buffer_load(b_scale_ptr + k0 // GROUP, bs_offsets)
    return a, b, b_s


@gluon.jit
def _mma_stage(stage, acc, dot_a: gl.constexpr, a_scale_layout: gl.constexpr):
    """Quantize one BF16 A tile and accumulate it."""
    a, b, b_s = stage
    a_q, a_s = _quantize_operand(a, dot_a, a_scale_layout)
    return cdna4.mfma_scaled(a_q, a_s, "e2m1", b, b_s, "e2m1", acc)


@gluon.jit
def _mxfp4_linear_body(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    stride_am,
    stride_cm,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_bsn: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    SWIGLU: gl.constexpr,
):
    """Shared main loop and epilogues.

    Program ids enumerate (split, m tile, n tile) with the N tile fastest, so
    programs reading the same activation rows are adjacent and, after the
    XCD remap, share one XCD's L2. The next K tile's loads are in flight in
    registers while the current one is quantized and multiplied.
    """
    gl.static_assert(BLOCK_K % 128 == 0)
    K_TILES: gl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    gl.static_assert(K_TILES % SPLIT_K == 0)
    TILES_PER_SPLIT: gl.constexpr = K_TILES // SPLIT_K
    K_TAIL: gl.constexpr = K % BLOCK_K != 0
    NUM_WARPS: gl.constexpr = WARPS_M * WARPS_N
    HALF: gl.constexpr = BLOCK_N // 2
    if SWIGLU:
        N_OUT: gl.constexpr = N // 2
        num_pid_n: gl.constexpr = (N_OUT + HALF - 1) // HALF
    else:
        N_OUT: gl.constexpr = N
        num_pid_n: gl.constexpr = (N + BLOCK_N - 1) // BLOCK_N

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_tiles = num_pid_m * num_pid_n
    if NUM_XCDS > 1:
        pid = _remap_xcd(pid, num_tiles * SPLIT_K, NUM_XCDS)
    tile = pid % num_tiles
    split = pid // num_tiles
    pid_n = tile % num_pid_n
    pid_m = tile // num_pid_n
    k_begin = split * (TILES_PER_SPLIT * BLOCK_K)

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    a_scale_layout: gl.constexpr = cdna4.get_mfma_scale_layout(
        dot_a, [BLOCK_M, BLOCK_K // GROUP]
    )
    b_scale_layout: gl.constexpr = cdna4.get_mfma_scale_layout(
        dot_b, [BLOCK_N, BLOCK_K // GROUP]
    )
    quant_layout: gl.constexpr = _quant_layout(BLOCK_M, BLOCK_K, NUM_WARPS)

    # Activation rows past M re-read the last row instead of paying masks;
    # the epilogue drops them.
    a_rows = gl.minimum(
        pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, quant_layout)),
        M - 1,
    )
    a_k_offs = gl.arange(0, BLOCK_K, gl.SliceLayout(0, quant_layout))
    a_offsets = a_rows[:, None] * stride_am + (k_begin + a_k_offs)[None, :]

    # Weight rows of this tile's columns. SwiGLU tiles pair HALF gate rows
    # with the up rows N_OUT further down. Rows past the matrix are clamped
    # to its last row (gate or up); their columns are dropped by the store.
    b_cols = gl.arange(0, BLOCK_N, gl.SliceLayout(0, dot_b))
    bs_cols = gl.arange(0, BLOCK_N, gl.SliceLayout(1, b_scale_layout))
    if SWIGLU:
        b_rows = gl.minimum(pid_n * HALF + (b_cols % HALF), N_OUT - 1) + gl.where(
            b_cols < HALF, 0, N_OUT
        )
        bs_rows = gl.minimum(pid_n * HALF + (bs_cols % HALF), N_OUT - 1) + gl.where(
            bs_cols < HALF, 0, N_OUT
        )
    else:
        b_rows = gl.minimum(pid_n * BLOCK_N + b_cols, N - 1)
        bs_rows = gl.minimum(pid_n * BLOCK_N + bs_cols, N - 1)
    # Byte (n, kp) of the preshuffled weight (see preshuffle_mxfp4_weight).
    b_k_offs = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(1, dot_b))
    b_offsets = (
        (b_rows // 16)[None, :] * (16 * (K // 2))
        + (b_k_offs // 32)[:, None] * 512
        + ((b_k_offs // 16) % 2)[:, None] * 256
        + (b_rows % 16)[None, :] * 16
        + (b_k_offs % 16)[:, None]
        + 8 * k_begin
    )
    bs_k_offs = gl.arange(0, BLOCK_K // GROUP, gl.SliceLayout(0, b_scale_layout))
    bs_offsets = bs_rows[:, None] * stride_bsn + (k_begin // GROUP + bs_k_offs)[None, :]
    if K_TAIL:
        a_k_offs = a_k_offs + k_begin
        b_k_offs = b_k_offs + k_begin // 2
        bs_k_offs = bs_k_offs + k_begin // GROUP

    # Register double buffer: the next K tile's loads are in flight while the
    # current one is consumed. The loop body handles two tiles so the buffers
    # swap roles instead of being copied each iteration.
    cur = _load_stage(
        a_ptr,
        b_ptr,
        b_scale_ptr,
        a_offsets,
        b_offsets,
        bs_offsets,
        a_k_offs,
        b_k_offs,
        bs_k_offs,
        0,
        K,
        K_TAIL,
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    for t in range((TILES_PER_SPLIT - 1) // 2):
        k0 = (2 * t + 1) * BLOCK_K
        nxt = _load_stage(
            a_ptr,
            b_ptr,
            b_scale_ptr,
            a_offsets,
            b_offsets,
            bs_offsets,
            a_k_offs,
            b_k_offs,
            bs_k_offs,
            k0,
            K,
            K_TAIL,
        )
        acc = _mma_stage(cur, acc, dot_a, a_scale_layout)
        cur = _load_stage(
            a_ptr,
            b_ptr,
            b_scale_ptr,
            a_offsets,
            b_offsets,
            bs_offsets,
            a_k_offs,
            b_k_offs,
            bs_k_offs,
            k0 + BLOCK_K,
            K,
            K_TAIL,
        )
        acc = _mma_stage(nxt, acc, dot_a, a_scale_layout)
    if TILES_PER_SPLIT % 2 == 0:
        nxt = _load_stage(
            a_ptr,
            b_ptr,
            b_scale_ptr,
            a_offsets,
            b_offsets,
            bs_offsets,
            a_k_offs,
            b_k_offs,
            bs_k_offs,
            (TILES_PER_SPLIT - 1) * BLOCK_K,
            K,
            K_TAIL,
        )
        acc = _mma_stage(cur, acc, dot_a, a_scale_layout)
        acc = _mma_stage(nxt, acc, dot_a, a_scale_layout)
    else:
        acc = _mma_stage(cur, acc, dot_a, a_scale_layout)

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    is_last = True
    if SPLIT_K > 1:
        tile_elems: gl.constexpr = BLOCK_M * BLOCK_N
        partial_offsets = offs_cm[:, None] * BLOCK_N + offs_cn[None, :]
        cdna4.buffer_store(
            ptr=partial_ptr + (split * num_tiles + tile) * tile_elems,
            offsets=partial_offsets,
            stored_value=acc,
            cache=".wt",
        )
        # Publish the partial without an acq_rel atomic, whose L2 writeback
        # and invalidate cost more than this GEMM: ".wt" writes through, every
        # wave waits for its stores (vmcnt), and the barrier orders all waves'
        # stores before the counter bump.
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
        is_last = old == SPLIT_K - 1
        if is_last:
            # Sum in split order, not arrival order, so results are
            # deterministic. ".cv" bypasses caches so partials written from
            # other XCDs are not stale.
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

    if is_last:
        if SWIGLU:
            # Round like the unfused path: GEMM output in BF16, then
            # SiLU(gate) * up in FP32 in the Triton silu_and_mul operation
            # order, rounded to BF16.
            pairs = gl.permute(gl.reshape(acc, [BLOCK_M, 2, HALF]), [0, 2, 1])
            pair_layout: gl.constexpr = _gate_up_pair_layout(BLOCK_M, HALF, NUM_WARPS)
            gate, up = gl.split(gl.convert_layout(pairs, pair_layout))
            gate = gate.to(gl.bfloat16).to(gl.float32)
            up = up.to(gl.bfloat16).to(gl.float32)
            h = gate * (1.0 / (1.0 + gl.exp(-gate))) * up
            h_layout: gl.constexpr = h.type.layout
            hm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, h_layout))
            hn = pid_n * HALF + gl.arange(0, HALF, gl.SliceLayout(0, h_layout))
            cdna4.buffer_store(
                ptr=c_ptr,
                offsets=hm[:, None] * stride_cm + hn[None, :],
                stored_value=h.to(c_ptr.dtype.element_ty),
                mask=(hm < M)[:, None] & (hn < N_OUT)[None, :],
            )
        else:
            cm = pid_m * BLOCK_M + offs_cm
            cn = pid_n * BLOCK_N + offs_cn
            cdna4.buffer_store(
                ptr=c_ptr,
                offsets=cm[:, None] * stride_cm + cn[None, :],
                stored_value=acc.to(c_ptr.dtype.element_ty),
                mask=(cm < M)[:, None] & (cn < N)[None, :],
            )


# M follows the batch; without do_not_specialize its value-1 and
# divisible-by-16 classes would each compile another binary.
@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_mxfp4_linear_gfx950(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    stride_am,
    stride_cm,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_bsn: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """``y = mxfp4(x) @ W.T`` in BF16."""
    _mxfp4_linear_body(
        a_ptr,
        b_ptr,
        b_scale_ptr,
        c_ptr,
        partial_ptr,
        counter_ptr,
        M,
        stride_am,
        stride_cm,
        N,
        K,
        stride_bsn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        WARPS_M,
        WARPS_N,
        SPLIT_K,
        NUM_XCDS,
        False,
    )


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M",))
def gluon_mxfp4_swiglu_linear_gfx950(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    stride_am,
    stride_cm,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_bsn: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    """``silu(mxfp4(x) @ W_gate.T) * (mxfp4(x) @ W_up.T)`` in BF16."""
    _mxfp4_linear_body(
        a_ptr,
        b_ptr,
        b_scale_ptr,
        c_ptr,
        partial_ptr,
        counter_ptr,
        M,
        stride_am,
        stride_cm,
        N,
        K,
        stride_bsn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        WARPS_M,
        WARPS_N,
        SPLIT_K,
        NUM_XCDS,
        True,
    )


def preshuffle_mxfp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Permute a packed ``[N, K/2]`` MXFP4 weight into MFMA-tile order.

    Byte ``(n, kp)`` moves to ``(n // 16) * 16 * K/2 + (kp // 32) * 512 +
    ((kp // 16) % 2) * 256 + (n % 16) * 16 + kp % 16``: each 16-row group
    stores 32-byte K chunks of its rows back to back, so the 16 bytes one
    lane feeds a ``16x16x128`` MFMA are contiguous and a warp's operand load
    for one MFMA reads 1 KiB contiguously. Same order as the MoE kernels'
    ``(16, 16)`` weight tiles.
    """
    n, kp = weight.shape
    if weight.dtype != torch.uint8 or n % 16 or kp % 32:
        raise ValueError(
            "MXFP4 weight preshuffle needs uint8 [N, K/2] with N % 16 == 0 and "
            f"K % 64 == 0, got {weight.dtype} {tuple(weight.shape)}"
        )
    return (
        weight.view(n // 16, 16, kp // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(n, kp)
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


# Tile configuration: (BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N, SPLIT_K).
# A call uses the config of the smallest M bucket covering its M (None covers
# the rest), so M stays a runtime argument within a bucket.
_Config = tuple[int, int, int, int, int, int]
_NUM_XCDS = 8

# Cold-cache sweeps on gfx950 at the Qwen3.8-Flash-Next TP2 shared expert,
# keyed by (SwiGLU, N, K): gate_up [640, 2560] and down [2560, 320].
_TUNED_CONFIGS: dict[tuple[bool, int, int], dict[int | None, _Config]] = {
    (True, 640, 2560): {
        64: (16, 64, 512, 1, 4, 5),
        256: (16, 64, 256, 1, 4, 2),
        512: (32, 64, 256, 1, 4, 2),
        2048: (32, 128, 256, 1, 4, 1),
        None: (64, 256, 128, 1, 4, 1),
    },
    (False, 2560, 320): {
        512: (16, 128, 128, 1, 4, 1),
        2048: (32, 256, 128, 1, 4, 1),
        None: (64, 256, 128, 1, 4, 1),
    },
}

# Split K until the grid reaches this many programs (small M only).
_MIN_PROGRAMS = 128


def _heuristic_config(m: int, n: int, k: int, swiglu: bool) -> _Config:
    """Untuned fallback following the tuned shapes' winners."""
    if m <= 64:
        block_m, block_n, block_k = 16, 64, 256
    elif m <= 512:
        block_m, block_n, block_k = 32, 64, 256
    else:
        block_m, block_n, block_k = 64, 256, 128
    block_n = max(64, min(block_n, triton.next_power_of_2(n)))
    block_k = max(128, min(block_k, triton.next_power_of_2(k) // 2))
    k_tiles = triton.cdiv(k, block_k)
    n_tiles = triton.cdiv(n // 2, block_n // 2) if swiglu else triton.cdiv(n, block_n)
    tiles = triton.cdiv(m, block_m) * n_tiles
    split_k = 1
    if m <= 512:
        for candidate in (2, 4, 5, 8):
            if tiles * split_k >= _MIN_PROGRAMS:
                break
            if k_tiles % candidate == 0 and k_tiles // candidate >= 2:
                split_k = candidate
    return (block_m, block_n, block_k, 1, 4, split_k)


def _choose_config(m: int, n: int, k: int, swiglu: bool) -> _Config:
    tuned = _TUNED_CONFIGS.get((swiglu, n, k))
    if tuned is not None:
        for bound, config in tuned.items():
            if bound is None or m <= bound:
                return config
    return _heuristic_config(m, n, k, swiglu)


def supports_gluon_mxfp4_linear_gfx950(n: int, k: int, swiglu: bool) -> bool:
    """Whether the gfx950 MXFP4 linear covers an ``[N, K]`` weight."""
    return (
        n % (32 if swiglu else 16) == 0
        and k % 64 == 0
        # Buffer offsets are 32-bit.
        and n * k < 2**31
    )


def _num_tiles(m: int, n: int, config: _Config, swiglu: bool) -> int:
    block_m, block_n = config[0], config[1]
    n_tiles = triton.cdiv(n // 2, block_n // 2) if swiglu else triton.cdiv(n, block_n)
    return triton.cdiv(m, block_m) * n_tiles


def gluon_mxfp4_linear_num_counters_gfx950(m: int, n: int, k: int, swiglu: bool) -> int:
    """Split-K arrival counters an MXFP4 linear launch needs; 0 without split-K.

    Args:
        m: Activation rows.
        n: Weight rows (gate and up rows together for SwiGLU).
        k: Reduction width.
        swiglu: Whether the launch is the fused SwiGLU linear.
    """
    config = _choose_config(m, n, k, swiglu)
    return 0 if config[5] == 1 else _num_tiles(m, n, config, swiglu)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    counters: torch.Tensor,
    out: torch.Tensor | None,
    swiglu: bool,
) -> torch.Tensor:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError("gfx950 MXFP4 linear needs a BF16 [M, K] x with unit stride")
    m, k = x.shape
    n = weight.shape[0]
    if weight.dtype != torch.uint8 or tuple(weight.shape) != (n, k // 2):
        raise ValueError(
            f"gfx950 MXFP4 linear needs a uint8 [N, {k // 2}] weight, got "
            f"{weight.dtype} {tuple(weight.shape)}"
        )
    if not weight.is_contiguous():
        raise ValueError("gfx950 MXFP4 linear needs a contiguous preshuffled weight")
    if (
        weight_scale.dtype != torch.uint8
        or tuple(weight_scale.shape) != (n, k // MXFP4_GROUP)
        or weight_scale.stride(1) != 1
    ):
        raise ValueError(
            "gfx950 MXFP4 linear needs uint8 [N, K/32] weight scales with unit "
            f"inner stride, got {weight_scale.dtype} {tuple(weight_scale.shape)}"
        )
    if not supports_gluon_mxfp4_linear_gfx950(n, k, swiglu):
        raise ValueError(f"gfx950 MXFP4 linear does not cover N={n}, K={k}")
    out_cols = n // 2 if swiglu else n
    if out is None:
        out = torch.empty((m, out_cols), device=x.device, dtype=torch.bfloat16)
    elif (
        tuple(out.shape) != (m, out_cols)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"gfx950 MXFP4 linear out must be BF16 [{m}, {out_cols}] with unit "
            "inner stride"
        )
    if m == 0:
        return out

    config = _choose_config(m, n, k, swiglu)
    block_m, block_n, block_k, warps_m, warps_n, split_k = config
    num_tiles = _num_tiles(m, n, config, swiglu)
    if split_k > 1:
        check_splitk_counters(counters, num_tiles, x.device, "gfx950 MXFP4 linear")
        partial = _get_partial_scratch(
            x.device, num_tiles * split_k * block_m * block_n
        )
    else:
        partial = counters = out
    kernel = gluon_mxfp4_swiglu_linear_gfx950 if swiglu else gluon_mxfp4_linear_gfx950
    kernel[(num_tiles * split_k,)](
        x,
        weight,
        weight_scale,
        out,
        partial,
        counters,
        m,
        x.stride(0),
        out.stride(0),
        n,
        k,
        weight_scale.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        SPLIT_K=split_k,
        NUM_XCDS=_NUM_XCDS,
        num_warps=warps_m * warps_n,
    )
    return out


def launch_gluon_mxfp4_linear_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    counters: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute ``mxfp4(x) @ W.T`` on gfx950.

    Args:
        x: BF16 activation ``[M, K]`` with unit inner stride; it is quantized
            to MXFP4 (per-32 UE8M0 scales) inside the kernel.
        weight: Contiguous uint8 ``[N, K/2]`` packed E2M1 weight in the
            order of ``preshuffle_mxfp4_weight``. ``N % 16 == 0`` and
            ``K % 64 == 0``.
        weight_scale: uint8 UE8M0 ``[N, K/32]`` with unit inner stride.
        counters: Zeroed int32 split-K arrival counters, at least
            ``gluon_mxfp4_linear_num_counters_gfx950(M, N, K, False)`` of
            them; the launch leaves them zeroed (see ``splitk.py``).
        out: Optional BF16 ``[M, N]`` destination with unit inner stride.

    Returns:
        BF16 ``[M, N]``; ``out`` when provided.
    """
    return _launch(x, weight, weight_scale, counters, out, swiglu=False)


def launch_gluon_mxfp4_swiglu_linear_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    counters: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute ``silu(mxfp4(x) @ W_gate.T) * (mxfp4(x) @ W_up.T)`` on gfx950.

    Args:
        x: BF16 activation ``[M, K]`` with unit inner stride.
        weight: Contiguous preshuffled uint8 ``[N, K/2]`` weight whose rows
            ``[0, N/2)`` are the gate projection and ``[N/2, N)`` the up
            projection. ``N % 32 == 0`` and ``K % 64 == 0``.
        weight_scale: uint8 UE8M0 ``[N, K/32]`` with unit inner stride.
        counters: Zeroed int32 split-K arrival counters, at least
            ``gluon_mxfp4_linear_num_counters_gfx950(M, N, K, True)`` of them;
            the launch leaves them zeroed (see ``splitk.py``).
        out: Optional BF16 ``[M, N/2]`` destination with unit inner stride.

    Returns:
        BF16 ``[M, N/2]``, rounded like the unfused GEMM, SiLU-and-mul path;
        ``out`` when provided.
    """
    return _launch(x, weight, weight_scale, counters, out, swiglu=True)


__all__ = [
    "gluon_mxfp4_linear_gfx950",
    "gluon_mxfp4_linear_num_counters_gfx950",
    "gluon_mxfp4_swiglu_linear_gfx950",
    "launch_gluon_mxfp4_linear_gfx950",
    "launch_gluon_mxfp4_swiglu_linear_gfx950",
    "preshuffle_mxfp4_weight",
    "supports_gluon_mxfp4_linear_gfx950",
]
