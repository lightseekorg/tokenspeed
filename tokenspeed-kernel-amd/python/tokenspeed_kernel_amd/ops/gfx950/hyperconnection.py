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

"""GFX950 Qwen4 gated hyper-connection mix kernels.

The mix runs as two launches instead of two GEMMs and two epilogues: a split-K
down projection whose last split applies the projection scale and SiLU (and
emits the inject logits), and an up projection whose epilogue gates and sums
the normalized branches.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

cdna4 = gl.amd.cdna4

__all__ = [
    "launch_gluon_hyperconnection_mix_gfx950",
    "supports_gluon_hyperconnection_mix_gfx950",
]

# Above this many rows the GEMM-plus-epilogue path is faster.
MAX_ROWS = 256
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_workspaces: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _mix_down_launch_metadata(grid, kernel, args):
    """Report the down projection work and traffic, partials included."""
    m = args["M"]
    k = args["K"]
    n = args["NP"]
    elem = args["a_ptr"].element_size()
    partials = grid[0] * args["BLOCK_M"] * args["BLOCK_N"] * 4
    return {
        "name": kernel.name,
        "flops16": 2 * m * n * k,
        "bytes": m * k * elem + n * k * elem + m * n * elem + 2 * partials,
    }


def _mix_up_launch_metadata(grid, kernel, args):
    """Report the up projection plus the gated branch sum."""
    m = args["M"]
    rank = args["R"]
    hidden = args["H"]
    branches = args["HC"]
    elem = args["act_ptr"].element_size()
    return {
        "name": kernel.name,
        "flops16": 2 * m * branches * hidden * rank,
        "flops32": 3 * m * branches * hidden,
        "bytes": (
            m * rank + branches * hidden * rank + m * branches * hidden + m * hidden
        )
        * elem,
    }


@gluon.constexpr_function
def _coalesced_layout(rows, cols, num_warps, k_dim):
    """Blocked layout giving each lane 16 contiguous bytes along ``k_dim``.

    Operands loaded this way are read from memory once per workgroup; the
    conversion to the MFMA operand layout broadcasts them through LDS to
    every wave that needs them.
    """
    if k_dim == 1:
        lanes_k = min(64, cols // 8)
        return gl.BlockedLayout(
            [1, 8], [64 // lanes_k, lanes_k], [num_warps, 1], [1, 0]
        )
    lanes_k = min(64, rows // 8)
    return gl.BlockedLayout([8, 1], [lanes_k, 64 // lanes_k], [1, num_warps], [0, 1])


@gluon.jit(
    launch_metadata=_mix_down_launch_metadata,
    do_not_specialize=("M",),
)
def gluon_hyperconnection_mix_gfx950(
    a_ptr,
    w_ptr,
    act_ptr,
    inject_ptr,
    partial_ptr,
    counter_ptr,
    M,
    stride_am,
    stride_actm,
    stride_injm,
    projection_scale,
    K: gl.constexpr,
    NP: gl.constexpr,
    LOWRANK: gl.constexpr,
    N_TILES: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
):
    """Down projection ``a @ w.T`` with a scale-SiLU epilogue.

    Programs are ``(m_tile, n_tile, split)``. Each split stores its FP32
    partial and bumps the tile counter; the last split sums the partials in
    split order (deterministic), rounds the projection to the input dtype as
    a standalone GEMM would, applies ``projection_scale``, writes
    ``silu(.)`` for the first ``LOWRANK`` columns and the scaled inject logits
    for the rest, and resets the counter.
    """
    pid = gl.program_id(0)
    split = pid % SPLIT_K
    tile = pid // SPLIT_K
    m_tile = tile // N_TILES
    n_tile = tile % N_TILES

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)

    # One K tile per program, so every load is issued before the first MFMA.
    gl.static_assert(K == SPLIT_K * BLOCK_K, "each split must be one K tile")
    k_base = split * BLOCK_K
    a_layout: gl.constexpr = _coalesced_layout(BLOCK_M, BLOCK_K, gl.num_warps(), 1)
    b_layout: gl.constexpr = _coalesced_layout(BLOCK_K, BLOCK_N, gl.num_warps(), 0)
    # Rows past M and weight rows past NP are clamped instead of masked; the
    # epilogue masks drop them.
    a_rows = gl.minimum(
        m_tile * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, a_layout)), M - 1
    )
    b_rows = gl.minimum(
        n_tile * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, b_layout)), NP - 1
    )
    a = cdna4.buffer_load(
        a_ptr,
        gl.expand_dims(a_rows * stride_am + k_base, 1)
        + gl.expand_dims(gl.arange(0, BLOCK_K, gl.SliceLayout(0, a_layout)), 0),
    )
    b = cdna4.buffer_load(
        w_ptr,
        gl.expand_dims(gl.arange(0, BLOCK_K, gl.SliceLayout(1, b_layout)) + k_base, 1)
        + gl.expand_dims(b_rows * K, 0),
    )
    acc = cdna4.mfma(
        gl.convert_layout(a, dot_a),
        gl.convert_layout(b, dot_b),
        gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mfma),
    )

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    if SPLIT_K > 1:
        tile_elems: gl.constexpr = BLOCK_M * BLOCK_N
        partial_offsets = gl.expand_dims(offs_cm * BLOCK_N, 1) + gl.expand_dims(
            offs_cn, 0
        )
        cdna4.buffer_store(
            acc,
            partial_ptr + (tile * SPLIT_K + split) * tile_elems,
            partial_offsets,
            cache=".wt",
        )
        # Publish the partial without an acq_rel atomic: ".wt" writes
        # through, each wave waits for its stores (vmcnt), and the barrier
        # orders every wave's stores before the counter bump.
        gl.inline_asm_elementwise(
            asm="s_waitcnt vmcnt(0)",
            constraints="=v,0",
            args=[offs_cm],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.barrier()
        arrived = gl.atomic_add(counter_ptr + tile, 1, sem="relaxed", scope="gpu")
        if arrived != SPLIT_K - 1:
            return
        # Sum in split order; ".cv" bypasses caches holding stale lines from
        # other XCDs.
        acc = cdna4.buffer_load(
            partial_ptr + tile * SPLIT_K * tile_elems, partial_offsets, cache=".cv"
        )
        for s in gl.static_range(1, SPLIT_K):
            acc += cdna4.buffer_load(
                partial_ptr + (tile * SPLIT_K + s) * tile_elems,
                partial_offsets,
                cache=".cv",
            )
        gl.store(counter_ptr + tile, 0)

    cm = m_tile * BLOCK_M + offs_cm
    cn = n_tile * BLOCK_N + offs_cn
    row_ok = gl.expand_dims(cm < M, 1)
    value = acc.to(act_ptr.dtype.element_ty).to(gl.float32) * projection_scale
    is_down = gl.expand_dims(cn < LOWRANK, 0)
    activated = value / (1.0 + gl.exp(-value))
    act_offs = gl.expand_dims(cm * stride_actm, 1) + gl.expand_dims(cn, 0)
    cdna4.buffer_store(
        activated.to(act_ptr.dtype.element_ty),
        act_ptr,
        act_offs,
        mask=row_ok & is_down,
    )
    if NP > LOWRANK:
        is_inject = gl.expand_dims((cn >= LOWRANK) & (cn < NP), 0)
        inj_offs = gl.expand_dims(cm * stride_injm, 1) + gl.expand_dims(cn - LOWRANK, 0)
        cdna4.buffer_store(
            value.to(inject_ptr.dtype.element_ty),
            inject_ptr,
            inj_offs,
            mask=row_ok & is_inject,
        )


@gluon.jit(
    launch_metadata=_mix_up_launch_metadata,
    do_not_specialize=("M",),
)
def gluon_hyperconnection_mix_up_gfx950(
    act_ptr,
    up_ptr,
    normalized_ptr,
    out_ptr,
    M,
    stride_actm,
    stride_nm,
    stride_om,
    R: gl.constexpr,
    H: gl.constexpr,
    HC: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_R: gl.constexpr,
    WARPS_M: gl.constexpr,
):
    """Up projection with the gated branch sum as its epilogue.

    Program ``(m_tile, h_tile)`` computes the gate logits of hidden columns
    ``h`` of every branch, ``act @ up[b * H + h].T``, as one MFMA tile whose
    columns are ``(branch, h)`` with one branch per wave column. It rounds
    them to the input dtype as a standalone GEMM would, and writes
    ``mean_b(sigmoid(gate_b) * normalized[:, b * H + h])``.
    """
    pid = gl.program_id(0)
    H_TILES: gl.constexpr = H // BLOCK_H
    m_tile = pid // H_TILES
    h_tile = pid % H_TILES
    BLOCK_N: gl.constexpr = HC * BLOCK_H

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, HC],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)

    # The rank is covered by a power-of-two head and tail so that every load
    # is issued before the first MFMA (one memory round trip). A tail must
    # fill at least one 32-deep MFMA K step.
    R_HEAD: gl.constexpr = BLOCK_R
    R_TAIL: gl.constexpr = R - BLOCK_R
    gl.static_assert(
        R_TAIL == 0 or (R_TAIL >= 32 and (R_TAIL & (R_TAIL - 1)) == 0),
        "rank must be the head tile plus a power-of-two tail of at least 32",
    )
    offs_bc = gl.arange(0, BLOCK_N, gl.SliceLayout(0, dot_b))
    b_rows = (offs_bc // BLOCK_H) * H + h_tile * BLOCK_H + offs_bc % BLOCK_H
    head_layout: gl.constexpr = _coalesced_layout(BLOCK_M, R_HEAD, gl.num_warps(), 1)
    head_rows = gl.minimum(
        m_tile * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, head_layout)), M - 1
    )
    a_head = cdna4.buffer_load(
        act_ptr,
        gl.expand_dims(head_rows * stride_actm, 1)
        + gl.expand_dims(gl.arange(0, R_HEAD, gl.SliceLayout(0, head_layout)), 0),
    )
    b_head = cdna4.buffer_load(
        up_ptr,
        gl.expand_dims(gl.arange(0, R_HEAD, gl.SliceLayout(1, dot_b)), 1)
        + gl.expand_dims(b_rows * R, 0),
    )
    if R_TAIL > 0:
        tail_layout: gl.constexpr = _coalesced_layout(
            BLOCK_M, R_TAIL, gl.num_warps(), 1
        )
        tail_rows = gl.minimum(
            m_tile * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, tail_layout)),
            M - 1,
        )
        a_tail = cdna4.buffer_load(
            act_ptr,
            gl.expand_dims(tail_rows * stride_actm + R_HEAD, 1)
            + gl.expand_dims(gl.arange(0, R_TAIL, gl.SliceLayout(0, tail_layout)), 0),
        )
        b_tail = cdna4.buffer_load(
            up_ptr,
            gl.expand_dims(gl.arange(0, R_TAIL, gl.SliceLayout(1, dot_b)), 1)
            + gl.expand_dims(b_rows * R + R_HEAD, 0),
        )

    offs_cm = m_tile * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    offs_cc = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    c_cols = (offs_cc // BLOCK_H) * H + h_tile * BLOCK_H + offs_cc % BLOCK_H
    c_rows = gl.minimum(offs_cm, M - 1)
    normalized = cdna4.buffer_load(
        normalized_ptr,
        gl.expand_dims(c_rows * stride_nm, 1) + gl.expand_dims(c_cols, 0),
    )

    acc = cdna4.mfma(
        gl.convert_layout(a_head, dot_a),
        b_head,
        gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mfma),
    )
    if R_TAIL > 0:
        acc = cdna4.mfma(gl.convert_layout(a_tail, dot_a), b_tail, acc)
    gate = acc.to(out_ptr.dtype.element_ty).to(gl.float32)
    gated = normalized.to(gl.float32) / (1.0 + gl.exp(-gate))
    mixed = gl.sum(gl.reshape(gated, [BLOCK_M, HC, BLOCK_H]), axis=1) / HC

    out_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [64 // BLOCK_H, BLOCK_H], [WARPS_M * HC, 1], [1, 0]
    )
    mixed = gl.convert_layout(mixed, out_layout)
    offs_om = m_tile * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))
    offs_oh = h_tile * BLOCK_H + gl.arange(0, BLOCK_H, gl.SliceLayout(0, out_layout))
    cdna4.buffer_store(
        mixed.to(out_ptr.dtype.element_ty),
        out_ptr,
        gl.expand_dims(offs_om * stride_om, 1) + gl.expand_dims(offs_oh, 0),
        mask=gl.expand_dims(offs_om < M, 1),
    )


def _workspace(
    device: torch.device, split_k: int, block_m: int, block_n: int, n_tiles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 split partials and zeroed arrival counters for a tiling.

    Sized for ``MAX_ROWS`` so one buffer serves every row count of the
    tiling. The last split of each tile resets its counter, so the counters
    are zero again whenever a launch completes; launches sharing a workspace
    (one per device and tiling) must therefore be ordered on one stream, as
    the model's hyper-connection layers are.

    The first request for a tiling must run eagerly: under graph capture the
    counter zero fill would only run when that one graph replays, while every
    later launch of the tiling reuses the cached counters. Serving's eager
    warmups before each capture allocate every tiling a graph uses.
    """
    index = torch.cuda.current_device() if device.index is None else device.index
    key = (index, split_k, block_m, block_n, n_tiles)
    workspace = _workspaces.get(key)
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "gfx950 hyper-connection mix workspace is not initialized for "
                "graph capture; run this row count eagerly before capture"
            )
        tiles = -(-MAX_ROWS // block_m) * n_tiles
        workspace = (
            torch.empty(
                (tiles * split_k * block_m * block_n,),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros((tiles,), dtype=torch.int32, device=device),
        )
        _workspaces[key] = workspace
    return workspace


# Cold-cache rocprofv3 sweeps at the Qwen3.8 HC4/H2560/R320 shapes. Down:
# (BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N), one K tile per split; 16-wide
# weight tiles and 1024-deep splits keep both the per-workgroup traffic and the last
# split's partial reduction small. Up: (BLOCK_M, BLOCK_H, BLOCK_R, WARPS_M).
def _down_config(rows: int) -> tuple[int, int, int, int, int]:
    if rows <= 32:
        return 16, 16, 1024, 1, 1
    return 64, 16, 1024, 4, 1


def _up_config(rows: int) -> tuple[int, int, int, int]:
    if rows <= 16:
        return 16, 16, 256, 1
    if rows <= 32:
        return 32, 16, 256, 1
    return 64, 16, 256, 1


def supports_gluon_hyperconnection_mix_gfx950(
    rows: int, hc_count: int, hidden_size: int, lowrank: int
) -> bool:
    """Whether the two-launch mix supports (and was tuned for) this shape."""
    wide = hc_count * hidden_size
    # The up kernel splits the rank into a 256 head and a power-of-two tail
    # that fills at least one 32-deep MFMA K step.
    tail = lowrank - 256
    return (
        1 <= rows <= MAX_ROWS
        and hc_count == 4
        and hidden_size % 16 == 0
        and wide % 10240 == 0
        and tail >= 32
        and tail & (tail - 1) == 0
    )


def launch_gluon_hyperconnection_mix_gfx950(
    normalized: torch.Tensor,
    projection_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    projection_rows: int,
    projection_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Mix normalized hyper-connection branches in two launches.

    Args:
        normalized: Contiguous ``[rows, hc_count * hidden_size]`` FP16/BF16
            branches, ``1 <= rows <= MAX_ROWS``.
        projection_weight: Contiguous ``[stored_rows, wide]`` fused mix-down /
            inject weight; ``stored_rows`` is ``projection_rows`` or that
            rounded up to eight rows by model-loading padding, which is never
            read.
        up_weight: Contiguous ``[wide, lowrank]`` mix-up weight.
        hc_count: Number of branches.
        hidden_size: Width of one branch.
        lowrank: Rank of the mix gate bottleneck.
        projection_rows: Logical weight rows: ``lowrank``, or ``lowrank +
            hc_count`` with inject logits.
        projection_scale: Scale applied to the down and inject projections.

    Returns:
        The mixed ``[rows, hidden_size]`` tensor and the ``[rows, hc_count]``
        inject logits (``None`` without inject rows), in the input dtype.
    """
    rows, wide = normalized.shape
    padded_rows = -(-projection_rows // 8) * 8
    if (
        normalized.dtype not in _SUPPORTED_DTYPES
        or projection_weight.dtype != normalized.dtype
        or up_weight.dtype != normalized.dtype
        or not all(
            t.is_contiguous() for t in (normalized, projection_weight, up_weight)
        )
        or wide != hc_count * hidden_size
        or up_weight.shape != (wide, lowrank)
        or projection_rows not in (lowrank, lowrank + hc_count)
        or projection_weight.shape not in ((projection_rows, wide), (padded_rows, wide))
        or not supports_gluon_hyperconnection_mix_gfx950(
            rows, hc_count, hidden_size, lowrank
        )
    ):
        raise ValueError(
            "GFX950 hyper-connection mix requires contiguous FP16/BF16 tensors, "
            f"1..{MAX_ROWS} rows, four branches, a wide dimension divisible by "
            "10240 and a rank of 256 plus a power of two of at least 32"
        )
    device = normalized.device
    dtype = normalized.dtype
    activated = torch.empty((rows, lowrank), dtype=dtype, device=device)
    inject = (
        torch.empty((rows, hc_count), dtype=dtype, device=device)
        if projection_rows > lowrank
        else None
    )
    mixed = torch.empty((rows, hidden_size), dtype=dtype, device=device)

    block_m, block_n, block_k, warps_m, warps_n = _down_config(rows)
    split_k = wide // block_k
    n_tiles = -(-projection_rows // block_n)
    partial, counters = _workspace(device, split_k, block_m, block_n, n_tiles)
    gluon_hyperconnection_mix_gfx950[(-(-rows // block_m) * n_tiles * split_k,)](
        normalized,
        projection_weight,
        activated,
        activated if inject is None else inject,
        partial,
        counters,
        rows,
        normalized.stride(0),
        activated.stride(0),
        0 if inject is None else inject.stride(0),
        float(projection_scale),
        K=wide,
        NP=projection_rows,
        LOWRANK=lowrank,
        N_TILES=n_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        WARPS_M=warps_m,
        WARPS_N=warps_n,
        num_warps=warps_m * warps_n,
    )

    block_m, block_h, block_r, warps_m = _up_config(rows)
    gluon_hyperconnection_mix_up_gfx950[
        (-(-rows // block_m) * (hidden_size // block_h),)
    ](
        activated,
        up_weight,
        normalized,
        mixed,
        rows,
        activated.stride(0),
        normalized.stride(0),
        mixed.stride(0),
        R=lowrank,
        H=hidden_size,
        HC=hc_count,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        BLOCK_R=block_r,
        WARPS_M=warps_m,
        num_warps=warps_m * hc_count,
    )
    return mixed, inject
